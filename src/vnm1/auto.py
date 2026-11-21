"""Auto mode orchestration (design doc section 6.2).

Auto mode tries the capacity ladder ``standard -> standard8 -> quality
-> quality8`` under the fixed judgment conditions (recipe v2, 50,000
iterations, default weights), evaluating every trial from its read-back
artifact and stopping at the first full gate pass.  If the quality8
default trial still fails, at most one adjustment trial is considered,
under the narrow-miss definition of section 6.2.2 and the pattern table
of section 6.2.3.  When nothing passes, the report's representative
candidate is chosen with the section 7.2 A tie-break.

State and resume (section 6.2.5): after every trial transition the run
state is written to ``work/auto_state.json``.  The resume hash covers
only the resolved map files' SHA-256 bytes, the normalized parameters
and the training core's identity -- never CLI path strings -- so
re-invoking the same conversion from a different working directory or
with differently spelled paths resumes instead of restarting.
Completed (evaluated) trials are never retrained on resume; a trial
that was left mid-run is marked interrupted and rerun from scratch at
trial granularity (its checkpoint path stays recorded in the state).

This module imports torch only indirectly, through the trial runner
the CLI injects (``vnm1.driver.run_trial`` in production); the decision
logic itself is pure and unit-tested without torch.
"""

from __future__ import annotations

import hashlib
import json
import os
from datetime import datetime, timezone
from pathlib import Path

from . import __version__
from . import core as _core
from .driver import GATE
from .errors import ExportVerificationError, TrainingError
from .inputs import LOGICAL_MAPS
from .report import core_provenance

#: The basic ladder (design doc 6.2.1), in order.
LADDER = ("standard", "standard8", "quality", "quality8")

#: Fixed ladder conditions (design doc 6.2.1 / 5.1).
LADDER_ITERATIONS = 50000
BASE_COLOR_WEIGHT = 4.0
BASE_SCALAR_WEIGHT = 1.0

#: Narrow-miss draft definition (design doc 6.2.2).  These are initial
#: operating values derived from the judgment run, not universal
#: constants; the step C desk check (step-c-records/) applied them to
#: all 30 capability-table materials before implementation.
NARROW_PSNR_MIN = 33.5
NARROW_MAE_MAX = 0.030

#: Adjustment patterns (design doc 6.2.3): failing pattern ->
#: (color_weight, scalar_weight, iterations).
ADJUSTMENT_PATTERNS = {
    "scalar_only": (4.0, 3.0, 50000),
    "color_only": (6.0, 1.0, 150000),
    "color_and_scalar": (6.0, 3.0, 150000),
}

#: Per-rung wall-time estimate ranges in minutes on a T4-class GPU
#: (design doc 8.2, from judgment-run measurements; shown as estimates,
#: never as guarantees).
RUN_ESTIMATE_MINUTES = {
    "standard": (7, 9),
    "standard8": (7, 9),
    "quality": (9, 10),
    "quality8": (9, 10),
}
ADJUSTMENT_ESTIMATE_MINUTES = 35
AUTO_TOTAL_ESTIMATE = "about 70-80 min"

STATE_FILENAME = "auto_state.json"
STATE_VERSION = 1

#: Trial statuses (design doc 6.2.5): pending / running / interrupted /
#: evaluated are the lifecycle; the evaluated outcome is carried by the
#: trial record's gate_passed flag and by status passed / failed.
STATUS_PENDING = "pending"
STATUS_RUNNING = "running"
STATUS_INTERRUPTED = "interrupted"
STATUS_PASSED = "passed"
STATUS_FAILED = "failed"
STATUS_ERROR = "error"
_EVALUATED = (STATUS_PASSED, STATUS_FAILED)

_PSRN_CHECK = "psnr_base_color_db"
_SCALAR_CHECKS = ("mae_roughness", "mae_metalness", "mae_ao")
_NORMAL_CHECKS = ("normal_angle_mean_deg", "normal_angle_p95_deg")


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


# ---------------------------------------------------------------------------
# Gate-failure analysis over driver.gate_verdict checks
# ---------------------------------------------------------------------------
def failing_checks(checks: dict) -> dict:
    """Applicable, failed checks -> overshoot ratio.

    The ratio is (shortfall / threshold) for the PSNR (a >= check) and
    (excess / threshold) for the <= checks -- the "amount missed,
    divided by the threshold" of the section 7.2 A tie-break.
    """
    out = {}
    for name, c in checks.items():
        if not c.get("gate_applicable", True) or c.get("passed", True):
            continue
        threshold = c["threshold"]
        if name == _PSRN_CHECK:
            out[name] = (threshold - c["value"]) / threshold
        else:
            out[name] = (c["value"] - threshold) / threshold
    return out


def adjustment_decision(checks: dict) -> dict:
    """Section 6.2.2 firing conditions + 6.2.3 pattern selection.

    Applied to the quality8 default-weights trial's checks.  Returns a
    dict with ``eligible`` and, when eligible, the pattern and the
    adjustment trial's weights / iterations; when not eligible, a
    machine-readable ``reason`` (also shown in the report so a failed
    run records *why* no adjustment ran).
    """
    fails = failing_checks(checks)
    if not fails:
        return {"eligible": False, "reason": "gate_passed",
                "pattern": None}
    normal_fails = [n for n in fails if n in _NORMAL_CHECKS]
    if normal_fails:
        # Color weight increases have worsened normals in the judgment
        # run; section 6.2.2 excludes such materials from auto
        # adjustment entirely.
        return {"eligible": False, "reason": "normal_not_passing",
                "pattern": None,
                "detail": f"failing normal checks: {normal_fails}"}
    color_fail = _PSRN_CHECK in fails
    scalar_fails = [n for n in fails if n in _SCALAR_CHECKS]
    if color_fail:
        psnr = checks[_PSRN_CHECK]["value"]
        if psnr < NARROW_PSNR_MIN:
            return {"eligible": False, "reason": "color_not_narrow",
                    "pattern": None,
                    "detail": f"psnr {psnr:.2f} < narrow floor "
                              f"{NARROW_PSNR_MIN}"}
    for name in scalar_fails:
        value = checks[name]["value"]
        if value > NARROW_MAE_MAX:
            return {"eligible": False, "reason": "scalar_not_narrow",
                    "pattern": None,
                    "detail": f"{name} {value:.4f} > narrow ceiling "
                              f"{NARROW_MAE_MAX}"}
    if color_fail and scalar_fails:
        pattern = "color_and_scalar"
    elif color_fail:
        pattern = "color_only"
    else:
        pattern = "scalar_only"
    color_weight, scalar_weight, iterations = ADJUSTMENT_PATTERNS[pattern]
    return {"eligible": True, "reason": "narrow_miss", "pattern": pattern,
            "color_weight": color_weight, "scalar_weight": scalar_weight,
            "iterations": iterations}


# ---------------------------------------------------------------------------
# Representative candidate (design doc 7.2 A tie-break)
# ---------------------------------------------------------------------------
def _profile_rank(profile: str) -> int:
    return LADDER.index(profile) if profile in LADDER else len(LADDER)


def candidate_key(record: dict, trial_index: int) -> tuple:
    """Sort key for the representative tie-break; smaller is 'further
    along'.  (1) fewest failing applicable checks, (2) smallest
    maximum overshoot ratio, (3) higher profile preferred, then the
    later trial (an adjustment trial of the same profile counts as
    further along than the default trial it followed)."""
    fails = failing_checks(record["checks"])
    max_overshoot = max(fails.values()) if fails else 0.0
    return (len(fails), max_overshoot,
            -_profile_rank(record["profile"]), -trial_index)


def pick_representative(records: list) -> dict | None:
    """records: trial record dicts in run order.  Returns the record
    chosen as representative candidate, or None if there are none."""
    if not records:
        return None
    return min(enumerate(records),
               key=lambda pair: candidate_key(pair[1], pair[0]))[1]


# ---------------------------------------------------------------------------
# Resume hash (design doc 6.2.5)
# ---------------------------------------------------------------------------
def normalized_run_params(seed: int, normal_convention: str, *,
                          mip_train: bool = False,
                          fast_eval: bool = False) -> dict:
    """The normalized parameter set the resume hash covers: the fixed
    ladder, the adjustment rules and the gate -- everything that would
    change what a rerun computes.  No paths, no device.

    ``mip_train`` (2026-11-21) is recorded only when enabled, so a
    mip-0 run's hash -- and therefore its resume compatibility with
    state files written before the option existed -- is unchanged,
    while a mip-trained run can never resume-skip mip-0 trials (or
    the reverse).  ``fast_eval`` (Neon O4, 2026-11-21) follows the
    same rule: it changes what a trial *records* (no multi-mip
    measurement), so a fast-eval run's lighter trial records must
    never be resume-skipped by a full run, or the reverse."""
    params = {
        "mode": "auto",
        "seed": int(seed),
        "normal_convention": normal_convention,
        "ladder": [
            {"profile": p, "iterations": LADDER_ITERATIONS,
             "color_weight": BASE_COLOR_WEIGHT,
             "scalar_weight": BASE_SCALAR_WEIGHT}
            for p in LADDER],
        "adjustment": {
            "narrow_psnr_min": NARROW_PSNR_MIN,
            "narrow_mae_max": NARROW_MAE_MAX,
            "patterns": {k: list(v)
                         for k, v in ADJUSTMENT_PATTERNS.items()},
        },
        "gate": GATE,
    }
    if mip_train:
        params["mip_train"] = True
    if fast_eval:
        params["fast_eval"] = True
    return params


def compute_resume_hash(spec, teacher, *, seed: int,
                        mip_train: bool = False,
                        fast_eval: bool = False) -> str:
    """SHA-256 over resolved map bytes + normalized parameters + core
    identity (design doc 6.2.5).  Map identity is the SHA-256 of the
    file bytes (or the constant marker), exactly as recorded by the
    input layer; CLI path spellings never enter the hash."""
    h = hashlib.sha256()
    for logical in LOGICAL_MAPS:
        record = teacher.records[logical]
        h.update(logical.encode())
        h.update(b"\0")
        identity = record.sha256 or f"constant:{record.constant_value}"
        h.update(identity.encode())
        h.update(b"\0")
    h.update(json.dumps(
        normalized_run_params(seed, spec.normal_convention,
                              mip_train=mip_train,
                              fast_eval=fast_eval),
        sort_keys=True).encode())
    core = core_provenance()
    h.update(core["sha256"].encode())
    h.update(f"recipe_version={_core.RECIPE_VERSION}".encode())
    h.update(f"cli_version={__version__}".encode())
    return h.hexdigest()


# ---------------------------------------------------------------------------
# Pre-start display (design doc 6.2.4)
# ---------------------------------------------------------------------------
def format_auto_plan(spec, teacher, output_dir, *, seed: int,
                     mip_train: bool = False,
                     fast_eval: bool = False) -> str:
    lines = []
    lines.append("Mode: auto")
    if mip_train:
        lines.append("Mip training: on (--mip-train; every ladder "
                     "and adjustment trial trains on the SPEC 5.3 "
                     "mip pyramid, batch5 proposal 1)")
    if fast_eval:
        lines.append("Fast evaluation: on (--fast-eval; the "
                     "multi-mip measurement and the visual sheets "
                     "are skipped, the gate evaluation is unchanged)")
    maps = []
    constants = []
    for logical in LOGICAL_MAPS:
        record = teacher.records[logical]
        if record.source == "map":
            maps.append(logical)
        else:
            constants.append(
                f"{logical} = {record.constant_value} (gate exempt)")
    lines.append(f"Input: {spec.name} {teacher.width}x{teacher.height}, "
                 f"maps: {', '.join(maps)}")
    if constants:
        lines.append("Constant channels: " + "; ".join(constants))
    lines.append("")
    lines.append("Plan:")
    total_lo = total_hi = 0
    for i, profile in enumerate(LADDER, start=1):
        lo, hi = RUN_ESTIMATE_MINUTES[profile]
        total_lo += lo
        total_hi += hi
        lines.append(f"  {i}. {profile:<10} {LADDER_ITERATIONS:,} "
                     f"iterations  estimated {lo}-{hi} min")
    lines.append(f"  5. adjustment (only if the quality8 default trial "
                 f"fails narrow, design doc 6.2.2)  up to about "
                 f"{ADJUSTMENT_ESTIMATE_MINUTES} min")
    lines.append("")
    lines.append(f"Estimated ladder total: about {total_lo}-{total_hi} "
                 f"min; if all rungs and the adjustment run: "
                 f"{AUTO_TOTAL_ESTIMATE} on a T4-class GPU.")
    lines.append("These are estimates from the judgment run, not "
                 "guarantees.")
    lines.append(f"Output: {output_dir}")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Trial definitions and state
# ---------------------------------------------------------------------------
def ladder_trial_def(index: int, profile: str, *, seed: int) -> dict:
    return {"index": index, "kind": "ladder", "profile": profile,
            "iterations": LADDER_ITERATIONS,
            "color_weight": BASE_COLOR_WEIGHT,
            "scalar_weight": BASE_SCALAR_WEIGHT, "seed": int(seed)}


def adjustment_trial_def(index: int, decision: dict, *, seed: int) -> dict:
    return {"index": index, "kind": "adjustment", "profile": "quality8",
            "iterations": decision["iterations"],
            "color_weight": decision["color_weight"],
            "scalar_weight": decision["scalar_weight"],
            "seed": int(seed), "pattern": decision["pattern"]}


def _def_identity(trial_def: dict) -> tuple:
    return (trial_def["index"], trial_def["kind"], trial_def["profile"],
            trial_def["iterations"], trial_def["color_weight"],
            trial_def["scalar_weight"], trial_def["seed"])


def _new_state(spec, resume_hash: str) -> dict:
    return {"state_version": STATE_VERSION, "material": spec.name,
            "resume_hash": resume_hash, "created_at": _now_iso(),
            "updated_at": _now_iso(), "trials": [],
            "adjustment": None, "decision": None}


def _save_state(work_dir: Path, state: dict) -> Path:
    state["updated_at"] = _now_iso()
    path = work_dir / STATE_FILENAME
    tmp = work_dir / (STATE_FILENAME + ".tmp")
    tmp.write_text(json.dumps(state, indent=2), encoding="utf-8")
    os.replace(tmp, path)
    return path


def _find_entry(state: dict, trial_def: dict) -> dict | None:
    for entry in state["trials"]:
        if _def_identity(entry["def"]) == _def_identity(trial_def):
            return entry
    return None


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------
class AutoResult:
    def __init__(self, *, status, trials, selected, representative,
                 adjustment, state_path, resume_hash, resumed_trials):
        self.status = status                # "passed" | "failed"
        self.trials = trials                # trial record dicts, in order
        self.selected = selected            # passing trial record | None
        self.representative = representative  # record | None (on failure)
        self.adjustment = adjustment        # decision dict (+ ran flag)
        self.state_path = state_path
        self.resume_hash = resume_hash
        self.resumed_trials = resumed_trials  # count of skipped trials


def run_auto(*, spec, teacher, output_dir: Path, seed: int,
             trial_runner, progress_cb=None,
             keep_work: bool = False,
             mip_train: bool = False,
             fast_eval: bool = False) -> AutoResult:
    """Run (or resume) the auto ladder for one material.

    ``trial_runner(trial_def, trial_work_dir)`` must run one trial and
    return a ``driver.TrialOutcome`` whose artifact still sits in the
    trial work dir's staging area; placement is handled here and
    recorded in the state file.  Training / export failures propagate
    as TrainingError / ExportVerificationError after the state has
    recorded the abnormal trial.
    """
    def progress(msg: str) -> None:
        if progress_cb is not None:
            progress_cb(msg)

    material_out = output_dir / spec.name
    work_dir = material_out / "work"
    work_dir.mkdir(parents=True, exist_ok=True)
    resume_hash = compute_resume_hash(spec, teacher, seed=seed,
                                      mip_train=mip_train,
                                      fast_eval=fast_eval)

    state = None
    state_path = work_dir / STATE_FILENAME
    if state_path.is_file():
        try:
            loaded = json.loads(state_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            loaded = None
        if loaded is not None and loaded.get("resume_hash") == resume_hash:
            state = loaded
            progress("auto_state= resumed (input hash matches)")
        elif loaded is not None:
            stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
            archived = work_dir / f"auto_state.superseded-{stamp}.json"
            os.replace(state_path, archived)
            progress(f"auto_state= previous state archived to {archived} "
                     f"(input or settings changed; not reused)")
    if state is None:
        state = _new_state(spec, resume_hash)
    # A trial left mid-run by an interruption is rerun from scratch at
    # trial granularity; its recorded status becomes 'interrupted'.
    for entry in state["trials"]:
        if entry["status"] in (STATUS_RUNNING, STATUS_ERROR):
            entry["status"] = STATUS_INTERRUPTED
    _save_state(work_dir, state)

    records: list[dict] = []
    resumed = 0
    selected = None

    def place_and_record(trial_def: dict, outcome, entry: dict) -> dict:
        """Move the trial's staging artifact to its final location and
        build the serializable trial record."""
        from .report import trial_record
        staging = outcome.artifact_dir
        if outcome.gate_passed:
            dest = material_out / f"{spec.name}.vnm"
            if dest.exists():
                stamp = datetime.now(timezone.utc).strftime(
                    "%Y%m%dT%H%M%SZ")
                previous = work_dir / "attempts" / f"previous-pass-{stamp}"
                previous.parent.mkdir(parents=True, exist_ok=True)
                os.rename(dest, previous)
                progress(f"previous_pass_archived={previous}")
            os.rename(staging, dest)
            location = str(dest)
        else:
            dest = work_dir / "attempts" / \
                f"trial-{trial_def['index']:03d}-{trial_def['profile']}"
            dest.parent.mkdir(parents=True, exist_ok=True)
            if dest.exists():
                stamp = datetime.now(timezone.utc).strftime(
                    "%Y%m%dT%H%M%SZ")
                dest = work_dir / "attempts" / (
                    f"trial-{trial_def['index']:03d}-"
                    f"{trial_def['profile']}-{stamp}")
            os.rename(staging, dest)
            location = str(dest)
        trial_params = {"iterations": trial_def["iterations"],
                        "color_weight": trial_def["color_weight"],
                        "scalar_weight": trial_def["scalar_weight"],
                        "seed": trial_def["seed"]}
        if fast_eval:
            trial_params["fast_eval"] = True
        record = trial_record(
            outcome, kind=trial_def["kind"],
            trial_params=trial_params,
            artifact_location=location)
        if trial_def["kind"] == "adjustment":
            record["adjustment_pattern"] = trial_def.get("pattern")
        entry["record"] = record
        entry["artifact_location"] = location
        entry["status"] = STATUS_PASSED if outcome.gate_passed \
            else STATUS_FAILED
        entry["finished_at"] = _now_iso()
        _save_state(work_dir, state)
        progress(f"trial_{trial_def['index']:03d}_{trial_def['profile']} "
                 f"gate_passed={outcome.gate_passed} "
                 f"artifact={location}")
        return record

    def run_one(trial_def: dict) -> dict:
        nonlocal resumed
        entry = _find_entry(state, trial_def)
        if entry is not None and entry["status"] in _EVALUATED \
                and entry.get("record") is not None:
            resumed += 1
            progress(f"trial_{trial_def['index']:03d}_"
                     f"{trial_def['profile']} resume_skip "
                     f"(already evaluated)")
            return entry["record"]
        if entry is None:
            entry = {"def": trial_def, "status": STATUS_PENDING,
                     "started_at": None, "finished_at": None,
                     "record": None, "artifact_location": None,
                     "error": None}
            state["trials"].append(entry)
        entry["status"] = STATUS_RUNNING
        entry["started_at"] = _now_iso()
        entry["error"] = None
        _save_state(work_dir, state)
        trial_work = work_dir / "trials" / \
            f"trial-{trial_def['index']:03d}-{trial_def['profile']}"
        trial_work.mkdir(parents=True, exist_ok=True)
        entry["checkpoint"] = str(
            trial_work / "checkpoints" / "checkpoint_latest.pt")
        _save_state(work_dir, state)
        progress(f"trial_{trial_def['index']:03d}_{trial_def['profile']} "
                 f"start (kind={trial_def['kind']}, "
                 f"iterations={trial_def['iterations']}, "
                 f"color_weight={trial_def['color_weight']}, "
                 f"scalar_weight={trial_def['scalar_weight']})")
        try:
            outcome = trial_runner(trial_def, trial_work)
        except (TrainingError, ExportVerificationError) as exc:
            entry["status"] = STATUS_ERROR
            entry["error"] = f"{type(exc).__name__}: {exc}"
            entry["finished_at"] = _now_iso()
            _save_state(work_dir, state)
            raise
        record = place_and_record(trial_def, outcome, entry)
        if not keep_work:
            for ckpt in (trial_work / "checkpoints").glob("*.pt"):
                ckpt.unlink()
        return record

    # ---- basic ladder -------------------------------------------------
    q8_record = None
    for i, profile in enumerate(LADDER, start=1):
        record = run_one(ladder_trial_def(i, profile, seed=seed))
        records.append(record)
        if profile == "quality8":
            q8_record = record
        if record["gate_passed"]:
            selected = record
            break

    adjustment_info = {"eligible": False, "reason": "ladder_passed",
                       "pattern": None, "ran": False}
    representative = None
    if selected is None:
        decision = adjustment_decision(q8_record["checks"])
        adjustment_info = dict(decision)
        adjustment_info["ran"] = False
        state["adjustment"] = adjustment_info
        _save_state(work_dir, state)
        if decision["eligible"]:
            trial_def = adjustment_trial_def(
                len(LADDER) + 1, decision, seed=seed)
            record = run_one(trial_def)
            records.append(record)
            adjustment_info["ran"] = True
            state["adjustment"] = adjustment_info
            _save_state(work_dir, state)
            if record["gate_passed"]:
                selected = record
        if selected is None:
            representative = pick_representative(records)

    status = "passed" if selected is not None else "failed"
    state["decision"] = {
        "status": status,
        "selected_profile": selected["profile"] if selected else None,
        "representative_profile":
            representative["profile"] if representative else None,
        "adjustment": adjustment_info,
        "finished_at": _now_iso(),
    }
    state_path = _save_state(work_dir, state)
    progress(f"auto_done status={status} "
             f"selected={selected['profile'] if selected else None} "
             f"representative="
             f"{representative['profile'] if representative else None}")
    return AutoResult(status=status, trials=records, selected=selected,
                      representative=representative,
                      adjustment=adjustment_info, state_path=state_path,
                      resume_hash=resume_hash, resumed_trials=resumed)


__all__ = [
    "ADJUSTMENT_ESTIMATE_MINUTES", "ADJUSTMENT_PATTERNS",
    "AUTO_TOTAL_ESTIMATE", "BASE_COLOR_WEIGHT", "BASE_SCALAR_WEIGHT",
    "LADDER", "LADDER_ITERATIONS", "NARROW_MAE_MAX", "NARROW_PSNR_MIN",
    "RUN_ESTIMATE_MINUTES", "STATE_FILENAME", "adjustment_decision",
    "adjustment_trial_def", "candidate_key", "compute_resume_hash",
    "failing_checks", "format_auto_plan", "ladder_trial_def",
    "normalized_run_params", "pick_representative", "run_auto",
]

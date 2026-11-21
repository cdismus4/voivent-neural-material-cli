"""report.json builders (manual mode from step B, auto mode from
step C).

Implements the design doc section 7.2 content set (A conclusion,
B input record, C trial record(s), D capacity, E format and
reproducibility, F environment and compatibility) as one JSON
document.  Step D froze this schema (the auto conclusion records the
chosen trial's index), added the Markdown summary
(:mod:`vnm1.report_md`) and the ``visual_check`` record the CLI
attaches after rendering the section 7.4 sheets
(:mod:`vnm1.visual`).
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

from . import __version__
from .capacity_check import capacity_warnings
from .driver import NORMAL_WEIGHT, TrialOutcome
from .imaging import TeacherResult, packed_suspicion_warnings
from .inputs import InputSpec, LOGICAL_MAPS

#: GPU decoder compatibility wording (design doc section 7.2 F).
#: P2 completed 2026-11-20: the TS GPU decoder (TslMlpDecoderNode)
#: supports all four profiles, with GPU/CPU conformance verified on
#: desktop WebGPU (Apple Metal-3) and on a Pixel 9a (ARM Valhall).
#: The only per-profile difference left to record is the 8-channel
#: load path: those latents are uploaded as texture pairs.
_PROFILES_8CH = ("standard8", "quality8")
_GPU_NOTE_4CH = (
    "CPU reference decoder: verified by this run's read-back decode. "
    "GPU decoder: verified; the TS GPU decoder (TslMlpDecoderNode) "
    "supports all four profiles (standard, quality, standard8, "
    "quality8), with conformance verified on desktop WebGPU (Apple "
    "Metal-3) and Pixel 9a (ARM Valhall).")
_GPU_NOTE_8CH = (
    "CPU reference decoder: verified by this run's read-back decode. "
    "GPU decoder: verified; the TS GPU decoder (TslMlpDecoderNode) "
    "supports all four profiles (standard, quality, standard8, "
    "quality8), with conformance verified on desktop WebGPU (Apple "
    "Metal-3) and Pixel 9a (ARM Valhall). This profile's latents "
    "load as texture pairs (two RGBA8 textures per grid).")


def core_provenance() -> dict:
    """Identity of the imported training core.

    ``vnm1.core`` is a byte-identical copy of the judgment port
    (``stage1/src/vnm1_torch.py``); its hashes are the provenance.
    """
    core_path = Path(__file__).resolve().parent / "core.py"
    data = core_path.read_bytes()
    return {
        "module": "vnm1.core",
        "sha256": hashlib.sha256(data).hexdigest(),
        "md5": hashlib.md5(data).hexdigest(),
        "provenance": "byte-identical copy of the judgment-run port "
                      "stage1/src/vnm1_torch.py (implementation "
                      "plan section 2); numeric conventions unchanged",
    }


def input_config_hash(spec: InputSpec, teacher: TeacherResult,
                      trial_params: dict) -> str:
    """Reproducibility hash over resolved input bytes + normalized
    parameters (design doc section 6.2.5 scoping: file bytes and
    normalized parameters, never CLI path strings)."""
    h = hashlib.sha256()
    for logical in LOGICAL_MAPS:
        record = teacher.records[logical]
        h.update(logical.encode())
        h.update((record.sha256 or f"constant:{record.constant_value}")
                 .encode())
    h.update(json.dumps(trial_params, sort_keys=True).encode())
    h.update(spec.normal_convention.encode())
    return h.hexdigest()


def _input_record(spec: InputSpec, teacher: TeacherResult) -> dict:
    """Section 7.2 B input record, shared by both report builders."""
    maps_record = {}
    for logical in LOGICAL_MAPS:
        record = teacher.records[logical]
        entry = {"source": record.source,
                 "gate_exempt": record.gate_exempt}
        if record.source == "map":
            entry.update({
                "file": Path(record.file).name if record.file else None,
                "sha256": record.sha256,
                "width": record.width, "height": record.height,
                "bit_depth": record.bit_depth,
                "pil_mode": record.pil_mode,
            })
        else:
            entry["constant_value"] = record.constant_value
        maps_record[logical] = entry
    record = {
        "name": spec.name,
        "normal_convention": spec.normal_convention,
        "base_color_space_assumption":
            "sRGB-encoded; converted to linear before training",
        "size": [teacher.width, teacher.height],
        "size_verified_baseline": teacher.size_verified,
        "maps": maps_record,
        "warnings": list(teacher.warnings),
    }
    if spec.unsupported_files:
        # Neon O5 (2026-11-21): files that were in the input folder
        # but are not VNM1 inputs are recorded as excluded, never
        # silently dropped.  Present only when non-empty, so a run
        # without extra files keeps the pre-existing record shape.
        record["unsupported_files"] = list(spec.unsupported_files)
    return record


def _conclusion_warnings(teacher: TeacherResult) -> dict:
    """Conclusion-level warning fields (Neon O5, 2026-11-21).

    A packed-map suspicion is promoted out of the input warning
    list into the conclusion: a run can pass the gate while grading
    the wrong channel, so the suspicion belongs next to the verdict.
    The field is present only when a suspicion actually fired, so
    unaffected reports keep their previous shape byte for byte.
    """
    suspicion = packed_suspicion_warnings(teacher.warnings)
    if not suspicion:
        return {}
    return {"packed_map_suspicion": suspicion}


def _mip_fields(outcome: TrialOutcome) -> dict:
    """Multi-mip record fields (2026-11-21), present only when the
    run produced them: ``mip_training`` + ``mip_evaluation`` for a
    --mip-train trial, ``mip_advisory`` when the far-field advisory
    fired on a mip-0-trained trial.  A plain mip-0 trial's record is
    unchanged from before the option existed."""
    fields = {}
    if outcome.mip_training is not None:
        fields["mip_training"] = outcome.mip_training
    if outcome.mip_evaluation is not None:
        fields["mip_evaluation"] = outcome.mip_evaluation
    if outcome.mip_advisory is not None:
        fields["mip_advisory"] = outcome.mip_advisory
    return fields


def trial_record(outcome: TrialOutcome, *, kind: str,
                 trial_params: dict, artifact_location: str) -> dict:
    """Serializable per-trial record (section 7.2 C) for one trial.

    Used by the auto-mode state file and report so that a resumed run
    can rebuild its report from stored records alone, without the
    in-memory TrialOutcome objects of the original process.
    """
    return {
        "kind": kind,
        "profile": outcome.profile,
        "fine_size": list(outcome.fine_size),
        "coarse_size": list(outcome.coarse_size),
        "fine_channels": outcome.fine_channels,
        "coarse_channels": outcome.coarse_channels,
        "iterations": trial_params["iterations"],
        "color_weight": trial_params["color_weight"],
        "scalar_weight": trial_params["scalar_weight"],
        "normal_weight": NORMAL_WEIGHT,
        "seed": trial_params["seed"],
        "recipe": outcome.recipe,
        "started_at": outcome.started_at,
        "finished_at": outcome.finished_at,
        "elapsed_seconds": outcome.elapsed_seconds,
        "training_seconds": outcome.training_seconds,
        "final_loss": outcome.final_loss,
        "device": outcome.environment.get("device"),
        "environment": outcome.environment,
        "artifact_location": artifact_location,
        "artifact_files": outcome.artifact_files,
        "metrics": outcome.metrics,
        "checks": outcome.checks,
        "gate_passed": outcome.gate_passed,
        "capacity": outcome.capacity,
        "roundtrip_verification": outcome.roundtrip,
        **_mip_fields(outcome),
        **({"fast_eval": True} if trial_params.get("fast_eval")
           else {}),
    }


def build_report(spec: InputSpec, teacher: TeacherResult,
                 outcome: TrialOutcome, *, mode: str,
                 artifact_location: str,
                 trial_params: dict,
                 capacity_check: dict | None = None) -> dict:
    checks = outcome.checks
    failed_checks = [name for name, c in checks.items()
                     if c["gate_applicable"] and not c["passed"]]
    exempt = [name for name, c in checks.items()
              if not c["gate_applicable"]]
    gpu_note = (_GPU_NOTE_8CH if outcome.profile in _PROFILES_8CH
                else _GPU_NOTE_4CH)
    cap_warnings = capacity_warnings(
        outcome.capacity.get("ratio_vs_bcn_mip_chain")
        if outcome.capacity else None)

    return {
        "report_version": 1,
        "cli_version": __version__,
        "material": spec.name,
        # --- A. conclusion (7.2 A) -----------------------------------
        "conclusion": {
            "result": "passed" if outcome.gate_passed else "failed",
            "gate_passed": outcome.gate_passed,
            "selected_profile": outcome.profile,
            "failed_checks": failed_checks,
            "gate_exempt_checks": exempt,
            "artifact_location": artifact_location,
            "compatibility_note": gpu_note,
            "capacity_warnings": cap_warnings,
            **_conclusion_warnings(teacher),
        },
        # --- B. input record (7.2 B) ---------------------------------
        "input": _input_record(spec, teacher),
        # --- C. trial record (7.2 C) ---------------------------------
        "trial": {
            "mode": mode,
            "profile": outcome.profile,
            "fine_size": list(outcome.fine_size),
            "coarse_size": list(outcome.coarse_size),
            "fine_channels": outcome.fine_channels,
            "coarse_channels": outcome.coarse_channels,
            "iterations": trial_params["iterations"],
            "color_weight": trial_params["color_weight"],
            "scalar_weight": trial_params["scalar_weight"],
            "normal_weight": NORMAL_WEIGHT,
            "seed": trial_params["seed"],
            "recipe": outcome.recipe,
            "started_at": outcome.started_at,
            "finished_at": outcome.finished_at,
            "elapsed_seconds": outcome.elapsed_seconds,
            "training_seconds": outcome.training_seconds,
            "final_loss": outcome.final_loss,
            "device": outcome.environment.get("device"),
            "artifact_files": outcome.artifact_files,
            "metrics": outcome.metrics,
            "checks": checks,
            **_mip_fields(outcome),
            **({"fast_eval": True}
               if trial_params.get("fast_eval") else {}),
        },
        # --- D. capacity (7.2 D) --------------------------------------
        "capacity": outcome.capacity,
        # Entry check record (2026-11-19): the pre-training estimate
        # at the cheapest profile, and whether the user overrode a
        # would-expand verdict with --allow-capacity-expansion.
        "capacity_check": capacity_check,
        # --- E. format and reproducibility (7.2 E) --------------------
        "format": {
            "format": "VNM1",
            "version": 1,
            "profile": outcome.profile,
            "cli_version": __version__,
            "core": core_provenance(),
            "input_config_hash": input_config_hash(
                spec, teacher, trial_params),
            "roundtrip_verification": outcome.roundtrip,
        },
        # --- F. environment and compatibility (7.2 F) -----------------
        "environment": outcome.environment,
        "compatibility": {
            "cpu_reference_decoder":
                "verified (read-back decode in this run)",
            "gpu_decoder": gpu_note,
        },
    }


def build_auto_report(spec: InputSpec, teacher: TeacherResult,
                      result, *, state_path: str,
                      capacity_check: dict | None = None) -> dict:
    """Report for an auto-mode run (design doc 7.2 A-F, step C).

    ``result`` is a :class:`vnm1.auto.AutoResult`; its trial records
    are the same dicts the state file stores, so this report can be
    rebuilt identically after a resume.
    """
    chosen = result.selected or result.representative
    chosen_index = next(
        (i for i, record in enumerate(result.trials)
         if record is chosen), None)
    checks = chosen["checks"] if chosen else {}
    failed_checks = [name for name, c in checks.items()
                     if c["gate_applicable"] and not c["passed"]]
    exempt = [name for name, c in checks.items()
              if not c["gate_applicable"]]
    gpu_note = (_GPU_NOTE_8CH
                if chosen and chosen["profile"] in _PROFILES_8CH
                else _GPU_NOTE_4CH)
    chosen_capacity = chosen["capacity"] if chosen else None
    cap_warnings = capacity_warnings(
        chosen_capacity.get("ratio_vs_bcn_mip_chain")
        if chosen_capacity else None)
    adjustment = dict(result.adjustment or {})
    if adjustment.get("reason") in ("color_not_narrow",
                                    "scalar_not_narrow",
                                    "normal_not_passing"):
        # Section 6.2.2: a run whose quality8 failure never qualifies
        # for adjustment records that it was outside the adjustment
        # scope (a ladder pass needs no such note).
        adjustment.setdefault("note",
                              "outside auto-adjustment scope "
                              "(design doc 6.2.2)")

    return {
        "report_version": 1,
        "cli_version": __version__,
        "material": spec.name,
        # --- A. conclusion (7.2 A) -----------------------------------
        "conclusion": {
            "result": result.status,
            "gate_passed": result.status == "passed",
            "mode": "auto",
            "selected_profile":
                result.selected["profile"] if result.selected else None,
            "representative_profile":
                result.representative["profile"]
                if result.representative else None,
            "representative_is_candidate_only":
                result.selected is None and result.representative
                is not None,
            "chosen_trial_index": chosen_index,
            "failed_checks": failed_checks,
            "gate_exempt_checks": exempt,
            "adjustment": adjustment,
            "artifact_location":
                chosen["artifact_location"] if chosen else None,
            "compatibility_note": gpu_note,
            "capacity_warnings": cap_warnings,
            **_conclusion_warnings(teacher),
        },
        # --- B. input record (7.2 B) ---------------------------------
        "input": _input_record(spec, teacher),
        # --- C. trial records (7.2 C) --------------------------------
        "trials": result.trials,
        # --- D. capacity (7.2 D), of the chosen artifact -------------
        "capacity": chosen["capacity"] if chosen else None,
        # Entry check record (2026-11-19), as in build_report.
        "capacity_check": capacity_check,
        # --- E. format and reproducibility (7.2 E) -------------------
        "format": {
            "format": "VNM1",
            "version": 1,
            "profile": chosen["profile"] if chosen else None,
            "cli_version": __version__,
            "core": core_provenance(),
            "input_config_hash": result.resume_hash,
            "auto_state": state_path,
            "roundtrip_verification":
                chosen["roundtrip_verification"] if chosen else None,
        },
        # --- F. environment and compatibility (7.2 F) ----------------
        "environment": chosen["environment"] if chosen else None,
        "compatibility": {
            "cpu_reference_decoder":
                "verified (read-back decode in this run)",
            "gpu_decoder": gpu_note,
        },
    }


__all__ = ["build_auto_report", "build_report", "core_provenance",
           "input_config_hash", "trial_record"]

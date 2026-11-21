"""Command line entry point for the vnm1 learning CLI.

Design doc section 3: subcommands ``train``, ``preflight``, ``inspect``
and exit codes 0/2/3/4/5.  Implementation step A provided the argument
parser, the full ``preflight`` (input validation only, section 4) and
stubs.  Step B wires ``train`` in manual mode end to end (one trial:
train -> export -> read back -> evaluate -> gate verdict -> minimal
report); step C adds ``train`` auto mode (the section 6.2 ladder, one
adjustment trial, state file and resume, in :mod:`vnm1.auto`).
Step D completes the outputs: every ``train`` run also renders the
section 7.4 visual sheets (with the float-buffer PSNR recheck) and
the section 7.3 ``report.md`` summary, and ``inspect`` (report
summary + artifact format verification, in :mod:`vnm1.inspect`) is
implemented.  :data:`EXIT_NOT_IMPLEMENTED` remains only as the
fallback for subcommands without a handler.

The 2026-11-19 capacity addition wires the two-stage capacity
discipline of :mod:`vnm1.capacity_check` into ``preflight`` and
``train`` (both modes): an entry check that rejects inputs even the
cheapest profile cannot make smaller than BCn (exit 2, overridable
with ``--allow-capacity-expansion`` and recorded in the report),
and outcome warnings on the finished report.

The 2026-11-21 multi-mip addition adds the opt-in ``--mip-train``
flag (batch5 proposal 1, in :mod:`vnm1.driver`): every trial of
either mode then trains on the SPEC 5.3 teacher mip pyramid and the
manifest records ``mipMax``.  Runs without the flag also measure
the finished artifact with the multi-mip ruler, and a far-field
advisory recommending the flag is printed and recorded when the
measured degradation crosses the advisory thresholds -- advisory
only, never part of the gate verdict or the exit code.

The 2026-11-21 Neon O4/O5 fixes add the opt-in ``--fast-eval``
flag (skip the multi-mip measurement and the visual sheets; the
gate evaluation is unchanged) and carry the input layer's warnings
to the train conclusion: a packed-map suspicion turns the result
line into ``PASS WITH WARNINGS: packed-map suspicion`` (verdict
and exit code unchanged), and image files in the input folder
that no map consumed are stated as excluded as unsupported, in
preflight, in train and in the report.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

from . import __version__
from .errors import (ExportVerificationError, InputValidationError,
                     TrainingError)
from .inputs import LOGICAL_MAPS, resolve_inputs

#: Exit code for subcommands / modes that are not implemented yet.
#: Distinct from the design doc section 3.3 codes (0, 2, 3, 4, 5).
EXIT_NOT_IMPLEMENTED = 69

#: Exit codes (design doc section 3.3).
EXIT_OK = 0
EXIT_INPUT_ERROR = 2
EXIT_GATE_FAILED = 3
EXIT_TRAINING_ERROR = 4
EXIT_EXPORT_ERROR = 5

#: Manual-mode defaults (design doc sections 5.1 and 6.1).
DEFAULT_PROFILE = "standard"
DEFAULT_ITERATIONS = 50000
DEFAULT_COLOR_WEIGHT = 4.0
DEFAULT_SCALAR_WEIGHT = 1.0
DEFAULT_SEED = 1337
DEFAULT_OUTPUT = "dist"

_FINE_PATTERN_NOTICE = (
    "Note: VNM1 can struggle with the fine detail of densely packed "
    "patterns (fine leaves, grass, needles, mesh, small text). Even "
    "with the largest profile and adjustment, such textures may not "
    "meet the quality gate. For important materials, check the visual "
    "sheet and keep the conventional texture as well. (design doc 11.2)"
)


def _add_input_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("input", nargs="?", default=None,
                        help="material input folder")
    parser.add_argument("--config", default=None,
                        help="JSON config file (design doc 4.2 C)")
    parser.add_argument("--name", default=None,
                        help="material name (default: input folder name)")
    for logical in LOGICAL_MAPS:
        parser.add_argument(
            f"--{logical.replace('_', '-')}", default=None,
            metavar="PATH",
            help=f"explicit path for the {logical} map (overrides the "
                 "standard file name and the config file)")
    parser.add_argument(
        "--normal-convention", choices=("opengl", "directx"), default=None,
        help="normal map convention (required; may also come from the "
             "config file)")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="vnm1",
        description="Convert PBR texture sets to the VNM1 neural "
                    "material format (productization P1).")
    parser.add_argument("--version", action="version",
                        version=f"vnm1 {__version__}")
    sub = parser.add_subparsers(dest="command", required=True)

    p_train = sub.add_parser(
        "train", help="validate, train, evaluate and export one material")
    _add_input_arguments(p_train)
    p_train.add_argument("--output", default=None, help="output folder")
    p_train.add_argument("--mode", choices=("auto", "manual"),
                         default="auto")
    p_train.add_argument("--profile",
                         choices=("standard", "standard8", "quality",
                                  "quality8"),
                         default=None)
    p_train.add_argument("--iterations", type=int, default=None)
    p_train.add_argument("--color-weight", type=float, default=None)
    p_train.add_argument("--scalar-weight", type=float, default=None)
    p_train.add_argument("--seed", type=int, default=None)
    p_train.add_argument("--device", default=None)
    p_train.add_argument("--allow-unverified-size", action="store_true",
                         help="allow sizes outside the verified "
                              "2048x2048 baseline (design doc 4.5)")
    p_train.add_argument("--allow-capacity-expansion",
                         action="store_true",
                         help="convert even when even the cheapest "
                              "profile is estimated to be no smaller "
                              "than BCn (capacity entry check, "
                              "2026-11-19); the override is recorded "
                              "in the report")
    p_train.add_argument("--mip-train", action="store_true",
                         help="train on the SPEC 5.3 teacher mip "
                              "pyramid (multi-mip training, batch5 "
                              "proposal 1, 2026-11-21): half the "
                              "samples come from mip 0, half from "
                              "mips 1-6, with normalizedMip = m/6, "
                              "and the manifest records the optional "
                              "mipMax field. Measured on 3 materials: "
                              "far-mip PSNR +7.7..+21.5 dB vs "
                              "mip-0-only training, mip-0 cost "
                              "-1.0..-2.2 dB at the same recipe. "
                              "Default off; the format is unchanged")
    p_train.add_argument("--fast-eval", action="store_true",
                         help="lighten the post-training evaluation "
                              "(Neon O4, 2026-11-21): skip the "
                              "multi-mip measurement (and with it the "
                              "far-field advisory) and the visual "
                              "sheets. The gate evaluation, the "
                              "artifact and the verdict are unchanged; "
                              "the report records that the lighter "
                              "evaluation was used")
    p_train.add_argument("--keep-work", action="store_true",
                         help="keep checkpoint files under work/ after "
                              "the run (default: checkpoints are removed, "
                              "logs and attempts are kept)")
    p_train.set_defaults(func=_cmd_train, subcommand="train")

    p_pre = sub.add_parser(
        "preflight",
        help="validate inputs only; no training (design doc 4, 11.2)")
    _add_input_arguments(p_pre)
    p_pre.add_argument("--material-kind",
                       choices=("general", "foliage", "fine-pattern"),
                       default="general",
                       help="material kind hint; foliage / fine-pattern "
                            "strengthens the fine-detail warning")
    p_pre.add_argument("--allow-capacity-expansion",
                       action="store_true",
                       help="report the capacity entry check verdict "
                            "without failing on it (2026-11-19)")
    p_pre.set_defaults(func=_cmd_preflight)

    p_inspect = sub.add_parser(
        "inspect", help="summarise a report / verify an artifact")
    p_inspect.add_argument("target", help="report or artifact path")
    p_inspect.set_defaults(func=_cmd_inspect, subcommand="inspect")

    return parser


def _explicit_maps_from_args(args: argparse.Namespace) -> dict:
    return {logical: getattr(args, logical, None) for logical in LOGICAL_MAPS}


def _cmd_not_implemented(args: argparse.Namespace) -> int:
    print(f"vnm1 {args.subcommand}: not implemented yet "
          f"(implementation step A provides preflight only).",
          file=sys.stderr)
    return EXIT_NOT_IMPLEMENTED


def _cmd_inspect(args: argparse.Namespace) -> int:
    """inspect (design doc 3.1, step D): report summary + artifact
    format verification.  Exit codes reuse the section 3.3 meanings
    (see :mod:`vnm1.inspect`)."""
    from .inspect import inspect_target

    code, lines = inspect_target(Path(args.target))
    for line in lines:
        print(line)
    return code


def _finish_run_reports(*, spec, teacher, report, material_out,
                        artifact_dir, expected_psnr_db, progress,
                        fast_eval: bool = False):
    """Step D outputs shared by both train modes: render the section
    7.4 visual sheets for the placed artifact (verifying the sheet
    decode's float-buffer PSNR against the report), attach the
    ``visual_check`` record, then write report.json and report.md.

    Raises ExportVerificationError when the sheet PSNR recheck or a
    sheet write fails (the CLI reports it as exit 5).

    With ``fast_eval`` (Neon O4, 2026-11-21) the sheets -- a second
    full CPU decode of the artifact plus the PNG rendering -- are
    skipped and the ``visual_check`` record states that plainly.
    """
    from . import visual as visual_mod
    from .report_md import render_report_markdown

    if fast_eval:
        visual_info = {
            "files": {},
            "fast_eval": True,
            "skipped": True,
            "note": "Visual sheets were skipped (--fast-eval): no "
                    "sheet decode was run, so there is no sheet "
                    "PSNR recheck for this run. The gate metrics "
                    "come from the read-back evaluation in the "
                    "training trial itself.",
        }
        progress("visual_sheets_skipped=fast_eval")
    else:
        visual_info = visual_mod.generate_visual_sheets(
            Path(artifact_dir), teacher, spec.name, material_out,
            expected_psnr_db=expected_psnr_db)
        progress("visual_sheets_written="
                 + ",".join(sorted(visual_info["files"])))
    report["visual_check"] = visual_info
    report_path = material_out / f"{spec.name}.report.json"
    report_path.write_text(json.dumps(report, indent=2),
                           encoding="utf-8")
    progress(f"report_written={report_path}")
    md_path = material_out / f"{spec.name}.report.md"
    md_path.write_text(render_report_markdown(report),
                       encoding="utf-8")
    progress(f"report_md_written={md_path}")
    return report_path, md_path, visual_info


def _capacity_entry_gate(spec, teacher, *, allow: bool):
    """Capacity entry check (2026-11-19, :mod:`vnm1.capacity_check`).

    Returns ``(record, error)``: ``record`` is the check result plus
    the ``override`` flag, for the report; ``error`` is a rejection
    message when even the cheapest profile cannot beat the BCn
    estimate and the user did not override, else ``None``.  Runs
    after the section 4 input validation (so an input that fails
    validation -- e.g. real alpha in the base color -- is still
    rejected for that reason, never for capacity).
    """
    from .capacity_check import capacity_entry_check

    channel_sources = {logical: teacher.records[logical].source
                       for logical in LOGICAL_MAPS}
    entry = capacity_entry_check(teacher.width, teacher.height,
                                 channel_sources, spec.name)
    record = dict(entry)
    record["override"] = bool(allow and entry["would_expand"])
    if entry["would_expand"] and not allow:
        return record, (
            f"capacity entry check: even the cheapest profile "
            f"({entry['entry_profile']}) is estimated at "
            f"{entry['estimated_vnm1_total_bytes']:,} bytes, which "
            f"does not beat the BCn full mip chain estimate of "
            f"{entry['bcn_mip_chain_bytes']:,} bytes for this input "
            f"(ratio {entry['estimated_ratio_vs_bcn_mip_chain']:.3f}); "
            f"converting it cannot save capacity at any profile. "
            f"Pass --allow-capacity-expansion to convert anyway.")
    return record, None


def _print_mip_notes(*, mip_training, mip_evaluation,
                     mip_advisory) -> None:
    """Console summary of a run's multi-mip records (2026-11-21):
    the per-mip PSNR line for a --mip-train run, and the far-field
    advisory for a mip-0-trained artifact whose measured far-field
    degradation crossed the advisory thresholds.  Advisory only;
    printed after the gate result and never changes it."""
    if mip_training is not None and mip_evaluation is not None:
        per_mip = mip_evaluation["per_mip"]
        parts = ", ".join(
            f"mip {level}: {per_mip[str(level)]['psnr_db']:.2f} dB"
            for level in mip_evaluation["levels"])
        print(f"Mip training: on (multi-mip; mipMax "
              f"{mip_evaluation['mip_max']} recorded in the "
              f"manifest)")
        print(f"Multi-mip PSNR (linear): {parts}")
    if mip_advisory is not None:
        psnr = mip_advisory["psnr_db"]
        print(f"Far-field advisory: this material degrades sharply "
              f"at distance when trained at mip 0 only (base-color "
              f"PSNR {psnr['0']:.2f} dB at mip 0, {psnr['2']:.2f} "
              f"dB at mip 2, {psnr['4']:.2f} dB at mip 4). "
              f"Re-running with --mip-train is recommended. "
              f"(Advisory only; the gate result is unaffected.)")


def _print_capacity_warnings(capacity: dict) -> None:
    from .capacity_check import capacity_warnings

    for warning in capacity_warnings(
            capacity.get("ratio_vs_bcn_mip_chain")):
        print(f"Capacity warning: {warning['message']}")


def _visual_line(visual_info: dict) -> str:
    """The console line naming the visual sheets of a run (or the
    fast-eval skip, Neon O4)."""
    if visual_info.get("skipped"):
        return "Visual sheets: skipped (--fast-eval)"
    return ("Visual sheets: "
            + ", ".join(sorted(visual_info["files"])))


def _packed_conclusion_suffix(teacher) -> str:
    """The conclusion-line marker for a packed-map suspicion
    (Neon O5, 2026-11-21): empty when no scalar map triggered the
    suspicion, else the ``WITH WARNINGS`` suffix the train result
    line carries.  The gate verdict and the exit code are unaffected;
    what changes is that a passing run can no longer pass silently
    while grading what may be the wrong channel."""
    from .imaging import packed_suspicion_warnings

    if not packed_suspicion_warnings(teacher.warnings):
        return ""
    return (" WITH WARNINGS: packed-map suspicion — a scalar map's "
            "RGB channels differ, so a packed map (e.g. ORM/ARM) may "
            "have been supplied by mistake; the gate graded the "
            "first channel only (see the report)")


def _cmd_preflight(args: argparse.Namespace) -> int:
    # Imported here so that `vnm1 --help` and the stubs work even in an
    # environment without numpy/Pillow installed.
    from .imaging import build_teacher

    try:
        spec = resolve_inputs(
            args.input,
            name=args.name,
            explicit_maps=_explicit_maps_from_args(args),
            normal_convention=args.normal_convention,
            config_path=args.config,
        )
        result = build_teacher(spec)
    except InputValidationError as exc:
        print(f"preflight: input validation error: {exc}", file=sys.stderr)
        return EXIT_INPUT_ERROR

    entry, entry_error = _capacity_entry_gate(
        spec, result, allow=args.allow_capacity_expansion)
    if entry_error is not None:
        print(f"preflight: input validation error: {entry_error}",
              file=sys.stderr)
        return EXIT_INPUT_ERROR

    print(f"Material: {spec.name}")
    print(f"Size: {result.width}x{result.height}")
    if result.size_verified:
        print("Size status: verified baseline (2048x2048)")
    else:
        print("Size status: OUTSIDE the verified range of the judgment "
              "run (2048x2048); the capability table does not directly "
              "apply (design doc 4.5)")
    print(f"Normal convention: {spec.normal_convention}")
    print("Maps:")
    for logical in LOGICAL_MAPS:
        record = result.records[logical]
        if record.source == "map":
            print(f"  {logical}: map ({Path(record.file).name}, "
                  f"{record.width}x{record.height}, "
                  f"{record.bit_depth} bit, mode {record.pil_mode})")
        else:
            print(f"  {logical}: constant {record.constant_value} "
                  "(gate exempt, not recovered data)")
    for warning in result.warnings:
        print(f"Warning: {warning}")
    print(f"Capacity entry check (cheapest profile "
          f"{entry['entry_profile']}): estimated VNM1 "
          f"{entry['estimated_vnm1_total_bytes']:,} bytes vs BCn "
          f"full mip chain {entry['bcn_mip_chain_bytes']:,} bytes "
          f"(ratio "
          f"{entry['estimated_ratio_vs_bcn_mip_chain']:.3f})")
    if entry["override"]:
        print("Note: the entry check verdict was overridden by "
              "--allow-capacity-expansion; this conversion cannot "
              "save capacity.")
    if args.material_kind in ("foliage", "fine-pattern"):
        print("Warning (material kind): fine-detail materials are a "
              "known weak case for VNM1; see the note below.")
    print(_FINE_PATTERN_NOTICE)
    return 0


def _print_env_diagnostics(device_request: str):
    """Print the section 8.1 environment block (without the per-mode
    Material line) and resolve the device.

    Returns ``(status, resolved_device)``, or ``None`` when torch is
    missing (the caller then exits with EXIT_TRAINING_ERROR).
    """
    from . import driver

    status = driver.torch_status()
    print(f"Python: {status['python_version']} "
          f"({status['python_implementation']}, {status['os']})")
    print(f"PyTorch: {status['torch_version'] or 'not installed'}")
    print(f"CUDA available: {status['cuda_available']}"
          + (f" (GPU: {status['gpu_name']})" if status["gpu_name"]
             else ""))
    resolved_device = device_request
    if resolved_device == "auto":
        resolved_device = "cuda" if status["cuda_available"] else "cpu"
    print(f"Device: {resolved_device} (requested: {device_request})")
    if resolved_device == "cpu":
        print("CPU fallback: yes. Note: CPU training time at "
              "production iteration counts is unmeasured (design doc "
              "8.3); CPU is a checking environment, not the standard "
              "conversion environment.")
    if not status["torch_available"]:
        print("train: torch is not installed; install the 'train' "
              "extra to run training (design doc 8.1).",
              file=sys.stderr)
        return None
    return status, resolved_device


def _run_auto_mode(args, spec, teacher, *, output_dir, device_request,
                   seed, material_out, progress, channel_sources,
                   log_fh, capacity_check_record=None) -> int:
    """Auto mode (design doc 6.2): plan display, ladder + adjustment
    via :mod:`vnm1.auto`, auto report, exit code 0 / 3 / 4 / 5."""
    from . import auto as auto_mod
    from . import driver, report as report_mod

    print(auto_mod.format_auto_plan(spec, teacher, output_dir,
                                    seed=seed,
                                    mip_train=args.mip_train,
                                    fast_eval=args.fast_eval))
    print(_FINE_PATTERN_NOTICE)

    def trial_runner(trial_def, trial_work_dir):
        options = driver.TrialOptions(
            material_name=spec.name, profile=trial_def["profile"],
            iterations=trial_def["iterations"],
            color_weight=trial_def["color_weight"],
            scalar_weight=trial_def["scalar_weight"],
            seed=trial_def["seed"], device=device_request,
            work_dir=trial_work_dir, mip_train=args.mip_train,
            fast_eval=args.fast_eval)
        return driver.run_trial(teacher.teacher, channel_sources,
                                options, progress_cb=progress)

    try:
        result = auto_mod.run_auto(
            spec=spec, teacher=teacher, output_dir=output_dir,
            seed=seed, trial_runner=trial_runner, progress_cb=progress,
            keep_work=args.keep_work, mip_train=args.mip_train,
            fast_eval=args.fast_eval)
    except TrainingError as exc:
        print(f"train: training error: {exc}", file=sys.stderr)
        log_fh.close()
        return EXIT_TRAINING_ERROR
    except ExportVerificationError as exc:
        print(f"train: export/verification error: {exc}",
              file=sys.stderr)
        log_fh.close()
        return EXIT_EXPORT_ERROR

    report = report_mod.build_auto_report(
        spec, teacher, result, state_path=str(result.state_path),
        capacity_check=capacity_check_record)
    chosen = result.selected or result.representative
    try:
        report_path, md_path, visual_info = _finish_run_reports(
            spec=spec, teacher=teacher, report=report,
            material_out=material_out,
            artifact_dir=chosen["artifact_location"],
            expected_psnr_db=chosen["metrics"]
            ["psnr_base_color_db"],
            progress=progress, fast_eval=args.fast_eval)
    except ExportVerificationError as exc:
        print(f"train: export/verification error: {exc}",
              file=sys.stderr)
        log_fh.close()
        return EXIT_EXPORT_ERROR
    log_fh.close()
    warn_suffix = _packed_conclusion_suffix(teacher)
    if result.status == "passed":
        print(f"Result: PASS{warn_suffix} "
              f"(profile {result.selected['profile']}"
              + (", adjustment trial"
                 if result.selected["kind"] == "adjustment" else "")
              + f", {len(result.trials)} trial(s) run"
              + (f", {result.resumed_trials} resumed from state"
                 if result.resumed_trials else "")
              + ")")
    else:
        print(f"Result: FAIL{warn_suffix} (no configuration passed; "
              f"representative candidate: "
              f"{result.representative['profile']}, kept "
              f"under work/attempts/)")
        adj = result.adjustment or {}
        if not adj.get("ran"):
            print(f"Adjustment: not run ({adj.get('reason', 'n/a')}"
                  + (f": {adj['detail']}" if adj.get("detail") else "")
                  + ")")
    if chosen is not None:
        print(f"Base Color PSNR (linear): "
              f"{chosen['metrics']['psnr_base_color_db']:.2f} dB")
        cap = chosen["capacity"]
        print(f"Capacity: {cap['vnm1_total_bytes']} bytes vs BCn mip "
              f"chain {cap['bcn']['total_mip_chain_bytes']} bytes "
              f"(ratio {cap['ratio_vs_bcn_mip_chain']:.4f})")
        _print_capacity_warnings(cap)
        _print_mip_notes(
            mip_training=chosen.get("mip_training"),
            mip_evaluation=chosen.get("mip_evaluation"),
            mip_advisory=chosen.get("mip_advisory"))
        print(f"Artifact: {chosen['artifact_location']}")
    print(_visual_line(visual_info))
    print(f"Report: {report_path}")
    print(f"Report (Markdown): {md_path}")
    print(f"State: {result.state_path}")
    return EXIT_OK if result.status == "passed" else EXIT_GATE_FAILED


def _cmd_train(args: argparse.Namespace) -> int:
    # Lazy imports: `vnm1 --help` and the stubs must work without
    # numpy/Pillow/torch installed (same pattern as _cmd_preflight).
    from . import auto as auto_mod
    from . import driver, report as report_mod
    from .imaging import build_teacher

    # ---- input resolution + validation (exit 2) ----------------------
    try:
        spec = resolve_inputs(
            args.input,
            name=args.name,
            explicit_maps=_explicit_maps_from_args(args),
            normal_convention=args.normal_convention,
            config_path=args.config,
        )
        teacher = build_teacher(spec)
    except InputValidationError as exc:
        print(f"train: input validation error: {exc}", file=sys.stderr)
        return EXIT_INPUT_ERROR

    profile = args.profile or DEFAULT_PROFILE
    iterations = args.iterations
    if iterations is None:
        iterations = spec.extra_config.get("iterations",
                                           DEFAULT_ITERATIONS)
    color_weight = args.color_weight
    if color_weight is None:
        color_weight = DEFAULT_COLOR_WEIGHT
    scalar_weight = args.scalar_weight
    if scalar_weight is None:
        scalar_weight = DEFAULT_SCALAR_WEIGHT
    seed = args.seed
    if seed is None:
        seed = int(spec.extra_config.get("seed", DEFAULT_SEED))
    output_raw = args.output or spec.extra_config.get("output_dir") \
        or DEFAULT_OUTPUT
    output_dir = Path(output_raw).expanduser().resolve()
    device_request = args.device or "auto"

    if iterations < 1:
        print("train: input validation error: --iterations must be "
              ">= 1", file=sys.stderr)
        return EXIT_INPUT_ERROR
    if color_weight <= 0 or scalar_weight <= 0:
        print("train: input validation error: loss weights must be "
              "positive", file=sys.stderr)
        return EXIT_INPUT_ERROR
    if device_request not in ("auto", "cpu", "cuda"):
        print(f"train: input validation error: --device must be one "
              f"of auto/cpu/cuda, got {device_request!r}",
              file=sys.stderr)
        return EXIT_INPUT_ERROR
    if args.mode == "auto":
        # Section 6.2 fixes the trial conditions; the manual knobs are
        # rejected outright instead of being silently ignored.
        conflicts = [flag for flag, value in (
            ("--profile", args.profile),
            ("--iterations", args.iterations),
            ("--color-weight", args.color_weight),
            ("--scalar-weight", args.scalar_weight),
        ) if value is not None]
        if conflicts:
            print("train: input validation error: auto mode fixes the "
                  "trial conditions (design doc 6.2); "
                  f"{', '.join(conflicts)} cannot be combined with "
                  "--mode auto (use --mode manual for a single "
                  "configured trial)", file=sys.stderr)
            return EXIT_INPUT_ERROR
    if not teacher.size_verified and not args.allow_unverified_size:
        print(f"train: input validation error: input size "
              f"{teacher.width}x{teacher.height} is outside the "
              f"verified baseline (2048x2048, design doc 4.5); pass "
              f"--allow-unverified-size to train anyway",
              file=sys.stderr)
        return EXIT_INPUT_ERROR
    if not teacher.size_verified:
        print("Warning: input size is outside the judgment run's "
              "verified range; the capability table does not directly "
              "apply (design doc 4.5).")
    capacity_check_record, entry_error = _capacity_entry_gate(
        spec, teacher, allow=args.allow_capacity_expansion)
    if entry_error is not None:
        print(f"train: input validation error: {entry_error}",
              file=sys.stderr)
        return EXIT_INPUT_ERROR
    if capacity_check_record["override"]:
        print("Note: the capacity entry check was overridden by "
              "--allow-capacity-expansion; this conversion cannot "
              "save capacity. The override is recorded in the "
              "report.")
    # Input-layer warnings are part of the run's visible record
    # (Neon O5, 2026-11-21): packed-map suspicions and unsupported
    # files excluded from the input folder are stated here, not
    # only inside the report file.
    for warning in teacher.warnings:
        print(f"Warning: {warning}")

    # ---- environment diagnostics (design doc 8.1) --------------------
    if args.mode == "auto":
        print(f"Material: {spec.name} "
              f"({teacher.width}x{teacher.height}, mode auto)")
    else:
        print(f"Material: {spec.name} "
              f"({teacher.width}x{teacher.height}, profile {profile}, "
              f"{iterations} iterations)")
    if _print_env_diagnostics(device_request) is None:
        return EXIT_TRAINING_ERROR

    # ---- shared run setup (log file + progress sink) -----------------
    material_out = output_dir / spec.name
    work_dir = material_out / "work"
    logs_dir = work_dir / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)
    log_path = logs_dir / "train.log"
    log_fh = open(log_path, "a", encoding="utf-8")

    def progress(msg: str) -> None:
        line = f"{datetime.now(timezone.utc).isoformat()} {msg}"
        print(line, flush=True)
        log_fh.write(line + "\n")
        log_fh.flush()

    channel_sources = {logical: teacher.records[logical].source
                       for logical in LOGICAL_MAPS}

    # ---- auto mode (design doc 6.2) ----------------------------------
    if args.mode == "auto":
        return _run_auto_mode(
            args, spec, teacher, output_dir=output_dir,
            device_request=device_request, seed=seed,
            material_out=material_out, progress=progress,
            channel_sources=channel_sources, log_fh=log_fh,
            capacity_check_record=capacity_check_record)

    # ---- one manual trial (design doc 6.1) ---------------------------
    options = driver.TrialOptions(
        material_name=spec.name, profile=profile,
        iterations=int(iterations), color_weight=float(color_weight),
        scalar_weight=float(scalar_weight), seed=int(seed),
        device=device_request, work_dir=work_dir,
        mip_train=args.mip_train, fast_eval=args.fast_eval)
    trial_params = {"profile": profile, "iterations": int(iterations),
                    "color_weight": float(color_weight),
                    "scalar_weight": float(scalar_weight),
                    "seed": int(seed)}
    if args.mip_train:
        # Recorded (and covered by the input/config hash) only when
        # enabled, so a mip-0 run's records are unchanged.
        trial_params["mip_train"] = True
    if args.fast_eval:
        # Same rule as mip_train (Neon O4): recorded only when
        # enabled, so a full-evaluation run's records are unchanged.
        trial_params["fast_eval"] = True
    try:
        outcome = driver.run_trial(teacher.teacher, channel_sources,
                                   options, progress_cb=progress)
    except TrainingError as exc:
        print(f"train: training error: {exc}", file=sys.stderr)
        log_fh.close()
        return EXIT_TRAINING_ERROR
    except ExportVerificationError as exc:
        print(f"train: export/verification error: {exc}",
              file=sys.stderr)
        log_fh.close()
        return EXIT_EXPORT_ERROR

    # ---- placement (design doc 4.6: dist gets passes only) -----------
    staging = outcome.artifact_dir
    if outcome.gate_passed:
        dest = material_out / f"{spec.name}.vnm"
        if dest.exists():
            stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
            previous = work_dir / "attempts" / f"previous-pass-{stamp}"
            previous.parent.mkdir(parents=True, exist_ok=True)
            os.rename(dest, previous)
            progress(f"previous_pass_archived={previous}")
        os.rename(staging, dest)
        artifact_location = str(dest)
    else:
        dest = work_dir / "attempts" / f"trial-001-{profile}"
        dest.parent.mkdir(parents=True, exist_ok=True)
        if dest.exists():
            stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
            dest = work_dir / "attempts" / \
                f"trial-001-{profile}-{stamp}"
        os.rename(staging, dest)
        artifact_location = str(dest)
    progress(f"artifact_placed={artifact_location}")

    # ---- reports + visual sheets (design doc 7.2, 7.3, 7.4) ---------
    report = report_mod.build_report(
        spec, teacher, outcome, mode="manual",
        artifact_location=artifact_location, trial_params=trial_params,
        capacity_check=capacity_check_record)
    try:
        report_path, md_path, visual_info = _finish_run_reports(
            spec=spec, teacher=teacher, report=report,
            material_out=material_out, artifact_dir=artifact_location,
            expected_psnr_db=outcome.metrics["psnr_base_color_db"],
            progress=progress, fast_eval=args.fast_eval)
    except ExportVerificationError as exc:
        print(f"train: export/verification error: {exc}",
              file=sys.stderr)
        log_fh.close()
        return EXIT_EXPORT_ERROR

    if not args.keep_work:
        for ckpt in (work_dir / "checkpoints").glob("*.pt"):
            ckpt.unlink()
    log_fh.close()

    cap = outcome.capacity
    print(f"Result: {'PASS' if outcome.gate_passed else 'FAIL'}"
          f"{_packed_conclusion_suffix(teacher)} "
          f"(profile {profile})")
    print(f"Base Color PSNR (linear): "
          f"{outcome.metrics['psnr_base_color_db']:.2f} dB")
    print(f"Capacity: {cap['vnm1_total_bytes']} bytes vs BCn mip "
          f"chain {cap['bcn']['total_mip_chain_bytes']} bytes "
          f"(ratio {cap['ratio_vs_bcn_mip_chain']:.4f})")
    _print_capacity_warnings(cap)
    _print_mip_notes(mip_training=outcome.mip_training,
                     mip_evaluation=outcome.mip_evaluation,
                     mip_advisory=outcome.mip_advisory)
    print(f"Artifact: {artifact_location}")
    print(_visual_line(visual_info))
    print(f"Report: {report_path}")
    print(f"Report (Markdown): {md_path}")
    return EXIT_OK if outcome.gate_passed else EXIT_GATE_FAILED


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())

"""Human-readable Markdown summary (``report.md``) from a report dict.

Implements design doc 7.3 on top of the section 7.2 JSON report built
by :mod:`vnm1.report` (manual mode: ``report["trial"]``; auto mode:
``report["trials"]``).  The summary is written so a developer can
decide in a short read whether to use the VNM1 artifact:

- the verdict comes first; on failure the failing items and their
  distance from the gate line are shown before anything else
  (section 7.3), and the candidate is never worded as a recommended
  artifact (section 11.3);
- gate-exempt (constant) channels are shown as constants, never as
  passes -- the information was not recovered (section 7.1);
- the GPU-decoder compatibility note is carried verbatim from the
  JSON report (section 7.2 F): the GPU decoder supports all four
  profiles, verified on desktop WebGPU and Pixel 9a (P2, 2026-11);
- known limitations (sections 11.1 and 11.4) are stated in measured
  terms.  Nothing in this module upgrades a result: wording follows
  the recorded numbers only.

Pure text generation from the report dict -- no torch, no file IO --
so it is unit-testable anywhere.
"""

from __future__ import annotations

#: Check presentation: key -> (label, unit, kind).  kind "min" is a
#: >= gate (PSNR), kind "max" is a <= gate (MAE, angles).  The stored
#: ``difference`` is value - threshold: negative means shortfall for
#: "min", positive means excess for "max" (design doc 7.2 C).
_CHECK_PRESENTATION = {
    "psnr_base_color_db": ("Base Color PSNR (linear)", "dB", "min"),
    "normal_angle_mean_deg": ("Normal angle error, mean", "°", "max"),
    "normal_angle_p95_deg": ("Normal angle error, p95", "°", "max"),
    "mae_roughness": ("Roughness MAE", "", "max"),
    "mae_metalness": ("Metalness MAE", "", "max"),
    "mae_ao": ("AO MAE", "", "max"),
}

#: Which logical input map each check belongs to (for the constant
#: display of gate-exempt checks).
_CHECK_MAP = {
    "normal_angle_mean_deg": "normal",
    "normal_angle_p95_deg": "normal",
    "mae_roughness": "roughness",
    "mae_metalness": "metalness",
    "mae_ao": "ao",
}

#: Adjustment pattern names -> short human description (the weights
#: and iterations themselves are read from the adjustment trial
#: record, not restated here).
_PATTERN_LABEL = {
    "scalar_only": "scalar weights only",
    "color_only": "color weight and longer training",
    "color_and_scalar": "color + scalar weights and longer training",
}

#: Why no adjustment ran -> human wording (design doc 6.2.2 / 11.3).
_ADJUSTMENT_REASON = {
    "ladder_passed": "a ladder trial passed, so no adjustment was "
                     "needed",
    "gate_passed": "the quality8 trial passed, so no adjustment was "
                   "needed",
    "color_not_narrow": "the base-color miss was wider than the "
                        "narrow-miss range where adjustment has a "
                        "measured basis",
    "scalar_not_narrow": "a scalar miss was wider than the "
                         "narrow-miss range where adjustment has a "
                         "measured basis",
    "normal_not_passing": "a normal check was failing; raising the "
                          "color weight has worsened normals in the "
                          "judgment run, so auto adjustment does not "
                          "run in that case",
    "narrow_miss": "the miss was within the narrow-miss range",
}


def _fmt_seconds(seconds: float) -> str:
    seconds = float(seconds)
    if seconds < 60:
        return f"{seconds:.1f} sec"
    minutes = int(seconds // 60)
    rest = int(round(seconds - minutes * 60))
    if rest == 60:
        minutes += 1
        rest = 0
    return f"{minutes} min {rest} sec"


def _fmt_value(key: str, value: float) -> str:
    if key == "psnr_base_color_db":
        return f"{value:.2f} dB"
    if key.startswith("normal_angle"):
        return f"{value:.2f}°"
    return f"{value:.4f}"


def _fmt_difference(key: str, check: dict) -> str:
    diff = float(check["difference"])
    _label, unit, _kind = _CHECK_PRESENTATION[key]
    if key == "psnr_base_color_db":
        return f"{diff:+.2f} dB vs the {check['threshold']:.2f} dB line"
    if key.startswith("normal_angle"):
        return f"{diff:+.2f}° vs the {check['threshold']:.2f}° limit"
    return f"{diff:+.4f} vs the {check['threshold']:.4f} limit"


def _check_lines(checks: dict, report: dict) -> list[str]:
    """One line per gate check, in gate order (design doc 7.3)."""
    maps = report.get("input", {}).get("maps", {})
    lines = []
    for key, (label, _unit, _kind) in _CHECK_PRESENTATION.items():
        check = checks.get(key)
        if check is None:
            continue
        if not check["gate_applicable"]:
            logical = _CHECK_MAP.get(key)
            entry = maps.get(logical, {}) if logical else {}
            constant = entry.get("constant_value")
            if isinstance(constant, (list, tuple)):
                shown = "(" + ", ".join(f"{v:g}" for v in constant) \
                        + ") flat"
            elif constant is not None:
                shown = f"{constant:g}"
            else:
                shown = "n/a"
            lines.append(
                f"- {label}: constant {shown} — gate exempt "
                f"(no input map; not recovered data)")
        elif check["passed"]:
            lines.append(
                f"- {label}: {_fmt_value(key, check['value'])} — PASS")
        else:
            lines.append(
                f"- {label}: {_fmt_value(key, check['value'])} — FAIL "
                f"({_fmt_difference(key, check)})")
    return lines


def _failed_summary_lines(report: dict, checks: dict) -> list[str]:
    """The failure block shown at the top on a failed run (7.3/11.3)."""
    conclusion = report["conclusion"]
    failed = conclusion.get("failed_checks", [])
    lines = ["Gate: NOT MET", ""]
    if conclusion.get("representative_is_candidate_only"):
        lines.append(
            "No configuration passed the quality gate. The artifact "
            "named below is the furthest-along candidate only; it is "
            "kept under work/attempts/ and is not a passing artifact. "
            "Do not ship it as one.")
        lines.append("")
    lines.append("Failing items:")
    for key in failed:
        check = checks.get(key)
        if check is None:
            continue
        label = _CHECK_PRESENTATION[key][0]
        lines.append(
            f"- {label}: {_fmt_value(key, check['value'])} "
            f"({_fmt_difference(key, check)})")
    lines.append("")
    if failed == ["psnr_base_color_db"]:
        lines.append(
            "The normal and scalar conditions were met; the miss is "
            "in base-color detail only. This matches the pattern "
            "measured in the judgment run for densely packed fine "
            "detail (the leaves material), where colour reproduction "
            "is the confirmed limit of the current format. The same "
            "result is not guaranteed for other materials, and this "
            "candidate is not treated as a passing artifact.")
        lines.append("")
    return lines


def _adjustment_lines(report: dict) -> list[str]:
    conclusion = report["conclusion"]
    adjustment = conclusion.get("adjustment")
    if not adjustment:
        return []
    lines = []
    if adjustment.get("ran"):
        pattern = adjustment.get("pattern")
        label = _PATTERN_LABEL.get(pattern, pattern or "")
        trial = next((t for t in report.get("trials", [])
                      if t.get("kind") == "adjustment"), None)
        detail = ""
        if trial is not None:
            detail = (f" (color weight {trial['color_weight']:g}, "
                      f"scalar weight {trial['scalar_weight']:g}, "
                      f"{trial['iterations']:,} iterations, "
                      f"profile {trial['profile']})")
        lines.append(
            f"Adjustment: ran once — {label}{detail}. Result: "
            + ("passed" if trial and trial["gate_passed"]
               else "still below the gate"))
    else:
        reason = adjustment.get("reason", "")
        wording = _ADJUSTMENT_REASON.get(reason, reason)
        lines.append(f"Adjustment: not run — {wording}.")
        if adjustment.get("detail"):
            lines.append(f"Adjustment detail: {adjustment['detail']}")
    lines.append("")
    return lines


def _mip_lines(chosen: dict | None) -> list[str]:
    """Multi-mip section (2026-11-21): the batch4/5 ruler's mip
    0 / 2 / 4 measurements for the chosen artifact.

    For a --mip-train artifact: the per-mip table.  For a
    mip-0-trained artifact whose far field degrades past the
    advisory thresholds (driver.ADVISORY_*): the recommendation to
    re-run with --mip-train, with the measured numbers.  Advisory
    wording only -- nothing here upgrades or downgrades the
    verdict."""
    if not chosen:
        return []
    lines = []
    evaluation = chosen.get("mip_evaluation")
    if evaluation:
        per_mip = evaluation["per_mip"]
        lines.append("Multi-mip behaviour (trained with "
                     "--mip-train, multi-mip training):")
        lines.append("")
        lines.append("| mip | Base Color PSNR (linear) | Normal "
                     "angle error, mean |")
        lines.append("| --- | --- | --- |")
        for level in evaluation["levels"]:
            met = per_mip[str(level)]
            lines.append(f"| {level} | {met['psnr_db']:.2f} dB | "
                         f"{met['normal_mean_deg']:.2f}° |")
        lines.append("")
        lines.append("Measured against the SPEC 5.3 teacher mips "
                     "with the batch4/5 multi-mip ruler; the "
                     "quality gate above remains a mip-0 gate.")
        lines.append("")
    advisory = chosen.get("mip_advisory")
    if advisory:
        psnr = advisory["psnr_db"]
        normal = advisory["normal_mean_deg"]
        lines.append("Far-field advisory:")
        lines.append(
            f"- This artifact was trained at mip 0 only and "
            f"degrades sharply at distance (measured with the "
            f"batch4/5 multi-mip ruler): base-color PSNR "
            f"{psnr['0']:.2f} dB at mip 0, {psnr['2']:.2f} dB at "
            f"mip 2, {psnr['4']:.2f} dB at mip 4; "
            f"normal angle error, mean {normal['2']:.2f}° at mip 2 "
            f"and {normal['4']:.2f}° at mip 4; high-frequency "
            f"energy ratio up to {advisory['rhf_max_far_mip']:.2f} "
            f"at far mips (pass band 0.65-1.35). Re-running with "
            f"--mip-train is recommended for this material; in "
            f"batch5 that improved far-mip PSNR by +7.7 to +21.5 "
            f"dB at a mip-0 cost of -1.0 to -2.2 dB (3 materials, "
            f"same recipe). This advisory does not affect the "
            f"gate result above.")
        lines.append("")
    return lines


def _packed_suspicion_lines(report: dict, passed: bool) -> list[str]:
    """The packed-map suspicion banner at the top of the summary
    (Neon O5, 2026-11-21).

    Read from the conclusion record when present, falling back to
    the input warning list (reports written before the conclusion
    field existed).  A run can pass the numeric gate while grading
    the wrong channel -- the round-2 ARM case graded an AO channel
    as roughness and passed -- so the suspicion is stated next to
    the verdict, not buried in the input section.  The verdict
    itself is not changed by this banner.
    """
    suspicion = report.get("conclusion", {}).get(
        "packed_map_suspicion")
    if not suspicion:
        suspicion = [w for w in report.get("input", {}).get(
            "warnings", []) if "packed-map suspicion" in w]
    if not suspicion:
        return []
    head = ("PASS WITH WARNINGS: packed-map suspicion"
            if passed else "WARNING: packed-map suspicion")
    return [
        f"**{head}** — a scalar map's RGB channels differ, so a "
        "packed map (e.g. ORM/ARM) may have been supplied by "
        "mistake. The gate graded the first channel only; if that "
        "channel is not the intended map, the numbers below grade "
        "the wrong data even when they pass.",
        "",
    ]


def _capacity_warning_lines(report: dict) -> list[str]:
    """Outcome capacity warnings, stated in the conclusion block
    (2026-11-19): a passing artifact can still be a capacity loss
    (thin margin at ratio >= 0.9, an outright expansion at >= 1.0),
    and that belongs next to the verdict, not buried in the numbers.
    """
    warnings = report.get("conclusion", {}).get(
        "capacity_warnings") or []
    lines = []
    for warning in warnings:
        lines.append(f"Capacity warning: {warning['message']}")
    entry = report.get("capacity_check") or {}
    if entry.get("override"):
        lines.append(
            "Capacity note: the entry check estimated even the "
            f"cheapest profile ({entry['entry_profile']}) at "
            f"{entry['estimated_vnm1_total_bytes']:,} bytes vs a "
            f"BCn full mip chain estimate of "
            f"{entry['bcn_mip_chain_bytes']:,} bytes; this run "
            "proceeded under --allow-capacity-expansion.")
    if lines:
        lines.append("")
    return lines


def _capacity_lines(capacity: dict | None) -> list[str]:
    if not capacity:
        return []
    bcn = capacity.get("bcn", {})
    lines = ["Capacity:"]
    lines.append(f"- VNM1: {capacity['vnm1_total_bytes']:,} bytes "
                 f"(manifest.json included)")
    if bcn.get("total_mip_chain_bytes"):
        lines.append(
            f"- BCn full mip chain estimate: "
            f"{bcn['total_mip_chain_bytes']:,} bytes")
        lines.append(
            f"- Ratio vs BCn mip chain: "
            f"{capacity['ratio_vs_bcn_mip_chain']:.3f}")
    if bcn.get("total_mip0_bytes"):
        lines.append(
            f"- BCn mip 0 only (reference): "
            f"{bcn['total_mip0_bytes']:,} bytes, ratio "
            f"{capacity['ratio_vs_bcn_mip0_reference']:.3f}")
    lines.append("")
    return lines


def _input_lines(report: dict) -> list[str]:
    inp = report.get("input", {})
    if not inp:
        return []
    lines = ["Inputs:"]
    size = inp.get("size")
    if size:
        lines.append(f"- Size: {size[0]}x{size[1]}")
        if not inp.get("size_verified_baseline", True):
            lines.append(
                "- Size note: outside the judgment run's verified "
                "range (2048x2048); the capability table does not "
                "directly apply (design doc 4.5)")
    lines.append(
        f"- Normal convention: {inp.get('normal_convention')}")
    for logical, entry in inp.get("maps", {}).items():
        if entry.get("source") == "map":
            lines.append(f"- {logical}: map ({entry.get('file')})")
        else:
            constant = entry.get("constant_value")
            if isinstance(constant, (list, tuple)):
                shown = "(" + ", ".join(f"{v:g}" for v in constant) \
                        + ")"
            else:
                shown = f"{constant:g}"
            lines.append(
                f"- {logical}: constant {shown} (no input map; gate "
                f"exempt, not recovered data)")
    for name in inp.get("unsupported_files", []):
        lines.append(
            f"- Unsupported file excluded: {name} (in the input "
            "folder but not a VNM1 input map; extra maps such as "
            "emissive, displacement and height are not supported "
            "in P1)")
    for warning in inp.get("warnings", []):
        lines.append(f"- Warning: {warning}")
    lines.append("")
    return lines


def _visual_lines(report: dict) -> list[str]:
    visual = report.get("visual_check")
    if not visual:
        return []
    if visual.get("skipped"):
        return ["Visual check:",
                "- Skipped (--fast-eval): no sheets were rendered "
                "and no sheet PSNR recheck was run for this run.",
                ""]
    lines = ["Visual check:"]
    for name in visual.get("files", {}):
        lines.append(f"- {name}")
    lines.append(
        f"- Sheet PSNR recheck: {visual['recomputed_psnr_db']:.2f} dB "
        f"vs report {visual['report_psnr_db']:.2f} dB "
        f"(delta {visual['psnr_delta_db']:+.3f} dB, float32 buffer, "
        f"before PNG encoding)")
    lines.append(
        f"- Note: {visual['note']}")
    lines.append("")
    return lines


def _limitations_lines(report: dict) -> list[str]:
    lines = ["Known limitations:"]
    lines.append(
        "- Fine detail: densely packed small elements (fine leaves, "
        "grass, needles, mesh, small text) can miss the gate on base "
        "color even at the largest profile with adjustment. In the "
        "judgment run this was the one failing pattern (leaves: "
        "31.46 dB after the strongest adjustment, with normals and "
        "scalars inside the gate). That is an observation from a "
        "small sample, not a rule that every such material fails.")
    lines.append(
        "- The capability figures behind this tool come from 30 "
        "materials, a single seed and one run per configuration; "
        "they are not a statistical guarantee for other textures.")
    lines.append(
        "- Passing the numeric gate does not by itself confirm the "
        "in-game appearance under lighting, distance, angle and "
        "motion. Check the visual sheets and the material in "
        "context before relying on it.")
    maps = report.get("input", {}).get("maps", {})
    constants = [name for name, entry in maps.items()
                 if entry.get("source") == "constant"]
    if constants:
        lines.append(
            "- Filled-in channels: " + ", ".join(constants) +
            " had no input map and were stored as constants; that "
            "information was not recovered by the conversion.")
    lines.append("")
    return lines


def render_report_markdown(report: dict) -> str:
    """Render the section 7.3 Markdown summary for a report dict."""
    conclusion = report["conclusion"]
    material = report["material"]
    passed = bool(conclusion.get("gate_passed"))
    is_auto = "trials" in report
    if is_auto:
        chosen = None
        index = conclusion.get("chosen_trial_index")
        if index is not None and 0 <= index < len(report["trials"]):
            chosen = report["trials"][index]
        if chosen is None:
            # Reports written before the index was recorded: fall
            # back to the last trial of the chosen profile.
            chosen_profile = conclusion.get("selected_profile") or \
                conclusion.get("representative_profile")
            chosen = next(
                (t for t in reversed(report["trials"])
                 if t["profile"] == chosen_profile), None)
            if chosen is None and report["trials"]:
                chosen = report["trials"][-1]
    else:
        chosen = report["trial"]
    checks = chosen["checks"] if chosen else {}

    lines = [f"# {material} — VNM1 conversion", ""]
    if passed:
        lines.append("Result: PASS")
    else:
        lines.append("Result: FAIL")
    suspicion_lines = _packed_suspicion_lines(report, passed)
    lines.extend(suspicion_lines)
    if not passed:
        if not suspicion_lines:
            lines.append("")
        lines.extend(_failed_summary_lines(report, checks))
    lines.append("")

    if is_auto:
        if passed:
            lines.append(
                f"Selected profile: {conclusion['selected_profile']}"
                + (" (adjustment trial)"
                   if chosen and chosen.get("kind") == "adjustment"
                   else ""))
        else:
            lines.append(
                f"Representative candidate: "
                f"{conclusion.get('representative_profile')} "
                f"(candidate only, under work/attempts/)")
        lines.extend(_adjustment_lines(report))
    else:
        lines.append(f"Selected profile: {chosen['profile']}")
    if chosen:
        lines.append(f"Iterations: {chosen['iterations']:,}")
        device = chosen.get("device") or ""
        env = chosen.get("environment") or report.get("environment") \
            or {}
        gpu = env.get("gpu_name")
        device_shown = f"{device} ({gpu})" if gpu else device
        if is_auto:
            total = sum(t["elapsed_seconds"]
                        for t in report["trials"])
            lines.append(
                f"Time: {_fmt_seconds(total)} across "
                f"{len(report['trials'])} trial(s) ({device_shown})")
        else:
            lines.append(
                f"Time: {_fmt_seconds(chosen['elapsed_seconds'])} "
                f"({device_shown})")
    lines.append("")

    lines.extend(_capacity_warning_lines(report))
    lines.extend(_capacity_lines(report.get("capacity")))

    lines.append("Quality:")
    lines.extend(_check_lines(checks, report))
    lines.append("")

    lines.extend(_mip_lines(chosen))

    if is_auto and len(report["trials"]) > 1:
        lines.append("Trials:")
        for trial in report["trials"]:
            verdict = "PASS" if trial["gate_passed"] else "FAIL"
            kind = " (adjustment)" if trial.get("kind") == \
                "adjustment" else ""
            psnr = trial["metrics"]["psnr_base_color_db"]
            lines.append(
                f"- {trial['profile']}{kind}: PSNR {psnr:.2f} dB — "
                f"{verdict}")
        lines.append("")

    lines.extend(_input_lines(report))
    lines.extend(_visual_lines(report))

    compatibility = report.get("compatibility", {})
    gpu_note = compatibility.get("gpu_decoder") or \
        conclusion.get("compatibility_note")
    if gpu_note:
        lines.append("Compatibility:")
        lines.append(f"- CPU reference decoder: "
                     f"{compatibility.get('cpu_reference_decoder', 'verified (read-back decode in this run)')}")
        lines.append(f"- {gpu_note}")
        lines.append("")

    lines.extend(_limitations_lines(report))

    fmt = report.get("format", {})
    core = fmt.get("core", {})
    lines.append("Reproducibility:")
    lines.append(f"- Format: {fmt.get('format', 'VNM1')} version "
                 f"{fmt.get('version', 1)}")
    lines.append(f"- CLI version: {report.get('cli_version')}")
    if core.get("md5"):
        lines.append(f"- Training core MD5: {core['md5']}")
    if fmt.get("input_config_hash"):
        lines.append(
            f"- Input/config hash: {fmt['input_config_hash']}")
    lines.append("")
    return "\n".join(lines)


__all__ = ["render_report_markdown"]

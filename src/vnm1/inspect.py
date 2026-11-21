"""``vnm1 inspect``: report summary + artifact format verification.

Design doc 3.1: inspect shows a summary of an existing report and
verifies the format of an artifact.  The target may be

- a ``*.report.json`` file (summary, plus verification of the
  artifact directories found next to it),
- a material output directory (its report plus its ``*.vnm``
  artifact and any ``work/attempts/`` candidates), or
- an artifact directory itself (the 4-file ``.vnm`` directory).

Format verification re-checks, from the files on disk alone: the
manifest (``format: 'VNM1'``, ``version: 1``, a registered profile),
the manifest's byte-length / SHA-256 references against the actual
files, the weights chunk against the profile's shapes, the latents
chunk (sizes, profile-derived channel counts, and the unified
storage rule that the subheader reserved bytes are zero), the tail
chunk, and -- when torch is installed -- a spot decode of a coarse
UV lattice confirming finite outputs and unit normals.  The chunk
parsers are the core's own, so a violation raises exactly as it does
in the training read-back path.

Exit codes (inspect has no row in the section 3.3 table; these reuse
its meanings): 0 = summary shown and every found artifact verified
(and the report, when present, passed); 2 = target missing or not a
report / artifact; 3 = the report records a gate failure; 5 = an
artifact failed format verification.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np

from . import core as vt

_ARTIFACT_FILES = ("manifest.json", "weights.bin", "level-0.bin",
                   "tail.bin")


def _sha256_hex(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def verify_artifact(artifact_dir: Path) -> dict:
    """Verify one 4-file artifact directory from disk.

    Returns ``{"ok": bool, "checks": [{name, ok, detail}], ...}``;
    every check is recorded, including the ones that could not run
    because an earlier one failed (marked not ok with the reason).
    """
    artifact_dir = Path(artifact_dir)
    checks: list[dict] = []

    def record(name: str, ok: bool, detail: str = "") -> bool:
        checks.append({"name": name, "ok": bool(ok),
                       "detail": detail})
        return bool(ok)

    present = {name: (artifact_dir / name).is_file()
               for name in _ARTIFACT_FILES}
    if not record("files_present", all(present.values()),
                  "missing: " + ", ".join(
                      n for n, p in present.items() if not p)
                  if not all(present.values()) else
                  "manifest.json, weights.bin, level-0.bin, tail.bin"):
        return {"ok": False, "checks": checks, "manifest": None}

    raw = {name: (artifact_dir / name).read_bytes()
           for name in _ARTIFACT_FILES}
    try:
        manifest = json.loads(raw["manifest.json"].decode("utf-8"))
        if not isinstance(manifest, dict):
            raise ValueError("manifest is not a JSON object")
    except (UnicodeDecodeError, json.JSONDecodeError,
            ValueError) as exc:
        record("manifest_parses", False, str(exc))
        return {"ok": False, "checks": checks, "manifest": None}
    record("manifest_parses", True, "JSON object")
    record("format_and_version",
           manifest.get("format") == "VNM1"
           and manifest.get("version") == 1,
           f"format={manifest.get('format')!r} "
           f"version={manifest.get('version')!r}")
    profile = manifest.get("profile")
    try:
        canonical = vt.resolve_profile(profile) \
            if isinstance(profile, str) else None
    except Exception:
        canonical = None
    record("profile_registered",
           canonical is not None and canonical == profile
           and manifest.get("vnm1Compatible") is not False,
           f"profile={profile!r}")
    if canonical is None:
        return {"ok": False, "checks": checks, "manifest": manifest}

    levels = manifest.get("levels") or [{}]
    level0 = levels[0] if isinstance(levels[0], dict) else {}
    refs = {"weights.bin": manifest.get("weights", {}),
            "level-0.bin": level0.get("chunk", {}),
            "tail.bin": manifest.get("tail", {})}
    ref_ok, ref_detail = True, []
    for name, ref in refs.items():
        if ref.get("byteLength") != len(raw[name]):
            ref_ok = False
            ref_detail.append(
                f"{name}: byteLength {ref.get('byteLength')} != "
                f"actual {len(raw[name])}")
        elif ref.get("sha256") != _sha256_hex(raw[name]):
            ref_ok = False
            ref_detail.append(f"{name}: sha256 mismatch")
    record("manifest_references_match", ref_ok,
           "; ".join(ref_detail) if ref_detail else
           "byteLength + sha256 match for all 3 chunks")

    try:
        vt.parse_weights_chunk(raw["weights.bin"], profile=profile)
        record("weights_chunk_parses", True,
               "payload matches the profile's weight shapes")
    except ValueError as exc:
        record("weights_chunk_parses", False, str(exc))

    latents = raw["level-0.bin"]
    try:
        fine_ch, coarse_ch = vt.latents_chunk_channels(
            latents, profile=profile)
        expected = vt.profile_latent_channels(profile)
        record("latents_reserved_bytes_zero",
               (fine_ch, coarse_ch) == expected,
               f"reserved bytes zero; channels {fine_ch}+{coarse_ch} "
               f"derived from profile")
    except ValueError as exc:
        record("latents_reserved_bytes_zero", False, str(exc))
        fine_ch = coarse_ch = None
    try:
        _fine, fine_size, _coarse, coarse_size = \
            vt.parse_latents_chunk(latents, profile=profile)
        sizes_ok = list(fine_size) == level0.get("fineSize") and \
            list(coarse_size) == level0.get("coarseSize")
        record("latents_chunk_parses", sizes_ok,
               f"fine {list(fine_size)}, coarse {list(coarse_size)}"
               + ("" if sizes_ok else
                  " (manifest level sizes differ)"))
    except ValueError as exc:
        record("latents_chunk_parses", False, str(exc))

    from .driver import parse_tail_chunk
    try:
        tail = parse_tail_chunk(raw["tail.bin"])
        record("tail_chunk_parses",
               tail["tail_first_mip"] == manifest.get("tailFirstMip"),
               f"tail_first_mip={tail['tail_first_mip']}")
    except Exception as exc:
        record("tail_chunk_parses", False,
               f"{type(exc).__name__}: {exc}")

    spot = _spot_decode(raw, manifest, profile)
    record("spot_decode", spot["ok"], spot["detail"])

    return {"ok": all(c["ok"] for c in checks), "checks": checks,
            "manifest": manifest}


def _spot_decode(raw: dict, manifest: dict, profile: str) -> dict:
    """Decode a coarse texel-centre UV lattice and sanity-check the
    outputs.  Skipped (counted as ok, with the reason recorded) when
    torch is not installed: the byte-level checks above do not need
    it, and inspect must stay usable in a minimal environment."""
    if vt.torch is None:
        return {"ok": True,
                "detail": "skipped (torch not installed); byte-level "
                          "checks only"}
    try:
        weights = vt.parse_weights_chunk(raw["weights.bin"],
                                         profile=profile)
        fine, fine_size, coarse, coarse_size = vt.parse_latents_chunk(
            raw["level-0.bin"], profile=profile)
        fine_ch, coarse_ch = vt.latents_chunk_channels(
            raw["level-0.bin"], profile=profile)
        decoder = vt.VnmDecoder.from_u8_numpy(
            weights, fine, coarse, fine_size, coarse_size,
            device="cpu", profile=profile, fine_channels=fine_ch,
            coarse_channels=coarse_ch)
        n = 33
        axis = (np.arange(n, dtype=np.float32) + 0.5) / n
        uu, vv = np.meshgrid(axis, axis)
        uv = np.stack([uu.ravel(), vv.ravel()], axis=1)
        torch = vt.require_torch()
        with torch.no_grad():
            pbr = decoder.decode_pbr(
                torch.tensor(uv, dtype=torch.float32),
                normalized_mip=0.0)
        base = pbr["baseColor"].numpy()
        normal = pbr["normal"].numpy()
        finite = bool(np.all(np.isfinite(base))
                      and np.all(np.isfinite(normal)))
        lengths = np.linalg.norm(normal, axis=1)
        unit = bool(np.max(np.abs(lengths - 1.0)) < 1e-2)
        in_range = bool(np.min(base) >= 0.0 and np.max(base) <= 1.0)
        ok = finite and unit and in_range
        return {"ok": ok,
                "detail": f"{n * n} points: finite={finite}, "
                          f"unit normals={unit}, "
                          f"base color in [0,1]={in_range}"}
    except Exception as exc:
        return {"ok": False,
                "detail": f"{type(exc).__name__}: {exc}"}


def _chosen_trial(report: dict) -> dict | None:
    if "trial" in report:
        return report["trial"]
    trials = report.get("trials") or []
    if not trials:
        return None
    index = report["conclusion"].get("chosen_trial_index")
    if index is not None and 0 <= index < len(trials):
        return trials[index]
    profile = report["conclusion"].get("selected_profile") or \
        report["conclusion"].get("representative_profile")
    for trial in reversed(trials):
        if trial["profile"] == profile:
            return trial
    return trials[-1]


def summarize_report(report: dict) -> list[str]:
    """Compact text summary of a report dict (inspect output)."""
    conclusion = report["conclusion"]
    chosen = _chosen_trial(report)
    lines = [f"Material: {report['material']}"]
    result = conclusion["result"]
    mode = conclusion.get("mode", "manual")
    lines.append(f"Result: {result.upper()} ({mode} mode)")
    if conclusion.get("selected_profile"):
        lines.append(
            f"Selected profile: {conclusion['selected_profile']}")
    if conclusion.get("representative_profile"):
        lines.append(
            f"Representative candidate: "
            f"{conclusion['representative_profile']} "
            f"(candidate only, not a passing artifact)")
    if chosen:
        metrics = chosen["metrics"]
        lines.append(
            f"Base Color PSNR (linear): "
            f"{metrics['psnr_base_color_db']:.2f} dB "
            f"(gate >= 35.00 dB)")
        lines.append(
            f"Iterations: {chosen['iterations']:,}, "
            f"profile {chosen['profile']}")
    if conclusion.get("failed_checks"):
        parts = []
        checks = chosen["checks"] if chosen else {}
        for key in conclusion["failed_checks"]:
            check = checks.get(key)
            if check:
                parts.append(
                    f"{key} ({check['difference']:+.4g} vs line)")
            else:
                parts.append(key)
        lines.append("Failed checks: " + ", ".join(parts))
    if conclusion.get("gate_exempt_checks"):
        lines.append("Gate-exempt checks: "
                     + ", ".join(conclusion["gate_exempt_checks"])
                     + " (constant channels, not recovered data)")
    adjustment = conclusion.get("adjustment")
    if adjustment:
        if adjustment.get("ran"):
            lines.append(
                f"Adjustment: ran (pattern "
                f"{adjustment.get('pattern')})")
        else:
            lines.append(
                f"Adjustment: not run ({adjustment.get('reason')})")
    capacity = report.get("capacity")
    if capacity:
        lines.append(
            f"Capacity: {capacity['vnm1_total_bytes']:,} bytes, "
            f"ratio {capacity['ratio_vs_bcn_mip_chain']:.4f} vs BCn "
            f"mip chain")
    if conclusion.get("artifact_location"):
        lines.append(
            f"Artifact location (as recorded): "
            f"{conclusion['artifact_location']}")
    return lines


def _find_artifacts_near(base: Path, report: dict | None) -> list[Path]:
    """Artifact directories next to a report / inside a material
    output directory: the distribution ``<material>.vnm`` plus any
    ``work/attempts/`` candidates."""
    found: list[Path] = []
    for candidate in sorted(base.glob("*.vnm")):
        if candidate.is_dir() and (candidate / "manifest.json").is_file():
            found.append(candidate)
    attempts = base / "work" / "attempts"
    if attempts.is_dir():
        for candidate in sorted(attempts.iterdir()):
            if candidate.is_dir() and \
                    (candidate / "manifest.json").is_file():
                found.append(candidate)
    return found


def inspect_target(target: Path) -> tuple[int, list[str]]:
    """Run inspect on a target path.  Returns (exit_code, lines)."""
    target = Path(target)
    if not target.exists():
        return 2, [f"inspect: target not found: {target}"]

    report = None
    artifact_dirs: list[Path] = []
    if target.is_file():
        if target.suffix != ".json":
            return 2, [f"inspect: not a report or artifact: {target}"]
        try:
            report = json.loads(target.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            return 2, [f"inspect: cannot read report {target}: {exc}"]
        if "conclusion" not in report or "material" not in report:
            return 2, [f"inspect: {target} is not a vnm1 report"]
        artifact_dirs = _find_artifacts_near(target.parent, report)
    elif (target / "manifest.json").is_file():
        artifact_dirs = [target]
    else:
        reports = sorted(target.glob("*.report.json"))
        if reports:
            try:
                report = json.loads(
                    reports[0].read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError) as exc:
                return 2, [f"inspect: cannot read report "
                           f"{reports[0]}: {exc}"]
        artifact_dirs = _find_artifacts_near(target, report)
        if report is None and not artifact_dirs:
            return 2, [f"inspect: no report or artifact found in "
                       f"{target}"]

    lines: list[str] = []
    if report is not None:
        lines.extend(summarize_report(report))
        lines.append("")
    verification_failed = False
    if artifact_dirs:
        for artifact_dir in artifact_dirs:
            result = verify_artifact(artifact_dir)
            status = "OK" if result["ok"] else "FAILED"
            lines.append(f"Artifact verification: {artifact_dir} "
                         f"— {status}")
            for check in result["checks"]:
                mark = "ok" if check["ok"] else "FAIL"
                lines.append(f"  [{mark}] {check['name']}: "
                             f"{check['detail']}")
            if not result["ok"]:
                verification_failed = True
            lines.append("")
    elif report is not None:
        lines.append("Artifact verification: no artifact directory "
                     "found next to the report; skipped.")
        lines.append("")

    if verification_failed:
        return 5, lines
    if report is not None and not report["conclusion"].get(
            "gate_passed"):
        return 3, lines
    return 0, lines


__all__ = ["inspect_target", "summarize_report", "verify_artifact"]

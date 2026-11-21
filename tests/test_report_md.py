"""Unit tests for the Markdown summary (vnm1.report_md, step D).

Covers design doc 7.3 / 11.3: verdict first, failing items with
their distance from the gate line shown up front on failure,
gate-exempt constants never shown as passes, the GPU-decoder
compatibility wording (7.2 F), the adjustment record, and the
visual-check section.  Pure dict -> text; no torch.
"""

import shutil
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np
from PIL import Image

from vnm1 import driver, report as report_mod
from vnm1.imaging import build_teacher
from vnm1.inputs import resolve_inputs
from vnm1.report_md import render_report_markdown

CAPACITY = {
    "vnm1_breakdown_bytes": {"manifest.json": 1000,
                             "weights.bin": 8000,
                             "level-0.bin": 1000,
                             "tail.bin": 2000},
    "vnm1_total_bytes": 12000,
    "bcn": {"per_map": {}, "total_mip0_bytes": 30000,
            "total_mip_chain_bytes": 40000},
    "ratio_vs_bcn_mip_chain": 0.3,
    "ratio_vs_bcn_mip0_reference": 0.4,
}


def _make_spec(tmp: Path):
    folder = tmp / "oak"
    folder.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(5)
    Image.fromarray(
        rng.integers(0, 255, size=(16, 16, 3), dtype=np.uint8)
    ).save(folder / "basecolor.png")
    Image.fromarray(
        np.full((16, 16), 128, dtype=np.uint8)).save(
            folder / "roughness.png")
    spec = resolve_inputs(folder, normal_convention="opengl")
    return spec, build_teacher(spec)


def _outcome(profile="standard", *, psnr=38.9, ao=0.009):
    metrics = {"psnr_base_color_db": psnr,
               "psnr_base_color_db_srgb_space_reference": 30.0,
               "mae": {"roughness": 0.002, "metalness": 0.0, "ao": ao},
               "normal_angle_mean_deg": 0.37,
               "normal_angle_p95_deg": 0.76,
               "texel_count": 256, "eval_seconds": 0.01,
               "decode_texels_per_second": 1.0}
    sources = {"base_color": "map", "normal": "constant",
               "roughness": "map", "metalness": "constant",
               "ao": "constant"}
    verdict = driver.gate_verdict(metrics, sources)
    return driver.TrialOutcome(
        profile=profile, fine_size=(512, 512), coarse_size=(256, 256),
        fine_channels=4, coarse_channels=4, metrics=metrics,
        checks=verdict["checks"], gate_passed=verdict["passed"],
        capacity=CAPACITY, recipe={"recipe_version": 2},
        environment={"device": "cuda", "gpu_name": "T4"},
        started_at="t0", finished_at="t1", elapsed_seconds=582.0,
        training_seconds=500.0, final_loss=0.01,
        artifact_dir=Path("/nonexistent"),
        artifact_files={}, manifest={},
        roundtrip={"weights_parse_ok": True})


def _manual_report(tmp, **kw):
    spec, teacher = _make_spec(tmp)
    params = {"profile": "standard", "iterations": 50000,
              "color_weight": 4.0, "scalar_weight": 1.0, "seed": 1337}
    return report_mod.build_report(
        spec, teacher, _outcome(**kw), mode="manual",
        artifact_location="/dist/oak/oak.vnm", trial_params=params)


def _auto_result(trials, *, selected=None, representative=None,
                 adjustment=None):
    return SimpleNamespace(
        status="passed" if selected else "failed",
        trials=trials, selected=selected,
        representative=representative,
        adjustment=adjustment or {"eligible": False,
                                  "reason": "ladder_passed",
                                  "pattern": None, "ran": False},
        state_path="/x/auto_state.json", resume_hash="ab" * 32,
        resumed_trials=0)


def _trial_record(tmp, *, kind="ladder", **kw):
    spec, teacher = _make_spec(tmp)
    outcome = _outcome(**kw)
    return report_mod.trial_record(
        outcome, kind=kind,
        trial_params={"iterations": 50000, "color_weight": 4.0,
                      "scalar_weight": 1.0, "seed": 1337},
        artifact_location="/dist/oak/oak.vnm")


class ManualMarkdownTest(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp(prefix="vnm1-md-test-"))
        self.addCleanup(shutil.rmtree, self.tmp, True)

    def test_pass_markdown(self):
        md = render_report_markdown(_manual_report(self.tmp))
        self.assertIn("# oak — VNM1 conversion", md)
        self.assertIn("Result: PASS", md)
        self.assertIn("Selected profile: standard", md)
        self.assertIn("Time: 9 min 42 sec (cuda (T4))", md)
        self.assertIn("VNM1: 12,000 bytes", md)
        self.assertIn("Ratio vs BCn mip chain: 0.300", md)
        self.assertIn("Base Color PSNR (linear): 38.90 dB — PASS", md)
        # Constants are shown as constants, never as passes.
        self.assertIn("Metalness MAE: constant 0 — gate exempt", md)
        self.assertIn("not recovered data", md)
        # All profiles are GPU-verified since P2: no P2 warning.
        self.assertNotIn("P2 work", md)
        self.assertIn("Known limitations:", md)
        self.assertIn("Training core MD5: "
                      "c1f9935a8912842a10072ae6df1dc725", md)

    def test_fail_color_only_markdown(self):
        md = render_report_markdown(
            _manual_report(self.tmp, psnr=34.77))
        self.assertIn("Result: FAIL", md)
        self.assertIn("Gate: NOT MET", md)
        self.assertIn("Base Color PSNR (linear): 34.77 dB "
                      "(-0.23 dB vs the 35.00 dB line)", md)
        # Section 11.3 colour-only pattern wording.
        self.assertIn("miss is in base-color detail only", md)
        self.assertIn("not treated as a passing artifact", md)

    def test_quality_profile_carries_gpu_verified_wording(self):
        # Updated 2026-11-20 to follow the facts after P2 completed
        # (see fix-log-gpu-note-20261120.md).
        spec, teacher = _make_spec(self.tmp)
        params = {"profile": "quality", "iterations": 50000,
                  "color_weight": 4.0, "scalar_weight": 1.0,
                  "seed": 1337}
        report = report_mod.build_report(
            spec, teacher, _outcome(profile="quality"), mode="manual",
            artifact_location="/x", trial_params=params)
        md = render_report_markdown(report)
        self.assertIn("supports all four profiles", md)
        self.assertNotIn("P2 work", md)

    def test_visual_section_rendered_when_present(self):
        report = _manual_report(self.tmp)
        report["visual_check"] = {
            "files": {"oak_sheet.png": {"bytes": 10, "kind": "sheet"},
                      "oak_crop.png": {"bytes": 5, "kind": "crop"}},
            "recomputed_psnr_db": 38.901, "report_psnr_db": 38.9,
            "psnr_delta_db": 0.001, "psnr_match": True,
            "heat_gain": 8.0, "note": "heat note text"}
        md = render_report_markdown(report)
        self.assertIn("Visual check:", md)
        self.assertIn("- oak_sheet.png", md)
        self.assertIn("Sheet PSNR recheck: 38.90 dB vs report "
                      "38.90 dB", md)
        self.assertIn("heat note text", md)


class AutoMarkdownTest(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp(prefix="vnm1-md-auto-"))
        self.addCleanup(shutil.rmtree, self.tmp, True)

    def _report(self, result):
        spec, teacher = _make_spec(self.tmp)
        return report_mod.build_auto_report(
            spec, teacher, result, state_path="/x/auto_state.json")

    def test_auto_fail_markdown(self):
        ladder = _trial_record(self.tmp, psnr=30.20, ao=0.0271)
        result = _auto_result(
            [ladder], representative=ladder,
            adjustment={"eligible": False, "reason": "color_not_narrow",
                        "pattern": None, "ran": False,
                        "detail": "psnr 30.20 < narrow floor 33.5"})
        report = self._report(result)
        self.assertEqual(report["conclusion"]["chosen_trial_index"], 0)
        md = render_report_markdown(report)
        self.assertIn("Result: FAIL", md)
        self.assertIn("Representative candidate: standard "
                      "(candidate only, under work/attempts/)", md)
        self.assertIn("not a passing artifact", md)
        self.assertIn("Adjustment: not run", md)
        self.assertIn("narrow-miss range", md)

    def test_auto_adjustment_pass_markdown(self):
        ladder = _trial_record(self.tmp, psnr=34.38, ao=0.0219)
        adj = _trial_record(self.tmp, kind="adjustment", psnr=35.31,
                            ao=0.0138)
        result = _auto_result(
            [ladder, adj], selected=adj,
            adjustment={"eligible": True, "reason": "narrow_miss",
                        "pattern": "color_and_scalar", "ran": True})
        report = self._report(result)
        self.assertEqual(report["conclusion"]["chosen_trial_index"], 1)
        md = render_report_markdown(report)
        self.assertIn("Result: PASS", md)
        self.assertIn("Selected profile: standard "
                      "(adjustment trial)", md)
        self.assertIn("Adjustment: ran once", md)
        self.assertIn("Trials:", md)


if __name__ == "__main__":
    unittest.main()

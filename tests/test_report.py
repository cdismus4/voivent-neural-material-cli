"""Unit tests for the minimal report builder (vnm1.report, step B).

Covers the design doc 7.2 content set: conclusion with failed-check
list and margins, input record with gate-exempt constants, capacity
passthrough, core provenance, and the GPU-decoder compatibility
wording (standard vs other profiles, 7.2 F).
"""

import shutil
import tempfile
import unittest
from pathlib import Path

import numpy as np
from PIL import Image

from vnm1 import driver, report as report_mod
from vnm1.imaging import build_teacher
from vnm1.inputs import resolve_inputs


def _make_spec(tmp: Path):
    folder = tmp / "oak"
    folder.mkdir(parents=True)
    rng = np.random.default_rng(5)
    Image.fromarray(
        rng.integers(0, 255, size=(16, 16, 3), dtype=np.uint8)
    ).save(folder / "basecolor.png")
    Image.fromarray(
        np.full((16, 16), 128, dtype=np.uint8)).save(
            folder / "roughness.png")
    spec = resolve_inputs(folder, normal_convention="opengl")
    return spec, build_teacher(spec)


def _outcome(profile="standard", gate_passed=True):
    metrics = {"psnr_base_color_db": 38.9 if gate_passed else 34.77,
               "psnr_base_color_db_srgb_space_reference": 30.0,
               "mae": {"roughness": 0.002, "metalness": 0.0,
                       "ao": 0.009},
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
        capacity={"vnm1_total_bytes": 100,
                  "ratio_vs_bcn_mip_chain": 0.08},
        recipe={"recipe_version": 2},
        environment={"device": "cuda", "gpu_name": "T4"},
        started_at="t0", finished_at="t1", elapsed_seconds=1.0,
        training_seconds=0.9, final_loss=0.01,
        artifact_dir=Path("/nonexistent"),
        artifact_files={}, manifest={},
        roundtrip={"weights_parse_ok": True})


class BuildReportTest(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp(prefix="vnm1-report-test-"))
        self.addCleanup(shutil.rmtree, self.tmp, True)
        self.spec, self.teacher = _make_spec(self.tmp)
        self.params = {"profile": "standard", "iterations": 50000,
                       "color_weight": 4.0, "scalar_weight": 1.0,
                       "seed": 1337}

    def test_pass_report_structure(self):
        rep = report_mod.build_report(
            self.spec, self.teacher, _outcome(), mode="manual",
            artifact_location="/dist/oak/oak.vnm",
            trial_params=self.params)
        self.assertEqual(rep["conclusion"]["result"], "passed")
        self.assertEqual(rep["conclusion"]["failed_checks"], [])
        self.assertIn("mae_metalness",
                      rep["conclusion"]["gate_exempt_checks"])
        self.assertEqual(rep["format"]["format"], "VNM1")
        self.assertEqual(rep["format"]["version"], 1)
        self.assertEqual(rep["format"]["core"]["md5"],
                         "c1f9935a8912842a10072ae6df1dc725")
        self.assertEqual(rep["trial"]["normal_weight"], 2.0)
        self.assertEqual(rep["input"]["maps"]["ao"]["source"],
                         "constant")
        self.assertNotIn("P2 work",
                         rep["compatibility"]["gpu_decoder"])

    def test_fail_report_lists_failed_checks_with_margins(self):
        rep = report_mod.build_report(
            self.spec, self.teacher, _outcome(gate_passed=False),
            mode="manual", artifact_location="/dist/oak/work/attempts/x",
            trial_params=self.params)
        self.assertEqual(rep["conclusion"]["result"], "failed")
        self.assertEqual(rep["conclusion"]["failed_checks"],
                         ["psnr_base_color_db"])
        diff = rep["trial"]["checks"]["psnr_base_color_db"]["difference"]
        self.assertAlmostEqual(diff, -0.23, places=2)

    def test_quality_profile_carries_gpu_verified_note(self):
        # Updated 2026-11-20 to follow the facts, not a change of
        # claim: P2 completed and all four profiles are GPU-verified
        # (see fix-log-gpu-note-20261120.md).
        rep = report_mod.build_report(
            self.spec, self.teacher, _outcome(profile="quality"),
            mode="manual", artifact_location="/x",
            trial_params=self.params)
        self.assertIn("all four profiles",
                      rep["compatibility"]["gpu_decoder"])
        self.assertIn("verified",
                      rep["compatibility"]["gpu_decoder"])
        self.assertNotIn("P2 work",
                         rep["compatibility"]["gpu_decoder"])
        self.assertIn("Pixel 9a",
                      rep["conclusion"]["compatibility_note"])

    def test_input_config_hash_is_stable_and_seed_sensitive(self):
        h1 = report_mod.input_config_hash(self.spec, self.teacher,
                                          self.params)
        h2 = report_mod.input_config_hash(self.spec, self.teacher,
                                          self.params)
        self.assertEqual(h1, h2)
        other = dict(self.params, seed=42)
        self.assertNotEqual(
            h1, report_mod.input_config_hash(self.spec, self.teacher,
                                             other))


if __name__ == "__main__":
    unittest.main()

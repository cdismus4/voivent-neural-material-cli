"""Unit tests for `vnm1 train` manual mode (step B): argument
validation, exit codes (design doc 3.3) and artifact placement
(design doc 4.6).

The heavy trial itself is replaced by a monkeypatched
``vnm1.driver.run_trial`` so these tests run without torch; the real
trial is exercised on Kaggle (see ../manifest.json).  Input
resolution and teacher construction run for real on tiny synthetic
PNGs.
"""

import json
import shutil
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np
from PIL import Image

from vnm1 import driver
from vnm1.cli import main
from vnm1.errors import ExportVerificationError, TrainingError


def _write_png(path: Path, array: np.ndarray) -> None:
    Image.fromarray(array).save(path)


def make_material(folder: Path, size: int = 32) -> Path:
    folder.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(11)
    base = rng.integers(64, 192, size=(size, size, 3), dtype=np.uint8)
    _write_png(folder / "basecolor.png", base)
    normal = np.zeros((size, size, 3), dtype=np.uint8)
    normal[..., 0] = 128
    normal[..., 1] = 128
    normal[..., 2] = 255
    _write_png(folder / "normal.png", normal)
    _write_png(folder / "roughness.png",
               np.full((size, size), 200, dtype=np.uint8))
    _write_png(folder / "ao.png",
               np.full((size, size), 230, dtype=np.uint8))
    return folder


def fake_outcome(*, gate_passed: bool, profile: str = "standard"):
    """A TrialOutcome stand-in whose staging dir the fake run_trial
    has already filled with the 4 artifact files."""
    metrics = {
        "psnr_base_color_db": 38.9 if gate_passed else 31.0,
        "psnr_base_color_db_srgb_space_reference": 30.0,
        "mae": {"roughness": 0.002, "metalness": 0.0, "ao": 0.009},
        "normal_angle_mean_deg": 0.37,
        "normal_angle_p95_deg": 0.76,
        "texel_count": 1024, "eval_seconds": 0.1,
        "decode_texels_per_second": 10000.0,
    }
    sources = {"base_color": "map", "normal": "map", "roughness": "map",
               "metalness": "constant", "ao": "map"}
    verdict = driver.gate_verdict(metrics, sources)
    assert verdict["passed"] == gate_passed
    return metrics, verdict


class TrainTestBase(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp(prefix="vnm1-train-test-"))
        self.addCleanup(shutil.rmtree, self.tmp, True)
        self.material = make_material(self.tmp / "oak")
        self.out = self.tmp / "dist"

    def argv(self, *extra):
        # --allow-capacity-expansion: the fixture is a tiny synthetic
        # stand-in for the flow tests; the capacity entry check
        # (2026-11-19) is covered by test_capacity_check.
        return ["train", str(self.material), "--mode", "manual",
                "--normal-convention", "opengl",
                "--allow-unverified-size",
                "--allow-capacity-expansion",
                "--output", str(self.out),
                *extra]

    def patch_status(self, *, torch_available=True):
        status = {"python_version": "3.12.3",
                  "python_implementation": "CPython", "os": "Linux",
                  "torch_available": torch_available,
                  "torch_version": "2.x" if torch_available else None,
                  "cuda_available": False, "gpu_name": None}
        return mock.patch.object(driver, "torch_status",
                                 return_value=status)

    def patch_trial(self, *, gate_passed=True, raises=None):
        metrics, verdict = fake_outcome(gate_passed=gate_passed)

        def run_trial(teacher, channel_sources, options, progress_cb=None):
            if raises is not None:
                raise raises
            staging = options.work_dir / "staging" / \
                f"{options.material_name}.vnm"
            staging.mkdir(parents=True, exist_ok=True)
            files = {}
            for name in ("manifest.json", "weights.bin", "level-0.bin",
                         "tail.bin"):
                data = f"fake-{name}".encode()
                (staging / name).write_bytes(data)
                import hashlib
                files[name] = {"bytes": len(data),
                               "sha256": hashlib.sha256(data).hexdigest()}
            return driver.TrialOutcome(
                profile=options.profile,
                fine_size=(512, 512), coarse_size=(256, 256),
                fine_channels=4, coarse_channels=4,
                metrics=metrics, checks=verdict["checks"],
                gate_passed=gate_passed,
                capacity={
                    "vnm1_breakdown_bytes": {
                        "manifest.json": 15, "weights.bin": 15,
                        "level-0.bin": 15, "tail.bin": 12},
                    "vnm1_total_bytes": 57,
                    "bcn": {"per_map": {}, "total_mip0_bytes": 100,
                            "total_mip_chain_bytes": 120},
                    "ratio_vs_bcn_mip_chain": 57 / 120,
                    "ratio_vs_bcn_mip0_reference": 57 / 100,
                },
                recipe={"recipe_version": 2},
                environment={"device": "cpu", "cpu_fallback": True},
                started_at="2026-11-19T00:00:00+00:00",
                finished_at="2026-11-19T00:01:00+00:00",
                elapsed_seconds=60.0, training_seconds=55.0,
                final_loss=0.01, artifact_dir=staging,
                artifact_files=files, manifest={"format": "VNM1"},
                roundtrip={"weights_parse_ok": True,
                           "latents_parse_ok": True,
                           "tail_parse_ok": True,
                           "latents_reserved_bytes_zero": True,
                           "manifest_sha_match": True})

        return mock.patch.object(driver, "run_trial", run_trial)


def fake_visual_sheets(artifact_dir, teacher, material_name, out_dir,
                       *, expected_psnr_db):
    """Stand-in for vnm1.visual.generate_visual_sheets (step D): the
    fake artifacts written by patch_trial are not decodable, so the
    sheet step is faked the same way the trial itself is.  The real
    sheet generation is covered by tests/test_visual.py."""
    files = {}
    for kind in ("sheet", "crop"):
        path = Path(out_dir) / f"{material_name}_{kind}.png"
        Image.new("RGB", (8, 8)).save(path)
        files[path.name] = {"bytes": path.stat().st_size, "kind": kind}
    return {"files": files, "artifact_dir": str(artifact_dir),
            "recomputed_psnr_db": expected_psnr_db,
            "report_psnr_db": expected_psnr_db, "psnr_delta_db": 0.0,
            "psnr_tolerance_db": 0.05, "psnr_match": True,
            "heat_gain": 8.0, "note": "fake sheets (flow test)"}


def patch_visual():
    return mock.patch("vnm1.visual.generate_visual_sheets",
                      fake_visual_sheets)


class ArgumentValidationTest(TrainTestBase):
    def test_missing_normal_convention_is_exit_2(self):
        argv = ["train", str(self.material), "--mode", "manual",
                "--allow-unverified-size", "--output", str(self.out)]
        self.assertEqual(main(argv), 2)

    def test_unverified_size_without_flag_is_exit_2(self):
        argv = ["train", str(self.material), "--mode", "manual",
                "--normal-convention", "opengl",
                "--output", str(self.out)]
        self.assertEqual(main(argv), 2)

    def test_invalid_iterations_is_exit_2(self):
        with self.patch_status():
            self.assertEqual(main(self.argv("--iterations", "0")), 2)

    def test_invalid_device_is_exit_2(self):
        with self.patch_status():
            self.assertEqual(main(self.argv("--device", "tpu")), 2)

    def test_auto_mode_is_wired_not_a_stub(self):
        # Step C replaced the step B "not implemented" stub (exit 69):
        # auto mode now runs, so a validation failure surfaces as the
        # section 3.3 input error (2) -- here, the missing required
        # --normal-convention.  Auto-mode behaviour itself is covered
        # by tests/test_auto.py.
        argv = ["train", str(self.material), "--mode", "auto"]
        self.assertEqual(main(argv), 2)


class ExitCodeFlowTest(TrainTestBase):
    def test_torch_missing_is_exit_4(self):
        with self.patch_status(torch_available=False):
            self.assertEqual(main(self.argv()), 4)

    def test_training_error_is_exit_4(self):
        with self.patch_status(), self.patch_trial(
                raises=TrainingError("diverged")):
            self.assertEqual(main(self.argv()), 4)

    def test_export_error_is_exit_5(self):
        with self.patch_status(), self.patch_trial(
                raises=ExportVerificationError("sha mismatch")):
            self.assertEqual(main(self.argv()), 5)

    def test_pass_is_exit_0_artifact_in_dist_report_written(self):
        with self.patch_status(), self.patch_trial(gate_passed=True), \
                patch_visual():
            self.assertEqual(main(self.argv()), 0)
        artifact = self.out / "oak" / "oak.vnm"
        self.assertTrue((artifact / "manifest.json").is_file())
        self.assertTrue((artifact / "level-0.bin").is_file())
        report = json.loads(
            (self.out / "oak" / "oak.report.json").read_text())
        self.assertTrue(report["conclusion"]["gate_passed"])
        self.assertEqual(report["conclusion"]["result"], "passed")
        self.assertEqual(report["trial"]["profile"], "standard")
        self.assertEqual(report["capacity"]["vnm1_total_bytes"], 57)
        self.assertEqual(report["format"]["core"]["md5"],
                         "c1f9935a8912842a10072ae6df1dc725")
        self.assertEqual(
            report["input"]["maps"]["metalness"]["source"], "constant")
        self.assertTrue(
            report["input"]["maps"]["metalness"]["gate_exempt"])
        # Step D: visual_check record + Markdown summary are written.
        self.assertTrue(report["visual_check"]["psnr_match"])
        self.assertIn("oak_sheet.png", report["visual_check"]["files"])
        md = (self.out / "oak" / "oak.report.md").read_text()
        self.assertIn("Result: PASS", md)
        self.assertIn("Metalness MAE: constant 0 — gate exempt", md)

    def test_fail_is_exit_3_artifact_only_in_attempts(self):
        with self.patch_status(), self.patch_trial(gate_passed=False), \
                patch_visual():
            self.assertEqual(main(self.argv()), 3)
        self.assertFalse((self.out / "oak" / "oak.vnm").exists())
        attempt = self.out / "oak" / "work" / "attempts" / \
            "trial-001-standard"
        self.assertTrue((attempt / "manifest.json").is_file())
        report = json.loads(
            (self.out / "oak" / "oak.report.json").read_text())
        self.assertFalse(report["conclusion"]["gate_passed"])
        self.assertIn("psnr_base_color_db",
                      report["conclusion"]["failed_checks"])
        self.assertIn("attempts",
                      report["conclusion"]["artifact_location"])
        md = (self.out / "oak" / "oak.report.md").read_text()
        self.assertIn("Result: FAIL", md)
        self.assertIn("Gate: NOT MET", md)


if __name__ == "__main__":
    unittest.main()

"""Tests for the 2026-11-21 Neon O4 / O5 fixes.

O5 (warning visibility):
- a packed-map suspicion is carried to the train conclusion (the
  console result line and the top of report.md) instead of staying
  buried in preflight, without changing the verdict or exit code;
- image files in the input folder that no map consumed are stated
  as excluded as unsupported, in preflight, in train and in the
  report, instead of being silently ignored.

O4 (post-processing cost):
- the opt-in ``--fast-eval`` skips the multi-mip measurement in the
  driver and the visual sheets in the CLI finish step, while the
  gate evaluation, the artifact and the verdict are unchanged;
- the flag is recorded (report trial record, auto resume hash)
  only when enabled, mirroring the --mip-train rule, so default
  runs keep their previous records byte for byte.

The train flow tests reuse the test_train.py pattern (monkeypatched
``driver.run_trial`` and sheet generation); the driver-level test
runs one tiny real trial on CPU and is skipped without torch.
"""

import contextlib
import hashlib
import io
import json
import shutil
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np
from PIL import Image

from vnm1 import auto, core as vt, driver
from vnm1.cli import build_parser, main
from vnm1.imaging import build_teacher, packed_suspicion_warnings
from vnm1.inputs import resolve_inputs
from vnm1.report_md import render_report_markdown

TORCH_AVAILABLE = vt.torch is not None


def _write_png(path: Path, array: np.ndarray) -> None:
    Image.fromarray(array).save(path)


def make_material(folder: Path, size: int = 32, *, packed: bool = False,
                  extras: tuple = ()) -> Path:
    """A tiny material; ``packed`` makes roughness an RGB map with
    differing channels (the packed-map suspicion trigger), ``extras``
    adds unconsumed files to the folder."""
    folder.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(11)
    base = rng.integers(64, 192, size=(size, size, 3), dtype=np.uint8)
    _write_png(folder / "basecolor.png", base)
    normal = np.zeros((size, size, 3), dtype=np.uint8)
    normal[..., 0] = 128
    normal[..., 1] = 128
    normal[..., 2] = 255
    _write_png(folder / "normal.png", normal)
    if packed:
        rough = np.zeros((size, size, 3), dtype=np.uint8)
        rough[..., 0] = 200  # would-be AO channel of an ARM map
        rough[..., 1] = 90   # would-be roughness
        rough[..., 2] = 0    # would-be metalness
        _write_png(folder / "roughness.png", rough)
    else:
        _write_png(folder / "roughness.png",
                   np.full((size, size), 200, dtype=np.uint8))
    _write_png(folder / "ao.png",
               np.full((size, size), 230, dtype=np.uint8))
    for name in extras:
        _write_png(folder / name,
                   np.full((size, size), 128, dtype=np.uint8))
    return folder


def _fake_outcome(*, gate_passed: bool):
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


class FlowBase(unittest.TestCase):
    """Manual-mode train through the CLI with the trial and the
    sheet generation faked (the test_train.py pattern)."""

    packed = False
    extras: tuple = ()

    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp(prefix="vnm1-neon-test-"))
        self.addCleanup(shutil.rmtree, self.tmp, True)
        self.material = make_material(self.tmp / "oak",
                                      packed=self.packed,
                                      extras=self.extras)
        self.out = self.tmp / "dist"
        self.seen_options = []
        self.visual_calls = []

    def argv(self, *extra):
        return ["train", str(self.material), "--mode", "manual",
                "--normal-convention", "opengl",
                "--allow-unverified-size",
                "--allow-capacity-expansion",
                "--output", str(self.out),
                *extra]

    def _patch_status(self):
        status = {"python_version": "3.12.3",
                  "python_implementation": "CPython", "os": "Linux",
                  "torch_available": True, "torch_version": "2.x",
                  "cuda_available": False, "gpu_name": None}
        return mock.patch.object(driver, "torch_status",
                                 return_value=status)

    def _patch_trial(self, *, gate_passed=True):
        metrics, verdict = _fake_outcome(gate_passed=gate_passed)
        seen = self.seen_options

        def run_trial(teacher, channel_sources, options,
                      progress_cb=None):
            seen.append(options)
            staging = options.work_dir / "staging" / \
                f"{options.material_name}.vnm"
            staging.mkdir(parents=True, exist_ok=True)
            files = {}
            for name in ("manifest.json", "weights.bin",
                         "level-0.bin", "tail.bin"):
                data = f"fake-{name}".encode()
                (staging / name).write_bytes(data)
                files[name] = {
                    "bytes": len(data),
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
                started_at="2026-11-21T00:00:00+00:00",
                finished_at="2026-11-21T00:01:00+00:00",
                elapsed_seconds=60.0, training_seconds=55.0,
                final_loss=0.01, artifact_dir=staging,
                artifact_files=files, manifest={"format": "VNM1"},
                roundtrip={"weights_parse_ok": True,
                           "latents_parse_ok": True,
                           "tail_parse_ok": True,
                           "latents_reserved_bytes_zero": True,
                           "manifest_sha_match": True})

        return mock.patch.object(driver, "run_trial", run_trial)

    def _patch_visual(self):
        calls = self.visual_calls

        def fake_sheets(artifact_dir, teacher, material_name, out_dir,
                        *, expected_psnr_db):
            calls.append(material_name)
            files = {}
            for kind in ("sheet", "crop"):
                path = Path(out_dir) / f"{material_name}_{kind}.png"
                Image.new("RGB", (8, 8)).save(path)
                files[path.name] = {"bytes": path.stat().st_size,
                                    "kind": kind}
            return {"files": files, "artifact_dir": str(artifact_dir),
                    "recomputed_psnr_db": expected_psnr_db,
                    "report_psnr_db": expected_psnr_db,
                    "psnr_delta_db": 0.0, "psnr_tolerance_db": 0.05,
                    "psnr_match": True, "heat_gain": 8.0,
                    "note": "fake sheets (flow test)"}

        return mock.patch("vnm1.visual.generate_visual_sheets",
                          fake_sheets)

    def run_train(self, *extra, gate_passed=True):
        buf = io.StringIO()
        with self._patch_status(), self._patch_trial(
                gate_passed=gate_passed), self._patch_visual(), \
                contextlib.redirect_stdout(buf):
            code = main(self.argv(*extra))
        return code, buf.getvalue()

    def read_report(self):
        path = self.out / "oak" / "oak.report.json"
        return json.loads(path.read_text(encoding="utf-8"))

    def read_md(self):
        path = self.out / "oak" / "oak.report.md"
        return path.read_text(encoding="utf-8")


# ---------------------------------------------------------------------------
# O5: packed-map suspicion reaches the conclusion
# ---------------------------------------------------------------------------
class PackedSuspicionConclusionTest(FlowBase):
    packed = True

    def test_passing_run_conclusion_carries_the_warning(self):
        code, out = self.run_train(gate_passed=True)
        self.assertEqual(code, 0)  # verdict / exit code unchanged
        self.assertIn("PASS WITH WARNINGS: packed-map suspicion", out)
        report = self.read_report()
        suspicion = report["conclusion"].get("packed_map_suspicion")
        self.assertTrue(suspicion)
        self.assertTrue(any("roughness" in w for w in suspicion))
        md = self.read_md()
        self.assertIn("**PASS WITH WARNINGS: packed-map suspicion**",
                      md)
        # The banner sits at the top, before the quality section.
        self.assertLess(md.index("packed-map suspicion"),
                        md.index("Quality:"))

    def test_failing_run_conclusion_carries_the_warning(self):
        code, out = self.run_train(gate_passed=False)
        self.assertEqual(code, 3)
        self.assertIn("FAIL WITH WARNINGS: packed-map suspicion", out)
        md = self.read_md()
        self.assertIn("**WARNING: packed-map suspicion**", md)


class NoSuspicionControlTest(FlowBase):
    packed = False

    def test_clean_run_has_no_warning_marks(self):
        code, out = self.run_train(gate_passed=True)
        self.assertEqual(code, 0)
        self.assertIn("Result: PASS (profile standard)", out)
        self.assertNotIn("WITH WARNINGS", out)
        report = self.read_report()
        self.assertNotIn("packed_map_suspicion",
                         report["conclusion"])
        self.assertNotIn("packed-map suspicion", self.read_md())


# ---------------------------------------------------------------------------
# O5: unsupported files are excluded explicitly
# ---------------------------------------------------------------------------
class UnsupportedFilesTest(FlowBase):
    extras = ("emissive.png", "height.png")

    def test_resolution_records_unconsumed_images(self):
        spec = resolve_inputs(self.material,
                              normal_convention="opengl")
        self.assertEqual(spec.unsupported_files,
                         ["emissive.png", "height.png"])

    def test_teacher_warnings_state_the_exclusion(self):
        spec = resolve_inputs(self.material,
                              normal_convention="opengl")
        result = build_teacher(spec)
        joined = "\n".join(result.warnings)
        self.assertIn("unsupported map excluded: emissive.png",
                      joined)
        self.assertIn("unsupported map excluded: height.png", joined)

    def test_preflight_states_the_exclusion(self):
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            code = main(["preflight", str(self.material),
                         "--normal-convention", "opengl",
                         "--allow-capacity-expansion"])
        self.assertEqual(code, 0)
        out = buf.getvalue()
        self.assertIn("unsupported map excluded: emissive.png", out)
        self.assertIn("unsupported map excluded: height.png", out)

    def test_train_and_report_state_the_exclusion(self):
        code, out = self.run_train(gate_passed=True)
        self.assertEqual(code, 0)
        self.assertIn("unsupported map excluded: emissive.png", out)
        report = self.read_report()
        self.assertEqual(report["input"]["unsupported_files"],
                         ["emissive.png", "height.png"])
        self.assertIn("Unsupported file excluded: emissive.png",
                      self.read_md())


class NoExtrasControlTest(FlowBase):
    def test_no_unsupported_records_without_extra_files(self):
        spec = resolve_inputs(self.material,
                              normal_convention="opengl")
        self.assertEqual(spec.unsupported_files, [])
        code, _out = self.run_train(gate_passed=True)
        self.assertEqual(code, 0)
        self.assertNotIn("unsupported_files",
                         self.read_report()["input"])


# ---------------------------------------------------------------------------
# O4: --fast-eval
# ---------------------------------------------------------------------------
class FastEvalParserTest(unittest.TestCase):
    def test_fast_eval_defaults_off(self):
        args = build_parser().parse_args(["train", "some-material"])
        self.assertFalse(args.fast_eval)

    def test_fast_eval_flag_sets_true(self):
        args = build_parser().parse_args(
            ["train", "some-material", "--fast-eval"])
        self.assertTrue(args.fast_eval)

    def test_resume_params_record_fast_eval_only_when_on(self):
        off = auto.normalized_run_params(1337, "opengl")
        self.assertNotIn("fast_eval", off)
        on = auto.normalized_run_params(1337, "opengl", fast_eval=True)
        self.assertEqual(on["fast_eval"], True)
        self.assertEqual(
            {k: v for k, v in on.items() if k != "fast_eval"}, off)


class FastEvalFlowTest(FlowBase):
    def test_fast_eval_skips_sheets_and_records_itself(self):
        code, out = self.run_train("--fast-eval", gate_passed=True)
        self.assertEqual(code, 0)
        self.assertEqual(self.visual_calls, [])
        self.assertTrue(self.seen_options[0].fast_eval)
        self.assertIn("Visual sheets: skipped (--fast-eval)", out)
        report = self.read_report()
        self.assertTrue(report["visual_check"]["skipped"])
        self.assertEqual(report["trial"]["fast_eval"], True)
        self.assertIn("Skipped (--fast-eval)", self.read_md())
        # No sheet files were written.
        self.assertFalse(
            (self.out / "oak" / "oak_sheet.png").exists())

    def test_default_run_keeps_sheets_and_records_nothing(self):
        code, out = self.run_train(gate_passed=True)
        self.assertEqual(code, 0)
        self.assertEqual(len(self.visual_calls), 1)
        self.assertFalse(self.seen_options[0].fast_eval)
        report = self.read_report()
        self.assertNotIn("fast_eval", report["trial"])
        self.assertNotIn("skipped", report["visual_check"])


@unittest.skipUnless(TORCH_AVAILABLE, "torch not installed")
class FastEvalDriverTest(unittest.TestCase):
    """One tiny real trial: --fast-eval must not call the multi-mip
    measurement at all, and must not change the gate evaluation."""

    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp(prefix="vnm1-neon-driver-"))
        self.addCleanup(shutil.rmtree, self.tmp, True)
        self.material = make_material(self.tmp / "tiny", size=64)
        spec = resolve_inputs(self.material,
                              normal_convention="opengl")
        self.teacher = build_teacher(spec)
        self.sources = {k: self.teacher.records[k].source
                        for k in self.teacher.records}

    def _options(self, work, *, fast_eval):
        return driver.TrialOptions(
            material_name="tiny", profile="standard", iterations=2,
            device="cpu", work_dir=work, fast_eval=fast_eval)

    def test_fast_eval_never_calls_evaluate_mips(self):
        with mock.patch.object(
                driver, "evaluate_mips",
                side_effect=AssertionError("must be skipped")):
            outcome = driver.run_trial(
                self.teacher.teacher, self.sources,
                self._options(self.tmp / "w-fast", fast_eval=True))
        self.assertIsNone(outcome.mip_evaluation)
        self.assertIsNone(outcome.mip_advisory)
        self.assertIn("psnr_base_color_db", outcome.metrics)

    def test_default_run_calls_evaluate_mips(self):
        calls = []
        real = driver.evaluate_mips

        def counting(*a, **kw):
            calls.append(1)
            return real(*a, **kw)

        with mock.patch.object(driver, "evaluate_mips", counting):
            outcome = driver.run_trial(
                self.teacher.teacher, self.sources,
                self._options(self.tmp / "w-full", fast_eval=False))
        self.assertEqual(len(calls), 1)
        self.assertIn("psnr_base_color_db", outcome.metrics)


class ReportMdFallbackTest(unittest.TestCase):
    """report.md renders the banner even for a report dict that only
    carries the suspicion in the input warnings (older reports)."""

    def test_banner_from_input_warnings(self):
        report = {
            "material": "oak",
            "conclusion": {"gate_passed": True,
                           "selected_profile": "standard",
                           "failed_checks": []},
            "trial": {"profile": "standard", "iterations": 10,
                      "checks": {}, "device": "cpu",
                      "elapsed_seconds": 1.0},
            "input": {"warnings": [
                "WARNING (packed-map suspicion): roughness: "
                "rough.png has differing RGB channels"]},
            "capacity": None,
        }
        md = render_report_markdown(report)
        self.assertIn("**PASS WITH WARNINGS: packed-map suspicion**",
                      md)


if __name__ == "__main__":
    unittest.main()

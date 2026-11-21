"""Tests for the capacity entry check + outcome warnings
(vnm1.capacity_check, 2026-11-19 addition).

Covers: the artifact-size estimate against the byte counts of real
artifacts from the P2 handoff package; the entry verdict on the
shapes that motivated the feature (tiny single-map inputs rejected,
a 2K full-map input passing, the prototype shape passing with a
thin margin); the --allow-capacity-expansion override at the gate,
preflight and report levels; and the outcome-warning thresholds
(0.9 thin margin / 1.0 expansion).
"""

import contextlib
import io
import shutil
import tempfile
import unittest
from pathlib import Path

import numpy as np
from PIL import Image

from vnm1 import cli, driver, report as report_mod
from vnm1.capacity_check import (capacity_entry_check,
                                 capacity_warnings,
                                 estimate_artifact_bytes)
from vnm1.imaging import build_teacher
from vnm1.inputs import resolve_inputs
from vnm1.report_md import render_report_markdown

ALL_MAP = {"base_color": "map", "normal": "map", "roughness": "map",
           "metalness": "map", "ao": "map"}
BASE_ONLY = {"base_color": "map", "normal": "constant",
             "roughness": "constant", "metalness": "constant",
             "ao": "constant"}
SNOW_LIKE = {"base_color": "map", "normal": "map",
             "roughness": "map", "metalness": "constant",
             "ao": "map"}


class EstimateTest(unittest.TestCase):
    """The estimate is derived from the format parameters via the
    real serializers; these totals are the byte-exact sizes of the
    real artifacts shipped in the P2 handoff package (same material
    names, so even the manifest matches)."""

    def test_standard_matches_real_marble(self):
        est = estimate_artifact_bytes("standard", 2048, 2048,
                                      "marble")
        self.assertEqual(est["total_bytes"], 1318407)
        self.assertEqual(est["breakdown_bytes"], {
            "manifest.json": 1111, "weights.bin": 4464,
            "level-0.bin": 1310752, "tail.bin": 2080})

    def test_standard8_matches_real_fabric(self):
        est = estimate_artifact_bytes(
            "standard8", 1024, 1024, "ambientcg-fabric030-1k")
        self.assertEqual(est["total_bytes"], 2629912)

    def test_quality8_matches_real_sand(self):
        est = estimate_artifact_bytes("quality8", 2048, 2048, "sand")
        self.assertEqual(est["total_bytes"], 10496968)

    def test_profile_sizes_grow_monotonically(self):
        totals = [estimate_artifact_bytes(p, 2048, 2048, "m")
                  ["total_bytes"]
                  for p in ("standard", "standard8", "quality",
                            "quality8")]
        self.assertEqual(totals, sorted(totals))
        self.assertLess(totals[0], totals[-1])


class EntryCheckTest(unittest.TestCase):
    def test_tiny_single_map_rejected(self):
        entry = capacity_entry_check(64, 64, BASE_ONLY, "retro")
        self.assertTrue(entry["would_expand"])
        self.assertGreater(
            entry["estimated_ratio_vs_bcn_mip_chain"], 100.0)

    def test_256_single_map_rejected(self):
        entry = capacity_entry_check(256, 256, BASE_ONLY, "pattern")
        self.assertTrue(entry["would_expand"])

    def test_tilesheet_rejected(self):
        entry = capacity_entry_check(768, 512, BASE_ONLY, "road")
        self.assertTrue(entry["would_expand"])

    def test_prototype_shape_passes_thin(self):
        # 1024x1024 base color only: standard just barely beats BCn
        # (measured ratio 0.943 in the step E run).
        entry = capacity_entry_check(1024, 1024, BASE_ONLY, "proto")
        self.assertFalse(entry["would_expand"])
        self.assertGreaterEqual(
            entry["estimated_ratio_vs_bcn_mip_chain"], 0.9)
        self.assertLess(
            entry["estimated_ratio_vs_bcn_mip_chain"], 1.0)

    def test_2k_full_maps_pass_clean(self):
        entry = capacity_entry_check(2048, 2048, ALL_MAP, "metal")
        self.assertFalse(entry["would_expand"])
        self.assertLess(
            entry["estimated_ratio_vs_bcn_mip_chain"], 0.1)

    def test_snow_shape_passes_entry(self):
        # Snow015's shape: the entry check passes at standard; its
        # measured expansion came from needing quality to pass the
        # quality gate, which is the outcome warning's job.
        entry = capacity_entry_check(1024, 1024, SNOW_LIKE, "snow")
        self.assertFalse(entry["would_expand"])
        self.assertLess(
            entry["estimated_ratio_vs_bcn_mip_chain"], 0.9)


class WarningsTest(unittest.TestCase):
    def test_below_thin_margin_no_warning(self):
        self.assertEqual(capacity_warnings(0.08), [])
        self.assertEqual(capacity_warnings(0.8999), [])
        self.assertEqual(capacity_warnings(None), [])

    def test_thin_margin_threshold(self):
        for ratio in (0.9, 0.943, 0.9999):
            warnings = capacity_warnings(ratio)
            self.assertEqual(len(warnings), 1)
            self.assertEqual(warnings[0]["kind"],
                             "capacity_thin_margin")

    def test_expansion_threshold(self):
        for ratio in (1.0, 1.2524, 240.2):
            warnings = capacity_warnings(ratio)
            self.assertEqual(len(warnings), 1)
            self.assertEqual(warnings[0]["kind"],
                             "capacity_expansion")


def _make_folder(tmp: Path, name: str, size: int, *, full: bool):
    folder = tmp / name
    folder.mkdir(parents=True)
    rng = np.random.default_rng(7)
    Image.fromarray(
        rng.integers(0, 255, size=(size, size, 3), dtype=np.uint8)
    ).save(folder / "basecolor.png")
    if full:
        Image.fromarray(
            rng.integers(0, 255, size=(size, size, 3),
                         dtype=np.uint8)).save(folder / "normal.png")
        for logical in ("roughness", "metalness", "ao"):
            Image.fromarray(
                np.full((size, size), 128, dtype=np.uint8)).save(
                    folder / f"{logical}.png")
    return folder


def _run_cli(argv):
    out, err = io.StringIO(), io.StringIO()
    with contextlib.redirect_stdout(out), \
            contextlib.redirect_stderr(err):
        code = cli.main(argv)
    return code, out.getvalue(), err.getvalue()


class CliGateTest(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp(prefix="vnm1-cap-test-"))
        self.addCleanup(shutil.rmtree, self.tmp, True)

    def test_preflight_rejects_tiny_single_map(self):
        folder = _make_folder(self.tmp, "tiny", 64, full=False)
        code, _out, err = _run_cli(
            ["preflight", str(folder), "--normal-convention",
             "opengl"])
        self.assertEqual(code, cli.EXIT_INPUT_ERROR)
        self.assertIn("capacity entry check", err)

    def test_preflight_override_passes_tiny(self):
        folder = _make_folder(self.tmp, "tiny", 64, full=False)
        code, out, _err = _run_cli(
            ["preflight", str(folder), "--normal-convention",
             "opengl", "--allow-capacity-expansion"])
        self.assertEqual(code, 0)
        self.assertIn("overridden by --allow-capacity-expansion",
                      out)

    def test_preflight_passes_2k_full_maps(self):
        folder = _make_folder(self.tmp, "big", 2048, full=True)
        code, out, _err = _run_cli(
            ["preflight", str(folder), "--normal-convention",
             "opengl"])
        self.assertEqual(code, 0)
        self.assertIn("Capacity entry check", out)

    def test_train_rejects_tiny_in_both_modes(self):
        folder = _make_folder(self.tmp, "tiny", 64, full=False)
        for mode in ("manual", "auto"):
            code, _out, err = _run_cli(
                ["train", str(folder), "--normal-convention",
                 "opengl", "--mode", mode,
                 "--allow-unverified-size",
                 "--output", str(self.tmp / "dist")])
            self.assertEqual(code, cli.EXIT_INPUT_ERROR, mode)
            self.assertIn("capacity entry check", err, mode)

    def test_gate_record_marks_override(self):
        folder = _make_folder(self.tmp, "tiny", 64, full=False)
        spec = resolve_inputs(folder, normal_convention="opengl")
        teacher = build_teacher(spec)
        record, error = cli._capacity_entry_gate(
            spec, teacher, allow=False)
        self.assertIsNotNone(error)
        self.assertFalse(record["override"])
        self.assertTrue(record["would_expand"])
        record, error = cli._capacity_entry_gate(
            spec, teacher, allow=True)
        self.assertIsNone(error)
        self.assertTrue(record["override"])


def _outcome(ratio):
    metrics = {"psnr_base_color_db": 38.9,
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
    capacity = {"vnm1_total_bytes": 100,
                "bcn": {"per_map": {}, "total_mip0_bytes": 100,
                        "total_mip_chain_bytes": 100},
                "ratio_vs_bcn_mip_chain": ratio,
                "ratio_vs_bcn_mip0_reference": ratio}
    return driver.TrialOutcome(
        profile="standard", fine_size=(512, 512),
        coarse_size=(256, 256), fine_channels=4, coarse_channels=4,
        metrics=metrics, checks=verdict["checks"],
        gate_passed=verdict["passed"], capacity=capacity,
        recipe={"recipe_version": 2},
        environment={"device": "cuda", "gpu_name": "T4"},
        started_at="t0", finished_at="t1", elapsed_seconds=1.0,
        training_seconds=0.9, final_loss=0.01,
        artifact_dir=Path("/nonexistent"), artifact_files={},
        manifest={}, roundtrip={"weights_parse_ok": True})


class ReportWarningTest(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp(prefix="vnm1-cap-rep-"))
        self.addCleanup(shutil.rmtree, self.tmp, True)
        folder = _make_folder(self.tmp, "oak", 16, full=False)
        self.spec = resolve_inputs(folder,
                                   normal_convention="opengl")
        self.teacher = build_teacher(self.spec)
        self.params = {"profile": "standard", "iterations": 50000,
                       "color_weight": 4.0, "scalar_weight": 1.0,
                       "seed": 1337}

    def _report(self, ratio, capacity_check=None):
        return report_mod.build_report(
            self.spec, self.teacher, _outcome(ratio), mode="manual",
            artifact_location="/dist/oak/oak.vnm",
            trial_params=self.params,
            capacity_check=capacity_check)

    def test_expansion_warning_in_json_and_markdown(self):
        rep = self._report(1.2524)
        warnings = rep["conclusion"]["capacity_warnings"]
        self.assertEqual([w["kind"] for w in warnings],
                         ["capacity_expansion"])
        md = render_report_markdown(rep)
        self.assertIn("Capacity warning:", md)
        self.assertIn("larger than the BCn", md)

    def test_thin_margin_warning(self):
        rep = self._report(0.95)
        warnings = rep["conclusion"]["capacity_warnings"]
        self.assertEqual([w["kind"] for w in warnings],
                         ["capacity_thin_margin"])
        self.assertIn("Capacity warning:",
                      render_report_markdown(rep))

    def test_clean_ratio_no_warning(self):
        rep = self._report(0.08)
        self.assertEqual(rep["conclusion"]["capacity_warnings"], [])
        self.assertIsNone(rep["capacity_check"])
        self.assertNotIn("Capacity warning:",
                         render_report_markdown(rep))

    def test_entry_record_and_override_in_report(self):
        record = {"entry_profile": "standard",
                  "estimated_vnm1_total_bytes": 1318407,
                  "estimated_breakdown_bytes": {},
                  "bcn_mip_chain_bytes": 5461,
                  "estimated_ratio_vs_bcn_mip_chain": 241.4,
                  "would_expand": True, "override": True}
        rep = self._report(240.2, capacity_check=record)
        self.assertTrue(rep["capacity_check"]["override"])
        md = render_report_markdown(rep)
        self.assertIn("--allow-capacity-expansion", md)


if __name__ == "__main__":
    unittest.main()

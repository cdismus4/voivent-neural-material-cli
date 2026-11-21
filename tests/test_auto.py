"""Unit tests for auto mode (step C, design doc section 6.2).

The decision logic (ladder stop, adjustment firing and pattern
selection, representative tie-break, resume hash) is tested directly;
the orchestration is tested through ``vnm1.auto.run_auto`` with a fake
trial runner, and the CLI wiring with a monkeypatched
``vnm1.driver.run_trial`` -- no torch training anywhere.  The real
trials are exercised on Kaggle (see ../manifest.json).
"""

import hashlib
import json
import shutil
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import numpy as np
from PIL import Image

from vnm1 import auto, driver
from vnm1.cli import main
from vnm1.errors import TrainingError
from vnm1.inputs import LOGICAL_MAPS

SOURCES = {"base_color": "map", "normal": "map", "roughness": "map",
           "metalness": "constant", "ao": "map"}


def metrics(psnr=40.0, rough=0.010, metal=0.0, ao=0.010,
            n_mean=1.0, n_p95=2.0):
    return {"psnr_base_color_db": psnr,
            "psnr_base_color_db_srgb_space_reference": 30.0,
            "mae": {"roughness": rough, "metalness": metal, "ao": ao},
            "normal_angle_mean_deg": n_mean,
            "normal_angle_p95_deg": n_p95,
            "texel_count": 1024, "eval_seconds": 0.1,
            "decode_texels_per_second": 10000.0}


def checks_of(m):
    return driver.gate_verdict(m, SOURCES)["checks"]


PASS_M = metrics()
FAIL_FAR_M = metrics(psnr=28.0, rough=0.05, ao=0.04, n_mean=5.0,
                     n_p95=12.0)
COLOR_NARROW_M = metrics(psnr=34.4, n_mean=2.3, n_p95=5.9)
SCALAR_NARROW_M = metrics(psnr=44.0, ao=0.025)
BOTH_NARROW_M = metrics(psnr=34.38, ao=0.0219, n_mean=1.96, n_p95=4.86)
COLOR_FAR_SCALAR_NARROW_M = metrics(psnr=30.20, ao=0.0271,
                                    n_mean=2.63, n_p95=6.57)


def fake_outcome(profile, m, work_dir, material="oak"):
    verdict = driver.gate_verdict(m, SOURCES)
    staging = work_dir / "staging" / f"{material}.vnm"
    staging.mkdir(parents=True, exist_ok=True)
    files = {}
    for name in ("manifest.json", "weights.bin", "level-0.bin",
                 "tail.bin"):
        data = f"fake-{profile}-{name}".encode()
        (staging / name).write_bytes(data)
        files[name] = {"bytes": len(data),
                       "sha256": hashlib.sha256(data).hexdigest()}
    return driver.TrialOutcome(
        profile=profile, fine_size=(512, 512), coarse_size=(256, 256),
        fine_channels=4, coarse_channels=4,
        metrics=m, checks=verdict["checks"],
        gate_passed=verdict["passed"],
        capacity={"vnm1_breakdown_bytes": {}, "vnm1_total_bytes": 57,
                  "bcn": {"per_map": {}, "total_mip0_bytes": 100,
                          "total_mip_chain_bytes": 120},
                  "ratio_vs_bcn_mip_chain": 57 / 120,
                  "ratio_vs_bcn_mip0_reference": 57 / 100},
        recipe={"recipe_version": 2},
        environment={"device": "cpu", "cpu_fallback": True},
        started_at="2026-11-19T00:00:00+00:00",
        finished_at="2026-11-19T00:01:00+00:00",
        elapsed_seconds=60.0, training_seconds=55.0, final_loss=0.01,
        artifact_dir=staging, artifact_files=files,
        manifest={"format": "VNM1"},
        roundtrip={"weights_parse_ok": True})


def make_spec_teacher(*, seed_maps=None):
    records = {}
    for logical in LOGICAL_MAPS:
        if logical == "metalness":
            records[logical] = SimpleNamespace(
                source="constant", constant_value=0.0, sha256=None,
                file=None, gate_exempt=True)
        else:
            records[logical] = SimpleNamespace(
                source="map", constant_value=None,
                sha256=hashlib.sha256(f"bytes-{logical}".encode())
                .hexdigest(),
                file=f"/some/dir/{logical}.png", gate_exempt=False)
    if seed_maps:
        for logical, sha in seed_maps.items():
            records[logical].sha256 = sha
    spec = SimpleNamespace(name="oak", normal_convention="opengl")
    teacher = SimpleNamespace(records=records, width=2048, height=2048)
    return spec, teacher


class FakeRunner:
    """trial_runner stand-in: outcomes keyed by (kind, profile)."""

    def __init__(self, outcomes, *, raises_at=None):
        self.outcomes = outcomes
        self.raises_at = raises_at
        self.calls = []

    def __call__(self, trial_def, trial_work_dir):
        self.calls.append(dict(trial_def))
        key = (trial_def["kind"], trial_def["profile"])
        if self.raises_at == key:
            raise TrainingError("simulated divergence")
        return fake_outcome(trial_def["profile"],
                            self.outcomes[key], trial_work_dir)


class AdjustmentDecisionTest(unittest.TestCase):
    def decide(self, m):
        return auto.adjustment_decision(checks_of(m))

    def test_color_only_narrow(self):
        d = self.decide(COLOR_NARROW_M)
        self.assertTrue(d["eligible"])
        self.assertEqual(d["pattern"], "color_only")
        self.assertEqual((d["color_weight"], d["scalar_weight"],
                          d["iterations"]), (6.0, 1.0, 150000))

    def test_scalar_only_narrow(self):
        d = self.decide(SCALAR_NARROW_M)
        self.assertTrue(d["eligible"])
        self.assertEqual(d["pattern"], "scalar_only")
        self.assertEqual((d["color_weight"], d["scalar_weight"],
                          d["iterations"]), (4.0, 3.0, 50000))

    def test_both_narrow_sand_pattern(self):
        d = self.decide(BOTH_NARROW_M)
        self.assertTrue(d["eligible"])
        self.assertEqual(d["pattern"], "color_and_scalar")
        self.assertEqual((d["color_weight"], d["scalar_weight"],
                          d["iterations"]), (6.0, 3.0, 150000))

    def test_leaves_color_not_narrow(self):
        d = self.decide(COLOR_FAR_SCALAR_NARROW_M)
        self.assertFalse(d["eligible"])
        self.assertEqual(d["reason"], "color_not_narrow")

    def test_scalar_too_wide(self):
        d = self.decide(metrics(psnr=44.0, ao=0.035))
        self.assertFalse(d["eligible"])
        self.assertEqual(d["reason"], "scalar_not_narrow")

    def test_normal_failure_blocks_even_when_color_narrow(self):
        d = self.decide(metrics(psnr=34.4, n_mean=3.4))
        self.assertFalse(d["eligible"])
        self.assertEqual(d["reason"], "normal_not_passing")

    def test_boundaries_are_narrow(self):
        d = self.decide(metrics(psnr=33.5, ao=0.030))
        self.assertTrue(d["eligible"])
        self.assertEqual(d["pattern"], "color_and_scalar")

    def test_gate_passed_is_not_eligible(self):
        d = self.decide(PASS_M)
        self.assertFalse(d["eligible"])
        self.assertEqual(d["reason"], "gate_passed")


class TieBreakTest(unittest.TestCase):
    def rec(self, profile, m):
        return {"profile": profile, "checks": checks_of(m)}

    def test_fewest_failing_checks_wins(self):
        a = self.rec("quality8", FAIL_FAR_M)
        b = self.rec("standard", COLOR_NARROW_M)
        self.assertIs(auto.pick_representative([a, b]), b)

    def test_smallest_max_overshoot_wins(self):
        a = self.rec("quality8", metrics(psnr=30.0))
        b = self.rec("standard", metrics(psnr=33.0))
        self.assertIs(auto.pick_representative([a, b]), b)

    def test_higher_profile_wins_on_full_tie(self):
        a = self.rec("standard", COLOR_NARROW_M)
        b = self.rec("quality8", COLOR_NARROW_M)
        self.assertIs(auto.pick_representative([a, b]), b)

    def test_later_trial_wins_same_profile_tie(self):
        a = self.rec("quality8", BOTH_NARROW_M)
        b = self.rec("quality8", BOTH_NARROW_M)
        self.assertIs(auto.pick_representative([a, b]), b)


class ResumeHashTest(unittest.TestCase):
    def test_path_spelling_does_not_change_hash(self):
        spec, teacher = make_spec_teacher()
        h1 = auto.compute_resume_hash(spec, teacher, seed=1337)
        for logical in LOGICAL_MAPS:
            rec = teacher.records[logical]
            if rec.file:
                rec.file = f"relative/other/{logical}.png"
        h2 = auto.compute_resume_hash(spec, teacher, seed=1337)
        self.assertEqual(h1, h2)

    def test_changed_map_bytes_change_hash(self):
        spec, teacher = make_spec_teacher()
        h1 = auto.compute_resume_hash(spec, teacher, seed=1337)
        teacher.records["ao"].sha256 = "0" * 64
        self.assertNotEqual(
            h1, auto.compute_resume_hash(spec, teacher, seed=1337))

    def test_changed_seed_changes_hash(self):
        spec, teacher = make_spec_teacher()
        self.assertNotEqual(
            auto.compute_resume_hash(spec, teacher, seed=1337),
            auto.compute_resume_hash(spec, teacher, seed=999))

    def test_changed_convention_changes_hash(self):
        spec, teacher = make_spec_teacher()
        h1 = auto.compute_resume_hash(spec, teacher, seed=1337)
        spec.normal_convention = "directx"
        self.assertNotEqual(
            h1, auto.compute_resume_hash(spec, teacher, seed=1337))

    def test_changed_constant_changes_hash(self):
        spec, teacher = make_spec_teacher()
        h1 = auto.compute_resume_hash(spec, teacher, seed=1337)
        teacher.records["metalness"].constant_value = 1.0
        self.assertNotEqual(
            h1, auto.compute_resume_hash(spec, teacher, seed=1337))


class RunAutoTest(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp(prefix="vnm1-auto-test-"))
        self.addCleanup(shutil.rmtree, self.tmp, True)
        self.spec, self.teacher = make_spec_teacher()
        self.out = self.tmp / "dist"

    def run_it(self, runner, **kw):
        return auto.run_auto(spec=self.spec, teacher=self.teacher,
                             output_dir=self.out, seed=1337,
                             trial_runner=runner, **kw)

    def test_ladder_stops_at_first_pass(self):
        runner = FakeRunner({
            ("ladder", "standard"): FAIL_FAR_M,
            ("ladder", "standard8"): PASS_M,
        })
        result = self.run_it(runner)
        self.assertEqual([c["profile"] for c in runner.calls],
                         ["standard", "standard8"])
        self.assertEqual(result.status, "passed")
        self.assertEqual(result.selected["profile"], "standard8")
        self.assertTrue((self.out / "oak" / "oak.vnm"
                         / "manifest.json").is_file())
        attempt = self.out / "oak" / "work" / "attempts" / \
            "trial-001-standard"
        self.assertTrue((attempt / "manifest.json").is_file())

    def test_adjustment_fires_with_sand_pattern_and_passes(self):
        runner = FakeRunner({
            ("ladder", "standard"): FAIL_FAR_M,
            ("ladder", "standard8"): FAIL_FAR_M,
            ("ladder", "quality"): FAIL_FAR_M,
            ("ladder", "quality8"): BOTH_NARROW_M,
            ("adjustment", "quality8"): PASS_M,
        })
        result = self.run_it(runner)
        self.assertEqual(len(runner.calls), 5)
        adj = runner.calls[4]
        self.assertEqual(adj["kind"], "adjustment")
        self.assertEqual((adj["color_weight"], adj["scalar_weight"],
                          adj["iterations"]), (6.0, 3.0, 150000))
        self.assertEqual(result.status, "passed")
        self.assertEqual(result.selected["kind"], "adjustment")
        self.assertTrue(result.adjustment["ran"])

    def test_leaves_like_no_adjustment_representative_quality8(self):
        runner = FakeRunner({
            ("ladder", "standard"): FAIL_FAR_M,
            ("ladder", "standard8"): FAIL_FAR_M,
            ("ladder", "quality"): FAIL_FAR_M,
            ("ladder", "quality8"): COLOR_FAR_SCALAR_NARROW_M,
        })
        result = self.run_it(runner)
        self.assertEqual(len(runner.calls), 4)
        self.assertEqual(result.status, "failed")
        self.assertEqual(result.representative["profile"], "quality8")
        self.assertEqual(result.adjustment["reason"],
                         "color_not_narrow")
        self.assertFalse((self.out / "oak" / "oak.vnm").exists())
        state = json.loads(result.state_path.read_text())
        self.assertEqual(state["decision"]["status"], "failed")
        self.assertEqual(state["decision"]["representative_profile"],
                         "quality8")

    def test_resume_skips_evaluated_trials(self):
        runner1 = FakeRunner(
            {("ladder", "standard"): FAIL_FAR_M},
            raises_at=("ladder", "standard8"))
        with self.assertRaises(TrainingError):
            self.run_it(runner1)
        state = json.loads(
            (self.out / "oak" / "work" / "auto_state.json").read_text())
        self.assertEqual(state["trials"][0]["status"], "failed")
        self.assertEqual(state["trials"][1]["status"], "error")

        runner2 = FakeRunner({("ladder", "standard8"): PASS_M})
        result = self.run_it(runner2)
        self.assertEqual([c["profile"] for c in runner2.calls],
                         ["standard8"])
        self.assertEqual(result.resumed_trials, 1)
        self.assertEqual(result.status, "passed")

    def test_resume_of_completed_run_reruns_nothing(self):
        runner = FakeRunner({("ladder", "standard"): PASS_M})
        first = self.run_it(runner)
        self.assertEqual(first.status, "passed")
        runner2 = FakeRunner({})
        second = self.run_it(runner2)
        self.assertEqual(runner2.calls, [])
        self.assertEqual(second.status, "passed")
        self.assertEqual(second.resumed_trials, 1)

    def test_hash_mismatch_archives_state_and_reruns(self):
        runner = FakeRunner({("ladder", "standard"): PASS_M})
        self.run_it(runner)
        runner2 = FakeRunner({("ladder", "standard"): PASS_M})
        auto.run_auto(spec=self.spec, teacher=self.teacher,
                      output_dir=self.out, seed=999,
                      trial_runner=runner2)
        self.assertEqual(len(runner2.calls), 1)
        archived = list((self.out / "oak" / "work").glob(
            "auto_state.superseded-*.json"))
        self.assertEqual(len(archived), 1)


# ---------------------------------------------------------------------------
# CLI-level tests on a tiny synthetic material
# ---------------------------------------------------------------------------
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


def fake_visual_sheets(artifact_dir, teacher, material_name, out_dir,
                       *, expected_psnr_db):
    """Stand-in for vnm1.visual.generate_visual_sheets (step D): the
    fake artifacts are not decodable, so the sheet step is faked the
    same way the trial itself is.  Real sheet generation is covered
    by tests/test_visual.py."""
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


class CliAutoTest(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp(prefix="vnm1-cli-auto-"))
        self.addCleanup(shutil.rmtree, self.tmp, True)
        self.material = make_material(self.tmp / "oak")
        self.out = self.tmp / "dist"

    def argv(self, *extra):
        # --allow-capacity-expansion: as in test_train -- the fixture
        # is a tiny synthetic stand-in; the capacity entry check
        # (2026-11-19) is covered by test_capacity_check.
        return ["train", str(self.material), "--mode", "auto",
                "--normal-convention", "opengl",
                "--allow-unverified-size",
                "--allow-capacity-expansion",
                "--output", str(self.out),
                *extra]

    def patch_status(self):
        status = {"python_version": "3.12.3",
                  "python_implementation": "CPython", "os": "Linux",
                  "torch_available": True, "torch_version": "2.x",
                  "cuda_available": False, "gpu_name": None}
        return mock.patch.object(driver, "torch_status",
                                 return_value=status)

    def patch_trials(self, outcomes_by_profile):
        calls = []

        def run_trial(teacher, channel_sources, options,
                      progress_cb=None):
            calls.append(options.profile)
            staging_holder = Path(options.work_dir)
            return fake_outcome(
                options.profile,
                outcomes_by_profile[options.profile],
                staging_holder, material=options.material_name)

        patcher = mock.patch.object(driver, "run_trial", run_trial)
        return patcher, calls

    def test_conflicting_manual_option_is_exit_2(self):
        self.assertEqual(main(self.argv("--profile", "standard")), 2)
        self.assertEqual(main(self.argv("--iterations", "100")), 2)
        self.assertEqual(main(self.argv("--color-weight", "6")), 2)

    def test_auto_pass_at_standard8_end_to_end(self):
        patcher, calls = self.patch_trials(
            {"standard": FAIL_FAR_M, "standard8": PASS_M})
        with self.patch_status(), patcher, patch_visual():
            self.assertEqual(main(self.argv()), 0)
        self.assertEqual(calls, ["standard", "standard8"])
        self.assertTrue((self.out / "oak" / "oak.vnm"
                         / "manifest.json").is_file())
        report = json.loads(
            (self.out / "oak" / "oak.report.json").read_text())
        self.assertEqual(report["conclusion"]["mode"], "auto")
        self.assertEqual(report["conclusion"]["selected_profile"],
                         "standard8")
        self.assertEqual(len(report["trials"]), 2)
        self.assertTrue(report["visual_check"]["psnr_match"])
        self.assertIsNotNone(report["conclusion"]["chosen_trial_index"])
        md = (self.out / "oak" / "oak.report.md").read_text()
        self.assertIn("Result: PASS", md)
        self.assertIn("Selected profile: standard8", md)
        self.assertTrue(
            (self.out / "oak" / "work" / "auto_state.json").is_file())
        # Second invocation resumes from state: no trial runs again.
        patcher2, calls2 = self.patch_trials({})
        with self.patch_status(), patcher2, patch_visual():
            self.assertEqual(main(self.argv()), 0)
        self.assertEqual(calls2, [])

    def test_auto_fail_exit_3_representative_in_report(self):
        patcher, calls = self.patch_trials({
            "standard": FAIL_FAR_M, "standard8": FAIL_FAR_M,
            "quality": FAIL_FAR_M,
            "quality8": COLOR_FAR_SCALAR_NARROW_M})
        with self.patch_status(), patcher, patch_visual():
            self.assertEqual(main(self.argv()), 3)
        self.assertEqual(len(calls), 4)
        self.assertFalse((self.out / "oak" / "oak.vnm").exists())
        report = json.loads(
            (self.out / "oak" / "oak.report.json").read_text())
        self.assertEqual(report["conclusion"]["result"], "failed")
        self.assertEqual(report["conclusion"]["representative_profile"],
                         "quality8")
        self.assertTrue(report["conclusion"]
                        ["representative_is_candidate_only"])
        self.assertEqual(
            report["conclusion"]["adjustment"]["reason"],
            "color_not_narrow")
        md = (self.out / "oak" / "oak.report.md").read_text()
        self.assertIn("Result: FAIL", md)
        self.assertIn("not a passing artifact", md)
        self.assertIn("Adjustment: not run", md)


if __name__ == "__main__":
    unittest.main()

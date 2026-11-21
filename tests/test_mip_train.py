"""Unit tests for the 2026-11-21 multi-mip addition (batch5
proposal 1): the ``--mip-train`` opt-in, the SPEC 5.3 teacher
pyramid, the mip sampling distribution, the manifest ``mipMax``
field, the far-field advisory decision (threshold boundaries), the
report / report.md mip records, and one short real mip-trained run
whose artifact must read back through the product decoder.

The advisory tests are pure (fabricated metric dicts); the real-run
test is CPU-short and skipped without torch, like test_visual's.
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

from vnm1 import auto, core as vt, driver, report as report_mod
from vnm1.cli import build_parser, main
from vnm1.imaging import build_teacher
from vnm1.inputs import resolve_inputs
from vnm1.report_md import render_report_markdown

TORCH_AVAILABLE = vt.torch is not None


# ---------------------------------------------------------------------------
# Option parsing / resume-hash plumbing
# ---------------------------------------------------------------------------
class ParserTest(unittest.TestCase):
    def test_mip_train_defaults_off(self):
        args = build_parser().parse_args(["train", "some-material"])
        self.assertFalse(args.mip_train)

    def test_mip_train_flag_sets_true(self):
        args = build_parser().parse_args(
            ["train", "some-material", "--mip-train"])
        self.assertTrue(args.mip_train)

    def test_resume_params_record_mip_train_only_when_on(self):
        off = auto.normalized_run_params(1337, "opengl")
        self.assertNotIn("mip_train", off)
        on = auto.normalized_run_params(1337, "opengl", mip_train=True)
        self.assertEqual(on["mip_train"], True)
        # Everything else identical: the flag is the only difference
        # a mip-trained auto run's resume hash sees.
        self.assertEqual(
            {k: v for k, v in on.items() if k != "mip_train"}, off)


# ---------------------------------------------------------------------------
# Teacher pyramid (SPEC 5.3)
# ---------------------------------------------------------------------------
class TeacherPyramidTest(unittest.TestCase):
    def _teacher(self, size=8):
        teacher = np.zeros((size, size, 8), dtype=np.float32)
        yy, xx = np.mgrid[0:size, 0:size]
        teacher[..., 0] = xx / size            # ramp in x
        teacher[..., 1] = yy / size            # ramp in y
        teacher[..., 2] = 0.25
        teacher[..., 3] = 0.0                  # normal xy = (0, 0)
        teacher[..., 4] = 0.0
        teacher[..., 5] = 0.5
        teacher[..., 6] = 0.0
        teacher[..., 7] = 1.0
        return teacher

    def test_level_count_and_sizes(self):
        mips = driver.build_teacher_mips(self._teacher(64))
        self.assertEqual(len(mips), driver.MIP_MAX + 1)
        self.assertEqual(
            [tuple(m.shape[:2]) for m in mips],
            [(64, 64), (32, 32), (16, 16), (8, 8), (4, 4), (2, 2),
             (1, 1)])

    def test_sizes_clamp_at_one(self):
        mips = driver.build_teacher_mips(self._teacher(48))
        self.assertEqual(tuple(mips[5].shape[:2]), (1, 1))
        self.assertEqual(tuple(mips[6].shape[:2]), (1, 1))

    def test_mip0_is_the_teacher(self):
        teacher = self._teacher()
        mips = driver.build_teacher_mips(teacher)
        self.assertTrue(np.array_equal(mips[0], teacher))

    def test_box_average_values(self):
        teacher = self._teacher(4)
        mip1 = driver.build_teacher_mips(teacher)[1]
        self.assertEqual(mip1.shape, (2, 2, 8))
        # Channel 0 is xx/4: the top-left 2x2 block holds 0, 0.25 in
        # both rows -> mean 0.125.  Top-right block: 0.5, 0.75 -> 0.625.
        self.assertAlmostEqual(float(mip1[0, 0, 0]), 0.125, places=6)
        self.assertAlmostEqual(float(mip1[0, 1, 0]), 0.625, places=6)
        # Constant channels stay constant.
        self.assertTrue(np.allclose(mip1[..., 5], 0.5))
        self.assertTrue(np.allclose(mip1[..., 7], 1.0))

    def test_normals_averaged_as_vectors_and_renormalized(self):
        teacher = np.zeros((4, 4, 8), dtype=np.float32)
        # Alternating columns: xy = (0.6, 0) on even columns,
        # (0, 0.6) on odd ones, so every 2x2 block holds two of each.
        xx = np.arange(4)[None, :]
        teacher[..., 3] = np.where(xx % 2 == 0, 0.6, 0.0)
        teacher[..., 4] = np.where(xx % 2 == 0, 0.0, 0.6)
        mip1 = driver.build_teacher_mips(teacher)[1]
        # Vector average: (0.6,0,0.8) and (0,0.6,0.8) average to
        # (0.3, 0.3, 0.8), whose length is sqrt(0.82); renormalized
        # xy is (0.3, 0.3) / sqrt(0.82) = 0.3313 -- NOT the plain xy
        # average (0.3, 0.3).  That gap is what proves the SPEC 5.3
        # vector path (and renormalization) is the one taken.
        expected = 0.3 / np.sqrt(0.82)
        self.assertTrue(
            np.allclose(mip1[..., 3], expected, atol=1e-5),
            f"nx {mip1[..., 3]} != {expected}")
        self.assertTrue(
            np.allclose(mip1[..., 4], expected, atol=1e-5))

    def test_constant_normal_survives(self):
        teacher = self._teacher()
        teacher[..., 3] = 0.28
        teacher[..., 4] = -0.36
        mip2 = driver.build_teacher_mips(teacher)[2]
        self.assertTrue(np.allclose(mip2[..., 3], 0.28, atol=1e-5))
        self.assertTrue(np.allclose(mip2[..., 4], -0.36, atol=1e-5))


# ---------------------------------------------------------------------------
# Sampling distribution (batch5 p1: 50% mip 0, 50% mips 1..6 uniform)
# ---------------------------------------------------------------------------
@unittest.skipUnless(TORCH_AVAILABLE, "torch not installed")
class MipSamplingTest(unittest.TestCase):
    def _mips(self):
        # Level m's texels encode (column, row) in channels 0/1 and
        # the level itself in channel 2, so target/uv alignment and
        # the drawn level are both directly checkable.
        torch = vt.torch
        mips = []
        for m in range(driver.MIP_MAX + 1):
            size = max(1, 64 // (2 ** m))
            tex = torch.zeros((size, size, 8), dtype=torch.float32)
            xs = torch.arange(size, dtype=torch.float32)
            tex[..., 0] = xs[None, :].expand(size, size)
            tex[..., 1] = xs[:, None].expand(size, size)
            tex[..., 2] = float(m)
            mips.append(tex)
        return mips

    def test_distribution_and_texel_centres(self):
        torch = vt.torch
        gen = torch.Generator(device="cpu")
        gen.manual_seed(20261121)
        uv, tgt, mip_f = driver.draw_mip_samples(
            gen, self._mips(), 12000, "cpu")
        levels = mip_f.to(torch.long)
        frac0 = float((levels == 0).float().mean())
        self.assertTrue(0.46 < frac0 < 0.54,
                        f"mip 0 fraction {frac0}")
        for m in range(1, driver.MIP_MAX + 1):
            frac = float((levels == m).float().mean())
            self.assertTrue(0.055 < frac < 0.115,
                            f"mip {m} fraction {frac}")
        # Target channel 2 is the level constant.
        self.assertTrue(torch.equal(tgt[:, 2], mip_f))
        # Texel-centre sampling: uv * size - 0.5 recovers the integer
        # texel that the target was read from.
        for m in range(driver.MIP_MAX + 1):
            idx = (levels == m).nonzero(as_tuple=True)[0]
            if idx.numel() == 0:
                continue
            size = max(1, 64 // (2 ** m))
            px = torch.round(uv[idx, 0] * size - 0.5)
            py = torch.round(uv[idx, 1] * size - 0.5)
            self.assertTrue(torch.equal(px, tgt[idx, 0]))
            self.assertTrue(torch.equal(py, tgt[idx, 1]))

    def test_normalized_mip_values_are_level_over_max(self):
        # The trainer input uses mip_f / MIP_MAX; the draw itself
        # must hand back exact level indices to scale.
        torch = vt.torch
        gen = torch.Generator(device="cpu")
        gen.manual_seed(7)
        _uv, _tgt, mip_f = driver.draw_mip_samples(
            gen, self._mips(), 512, "cpu")
        uniq = sorted(set(mip_f.tolist()))
        self.assertTrue(all(u == int(u) for u in uniq))
        self.assertTrue(min(uniq) >= 0 and max(uniq) <= driver.MIP_MAX)


# ---------------------------------------------------------------------------
# Manifest mipMax
# ---------------------------------------------------------------------------
class ManifestMipMaxTest(unittest.TestCase):
    def _manifest(self, mip_max):
        return driver.build_manifest(
            "oak", 64, 64, b"weights", b"latents", b"tail!", 7,
            (512, 512), (256, 256), profile="standard",
            fine_channels=4, coarse_channels=4, mip_max=mip_max)

    def test_no_mip_max_by_default(self):
        self.assertNotIn("mipMax", self._manifest(None))

    def test_mip_max_recorded_when_given(self):
        manifest = self._manifest(driver.MIP_MAX)
        self.assertEqual(manifest["mipMax"], 6)


# ---------------------------------------------------------------------------
# Advisory decision (pure; thresholds derived in driver comments)
# ---------------------------------------------------------------------------
def _met(psnr, normal_mean, rhf=(1.0, 1.0, 1.0)):
    return {"psnr_db": psnr, "normal_mean_deg": normal_mean,
            "rhf_lum": rhf[0], "rhf_nx": rhf[1], "rhf_ny": rhf[2]}


def _per_mip(m0, m2, m4):
    return {"0": m0, "2": m2, "4": m4}


class AdvisoryDecisionTest(unittest.TestCase):
    def test_calm_material_no_advisory(self):
        per_mip = _per_mip(_met(40.0, 1.0), _met(39.0, 1.2),
                           _met(38.0, 1.3))
        self.assertIsNone(driver.mip_advisory_decision(per_mip))

    def test_psnr_drop_leg_boundary(self):
        # Drop exactly at the threshold fires; just under does not.
        fires = _per_mip(_met(40.0, 1.0), _met(38.0, 1.0),
                         _met(40.0 - driver.ADVISORY_PSNR_DROP_DB, 1.0))
        adv = driver.mip_advisory_decision(fires)
        self.assertIsNotNone(adv)
        self.assertEqual([r["kind"] for r in adv["reasons"]],
                         ["psnr_drop_mip0_to_mip4"])
        under = _per_mip(_met(40.0, 1.0), _met(38.0, 1.0),
                         _met(40.0 - driver.ADVISORY_PSNR_DROP_DB
                              + 0.01, 1.0))
        self.assertIsNone(driver.mip_advisory_decision(under))

    def test_normal_leg_boundary(self):
        fires = _per_mip(_met(40.0, 1.0),
                         _met(39.5, driver.ADVISORY_NORMAL_MEAN_DEG),
                         _met(39.0, 1.0))
        adv = driver.mip_advisory_decision(fires)
        self.assertIsNotNone(adv)
        self.assertIn("normal_mean_far_mip",
                      [r["kind"] for r in adv["reasons"]])
        under = _per_mip(
            _met(40.0, 1.0),
            _met(39.5, driver.ADVISORY_NORMAL_MEAN_DEG - 0.01),
            _met(39.0, 1.0))
        self.assertIsNone(driver.mip_advisory_decision(under))

    def test_rhf_leg_boundary(self):
        fires = _per_mip(_met(40.0, 1.0),
                         _met(39.5, 1.0, rhf=(1.0,
                                              driver.ADVISORY_RHF_MAX,
                                              1.0)),
                         _met(39.0, 1.0))
        adv = driver.mip_advisory_decision(fires)
        self.assertIsNotNone(adv)
        self.assertIn("rhf_far_mip", [r["kind"] for r in adv["reasons"]])
        under = _per_mip(_met(40.0, 1.0),
                         _met(39.5, 1.0,
                              rhf=(1.0, driver.ADVISORY_RHF_MAX - 0.01,
                                   1.0)),
                         _met(39.0, 1.0))
        self.assertIsNone(driver.mip_advisory_decision(under))

    def test_record_carries_measurements_and_recommendation(self):
        per_mip = _per_mip(_met(38.0, 2.0), _met(25.0, 12.0),
                           _met(22.0, 20.0))
        adv = driver.mip_advisory_decision(per_mip)
        self.assertEqual(adv["recommendation"], "--mip-train")
        self.assertAlmostEqual(adv["psnr_drop_mip0_to_mip4_db"], 16.0)
        self.assertEqual(adv["psnr_db"]["4"], 22.0)
        self.assertEqual(adv["thresholds"]["rhf_far_mip"],
                         driver.ADVISORY_RHF_MAX)


# ---------------------------------------------------------------------------
# Report records + report.md rendering
# ---------------------------------------------------------------------------
def _make_spec_teacher(tmp: Path):
    folder = tmp / "oak"
    folder.mkdir(parents=True)
    rng = np.random.default_rng(5)
    Image.fromarray(
        rng.integers(0, 255, size=(16, 16, 3), dtype=np.uint8)
    ).save(folder / "basecolor.png")
    Image.fromarray(np.full((16, 16), 128, dtype=np.uint8)).save(
        folder / "roughness.png")
    spec = resolve_inputs(folder, normal_convention="opengl")
    return spec, build_teacher(spec)


def _mip_evaluation():
    def met(psnr, nmean):
        return {"psnr_db": psnr,
                "mae": {"roughness": 0.001, "metalness": 0.0,
                        "ao": 0.001},
                "normal_mean_deg": nmean, "normal_p95_deg": nmean * 2,
                "rhf_lum": 1.0, "rhf_nx": 1.0, "rhf_ny": 1.0,
                "gate_pass": False}
    return {"mip_max": 6, "levels": [0, 2, 4],
            "decode_normalized_mip": "m/mip_max", "rhf_floor": 2e-4,
            "per_mip": {"0": met(36.0, 1.0), "2": met(38.0, 0.8),
                        "4": met(41.0, 0.5)}}


def _outcome(**extra):
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
    return driver.TrialOutcome(
        profile="standard", fine_size=(512, 512),
        coarse_size=(256, 256), fine_channels=4, coarse_channels=4,
        metrics=metrics, checks=verdict["checks"],
        gate_passed=verdict["passed"],
        capacity={"vnm1_total_bytes": 100,
                  "ratio_vs_bcn_mip_chain": 0.08},
        recipe={"recipe_version": 2},
        environment={"device": "cuda", "gpu_name": "T4"},
        started_at="t0", finished_at="t1", elapsed_seconds=1.0,
        training_seconds=0.9, final_loss=0.01,
        artifact_dir=Path("/nonexistent"), artifact_files={},
        manifest={}, roundtrip={"weights_parse_ok": True}, **extra)


class ReportMipTest(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp(prefix="vnm1-mip-report-"))
        self.addCleanup(shutil.rmtree, self.tmp, True)
        self.spec, self.teacher = _make_spec_teacher(self.tmp)
        self.params = {"profile": "standard", "iterations": 50000,
                       "color_weight": 4.0, "scalar_weight": 1.0,
                       "seed": 1337}

    def _render(self, outcome):
        rep = report_mod.build_report(
            self.spec, self.teacher, outcome, mode="manual",
            artifact_location="/dist/oak/oak.vnm",
            trial_params=self.params)
        return rep, render_report_markdown(rep)

    def test_plain_run_has_no_mip_fields(self):
        rep, md = self._render(_outcome())
        self.assertNotIn("mip_evaluation", rep["trial"])
        self.assertNotIn("mip_advisory", rep["trial"])
        self.assertNotIn("Multi-mip behaviour", md)
        self.assertNotIn("Far-field advisory", md)

    def test_mip_trained_run_records_evaluation_and_table(self):
        outcome = _outcome(
            mip_training={"enabled": True, "mip_max": 6},
            mip_evaluation=_mip_evaluation())
        rep, md = self._render(outcome)
        self.assertEqual(
            rep["trial"]["mip_evaluation"]["per_mip"]["4"]
            ["psnr_db"], 41.0)
        self.assertIn("Multi-mip behaviour", md)
        self.assertIn("| 0 | 36.00 dB | 1.00° |", md)
        self.assertIn("| 4 | 41.00 dB | 0.50° |", md)

    def test_advisory_appears_in_report_md_with_numbers(self):
        per_mip = _per_mip(_met(38.0, 2.0), _met(25.0, 12.0),
                           _met(22.0, 20.0))
        outcome = _outcome(
            mip_advisory=driver.mip_advisory_decision(per_mip))
        rep, md = self._render(outcome)
        self.assertIn("mip_advisory", rep["trial"])
        self.assertIn("Far-field advisory", md)
        self.assertIn("--mip-train is recommended", md)
        self.assertIn("38.00 dB at mip 0", md)
        self.assertIn("22.00 dB at mip 4", md)
        # Advisory only: the verdict is untouched.
        self.assertEqual(rep["conclusion"]["result"], "passed")


# ---------------------------------------------------------------------------
# Short real runs (CPU): --mip-train artifact must read back and
# decode; a mip-0 run must keep the pre-addition manifest shape.
# ---------------------------------------------------------------------------
@unittest.skipUnless(TORCH_AVAILABLE, "torch not installed")
class ShortRealRunTest(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp(prefix="vnm1-mip-run-"))
        self.addCleanup(shutil.rmtree, self.tmp, True)
        rng = np.random.default_rng(3)
        teacher = rng.random((64, 64, 8), dtype=np.float32)
        teacher[..., 3] = teacher[..., 3] * 2 - 1
        teacher[..., 4] = teacher[..., 4] * 2 - 1
        self.teacher = teacher.astype(np.float32)
        self.sources = {"base_color": "map", "normal": "map",
                        "roughness": "map", "metalness": "constant",
                        "ao": "constant"}

    def _run(self, mip_train):
        options = driver.TrialOptions(
            material_name="mipcheck", profile="standard",
            iterations=24, seed=1337, device="cpu",
            work_dir=self.tmp / ("on" if mip_train else "off"),
            mip_train=mip_train)
        return driver.run_trial(self.teacher, self.sources, options)

    def test_mip_trained_artifact_reads_back(self):
        outcome = self._run(mip_train=True)
        self.assertIsNotNone(outcome.mip_training)
        self.assertEqual(outcome.mip_training["mip_max"],
                         driver.MIP_MAX)
        self.assertIsNotNone(outcome.mip_evaluation)
        self.assertEqual(
            sorted(outcome.mip_evaluation["per_mip"].keys()),
            ["0", "2", "4"])
        self.assertIsNone(outcome.mip_advisory)
        self.assertEqual(outcome.manifest["mipMax"], driver.MIP_MAX)
        # Read the artifact back from disk and decode at a mid
        # normalizedMip, as a runtime would.
        staging = outcome.artifact_dir
        weights = vt.parse_weights_chunk(
            (staging / "weights.bin").read_bytes(), profile="standard")
        fine, fine_size, coarse, coarse_size = vt.parse_latents_chunk(
            (staging / "level-0.bin").read_bytes(), profile="standard")
        fine_ch, coarse_ch = vt.latents_chunk_channels(
            (staging / "level-0.bin").read_bytes(), profile="standard")
        decoder = vt.VnmDecoder.from_u8_numpy(
            weights, fine, coarse, fine_size, coarse_size,
            device="cpu", profile="standard",
            fine_channels=fine_ch, coarse_channels=coarse_ch)
        torch = vt.torch
        uv = torch.tensor([[0.25, 0.25], [0.75, 0.6]],
                          dtype=torch.float32)
        pbr = decoder.decode_pbr(uv, normalized_mip=0.5)
        self.assertTrue(
            bool(torch.isfinite(pbr["baseColor"]).all()))
        self.assertTrue(bool(torch.isfinite(pbr["normal"]).all()))

    def test_mip0_run_keeps_manifest_shape(self):
        outcome = self._run(mip_train=False)
        self.assertNotIn("mipMax", outcome.manifest)
        self.assertIsNone(outcome.mip_training)
        self.assertIsNone(outcome.mip_evaluation)
        # The advisory may or may not fire on this synthetic teacher;
        # when it fires it must carry the full record.
        if outcome.mip_advisory is not None:
            self.assertEqual(outcome.mip_advisory["recommendation"],
                             "--mip-train")
            self.assertIn("reasons", outcome.mip_advisory)


# ---------------------------------------------------------------------------
# CLI wiring: --mip-train must reach TrialOptions in BOTH modes, be
# recorded in the manual trial params only when on, and the chosen
# record's mip notes must print without disturbing the flow.  The
# trial itself is faked (the pattern of tests/test_train.py); the
# real training paths are covered by ShortRealRunTest above.
# ---------------------------------------------------------------------------
def _write_png(path: Path, array: np.ndarray):
    Image.fromarray(array).save(path)


def _make_material(folder: Path) -> Path:
    folder.mkdir(parents=True)
    size = 32
    rng = np.random.default_rng(11)
    _write_png(folder / "basecolor.png",
               rng.integers(0, 255, size=(size, size, 3),
                            dtype=np.uint8))
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


def _fake_visual_sheets(artifact_dir, teacher, material_name, out_dir,
                        *, expected_psnr_db):
    files = {}
    for kind in ("sheet", "crop"):
        path = Path(out_dir) / f"{material_name}_{kind}.png"
        Image.new("RGB", (8, 8)).save(path)
        files[path.name] = {"bytes": path.stat().st_size,
                            "kind": kind}
    return {"files": files, "artifact_dir": str(artifact_dir),
            "recomputed_psnr_db": expected_psnr_db,
            "report_psnr_db": expected_psnr_db, "psnr_delta_db": 0.0,
            "psnr_tolerance_db": 0.05, "psnr_match": True,
            "heat_gain": 8.0, "note": "fake sheets (wiring test)"}


class CliWiringTest(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp(prefix="vnm1-mip-cli-"))
        self.addCleanup(shutil.rmtree, self.tmp, True)
        self.material = _make_material(self.tmp / "oak")
        self.out = self.tmp / "dist"
        self.seen_options = []

    def _patches(self):
        status = {"python_version": "3.12.3",
                  "python_implementation": "CPython", "os": "Linux",
                  "torch_available": True, "torch_version": "2.x",
                  "cuda_available": False, "gpu_name": None}

        def run_trial(teacher, channel_sources, options,
                      progress_cb=None):
            self.seen_options.append(options)
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
            metrics = {
                "psnr_base_color_db": 38.9,
                "psnr_base_color_db_srgb_space_reference": 30.0,
                "mae": {"roughness": 0.002, "metalness": 0.0,
                        "ao": 0.009},
                "normal_angle_mean_deg": 0.37,
                "normal_angle_p95_deg": 0.76,
                "texel_count": 1024, "eval_seconds": 0.1,
                "decode_texels_per_second": 10000.0}
            sources = {"base_color": "map", "normal": "map",
                       "roughness": "map", "metalness": "constant",
                       "ao": "map"}
            verdict = driver.gate_verdict(metrics, sources)
            extra = {}
            if options.mip_train:
                extra = {
                    "mip_training": {"enabled": True, "mip_max": 6},
                    "mip_evaluation": _mip_evaluation()}
            return driver.TrialOutcome(
                profile=options.profile, fine_size=(512, 512),
                coarse_size=(256, 256), fine_channels=4,
                coarse_channels=4, metrics=metrics,
                checks=verdict["checks"], gate_passed=True,
                capacity={
                    "vnm1_breakdown_bytes": {
                        "manifest.json": 15, "weights.bin": 15,
                        "level-0.bin": 15, "tail.bin": 12},
                    "vnm1_total_bytes": 57,
                    "bcn": {"per_map": {}, "total_mip0_bytes": 100,
                            "total_mip_chain_bytes": 120},
                    "ratio_vs_bcn_mip_chain": 57 / 120,
                    "ratio_vs_bcn_mip0_reference": 57 / 100},
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
                           "manifest_sha_match": True},
                **extra)

        return (mock.patch.object(driver, "torch_status",
                                  return_value=status),
                mock.patch.object(driver, "run_trial", run_trial),
                mock.patch("vnm1.visual.generate_visual_sheets",
                           _fake_visual_sheets))

    def _run_main(self, argv):
        patches = self._patches()
        buf = io.StringIO()
        with patches[0], patches[1], patches[2], \
                contextlib.redirect_stdout(buf):
            code = main(argv)
        return code, buf.getvalue()

    def _argv(self, mode, *extra):
        return ["train", str(self.material), "--mode", mode,
                "--normal-convention", "opengl",
                "--allow-unverified-size",
                "--allow-capacity-expansion",
                "--output", str(self.out), *extra]

    def test_manual_mip_train_reaches_options_and_report(self):
        code, out = self._run_main(
            self._argv("manual", "--profile", "standard",
                       "--iterations", "10", "--mip-train"))
        self.assertEqual(code, 0)
        self.assertEqual(len(self.seen_options), 1)
        self.assertTrue(self.seen_options[0].mip_train)
        self.assertIn("Mip training: on", out)
        report = json.loads(
            (self.out / "oak" / "oak.report.json").read_text())
        self.assertIn("mip_training", report["trial"])
        self.assertIn("mip_evaluation", report["trial"])
        self.on_report = report

    def test_manual_without_flag_records_nothing(self):
        # Run the ON variant first (separate output dir) so the
        # reproducibility hash can be compared: mip_train=True is
        # part of the manual trial params only when the flag is on,
        # so the two runs' input/config hashes must differ while the
        # OFF report itself gains no mip keys at all.
        argv_on = ["train", str(self.material), "--mode", "manual",
                   "--normal-convention", "opengl",
                   "--allow-unverified-size",
                   "--allow-capacity-expansion",
                   "--output", str(self.tmp / "dist-on"),
                   "--profile", "standard", "--iterations", "10",
                   "--mip-train"]
        code_on, _ = self._run_main(argv_on)
        self.assertEqual(code_on, 0)
        report_on = json.loads(
            (self.tmp / "dist-on" / "oak" / "oak.report.json")
            .read_text())
        code, out = self._run_main(
            self._argv("manual", "--profile", "standard",
                       "--iterations", "10"))
        self.assertEqual(code, 0)
        self.assertFalse(self.seen_options[-1].mip_train)
        self.assertNotIn("Mip training: on", out)
        report = json.loads(
            (self.out / "oak" / "oak.report.json").read_text())
        self.assertNotIn("mip_training", report["trial"])
        self.assertNotIn("mip_evaluation", report["trial"])
        self.assertNotEqual(
            report["format"]["input_config_hash"],
            report_on["format"]["input_config_hash"])

    def test_auto_mip_train_covers_the_ladder(self):
        # The fake trial passes, so the ladder stops after trial 1;
        # every trial it did run must have carried the flag.
        code, out = self._run_main(
            self._argv("auto", "--mip-train"))
        self.assertEqual(code, 0)
        self.assertTrue(self.seen_options)
        self.assertTrue(
            all(o.mip_train for o in self.seen_options))
        self.assertIn("Mip training: on", out)


if __name__ == "__main__":
    unittest.main()

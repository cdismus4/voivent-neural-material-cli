"""Unit tests for the step B training core import (vnm1.core) and the
ported driver helpers (vnm1.driver).

These tests use only the numpy code paths (chunk IO, quantization,
gate verdicts, BCn estimate, LR schedule, tail chunk) and therefore
run without torch, like the step A input-layer tests.
"""

import hashlib
import unittest
from pathlib import Path

import numpy as np

from vnm1 import core, driver

ROOT = Path(__file__).resolve().parents[3]  # vnm1-judgment-run/
JUDGMENT_CORE = ROOT / "stage1" / "src" / "vnm1_torch.py"
PRODUCT_CORE = Path(core.__file__)


def _md5(path: Path) -> str:
    return hashlib.md5(path.read_bytes()).hexdigest()


class CoreIdentityTest(unittest.TestCase):
    def test_core_is_byte_identical_to_judgment_port(self):
        if not JUDGMENT_CORE.exists():
            self.skipTest("judgment-run tree not present")
        self.assertEqual(_md5(PRODUCT_CORE), _md5(JUDGMENT_CORE))
        self.assertEqual(_md5(PRODUCT_CORE),
                         "c1f9935a8912842a10072ae6df1dc725")


class ProfileTest(unittest.TestCase):
    def test_channel_derivation(self):
        self.assertEqual(core.profile_latent_channels("standard"), (4, 4))
        self.assertEqual(core.profile_latent_channels("quality"), (4, 4))
        self.assertEqual(core.profile_latent_channels("standard8"), (8, 8))
        self.assertEqual(core.profile_latent_channels("quality8"), (8, 8))

    def test_aliases_resolve_to_canonical_names(self):
        self.assertEqual(core.resolve_profile("standard8ch"), "standard8")
        self.assertEqual(core.resolve_profile("quality8ch"), "quality8")

    def test_presets_cover_design_profiles(self):
        self.assertEqual(set(driver.PROFILE_PRESETS),
                         {"standard", "standard8", "quality", "quality8"})
        self.assertEqual(driver.PROFILE_PRESETS["standard"]["fine_size"],
                         (512, 512))
        self.assertEqual(driver.PROFILE_PRESETS["quality8"]["fine_size"],
                         (1024, 1024))


class ChunkRoundTripTest(unittest.TestCase):
    def test_weights_roundtrip_exact_all_profiles(self):
        rng = np.random.default_rng(42)
        for profile in ("standard", "standard8", "quality", "quality8"):
            shapes = core.weight_shapes(profile)
            weights = {k: rng.standard_normal(shapes[k]).astype(np.float32)
                       for k in core.WEIGHT_KEYS}
            back = core.parse_weights_chunk(
                core.serialize_weights_chunk(weights, profile=profile),
                profile=profile)
            for k in core.WEIGHT_KEYS:
                self.assertTrue(np.array_equal(back[k], weights[k]),
                                f"{profile}/{k}")

    def test_latents_roundtrip_8ch_and_reserved_bytes(self):
        rng = np.random.default_rng(7)
        fine = rng.integers(0, 256, size=64 * 64 * 8, dtype=np.uint8)
        coarse = rng.integers(0, 256, size=32 * 32 * 8, dtype=np.uint8)
        chunk = core.serialize_latents_chunk(
            fine, (64, 64), coarse, (32, 32),
            fine_channels=8, coarse_channels=8)
        # Unified storage rule: subheader reserved bytes are zero.
        self.assertTrue(all(b == 0 for b in chunk[26:32]))
        f2, fs2, c2, cs2 = core.parse_latents_chunk(chunk,
                                                    profile="standard8")
        self.assertEqual(fs2, (64, 64))
        self.assertEqual(cs2, (32, 32))
        self.assertTrue(np.array_equal(f2, fine))
        self.assertTrue(np.array_equal(c2, coarse))
        # A non-zero reserved byte is a spec violation.
        tampered = bytearray(chunk)
        tampered[27] = 8
        with self.assertRaises(ValueError):
            core.latents_chunk_channels(bytes(tampered),
                                         profile="standard8")

    def test_quantize_dequantize_bounds(self):
        values = np.linspace(-1.0, 1.0, 2001, dtype=np.float32)
        back = core.dequantize_u8(core.quantize_float_to_u8(values))
        err = float(np.max(np.abs(back.astype(np.float64)
                                  - values.astype(np.float64))))
        self.assertLessEqual(err, 1.0 / 255.0 + 1e-9)


class TailChunkTest(unittest.TestCase):
    def test_tail_roundtrip(self):
        rng = np.random.default_rng(3)
        teacher = rng.random((32, 48, 8), dtype=np.float32)
        chunk = driver.serialize_tail_chunk(teacher)
        info = driver.parse_tail_chunk(chunk)
        # 48x32 -> box downsample to max side <= 16 lands at 12x8, mip 2.
        self.assertEqual(info["tail_size"], (12, 8))
        self.assertEqual(info["tail_first_mip"], 2)
        self.assertEqual(info["pixels"].shape, (8, 12, 8))


class GateVerdictTest(unittest.TestCase):
    SOURCES = {"base_color": "map", "normal": "map", "roughness": "map",
               "metalness": "constant", "ao": "map"}

    def _metrics(self, **over):
        metrics = {"psnr_base_color_db": 38.9,
                   "mae": {"roughness": 0.002, "metalness": 0.5,
                           "ao": 0.009},
                   "normal_angle_mean_deg": 0.37,
                   "normal_angle_p95_deg": 0.76}
        for key, value in over.items():
            if key == "mae":
                metrics["mae"].update(value)
            else:
                metrics[key] = value
        return metrics

    def test_pass_with_constant_channel_exempt(self):
        # metalness MAE is huge but the channel is constant (no source
        # map), so it is gate-exempt and cannot fail the verdict.
        verdict = driver.gate_verdict(self._metrics(), self.SOURCES)
        self.assertTrue(verdict["passed"])
        self.assertFalse(
            verdict["checks"]["mae_metalness"]["gate_applicable"])

    def test_psnr_shortfall_fails_with_negative_difference(self):
        verdict = driver.gate_verdict(
            self._metrics(psnr_base_color_db=34.77), self.SOURCES)
        self.assertFalse(verdict["passed"])
        self.assertAlmostEqual(
            verdict["checks"]["psnr_base_color_db"]["difference"],
            -0.23, places=2)

    def test_mae_excess_fails_with_positive_difference(self):
        verdict = driver.gate_verdict(
            self._metrics(mae={"ao": 0.0219}), self.SOURCES)
        self.assertFalse(verdict["passed"])
        self.assertAlmostEqual(
            verdict["checks"]["mae_ao"]["difference"], 0.0019, places=4)


class ScheduleAndCapacityTest(unittest.TestCase):
    def test_lr_factor(self):
        self.assertEqual(driver.lr_factor(0, 50000), 1.0)
        self.assertEqual(driver.lr_factor(24999, 50000), 1.0)
        self.assertAlmostEqual(driver.lr_factor(49999, 50000), 0.1,
                               places=2)
        mid = driver.lr_factor(37500, 50000)
        self.assertTrue(0.1 < mid < 1.0)

    def test_bcn_estimate_matches_stage3_marble_record(self):
        # Stage 3 marble-std record: BCn full mip chain 16,777,296
        # bytes for 2048x2048 with metalness constant (no map).
        sources = {"base_color": "map", "normal": "map",
                   "roughness": "map", "metalness": "constant",
                   "ao": "map"}
        est = driver.bcn_estimate(2048, 2048, sources)
        self.assertEqual(est["total_mip_chain_bytes"], 16777296)
        self.assertEqual(est["per_map"]["metalness"]["mip_chain_bytes"],
                         0)


if __name__ == "__main__":
    unittest.main()

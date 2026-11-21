"""Unit tests for the visual sheets (vnm1.visual, step D).

The pure rendering helpers (crop clamping, heat palette, colour
conversions) run anywhere.  The end-to-end sheet generation needs
torch (the core decoder is the torch path) and is skipped without
it; it trains a tiny real trial on CPU so the sheets, the float-
buffer PSNR recheck and the mismatch failure are all exercised
against a real artifact.
"""

import shutil
import tempfile
import unittest
from pathlib import Path

import numpy as np
from PIL import Image

from vnm1 import core as vt
from vnm1 import visual
from vnm1.errors import ExportVerificationError

TORCH_AVAILABLE = vt.torch is not None


class PureHelpersTest(unittest.TestCase):
    def test_crop_bounds_full_size_input(self):
        self.assertEqual(visual.crop_bounds(2048, 2048),
                         (768, 768, 512, 512))

    def test_crop_bounds_clamps_to_small_input(self):
        # Section 7.4 revision: smaller than the nominal crop ->
        # crop at the real size instead of failing.
        self.assertEqual(visual.crop_bounds(256, 256), (0, 0, 256, 256))
        self.assertEqual(visual.crop_bounds(100, 600), (0, 44, 100, 512))

    def test_heat_rgb_shape_and_endpoints(self):
        err = np.array([[0.0, 0.5, 1.0]], dtype=np.float32)
        img = visual.heat_rgb(err)
        self.assertEqual(img.shape, (1, 3, 3))
        self.assertEqual(img.dtype, np.uint8)
        self.assertTrue(np.all(img[0, 0] == 0))          # no error: black
        self.assertTrue(np.all(img[0, 2] == 255))        # max: white

    def test_linear_to_srgb_endpoints(self):
        out = visual.linear_to_srgb(np.array([0.0, 1.0],
                                             dtype=np.float32))
        self.assertAlmostEqual(float(out[0]), 0.0, places=5)
        self.assertAlmostEqual(float(out[1]), 1.0, places=5)

    def test_load_decoder_rejects_garbage(self):
        tmp = Path(tempfile.mkdtemp(prefix="vnm1-visual-bad-"))
        self.addCleanup(shutil.rmtree, tmp, True)
        (tmp / "manifest.json").write_text("not json")
        with self.assertRaises(ExportVerificationError):
            visual.load_decoder(tmp)


@unittest.skipUnless(TORCH_AVAILABLE, "torch not installed")
class SheetGenerationTest(unittest.TestCase):
    """Real tiny trial -> real sheets (CPU, a few seconds)."""

    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp(prefix="vnm1-visual-test-"))
        self.addCleanup(shutil.rmtree, self.tmp, True)
        from vnm1 import driver
        from vnm1.imaging import build_teacher
        from vnm1.inputs import resolve_inputs

        folder = self.tmp / "oak"
        folder.mkdir(parents=True)
        rng = np.random.default_rng(7)
        base = rng.integers(64, 192, size=(64, 64, 3), dtype=np.uint8)
        Image.fromarray(base).save(folder / "basecolor.png")
        normal = np.zeros((64, 64, 3), dtype=np.uint8)
        normal[..., 0] = 128
        normal[..., 1] = 140
        normal[..., 2] = 255
        Image.fromarray(normal).save(folder / "normal.png")
        Image.fromarray(np.full((64, 64), 200, dtype=np.uint8)).save(
            folder / "roughness.png")
        spec = resolve_inputs(folder, normal_convention="opengl")
        self.teacher = build_teacher(spec)
        options = driver.TrialOptions(
            material_name="oak", profile="standard", iterations=30,
            device="cpu", work_dir=self.tmp / "work")
        sources = {"base_color": "map", "normal": "map",
                   "roughness": "map", "metalness": "constant",
                   "ao": "constant"}
        self.outcome = driver.run_trial(
            self.teacher.teacher, sources, options)

    def test_sheets_written_and_psnr_matches(self):
        out_dir = self.tmp / "sheets"
        info = visual.generate_visual_sheets(
            self.outcome.artifact_dir, self.teacher, "oak", out_dir,
            expected_psnr_db=self.outcome.metrics[
                "psnr_base_color_db"])
        self.assertTrue(info["psnr_match"])
        self.assertLessEqual(abs(info["psnr_delta_db"]), 0.05)
        with Image.open(out_dir / "oak_sheet.png") as sheet:
            self.assertEqual(sheet.size, (3072, 1024))
        # 64x64 input: the crop clamps to the real size (128x64).
        with Image.open(out_dir / "oak_crop.png") as crop:
            self.assertEqual(crop.size, (128, 64))
        with Image.open(out_dir / "oak_normal_sheet.png") as normal_sheet:
            self.assertEqual(normal_sheet.size, (2048, 1024))
        self.assertEqual(set(info["files"]),
                         {"oak_sheet.png", "oak_crop.png",
                          "oak_normal_sheet.png"})

    def test_psnr_mismatch_raises(self):
        with self.assertRaises(ExportVerificationError):
            visual.generate_visual_sheets(
                self.outcome.artifact_dir, self.teacher, "oak",
                self.tmp / "sheets2",
                expected_psnr_db=self.outcome.metrics[
                    "psnr_base_color_db"] + 5.0)


if __name__ == "__main__":
    unittest.main()

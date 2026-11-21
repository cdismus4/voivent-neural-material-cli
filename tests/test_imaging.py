"""Unit tests for vnm1.imaging (loading / validation / teacher, doc 4.3-4.5).

All images are small synthetic PNGs written with Pillow.  The numeric
expectations pin the judgment-run driver's conventions: dtype-based
bit-depth normalisation, sRGB->linear base color, normal xy = rgb*2-1,
and the constant values for missing maps.
"""

import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from vnm1.errors import InputValidationError
from vnm1.imaging import build_teacher, load_scalar, srgb_to_linear
from vnm1.inputs import resolve_inputs

W = H = 8


def _save(img: Image.Image, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    img.save(path)
    return path


def _solid(mode: str, size, value) -> Image.Image:
    return Image.new(mode, size, value)


class BitDepthNormalisationTest(unittest.TestCase):
    def test_8bit_grayscale(self):
        with tempfile.TemporaryDirectory() as td:
            p = _save(_solid("L", (W, H), 128), Path(td) / "roughness.png")
            arr, warnings = load_scalar(p, "roughness")
            self.assertEqual(warnings, [])
            self.assertAlmostEqual(float(arr[0, 0]), 128 / 255, places=6)

    def test_16bit_grayscale(self):
        with tempfile.TemporaryDirectory() as td:
            data = np.full((H, W), 32768, dtype=np.uint16)
            img = Image.fromarray(data, mode="I;16")
            p = _save(img, Path(td) / "ao.png")
            arr, _ = load_scalar(p, "ao")
            # 16-bit data must be normalised by 65535, not 255
            # (the stage2 driver's 16-bit fix).
            self.assertAlmostEqual(float(arr[0, 0]), 32768 / 65535, places=5)

    def test_16bit_full_scale_is_one(self):
        with tempfile.TemporaryDirectory() as td:
            data = np.full((H, W), 65535, dtype=np.uint16)
            p = _save(Image.fromarray(data, mode="I;16"), Path(td) / "ao.png")
            arr, _ = load_scalar(p, "ao")
            self.assertAlmostEqual(float(arr[0, 0]), 1.0, places=6)


def _write_png16_rgb(path: Path, w: int, h: int, value16: int) -> Path:
    """Write a minimal 16-bit RGB PNG by hand.

    Pillow cannot save 16-bit RGB (it truncates such files to uint8 on
    load), so the fixture is assembled from raw chunks: IHDR bit depth
    16, colour type 2, filter-0 scanlines of big-endian uint16.
    """
    import binascii
    import struct
    import zlib

    def chunk(tag: bytes, data: bytes) -> bytes:
        return (struct.pack(">I", len(data)) + tag + data
                + struct.pack(">I", binascii.crc32(tag + data) & 0xFFFFFFFF))

    ihdr = struct.pack(">IIBBBBB", w, h, 16, 2, 0, 0, 0)
    row = b"\x00" + struct.pack(">H", value16) * (w * 3)
    png = (b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", ihdr)
           + chunk(b"IDAT", zlib.compress(row * h)) + chunk(b"IEND", b""))
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(png)
    return path


class OneBitScalarTest(unittest.TestCase):
    """1-bit PNGs (PIL mode "1", bool) must load as exact 0.0 / 1.0.

    Regression test for the step-E finding: a bool array fell into the
    "anything else" branch of _to_unit_float and was divided by 65535,
    silently scaling a constant-1.0 metalness map to ~1.5e-5 while the
    run still reported a gate pass.
    """

    def test_1bit_all_true_is_one(self):
        with tempfile.TemporaryDirectory() as td:
            p = _save(_solid("1", (W, H), 1), Path(td) / "metalness.png")
            arr, warnings = load_scalar(p, "metalness")
            self.assertEqual(warnings, [])
            self.assertTrue(np.all(arr == 1.0))

    def test_1bit_all_false_is_zero(self):
        with tempfile.TemporaryDirectory() as td:
            p = _save(_solid("1", (W, H), 0), Path(td) / "metalness.png")
            arr, _ = load_scalar(p, "metalness")
            self.assertTrue(np.all(arr == 0.0))

    def test_1bit_mixed_pattern_is_exact(self):
        with tempfile.TemporaryDirectory() as td:
            pattern = (np.indices((H, W)).sum(axis=0) % 2) == 0
            p = _save(Image.fromarray(pattern), Path(td) / "metalness.png")
            arr, _ = load_scalar(p, "metalness")
            self.assertTrue(np.array_equal(arr, pattern.astype(np.float32)))

    def test_1bit_metalness_reaches_teacher_as_one(self):
        with tempfile.TemporaryDirectory() as td:
            folder = Path(td) / "mat"
            _save(_solid("RGB", (W, H), (10, 20, 30)),
                  folder / "basecolor.png")
            _save(_solid("1", (W, H), 1), folder / "metalness.png")
            spec = resolve_inputs(folder, normal_convention="opengl")
            result = build_teacher(spec)
            self.assertTrue(np.all(result.teacher[..., 6] == 1.0))
            record = result.records["metalness"]
            self.assertEqual(record.source, "map")
            self.assertFalse(record.gate_exempt)
            self.assertEqual(record.pil_mode, "1")
            self.assertEqual(record.bit_depth, 1)


class BitDepthRecordTest(unittest.TestCase):
    """MapRecord.bit_depth is the file's IHDR bit depth, not a guess
    from the post-load dtype (Pillow loads 16-bit RGB as uint8 and
    1-bit data as bool, so the dtype rule misreported both)."""

    def _record_for(self, td: str, logical: str, path: Path):
        folder = Path(td) / "mat"
        if logical != "base_color":
            _save(_solid("RGB", (W, H), (0, 0, 0)),
                  folder / "basecolor.png")
        folder.mkdir(parents=True, exist_ok=True)
        dest = folder / path.name
        if path != dest:
            dest.write_bytes(path.read_bytes())
        spec = resolve_inputs(folder, normal_convention="opengl")
        return build_teacher(spec).records[logical]

    def test_16bit_rgb_records_16_not_8(self):
        with tempfile.TemporaryDirectory() as td:
            src = _write_png16_rgb(Path(td) / "basecolor.png", W, H, 32768)
            record = self._record_for(td, "base_color", src)
            self.assertEqual(record.bit_depth, 16)

    def test_8bit_grayscale_records_8(self):
        with tempfile.TemporaryDirectory() as td:
            src = _save(_solid("L", (W, H), 128), Path(td) / "roughness.png")
            record = self._record_for(td, "roughness", src)
            self.assertEqual(record.bit_depth, 8)

    def test_16bit_grayscale_records_16(self):
        with tempfile.TemporaryDirectory() as td:
            data = np.full((H, W), 32768, dtype=np.uint16)
            src = _save(Image.fromarray(data, mode="I;16"),
                        Path(td) / "ao.png")
            record = self._record_for(td, "ao", src)
            self.assertEqual(record.bit_depth, 16)


class ScalarAcceptanceRulesTest(unittest.TestCase):
    def test_rgb_with_equal_channels_accepted_silently(self):
        with tempfile.TemporaryDirectory() as td:
            p = _save(_solid("RGB", (W, H), (90, 90, 90)),
                      Path(td) / "roughness.png")
            arr, warnings = load_scalar(p, "roughness")
            self.assertEqual(warnings, [])
            self.assertAlmostEqual(float(arr[0, 0]), 90 / 255, places=6)

    def test_rgba_with_opaque_alpha_accepted_silently(self):
        with tempfile.TemporaryDirectory() as td:
            p = _save(_solid("RGBA", (W, H), (90, 90, 90, 255)),
                      Path(td) / "roughness.png")
            _arr, warnings = load_scalar(p, "roughness")
            self.assertEqual(warnings, [])

    def test_differing_channels_raise_orm_suspicion_warning(self):
        with tempfile.TemporaryDirectory() as td:
            # ORM-style packing: R=occlusion, G=roughness, B=metalness.
            p = _save(_solid("RGB", (W, H), (10, 200, 30)),
                      Path(td) / "roughness.png")
            arr, warnings = load_scalar(p, "roughness")
            self.assertEqual(len(warnings), 1)
            self.assertIn("packed", warnings[0])
            # First channel is used, as in the driver.
            self.assertAlmostEqual(float(arr[0, 0]), 10 / 255, places=6)

    def test_differing_channels_are_an_error_under_error_policy(self):
        with tempfile.TemporaryDirectory() as td:
            p = _save(_solid("RGB", (W, H), (10, 200, 30)),
                      Path(td) / "roughness.png")
            with self.assertRaises(InputValidationError):
                load_scalar(p, "roughness", packed_policy="error")

    def test_scalar_alpha_non_opaque_warns(self):
        with tempfile.TemporaryDirectory() as td:
            p = _save(_solid("RGBA", (W, H), (90, 90, 90, 128)),
                      Path(td) / "ao.png")
            _arr, warnings = load_scalar(p, "ao")
            self.assertTrue(any("alpha" in w for w in warnings))


class BaseColorAlphaTest(unittest.TestCase):
    def _folder_with_base(self, td: str, img: Image.Image) -> Path:
        folder = Path(td) / "mat"
        _save(img, folder / "basecolor.png")
        return folder

    def test_opaque_alpha_warns_but_loads(self):
        with tempfile.TemporaryDirectory() as td:
            folder = self._folder_with_base(
                td, _solid("RGBA", (W, H), (200, 100, 50, 255)))
            spec = resolve_inputs(folder, normal_convention="opengl")
            result = build_teacher(spec)
            self.assertTrue(any("alpha" in w for w in result.warnings))

    def test_non_opaque_alpha_is_an_error(self):
        with tempfile.TemporaryDirectory() as td:
            folder = self._folder_with_base(
                td, _solid("RGBA", (W, H), (200, 100, 50, 128)))
            spec = resolve_inputs(folder, normal_convention="opengl")
            with self.assertRaises(InputValidationError):
                build_teacher(spec)


class FormatAndDimensionTest(unittest.TestCase):
    def test_jpeg_is_rejected(self):
        with tempfile.TemporaryDirectory() as td:
            folder = Path(td) / "mat"
            jpg = folder / "basecolor.png"  # PNG name, JPEG content
            jpg.parent.mkdir(parents=True)
            _solid("RGB", (W, H), (1, 2, 3)).save(jpg, format="JPEG")
            spec = resolve_inputs(folder, normal_convention="opengl")
            with self.assertRaises(InputValidationError):
                build_teacher(spec)

    def test_dimension_mismatch_is_an_error(self):
        with tempfile.TemporaryDirectory() as td:
            folder = Path(td) / "mat"
            _save(_solid("RGB", (W, H), (10, 20, 30)),
                  folder / "basecolor.png")
            _save(_solid("RGB", (4, 4), (128, 128, 255)),
                  folder / "normal.png")
            spec = resolve_inputs(folder, normal_convention="opengl")
            with self.assertRaises(InputValidationError):
                build_teacher(spec)


class TeacherConstructionTest(unittest.TestCase):
    def _base_only_folder(self, td: str, size=(W, H), value=(255, 255, 255)):
        folder = Path(td) / "mat"
        _save(_solid("RGB", size, value), folder / "basecolor.png")
        return folder

    def test_missing_maps_become_gate_exempt_constants(self):
        with tempfile.TemporaryDirectory() as td:
            folder = self._base_only_folder(td)
            spec = resolve_inputs(folder, normal_convention="opengl")
            result = build_teacher(spec)
            t = result.teacher
            self.assertEqual(t.shape, (H, W, 8))
            self.assertEqual(t.dtype, np.float32)
            # Flat normal (0, 0), roughness 1.0, metalness 0.0, AO 1.0.
            self.assertTrue(np.all(t[..., 3] == 0.0))
            self.assertTrue(np.all(t[..., 4] == 0.0))
            self.assertTrue(np.all(t[..., 5] == 1.0))
            self.assertTrue(np.all(t[..., 6] == 0.0))
            self.assertTrue(np.all(t[..., 7] == 1.0))
            for logical in ("normal", "roughness", "metalness", "ao"):
                record = result.records[logical]
                self.assertEqual(record.source, "constant")
                self.assertTrue(record.gate_exempt)
            self.assertEqual(result.records["base_color"].source, "map")
            self.assertIsNotNone(result.records["base_color"].sha256)

    def test_base_color_is_linearised(self):
        with tempfile.TemporaryDirectory() as td:
            folder = self._base_only_folder(td, value=(255, 0, 0))
            spec = resolve_inputs(folder, normal_convention="opengl")
            t = build_teacher(spec).teacher
            self.assertAlmostEqual(float(t[0, 0, 0]), 1.0, places=6)
            self.assertAlmostEqual(float(t[0, 0, 1]), 0.0, places=6)
        # Mid grey: sRGB 0.5 -> linear ~0.2140, not 0.5.
        self.assertAlmostEqual(float(srgb_to_linear(np.float32(0.5))),
                               0.21404114, places=5)

    def test_normal_xy_mapping(self):
        with tempfile.TemporaryDirectory() as td:
            folder = Path(td) / "mat"
            _save(_solid("RGB", (W, H), (0, 0, 0)), folder / "basecolor.png")
            # rgb (1.0, 0.0) -> xy (+1, -1) under OpenGL.
            _save(_solid("RGB", (W, H), (255, 0, 128)),
                  folder / "normal.png")
            spec = resolve_inputs(folder, normal_convention="opengl")
            t = build_teacher(spec).teacher
            self.assertAlmostEqual(float(t[0, 0, 3]), 1.0, places=5)
            self.assertAlmostEqual(float(t[0, 0, 4]), -1.0, places=5)

    def test_directx_flips_g_only(self):
        with tempfile.TemporaryDirectory() as td:
            folder = Path(td) / "mat"
            _save(_solid("RGB", (W, H), (0, 0, 0)), folder / "basecolor.png")
            # G = 0.25 in unit float -> y = -0.5 (OpenGL) / +0.5 (DirectX).
            _save(_solid("RGB", (W, H), (128, 64, 128)),
                  folder / "normal.png")
            gl = build_teacher(
                resolve_inputs(folder, normal_convention="opengl")).teacher
            dx = build_teacher(
                resolve_inputs(folder, normal_convention="directx")).teacher
            self.assertAlmostEqual(float(gl[0, 0, 4]), 64 / 255 * 2 - 1,
                                   places=5)
            self.assertAlmostEqual(float(dx[0, 0, 4]),
                                   -(64 / 255 * 2 - 1), places=5)
            # X is untouched by the convention flip.
            self.assertAlmostEqual(float(dx[0, 0, 3]), float(gl[0, 0, 3]),
                                   places=6)

    def test_scalar_maps_land_in_their_teacher_channels(self):
        with tempfile.TemporaryDirectory() as td:
            folder = Path(td) / "mat"
            _save(_solid("RGB", (W, H), (0, 0, 0)), folder / "basecolor.png")
            _save(_solid("L", (W, H), 51), folder / "roughness.png")
            _save(_solid("L", (W, H), 102), folder / "metalness.png")
            _save(_solid("L", (W, H), 153), folder / "ao.png")
            spec = resolve_inputs(folder, normal_convention="opengl")
            t = build_teacher(spec).teacher
            self.assertAlmostEqual(float(t[0, 0, 5]), 51 / 255, places=6)
            self.assertAlmostEqual(float(t[0, 0, 6]), 102 / 255, places=6)
            self.assertAlmostEqual(float(t[0, 0, 7]), 153 / 255, places=6)

    def test_unverified_size_is_flagged_not_rejected(self):
        with tempfile.TemporaryDirectory() as td:
            folder = self._base_only_folder(td)  # 8x8, not 2048x2048
            spec = resolve_inputs(folder, normal_convention="opengl")
            result = build_teacher(spec)
            self.assertFalse(result.size_verified)
            self.assertEqual((result.width, result.height), (W, H))

    def test_verified_size_flag(self):
        with tempfile.TemporaryDirectory() as td:
            folder = Path(td) / "mat"
            _save(_solid("L", (2048, 2048), 200), folder / "basecolor.png")
            spec = resolve_inputs(folder, normal_convention="opengl")
            result = build_teacher(spec)
            self.assertTrue(result.size_verified)


class TruncatedPngTest(unittest.TestCase):
    """A PNG cut short (e.g. an interrupted download) passes header-level
    checks but fails the pixel decode.  That must surface as an
    InputValidationError (CLI exit 2 with a reason), not as an unhandled
    OSError traceback from teacher construction."""

    def _noise_png_bytes(self) -> bytes:
        rng = np.random.default_rng(3)
        img = Image.fromarray(
            rng.integers(0, 255, size=(64, 64, 3), dtype=np.uint8))
        with tempfile.SpooledTemporaryFile() as buf:
            img.save(buf, format="PNG")
            buf.seek(0)
            return buf.read()

    def _folder_with(self, td: str, base_bytes: bytes,
                     roughness_bytes: bytes | None = None) -> Path:
        folder = Path(td) / "mat"
        folder.mkdir(parents=True)
        (folder / "basecolor.png").write_bytes(base_bytes)
        if roughness_bytes is not None:
            (folder / "roughness.png").write_bytes(roughness_bytes)
        return folder

    def test_truncated_base_color_is_input_error(self):
        data = self._noise_png_bytes()
        cut = data[: int(len(data) * 0.55)]
        with tempfile.TemporaryDirectory() as td:
            folder = self._folder_with(td, cut)
            spec = resolve_inputs(folder, normal_convention="opengl")
            with self.assertRaises(InputValidationError) as ctx:
                build_teacher(spec)
            self.assertIn("truncated or corrupt", str(ctx.exception))

    def test_truncated_scalar_map_is_input_error(self):
        good = self._noise_png_bytes()
        cut = good[: int(len(good) * 0.55)]
        with tempfile.TemporaryDirectory() as td:
            folder = self._folder_with(td, good, roughness_bytes=cut)
            spec = resolve_inputs(folder, normal_convention="opengl")
            with self.assertRaises(InputValidationError) as ctx:
                build_teacher(spec)
            self.assertIn("truncated or corrupt", str(ctx.exception))

    def test_intact_png_still_loads(self):
        data = self._noise_png_bytes()
        with tempfile.TemporaryDirectory() as td:
            folder = self._folder_with(td, data)
            spec = resolve_inputs(folder, normal_convention="opengl")
            result = build_teacher(spec)
            self.assertEqual((result.width, result.height), (64, 64))

    def test_preflight_exit_code_2_for_truncated(self):
        from vnm1 import cli

        data = self._noise_png_bytes()
        cut = data[: int(len(data) * 0.55)]
        with tempfile.TemporaryDirectory() as td:
            folder = self._folder_with(td, cut)
            code = cli.main(["preflight", str(folder),
                             "--normal-convention", "opengl"])
            self.assertEqual(code, cli.EXIT_INPUT_ERROR)


if __name__ == "__main__":
    unittest.main()

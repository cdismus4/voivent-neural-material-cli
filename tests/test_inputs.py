"""Unit tests for vnm1.inputs (texture-set resolution, design doc 4.1/4.2).

Resolution works on paths only, so dummy (empty) files are enough here;
image content is exercised in test_imaging.py.
"""

import json
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from vnm1.errors import InputValidationError
from vnm1.inputs import resolve_inputs, validate_material_name


def _touch(path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"dummy")
    return path


class ResolveStandardNamesTest(unittest.TestCase):
    def test_standard_filenames_and_folder_name(self):
        with tempfile.TemporaryDirectory() as td:
            folder = Path(td) / "oak_planks"
            for fname in ("basecolor.png", "normal.png", "roughness.png"):
                _touch(folder / fname)
            spec = resolve_inputs(folder, normal_convention="opengl")
            self.assertEqual(spec.name, "oak_planks")
            self.assertEqual(spec.maps["base_color"], folder / "basecolor.png")
            self.assertEqual(spec.sources["base_color"], "standard")
            self.assertEqual(spec.maps["normal"], folder / "normal.png")
            # Absent optional maps resolve to None (constants downstream).
            self.assertIsNone(spec.maps["metalness"])
            self.assertIsNone(spec.maps["ao"])
            self.assertEqual(spec.sources["ao"], "absent")
            self.assertEqual(spec.absent_maps(), ["metalness", "ao"])

    def test_missing_base_color_is_an_error(self):
        with tempfile.TemporaryDirectory() as td:
            folder = Path(td) / "empty_mat"
            folder.mkdir()
            with self.assertRaises(InputValidationError):
                resolve_inputs(folder, normal_convention="opengl")

    def test_missing_input_folder_is_an_error(self):
        with tempfile.TemporaryDirectory() as td:
            with self.assertRaises(InputValidationError):
                resolve_inputs(Path(td) / "does_not_exist",
                               normal_convention="opengl")


class ExplicitPathsTest(unittest.TestCase):
    def test_explicit_path_overrides_standard_name(self):
        with tempfile.TemporaryDirectory() as td:
            folder = Path(td) / "mat"
            _touch(folder / "basecolor.png")
            _touch(folder / "normal.png")
            custom = _touch(Path(td) / "elsewhere" / "my_normal.png")
            spec = resolve_inputs(
                folder, normal_convention="directx",
                explicit_maps={"normal": custom})
            self.assertEqual(spec.maps["normal"], custom)
            self.assertEqual(spec.sources["normal"], "explicit")
            self.assertEqual(spec.normal_convention, "directx")

    def test_explicit_path_must_exist(self):
        with tempfile.TemporaryDirectory() as td:
            folder = Path(td) / "mat"
            _touch(folder / "basecolor.png")
            with self.assertRaises(InputValidationError):
                resolve_inputs(
                    folder, normal_convention="opengl",
                    explicit_maps={"roughness": Path(td) / "nope.png"})

    def test_unknown_logical_map_rejected(self):
        with tempfile.TemporaryDirectory() as td:
            folder = Path(td) / "mat"
            _touch(folder / "basecolor.png")
            with self.assertRaises(InputValidationError):
                resolve_inputs(folder, normal_convention="opengl",
                               explicit_maps={"emissive": "x.png"})


class ConfigFileTest(unittest.TestCase):
    def _write_config(self, folder: Path, cfg: dict) -> Path:
        cfg_path = folder / "job.json"
        cfg_path.write_text(json.dumps(cfg), encoding="utf-8")
        return cfg_path

    def test_config_paths_resolve_relative_to_config_folder(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            maps = root / "maps"
            albedo = _touch(maps / "oak_albedo.png")
            normal = _touch(maps / "oak_normal_gl.png")
            cfg_path = self._write_config(root, {
                "name": "oak_planks",
                "input": {
                    "base_color": "./maps/oak_albedo.png",
                    "normal": "./maps/oak_normal_gl.png",
                    "normal_convention": "opengl",
                },
                "mode": "auto",
                "seed": 1337,
            })
            spec = resolve_inputs(config_path=cfg_path)
            self.assertEqual(spec.name, "oak_planks")
            self.assertEqual(spec.maps["base_color"], albedo)
            self.assertEqual(spec.maps["normal"], normal)
            self.assertEqual(spec.sources["base_color"], "config")
            self.assertEqual(spec.normal_convention, "opengl")
            self.assertEqual(spec.extra_config["seed"], 1337)

    def test_cli_values_override_config(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            _touch(root / "maps" / "a.png")
            override = _touch(root / "maps" / "b.png")
            cfg_path = self._write_config(root, {
                "name": "from_config",
                "input": {"base_color": "./maps/a.png",
                          "normal_convention": "opengl"},
            })
            spec = resolve_inputs(
                config_path=cfg_path, name="from_cli",
                normal_convention="directx",
                explicit_maps={"base_color": override})
            self.assertEqual(spec.name, "from_cli")
            self.assertEqual(spec.normal_convention, "directx")
            self.assertEqual(spec.maps["base_color"], override)
            self.assertEqual(spec.sources["base_color"], "explicit")

    def test_config_without_normal_convention_is_an_error(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            _touch(root / "maps" / "a.png")
            cfg_path = self._write_config(root, {
                "name": "x",
                "input": {"base_color": "./maps/a.png"},
            })
            with self.assertRaises(InputValidationError):
                resolve_inputs(config_path=cfg_path)


class NormalConventionTest(unittest.TestCase):
    def _folder(self, td: str) -> Path:
        folder = Path(td) / "mat"
        _touch(folder / "basecolor.png")
        return folder

    def test_missing_convention_is_an_error(self):
        with tempfile.TemporaryDirectory() as td:
            with self.assertRaises(InputValidationError):
                resolve_inputs(self._folder(td))

    def test_invalid_convention_is_an_error(self):
        with tempfile.TemporaryDirectory() as td:
            with self.assertRaises(InputValidationError):
                resolve_inputs(self._folder(td),
                               normal_convention="blender")


class MaterialNameTest(unittest.TestCase):
    def test_valid_names_pass_unchanged(self):
        for name in ("oak_planks", "石-01", "a.b_c", "Material (2)"):
            self.assertEqual(validate_material_name(name), name)

    def test_invalid_names_are_errors_not_replacements(self):
        for name in ("", "   ", ".", "..", "a/b", "a\\b", "a:b", "a?b",
                     "a*b", 'a"b', "a<b", "a|b", " name", "name ",
                     "name.", "a\tb"):
            with self.assertRaises(InputValidationError, msg=repr(name)):
                validate_material_name(name)

    def test_name_override(self):
        with tempfile.TemporaryDirectory() as td:
            folder = Path(td) / "folder_name"
            _touch(folder / "basecolor.png")
            spec = resolve_inputs(folder, name="chosen_name",
                                  normal_convention="opengl")
            self.assertEqual(spec.name, "chosen_name")


if __name__ == "__main__":
    unittest.main()

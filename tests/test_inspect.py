"""Unit tests for inspect (vnm1.inspect, step D): artifact format
verification from disk alone, tamper detection, report summary and
the inspect exit codes.

Artifacts are built with the product core's own serializers (no
training, no torch needed for the byte-level checks; the spot decode
runs when torch is present and is recorded as skipped otherwise).
"""

import json
import shutil
import tempfile
import unittest
from pathlib import Path

import numpy as np

from vnm1 import core as vt
from vnm1 import driver
from vnm1.inspect import inspect_target, verify_artifact


def build_artifact(dir_path: Path, *, profile: str = "standard") -> Path:
    """Write a structurally valid 4-file artifact (zero weights)."""
    dir_path.mkdir(parents=True, exist_ok=True)
    shapes = vt.weight_shapes(profile)
    weights = {k: np.zeros(shapes[k], dtype=np.float32)
               for k in vt.WEIGHT_KEYS}
    fine_ch, coarse_ch = vt.profile_latent_channels(profile)
    fine_size, coarse_size = (512, 512), (256, 256)
    fine_u8 = np.zeros((fine_size[1], fine_size[0], fine_ch),
                       dtype=np.uint8)
    coarse_u8 = np.zeros((coarse_size[1], coarse_size[0], coarse_ch),
                         dtype=np.uint8)
    weights_chunk = vt.serialize_weights_chunk(weights,
                                               profile=profile)
    latents_chunk = vt.serialize_latents_chunk(
        fine_u8, fine_size, coarse_u8, coarse_size,
        fine_channels=fine_ch, coarse_channels=coarse_ch)
    rng = np.random.default_rng(3)
    teacher = rng.random((64, 64, 8), dtype=np.float32)
    tail_chunk = driver.serialize_tail_chunk(teacher)
    tail_info = driver.parse_tail_chunk(tail_chunk)
    manifest = driver.build_manifest(
        "oak", 64, 64, weights_chunk, latents_chunk, tail_chunk,
        tail_info["tail_first_mip"], fine_size, coarse_size,
        profile=profile, fine_channels=fine_ch,
        coarse_channels=coarse_ch)
    (dir_path / "manifest.json").write_text(json.dumps(manifest,
                                                       indent=2))
    (dir_path / "weights.bin").write_bytes(weights_chunk)
    (dir_path / "level-0.bin").write_bytes(latents_chunk)
    (dir_path / "tail.bin").write_bytes(tail_chunk)
    return dir_path


def _report_dict(*, gate_passed: bool) -> dict:
    return {
        "material": "oak",
        "conclusion": {
            "result": "passed" if gate_passed else "failed",
            "gate_passed": gate_passed,
            "selected_profile": "standard" if gate_passed else None,
            "representative_profile": None if gate_passed
            else "standard",
            "representative_is_candidate_only": not gate_passed,
            "failed_checks": [] if gate_passed
            else ["psnr_base_color_db"],
            "gate_exempt_checks": ["mae_metalness"],
            "artifact_location": "/elsewhere/oak.vnm",
        },
        "trial": {"profile": "standard", "iterations": 50000,
                  "metrics": {"psnr_base_color_db": 38.9},
                  "checks": {"psnr_base_color_db": {
                      "value": 38.9, "threshold": 35.0, "passed": True,
                      "gate_applicable": True, "difference": 3.9}}},
        "capacity": {"vnm1_total_bytes": 12000,
                     "ratio_vs_bcn_mip_chain": 0.3},
    }


class VerifyArtifactTest(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp(prefix="vnm1-inspect-test-"))
        self.addCleanup(shutil.rmtree, self.tmp, True)
        self.artifact = build_artifact(self.tmp / "oak.vnm")

    def test_valid_artifact_verifies(self):
        result = verify_artifact(self.artifact)
        self.assertTrue(result["ok"],
                        [c for c in result["checks"] if not c["ok"]])
        names = {c["name"] for c in result["checks"]}
        self.assertIn("latents_reserved_bytes_zero", names)
        self.assertIn("manifest_references_match", names)
        self.assertIn("spot_decode", names)

    def test_missing_file_fails(self):
        (self.artifact / "tail.bin").unlink()
        result = verify_artifact(self.artifact)
        self.assertFalse(result["ok"])
        self.assertFalse(result["checks"][0]["ok"])

    def test_reserved_byte_violation_detected(self):
        data = bytearray((self.artifact / "level-0.bin").read_bytes())
        data[26] = 1  # subheader reserved byte (unified storage rule)
        (self.artifact / "level-0.bin").write_bytes(bytes(data))
        result = verify_artifact(self.artifact)
        self.assertFalse(result["ok"])
        reserved = next(c for c in result["checks"]
                        if c["name"] == "latents_reserved_bytes_zero")
        self.assertFalse(reserved["ok"])

    def test_tampered_weights_fail_reference_match(self):
        data = bytearray((self.artifact / "weights.bin").read_bytes())
        data[-1] ^= 0xFF
        (self.artifact / "weights.bin").write_bytes(bytes(data))
        result = verify_artifact(self.artifact)
        self.assertFalse(result["ok"])
        refs = next(c for c in result["checks"]
                    if c["name"] == "manifest_references_match")
        self.assertFalse(refs["ok"])

    def test_wrong_profile_in_manifest_fails(self):
        manifest = json.loads(
            (self.artifact / "manifest.json").read_text())
        manifest["profile"] = "quality8"
        (self.artifact / "manifest.json").write_text(
            json.dumps(manifest, indent=2))
        result = verify_artifact(self.artifact)
        self.assertFalse(result["ok"])


class InspectTargetTest(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp(prefix="vnm1-inspect-cli-"))
        self.addCleanup(shutil.rmtree, self.tmp, True)

    def _material_dir(self, *, gate_passed: bool) -> Path:
        material = self.tmp / "oak"
        material.mkdir(parents=True)
        build_artifact(material / "oak.vnm")
        (material / "oak.report.json").write_text(
            json.dumps(_report_dict(gate_passed=gate_passed)))
        return material

    def test_artifact_dir_exit_0(self):
        artifact = build_artifact(self.tmp / "oak.vnm")
        code, lines = inspect_target(artifact)
        self.assertEqual(code, 0)
        self.assertTrue(any("OK" in line for line in lines))

    def test_missing_target_exit_2(self):
        code, _lines = inspect_target(self.tmp / "nope")
        self.assertEqual(code, 2)

    def test_material_dir_passed_exit_0(self):
        code, lines = inspect_target(
            self._material_dir(gate_passed=True))
        self.assertEqual(code, 0)
        self.assertTrue(any("Material: oak" in line for line in lines))
        self.assertTrue(any("Result: PASSED" in line for line in lines))

    def test_material_dir_failed_report_exit_3(self):
        code, lines = inspect_target(
            self._material_dir(gate_passed=False))
        self.assertEqual(code, 3)
        self.assertTrue(any("Failed checks" in line for line in lines))

    def test_report_file_target(self):
        material = self._material_dir(gate_passed=True)
        code, _lines = inspect_target(material / "oak.report.json")
        self.assertEqual(code, 0)

    def test_broken_artifact_exit_5(self):
        material = self._material_dir(gate_passed=True)
        data = bytearray(
            (material / "oak.vnm" / "level-0.bin").read_bytes())
        data[26] = 7
        (material / "oak.vnm" / "level-0.bin").write_bytes(bytes(data))
        code, _lines = inspect_target(material)
        self.assertEqual(code, 5)


if __name__ == "__main__":
    unittest.main()

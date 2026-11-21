"""Visual check sheets for a trained artifact (design doc 7.4, step D).

Generalises the judgment run's visual-check scripts
(``visual-check/make_visual_check.py`` and the step C variant) into the
product package.  The sheets are a standard output of ``vnm1 train``,
not an optional extra: the numeric gate cannot show softening,
pattern loss or contour changes, so every run renders

- ``<material>_sheet.png`` -- [original | VNM decode | error heat map],
  each panel the native-resolution decode resized to 1024x1024;
- ``<material>_crop.png`` -- the centre crop at native resolution,
  [original | decode] (clamped to the input size when the input is
  smaller than 512x512, per the section 7.4 revision);
- ``<material>_normal_sheet.png`` -- [original | decode] normal maps
  as RGB visualisations, only when a real normal map was supplied.

The decode used here is the same read-back path as the quality
evaluation: the exported 4-file artifact is parsed from disk and
decoded with the product core at texel centres, mip 0.  Nothing is
drawn through a different, better-looking path.

Internal verification (section 7.4, as revised): the linear-space
base-color PSNR is recomputed on the float32 decode buffer *before*
any PNG encoding and must agree with the PSNR recorded in the report
(tolerance :data:`PSNR_TOLERANCE_DB`).  Recomputing from PNG pixels
would never agree (8-bit quantisation and the sRGB conversion shift
the values), so the check is deliberately done on the float buffer.
The heat map is a display-only visualisation (mean absolute sRGB
error amplified by :data:`HEAT_GAIN`, the same gain for every
material); it is never used for the verdict.

The "original" panels show the teacher the gate evaluated against
(the input maps after the input layer's convention unification, e.g.
a DirectX normal already G-flipped into the internal OpenGL
convention), so the comparison is exactly original-vs-decoded for the
data the numbers describe.

Requires torch (the core decoder is the torch path); the CLI only
calls this after a training run, where torch is already present.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
from PIL import Image

from . import core as vt
from .errors import ExportVerificationError
from .imaging import TeacherResult

#: Rows decoded per block (bounds the transient UV tensor size).
BLOCK_ROWS = 32

#: Display panel edge for the full sheet and the normal sheet.
SHEET_PANEL = 1024

#: Native-resolution centre crop edge (clamped to the input size).
CROP_SIZE = 512

#: Heat map amplification: mean |sRGB error| x 8, same for all
#: materials (design doc 7.4).
HEAT_GAIN = 8.0

#: Agreement required between the sheet decode's float-buffer PSNR
#: and the report PSNR (design doc 7.4 / section 13 item 6).  The
#: step C visual check observed deltas of ~0.001 dB between the
#: Kaggle GPU evaluation and a local CPU recompute; 0.05 dB leaves
#: room for device float differences without hiding a real mismatch.
PSNR_TOLERANCE_DB = 0.05

_HEAT_ANCHORS = np.array([
    [0, 0, 0], [120, 0, 0], [220, 40, 0], [255, 140, 0],
    [255, 220, 60], [255, 255, 255]], dtype=np.float32)
_HEAT_POS = np.array([0.0, 0.2, 0.4, 0.6, 0.8, 1.0], dtype=np.float32)


def linear_to_srgb(c: np.ndarray) -> np.ndarray:
    """Linear -> sRGB encoding (the judgment driver's formula)."""
    c = np.clip(c, 0.0, 1.0)
    return np.where(c <= 0.0031308, c * 12.92,
                    1.055 * np.power(c, 1.0 / 2.4) - 0.055
                    ).astype(np.float32)


def to_u8(x: np.ndarray) -> np.ndarray:
    return (np.clip(x, 0.0, 1.0) * 255.0).round().astype(np.uint8)


def heat_rgb(err: np.ndarray) -> np.ndarray:
    """(H, W) error in 0..1 -> (H, W, 3) uint8 hot-palette image."""
    idx = np.clip(err, 0.0, 1.0) * (len(_HEAT_POS) - 1)
    lo = np.floor(idx).astype(np.int32)
    hi = np.minimum(lo + 1, len(_HEAT_POS) - 1)
    frac = (idx - lo)[..., None]
    img = (_HEAT_ANCHORS[lo] * (1.0 - frac) + _HEAT_ANCHORS[hi] * frac)
    return img.round().astype(np.uint8)


def crop_bounds(height: int, width: int,
                size: int = CROP_SIZE) -> tuple[int, int, int, int]:
    """Centre crop (y0, x0, crop_h, crop_w), clamped to the input.

    Section 7.4 revision: an input smaller than the nominal crop is
    cropped at its real size instead of failing sheet generation.
    """
    crop_h = min(size, height)
    crop_w = min(size, width)
    return (height - crop_h) // 2, (width - crop_w) // 2, crop_h, crop_w


def load_decoder(artifact_dir: Path):
    """Parse a 4-file artifact directory from disk and build the CPU
    decoder -- the same read-back path as the evaluation."""
    artifact_dir = Path(artifact_dir)
    try:
        manifest = json.loads(
            (artifact_dir / "manifest.json").read_text(encoding="utf-8"))
        profile = manifest["profile"]
        weights = vt.parse_weights_chunk(
            (artifact_dir / "weights.bin").read_bytes(), profile=profile)
        latents = (artifact_dir / "level-0.bin").read_bytes()
        fine, fine_size, coarse, coarse_size = vt.parse_latents_chunk(
            latents, profile=profile)
        fine_ch, coarse_ch = vt.latents_chunk_channels(
            latents, profile=profile)
    except (OSError, ValueError, KeyError, json.JSONDecodeError) as exc:
        raise ExportVerificationError(
            f"visual sheets: cannot read artifact at {artifact_dir}: "
            f"{exc}") from exc
    decoder = vt.VnmDecoder.from_u8_numpy(
        weights, fine, coarse, fine_size, coarse_size, device="cpu",
        profile=profile, fine_channels=fine_ch,
        coarse_channels=coarse_ch)
    return decoder, manifest


def decode_artifact(artifact_dir: Path, teacher: np.ndarray,
                    want_normal: bool) -> dict:
    """Decode every texel of an artifact (texel centres, mip 0) in row
    blocks on CPU.

    Returns the float32 base-color buffer (linear, HxWx3), the float32
    normal buffer (HxWx3) or None, and the linear PSNR recomputed on
    that buffer against ``teacher`` (the section 7.4 internal check).
    """
    torch = vt.require_torch()
    decoder, manifest = load_decoder(artifact_dir)
    height, width = int(teacher.shape[0]), int(teacher.shape[1])
    base = np.empty((height, width, 3), dtype=np.float32)
    normal = (np.empty((height, width, 3), dtype=np.float32)
              if want_normal else None)
    sse = 0.0
    with torch.no_grad():
        for y0 in range(0, height, BLOCK_ROWS):
            ys = np.arange(y0, min(y0 + BLOCK_ROWS, height))
            xs = np.arange(width)
            uu, vv = np.meshgrid((xs + 0.5) / width, (ys + 0.5) / height)
            uv = torch.tensor(
                np.stack([uu.ravel(), vv.ravel()], axis=1),
                dtype=torch.float32)
            pbr = decoder.decode_pbr(uv, normalized_mip=0.0)
            block = pbr["baseColor"].numpy().reshape(len(ys), width, 3)
            base[y0:y0 + len(ys)] = block
            diff = teacher[y0:y0 + len(ys), :, 0:3] - block
            sse += float(np.sum(diff * diff))
            if want_normal:
                normal[y0:y0 + len(ys)] = pbr["normal"].numpy().reshape(
                    len(ys), width, 3)
    mse = sse / (width * height * 3)
    psnr = -10.0 * math.log10(max(mse, 1e-12))
    return {"base_linear": base, "normal": normal,
            "psnr_db": float(psnr), "manifest": manifest}


def _panel(img_u8: np.ndarray, size: int = SHEET_PANEL) -> Image.Image:
    return Image.fromarray(img_u8).resize((size, size), Image.LANCZOS)


def _teacher_normal_rgb(teacher: np.ndarray) -> np.ndarray:
    """Original-normal RGB visualisation from the teacher xy channels
    (z reconstructed by the decoder convention, then normalised)."""
    xy = teacher[..., 3:5].astype(np.float32)
    z = np.sqrt(np.maximum(0.0, 1.0 - np.sum(xy * xy, axis=-1)))
    n = np.concatenate([xy, z[..., None]], axis=-1)
    length = np.maximum(np.linalg.norm(n, axis=-1, keepdims=True), 1e-12)
    return to_u8((n / length) * 0.5 + 0.5)


def generate_visual_sheets(artifact_dir: Path, teacher: TeacherResult,
                           material_name: str, out_dir: Path, *,
                           expected_psnr_db: float) -> dict:
    """Render the section 7.4 sheets for one artifact and verify the
    sheet decode against the report PSNR.

    ``teacher`` is the input layer's TeacherResult for the run;
    ``expected_psnr_db`` is the chosen trial's recorded PSNR.  Raises
    :class:`ExportVerificationError` when the float-buffer PSNR
    disagrees beyond :data:`PSNR_TOLERANCE_DB` (a real decode-path
    mismatch, reported as exit 5 by the CLI) or when a sheet cannot
    be written.  Returns the ``visual_check`` record for the report.
    """
    out_dir = Path(out_dir)
    want_normal = teacher.records["normal"].source == "map"
    decoded = decode_artifact(artifact_dir, teacher.teacher, want_normal)
    recomputed = decoded["psnr_db"]
    delta = recomputed - float(expected_psnr_db)
    if abs(delta) > PSNR_TOLERANCE_DB:
        raise ExportVerificationError(
            f"visual sheet PSNR check failed for {material_name}: "
            f"sheet decode gives {recomputed:.4f} dB, report records "
            f"{expected_psnr_db:.4f} dB (delta {delta:+.4f} dB, "
            f"tolerance {PSNR_TOLERANCE_DB} dB); the sheets were not "
            f"written")

    base_linear = decoded["base_linear"]
    orig_srgb = linear_to_srgb(teacher.teacher[..., 0:3])
    dec_srgb = linear_to_srgb(base_linear)
    err = np.mean(np.abs(orig_srgb - dec_srgb), axis=2)
    orig_u8, dec_u8 = to_u8(orig_srgb), to_u8(dec_srgb)
    height, width = orig_u8.shape[:2]

    files: dict[str, dict] = {}
    try:
        out_dir.mkdir(parents=True, exist_ok=True)

        heat = heat_rgb(np.clip(err * HEAT_GAIN, 0.0, 1.0))
        sheet = Image.new("RGB", (SHEET_PANEL * 3, SHEET_PANEL))
        sheet.paste(_panel(orig_u8), (0, 0))
        sheet.paste(_panel(dec_u8), (SHEET_PANEL, 0))
        sheet.paste(Image.fromarray(heat).resize(
            (SHEET_PANEL, SHEET_PANEL), Image.BILINEAR),
            (SHEET_PANEL * 2, 0))
        path = out_dir / f"{material_name}_sheet.png"
        sheet.save(path)
        files[path.name] = {"bytes": path.stat().st_size,
                            "kind": "sheet"}

        y0, x0, crop_h, crop_w = crop_bounds(height, width)
        crop_img = Image.new("RGB", (crop_w * 2, crop_h))
        crop_img.paste(Image.fromarray(
            orig_u8[y0:y0 + crop_h, x0:x0 + crop_w]), (0, 0))
        crop_img.paste(Image.fromarray(
            dec_u8[y0:y0 + crop_h, x0:x0 + crop_w]), (crop_w, 0))
        path = out_dir / f"{material_name}_crop.png"
        crop_img.save(path)
        files[path.name] = {"bytes": path.stat().st_size,
                            "kind": "crop"}

        if want_normal:
            dec_n_u8 = to_u8(decoded["normal"] * 0.5 + 0.5)
            n_sheet = Image.new("RGB", (SHEET_PANEL * 2, SHEET_PANEL))
            n_sheet.paste(_panel(_teacher_normal_rgb(teacher.teacher)),
                          (0, 0))
            n_sheet.paste(_panel(dec_n_u8), (SHEET_PANEL, 0))
            path = out_dir / f"{material_name}_normal_sheet.png"
            n_sheet.save(path)
            files[path.name] = {"bytes": path.stat().st_size,
                                "kind": "normal_sheet"}
    except OSError as exc:
        raise ExportVerificationError(
            f"visual sheets: cannot write sheets under {out_dir}: "
            f"{exc}") from exc

    return {
        "files": files,
        "artifact_dir": str(artifact_dir),
        "recomputed_psnr_db": recomputed,
        "report_psnr_db": float(expected_psnr_db),
        "psnr_delta_db": delta,
        "psnr_tolerance_db": PSNR_TOLERANCE_DB,
        "psnr_match": True,
        "heat_gain": HEAT_GAIN,
        "note": "The heat map is the mean absolute sRGB error "
                f"amplified x{HEAT_GAIN:g} for visibility; it is not "
                "an actual colour and is not used for the verdict. "
                "Sheet PSNR is recomputed on the float32 decode "
                "buffer before PNG encoding (design doc 7.4).",
    }


__all__ = [
    "HEAT_GAIN", "PSNR_TOLERANCE_DB", "crop_bounds", "decode_artifact",
    "generate_visual_sheets", "heat_rgb", "linear_to_srgb",
    "load_decoder", "to_u8",
]

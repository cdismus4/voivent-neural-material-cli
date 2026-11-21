"""Image loading, validation and teacher construction (design doc 4.3-4.5).

numpy + Pillow only; this module never imports torch.  The numeric
conventions deliberately match the judgment-run driver
(``stage2/src/train_stage2.py``) so that product results stay comparable
with the judgment run:

- bit-depth normalisation by dtype: bool (1-bit data, PIL mode "1")
  -> 0.0 / 1.0 as-is, uint8 -> /255, anything else (16-bit integer
  data) -> /65535 (driver ``_to_unit_float``, plus the 1-bit case the
  driver never met: a bool array divided by 65535 silently scales a
  constant 1.0 metalness to ~1.5e-5);
- base color is read as sRGB and converted to linear before training;
- normal xy is read as ``rgb * 2 - 1`` and Z is reconstructed by the
  decoder convention (it is not stored in the teacher);
- missing maps become constants (normal ``(0, 0)``, roughness ``1.0``,
  metalness ``0.0``, AO ``1.0``) and are recorded as gate-exempt, never
  as "recovered" data.

On top of the driver's behaviour, the product input layer enforces the
section 4 rules the driver did not need: PNG-only inputs, dimension
agreement across maps, base-color alpha rules, and the scalar-map
RGB/RGBA acceptance rules with the packed-map (ORM) suspicion warning.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from PIL import Image

from .errors import InputValidationError
from .inputs import InputSpec, LOGICAL_MAPS

#: The resolution the judgment run's capability table was measured at.
VERIFIED_SIZE = (2048, 2048)

#: Teacher constants for maps with no source file (driver CONSTANT_CHANNELS).
CONSTANT_VALUES = {
    "normal": (0.0, 0.0),
    "roughness": 1.0,
    "metalness": 0.0,
    "ao": 1.0,
}

#: Teacher channel index of each scalar map.
_SCALAR_CHANNEL = {"roughness": 5, "metalness": 6, "ao": 7}

#: What to do when a scalar map's RGB channels differ (packed-map
#: suspicion, section 4.3): "warn" (default) or "error".
PACKED_POLICIES = ("warn", "error")


@dataclass
class MapRecord:
    """Per-map record for the report (design doc section 7.2 B)."""

    logical: str
    source: str  # "map" or "constant"
    file: str | None = None
    sha256: str | None = None
    width: int | None = None
    height: int | None = None
    pil_mode: str | None = None
    bit_depth: int | None = None
    constant_value: float | tuple | None = None
    gate_exempt: bool = False


@dataclass
class TeacherResult:
    """The 8-channel teacher plus everything the report needs."""

    teacher: np.ndarray  # float32 (H, W, 8)
    width: int
    height: int
    size_verified: bool  # True only for the verified 2048x2048 baseline
    records: dict[str, MapRecord]
    warnings: list[str] = field(default_factory=list)


def packed_suspicion_warnings(warnings: list[str]) -> list[str]:
    """The packed-map suspicion warnings among ``warnings``.

    These are the warnings :func:`load_scalar` emits when a scalar
    map's RGB channels differ.  They matter beyond preflight: a run
    can pass the numeric gate while grading the wrong channel (the
    round-2 ARM case), so the CLI carries them to the train
    conclusion and the report summary (Neon O5, 2026-11-21).
    """
    return [w for w in warnings if "packed-map suspicion" in w]


def unsupported_file_warning(filename: str) -> str:
    """The exclusion notice for one unconsumed input-folder file."""
    return (f"unsupported map excluded: {filename} is in the input "
            "folder but is not one of the VNM1 input maps "
            "(basecolor / normal / roughness / metalness / ao); "
            "extra maps such as emissive, displacement and height "
            "are not supported in P1, so the file was excluded")


def file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _open_png(path: Path) -> Image.Image:
    """Open an image and enforce the PNG-only rule (section 4.4)."""
    try:
        img = Image.open(path)
    except FileNotFoundError:
        raise InputValidationError(f"map file not found: {path}")
    except Exception as exc:  # unreadable / not an image at all
        raise InputValidationError(f"cannot read image {path}: {exc}")
    if img.format != "PNG":
        img.close()
        raise InputValidationError(
            f"only PNG input is supported in P1, got "
            f"{img.format or 'unknown'}: {path}")
    # Header-level checks pass for a truncated file; only a full decode
    # proves the pixel data is all there.  A download cut short is a
    # realistic input, so a decode failure here is an input error
    # (exit 2), not an unhandled exception.
    try:
        img.load()
    except (OSError, EOFError) as exc:
        img.close()
        raise InputValidationError(
            f"image data cannot be fully read (the file may be "
            f"truncated or corrupt): {path}: {exc}") from exc
    return img


def _to_unit_float(img: Image.Image) -> np.ndarray:
    """PIL image -> float32 array in 0..1, honouring the source bit depth.

    Same rule as the judgment driver for integer data: uint8 is divided
    by 255, anything else (16-bit integer data) by 65535.  Bool arrays
    (1-bit PNGs, PIL mode "1") are handled first: their values already
    are 0.0 / 1.0, and dividing them by 65535 would silently scale a
    constant-1.0 map to ~1.5e-5.
    """
    arr = np.asarray(img)
    if arr.dtype == np.bool_:
        return arr.astype(np.float32)
    if arr.dtype == np.uint8:
        return arr.astype(np.float32) / 255.0
    return arr.astype(np.float32) / 65535.0


_PNG_SIGNATURE = b"\x89PNG\r\n\x1a\n"


def _bit_depth(img: Image.Image, path: Path | None = None) -> int:
    """Bit depth per channel as stored in the file (PNG IHDR).

    The post-load dtype is not a faithful record of the file: Pillow
    loads a 16-bit RGB PNG as uint8 (so the dtype guess reported 8 for
    a 16-bit file) and a 1-bit PNG as bool (reported as 16).  The
    report therefore reads the IHDR bit-depth byte from the file
    itself; the dtype rule remains as a fallback for images without a
    readable file header.
    """
    if path is not None:
        try:
            with open(path, "rb") as fh:
                header = fh.read(26)
        except OSError:
            header = b""
        if len(header) == 26 and header[:8] == _PNG_SIGNATURE \
                and header[12:16] == b"IHDR":
            return int(header[24])
    return 8 if np.asarray(img).dtype == np.uint8 else 16


def _normalised(img: Image.Image) -> Image.Image:
    """Palette images carry indices, not colours; expand them to RGB(A).

    Every other mode (L, LA, RGB, RGBA, I, I;16) is passed through so the
    driver's dtype-based normalisation sees the original values.
    """
    if img.mode == "P":
        return img.convert("RGBA" if "transparency" in img.info else "RGB")
    return img


def _split_alpha(arr: np.ndarray, img: Image.Image):
    """Return (colour array without alpha, alpha array or None)."""
    bands = img.getbands()
    alpha_band = "A" if "A" in bands else ("a" if "a" in bands else None)
    if alpha_band is not None:
        idx = bands.index(alpha_band)
        alpha = np.take(arr, idx, axis=-1) if arr.ndim == 3 else None
        if arr.ndim == 3 and arr.shape[-1] == len(bands):
            colour = np.delete(arr, idx, axis=-1)
        else:
            colour = arr
        return colour, alpha
    if img.mode == "P" and "transparency" in img.info:
        rgba = np.asarray(img.convert("RGBA")).astype(np.float32) / 255.0
        return rgba[..., :3], rgba[..., 3]
    return arr, None


def _load_rgb_unit(path: Path):
    """Load an image as unit-float RGB (H, W, 3) plus alpha and metadata.

    Mirrors the driver ``_load_rgb``: a 2-D image is stacked to 3
    channels and only the first 3 channels are used.
    """
    with _open_png(path) as raw:
        img = _normalised(raw)
        mode = img.mode
        depth = _bit_depth(img, path)
        arr = _to_unit_float(img)
        colour, alpha = _split_alpha(arr, img)
    if colour.ndim == 3 and colour.shape[-1] == 1:
        colour = colour[..., 0]
    if colour.ndim == 2:
        colour = np.stack([colour, colour, colour], axis=-1)
    rgb = colour[..., :3]
    return rgb, alpha, mode, depth


def srgb_to_linear(c: np.ndarray) -> np.ndarray:
    """sRGB -> linear, same formula as the judgment driver."""
    return np.where(c <= 0.04045, c / 12.92,
                    ((c + 0.055) / 1.055) ** 2.4).astype(np.float32)


def load_base_color(path: Path):
    """Load the base color map (section 4.3).

    Returns (rgb unit-float (H, W, 3) in sRGB encoding, warnings).
    An alpha channel always produces a warning; any non-opaque pixel is
    an error because alpha is not part of the VNM1 teacher in P1.
    """
    rgb, alpha, _mode, _depth = _load_rgb_unit(path)
    warnings: list[str] = []
    if alpha is not None:
        warnings.append(
            f"base_color: alpha channel present in {path.name}; alpha is "
            "not part of the VNM1 teacher and is ignored")
        if float(alpha.min()) < 1.0:
            raise InputValidationError(
                f"base_color: {path.name} has non-opaque pixels; alpha / "
                "transparency is not supported in P1 (design doc 4.3)")
    return rgb, warnings


def load_scalar(path: Path, logical: str, *, packed_policy: str = "warn"):
    """Load a roughness / metalness / AO map as a unit-float (H, W) array.

    Section 4.3 acceptance rules: a grayscale map is used directly; an
    RGB map with ``R == G == B`` and an RGBA map with fully opaque alpha
    are accepted silently as a single channel (production tools often
    export scalar maps that way).  If the RGB channels differ, a packed
    map (e.g. ORM) may have been supplied by mistake: emit a strong
    warning, or raise when ``packed_policy == "error"``.  Pack splitting
    is not done in P1; the first channel is used, as in the driver.
    """
    if packed_policy not in PACKED_POLICIES:
        raise ValueError(f"packed_policy must be one of {PACKED_POLICIES}")
    with _open_png(path) as raw:
        img = _normalised(raw)
        arr = _to_unit_float(img)
        colour, alpha = _split_alpha(arr, img)
    warnings: list[str] = []
    if colour.ndim == 2:
        return colour, warnings
    if colour.shape[-1] == 1:
        return colour[..., 0], warnings
    rgb = colour[..., :3]
    if np.array_equal(rgb[..., 0], rgb[..., 1]) and \
            np.array_equal(rgb[..., 0], rgb[..., 2]):
        scalar = rgb[..., 0]
    else:
        msg = (f"{logical}: {path.name} has differing RGB channels; a "
               "packed map (e.g. ORM) may have been supplied by mistake. "
               "P1 does not split packed maps; only the first channel "
               "would be used")
        if packed_policy == "error":
            raise InputValidationError(msg)
        warnings.append("WARNING (packed-map suspicion): " + msg)
        scalar = rgb[..., 0]
    if alpha is not None and float(alpha.min()) < 1.0:
        warnings.append(
            f"{logical}: {path.name} has a non-opaque alpha channel; "
            "alpha is ignored for scalar maps")
    return scalar, warnings


def build_teacher(spec: InputSpec, *, packed_policy: str = "warn") -> TeacherResult:
    """Build the (H, W, 8) teacher and the per-map records for a spec.

    Teacher layout (same as the judgment driver): channels 0-2 base
    color (linear), 3-4 normal xy, 5 roughness, 6 metalness, 7 AO.
    A ``"directx"`` normal convention flips the G channel on input so
    the teacher is always in the OpenGL convention internally
    (section 4.3).
    """
    warnings: list[str] = []
    records: dict[str, MapRecord] = {}

    # Unconsumed files in the input folder are excluded explicitly,
    # never silently ignored (Neon O5, 2026-11-21).
    for filename in spec.unsupported_files:
        warnings.append(unsupported_file_warning(filename))

    base_path = spec.maps["base_color"]
    assert base_path is not None  # guaranteed by resolve_inputs
    base_srgb, base_warnings = load_base_color(base_path)
    warnings.extend(base_warnings)
    height, width = int(base_srgb.shape[0]), int(base_srgb.shape[1])

    def _record(logical: str, path: Path) -> MapRecord:
        with _open_png(path) as raw:
            img = _normalised(raw)
            w, h = img.size
            mode, depth = img.mode, _bit_depth(img, path)
        return MapRecord(logical=logical, source="map", file=str(path),
                         sha256=file_sha256(path), width=w, height=h,
                         pil_mode=mode, bit_depth=depth)

    def _check_dims(logical: str, shape) -> None:
        if tuple(shape[:2]) != (height, width):
            raise InputValidationError(
                f"{logical} map size {tuple(shape[:2])} does not match "
                f"base_color size {(height, width)}; automatic resizing "
                "is not performed (design doc 4.4)")

    teacher = np.zeros((height, width, 8), dtype=np.float32)
    teacher[..., 0:3] = srgb_to_linear(base_srgb)
    records["base_color"] = _record("base_color", base_path)

    normal_path = spec.maps.get("normal")
    if normal_path is not None:
        n_rgb, _alpha, _m, _d = _load_rgb_unit(normal_path)
        _check_dims("normal", n_rgb.shape)
        xy = n_rgb[..., 0:2] * 2.0 - 1.0
        if spec.normal_convention == "directx":
            # G flip only: y -> -y turns Y- (DirectX) into Y+ (OpenGL).
            xy = xy.copy()
            xy[..., 1] = -xy[..., 1]
        teacher[..., 3:5] = xy
        records["normal"] = _record("normal", normal_path)
    else:
        teacher[..., 3] = CONSTANT_VALUES["normal"][0]
        teacher[..., 4] = CONSTANT_VALUES["normal"][1]
        records["normal"] = MapRecord(
            logical="normal", source="constant",
            constant_value=CONSTANT_VALUES["normal"], gate_exempt=True)

    for logical in ("roughness", "metalness", "ao"):
        path = spec.maps.get(logical)
        channel = _SCALAR_CHANNEL[logical]
        if path is not None:
            scalar, scalar_warnings = load_scalar(
                path, logical, packed_policy=packed_policy)
            warnings.extend(scalar_warnings)
            _check_dims(logical, scalar.shape)
            teacher[..., channel] = scalar
            records[logical] = _record(logical, path)
        else:
            teacher[..., channel] = CONSTANT_VALUES[logical]
            records[logical] = MapRecord(
                logical=logical, source="constant",
                constant_value=CONSTANT_VALUES[logical], gate_exempt=True)

    return TeacherResult(
        teacher=teacher,
        width=width,
        height=height,
        size_verified=(width, height) == VERIFIED_SIZE,
        records=records,
        warnings=warnings,
    )

"""Texture-set resolution for the vnm1 CLI (design doc section 4.1/4.2).

This module decides *which files* make up one material's texture set and
validates the material name and the normal convention.  It works purely
on paths and JSON; it never opens image data (that is
:mod:`vnm1.imaging`) and never imports torch or numpy, so the resolution
rules can be unit-tested in isolation.

Resolution rules (design doc section 4.2), highest priority first:

1. an explicitly given per-map path (CLI option or config entry),
2. the standard file name inside the input folder,
3. "absent" for optional maps (they become constants downstream and are
   recorded as gate-exempt; the required base color map has no fallback).

``normal_convention`` is mandatory (section 4.3): it must come from the
CLI or the config file, must be ``"opengl"`` or ``"directx"``, and is
never silently defaulted.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

from .errors import InputValidationError

#: Logical map names, in teacher-channel order.
LOGICAL_MAPS = ("base_color", "normal", "roughness", "metalness", "ao")

#: Standard file names inside a material folder (section 4.2 A).
#: Same convention as the judgment-run driver (stage2 MAP_FILES).
STANDARD_FILENAMES = {
    "base_color": "basecolor.png",
    "normal": "normal.png",
    "roughness": "roughness.png",
    "metalness": "metalness.png",
    "ao": "ao.png",
}

#: Maps that must resolve to a real file.
REQUIRED_MAPS = ("base_color",)

#: Accepted values for ``normal_convention`` (section 4.3).
NORMAL_CONVENTIONS = ("opengl", "directx")

#: File suffixes treated as image files when scanning an input folder
#: for unconsumed extras (Neon O5, 2026-11-21).  A file with one of
#: these suffixes that no resolution rule consumed is almost always
#: an extra map the user expected to be used (emissive, displacement,
#: height, opacity, ...); P1 supports none of those, so the file is
#: recorded as excluded instead of being silently ignored.
IMAGE_SUFFIXES = (".png", ".jpg", ".jpeg", ".tif", ".tiff", ".tga",
                  ".bmp", ".exr", ".webp", ".hdr")

# Characters that must never appear in a material name: path separators
# and characters that are invalid in file names on common filesystems.
_INVALID_NAME_CHARS = set('/\\:*?"<>|')


def validate_material_name(name: str) -> str:
    """Return ``name`` unchanged, or raise :class:`InputValidationError`.

    Section 4.1: a material name that cannot be used as a file name is
    an error at validation time; it is never silently replaced.
    """
    if not isinstance(name, str) or not name.strip():
        raise InputValidationError("material name must not be empty")
    if name != name.strip():
        raise InputValidationError(
            f"material name has leading/trailing whitespace: {name!r}")
    if name in (".", ".."):
        raise InputValidationError(f"invalid material name: {name!r}")
    bad = sorted({c for c in name
                  if c in _INVALID_NAME_CHARS or ord(c) < 32 or ord(c) == 127})
    if bad:
        raise InputValidationError(
            f"material name {name!r} contains characters that cannot be "
            f"used in file names: {''.join(bad)!r}")
    if name.endswith(".") or name.endswith(" "):
        raise InputValidationError(
            f"material name must not end with '.' or a space: {name!r}")
    return name


@dataclass
class InputSpec:
    """A fully resolved texture set for one material.

    ``maps`` maps every logical map name to a resolved absolute
    :class:`~pathlib.Path`, or to ``None`` when an optional map is
    absent (constant downstream).  ``sources`` records where each path
    came from: ``"explicit"``, ``"config"``, ``"standard"`` or
    ``"absent"``.
    """

    name: str
    normal_convention: str
    maps: dict[str, Path | None]
    sources: dict[str, str]
    input_dir: Path | None = None
    config_path: Path | None = None
    extra_config: dict = field(default_factory=dict)
    #: Image files found in the input folder that no resolution rule
    #: consumed (file names only, sorted).  Recorded so the input
    #: layer can state explicitly that they were excluded as
    #: unsupported instead of silently ignoring them (Neon O5).
    unsupported_files: list[str] = field(default_factory=list)

    def present_maps(self) -> list[str]:
        return [m for m in LOGICAL_MAPS if self.maps.get(m) is not None]

    def absent_maps(self) -> list[str]:
        return [m for m in LOGICAL_MAPS if self.maps.get(m) is None]


def _load_config(config_path: Path) -> dict:
    try:
        with open(config_path, "r", encoding="utf-8") as fh:
            data = json.load(fh)
    except FileNotFoundError:
        raise InputValidationError(f"config file not found: {config_path}")
    except json.JSONDecodeError as exc:
        raise InputValidationError(
            f"config file is not valid JSON: {config_path}: {exc}")
    if not isinstance(data, dict):
        raise InputValidationError(
            f"config file must contain a JSON object: {config_path}")
    return data


def resolve_inputs(
    input_dir: str | Path | None = None,
    *,
    name: str | None = None,
    explicit_maps: dict[str, str | Path | None] | None = None,
    normal_convention: str | None = None,
    config_path: str | Path | None = None,
) -> InputSpec:
    """Resolve one material's texture set (design doc section 4.2).

    CLI-level values (``explicit_maps``, ``normal_convention``, ``name``)
    take priority over config-file values (section 3.2).  Relative paths
    inside a config file are resolved against the config file's folder
    (section 4.2 C); relative CLI paths are resolved against the current
    working directory.
    """
    cfg: dict = {}
    cfg_dir: Path | None = None
    cfg_file: Path | None = None
    if config_path is not None:
        cfg_file = Path(config_path).expanduser().resolve()
        cfg = _load_config(cfg_file)
        cfg_dir = cfg_file.parent

    cfg_input = cfg.get("input") or {}
    if not isinstance(cfg_input, dict):
        raise InputValidationError("config 'input' must be a JSON object")

    def _resolve_path(raw: str | Path, base: Path | None) -> Path:
        p = Path(raw).expanduser()
        if not p.is_absolute():
            p = (base or Path.cwd()) / p
        return p.resolve()

    # --- input folder -------------------------------------------------
    raw_dir = input_dir if input_dir is not None else cfg.get("input_dir")
    dir_base = None if input_dir is not None else cfg_dir
    resolved_dir: Path | None = None
    if raw_dir is not None:
        resolved_dir = _resolve_path(raw_dir, dir_base)
        if not resolved_dir.is_dir():
            raise InputValidationError(
                f"input folder does not exist: {resolved_dir}")

    # --- per-map paths ------------------------------------------------
    explicit_maps = explicit_maps or {}
    unknown = set(explicit_maps) - set(LOGICAL_MAPS)
    if unknown:
        raise InputValidationError(
            f"unknown logical map name(s): {sorted(unknown)}")
    unknown_cfg = set(cfg_input) - set(LOGICAL_MAPS) - {"normal_convention"}
    if unknown_cfg:
        raise InputValidationError(
            f"unknown key(s) in config 'input': {sorted(unknown_cfg)}")

    maps: dict[str, Path | None] = {}
    sources: dict[str, str] = {}
    for logical in LOGICAL_MAPS:
        chosen: Path | None = None
        source = "absent"
        if explicit_maps.get(logical) is not None:
            chosen = _resolve_path(explicit_maps[logical], None)
            source = "explicit"
        elif cfg_input.get(logical) is not None:
            chosen = _resolve_path(cfg_input[logical], cfg_dir)
            source = "config"
        elif resolved_dir is not None:
            candidate = resolved_dir / STANDARD_FILENAMES[logical]
            if candidate.is_file():
                chosen = candidate
                source = "standard"
        if chosen is not None and not chosen.is_file():
            raise InputValidationError(
                f"{logical} map file does not exist: {chosen}")
        if chosen is None and logical in REQUIRED_MAPS:
            raise InputValidationError(
                f"{logical} map is required but was not found "
                f"(standard file name: {STANDARD_FILENAMES[logical]})")
        maps[logical] = chosen
        sources[logical] = source

    # --- normal convention (mandatory, section 4.3) -------------------
    convention = normal_convention
    if convention is None:
        convention = cfg_input.get("normal_convention")
    if convention is None:
        raise InputValidationError(
            "normal_convention is required: specify \"opengl\" or "
            "\"directx\" (there is no default; see design doc 4.3)")
    if convention not in NORMAL_CONVENTIONS:
        raise InputValidationError(
            f"normal_convention must be one of {NORMAL_CONVENTIONS}, "
            f"got {convention!r}")

    # --- material name --------------------------------------------------
    raw_name = name if name is not None else cfg.get("name")
    if raw_name is None:
        if resolved_dir is None:
            raise InputValidationError(
                "material name could not be determined: pass an input "
                "folder, --name, or a config 'name'")
        raw_name = resolved_dir.name
    material_name = validate_material_name(str(raw_name))

    extra = {k: v for k, v in cfg.items()
             if k not in ("input", "name", "input_dir")}

    # --- unconsumed image files in the input folder (Neon O5) -------
    # Extra maps (emissive, displacement, height, ...) are not P1
    # inputs.  They used to be ignored without a word; now they are
    # recorded on the spec so preflight / train / the report can state
    # that they were excluded as unsupported.
    unsupported: list[str] = []
    if resolved_dir is not None:
        consumed = {p for p in maps.values() if p is not None}
        for entry in sorted(resolved_dir.iterdir()):
            if entry.is_file() \
                    and entry.suffix.lower() in IMAGE_SUFFIXES \
                    and entry.resolve() not in consumed:
                unsupported.append(entry.name)

    return InputSpec(
        name=material_name,
        normal_convention=convention,
        maps=maps,
        sources=sources,
        input_dir=resolved_dir,
        config_path=cfg_file,
        extra_config=extra,
        unsupported_files=unsupported,
    )

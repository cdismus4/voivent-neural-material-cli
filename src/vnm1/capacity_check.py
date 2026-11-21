"""Capacity entry check and outcome warnings (2026-11-19 addition).

Two-stage capacity discipline, approved by the user on 2026-11-19
after the step E corpus run showed that small / single-map inputs
silently *expand* under VNM1 (a 64x64 base-color-only texture came
out at 240x its BCn size): a VNM1 artifact's size is set by the
profile's format parameters (latent grids + MLP weights), not by the
input size, while the BCn estimate scales with pixels x maps.  Below
a crossover the conversion can never save capacity, no matter which
profile passes the quality gate (profile sizes grow monotonically
along the ladder: standard < standard8 < quality < quality8).

Stage 1 -- entry check (this module's :func:`capacity_entry_check`),
run by the CLI after input validation and before any training: if
even the cheapest profile (``standard``) is estimated at or above
the BCn full mip chain estimate, the material is rejected as an
input error (exit 2, the same route as the alpha rejection) unless
the user overrides with ``--allow-capacity-expansion``; the
override is recorded in the report.

Stage 2 -- outcome warnings (:func:`capacity_warnings`), attached
to the report of a finished run: a passing artifact whose measured
capacity ratio is >= 0.9 carries a "thin margin" warning, and one
at >= 1.0 an "expansion" warning (e.g. a 1K material that only
passed at quality can end up larger than BCn even though it cleared
the entry check at standard).

Both stages reuse the report's existing BCn estimate
(:func:`vnm1.driver.bcn_estimate`) verbatim -- no second formula.
The VNM1 side is not a hard-coded size either: the estimate is
derived from the profile's format parameters by running the real
serializers on zero-filled tensors and the real manifest builder,
so it tracks the format if the format ever changes.  (Verified
byte-exact against the four real artifacts in the P2 handoff
package, given the same material name.)
"""

from __future__ import annotations

import json
import struct

import numpy as np

from . import core as vt
from .driver import (PROFILE_PRESETS, bcn_estimate, build_manifest,
                     derive_tail_layout)

#: The cheapest profile, and therefore the entry check's yardstick:
#: if standard cannot beat BCn, no profile can.
ENTRY_PROFILE = "standard"

#: Outcome-warning thresholds on the measured capacity ratio
#: (VNM1 bytes / BCn full mip chain bytes) of the chosen artifact.
THIN_MARGIN_RATIO = 0.9
EXPANSION_RATIO = 1.0

#: Tail subheader layout, as written by
#: :func:`vnm1.driver.serialize_tail_chunk`
#: (``struct.pack("<HHBBB9x", tw, th, mip, 8, 1)`` + tw*th*8 bytes).
_TAIL_SUBHEADER = "<HHBBB9x"


def estimate_artifact_bytes(profile: str, width: int, height: int,
                            material_name: str) -> dict:
    """Predicted 4-file artifact size for `profile` at input size
    ``width`` x ``height``, derived from the format parameters.

    The chunk byte counts come from the format's own serializers
    (weights from the profile's tensor shapes, latents from the
    profile preset's grid sizes and the profile-derived channel
    counts, tail from the tail layout rule for the input size) and
    the manifest from the real manifest builder, so the total is
    the exact size the exporter would write for this material name.
    """
    preset = PROFILE_PRESETS[profile]
    fine_size = tuple(preset["fine_size"])
    coarse_size = tuple(preset["coarse_size"])
    fine_channels, coarse_channels = vt.profile_latent_channels(
        profile)
    weights = {key: np.zeros(shape, dtype=np.float32)
               for key, shape in vt.weight_shapes(profile).items()}
    weights_chunk = vt.serialize_weights_chunk(weights,
                                               profile=profile)
    fine_w, fine_h = fine_size
    coarse_w, coarse_h = coarse_size
    latents_chunk = vt.serialize_latents_chunk(
        np.zeros(fine_w * fine_h * fine_channels, dtype=np.uint8),
        fine_size,
        np.zeros(coarse_w * coarse_h * coarse_channels,
                 dtype=np.uint8),
        coarse_size,
        fine_channels=fine_channels, coarse_channels=coarse_channels)
    tail_w, tail_h, tail_mip = derive_tail_layout(width, height)
    tail_bytes = (len(vt._write_header(vt.CHUNK_TAIL, 0))
                  + struct.calcsize(_TAIL_SUBHEADER)
                  + tail_w * tail_h * 8)
    manifest = build_manifest(
        material_name, width, height, weights_chunk, latents_chunk,
        bytes(tail_bytes), tail_mip, fine_size, coarse_size,
        profile=profile, fine_channels=fine_channels,
        coarse_channels=coarse_channels)
    manifest_bytes = json.dumps(manifest, indent=2).encode("utf-8")
    breakdown = {
        "manifest.json": len(manifest_bytes),
        "weights.bin": len(weights_chunk),
        "level-0.bin": len(latents_chunk),
        "tail.bin": tail_bytes,
    }
    return {"breakdown_bytes": breakdown,
            "total_bytes": sum(breakdown.values())}


def capacity_entry_check(width: int, height: int,
                         channel_sources: dict,
                         material_name: str) -> dict:
    """Stage 1: can the cheapest profile beat BCn for this input?

    ``channel_sources`` maps each logical map name to ``"map"`` or
    ``"constant"`` (the same dict the BCn estimate and the gate
    already use).  ``would_expand`` is True when the estimated
    standard-profile artifact is at or above the BCn full mip chain
    estimate -- the conversion then cannot save capacity at any
    profile, because every other profile is larger.
    """
    bcn = bcn_estimate(width, height, channel_sources)
    estimate = estimate_artifact_bytes(ENTRY_PROFILE, width, height,
                                       material_name)
    chain = bcn["total_mip_chain_bytes"]
    return {
        "entry_profile": ENTRY_PROFILE,
        "estimated_vnm1_total_bytes": estimate["total_bytes"],
        "estimated_breakdown_bytes": estimate["breakdown_bytes"],
        "bcn_mip_chain_bytes": chain,
        "estimated_ratio_vs_bcn_mip_chain":
            estimate["total_bytes"] / max(1, chain),
        "would_expand": estimate["total_bytes"] >= chain,
    }


def capacity_warnings(ratio: float | None) -> list[dict]:
    """Stage 2: outcome warnings for a finished run's measured
    capacity ratio (chosen artifact, VNM1 / BCn mip chain).

    >= 1.0 is an expansion (the artifact is larger than the BCn
    estimate); >= 0.9 is a thin margin (the conversion buys little
    capacity).  Below 0.9 -- or with no capacity record -- the list
    is empty.
    """
    if ratio is None:
        return []
    ratio = float(ratio)
    if ratio >= EXPANSION_RATIO:
        return [{
            "kind": "capacity_expansion",
            "ratio": ratio,
            "message": (
                "the converted artifact is larger than the BCn "
                f"full mip chain estimate (ratio {ratio:.3f}); "
                "VNM1 expands this material instead of shrinking "
                "it. Keeping the conventional texture is the "
                "capacity-sane choice unless there is a "
                "non-capacity reason to convert."),
        }]
    if ratio >= THIN_MARGIN_RATIO:
        return [{
            "kind": "capacity_thin_margin",
            "ratio": ratio,
            "message": (
                "the saving over the BCn full mip chain estimate "
                f"is thin (ratio {ratio:.3f}); the conversion "
                "buys little capacity for its training and "
                "decode cost."),
        }]
    return []


__all__ = ["ENTRY_PROFILE", "EXPANSION_RATIO", "THIN_MARGIN_RATIO",
           "capacity_entry_check", "capacity_warnings",
           "estimate_artifact_bytes"]

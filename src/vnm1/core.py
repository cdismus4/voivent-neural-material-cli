"""
VNM1 PyTorch port (Stage 1, chunk A/B/C).

Canonical source (TypeScript, read-only reference):
  - reference/CpuReferenceDecoder.ts
      evaluateMlpStandardCpu / decodeNeuralMaterialRawCpu / decodeNeuralMaterialCpu
  - training/LossFunctions.ts        computeMaterialLoss / reconstructNormal
  - training/NeuralMaterialTrainer.ts (Adam training, LCG init, grid clamp)
  - exporter/VnmExporter.ts          quantizeFloatToU8Grid
  - format/binary.ts                 serialize/parse weights & latents chunks

Precision spec (manifest.json precision_spec is authoritative):
  - compute dtype float32, TF32 disabled (see disable_tf32)
  - bilinear: texel-centre convention p = uv * size - 0.5,
    local coordinate r = 2 * fract(p) - 1, wrap repeat/clamp as canonical
  - latent quantization: q = clamp(round(255 * (clamp(f,-1,1)+1)/2), 0, 255),
    dequantize f = 2q/255 - 1   (JS Math.round semantics: round-half-up,
    implemented as floor(x + 0.5); Python's round() is banker's rounding
    and is NOT used)
  - weight layout: layer order w1,b1,w2,b2,w3,b3, each matrix row-major,
    output-neuron-major.  w1:24x12, w2:24x24, w3:8x24 (Standard profile)
  - normal reconstruction: clamp xy to [-1,1]; if x^2+y^2 > 1 scale xy by
    1/sqrt(x^2+y^2); nz = sqrt(max(0, 1-x^2-y^2)); normalize.
    (LossFunctions.reconstructNormal omits the explicit disk-shrink step,
    but for x^2+y^2 > 1 it sets nz=0 and normalizes by sqrt(x^2+y^2), which
    yields exactly the same unit vector.  The two forms are equivalent.)

Loss / autograd approximations (differences from the TS analytical gradient):
  - TS computeMaterialLoss ships a hand-written exact gradient.  In
    particular, for the normal term it (a) selects gradient 0 when a
    predicted component is <= -1 or >= 1 (chain rule through the clamp),
    (b) uses a dedicated outside-disk branch, and (c) treats
    normalLoss = 2 * max(0, 1 - dot) as exactly 0 when dot >= 1.
  - This port expresses the same *forward* loss with differentiable torch
    ops and lets autograd differentiate it.  Consequences:
      * torch.clamp has gradient 0 outside [-1, 1] and an (arbitrary)
        subgradient at exactly +/-1, matching (a) everywhere except
        possibly at the exact boundary points.
      * sqrt(max(0, 1-x^2-y^2)) has an unbounded derivative as
        x^2+y^2 -> 1 from inside; autograd can therefore produce large /
        non-finite gradients in a thin shell near the unit circle where
        the TS code switches to its finite outside-disk branch.  In
        practice predicted normals are produced by a linear layer and the
        shell is measure-zero, but this IS an approximation: if training
        instability appears near |n_xy| ~= 1, the cause should be looked
        for here first.
      * max(0, 1-dot) is implemented as clamp(1-dot, min=0); autograd
        uses subgradient 0 at dot == 1, matching TS.
  - Quantize-aware training (final 20%) uses a straight-through estimator
    (STE): forward evaluates dequant(quant(grid)) while the gradient is
    passed through as if the operation were the identity.  TS has no
    training-time quantization-aware phase in code (it is a design
    statement in NTC_PLAN.md: "the final 20% of iterations adjust through
    the restoration path including u8 quantization and f16 weight
    rounding"); this port follows the *intent* for latents only.
    Weights stay float32 because manifest precision_spec fixes weight
    storage at f32 for this run; f16 rounding is deliberately NOT applied.

Recipe v2 (2026-11-18, stage-2 stabilization; see stage2/RECIPE_V2.md):
  Stage 2 diverged (loss -> NaN) in 3 of 6 materials at unrelated
  iterations (<=500 / 2,000 / 35,000) after long, smooth loss decreases.
  The only unbounded derivative in the training graph is the sqrt in
  normal reconstruction, exactly the hazard flagged above: as the
  post-shrink s = nx^2+ny^2 -> 1, d nz/ds = -1/(2 sqrt(1-s)) -> -inf,
  and at s == 1.0 exactly (the shrink branch maps onto the unit circle)
  autograd yields an inf gradient, which poisons Adam's state (m, v =
  inf -> update = inf/inf = NaN) and every parameter with it.  Recipe v2
  changes TRAINING ONLY:
    * material_loss_t now uses reconstruct_normal_train_t, which floors
      the sqrt argument at NORMAL_SQRT_EPS (forward value changes by at
      most sqrt(eps) = 1e-3 in nz, and only inside the thin shell
      1-s < eps; the canonical reconstruct_normal_t and every decode
      path are byte-identical to recipe v1, so the Stage 1 equivalence
      gate is unaffected);
    * VnmTrainer.train_steps clips the global gradient norm
      (GRAD_CLIP_MAX_NORM) and, if the pre-clip norm is non-finite,
      skips the optimizer step entirely (gradients discarded, Adam
      state untouched) and records the event on the trainer.
  Profile, iteration count, batch size, learning rates, loss weights,
  quantize-aware fraction and all evaluation code are unchanged.

Profile support (2026-11-18, stage2-quality profile test):
  MLP_PROFILES mirrors format/types.ts (small 16/16, standard 24/24,
  quality 32/32; inputDim 12 for all).  Weight SHAPES are derived from
  TrainerConfig.profile / explicit profile arguments instead of the
  former hardcoded Standard constants.  "standard" is the default in
  every signature, and standard initialization (LCG order + He scales
  sqrt(2/fan_in)) is bit-identical to the pre-profile code -- verified
  by hash comparison and the Stage 1 local equivalence regression.
  No numerical convention changes with the profile.

Format-variant probe (2026-11-18, stage2-stone-format):
  The stage2-color sweep showed stone's recipe-side trade-off curve
  passes just outside the quality gate, leaving latent REPRESENTATION
  as the suspected bottleneck: either the Fine grid resolution
  (1024 vs the 2K albedo detail) or the latent channel count (4+4
  channels shared by all 8 outputs).  This probe varies exactly one
  format axis per arm, so the latent channel counts become a
  TrainerConfig axis (fine_channels / coarse_channels, default 4/4)
  and a variant profile "quality8ch" (inputDim 20 = 8+8 latent
  channels + r2 + mip1 + const1, hidden widths kept at quality's
  32/32) is added.  "quality8ch" is NOT part of the canonical format
  (format/types.ts fixes inputDim 12); artifacts produced with it are
  format variants and are labelled as such by the driver.
  The latents chunk carries the channel counts in the first two
  subheader reserved bytes when they differ from 4/4; the canonical
  4/4 chunk is byte-identical to before (reserved bytes all zero).
  Defaults everywhere reproduce all previous runs bit-for-bit
  (hash-verified).  No numerical convention (sampling, quantization,
  normal reconstruction, loss) changes -- only tensor shapes.

8ch generalization (2026-11-18, stage2-ch8):
  A second variant profile "standard8ch" (inputDim 20, hidden widths
  kept at standard's 24/24) is added so the channel axis can be varied
  on Standard-profile bases (wood, stone-lowres) exactly as
  "quality8ch" did on the quality base.  Same status: NOT part of the
  canonical format, labelled a format variant by the driver.  Existing
  profiles are untouched, so all previous runs still reproduce
  bit-for-bit.

Format revision (2026-11-18, user decision; format-revision/
REVISION_DECISION.md):
  The two 8+8ch profiles were adopted into the format as FORMAL VNM1
  profiles (revision plan A: profile-registry extension; the format
  name stays VNM1 and the version stays 1).  Their canonical names
  are "standard8" (hidden 24/24) and "quality8" (hidden 32/32).  The
  probe-era names "standard8ch" / "quality8ch" survive only as
  PROFILE_ALIASES, so records and artifacts written during the
  probes (which use the old names and carry "format variant"
  labels) remain readable; the labels in those historical files are
  NOT rewritten.  Under the revised spec the per-grid latent channel
  count derives from the profile, (inputDim - 4) / 2.  Conformance
  of the two new profiles against an independently written spec
  reference is recorded in format-revision/conformance_ch8.json.

Storage-rule unification (2026-11-18, user decision;
format-revision/REVISION_DECISION.md):
  The channel counts derive from the profile ONLY -- latents chunks
  never carry them.  The latents subheader reserved bytes (subheader
  bytes 10..15, file bytes 26..31) must be all zero; a non-zero byte
  is a spec violation.  serialize_latents_chunk therefore writes
  zero reserved bytes for every channel shape (the 4/4 output is
  unchanged, byte-identical to before), and the parse path derives
  the counts from the profile, never reading bytes 26/27 for
  information.  The probe-era 8ch assets that carried 8/8 in those
  bytes (stage2-stone-format / stage2-ch8 outputs) were rewritten in
  place under this rule -- payload bytes untouched, originals kept
  next to them as *.pre-unify -- so every latents chunk on disk now
  satisfies the unified rule.  This also matches the current TS
  parser, which rejects non-zero reserved bytes
  (format-revision/TS_ADOPTION_NOTES.md, item 2).
"""

from __future__ import annotations

import io
import json
import math
import struct
from dataclasses import dataclass, field
from typing import Dict, Optional, Tuple

import numpy as np

try:
    import torch
    from torch import nn
except Exception:  # pragma: no cover - local VM has no torch
    torch = None
    nn = None

# --------------------------------------------------------------------------
# Constants (format/binary.ts)
# --------------------------------------------------------------------------
VNM_MAGIC = 0x314D4E56  # 'VNM1' little-endian
VNM_VERSION = 1
CHUNK_WEIGHTS = 1
CHUNK_LATENTS = 2
CHUNK_TAIL = 3

WEIGHT_KEYS = ("w1", "b1", "w2", "b2", "w3", "b3")

# MLP profiles (format/types.ts MLP_PROFILES, as extended by the
# 2026-11-18 revision).  The original three profiles have inputDim 12
# (4+4 latent channels); only the hidden widths change between them.
# "standard" is the profile proven by the Stage 1 equivalence gate
# and used by the Stage 2 runs; it remains the default everywhere so
# existing callers are bit-identical.  The stage2-quality profile
# test (2026-11-18) exercises "quality" (12->32->32->8) with larger
# latent grids; the latent grid sizes themselves are a trainer-config
# axis and are NOT part of the profile.  Only tensor SHAPES depend on
# the profile -- every numerical convention (sampling, quantization,
# normal reconstruction, loss, recipe-v2 guards) is
# profile-independent.
#
# "quality8" and "standard8" pair the quality / standard hidden
# widths (32/32, 24/24) with an inputDim of 20 for 8+8 latent channels
# (TrainerConfig fine_channels/coarse_channels = 8/8).  They entered
# this code as the format-variant probes "quality8ch"
# (stage2-stone-format) and "standard8ch" (stage2-ch8) and were
# adopted as formal VNM1 profiles by the 2026-11-18 revision
# (format-revision/REVISION_DECISION.md); format/types.ts on the TS
# side still fixes inputDim 12 until the adoption patch
# (format-revision/ts-profile8.patch) is applied there.
MLP_PROFILES = {
    "small": {"input_dim": 12, "hidden1": 16, "hidden2": 16,
              "output_dim": 8},
    "standard": {"input_dim": 12, "hidden1": 24, "hidden2": 24,
                 "output_dim": 8},
    "quality": {"input_dim": 12, "hidden1": 32, "hidden2": 32,
                "output_dim": 8},
    "quality8": {"input_dim": 20, "hidden1": 32, "hidden2": 32,
                   "output_dim": 8},
    "standard8": {"input_dim": 20, "hidden1": 24, "hidden2": 24,
                    "output_dim": 8},
}
DEFAULT_PROFILE = "standard"

# Probe-era profile names, kept ONLY as aliases so that records,
# manifests and checkpoints written during the stage2-stone-format /
# stage2-ch8 probes (which name the profiles "quality8ch" /
# "standard8ch") remain readable after the 2026-11-18 formalization.
# New code and new artifacts must use the canonical names.
PROFILE_ALIASES = {
    "quality8ch": "quality8",
    "standard8ch": "standard8",
}


def resolve_profile(profile: str) -> str:
    """Canonical profile name for `profile` (resolves probe-era
    aliases; raises KeyError for unknown names, as a direct
    MLP_PROFILES lookup would)."""
    canonical = PROFILE_ALIASES.get(profile, profile)
    if canonical not in MLP_PROFILES:
        raise KeyError(f"unknown profile: {profile!r}")
    return canonical


def profile_latent_channels(profile: str) -> Tuple[int, int]:
    """(fine_channels, coarse_channels) for `profile`, derived from
    its input_dim: (inputDim - 4) / 2 per grid (the remaining 4
    inputs are r2 + normalizedMip + const).  Under the unified
    storage rule (2026-11-18, format-revision/REVISION_DECISION.md)
    this derivation is the ONLY source of latent channel counts --
    latents chunks never carry them."""
    spec = MLP_PROFILES[resolve_profile(profile)]
    c = (spec["input_dim"] - 4) // 2
    return c, c


def weight_shapes(profile: str = DEFAULT_PROFILE) -> Dict[str, Tuple[int, ...]]:
    spec = MLP_PROFILES[resolve_profile(profile)]
    i, h1 = spec["input_dim"], spec["hidden1"]
    h2, o = spec["hidden2"], spec["output_dim"]
    return {
        "w1": (h1, i), "b1": (h1,),
        "w2": (h2, h1), "b2": (h2,),
        "w3": (o, h2), "b3": (o,),
    }


def profile_for_weight_count(total_floats: int) -> Optional[str]:
    """Infer the profile from a weights payload's float count.  The
    profiles have distinct totals (616 / 1112 / 1736 / 1992 for
    quality8 / 1304 for standard8), so a chunk's profile is
    recoverable from its length alone.  Returns canonical names."""
    for name in MLP_PROFILES:
        shapes = weight_shapes(name)
        if sum(int(np.prod(shapes[k])) for k in WEIGHT_KEYS) == total_floats:
            return name
    return None


WEIGHT_SHAPES = weight_shapes(DEFAULT_PROFILE)


def require_torch():
    if torch is None:
        raise RuntimeError(
            "torch is not available in this environment; "
            "use the numpy fallback in check_equivalence.py for local checks"
        )
    return torch


def disable_tf32() -> dict:
    """Disable TF32 everywhere (manifest precision_spec: tf32_matmul disabled).

    Fix (2026-11-17, Kaggle attempt 1): ``torch.backends.cuda.cudnn`` does not
    exist in current torch (AttributeError at kernel start, before any
    compute). The cuDNN backend lives at ``torch.backends.cudnn``; fall back
    to the cuda-namespaced attribute if a future version moves it.
    """
    t = require_torch()
    t.backends.cuda.matmul.allow_tf32 = False
    cudnn_backend = getattr(t.backends, "cudnn", None)
    if cudnn_backend is None:
        cudnn_backend = getattr(t.backends.cuda, "cudnn", None)
    if cudnn_backend is not None:
        cudnn_backend.allow_tf32 = False
    try:
        t.set_float32_matmul_precision("highest")
    except Exception:
        pass
    return {
        "matmul_allow_tf32": bool(t.backends.cuda.matmul.allow_tf32),
        "cudnn_allow_tf32": (
            bool(cudnn_backend.allow_tf32) if cudnn_backend is not None else None
        ),
        "float32_matmul_precision": (
            t.get_float32_matmul_precision()
            if hasattr(t, "get_float32_matmul_precision") else "unknown"
        ),
    }


# --------------------------------------------------------------------------
# Quantization (exporter/VnmExporter.ts quantizeFloatToU8Grid)
# --------------------------------------------------------------------------
def quantize_float_to_u8(values):
    """q = clamp(round-half-up(255 * (clamp(f,-1,1)+1)/2), 0, 255).

    Works for numpy arrays and torch tensors.
    """
    if torch is not None and isinstance(values, torch.Tensor):
        f = torch.clamp(values, -1.0, 1.0)
        q = torch.floor((f + 1.0) * 0.5 * 255.0 + 0.5)
        return torch.clamp(q, 0, 255)
    f = np.clip(np.asarray(values, dtype=np.float64), -1.0, 1.0)
    q = np.floor((f + 1.0) * 0.5 * 255.0 + 0.5)
    return np.clip(q, 0, 255).astype(np.uint8)


def dequantize_u8(q):
    """f = 2q/255 - 1.  Works for numpy arrays and torch tensors."""
    if torch is not None and isinstance(q, torch.Tensor):
        return q.to(torch.float32) / 255.0 * 2.0 - 1.0
    return (np.asarray(q, dtype=np.float64) / 255.0) * 2.0 - 1.0


# --------------------------------------------------------------------------
# LCG used by the TS trainer for initialization (NeuralMaterialTrainer.ts)
#   state = (imul(state, 1664525) + 1013904223) >>> 0 ; rand = state / 2^32
#   initUniform(arr, scale): arr[i] = (rand * 2 - 1) * scale
#   Order: w1 (sqrt(2/12)), w2 (sqrt(2/24)), w3 (sqrt(2/24)),
#          fine grid (0.1), coarse grid (0.1).  Biases are zero and do
#          NOT consume random numbers.
# --------------------------------------------------------------------------
class Lcg:
    def __init__(self, seed: int = 1337):
        self.state = seed & 0xFFFFFFFF

    def next(self) -> float:
        self.state = (self.state * 1664525 + 1013904223) & 0xFFFFFFFF
        # NOTE: JS uses Math.imul (32-bit signed multiply, low 32 bits);
        # masking the full product to 32 bits gives identical low bits.
        return self.state / 4294967296.0

    def init_uniform(self, size: int, scale: float) -> np.ndarray:
        out = np.empty(size, dtype=np.float64)
        for i in range(size):
            out[i] = (self.next() * 2.0 - 1.0) * scale
        return out.astype(np.float32)


def initial_weights_and_grids(seed: int = 1337, fine_w: int = 64, fine_h: int = 64,
                              coarse_w: int = 32, coarse_h: int = 32,
                              profile: str = DEFAULT_PROFILE,
                              fine_channels: int = 4, coarse_channels: int = 4):
    """Reproduce NeuralMaterialTrainer constructor initialization exactly.

    He scale is sqrt(2 / fan_in) per layer (fan_in = input_dim for w1,
    hidden1 for w2, hidden2 for w3); for "standard" this reproduces
    the original constants sqrt(2/12), sqrt(2/24), sqrt(2/24) and the
    exact same LCG consumption order (w1, w2, w3, fine, coarse), so
    standard initialization is bit-identical to the pre-profile code.

    fine_channels / coarse_channels (stage2-stone-format probe) scale
    the grid element counts; at the default 4/4 the counts -- and thus
    the LCG consumption -- are exactly the canonical ones.
    """
    shapes = weight_shapes(profile)
    spec = MLP_PROFILES[resolve_profile(profile)]
    h1, h2 = spec["hidden1"], spec["hidden2"]
    lcg = Lcg(seed)
    weights = {
        "w1": lcg.init_uniform(int(np.prod(shapes["w1"])),
                               math.sqrt(2.0 / spec["input_dim"])
                               ).reshape(shapes["w1"]),
        "b1": np.zeros(h1, dtype=np.float32),
        "w2": lcg.init_uniform(int(np.prod(shapes["w2"])),
                               math.sqrt(2.0 / h1)).reshape(shapes["w2"]),
        "b2": np.zeros(h2, dtype=np.float32),
        "w3": lcg.init_uniform(int(np.prod(shapes["w3"])),
                               math.sqrt(2.0 / h2)).reshape(shapes["w3"]),
        "b3": np.zeros(spec["output_dim"], dtype=np.float32),
    }
    fine = lcg.init_uniform(fine_w * fine_h * fine_channels, 0.1)
    coarse = lcg.init_uniform(coarse_w * coarse_h * coarse_channels, 0.1)
    return weights, fine, coarse, lcg.state


# --------------------------------------------------------------------------
# Bilinear sampling (CpuReferenceDecoder.ts sampleGridBilinear)
# grid: torch tensor (H, W, C); uv: (N, 2) float tensor.
# Returns (N, C).  Differentiable w.r.t. grid values (gather + weighted sum,
# backward scatter-adds into the four corner texels).  C is 4 for the
# inputDim-12 profiles and 8 for standard8/quality8 (2026-11-18
# revision).  The sampler is channel-count agnostic (shape-driven),
# so the numeric convention is identical for any C.
# --------------------------------------------------------------------------
def _wrap_coord_t(coord, mode: str):
    t = require_torch()
    if mode == "repeat":
        return coord - t.floor(coord)  # coord % 1 in [0,1), matches JS wrap
    return t.clamp(coord, 0.0, 1.0)


def sample_grid_bilinear_t(grid, uv, wrap_u: str = "repeat", wrap_v: str = "repeat"):
    t = require_torch()
    h, w = int(grid.shape[0]), int(grid.shape[1])
    u = _wrap_coord_t(uv[:, 0], wrap_u)
    v = _wrap_coord_t(uv[:, 1], wrap_v)
    px = u * w - 0.5
    py = v * h - 0.5
    x0 = t.floor(px)
    y0 = t.floor(py)
    fx = (px - x0).unsqueeze(1)
    fy = (py - y0).unsqueeze(1)

    def wrap_idx(idx, size, mode):
        if mode == "repeat":
            return idx % size
        return t.clamp(idx, 0, size - 1)

    xi0 = wrap_idx(x0.long(), w, wrap_u)
    xi1 = wrap_idx(x0.long() + 1, w, wrap_u)
    yi0 = wrap_idx(y0.long(), h, wrap_v)
    yi1 = wrap_idx(y0.long() + 1, h, wrap_v)
    flat = grid.reshape(h * w, grid.shape[-1])
    c00 = flat[yi0 * w + xi0]
    c10 = flat[yi0 * w + xi1]
    c01 = flat[yi1 * w + xi0]
    c11 = flat[yi1 * w + xi1]
    top = c00 * (1 - fx) + c10 * fx
    bottom = c01 * (1 - fx) + c11 * fx
    return top * (1 - fy) + bottom * fy


def local_coord_t(uv, fine_w: int, fine_h: int, wrap_u="repeat", wrap_v="repeat"):
    """r = 2 * fract(p) - 1 with p = uv_wrapped * size(Fine) - 0.5."""
    t = require_torch()
    u = _wrap_coord_t(uv[:, 0], wrap_u)
    v = _wrap_coord_t(uv[:, 1], wrap_v)
    px = u * fine_w - 0.5
    py = v * fine_h - 0.5
    rx = 2.0 * (px - t.floor(px)) - 1.0
    ry = 2.0 * (py - t.floor(py)) - 1.0
    return rx, ry


# --------------------------------------------------------------------------
# MLP + decoder
# --------------------------------------------------------------------------
def weights_from_numpy(weights: Dict[str, np.ndarray], device="cpu", dtype=None,
                       profile: str = DEFAULT_PROFILE):
    t = require_torch()
    dtype = dtype or t.float32
    shapes = weight_shapes(profile)
    out = {}
    for k in WEIGHT_KEYS:
        arr = np.asarray(weights[k], dtype=np.float32).reshape(shapes[k])
        out[k] = t.as_tensor(arr, dtype=dtype, device=device)
    return out


def mlp_forward_t(x, w):
    """I -> H1 (ReLU) -> H2 (ReLU) -> 8 (linear); I/H1/H2 from the
    profile of the weight tensors (12->24/24 standard, 12->32/32
    quality, 20->32/32 quality8, 20->24/24 standard8).
    x: (N,I).  Shape-driven, so one code path serves all profiles."""
    t = require_torch()
    h1 = t.relu(x @ w["w1"].T + w["b1"])
    h2 = t.relu(h1 @ w["w2"].T + w["b2"])
    return h2 @ w["w3"].T + w["b3"]


def build_input_t(fine_sample, coarse_sample, rx, ry, normalized_mip: float = 0.0):
    t = require_torch()
    n = fine_sample.shape[0]
    mip = t.full((n, 1), float(normalized_mip), dtype=fine_sample.dtype,
                 device=fine_sample.device)
    one = t.ones((n, 1), dtype=fine_sample.dtype, device=fine_sample.device)
    return t.cat([fine_sample, coarse_sample,
                  rx.unsqueeze(1), ry.unsqueeze(1), mip, one], dim=1)


def reconstruct_normal_t(xy):
    """xy: (N,2) -> (N,3) unit normals, canonical decoder form."""
    t = require_torch()
    nx = t.clamp(xy[:, 0], -1.0, 1.0)
    ny = t.clamp(xy[:, 1], -1.0, 1.0)
    sq = nx * nx + ny * ny
    scale = t.where(sq > 1.0, 1.0 / t.sqrt(t.clamp(sq, min=1e-20)),
                    t.ones_like(sq))
    nx = nx * scale
    ny = ny * scale
    nz = t.sqrt(t.clamp(1.0 - (nx * nx + ny * ny), min=0.0))
    vec = t.stack([nx, ny, nz], dim=1)
    length = t.linalg.vector_norm(vec, dim=1, keepdim=True)
    length = t.where(length == 0, t.ones_like(length), length)
    return vec / length


# --------------------------------------------------------------------------
# Recipe v2 training-side stabilization (2026-11-18; see the module
# docstring and stage2/RECIPE_V2.md for the full diagnosis).
#
# RECIPE_VERSION is stamped into checkpoints and, via the stage-2
# driver, into every records file, so a run's recipe is identifiable
# from its artifacts alone.
# --------------------------------------------------------------------------
RECIPE_VERSION = 2

# Floor for the sqrt argument in the TRAINING-side normal
# reconstruction.  Rationale: float32 has ~1.19e-7 ulp near 1.0, so
# 1-s below ~1e-6 is within a few ulp of the unit circle and carries no
# usable geometric information; flooring there bounds the derivative
# |d nz / ds| <= 1 / (2 sqrt(eps)) = 500 (finite), while the forward
# value of nz changes by at most sqrt(eps) = 1e-3 and only for samples
# inside that shell (at the measured operating point of the stage-2
# runs, max predicted s was ~0.96, so the guard is inactive there).
NORMAL_SQRT_EPS = 1e-6

# Global-norm gradient clipping threshold for the trainer.  Rationale:
# the healthy total gradient norm measured from the stage-2 highfreq
# checkpoint's Adam exp_avg_sq (iteration 30,000) is ~0.04; 1.0 is ~25x
# that, so clipping is not expected to bind in healthy training (Adam
# is in any case invariant to a uniform gradient rescaling in steady
# state), while finite spikes from any residual pathology are capped.
GRAD_CLIP_MAX_NORM = 1.0

STABILITY_FEATURES = (
    "normal_sqrt_eps_guard",
    "grad_clip_global_norm",
    "nonfinite_step_skip",
)


def reconstruct_normal_train_t(xy):
    """Training-side normal reconstruction (recipe v2).

    Identical to reconstruct_normal_t except the nz sqrt argument is
    floored at NORMAL_SQRT_EPS, removing the unbounded derivative at
    s -> 1.  Used ONLY by material_loss_t; decode/evaluation keep the
    canonical reconstruct_normal_t so the Stage 1 equivalence-proven
    numeric path is untouched.
    """
    t = require_torch()
    nx = t.clamp(xy[:, 0], -1.0, 1.0)
    ny = t.clamp(xy[:, 1], -1.0, 1.0)
    sq = nx * nx + ny * ny
    scale = t.where(sq > 1.0, 1.0 / t.sqrt(t.clamp(sq, min=1e-20)),
                    t.ones_like(sq))
    nx = nx * scale
    ny = ny * scale
    nz = t.sqrt(t.clamp(1.0 - (nx * nx + ny * ny), min=NORMAL_SQRT_EPS))
    vec = t.stack([nx, ny, nz], dim=1)
    length = t.linalg.vector_norm(vec, dim=1, keepdim=True)
    length = t.where(length == 0, t.ones_like(length), length)
    return vec / length


def decode_pbr_from_raw_t(raw):
    """CpuReferenceDecoder decodeNeuralMaterialCpu post-processing."""
    t = require_torch()
    base = t.clamp(raw[:, 0:3], 0.0, 1.0)
    normal = reconstruct_normal_t(raw[:, 3:5])
    scalars = t.clamp(raw[:, 5:8], 0.0, 1.0)
    return {
        "raw": raw,
        "baseColor": base,
        "normal": normal,
        "roughness": scalars[:, 0],
        "metalness": scalars[:, 1],
        "ao": scalars[:, 2],
    }


class VnmDecoder:
    """Decode raw/PBR values from latent grids + MLP weights (torch)."""

    def __init__(self, weights, fine_grid, coarse_grid,
                 fine_size, coarse_size, wrap_u="repeat", wrap_v="repeat"):
        """weights: dict of torch tensors; grids: (H,W,C) torch tensors
        (float, or uint8 when from_u8=True via classmethod).  C is 4
        for the inputDim-12 profiles, 8 for standard8/quality8."""
        self.weights = weights
        self.fine = fine_grid
        self.coarse = coarse_grid
        self.fine_size = tuple(fine_size)
        self.coarse_size = tuple(coarse_size)
        self.wrap_u = wrap_u
        self.wrap_v = wrap_v

    @classmethod
    def from_numpy(cls, weights_np, fine_flat, coarse_flat, fine_size,
                   coarse_size, wrap_u="repeat", wrap_v="repeat",
                   device="cpu", profile: str = DEFAULT_PROFILE,
                   fine_channels: int = 4, coarse_channels: int = 4):
        t = require_torch()
        w = weights_from_numpy(weights_np, device=device, profile=profile)
        fw, fh = fine_size
        cw, ch = coarse_size
        fine = t.as_tensor(np.asarray(fine_flat, dtype=np.float32),
                           device=device).reshape(fh, fw, fine_channels)
        coarse = t.as_tensor(np.asarray(coarse_flat, dtype=np.float32),
                             device=device).reshape(ch, cw, coarse_channels)
        return cls(w, fine, coarse, fine_size, coarse_size, wrap_u, wrap_v)

    @classmethod
    def from_u8_numpy(cls, weights_np, fine_u8, coarse_u8, fine_size,
                      coarse_size, wrap_u="repeat", wrap_v="repeat",
                      device="cpu", profile: str = DEFAULT_PROFILE,
                      fine_channels: int = 4, coarse_channels: int = 4):
        fine_f = dequantize_u8(np.asarray(fine_u8))
        coarse_f = dequantize_u8(np.asarray(coarse_u8))
        return cls.from_numpy(weights_np, fine_f.astype(np.float32),
                              coarse_f.astype(np.float32), fine_size,
                              coarse_size, wrap_u, wrap_v, device=device,
                              profile=profile,
                              fine_channels=fine_channels,
                              coarse_channels=coarse_channels)

    def decode_raw(self, uv, normalized_mip: float = 0.0):
        t = require_torch()
        if not isinstance(uv, t.Tensor):
            uv = t.as_tensor(np.asarray(uv, dtype=np.float32),
                             device=self.fine.device)
        fine_s = sample_grid_bilinear_t(self.fine, uv, self.wrap_u, self.wrap_v)
        coarse_s = sample_grid_bilinear_t(self.coarse, uv, self.wrap_u,
                                          self.wrap_v)
        rx, ry = local_coord_t(uv, self.fine_size[0], self.fine_size[1],
                               self.wrap_u, self.wrap_v)
        x = build_input_t(fine_s, coarse_s, rx, ry, normalized_mip)
        return mlp_forward_t(x, self.weights)

    def decode_pbr(self, uv, normalized_mip: float = 0.0):
        return decode_pbr_from_raw_t(self.decode_raw(uv, normalized_mip))


# --------------------------------------------------------------------------
# Loss (training/LossFunctions.ts computeMaterialLoss)
#   L = 4 * colorMSE + 2 * (1 - n.t) + roughMSE + metalMSE + aoMSE
# Per-sample values are averaged over the batch by the trainer, matching
# TS batchLoss /= batchSize.  Color MSE is the mean over the 3 channels;
# scalar "MSE" terms are single-channel squared errors.
# --------------------------------------------------------------------------
def material_loss_t(pred, target, color_weight=4.0, scalar_weight=1.0):
    # Recipe v2: the loss is training-only, so it uses the eps-guarded
    # reconstruction for both sides (symmetric, and the forward change
    # is confined to the 1-s < NORMAL_SQRT_EPS shell).  Decode paths
    # continue to use reconstruct_normal_t unchanged.
    # color_weight defaults to the canonical 4.0 (TS LossFunctions);
    # the stage2-color discrimination runs override it via
    # TrainerConfig to test the loss-balance hypothesis (arm W).
    # scalar_weight (stage2-woodmetal) multiplies the three scalar MSE
    # terms (roughness / metalness / AO) as one group; its default 1.0
    # is the canonical TS weighting, so default runs are unchanged.
    t = require_torch()
    color_mse = t.mean((pred[:, 0:3] - target[:, 0:3]) ** 2, dim=1)
    color_loss = color_weight * color_mse
    n = reconstruct_normal_train_t(pred[:, 3:5])
    tt = reconstruct_normal_train_t(target[:, 3:5])
    dot = t.sum(n * tt, dim=1)
    normal_loss = 2.0 * t.clamp(1.0 - dot, min=0.0)
    rough_loss = scalar_weight * (pred[:, 5] - target[:, 5]) ** 2
    metal_loss = scalar_weight * (pred[:, 6] - target[:, 6]) ** 2
    ao_loss = scalar_weight * (pred[:, 7] - target[:, 7]) ** 2
    total = color_loss + normal_loss + rough_loss + metal_loss + ao_loss
    return {
        "total": t.mean(total),
        "color": t.mean(color_loss),
        "normal": t.mean(normal_loss),
        "roughness": t.mean(rough_loss),
        "metalness": t.mean(metal_loss),
        "ao": t.mean(ao_loss),
    }


# --------------------------------------------------------------------------
# Trainer: joint Adam optimization of latent grids + MLP weights.
# TS defaults: lr weights 1e-3, lr grids 2e-3, Adam beta1 .9 beta2 .999
# eps 1e-8 (torch Adam uses the same defaults), grids clamped to [-1,1]
# after each step.
# --------------------------------------------------------------------------
@dataclass
class TrainerConfig:
    profile: str = DEFAULT_PROFILE
    fine_size: Tuple[int, int] = (64, 64)
    coarse_size: Tuple[int, int] = (32, 32)
    # Latent channel counts (axis added by the stage2-stone-format
    # probe).  Registered VNM1 shapes are 4/4 (inputDim-12 profiles)
    # and 8/8 (standard8/quality8, formal since the 2026-11-18
    # revision).  The MLP input is fine+coarse+4 scalars
    # (r2 + normalizedMip + const), so profile input_dim must equal
    # fine_channels + coarse_channels + 4; VnmTrainer asserts this.
    fine_channels: int = 4
    coarse_channels: int = 4
    lr_weights: float = 1e-3
    lr_grid: float = 2e-3
    color_weight: float = 4.0  # stage2-color arm knob; 4.0 = canonical
    scalar_weight: float = 1.0  # stage2-woodmetal arm knob; 1.0 = canonical
    batch_size: int = 512
    iterations: int = 300
    seed: int = 1337
    wrap_u: str = "repeat"
    wrap_v: str = "repeat"
    quantize_aware: bool = False  # final 20%: forward through u8 round-trip (STE)


class VnmTrainer:
    def __init__(self, config: TrainerConfig, device="cpu",
                 weights_np=None, fine_flat=None, coarse_flat=None):
        t = require_torch()
        self.config = config
        self.device = device
        expected_in = config.fine_channels + config.coarse_channels + 4
        actual_in = MLP_PROFILES[resolve_profile(config.profile)]["input_dim"]
        if expected_in != actual_in:
            raise ValueError(
                f"profile {config.profile} input_dim {actual_in} != "
                f"fine_channels {config.fine_channels} + coarse_channels "
                f"{config.coarse_channels} + 4 scalars = {expected_in}")
        if weights_np is None:
            weights_np, fine_flat, coarse_flat, _ = initial_weights_and_grids(
                seed=config.seed, fine_w=config.fine_size[0],
                fine_h=config.fine_size[1], coarse_w=config.coarse_size[0],
                coarse_h=config.coarse_size[1], profile=config.profile,
                fine_channels=config.fine_channels,
                coarse_channels=config.coarse_channels)
        fw, fh = config.fine_size
        cw, ch = config.coarse_size
        self.weight_params = {}
        for k in WEIGHT_KEYS:
            arr = np.asarray(weights_np[k], dtype=np.float32)
            self.weight_params[k] = t.nn.Parameter(
                t.as_tensor(arr.copy(), device=device))
        self.fine = t.nn.Parameter(
            t.as_tensor(np.asarray(fine_flat, dtype=np.float32).copy(),
                        device=device).reshape(fh, fw, config.fine_channels))
        self.coarse = t.nn.Parameter(
            t.as_tensor(np.asarray(coarse_flat, dtype=np.float32).copy(),
                        device=device).reshape(
                            ch, cw, config.coarse_channels))
        self.optimizer = t.optim.Adam(
            [
                {"params": list(self.weight_params.values()),
                 "lr": config.lr_weights},
                {"params": [self.fine, self.coarse], "lr": config.lr_grid},
            ],
            betas=(0.9, 0.999), eps=1e-8)
        self.iteration = 0
        self.generator = t.Generator(device="cpu")
        self.generator.manual_seed(config.seed)
        self.loss_history = []
        # Recipe v2 stability bookkeeping (see module docstring):
        # per-step pre-clip gradient norms are summarised, steps whose
        # gradient norm exceeded GRAD_CLIP_MAX_NORM are counted, and
        # steps skipped for a non-finite gradient norm are recorded
        # with their iteration so a run's artifacts show exactly where
        # the guards fired.
        self.grad_norm_last = None
        self.grad_norm_max = 0.0
        self.grad_clipped_steps = 0
        self.grad_events = []

    def recipe_info(self) -> dict:
        """Recipe version + active stabilization parts (for records)."""
        return {
            "recipe_version": RECIPE_VERSION,
            "stability_features": list(STABILITY_FEATURES),
            "normal_sqrt_eps": NORMAL_SQRT_EPS,
            "grad_clip_max_norm": GRAD_CLIP_MAX_NORM,
            "grad_clipped_steps": int(self.grad_clipped_steps),
            "grad_skipped_steps": len(self.grad_events),
            "grad_norm_max_seen": float(self.grad_norm_max),
            "grad_events": list(self.grad_events),
        }

    # -- forward through (optionally quantized) grids ----------------------
    def _effective_grids(self, quantize_aware_now: bool):
        if not quantize_aware_now:
            return self.fine, self.coarse
        def ste(grid):
            q = quantize_float_to_u8(grid)
            dq = dequantize_u8(q)
            return grid + (dq - grid).detach()  # STE: identity gradient
        return ste(self.fine), ste(self.coarse)

    def train(self, texture_data, width: int, height: int,
              iterations: Optional[int] = None, on_progress=None):
        """texture_data: numpy float32 flat (width*height*8,) or (H,W,8)."""
        t = require_torch()
        total_iters = iterations if iterations is not None else self.config.iterations
        tex = t.as_tensor(np.asarray(texture_data, dtype=np.float32),
                          device=self.device).reshape(height, width, 8)
        start = self.iteration
        for it in range(total_iters):
            global_it = start + it
            qa_now = (self.config.quantize_aware
                      and global_it >= int(0.8 * (start + total_iters))
                      ) if total_iters > 0 else False
            # Quantize-aware phase is defined relative to the whole run:
            # caller running in segments should pass the full plan; the
            # simple rule above treats the current call as the run.  For
            # exact run-level control use train_steps(..., qa_now=...).
            self.train_steps(tex, width, height, 1, qa_now=qa_now)
            if on_progress and (it % 20 == 0 or it == total_iters - 1):
                on_progress(global_it, self.loss_history[-1])
        return {"loss_history": list(self.loss_history),
                "final_loss": self.loss_history[-1] if self.loss_history else None}

    def train_steps(self, tex, width: int, height: int, steps: int,
                    qa_now: bool = False):
        t = require_torch()
        for _ in range(steps):
            px = t.randint(0, width, (self.config.batch_size,),
                           generator=self.generator)
            py = t.randint(0, height, (self.config.batch_size,),
                           generator=self.generator)
            uv = t.stack([(px.to(t.float32) + 0.5) / width,
                          (py.to(t.float32) + 0.5) / height],
                         dim=1).to(self.device)
            target = tex[py.to(self.device), px.to(self.device)]
            fine_g, coarse_g = self._effective_grids(qa_now)
            fine_s = sample_grid_bilinear_t(fine_g, uv, self.config.wrap_u,
                                            self.config.wrap_v)
            coarse_s = sample_grid_bilinear_t(coarse_g, uv, self.config.wrap_u,
                                              self.config.wrap_v)
            rx, ry = local_coord_t(uv, self.config.fine_size[0],
                                   self.config.fine_size[1],
                                   self.config.wrap_u, self.config.wrap_v)
            x = build_input_t(fine_s, coarse_s, rx, ry, 0.0)
            w = {k: self.weight_params[k] for k in WEIGHT_KEYS}
            pred = mlp_forward_t(x, w)
            losses = material_loss_t(pred, target,
                                     color_weight=self.config.color_weight,
                                     scalar_weight=self.config.scalar_weight)
            self.optimizer.zero_grad(set_to_none=True)
            losses["total"].backward()
            # Recipe v2: clip the global gradient norm, and never let a
            # non-finite gradient reach Adam -- a single inf gradient
            # (e.g. from the v1 sqrt singularity at s == 1) makes the
            # Adam state inf/NaN and poisons every parameter at once.
            # clip_grad_norm_ returns the PRE-clip total norm; if that
            # norm is non-finite the (already NaN-scaled) gradients are
            # discarded by skipping the step and re-zeroing, leaving
            # parameters and Adam state untouched.
            all_params = list(self.weight_params.values()) + \
                [self.fine, self.coarse]
            total_norm = t.nn.utils.clip_grad_norm_(
                all_params, GRAD_CLIP_MAX_NORM)
            norm_value = float(total_norm)
            self.grad_norm_last = norm_value
            if math.isfinite(norm_value):
                if norm_value > self.grad_norm_max:
                    self.grad_norm_max = norm_value
                if norm_value > GRAD_CLIP_MAX_NORM:
                    self.grad_clipped_steps += 1
                self.optimizer.step()
                with t.no_grad():
                    self.fine.clamp_(-1.0, 1.0)
                    self.coarse.clamp_(-1.0, 1.0)
            else:
                self.grad_events.append({
                    "iteration": int(self.iteration),
                    "total_grad_norm": norm_value,
                    "action": "optimizer_step_skipped_nonfinite_grad",
                })
                self.optimizer.zero_grad(set_to_none=True)
            self.iteration += 1
            self.loss_history.append(float(losses["total"].detach().cpu()))

    # -- export helpers ----------------------------------------------------
    def weights_numpy(self) -> Dict[str, np.ndarray]:
        return {k: self.weight_params[k].detach().cpu().numpy().astype(np.float32)
                for k in WEIGHT_KEYS}

    def grids_numpy(self):
        fine = self.fine.detach().cpu().numpy().reshape(-1).astype(np.float32)
        coarse = self.coarse.detach().cpu().numpy().reshape(-1).astype(np.float32)
        return fine, coarse

    def decoder(self) -> VnmDecoder:
        fine, coarse = self.grids_numpy()
        return VnmDecoder.from_numpy(self.weights_numpy(), fine, coarse,
                                     self.config.fine_size,
                                     self.config.coarse_size,
                                     self.config.wrap_u, self.config.wrap_v,
                                     device=self.device,
                                     profile=self.config.profile,
                                     fine_channels=self.config.fine_channels,
                                     coarse_channels=self.config.coarse_channels)

    # -- checkpoint (kill-test) -------------------------------------------
    def save_checkpoint(self, path: str):
        t = require_torch()
        ckpt = {
            "weights": {k: v.detach().cpu() for k, v in self.weight_params.items()},
            "fine": self.fine.detach().cpu(),
            "coarse": self.coarse.detach().cpu(),
            "optimizer_state": self.optimizer.state_dict(),
            "iteration": self.iteration,
            "loss_history": list(self.loss_history),
            # Recipe v2: stamp the recipe so a checkpoint's provenance
            # is identifiable from the artifact alone.
            "recipe": self.recipe_info(),
            "generator_state": self.generator.get_state(),
            "config": {
                "profile": self.config.profile,
                "fine_size": list(self.config.fine_size),
                "coarse_size": list(self.config.coarse_size),
                "fine_channels": self.config.fine_channels,
                "coarse_channels": self.config.coarse_channels,
                "lr_weights": self.config.lr_weights,
                "lr_grid": self.config.lr_grid,
                "color_weight": self.config.color_weight,
                "scalar_weight": self.config.scalar_weight,
                "batch_size": self.config.batch_size,
                "iterations": self.config.iterations,
                "seed": self.config.seed,
                "wrap_u": self.config.wrap_u,
                "wrap_v": self.config.wrap_v,
                "quantize_aware": self.config.quantize_aware,
            },
        }
        t.save(ckpt, path)

    @classmethod
    def load_checkpoint(cls, path: str, device="cpu"):
        t = require_torch()
        ckpt = t.load(path, map_location=device, weights_only=False)
        cfg = TrainerConfig(
            profile=ckpt["config"].get("profile", DEFAULT_PROFILE),
            fine_size=tuple(ckpt["config"]["fine_size"]),
            coarse_size=tuple(ckpt["config"]["coarse_size"]),
            fine_channels=ckpt["config"].get("fine_channels", 4),
            coarse_channels=ckpt["config"].get("coarse_channels", 4),
            lr_weights=ckpt["config"]["lr_weights"],
            lr_grid=ckpt["config"]["lr_grid"],
            color_weight=ckpt["config"].get("color_weight", 4.0),
            scalar_weight=ckpt["config"].get("scalar_weight", 1.0),
            batch_size=ckpt["config"]["batch_size"],
            iterations=ckpt["config"]["iterations"],
            seed=ckpt["config"]["seed"],
            wrap_u=ckpt["config"]["wrap_u"],
            wrap_v=ckpt["config"]["wrap_v"],
            quantize_aware=ckpt["config"]["quantize_aware"],
        )
        obj = cls.__new__(cls)
        obj.config = cfg
        obj.device = device
        obj.weight_params = {}
        for k in WEIGHT_KEYS:
            obj.weight_params[k] = t.nn.Parameter(
                ckpt["weights"][k].to(device).clone())
        obj.fine = t.nn.Parameter(ckpt["fine"].to(device).clone())
        obj.coarse = t.nn.Parameter(ckpt["coarse"].to(device).clone())
        obj.optimizer = t.optim.Adam(
            [
                {"params": list(obj.weight_params.values()), "lr": cfg.lr_weights},
                {"params": [obj.fine, obj.coarse], "lr": cfg.lr_grid},
            ], betas=(0.9, 0.999), eps=1e-8)
        obj.optimizer.load_state_dict(ckpt["optimizer_state"])
        obj.iteration = int(ckpt["iteration"])
        obj.loss_history = list(ckpt["loss_history"])
        # Recipe v2 bookkeeping (absent in v1 checkpoints; start fresh
        # counters on resume -- the checkpoint's own "recipe" stamp, if
        # any, stays in the file as provenance).
        obj.grad_norm_last = None
        obj.grad_norm_max = 0.0
        obj.grad_clipped_steps = 0
        obj.grad_events = []
        obj.generator = t.Generator(device="cpu")
        # Fix (2026-11-17, Kaggle attempt 2): t.load(map_location="cuda")
        # remaps every tensor in the checkpoint -- including the CPU
        # generator's ByteTensor state -- to CUDA, and Generator.set_state
        # then raises "RNG state must be a torch.ByteTensor". Move the
        # state back to CPU before restoring. (Fix prepared and recorded;
        # re-run requires parent decision -- retry cap already used.)
        gen_state = ckpt["generator_state"]
        if hasattr(gen_state, "cpu"):
            gen_state = gen_state.cpu()
        obj.generator.set_state(gen_state)
        return obj


# --------------------------------------------------------------------------
# VNM1 chunk IO (format/binary.ts)
# Header 16B: magic u32 LE, version u16, kind u16, payload u32, reserved u32=0
# --------------------------------------------------------------------------
def _write_header(kind: int, payload_len: int) -> bytes:
    return struct.pack("<IHHII", VNM_MAGIC, VNM_VERSION, kind, payload_len, 0)


def _parse_header(data: bytes):
    if len(data) < 16:
        raise ValueError("chunk too small")
    magic, version, kind, payload_len, reserved = struct.unpack(
        "<IHHII", data[:16])
    if magic != VNM_MAGIC:
        raise ValueError(f"bad magic 0x{magic:08x}")
    if version != VNM_VERSION:
        raise ValueError(f"bad version {version}")
    if reserved != 0:
        raise ValueError("reserved field must be 0")
    if len(data) != 16 + payload_len:
        raise ValueError("chunk length mismatch")
    return kind, payload_len


def serialize_weights_chunk(weights: Dict[str, np.ndarray],
                            profile: str = None) -> bytes:
    """Serialize in WEIGHT_KEYS order.  The chunk format carries no
    profile field; the profile is recoverable from the payload length
    (distinct totals per profile).  When `profile` is given, each
    array's element count is validated against that profile."""
    if profile is not None:
        shapes = weight_shapes(profile)
        for k in WEIGHT_KEYS:
            if int(np.size(weights[k])) != int(np.prod(shapes[k])):
                raise ValueError(
                    f"weights[{k}] size {np.size(weights[k])} does not "
                    f"match profile {profile} shape {shapes[k]}")
    payload = b"".join(
        np.asarray(weights[k], dtype="<f4").tobytes(order="C")
        for k in WEIGHT_KEYS)
    return _write_header(CHUNK_WEIGHTS, len(payload)) + payload


def parse_weights_chunk(data: bytes,
                        profile: str = None) -> Dict[str, np.ndarray]:
    kind, payload_len = _parse_header(data)
    if kind != CHUNK_WEIGHTS:
        raise ValueError(f"expected weights chunk, got kind {kind}")
    if payload_len % 4 != 0:
        raise ValueError("weights payload size mismatch")
    if profile is None:
        profile = profile_for_weight_count(payload_len // 4)
        if profile is None:
            raise ValueError("weights payload size mismatch")
    shapes = weight_shapes(profile)
    total = sum(int(np.prod(shapes[k])) for k in WEIGHT_KEYS)
    if payload_len != total * 4:
        raise ValueError("weights payload size mismatch")
    out = {}
    off = 16
    for k in WEIGHT_KEYS:
        n = int(np.prod(shapes[k]))
        arr = np.frombuffer(data, dtype="<f4", count=n, offset=off)
        out[k] = arr.astype(np.float32).reshape(shapes[k]).copy()
        off += n * 4
    return out


def serialize_latents_chunk(fine_u8: np.ndarray, fine_size,
                            coarse_u8: np.ndarray, coarse_size,
                            fine_channels: int = 4,
                            coarse_channels: int = 4) -> bytes:
    fw, fh = fine_size
    cw, ch = coarse_size
    fine_b = np.asarray(fine_u8, dtype=np.uint8).tobytes(order="C")
    coarse_b = np.asarray(coarse_u8, dtype=np.uint8).tobytes(order="C")
    if len(fine_b) != fw * fh * fine_channels or \
            len(coarse_b) != cw * ch * coarse_channels:
        raise ValueError("latent byte length mismatch")
    # Unified storage rule (2026-11-18, user decision;
    # format-revision/REVISION_DECISION.md): the subheader reserved
    # bytes are ALWAYS zero, whatever the channel shape -- the counts
    # derive from the profile and never ride in the chunk.  (The
    # stage2-stone-format probe era wrote the counts into reserved
    # bytes 0/1 for non-4/4 shapes; that signalling is abolished and
    # the probe-era assets were rewritten to this form.)  For 4/4
    # this output is byte-identical to every chunk written before
    # the probe.
    sub = struct.pack("<HHHHBB6x", fw, fh, cw, ch, 1, 1)
    payload = sub + fine_b + coarse_b
    return _write_header(CHUNK_LATENTS, len(payload)) + payload


def latents_chunk_channels(data: bytes, profile: str = None):
    """(fine_channels, coarse_channels) for a latents chunk.

    The counts derive from the profile (profile_latent_channels) --
    never from the chunk bytes.  The subheader reserved bytes (chunk
    bytes 26..31) are only VALIDATED: under the unified storage rule
    (2026-11-18) they must all be zero, and a non-zero byte is a
    spec violation that raises ValueError.  When `profile` is
    omitted, the count is recovered arithmetically from the header
    payload length (every registered profile has
    fine_channels == coarse_channels, so the symmetric count is
    determined by the payload size); the reserved bytes are not
    consulted for information in that path either.
    """
    reserved = data[26:32]
    if any(b != 0 for b in reserved):
        raise ValueError(
            "spec violation: non-zero reserved byte in latents "
            "subheader (channel counts derive from the profile, "
            "never from chunk bytes)")
    if profile is not None:
        return profile_latent_channels(profile)
    _, payload_len = _parse_header(data)
    fw, fh, cw, ch = struct.unpack("<HHHH", data[16:24])
    texels = fw * fh + cw * ch
    latent_bytes = payload_len - 16
    if texels <= 0 or latent_bytes < 0 or latent_bytes % texels != 0:
        raise ValueError("latents payload size mismatch")
    c = latent_bytes // texels
    return c, c


def parse_latents_chunk(data: bytes, profile: str = None):
    kind, payload_len = _parse_header(data)
    if kind != CHUNK_LATENTS:
        raise ValueError(f"expected latents chunk, got kind {kind}")
    fw, fh, cw, ch, ff, cf = struct.unpack("<HHHHBB", data[16:26])
    if ff != 1 or cf != 1:
        raise ValueError("unsupported latent quantization flag")
    fine_channels, coarse_channels = latents_chunk_channels(
        data, profile)
    fine_n = fw * fh * fine_channels
    coarse_n = cw * ch * coarse_channels
    if payload_len != 16 + fine_n + coarse_n:
        raise ValueError("latents payload size mismatch")
    fine = np.frombuffer(data, dtype=np.uint8, count=fine_n,
                         offset=32).copy()
    coarse = np.frombuffer(data, dtype=np.uint8, count=coarse_n,
                           offset=32 + fine_n).copy()
    return fine, (fw, fh), coarse, (cw, ch)

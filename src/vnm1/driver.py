"""Training / export / evaluation driver for the vnm1 CLI (step B).

This module is the product-side integration of the judgment-run
pipeline.  The numeric core is :mod:`vnm1.core`, a byte-identical copy
of the judgment port (``stage1/src/vnm1_torch.py``); the pipeline
logic here (tail chunk, BCn estimate, LR schedule, manifest, full-texel
read-back evaluation, gate verdict) is ported from the judgment driver
``stage2/src/train_stage2.py`` with the same constants and the same
loop structure, so a product run under the judgment conditions
reproduces the judgment numbers.  What was deliberately NOT ported:
the driver's Kaggle-specific fill-in constants, material auto-detection,
startup asserts and tar packing (those belong to the execution
environment, design doc section 9.1).

The teacher comes from the step A input layer (:mod:`vnm1.imaging`),
never from a second teacher implementation here.

Phase / exit-code contract (design doc section 3.3), enforced by the
CLI layer from the exception types raised here:

- training phase failures raise :class:`TrainingError` (exit 4);
- export, read-back and evaluation failures raise
  :class:`ExportVerificationError` (exit 5).
"""

from __future__ import annotations

import hashlib
import json
import math
import platform
import struct
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from . import core as vt
from .errors import ExportVerificationError, TrainingError

# ---------------------------------------------------------------------------
# Judgment-driver constants (stage2/src/train_stage2.py).  Design doc
# section 5.1: these defaults are NOT changed in productization without
# a regression measurement.
# ---------------------------------------------------------------------------
BATCH_SIZE = 8192
LR_WEIGHTS = 1e-3
LR_GRID = 2e-3
LR_DECAY_FLOOR = 0.1        # linear decay over the second half, to 10%
QA_FRACTION = 0.2           # final 20% quantize-aware (STE)
BLOCK_STEPS = 500           # progress / LR-schedule granularity
CHECKPOINT_EVERY = 10000
EVAL_ROWS_PER_BLOCK = 64
ANGLE_BIN_DEG = 0.01

#: Quality gate (design doc section 7.1; judgment manifest quality_gate).
GATE = {
    "psnr_base_color_db_min": 35.0,
    "mae_roughness_max": 0.02,
    "mae_metalness_max": 0.02,
    "mae_ao_max": 0.02,
    "normal_angle_mean_deg_max": 3.0,
    "normal_angle_p95_deg_max": 8.0,
}

#: Profile presets (design doc section 5).  Latent channels derive from
#: the profile via the core ((inputDim - 4) / 2 per grid).
PROFILE_PRESETS = {
    "standard": {"fine_size": (512, 512), "coarse_size": (256, 256)},
    "standard8": {"fine_size": (512, 512), "coarse_size": (256, 256)},
    "quality": {"fine_size": (1024, 1024), "coarse_size": (512, 512)},
    "quality8": {"fine_size": (1024, 1024), "coarse_size": (512, 512)},
}

#: Normal loss weight: fixed at 2.0 (canonical TS LossFunctions).  The
#: adjustment knobs never move it (design doc section 5.1).
NORMAL_WEIGHT = 2.0

# ---------------------------------------------------------------------------
# Multi-mip training (2026-11-21, batch5 proposal 1; opt-in via
# ``--mip-train``).  Ported from the batch5 experiment fork
# (experiments/batch5/train_mip.py mode "p1" and eval_mip.py), which
# measured the method on sand / cloth / fabric030 under the canonical
# recipe: far-mip (2-4) base-color PSNR improves by +7.7 to +21.5 dB
# over mip-0-only training, at a same-recipe mip-0 cost of -1.0 to
# -2.2 dB.  The format is unchanged; the only artifact difference is
# the optional ``mipMax`` manifest field (SPEC section 3).  The
# numeric core (:mod:`vnm1.core`) is NOT modified -- exactly as the
# experiment fork was built on top of the unmodified port, the mip
# sampling lives here in the product integration layer and reuses the
# core's sampler / loss / guards verbatim.
# ---------------------------------------------------------------------------

#: Deepest teacher mip level used by multi-mip training and by the
#: multi-mip evaluation (batch5 MIP_MAX; normalizedMip = m / MIP_MAX).
MIP_MAX = 6

#: Mip levels the multi-mip evaluation reports (batch5 ruler levels
#: 0 / 2 / 4; the full 0..6 sweep stays in the experiment records).
MIP_EVAL_LEVELS = (0, 2, 4)

#: R_hf denominator floor (batch5 DESIGN section 0): twice the
#: Laplacian variance of u8 quantization noise for a [-1, 1] channel
#: (20 * (2/255)^2 / 12 = 1.03e-4).  Without the floor, near-flat
#: channels make R_hf explode on quantization noise alone (batch4:
#: the PNG u8 control itself failed marble mip 4 at 1.69).  The pass
#: band [0.65, 1.35] is unchanged from batch4.
RHF_FLOOR = 2e-4
RHF_BAND = (0.65, 1.35)

#: Advisory thresholds for mip-0-trained artifacts (2026-11-21).
#: When a run does NOT use --mip-train, the finished artifact is
#: measured with the multi-mip ruler; if its far-field degradation
#: crosses any leg below, the report and the console summary carry an
#: advisory recommending --mip-train.  The advisory never affects
#: the gate verdict or the exit code.
#:
#: Derivation (same ruler, mip-0-trained instances only): batch4's
#: four shipped materials (marble / sand / cloth / fabric030) plus
#: batch5's three canonical-recipe p0 controls (sand / cloth /
#: fabric030) -- seven instances, all measured in the experiment
#: records:
#:   * PSNR drop mip 0 -> mip 4: the shipped four fall 4.21..18.23 dB
#:     (fabric030 4.21, marble 7.19, sand 10.93, cloth 18.23); the p0
#:     controls fall 0.57..6.37 dB (cloth 0.57, fabric030 2.29, sand
#:     6.37).  A drop-only trigger would miss the low-mip-0 p0
#:     instances, so this leg is never used alone.  4.0 dB sits just
#:     under the smallest shipped drop.
#:   * Normal mean over mips 2/4 (max): the normal-collapsing
#:     instances (sand and fabric030, both recipes) sit at
#:     10.9..35.6 deg; the other three stay <= 3.9 deg.  8.0 deg (the
#:     gate's p95 line) sits inside that gap.
#:   * Max R_hf component over mips 2/4: all seven instances are
#:     >= 2.26 (weakest: sand shipped, mip 2); the three mip-trained
#:     batch5 p1 instances stay <= 1.90.  2.0 sits between the two
#:     groups, at about 1.5x the ruler's pass-band edge (1.35).
#: These are heuristics fitted to seven materials, not physical
#: constants; they are expected to be revisited as more materials
#: are measured.
ADVISORY_PSNR_DROP_DB = 4.0
ADVISORY_NORMAL_MEAN_DEG = 8.0
ADVISORY_RHF_MAX = 2.0


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _sha256_hex(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def torch_status() -> dict:
    """Environment facts for the section 8.1 diagnostics.

    Never raises: a missing torch is reported, not thrown, so the CLI
    can print the full diagnostic block before failing.
    """
    status = {
        "python_version": platform.python_version(),
        "python_implementation": platform.python_implementation(),
        "os": f"{platform.system()} {platform.release()}",
        "torch_available": vt.torch is not None,
        "torch_version": None,
        "cuda_available": False,
        "gpu_name": None,
    }
    if vt.torch is not None:
        status["torch_version"] = vt.torch.__version__
        try:
            status["cuda_available"] = bool(vt.torch.cuda.is_available())
            if status["cuda_available"]:
                status["gpu_name"] = vt.torch.cuda.get_device_name(0)
        except Exception:
            status["cuda_available"] = False
    return status


# ---------------------------------------------------------------------------
# Tail chunk (ported from the judgment driver; format/binary.ts
# createTailChunk / parseTailChunk).  Box downsample to max side <= 16,
# normals averaged as 3D vectors and renormalized, u8 with round-half-up.
# ---------------------------------------------------------------------------
def derive_tail_layout(width: int, height: int):
    mip = 0
    w, h = width, height
    while max(w, h) > 16:
        mip += 1
        w = max(1, width // (2 ** mip))
        h = max(1, height // (2 ** mip))
    return w, h, mip


def _axis_weights(src_len: int, dst_len: int):
    """(dst_len, src_len) fractional-overlap weights (TS box filter)."""
    w = np.zeros((dst_len, src_len), dtype=np.float64)
    for d in range(dst_len):
        lo = d * src_len / dst_len
        hi = (d + 1) * src_len / dst_len
        for s in range(int(math.floor(lo)), int(math.ceil(hi))):
            w[d, s] = min(s + 1, hi) - max(s, lo)
    return w


def _round_u8(x):
    return np.clip(np.floor(x + 0.5), 0, 255).astype(np.uint8)


def serialize_tail_chunk(teacher) -> bytes:
    """teacher: (H,W,8) float32 numpy."""
    height, width = int(teacher.shape[0]), int(teacher.shape[1])
    tw, th, mip = derive_tail_layout(width, height)
    wx = _axis_weights(width, tw)
    wy = _axis_weights(height, th)
    area = (width / tw) * (height / th)
    xy = teacher[..., 3:5].astype(np.float64)
    nx = np.clip(xy[..., 0], -1.0, 1.0)
    ny = np.clip(xy[..., 1], -1.0, 1.0)
    sq = nx * nx + ny * ny
    scale = np.where(sq > 1.0, 1.0 / np.sqrt(np.maximum(sq, 1e-20)), 1.0)
    nx, ny = nx * scale, ny * scale
    nz = np.sqrt(np.maximum(0.0, 1.0 - (nx * nx + ny * ny)))
    vec = np.stack([nx, ny, nz], axis=-1)
    length = np.linalg.norm(vec, axis=-1, keepdims=True)
    vec = vec / np.where(length == 0, 1.0, length)

    def box(img):
        return (wy @ img @ wx.T) / area

    out = np.zeros((th, tw, 8), dtype=np.uint8)
    for ch in (0, 1, 2, 5, 6, 7):
        out[..., ch] = _round_u8(box(teacher[..., ch].astype(np.float64))
                                 * 255.0)
    avg_n = np.stack([box(vec[..., i]) for i in range(3)], axis=-1)
    nlen = np.linalg.norm(avg_n, axis=-1, keepdims=True)
    avg_n = avg_n / np.where(nlen == 0, 1.0, nlen)
    out[..., 3] = _round_u8((avg_n[..., 0] + 1.0) * 0.5 * 255.0)
    out[..., 4] = _round_u8((avg_n[..., 1] + 1.0) * 0.5 * 255.0)

    sub = struct.pack("<HHBBB9x", tw, th, mip, 8, 1)
    payload = sub + out.tobytes(order="C")
    return vt._write_header(vt.CHUNK_TAIL, len(payload)) + payload


def parse_tail_chunk(data: bytes) -> dict:
    kind, payload_len = vt._parse_header(data)
    if kind != vt.CHUNK_TAIL:
        raise ValueError(f"expected tail chunk, got kind {kind}")
    tw, th, mip, channels, fmt = struct.unpack("<HHBBB", data[16:23])
    if channels != 8 or fmt != 1:
        raise ValueError("unsupported tail channels/format")
    if any(b != 0 for b in data[23:32]):
        raise ValueError("non-zero reserved byte in tail subheader")
    pixels = np.frombuffer(data, dtype=np.uint8, count=tw * th * 8,
                           offset=32).copy()
    if payload_len != 16 + tw * th * 8:
        raise ValueError("tail payload size mismatch")
    return {"tail_size": (tw, th), "tail_first_mip": mip,
            "pixels": pixels.reshape(th, tw, 8)}


# ---------------------------------------------------------------------------
# BCn capacity estimate (ported; same source dimensions).
#   BaseColor BC7 (16B/4x4), Normal BC5 (16B), Rough/Metal/AO BC4 (8B).
#   Constant channels ship no map on the BCn side either (0 bytes).
# ---------------------------------------------------------------------------
BCN_BLOCK_BYTES = {"BC7": 16, "BC5": 16, "BC4": 8}
BCN_ASSIGNMENT = {"base_color": "BC7", "normal": "BC5",
                  "roughness": "BC4", "metalness": "BC4", "ao": "BC4"}


def _bcn_level_bytes(width: int, height: int, block_bytes: int,
                     mip_chain: bool) -> int:
    total = 0
    w, h = width, height
    while True:
        total += max(1, math.ceil(w / 4)) * max(1, math.ceil(h / 4)) \
            * block_bytes
        if not mip_chain or (w == 1 and h == 1):
            return total
        w, h = max(1, w // 2), max(1, h // 2)


def bcn_estimate(width: int, height: int, channel_sources: dict) -> dict:
    """channel_sources: logical map name -> "map" | "constant"."""
    per_map = {}
    mip0_total = 0
    chain_total = 0
    for name, fmt in BCN_ASSIGNMENT.items():
        if channel_sources[name] == "constant":
            per_map[name] = {"format": fmt, "source": "constant",
                             "mip0_bytes": 0, "mip_chain_bytes": 0}
            continue
        bb = BCN_BLOCK_BYTES[fmt]
        m0 = _bcn_level_bytes(width, height, bb, mip_chain=False)
        mc = _bcn_level_bytes(width, height, bb, mip_chain=True)
        per_map[name] = {"format": fmt, "source": "map",
                         "mip0_bytes": m0, "mip_chain_bytes": mc}
        mip0_total += m0
        chain_total += mc
    return {"per_map": per_map, "total_mip0_bytes": mip0_total,
            "total_mip_chain_bytes": chain_total}


# ---------------------------------------------------------------------------
# Teacher mip pyramid (SPEC section 5.3 box convention) and the
# multi-mip training sampler.  Ported from batch5 train_mip.py; the
# pyramid is built from the step A teacher array the driver already
# holds, with the same area-weighted box filter (_axis_weights) and
# the same normal handling (3D vectors averaged, then renormalized).
# ---------------------------------------------------------------------------
def _reconstruct3_np(xy):
    """(…, 2) normal xy -> (…, 3) unit vectors (numpy; mirrors the
    canonical reconstruction used by the tail chunk and by the
    batch4/5 evaluation)."""
    nx = np.clip(xy[..., 0], -1.0, 1.0)
    ny = np.clip(xy[..., 1], -1.0, 1.0)
    sq = nx * nx + ny * ny
    scale = np.where(sq > 1.0, 1.0 / np.sqrt(np.maximum(sq, 1e-20)), 1.0)
    nx, ny = nx * scale, ny * scale
    nz = np.sqrt(np.maximum(0.0, 1.0 - (nx * nx + ny * ny)))
    vec = np.stack([nx, ny, nz], axis=-1)
    length = np.linalg.norm(vec, axis=-1, keepdims=True)
    return vec / np.where(length == 0, 1.0, length)


def build_teacher_mips(teacher):
    """[mip 0 .. MIP_MAX] float32 teacher arrays (SPEC 5.3).

    Mip m has size max(1, W // 2**m) x max(1, H // 2**m); mip 0 is
    the teacher itself.  Color and scalar channels are area-weighted
    box averages in value space; normals are averaged as 3D vectors
    and renormalized.  No quantization at any level.
    """
    height, width = int(teacher.shape[0]), int(teacher.shape[1])

    def box_downsample(img, tw, th):
        wx = _axis_weights(img.shape[1], tw)
        wy = _axis_weights(img.shape[0], th)
        area = (img.shape[1] / tw) * (img.shape[0] / th)
        t = img.astype(np.float64)

        def box(a):
            return (wy @ a @ wx.T) / area

        out = np.zeros((th, tw, 8), dtype=np.float64)
        for ch in (0, 1, 2, 5, 6, 7):
            out[..., ch] = box(t[..., ch])
        vec = _reconstruct3_np(t[..., 3:5])
        avg = np.stack([box(vec[..., i]) for i in range(3)], axis=-1)
        nlen = np.linalg.norm(avg, axis=-1, keepdims=True)
        avg = avg / np.where(nlen == 0, 1.0, nlen)
        out[..., 3] = avg[..., 0]
        out[..., 4] = avg[..., 1]
        return out.astype(np.float32)

    mips = []
    for m in range(0, MIP_MAX + 1):
        wm = max(1, width // (2 ** m))
        hm = max(1, height // (2 ** m))
        mips.append(teacher if m == 0
                    else box_downsample(teacher, wm, hm))
    return mips


def draw_mip_samples(generator, mips, batch_size, device):
    """One multi-mip training batch (batch5 train_mip.py, mode p1).

    ``mips`` is a list of (H, W, 8) torch tensors on ``device``.
    Half the batch is drawn from mip 0; the other half from mips
    1..MIP_MAX uniformly over levels (log-uniform over scales).
    Samples sit at texel centres of their mip level, exactly as the
    batch4/5 evaluation samples.  Returns (uv, target, mip_float)
    on ``device``; all randomness comes from ``generator`` (the
    trainer's CPU generator, as in the core trainer).
    """
    t = vt.require_torch()
    n = batch_size
    mip_idx = t.empty(n, dtype=t.long)
    half = n // 2
    mip_idx[:half] = 0
    mip_idx[half:] = t.randint(1, MIP_MAX + 1, (n - half,),
                               generator=generator)
    uv_all = t.empty((n, 2), dtype=t.float32, device=device)
    tgt_all = t.empty((n, 8), dtype=t.float32, device=device)
    for m in range(0, MIP_MAX + 1):
        idx_cpu = (mip_idx == m).nonzero(as_tuple=True)[0]
        if idx_cpu.numel() == 0:
            continue
        tex = mips[m]
        hm, wm = int(tex.shape[0]), int(tex.shape[1])
        px = t.randint(0, wm, (idx_cpu.numel(),), generator=generator)
        py = t.randint(0, hm, (idx_cpu.numel(),), generator=generator)
        uv = t.stack([(px.to(t.float32) + 0.5) / wm,
                      (py.to(t.float32) + 0.5) / hm], dim=1)
        tgt = tex[py.to(device), px.to(device)]
        idx = idx_cpu.to(device)
        uv_all[idx] = uv.to(device)
        tgt_all[idx] = tgt
    return uv_all, tgt_all, mip_idx.to(device).to(t.float32)


def _train_steps_mip(trainer, mips_t, steps, qa_now=False):
    """Multi-mip variant of ``VnmTrainer.train_steps`` (core).

    Identical to the core step -- same loss, same recipe-v2 guards
    (global-norm clip, non-finite step skip, grid clamp), same
    bookkeeping -- except samples come from the teacher mip pyramid
    via :func:`draw_mip_samples` and the MLP input carries the
    per-sample normalizedMip = m / MIP_MAX instead of the constant
    0.0 the core step passes.  Lives in the driver (not the core) so
    the core stays the byte-identical judgment port, mirroring how
    the batch5 fork was built on the unmodified port.
    """
    t = vt.require_torch()
    cfg = trainer.config
    for _ in range(steps):
        uv, target, mip_f = draw_mip_samples(
            trainer.generator, mips_t, cfg.batch_size, trainer.device)
        fine_g, coarse_g = trainer._effective_grids(qa_now)
        fine_s = vt.sample_grid_bilinear_t(fine_g, uv, cfg.wrap_u,
                                           cfg.wrap_v)
        coarse_s = vt.sample_grid_bilinear_t(coarse_g, uv, cfg.wrap_u,
                                             cfg.wrap_v)
        rx, ry = vt.local_coord_t(uv, cfg.fine_size[0],
                                  cfg.fine_size[1], cfg.wrap_u,
                                  cfg.wrap_v)
        nmip = (mip_f / float(MIP_MAX)).unsqueeze(1)
        one = t.ones((uv.shape[0], 1), dtype=uv.dtype,
                     device=trainer.device)
        x = t.cat([fine_s, coarse_s, rx.unsqueeze(1), ry.unsqueeze(1),
                   nmip, one], dim=1)
        w = {k: trainer.weight_params[k] for k in vt.WEIGHT_KEYS}
        pred = vt.mlp_forward_t(x, w)
        losses = vt.material_loss_t(
            pred, target, color_weight=cfg.color_weight,
            scalar_weight=cfg.scalar_weight)
        trainer.optimizer.zero_grad(set_to_none=True)
        losses["total"].backward()
        all_params = list(trainer.weight_params.values()) + \
            [trainer.fine, trainer.coarse]
        total_norm = t.nn.utils.clip_grad_norm_(
            all_params, vt.GRAD_CLIP_MAX_NORM)
        norm_value = float(total_norm)
        trainer.grad_norm_last = norm_value
        if math.isfinite(norm_value):
            if norm_value > trainer.grad_norm_max:
                trainer.grad_norm_max = norm_value
            if norm_value > vt.GRAD_CLIP_MAX_NORM:
                trainer.grad_clipped_steps += 1
            trainer.optimizer.step()
            with t.no_grad():
                trainer.fine.clamp_(-1.0, 1.0)
                trainer.coarse.clamp_(-1.0, 1.0)
        else:
            trainer.grad_events.append({
                "iteration": int(trainer.iteration),
                "total_grad_norm": norm_value,
                "action": "optimizer_step_skipped_nonfinite_grad",
            })
            trainer.optimizer.zero_grad(set_to_none=True)
        trainer.iteration += 1
        trainer.loss_history.append(
            float(losses["total"].detach().cpu()))


# ---------------------------------------------------------------------------
# LR schedule: constant for the first half, then linear decay to
# LR_DECAY_FLOOR at the final iteration (judgment driver, unchanged).
# ---------------------------------------------------------------------------
def lr_factor(iteration: int, total: int) -> float:
    if total <= 1 or iteration < total / 2:
        return 1.0
    frac = (iteration - total / 2) / (total / 2)
    return 1.0 - (1.0 - LR_DECAY_FLOOR) * min(1.0, frac)


# ---------------------------------------------------------------------------
# Manifest (format/types.ts NeuralMaterialManifest; ported).
# ---------------------------------------------------------------------------
def format_variant_info(fine_channels: int, coarse_channels: int):
    """None for registered VNM1 shapes (4+4 and, since the 2026-11-18
    revision, 8+8); an explicit not-compatible label otherwise."""
    if (fine_channels, coarse_channels) in ((4, 4), (8, 8)):
        return None
    return {
        "vnm1_compatible": False,
        "kind": "latent_channels",
        "note": "FORMAT VARIANT: latent channel counts match no "
                "registered VNM1 profile (registered pairs are 4+4 "
                "and, since the 2026-11-18 revision, 8+8).  This "
                "artifact must not be called VNM1-compatible.",
        "mlp_input_dim": fine_channels + coarse_channels + 4,
        "input_dim_breakdown": {
            "fine_latent_channels": fine_channels,
            "coarse_latent_channels": coarse_channels,
            "local_coord_r": 2,
            "normalized_mip": 1,
            "constant": 1,
        },
    }


def build_manifest(material: str, width: int, height: int,
                   weights_chunk: bytes, latents_chunk: bytes,
                   tail_chunk: bytes, tail_first_mip: int,
                   fine_size, coarse_size, profile: str,
                   fine_channels: int, coarse_channels: int,
                   mip_max: int | None = None) -> dict:
    def ref(name, data):
        return {"url": name, "sha256": _sha256_hex(data),
                "byteLength": len(data)}
    manifest = {
        "format": "VNM1",
        "version": 1,
        "assetId": f"vnm1-p1-{material}",
        "sourceSize": [width, height],
        "profile": profile,
        "quantization": "u8",
        "weightStoragePrecision": "f32",
        "computePrecision": "f32",
        "channels": "base3-normal2-rough-metal-ao",
        "baseColorSpace": "linear-srgb",
        "normalConvention": "tangent-positive-y-positive-z",
        "wrapU": "repeat",
        "wrapV": "repeat",
        "weights": ref("weights.bin", weights_chunk),
        "levels": [{
            "sourceMip": 0,
            "sourceSize": [width, height],
            "fineSize": list(fine_size),
            "coarseSize": list(coarse_size),
            "chunk": ref("level-0.bin", latents_chunk),
        }],
        "tailFirstMip": tail_first_mip,
        "tail": ref("tail.bin", tail_chunk),
    }
    if mip_max is not None:
        # Optional SPEC section 3 field: the normalizedMip
        # normalization denominator.  Written only for artifacts
        # trained with --mip-train (batch5 proposal 1); mip-0-trained
        # artifacts keep the exact pre-2026-11-21 manifest shape.
        manifest["mipMax"] = mip_max
    variant = format_variant_info(fine_channels, coarse_channels)
    if variant is not None:
        manifest["vnm1Compatible"] = False
        manifest["formatVariant"] = variant
        manifest["levels"][0]["fineChannels"] = fine_channels
        manifest["levels"][0]["coarseChannels"] = coarse_channels
    return manifest


# ---------------------------------------------------------------------------
# Full-texel evaluation against the source teacher (ported).  Decodes
# from the *parsed export chunks* at texel centres, mip 0.
# ---------------------------------------------------------------------------
def _linear_to_srgb_t(x_t, torch):
    c = torch.clamp(x_t, 0.0, 1.0)
    return torch.where(c <= 0.0031308, c * 12.92,
                       1.055 * torch.pow(c, 1.0 / 2.4) - 0.055)


def evaluate(decoder, teacher, device, progress_cb=None):
    torch = vt.require_torch()
    height, width = int(teacher.shape[0]), int(teacher.shape[1])
    tex_t = torch.as_tensor(teacher, device=device)
    sse_linear = 0.0
    sse_srgb = 0.0
    mae_sum = {"roughness": 0.0, "metalness": 0.0, "ao": 0.0}
    angle_hist = np.zeros(int(180.0 / ANGLE_BIN_DEG) + 1, dtype=np.int64)
    angle_sum = 0.0
    n_texel = width * height
    t0 = time.perf_counter()
    with torch.no_grad():
        for y0 in range(0, height, EVAL_ROWS_PER_BLOCK):
            ys = np.arange(y0, min(y0 + EVAL_ROWS_PER_BLOCK, height))
            xs = np.arange(width)
            uu, vv = np.meshgrid((xs + 0.5) / width, (ys + 0.5) / height)
            uv = torch.tensor(
                np.stack([uu.ravel(), vv.ravel()], axis=1),
                dtype=torch.float32, device=device)
            pbr = decoder.decode_pbr(uv, normalized_mip=0.0)
            tgt = tex_t[torch.as_tensor(ys, device=device)]
            tgt = tgt.reshape(-1, 8)
            pred_base = pbr["baseColor"]
            diff = tgt[:, 0:3] - pred_base
            sse_linear += float(torch.sum(diff * diff))
            diff_s = (_linear_to_srgb_t(tgt[:, 0:3], torch)
                      - _linear_to_srgb_t(pred_base, torch))
            sse_srgb += float(torch.sum(diff_s * diff_s))
            for name, col in (("roughness", 5), ("metalness", 6),
                              ("ao", 7)):
                mae_sum[name] += float(torch.sum(torch.abs(
                    tgt[:, col] - pbr[name])))
            tgt_n = torch.stack([
                torch.clamp(tgt[:, 3], -1.0, 1.0),
                torch.clamp(tgt[:, 4], -1.0, 1.0)], dim=1)
            tgt_n3 = vt.reconstruct_normal_t(tgt_n)
            dot = torch.sum(pbr["normal"] * tgt_n3, dim=1)
            dot = torch.clamp(dot, -1.0, 1.0)
            ang = torch.rad2deg(torch.acos(dot)).cpu().numpy()
            angle_sum += float(np.sum(ang))
            bins = np.minimum((ang / ANGLE_BIN_DEG).astype(np.int64),
                              len(angle_hist) - 1)
            angle_hist += np.bincount(bins, minlength=len(angle_hist))
            if progress_cb:
                done_rows = y0 + len(ys)
                if done_rows % (EVAL_ROWS_PER_BLOCK * 16) == 0 \
                        or done_rows == height:
                    progress_cb(f"eval_rows={done_rows}/{height}")
    eval_seconds = time.perf_counter() - t0
    mse_linear = sse_linear / (n_texel * 3)
    mse_srgb = sse_srgb / (n_texel * 3)
    cum = np.cumsum(angle_hist)
    p95_idx = int(np.searchsorted(cum, math.ceil(0.95 * n_texel)))
    p95_deg = min(180.0, (p95_idx + 1) * ANGLE_BIN_DEG)
    return {
        "psnr_base_color_db": float(
            -10.0 * math.log10(max(mse_linear, 1e-12))),
        "psnr_base_color_db_srgb_space_reference": float(
            -10.0 * math.log10(max(mse_srgb, 1e-12))),
        "mae": {k: v / n_texel for k, v in mae_sum.items()},
        "normal_angle_mean_deg": angle_sum / n_texel,
        "normal_angle_p95_deg": float(p95_deg),
        "texel_count": n_texel,
        "eval_seconds": eval_seconds,
        "decode_texels_per_second": n_texel / max(eval_seconds, 1e-9),
    }


def gate_verdict(metrics: dict, channel_sources: dict) -> dict:
    """Gate checks (design doc section 7.1).

    ``difference`` is value - threshold throughout: for the PSNR
    (a >= check) a negative difference is the shortfall (the design
    doc's "-0.23 dB" example); for the <= checks a positive difference
    is the excess (the "+0.0019" example).
    """
    def check(value, threshold, passed, applicable):
        return {"value": value, "threshold": threshold,
                "passed": bool(passed), "gate_applicable": bool(applicable),
                "difference": value - threshold}

    checks = {
        "psnr_base_color_db": check(
            metrics["psnr_base_color_db"], GATE["psnr_base_color_db_min"],
            metrics["psnr_base_color_db"] >= GATE["psnr_base_color_db_min"],
            True),
        "normal_angle_mean_deg": check(
            metrics["normal_angle_mean_deg"],
            GATE["normal_angle_mean_deg_max"],
            metrics["normal_angle_mean_deg"]
            <= GATE["normal_angle_mean_deg_max"],
            channel_sources["normal"] == "map"),
        "normal_angle_p95_deg": check(
            metrics["normal_angle_p95_deg"],
            GATE["normal_angle_p95_deg_max"],
            metrics["normal_angle_p95_deg"]
            <= GATE["normal_angle_p95_deg_max"],
            channel_sources["normal"] == "map"),
    }
    for name in ("roughness", "metalness", "ao"):
        key = f"mae_{name}"
        checks[key] = check(
            metrics["mae"][name], GATE[key + "_max"],
            metrics["mae"][name] <= GATE[key + "_max"],
            channel_sources[name] == "map")
    applicable = [c for c in checks.values() if c["gate_applicable"]]
    return {"checks": checks,
            "passed": all(c["passed"] for c in applicable)}


# ---------------------------------------------------------------------------
# Static multi-mip evaluation (batch4/5 ruler, ported from
# experiments/batch5/eval_mip.py).  The artifact is decoded on each
# evaluated mip level's texel grid and compared against the SPEC 5.3
# teacher mip at the same resolution: base-color PSNR (linear),
# scalar MAE, normal angle mean / p95, and R_hf (high-frequency
# energy ratio, denominator floored -- see RHF_FLOOR).  The per-mip
# gate_pass uses the batch4/5 thresholds; it is recorded for
# information only and never feeds the CLI's own gate verdict
# (design doc section 7.1 stays a mip-0 gate).
#
# Decode convention follows the artifact's training: a mip-trained
# artifact is decoded with normalizedMip = m / MIP_MAX (how it was
# taught); a mip-0-trained artifact is decoded with normalizedMip = 0
# (how the current runtime drives it).
# ---------------------------------------------------------------------------
def _laplacian_np(a):
    p = np.pad(a.astype(np.float64), 1, mode="edge")
    return (p[:-2, 1:-1] + p[2:, 1:-1] + p[1:-1, :-2] + p[1:-1, 2:]
            - 4.0 * p[1:-1, 1:-1])


def _rhf(pred2d, gt2d):
    vg = float(np.var(_laplacian_np(gt2d)))
    return float(np.var(_laplacian_np(pred2d)) / max(vg, RHF_FLOOR))


def _mip_gate_pass(m: dict) -> bool:
    """The batch4/5 multi-mip gate (informational; see header)."""
    if m["psnr_db"] < GATE["psnr_base_color_db_min"]:
        return False
    if m["normal_mean_deg"] > GATE["normal_angle_mean_deg_max"]:
        return False
    if any(v > 0.02 for v in m["mae"].values()):
        return False
    for k in ("rhf_lum", "rhf_nx", "rhf_ny"):
        if not (RHF_BAND[0] <= m[k] <= RHF_BAND[1]):
            return False
    return True


def _metrics_vs_teacher_mip(pred, gt):
    """eval_mip.py metrics_vs_teacher, unchanged definitions."""
    diff = gt[..., 0:3].astype(np.float64) \
        - pred[..., 0:3].astype(np.float64)
    mse = float(np.mean(diff * diff))
    psnr = float(-10.0 * math.log10(max(mse, 1e-12)))
    mae = {}
    for name, ch in (("roughness", 5), ("metalness", 6), ("ao", 7)):
        mae[name] = float(np.mean(np.abs(
            gt[..., ch].astype(np.float64)
            - pred[..., ch].astype(np.float64))))
    pn = _reconstruct3_np(pred[..., 3:5].astype(np.float64))
    gn = _reconstruct3_np(gt[..., 3:5].astype(np.float64))
    dot = np.clip(np.sum(pn * gn, axis=-1), -1.0, 1.0)
    ang = np.degrees(np.arccos(dot))
    lum_p = (pred[..., 0] * 0.2126 + pred[..., 1] * 0.7152
             + pred[..., 2] * 0.0722)
    lum_g = (gt[..., 0] * 0.2126 + gt[..., 1] * 0.7152
             + gt[..., 2] * 0.0722)
    met = {
        "psnr_db": psnr,
        "mae": mae,
        "normal_mean_deg": float(np.mean(ang)),
        "normal_p95_deg": float(np.percentile(ang, 95)),
        "rhf_lum": _rhf(lum_p, lum_g),
        "rhf_nx": _rhf(pred[..., 3], gt[..., 3]),
        "rhf_ny": _rhf(pred[..., 4], gt[..., 4]),
    }
    met["gate_pass"] = _mip_gate_pass(met)
    return met


def _pbr_to_teacher_layout(pbr, nrow, wm):
    out = np.zeros((nrow, wm, 8), dtype=np.float32)
    out[..., 0:3] = pbr["baseColor"].cpu().numpy().reshape(nrow, wm, 3)
    nrm = pbr["normal"].cpu().numpy().reshape(nrow, wm, 3)
    out[..., 3] = nrm[..., 0]
    out[..., 4] = nrm[..., 1]
    for name, ch in (("roughness", 5), ("metalness", 6), ("ao", 7)):
        out[..., ch] = pbr[name].cpu().numpy().reshape(nrow, wm)
    return out


def evaluate_mips(decoder, teacher, device, *, mip_trained: bool,
                  progress_cb=None):
    """Multi-mip metrics for a read-back decoder at MIP_EVAL_LEVELS.

    Returns the record stored in the report (``mip_evaluation`` for
    mip-trained runs; the source of the advisory decision otherwise).
    """
    torch = vt.require_torch()
    mips = build_teacher_mips(teacher)
    per_mip = {}
    for m in MIP_EVAL_LEVELS:
        gt = mips[m]
        hm, wm = int(gt.shape[0]), int(gt.shape[1])
        nmip = (m / float(MIP_MAX)) if mip_trained else 0.0
        pred = np.zeros((hm, wm, 8), dtype=np.float32)
        with torch.no_grad():
            for y0 in range(0, hm, EVAL_ROWS_PER_BLOCK):
                ys = np.arange(y0, min(y0 + EVAL_ROWS_PER_BLOCK, hm))
                xs = np.arange(wm)
                uu, vv = np.meshgrid((xs + 0.5) / wm,
                                     (ys + 0.5) / hm)
                uv = torch.tensor(
                    np.stack([uu.ravel(), vv.ravel()], axis=1),
                    dtype=torch.float32, device=device)
                pbr = decoder.decode_pbr(uv, normalized_mip=nmip)
                pred[y0:y0 + len(ys)] = _pbr_to_teacher_layout(
                    pbr, len(ys), wm)
        per_mip[str(m)] = _metrics_vs_teacher_mip(pred, gt)
        if progress_cb:
            progress_cb(f"mip_eval mip={m} "
                        f"psnr={per_mip[str(m)]['psnr_db']:.2f}")
    return {
        "mip_max": MIP_MAX,
        "levels": list(MIP_EVAL_LEVELS),
        "decode_normalized_mip": "m/mip_max" if mip_trained else "0",
        "rhf_floor": RHF_FLOOR,
        "per_mip": per_mip,
    }


def mip_advisory_decision(per_mip: dict) -> dict | None:
    """Advisory decision for a mip-0-trained artifact (pure function).

    ``per_mip`` maps "0" / "2" / "4" to evaluate_mips metric dicts.
    Returns None when the far field holds up, else a record with the
    measured values, the thresholds and the legs that fired.  See
    the ADVISORY_* constants for the derivation; this function is
    advisory only and must never feed a gate verdict.
    """
    m0, m2, m4 = per_mip["0"], per_mip["2"], per_mip["4"]
    reasons = []
    drop = m0["psnr_db"] - m4["psnr_db"]
    if drop >= ADVISORY_PSNR_DROP_DB:
        reasons.append({
            "kind": "psnr_drop_mip0_to_mip4",
            "value": drop, "threshold": ADVISORY_PSNR_DROP_DB})
    normal_far = max(m2["normal_mean_deg"], m4["normal_mean_deg"])
    if normal_far >= ADVISORY_NORMAL_MEAN_DEG:
        reasons.append({
            "kind": "normal_mean_far_mip",
            "value": normal_far, "threshold": ADVISORY_NORMAL_MEAN_DEG})
    rhf_far = max(m[k] for m in (m2, m4)
                  for k in ("rhf_lum", "rhf_nx", "rhf_ny"))
    if rhf_far >= ADVISORY_RHF_MAX:
        reasons.append({
            "kind": "rhf_far_mip",
            "value": rhf_far, "threshold": ADVISORY_RHF_MAX})
    if not reasons:
        return None
    return {
        "recommendation": "--mip-train",
        "reasons": reasons,
        "psnr_db": {k: per_mip[k]["psnr_db"] for k in ("0", "2", "4")},
        "psnr_drop_mip0_to_mip4_db": drop,
        "normal_mean_deg": {k: per_mip[k]["normal_mean_deg"]
                            for k in ("0", "2", "4")},
        "rhf_max_far_mip": rhf_far,
        "thresholds": {
            "psnr_drop_mip0_to_mip4_db": ADVISORY_PSNR_DROP_DB,
            "normal_mean_far_mip_deg": ADVISORY_NORMAL_MEAN_DEG,
            "rhf_far_mip": ADVISORY_RHF_MAX,
        },
        "basis": "thresholds derived from 7 mip-0-trained materials "
                 "measured in batch4/batch5 (heuristic, see driver "
                 "ADVISORY_* constants)",
    }


# ---------------------------------------------------------------------------
# One manual-mode trial (design doc section 6.1)
# ---------------------------------------------------------------------------
@dataclass
class TrialOptions:
    material_name: str
    profile: str = "standard"
    iterations: int = 50000
    color_weight: float = 4.0
    scalar_weight: float = 1.0
    seed: int = 1337
    device: str = "auto"          # "auto" | "cpu" | "cuda"
    work_dir: Path = Path("work")  # checkpoints + staging live under here
    #: Opt-in multi-mip training (batch5 proposal 1, 2026-11-21).
    #: Default False: training, manifest and reports are then exactly
    #: the pre-2026-11-21 behaviour.
    mip_train: bool = False
    #: Lightweight post-training evaluation (Neon O4, 2026-11-21).
    #: Default False: the full post-processing (gate evaluation,
    #: multi-mip ruler, visual sheets) runs exactly as before.  When
    #: True, the multi-mip measurement is skipped (the mip records
    #: stay unset, so no far-field advisory is produced either) and
    #: the CLI skips the visual sheets; the gate evaluation itself
    #: is unchanged, so the artifact and the verdict are identical.
    fast_eval: bool = False


@dataclass
class TrialOutcome:
    profile: str
    fine_size: tuple
    coarse_size: tuple
    fine_channels: int
    coarse_channels: int
    metrics: dict
    checks: dict
    gate_passed: bool
    capacity: dict
    recipe: dict
    environment: dict
    started_at: str
    finished_at: str
    elapsed_seconds: float
    training_seconds: float
    final_loss: float
    artifact_dir: Path            # staging dir holding the 4 files
    artifact_files: dict          # name -> {"bytes": int, "sha256": str}
    manifest: dict
    roundtrip: dict
    # Multi-mip records (2026-11-21).  All None for a mip-0-trained
    # artifact whose far field holds up; mip_training + mip_evaluation
    # are set for --mip-train runs, mip_advisory when the advisory
    # fires on a mip-0-trained artifact.
    mip_training: dict | None = None
    mip_evaluation: dict | None = None
    mip_advisory: dict | None = None


def _resolve_device(requested: str, status: dict) -> str:
    if requested == "cpu":
        return "cpu"
    if requested == "cuda":
        if not status["cuda_available"]:
            raise TrainingError(
                "device 'cuda' was requested but CUDA is not available "
                "in this environment")
        return "cuda"
    return "cuda" if status["cuda_available"] else "cpu"


def run_trial(teacher: np.ndarray, channel_sources: dict,
              options: TrialOptions, progress_cb=None) -> TrialOutcome:
    """Train one trial, export it, read it back and evaluate it.

    The artifact is written to ``work_dir/staging/<material>.vnm/``
    (the 4-file directory format); placement into the distribution
    folder (or ``work/attempts/``) is the CLI layer's job, driven by
    the returned gate verdict.
    """
    def progress(msg: str):
        if progress_cb is not None:
            progress_cb(msg)

    if options.profile not in PROFILE_PRESETS:
        raise TrainingError(f"unknown profile preset: {options.profile!r}")
    preset = PROFILE_PRESETS[options.profile]
    fine_size = preset["fine_size"]
    coarse_size = preset["coarse_size"]
    fine_channels, coarse_channels = vt.profile_latent_channels(
        options.profile)

    status = torch_status()
    if not status["torch_available"]:
        raise TrainingError(
            "torch is not installed in this environment; the training "
            "core needs the 'train' extra (design doc section 8.1)")
    torch = vt.require_torch()
    device = _resolve_device(options.device, status)
    environment = dict(status)
    environment["device"] = device
    environment["cpu_fallback"] = device == "cpu"
    environment["tf32"] = vt.disable_tf32()

    height, width = int(teacher.shape[0]), int(teacher.shape[1])
    started_at = _now_iso()
    t_start = time.perf_counter()

    # ---- training phase (failures -> TrainingError, exit 4) -----------
    try:
        trainer, train_seconds = _train(
            teacher, width, height, options, fine_size, coarse_size,
            fine_channels, coarse_channels, device, progress)
    except TrainingError:
        raise
    except Exception as exc:
        raise TrainingError(
            f"training failed: {type(exc).__name__}: {exc}") from exc

    # ---- export + read-back + evaluation (failures -> exit 5) ---------
    try:
        outcome_tail = _export_and_evaluate(
            trainer, teacher, width, height, options, fine_size,
            coarse_size, fine_channels, coarse_channels, device,
            channel_sources, progress)
    except ExportVerificationError:
        raise
    except Exception as exc:
        raise ExportVerificationError(
            f"export / read-back / evaluation failed: "
            f"{type(exc).__name__}: {exc}") from exc

    elapsed = time.perf_counter() - t_start
    return TrialOutcome(
        profile=options.profile,
        fine_size=fine_size, coarse_size=coarse_size,
        fine_channels=fine_channels, coarse_channels=coarse_channels,
        metrics=outcome_tail["metrics"],
        checks=outcome_tail["checks"],
        gate_passed=outcome_tail["gate_passed"],
        capacity=outcome_tail["capacity"],
        recipe=trainer.recipe_info(),
        environment=environment,
        started_at=started_at, finished_at=_now_iso(),
        elapsed_seconds=elapsed, training_seconds=train_seconds,
        final_loss=float(trainer.loss_history[-1]),
        artifact_dir=outcome_tail["artifact_dir"],
        artifact_files=outcome_tail["artifact_files"],
        manifest=outcome_tail["manifest"],
        roundtrip=outcome_tail["roundtrip"],
        mip_training=outcome_tail["mip_training"],
        mip_evaluation=outcome_tail["mip_evaluation"],
        mip_advisory=outcome_tail["mip_advisory"],
    )


def _train(teacher, width, height, options, fine_size, coarse_size,
           fine_channels, coarse_channels, device, progress):
    """The judgment driver's training loop, structure unchanged:
    blocks of BLOCK_STEPS, LR factor per block, quantize-aware final
    QA_FRACTION, checkpoint every CHECKPOINT_EVERY iterations, and a
    hard stop (no retry) on a non-finite loss."""
    torch = vt.require_torch()
    total_iters = options.iterations
    tcfg = vt.TrainerConfig(
        profile=options.profile,
        fine_size=fine_size, coarse_size=coarse_size,
        fine_channels=fine_channels, coarse_channels=coarse_channels,
        lr_weights=LR_WEIGHTS, lr_grid=LR_GRID,
        color_weight=options.color_weight,
        scalar_weight=options.scalar_weight,
        batch_size=BATCH_SIZE, iterations=total_iters,
        seed=options.seed, quantize_aware=True)
    trainer = vt.VnmTrainer(tcfg, device=device)
    tex_t = torch.as_tensor(teacher, device=device)
    mips_t = None
    if options.mip_train:
        # Multi-mip training (batch5 proposal 1): build the SPEC 5.3
        # teacher pyramid once; the block loop below then draws from
        # it instead of the mip-0 teacher.  Everything else (LR
        # schedule, QA phase, checkpoints) is shared with the
        # mip-0 path.
        mips_t = [torch.as_tensor(m, device=device)
                  for m in build_teacher_mips(teacher)]
        progress(f"mip_train=on mip_max={MIP_MAX} "
                 f"pyramid_levels={len(mips_t)}")
    base_lrs = [g["lr"] for g in trainer.optimizer.param_groups]
    qa_start = int((1.0 - QA_FRACTION) * total_iters)
    ckpt_dir = options.work_dir / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = ckpt_dir / "checkpoint_latest.pt"
    train_t0 = time.perf_counter()
    done_iters = 0
    grad_events_seen = 0
    while done_iters < total_iters:
        steps = min(BLOCK_STEPS, total_iters - done_iters)
        factor = lr_factor(trainer.iteration, total_iters)
        for group, base in zip(trainer.optimizer.param_groups, base_lrs):
            group["lr"] = base * factor
        qa_now = trainer.iteration >= qa_start
        if mips_t is not None:
            _train_steps_mip(trainer, mips_t, steps, qa_now=qa_now)
        else:
            trainer.train_steps(tex_t, width, height, steps,
                                qa_now=qa_now)
        if device == "cuda":
            torch.cuda.synchronize()
        done_iters += steps
        loss_now = trainer.loss_history[-1]
        if not math.isfinite(loss_now):
            raise TrainingError(
                f"loss became non-finite at iteration "
                f"{trainer.iteration}: {loss_now} (divergence; the "
                f"trial is stopped, not retried)")
        new_events = trainer.grad_events[grad_events_seen:]
        grad_events_seen = len(trainer.grad_events)
        for ev in new_events:
            progress(f"grad_event iter={ev['iteration']} "
                     f"norm={ev['total_grad_norm']} "
                     f"action={ev['action']}")
        gnorm_note = (f"gnorm_last={trainer.grad_norm_last:.4g} "
                      f"gnorm_max={trainer.grad_norm_max:.4g} "
                      f"clipped={trainer.grad_clipped_steps} "
                      f"skipped={len(trainer.grad_events)}")
        if trainer.iteration % CHECKPOINT_EVERY == 0 \
                or done_iters == total_iters:
            trainer.save_checkpoint(str(ckpt_path))
            progress(f"train_iter={trainer.iteration} "
                     f"loss={loss_now:.4f} qa={qa_now} "
                     f"{gnorm_note} checkpoint_saved")
        else:
            progress(f"train_iter={trainer.iteration} "
                     f"loss={loss_now:.4f} qa={qa_now} {gnorm_note}")
    train_seconds = time.perf_counter() - train_t0
    progress(f"train_done seconds={train_seconds:.1f}")
    return trainer, train_seconds


def _export_and_evaluate(trainer, teacher, width, height, options,
                         fine_size, coarse_size, fine_channels,
                         coarse_channels, device, channel_sources,
                         progress):
    """Export the 4-file artifact to staging, read the files back from
    disk, parse and verify them, then evaluate from the parsed data
    only (design doc section 5.1: evaluation is of the read-back
    artifact, never of in-training tensors)."""
    material = options.material_name
    w_np = trainer.weights_numpy()
    fine_f, coarse_f = trainer.grids_numpy()
    fine_u8 = vt.quantize_float_to_u8(fine_f)
    coarse_u8 = vt.quantize_float_to_u8(coarse_f)
    weights_chunk = vt.serialize_weights_chunk(w_np,
                                               profile=options.profile)
    latents_chunk = vt.serialize_latents_chunk(
        fine_u8, fine_size, coarse_u8, coarse_size,
        fine_channels=fine_channels, coarse_channels=coarse_channels)
    tail_chunk = serialize_tail_chunk(teacher)
    tail_info = parse_tail_chunk(tail_chunk)
    manifest = build_manifest(
        material, width, height, weights_chunk, latents_chunk,
        tail_chunk, tail_info["tail_first_mip"], fine_size, coarse_size,
        profile=options.profile, fine_channels=fine_channels,
        coarse_channels=coarse_channels,
        mip_max=MIP_MAX if options.mip_train else None)
    manifest_bytes = json.dumps(manifest, indent=2).encode("utf-8")

    staging = options.work_dir / "staging" / f"{material}.vnm"
    staging.mkdir(parents=True, exist_ok=True)
    files = {"manifest.json": manifest_bytes,
             "weights.bin": weights_chunk,
             "level-0.bin": latents_chunk,
             "tail.bin": tail_chunk}
    for name, data in files.items():
        (staging / name).write_bytes(data)
    progress("export_saved")

    # Read back from disk and verify against the manifest references.
    back = {name: (staging / name).read_bytes() for name in files}
    artifact_files = {
        name: {"bytes": len(data), "sha256": _sha256_hex(data)}
        for name, data in back.items()}
    manifest_back = json.loads(back["manifest.json"].decode("utf-8"))
    sha_match = (
        manifest_back["weights"]["sha256"]
        == artifact_files["weights.bin"]["sha256"]
        and manifest_back["levels"][0]["chunk"]["sha256"]
        == artifact_files["level-0.bin"]["sha256"]
        and manifest_back["tail"]["sha256"]
        == artifact_files["tail.bin"]["sha256"])
    if not sha_match:
        raise ExportVerificationError(
            "artifact file hashes do not match the manifest references")

    w_back = vt.parse_weights_chunk(back["weights.bin"],
                                    profile=options.profile)
    f_back, fs_back, c_back, cs_back = vt.parse_latents_chunk(
        back["level-0.bin"], profile=options.profile)
    if tuple(fs_back) != tuple(fine_size) \
            or tuple(cs_back) != tuple(coarse_size):
        raise ExportVerificationError(
            f"latents chunk sizes {fs_back}/{cs_back} != exported "
            f"{fine_size}/{coarse_size}")
    # This call also enforces the unified storage rule: a non-zero
    # reserved byte in the latents subheader raises (spec violation).
    fc_back, cc_back = vt.latents_chunk_channels(
        back["level-0.bin"], profile=options.profile)
    if (fc_back, cc_back) != (fine_channels, coarse_channels):
        raise ExportVerificationError(
            f"latents chunk channels {(fc_back, cc_back)} != "
            f"{(fine_channels, coarse_channels)}")
    tail_back = parse_tail_chunk(back["tail.bin"])
    decoder = vt.VnmDecoder.from_u8_numpy(
        w_back, f_back, c_back, fs_back, cs_back, device=device,
        profile=options.profile,
        fine_channels=fc_back, coarse_channels=cc_back)
    roundtrip = {
        "weights_parse_ok": True,
        "latents_parse_ok": True,
        "tail_parse_ok": tail_back["tail_first_mip"]
        == tail_info["tail_first_mip"],
        "latents_reserved_bytes_zero": True,
        "manifest_sha_match": True,
    }
    progress("roundtrip_decoder_ready")

    metrics = evaluate(decoder, teacher, device, progress_cb=progress)
    verdict = gate_verdict(metrics, channel_sources)
    progress(f"eval_done psnr={metrics['psnr_base_color_db']:.2f} "
             f"gate_passed={verdict['passed']}")

    vnm_total = sum(f["bytes"] for f in artifact_files.values())
    bcn = bcn_estimate(width, height, channel_sources)
    capacity = {
        "vnm1_breakdown_bytes": {
            "manifest.json": artifact_files["manifest.json"]["bytes"],
            "weights.bin": artifact_files["weights.bin"]["bytes"],
            "level-0.bin": artifact_files["level-0.bin"]["bytes"],
            "tail.bin": artifact_files["tail.bin"]["bytes"],
        },
        "vnm1_total_bytes": vnm_total,
        "bcn": bcn,
        "ratio_vs_bcn_mip_chain":
            vnm_total / max(1, bcn["total_mip_chain_bytes"]),
        "ratio_vs_bcn_mip0_reference":
            vnm_total / max(1, bcn["total_mip0_bytes"]),
    }
    # ---- multi-mip records (2026-11-21) ------------------------------
    # Every trial is measured with the batch4/5 multi-mip ruler.  A
    # mip-trained artifact stores the full evaluation in its report;
    # a mip-0-trained artifact stores only the advisory record, and
    # only when the far-field degradation crosses the advisory
    # thresholds (otherwise its report is byte-identical to the
    # pre-2026-11-21 shape).  Neither record feeds the gate verdict.
    # Under --fast-eval (Neon O4) the measurement is skipped: it is
    # the single most expensive post-training step and none of its
    # outputs feed the gate.
    mip_eval = None
    if not options.fast_eval:
        mip_eval = evaluate_mips(decoder, teacher, device,
                                 mip_trained=options.mip_train,
                                 progress_cb=progress)
    else:
        progress("fast_eval=on multi-mip evaluation skipped")
    mip_training = None
    mip_evaluation = None
    mip_advisory = None
    if options.mip_train:
        mip_training = {
            "enabled": True,
            "method": "multi-mip training (batch5 proposal 1, "
                      "opt-in 2026-11-21)",
            "mip_max": MIP_MAX,
            "sampling": "50% mip 0, 50% mips 1-6 uniform over "
                        "levels (log-uniform over scales)",
            "teacher_pyramid": "SPEC 5.3 area-weighted box; normals "
                               "averaged as 3D vectors and "
                               "renormalized",
            "measured_tradeoff": "batch5, 3 materials: far-mip "
                                 "(2-4) PSNR +7.7..+21.5 dB vs "
                                 "mip-0-only training; mip-0 cost "
                                 "-1.0..-2.2 dB at the same recipe",
        }
        mip_evaluation = mip_eval
    elif mip_eval is not None:
        mip_advisory = mip_advisory_decision(mip_eval["per_mip"])
        if mip_advisory is not None:
            progress("mip_advisory=far-field degradation above the "
                     "advisory thresholds; --mip-train recommended "
                     "(advisory only, gate unaffected)")

    return {"metrics": metrics, "checks": verdict["checks"],
            "gate_passed": verdict["passed"], "capacity": capacity,
            "artifact_dir": staging, "artifact_files": artifact_files,
            "manifest": manifest_back, "roundtrip": roundtrip,
            "mip_training": mip_training,
            "mip_evaluation": mip_evaluation,
            "mip_advisory": mip_advisory}


__all__ = [
    "ADVISORY_NORMAL_MEAN_DEG", "ADVISORY_PSNR_DROP_DB",
    "ADVISORY_RHF_MAX", "BATCH_SIZE", "GATE", "MIP_EVAL_LEVELS",
    "MIP_MAX", "NORMAL_WEIGHT", "PROFILE_PRESETS", "RHF_FLOOR",
    "TrialOptions", "TrialOutcome", "bcn_estimate", "build_manifest",
    "build_teacher_mips", "draw_mip_samples", "evaluate",
    "evaluate_mips", "gate_verdict", "lr_factor",
    "mip_advisory_decision", "parse_tail_chunk", "run_trial",
    "serialize_tail_chunk", "torch_status",
]

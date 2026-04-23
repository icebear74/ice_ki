#!/usr/bin/env python3
"""
VSR++ Training Configuration Benchmark — Hardware-Specific Evaluation
======================================================================

Purpose
-------
Systematically evaluate training configurations (resolution, frame count,
model capacity) on limited GPU hardware (Tesla P100 / P4 / similar)
**without** generating or loading the real dataset.

Synthetic batches that closely mirror the real training pipeline are created
on-the-fly:
  • GT  — smooth structured random content (not pure white noise)
  • LR  — derived via 3× bicubic downscale + per-frame random pixel shifts
           (simulating inter-frame motion) + mild Gaussian noise + optional blur
  • Format: [B, T, 3, H_lr, W_lr] input  →  [B, 3, H_gt, W_gt] target

Questions answered
------------------
  Q1  What is the highest-quality configuration that fits in available VRAM?
  Q2  How does frame count (7 vs 9 vs 11) affect memory and throughput?
  Q3  What GT resolution (720×405 vs 960×540 vs 1920×1080) is feasible?
  Q4  What n_feats / n_blocks combination is the best capacity/cost trade-off?
  Q5  Does FP16 mixed precision provide a meaningful VRAM/speed benefit?

Architecture note — 9/11-frame support
---------------------------------------
The production model `VSRBidirectional_7frames_3x` (core/model_7frame.py) is
hard-coded for 7 frames.  This script defines `VSRBidirectional_Nframes_3x`,
which generalises the same bidirectional propagation architecture to any odd
frame count N ≥ 5.  The weights and structural components are identical; only
the loop indices change (center = N // 2, backward = [c+1 … N-1], forward =
[c-1 … 0]).  This variant is used only for benchmarking — no production code
is modified.

Usage
-----
  # Focused scan (recommended first run — 16:9 sizes, FP16 only, ~10-20 min)
  python benchmark_training_configs.py --quick

  # Full sweep (all combinations, can take 60-120 min on a P100)
  python benchmark_training_configs.py --full

  # Custom resolution focus (e.g. only 960×540)
  python benchmark_training_configs.py --gt-sizes 960x540

  # Save results to a specific directory
  python benchmark_training_configs.py --output-dir /tmp/benchmark_results

  # Dry run — print planned config list, no GPU execution
  python benchmark_training_configs.py --dry-run

Output
------
  benchmark_results.csv   — machine-readable per-config results
  benchmark_results.json  — same data as JSON
  Terminal                — live progress + summary table

Interpreting results
--------------------
  • VRAM_GB  — peak allocation during forward + backward + optimizer step.
               P100 = 16 GB total; keep below ~14.5 GB for training stability.
  • s/iter   — wall-clock time for one full optimizer step (all accum steps).
               Multiply by planned MAX_STEPS to estimate total training time.
  • OOM      — configuration exceeds available VRAM; choose smaller batch/feat.
  • SKIPPED  — pre-known OOM combination, not executed.

Limitations
-----------
  • Synthetic data is smooth random content.  Real training with real video
    will show slightly higher peak VRAM (~100–300 MB) due to dataloader
    buffers, TensorBoard, and web monitoring overhead.
  • Perceptual loss (VGG16) is included to match real training memory usage.
  • Timing accuracy is ±0.5–1 s per iteration.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import sys
import time
from datetime import datetime
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

# ---------------------------------------------------------------------------
# Allow imports from the vsr_plusplus_NEU package (loss, model helpers)
# ---------------------------------------------------------------------------
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _THIS_DIR)

# ---------------------------------------------------------------------------
# Import shared architecture components from production model code
# ---------------------------------------------------------------------------
try:
    from core.model_7frame import (
        AttentionGate,
        GatedFusionBlock,
        ResidualBlock,
        TemporalAlignBlock,
        VSRBidirectional_7frames_3x,
    )
    from core.loss import HybridLoss
    _IMPORTS_OK = True
except ImportError as _import_err:
    _IMPORTS_OK = False
    _IMPORT_ERROR = str(_import_err)

# ===========================================================================
# BENCHMARK PARAMETER SPACE
# ===========================================================================

# ── GT sizes stored as (H, W) — PyTorch convention (height first) ────────────
# Keys follow display convention "WxH" (width × height, e.g. "720x405").
# All 16:9 sizes chosen so that both H and W are divisible by 3 (clean 3× LR).
# Square crops kept for continuity with existing production training.
ALL_GT_SIZES: Dict[str, Tuple[int, int]] = {
    # Existing production sizes  →  key = "WxH",  value = (H, W)
    "720x405":   ( 405,  720),   # 16:9, LR = 135×240   — production standard
    "540x540":   ( 540,  540),   # square crop          — production standard
    "720x720":   ( 720,  720),   # square crop          — production standard
    # New 16:9 sizes for evaluation
    "960x540":   ( 540,  960),   # 16:9, LR = 180×320   — ½ FullHD
    "1920x1080": (1080, 1920),   # 16:9, LR = 360×640   — FullHD (large; likely needs FP16+small model)
}

# ── Quick mode: smaller search space for a first orientation ─────────────────
QUICK_GT_SIZES    = ["720x405", "960x540"]
QUICK_FRAMES      = [7, 9]
QUICK_N_FEATS     = [60, 72]
QUICK_N_BLOCKS    = [24, 26]
QUICK_BATCH_SIZES = [1, 2]
QUICK_PRECISIONS  = ["float16"]

# ── Full mode: exhaustive sweep ───────────────────────────────────────────────
FULL_GT_SIZES     = list(ALL_GT_SIZES.keys())
FULL_FRAMES       = [7, 9, 11]
FULL_N_FEATS      = [48, 60, 72, 80]
FULL_N_BLOCKS     = [20, 24, 26]
FULL_BATCH_SIZES  = [1, 2]
FULL_PRECISIONS   = ["float16", "float32"]

# ── Gradient accumulation map: (batch_size, gt_key) → accum_steps ────────────
# Targets effective batch = 8 to match production training.
# None = known-OOM, skip.
ACCUM_MAP: Dict[Tuple[int, str], Optional[int]] = {
    # 720×405
    (1, "720x405"):   8,
    (2, "720x405"):   4,
    # 540×540
    (1, "540x540"):   8,
    (2, "540x540"):   4,
    # 720×720
    (1, "720x720"):   8,
    (2, "720x720"):   None,   # known OOM at BS=2 on P100 for large models
    # 960×540  (larger patch — be conservative)
    (1, "960x540"):   8,
    (2, "960x540"):   4,
    # 1920×1080  (very large — BS=2 almost certainly OOM)
    (1, "1920x1080"): 4,
    (2, "1920x1080"): None,
}

# ── Benchmark iteration counts ────────────────────────────────────────────────
WARMUP_ITERS = 1   # discarded (GPU warm-up)
TIMING_ITERS = 5   # averaged for reported s/iter

# ── P100 VRAM for utilisation display ────────────────────────────────────────
P100_VRAM_GB = 16.0


# ===========================================================================
# GENERALISED N-FRAME BIDIRECTIONAL VSR MODEL
# ===========================================================================

class VSRBidirectional_Nframes_3x(nn.Module):
    """
    N-Frame Bidirectional VSR — generalisation of VSRBidirectional_7frames_3x.

    Supports any odd frame count N ≥ 5 (production: N=7; benchmark: N=9, 11).
    Architecture is identical; only the propagation loop indices change:
        center     = N // 2
        backward   = [center+1, …, N-1]
        forward    = [center-1, …, 0]

    Input:   [B, N, 3, H,   W  ]
    Output:  [B, 3, H*3, W*3]  (3× upscale of center frame)

    Note: This class is **benchmark-only**.  The production codebase uses
    `VSRBidirectional_7frames_3x` from core/model_7frame.py directly.
    When N=7 this model is functionally equivalent to the production model.
    """

    def __init__(self, n_frames: int = 7, n_feats: int = 72, n_blocks: int = 26,
                 use_checkpointing: bool = False):
        if n_frames < 5 or n_frames % 2 == 0:
            raise ValueError(
                f"n_frames must be an odd number ≥ 5, got {n_frames}. "
                "Supported: 5, 7, 9, 11, …"
            )
        super().__init__()
        self.n_frames = n_frames
        self.n_feats  = n_feats
        self.n_blocks = n_blocks

        half_blocks = max(1, n_blocks // 2)

        # 1. Shared feature extraction (one Conv2d applied to all frames)
        self.feat_extract = nn.Conv2d(3, n_feats, 3, 1, 1)

        # 2. Temporal alignment blocks (shared across all propagation steps)
        self.backward_align = TemporalAlignBlock(n_feats)
        self.forward_align  = TemporalAlignBlock(n_feats)

        # 3. Gated fusion layers (shared across all propagation steps)
        self.backward_fuse = GatedFusionBlock(n_feats * 2, n_feats)
        self.forward_fuse  = GatedFusionBlock(n_feats * 2, n_feats)

        # 4. Propagation trunks with attention-gated skip connections
        self.backward_trunk = nn.ModuleList([
            ResidualBlock(n_feats, use_checkpointing=use_checkpointing, use_attention=True)
            for _ in range(half_blocks)
        ])
        self.forward_trunk = nn.ModuleList([
            ResidualBlock(n_feats, use_checkpointing=use_checkpointing, use_attention=True)
            for _ in range(half_blocks)
        ])

        # 5. Final gated fusion of bidirectional results
        self.fusion = GatedFusionBlock(n_feats * 2, n_feats)

        # 6. 3× PixelShuffle upsampler
        self.upsample = nn.Sequential(
            nn.Conv2d(n_feats, n_feats * 9, 3, 1, 1),
            nn.PixelShuffle(3),
            nn.Conv2d(n_feats, 3, 3, 1, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [B, N, 3, H, W]

        Returns:
            [B, 3, H*3, W*3]
        """
        B, T, C, H, W = x.size()
        center = T // 2   # e.g. 3 for N=7, 4 for N=9, 5 for N=11

        # Extract features from all N frames in a single batched pass
        feats = self.feat_extract(x.view(-1, C, H, W))
        feats = feats.view(B, T, self.n_feats, H, W)

        # Initialise both propagation streams from the center frame
        center_feat = feats[:, center].clone()

        # ── Backward propagation: center → frames AFTER center ─────────────
        back_prop = center_feat
        for i in range(center + 1, T):
            aligned   = self.backward_align(back_prop, feats[:, i])
            fused     = self.backward_fuse(torch.cat([back_prop, aligned], dim=1))
            for block in self.backward_trunk:
                fused = block(fused)
            back_prop = fused

        # ── Forward propagation: center → frames BEFORE center ─────────────
        forw_prop = center_feat
        for i in range(center - 1, -1, -1):
            aligned   = self.forward_align(forw_prop, feats[:, i])
            fused     = self.forward_fuse(torch.cat([forw_prop, aligned], dim=1))
            for block in self.forward_trunk:
                fused = block(fused)
            forw_prop = fused

        # ── Final bidirectional fusion ──────────────────────────────────────
        fused     = self.fusion(torch.cat([back_prop, forw_prop], dim=1))

        # ── Upsample + bilinear residual from center frame ─────────────────
        base      = F.interpolate(x[:, center], scale_factor=3, mode='bilinear', align_corners=False)
        upsampled = self.upsample(fused)
        return upsampled + base


# ===========================================================================
# SYNTHETIC DATA GENERATION
# ===========================================================================

def _smooth_random(batch: int, c: int, h: int, w: int, device: torch.device,
                   dtype: torch.dtype) -> torch.Tensor:
    """
    Return a smoothed random image [B, C, H, W] in [0, 1].

    Smoothing is achieved by generating a small random image and upsampling it,
    which produces low-frequency structure that more closely resembles real video
    frames than pure Gaussian noise.
    """
    factor = 8
    small = torch.rand(batch, c, max(1, h // factor), max(1, w // factor),
                       device=device, dtype=dtype)
    return F.interpolate(small, size=(h, w), mode='bilinear', align_corners=False).clamp(0.0, 1.0)


def generate_synthetic_batch(
    n_frames:   int,
    batch_size: int,
    gt_size:    Tuple[int, int],
    precision:  str,
    device:     torch.device,
    noise_sigma: float = 1.5,
    blur_sigma:  float = 0.3,
    max_shift:   int   = 2,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Generate a synthetic (LR input, GT target) batch that mirrors the real
    training pipeline.

    Strategy
    --------
    1. Create a smooth random "ground-truth" image at gt_size.
    2. For each of the N frames:
       a. Apply a small random spatial shift (±max_shift pixels) to simulate
          inter-frame motion (real video = consecutive frames with slight motion).
       b. Downsample the shifted frame by 3× (bicubic) to obtain the LR frame.
       c. Add mild Gaussian noise (σ = noise_sigma / 255) to the LR frame.
       d. Optionally apply a 3×3 Gaussian-like blur to the LR frame.
    3. Stack N LR frames → [B, T, 3, H_lr, W_lr].
    4. GT = original smooth frame (no shift, no degradation) → [B, 3, H_gt, W_gt].

    This is significantly more realistic than pure torch.randn() because:
      • LR is correlated with GT via the 3× downscale.
      • Adjacent frames share content with slight motion, as in real video.
      • Mild degradation represents the generator's quality settings.

    Args:
        n_frames:   Number of LR frames in the stack.
        batch_size: Batch size.
        gt_size:    (H_gt, W_gt) — GT resolution.
        precision:  'float16' or 'float32'.
        device:     Target CUDA device.
        noise_sigma: Standard deviation of per-pixel noise (in [0, 255] range).
        blur_sigma:  Unused (reserved for future depth-of-field simulation).
        max_shift:   Maximum ±pixel shift between consecutive frames.

    Returns:
        lr_input:  [B, T, 3, H_lr, W_lr]
        gt_target: [B, 3, H_gt, W_gt]
    """
    dtype    = torch.float16 if precision == 'float16' else torch.float32
    gt_h, gt_w = gt_size
    lr_h, lr_w = gt_h // 3, gt_w // 3

    # ── 1. Smooth GT content ─────────────────────────────────────────────────
    gt = _smooth_random(batch_size, 3, gt_h, gt_w, device, dtype)

    # ── 2. Build per-frame LR stack ──────────────────────────────────────────
    frames: List[torch.Tensor] = []
    for t in range(n_frames):
        # 2a. Random sub-pixel / pixel shift to simulate frame-to-frame motion
        shift_x = (t - n_frames // 2) * max_shift // max(1, n_frames // 2)
        shift_y = (t - n_frames // 2) * max_shift // max(1, n_frames // 2)
        # Small additive per-frame jitter (different per batch item)
        jitter_x = torch.randint(-max_shift, max_shift + 1, (1,)).item()
        jitter_y = torch.randint(-max_shift, max_shift + 1, (1,)).item()
        dx = int(shift_x) + int(jitter_x)
        dy = int(shift_y) + int(jitter_y)

        if dx != 0 or dy != 0:
            # Shift via affine grid (handles all batch items identically here)
            theta = torch.tensor(
                [[1.0, 0.0, 2.0 * dx / gt_w],
                 [0.0, 1.0, 2.0 * dy / gt_h]],
                dtype=torch.float32, device=device,
            ).unsqueeze(0).expand(batch_size, -1, -1)
            grid     = F.affine_grid(theta, gt.float().size(), align_corners=False).to(dtype)
            shifted  = F.grid_sample(gt.float(), grid.float(), mode='bilinear',
                                     padding_mode='border', align_corners=False).to(dtype)
        else:
            shifted = gt.clone()

        # 2b. Downsample to LR by 3×
        lr_frame = F.interpolate(shifted.float(), size=(lr_h, lr_w),
                                 mode='bicubic', align_corners=False,
                                 antialias=True).clamp(0.0, 1.0).to(dtype)

        # 2c. Mild noise (matches generator's lr_noise_sigma in [0.5, 2.0])
        noise = torch.randn_like(lr_frame) * (noise_sigma / 255.0)
        lr_frame = (lr_frame + noise).clamp(0.0, 1.0)

        frames.append(lr_frame)

    # Stack → [B, T, 3, H_lr, W_lr]
    lr_input = torch.stack(frames, dim=1)

    return lr_input, gt


# ===========================================================================
# MODEL FACTORY
# ===========================================================================

def create_model(n_frames: int, n_feats: int, n_blocks: int,
                 precision: str) -> nn.Module:
    """
    Instantiate the appropriate VSR model on CUDA.

    For N=7 the production `VSRBidirectional_7frames_3x` is used directly.
    For N=9 and N=11 the generalised `VSRBidirectional_Nframes_3x` is used.
    """
    if n_frames == 7:
        # Use the production model for N=7 to validate identical behaviour
        model = VSRBidirectional_7frames_3x(n_feats=n_feats, n_blocks=n_blocks)
    else:
        model = VSRBidirectional_Nframes_3x(n_frames=n_frames,
                                             n_feats=n_feats,
                                             n_blocks=n_blocks)
    model = model.cuda()
    if precision == 'float16':
        model = model.half()
    return model


# ===========================================================================
# SINGLE-CONFIG TEST
# ===========================================================================

def test_config(
    n_frames:   int,
    batch_size: int,
    n_feats:    int,
    n_blocks:   int,
    gt_key:     str,
    gt_size:    Tuple[int, int],
    precision:  str,
    accum:      int,
    timing_iters: int,
    warmup_iters: int,
) -> Dict:
    """
    Run a single training-like benchmark configuration.

    Performs:
      1. Model creation + Adam optimizer
      2. HybridLoss instantiation (includes VGG16 perceptual network)
      3. warm-up iterations (discarded)
      4. timed iterations: forward → loss → backward → optimizer.step()
      5. Peak VRAM measurement

    Returns a result dict with keys:
        success, vram_gb, time_per_iter, frames, batch_size, accum,
        n_feats, n_blocks, gt_key, gt_size, precision, error
    """
    result_base = dict(
        n_frames=n_frames, batch_size=batch_size, accum=accum,
        n_feats=n_feats, n_blocks=n_blocks,
        gt_key=gt_key, gt_size=gt_size, precision=precision,
    )

    try:
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()

        # ── Model + optimiser ──────────────────────────────────────────────
        model     = create_model(n_frames, n_feats, n_blocks, precision)
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
        model.train()

        # ── HybridLoss — matches real training (VGG16 perceptual included) ─
        criterion = HybridLoss(
            l1_weight=0.60,
            ms_weight=0.20,
            grad_weight=0.20,
            perceptual_weight=0.10,   # VGG16 active to get realistic VRAM footprint
        ).cuda()

        # ── Warm-up ────────────────────────────────────────────────────────
        for _ in range(warmup_iters):
            optimizer.zero_grad()
            for _ in range(accum):
                lr_in, gt_tgt = generate_synthetic_batch(
                    n_frames, batch_size, gt_size, precision, device=torch.device('cuda'))
                out     = model(lr_in)
                loss_d  = criterion(out, gt_tgt)
                (loss_d['total'] / accum).backward()
            optimizer.step()

        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()

        # ── Timed iterations ───────────────────────────────────────────────
        timings = []
        for _ in range(timing_iters):
            optimizer.zero_grad()
            t0 = time.perf_counter()

            for _ in range(accum):
                lr_in, gt_tgt = generate_synthetic_batch(
                    n_frames, batch_size, gt_size, precision, device=torch.device('cuda'))
                out    = model(lr_in)
                loss_d = criterion(out, gt_tgt)
                (loss_d['total'] / accum).backward()

            optimizer.step()
            torch.cuda.synchronize()
            timings.append(time.perf_counter() - t0)

        vram_gb = torch.cuda.max_memory_allocated() / (1024 ** 3)
        avg_t   = sum(timings) / len(timings)

        # ── Cleanup ────────────────────────────────────────────────────────
        del model, optimizer, criterion
        torch.cuda.empty_cache()

        return dict(**result_base, success=True, vram_gb=vram_gb,
                    time_per_iter=avg_t, error=None)

    except RuntimeError as exc:
        torch.cuda.empty_cache()
        oom = 'out of memory' in str(exc).lower()
        return dict(**result_base, success=False,
                    vram_gb=0.0, time_per_iter=0.0,
                    error='OOM' if oom else str(exc))


# ===========================================================================
# CLI + MAIN
# ===========================================================================

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description='VSR++ Training Configuration Benchmark',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__.split('Usage')[0].strip(),
    )
    mode = p.add_mutually_exclusive_group()
    mode.add_argument('--quick', action='store_true',
                      help='Quick mode: focused subset (~10-20 min)')
    mode.add_argument('--full',  action='store_true',
                      help='Full mode: exhaustive sweep (~60-120 min)')

    p.add_argument('--gt-sizes', nargs='+', metavar='HxW',
                   help='Override GT sizes to test (e.g. 960x540 1920x1080)')
    p.add_argument('--frames',   nargs='+', type=int, metavar='N',
                   help='Frame counts to test (e.g. 7 9 11)')
    p.add_argument('--n-feats',  nargs='+', type=int, metavar='F',
                   help='n_feats values (e.g. 60 72)')
    p.add_argument('--n-blocks', nargs='+', type=int, metavar='B',
                   help='n_blocks values (e.g. 24 26)')
    p.add_argument('--batch-sizes', nargs='+', type=int, metavar='BS',
                   help='Batch sizes (e.g. 1 2)')
    p.add_argument('--precisions', nargs='+', choices=['float16', 'float32'],
                   help='Precision modes')
    p.add_argument('--output-dir', default=_THIS_DIR, metavar='DIR',
                   help='Directory for output CSV / JSON (default: script dir)')
    p.add_argument('--timing-iters', type=int, default=TIMING_ITERS,
                   help=f'Timed iterations per config (default: {TIMING_ITERS})')
    p.add_argument('--dry-run', action='store_true',
                   help='Print planned configs without running GPU tests')
    p.add_argument('--no-csv',  action='store_true', help='Skip CSV output')
    p.add_argument('--no-json', action='store_true', help='Skip JSON output')
    return p.parse_args()


def _build_config_list(args: argparse.Namespace) -> List[Dict]:
    """Build the list of configurations to benchmark from CLI args."""
    # ── Choose base parameter sets ─────────────────────────────────────────
    if args.full:
        gt_keys     = FULL_GT_SIZES
        frames      = FULL_FRAMES
        n_feats_l   = FULL_N_FEATS
        n_blocks_l  = FULL_N_BLOCKS
        batch_sizes = FULL_BATCH_SIZES
        precisions  = FULL_PRECISIONS
    else:
        # Default = quick
        gt_keys     = QUICK_GT_SIZES
        frames      = QUICK_FRAMES
        n_feats_l   = QUICK_N_FEATS
        n_blocks_l  = QUICK_N_BLOCKS
        batch_sizes = QUICK_BATCH_SIZES
        precisions  = QUICK_PRECISIONS

    # ── Apply per-parameter CLI overrides ──────────────────────────────────
    if args.gt_sizes:
        gt_keys = []
        for s in args.gt_sizes:
            if 'x' in s.lower():
                # Input format is "WxH" (width × height, display convention)
                w_str, h_str = s.lower().split('x')
                key = f"{w_str}x{h_str}"
                # Accept either known key or ad-hoc size
                if key not in ALL_GT_SIZES:
                    try:
                        # Store as (H, W) — PyTorch convention (height first)
                        w, h = int(w_str), int(h_str)
                        ALL_GT_SIZES[key] = (h, w)
                    except ValueError:
                        print(f"[WARN] Cannot parse GT size '{s}', skipping.")
                        continue
                gt_keys.append(key)

    if args.frames:    frames      = args.frames
    if args.n_feats:   n_feats_l   = args.n_feats
    if args.n_blocks:  n_blocks_l  = args.n_blocks
    if args.batch_sizes: batch_sizes = args.batch_sizes
    if args.precisions:  precisions  = args.precisions

    # ── Enumerate all combinations ─────────────────────────────────────────
    configs = []
    for gt_key in gt_keys:
        if gt_key not in ALL_GT_SIZES:
            print(f"[WARN] Unknown GT size key '{gt_key}', skipping.")
            continue
        gt_size = ALL_GT_SIZES[gt_key]
        for n_f in frames:
            for bs in batch_sizes:
                accum = ACCUM_MAP.get((bs, gt_key))
                if accum is None:
                    # Check if user provided an ad-hoc size not in ACCUM_MAP
                    if (bs, gt_key) not in ACCUM_MAP:
                        # Default conservative accumulation
                        gt_h, gt_w = gt_size
                        pixels = gt_h * gt_w
                        # Use smaller accum for larger patches
                        if pixels >= 1920 * 1080:
                            accum = 2 if bs == 1 else None
                        elif pixels >= 960 * 540:
                            accum = 4 if bs == 1 else 2
                        else:
                            accum = 8 if bs == 1 else 4
                for prec in precisions:
                    for n_feats in n_feats_l:
                        for n_blocks in n_blocks_l:
                            configs.append(dict(
                                n_frames=n_f, batch_size=bs,
                                n_feats=n_feats, n_blocks=n_blocks,
                                gt_key=gt_key, gt_size=gt_size,
                                precision=prec,
                                accum=accum,   # None = skip (known OOM)
                            ))
    return configs


def _config_label(c: Dict) -> str:
    """Short human-readable label for a config."""
    gt_h, gt_w = c['gt_size']
    return (f"{c['n_frames']}f | B{c['batch_size']}×A{c['accum'] or '?'} | "
            f"{c['n_blocks']}b | {c['n_feats']}feat | "
            f"{gt_w}×{gt_h} | {c['precision'].upper()}")


def _print_table(results: List[Dict], p100_gb: float) -> None:
    """Print a formatted summary table to stdout."""
    ok  = [r for r in results if r.get('success')]
    oom = [r for r in results if not r.get('success') and r.get('error') not in (None, 'SKIPPED')]
    skipped = [r for r in results if r.get('error') == 'SKIPPED']

    ok_sorted = sorted(ok, key=lambda x: x['vram_gb'])

    sep = "─" * 108
    print(f"\n{'═' * 108}")
    print("  VSR++ BENCHMARK — RESULTS SUMMARY")
    print(f"{'═' * 108}")
    print(f"  Tested:   {len(results)}  |  OK: {len(ok)}  |  OOM: {len(oom)}  |  Skipped: {len(skipped)}")
    print(f"  P100 VRAM budget: {p100_gb:.0f} GB  |  Safe threshold: {p100_gb * 0.90:.1f} GB")
    print()

    # ── Successful configs ─────────────────────────────────────────────────
    print(f"  {'#':>3}  {'Config':<52}  {'VRAM GB':>8}  {'% P100':>7}  {'s/iter':>8}  {'Fit?':>6}")
    print(f"  {sep}")
    for idx, r in enumerate(ok_sorted, 1):
        vram    = r['vram_gb']
        pct     = 100.0 * vram / p100_gb
        siter   = r['time_per_iter']
        fit_sym = '✅' if vram <= p100_gb * 0.90 else ('⚠️ ' if vram <= p100_gb else '❌')
        label   = _config_label(r)
        print(f"  {idx:>3}  {label:<52}  {vram:>8.2f}  {pct:>6.1f}%  {siter:>8.3f}  {fit_sym}")

    if ok:
        print(f"\n  ── Top 5 by best VRAM/speed trade-off (lightest first) ────────────────────────────────────────────────────")
        for r in ok_sorted[:5]:
            eff = r.get('batch_size', 1) * (r.get('accum') or 1)
            est_hours = r['time_per_iter'] * 150_000 / 3600
            print(f"     {_config_label(r)}")
            print(f"     → VRAM {r['vram_gb']:.2f} GB  |  {r['time_per_iter']:.3f} s/iter  |  eff.BS={eff}  |  ~{est_hours:.0f} h for 150k steps")

    # ── OOM configs ────────────────────────────────────────────────────────
    if oom:
        print(f"\n  ── OOM / Error configs ({len(oom)}) ─────────────────────────────────────────────────────────────────────")
        for r in oom:
            print(f"     ❌  {_config_label(r)}  → {r.get('error', '?')}")

    print(f"\n{'═' * 108}")


def _save_csv(results: List[Dict], path: str) -> None:
    """Save results as CSV."""
    if not results:
        return
    fields = ['n_frames', 'batch_size', 'accum', 'n_feats', 'n_blocks',
              'gt_key', 'gt_h', 'gt_w', 'precision',
              'success', 'vram_gb', 'time_per_iter', 'error']
    with open(path, 'w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction='ignore')
        w.writeheader()
        for r in results:
            row = dict(r)
            gt  = row.pop('gt_size', (0, 0))
            row['gt_h'] = gt[0]
            row['gt_w'] = gt[1]
            w.writerow(row)


def _save_json(results: List[Dict], path: str) -> None:
    """Save results as JSON."""
    serialisable = []
    for r in results:
        row = dict(r)
        row['gt_size'] = list(row.get('gt_size', [0, 0]))
        serialisable.append(row)
    with open(path, 'w', encoding='utf-8') as f:
        json.dump({'generated': datetime.now().isoformat(),
                   'p100_vram_gb': P100_VRAM_GB,
                   'results': serialisable}, f, indent=2)


def main() -> None:
    args = _parse_args()

    # ── Sanity checks ──────────────────────────────────────────────────────
    if not _IMPORTS_OK:
        print(f"[ERROR] Could not import production model/loss components:\n  {_IMPORT_ERROR}")
        print("  Make sure you run this script from inside the vsr_plusplus_NEU directory")
        print("  or that the package is on PYTHONPATH.")
        sys.exit(1)

    if not torch.cuda.is_available():
        print("[ERROR] No CUDA device found.  This benchmark requires a CUDA-capable GPU.")
        sys.exit(1)

    # ── Header ─────────────────────────────────────────────────────────────
    device_name = torch.cuda.get_device_name(0)
    vram_total  = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)
    ts          = datetime.now().strftime('%Y-%m-%d %H:%M:%S')

    print(f"\n{'═' * 108}")
    print("  VSR++ Training Configuration Benchmark")
    print(f"{'═' * 108}")
    print(f"  Started  : {ts}")
    print(f"  Device   : {device_name}  ({vram_total:.1f} GB VRAM detected)")
    print(f"  Mode     : {'full' if args.full else 'quick (default)'}")
    print(f"  Timing   : {WARMUP_ITERS} warm-up + {args.timing_iters} timed iterations per config")
    print()
    print("  What this measures:")
    print("    • Full training step: forward → HybridLoss (incl. VGG16) → backward → Adam step")
    print("    • Peak VRAM allocation (includes perceptual network ~400-650 MB)")
    print("    • Wall-clock time per optimizer step (with gradient accumulation)")
    print()
    print("  Synthetic data notes:")
    print("    • GT = smooth structured random content (not pure noise)")
    print("    • LR = 3× bicubic downscale of GT + per-frame motion shifts + mild noise")
    print("    • More realistic than randn() — LR is correlated with GT as in real training")
    print(f"{'═' * 108}\n")

    # ── Build config list ──────────────────────────────────────────────────
    configs = _build_config_list(args)
    total   = len(configs)

    if total == 0:
        print("[ERROR] No configurations to test.  Check --gt-sizes / --frames arguments.")
        sys.exit(1)

    # Notify about 9/11-frame support
    extra_frames = [c['n_frames'] for c in configs if c['n_frames'] != 7]
    if extra_frames:
        unique_extra = sorted(set(extra_frames))
        print(f"  ℹ️  Frame counts {unique_extra} use VSRBidirectional_Nframes_3x")
        print("     (same architecture as production 7-frame model, generalised loop indices)")
        print(f"     N=7 uses production VSRBidirectional_7frames_3x directly.\n")

    print(f"  Total configurations: {total}")

    # ── Dry run ─────────────────────────────────────────────────────────────
    if args.dry_run:
        print("\n  DRY RUN — planned configurations:\n")
        for idx, c in enumerate(configs, 1):
            skipped = '  [SKIP-OOM]' if c['accum'] is None else ''
            print(f"    [{idx:>3}/{total}] {_config_label(c)}{skipped}")
        print()
        return

    # ── Run benchmarks ─────────────────────────────────────────────────────
    results = []
    for idx, c in enumerate(configs, 1):
        label = _config_label(c)
        print(f"  [{idx:>3}/{total}]  {label}")

        if c['accum'] is None:
            print(f"          ⏭  SKIPPED (pre-identified OOM for this batch/resolution)")
            results.append(dict(**c, success=False, vram_gb=0.0,
                                time_per_iter=0.0, error='SKIPPED'))
            continue

        r = test_config(
            n_frames=c['n_frames'],
            batch_size=c['batch_size'],
            n_feats=c['n_feats'],
            n_blocks=c['n_blocks'],
            gt_key=c['gt_key'],
            gt_size=c['gt_size'],
            precision=c['precision'],
            accum=c['accum'],
            timing_iters=args.timing_iters,
            warmup_iters=WARMUP_ITERS,
        )

        if r['success']:
            pct = 100.0 * r['vram_gb'] / vram_total
            print(f"          ✅  {r['vram_gb']:.2f} GB ({pct:.1f}%)  |  {r['time_per_iter']:.3f} s/iter")
        else:
            print(f"          ❌  {r.get('error', 'unknown error')}")

        results.append(r)

    # ── Print summary table ────────────────────────────────────────────────
    _print_table(results, P100_VRAM_GB)

    # ── Save outputs ───────────────────────────────────────────────────────
    os.makedirs(args.output_dir, exist_ok=True)

    if not args.no_csv:
        csv_path = os.path.join(args.output_dir, 'benchmark_results.csv')
        _save_csv(results, csv_path)
        print(f"  CSV  → {csv_path}")

    if not args.no_json:
        json_path = os.path.join(args.output_dir, 'benchmark_results.json')
        _save_json(results, json_path)
        print(f"  JSON → {json_path}")

    print()


if __name__ == '__main__':
    main()

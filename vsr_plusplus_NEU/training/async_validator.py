#!/usr/bin/env python3
"""
Async Validation Process

Runs as a standalone process on a secondary GPU (e.g. cuda:1) and monitors
for new validation requests written by the training process.

Communication protocol (file-based, in the checkpoint directory):
  async_val_request.json  - Written by training process to trigger validation.
                             Contains 'checkpoint_path', 'step', 'log_dir',
                             'data_root', 'dataset_name', 'val_sizes',
                             'config_snapshot'.
  async_val_result.json   - Written by this process after validation completes.
                             Contains the same metrics dict that VSRValidator
                             normally returns, plus 'step' and 'timestamp'.

Usage (invoked from train.py):
    python -m vsr_plusplus_NEU.training.async_validator \\
        --checkpoint-dir /path/to/checkpoints \\
        --data-root      /path/to/data \\
        --dataset-name   master \\
        --log-dir        /path/to/logs \\
        --gpu            1

The process runs until a file named ``async_val_stop`` appears in the
checkpoint directory (written by the training process on exit).
"""

import argparse
import json
import os
import sys
import time
import traceback

import cv2
import numpy as np
import torch
import torch.nn.functional as F

# ---------------------------------------------------------------------------
# Resolve imports whether launched as script or module
# ---------------------------------------------------------------------------
_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(os.path.dirname(_HERE))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)
if os.path.dirname(_HERE) not in sys.path:
    sys.path.insert(0, os.path.dirname(_HERE))


# ---------------------------------------------------------------------------
# Adaptive batch-size probe
# ---------------------------------------------------------------------------

def _find_safe_batch_size(model, val_dataset, device, config_snapshot):
    """
    Try progressively smaller batch sizes (starting from 4) until the
    forward pass on a 720x720 resolution batch succeeds without OOM.

    Rationale:
    - The validation process runs without gradients or optimiser states,
      so it needs ~50% of the training VRAM.
    - 720x720 is the most memory-intensive resolution, so probing with it
      gives a safe upper bound that works for all smaller resolutions too.
    - If a CUDA OOM is raised the batch size is halved and the probe is
      retried until it reaches 1 (which always fits).

    Returns:
        int: Safe batch size (≥ 1)
    """
    # Prefer to probe with the largest resolution available in the dataset.
    # The caller already passes the correct dataset for the probe.
    probe_batch = 4  # start with a reasonably high value

    model.eval()
    while probe_batch >= 1:
        lr_batch = None
        try:
            # Build a dummy batch of the right spatial size from dataset[0].
            sample = val_dataset[0]
            lr_sample, gt_sample, _ = sample
            lr_batch = lr_sample.unsqueeze(0).expand(probe_batch, -1, -1, -1).to(device)
            with torch.no_grad():
                _ = model(lr_batch)
            del lr_batch
            torch.cuda.empty_cache()
            print(f"[AsyncVal] Probe succeeded: batch_size={probe_batch}")
            return probe_batch
        except torch.cuda.OutOfMemoryError:
            if lr_batch is not None:
                del lr_batch
            torch.cuda.empty_cache()
            probe_batch //= 2
            if probe_batch < 1:
                break
            print(f"[AsyncVal] OOM at probe batch, retrying with batch_size={probe_batch}")
        except Exception as e:
            # Non-OOM errors should not silently swallow – fall back to 1.
            if lr_batch is not None:
                del lr_batch
            print(f"[AsyncVal] Probe error ({e}), using batch_size=1")
            torch.cuda.empty_cache()
            return 1

    print("[AsyncVal] Could not find safe batch size via probe, using batch_size=1")
    return 1


# ---------------------------------------------------------------------------
# Core validation logic (GPU-agnostic helper)
# ---------------------------------------------------------------------------

def _run_validation_on_device(model, val_loaders, loss_fn, device, global_step):
    """
    Run multi-size validation and return combined metrics dict.

    Mirrors the logic in VSRTrainer._run_multi_size_validation() /
    VSRValidator.validate() but is self-contained so that it can run in
    a separate process without importing VSRTrainer.

    Args:
        model:       VSR model (already on *device*, in eval mode).
        val_loaders: List of (size_key, DataLoader) tuples.
        loss_fn:     HybridLoss instance (on *device*).
        device:      torch.device
        global_step: Training step the checkpoint belongs to (for logging).

    Returns:
        dict: Combined metrics (same keys as VSRValidator.validate()).
    """
    from vsr_plusplus_NEU.utils.metrics import calculate_psnr, calculate_ssim, quality_to_percent

    all_metrics = []
    all_labeled_images = []

    model.eval()

    for size_key, val_loader in val_loaders:
        print(f"[AsyncVal] Validating {size_key} ({len(val_loader)} batches)…")

        total_loss = 0.0
        total_lr_psnr = total_lr_ssim = total_ki_psnr = total_ki_ssim = 0.0
        total_improvement = total_ki_to_gt = total_lr_to_gt = 0.0
        num_samples = 0

        val_total = len(val_loader)
        val_start = time.time()

        with torch.no_grad():
            for batch_idx, batch in enumerate(val_loader):
                lr_stack, gt, filenames = batch
                num_samples += lr_stack.size(0)

                # Minimal progress indicator (no ANSI needed – stdout goes to log)
                if batch_idx % max(1, val_total // 10) == 0:
                    pct = (batch_idx + 1) / val_total * 100
                    elapsed = time.time() - val_start
                    eta = elapsed / (batch_idx + 1) * (val_total - batch_idx - 1) if batch_idx > 0 else 0
                    print(f"[AsyncVal]   {size_key}: {pct:.0f}% ({batch_idx+1}/{val_total}) ETA {eta:.0f}s")

                lr_stack = lr_stack.to(device, non_blocking=True)
                gt = gt.to(device, non_blocking=True)

                ki_output = model(lr_stack)

                loss_dict = loss_fn(ki_output, gt)
                total_loss += loss_dict['total'].item() if torch.is_tensor(loss_dict['total']) else loss_dict['total']
                del loss_dict

                lr_center = lr_stack[:, 3]
                lr_upscaled = F.interpolate(lr_center, scale_factor=3, mode='bilinear', align_corners=False)
                del lr_center

                for i in range(lr_stack.size(0)):
                    lr_psnr = calculate_psnr(lr_upscaled[i], gt[i])
                    lr_ssim = calculate_ssim(lr_upscaled[i], gt[i])
                    ki_psnr = calculate_psnr(ki_output[i], gt[i])
                    ki_ssim = calculate_ssim(ki_output[i], gt[i])

                    total_lr_psnr += lr_psnr
                    total_lr_ssim += lr_ssim
                    total_ki_psnr += ki_psnr
                    total_ki_ssim += ki_ssim

                    lr_qual = quality_to_percent(lr_psnr, lr_ssim)
                    ki_qual = quality_to_percent(ki_psnr, ki_ssim)
                    gt_qual = 1.0

                    total_improvement += (ki_qual - lr_qual)
                    total_ki_to_gt += (ki_qual - gt_qual)
                    total_lr_to_gt += (lr_qual - gt_qual)

                    # Build labeled comparison images for TensorBoard
                    lr_img = lr_upscaled[i].cpu().permute(1, 2, 0).numpy()
                    ki_img = ki_output[i].cpu().permute(1, 2, 0).numpy()
                    gt_img = gt[i].cpu().permute(1, 2, 0).numpy()

                    lr_img = np.clip(lr_img * 255, 0, 255).astype(np.uint8).copy()
                    ki_img = np.clip(ki_img * 255, 0, 255).astype(np.uint8).copy()
                    gt_img = np.clip(gt_img * 255, 0, 255).astype(np.uint8).copy()

                    font = cv2.FONT_HERSHEY_SIMPLEX
                    font_scale = 1.5
                    thickness = 3

                    cv2.putText(lr_img, f"LR {lr_qual*100:.1f}%", (10, 40), font, font_scale, (255, 255, 255), thickness)
                    cv2.putText(lr_img, f"LR {lr_qual*100:.1f}%", (10, 40), font, font_scale, (0, 255, 0), thickness - 1)

                    cv2.putText(ki_img, f"KI {ki_qual*100:.1f}%", (10, 40), font, font_scale, (255, 255, 255), thickness)
                    cv2.putText(ki_img, f"KI {ki_qual*100:.1f}%", (10, 40), font, font_scale, (0, 255, 255), thickness - 1)

                    cv2.putText(gt_img, "GT 100.0%", (10, 40), font, font_scale, (255, 255, 255), thickness)
                    cv2.putText(gt_img, "GT 100.0%", (10, 40), font, font_scale, (255, 0, 0), thickness - 1)

                    border_width = 3
                    lr_bordered = cv2.copyMakeBorder(lr_img, 0, 0, 0, border_width, cv2.BORDER_CONSTANT, value=(0, 0, 0))
                    ki_bordered = cv2.copyMakeBorder(ki_img, 0, 0, 0, border_width, cv2.BORDER_CONSTANT, value=(0, 0, 0))
                    combined = np.concatenate([lr_bordered, ki_bordered, gt_img], axis=1)

                    combined_tensor = torch.from_numpy(combined).permute(2, 0, 1).float() / 255.0
                    name = os.path.splitext(os.path.basename(filenames[i]))[0]
                    all_labeled_images.append((f"val_{size_key}/{name}", combined_tensor.contiguous()))

                del lr_stack, gt, ki_output, lr_upscaled
                torch.cuda.empty_cache()

        avg_loss = total_loss / max(1, val_total)
        avg_lr_psnr = total_lr_psnr / max(1, num_samples)
        avg_lr_ssim = total_lr_ssim / max(1, num_samples)
        avg_ki_psnr = total_ki_psnr / max(1, num_samples)
        avg_ki_ssim = total_ki_ssim / max(1, num_samples)

        lr_quality = quality_to_percent(avg_lr_psnr, avg_lr_ssim)
        ki_quality = quality_to_percent(avg_ki_psnr, avg_ki_ssim)

        m = {
            'val_loss': avg_loss,
            'lr_quality': lr_quality,
            'ki_quality': ki_quality,
            'improvement': total_improvement,
            'ki_to_gt': total_ki_to_gt,
            'lr_to_gt': total_lr_to_gt,
            'lr_psnr': avg_lr_psnr,
            'lr_ssim': avg_lr_ssim,
            'ki_psnr': avg_ki_psnr,
            'ki_ssim': avg_ki_ssim,
        }
        all_metrics.append((size_key, m))
        print(f"[AsyncVal]   ✓ {size_key}: KI {ki_quality*100:.1f}%  PSNR {avg_ki_psnr:.2f} dB")

    if not all_metrics:
        return {}

    combined = {
        'val_loss':    sum(m['val_loss']    for _, m in all_metrics) / len(all_metrics),
        'lr_quality':  sum(m['lr_quality']  for _, m in all_metrics) / len(all_metrics),
        'ki_quality':  sum(m['ki_quality']  for _, m in all_metrics) / len(all_metrics),
        'improvement': sum(m['improvement'] for _, m in all_metrics) / len(all_metrics),
        'ki_to_gt':    sum(m['ki_to_gt']    for _, m in all_metrics) / len(all_metrics),
        'lr_to_gt':    sum(m['lr_to_gt']    for _, m in all_metrics) / len(all_metrics),
        'lr_psnr':     sum(m['lr_psnr']     for _, m in all_metrics) / len(all_metrics),
        'lr_ssim':     sum(m['lr_ssim']     for _, m in all_metrics) / len(all_metrics),
        'ki_psnr':     sum(m['ki_psnr']     for _, m in all_metrics) / len(all_metrics),
        'ki_ssim':     sum(m['ki_ssim']     for _, m in all_metrics) / len(all_metrics),
        'labeled_images': all_labeled_images,
        'per_size_metrics': {sk: m for sk, m in all_metrics},
    }
    return combined


# ---------------------------------------------------------------------------
# Error-result helper
# ---------------------------------------------------------------------------

def _write_error_result(result_file: str, step: int, error_message: str) -> None:
    """
    Write an error sentinel to ``result_file`` so the training process can
    detect and report failures instead of silently ignoring them.

    The training process recognises an error result by the presence of the
    ``'error'`` key.  It logs the message and does **not** update quality
    metrics (which must remain at their last valid values).
    """
    payload = {
        'step':       step,
        'error':      error_message,
        'timestamp':  time.time(),
    }
    tmp = result_file + '.tmp'
    try:
        with open(tmp, 'w') as f:
            json.dump(payload, f, indent=2)
        os.replace(tmp, result_file)
        print(f"[AsyncVal] Error result written for step {step}: {error_message}")
    except OSError as e:
        print(f"[AsyncVal] ⚠ Could not write error result file: {e}")


# ---------------------------------------------------------------------------
# Main async-validator loop
# ---------------------------------------------------------------------------

def run_async_validator(checkpoint_dir, data_root, dataset_name, log_dir, gpu_index,
                        config_snapshot=None):
    """
    Main entry point for the async validation process.

    Runs in an infinite loop, polling for ``async_val_request.json``.
    Writes results to ``async_val_result.json`` and TensorBoard.

    Args:
        checkpoint_dir: Directory that contains checkpoint files and the
                        sentinel files used for IPC.
        data_root:      Root of the dataset (e.g. ``/mnt/data/training/…``).
        dataset_name:   Sub-folder name (e.g. ``master``).
        log_dir:        TensorBoard log directory.
        gpu_index:      Integer index of the GPU to use (0-based).
        config_snapshot: Optional dict with model/training config (used to
                         reconstruct the model if not embedded in checkpoint).
    """
    from vsr_plusplus_NEU.core.model_7frame import VSRBidirectional_7frames_3x
    from vsr_plusplus_NEU.core.loss import HybridLoss
    from vsr_plusplus_NEU.core.dataset import VSRDataset
    from torch.utils.data import DataLoader
    from vsr_plusplus_NEU.systems.logger import TensorBoardLogger

    device = torch.device(f'cuda:{gpu_index}' if torch.cuda.is_available() else 'cpu')
    print(f"[AsyncVal] Starting on {device}  (checkpoint_dir={checkpoint_dir})")

    request_file = os.path.join(checkpoint_dir, 'async_val_request.json')
    result_file  = os.path.join(checkpoint_dir, 'async_val_result.json')
    stop_file    = os.path.join(checkpoint_dir, 'async_val_stop')
    done_file    = os.path.join(checkpoint_dir, 'async_val_done.json')

    tb_logger = TensorBoardLogger(log_dir)

    last_processed_step = -1

    while True:
        # ── Stop signal ──────────────────────────────────────────────────────
        if os.path.exists(stop_file):
            print("[AsyncVal] Stop signal received – shutting down.")
            try:
                os.unlink(stop_file)
            except OSError:
                pass
            break

        # ── Poll for new request ─────────────────────────────────────────────
        if not os.path.exists(request_file):
            time.sleep(2.0)
            continue

        # Read request
        try:
            with open(request_file, 'r') as f:
                request = json.load(f)
        except (json.JSONDecodeError, OSError):
            time.sleep(1.0)
            continue

        step           = request.get('step', 0)
        checkpoint_path = request.get('checkpoint_path', '')
        req_data_root  = request.get('data_root', data_root)
        req_ds_name    = request.get('dataset_name', dataset_name)
        val_sizes      = request.get('val_sizes', ['540'])
        req_config     = request.get('config_snapshot', config_snapshot or {})

        # Skip duplicates (same step processed before)
        if step == last_processed_step:
            time.sleep(2.0)
            continue

        # Consume the request file immediately to signal we accepted it
        try:
            os.unlink(request_file)
        except OSError:
            pass

        print(f"[AsyncVal] Processing step {step}  checkpoint={checkpoint_path}")

        # ── Verify checkpoint file exists before loading ──────────────────────
        if not os.path.exists(checkpoint_path):
            err_msg = f"Checkpoint file not found: {checkpoint_path}"
            print(f"[AsyncVal] ❌ {err_msg}")
            _write_error_result(result_file, step, err_msg)
            time.sleep(2.0)
            continue

        # ── Load model ───────────────────────────────────────────────────────
        try:
            n_feats  = req_config.get('N_FEATS',  72)
            n_blocks = req_config.get('N_BLOCKS', 28)
            model = VSRBidirectional_7frames_3x(n_feats=n_feats, n_blocks=n_blocks)

            # Activate gradient checkpointing if training used it (saves VRAM)
            if req_config.get('USE_CHECKPOINTING', False) and hasattr(model, 'enable_checkpointing'):
                model.enable_checkpointing()

            # Use weights_only=False for compatibility with PyTorch 2.6+.
            # The training process also uses weights_only=False (see train.py).
            # weights_only=True would reject some tensor types present in model
            # state dicts saved by PyTorch 2.6+ and cause a silent failure.
            ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
            if 'model_state_dict' in ckpt:
                model.load_state_dict(ckpt['model_state_dict'])
            else:
                # Lightweight weights-only state dict saved by _request_async_validation
                model.load_state_dict(ckpt)

            model = model.to(device)
            model.eval()
            print(f"[AsyncVal] Model loaded from {os.path.basename(checkpoint_path)}")
        except Exception as e:
            err_msg = f"Failed to load model: {e}"
            print(f"[AsyncVal] ❌ {err_msg}")
            traceback.print_exc()
            _write_error_result(result_file, step, err_msg)
            time.sleep(5.0)
            continue

        # ── Load loss function ───────────────────────────────────────────────
        try:
            loss_fn = HybridLoss(
                l1_weight=req_config.get('L1_WEIGHT', 0.60),
                ms_weight=req_config.get('MS_WEIGHT', 0.20),
                grad_weight=req_config.get('GRAD_WEIGHT', 0.20),
                perceptual_weight=req_config.get('PERCEPTUAL_WEIGHT', 0.0),
            ).to(device)
        except Exception as e:
            err_msg = f"Failed to create loss function: {e}"
            print(f"[AsyncVal] ❌ {err_msg}")
            traceback.print_exc()
            _write_error_result(result_file, step, err_msg)
            time.sleep(5.0)
            continue

        # ── Load validation datasets ─────────────────────────────────────────
        try:
            val_loaders = []
            for size_key in val_sizes:
                val_ds = VSRDataset(
                    root=req_data_root,
                    dataset_name=req_ds_name,
                    size_key=size_key,
                    mode='val',
                    augment=False,
                    paths_config=None,
                )
                if len(val_ds) == 0:
                    print(f"[AsyncVal]   ⚠ {size_key}: 0 samples – skipping")
                    continue

                # Determine safe batch size once, using the most memory-intensive
                # size key available (prefer '720', then '540', then '720_169').
                # We probe only for the first dataset to keep startup latency low.
                if not val_loaders and size_key == '720':
                    bs = _find_safe_batch_size(model, val_ds, device, req_config)
                else:
                    bs = 1  # safe default for other sizes / after first probe

                loader = DataLoader(
                    val_ds, batch_size=bs,
                    shuffle=False, num_workers=2, pin_memory=False,
                )
                val_loaders.append((size_key, loader))
                print(f"[AsyncVal]   {size_key}: {len(val_ds)} samples, batch_size={bs}")
        except Exception as e:
            err_msg = f"Failed to build validation datasets: {e}"
            print(f"[AsyncVal] ❌ {err_msg}")
            traceback.print_exc()
            _write_error_result(result_file, step, err_msg)
            del model, loss_fn
            torch.cuda.empty_cache()
            time.sleep(5.0)
            continue

        if not val_loaders:
            err_msg = "No validation datasets available (0 samples in all size keys)"
            print(f"[AsyncVal] ⚠ {err_msg}")
            _write_error_result(result_file, step, err_msg)
            del model, loss_fn
            torch.cuda.empty_cache()
            time.sleep(5.0)
            continue

        # ── Run validation ───────────────────────────────────────────────────
        t_start = time.time()
        try:
            metrics = _run_validation_on_device(model, val_loaders, loss_fn, device, step)
        except Exception as e:
            err_msg = f"Validation inference failed: {e}"
            print(f"[AsyncVal] ❌ {err_msg}")
            traceback.print_exc()
            _write_error_result(result_file, step, err_msg)
            # Clean up model to free VRAM before retrying
            del model, loss_fn
            torch.cuda.empty_cache()
            time.sleep(5.0)
            continue

        elapsed = time.time() - t_start
        ki_q = metrics.get('ki_quality', 0.0)
        print(f"[AsyncVal] ✅ Step {step} done in {elapsed:.1f}s – KI Quality {ki_q*100:.1f}%")

        # ── Write results for training process ───────────────────────────────
        serialisable = {k: v for k, v in metrics.items()
                        if k not in ('labeled_images', 'per_size_metrics')}
        serialisable['step']                = step
        serialisable['timestamp']           = time.time()
        # Include total elapsed time so the training process can update its
        # validation-speed tracker (samples/s incl. TensorBoard I/O).
        serialisable['val_elapsed_seconds'] = elapsed

        # Write per-size metrics (scalars only) for detailed logging
        if 'per_size_metrics' in metrics:
            per_size_out = {}
            for sk, m in metrics['per_size_metrics'].items():
                per_size_out[sk] = {kk: vv for kk, vv in m.items()
                                    if kk not in ('labeled_images',)}
            serialisable['per_size_metrics'] = per_size_out

        # Atomic write: write to tmp file then rename
        tmp_result = result_file + '.tmp'
        try:
            with open(tmp_result, 'w') as f:
                json.dump(serialisable, f, indent=2)
            os.replace(tmp_result, result_file)
            print(f"[AsyncVal] Results written to {result_file}")
        except OSError as e:
            print(f"[AsyncVal] ⚠ Could not write result file: {e}")

        # ── Log to TensorBoard ───────────────────────────────────────────────
        try:
            tb_logger.log_quality(step, metrics)
            tb_logger.log_metrics(step, metrics)
            tb_logger.log_validation_loss(step, metrics.get('val_loss', 0.0))

            labeled_images = metrics.get('labeled_images', [])
            for tag, img_tensor in labeled_images:
                if img_tensor.device.type != 'cpu':
                    img_tensor = img_tensor.cpu()
                tb_logger.writer.add_image(tag, img_tensor, step)
            if labeled_images:
                tb_logger.writer.flush()
                print(f"[AsyncVal] Logged {len(labeled_images)} images to TensorBoard")
        except Exception as e:
            print(f"[AsyncVal] ⚠ TensorBoard logging error: {e}")

        # ── Clean up GPU memory for next round ───────────────────────────────
        del model, loss_fn, metrics
        torch.cuda.empty_cache()

        last_processed_step = step

    print("[AsyncVal] Process exiting.")


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def _parse_args():
    p = argparse.ArgumentParser(description="Async validation process for VSR++ training")
    p.add_argument('--checkpoint-dir', required=True, help='Checkpoint directory (IPC location)')
    p.add_argument('--data-root',      required=True, help='Dataset root directory')
    p.add_argument('--dataset-name',   required=True, help='Dataset name (e.g. master)')
    p.add_argument('--log-dir',        required=True, help='TensorBoard log directory')
    p.add_argument('--gpu',            type=int, default=1, help='GPU index to use (default: 1)')
    p.add_argument('--config-json',    default=None,
                   help='Path to JSON file with config snapshot (optional)')
    return p.parse_args()


if __name__ == '__main__':
    args = _parse_args()

    config_snapshot = {}
    if args.config_json and os.path.exists(args.config_json):
        with open(args.config_json, 'r') as f:
            config_snapshot = json.load(f)

    run_async_validator(
        checkpoint_dir=args.checkpoint_dir,
        data_root=args.data_root,
        dataset_name=args.dataset_name,
        log_dir=args.log_dir,
        gpu_index=args.gpu,
        config_snapshot=config_snapshot,
    )

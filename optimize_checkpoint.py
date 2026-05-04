#!/usr/bin/env python3
"""
optimize_checkpoint.py — VSR++ Modell-Konvertierung

Konvertiert einen trainierten VSR++ Checkpoint in optimierte Formate:

  1. TensorRT (FP16/FP32) — beste NVIDIA-Performance
       Pfad: PyTorch → ONNX (temp) → TensorRT Engine
       Benötigt: pip install tensorrt onnx
       Hinweis: tensorrt-Paket muss zur CUDA-Toolkit-Version passen

  2. ONNX — portables offenes Format
       Benötigt: pip install onnx onnxruntime-gpu

  3. TorchScript — PyTorch JIT, läuft ohne Originalcode
       Benötigt: nur torch (eingebaut)

  4. Pruning — entfernt unwichtige Weights (kleineres Modell)
       Benötigt: nur torch (eingebaut)

Verwendung:
    # TensorRT FP16 (empfohlen für P100)
    python optimize_checkpoint.py -c model.pth -o model_fp16.engine -f tensorrt -p fp16

    # ONNX (480×270 LR → 1440×810 SR)
    python optimize_checkpoint.py -c model.pth -o model.onnx -f onnx --width 480 --height 270

    # TorchScript
    python optimize_checkpoint.py -c model.pth -o model.pt -f torchscript

    # Pruning 30% strukturiert
    python optimize_checkpoint.py -c model.pth -o model_pruned.pth -f pruned --prune-amount 0.3

Hinweis P100 / CC 6.0:
    FP16 wird vollständig unterstützt.
    torch.compile/Triton ist NICHT verfügbar (erfordert CC ≥ 7.0).
"""

import argparse
import json
import os
import sys
import tempfile
import time
import traceback
from pathlib import Path

import numpy as np
import torch

# Repo-Root und vsr_plusplus_NEU in Suchpfad aufnehmen
_REPO_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(_REPO_ROOT))
sys.path.insert(0, str(_REPO_ROOT / "vsr_plusplus_NEU"))


# ─────────────────────────────────────────────────────────────────────────────
# Modell laden
# ─────────────────────────────────────────────────────────────────────────────

def load_model(checkpoint_path: str, device: str):
    """
    Lädt VSRBidirectional_7frames_3x aus einem .pth-Checkpoint.

    Gibt (model, info_dict) zurück.
    model ist im eval()-Modus ohne Gradienten.
    """
    ckpt_path = Path(checkpoint_path)
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint nicht gefunden: {ckpt_path}")

    print(f"📦 Lade Checkpoint: {ckpt_path}")
    ckpt = torch.load(str(ckpt_path), map_location=device, weights_only=False)

    # Konfig — aus lokalem config.py falls vorhanden, sonst Defaults
    n_feats, n_blocks = 72, 28
    try:
        import config as cfg
        c = cfg.get_config()
        n_feats  = c.get("N_FEATS",  n_feats)
        n_blocks = c.get("N_BLOCKS", n_blocks)
        print(f"   ✅ config.py: N_FEATS={n_feats}, N_BLOCKS={n_blocks}")
    except Exception:
        print(f"   ℹ️  config.py nicht gefunden — Defaults: N_FEATS={n_feats}, N_BLOCKS={n_blocks}")

    from vsr_plusplus_NEU.core.model_7frame import VSRBidirectional_7frames_3x
    model = VSRBidirectional_7frames_3x(n_feats=n_feats, n_blocks=n_blocks).to(device)

    # state_dict kann direkt oder in 'model_state_dict' liegen
    state = ckpt.get("model_state_dict", ckpt)
    model.load_state_dict(state)
    model.eval()

    info = {
        "step":    ckpt.get("step",  "?"),
        "epoch":   ckpt.get("epoch", "?"),
        "n_feats": n_feats,
        "n_blocks": n_blocks,
    }
    print(f"✅ Modell geladen — Step {info['step']}, Epoch {info['epoch']}, "
          f"N_FEATS={n_feats}, N_BLOCKS={n_blocks}")
    return model, info


# ─────────────────────────────────────────────────────────────────────────────
# Benchmark
# ─────────────────────────────────────────────────────────────────────────────

def _benchmark(model, device: str, input_shape: tuple, n: int = 20) -> float:
    """Gibt mittlere Inferenzzeit in ms zurück."""
    dummy = torch.randn(*input_shape).to(device)
    # Warmup
    with torch.no_grad():
        for _ in range(5):
            model(dummy)
    if device == "cuda":
        torch.cuda.synchronize()

    times = []
    with torch.no_grad():
        for _ in range(n):
            if device == "cuda":
                torch.cuda.synchronize()
            t0 = time.perf_counter()
            model(dummy)
            if device == "cuda":
                torch.cuda.synchronize()
            times.append((time.perf_counter() - t0) * 1000)

    avg, std = float(np.mean(times)), float(np.std(times))
    print(f"   ⏱  {avg:.1f} ms ±{std:.1f} ms  ({1000/avg:.1f} fps)  —  shape {input_shape}")
    return avg


def _write_meta(path: str, data: dict):
    meta = Path(path).with_suffix(Path(path).suffix + ".meta")
    with open(meta, "w") as f:
        json.dump(data, f, indent=2)
    print(f"   📄 Metadaten: {meta}")


# ─────────────────────────────────────────────────────────────────────────────
# TorchScript
# ─────────────────────────────────────────────────────────────────────────────

def convert_torchscript(model, output_path: str, input_shape: tuple, device: str) -> bool:
    """Konvertiert zu TorchScript via torch.jit.trace."""
    print(f"\n🔷 TorchScript — Input {input_shape}")

    print("📊 Baseline:")
    t_orig = _benchmark(model, device, input_shape)

    dummy = torch.randn(*input_shape).to(device)
    print("🔄 Tracing...")
    try:
        with torch.no_grad():
            scripted = torch.jit.trace(model, dummy)
        scripted = torch.jit.optimize_for_inference(scripted)
        print("✅ Tracing erfolgreich")
    except Exception as e:
        print(f"❌ Tracing fehlgeschlagen: {e}")
        traceback.print_exc()
        return False

    print("📊 TorchScript:")
    t_jit = _benchmark(scripted, device, input_shape)

    scripted.save(output_path)
    print(f"💾 Gespeichert: {output_path}")
    _write_meta(output_path, {
        "format": "torchscript",
        "input_shape": list(input_shape),
        "baseline_ms": round(t_orig, 2),
        "torchscript_ms": round(t_jit, 2),
        "speedup": round(t_orig / t_jit, 3),
    })
    print(f"🎉 Speedup: {t_orig / t_jit:.2f}×")
    return True


# ─────────────────────────────────────────────────────────────────────────────
# ONNX
# ─────────────────────────────────────────────────────────────────────────────

def convert_onnx(model, output_path: str, input_shape: tuple, device: str,
                 opset: int = 17) -> bool:
    """
    Exportiert das Modell nach ONNX.

    Input-Shape: (1, 7, 3, H, W)
    Batch-Dimension wird dynamisch exportiert.
    """
    try:
        import onnx
    except ImportError:
        print("❌ onnx fehlt — pip install onnx onnxruntime-gpu")
        return False

    print(f"\n🔷 ONNX Export — opset {opset}, Input {input_shape}")

    print("📊 Baseline:")
    t_orig = _benchmark(model, device, input_shape)

    dummy = torch.randn(*input_shape).to(device)
    print("🔄 Exportiere...")
    try:
        with torch.no_grad():
            torch.onnx.export(
                model,
                dummy,
                output_path,
                export_params=True,
                opset_version=opset,
                do_constant_folding=True,
                input_names=["frames"],
                output_names=["sr_frame"],
                dynamic_axes={
                    "frames":    {0: "batch"},
                    "sr_frame":  {0: "batch"},
                },
            )
    except Exception as e:
        print(f"❌ ONNX Export fehlgeschlagen: {e}")
        traceback.print_exc()
        return False

    # Validierung
    try:
        m = onnx.load(output_path)
        onnx.checker.check_model(m)
        print("✅ ONNX-Modell verifiziert")
    except Exception as e:
        print(f"⚠️  ONNX-Verifikation: {e}")

    print(f"💾 Gespeichert: {output_path}")
    _write_meta(output_path, {
        "format": "onnx",
        "opset": opset,
        "input_shape": list(input_shape),
        "output_shape": [input_shape[0], 3, input_shape[3] * 3, input_shape[4] * 3],
        "baseline_ms": round(t_orig, 2),
    })
    return True


# ─────────────────────────────────────────────────────────────────────────────
# TensorRT  (PyTorch → ONNX → TRT Engine)
# ─────────────────────────────────────────────────────────────────────────────

def _build_trt_engine(onnx_path: str, engine_path: str,
                      precision: str, workspace_gb: int = 2,
                      input_shape: tuple = None) -> bool:
    """
    Baut eine TensorRT-Engine aus einer ONNX-Datei.

    Benötigt das 'tensorrt' PyPI-Paket (muss zur CUDA-Toolkit-Version passen).
    Für CUDA 12.0: pip install tensorrt==8.6.*  (letzte Version mit CC 6.0-Support)
    """
    try:
        import tensorrt as trt
    except ImportError:
        print("❌ tensorrt fehlt oder nicht importierbar!")
        print("   Alle pip-Wheels für TRT 8.6.x sind Stubs ohne gebündelte .so-Dateien.")
        print("   System-Installation erforderlich — eine der folgenden Optionen:")
        print("   Option 1 (apt):  sudo apt-get install libnvinfer8 libnvinfer-plugin8 libnvonnxparser8 python3-libnvinfer")
        print("   Option 2 (lokal): TensorRT-8.6.1.6.Linux.x86_64-gnu.cuda-12.0.tar.gz von")
        print("                     developer.nvidia.com herunterladen, dann das enthaltene Wheel installieren:")
        print("                     pip install tensorrt-8.6.1-cp311-none-linux_x86_64.whl")
        return False

    TRT_LOGGER = trt.Logger(trt.Logger.WARNING)
    network_flags = 1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)

    print(f"🔄 Baue TensorRT-Engine ({precision.upper()})...")
    print(f"   TensorRT-Version: {trt.__version__}")
    print(f"   Workspace: {workspace_gb} GB")

    try:
        with trt.Builder(TRT_LOGGER) as builder, \
             builder.create_network(network_flags) as network, \
             trt.OnnxParser(network, TRT_LOGGER) as parser, \
             builder.create_builder_config() as config:

            # Workspace
            config.set_memory_pool_limit(
                trt.MemoryPoolType.WORKSPACE, workspace_gb << 30
            )

            # Precision
            if precision == "fp16":
                if not builder.platform_has_fast_fp16:
                    print("   ⚠️  GPU meldet kein 'fast_fp16' — Engine wird trotzdem mit FP16 gebaut")
                config.set_flag(trt.BuilderFlag.FP16)
                print("   ✅ FP16 aktiviert")

            # ONNX einlesen
            with open(onnx_path, "rb") as f:
                raw = f.read()
            if not parser.parse(raw):
                print("❌ ONNX-Parser Fehler:")
                for i in range(parser.num_errors):
                    print(f"   [{i}] {parser.get_error(i)}")
                return False

            print(f"   ONNX gelesen — {network.num_layers} Layer")

            # Optimization Profile — immer erforderlich wenn dynamic_axes im ONNX gesetzt sind
            # Echten TRT-Input-Namen aus dem geparsten Netzwerk lesen (nicht "frames" hardcoden)
            profile = builder.create_optimization_profile()
            inp_tensor = network.get_input(0)
            inp_name = inp_tensor.name
            fixed = tuple(input_shape) if input_shape is not None else tuple(
                abs(d) for d in inp_tensor.shape)
            ok = profile.set_shape(inp_name, min=fixed, opt=fixed, max=fixed)
            # TRT 8.6.x: set_shape() gibt None zurück (C++ void → Python None),
            # kein bool. Nur explizites False als Fehler werten.
            if ok is False:
                print(f"   ❌ set_shape für '{inp_name}' fehlgeschlagen — shape={fixed}")
                return False
            idx = config.add_optimization_profile(profile)
            if idx < 0:
                print(f"   ❌ add_optimization_profile fehlgeschlagen (idx={idx})")
                return False
            print(f"   ✅ Optimization Profile gesetzt [{inp_name}]: {fixed}")

            # Engine bauen (kann mehrere Minuten dauern)
            print("   ⏳ Kompiliere Engine (kann einige Minuten dauern)...")
            serialized = builder.build_serialized_network(network, config)
            if serialized is None:
                print("❌ build_serialized_network gab None zurück")
                return False

            with open(engine_path, "wb") as f:
                f.write(serialized)

            size_mb = Path(engine_path).stat().st_size / (1024 * 1024)
            print(f"✅ Engine gebaut — {size_mb:.1f} MB")
            return True

    except Exception as e:
        print(f"❌ TensorRT-Engine-Bau fehlgeschlagen: {e}")
        traceback.print_exc()
        return False


def _benchmark_trt_engine(engine_path: str, input_shape: tuple, device: str,
                           n: int = 20) -> float:
    """Benchmarkt eine gespeicherte TRT-Engine über IExecutionContext."""
    try:
        import tensorrt as trt
        import pycuda.driver as cuda
        import pycuda.autoinit  # noqa: F401
    except ImportError:
        print("   ⚠️  pycuda nicht verfügbar — TRT-Benchmark übersprungen")
        return 0.0

    TRT_LOGGER = trt.Logger(trt.Logger.WARNING)
    runtime = trt.Runtime(TRT_LOGGER)
    with open(engine_path, "rb") as f:
        engine = runtime.deserialize_cuda_engine(f.read())
    context = engine.create_execution_context()

    dummy = np.random.randn(*input_shape).astype(np.float32)
    out_shape = (input_shape[0], 3, input_shape[3] * 3, input_shape[4] * 3)
    out = np.empty(out_shape, dtype=np.float32)

    d_in  = cuda.mem_alloc(dummy.nbytes)
    d_out = cuda.mem_alloc(out.nbytes)

    # Warmup
    for _ in range(5):
        cuda.memcpy_htod(d_in, dummy)
        context.execute_v2([int(d_in), int(d_out)])
        cuda.memcpy_dtoh(out, d_out)

    times = []
    for _ in range(n):
        cuda.memcpy_htod(d_in, dummy)
        t0 = time.perf_counter()
        context.execute_v2([int(d_in), int(d_out)])
        cuda.memcpy_dtoh(out, d_out)
        times.append((time.perf_counter() - t0) * 1000)

    avg = float(np.mean(times))
    print(f"   ⏱  TRT Engine: {avg:.1f} ms ±{np.std(times):.1f} ms  ({1000/avg:.1f} fps)")
    return avg


def convert_tensorrt(model, output_path: str, input_shape: tuple, device: str,
                     precision: str = "fp16", workspace_gb: int = 2) -> bool:
    """
    Konvertiert das VSR++-Modell zu einer TensorRT-Engine.

    Ablauf:
      1. ONNX-Export in eine temporäre Datei
      2. TRT-Engine aus ONNX bauen
      3. Benchmark (falls pycuda verfügbar)
    """
    if device != "cuda":
        print("❌ TensorRT benötigt CUDA — bitte --device cuda verwenden")
        return False

    print(f"\n🔷 TensorRT ({precision.upper()}) — Input {input_shape}")

    print("📊 Baseline PyTorch:")
    t_orig = _benchmark(model, device, input_shape)

    # Schritt 1: ONNX temporär
    with tempfile.NamedTemporaryFile(suffix=".onnx", delete=False) as tmp:
        tmp_onnx = tmp.name

    try:
        print("🔄 Schritt 1/2 — ONNX Export...")
        ok = convert_onnx(model, tmp_onnx, input_shape, device, opset=17)
        if not ok:
            return False

        # Schritt 2: TRT Engine
        print("🔄 Schritt 2/2 — TensorRT Engine bauen...")
        ok = _build_trt_engine(tmp_onnx, output_path, precision, workspace_gb,
                               input_shape=input_shape)
        if not ok:
            return False

    finally:
        if os.path.exists(tmp_onnx):
            os.unlink(tmp_onnx)
        meta_tmp = tmp_onnx + ".meta"
        if os.path.exists(meta_tmp):
            os.unlink(meta_tmp)

    print(f"💾 Engine gespeichert: {output_path}")

    # Benchmark (optional, braucht pycuda)
    t_trt = _benchmark_trt_engine(output_path, input_shape, device)

    meta: dict = {
        "format": "tensorrt",
        "precision": precision,
        "input_shape": list(input_shape),
        "output_shape": [input_shape[0], 3, input_shape[3] * 3, input_shape[4] * 3],
        "baseline_ms": round(t_orig, 2),
        "workspace_gb": workspace_gb,
    }
    if t_trt > 0:
        meta["trt_ms"] = round(t_trt, 2)
        meta["speedup"] = round(t_orig / t_trt, 3)
        print(f"🎉 Speedup: {t_orig / t_trt:.2f}×")
    _write_meta(output_path, meta)
    return True


# ─────────────────────────────────────────────────────────────────────────────
# Pruning
# ─────────────────────────────────────────────────────────────────────────────

def convert_pruned(model, checkpoint_path: str, output_path: str,
                   prune_amount: float, prune_type: str,
                   input_shape: tuple, device: str) -> bool:
    """
    Pruning — entfernt unwichtige Conv2d-Weights.

    prune_type 'structured'   → ganze Ausgangskanäle (echt kleiner nach Re-Training)
    prune_type 'unstructured' → einzelne Weights auf 0 (Sparsity, kein Größenvorteil)
    """
    import torch.nn.utils.prune as prune_utils

    print(f"\n🔷 Pruning ({prune_type}, {prune_amount*100:.0f}%) — Input {input_shape}")

    print("📊 Baseline:")
    t_orig = _benchmark(model, device, input_shape)

    total_params = sum(p.numel() for p in model.parameters())
    print(f"   Parameter gesamt: {total_params:,}")

    conv_layers = [
        (m, "weight")
        for m in model.modules()
        if isinstance(m, torch.nn.Conv2d)
    ]
    print(f"   Conv2d-Layer: {len(conv_layers)}")

    try:
        if prune_type == "structured":
            for m, name in conv_layers:
                prune_utils.ln_structured(m, name=name, amount=prune_amount, n=2, dim=0)
        elif prune_type == "unstructured":
            for m, name in conv_layers:
                prune_utils.l1_unstructured(m, name=name, amount=prune_amount)
        else:
            raise ValueError(f"Unbekannter prune_type: {prune_type!r}")
    except Exception as e:
        print(f"❌ Pruning fehlgeschlagen: {e}")
        traceback.print_exc()
        return False

    # Null-Anteil messen
    zero = sum(
        (getattr(m, name + "_mask") == 0).sum().item()
        for m, name in conv_layers
        if hasattr(m, name + "_mask")
    )
    total_conv = sum(
        getattr(m, name + "_mask").numel()
        for m, name in conv_layers
        if hasattr(m, name + "_mask")
    )
    ratio = zero / total_conv if total_conv > 0 else 0.0
    print(f"✅ Gepruned: {ratio*100:.1f}% der Conv-Weights ({zero:,}/{total_conv:,} auf 0)")

    # Masken permanent machen
    for m, name in conv_layers:
        try:
            prune_utils.remove(m, name)
        except Exception:
            pass

    print("📊 Nach Pruning:")
    t_pruned = _benchmark(model, device, input_shape)

    # Checkpoint mit Metadaten speichern
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    ckpt["model_state_dict"]  = model.state_dict()
    ckpt["pruned"]            = True
    ckpt["prune_amount"]      = prune_amount
    ckpt["prune_type"]        = prune_type
    ckpt["actual_prune_ratio"] = ratio

    torch.save(ckpt, output_path)
    print(f"💾 Gespeichert: {output_path}")

    meta = {
        "format": "pruned",
        "prune_type": prune_type,
        "prune_amount": prune_amount,
        "actual_ratio": round(ratio, 4),
        "input_shape": list(input_shape),
        "baseline_ms": round(t_orig, 2),
        "pruned_ms": round(t_pruned, 2),
        "speedup": round(t_orig / t_pruned, 3) if t_pruned > 0 else None,
        "total_params": total_params,
        "zero_conv_weights": zero,
        "total_conv_weights": total_conv,
    }
    _write_meta(output_path, meta)

    if prune_type == "unstructured":
        print("⚠️  Hinweis: Unstrukturiertes Pruning erzeugt Sparsity, aber keine echte "
              "Modell-Größenreduktion.\n"
              "   Für kleinere Modelle: --prune-type structured verwenden.")
    return True


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="VSR++ Modell-Konvertierung (TensorRT / ONNX / TorchScript / Pruning)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Beispiele:
  # TensorRT FP16 bei 480×270 LR-Auflösung
  python optimize_checkpoint.py -c model.pth -o model_fp16.engine -f tensorrt -p fp16 --width 480 --height 270

  # ONNX
  python optimize_checkpoint.py -c model.pth -o model.onnx -f onnx --width 480 --height 270

  # TorchScript
  python optimize_checkpoint.py -c model.pth -o model.pt -f torchscript

  # Pruning strukturiert 30%
  python optimize_checkpoint.py -c model.pth -o model_pruned.pth -f pruned --prune-amount 0.3
        """,
    )
    parser.add_argument("--checkpoint", "-c", required=True,
                        help="Pfad zum .pth-Checkpoint")
    parser.add_argument("--output", "-o", required=True,
                        help="Ausgabe-Pfad (Endung: .engine / .onnx / .pt / .pth)")
    parser.add_argument("--format", "-f",
                        choices=["tensorrt", "onnx", "torchscript", "pruned"],
                        default="tensorrt",
                        help="Konvertierungsformat (Standard: tensorrt)")
    parser.add_argument("--precision", "-p", choices=["fp16", "fp32"],
                        default="fp16",
                        help="TensorRT Precision (Standard: fp16)")
    parser.add_argument("--width", type=int, default=480,
                        help="LR-Eingabebreite in Pixeln (Standard: 480)")
    parser.add_argument("--height", type=int, default=270,
                        help="LR-Eingabehöhe in Pixeln (Standard: 270)")
    parser.add_argument("--prune-amount", type=float, default=0.3,
                        help="Pruning-Anteil 0.0–1.0 (Standard: 0.3)")
    parser.add_argument("--prune-type", choices=["structured", "unstructured"],
                        default="structured",
                        help="Pruning-Typ (Standard: structured)")
    parser.add_argument("--workspace-gb", type=int, default=2,
                        help="TensorRT Workspace-Größe in GB (Standard: 2)")
    parser.add_argument("--device", "-d", choices=["cuda", "cpu"],
                        default="cuda" if torch.cuda.is_available() else "cpu",
                        help="Device (Standard: cuda falls verfügbar)")
    parser.add_argument("--benchmark-iters", type=int, default=20,
                        help="Anzahl Benchmark-Iterationen (Standard: 20)")

    args = parser.parse_args()
    input_shape = (1, 7, 3, args.height, args.width)

    print("=" * 68)
    print("🚀  VSR++ Modell-Konvertierung")
    print("=" * 68)
    print(f"  Checkpoint  : {args.checkpoint}")
    print(f"  Ausgabe     : {args.output}")
    print(f"  Format      : {args.format}")
    print(f"  Input-Shape : {input_shape}  →  SR {args.height*3}×{args.width*3}")
    if args.format == "tensorrt":
        print(f"  Precision   : {args.precision.upper()}")
        print(f"  Workspace   : {args.workspace_gb} GB")
    if args.format == "pruned":
        print(f"  Prune       : {args.prune_amount*100:.0f}% {args.prune_type}")
    print(f"  Device      : {args.device}")
    print("=" * 68)

    # Validierung
    if not Path(args.checkpoint).exists():
        print(f"❌ Checkpoint nicht gefunden: {args.checkpoint}")
        return 1

    if args.device == "cuda" and not torch.cuda.is_available():
        print("⚠️  CUDA nicht verfügbar — wechsle auf CPU")
        args.device = "cpu"

    if args.format == "tensorrt" and args.device != "cuda":
        print("❌ TensorRT benötigt CUDA")
        return 1

    if args.format == "pruned" and not (0.0 < args.prune_amount < 1.0):
        print(f"❌ --prune-amount muss zwischen 0 und 1 liegen (ist: {args.prune_amount})")
        return 1

    # Modell laden
    try:
        model, info = load_model(args.checkpoint, args.device)
    except Exception as e:
        print(f"❌ Modell laden fehlgeschlagen: {e}")
        traceback.print_exc()
        return 1

    # GPU-Info + P100/CC-6.0-Warnungen ausgeben
    if args.device == "cuda":
        cc = torch.cuda.get_device_capability(0)
        gpu_name = torch.cuda.get_device_name(0)
        print(f"\n  GPU : {gpu_name}  (CC {cc[0]}.{cc[1]})")
        if cc < (7, 0):
            print("  ⚠️  CC < 7.0 — torch.compile/Triton nicht verfügbar (nicht benötigt)")
            if args.format == "tensorrt":
                print("  ⚠️  CC 6.0: TensorRT ≥ 9.0 unterstützt diese GPU NICHT!")
                print("       Benötigt: pip install 'tensorrt==8.6.*'")
                print("       Kompatible Versionen: tensorrt 8.5.x / 8.6.x")
        print()

    # Konvertierung starten
    success = False

    if args.format == "tensorrt":
        # Zusätzlicher Vorab-Check: tensorrt-Version für CC 6.0 validieren
        try:
            import tensorrt as trt
            trt_major = int(trt.__version__.split(".")[0])
            if trt_major >= 9 and args.device == "cuda":
                cc = torch.cuda.get_device_capability(0)
                if cc < (7, 0):
                    print(f"❌ tensorrt {trt.__version__} unterstützt CC {cc[0]}.{cc[1]} nicht!")
                    print("   TensorRT 9.x erfordert CC ≥ 7.0.")
                    print("   Lösung: pip install 'tensorrt==8.6.*'")
                    return 1
        except ImportError:
            pass  # Fehlermeldung kommt aus convert_tensorrt

        success = convert_tensorrt(
            model, args.output, input_shape, args.device,
            precision=args.precision,
            workspace_gb=args.workspace_gb,
        )

    elif args.format == "torchscript":
        success = convert_torchscript(model, args.output, input_shape, args.device)

    elif args.format == "onnx":
        success = convert_onnx(model, args.output, input_shape, args.device)

    elif args.format == "pruned":
        success = convert_pruned(
            model, args.checkpoint, args.output,
            prune_amount=args.prune_amount,
            prune_type=args.prune_type,
            input_shape=input_shape,
            device=args.device,
        )

    # Zusammenfassung
    print("\n" + "=" * 68)
    if success:
        print(f"✅ Konvertierung erfolgreich!")
        print(f"   Ausgabe : {args.output}")
        meta = Path(args.output).with_suffix(Path(args.output).suffix + ".meta")
        if meta.exists():
            print(f"   Metadaten: {meta}")
        print("=" * 68)
        return 0
    else:
        print("❌ Konvertierung fehlgeschlagen — siehe Ausgabe oben.")
        print("=" * 68)
        return 1


if __name__ == "__main__":
    sys.exit(main())

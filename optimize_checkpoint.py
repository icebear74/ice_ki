#!/usr/bin/env python3
"""
Model Optimization Script - Konvertiert PyTorch Checkpoint zu optimierten Formaten

Unterstützt mehrere Optimierungs-Optionen:
1. TensorRT (FP32/FP16) - Beste Performance auf NVIDIA GPUs
2. TorchScript - PyTorch JIT Compiler (portabel, gute Performance)
3. ONNX - Offenes Format (portabel)
4. Pruning - Entfernt unwichtige Weights für kleineres/schnelleres Modell

Verwendung:
    # TensorRT FP16 (empfohlen für beste Performance)
    python optimize_checkpoint.py --checkpoint model.pth --output model_trt_fp16.engine --format tensorrt --precision fp16
    
    # TensorRT FP32
    python optimize_checkpoint.py --checkpoint model.pth --output model_trt_fp32.engine --format tensorrt --precision fp32
    
    # TorchScript
    python optimize_checkpoint.py --checkpoint model.pth --output model_scripted.pt --format torchscript
    
    # ONNX
    python optimize_checkpoint.py --checkpoint model.pth --output model.onnx --format onnx
    
    # Pruning (strukturiert, 30% der Kanäle entfernen)
    python optimize_checkpoint.py --checkpoint model.pth --output model_pruned.pth --format pruned --prune-amount 0.3 --prune-type structured
    
    # Pruning (unstrukturiert, 50% der Weights entfernen)
    python optimize_checkpoint.py --checkpoint model.pth --output model_pruned.pth --format pruned --prune-amount 0.5 --prune-type unstructured

Installations-Voraussetzungen:
    # Basis
    pip install torch torchvision
    
    # Für TensorRT (NVIDIA GPU erforderlich)
    pip install torch2trt
    # Oder: https://developer.nvidia.com/tensorrt
    
    # Für ONNX
    pip install onnx onnxruntime-gpu
    
    # Pruning ist in PyTorch eingebaut (torch.nn.utils.prune)
"""

import argparse
import os
import sys
import time
import torch
import numpy as np

# Add vsr_plusplus_NEU to path
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), 'vsr_plusplus_NEU'))


def load_pytorch_model(checkpoint_path, device='cuda'):
    """
    Lädt das PyTorch Modell aus einem Checkpoint
    
    Args:
        checkpoint_path: Pfad zum Checkpoint (.pth Datei)
        device: Device für Modell
        
    Returns:
        model: Geladenes PyTorch Modell im eval() Modus
        checkpoint_info: Checkpoint-Informationen
    """
    print(f"📦 Lade PyTorch Checkpoint: {checkpoint_path}")
    
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint nicht gefunden: {checkpoint_path}")
    
    # Checkpoint laden
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    
    # Modell-Konfiguration laden
    n_feats = 72
    n_blocks = 28
    
    try:
        import config as cfg
        config = cfg.get_config()
        n_feats = config.get('N_FEATS', 72)
        n_blocks = config.get('N_BLOCKS', 28)
        print(f"   ✅ Config geladen: n_feats={n_feats}, n_blocks={n_blocks}")
    except:
        print(f"   ℹ️  Verwende Standard-Config: n_feats={n_feats}, n_blocks={n_blocks}")
    
    # Modell erstellen
    from vsr_plusplus_NEU.core.model_7frame import VSRBidirectional_7frames_3x
    model = VSRBidirectional_7frames_3x(
        n_feats=n_feats,
        n_blocks=n_blocks
    ).to(device)
    
    # Weights laden
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    checkpoint_info = {
        'step': checkpoint.get('step', 'unknown'),
        'epoch': checkpoint.get('epoch', 'unknown'),
        'n_feats': n_feats,
        'n_blocks': n_blocks
    }
    
    print(f"✅ Modell geladen (Step: {checkpoint_info['step']}, Epoch: {checkpoint_info['epoch']})")
    
    return model, checkpoint_info


def benchmark_model(model, device='cuda', input_shape=(1, 7, 3, 180, 180), iterations=10):
    """
    Benchmark-Test für Modell-Performance
    
    Args:
        model: Zu testendes Modell
        device: Device
        input_shape: Input-Tensor Shape
        iterations: Anzahl Durchläufe
        
    Returns:
        avg_time: Durchschnittliche Inferenz-Zeit in ms
    """
    print(f"\n⏱️  Benchmark mit {iterations} Iterationen...")
    print(f"   Input Shape: {input_shape}")
    
    # Dummy Input erstellen
    dummy_input = torch.randn(*input_shape).to(device)
    
    # Warmup
    with torch.no_grad():
        for _ in range(3):
            _ = model(dummy_input)
    
    # Benchmark
    times = []
    with torch.no_grad():
        for i in range(iterations):
            if device == 'cuda':
                torch.cuda.synchronize()
            
            start = time.time()
            _ = model(dummy_input)
            
            if device == 'cuda':
                torch.cuda.synchronize()
            
            end = time.time()
            times.append((end - start) * 1000)  # Convert to ms
    
    avg_time = np.mean(times)
    std_time = np.std(times)
    
    print(f"   ⏱️  Durchschnitt: {avg_time:.2f} ms (±{std_time:.2f} ms)")
    print(f"   ⏱️  FPS: {1000/avg_time:.2f}")
    
    return avg_time


def optimize_tensorrt(model, output_path, precision='fp16', input_shape=(1, 7, 3, 180, 180), device='cuda'):
    """
    Konvertiert zu TensorRT Engine
    
    Args:
        model: PyTorch Modell
        output_path: Ausgabe-Pfad für .engine Datei
        precision: 'fp32' oder 'fp16'
        input_shape: Input Shape für Engine
        device: Device
    """
    try:
        from torch2trt import torch2trt
    except ImportError:
        print("❌ torch2trt nicht installiert!")
        print("   Installation: pip install torch2trt")
        print("   Oder von: https://github.com/NVIDIA-AI-IOT/torch2trt")
        return False
    
    print(f"\n🚀 Konvertiere zu TensorRT ({precision.upper()})...")
    print(f"   Input Shape: {input_shape}")
    
    # Benchmark Original
    print("\n📊 Original PyTorch Modell:")
    original_time = benchmark_model(model, device, input_shape)
    
    # Erstelle Dummy Input
    dummy_input = torch.randn(*input_shape).to(device)
    
    # Konvertiere zu TensorRT
    print(f"\n🔄 TensorRT Konvertierung...")
    
    fp16_mode = (precision == 'fp16')
    
    try:
        model_trt = torch2trt(
            model,
            [dummy_input],
            fp16_mode=fp16_mode,
            max_workspace_size=1 << 30,  # 1GB
            log_level=torch2trt.trt.Logger.INFO
        )
        
        print(f"✅ TensorRT Konvertierung erfolgreich!")
        
        # Benchmark TensorRT
        print(f"\n📊 TensorRT {precision.upper()} Modell:")
        trt_time = benchmark_model(model_trt, device, input_shape)
        
        # Speedup
        speedup = original_time / trt_time
        print(f"\n🎉 Speedup: {speedup:.2f}x schneller!")
        
        # Speichern
        torch.save(model_trt.state_dict(), output_path)
        print(f"💾 TensorRT Engine gespeichert: {output_path}")
        
        # Metadaten speichern
        meta_path = output_path + '.meta'
        with open(meta_path, 'w') as f:
            f.write(f"precision: {precision}\n")
            f.write(f"input_shape: {input_shape}\n")
            f.write(f"original_time_ms: {original_time:.2f}\n")
            f.write(f"trt_time_ms: {trt_time:.2f}\n")
            f.write(f"speedup: {speedup:.2f}x\n")
        
        print(f"💾 Metadaten gespeichert: {meta_path}")
        
        return True
        
    except Exception as e:
        print(f"❌ TensorRT Konvertierung fehlgeschlagen: {e}")
        import traceback
        traceback.print_exc()
        return False


def optimize_torchscript(model, output_path, input_shape=(1, 7, 3, 180, 180), device='cuda'):
    """
    Konvertiert zu TorchScript
    
    Args:
        model: PyTorch Modell
        output_path: Ausgabe-Pfad für .pt Datei
        input_shape: Input Shape
        device: Device
    """
    print(f"\n🚀 Konvertiere zu TorchScript...")
    print(f"   Input Shape: {input_shape}")
    
    # Benchmark Original
    print("\n📊 Original PyTorch Modell:")
    original_time = benchmark_model(model, device, input_shape)
    
    # Erstelle Dummy Input
    dummy_input = torch.randn(*input_shape).to(device)
    
    # Trace Modell
    print(f"\n🔄 TorchScript Tracing...")
    
    try:
        model_scripted = torch.jit.trace(model, dummy_input)
        
        # Optimierungen
        model_scripted = torch.jit.optimize_for_inference(model_scripted)
        
        print(f"✅ TorchScript Konvertierung erfolgreich!")
        
        # Benchmark TorchScript
        print(f"\n📊 TorchScript Modell:")
        scripted_time = benchmark_model(model_scripted, device, input_shape)
        
        # Speedup
        speedup = original_time / scripted_time
        print(f"\n🎉 Speedup: {speedup:.2f}x schneller!")
        
        # Speichern
        model_scripted.save(output_path)
        print(f"💾 TorchScript Modell gespeichert: {output_path}")
        
        # Metadaten
        meta_path = output_path + '.meta'
        with open(meta_path, 'w') as f:
            f.write(f"format: torchscript\n")
            f.write(f"input_shape: {input_shape}\n")
            f.write(f"original_time_ms: {original_time:.2f}\n")
            f.write(f"scripted_time_ms: {scripted_time:.2f}\n")
            f.write(f"speedup: {speedup:.2f}x\n")
        
        print(f"💾 Metadaten gespeichert: {meta_path}")
        
        return True
        
    except Exception as e:
        print(f"❌ TorchScript Konvertierung fehlgeschlagen: {e}")
        import traceback
        traceback.print_exc()
        return False


def optimize_onnx(model, output_path, input_shape=(1, 7, 3, 180, 180), device='cuda'):
    """
    Konvertiert zu ONNX
    
    Args:
        model: PyTorch Modell
        output_path: Ausgabe-Pfad für .onnx Datei
        input_shape: Input Shape
        device: Device
    """
    try:
        import onnx
    except ImportError:
        print("❌ onnx nicht installiert!")
        print("   Installation: pip install onnx onnxruntime-gpu")
        return False
    
    print(f"\n🚀 Konvertiere zu ONNX...")
    print(f"   Input Shape: {input_shape}")
    
    # Benchmark Original
    print("\n📊 Original PyTorch Modell:")
    original_time = benchmark_model(model, device, input_shape)
    
    # Erstelle Dummy Input
    dummy_input = torch.randn(*input_shape).to(device)
    
    # Export zu ONNX
    print(f"\n🔄 ONNX Export...")
    
    try:
        torch.onnx.export(
            model,
            dummy_input,
            output_path,
            export_params=True,
            opset_version=17,
            do_constant_folding=True,
            input_names=['input'],
            output_names=['output'],
            dynamic_axes={
                'input': {0: 'batch_size'},
                'output': {0: 'batch_size'}
            }
        )
        
        print(f"✅ ONNX Export erfolgreich!")
        
        # Verifiziere ONNX Modell
        onnx_model = onnx.load(output_path)
        onnx.checker.check_model(onnx_model)
        print(f"✅ ONNX Modell verifiziert!")
        
        print(f"💾 ONNX Modell gespeichert: {output_path}")
        
        # Metadaten
        meta_path = output_path + '.meta'
        with open(meta_path, 'w') as f:
            f.write(f"format: onnx\n")
            f.write(f"input_shape: {input_shape}\n")
            f.write(f"original_time_ms: {original_time:.2f}\n")
            f.write(f"opset_version: 17\n")
        
        print(f"💾 Metadaten gespeichert: {meta_path}")
        
        return True
        
    except Exception as e:
        print(f"❌ ONNX Export fehlgeschlagen: {e}")
        import traceback
        traceback.print_exc()
        return False


def optimize_pruned(model, checkpoint_path, output_path, prune_amount=0.3, prune_type='structured', 
                    input_shape=(1, 7, 3, 180, 180), device='cuda'):
    """
    Pruning - Entfernt unwichtige Weights/Kanäle
    
    Args:
        model: PyTorch Modell
        checkpoint_path: Original Checkpoint (für metadata)
        output_path: Ausgabe-Pfad für .pth Datei
        prune_amount: Anteil zu entfernender Weights/Kanäle (0.0-1.0)
        prune_type: 'structured' (ganze Kanäle) oder 'unstructured' (einzelne Weights)
        input_shape: Input Shape
        device: Device
    """
    import torch.nn.utils.prune as prune
    
    print(f"\n🚀 Model Pruning ({prune_type})...")
    print(f"   Prune Amount: {prune_amount*100:.1f}%")
    print(f"   Input Shape: {input_shape}")
    
    # Benchmark Original
    print("\n📊 Original Modell:")
    original_time = benchmark_model(model, device, input_shape)
    
    # Original Modell-Größe
    original_size = sum(p.numel() for p in model.parameters())
    print(f"   Parameter: {original_size:,}")
    
    # Pruning anwenden
    print(f"\n🔄 Wende Pruning an...")
    
    modules_to_prune = []
    
    # Sammle alle Conv2d Layer
    for name, module in model.named_modules():
        if isinstance(module, torch.nn.Conv2d):
            modules_to_prune.append((module, 'weight'))
    
    print(f"   Gefunden: {len(modules_to_prune)} Conv2d Layer")
    
    try:
        if prune_type == 'structured':
            # Strukturiertes Pruning - entfernt ganze Kanäle
            for module, param_name in modules_to_prune:
                # Prune output channels (dim=0)
                prune.ln_structured(
                    module, 
                    name=param_name,
                    amount=prune_amount,
                    n=2,  # L2 norm
                    dim=0  # Output channels
                )
        
        elif prune_type == 'unstructured':
            # Unstrukturiertes Pruning - entfernt einzelne Weights
            for module, param_name in modules_to_prune:
                prune.l1_unstructured(
                    module,
                    name=param_name,
                    amount=prune_amount
                )
        
        else:
            raise ValueError(f"Unbekannter Prune-Typ: {prune_type}")
        
        print(f"✅ Pruning erfolgreich angewendet!")
        
        # Zähle tatsächlich geprunte Parameter
        zero_params = 0
        total_params = 0
        
        for module, param_name in modules_to_prune:
            # Zugriff auf die Maske
            if hasattr(module, param_name + '_mask'):
                mask = getattr(module, param_name + '_mask')
                zero_params += (mask == 0).sum().item()
                total_params += mask.numel()
        
        actual_prune_ratio = zero_params / total_params if total_params > 0 else 0
        print(f"   Tatsächlich gepruned: {actual_prune_ratio*100:.1f}% der Conv-Weights")
        print(f"   Null-Parameter: {zero_params:,} von {total_params:,}")
        
        # Pruning permanent machen (entfernt Masken, setzt Weights auf 0)
        print(f"\n🔄 Mache Pruning permanent...")
        for module, param_name in modules_to_prune:
            prune.remove(module, param_name)
        
        print(f"✅ Pruning permanent gemacht!")
        
        # Benchmark nach Pruning
        print(f"\n📊 Gepruntes Modell:")
        pruned_time = benchmark_model(model, device, input_shape)
        
        # Statistiken
        speedup = original_time / pruned_time
        print(f"\n📈 Pruning-Statistiken:")
        print(f"   Speedup: {speedup:.2f}x schneller")
        print(f"   Kompression: {actual_prune_ratio*100:.1f}% Weights entfernt")
        
        # Modell speichern
        print(f"\n💾 Speichere gepruntes Modell...")
        
        # Lade original checkpoint für metadata
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
        
        # Update model state dict
        checkpoint['model_state_dict'] = model.state_dict()
        checkpoint['pruned'] = True
        checkpoint['prune_amount'] = prune_amount
        checkpoint['prune_type'] = prune_type
        checkpoint['actual_prune_ratio'] = actual_prune_ratio
        
        # Speichern
        torch.save(checkpoint, output_path)
        print(f"💾 Gepruntes Modell gespeichert: {output_path}")
        
        # Metadaten
        meta_path = output_path + '.meta'
        with open(meta_path, 'w') as f:
            f.write(f"format: pruned\n")
            f.write(f"prune_type: {prune_type}\n")
            f.write(f"prune_amount: {prune_amount}\n")
            f.write(f"actual_prune_ratio: {actual_prune_ratio:.4f}\n")
            f.write(f"input_shape: {input_shape}\n")
            f.write(f"original_time_ms: {original_time:.2f}\n")
            f.write(f"pruned_time_ms: {pruned_time:.2f}\n")
            f.write(f"speedup: {speedup:.2f}x\n")
            f.write(f"zero_params: {zero_params}\n")
            f.write(f"total_conv_params: {total_params}\n")
        
        print(f"💾 Metadaten gespeichert: {meta_path}")
        
        # Warnung bei unstrukturiertem Pruning
        if prune_type == 'unstructured':
            print(f"\n⚠️  Hinweis: Unstrukturiertes Pruning setzt Weights auf 0,")
            print(f"    aber die Modell-Größe bleibt gleich. Für echte Kompression")
            print(f"    verwende strukturiertes Pruning (--prune-type structured)")
        
        return True
        
    except Exception as e:
        print(f"❌ Pruning fehlgeschlagen: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    parser = argparse.ArgumentParser(
        description='Optimiert PyTorch Checkpoint für schnellere Inferenz',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Beispiele:
  # TensorRT FP16 (empfohlen)
  python optimize_checkpoint.py --checkpoint model.pth --output model_trt_fp16.engine --format tensorrt --precision fp16
  
  # TorchScript
  python optimize_checkpoint.py --checkpoint model.pth --output model_scripted.pt --format torchscript
  
  # ONNX
  python optimize_checkpoint.py --checkpoint model.pth --output model.onnx --format onnx
  
  # Pruning (strukturiert, 30%)
  python optimize_checkpoint.py --checkpoint model.pth --output model_pruned.pth --format pruned --prune-amount 0.3

Installations-Voraussetzungen:
  TensorRT: pip install torch2trt
  ONNX: pip install onnx onnxruntime-gpu
  Pruning: In PyTorch eingebaut (torch.nn.utils.prune)
        """
    )
    
    parser.add_argument('--checkpoint', '-c', required=True,
                        help='Pfad zum PyTorch Checkpoint (.pth)')
    parser.add_argument('--output', '-o', required=True,
                        help='Pfad für optimiertes Modell')
    parser.add_argument('--format', '-f', choices=['tensorrt', 'torchscript', 'onnx', 'pruned'],
                        default='tensorrt',
                        help='Optimierungs-Format (Standard: tensorrt)')
    parser.add_argument('--precision', '-p', choices=['fp32', 'fp16'],
                        default='fp16',
                        help='Precision für TensorRT (Standard: fp16)')
    parser.add_argument('--prune-amount', type=float, default=0.3,
                        help='Pruning: Anteil zu entfernender Weights (0.0-1.0, Standard: 0.3)')
    parser.add_argument('--prune-type', choices=['structured', 'unstructured'],
                        default='structured',
                        help='Pruning-Typ (Standard: structured)')
    parser.add_argument('--device', '-d', choices=['cuda', 'cpu'],
                        default='cuda' if torch.cuda.is_available() else 'cpu',
                        help='Device (Standard: cuda falls verfügbar)')
    parser.add_argument('--input-size', type=int, default=180,
                        help='Input-Größe (Höhe/Breite) (Standard: 180)')
    
    args = parser.parse_args()
    
    # Header
    print("=" * 70)
    print("🚀 Model Optimization Tool")
    print("=" * 70)
    print(f"Checkpoint: {args.checkpoint}")
    print(f"Output: {args.output}")
    print(f"Format: {args.format}")
    if args.format == 'tensorrt':
        print(f"Precision: {args.precision}")
    elif args.format == 'pruned':
        print(f"Prune Amount: {args.prune_amount*100:.1f}%")
        print(f"Prune Type: {args.prune_type}")
    print(f"Device: {args.device}")
    print(f"Input Size: {args.input_size}x{args.input_size}")
    print("=" * 70)
    
    # Validierung
    if not os.path.exists(args.checkpoint):
        print(f"❌ Checkpoint nicht gefunden: {args.checkpoint}")
        return 1
    
    if args.device == 'cuda' and not torch.cuda.is_available():
        print("⚠️  CUDA nicht verfügbar, verwende CPU")
        args.device = 'cpu'
    
    if args.format == 'pruned' and not (0.0 < args.prune_amount < 1.0):
        print(f"❌ Prune-Amount muss zwischen 0.0 und 1.0 liegen, ist: {args.prune_amount}")
        return 1
    
    # Modell laden
    try:
        model, checkpoint_info = load_pytorch_model(args.checkpoint, args.device)
    except Exception as e:
        print(f"❌ Fehler beim Laden des Modells: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    # Input Shape
    input_shape = (1, 7, 3, args.input_size, args.input_size)
    
    # Optimierung
    success = False
    
    if args.format == 'tensorrt':
        success = optimize_tensorrt(model, args.output, args.precision, input_shape, args.device)
    elif args.format == 'torchscript':
        success = optimize_torchscript(model, args.output, input_shape, args.device)
    elif args.format == 'onnx':
        success = optimize_onnx(model, args.output, input_shape, args.device)
    elif args.format == 'pruned':
        success = optimize_pruned(model, args.checkpoint, args.output, args.prune_amount, 
                                 args.prune_type, input_shape, args.device)
    
    # Abschluss
    print("\n" + "=" * 70)
    if success:
        print("✅ Optimierung erfolgreich!")
        print("=" * 70)
        print(f"📄 Optimiertes Modell: {args.output}")
        print(f"📄 Metadaten: {args.output}.meta")
        print("\nNächster Schritt:")
        print(f"  python run_video_inference_optimized.py --model {args.output} --input video.mkv --output result.mkv")
        print("=" * 70)
        return 0
    else:
        print("❌ Optimierung fehlgeschlagen!")
        print("=" * 70)
        return 1


if __name__ == '__main__':
    sys.exit(main())

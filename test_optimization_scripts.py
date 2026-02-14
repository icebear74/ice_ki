#!/usr/bin/env python3
"""
Test für Optimierungs-Scripts

Testet die Funktionalität ohne echtes Modell
"""

import sys
import os

def test_imports():
    """Test dass alle benötigten Module importierbar sind"""
    print("Testing imports...")
    
    try:
        import torch
        print(f"  ✅ PyTorch {torch.__version__}")
    except ImportError as e:
        print(f"  ❌ PyTorch nicht gefunden: {e}")
        return False
    
    try:
        import numpy
        print(f"  ✅ NumPy")
    except ImportError:
        print(f"  ❌ NumPy nicht gefunden")
        return False
    
    try:
        import cv2
        print(f"  ✅ OpenCV")
    except ImportError:
        print(f"  ❌ OpenCV nicht gefunden")
        return False
    
    # Optional dependencies
    try:
        from torch2trt import torch2trt
        print(f"  ✅ torch2trt (optional)")
    except ImportError:
        print(f"  ⚠️  torch2trt nicht installiert (optional für TensorRT)")
    
    try:
        import onnx
        print(f"  ✅ ONNX (optional)")
    except ImportError:
        print(f"  ⚠️  ONNX nicht installiert (optional für ONNX export)")
    
    try:
        import onnxruntime
        print(f"  ✅ ONNX Runtime (optional)")
    except ImportError:
        print(f"  ⚠️  ONNX Runtime nicht installiert (optional für ONNX)")
    
    return True


def test_script_syntax():
    """Test dass Scripts syntaktisch korrekt sind"""
    print("\nTesting script syntax...")
    
    scripts = [
        'optimize_checkpoint.py',
        'run_video_inference_optimized.py'
    ]
    
    import py_compile
    
    for script in scripts:
        try:
            py_compile.compile(script, doraise=True)
            print(f"  ✅ {script}")
        except py_compile.PyCompileError as e:
            print(f"  ❌ {script}: {e}")
            return False
    
    return True


def test_help_output():
    """Test dass Help-Output funktioniert"""
    print("\nTesting help output...")
    
    import subprocess
    
    scripts = [
        ('optimize_checkpoint.py', '--help'),
        ('run_video_inference_optimized.py', '--help')
    ]
    
    for script, arg in scripts:
        try:
            result = subprocess.run(
                ['python3', script, arg],
                capture_output=True,
                timeout=5
            )
            if result.returncode == 0:
                print(f"  ✅ {script} {arg}")
            else:
                print(f"  ❌ {script} {arg}: exit code {result.returncode}")
                return False
        except subprocess.TimeoutExpired:
            print(f"  ❌ {script} {arg}: timeout")
            return False
        except Exception as e:
            print(f"  ❌ {script} {arg}: {e}")
            return False
    
    return True


def test_pruning_availability():
    """Test dass Pruning verfügbar ist"""
    print("\nTesting pruning availability...")
    
    try:
        import torch.nn.utils.prune as prune
        print(f"  ✅ torch.nn.utils.prune verfügbar")
        
        # Test simple pruning
        import torch.nn as nn
        conv = nn.Conv2d(3, 64, 3)
        prune.l1_unstructured(conv, name='weight', amount=0.3)
        print(f"  ✅ Pruning funktioniert")
        
        return True
    except Exception as e:
        print(f"  ❌ Pruning nicht verfügbar: {e}")
        return False


def test_documentation_exists():
    """Test dass Dokumentation existiert"""
    print("\nTesting documentation...")
    
    docs = [
        'OPTIMIERUNG_ANLEITUNG_DE.md'
    ]
    
    for doc in docs:
        if os.path.exists(doc):
            size = os.path.getsize(doc)
            print(f"  ✅ {doc} ({size} bytes)")
        else:
            print(f"  ❌ {doc} nicht gefunden")
            return False
    
    return True


def main():
    print("=" * 70)
    print("Optimierungs-Scripts Test Suite")
    print("=" * 70)
    
    tests = [
        ("Imports", test_imports),
        ("Script Syntax", test_script_syntax),
        ("Help Output", test_help_output),
        ("Pruning", test_pruning_availability),
        ("Documentation", test_documentation_exists)
    ]
    
    results = {}
    
    for name, test_func in tests:
        try:
            results[name] = test_func()
        except Exception as e:
            print(f"\n❌ {name} Test fehlgeschlagen: {e}")
            import traceback
            traceback.print_exc()
            results[name] = False
    
    # Summary
    print("\n" + "=" * 70)
    print("Test Summary")
    print("=" * 70)
    
    passed = sum(1 for v in results.values() if v)
    total = len(results)
    
    for name, result in results.items():
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{status} - {name}")
    
    print("=" * 70)
    print(f"Ergebnis: {passed}/{total} Tests bestanden")
    print("=" * 70)
    
    if passed == total:
        print("✅ Alle Tests bestanden!")
        return 0
    else:
        print(f"❌ {total - passed} Test(s) fehlgeschlagen")
        return 1


if __name__ == '__main__':
    sys.exit(main())

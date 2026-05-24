#!/usr/bin/env python3
"""
Sanity checks for dynamic center-frame handling and n_frames configuration.
"""

from pathlib import Path
import json
import sys


ROOT = Path(__file__).resolve().parent


def test_center_frame_index_is_dynamic() -> bool:
    validator_file = ROOT / "training" / "validator.py"
    content = validator_file.read_text(encoding="utf-8")

    ok = "center_idx      = lr_stack.size(1) // 2" in content
    if ok:
        print("✅ PASS: validator center-frame index is derived dynamically (T // 2)")
    else:
        print("❌ FAIL: validator center-frame index is not dynamic")
    return ok


def test_runtime_config_paths_are_dynamic() -> bool:
    runtime_path = ROOT / "runtime_config.json"
    cfg = json.loads(runtime_path.read_text(encoding="utf-8"))
    paths = cfg.get("data", {}).get("paths", {})
    train_lr = paths.get("train_lr", "")
    val_lr = paths.get("val_lr", "")
    ok = "LR_{n_frames}frames" in train_lr and "LR_{n_frames}frames" in val_lr
    if ok:
        print("✅ PASS: runtime config LR paths use dynamic frame placeholder")
    else:
        print("❌ FAIL: runtime config LR paths are still hardcoded")
    return ok


def main() -> int:
    results = [
        test_center_frame_index_is_dynamic(),
        test_runtime_config_paths_are_dynamic(),
    ]
    passed = sum(results)
    total = len(results)
    print(f"\nTests Passed: {passed}/{total}")
    return 0 if all(results) else 1


if __name__ == "__main__":
    sys.exit(main())

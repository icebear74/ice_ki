"""
Training Run Lock — checkpoint compatibility guard.

On first start a ``training_run_locked.json`` file is written to the
category-specific run directory (e.g. ``{dataset_root}/{category}/``).

On resume the locked values are loaded and compared against the current
configuration.  Any incompatible difference causes an early, clear abort so
that checkpoints are never silently mixed with a different model architecture
or dataset layout.

Fields that are locked (must be identical on resume):
    n_feats       — model width
    n_blocks      — model depth
    n_frames      — temporal context
    scale         — upscaling factor (fixed 3)
    dataset_root  — absolute path to dataset root
    category      — dataset category (e.g. 'master', 'space')
    templates     — sorted list of template/size keys used for this run
"""

import json
import os
import sys
from typing import Dict, Any, List, Optional

LOCK_FILE_NAME = "training_run_locked.json"

# Fields that are compared strictly on resume.
# All must be present in both the locked file and the provided config_dict.
_REQUIRED_FIELDS = ["n_feats", "n_blocks", "n_frames", "scale",
                    "dataset_root", "category", "templates"]


def _lock_path(run_dir: str) -> str:
    return os.path.join(run_dir, LOCK_FILE_NAME)


def save_run_lock(
    run_dir: str,
    n_feats: int,
    n_blocks: int,
    n_frames: int,
    scale: int,
    dataset_root: str,
    category: str,
    templates: List[str],
) -> str:
    """
    Write ``training_run_locked.json`` to *run_dir* on a fresh training start.

    This is a no-op if the file already exists (prevents accidental overwrites
    when ``save_run_lock`` is called more than once in the same process).

    Args:
        run_dir:      Category-specific run directory (checkpoints live here).
        n_feats:      Number of feature channels.
        n_blocks:     Number of residual blocks.
        n_frames:     Number of input frames.
        scale:        Upscaling factor.
        dataset_root: Absolute path to dataset root.
        category:     Dataset category string.
        templates:    Sorted list of template keys used for this run.

    Returns:
        Path to the lock file (whether created or already existing).
    """
    path = _lock_path(run_dir)

    if os.path.exists(path):
        return path  # Already locked — caller should verify, not overwrite

    os.makedirs(run_dir, exist_ok=True)

    lock_data: Dict[str, Any] = {
        "n_feats":      n_feats,
        "n_blocks":     n_blocks,
        "n_frames":     n_frames,
        "scale":        scale,
        "dataset_root": os.path.abspath(dataset_root),
        "category":     category,
        "templates":    sorted(templates),
    }

    try:
        with open(path, "w", encoding="utf-8") as fh:
            json.dump(lock_data, fh, indent=2)
    except OSError as exc:
        print(f"⚠  Could not write run lock file: {exc}")

    return path


def load_and_verify_run_lock(
    run_dir: str,
    n_feats: int,
    n_blocks: int,
    n_frames: int,
    scale: int,
    dataset_root: str,
    category: str,
    templates: List[str],
    abort: bool = True,
) -> bool:
    """
    Load ``training_run_locked.json`` from *run_dir* and verify compatibility.

    If the locked file does not exist the function returns ``True`` (nothing
    to verify — the caller should then call ``save_run_lock`` to create it).

    If the file exists but a critical mismatch is detected the function prints
    a clear error message.  When *abort* is ``True`` (default) it calls
    ``sys.exit(1)`` so training never starts with an incompatible checkpoint.

    Args:
        run_dir:      Category-specific run directory.
        n_feats:      Current number of feature channels.
        n_blocks:     Current number of residual blocks.
        n_frames:     Current number of input frames.
        scale:        Current upscaling factor.
        dataset_root: Current absolute dataset root path.
        category:     Current dataset category.
        templates:    Current sorted list of template keys.
        abort:        When True, call sys.exit(1) on mismatch (default True).

    Returns:
        True  — no lock file found (fresh run) or verification passed.
        False — mismatch found (only when *abort=False*).
    """
    path = _lock_path(run_dir)

    if not os.path.exists(path):
        return True  # No lock yet — fresh run

    try:
        with open(path, "r", encoding="utf-8") as fh:
            locked: Dict[str, Any] = json.load(fh)
    except (OSError, json.JSONDecodeError) as exc:
        print(f"⚠  Could not read run lock file ({path}): {exc}")
        print("   Continuing without compatibility check.")
        return True

    current: Dict[str, Any] = {
        "n_feats":      n_feats,
        "n_blocks":     n_blocks,
        "n_frames":     n_frames,
        "scale":        scale,
        "dataset_root": os.path.abspath(dataset_root),
        "category":     category,
        "templates":    sorted(templates),
    }

    mismatches: List[str] = []
    for field in _REQUIRED_FIELDS:
        locked_val = locked.get(field)
        current_val = current.get(field)
        if locked_val != current_val:
            mismatches.append(
                f"  {field}: locked={locked_val!r}  current={current_val!r}"
            )

    if not mismatches:
        print(f"✅ Run lock verified: {path}")
        return True

    # Mismatch detected — print a clear error
    print()
    print("=" * 72)
    print("❌  CHECKPOINT COMPATIBILITY ERROR")
    print("-" * 72)
    print(f"  Lock file: {path}")
    print()
    print("  The following parameters differ from the locked run config:")
    for msg in mismatches:
        print(msg)
    print()
    print("  Resuming with different parameters would corrupt the checkpoint.")
    print("  Options:")
    print("    1. Restore the original config values shown above.")
    print("    2. Start fresh (choose 'L' at startup to create a new run).")
    print("       WARNING: Starting fresh will delete or backup existing checkpoints.")
    print("=" * 72)
    print()

    if abort:
        sys.exit(1)
    return False

"""
Dataset Architecture Loader

Loads and parses the ``dataset_architecture.json`` file written by
Dataset Generator V2.  This file is located at::

    {dataset_root}/dataset_architecture.json

and contains the complete description of how the dataset was built:
categories, format templates (gt_size / lr_size / scale), number of
frames, output image format (bmp / png), etc.

Typical usage
-------------
::

    from vsr_plusplus_NEU.utils.dataset_architecture import load_dataset_architecture

    arch = load_dataset_architecture("/mnt/data/training/Dataset")
    if arch:
        n_frames      = arch["n_frames"]        # e.g. 7
        img_ext       = arch["output_format"]   # "bmp" or "png"
        templates     = arch.get_templates_for_category("master")
        # → ["1152_169", "960_43", ...]

Architecture JSON structure (written by make_dataset_v2_uhd.py)
---------------------------------------------------------------
::

    {
      "generated_at":   "...",
      "generator_version": "dataset_generator_v2",
      "root_path":      "/mnt/data/training/Dataset",
      "n_frames":       7,
      "output_format":  "bmp",
      "category_targets": {"master": 300000, ...},
      "categories": {
        "master": {
          "target_total": 300000,
          "formats": [
            {
              "template":     "1152_169",
              "weight":       1,
              "source_mode":  "resize",
              "gt_size":      [1152, 648],
              "lr_size":      [384, 216],
              "scale":        3,
              "aspect_ratio": "16:9",
              "description":  "...",
              "degradation_mix": {}
            },
            ...
          ]
        },
        ...
      },
      "format_templates":  {...},
      "degradation_templates": {...}
    }
"""

import json
import os
from typing import Dict, List, Optional, Any


_ARCH_FILENAME = "dataset_architecture.json"

# Supported image extensions that the dataset generator can produce.
SUPPORTED_EXTENSIONS = ("bmp", "png")


class DatasetArchitecture:
    """Parsed view of a ``dataset_architecture.json`` file.

    Provides helper accessors used by the trainer so that callers do not need
    to navigate the raw JSON dict themselves.
    """

    def __init__(self, data: Dict[str, Any], source_path: str):
        self._data = data
        self.source_path = source_path

        # Top-level metadata
        self.n_frames: int = int(data.get("n_frames", 7))
        raw_fmt = data.get("output_format", "bmp").lower().lstrip(".")
        self.output_format: str = raw_fmt if raw_fmt in SUPPORTED_EXTENSIONS else "bmp"
        self.img_ext: str = f".{self.output_format}"  # e.g. ".bmp" or ".png"

        self.categories: Dict[str, Any] = data.get("categories", {})
        self.format_templates: Dict[str, Any] = data.get("format_templates", {})

    # ------------------------------------------------------------------
    # Accessors
    # ------------------------------------------------------------------

    def get_templates_for_category(self, category: str) -> List[str]:
        """Return ordered list of format template names used by *category*.

        Returns an empty list when *category* is not in the architecture file.
        """
        cat = self.categories.get(category, {})
        return [f["template"] for f in cat.get("formats", [])]

    def get_format_entry(self, category: str, template: str) -> Optional[Dict[str, Any]]:
        """Return the format entry dict for a given category + template, or None."""
        cat = self.categories.get(category, {})
        for entry in cat.get("formats", []):
            if entry["template"] == template:
                return entry
        return None

    def get_lr_dir_name(self) -> str:
        """Return the LR subdirectory name for the configured frame count.

        The Dataset Generator V2 uses ``LR_{n}frames`` for all frame counts
        **except** the original 5-frame legacy mode, which used the bare
        name ``LR`` (no frame count suffix) for historical reasons.
        All other counts (7, 9, …) use ``LR_{n}frames``.
        """
        n = self.n_frames
        return "LR" if n == 5 else f"LR_{n}frames"

    def all_categories(self) -> List[str]:
        """Return list of all category names present in the architecture."""
        return list(self.categories.keys())

    def __getitem__(self, key: str) -> Any:
        """Allow dict-style access to the raw architecture data."""
        return self._data[key]

    def get(self, key: str, default: Any = None) -> Any:
        return self._data.get(key, default)

    def __repr__(self) -> str:
        cats = list(self.categories.keys())
        return (
            f"DatasetArchitecture(n_frames={self.n_frames}, "
            f"output_format={self.output_format!r}, "
            f"categories={cats})"
        )


def load_dataset_architecture(dataset_root: str) -> Optional[DatasetArchitecture]:
    """Load and parse the ``dataset_architecture.json`` from *dataset_root*.

    Args:
        dataset_root: Root directory of the dataset (the directory that
                      contains the ``dataset_architecture.json`` file and
                      the category subdirectories).

    Returns:
        A :class:`DatasetArchitecture` instance on success, or ``None`` when
        the file does not exist or cannot be parsed.  A missing file is treated
        as a warning (not an error) so that legacy datasets without the file
        continue to work.
    """
    arch_path = os.path.join(dataset_root, _ARCH_FILENAME)
    if not os.path.isfile(arch_path):
        return None
    try:
        with open(arch_path, "r", encoding="utf-8") as fh:
            data = json.load(fh)
        return DatasetArchitecture(data, arch_path)
    except Exception as exc:
        print(f"⚠️  Could not parse {arch_path}: {exc}")
        return None


def get_size_keys_for_category(
    dataset_root: str,
    category: str,
) -> List[str]:
    """Convenience wrapper: return template names for *category* from the
    architecture file, or an empty list when the file is absent.

    This is the primary function used by ``train.py`` to replace the
    hardcoded ``KNOWN_SIZE_KEYS`` list.
    """
    arch = load_dataset_architecture(dataset_root)
    if arch is None:
        return []
    return arch.get_templates_for_category(category)

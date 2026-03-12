#!/usr/bin/env python3
"""
Generation plan manager for dataset_generator_v2.

The plan file (version 3.2) tracks per-video progress using video *paths* as
identifiers rather than integer indices.  This is more robust because it
survives reordering of the video list, additions, removals, and any change
to the ``forced_frames`` sort.

Plan file format (version 3.2)
-------------------------------
{
  "version": "3.2",
  "plan_created_at": "2026-03-11T23:45:20.902002",
  "current_phase": "phase_1",
  "phase_1": {
    "description": "Dataset generation",
    "status": "in_progress",
    "videos": [
      {
        "path": "/mnt/data/video/…/S01E01.mkv",
        "name": "S01E01 - ...",
        "status": "done",
        "patches_created": {"master": 4933}
      },
      {
        "path": "/mnt/data/video/…/S01E02.mkv",
        "name": "S01E02 - ...",
        "status": "pending",
        "patches_created": {}
      }
    ]
  }
}
"""

import json
import logging
import os
from datetime import datetime
from typing import Dict, List, Optional, Any

logger = logging.getLogger(__name__)

PLAN_VERSION = "3.2"
_DEFAULT_PHASE_NAME = "phase_1"


class GenerationPlan:
    """
    Plan file manager for the dataset generator.

    Tracks which videos have been successfully processed using their file
    *paths* as identifiers.  This is more robust than an integer index
    because it survives changes to video ordering or the video list itself.

    The plan file stores video entries under the phase key named by
    ``current_phase``.  When an existing plan is loaded the phase key is
    read from the file so that plans with any phase name (e.g. ``phase_169``)
    are handled correctly.

    Usage
    -----
    1. Create the plan (or load it if one already exists on disk)::

           plan = GenerationPlan("/path/to/.generation_plan.json")

    2. Populate it with the full list of videos (idempotent — already-known
       entries, including those marked "done", are preserved)::

           plan.initialize(self.videos)

    3. In the processing loop, skip videos that are already done::

           if plan.is_video_done(video_path):
               continue

    4. After a video finishes successfully, mark it done::

           plan.mark_video_done(video_path, patches_created_dict)
    """

    def __init__(self, plan_file: str) -> None:
        self.plan_file = plan_file
        self.plan: Dict[str, Any] = self._load_or_create()
        # Effective phase key — read from the plan so we handle any phase name.
        self._phase_name: str = self.plan.get("current_phase", _DEFAULT_PHASE_NAME)
        # Fast path → entry lookup (rebuilt from plan on every load/save)
        self._index: Dict[str, Dict[str, Any]] = {}
        self._rebuild_index()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _load_or_create(self) -> Dict[str, Any]:
        """Load an existing plan or return an empty skeleton."""
        if os.path.exists(self.plan_file):
            try:
                with open(self.plan_file, "r", encoding="utf-8") as fh:
                    data = json.load(fh)
                # Accept any 3.x plan (path-indexed)
                if isinstance(data, dict) and str(data.get("version", "")).split(".")[0] == "3":
                    logger.info(
                        f"Loaded generation plan (version {data.get('version')}, "
                        f"phase {data.get('current_phase')!r}) "
                        f"from {self.plan_file}"
                    )
                    return data
            except Exception as exc:
                logger.warning(f"Could not read plan file ({self.plan_file}): {exc}")

        logger.info(f"Creating new generation plan at {self.plan_file}")
        return {
            "version": PLAN_VERSION,
            "plan_created_at": datetime.now().isoformat(),
            "current_phase": _DEFAULT_PHASE_NAME,
            _DEFAULT_PHASE_NAME: {
                "description": "Dataset generation",
                "status": "in_progress",
                "videos": [],
            },
        }

    def _rebuild_index(self) -> None:
        """Rebuild the in-memory path → entry lookup from the stored list."""
        self._index = {}
        phase = self.plan.get(self._phase_name, {})
        for entry in phase.get("videos", []):
            path = entry.get("path")
            if path:
                self._index[path] = entry

    def _phase(self) -> Dict[str, Any]:
        return self.plan.setdefault(
            self._phase_name,
            {
                "description": "Dataset generation",
                "status": "in_progress",
                "videos": [],
            },
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def initialize(self, videos: List[Dict[str, Any]]) -> None:
        """
        Populate the plan with the given video list (idempotent).

        Only videos that are *not* yet tracked are added.  Existing entries,
        including those already marked "done", are left untouched so that
        a resume always preserves previously recorded progress.
        """
        phase = self._phase()
        video_list: List[Dict[str, Any]] = phase.setdefault("videos", [])

        known = {e["path"] for e in video_list if "path" in e}
        added = 0
        for v in videos:
            path = v.get("path")
            if path and path not in known:
                entry: Dict[str, Any] = {
                    "path": path,
                    "name": v.get("name", os.path.basename(path)),
                    "status": "pending",
                    "patches_created": {},
                }
                video_list.append(entry)
                known.add(path)
                added += 1

        if added:
            logger.info(f"Plan: added {added} new video(s) to track")

        self._rebuild_index()
        self.save()

    def is_video_done(self, video_path: str) -> bool:
        """Return ``True`` when the video has already been processed."""
        entry = self._index.get(video_path)
        return entry is not None and entry.get("status") == "done"

    def mark_video_done(
        self,
        video_path: str,
        patches_created: Optional[Dict[str, int]] = None,
    ) -> None:
        """Mark a video as done and record patch counts."""
        entry = self._index.get(video_path)
        if entry is None:
            # Video was not in the original plan – add it on the fly
            entry = {
                "path": video_path,
                "name": os.path.basename(video_path),
                "status": "pending",
                "patches_created": {},
            }
            self._phase()["videos"].append(entry)
            self._index[video_path] = entry

        entry["status"] = "done"
        if patches_created:
            entry["patches_created"] = dict(patches_created)
        self.save()

    def mark_video_pending(self, video_path: str) -> None:
        """Reset a video to *pending* so it will be retried on the next run."""
        entry = self._index.get(video_path)
        if entry is not None and entry.get("status") != "pending":
            entry["status"] = "pending"
            self.save()

    def count_done(self) -> int:
        """Return the number of videos already marked done."""
        return sum(1 for e in self._index.values() if e.get("status") == "done")

    def count_total(self) -> int:
        """Return the total number of videos tracked in the plan."""
        return len(self._index)

    def save(self) -> None:
        """Atomically persist the plan to disk."""
        try:
            plan_dir = os.path.dirname(self.plan_file)
            if plan_dir:
                os.makedirs(plan_dir, exist_ok=True)
            tmp = self.plan_file + ".tmp"
            with open(tmp, "w", encoding="utf-8") as fh:
                json.dump(self.plan, fh, indent=2, ensure_ascii=False)
            os.replace(tmp, self.plan_file)
        except Exception as exc:
            logger.error(f"Failed to save plan to {self.plan_file}: {exc}")

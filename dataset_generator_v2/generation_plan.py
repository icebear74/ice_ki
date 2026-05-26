#!/usr/bin/env python3
"""
Generation plan manager for dataset_generator_v2  –  Plan v4.0.

Architecture
============
The generator follows a strict  *first plan, then execute*  discipline:

1. Before any extraction begins :meth:`create_full_plan` is called with the
   complete intended work for every video.  The plan is immediately persisted.
2. Execution reads the plan and updates it as work completes.
3. After an interruption the generator resumes by loading the same plan file
   and skipping items whose ``status`` is ``"done"``.

Plan file format  (version 4.0)
--------------------------------
{
  "version": "4.0",
  "plan_created_at": "2026-04-26T17:00:00",
  "plan_updated_at": "2026-04-26T18:30:00",

  // ── Global aggregates (recomputed on every save) ──────────────────────── //
  "global": {
    "n_items_total":   63,
    "n_items_done":    5,
    "n_items_pending": 55,
    "n_items_running": 2,
    "n_items_failed":  1,

    "planned_total":                  165000,
    "planned_per_category":           {"general": 80000, "space": 55000, "toon": 30000},
    "planned_per_format_template":    {"uhd_169": 60000, "hd_169": 50000, ...},
    "planned_per_degradation_template": {"mpeg2_heavy": 30000, "web_medium": 80000, ...},
    "category_format_degradation": {
      "general": {
        "uhd_169": {"web_medium": 40000, "mpeg2_heavy": 15000},
        ...
      }
    },

    "completed_total":                  12000,
    "completed_per_category":           {"general": 8000},
    "completed_per_format_template":    {"uhd_169": 4000},
    "completed_per_degradation_template": {"web_medium": 6000}
  },

  // ── Per-video items ────────────────────────────────────────────────────── //
  "items": [
    {
      "plan_item_id":  "abc123def456",   // stable SHA-256[:16] of video_path
      "queue_position": 1,
      "video_path":    "/mnt/.../S01E01.mkv",
      "video_name":    "S01E01 – Title",
      "status":        "done",           // pending | in_progress | done | failed
      "created_at":    "...",
      "updated_at":    "...",
      "retry_count":   0,
      "failed_reason": null,

      "planned": {
        "total": 2500,
        "per_category":           {"general": 1500, "toon": 1000},
        "per_format_template":    {"uhd_169": 1250, "hd_169": 1250},
        "per_degradation_template": {"web_medium": 1500, "mpeg2_heavy": 1000},
        "category_format_degradation": {
          "general": {
            "uhd_169": {"web_medium": 500, "mpeg2_heavy": 250},
            "hd_169":  {"web_medium": 500, "mpeg2_heavy": 250}
          },
          "toon": {
            "uhd_169": {"web_medium": 500, "mpeg2_heavy": 500}
          }
        }
      },

      "completed": {
        "total": 2487,
        "per_category":           {"general": 1487, "toon": 1000},
        "per_format_template":    {"uhd_169": 1240, "hd_169": 1247},
        "per_degradation_template": {"web_medium": 1490, "mpeg2_heavy": 997}
      }
    }
  ]
}
"""

import hashlib
import json
import logging
import os
from datetime import datetime
from typing import Dict, List, Optional, Any

logger = logging.getLogger(__name__)

PLAN_VERSION = "4.0"


def _stable_id(video_path: str) -> str:
    """Return a 16-character stable identifier for *video_path*."""
    return hashlib.sha256(video_path.encode()).hexdigest()[:16]


class GenerationPlan:
    """
    Full execution plan manager for the dataset generator (plan v4.0).

    Lifecycle
    ---------
    1. ``create_full_plan(plan_items)`` — called **once** before extraction
       starts.  Persists the complete intended work for every video.
    2. ``update_item_started(plan_item_id)`` — called when a stream worker
       picks up a video.
    3. ``update_item_completed(...)`` — called when a stream worker finishes
       a video successfully.
    4. ``update_item_failed(...)`` — called on unrecoverable error.

    The plan is human-inspectable JSON on disk.  :meth:`save` uses an atomic
    rename so a crash never leaves a partial file.

    Backwards-compatible helpers
    ----------------------------
    ``is_video_done(video_path)`` and ``mark_video_done(video_path, patches)``
    are retained so existing call sites that have not yet been migrated
    continue to work.
    """

    def __init__(self, plan_file: str) -> None:
        self.plan_file = plan_file
        self.plan: Dict[str, Any] = self._load_or_create()
        # Fast lookup: plan_item_id → item dict
        self._index_by_id: Dict[str, Dict[str, Any]] = {}
        # Fast lookup: video_path → item dict
        self._index_by_path: Dict[str, Dict[str, Any]] = {}
        self._rebuild_index()

    # ── Internal helpers ──────────────────────────────────────────────────────

    def _load_or_create(self) -> Dict[str, Any]:
        """Load an existing v4 plan or return a bare skeleton."""
        if os.path.exists(self.plan_file):
            try:
                with open(self.plan_file, "r", encoding="utf-8") as fh:
                    data = json.load(fh)
                ver = str(data.get("version", "")).split(".")[0]
                if isinstance(data, dict) and ver == "4":
                    logger.info(
                        f"Loaded generation plan v{data.get('version')} "
                        f"({data.get('global', {}).get('n_items_total', '?')} items) "
                        f"from {self.plan_file}"
                    )
                    return data
                else:
                    logger.info(
                        f"Existing plan file has version {data.get('version')!r}; "
                        "upgrading to v4.0 (previous progress preserved where possible)"
                    )
                    # Try to preserve done status from old v3.x plans.
                    return self._migrate_from_v3(data)
            except Exception as exc:
                logger.warning(f"Could not read plan file ({self.plan_file}): {exc}")

        logger.info(f"Creating new generation plan (v4.0) at {self.plan_file}")
        return self._empty_plan()

    def _empty_plan(self) -> Dict[str, Any]:
        now = datetime.now().isoformat()
        return {
            "version": PLAN_VERSION,
            "plan_created_at": now,
            "plan_updated_at": now,
            "global": self._empty_global(),
            "items": [],
        }

    @staticmethod
    def _empty_global() -> Dict[str, Any]:
        return {
            "n_items_total":   0,
            "n_items_done":    0,
            "n_items_pending": 0,
            "n_items_running": 0,
            "n_items_failed":  0,
            "planned_total":                    0,
            "planned_per_category":             {},
            "planned_per_format_template":      {},
            "planned_per_degradation_template": {},
            "category_format_degradation":      {},
            "completed_total":                    0,
            "completed_per_category":             {},
            "completed_per_format_template":      {},
            "completed_per_degradation_template": {},
        }

    def _migrate_from_v3(self, old: Dict[str, Any]) -> Dict[str, Any]:
        """Build a v4 skeleton from an old v3.x plan, preserving done flags."""
        new = self._empty_plan()
        new["plan_created_at"] = old.get("plan_created_at", new["plan_created_at"])
        # Collect videos from the old phase structure.
        phase_key = old.get("current_phase", "phase_1")
        old_videos = old.get(phase_key, {}).get("videos", [])
        now = datetime.now().isoformat()
        for v in old_videos:
            path = v.get("path")
            if not path:
                continue
            status = v.get("status", "pending")
            patches = v.get("patches_created", {})
            new["items"].append({
                "plan_item_id":  _stable_id(path),
                "queue_position": 0,
                "video_path":    path,
                "video_name":    v.get("name", os.path.basename(path)),
                "status":        status if status in ("done", "pending", "failed") else "pending",
                "created_at":    now,
                "updated_at":    now,
                "retry_count":   0,
                "failed_reason": None,
                "planned":       {
                    "total": sum(patches.values()) if patches else 0,
                    "per_category": {},
                    "per_format_template": {},
                    "per_degradation_template": {},
                    "category_format_degradation": {},
                },
                "completed": {
                    "total": sum(patches.values()) if patches else 0,
                    "per_category": dict(patches) if patches else {},
                    "per_format_template": {},
                    "per_degradation_template": {},
                },
            })
        return new

    def _rebuild_index(self) -> None:
        self._index_by_id   = {}
        self._index_by_path = {}
        for item in self.plan.get("items", []):
            pid  = item.get("plan_item_id")
            path = item.get("video_path")
            if pid:
                self._index_by_id[pid] = item
            if path:
                self._index_by_path[path] = item

    def _recompute_global(self) -> None:
        """Recompute the ``"global"`` section from all items."""
        g: Dict[str, Any] = self._empty_global()

        for item in self.plan.get("items", []):
            status = item.get("status", "pending")
            if status == "done":
                g["n_items_done"] += 1
            elif status == "in_progress":
                g["n_items_running"] += 1
            elif status == "failed":
                g["n_items_failed"] += 1
            else:
                g["n_items_pending"] += 1

            # Planned aggregates
            pl = item.get("planned", {})
            g["planned_total"] += pl.get("total", 0)

            for cat, cnt in pl.get("per_category", {}).items():
                g["planned_per_category"][cat] = g["planned_per_category"].get(cat, 0) + cnt

            for fmt, cnt in pl.get("per_format_template", {}).items():
                g["planned_per_format_template"][fmt] = (
                    g["planned_per_format_template"].get(fmt, 0) + cnt
                )

            for deg, cnt in pl.get("per_degradation_template", {}).items():
                g["planned_per_degradation_template"][deg] = (
                    g["planned_per_degradation_template"].get(deg, 0) + cnt
                )

            for cat, fmt_map in pl.get("category_format_degradation", {}).items():
                g_cfd = g["category_format_degradation"].setdefault(cat, {})
                for fmt, deg_map in fmt_map.items():
                    g_cfd_f = g_cfd.setdefault(fmt, {})
                    for deg, cnt in deg_map.items():
                        g_cfd_f[deg] = g_cfd_f.get(deg, 0) + cnt

            # Completed aggregates (only for done items)
            if status == "done":
                co = item.get("completed", {})
                g["completed_total"] += co.get("total", 0)

                for cat, cnt in co.get("per_category", {}).items():
                    g["completed_per_category"][cat] = (
                        g["completed_per_category"].get(cat, 0) + cnt
                    )

                for fmt, cnt in co.get("per_format_template", {}).items():
                    g["completed_per_format_template"][fmt] = (
                        g["completed_per_format_template"].get(fmt, 0) + cnt
                    )

                for deg, cnt in co.get("per_degradation_template", {}).items():
                    g["completed_per_degradation_template"][deg] = (
                        g["completed_per_degradation_template"].get(deg, 0) + cnt
                    )

        g["n_items_total"] = len(self.plan.get("items", []))
        self.plan["global"] = g

    # ── Public API ────────────────────────────────────────────────────────────

    def create_full_plan(self, plan_items: List[Dict[str, Any]]) -> None:
        """
        Create or update the full execution plan before extraction starts.

        This is the primary entry point.  It must be called with the complete
        list of intended work items **before** :meth:`_run_multi_stream` is
        called.

        Idempotency
        -----------
        * Items whose ``status`` is ``"done"`` are left untouched so that
          re-running after a partial completion correctly skips done videos.
        * Planning data (sizes, weights) is refreshed for pending/failed items
          in case the config changed since the last run.
        * Items present in the old plan but absent from *plan_items* are
          retained with their current status (they may have been removed from
          the config temporarily).

        Args:
            plan_items: List of dicts with keys ``plan_item_id``,
                ``queue_position``, ``video_path``, ``video_name``, and
                ``planned`` (full breakdown dict).
        """
        now = datetime.now().isoformat()
        existing_by_id: Dict[str, Dict[str, Any]] = {
            item["plan_item_id"]: item
            for item in self.plan.get("items", [])
            if "plan_item_id" in item
        }

        new_items: List[Dict[str, Any]] = []
        for pitem in plan_items:
            pid = pitem["plan_item_id"]
            if pid in existing_by_id:
                existing = existing_by_id[pid]
                # Preserve execution state; refresh planning data.
                existing["planned"]        = pitem["planned"]
                existing["queue_position"] = pitem["queue_position"]
                existing["video_name"]     = pitem["video_name"]
                existing["updated_at"]     = now
                new_items.append(existing)
            else:
                new_items.append({
                    "plan_item_id":  pid,
                    "queue_position": pitem["queue_position"],
                    "video_path":    pitem["video_path"],
                    "video_name":    pitem["video_name"],
                    "status":        "pending",
                    "created_at":    now,
                    "updated_at":    now,
                    "retry_count":   0,
                    "failed_reason": None,
                    "planned":       pitem["planned"],
                    "completed": {
                        "total": 0,
                        "per_category":             {},
                        "per_format_template":      {},
                        "per_degradation_template": {},
                    },
                })

        self.plan["items"] = new_items
        self.plan["plan_updated_at"] = now
        self._recompute_global()
        self._rebuild_index()
        self.save()

        n_done    = self.plan["global"]["n_items_done"]
        n_pending = self.plan["global"]["n_items_pending"]
        logger.info(
            f"Plan v4.0 persisted: {len(new_items)} items "
            f"({n_done} done, {n_pending} pending)  →  {self.plan_file}"
        )

    def update_item_started(self, plan_item_id: str) -> None:
        """Mark an item as *in_progress*."""
        item = self._index_by_id.get(plan_item_id)
        if item is None:
            return
        if item.get("status") != "done":
            item["status"]     = "in_progress"
            item["updated_at"] = datetime.now().isoformat()
            self._recompute_global()
            self.save()

    def update_item_completed(
        self,
        plan_item_id: str,
        completed_per_category:             Dict[str, int],
        completed_per_format_template:      Optional[Dict[str, int]] = None,
        completed_per_degradation_template: Optional[Dict[str, int]] = None,
    ) -> None:
        """
        Mark an item as *done* and record actual completion counts.

        Args:
            plan_item_id:                     Stable item ID.
            completed_per_category:           Actual ``{category: count}``
                                              from the extractor result.
            completed_per_format_template:    Approximate per-format counts
                                              (estimated from planned weights
                                              × actual completion ratio).
                                              May be ``None`` or ``{}``.
            completed_per_degradation_template: Actual per-degradation counts
                                              from ``degrade_counts`` timing.
                                              May be ``None`` or ``{}``.
        """
        item = self._index_by_id.get(plan_item_id)
        if item is None:
            return

        total = sum(v for v in completed_per_category.values() if isinstance(v, int))

        item["status"]     = "done"
        item["updated_at"] = datetime.now().isoformat()
        item["completed"]  = {
            "total":                      total,
            "per_category":               dict(completed_per_category),
            "per_format_template":        dict(completed_per_format_template  or {}),
            "per_degradation_template":   dict(completed_per_degradation_template or {}),
        }
        self._recompute_global()
        self.plan["plan_updated_at"] = datetime.now().isoformat()
        self.save()

    def update_item_failed(
        self,
        plan_item_id: str,
        reason: str = "",
    ) -> None:
        """Mark an item as *failed* so it will be retried on the next run."""
        item = self._index_by_id.get(plan_item_id)
        if item is None:
            return
        item["status"]        = "failed"
        item["failed_reason"] = reason or "unknown error"
        item["retry_count"]   = item.get("retry_count", 0) + 1
        item["updated_at"]    = datetime.now().isoformat()
        self._recompute_global()
        self.plan["plan_updated_at"] = datetime.now().isoformat()
        self.save()

    def accumulate_item_completed(
        self,
        plan_item_id: str,
        new_per_category:             Dict[str, int],
        new_per_format_template:      Optional[Dict[str, int]] = None,
        new_per_degradation_template: Optional[Dict[str, int]] = None,
    ) -> None:
        """
        Accumulate completed patch counts onto any previously stored counts.

        Unlike :meth:`update_item_completed` (which overwrites completed
        data), this method **adds** *new_* counts to whatever was already
        recorded.  Use this for the streaming-friendly resume workflow where
        only the *remaining* patches were processed in the current run and
        the prior partial completion must be preserved:

        1. Run is interrupted mid-video → ``status = "in_progress"``,
           ``completed.per_category = {cat: N}``
        2. On next run, :func:`build_remaining_assignments` plans only the
           remaining work.
        3. After the remaining extraction finishes, call
           ``accumulate_item_completed`` so the plan reflects
           ``completed = prior + new`` rather than only *new*.

        Args:
            plan_item_id:                 Stable item ID.
            new_per_category:             Freshly completed ``{category: count}``
                                          from the extractor result.
            new_per_format_template:      Freshly completed per-format counts.
            new_per_degradation_template: Freshly completed per-degradation counts.
        """
        item = self._index_by_id.get(plan_item_id)
        if item is None:
            return

        prior = item.get("completed", {})

        def _merge(old: Dict[str, int], new: Optional[Dict[str, int]]) -> Dict[str, int]:
            merged = dict(old)
            for k, v in (new or {}).items():
                merged[k] = merged.get(k, 0) + v
            return merged

        merged_cat = _merge(prior.get("per_category", {}),             new_per_category)
        merged_fmt = _merge(prior.get("per_format_template", {}),      new_per_format_template)
        merged_deg = _merge(prior.get("per_degradation_template", {}), new_per_degradation_template)

        item["status"]     = "done"
        item["updated_at"] = datetime.now().isoformat()
        item["completed"]  = {
            "total":                      sum(merged_cat.values()),
            "per_category":               merged_cat,
            "per_format_template":        merged_fmt,
            "per_degradation_template":   merged_deg,
        }
        self._recompute_global()
        self.plan["plan_updated_at"] = datetime.now().isoformat()
        self.save()

    def get_item_by_path(self, video_path: str) -> Optional[Dict[str, Any]]:
        """Return the plan item for *video_path*, or ``None``."""
        return self._index_by_path.get(video_path)

    def get_item_by_id(self, plan_item_id: str) -> Optional[Dict[str, Any]]:
        """Return the plan item with *plan_item_id*, or ``None``."""
        return self._index_by_id.get(plan_item_id)

    def get_global_stats(self) -> Dict[str, Any]:
        """Return a copy of the ``"global"`` section."""
        return dict(self.plan.get("global", self._empty_global()))

    # ── Backwards-compatible API ──────────────────────────────────────────────

    def initialize(self, videos: List[Dict[str, Any]]) -> None:
        """
        Legacy initialiser — adds videos not yet tracked as *pending* items.

        New code should call :meth:`create_full_plan` instead, which also
        populates the full planning breakdown before extraction starts.
        """
        now = datetime.now().isoformat()
        added = 0
        for v in videos:
            path = v.get("path")
            if not path:
                continue
            pid = _stable_id(path)
            if pid not in self._index_by_id:
                item = {
                    "plan_item_id":  pid,
                    "queue_position": 0,
                    "video_path":    path,
                    "video_name":    v.get("name", os.path.basename(path)),
                    "status":        "pending",
                    "created_at":    now,
                    "updated_at":    now,
                    "retry_count":   0,
                    "failed_reason": None,
                    "planned": {
                        "total": 0,
                        "per_category": {},
                        "per_format_template": {},
                        "per_degradation_template": {},
                        "category_format_degradation": {},
                    },
                    "completed": {
                        "total": 0,
                        "per_category": {},
                        "per_format_template": {},
                        "per_degradation_template": {},
                    },
                }
                self.plan.setdefault("items", []).append(item)
                self._index_by_id[pid]   = item
                self._index_by_path[path] = item
                added += 1

        if added:
            logger.info(f"Plan (legacy initialize): added {added} new video(s)")
            self._recompute_global()
            self.plan["plan_updated_at"] = now
            self.save()

    def is_video_done(self, video_path: str) -> bool:
        """Return ``True`` when the video has already been successfully processed."""
        item = self._index_by_path.get(video_path)
        return item is not None and item.get("status") == "done"

    def mark_video_done(
        self,
        video_path: str,
        patches_created: Optional[Dict[str, int]] = None,
    ) -> None:
        """Legacy helper — prefer :meth:`update_item_completed`."""
        pid = _stable_id(video_path)
        if pid not in self._index_by_id:
            # Auto-add if not tracked yet.
            now = datetime.now().isoformat()
            item = {
                "plan_item_id":  pid,
                "queue_position": 0,
                "video_path":    video_path,
                "video_name":    os.path.basename(video_path),
                "status":        "pending",
                "created_at":    now,
                "updated_at":    now,
                "retry_count":   0,
                "failed_reason": None,
                "planned":  {"total": 0, "per_category": {}, "per_format_template": {}, "per_degradation_template": {}, "category_format_degradation": {}},
                "completed": {"total": 0, "per_category": {}, "per_format_template": {}, "per_degradation_template": {}},
            }
            self.plan.setdefault("items", []).append(item)
            self._index_by_id[pid]        = item
            self._index_by_path[video_path] = item

        self.update_item_completed(
            pid,
            completed_per_category             = dict(patches_created or {}),
            completed_per_format_template      = None,
            completed_per_degradation_template = None,
        )

    def mark_video_pending(self, video_path: str) -> None:
        """Reset a video to *pending* so it will be retried on the next run."""
        item = self._index_by_path.get(video_path)
        if item is not None and item.get("status") != "pending":
            item["status"]     = "pending"
            item["updated_at"] = datetime.now().isoformat()
            self._recompute_global()
            self.plan["plan_updated_at"] = datetime.now().isoformat()
            self.save()

    def count_done(self) -> int:
        """Return the number of videos already marked done."""
        return self.plan.get("global", {}).get("n_items_done", 0)

    def count_total(self) -> int:
        """Return the total number of videos tracked in the plan."""
        return self.plan.get("global", {}).get("n_items_total", 0)

    # ── Persistence ───────────────────────────────────────────────────────────

    def save(self) -> None:
        """Atomically persist the plan to disk (tmp-then-rename)."""
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

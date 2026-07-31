"""On-demand replay frame reconstruction for dashboard as-of views."""
from __future__ import annotations

import os
import threading
from collections import OrderedDict
from dataclasses import dataclass
from typing import Any

from storage import snapshot as snap


@dataclass
class ReplayFrame:
    t: int
    products: dict[str, dict[str, Any]]
    agents: list[dict[str, Any]]
    survival_state: dict[str, Any]


class ReplayFrameCache:
    """Bounded per-process cache for checkpoint + delta replay frames.

    Frames hold mutable product overlays plus the latest full agents blob at or
    before ``t``. Static product columns stay in SQLite and are joined by callers.

    Performance optimizations:
    - File I/O (checkpoint decompression, delta reads) happens OUTSIDE the lock
      so concurrent requests for the same run don't serialize behind a single
      slow reconstruction.
    - No cloning on cache hits (callers must not mutate the returned frame).
    - Checkpoint base states cached separately to avoid re-reading json.gz files.
    - Default cache size increased to 32 frames.
    """

    def __init__(self, runs_root: str, max_frames: int = 32):
        self.runs_root = runs_root
        self.max_frames = max(1, int(max_frames))
        self._cache: OrderedDict[tuple[str, int], ReplayFrame] = OrderedDict()
        self._checkpoint_cache: dict[tuple[str, int], tuple[int, dict, list, dict]] = {}
        self._lock = threading.RLock()

    def frame(self, run_id: str, t: int) -> ReplayFrame:
        t = int(t)
        key = (run_id, t)

        # Phase 1: quick cache check (under lock)
        with self._lock:
            cached = self._cache.get(key)
            if cached is not None:
                self._cache.move_to_end(key)
                return self._clone(cached)

        # Phase 2: determine base state. We need to know where to start
        # reconstruction from -- a prior cached frame, or a checkpoint on disk.
        # Each helper acquires the lock only briefly for the lookup, then does
        # any file I/O outside the lock.
        prior_step = self._find_prior_cached_step(run_id, t)
        if prior_step is not None:
            with self._lock:
                base = self._cache.get((run_id, prior_step))
            if base is not None:
                base_t = prior_step
                products = {pid: dict(values) for pid, values in base.products.items()}
                agents = [dict(agent) for agent in base.agents]
                survival_state = dict(base.survival_state)
                start_after = base_t + 1
            else:
                # Race: another thread evicted it. Fall back to checkpoint.
                base_t, products, agents, survival_state, start_after = (
                    self._load_checkpoint_base(run_id, t)
                )
        else:
            base_t, products, agents, survival_state, start_after = (
                self._load_checkpoint_base(run_id, t)
            )

        # Phase 3: apply delta snapshots -- all I/O, NO lock held.
        for step in self._delta_steps(run_id, start_after, t):
            obj = snap.read_env_snapshot(self.runs_root, run_id, step)
            if not obj:
                continue
            self._apply_snapshot(products, obj)
            if "agents" in obj:
                agents = list(obj.get("agents") or [])
            if "survival_state" in obj:
                survival_state = dict(obj.get("survival_state") or {})

        frame = ReplayFrame(t=t, products=products, agents=agents, survival_state=survival_state)

        # Phase 4: store in frame cache (under lock). Another thread may have
        # beaten us to it; either result is equivalent so last-writer-wins is fine.
        with self._lock:
            existing = self._cache.get(key)
            if existing is not None:
                self._cache.move_to_end(key)
                return self._clone(existing)
            self._cache[key] = frame
            self._cache.move_to_end(key)
            while len(self._cache) > self.max_frames:
                self._cache.popitem(last=False)
        return frame

    def _clone(self, frame: ReplayFrame) -> ReplayFrame:
        return ReplayFrame(
            t=frame.t,
            products={pid: dict(values) for pid, values in frame.products.items()},
            agents=[dict(agent) for agent in frame.agents],
            survival_state=dict(frame.survival_state),
        )

    def _find_prior_cached_step(self, run_id: str, t: int) -> int | None:
        """Return the largest cached step for this run at or before t."""
        with self._lock:
            steps = [step for (rid, step) in self._cache if rid == run_id and step <= t]
        return max(steps) if steps else None

    def _load_checkpoint_base(
        self, run_id: str, t: int
    ) -> tuple[int, dict[str, dict[str, Any]], list[dict[str, Any]], dict[str, Any], int]:
        """Load the base state from the nearest checkpoint <= t.

        Returns (base_t, products, agents, survival_state, start_after).
        Checkpoint reads are cached so repeated seeks into the same checkpoint
        window skip the gzip decompression.
        """
        checkpoint_t = self._latest_checkpoint_t(run_id, t)
        if checkpoint_t is None:
            return -1, {}, [], {}, 0

        ck_key = (run_id, checkpoint_t)

        # Check checkpoint cache under lock.
        with self._lock:
            hit = self._checkpoint_cache.get(ck_key)
        if hit is not None:
            base_t, products, agents, survival_state = hit
            return (
                base_t,
                {pid: dict(values) for pid, values in products.items()},
                [dict(agent) for agent in agents],
                dict(survival_state),
                base_t + 1,
            )

        # I/O outside any lock: read + gzip-decompress the checkpoint file.
        checkpoint = snap.read_env_checkpoint(self.runs_root, run_id, checkpoint_t) or {}
        products = {
            pid: dict(values)
            for pid, values in (checkpoint.get("products") or {}).items()
        }
        agents = list(checkpoint.get("agents") or [])
        survival_state = dict(checkpoint.get("survival_state") or {})

        # Store in checkpoint cache under lock.
        with self._lock:
            # Another thread may have populated it; either copy is fine.
            self._checkpoint_cache[ck_key] = (checkpoint_t, products, agents, survival_state)
            while len(self._checkpoint_cache) > 8:
                oldest_key = next(iter(self._checkpoint_cache))
                del self._checkpoint_cache[oldest_key]

        # Return COPIES so caller mutations (delta apply) don't pollute the cache.
        return (
            checkpoint_t,
            {pid: dict(v) for pid, v in products.items()},
            [dict(a) for a in agents],
            dict(survival_state),
            checkpoint_t + 1,
        )

    def _latest_checkpoint_t(self, run_id: str, t: int) -> int | None:
        checkpoint_dir = os.path.join(snap.run_dir(self.runs_root, run_id), "env_checkpoint")
        if not os.path.isdir(checkpoint_dir):
            return None
        out: list[int] = []
        for name in os.listdir(checkpoint_dir):
            if name.startswith("t_") and name.endswith(".json.gz"):
                try:
                    step = int(name[2:-8])
                except ValueError:
                    continue
                if step <= t:
                    out.append(step)
        return max(out) if out else None

    def _delta_steps(self, run_id: str, t_from: int, t_to: int) -> list[int]:
        snap_dir = os.path.join(snap.run_dir(self.runs_root, run_id), "env_snapshot")
        if not os.path.isdir(snap_dir):
            return []
        out: list[int] = []
        for name in os.listdir(snap_dir):
            if name.startswith("t_") and name.endswith(".json"):
                try:
                    step = int(name[2:-5])
                except ValueError:
                    continue
                if t_from <= step <= t_to:
                    out.append(step)
        out.sort()
        return out

    def _apply_snapshot(self, products: dict[str, dict[str, Any]], obj: dict[str, Any]) -> None:
        if obj.get("kind") == "delta":
            for pid, delta in (obj.get("products_delta") or {}).items():
                current = dict(products.get(pid) or {})
                current.update(delta)
                products[pid] = current
            return

        raw_products = obj.get("products") or {}
        if isinstance(raw_products, dict):
            for pid, delta in raw_products.items():
                current = dict(products.get(pid) or {})
                current.update(delta)
                products[pid] = current
        elif isinstance(raw_products, list):
            for row in raw_products:
                pid = row.get("product_id")
                if pid:
                    products[pid] = dict(row)

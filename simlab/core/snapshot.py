"""Snapshot storage and JSON friendly action log records."""
from __future__ import annotations
from dataclasses import dataclass
from typing import Any
from simlab.core.types import Action, Snapshot

@dataclass(frozen=True)
class ActionLogEntry:
    step: int
    agent_id: str
    action: Action
    def to_dict(self) -> dict[str, Any]: return {"step": self.step, "agent_id": self.agent_id, "action": vars(self.action)}

class SnapshotManager:
    def __init__(self, max_snapshots: int = 2000) -> None: self._snapshots: dict[int, Snapshot] = {}; self.max_snapshots = max_snapshots
    def add(self, snapshot: Snapshot) -> None:
        self._snapshots[snapshot.step] = snapshot
        while len(self._snapshots) > self.max_snapshots: self._snapshots.pop(min(self._snapshots))
    @property
    def latest_snapshot(self) -> Snapshot | None: return self._snapshots[max(self._snapshots)] if self._snapshots else None
    def get_snapshot(self, step: int) -> Snapshot | None: return self._snapshots.get(step)
    def capture_snapshot(self, world: Any) -> Snapshot: snapshot = world.snapshot(); self.add(snapshot); return snapshot

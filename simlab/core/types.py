"""Small data objects shared by the simulation core."""

from __future__ import annotations

from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any, Mapping

import numpy as np

@dataclass(frozen=True)
class Action:
    turn: float = 0.0
    thrust: float = 0.0
    eat: bool = False
    action_id: str = ""
    def __post_init__(self) -> None:
        object.__setattr__(self, "turn", float(np.clip(self.turn, -1.0, 1.0)))
        object.__setattr__(self, "thrust", float(np.clip(self.thrust, 0.0, 1.0)))
    @property
    def move_direction(self) -> tuple[int, int]:
        return (int(round(self.thrust * np.cos(self.turn))), int(round(self.thrust * np.sin(self.turn))))
    @property
    def energy_spent(self) -> float:
        return self.thrust

@dataclass(frozen=True)
class AgentState:
    agent_id: str
    position: tuple[float, float]
    heading: float = 0.0
    speed: float = 0.0
    energy: float = 100.0
    health: float = 100.0
    age: float = 0.0
    alive: bool = True
    @property
    def is_alive(self) -> bool: return self.alive
    @property
    def max_energy(self) -> float: return 100.0

@dataclass(frozen=True)
class Observation:
    rays: tuple[float, ...] = ()
    ray_types: tuple[int, ...] = ()
    proprioception: tuple[float, float, float] = (1.0, 1.0, 0.0)
    agent_position: tuple[float, float] = (0.0, 0.0)
    agent_energy: float = 100.0
    step_count: int = 0
    @property
    def world_state(self) -> np.ndarray: return np.asarray(self.rays, dtype=np.float32)
    @property
    def food_locations(self) -> list[tuple[float, float]]: return []
    @property
    def hazard_locations(self) -> list[tuple[float, float]]: return []

@dataclass(frozen=True)
class Snapshot:
    step: int
    agent_states: Mapping[str, AgentState]
    food_patches: tuple[tuple[float, float], ...]
    hazard_patches: tuple[tuple[float, float], ...]
    rng_state: Mapping[str, Any]
    width: float
    height: float
    dt: float
    metadata: Mapping[str, Any] = field(default_factory=dict)
    @property
    def world_state(self) -> np.ndarray:
        raster = np.zeros((int(self.height) + 1, int(self.width) + 1), dtype=np.int8)
        for state in self.agent_states.values():
            if state.alive:
                x, y = int(state.position[0]), int(state.position[1])
                if 0 <= x < raster.shape[1] and 0 <= y < raster.shape[0]: raster[y, x] = 1
        raster.setflags(write=False); return raster
    def to_dict(self) -> dict[str, Any]:
        return {"step": self.step, "agent_states": {k: vars(v) for k, v in self.agent_states.items()},
                "food_patches": [list(p) for p in self.food_patches], "hazard_patches": [list(p) for p in self.hazard_patches],
                "rng_state": dict(self.rng_state), "width": self.width, "height": self.height, "dt": self.dt, "metadata": dict(self.metadata)}
    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Snapshot":
        agents = {aid: AgentState(**{**state, "position": tuple(state["position"])}) for aid, state in data["agent_states"].items()}
        return cls(int(data["step"]), MappingProxyType(agents), tuple(tuple(p) for p in data.get("food_patches", [])), tuple(tuple(p) for p in data.get("hazard_patches", [])), data["rng_state"], float(data["width"]), float(data["height"]), float(data["dt"]), MappingProxyType(dict(data.get("metadata", {}))))

@dataclass(frozen=True)
class StepResult:
    agent_states: Mapping[str, AgentState]
    observations: Mapping[str, Observation]
    snapshot: Snapshot
    terminated: bool
    truncated: bool = False
    reward: float | Mapping[str, float] = 0.0
    info: Mapping[str, Any] = field(default_factory=dict)

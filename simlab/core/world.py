"""Deterministic continuous 2D survival world."""
from __future__ import annotations
import copy, hashlib, json, math
from types import MappingProxyType
from typing import Any
import numpy as np
from simlab.core.rng import SimRNG
from simlab.core.types import Action, AgentState, Observation, Snapshot, StepResult

class World:
    """A walled continuous arena whose state is owned by this object."""
    def __init__(self, width: float = 100.0, height: float = 100.0, seed: int = 0, dt: float = 0.1) -> None:
        self.width, self.height, self.dt = float(width), float(height), float(dt)
        self._seed = int(seed); self._rng = SimRNG(self._seed); self._agents: dict[str, dict[str, Any]] = {}
        self._food_patches: list[tuple[float, float]] = []; self._hazard_patches: list[tuple[float, float]] = []; self._step_count = 0
    @property
    def agents(self) -> MappingProxyType: return MappingProxyType(self._agents)
    @property
    def food_patches(self) -> tuple[tuple[float, float], ...]: return tuple(self._food_patches)
    @property
    def hazard_patches(self) -> tuple[tuple[float, float], ...]: return tuple(self._hazard_patches)
    def reset(self, seed: int | None = None) -> None:
        if seed is not None: self._seed = int(seed)
        self._rng = SimRNG(self._seed); self._agents.clear(); self._food_patches.clear(); self._hazard_patches.clear(); self._step_count = 0
    def add_agent(self, agent_id: str, position: tuple[float, float], energy: float = 100.0, heading: float = 0.0, health: float = 100.0) -> None:
        self._agents[agent_id] = {"position": self._clamp_point(position), "heading": float(heading), "speed": 0.0, "energy": float(np.clip(energy, 0.0, 100.0)), "health": float(np.clip(health, 0.0, 100.0)), "age": 0.0, "is_alive": bool(health > 0 and energy > 0), "food_eaten": 0}
    def remove_agent(self, agent_id: str) -> None: self._agents.pop(agent_id, None)
    def add_food(self, position: tuple[float, float], count: int = 1) -> None: self._food_patches.extend([self._clamp_point(position)] * max(0, int(count)))
    def add_hazard(self, position: tuple[float, float], count: int = 1) -> None: self._hazard_patches.extend([self._clamp_point(position)] * max(0, int(count)))
    def step(self, actions: dict[str, Action]) -> StepResult:
        self._step_count += 1; rewards: dict[str, float] = {}
        for aid, state in self._agents.items():
            if not state["is_alive"]: rewards[aid] = 0.0; continue
            action = actions.get(aid, Action()); state["heading"] = (state["heading"] + action.turn * 3.0 * self.dt) % (2 * math.pi); state["speed"] = action.thrust * 8.0
            x, y = state["position"]; state["position"] = self._clamp_point((x + math.cos(state["heading"]) * state["speed"] * self.dt, y + math.sin(state["heading"]) * state["speed"] * self.dt))
            state["energy"] = max(0.0, state["energy"] - (0.08 + 0.04 * state["speed"]) * self.dt); state["age"] += self.dt; reward = -0.01
            if action.eat:
                for index, food in enumerate(self._food_patches):
                    if math.dist(state["position"], food) <= 1.5:
                        self._food_patches.pop(index); state["energy"] = min(100.0, state["energy"] + 20.0); state["food_eaten"] += 1; reward += 1.0; break
            if state["energy"] <= 0.0: state["health"] = max(0.0, state["health"] - 0.5)
            if state["health"] <= 0.0: state["is_alive"] = False
            rewards[aid] = reward
        states = self._state_view(); observations = {aid: self._observation(aid) for aid in self._agents}; snap = self.snapshot()
        return StepResult(MappingProxyType(states), MappingProxyType(observations), snap, not any(s.alive for s in states.values()), reward=MappingProxyType(rewards), info=MappingProxyType({"step": self._step_count}))
    def _state_view(self) -> dict[str, AgentState]:
        return {aid: AgentState(aid, tuple(s["position"]), s["heading"], s["speed"], s["energy"], s["health"], s["age"], s["is_alive"]) for aid, s in self._agents.items()}
    def _observation(self, aid: str) -> Observation:
        s = self._agents[aid]; rays = tuple(self._ray_distances(s["position"], s["heading"]))
        return Observation(rays, tuple(0 for _ in rays), (s["energy"] / 100.0, s["health"] / 100.0, s["speed"] / 8.0), tuple(s["position"]), s["energy"], self._step_count)
    def _ray_distances(self, position: tuple[float, float], heading: float, n: int = 16) -> list[float]:
        distances = []
        for angle in heading + np.linspace(-math.radians(135), math.radians(135), n):
            dx, dy = math.cos(angle), math.sin(angle); hits = []
            if dx > 0: hits.append((self.width - position[0]) / dx)
            elif dx < 0: hits.append(-position[0] / dx)
            if dy > 0: hits.append((self.height - position[1]) / dy)
            elif dy < 0: hits.append(-position[1] / dy)
            distances.append(float(min(hits)))
        return distances
    def snapshot(self) -> Snapshot:
        return Snapshot(self._step_count, MappingProxyType(self._state_view()), tuple(self._food_patches), tuple(self._hazard_patches), copy.deepcopy(self._rng.state), self.width, self.height, self.dt, MappingProxyType({"seed": self._seed}))
    def restore(self, snapshot: Snapshot) -> None:
        self._seed = int(snapshot.metadata.get("seed", self._seed)); self._step_count = snapshot.step; self._food_patches = list(snapshot.food_patches); self._hazard_patches = list(snapshot.hazard_patches)
        self._agents = {aid: {"position": tuple(s.position), "heading": s.heading, "speed": s.speed, "energy": s.energy, "health": s.health, "age": s.age, "is_alive": s.alive, "food_eaten": 0} for aid, s in snapshot.agent_states.items()}
        self._rng = SimRNG(0); self._rng.state = copy.deepcopy(dict(snapshot.rng_state))
    def state_hash(self) -> str: return hashlib.sha256(json.dumps(self.snapshot().to_dict(), sort_keys=True, default=str).encode()).hexdigest()
    def _clamp_point(self, point: tuple[float, float]) -> tuple[float, float]: return (float(np.clip(point[0], 0.0, self.width)), float(np.clip(point[1], 0.0, self.height)))
    def get_agent(self, agent_id: str) -> dict[str, Any] | None: return self._agents.get(agent_id)
    def get_step_count(self) -> int: return self._step_count
    def get_world_state(self) -> np.ndarray: return self.snapshot().world_state

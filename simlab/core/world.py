"""Deterministic continuous 2D survival world."""

from __future__ import annotations

import copy
import hashlib
import json
import math
from types import MappingProxyType
from typing import Any

import numpy as np

from simlab.core.rng import SimRNG
from simlab.core.types import (
    Action,
    AgentState,
    Observation,
    Snapshot,
    StepResult,
    canonicalize,
)


class World:
    """A walled continuous arena whose state is owned by this object."""

    def __init__(
        self,
        width: float = 100.0,
        height: float = 100.0,
        seed: int = 0,
        dt: float = 0.1,
    ) -> None:
        self.width = float(width)
        self.height = float(height)
        self.dt = float(dt)
        self._seed = int(seed)
        self._rng = SimRNG(self._seed)
        self._agents: dict[str, dict[str, Any]] = {}
        self._food_patches: list[dict[str, Any]] = []
        self._hazard_patches: list[dict[str, Any]] = []
        self._metadata: dict[str, Any] = {}
        self._step_count = 0

    @property
    def agents(self) -> MappingProxyType:
        return MappingProxyType(self._agents)

    @property
    def food_patches(self) -> tuple[MappingProxyType, ...]:
        return tuple(MappingProxyType(dict(patch)) for patch in self._food_patches)

    @property
    def hazard_patches(self) -> tuple[MappingProxyType, ...]:
        return tuple(MappingProxyType(dict(patch)) for patch in self._hazard_patches)

    def reset(self, seed: int | None = None) -> None:
        if seed is not None:
            self._seed = int(seed)
        self._rng = SimRNG(self._seed)
        self._agents.clear()
        self._food_patches.clear()
        self._hazard_patches.clear()
        self._metadata.clear()
        self._step_count = 0

    def add_agent(
        self,
        agent_id: str,
        position: tuple[float, float],
        energy: float = 100.0,
        heading: float = 0.0,
        health: float = 100.0,
    ) -> None:
        capacity = 100.0
        clipped_energy = float(np.clip(energy, 0.0, capacity))
        self._agents[agent_id] = {
            "position": self._clamp_point(position),
            "heading": float(heading),
            "speed": 0.0,
            "energy": clipped_energy,
            "max_energy": capacity,
            "health": float(np.clip(health, 0.0, 100.0)),
            "age": 0.0,
            "is_alive": bool(health > 0 and clipped_energy > 0),
            "food_eaten": 0,
        }

    def remove_agent(self, agent_id: str) -> None:
        self._agents.pop(agent_id, None)

    def add_food(
        self, position: tuple[float, float], count: int = 1, amount: float = 20.0
    ) -> None:
        point = self._clamp_point(position)
        for _ in range(max(0, int(count))):
            self._food_patches.append(
                {"position": point, "amount": float(max(0.0, amount))}
            )

    def add_hazard(
        self,
        position: tuple[float, float],
        count: int = 1,
        damage: float = 12.0,
        radius: float = 1.5,
        detection_radius: float = 30.0,
    ) -> None:
        point = self._clamp_point(position)
        for _ in range(max(0, int(count))):
            self._hazard_patches.append(
                {
                    "position": point,
                    "damage": float(max(0.0, damage)),
                    "radius": float(max(0.1, radius)),
                    "detection_radius": float(max(0.1, detection_radius)),
                    "heading": 0.0,
                }
            )

    def step(self, actions: dict[str, Action]) -> StepResult:
        self._step_count += 1
        self._move_hazards()
        rewards: dict[str, float] = {}
        for agent_id, state in self._agents.items():
            if not state["is_alive"]:
                rewards[agent_id] = 0.0
                continue
            action = actions.get(agent_id, Action())
            was_full = state["energy"] >= state["max_energy"]
            self._move_agent(state, action)
            reward = -0.01
            if action.eat:
                reward += self._try_eat(state, was_full)
            self._apply_hazard_damage(state)
            if state["energy"] <= 0.0:
                state["health"] = max(0.0, state["health"] - 0.5)
            if state["health"] <= 0.0:
                state["is_alive"] = False
            rewards[agent_id] = reward
        states = self._state_view()
        observations = {
            agent_id: self._observation(agent_id) for agent_id in self._agents
        }
        snapshot = self.snapshot()
        terminated = not any(state.alive for state in states.values())
        return StepResult(
            MappingProxyType(states),
            MappingProxyType(observations),
            snapshot,
            terminated,
            reward=MappingProxyType(rewards),
            info=MappingProxyType({"step": self._step_count}),
        )

    def _move_agent(self, state: dict[str, Any], action: Action) -> None:
        state["heading"] = (state["heading"] + action.turn * 3.0 * self.dt) % (
            2.0 * math.pi
        )
        state["speed"] = action.thrust * 8.0
        x, y = state["position"]
        state["position"] = self._clamp_point(
            (
                x + math.cos(state["heading"]) * state["speed"] * self.dt,
                y + math.sin(state["heading"]) * state["speed"] * self.dt,
            )
        )
        state["energy"] = max(
            0.0, state["energy"] - (0.08 + 0.04 * state["speed"]) * self.dt
        )
        state["age"] += self.dt

    def _move_hazards(self) -> None:
        alive = [state for state in self._agents.values() if state["is_alive"]]
        for hazard in self._hazard_patches:
            hx, hy = hazard["position"]
            nearby = [
                state
                for state in alive
                if math.dist((hx, hy), state["position"]) <= hazard["detection_radius"]
            ]
            if nearby:
                target = min(
                    nearby, key=lambda state: math.dist((hx, hy), state["position"])
                )
                hazard["heading"] = math.atan2(
                    target["position"][1] - hy, target["position"][0] - hx
                )
            else:
                hazard["heading"] += float(self._rng.uniform(-0.4, 0.4)) * self.dt
            speed = 2.0
            hazard["position"] = self._clamp_point(
                (
                    hx + math.cos(hazard["heading"]) * speed * self.dt,
                    hy + math.sin(hazard["heading"]) * speed * self.dt,
                )
            )

    def _try_eat(self, state: dict[str, Any], was_full: bool = False) -> float:
        if was_full:
            return 0.0
        for patch in self._food_patches:
            if (
                patch["amount"] <= 0.0
                or math.dist(state["position"], patch["position"]) > 1.5
            ):
                continue
            available = min(patch["amount"], state["max_energy"] - state["energy"])
            if available <= 0.0:
                return 0.0
            state["energy"] += available
            patch["amount"] -= available
            state["food_eaten"] += 1
            return 1.0
        return 0.0

    def _apply_hazard_damage(self, state: dict[str, Any]) -> None:
        for hazard in self._hazard_patches:
            if math.dist(state["position"], hazard["position"]) <= hazard["radius"]:
                state["health"] = max(0.0, state["health"] - hazard["damage"] * self.dt)

    def _state_view(self) -> dict[str, AgentState]:
        return {
            agent_id: AgentState(
                agent_id,
                tuple(state["position"]),
                state["heading"],
                state["speed"],
                state["energy"],
                state["max_energy"],
                state["health"],
                state["age"],
                state["is_alive"],
                state["food_eaten"],
            )
            for agent_id, state in self._agents.items()
        }

    def _observation(self, agent_id: str) -> Observation:
        state = self._agents[agent_id]
        distances, types = self._ray_sensors(state["position"], state["heading"])
        return Observation(
            tuple(distances),
            tuple(types),
            (
                state["energy"] / state["max_energy"],
                state["health"] / 100.0,
                state["speed"] / 8.0,
            ),
            tuple(state["position"]),
            state["energy"],
            self._step_count,
        )

    def _ray_sensors(
        self, position: tuple[float, float], heading: float, count: int = 16
    ) -> tuple[list[float], list[int]]:
        distances: list[float] = []
        types: list[int] = []
        offsets = np.linspace(-math.radians(135), math.radians(135), count)
        offsets[count // 2] = 0.0
        for angle in heading + offsets:
            dx, dy = math.cos(float(angle)), math.sin(float(angle))
            wall_distance = self._wall_distance(position, dx, dy)
            hit_distance, hit_type = wall_distance, 3
            for patch in self._food_patches:
                distance = self._ray_point_distance(
                    position, (dx, dy), patch["position"]
                )
                if distance < hit_distance and distance <= 20.0:
                    hit_distance, hit_type = distance, 1
            for patch in self._hazard_patches:
                distance = self._ray_point_distance(
                    position, (dx, dy), patch["position"]
                )
                if distance < hit_distance and distance <= 20.0:
                    hit_distance, hit_type = distance, 2
            for other in self._agents.values():
                distance = self._ray_point_distance(
                    position, (dx, dy), other["position"]
                )
                if distance < hit_distance and distance > 1e-6 and distance <= 20.0:
                    hit_distance, hit_type = distance, 4
            distances.append(float(min(hit_distance, 20.0)))
            types.append(hit_type)
        return distances, types

    def _wall_distance(
        self, position: tuple[float, float], dx: float, dy: float
    ) -> float:
        hits: list[float] = []
        if dx > 0.0:
            hits.append((self.width - position[0]) / dx)
        elif dx < 0.0:
            hits.append(-position[0] / dx)
        if dy > 0.0:
            hits.append((self.height - position[1]) / dy)
        elif dy < 0.0:
            hits.append(-position[1] / dy)
        return min(hits)

    @staticmethod
    def _ray_point_distance(
        origin: tuple[float, float],
        direction: tuple[float, float],
        point: tuple[float, float],
    ) -> float:
        vx, vy = point[0] - origin[0], point[1] - origin[1]
        projection = vx * direction[0] + vy * direction[1]
        perpendicular = abs(vx * direction[1] - vy * direction[0])
        return projection if projection > 0.0 and perpendicular <= 0.75 else math.inf

    def snapshot(self) -> Snapshot:
        food = tuple(
            MappingProxyType(copy.deepcopy(patch)) for patch in self._food_patches
        )
        hazards = tuple(
            MappingProxyType(copy.deepcopy(patch)) for patch in self._hazard_patches
        )
        return Snapshot(
            self._step_count,
            MappingProxyType(self._state_view()),
            food,
            hazards,
            copy.deepcopy(self._rng.state),
            self.width,
            self.height,
            self.dt,
            MappingProxyType({"seed": self._seed, **copy.deepcopy(self._metadata)}),
        )

    def restore(self, snapshot: Snapshot) -> None:
        self._seed = int(snapshot.metadata.get("seed", self._seed))
        self._step_count = snapshot.step
        self._food_patches = [dict(patch) for patch in snapshot.food_patches]
        self._hazard_patches = [dict(patch) for patch in snapshot.hazard_patches]
        self._agents = {
            agent_id: {
                "position": tuple(state.position),
                "heading": state.heading,
                "speed": state.speed,
                "energy": state.energy,
                "max_energy": state.max_energy,
                "health": state.health,
                "age": state.age,
                "is_alive": state.alive,
                "food_eaten": state.food_eaten,
            }
            for agent_id, state in snapshot.agent_states.items()
        }
        self._rng = SimRNG(0)
        self._rng.state = copy.deepcopy(dict(snapshot.rng_state))
        self._metadata = {
            key: copy.deepcopy(value)
            for key, value in snapshot.metadata.items()
            if key != "seed"
        }

    def state_hash(self) -> str:
        canonical = canonicalize(self.snapshot().to_dict())
        encoded = json.dumps(
            canonical, sort_keys=True, separators=(",", ":"), allow_nan=False
        ).encode("utf-8")
        return hashlib.sha256(encoded).hexdigest()

    def _clamp_point(self, point: tuple[float, float]) -> tuple[float, float]:
        return (
            float(np.clip(point[0], 0.0, self.width)),
            float(np.clip(point[1], 0.0, self.height)),
        )

    def get_agent(self, agent_id: str) -> dict[str, Any] | None:
        return self._agents.get(agent_id)

    def get_step_count(self) -> int:
        return self._step_count

    def get_world_state(self) -> np.ndarray:
        return self.snapshot().world_state

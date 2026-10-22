"""Policy protocol and deterministic built-in policies."""

from __future__ import annotations

import math
from typing import Any, Protocol

from simlab.core.types import Action


class Policy(Protocol):
    def act(self, obs: Any, drive_state: Any = None) -> Action: ...


class HeuristicPolicy:
    def act(self, obs: Any, drive_state: Any = None) -> Action:
        return Action(0.0, 0.8, True)

    def select_action(
        self,
        agent_state: dict[str, Any],
        world_state: dict[str, Any],
        available_actions: list[Any],
        step: int,
    ) -> Action:
        position = agent_state.get("position", (0.0, 0.0))
        food = world_state.get("food_locations", [])
        if not food:
            return Action(0.25, 0.5, False)
        target = min(
            food, key=lambda p: (p[0] - position[0]) ** 2 + (p[1] - position[1]) ** 2
        )
        desired = math.atan2(target[1] - position[1], target[0] - position[0])
        heading = agent_state.get("heading", 0.0)
        delta = (desired - heading + math.pi) % (2 * math.pi) - math.pi
        return Action(max(-1.0, min(1.0, delta)), 0.8, True, "heuristic")


class RandomPolicy:
    def __init__(self, seed: int = 0) -> None:
        self._rng = __import__("numpy").random.default_rng(seed)

    def act(self, obs: Any, drive_state: Any = None) -> Action:
        return Action(
            float(self._rng.uniform(-1, 1)), float(self._rng.random()), False, "random"
        )


class LearnedPolicy:
    def act(self, obs: Any, drive_state: Any = None) -> Action:
        return Action()

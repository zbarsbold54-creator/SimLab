"""Registered policies."""

import math
from typing import Any

from simlab.core.registry import register
from simlab.core.types import Action
from simlab.interfaces.policy import HeuristicPolicy, RandomPolicy

register("policy", "heuristic")(HeuristicPolicy)
register("policy", "random")(RandomPolicy)


class OracleHeuristicPolicy:
    """World-state heuristic (kept for comparison): reads World state directly."""

    needs_observations = False

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


register("policy", "oracle_heuristic")(OracleHeuristicPolicy)

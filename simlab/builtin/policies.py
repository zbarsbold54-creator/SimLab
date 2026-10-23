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
        # Mirror the sensor HeuristicPolicy control law, using world geometry
        # instead of rays. The sensor sees no depleted patches (they are
        # filtered from observations), so the oracle must apply the same filter.
        position = agent_state.get("position", (0.0, 0.0))
        food = [
            patch
            for patch in world_state.get("food_patches", [])
            if patch[1] > 0.0
        ]
        if not food:
            return Action(0.25, 0.5, False)
        target, _amount = min(
            food, key=lambda p: (p[0][0] - position[0]) ** 2 + (p[0][1] - position[1]) ** 2
        )
        desired = math.atan2(target[1] - position[1], target[0] - position[0])
        heading = agent_state.get("heading", 0.0)
        delta = (desired - heading + math.pi) % (2 * math.pi) - math.pi
        turn = max(-1.0, min(1.0, math.degrees(delta) / 135.0))
        dist2 = (position[0] - target[0]) ** 2 + (position[1] - target[1]) ** 2
        eat = dist2 < 2.25  # matches World._try_eat contact radius (1.5)
        return Action(turn, 0.8, eat, "heuristic")


register("policy", "oracle_heuristic")(OracleHeuristicPolicy)

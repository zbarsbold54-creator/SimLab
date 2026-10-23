"""Registered policies."""

import math
from typing import Any

from simlab.core.registry import register
from simlab.core.types import Action
from simlab.interfaces.policy import (
    HeuristicPolicy,
    HeuristicPolicyNoflee,
    RandomPolicy,
)

register("policy", "heuristic")(HeuristicPolicy)
register("policy", "heuristic_noflee")(HeuristicPolicyNoflee)
register("policy", "random")(RandomPolicy)


class OracleHeuristicPolicyNoflee:
    """World-state heuristic without hazard fleeing (historic behavior)."""

    needs_observations = False
    EAT_DIST = 1.5
    CLOSE_DIST = 5.0
    THRUST_SEEK = 0.8
    THRUST_CLOSE = 0.5
    STEER_GAIN = 60.0

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
        dist2 = (position[0] - target[0]) ** 2 + (position[1] - target[1]) ** 2
        dist = math.sqrt(dist2)
        if dist < self.EAT_DIST:
            turn = max(-1.0, min(1.0, math.degrees(delta) / 135.0))
            return Action(turn, 0.6, True, "heuristic")
        if dist < self.CLOSE_DIST:
            turn = max(-1.0, min(1.0, math.degrees(delta) / self.STEER_GAIN))
            return Action(turn, self.THRUST_CLOSE, False, "heuristic")
        turn = max(-1.0, min(1.0, math.degrees(delta) / 135.0))
        return Action(turn, self.THRUST_SEEK, False, "heuristic")


class OracleHeuristicPolicy(OracleHeuristicPolicyNoflee):
    """Default world-state heuristic: flee hazards when threat > 0.3."""

    def select_action(
        self,
        agent_state: dict[str, Any],
        world_state: dict[str, Any],
        available_actions: list[Any],
        step: int,
    ) -> Action:
        position = agent_state.get("position", (0.0, 0.0))
        hazards = world_state.get("hazard_locations", [])
        nearest = None
        for hx, hy in hazards:
            dist2 = (hx - position[0]) ** 2 + (hy - position[1]) ** 2
            if nearest is None or dist2 < nearest:
                nearest = dist2
        threat = 1.0 - (math.sqrt(nearest) if nearest is not None else 20.0) / 20.0
        if threat > 0.3:
            hx, hy = min(
                hazards,
                key=lambda h: (h[0] - position[0]) ** 2 + (h[1] - position[1]) ** 2,
            )
            desired = math.atan2(position[1] - hy, position[0] - hx)  # away
            heading = agent_state.get("heading", 0.0)
            delta = (desired - heading + math.pi) % (2 * math.pi) - math.pi
            turn = max(-1.0, min(1.0, math.degrees(delta) / 135.0))
            return Action(turn, 0.8, False, "oracle_flee")
        return super().select_action(agent_state, world_state, available_actions, step)


register("policy", "oracle_noflee")(OracleHeuristicPolicyNoflee)
register("policy", "oracle_heuristic")(OracleHeuristicPolicyNoflee)
register("policy", "oracle")(OracleHeuristicPolicy)

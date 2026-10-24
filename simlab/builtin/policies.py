"""Registered policies."""

import math
from typing import Any

from simlab.core.registry import register
from simlab.core.types import Action
from simlab.interfaces.policy import (
    HeuristicPolicy,
    HeuristicPolicyNoflee,
    HeuristicPolicyV2,
    RandomPolicy,
    _clamp,
    derive_eat_thrust,
)

register("policy", "heuristic")(HeuristicPolicy)
register("policy", "heuristic_noflee")(HeuristicPolicyNoflee)
register("policy", "heuristic_v2")(HeuristicPolicyV2)
register("policy", "random")(RandomPolicy)


class OracleHeuristicPolicyNoflee:
    """World-state heuristic without hazard fleeing (historic behavior)."""

    needs_observations = False
    CLOSE_DIST = 5.0
    THRUST_SEEK = 0.8
    STEER_GAIN = 60.0

    def __init__(self, seed: int = 0, config: Any = None) -> None:
        food = getattr(config, "food", None)
        agent = getattr(config, "agent", None)
        self.eat_dist = float(getattr(food, "eat_dist", 1.5))
        self.max_turn_rate = float(getattr(agent, "max_turn_rate", 3.0))
        self.speed_factor = float(getattr(agent, "speed_factor", 8.0))
        self.eat_thrust = derive_eat_thrust(
            self.eat_dist, self.max_turn_rate, self.speed_factor
        )
        self.close_thrust = self.eat_thrust
        self.eat_steer_divisor = self.STEER_GAIN
        self.close_steer_divisor = self.STEER_GAIN

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
        if dist < self.eat_dist:
            turn = _clamp(math.degrees(delta) / self.eat_steer_divisor)
            return Action(turn, self.eat_thrust, True, "heuristic")
        if dist < self.CLOSE_DIST:
            turn = _clamp(math.degrees(delta) / self.close_steer_divisor)
            return Action(turn, self.close_thrust, False, "heuristic")
        turn = _clamp(math.degrees(delta) / 135.0)
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
            turn = _clamp(math.degrees(delta) / 135.0)
            return Action(turn, 0.8, False, "oracle_flee")
        return super().select_action(agent_state, world_state, available_actions, step)


class OracleHeuristicPolicyV2(OracleHeuristicPolicy):
    """World-state heuristic from the world-v1 era (pre principled-thrust).

    Eat branch used thrust 0.6 with steer /135; close branch thrust 0.5.
    Frozen for reproduction of the world-v1 baseline table.
    """

    def __init__(self, seed: int = 0, config: Any = None) -> None:
        super().__init__(seed, config)
        self.eat_thrust = 0.6
        self.close_thrust = 0.5
        self.eat_steer_divisor = 135.0


register("policy", "oracle_noflee")(OracleHeuristicPolicyNoflee)
register("policy", "oracle_heuristic")(OracleHeuristicPolicyNoflee)
register("policy", "oracle")(OracleHeuristicPolicy)
register("policy", "oracle_v2")(OracleHeuristicPolicyV2)

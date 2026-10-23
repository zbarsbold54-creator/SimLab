"""Policy protocol and deterministic built-in policies."""

from __future__ import annotations

from typing import Any, Protocol

from simlab.core.types import Action


class Policy(Protocol):
    needs_observations: bool

    def act(self, obs: Any, drive_state: Any = None) -> Action: ...


class HeuristicPolicyNoflee:
    """Sensor-based heuristic without hazard fleeing (historic behavior).

    Uses only obs (rays + proprioception), no World access.
    """

    needs_observations = True

    EAT_DIST = 1.5
    CLOSE_DIST = 5.0
    THRUST_SEEK = 0.8
    THRUST_CLOSE = 0.5
    STEER_GAIN = 60.0

    @staticmethod
    def _best_food(rays, ray_types):
        best_dist = None
        best_index = None
        for index, (distance, kind) in enumerate(zip(rays, ray_types)):
            if kind == 1 and (best_dist is None or distance < best_dist):
                best_dist = distance
                best_index = index
        return best_dist, best_index

    def _act(self, obs):
        if obs is None:
            return Action(0.25, 0.5, False)
        rays = getattr(obs, "rays", ())
        ray_types = getattr(obs, "ray_types", ())
        best_dist, best_index = self._best_food(rays, ray_types)
        if best_dist is None or best_index is None:
            return Action(0.25, 0.5, False)

        count = len(rays)
        middle = count // 2
        offset_deg = (
            (best_index - middle) * (270.0 / (count - 1)) if count > 1 else 0.0
        )

        if best_dist < self.EAT_DIST:
            turn = max(-1.0, min(1.0, offset_deg / 135.0))
            return Action(turn, 0.6, True, "heuristic")
        if best_dist < self.CLOSE_DIST:
            # Near food: steer harder and ease off so the agent converges
            # instead of orbiting the patch just outside eat range.
            turn = max(-1.0, min(1.0, offset_deg / self.STEER_GAIN))
            return Action(turn, self.THRUST_CLOSE, False, "heuristic")
        turn = max(-1.0, min(1.0, offset_deg / 135.0))
        return Action(turn, self.THRUST_SEEK, False, "heuristic")

    def act(self, obs: Any, drive_state: Any = None) -> Action:
        return self._act(obs)


class HeuristicPolicy(HeuristicPolicyNoflee):
    """Default sensor heuristic: flee hazards when threat > 0.3, else seek/wander."""

    @staticmethod
    def _threat(rays, ray_types) -> float:
        hazards = [
            distance
            for distance, kind in zip(rays, ray_types)
            if kind == 2
        ]
        nearest = min(hazards) if hazards else 20.0
        return max(0.0, min(1.0, 1.0 - nearest / 20.0))

    def act(self, obs: Any, drive_state: Any = None) -> Action:
        if obs is None:
            return Action(0.25, 0.5, False)
        rays = getattr(obs, "rays", ())
        ray_types = getattr(obs, "ray_types", ())
        if self._threat(rays, ray_types) > 0.3:
            count = len(rays)
            middle = count // 2
            best = -1
            nearest = None
            for index, (distance, kind) in enumerate(zip(rays, ray_types)):
                if kind == 2 and (nearest is None or distance < nearest):
                    nearest = distance
                    best = index
            offset_deg = (
                (best - middle) * (270.0 / (count - 1)) if count > 1 else 0.0
            )
            if abs(offset_deg) < 15.0:
                turn = -1.0
            else:
                turn = max(-1.0, min(1.0, -offset_deg / 135.0))
            return Action(turn, 0.8, False, "heuristic_flee")
        return self._act(obs)


class RandomPolicy:
    needs_observations = False

    def __init__(self, seed: int = 0) -> None:
        self._rng = __import__("numpy").random.default_rng(seed)

    def act(self, obs: Any, drive_state: Any = None) -> Action:
        return Action(
            float(self._rng.uniform(-1, 1)), float(self._rng.random()), False, "random"
        )


class LearnedPolicy:
    needs_observations = False

    def act(self, obs: Any, drive_state: Any = None) -> Action:
        return Action()

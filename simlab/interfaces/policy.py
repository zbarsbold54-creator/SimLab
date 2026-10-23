"""Policy protocol and deterministic built-in policies."""

from __future__ import annotations

from typing import Any, Protocol

from simlab.core.types import Action


class Policy(Protocol):
    needs_observations: bool

    def act(self, obs: Any, drive_state: Any = None) -> Action: ...


class HeuristicPolicy:
    """Sensor-based heuristic: uses only obs (rays + proprioception), no World access."""

    needs_observations = True

    def act(self, obs: Any, drive_state: Any = None) -> Action:
        if obs is None:
            return Action(0.25, 0.5, False)
        rays = getattr(obs, "rays", ())
        ray_types = getattr(obs, "ray_types", ())

        # Find the nearest food ray (type code 1).
        best_dist = None
        best_index = None
        for index, (distance, kind) in enumerate(zip(rays, ray_types)):
            if kind == 1 and (best_dist is None or distance < best_dist):
                best_dist = distance
                best_index = index

        if best_dist is None:
            # No food visible: wander forward.
            return Action(0.25, 0.5, False)

        count = len(rays)
        middle = count // 2
        offset_deg = (
            (best_index - middle) * (270.0 / (count - 1)) if count > 1 else 0.0
        )
        turn = max(-1.0, min(1.0, offset_deg / 135.0))
        thrust = 0.8
        eat = best_dist < 1.5  # eat range matches World._try_eat contact radius
        return Action(turn, thrust, eat, "heuristic")


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

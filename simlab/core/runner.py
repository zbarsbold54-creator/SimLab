"""Headless simulation runner."""

from __future__ import annotations

import math
import time
from collections.abc import Callable
from typing import Any

from simlab.core.config import SimLabConfig, load_config
from simlab.core.snapshot import ActionLog, SnapshotManager
from simlab.core.types import Action, AgentState, Snapshot, StepResult
from simlab.core.world import World


class Runner:
    def __init__(
        self,
        config: SimLabConfig | None = None,
        seed: int | None = None,
        rng: Any = None,
        agents: int = 1,
    ) -> None:
        self._config = config or load_config()
        self._seed = self._config.world.seed if seed is None else seed
        self._agent_count = max(1, int(agents))
        self._world = World(
            self._config.world.width, self._config.world.height, self._seed
        )
        self._step_count = 0
        self._max_steps = 100000
        self._is_running = False
        self._snapshot_manager = SnapshotManager()
        self._snapshots: dict[int, Snapshot] = self._snapshot_manager._snapshots
        self.action_log: list[dict[str, Action]] = []

    def setup(self) -> None:
        self._world.reset(self._seed)
        self._step_count = 0
        self._snapshots.clear()
        self.action_log.clear()
        for index in range(self._agent_count):
            x = float(self._world._rng.uniform(10.0, self._world.width - 10.0))
            y = float(self._world._rng.uniform(10.0, self._world.height - 10.0))
            heading = float(self._world._rng.uniform(-3.14, 3.14))
            self._world.add_agent(
                f"agent_{index}",
                (x, y),
                self._config.agent.initial_energy,
                heading=heading,
            )
        count = min(
            self._config.food.max_food,
            max(
                1,
                int(
                    self._world.width
                    * self._world.height
                    * self._config.food.spawn_rate
                    / 10
                ),
            ),
        )
        for _ in range(count):
            self._world.add_food(
                (
                    float(self._world._rng.uniform(2, self._world.width - 2)),
                    float(self._world._rng.uniform(2, self._world.height - 2)),
                )
            )
        for _ in range(self._config.hazard.count):
            self._world.add_hazard(
                (
                    float(self._world._rng.uniform(2, self._world.width - 2)),
                    float(self._world._rng.uniform(2, self._world.height - 2)),
                ),
                damage=self._config.hazard.damage_per_step,
                detection_radius=self._config.hazard.detection_radius,
            )
        self._is_running = True
        self._snapshot_manager.add(self._world.snapshot())

    def _default_action(self, aid: str) -> Action:
        state = self._world.get_agent(aid)
        if state is None or not state["is_alive"]:
            return Action()
        if self._world._food_patches:
            target = min(
                self._world._food_patches,
                key=lambda patch: (
                    (patch["position"][0] - state["position"][0]) ** 2
                    + (patch["position"][1] - state["position"][1]) ** 2
                ),
            )
            desired = math.atan2(
                target["position"][1] - state["position"][1],
                target["position"][0] - state["position"][0],
            )
            delta = (desired - state["heading"] + math.pi) % (2 * math.pi) - math.pi
            return Action(float(max(-1.0, min(1.0, delta))), 0.8, True)
        return Action(0.25, 0.5, False)

    def run_step(self, agent_actions: dict[str, Action] | None = None) -> StepResult:
        if not self._is_running:
            raise RuntimeError("Runner is not running. Call setup() first.")
        actions = (
            dict(agent_actions)
            if agent_actions
            else {aid: self._default_action(aid) for aid in self._world._agents}
        )
        action_index = self._step_count
        result = self._world.step(actions)
        self._step_count = self._world.get_step_count()
        if action_index < len(self.action_log):
            self.action_log[action_index] = actions
        else:
            self.action_log.append(actions)
        self._snapshot_manager.add(result.snapshot)
        return result

    def run_episode(
        self,
        agent_actions_generator: Callable[
            [int, dict[str, AgentState]], dict[str, Action]
        ]
        | None = None,
    ) -> list[StepResult]:
        results = []
        while self._is_running and self._step_count < self._max_steps:
            actions = (
                agent_actions_generator(
                    self._step_count, dict(self._world._state_view())
                )
                if agent_actions_generator
                else None
            )
            result = self.run_step(actions)
            results.append(result)
            if result.terminated:
                break
        return results

    def run_bench(self, num_steps: int) -> list[StepResult]:
        started = time.perf_counter()
        results = [self.run_step() for _ in range(num_steps)]
        self.last_bench_steps_per_second = num_steps / max(
            time.perf_counter() - started, 1e-9
        )
        return results

    def snapshot(self) -> Snapshot | None:
        return self._world.snapshot() if self._is_running else None

    def restore(self, snapshot: Snapshot, preserve_action_log: bool = True) -> None:
        """Restore a checkpoint while optionally retaining recorded actions."""
        self._world.restore(snapshot)
        self._world.width = snapshot.width
        self._world.height = snapshot.height
        self._world.dt = snapshot.dt
        self._step_count = snapshot.step
        self._is_running = True
        if not preserve_action_log:
            self.action_log.clear()
            self._snapshots.clear()
        self._snapshot_manager.add(snapshot)

    def export_action_log(self) -> ActionLog:
        return ActionLog.from_steps(self._seed, self.action_log)

    def reset(self) -> None:
        self._is_running = False
        self._world.reset(self._seed)
        self._step_count = 0
        self._snapshots.clear()
        self.action_log.clear()

    def stop(self) -> None:
        self._is_running = False

    def get_stats(self) -> dict[str, Any]:
        return {
            "step_count": self._step_count,
            "is_running": self._is_running,
            "world_size": f"{self._world.width:g}x{self._world.height:g}",
        }


class SimEnv:
    """Thin single-agent environment with Gymnasium-like signatures."""

    def __init__(
        self, config: SimLabConfig | None = None, seed: int | None = None
    ) -> None:
        self.runner = Runner(config, seed)

    def reset(self, seed: int | None = None) -> tuple[Any, dict[str, Any]]:
        self.runner = Runner(
            self.runner._config, self.runner._seed if seed is None else seed
        )
        self.runner.setup()
        result = self.runner._world.snapshot()
        return result.agent_states, {"seed": self.runner._seed}

    def step(self, action: Action) -> tuple[Any, float, bool, bool, dict[str, Any]]:
        result = self.runner.run_step({"agent_0": action})
        reward = (
            float(result.reward.get("agent_0", 0.0))
            if hasattr(result.reward, "get")
            else float(result.reward)
        )
        return (
            result.observations.get("agent_0"),
            reward,
            result.terminated,
            result.truncated,
            dict(result.info),
        )

"""Simulation worker hosted by a dedicated QThread."""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass
from types import MappingProxyType

from PySide6.QtCore import QObject, Signal, Slot

from simlab.core.config import SimLabConfig, load_config
from simlab.core.registry import get, load_plugins
from simlab.core.registry import list_ as registry_list
from simlab.core.runner import Runner
from simlab.core.types import Snapshot
from simlab.gui import theme


@dataclass(frozen=True)
class AgentFrame:
    action: object
    observation: object
    drives: tuple[float, ...]


@dataclass(frozen=True)
class SimulationFrame:
    snapshot: Snapshot
    agents: object
    drive_names: tuple[str, ...]


class SimWorker(QObject):
    """Runs simulation steps off the GUI thread and publishes immutable state."""

    snapshotReady = Signal(object)
    frameReady = Signal(object)
    metricsReady = Signal(float, float, int, int)
    stopped = Signal()

    def __init__(
        self,
        config: SimLabConfig | None = None,
        seed: int = 1,
        agents: int | None = None,
    ) -> None:
        super().__init__()
        self._config = config or load_config()
        self._seed = int(seed)
        self._agent_count = max(
            1, int(self._config.agent.count if agents is None else agents)
        )
        self._runner = Runner(self._config, self._seed, agents=self._agent_count)
        self._runner.setup()
        self._lock = threading.RLock()
        self._wake = threading.Event()
        self._playing = True
        self._stopping = False
        self._step_budget = 0
        self._speed = theme.DEFAULT_SPEED
        self._requested_seed: int | None = None
        self._requested_snapshot: SimulationFrame | None = None
        self._requested_scenario: tuple[int, SimLabConfig] | None = None
        self._steps_per_second = 0.0
        self._latest_snapshot = self._runner._world.snapshot()
        self._policy_name = self._config.policy
        self._drive_name = self._config.drive
        self._drive_core = None
        self._policy = None
        self._latest_frame = None
        self._initialize_components()

    def _initialize_components(self) -> None:
        load_plugins()
        policy_type = get("policy", self._policy_name)
        drive_type = get("drive", self._drive_name)
        if policy_type is None:
            self._policy = None
        else:
            try:
                self._policy = policy_type(seed=self._seed)
            except TypeError:
                self._policy = policy_type()
        self._drive_core = drive_type() if drive_type else None
        if self._drive_core is not None:
            self._drive_core.reset(self._agent_count, self._runner._world._rng)
        self._latest_frame = SimulationFrame(
            self._latest_snapshot,
            MappingProxyType({}),
            tuple(getattr(self._drive_core, "names", ())),
        )

    @property
    def latest_frame(self) -> SimulationFrame:
        with self._lock:
            return self._latest_frame

    @property
    def registry_options(self) -> tuple[tuple[str, ...], tuple[str, ...]]:
        load_plugins()
        return tuple(registry_list("policy")), tuple(registry_list("drive"))

    @property
    def latest_snapshot(self) -> Snapshot:
        with self._lock:
            return self._latest_snapshot

    @property
    def seed(self) -> int:
        with self._lock:
            return self._seed

    @property
    def steps_per_second(self) -> float:
        with self._lock:
            return self._steps_per_second

    @property
    def target_steps_per_second(self) -> float | None:
        with self._lock:
            speed = self._speed
        if speed == "Max":
            return None
        return theme.BASE_STEPS_PER_SECOND * float(speed.removesuffix("x"))

    def set_playing(self, playing: bool) -> None:
        with self._lock:
            self._playing = playing
        self._wake.set()

    def set_speed(self, speed: str) -> None:
        with self._lock:
            self._speed = speed
        self._wake.set()

    def request_step(self) -> None:
        self.request_steps(1)

    def request_steps(self, count: int) -> None:
        with self._lock:
            self._step_budget += max(0, int(count))
        self._wake.set()

    def request_reset(self, seed: int) -> None:
        with self._lock:
            self._requested_seed = int(seed)
        self._wake.set()

    def request_scenario(self, seed: int, config: SimLabConfig) -> None:
        with self._lock:
            self._requested_scenario = (int(seed), config)
        self._wake.set()

    def request_restore(self, frame: SimulationFrame) -> None:
        with self._lock:
            self._requested_snapshot = frame
            self._playing = False
        self._wake.set()

    def stop(self) -> None:
        with self._lock:
            self._stopping = True
        self._wake.set()

    @Slot()
    def run(self) -> None:
        previous = time.perf_counter()
        last_publish = previous
        last_metrics = previous
        window_steps = 0
        step_credit = 0.0
        self.snapshotReady.emit(self.latest_snapshot)
        self.frameReady.emit(self.latest_frame)

        while True:
            now = time.perf_counter()
            elapsed = now - previous
            previous = now
            with self._lock:
                if self._stopping:
                    break
                requested_seed = self._requested_seed
                self._requested_seed = None
                requested_snapshot = self._requested_snapshot
                self._requested_snapshot = None
                requested_scenario = self._requested_scenario
                self._requested_scenario = None
                playing = self._playing
                speed = self._speed
                step_budget = self._step_budget
                self._step_budget = 0

            if requested_scenario is not None:
                requested_seed, self._config = requested_scenario
                self._agent_count = max(1, int(self._config.agent.count))
                with self._lock:
                    self._seed = requested_seed
                    self._runner = Runner(
                        self._config, requested_seed, agents=self._agent_count
                    )
                    self._runner.setup()
                    self._latest_snapshot = self._runner._world.snapshot()
                    self._initialize_components()
                    step_credit = 0.0
                    window_steps = 0
                self.snapshotReady.emit(self.latest_snapshot)
                self.frameReady.emit(self.latest_frame)
            elif requested_seed is not None:
                with self._lock:
                    self._seed = requested_seed
                    self._runner = Runner(
                        self._config, requested_seed, agents=self._agent_count
                    )
                    self._runner.setup()
                    self._latest_snapshot = self._runner._world.snapshot()
                    self._initialize_components()
                    step_credit = 0.0
                    window_steps = 0
                self.snapshotReady.emit(self.latest_snapshot)
                self.frameReady.emit(self.latest_frame)

            if requested_snapshot is not None:
                with self._lock:
                    self._runner._world.restore(requested_snapshot.snapshot)
                    self._runner._step_count = requested_snapshot.snapshot.step
                    self._latest_snapshot = requested_snapshot.snapshot
                    self._latest_frame = requested_snapshot
                self.snapshotReady.emit(requested_snapshot.snapshot)
                self.frameReady.emit(self._latest_frame)

            if playing:
                if speed == "Max":
                    steps_to_run = theme.MAX_BATCH_STEPS
                else:
                    multiplier = float(speed.removesuffix("x"))
                    step_credit += elapsed * theme.BASE_STEPS_PER_SECOND * multiplier
                    steps_to_run = int(step_credit)
                    step_credit -= steps_to_run
                    if steps_to_run == 0 and step_budget == 0:
                        self._wake.wait(0.002)
                        self._wake.clear()
                        continue
            else:
                steps_to_run = 0
                if step_budget == 0:
                    self._wake.wait(0.05)
                    self._wake.clear()
                    continue

            steps_to_run += step_budget
            for _ in range(steps_to_run):
                with self._lock:
                    if self._stopping:
                        break
                    world = self._runner._world
                    observations = {
                        aid: world._observation(aid) for aid in world._agents
                    }
                    live_ids = [
                        aid for aid, state in world._agents.items() if state["is_alive"]
                    ]
                    live_observations = [observations[aid] for aid in live_ids]
                    drive_values = (
                        self._drive_core.step(live_observations, {"rng": world._rng})
                        if self._drive_core is not None
                        else []
                    )
                    drive_by_agent = {
                        aid: tuple(float(value) for value in drive_values[index])
                        for index, aid in enumerate(live_ids)
                    }
                    actions = {}
                    for aid in live_ids:
                        drives = drive_by_agent.get(aid, ())
                        if hasattr(self._policy, "select_action"):
                            state = world._agents[aid]
                            action = self._policy.select_action(
                                state,
                                {
                                    "food_locations": [
                                        p["position"] for p in world._food_patches
                                    ]
                                },
                                [],
                                self._runner._step_count,
                            )
                        elif self._policy is not None:
                            action = self._policy.act(observations[aid], drives)
                        else:
                            action = self._runner._default_action(aid)
                        actions[aid] = action
                    result = self._runner.run_step(actions)
                    self._latest_snapshot = result.snapshot
                    details = {
                        aid: AgentFrame(
                            actions.get(aid),
                            result.observations.get(aid),
                            drive_by_agent.get(aid, ()),
                        )
                        for aid in result.snapshot.agent_states
                    }
                    self._latest_frame = SimulationFrame(
                        result.snapshot,
                        MappingProxyType(details),
                        tuple(getattr(self._drive_core, "names", ())),
                    )
                window_steps += 1
                if result.terminated:
                    with self._lock:
                        self._playing = False
                    break

            now = time.perf_counter()
            if now - last_publish >= theme.SNAPSHOT_EMIT_INTERVAL_S:
                self.snapshotReady.emit(self.latest_snapshot)
                self.frameReady.emit(self.latest_frame)
                last_publish = now
            if now - last_metrics >= theme.METRICS_INTERVAL_S:
                rate = window_steps / max(now - last_metrics, 1e-9)
                with self._lock:
                    self._steps_per_second = rate
                    current_step = self._runner._step_count
                    current_seed = self._seed
                self.metricsReady.emit(
                    rate, float(now - last_metrics), current_step, current_seed
                )
                window_steps = 0
                last_metrics = now
            if speed == "Max":
                time.sleep(theme.MAX_BATCH_YIELD_S)

        self.stopped.emit()

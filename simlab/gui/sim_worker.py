"""Simulation worker hosted by a dedicated QThread."""

from __future__ import annotations

import threading
import time

from PySide6.QtCore import QObject, Signal, Slot

from simlab.core.config import SimLabConfig, load_config
from simlab.core.runner import Runner
from simlab.core.types import Snapshot
from simlab.gui import theme


class SimWorker(QObject):
    """Runs simulation steps off the GUI thread and publishes immutable state."""

    snapshotReady = Signal(object)
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
        self._steps_per_second = 0.0
        self._latest_snapshot = self._runner._world.snapshot()

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
        with self._lock:
            self._step_budget += 1
        self._wake.set()

    def request_reset(self, seed: int) -> None:
        with self._lock:
            self._requested_seed = int(seed)
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

        while True:
            now = time.perf_counter()
            elapsed = now - previous
            previous = now
            with self._lock:
                if self._stopping:
                    break
                requested_seed = self._requested_seed
                self._requested_seed = None
                playing = self._playing
                speed = self._speed
                step_budget = self._step_budget
                self._step_budget = 0

            if requested_seed is not None:
                with self._lock:
                    self._seed = requested_seed
                    self._runner = Runner(
                        self._config, requested_seed, agents=self._agent_count
                    )
                    self._runner.setup()
                    self._latest_snapshot = self._runner._world.snapshot()
                    step_credit = 0.0
                    window_steps = 0
                self.snapshotReady.emit(self.latest_snapshot)

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
                    result = self._runner.run_step()
                    self._latest_snapshot = result.snapshot
                window_steps += 1
                if result.terminated:
                    with self._lock:
                        self._playing = False
                    break

            now = time.perf_counter()
            if now - last_publish >= theme.SNAPSHOT_EMIT_INTERVAL_S:
                self.snapshotReady.emit(self.latest_snapshot)
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

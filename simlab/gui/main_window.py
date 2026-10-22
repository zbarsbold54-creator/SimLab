"""Main Simulation Studio window."""

from __future__ import annotations

import time
from pathlib import Path

from PySide6.QtCore import Qt, QThread, QTimer
from PySide6.QtGui import QFont
from PySide6.QtWidgets import QApplication, QLabel, QMainWindow, QWidget

from simlab.core.config import load_config
from simlab.gui import theme
from simlab.gui.sim_worker import SimWorker
from simlab.gui.toolbar import SimulationToolbar
from simlab.gui.world_view import WorldView


class MainWindow(QMainWindow):
    def __init__(self, seed: int = 1, agent_count: int | None = None) -> None:
        super().__init__()
        stylesheet_path = Path(__file__).with_name("style.qss")
        stylesheet = stylesheet_path.read_text(encoding="utf-8")
        chevron_path = (stylesheet_path.parent / "icons" / "chevron.svg").as_posix()
        stylesheet = stylesheet.replace(
            "url(icons/chevron.svg)", f'url("{chevron_path}")'
        )
        application = QApplication.instance()
        if application is not None:
            application.setStyleSheet(stylesheet)
            application.setFont(QFont(theme.UI_FONT, theme.UI_FONT_SIZE_PT))
        self.setWindowTitle("SimLab - Simulation Studio")
        self.resize(theme.WINDOW_WIDTH, theme.WINDOW_HEIGHT)
        self.setMinimumSize(theme.WINDOW_MIN_WIDTH, theme.WINDOW_MIN_HEIGHT)
        self.toolbar = SimulationToolbar(self)
        self.addToolBar(self.toolbar)
        self.world_view = WorldView(self)
        self.setCentralWidget(self.world_view)
        self._render_frames = 0
        self._render_fps = 0.0
        self._last_fps_time = time.perf_counter()
        self._step_rate = 0.0
        self._step = 0
        self._seed = seed
        self._thread = QThread(self)
        config = load_config()
        self.worker = SimWorker(config, seed, agents=agent_count)
        self.worker.moveToThread(self._thread)
        self._thread.started.connect(self.worker.run)
        self.worker.snapshotReady.connect(self._accept_snapshot)
        self.worker.metricsReady.connect(self._accept_metrics)
        self.worker.stopped.connect(self._thread.quit)
        self.toolbar.playPause.connect(
            self.worker.set_playing, Qt.ConnectionType.DirectConnection
        )
        self.toolbar.singleStep.connect(
            self.worker.request_step, Qt.ConnectionType.DirectConnection
        )
        self.toolbar.speedChanged.connect(
            self.worker.set_speed, Qt.ConnectionType.DirectConnection
        )
        self.toolbar.resetRequested.connect(
            self.worker.request_reset, Qt.ConnectionType.DirectConnection
        )
        self._timer = QTimer(self)
        self._timer.setInterval(theme.TIMER_INTERVAL_MS)
        self._timer.timeout.connect(self._render_latest)
        self._thread.start()
        self._timer.start()
        self.status_label = QLabel("Starting simulation…")
        status_bar = self.statusBar()
        status_bar.layout().setContentsMargins(
            theme.STATUS_TEXT_INSET,
            0,
            theme.STATUS_TEXT_INSET,
            0,
        )
        self._status_inset = QWidget()
        self._status_inset.setObjectName("statusInset")
        self._status_inset.setFixedWidth(theme.STATUS_TEXT_INSET)
        status_bar.addWidget(self._status_inset)
        status_bar.addWidget(self.status_label, 1)

    def _accept_snapshot(self, snapshot) -> None:
        self._step = snapshot.step
        self._seed = self.worker.seed
        self.world_view.set_snapshot(snapshot)

    def _accept_metrics(
        self, steps_per_second: float, elapsed: float, step: int, seed: int
    ) -> None:
        del elapsed
        self._step_rate = steps_per_second
        self._step = step
        self._seed = seed

    def _render_latest(self) -> None:
        snapshot = self.worker.latest_snapshot
        self.world_view.set_snapshot(snapshot)
        self._render_frames += 1
        now = time.perf_counter()
        elapsed = now - self._last_fps_time
        if elapsed >= 1.0:
            self._render_fps = self._render_frames / elapsed
            self._render_frames = 0
            self._last_fps_time = now
        sim_time = self._step * snapshot.dt
        self.status_label.setText(
            f"Step {self._step} | Sim time {sim_time:.1f}s | "
            f"Render {self._render_fps:.1f} FPS | Sim {self._step_rate:.1f} steps/s | Seed {self._seed}"
        )

    def closeEvent(self, event) -> None:
        self._timer.stop()
        self.worker.stop()
        self._thread.quit()
        self._thread.wait(5000)
        super().closeEvent(event)

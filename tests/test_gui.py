"""Offscreen GUI smoke checks without pytest-qt."""

from __future__ import annotations

import os
import subprocess
import sys
import time

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtWidgets import QApplication

from simlab.gui import theme
from simlab.gui.main_window import MainWindow
from simlab.gui.sim_worker import SimWorker
from simlab.gui.toolbar import SimulationToolbar


def test_core_import_does_not_load_qt() -> None:
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            "import simlab.core, sys; print('PySide6' in sys.modules)",
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    assert result.stdout.strip() == "False"


def test_snapshots_are_deeply_immutable() -> None:
    from simlab.core.world import World

    snapshot = World(seed=8).snapshot()
    try:
        snapshot.rng_state["state"]["state"] = 0
    except TypeError:
        pass
    else:
        raise AssertionError("nested snapshot RNG state should be immutable")


def test_speed_selector_targets_requested_simulation_rate() -> None:
    app = QApplication.instance() or QApplication([])
    del app
    worker = SimWorker(agents=1)
    toolbar = SimulationToolbar()
    toolbar.speed.currentTextChanged.connect(worker.set_speed)
    assert toolbar.speed.currentText() == theme.DEFAULT_SPEED
    for speed, expected_rate in (
        ("0.25x", 2.5),
        ("0.5x", 5.0),
        ("1x", 10.0),
        ("2x", 20.0),
        ("5x", 50.0),
        ("10x", 100.0),
    ):
        toolbar.speed.setCurrentText(speed)
        assert worker.target_steps_per_second == expected_rate
    assert worker._agent_count == 1


def test_default_agent_count_is_configured_in_toml() -> None:
    from simlab.core.config import load_config

    assert load_config().agent.count == 5
    assert SimWorker()._agent_count == 5


def test_food_draw_radius_scales_with_amount() -> None:
    from simlab.gui.world_view import WorldView

    radii = [
        WorldView.food_draw_radius(amount)
        for amount in theme.FOOD_RADIUS_SAMPLE_AMOUNTS
    ]
    assert radii == sorted(radii)
    assert len(set(radii)) == len(radii)


def test_detection_ring_requires_hazard_hover_or_agent_in_range() -> None:
    from simlab.core.world import World
    from simlab.gui.world_view import WorldView

    world = World(seed=2)
    world.add_agent("agent_0", (10.0, 10.0))
    world.add_hazard((11.0, 10.0), detection_radius=5.0)
    snapshot = world.snapshot()
    view = WorldView()
    hazard = snapshot.hazard_patches[0]
    selected = snapshot.agent_states["agent_0"]
    assert not view._detection_ring_visible(0, hazard, None)
    assert view._detection_ring_visible(0, hazard, selected)
    view._hovered_hazard = 0
    assert view._detection_ring_visible(0, hazard, None)


def test_window_worker_steps_and_closes_cleanly() -> None:
    app = QApplication.instance() or QApplication([])
    window = MainWindow(seed=3, agent_count=2)
    window.worker.set_playing(False)
    window.show()
    window.worker.request_step()
    deadline = time.monotonic() + 3.0
    while window.worker.latest_snapshot.step < 1 and time.monotonic() < deadline:
        app.processEvents()
        time.sleep(0.01)
    assert window.worker.latest_snapshot.step >= 1
    for _ in range(2):
        window.worker.request_step()
    deadline = time.monotonic() + 3.0
    while window.worker.latest_snapshot.step < 3 and time.monotonic() < deadline:
        app.processEvents()
        time.sleep(0.01)
    assert window.worker.latest_snapshot.step >= 3
    thread = window._thread
    window.close()
    app.processEvents()
    assert not thread.isRunning()

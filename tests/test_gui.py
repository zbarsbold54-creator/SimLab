"""Offscreen GUI smoke checks without pytest-qt."""

from __future__ import annotations

import os
import subprocess
import sys
import time

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtWidgets import QApplication

from simlab.gui.main_window import MainWindow


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

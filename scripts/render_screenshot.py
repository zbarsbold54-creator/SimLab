"""Render the SimLab GUI to the Phase 3 review screenshot."""

from __future__ import annotations

import os
import sys
import time
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from PySide6.QtWidgets import QApplication

from simlab.gui.app import MainWindow


def main() -> None:
    app = QApplication.instance() or QApplication([])
    window = MainWindow(seed=1, agent_count=20)
    window.worker.set_playing(False)
    window.show()
    deadline = time.monotonic() + 0.25
    while time.monotonic() < deadline:
        app.processEvents()
        time.sleep(0.01)
    destination = (
        Path(__file__).resolve().parents[1] / "docs" / "screenshots" / "phase3.png"
    )
    destination.parent.mkdir(parents=True, exist_ok=True)
    if not window.grab().save(str(destination)):
        raise RuntimeError(f"could not save screenshot to {destination}")
    window.close()
    app.processEvents()
    print(destination)


if __name__ == "__main__":
    main()

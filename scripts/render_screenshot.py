"""Render the SimLab GUI to the Phase 3 review screenshot."""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from PySide6.QtWidgets import QApplication

from simlab.gui import theme
from simlab.gui.app import MainWindow


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--agents", type=int, default=None)
    parser.add_argument("--output", default="docs/screenshots/phase3.png")
    args = parser.parse_args()
    app = QApplication.instance() or QApplication([])
    window = MainWindow(seed=1, agent_count=args.agents)
    window.worker.set_playing(False)
    snapshot = window.worker.latest_snapshot
    for index, amount in enumerate(theme.FOOD_RADIUS_SAMPLE_AMOUNTS):
        radius = window.world_view.food_draw_radius(amount)
        print(
            f"food_patch_sample={index} amount={amount:.2f} drawn_radius={radius:.3f}"
        )
    window.show()
    deadline = time.monotonic() + 0.25
    while time.monotonic() < deadline:
        app.processEvents()
        time.sleep(0.01)
    destination = Path(args.output)
    if not destination.is_absolute():
        destination = Path(__file__).resolve().parents[1] / destination
    destination.parent.mkdir(parents=True, exist_ok=True)
    if not window.grab().save(str(destination)):
        raise RuntimeError(f"could not save screenshot to {destination}")
    window.close()
    app.processEvents()
    print(f"agents={len(snapshot.agent_states)} screenshot={destination}")


if __name__ == "__main__":
    main()

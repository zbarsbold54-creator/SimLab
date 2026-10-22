"""Rolling pyqtgraph drive histories and snapshot timeline."""

from __future__ import annotations

from collections import deque

import pyqtgraph as pg
from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import QHBoxLayout, QSlider, QWidget

from simlab.gui import theme
from simlab.gui.frame import SimulationFrame


class DrivePlotWidget(QWidget):
    timelineChanged = Signal(int)

    def __init__(self) -> None:
        super().__init__()
        layout = QHBoxLayout(self)
        self.plot = pg.PlotWidget()
        self.plot.setBackground(theme.PANEL)
        self.plot.showGrid(x=True, y=True, alpha=0.18)
        self.plot.setYRange(0, 1, padding=0)
        self.plot.addLegend()
        self.timeline = QSlider(Qt.Orientation.Horizontal)
        self.timeline.setRange(0, 0)
        self.timeline.setFixedWidth(theme.TIMELINE_WIDTH)
        layout.addWidget(self.plot, 1)
        layout.addWidget(self.timeline)
        self.timeline.valueChanged.connect(self.timelineChanged)
        self._selected: str | None = None
        self._series: dict[str, deque[tuple[int, float]]] = {}
        self._lines: dict[str, object] = {}

    def select_agent(self, agent_id: str | None) -> None:
        self._selected = agent_id
        self._series.clear()
        for line in self._lines.values():
            self.plot.removeItem(line)
        self._lines.clear()

    def set_timeline_range(self, first: int, last: int, current: int) -> None:
        self.timeline.blockSignals(True)
        self.timeline.setRange(first, max(first, last))
        self.timeline.setValue(min(max(current, first), max(first, last)))
        self.timeline.blockSignals(False)

    def update_frame(self, frame: SimulationFrame) -> None:
        detail = frame.agents.get(self._selected)
        if detail is None:
            return
        for index, name in enumerate(frame.drive_names):
            values = self._series.setdefault(name, deque(maxlen=theme.PLOT_HISTORY))
            values.append(
                (
                    frame.snapshot.step,
                    detail.drives[index] if index < len(detail.drives) else 0.0,
                )
            )
            line = self._lines.get(name)
            if line is None:
                line = self.plot.plot(
                    name=name,
                    pen=pg.mkPen(
                        theme.DRIVE_COLORS[index % len(theme.DRIVE_COLORS)], width=2
                    ),
                )
                self._lines[name] = line
                if self.plot.plotItem.legend is not None:
                    for sample, label in self.plot.plotItem.legend.items:
                        if label.text == name:
                            label.mousePressEvent = lambda event, item=line: (
                                item.setVisible(not item.isVisible())
                            )
            line.setData([point[0] for point in values], [point[1] for point in values])

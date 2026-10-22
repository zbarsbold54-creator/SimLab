"""Simulation controls toolbar."""

from __future__ import annotations

import secrets

from PySide6.QtCore import Signal
from PySide6.QtGui import QFont
from PySide6.QtWidgets import QComboBox, QLineEdit, QPushButton, QToolBar

from simlab.gui import theme


class SimulationToolbar(QToolBar):
    playPause = Signal(bool)
    singleStep = Signal()
    speedChanged = Signal(str)
    resetRequested = Signal(int)

    SPEEDS = ("0.25x", "0.5x", "1x", "2x", "5x", "10x", "Max")

    def __init__(self, parent=None) -> None:
        super().__init__("Simulation", parent)
        self.setMovable(False)
        self.play_button = QPushButton("Pause")
        self.step_button = QPushButton("Single-step")
        self.speed = QComboBox()
        self.speed.addItems(self.SPEEDS)
        self.seed = QLineEdit("1")
        self.seed.setFont(QFont(theme.NUMBER_FONT, theme.NUMBER_FONT_SIZE_PT))
        self.seed.setMaximumWidth(theme.SEED_FIELD_WIDTH)
        self.random_button = QPushButton("Random seed")
        self.reset_button = QPushButton("Reset")
        for widget in (
            self.play_button,
            self.step_button,
            self.speed,
            self.seed,
            self.random_button,
            self.reset_button,
        ):
            self.addWidget(widget)
        self.play_button.clicked.connect(self._toggle_play)
        self.step_button.clicked.connect(self.singleStep)
        self.speed.currentTextChanged.connect(self.speedChanged)
        self.random_button.clicked.connect(self._randomize_seed)
        self.reset_button.clicked.connect(self._reset)
        self._playing = True

    def _toggle_play(self) -> None:
        self._playing = not self._playing
        self.play_button.setText("Pause" if self._playing else "Play")
        self.playPause.emit(self._playing)

    def _randomize_seed(self) -> None:
        self.seed.setText(str(secrets.randbelow(2**31)))

    def _reset(self) -> None:
        try:
            seed = int(self.seed.text())
        except ValueError:
            seed = 1
            self.seed.setText(str(seed))
        self.resetRequested.emit(seed)

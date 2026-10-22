"""Painter-based world view with pan and zoom."""

from __future__ import annotations

import math

from PySide6.QtCore import QPointF, Qt
from PySide6.QtGui import (
    QBrush,
    QColor,
    QLinearGradient,
    QPainter,
    QPainterPath,
    QPen,
    QPolygonF,
)
from PySide6.QtWidgets import QWidget

from simlab.core.types import Snapshot
from simlab.gui import theme


class WorldView(QWidget):
    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._snapshot: Snapshot | None = None
        self._zoom = 1.0
        self._pan = QPointF(0.0, 0.0)
        self._drag_point: QPointF | None = None
        self.setMinimumSize(theme.VIEW_MIN_WIDTH, theme.VIEW_MIN_HEIGHT)

    def set_snapshot(self, snapshot: Snapshot) -> None:
        self._snapshot = snapshot
        self.update()

    def paintEvent(self, event: object) -> None:
        del event
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)
        painter.fillRect(self.rect(), QColor(theme.BACKGROUND))
        snapshot = self._snapshot
        if snapshot is None:
            painter.end()
            return
        margin = float(theme.WORLD_MARGIN)
        scale = (
            min(
                (self.width() - 2 * margin) / max(snapshot.width, 1.0),
                (self.height() - 2 * margin) / max(snapshot.height, 1.0),
            )
            * self._zoom
        )
        origin = QPointF(
            (self.width() - snapshot.width * scale) / 2 + self._pan.x(),
            (self.height() - snapshot.height * scale) / 2 + self._pan.y(),
        )
        painter.save()
        painter.translate(origin)
        painter.scale(scale, scale)
        painter.setPen(QPen(QColor(theme.BORDER), theme.WALL_WIDTH / scale))
        painter.setBrush(Qt.BrushStyle.NoBrush)
        painter.drawRect(0.0, 0.0, snapshot.width, snapshot.height)
        for patch in snapshot.food_patches:
            x, y = patch["position"]
            radius = theme.FOOD_SIZE_MIN + min(
                float(patch["amount"]) / theme.DEFAULT_FOOD_AMOUNT, 1
            ) * (theme.FOOD_SIZE_MAX - theme.FOOD_SIZE_MIN)
            painter.setPen(Qt.PenStyle.NoPen)
            painter.setBrush(QColor(theme.FOOD))
            painter.drawEllipse(QPointF(x, y), radius, radius)
        for patch in snapshot.hazard_patches:
            x, y = patch["position"]
            radius = float(patch.get("detection_radius", 0.0))
            ring = QColor(theme.HAZARD)
            ring.setAlpha(theme.HAZARD_RING_ALPHA)
            painter.setPen(QPen(ring, theme.WALL_WIDTH / scale))
            painter.setBrush(Qt.BrushStyle.NoBrush)
            painter.drawEllipse(QPointF(x, y), radius, radius)
            painter.setPen(Qt.PenStyle.NoPen)
            painter.setBrush(QColor(theme.HAZARD))
            size = theme.HAZARD_SIZE
            painter.drawPolygon(
                QPolygonF(
                    [
                        QPointF(x, y - size),
                        QPointF(x + size, y),
                        QPointF(x, y + size),
                        QPointF(x - size, y),
                    ]
                )
            )
        for state in snapshot.agent_states.values():
            if not state.alive:
                continue
            x, y = state.position
            angle = state.heading
            size = theme.AGENT_SIZE
            local = (
                (size, 0.0),
                (
                    -size * theme.ENERGY_GRADIENT_RADIUS,
                    size * theme.ENERGY_GRADIENT_RADIUS,
                ),
                (
                    -size * theme.ENERGY_GRADIENT_RADIUS,
                    -size * theme.ENERGY_GRADIENT_RADIUS,
                ),
            )
            points = [
                QPointF(
                    x + dx * math.cos(angle) - dy * math.sin(angle),
                    y + dx * math.sin(angle) + dy * math.cos(angle),
                )
                for dx, dy in local
            ]
            gradient = QLinearGradient(QPointF(x - size, y), QPointF(x + size, y))
            gradient.setColorAt(0.0, QColor(theme.ENERGY_LOW))
            gradient.setColorAt(1.0, QColor(theme.ENERGY_HIGH))
            painter.setPen(QPen(QColor(theme.BORDER), theme.WALL_WIDTH / scale))
            painter.setBrush(QBrush(gradient))
            path = QPainterPath()
            incoming = []
            outgoing = []
            for index, point in enumerate(points):
                previous_point = points[index - 1]
                next_point = points[(index + 1) % len(points)]
                incoming.append(
                    point + (previous_point - point) * theme.AGENT_CORNER_ROUNDING
                )
                outgoing.append(
                    point + (next_point - point) * theme.AGENT_CORNER_ROUNDING
                )
            path.moveTo(incoming[0])
            for index, point in enumerate(points):
                path.quadTo(point, outgoing[index])
                path.lineTo(incoming[(index + 1) % len(points)])
            path.closeSubpath()
            painter.drawPath(path)
        painter.restore()
        painter.end()

    def wheelEvent(self, event: object) -> None:
        factor = (
            theme.ZOOM_FACTOR if event.angleDelta().y() > 0 else 1.0 / theme.ZOOM_FACTOR
        )
        self._zoom = min(theme.ZOOM_MAX, max(theme.ZOOM_MIN, self._zoom * factor))
        self.update()

    def mousePressEvent(self, event: object) -> None:
        if event.button() == Qt.MouseButton.LeftButton:
            self._drag_point = event.position()
            self.setCursor(Qt.CursorShape.ClosedHandCursor)

    def mouseMoveEvent(self, event: object) -> None:
        if self._drag_point is not None:
            point = event.position()
            self._pan += point - self._drag_point
            self._drag_point = point
            self.update()

    def mouseReleaseEvent(self, event: object) -> None:
        del event
        self._drag_point = None
        self.setCursor(Qt.CursorShape.ArrowCursor)

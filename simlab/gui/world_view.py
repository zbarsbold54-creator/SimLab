"""Painter-based world view with arena clipping, hover details, pan, and zoom."""

from __future__ import annotations

import math

from PySide6.QtCore import QPointF, QRectF, Qt
from PySide6.QtGui import QColor, QPainter, QPainterPath, QPen, QPolygonF
from PySide6.QtWidgets import QWidget

from simlab.core.types import Snapshot
from simlab.gui import theme


def agent_triangle_vertices(
    position: tuple[float, float], heading: float, radius: float
) -> tuple[tuple[float, float], tuple[float, float], tuple[float, float]]:
    """Return the nose and symmetric rear vertices in world coordinates."""
    local_vertices = (
        (radius, 0.0),
        (theme.AGENT_TAIL_X * radius, theme.AGENT_TAIL_Y * radius),
        (theme.AGENT_TAIL_X * radius, -theme.AGENT_TAIL_Y * radius),
    )
    cosine, sine = math.cos(heading), math.sin(heading)
    px, py = position
    return tuple(
        (
            px + x * cosine - y * sine,
            py + x * sine + y * cosine,
        )
        for x, y in local_vertices
    )


class WorldView(QWidget):
    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._snapshot: Snapshot | None = None
        self._zoom = 1.0
        self._pan = QPointF(0.0, 0.0)
        self._drag_point: QPointF | None = None
        self._press_point: QPointF | None = None
        self._cursor_point: QPointF | None = None
        self._hovered_hazard: int | None = None
        self._selected_agent: str | None = None
        self.setMouseTracking(True)
        self.setMinimumSize(theme.VIEW_MIN_WIDTH, theme.VIEW_MIN_HEIGHT)

    @staticmethod
    def food_draw_radius(amount: float) -> float:
        ratio = min(max(float(amount) / theme.DEFAULT_FOOD_AMOUNT, 0.0), 1.0)
        return theme.FOOD_SIZE_MIN + ratio * (theme.FOOD_SIZE_MAX - theme.FOOD_SIZE_MIN)

    def set_snapshot(self, snapshot: Snapshot) -> None:
        self._snapshot = snapshot
        self._refresh_hover()
        self.update()

    def _transform(self, snapshot: Snapshot) -> tuple[float, QPointF]:
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
        return scale, origin

    def _world_point(self, point: QPointF) -> QPointF | None:
        if self._snapshot is None:
            return None
        scale, origin = self._transform(self._snapshot)
        return QPointF(
            (point.x() - origin.x()) / scale, (point.y() - origin.y()) / scale
        )

    def _refresh_hover(self) -> None:
        self._hovered_hazard = None
        if self._snapshot is None or self._cursor_point is None:
            return
        world_point = self._world_point(self._cursor_point)
        if world_point is None:
            return
        closest = math.inf
        for index, patch in enumerate(self._snapshot.hazard_patches):
            x, y = patch["position"]
            distance = math.hypot(world_point.x() - x, world_point.y() - y)
            if (
                distance <= float(patch["radius"]) + theme.HAZARD_HOVER_TOLERANCE
                and distance < closest
            ):
                self._hovered_hazard = index
                closest = distance

    def paintEvent(self, event: object) -> None:
        del event
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)
        painter.fillRect(self.rect(), QColor(theme.BACKGROUND))
        snapshot = self._snapshot
        if snapshot is None:
            painter.end()
            return

        scale, origin = self._transform(snapshot)
        painter.save()
        painter.translate(origin)
        painter.scale(scale, scale)
        arena = QRectF(0.0, 0.0, snapshot.width, snapshot.height)
        painter.setPen(
            QPen(QColor(theme.ARENA_BORDER), theme.ARENA_BORDER_WIDTH / scale)
        )
        painter.setBrush(QColor(theme.PANEL))
        painter.drawRect(arena)
        painter.setClipRect(arena, Qt.ClipOperation.IntersectClip)

        # Food is beneath hazards and agents so living entities remain legible.
        for patch in snapshot.food_patches:
            x, y = patch["position"]
            radius = self.food_draw_radius(float(patch["amount"]))
            outline = QColor(theme.FOOD_OUTLINE)
            outline.setAlpha(theme.FOOD_OUTLINE_ALPHA)
            painter.setPen(QPen(outline, theme.FOOD_OUTLINE_WIDTH / scale))
            painter.setBrush(QColor(theme.FOOD_FILL))
            painter.drawEllipse(QPointF(x, y), radius, radius)

        selected = snapshot.agent_states.get(self._selected_agent)
        for index, patch in enumerate(snapshot.hazard_patches):
            x, y = patch["position"]
            contact_radius = float(patch.get("radius", 0.0))
            detection_radius = float(patch.get("detection_radius", 0.0))
            ring = QColor(theme.HAZARD)
            ring.setAlpha(theme.HAZARD_RING_ALPHA)
            painter.setPen(QPen(ring, theme.WALL_WIDTH / scale))
            painter.setBrush(Qt.BrushStyle.NoBrush)
            painter.drawEllipse(QPointF(x, y), contact_radius, contact_radius)
            if self._detection_ring_visible(index, patch, selected):
                detection_ring = QColor(theme.HAZARD)
                detection_ring.setAlpha(theme.HAZARD_DETECTION_RING_ALPHA)
                painter.setPen(QPen(detection_ring, theme.WALL_WIDTH / scale))
                painter.drawEllipse(QPointF(x, y), detection_radius, detection_radius)

            outline = QColor(theme.HAZARD)
            outline.setAlpha(theme.HAZARD_OUTLINE_ALPHA)
            painter.setPen(QPen(outline, theme.WALL_WIDTH / scale))
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
            points = [
                QPointF(*point)
                for point in agent_triangle_vertices(state.position, angle, size)
            ]
            fraction = min(max(state.energy / max(state.max_energy, 1e-9), 0.0), 1.0)
            fill = self._energy_color(fraction)
            outline = QColor(theme.AGENT_OUTLINE)
            outline.setAlpha(theme.AGENT_OUTLINE_ALPHA)
            painter.setPen(QPen(outline, theme.AGENT_OUTLINE_WIDTH / scale))
            painter.setBrush(fill)
            painter.drawPath(self._rounded_triangle(points))

        painter.restore()
        painter.end()

    def _detection_ring_visible(self, index: int, hazard, selected_agent) -> bool:
        if self._hovered_hazard == index:
            return True
        return (
            selected_agent is not None
            and selected_agent.alive
            and math.dist(selected_agent.position, hazard["position"])
            <= float(hazard.get("detection_radius", 0.0))
        )

    @staticmethod
    def _energy_color(fraction: float) -> QColor:
        low, high = QColor(theme.ENERGY_LOW), QColor(theme.ENERGY_HIGH)
        return QColor(
            round(low.red() + fraction * (high.red() - low.red())),
            round(low.green() + fraction * (high.green() - low.green())),
            round(low.blue() + fraction * (high.blue() - low.blue())),
        )

    @staticmethod
    def _rounded_triangle(points: list[QPointF]) -> QPainterPath:
        path = QPainterPath()
        incoming = []
        outgoing = []
        for index, point in enumerate(points):
            previous_point = points[index - 1]
            next_point = points[(index + 1) % len(points)]
            incoming.append(
                point + (previous_point - point) * theme.AGENT_CORNER_ROUNDING
            )
            outgoing.append(point + (next_point - point) * theme.AGENT_CORNER_ROUNDING)
        path.moveTo(incoming[0])
        for index, point in enumerate(points):
            path.quadTo(point, outgoing[index])
            path.lineTo(incoming[(index + 1) % len(points)])
        path.closeSubpath()
        return path

    def wheelEvent(self, event: object) -> None:
        factor = (
            theme.ZOOM_FACTOR if event.angleDelta().y() > 0 else 1.0 / theme.ZOOM_FACTOR
        )
        self._zoom = min(theme.ZOOM_MAX, max(theme.ZOOM_MIN, self._zoom * factor))
        self._refresh_hover()
        self.update()

    def mousePressEvent(self, event: object) -> None:
        if event.button() == Qt.MouseButton.LeftButton:
            self._drag_point = event.position()
            self._press_point = event.position()
            self.setCursor(Qt.CursorShape.ClosedHandCursor)

    def mouseMoveEvent(self, event: object) -> None:
        point = event.position()
        if self._drag_point is not None:
            self._pan += point - self._drag_point
            self._drag_point = point
        self._cursor_point = point
        self._refresh_hover()
        self.update()

    def mouseReleaseEvent(self, event: object) -> None:
        point = event.position()
        if (
            self._press_point is not None
            and self._snapshot is not None
            and (point - self._press_point).manhattanLength()
            <= theme.CLICK_DRAG_THRESHOLD
        ):
            world_point = self._world_point(point)
            if world_point is not None:
                nearest = min(
                    (
                        (
                            math.hypot(
                                world_point.x() - state.position[0],
                                world_point.y() - state.position[1],
                            ),
                            agent_id,
                        )
                        for agent_id, state in self._snapshot.agent_states.items()
                        if state.alive
                    ),
                    default=(math.inf, None),
                )
                self._selected_agent = (
                    nearest[1] if nearest[0] <= theme.AGENT_SELECT_RADIUS else None
                )
        self._drag_point = None
        self._press_point = None
        self._cursor_point = point
        self.setCursor(Qt.CursorShape.ArrowCursor)
        self._refresh_hover()
        self.update()

    def leaveEvent(self, event: object) -> None:
        self._cursor_point = None
        self._refresh_hover()
        self.update()
        super().leaveEvent(event)

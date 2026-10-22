"""SimLab sensors module - ray-cast vision + proprioception."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class SensorReading:
    """Reading from a single sensor."""

    direction: tuple[int, int]  # (dx, dy) direction
    distance: int  # Distance to nearest obstacle/agent/food
    hit_type: str  # "wall", "agent", "food", "hazard", "empty"
    hit_position: tuple[int, int] | None = None  # (x, y) position of hit


@dataclass
class VisionResult:
    """Result from agent vision/ray-casting."""

    center: np.ndarray  # Center vision patch
    left: np.ndarray  # Left sector vision patch
    right: np.ndarray  # Right sector vision patch
    up: np.ndarray  # Up sector vision patch
    down: np.ndarray  # Down sector vision patch
    sensors: list[SensorReading]  # Individual sensor readings
    proprioception: dict[str, float]  # Agent's internal state (energy, etc.)


class RayCaster:
    """Ray-casting vision system for agent perception.

    Provides field-of-view vision by casting rays in multiple directions
    and detecting what's in front of the agent (walls, other agents,
    food, hazards, or empty space).
    """

    def __init__(self, vision_range: int = 5, num_rays: int = 8) -> None:
        """Initialize the ray caster.

        Args:
            vision_range: Maximum distance rays can travel.
            num_rays: Number of rays to cast in different directions.
        """
        self._vision_range = vision_range
        self._num_rays = num_rays

    def cast_all_rays(
        self,
        agent_position: tuple[int, int],
        world_state: np.ndarray,
        world_width: int,
        world_height: int,
        agent_positions: dict[str, tuple[int, int]] | None = None,
        food_positions: list[tuple[int, int]] | None = None,
        hazard_positions: list[tuple[int, int]] | None = None,
    ) -> list[SensorReading]:
        """Cast rays in all directions and return readings.

        Args:
            agent_position: (x, y) position of the agent.
            world_state: Current world state array.
            world_width: Width of the world.
            world_height: Height of the world.
            agent_positions: Optional dict of agent positions for collision detection.
            food_positions: Optional list of food patch positions.
            hazard_positions: Optional list of hazard patch positions.

        Returns:
            List of SensorReading for each ray direction.
        """
        x, y = agent_position
        readings = []

        # Generate ray directions in a circle
        angles = [2 * np.pi * i / self._num_rays for i in range(self._num_rays)]

        for angle in angles:
            dx = int(np.round(np.cos(angle)))
            dy = int(np.round(np.sin(angle)))
            reading = self._cast_ray(
                x,
                y,
                dx,
                dy,
                world_state,
                world_width,
                world_height,
                agent_positions,
                food_positions,
                hazard_positions,
            )
            readings.append(reading)

        return readings

    def _cast_ray(
        self,
        x: int,
        y: int,
        dx: int,
        dy: int,
        world_state: np.ndarray,
        world_width: int,
        world_height: int,
        agent_positions: dict[str, tuple[int, int]] | None = None,
        food_positions: list[tuple[int, int]] | None = None,
        hazard_positions: list[tuple[int, int]] | None = None,
    ) -> SensorReading:
        """Cast a single ray and return the reading.

        Args:
            x: Starting x position.
            y: Starting y position.
            dx: Ray direction x component.
            dy: Ray direction y component.
            world_state: Current world state array.
            world_width: Width of the world.
            world_height: Height of the world.
            agent_positions: Optional dict of agent positions.
            food_positions: Optional list of food patch positions.
            hazard_positions: Optional list of hazard patch positions.

        Returns:
            SensorReading for this ray.
        """
        distance = 0
        hit_type = "empty"
        hit_position: tuple[int, int] | None = None

        for step in range(1, self._vision_range + 1):
            check_x = x + dx * step
            check_y = y + dy * step

            # Check bounds
            if (
                check_x < 0
                or check_x >= world_width
                or check_y < 0
                or check_y >= world_height
            ):
                distance = step - 1
                hit_type = "wall"
                hit_position = (check_x, check_y) if step > 1 else None
                break

            # Check world state (wall/agent)
            if world_state[check_y, check_x] == 1:
                distance = step
                hit_type = "wall"
                hit_position = (check_x, check_y)
                break

            # Check agents
            if agent_positions:
                for ax, ay in agent_positions.values():
                    if ax == check_x and ay == check_y:
                        distance = step
                        hit_type = "agent"
                        hit_position = (check_x, check_y)
                        break
                else:
                    continue
                break

            # Check food
            if food_positions:
                for fx, fy in food_positions:
                    if fx == check_x and fy == check_y:
                        distance = step
                        hit_type = "food"
                        hit_position = (check_x, check_y)
                        break
                else:
                    continue
                break

            # Check hazards
            if hazard_positions:
                for hx, hy in hazard_positions:
                    if hx == check_x and hy == check_y:
                        distance = step
                        hit_type = "hazard"
                        hit_position = (check_x, check_y)
                        break
                else:
                    continue
                break

        # If no hit within range
        if distance == 0:
            distance = self._vision_range

        return SensorReading(
            direction=(dx, dy),
            distance=distance,
            hit_type=hit_type,
            hit_position=hit_position,
        )

    def compute_vision_patches(
        self,
        agent_position: tuple[int, int],
        world_state: np.ndarray,
        world_width: int,
        world_height: int,
    ) -> VisionResult:
        """Compute vision patches for different directions.

        Args:
            agent_position: (x, y) position of the agent.
            world_state: Current world state array.
            world_width: Width of the world.
            world_height: Height of the world.

        Returns:
            VisionResult containing vision patches and sensor readings.
        """
        x, y = agent_position

        # Get center patch (3x3 neighborhood around agent)
        center_patch = self._get_patch(
            world_state,
            world_width,
            world_height,
            x,
            y,
            radius=1,
        )

        # Get cardinal direction patches
        left_patch = self._get_patch(
            world_state,
            world_width,
            world_height,
            x,
            y,
            radius=1,
            direction=(-1, 0),
        )
        right_patch = self._get_patch(
            world_state,
            world_width,
            world_height,
            x,
            y,
            radius=1,
            direction=(1, 0),
        )
        up_patch = self._get_patch(
            world_state,
            world_width,
            world_height,
            x,
            y,
            radius=1,
            direction=(0, -1),
        )
        down_patch = self._get_patch(
            world_state,
            world_width,
            world_height,
            x,
            y,
            radius=1,
            direction=(0, 1),
        )

        # Cast all rays for sensor readings
        all_readings = self.cast_all_rays(
            agent_position,
            world_state,
            world_width,
            world_height,
        )

        # Compute proprioception
        proprioception = {
            "vision_range": float(self._vision_range),
            "num_rays": float(self._num_rays),
        }

        return VisionResult(
            center=center_patch,
            left=left_patch,
            right=right_patch,
            up=up_patch,
            down=down_patch,
            sensors=all_readings,
            proprioception=proprioception,
        )

    def _get_patch(
        self,
        world_state: np.ndarray,
        world_width: int,
        world_height: int,
        cx: int,
        cy: int,
        radius: int = 1,
        direction: tuple[int, int] = (0, 0),
    ) -> np.ndarray:
        """Get a vision patch around a position.

        Args:
            world_state: Current world state array.
            world_width: Width of the world.
            world_height: Height of the world.
            cx: Center x position.
            cy: Center y position.
            radius: Radius of the patch.
            direction: Direction offset (dx, dy).

        Returns:
            numpy array representing the vision patch.
        """
        x = cx + direction[0]
        y = cy + direction[1]

        # Ensure within bounds
        x = max(0, min(x, world_width - 1))
        y = max(0, min(y, world_height - 1))

        # Extract patch around position
        half = radius
        x_start = max(0, x - half)
        x_end = min(world_width, x + half + 1)
        y_start = max(0, y - half)
        y_end = min(world_height, y + half + 1)

        return world_state[y_start:y_end, x_start:x_end].copy()


def compute_proprioception(
    energy: float,
    max_energy: float = 100.0,
    step_count: int = 0,
) -> dict[str, float]:
    """Compute proprioception sensory data for an agent.

    Args:
        energy: Current energy level.
        max_energy: Maximum energy capacity.
        step_count: Current step number.

    Returns:
        Dictionary of proprioception sensory data.
    """
    return {
        "energy_ratio": energy / max_energy if max_energy > 0 else 0.0,
        "energy_level": energy,
        "max_energy": max_energy,
        "step_count": float(step_count),
    }

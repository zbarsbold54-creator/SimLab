"""Phase 2 survival, sensing, drives, and plugin gates."""

import numpy as np

from simlab.core.config import SimLabConfig
from simlab.core.registry import get, list_
from simlab.core.types import Action
from simlab.core.world import World
from simlab.interfaces.drive_core import BasicDriveCore, FixedDecay
from simlab.interfaces.policy import HeuristicPolicy


def test_hand_built_scene_has_food_and_hazard_ray_types() -> None:
    world = World(seed=4)
    world.add_agent("agent_0", (10.0, 10.0), heading=0.0)
    world.add_food((15.0, 10.0))
    world.add_hazard((15.0, 15.0), detection_radius=0.1)
    observation = world.step({"agent_0": Action()}).observations["agent_0"]
    assert 1 in observation.ray_types
    assert 2 in observation.ray_types
    assert all(0.0 <= value <= 20.0 for value in observation.rays)


def test_placeholder_drives_are_bounded() -> None:
    world = World(seed=5)
    world.add_agent("agent_0", (10.0, 10.0), energy=50.0, health=70.0)
    observation = world.step({"agent_0": Action()}).observations["agent_0"]
    drives = BasicDriveCore()
    drives.reset(1, world._rng)
    values = drives.step([observation], {"rng": world._rng})
    assert values.shape == (1, 8)
    assert np.all((values >= 0.0) & (values <= 1.0))


def test_retention_can_be_swapped_by_registered_name() -> None:
    assert get("retention", "fixed_decay") is not None
    assert get("retention", "passthrough") is not None
    fixed = FixedDecay(0.5)
    previous = np.zeros((1, 8), dtype=np.float32)
    drives = np.ones((1, 8), dtype=np.float32)
    assert np.allclose(fixed.step(previous, drives, {}), 0.5)
    config = SimLabConfig.from_dict({"components": {"retention": "passthrough"}})
    assert config.retention == "passthrough"


def test_plugins_register_without_core_edits() -> None:
    assert "example_idle" in list_("policy")
    assert get("policy", "heuristic") is HeuristicPolicy

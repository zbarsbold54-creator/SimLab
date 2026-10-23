"""Balanced preset: loads, enables regrowth, and is deterministic."""

import pytest
from PySide6.QtWidgets import QApplication

from simlab.core.config import load_config
from simlab.core.runner import Runner
from simlab.gui.main_window import MainWindow

CAP = 2000


@pytest.fixture(scope="module")
def app() -> QApplication:
    return QApplication.instance() or QApplication([])


def test_balanced_config_file_has_radius_16_and_regrow_0_02() -> None:
    cfg = load_config("configs/balanced.toml")
    assert cfg.hazard.detection_radius == 16.0
    assert cfg.food.regrow_rate == 0.02
    assert cfg.food.spawn_rate == 0.1


def test_balanced_preset_loads_into_scenario(app: QApplication) -> None:
    window = MainWindow(seed=7, agent_count=5)
    window.worker.set_playing(False)
    window.show()
    window.scenario.preset.setCurrentText("Balanced")
    window.scenario.apply()
    cfg = window._config
    assert cfg.hazard.detection_radius == 16.0
    assert cfg.food.regrow_rate == 0.02
    assert cfg.food.spawn_rate == 0.1
    assert cfg.agent.count == 5
    window.close()
    app.processEvents()


def test_balanced_determinism() -> None:
    cfg = load_config("configs/balanced.toml")
    hashes = []
    for _ in range(2):
        runner = Runner(cfg, seed=42, agents=5)
        runner.setup()
        world = runner._world
        for _ in range(CAP):
            runner.run_step(include_observations=False, include_agent_states=False)
        hashes.append(world.state_hash())
    assert hashes[0] == hashes[1]


def test_balanced_regrowth_is_active_and_capped() -> None:
    from simlab.core.world import World

    world = World(50, 50, seed=1, food_regrow_rate=0.02)
    world.add_food((10.0, 10.0), amount=5.0, max_amount=20.0)
    for _ in range(1000):
        world.step({})
    assert world._food_patches[0]["amount"] == pytest.approx(20.0)

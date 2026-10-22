"""Snapshot scrubbing, deterministic replay, and action-log coverage."""

from simlab.core.config import load_config
from simlab.core.runner import Runner
from simlab.core.snapshot import ActionLog
from simlab.core.types import Action


def test_scrub_back_500_steps_and_replay_matches_original_future() -> None:
    runner = Runner(load_config(), seed=71, agents=2)
    runner.setup()
    expected_hashes = []
    for step in range(700):
        actions = {
            agent_id: Action(
                turn=((step + index) % 9 - 4) / 4,
                thrust=((step + 2 * index) % 5) / 4,
                eat=step % 13 == 0,
            )
            for index, agent_id in enumerate(runner._world._agents)
        }
        runner.run_step(actions)
        expected_hashes.append(runner._world.state_hash())

    scrub_step = 700 - 500
    checkpoint = runner._snapshot_manager.get_snapshot(scrub_step)
    assert checkpoint is not None
    recorded_future = runner.action_log[scrub_step:]
    runner.restore(checkpoint)

    for offset, actions in enumerate(recorded_future):
        runner.run_step(actions)
        assert runner._world.state_hash() == expected_hashes[scrub_step + offset]


def test_action_log_round_trips_json_and_groups_by_step(tmp_path) -> None:
    steps = [
        {"agent_0": Action(0.25, 0.5, True), "agent_1": Action(-0.5, 0.75)},
        {"agent_0": Action(-1.0, 1.0)},
    ]
    recording = ActionLog.from_steps(19, steps)
    destination = tmp_path / "run.json"
    recording.save(destination)
    loaded = ActionLog.load(destination)
    assert loaded.seed == 19
    assert loaded.actions_by_step() == dict(enumerate(steps))

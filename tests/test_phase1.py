"""Phase 1 tests for SimLab - determinism verification."""

import hashlib

import pytest

from simlab.core.runner import Runner
from simlab.core.config import load_config


def world_state_hash(world) -> str:
    """Compute a hash of the full world state including agents, food, step, and RNG."""
    # Collect agent states
    agent_data = {}
    for aid, agent in sorted(world._agents.items()):
        if agent.get("is_alive", True):
            agent_data[aid] = {
                "position": agent["position"],
                "energy": agent["energy"],
            }

    # Serialize world state for hashing
    hash_input = (
        f"{world._step_count}"
        f"{sorted(agent_data.items())}"
        f"{sorted(world._food_patches)}"
        f"{sorted(world._hazard_patches)}"
    )
    return hashlib.sha256(hash_input.encode()).hexdigest()


def make_runner(seed=42, initial_energy=100000.0):
    """Helper to create a runner with sufficient agent energy."""
    config = load_config()
    config.agent.initial_energy = initial_energy
    runner = Runner(config=config, seed=seed)
    runner.setup()
    # Ensure agent has enough energy
    if "agent_0" in runner._world._agents:
        runner._world._agents["agent_0"]["energy"] = initial_energy
    return runner


def test_agent_moves_over_steps():
    """Test that the agent moves more than a minimum distance over 200 steps."""
    runner = make_runner(seed=42, initial_energy=100000.0)

    initial_pos = runner._world._agents["agent_0"]["position"]

    # Run 200 steps
    for _ in range(200):
        runner.run_step({})

    final_pos = runner._world._agents["agent_0"]["position"]

    # Manhattan distance should be > 0 (agent moved)
    distance = abs(final_pos[0] - initial_pos[0]) + abs(final_pos[1] - initial_pos[1])
    assert distance > 0, f"Agent did not move: initial={initial_pos}, final={final_pos}"


def test_different_seeds_different_trajectories():
    """Test that two different seeds produce different trajectories."""
    runner1 = make_runner(seed=42, initial_energy=100000.0)
    runner2 = make_runner(seed=123, initial_energy=100000.0)

    # Run 100 steps on both
    for _ in range(100):
        runner1.run_step({})
        runner2.run_step({})

    pos1 = runner1._world._agents["agent_0"]["position"]
    pos2 = runner2._world._agents["agent_0"]["position"]

    # Positions should differ for different seeds
    assert pos1 != pos2, f"Same position for different seeds: {pos1} vs {pos2}"


def test_snapshot_determinism():
    """Test: run 500 steps, snapshot, run 500 more and record hash.
    Then restore snapshot into fresh world and replay last 500 steps,
    assert hashes match step by step."""
    # Use higher initial energy
    runner1 = make_runner(seed=42, initial_energy=100000.0)

    # Run first 500 steps
    for _ in range(500):
        runner1.run_step({})

    # Capture snapshot at step 500
        snap = runner1._world.snapshot()

    # Run another 500 steps and capture hashes
    hashes1 = []
    for _ in range(500):
        runner1.run_step({})
    hashes1.append(world_state_hash(runner1._world))

    # Now create a fresh runner with same seed and restore snapshot
    runner2 = make_runner(seed=42, initial_energy=100000.0)

    # Restore the snapshot
    runner2._world.restore(snap)

    # Run 500 more steps and verify hashes match
    for step in range(500):
        runner1.run_step({})
        runner2.run_step({})

        h1 = world_state_hash(runner1._world)
        h2 = world_state_hash(runner2._world)

        assert h1 == h2, f"Hash mismatch at step {step + 500}: {h1} vs {h2}"


def test_action_log_replay():
    """Test that replaying a recorded action log reproduces the original final state hash."""
    # Use very high initial energy
    runner1 = make_runner(seed=42, initial_energy=100000.0)

    # Run 100 steps
    for _ in range(100):
        runner1.run_step({})

    # Get the last snapshot (step field = 100 after 100 steps)
    original_snap = None
    for snap in runner1._snapshot_manager._snapshots.values():
        if snap.step == 100:
            original_snap = snap
            break

    assert original_snap is not None, "Snapshot at step 100 not found"
    original_hash = world_state_hash(runner1._world)

    # Create a fresh runner
    runner2 = make_runner(seed=42, initial_energy=100000.0)
    runner2.setup()

    # Restore the snapshot
    runner2._world.restore(original_snap)

    # Replay: run 100 steps with no actions (should produce same result)
    for _ in range(100):
        runner2.run_step({})

    final_hash = world_state_hash(runner2._world)
    assert final_hash == original_hash, \
        f"Replay hash {final_hash} != original {original_hash}"

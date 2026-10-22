"""SimLab runner module - headless loop, no GUI."""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from simlab.core.types import (
    Action,
    Observation,
    AgentState,
    StepResult,
    Snapshot,
)
from simlab.core.rng import SimRNG
from simlab.core.config import SimLabConfig, load_config
from simlab.core.world import World
from simlab.core.entities import Agent, FoodPatch, Hazard
from simlab.core.physics import apply_movement, check_collision, compute_arena_bounds
from simlab.core.sensors import RayCaster, compute_proprioception, VisionResult, SensorReading
from simlab.core.snapshot import SnapshotManager
from simlab.interfaces.policy import HeuristicPolicy


class Runner:
    """Headless simulation runner - no GUI dependency.

    Runs the simulation in a headless mode, suitable for benchmarking,
    training, and headless operation. Emits snapshots at each step
    that can be consumed by GUI or analysis tools.
    """

    def __init__(
        self,
        config: Optional[SimLabConfig] = None,
        seed: Optional[int] = None,
        rng: Optional[SimRNG] = None,
    ) -> None:
        """Initialize the runner.

        Args:
            config: SimLab configuration. If None, loads default config.
            seed: Random seed for the simulation. If None, uses config seed.
            rng: Optional pre-initialized RNG. If None, creates one from seed.
        """
        self._config = config if config else load_config()
        self._seed = seed if seed is not None else self._config.world.seed
        self._rng = rng if rng else SimRNG(self._seed)

        # Initialize world
        self._world = World(
            width=self._config.world.width,
            height=self._config.world.height,
            seed=self._seed,
        )

        # Initialize ray caster for vision
        self._ray_caster = RayCaster(
            vision_range=self._config.world.width // 4,
            num_rays=8,
        )

        # Simulation state
        self._step_count = 0
        self._max_steps = 1000
        self._is_running = False
        self._snapshot_manager = None

    def setup(self) -> None:
        """Set up the initial simulation state."""
        # Reset world
        self._world.reset(seed=self._seed)

        # Add initial agents
        # Position agents in the center of the world
        center_x = self._config.world.width // 2
        center_y = self._config.world.height // 2

        if self._config.agent.initial_energy > 0:
            agent_id = "agent_0"
            agent = Agent(
                agent_id=agent_id,
                position=(center_x, center_y),
                energy=self._config.agent.initial_energy,
                max_energy=self._config.agent.initial_energy,
            )
            self._world.add_agent(agent_id, (center_x, center_y), agent.energy)

        # Add food patches
        if self._config.food.spawn_rate > 0:
            np.random.seed(self._seed + 1)
            num_food = max(1, int(self._config.world.width * self._config.world.height * self._config.food.spawn_rate))
            for _ in range(num_food):
                x = np.random.randint(0, self._config.world.width)
                y = np.random.randint(0, self._config.world.height)
                food_patch = FoodPatch(position=(x, y))
                self._world.add_food((x, y))

        # Add hazard patches
        if self._config.hazard.damage_per_step > 0:
            np.random.seed(self._seed + 2)
            num_hazards = max(1, int(self._config.world.width * self._config.world.height * 0.05))
            for _ in range(num_hazards):
                x = np.random.randint(0, self._config.world.width)
                y = np.random.randint(0, self._config.world.height)
                hazard = Hazard(position=(x, y))
                self._world.add_hazard((x, y))

        # Initialize snapshot manager
        self._snapshot_manager = SnapshotManager()

        self._step_count = 0
        self._is_running = True

    def run_step(
        self,
        agent_actions: Dict[str, Action],
    ) -> StepResult:
        """Run one simulation step.

        Args:
            agent_actions: Dictionary mapping agent_id to Action.

        Returns:
            StepResult containing updated agent states, observations,
            snapshot, termination flag, and reward.
        """
        if not self._is_running:
            raise RuntimeError("Runner is not running. Call setup() first.")

        if self._step_count >= self._max_steps:
            return StepResult(
                agent_states={},
                observations={},
                snapshot=self._snapshot_manager.capture_snapshot(
                    self._world.get_world_state(), self._step_count,
                ),
                terminated=True,
                reward=0.0,
            )

        # Use heuristic policy if no actions provided
        if not agent_actions:
            policy = HeuristicPolicy()
            agent_0_state = {
                "energy": self._config.agent.initial_energy,
                "position": self._world._agents.get("agent_0", {}).get("position", (0, 0)),
            }
            world_state = {
                "food_locations": list(self._world._food_patches),
                "hazard_locations": list(self._world._hazard_patches),
            }
            agent_actions = {"agent_0": policy.select_action(
                agent_0_state, world_state, [], self._step_count,
            )}
        
        # Step the world
        result = self._world.step(agent_actions)

        # Capture snapshot using World's snapshot method (contains ALL state)
        snapshot = self._world.snapshot()
        self._snapshot_manager._snapshots[self._step_count] = snapshot

        # Update step count
        self._step_count += 1

        # Build observations and agent states from world state
        observations = {}
        agent_states = {}

        for agent_id, agent_data in self._world._agents.items():
            if not agent_data.get("is_alive", True):
                continue

            # Get sensor readings for this agent
            vision = self._ray_caster.compute_vision_patches(
                (int(agent_data["position"][0]), int(agent_data["position"][1])),
                self._world.get_world_state(),
                self._config.world.width,
                self._config.world.height,
            )

            # Compute proprioception
            proprioception = compute_proprioception(
                agent_data["energy"],
                self._config.agent.initial_energy,
                self._step_count,
            )

            # Build observation
            observations[agent_id] = Observation(
                world_state=vision.center,
                agent_position=(int(agent_data["position"][0]), int(agent_data["position"][1])),
                agent_energy=agent_data["energy"],
                food_locations=list(self._world._food_patches),
                hazard_locations=list(self._world._hazard_patches),
                step_count=self._step_count,
            )

            # Build agent state
            agent_states[agent_id] = AgentState(
                position=(int(agent_data["position"][0]), int(agent_data["position"][1])),
                energy=agent_data["energy"],
                max_energy=self._config.agent.initial_energy,
                is_alive=agent_data.get("is_alive", True),
                agent_id=agent_id,
            )

        return StepResult(
            agent_states=agent_states,
            observations=observations,
            snapshot=snapshot,
            terminated=result.terminated,
            reward=result.reward,
        )

    def run_episode(
        self,
        agent_actions_generator: Optional[
            callable[[int, Dict[str, AgentState]], Dict[str, Action]]
        ] = None,
    ) -> List[StepResult]:
        """Run a complete episode (sequence of steps).

        Args:
            agent_actions_generator: Optional function that generates
                actions given step and agent states. If None, agents
                take no actions (default).

        Returns:
            List of StepResult for each step in the episode.
        """
        results = []

        while self._is_running:
            # Generate actions if provided, otherwise agents take no actions
            if agent_actions_generator is not None:
                states = {
                    agent_id: state
                    for agent_id, state in self._entity_registry.get_all_agents().items()
                    if state is not None
                }
                agent_actions = agent_actions_generator(self._step_count, states)
            else:
                agent_actions = {}

            step_result = self.run_step(agent_actions)
            results.append(step_result)

            if step_result.terminated:
                break

        return results

    def run_bench(
        self,
        num_steps: int,
    ) -> List[StepResult]:
        """Run a benchmark for a fixed number of steps.

        Args:
            num_steps: Number of steps to run.

        Returns:
            List of StepResult for each step.
        """
        results = []

        for _ in range(num_steps):
            agent_actions = {}
            step_result = self.run_step(agent_actions)
            results.append(step_result)

            if step_result.terminated:
                break

        return results

    def snapshot(self) -> Optional[Snapshot]:
        """Get the most recent snapshot.

        Returns:
            Most recent Snapshot, or None if no steps have been taken.
        """
        if self._snapshot_manager is None:
            return None
        return self._snapshot_manager.latest_snapshot

    def reset(self) -> None:
        """Reset the runner to initial state."""
        self._is_running = False
        self._step_count = 0
        self._world.reset(seed=self._seed)
        self._entity_registry = EntityRegistry()
        self._snapshot_manager = None

    def get_stats(self) -> Dict[str, Any]:
        """Get statistics about the simulation run.

        Returns:
            Dictionary of statistics.
        """
        return {
            "step_count": self._step_count,
            "is_running": self._is_running,
            "world_size": f"{self._config.world.width}x{self._config.world.height}",
            "is_running": self._is_running,
        }

    def stop(self) -> None:
        """Stop the runner."""
        self._is_running = False

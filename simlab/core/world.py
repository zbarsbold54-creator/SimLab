"""SimLab world module - World: reset(seed), step(actions) -> StepResult."""

import numpy as np
from typing import Any, Dict, List, Optional, Tuple

from simlab.core.types import (
    Action,
    Observation,
    AgentState,
    StepResult,
    Snapshot,
)


class World:
    """2D world for simulation where agents survive by managing energy.
    
    The world is headless-capable and deterministic: same seed + same actions
    = same result. Supports snapshot/restore for action-log record/replay.
    """
    
    def __init__(self, width: int = 50, height: int = 50, seed: int = 42) -> None:
        """Initialize the world.
        
        Args:
            width: Width of the world grid.
            height: Height of the world grid.
            seed: Random seed for deterministic initialization.
        """
        self._width = width
        self._height = height
        self._seed = seed
        self._step_count = 0
        self._world: Optional[np.ndarray] = None
        self._agents: Dict[str, Dict[str, Any]] = {}
        self._food_patches: List[Tuple[int, int]] = []
        self._hazard_patches: List[Tuple[int, int]] = []
        self._reset()
    
    def _reset(self) -> None:
        """Reset the world to initial state."""
        self._world = np.zeros((self._height, self._width), dtype=np.int8)
        self._agents = {}
        self._food_patches = []
        self._hazard_patches = []
        self._step_count = 0
        # Set random seed for reproducible initialization
        np.random.seed(self._seed)
    
    def reset(self, seed: Optional[int] = None) -> None:
        """Reset the world to initial state, optionally with a new seed.
        
        Args:
            seed: Optional new seed value. If None, uses the original seed.
        """
        if seed is not None:
            self._seed = seed
        self._reset()
    
    def add_agent(
        self,
        agent_id: str,
        position: Tuple[int, int],
        energy: Optional[float] = None,
    ) -> None:
        """Add an agent to the world.
        
        Args:
            agent_id: Unique identifier for the agent.
            position: (x, y) position in the world.
            energy: Initial energy level. If None, uses default.
        """
        x, y = position
        if energy is None:
            energy = 100.0
        
        # Ensure position is within bounds
        x = max(0, min(x, self._width - 1))
        y = max(0, min(y, self._height - 1))
        
        self._agents[agent_id] = {
            "position": (x, y),
            "energy": energy,
            "is_alive": True,
        }
    
    def remove_agent(self, agent_id: str) -> None:
        """Remove an agent from the world.
        
        Args:
            agent_id: Unique identifier for the agent.
        """
        if agent_id in self._agents:
            del self._agents[agent_id]
    
    def add_food(self, position: Tuple[int, int], count: int = 1) -> None:
        """Add food patches to the world.
        
        Args:
            position: (x, y) position for the food.
            count: Number of food patches at this position.
        """
        x, y = position
        x = max(0, min(x, self._width - 1))
        y = max(0, min(y, self._height - 1))
        
        for _ in range(count):
            self._food_patches.append((x, y))
    
    def add_hazard(self, position: Tuple[int, int], count: int = 1) -> None:
        """Add hazard patches to the world.
        
        Args:
            position: (x, y) position for the hazard.
            count: Number of hazard patches at this position.
        """
        x, y = position
        x = max(0, min(x, self._width - 1))
        y = max(0, min(y, self._height - 1))
        
        for _ in range(count):
            self._hazard_patches.append((x, y))
    
    def step(self, actions: Dict[str, Action]) -> StepResult:
        """Run one simulation step.
        
        Args:
            actions: Dictionary mapping agent_id to Action.
            
        Returns:
            StepResult containing updated agent states, observations,
            snapshot, termination flag, and reward.
        """
        self._step_count += 1
        
        # Process agent actions
        for agent_id, action in actions.items():
            if agent_id not in self._agents or not self._agents[agent_id]["is_alive"]:
                continue
            
            # Move agent
            dx, dy = action.move_direction
            x, y = self._agents[agent_id]["position"]
            new_x = max(0, min(x + dx, self._width - 1))
            new_y = max(0, min(y + dy, self._height - 1))
            
            self._agents[agent_id]["position"] = (new_x, new_y)
            self._agents[agent_id]["energy"] -= action.energy_spent
            
            # Check for energy depletion
            if self._agents[agent_id]["energy"] <= 0:
                self._agents[agent_id]["is_alive"] = False
        
        # Process food consumption
        for agent_id, agent in self._agents.items():
            if not agent["is_alive"]:
                continue
            
            x, y = agent["position"]
            # Check if position has food
            food_indices = [
                i for i, (fx, fy) in enumerate(self._food_patches)
                if fx == x and fy == y
            ]
            
            if food_indices:
                # Eat food and gain energy
                for idx in sorted(food_indices, reverse=True):
                    self._food_patches.pop(idx)
                agent["energy"] = min(
                    agent["energy"] + 10.0, 100.0
                )
        
        # Process hazard damage
        for agent_id, agent in self._agents.items():
            if not agent["is_alive"]:
                continue
            
            x, y = agent["position"]
            # Check if position has hazard
            for hx, hy in self._hazard_patches:
                if hx == x and hy == y:
                    agent["energy"] -= 1.0
                    break
            
            # Check for energy depletion after hazard
            if agent["energy"] <= 0:
                agent["is_alive"] = False
        
        # Remove dead agents
        dead_agents = [
            aid for aid, agent in self._agents.items()
            if not agent["is_alive"]
        ]
        for aid in dead_agents:
            self.remove_agent(aid)
        
        # Update world state
        self._world.fill(0)
        for agent_id, agent in self._agents.items():
            if agent["is_alive"]:
                x, y = agent["position"]
                if 0 <= x < self._width and 0 <= y < self._height:
                    self._world[y, x] = 1
        
        # Build observations for each agent
        observations = {}
        for agent_id, agent in self._agents.items():
            observations[agent_id] = self._build_observation(agent_id)
        
        # Create snapshot
        snapshot = Snapshot(
            step=self._step_count,
            world_state=self._world.copy(),
            agent_positions={
                aid: agent["position"] for aid, agent in self._agents.items()
            },
            agent_energies={
                aid: agent["energy"] for aid, agent in self._agents.items()
            },
            food_count=len(self._food_patches),
            hazard_count=len(self._hazard_patches),
        )
        
        # Calculate reward (simplified: sum of alive agent energies)
        reward = sum(
            agent["energy"] for agent in self._agents.values() if agent["is_alive"]
        )
        
        terminated = len([a for a in self._agents.values() if a["is_alive"]]) == 0
        
        # Build agent states
        agent_states = {}
        for agent_id, agent in self._agents.items():
            agent_states[agent_id] = AgentState(
                position=agent["position"],
                energy=agent["energy"],
                max_energy=100.0,
                is_alive=agent["is_alive"],
                agent_id=agent_id,
            )
        
        return StepResult(
            agent_states=agent_states,
            observations=observations,
            snapshot=snapshot,
            terminated=terminated,
            reward=reward,
        )
    
    def _build_observation(self, agent_id: str) -> Observation:
        """Build an observation for an agent.
        
        Args:
            agent_id: ID of the agent.
            
        Returns:
            Observation containing world state and relevant info.
        """
        agent = self._agents[agent_id]
        x, y = agent["position"]
        
        # Get surrounding area (3x3 neighborhood)
        half = 1
        x_start = max(0, x - half)
        x_end = min(self._width, x + half + 1)
        y_start = max(0, y - half)
        y_end = min(self._height, y + half + 1)
        
        world_patch = self._world[y_start:y_end, x_start:x_end].copy()
        
        # Find nearby food and hazards
        food_locations = [
            (fx, fy) for fx, fy in self._food_patches
            if abs(fx - x) <= half and abs(fy - y) <= half
        ]
        hazard_locations = [
            (hx, hy) for hx, hy in self._hazard_patches
            if abs(hx - x) <= half and abs(hy - y) <= half
        ]
        
        return Observation(
            world_state=world_patch,
            agent_position=(x, y),
            agent_energy=agent["energy"],
            food_locations=food_locations,
            hazard_locations=hazard_locations,
            step_count=self._step_count,
        )
    
    def snapshot(self) -> Snapshot:
        """Get a snapshot of the current world state.
        
        Returns:
            Snapshot of the current world state.
        """
        return Snapshot(
            step=self._step_count,
            world_state=self._world.copy(),
            agent_positions={
                aid: agent["position"] for aid, agent in self._agents.items()
            },
            agent_energies={
                aid: agent["energy"] for aid, agent in self._agents.items()
            },
            food_count=len(self._food_patches),
            hazard_count=len(self._hazard_patches),
        )
    
    def restore(self, snapshot: Snapshot) -> None:
        """Restore the world from a snapshot.
        
        Args:
            snapshot: Snapshot to restore from.
        """
        self._step_count = snapshot.step
        self._world = snapshot.world_state.copy()
        self._agents = {}
        self._food_patches = []
        self._hazard_patches = []
        
        # Restore agent positions and energies from snapshot
        for agent_id, pos in snapshot.agent_positions.items():
            energy = snapshot.agent_energies.get(agent_id, 100.0)
            self._agents[agent_id] = {
                "position": pos,
                "energy": energy,
                "is_alive": True,
            }
        
        # Restore food and hazard counts from snapshot if available
        if hasattr(snapshot, 'food_count') and snapshot.food_count > 0:
            # Food/hazard positions are not fully stored in snapshot;
            # reset to empty and regenerate if needed
            pass
    
    def get_agent(self, agent_id: str) -> Optional[Dict[str, Any]]:
        """Get agent state.
        
        Args:
            agent_id: ID of the agent.
            
        Returns:
            Agent dict or None if not found.
        """
        return self._agents.get(agent_id)
    
    def get_world_state(self) -> np.ndarray:
        """Get the current world state.
        
        Returns:
            numpy array representing the world state.
        """
        return self._world.copy()
    
    def get_step_count(self) -> int:
        """Get the current step count.
        
        Returns:
            Number of steps taken.
        """
        return self._step_count

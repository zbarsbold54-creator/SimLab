"""SimLab core types - Snapshot, AgentState, Action, Observation dataclasses."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np


@dataclass
class Observation:
    """Observation received by an agent from the world."""
    world_state: np.ndarray  # Current world state array
    agent_position: Tuple[int, int]  # Agent's (x, y) position
    agent_energy: float  # Agent's current energy level
    food_locations: List[Tuple[int, int]]  # Positions of food patches
    hazard_locations: List[Tuple[int, int]]  # Positions of hazards
    step_count: int  # Current step number


@dataclass
class AgentState:
    """State of a single agent in the simulation."""
    position: Tuple[int, int]  # Agent's (x, y) position
    energy: float  # Current energy level (0.0 = dead)
    max_energy: float  # Maximum energy capacity
    is_alive: bool  # Whether the agent is alive
    agent_id: str  # Unique identifier for the agent


@dataclass
class Action:
    """Action that an agent can take in the simulation."""
    move_direction: Tuple[int, int]  # Direction vector (dx, dy)
    energy_spent: float  # Energy cost of the action
    action_id: str  # Unique identifier for the action


@dataclass
class StepResult:
    """Result of running one simulation step."""
    agent_states: Dict[str, AgentState]  # Updated agent states
    observations: Dict[str, Observation]  # Observations for each agent
    snapshot: "Snapshot"  # World state snapshot
    terminated: bool  # Whether the simulation should terminate
    reward: float  # Reward received


@dataclass
class Snapshot:
    """Immutable snapshot of the world state at a given step.
    
    Contains ALL world state for deterministic replay.
    """
    step: int  # Step number
    world_state: np.ndarray  # World state array (shape: width x height)
    agent_states: Dict[str, AgentState]  # Agent states with position, energy, is_alive
    food_patches: List[Tuple[int, int]]  # Positions of food patches
    hazard_patches: List[Tuple[int, int]]  # Positions of hazard patches
    rng_state: Optional[np.random.RandomState] = None  # RNG state for determinism
    metadata: Dict[str, Any] = field(default_factory=dict)  # Additional metadata

    def to_dict(self) -> Dict[str, Any]:
        """Convert snapshot to serializable dictionary."""
        return {
            "step": self.step,
            "world_state": self.world_state.tolist(),
            "agent_states": {aid: {
                "position": state.position,
                "energy": state.energy,
                "max_energy": state.max_energy,
                "is_alive": state.is_alive,
                "agent_id": state.agent_id,
            } for aid, state in self.agent_states.items()},
            "food_patches": self.food_patches,
            "hazard_patches": self.hazard_patches,
            "rng_state": None,  # RNG state not serializable, set during restore
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Snapshot":
        """Create snapshot from serializable dictionary."""
        world_state = np.array(data["world_state"], dtype=np.int8)
        agent_states = {}
        for aid, adata in data.get("agent_states", {}).items():
            agent_states[aid] = AgentState(
                position=tuple(adata["position"]),
                energy=adata["energy"],
                max_energy=adata.get("max_energy", 100.0),
                is_alive=adata.get("is_alive", True),
                agent_id=adata.get("agent_id", aid),
            )
        rng_state = None
        if data.get("rng_state") is not None:
            rng_state = np.random.RandomState(data["rng_state"])
        return cls(
            step=data["step"],
            world_state=world_state,
            agent_states=agent_states,
            food_patches=data.get("food_patches", []),
            hazard_patches=data.get("hazard_patches", []),
            rng_state=rng_state,
            metadata=data.get("metadata", {}),
        )

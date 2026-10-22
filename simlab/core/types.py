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
    agent_positions: Dict[str, Tuple[int, int]]  # Agent positions
    agent_energies: Dict[str, float]  # Agent energy levels
    food_patches: List[Tuple[int, int]]  # Positions of food patches
    hazard_patches: List[Tuple[int, int]]  # Positions of hazard patches
    metadata: Dict[str, Any] = field(default_factory=dict)  # Additional metadata

    def to_dict(self) -> Dict[str, Any]:
        """Convert snapshot to serializable dictionary."""
        return {
            "step": self.step,
            "world_state": self.world_state.tolist(),
            "agent_positions": self.agent_positions,
            "agent_energies": self.agent_energies,
            "food_patches": self.food_patches,
            "hazard_patches": self.hazard_patches,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Snapshot":
        """Create snapshot from serializable dictionary."""
        world_state = np.array(data["world_state"], dtype=np.int8)
        return cls(
            step=data["step"],
            world_state=world_state,
            agent_positions=data["agent_positions"],
            agent_energies=data["agent_energies"],
            food_patches=data.get("food_patches", []),
            hazard_patches=data.get("hazard_patches", []),
            metadata=data.get("metadata", {}),
        )

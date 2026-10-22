"""SimLab snapshot module - snapshot(), restore(), action-log record/replay."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from simlab.core.types import Snapshot, StepResult


@dataclass
class ActionLogEntry:
    """A single entry in the action log."""
    step: int
    agent_id: str
    action: Any  # Action object
    world_state_before: np.ndarray  # World state before the action
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to serializable dictionary."""
        return {
            "step": self.step,
            "agent_id": self.agent_id,
            "action": {
                "move_direction": self.action.move_direction,
                "energy_spent": self.action.energy_spent,
                "action_id": self.action.action_id,
            },
            "world_state_before": self.world_state_before.tolist(),
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ActionLogEntry":
        """Create from serializable dictionary."""
        from simlab.core.types import Action
        action = Action(
            move_direction=tuple(data["action"]["move_direction"]),
            energy_spent=data["action"]["energy_spent"],
            action_id=data["action"]["action_id"],
        )
        return cls(
            step=data["step"],
            agent_id=data["agent_id"],
            action=action,
            world_state_before=np.array(data["world_state_before"], dtype=np.int8),
        )


@dataclass
class Snapshot:
    """Immutable snapshot of the world state at a given step.
    
    Contains ALL world state for deterministic replay:
    - step number
    - world state array
    - agent positions and energies
    - food patch positions
    - hazard patch positions
    - RNG state for full determinism
    """
    step: int  # Step number
    world_state: np.ndarray  # World state array (shape: width x height)
    agent_positions: Dict[str, Tuple[int, int]]  # Agent positions
    agent_energies: Dict[str, float]  # Agent energy levels
    food_patches: List[Tuple[int, int]]  # Positions of food patches
    hazard_patches: List[Tuple[int, int]]  # Positions of hazard patches
    rng_state: Optional[Any] = None  # RNG state for determinism
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
            "rng_state": self.rng_state,
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
            rng_state=data.get("rng_state"),
            metadata=data.get("metadata", {}),
        )


class SnapshotManager:
    """Manages world snapshots for record/replay functionality.
    
    Provides snapshot capture, restore, and action-log recording
    for deterministic simulation replay.
    """
    
    def __init__(self) -> None:
        """Initialize an empty snapshot manager."""
        self._snapshots: Dict[int, Snapshot] = {}  # step -> snapshot
        self._action_log: List[ActionLogEntry] = []
        self._current_step = 0
    
    def capture_snapshot(
    self,
    world_state: np.ndarray,
    step: int,
    agent_positions: Dict[str, Tuple[int, int]] | None = None,
    agent_energies: Dict[str, float] | None = None,
    food_patches: Optional[List[Tuple[int, int]]] = None,
    hazard_patches: Optional[List[Tuple[int, int]]] = None,
) -> Snapshot:
        """Capture a snapshot of the world state at a given step.
        
        Args:
            world_state: Current world state array.
            step: Step number.
            agent_positions: Optional dict of agent_id -> (x, y) position.
            agent_energies: Optional dict of agent_id -> energy.
            food_patches: Optional list of food patch positions.
            hazard_patches: Optional list of hazard patch positions.
            
        Returns:
            Snapshot of the world state.
        """
        if agent_positions is None:
            agent_positions = {}
        if agent_energies is None:
            agent_energies = {}
        if food_patches is None:
            food_patches = []
        if hazard_patches is None:
            hazard_patches = []
        
        snapshot = Snapshot(
            step=step,
            world_state=world_state.copy(),
            agent_positions=agent_positions,
            agent_energies=agent_energies,
            food_patches=food_patches,
            hazard_patches=hazard_patches,
        )
        self._snapshots[step] = snapshot
        self._current_step = step
        return snapshot
    
    def restore_snapshot(self, step: int) -> Optional[Snapshot]:
        """Restore a snapshot from a given step.
        
        Args:
            step: Step number to restore.
            
        Returns:
            Snapshot if found, None otherwise.
        """
        if step in self._snapshots:
            return self._snapshots[step]
        return None
    
    def record_action(
        self,
        step: int,
        agent_id: str,
        action: Any,
        world_state_before: np.ndarray,
    ) -> None:
        """Record an action in the action log.
        
        Args:
            step: Step number.
            agent_id: ID of the agent taking the action.
            action: Action that was taken.
            world_state_before: World state before the action was taken.
        """
        entry = ActionLogEntry(
            step=step,
            agent_id=agent_id,
            action=action,
            world_state_before=world_state_before.copy(),
        )
        self._action_log.append(entry)
    
    def get_action_log(self) -> List[ActionLogEntry]:
        """Get the complete action log.
        
        Returns:
            List of action log entries.
        """
        return self._action_log
    
    def get_action_log_until(self, step: int) -> List[ActionLogEntry]:
        """Get action log entries up to a given step.
        
        Args:
            step: Maximum step number.
            
        Returns:
            List of action log entries up to the given step.
        """
        return [entry for entry in self._action_log if entry.step <= step]
    
    def replay_actions(
        self,
        world: "World",
        start_step: int = 0,
        end_step: Optional[int] = None,
    ) -> List[StepResult]:
        """Replay recorded actions to reproduce simulation steps.
        
        Args:
            world: World instance to replay on.
            start_step: Step to start replay from.
            end_step: Step to end replay at (inclusive). If None, replays to end.
            
        Returns:
            List of StepResult for each replayed step.
        """
        if end_step is None:
            end_step = max(
                (entry.step for entry in self._action_log),
                default=start_step,
            )
        
        results = []
        current_world_state = world.get_world_state()
        
        # Reset world to initial state
        world.reset()
        current_world_state = world.get_world_state()
        
        # Get actions for each step
        actions_by_step: Dict[int, Dict[str, Any]] = {}
        for entry in self._action_log:
            if entry.step not in actions_by_step:
                actions_by_step[entry.step] = {}
            actions_by_step[entry.step][entry.agent_id] = entry.action
        
        # Replay each step
        for step in range(start_step, end_step + 1):
            # Get actions for this step
            step_actions = actions_by_step.get(step, {})
            
            # Capture world state before action
            world_state_before = world.get_world_state().copy()
            
            # Record action log
            self.record_action(
                step=step,
                agent_id=list(step_actions.keys())[0] if step_actions else "",
                action=list(step_actions.values())[0] if step_actions else None,
                world_state_before=world_state_before,
            )
            
            # Take step in world
            # Note: This is simplified - real replay would need proper agent ID mapping
            result = world.step(step_actions)
            
            results.append(result)
        
        return results
    
    def get_snapshot(self, step: int) -> Optional[Snapshot]:
        """Get a snapshot by step number.
        
        Args:
            step: Step number.
            
        Returns:
            Snapshot if found, None otherwise.
        """
        return self._snapshots.get(step)

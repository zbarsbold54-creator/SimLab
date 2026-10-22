"""SimLab interfaces/policy module - Policy protocol."""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple
from typing_extensions import runtime_checkable
from typing import Protocol

import numpy as np

from simlab.core.types import Action


@runtime_checkable
class Policy(Protocol):
    """Protocol for decision policies.
    
    A policy determines what action an agent should take given
    its current state and the world state. Policies can be rule-based,
    heuristic-based, or learned from data.
    """
    
    def select_action(
        self,
        agent_state: Dict[str, Any],
        world_state: Dict[str, Any],
        available_actions: List[Dict[str, Any]],
        step: int,
    ) -> Action:
        """Select an action for the agent based on current state.
        
        Args:
            agent_state: Dictionary containing agent's internal state.
            world_state: Dictionary containing world state.
            available_actions: List of available action options.
            step: Current simulation step number.
            
        Returns:
            Selected action as an Action dataclass.
        """
        ...


# Heuristic policy implementation
class HeuristicPolicy:
    """Heuristic-based policy that selects actions based on simple rules.
    
    Prioritizes avoiding hazards when close, seeking food when energy is low,
    and defaulting to no-op otherwise.
    """
    
    def select_action(
        self,
        agent_state: Dict[str, Any],
        world_state: Dict[str, Any],
        available_actions: List[Dict[str, Any]],
        step: int,
    ) -> Action:
        """Select an action for the agent based on current state.
        
        Args:
            agent_state: Dictionary containing agent's internal state.
            world_state: Dictionary containing world state.
            available_actions: List of available action options.
            step: Current simulation step number.
            
        Returns:
            Selected action as an Action dataclass.
        """
        energy = agent_state.get("energy", 1.0)
        position = agent_state.get("position", (0, 0))
        food_locations = world_state.get("food_locations", [])
        hazard_locations = world_state.get("hazard_locations", [])
        
        # Determine desired direction based on world state
        desired_dx, desired_dy = 0, 0  # Default: no movement
        
        # Priority 1: Avoid hazards if nearby
        if hazard_locations:
            nearest_hazard = min(
                hazard_locations,
                key=lambda h: abs(h[0] - position[0]) + abs(h[1] - position[1]),
            )
            dx = -np.sign(nearest_hazard[0] - position[0])
            dy = -np.sign(nearest_hazard[1] - position[1])
            # Only avoid if hazard is close (within 5 units)
            dist = abs(nearest_hazard[0] - position[0]) + abs(nearest_hazard[1] - position[1])
            if dist <= 5:
                desired_dx, desired_dy = dx, dy
        
        # Priority 2: Seek food if energy is low
        if energy < 0.5 and food_locations:
            nearest_food = min(
                food_locations,
                key=lambda f: abs(f[0] - position[0]) + abs(f[1] - position[1]),
            )
            dx = np.sign(nearest_food[0] - position[0])
            dy = np.sign(nearest_food[1] - position[1])
            # Only seek food if we're not already avoiding a hazard
            if desired_dx == 0 and desired_dy == 0:
                desired_dx, desired_dy = dx, dy
        
        # If we have a desired direction, use it
        if desired_dx != 0 or desired_dy != 0:
            return Action(
                move_direction=(int(desired_dx), int(desired_dy)),
                energy_spent=1.0,
                action_id="heuristic",
            )
        
        # Default: no-op if no movement needed
        return Action(
            move_direction=(0, 0), energy_spent=0.0, action_id="noop",
        )


# Example learned policy placeholder
class LearnedPolicy:
    """Placeholder for learned policy (neural network-based).
    
    This would be replaced by a neural network policy in future phases.
    Currently returns a no-op action.
    """
    
    def select_action(
        self,
        agent_state: Dict[str, Any],
        world_state: Dict[str, Any],
        available_actions: List[Dict[str, Any]],
        step: int,
    ) -> Dict[str, Any]:
        """Select an action for the agent based on current state.
        
        Args:
            agent_state: Dictionary containing agent's internal state.
            world_state: Dictionary containing world state.
            available_actions: List of available action options.
            step: Current simulation step number.
            
        Returns:
            Selected action dictionary (noop by default).
        """
        # Placeholder - return noop until neural network is implemented
        return {"move_direction": (0, 0), "energy_spent": 0.0, "action_id": "noop"}
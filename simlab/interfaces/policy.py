"""SimLab interfaces/policy module - Policy protocol."""

from __future__ import annotations

from typing import Any, Protocol, runtime_checkable


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
    ) -> Dict[str, Any]:
        """Select an action for the agent based on current state.
        
        Args:
            agent_state: Dictionary containing agent's internal state:
                - energy: Normalized energy level (0.0 - 1.0)
                - position: (x, y) position
                - step: Current step number
            world_state: Dictionary containing world state:
                - food_locations: List of (x, y) food positions
                - hazard_locations: List of (x, y) hazard positions
                - step: Current step number
            available_actions: List of available action options.
            step: Current simulation step number.
            
        Returns:
            Selected action as a dictionary with action parameters.
        """
        ...


# Example policy implementation
class HeuristicPolicy:
    """Heuristic-based policy that selects actions based on simple rules.
    
    Prioritizes eating when hungry, avoiding hazards, and moving
    toward food sources.
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
            Selected action dictionary.
        """
        energy = agent_state.get("energy", 1.0)
        position = agent_state.get("position", (0, 0))
        food_locations = world_state.get("food_locations", [])
        hazard_locations = world_state.get("hazard_locations", [])
        
        # No available actions - returnnoop
        if not available_actions:
            return {"move_direction": (0, 0), "energy_spent": 0.0, "action_id": "noop"}
        
        # Find nearest food if energy is low
        if energy < 0.5 and food_locations:
            nearest_food = min(
                food_locations,
                key=lambda f: abs(f[0] - position[0]) + abs(f[1] - position[1]),
            )
            dx = np.sign(nearest_food[0] - position[0])
            dy = np.sign(nearest_food[1] - position[1])
            
            # Find matching action that moves toward food
            for action in available_actions:
                action_dir = action.get("move_direction", (0, 0))
                if (action_dir[0] == dx and action_dir[1] == dy):
                    return action
        
        # Avoid hazards
        if hazard_locations:
            nearest_hazard = min(
                hazard_locations,
                key=lambda h: abs(h[0] - position[0]) + abs(h[1] - position[1]),
            )
            # Move away from hazard
            dx = -np.sign(nearest_hazard[0] - position[0])
            dy = -np.sign(nearest_hazard[1] - position[1])
            
            for action in available_actions:
                action_dir = action.get("move_direction", (0, 0))
                if (action_dir[0] == dx and action_dir[1] == dy):
                    return action
        
        # Default: move forward (first available action)
        return available_actions[0]


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

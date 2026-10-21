"""SimLab interfaces/drive_core module - DriveCore protocol + RetentionRule protocol."""

from __future__ import annotations

from typing import Any, Protocol, runtime_checkable


@runtime_checkable
class RetentionRule(Protocol):
    """Protocol for drive retention rules.
    
    Determines whether a drive should retain or discard a given state
    based on the agent's current situation.
    """
    
    def should_retain(
        self,
        agent_energy: float,
        step: int,
        context: Dict[str, Any],
    ) -> bool:
        """Determine if the drive should retain the current state.
        
        Args:
            agent_energy: Current agent energy level (0.0 - 1.0).
            step: Current simulation step number.
            context: Additional context about the agent's situation.
            
        Returns:
            True if the drive should retain the state, False otherwise.
        """
        ...


@runtime_checkable
class DriveCore(Protocol):
    """Protocol for drive cores.
    
    A drive core determines what drives an agent based on its
    internal state, needs, and environmental context.
    """
    
    def compute_drives(
        self,
        agent_state: Dict[str, Any],
        world_state: Dict[str, Any],
        retention_rule: RetentionRule,
    ) -> Dict[str, float]:
        """Compute the drive values for the agent.
        
        Args:
            agent_state: Dictionary containing agent's internal state:
                - energy: Normalized energy level (0.0 - 1.0)
                - position: (x, y) position
                - step: Current step number
            world_state: Dictionary containing world state:
                - food_locations: List of (x, y) food positions
                - hazard_locations: List of (x, y) hazard positions
                - step: Current step number
            retention_rule: Retention rule to apply.
            
        Returns:
            Dictionary mapping drive names to drive strength values (0.0 - 1.0).
        """
        ...


# Example retention rule implementation
class EnergyBasedRetention:
    """Retention rule based on agent energy levels.
    
    Retains states when agent has sufficient energy, discards when
    energy is low to prioritize survival.
    """
    
    def __init__(self, energy_threshold: float = 0.3) -> None:
        """Initialize the retention rule.
        
        Args:
            energy_threshold: Energy level below which states are discarded.
        """
        self._energy_threshold = energy_threshold
    
    def should_retain(
        self,
        agent_energy: float,
        step: int,
        context: Dict[str, Any],
    ) -> bool:
        """Determine if the drive should retain the current state.
        
        Args:
            agent_energy: Current agent energy level (0.0 - 1.0).
            step: Current simulation step number.
            context: Additional context about the agent's situation.
            
        Returns:
            True if the drive should retain the state, False otherwise.
        """
        return agent_energy >= self._energy_threshold


# Example drive core implementation
class BasicDriveCore:
    """Basic drive core that computes drives based on needs.
    
    Computes drives for eating, avoiding hazards, and resting
    based on the agent's current state and world conditions.
    """
    
    def compute_drives(
        self,
        agent_state: Dict[str, Any],
        world_state: Dict[str, Any],
        retention_rule: RetentionRule,
    ) -> Dict[str, float]:
        """Compute the drive values for the agent.
        
        Args:
            agent_state: Dictionary containing agent's internal state.
            world_state: Dictionary containing world state.
            retention_rule: Retention rule to apply.
            
        Returns:
            Dictionary mapping drive names to drive strength values (0.0 - 1.0).
        """
        energy = agent_state.get("energy", 1.0)
        position = agent_state.get("position", (0, 0))
        food_locations = world_state.get("food_locations", [])
        hazard_locations = world_state.get("hazard_locations", [])
        
        # Apply retention check
        if not retention_rule.should_retain(energy, agent_state.get("step", 0), {}):
            # If retention says discard, return minimal drives
            return {"drive_eat": 0.0, "drive_avoid": 0.0, "drive_rest": 0.0}
        
        # Compute individual drives
        drives = {}
        
        # Eat drive: seek food if hungry
        if energy < 0.7 and food_locations:
            # Find nearest food
            nearest_food = min(
                food_locations,
                key=lambda f: abs(f[0] - position[0]) + abs(f[1] - position[1]),
            )
            distance = abs(nearest_food[0] - position[0]) + abs(nearest_food[1] - position[1])
            drives["drive_eat"] = max(0.0, 1.0 - distance / 10.0)
        else:
            drives["drive_eat"] = 0.0
        
        # Avoid drive: stay away from hazards
        if hazard_locations:
            # Find nearest hazard
            nearest_hazard = min(
                hazard_locations,
                key=lambda h: abs(h[0] - position[0]) + abs(h[1] - position[1]),
            )
            distance = abs(nearest_hazard[0] - position[0]) + abs(nearest_hazard[1] - position[1])
            drives["drive_avoid"] = max(0.0, 1.0 - distance / 10.0)
        else:
            drives["drive_avoid"] = 0.0
        
        # Rest drive: rest when energy is high enough
        if energy < 1.0:
            drives["drive_rest"] = 0.1
        else:
            drives["drive_rest"] = 0.0
        
        # Normalize drives
        total = sum(drives.values())
        if total > 0:
            drives = {k: v / total for k, v in drives.items()}
        
        return drives

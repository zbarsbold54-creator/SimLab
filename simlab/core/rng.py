"""SimLab RNG module - single seeded numpy Generator, spawn child streams."""

import numpy as np
from typing import Any, Optional


class SimRNG:
    """Single seeded numpy Generator with child stream spawning.
    
    Provides deterministic random number generation for the simulation.
    Each RNG instance can spawn child streams for parallel/sub-simulation
    scenarios while maintaining reproducibility.
    """
    
    def __init__(self, seed: int = 42) -> None:
        """Initialize RNG with a specific seed.
        
        Args:
            seed: Random seed value for reproducibility.
        """
        self._rng = np.random.default_rng(seed)
    
    def spawn(self, stream_id: int) -> "SimRNG":
        """Spawn a child RNG stream.
        
        Creates a child RNG with a deterministic seed derived from
        the parent seed and stream ID, ensuring reproducible
        sub-simulations.
        
        Args:
            stream_id: Unique identifier for the child stream.
            
        Returns:
            A new SimRNG instance with a child stream seed.
        """
        child_seed = int(self._rng.integers(2**31)) ^ stream_id
        return SimRNG(child_seed)
    
    def random(self) -> float:
        """Generate a random float in [0.0, 1.0)."""
        return float(self._rng.random())
    
    def random_int(self, low: int, high: int) -> int:
        """Generate a random integer in [low, high).
        
        Args:
            low: Lower bound (inclusive).
            high: Upper bound (exclusive).
            
        Returns:
            Random integer in [low, high).
        """
        return int(self._rng.integers(low, high))
    
    def random_choice(self, population: list) -> Any:
        """Select a random element from a population.
        
        Args:
            population: List of elements to choose from.
            
        Returns:
            A randomly selected element.
        """
        return self._rng.choice(population)
    
    def random_sample(self, population: list, k: int = 1) -> list:
        """Sample k unique elements from a population.
        
        Args:
            population: List of elements to sample from.
            k: Number of elements to sample.
            
        Returns:
            List of k randomly selected unique elements.
        """
        return self._rng.choice(population, size=k, replace=False).tolist()
    
    def shuffle(self, array: np.ndarray) -> np.ndarray:
        """Shuffle an array in-place.
        
        Args:
            array: numpy array to shuffle.
            
        Returns:
            The shuffled array.
        """
        self._rng.shuffle(array)
        return array
    
    def choice_probabilistic(self, choices: list, p: list) -> Any:
        """Choose randomly from a list with given probabilities.
        
        Args:
            choices: List of possible choices.
            p: List of probabilities corresponding to each choice.
            
        Returns:
            A randomly chosen element based on the probability distribution.
        """
        return self._rng.choice(choices, p=p)
    
    @property
    def state(self) -> dict:
        """Get the current RNG state for reproducibility.
        
        Returns:
            Dictionary containing the RNG internal state.
        """
        return dict(self._rng)
    
    @state.setter
    def state(self, value: dict) -> None:
        """Set the RNG state from a dictionary.
        
        Args:
            value: Dictionary containing RNG internal state.
        """
        self._rng = np.random.default_rng(value)

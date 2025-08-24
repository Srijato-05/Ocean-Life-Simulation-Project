# src/agents/small_fish.py

import math
import random
from .base_agent import BaseAgent

class SmallFish(BaseAgent):
    """
    Represents a predator/omnivore agent.
    All simulation logic is handled by the vectorized SimulationManager.
    """

    def __init__(self, env, species_config, initial_position=None):
        # The 'movement_behavior' argument is removed as it's no longer used.
        super().__init__(env, species_config, initial_position)
        # All other methods (find_nearest_prey, eat, update) are removed as their
        # functionality is now handled more efficiently by the SimulationManager.
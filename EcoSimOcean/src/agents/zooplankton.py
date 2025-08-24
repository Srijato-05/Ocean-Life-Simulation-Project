# src/agents/zooplankton.py

from .base_agent import BaseAgent

class Zooplankton(BaseAgent):
    """
    Represents a simple herbivorous agent.
    All simulation logic is handled by the vectorized SimulationManager.
    """

    def __init__(self, env, species_config, initial_position=None):
        """
        Initializes a Zooplankton agent.
        """
        # The 'movement_behavior' argument is removed as it's no longer used.
        super().__init__(env, species_config, initial_position)
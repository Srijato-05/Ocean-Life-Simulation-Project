# src/agents/sea_turtle.py

from .base_agent import BaseAgent

class SeaTurtle(BaseAgent):
    """
    Represents a herbivore agent (Marine Reptile) that competes with Zooplankton.
    Its behavior is fully vectorized and managed by the SimulationManager.
    """
    def __init__(self, env, species_config, initial_position=None):
        # The 'movement_behavior' argument is removed as it's no longer used.
        super().__init__(env, species_config, initial_position)
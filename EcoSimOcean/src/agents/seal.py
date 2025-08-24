# src/agents/seal.py

from .base_agent import BaseAgent

class Seal(BaseAgent):
    """
    Represents an apex predator agent (Marine Mammal).
    Its behavior is fully vectorized and managed by the SimulationManager.
    """
    def __init__(self, env, species_config, initial_position=None):
        # The 'movement_behavior' argument is removed as it's no longer used.
        super().__init__(env, species_config, initial_position)
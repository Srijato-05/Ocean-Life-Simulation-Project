# src/agents/crab.py

from .base_agent import BaseAgent

class Crab(BaseAgent):
    """
    Represents a scavenger agent that primarily consumes marine snow.
    Its movement is handled by the simulation manager.
    """
    def __init__(self, env, species_config, initial_position=None):
        # The 'movement_behavior' argument is removed as it's no longer used.
        super().__init__(env, species_config, initial_position)
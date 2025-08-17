# src/agents/crab.py

from .base_agent import BaseAgent
from .behaviors import RandomWalk

class Crab(BaseAgent):
    """
    Represents a scavenger agent that primarily consumes marine snow.
    Its movement will be handled by the simulation manager.
    """
    def __init__(self, env, species_config, movement_behavior=None, initial_position=None):
        # Crabs have simple, manager-driven movement, so we default to RandomWalk.
        if movement_behavior is None:
            movement_behavior = RandomWalk()
        super().__init__(env, species_config, movement_behavior, initial_position)
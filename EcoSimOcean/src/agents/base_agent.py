# src/agents/base_agent.py

import random
import itertools

class BaseAgent:
    """
    The fundamental blueprint for all living organisms.
    It now delegates its movement logic to a swappable behavior object,
    following the Strategy design pattern.
    """
    id_counter = itertools.count()

    def __init__(self, env, species_config, movement_behavior, initial_position=None):
        """
        Initializes a new agent.

        Args:
            env (Environment): The simulation environment instance.
            species_config (dict): A dictionary of parameters for this species.
            movement_behavior (MovementBehavior): An object that defines how this
                                                  agent will move.
            initial_position (tuple, optional): The (x, y, z) starting position.
        """
        self.id = next(BaseAgent.id_counter)
        self.env = env
        self.config = species_config
        self.movement_behavior = movement_behavior  # Assign the movement strategy

        # Core Attributes
        self.species = self.config.get("species_name", "Generic")
        self.energy = self.config.get("initial_energy", 10.0)
        self.alive = True

        # Positional Attributes
        if initial_position:
            self.x, self.y, self.z = initial_position
        else:
            self.x = random.randint(0, self.env.width - 1)
            self.y = random.randint(0, self.env.height - 1)
            self.z = random.randint(0, self.env.depth - 1)

    def move(self):
        """
        Executes a move by calling the current movement_behavior's move method.
        """
        if not self.alive:
            return
        
        # Delegate the actual movement logic to the behavior object.
        self.movement_behavior.move(self)
        
        # Apply energy cost for movement, regardless of the strategy used.
        movement_cost = self.config.get("movement_cost", 0.1)
        self.energy -= movement_cost

    def update(self):
        """
        The main update loop for the agent, called once per simulation tick.
        """
        if not self.alive:
            return

        # Apply base metabolic cost
        metabolic_rate = self.config.get("metabolic_rate", 0.05)
        self.energy -= metabolic_rate
        
        # Perform the move action
        self.move()

        # Check for death from energy loss
        if self.energy <= 0:
            self.die()

    def die(self):
        """
        Handles the agent's death.
        """
        self.alive = False
        agent_size = self.config.get("size", 1.0)
        self.env.deposit_marine_snow(self.x, self.y, self.z, agent_size)

    def __repr__(self):
        """Provides a developer-friendly string representation of the agent."""
        return (f"{self.species}(id={self.id}, pos=({self.x},{self.y},{self.z}), "
                f"energy={self.energy:.2f}, alive={self.alive})")
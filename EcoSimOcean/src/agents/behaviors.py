# src/agents/behaviors.py

import random

class MovementBehavior:
    """
    An abstract base class for all movement strategies.
    This defines the interface that all concrete behaviors must follow.
    """
    def move(self, agent, neighbors):
        """
        Executes the movement logic for the given agent.

        Args:
            agent (BaseAgent): The agent instance to move.
            neighbors (list): A list of nearby agent objects.
        """
        raise NotImplementedError("Subclasses must implement this method.")

class RandomWalk(MovementBehavior):
    """
    Implements a simple random walk in 3D space.
    The agent moves one step in a random direction.
    """
    def move(self, agent, neighbors=None):
        dx = random.choice([-1, 0, 1])
        dy = random.choice([-1, 0, 1])
        dz = random.choice([-1, 0, 1])

        agent.x = (agent.x + dx) % agent.env.width
        agent.y = (agent.y + dy) % agent.env.height
        agent.z = max(0, min(agent.env.depth - 1, agent.z + dz))

class ChasePrey(MovementBehavior):
    """
    Implements a chase behavior to move towards the nearest prey.
    If no prey is visible, it defaults to a random walk.
    """
    def move(self, agent, neighbors):
        target = getattr(agent, 'target', None)

        if not target or not target.alive:
            # If no target, perform a random walk as a fallback.
            RandomWalk().move(agent)
            return

        # Move one step towards the target's coordinates.
        if target.x > agent.x: dx = 1
        elif target.x < agent.x: dx = -1
        else: dx = 0

        if target.y > agent.y: dy = 1
        elif target.y < agent.y: dy = -1
        else: dy = 0

        if target.z > agent.z: dz = 1
        elif target.z < agent.z: dz = -1
        else: dz = 0

        agent.x = (agent.x + dx) % agent.env.width
        agent.y = (agent.y + dy) % agent.env.height
        agent.z = max(0, min(agent.env.depth - 1, agent.z + dz))

# --- HIGHLIGHT: New FleeFromPredator Behavior ---
class FleeFromPredator(MovementBehavior):
    """
    Implements an evasion behavior to move away from the nearest predator.
    If no predator is near, it defaults to a random walk.
    """
    def move(self, agent, neighbors):
        """
        Calculates a vector away from the nearest predator and moves the
        agent in that direction.
        """
        from .small_fish import SmallFish # Local import to avoid circular dependency
        
        # Find all nearby predators from the list of neighbors
        predators = [n for n in neighbors if isinstance(n, SmallFish)]
        
        if not predators:
            # If no predators are near, perform a random walk.
            RandomWalk().move(agent)
            return
            
        # Find the single closest predator
        closest_predator = min(predators, key=lambda p: 
            (agent.x - p.x)**2 + (agent.y - p.y)**2 + (agent.z - p.z)**2)

        # Calculate the vector pointing *away* from the predator
        flee_dx = agent.x - closest_predator.x
        flee_dy = agent.y - closest_predator.y
        flee_dz = agent.z - closest_predator.z

        # Move one step in the direction of the flee vector
        dx = 1 if flee_dx > 0 else -1 if flee_dx < 0 else 0
        dy = 1 if flee_dy > 0 else -1 if flee_dy < 0 else 0
        dz = 1 if flee_dz > 0 else -1 if flee_dz < 0 else 0

        agent.x = (agent.x + dx) % agent.env.width
        agent.y = (agent.y + dy) % agent.env.height
        agent.z = max(0, min(agent.env.depth - 1, agent.z + dz))
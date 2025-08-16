# src/agents/zooplankton.py

from .base_agent import BaseAgent
from .behaviors import RandomWalk, FleeFromPredator # Import the new behavior

class Zooplankton(BaseAgent):
    """
    Represents a simple herbivorous agent that now has the ability to flee
    from nearby predators.
    """

    def __init__(self, env, species_config, movement_behavior, initial_position=None):
        """
        Initializes a Zooplankton agent.
        """
        super().__init__(env, species_config, movement_behavior, initial_position)
        # --- HIGHLIGHT: Add a new attribute for fear radius ---
        self.fear_radius = self.config.get("fear_radius", 10.0)

    def eat(self):
        """
        Consumes food from the current grid cell in the environment.
        """
        if not self.alive:
            return

        eating_rate = self.config.get("eating_rate", 0.5)
        energy_conversion = self.config.get("energy_conversion_factor", 5.0)

        food_available = self.env.food[self.x, self.y, self.z]
        
        amount_to_eat = min(food_available, eating_rate)
        
        if amount_to_eat > 0:
            self.env.food[self.x, self.y, self.z] -= amount_to_eat
            self.energy += amount_to_eat * energy_conversion

    def update(self, local_neighbors):
        """
        Overrides the BaseAgent's update method to include evasion logic.
        
        The agent first checks for predators, then decides its movement
        behavior before performing the standard updates.
        """
        if not self.alive:
            return
        
        # --- HIGHLIGHT: New Evasion Logic ---
        from .small_fish import SmallFish # Local import to avoid circular dependency
        
        # Scan local neighbors for predators
        nearby_predators = [n for n in local_neighbors if isinstance(n, SmallFish)]
        
        # Check if any of those predators are within the fear radius
        is_in_danger = False
        if nearby_predators:
            closest_predator = min(nearby_predators, key=lambda p:
                (self.x - p.x)**2 + (self.y - p.y)**2 + (self.z - p.z)**2)
            
            distance_to_predator = ( (self.x - closest_predator.x)**2 + 
                                     (self.y - closest_predator.y)**2 + 
                                     (self.z - closest_predator.z)**2 )**0.5
            
            if distance_to_predator < self.fear_radius:
                is_in_danger = True

        # Dynamically switch movement behavior based on threat
        if is_in_danger:
            self.movement_behavior = FleeFromPredator()
        else:
            self.movement_behavior = RandomWalk()
        # --- End of New Logic ---

        self.eat()
        
        # The call to super().update() will now use the chosen behavior
        super().update(local_neighbors)
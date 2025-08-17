# src/agents/small_fish.py

import math
import random
from .base_agent import BaseAgent
from .behaviors import ChasePrey, RandomWalk
from .zooplankton import Zooplankton

class SmallFish(BaseAgent):
    """
    Represents a predator agent that now gives up hunting when prey is scarce
    and has its vision impaired in refuge zones. This is the final, balanced version.
    """

    def __init__(self, env, species_config, movement_behavior, initial_position=None):
        super().__init__(env, species_config, movement_behavior, initial_position)
        self.target = None
        self.reproduction_cooldown = 0

    def find_nearest_prey(self, prey_list):
        """
        Scans for the closest prey, with vision impaired by refuge zones.
        """
        vision_radius = self.config.get("vision_radius", 15.0)
        
        # --- HIGHLIGHT: New Refuge Logic ---
        # Check if the fish is in a refuge zone and reduce vision if so
        current_pos_x, current_pos_y, current_pos_z = int(self.x), int(self.y), int(self.z)
        if self.env.refuge_map[current_pos_x, current_pos_y, current_pos_z] == 1.0:
            vision_radius *= self.config.get("refuge_vision_modifier", 0.5)
        # --- END: New Refuge Logic ---
            
        closest_prey = None
        min_distance = float('inf')

        for prey in prey_list:
            if prey.alive:
                distance = math.sqrt(
                    (self.x - prey.x)**2 +
                    (self.y - prey.y)**2 +
                    (self.z - prey.z)**2
                )
                if distance < vision_radius and distance < min_distance:
                    min_distance = distance
                    closest_prey = prey
        
        return closest_prey

    def eat(self, prey_agent):
        """
        Consumes a prey agent, transferring its energy.
        """
        if not self.alive or not prey_agent.alive:
            return
        efficiency = self.config.get("energy_transfer_efficiency", 0.8)
        self.energy += prey_agent.energy * efficiency
        prey_agent.die()
        self.target = None

    def update(self, local_neighbors, global_prey_count):
        """
        The update method now checks for prey scarcity before deciding to hunt.
        """
        if not self.alive:
            return

        if self.reproduction_cooldown > 0:
            self.reproduction_cooldown -= 1

        overcrowding_penalty = self.config.get("overcrowding_penalty", 0.05)
        nearby_predators = [n for n in local_neighbors if isinstance(n, SmallFish)]
        self.energy -= len(nearby_predators) * overcrowding_penalty

        # --- Prey Scarcity Logic ---
        scarcity_threshold = self.config.get("prey_scarcity_threshold", 50)
        should_hunt = global_prey_count > scarcity_threshold

        if should_hunt:
            if not self.target or not self.target.alive:
                potential_prey = [n for n in local_neighbors if isinstance(n, Zooplankton)]
                self.target = self.find_nearest_prey(potential_prey)
        else:
            # If prey is scarce, give up the chase to conserve energy
            self.target = None

        if self.target:
            self.movement_behavior = ChasePrey()
        else:
            self.movement_behavior = RandomWalk()

        super().update()
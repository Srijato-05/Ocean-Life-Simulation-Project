# src/optimizer/particle.py

"""
This file defines the Particle class used in the Particle Swarm Optimization.
Each particle represents a complete, testable ecosystem configuration.
This version can tune parameters for both prey and predators.
"""

import random
from copy import deepcopy

class Particle:
    """Represents one set of parameters (a particle) in the swarm."""
    
    def __init__(self, base_sim_config, base_fauna_config, param_bounds):
        self.sim_config = deepcopy(base_sim_config)
        self.fauna_config = deepcopy(base_fauna_config)
        self.param_bounds = param_bounds
        
        self._initialize_params()
        
        self.velocity = self._initialize_velocity()
        
        self.best_sim_config = deepcopy(self.sim_config)
        self.best_fauna_config = deepcopy(self.fauna_config)
        self.best_score = -1

    def _initialize_params(self):
        """Randomizes all tunable parameters within their defined bounds."""
        for key, bounds in self.param_bounds.items():
            min_val, max_val = bounds
            # --- HIGHLIGHT: Renamed keys to be specific ---
            sim_key = key.replace('_prey', '').replace('_predator', '')
            
            if sim_key in self.sim_config:
                self.sim_config[sim_key] = random.uniform(min_val, max_val)
            elif key.endswith('_prey'):
                self.fauna_config["Zooplankton"][sim_key] = random.uniform(min_val, max_val)
            elif key.endswith('_predator'):
                 self.fauna_config["SmallFish"][sim_key] = random.uniform(min_val, max_val)
            # Fallback for keys that aren't species-specific
            elif sim_key in self.fauna_config["Zooplankton"]:
                 self.fauna_config["Zooplankton"][sim_key] = random.uniform(min_val, max_val)
            elif sim_key in self.fauna_config["SmallFish"]:
                 self.fauna_config["SmallFish"][sim_key] = random.uniform(min_val, max_val)

    def _initialize_velocity(self):
        """Creates a small, random initial velocity for each parameter."""
        velocity = {}
        for key, bounds in self.param_bounds.items():
            min_val, max_val = bounds
            velocity[key] = (max_val - min_val) * random.uniform(-0.1, 0.1)
        return velocity

    def update_velocity(self, global_best_sim, global_best_fauna, pso_config):
        """Updates the particle's velocity based on its personal and global best."""
        w = pso_config["inertia"]
        c1 = pso_config["cognitive_weight"]
        c2 = pso_config["social_weight"]

        for key in self.velocity.keys():
            r1, r2 = random.random(), random.random()
            
            sim_key = key.replace('_prey', '').replace('_predator', '')
            p_best_val, g_best_val, current_val = 0, 0, 0
            
            # --- HIGHLIGHT: Added logic for Prey parameters ---
            if sim_key in self.sim_config:
                p_best_val = self.best_sim_config[sim_key]
                g_best_val = global_best_sim.get(sim_key, p_best_val)
                current_val = self.sim_config[sim_key]
            elif key.endswith('_prey'):
                p_best_val = self.best_fauna_config["Zooplankton"][sim_key]
                g_best_val = global_best_fauna.get("Zooplankton", {}).get(sim_key, p_best_val)
                current_val = self.fauna_config["Zooplankton"][sim_key]
            elif key.endswith('_predator'):
                p_best_val = self.best_fauna_config["SmallFish"][sim_key]
                g_best_val = global_best_fauna.get("SmallFish", {}).get(sim_key, p_best_val)
                current_val = self.fauna_config["SmallFish"][sim_key]
            # Fallback logic
            elif sim_key in self.fauna_config["Zooplankton"]:
                p_best_val = self.best_fauna_config["Zooplankton"][sim_key]
                g_best_val = global_best_fauna.get("Zooplankton", {}).get(sim_key, p_best_val)
                current_val = self.fauna_config["Zooplankton"][sim_key]
            elif sim_key in self.fauna_config["SmallFish"]:
                p_best_val = self.best_fauna_config["SmallFish"][sim_key]
                g_best_val = global_best_fauna.get("SmallFish", {}).get(sim_key, p_best_val)
                current_val = self.fauna_config["SmallFish"][sim_key]
            else:
                continue

            cognitive_component = c1 * r1 * (p_best_val - current_val)
            social_component = c2 * r2 * (g_best_val - current_val)
            self.velocity[key] = (w * self.velocity[key]) + cognitive_component + social_component

    def update_position(self):
        """Updates the particle's parameters by applying its velocity."""
        for key, vel in self.velocity.items():
            if key not in self.param_bounds: continue
            min_val, max_val = self.param_bounds[key]
            
            sim_key = key.replace('_prey', '').replace('_predator', '')

            # --- HIGHLIGHT: Added logic for Prey parameters ---
            if sim_key in self.sim_config:
                self.sim_config[sim_key] += vel
                self.sim_config[sim_key] = max(min_val, min(self.sim_config[sim_key], max_val))
            elif key.endswith('_prey'):
                self.fauna_config["Zooplankton"][sim_key] += vel
                self.fauna_config["Zooplankton"][sim_key] = max(min_val, min(self.fauna_config["Zooplankton"][sim_key], max_val))
            elif key.endswith('_predator'):
                self.fauna_config["SmallFish"][sim_key] += vel
                self.fauna_config["SmallFish"][sim_key] = max(min_val, min(self.fauna_config["SmallFish"][sim_key], max_val))
            # Fallback logic
            elif sim_key in self.fauna_config["Zooplankton"]:
                self.fauna_config["Zooplankton"][sim_key] += vel
                self.fauna_config["Zooplankton"][sim_key] = max(min_val, min(self.fauna_config["Zooplankton"][sim_key], max_val))
            elif sim_key in self.fauna_config["SmallFish"]:
                self.fauna_config["SmallFish"][sim_key] += vel
                self.fauna_config["SmallFish"][sim_key] = max(min_val, min(self.fauna_config["SmallFish"][sim_key], max_val))

            # Ensure integer-based parameters are correctly typed
            if any(k in key for k in ["period", "count", "threshold", "age"]):
                if key.endswith('_prey'):
                    self.fauna_config["Zooplankton"][sim_key] = int(self.fauna_config["Zooplankton"][sim_key])
                elif key.endswith('_predator'):
                    self.fauna_config["SmallFish"][sim_key] = int(self.fauna_config["SmallFish"][sim_key])
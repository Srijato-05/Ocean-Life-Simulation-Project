# src/optimizer/pso_config.py

"""
This file contains all the configuration settings for the Particle Swarm
Optimization (PSO) algorithm. This version provides a holistic search space
for all key environmental and species parameters.
"""

# --- Particle Swarm Optimization (PSO) Configuration ---
PSO_CONFIG = {
    "num_particles": 20,
    "num_iterations": 50,
    "inertia_start": 0.9,
    "inertia_end": 0.4,
    "cognitive_weight": 2.0,
    "social_weight": 2.0
}

# --- HOLISTIC PARAMETER BOUNDS ---
# This dictionary now defines the search space for all key parameters.
PARAM_BOUNDS = {
    # Environmental Parameters
    "food_growth_rate": (0.01, 0.2),
    
    # Initial Conditions
    "initial_zooplankton_count": (100, 1500),
    "initial_small_fish_count": (10, 100),

    # --- Zooplankton (Prey) Parameters ---
    "reproduction_threshold_prey": (15.0, 30.0),
    "carrying_capacity_threshold": (5, 15),
    "starvation_chance": (0.1, 0.6),
    "disease_threshold": (1000, 4000),
    "disease_chance": (0.05, 0.3),
    "eating_rate": (0.3, 1.0),
    "energy_conversion_factor": (4.0, 8.0),

    # --- SmallFish (Predator) Parameters ---
    "metabolic_rate": (0.1, 1.0),
    "movement_cost": (0.2, 1.5),
    "vision_radius": (8.0, 20.0),
    "predation_range": (1.0, 3.0),
    "energy_transfer_efficiency": (0.6, 0.9),
    "reproduction_threshold_predator": (40.0, 80.0),
    "overcrowding_penalty": (0.1, 0.8),
    "reproduction_cooldown_period": (20, 80),
    "hunt_success_chance": (0.5, 0.9),
    "satiation_period": (5, 40),
    "maturity_age": (15, 60),
    "juvenile_hunt_modifier": (0.3, 0.9),
    "prey_scarcity_threshold": (50, 200)
}
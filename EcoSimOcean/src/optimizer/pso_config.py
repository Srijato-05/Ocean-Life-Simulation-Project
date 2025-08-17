# src/optimizer/pso_config.py

"""
This file contains all the configuration settings for the Particle Swarm
Optimization (PSO) algorithm. This version provides a vastly expanded search
space to explore more extreme parameter combinations.
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

# --- VASTLY EXPANDED PARAMETER BOUNDS ---
PARAM_BOUNDS = {
    # Environmental Parameters
    "plankton_max_growth_rate": (0.1, 1.0),

    # Initial Conditions
    "initial_zooplankton_count": (500, 3000),
    "initial_small_fish_count": (20, 200),
    "initial_crab_count": (10, 100),

    # --- Zooplankton (Prey) Parameters ---
    "reproduction_threshold_prey": (10.0, 50.0),
    "carrying_capacity_threshold": (5, 25),
    "starvation_chance": (0.05, 0.7),
    "disease_threshold": (1000, 6000),
    "disease_chance": (0.01, 0.4),
    "eating_rate_prey": (0.3, 1.0),
    "energy_conversion_factor_prey": (3.0, 10.0),
    "max_lifespan_prey": (100, 500),

    # --- SmallFish (Predator) Parameters ---
    "metabolic_rate_predator": (0.1, 1.5),
    "reproduction_threshold_predator": (30.0, 120.0),
    "vision_radius": (5.0, 25.0),
    "predation_range": (1.0, 4.0),
    "energy_transfer_efficiency": (0.5, 0.95),
    "overcrowding_penalty": (0.1, 1.0),
    "reproduction_cooldown_period": (20, 100),
    "hunt_success_chance": (0.4, 0.95),
    "satiation_period": (5, 60),
    "maturity_age": (15, 80),
    "juvenile_hunt_modifier": (0.2, 0.9),
    "prey_scarcity_threshold": (50, 800),
    "max_lifespan_predator": (200, 800),

    # --- Crab (Scavenger) Parameters ---
    "metabolic_rate_scav": (0.01, 0.5),
    "reproduction_threshold_scav": (20.0, 60.0),
    "eating_rate_scav": (0.1, 0.8),
    "energy_conversion_factor_scav": (2.0, 8.0),
    "max_lifespan_scav": (400, 800),
    # --- HIGHLIGHT: Add crab carrying capacity to the optimizer's control ---
    "carrying_capacity_threshold_scav": (2, 10),
    "starvation_chance_scav": (0.1, 0.5)
}
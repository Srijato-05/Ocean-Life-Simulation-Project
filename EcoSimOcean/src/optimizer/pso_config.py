# src/optimizer/pso_config.py

"""
This file contains all the configuration settings for the Particle Swarm
Optimization (PSO) algorithm. This version is updated to include tunable parameters
for advanced ecological mechanics.
"""

# --- Particle Swarm Optimization (PSO) Configuration ---
PSO_CONFIG = {
    "num_particles": 25,
    "num_iterations": 100,
    "inertia_start": 0.9,
    "inertia_end": 0.4,
    "cognitive_weight": 2.0,
    "social_weight": 2.0
}

# --- PARAMETER BOUNDS (Fully Tunable) ---
PARAM_BOUNDS = {
    # Environmental Parameters
    "plankton_max_growth_rate": (0.7, 1.2),
    "initial_zooplankton_count": (4000, 6000),
    "initial_smallfish_count": (250, 450),
    "initial_crab_count": (80, 200),
    "initial_seal_count": (5, 15),
    "initial_seaturtle_count": (15, 40),
    "refuge_hunt_debuff": (0.1, 0.6),
    "low_plankton_threshold": (0.1, 0.4),
    "event_chance": (0.005, 0.02),
    "plankton_bloom_modifier": (1.5, 3.0),
    "disease_zone_modifier": (1.2, 2.5),

    # --- Zooplankton (Prey) Parameters ---
    "metabolic_rate_prey": (0.15, 0.3),
    "reproduction_threshold_prey": (15.0, 25.0),
    "eating_rate_prey": (0.9, 1.3),
    "energy_conversion_factor_prey": (4.0, 7.0),
    "max_lifespan_prey": (200, 350),
    "flee_distance_prey": (5.0, 15.0),
    "reproduction_fear_debuff_prey": (0.5, 1.0),
    "plankton_satiation_period_prey": (10, 30),
    "carrying_capacity_threshold_prey": (15, 30),
    "starvation_chance_prey": (0.05, 0.2),

    # --- SmallFish (Predator/Omnivore) Parameters ---
    "metabolic_rate_predator": (0.1, 0.2),
    "reproduction_threshold_predator": (30.0, 45.0),
    "reproduction_cooldown_period_predator": (50, 80),
    "vision_radius_predator": (18.0, 28.0),
    "predation_range_predator": (1.5, 4.0),
    "hunt_success_chance_predator": (0.7, 0.95),
    "satiation_period_predator": (30, 80),
    "maturity_age_predator": (30, 60),
    "max_lifespan_predator": (400, 900),
    "max_energy_transfer_efficiency_predator": (0.8, 0.95),
    "optimal_prey_size_predator": (0.8, 1.5),
    "prey_size_tolerance_predator": (1.0, 3.0),
    "juvenile_hunt_modifier_predator": (0.5, 0.9),
    "juvenile_metabolic_modifier_predator": (0.6, 1.0),
    "flee_distance_predator": (15.0, 25.0),
    "reproduction_fear_debuff_predator": (0.6, 1.0),
    "prey_scarcity_threshold_predator": (400, 1200),
    "eating_rate_predator": (0.1, 0.5),
    "energy_conversion_factor_predator": (1.0, 4.0),
    "plankton_satiation_period_predator": (10, 50),

    # --- Crab (Scavenger) Parameters ---
    "metabolic_rate_scav": (0.01, 0.2),
    "reproduction_threshold_scav": (30.0, 70.0),
    "eating_rate_scav": (0.2, 0.9),
    "energy_conversion_factor_scav": (5.0, 12.0),
    "max_lifespan_scav": (500, 900),
    "flee_distance_scav": (2.0, 10.0),
    "reproduction_fear_debuff_scav": (0.4, 1.0),

    # --- Seal (Apex Predator) Parameters ---
    "metabolic_rate_apex": (0.6, 1.2),
    "reproduction_threshold_apex": (300.0, 450.0),
    "reproduction_cooldown_period_apex": (180, 300),
    "vision_radius_apex": (22.0, 35.0),
    "predation_range_apex": (2.0, 5.0),
    "satiation_period_apex": (150, 250),
    "hunt_success_chance_apex": (0.25, 0.55),
    "maturity_age_apex": (100, 180),
    "max_lifespan_apex": (800, 1200),
    "max_energy_transfer_efficiency_apex": (0.8, 0.95),
    "optimal_prey_size_apex": (4.0, 10.0),
    "prey_size_tolerance_apex": (3.0, 8.0),
    "juvenile_hunt_modifier_apex": (0.4, 0.8),
    "juvenile_metabolic_modifier_apex": (0.7, 1.0),

    # --- Sea Turtle (Competitor) Parameters ---
    "metabolic_rate_turtle": (0.15, 0.3),
    "reproduction_threshold_turtle": (110.0, 180.0),
    "eating_rate_turtle": (0.2, 0.5),
    "energy_conversion_factor_turtle": (6.0, 10.0),
    "max_lifespan_turtle": (800, 1400),
    "flee_distance_turtle": (10.0, 25.0),
    "reproduction_fear_debuff_turtle": (0.5, 1.0),
    "plankton_satiation_period_turtle": (20, 50)
}
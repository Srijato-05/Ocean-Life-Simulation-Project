# parameter_sweep.py

import sys
import os
import random
from copy import deepcopy
import json

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '.'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from src.simulation.runner import run_headless_simulation
from src.utils.config_loader import load_fauna_config, load_sim_config
from src.optimizer.pso_config import PSO_CONFIG
from src.optimizer.scoring import fitness
from src.optimizer.logging import log_particle_performance, log_iteration_summary, log_final_results
from src.optimizer.particle import Particle

# --- EXPANDED PARAMETER BOUNDS ---
# The search now includes all key environmental and species parameters.
PARAM_BOUNDS = {
    # Environmental Parameters
    "food_growth_rate": (0.01, 0.1),
    "disease_threshold": (5, 15),
    "disease_chance": (0.05, 0.3),
    
    # --- HIGHLIGHT: Add new prey parameters to the search ---
    "carrying_capacity_threshold": (5, 12),
    "starvation_chance": (0.1, 0.5),

    # Predator (SmallFish) Parameters
    "metabolic_rate": (0.2, 0.6),
    "movement_cost": (0.3, 0.8),
    "vision_radius": (8.0, 15.0),
    "energy_transfer_efficiency": (0.6, 0.85),
    "reproduction_threshold": (50.0, 70.0),
    "overcrowding_penalty": (0.2, 0.6),
    "reproduction_cooldown_period": (30, 50),
    "hunt_success_chance": (0.6, 0.8),
    "satiation_period": (5, 20)
}


def run_pso():
    """
    Runs the final, enhanced Particle Swarm Optimization algorithm.
    """
    print("--- Starting Final Holistic Particle Swarm Optimization ---")
    
    base_fauna_configs = load_fauna_config()
    base_sim_config = load_sim_config()
    if not base_fauna_configs or not base_sim_config: return

    swarm = [Particle(base_sim_config, base_fauna_configs, PARAM_BOUNDS) for _ in range(PSO_CONFIG["num_particles"])]
    
    global_best_score = -1
    global_best_sim_config = None
    global_best_fauna_config = None

    for iteration in range(PSO_CONFIG["num_iterations"]):
        print(f"\n--- Iteration {iteration + 1}/{PSO_CONFIG['num_iterations']} ---")
        
        for i, particle in enumerate(swarm):
            history = run_headless_simulation(particle.sim_config, particle.fauna_config)
            score = fitness(history)
            
            # --- HIGHLIGHT: The params dictionary is now more comprehensive ---
            all_params = {**particle.sim_config, **particle.fauna_config["Zooplankton"], **particle.fauna_config["SmallFish"]}
            log_particle_performance(i, score, history, all_params, PARAM_BOUNDS)

            if score > particle.best_score:
                particle.best_score = score
                particle.best_sim_config = deepcopy(particle.sim_config)
                particle.best_fauna_config = deepcopy(particle.fauna_config)

            if score > global_best_score:
                global_best_score = score
                global_best_sim_config = deepcopy(particle.sim_config)
                global_best_fauna_config = deepcopy(particle.fauna_config)
        
        if global_best_fauna_config and global_best_sim_config:
            log_iteration_summary(iteration + 1, global_best_score, global_best_sim_config, global_best_fauna_config)
        else:
            print("  --- End of Iteration: No valid best score yet. ---")

        pso_config_iter = deepcopy(PSO_CONFIG)
        pso_config_iter["inertia"] = PSO_CONFIG["inertia_start"] - \
                                     (PSO_CONFIG["inertia_start"] - PSO_CONFIG["inertia_end"]) * \
                                     (iteration / PSO_CONFIG["num_iterations"])

        for particle in swarm:
            if global_best_sim_config and global_best_fauna_config:
                particle.update_velocity(global_best_sim_config, global_best_fauna_config, pso_config_iter)
                particle.update_position()

    log_final_results(global_best_score, global_best_sim_config, global_best_fauna_config)

if __name__ == "__main__":
    run_pso()
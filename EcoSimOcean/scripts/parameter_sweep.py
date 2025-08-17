# scripts/parameter_sweep.py

import sys
import os
import random
from copy import deepcopy
import json

# --- Path Correction Logic ---
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from src.simulation.runner import run_headless_simulation
from src.utils.config_loader import load_fauna_config, load_sim_config
from src.optimizer.pso_config import PSO_CONFIG, PARAM_BOUNDS
from src.optimizer.scoring import fitness
from src.optimizer.logging import log_particle_performance, log_iteration_summary, log_final_results
from src.optimizer.particle import Particle


def run_pso():
    """
    Runs the final, enhanced Particle Swarm Optimization algorithm sequentially
    with detailed, colored logging.
    """
    print("--- Starting Final Holistic Particle Swarm Optimization ---")
    
    base_fauna_configs = load_fauna_config()
    base_sim_config = load_sim_config()
    if not base_fauna_configs or not base_sim_config: return

    swarm = [Particle(base_sim_config, base_fauna_configs, PARAM_BOUNDS) for _ in range(PSO_CONFIG["num_particles"])]
    
    global_best_score = -1
    global_best_sim_config = None
    global_best_fauna_config = None
    global_best_history = None

    for iteration in range(PSO_CONFIG["num_iterations"]):
        print(f"\n--- Iteration {iteration + 1}/{PSO_CONFIG['num_iterations']} ---")
        
        # --- HIGHLIGHT: Reverted to a standard sequential loop ---
        for i, particle in enumerate(swarm):
            history = run_headless_simulation(particle.sim_config, particle.fauna_config)
            score = fitness(history)
            
            # --- HIGHLIGHT: Re-enabled the detailed, per-particle logger ---
            all_params = {
                **particle.sim_config, 
                **particle.fauna_config.get("Zooplankton", {}), 
                **particle.fauna_config.get("SmallFish", {}), 
                **particle.fauna_config.get("Crab", {})
            }
            log_particle_performance(i, score, history, all_params, PARAM_BOUNDS)

            if score > particle.best_score:
                particle.best_score = score
                particle.best_sim_config = deepcopy(particle.sim_config)
                particle.best_fauna_config = deepcopy(particle.fauna_config)

            if score > global_best_score:
                global_best_score = score
                global_best_sim_config = deepcopy(particle.sim_config)
                global_best_fauna_config = deepcopy(particle.fauna_config)
                global_best_history = history
        
        if global_best_fauna_config and global_best_sim_config:
            log_iteration_summary(iteration + 1, global_best_score, global_best_sim_config, global_best_fauna_config, global_best_history)
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
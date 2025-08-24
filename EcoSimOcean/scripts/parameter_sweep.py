# scripts/parameter_sweep.py

import sys
import os
import random
import json
import numpy as np
import multiprocessing
from copy import deepcopy
from datetime import datetime

# Path Correction Logic
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from src.simulation.runner import run_headless_simulation
from src.utils.config_loader import load_fauna_config, load_sim_config
from src.optimizer.pso_config import PSO_CONFIG, PARAM_BOUNDS
from src.optimizer.scoring import fitness
from src.optimizer.logging import (print_particle_performance, print_iteration_summary,
                                   print_final_results, print_message,
                                   create_particle_log, create_summary_log,
                                   create_final_log)
from src.optimizer.particle import Particle

CHECKPOINT_FILE = "pso_checkpoint.json"

def run_particle_simulation(particle_tuple):
    """A top-level function to run a simulation for a single particle's state."""
    particle_index, particle_state = particle_tuple
    
    base_sim = load_sim_config()
    base_fauna = load_fauna_config()
    temp_particle = Particle(base_sim, base_fauna, PARAM_BOUNDS)
    temp_particle.sim_config = particle_state['sim_config']
    temp_particle.fauna_config = particle_state['fauna_config']
    
    history = run_headless_simulation(temp_particle.sim_config, temp_particle.fauna_config)
    score = fitness(history, temp_particle.sim_config)
    
    return particle_index, score, history, particle_state

def convert_to_json_serializable(obj):
    """Recursively converts numpy types to native Python types in a dictionary."""
    if isinstance(obj, dict):
        return {k: convert_to_json_serializable(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_to_json_serializable(i) for i in obj]
    elif isinstance(obj, (np.int64, np.int32)):
        return int(obj)
    elif isinstance(obj, (np.float64, np.float32)):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    return obj

def save_checkpoint(state):
    """Saves the current state of the PSO to a JSON file."""
    try:
        serializable_state = convert_to_json_serializable(state)
        with open(CHECKPOINT_FILE, 'w') as f:
            json.dump(serializable_state, f, indent=4)
    except Exception as e:
        print(f"\nWARNING: Could not save checkpoint file: {e}")

def load_checkpoint():
    """Loads the PSO state from a JSON file if it exists."""
    if os.path.exists(CHECKPOINT_FILE):
        try:
            with open(CHECKPOINT_FILE, 'r') as f:
                return json.load(f)
        except Exception as e:
            print(f"\nWARNING: Could not load checkpoint file, starting new run. Error: {e}")
    return None

def run_pso():
    """
    Runs the PSO algorithm with checkpointing and resume functionality.
    """
    start_time = datetime.now()
    output_dir = 'results'
    os.makedirs(output_dir, exist_ok=True)
    log_filename = os.path.join(output_dir, f"pso_log_{start_time:%Y%m%d_%H%M%S}.json")

    total_cores = multiprocessing.cpu_count()
    num_processes = max(1, total_cores - 4)

    print("--- Starting Final Holistic Particle Swarm Optimization ---")
    
    checkpoint = load_checkpoint()
    
    if checkpoint:
        print(f"✅ Resuming from checkpoint at iteration {checkpoint['start_iteration']}")
        start_iteration = checkpoint['start_iteration']
        # ... (resume logic)
    else:
        print("🚀 Starting a new optimization run.")
        start_iteration = 0
        base_fauna_configs = load_fauna_config()
        base_sim_config = load_sim_config()
        if not base_fauna_configs or not base_sim_config: return
        
        # --- FIX: Swapped base_sim_config and base_fauna_configs to correct order ---
        swarm = [Particle(base_sim_config, base_fauna_configs, PARAM_BOUNDS) for _ in range(PSO_CONFIG["num_particles"])]
        
        global_best_score = -1
        global_best_sim_config = None
        global_best_fauna_config = None
        global_best_history = None
        full_log = { "run_metadata": { "start_time": start_time.isoformat(), "log_file": log_filename, "pso_config": PSO_CONFIG }, "iterations": [] }

    print(f"Using {num_processes} processes. Logging to: {log_filename}")

    with multiprocessing.Pool(processes=num_processes) as pool:
        for iteration in range(start_iteration, PSO_CONFIG["num_iterations"]):
            print_message(f"--- Iteration {iteration + 1}/{PSO_CONFIG['num_iterations']} ---")
            iteration_log = { "iteration": iteration + 1, "particle_performances": [] }

            particle_jobs = [(i, p.get_state()) for i, p in enumerate(swarm)]
            results = pool.map(run_particle_simulation, particle_jobs)

            for i, score, history, particle_state in results:
                print_particle_performance(i, score, history)
                
                all_params = {**particle_state['sim_config']}
                for species, conf in particle_state['fauna_config'].items():
                    for key, val in conf.items():
                        all_params[f"{key}_{species}"] = val
                
                particle_log_entry = create_particle_log(i, score, history, all_params, PARAM_BOUNDS)
                iteration_log["particle_performances"].append(particle_log_entry)

                if score > swarm[i].best_score:
                    swarm[i].best_score = score
                    swarm[i].best_sim_config = deepcopy(particle_state['sim_config'])
                    swarm[i].best_fauna_config = deepcopy(particle_state['fauna_config'])

                if score > global_best_score:
                    global_best_score = score
                    global_best_sim_config = deepcopy(particle_state['sim_config'])
                    global_best_fauna_config = deepcopy(particle_state['fauna_config'])
                    global_best_history = history
            
            summary_log_entry = create_summary_log(iteration + 1, global_best_score, global_best_sim_config, global_best_fauna_config, global_best_history)
            iteration_log["iteration_summary"] = summary_log_entry
            full_log["iterations"].append(iteration_log)
            
            print_iteration_summary(iteration + 1, global_best_score, global_best_history)

            pso_config_iter = deepcopy(PSO_CONFIG)
            pso_config_iter["inertia"] = PSO_CONFIG["inertia_start"] - (PSO_CONFIG["inertia_start"] - PSO_CONFIG["inertia_end"]) * (iteration / PSO_CONFIG["num_iterations"])

            for particle in swarm:
                if global_best_sim_config and global_best_fauna_config:
                    particle.update_velocity(global_best_sim_config, global_best_fauna_config, pso_config_iter)
                    particle.update_position()
            
            checkpoint_state = {
                'start_iteration': iteration + 1,
                'swarm_state': [p.get_state() for p in swarm],
                'global_best_score': global_best_score,
                'global_best_sim_config': global_best_sim_config,
                'global_best_fauna_config': global_best_fauna_config,
                'global_best_history': global_best_history,
                'full_log': full_log
            }
            save_checkpoint(checkpoint_state)

    final_log_entry = create_final_log(global_best_score, global_best_sim_config, global_best_fauna_config, global_best_history)
    full_log["final_result"] = final_log_entry
    print_final_results(global_best_score, global_best_history)

    try:
        with open(log_filename, 'w', encoding='utf-8') as f:
            json.dump(full_log, f, indent=4)
        print(f"\n✅ Successfully saved structured log to {log_filename}")
        if os.path.exists(CHECKPOINT_FILE):
            os.remove(CHECKPOINT_FILE)
    except Exception as e:
        print(f"\n❌ Error saving JSON log: {e}")


if __name__ == '__main__':
    # Add a method to the Particle class to get its JSON-serializable state
    def get_particle_state(self):
        return {
            'sim_config': self.sim_config,
            'fauna_config': self.fauna_config,
            'velocity': self.velocity,
            'best_score': self.best_score,
            'best_sim_config': self.best_sim_config,
            'best_fauna_config': self.best_fauna_config
        }
    Particle.get_state = get_particle_state
    
    run_pso()
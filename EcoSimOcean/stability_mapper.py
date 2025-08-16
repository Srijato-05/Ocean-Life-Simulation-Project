# stability_mapper.py

"""
This script performs a systematic parameter sweep to map the stability of the
ecosystem across a range of key parameters. It generates a heatmap to visualize
the "islands of stability."
"""

import sys
import os
import numpy as np
import matplotlib.pyplot as plt
from copy import deepcopy
import json

# Add the project root to the Python path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '.'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from src.simulation.runner import run_headless_simulation
from src.utils.config_loader import load_fauna_config, load_sim_config
from src.optimizer.scoring import fitness

# --- Stability Map Configuration ---
# HIGHLIGHT: Example updated to test a prey parameter vs. a predator parameter.
# You can change these to any two parameters from pso_config.py.
MAP_CONFIG = {
    "x_param": "reproduction_threshold_prey",
    "x_range": np.linspace(20, 50, 10), # Test 10 values for prey reproduction cost

    "y_param": "satiation_period",
    "y_range": np.linspace(10, 30, 10)  # Test 10 values for predator satiation
}


# --- HIGHLIGHT: New helper function to set parameters robustly ---
def set_param(sim_config, fauna_configs, param_key, value):
    """
    Finds and sets a parameter in the correct config dictionary, whether it's
    for the environment, prey, or predator.
    """
    key = param_key.replace('_prey', '').replace('_predator', '')
    
    if key in sim_config:
        sim_config[key] = value
    elif param_key.endswith('_prey') or key in fauna_configs["Zooplankton"]:
        fauna_configs["Zooplankton"][key] = value
    elif param_key.endswith('_predator') or key in fauna_configs["SmallFish"]:
        fauna_configs["SmallFish"][key] = value
    else:
        print(f"Warning: Parameter '{param_key}' not found in any config.")


def run_stability_map():
    """
    Runs the full simulation for every combination of parameters defined
    in the MAP_CONFIG and stores the fitness score for each run.
    """
    print("--- Starting Ecosystem Stability Mapping ---")
    
    base_fauna_configs = load_fauna_config()
    base_sim_config = load_sim_config()
    if not base_fauna_configs or not base_sim_config:
        print("Error: Could not load base configuration files. Exiting.")
        return

    x_param = MAP_CONFIG["x_param"]
    x_range = MAP_CONFIG["x_range"]
    y_param = MAP_CONFIG["y_param"]
    y_range = MAP_CONFIG["y_range"]
    
    results_grid = np.zeros((len(y_range), len(x_range)))
    total_runs = len(x_range) * len(y_range)
    current_run = 0

    for i, y_val in enumerate(y_range):
        for j, x_val in enumerate(x_range):
            current_run += 1
            print(f"Running simulation {current_run}/{total_runs} ( {y_param}={y_val:.2f}, {x_param}={x_val:.2f} )")
            
            test_sim_config = deepcopy(base_sim_config)
            test_fauna_configs = deepcopy(base_fauna_configs)
            
            # --- HIGHLIGHT: Simplified logic using the new helper function ---
            set_param(test_sim_config, test_fauna_configs, x_param, x_val)
            set_param(test_sim_config, test_fauna_configs, y_param, y_val)

            history = run_headless_simulation(test_sim_config, test_fauna_configs)
            
            score = fitness(history)
            results_grid[i, j] = score

    return results_grid

def plot_stability_map(results_grid):
    """
    Generates and saves a heatmap visualizing the stability of the ecosystem.
    """
    print("\n--- Generating Stability Map ---")
    
    plt.figure(figsize=(12, 10))
    plt.imshow(results_grid, cmap='viridis', origin='lower', aspect='auto', 
               norm=plt.matplotlib.colors.LogNorm(vmin=1, vmax=np.max(results_grid)))
    
    plt.colorbar(label='Ecosystem Fitness Score (Log Scale)')
    
    plt.xlabel(MAP_CONFIG["x_param"])
    plt.ylabel(MAP_CONFIG["y_param"])
    
    plt.xticks(ticks=np.arange(len(MAP_CONFIG["x_range"])), 
               labels=[f'{val:.2f}' for val in MAP_CONFIG["x_range"]], rotation=45)
    plt.yticks(ticks=np.arange(len(MAP_CONFIG["y_range"])), 
               labels=[f'{val:.2f}' for val in MAP_CONFIG["y_range"]])
    
    plt.title('Ecosystem Stability Map')
    plt.tight_layout()
    
    output_filename = "stability_map.png"
    plt.savefig(output_filename)
    print(f"Stability map saved as {output_filename}")
    plt.show()

if __name__ == "__main__":
    results = run_stability_map()
    if results is not None:
        plot_stability_map(results)
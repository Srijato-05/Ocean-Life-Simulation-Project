# scripts/stability_mapper.py

"""
This script performs a systematic, sequential parameter sweep and generates two
distinct, 4-panel dashboards (Core Results and Advanced Analytics) for each
experiment defined.
"""

import sys
import os
import numpy as np
import matplotlib.pyplot as plt
from copy import deepcopy
import json
from matplotlib.colors import ListedColormap, BoundaryNorm

# --- Path Correction Logic ---
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from src.simulation.runner import run_headless_simulation
from src.utils.config_loader import load_fauna_config, load_sim_config
from src.optimizer.scoring import fitness

# --- List to define multiple map experiments ---
MAP_CONFIGS = [
    {
        "basename": "pred_vs_scav_repro",
        "title": "Predator Reproduction vs. Scavenger Reproduction",
        "x_param": "reproduction_threshold_predator",
        "x_range": np.linspace(40, 80, 15),
        "y_param": "reproduction_threshold_scav",
        "y_range": np.linspace(20, 40, 15)
    },
    {
        "basename": "plankton_vs_prey_lifespan",
        "title": "Plankton Growth vs. Prey Lifespan",
        "x_param": "plankton_max_growth_rate",
        "x_range": np.linspace(0.1, 0.5, 15),
        "y_param": "max_lifespan_prey",
        "y_range": np.linspace(150, 300, 15)
    },
    # --- NEW: Configuration for Apex vs. Predator Analysis ---
    {
        "basename": "apex_vs_pred_repro",
        "title": "Apex (Seal) Reproduction vs. Predator (Fish) Reproduction",
        "x_param": "reproduction_threshold_apex",
        "x_range": np.linspace(100, 200, 10),
        "y_param": "reproduction_threshold_predator",
        "y_range": np.linspace(25, 45, 10)
    },
    # --- NEW: Configuration for Herbivore Competition Analysis ---
    {
        "basename": "herbivore_competition_eating_rate",
        "title": "Herbivore Competition: Turtle Eating Rate vs. Zooplankton Eating Rate",
        "x_param": "eating_rate_turtle",
        "x_range": np.linspace(0.3, 0.8, 10),
        "y_param": "eating_rate_prey",
        "y_range": np.linspace(0.8, 1.2, 10)
    }
]

def set_param(sim_config, fauna_configs, param_key, value):
    """Dynamically sets a parameter in the appropriate config dictionary."""
    key_map = {
        '_prey': "Zooplankton",
        '_predator': "SmallFish",
        '_scav': "Crab",
        '_apex': "Seal",
        '_turtle': "SeaTurtle"
    }

    for suffix, species_name in key_map.items():
        if param_key.endswith(suffix):
            key_without_suffix = param_key.replace(suffix, '')
            if species_name in fauna_configs and key_without_suffix in fauna_configs[species_name]:
                fauna_configs[species_name][key_without_suffix] = value
                return

    if param_key in sim_config:
        sim_config[param_key] = value
        return
    
    print(f"Warning: Parameter '{param_key}' not found in any config.")


def run_stability_map(map_config):
    print(f"\n--- Starting Analysis for: {map_config['title']} (Linear Mode) ---")
    base_fauna_configs = load_fauna_config()
    base_sim_config = load_sim_config()
    if not base_fauna_configs or not base_sim_config: return None

    x_param, x_range = map_config["x_param"], map_config["x_range"]
    y_param, y_range = map_config["y_param"], map_config["y_range"]
    
    grid_shape = (len(y_range), len(x_range))
    results = { "fitness": np.zeros(grid_shape), "prey_pop": np.zeros(grid_shape), 
                "pred_pop": np.zeros(grid_shape), "scav_pop": np.zeros(grid_shape),
                "apex_pop": np.zeros(grid_shape), "turtle_pop": np.zeros(grid_shape),
                "time_to_collapse": np.full(grid_shape, np.nan) }
    total_runs = len(x_range) * len(y_range)
    current_run = 0

    for i, y_val in enumerate(y_range):
        for j, x_val in enumerate(x_range):
            current_run += 1
            print(f"  Running simulation {current_run}/{total_runs}...")
            test_sim_config = deepcopy(base_sim_config)
            test_fauna_configs = deepcopy(base_fauna_configs)
            set_param(test_sim_config, test_fauna_configs, x_param, x_val)
            set_param(test_sim_config, test_fauna_configs, y_param, y_val)
            history = run_headless_simulation(test_sim_config, test_fauna_configs)
            score = fitness(history)
            results["fitness"][i, j] = score
            if score < 100000:
                results["time_to_collapse"][i, j] = score
            if history:
                final_state = history[-1]
                results["prey_pop"][i, j] = final_state.get('zooplankton', 0)
                results["pred_pop"][i, j] = final_state.get('smallfish', 0)
                results["scav_pop"][i, j] = final_state.get('crab', 0)
                results["apex_pop"][i, j] = final_state.get('seal', 0)
                results["turtle_pop"][i, j] = final_state.get('seaturtle', 0)

    return results

def create_dominance_grid(prey_grid, pred_grid, scav_grid, apex_grid, turtle_grid):
    dominance_grid = np.zeros_like(prey_grid)
    collapse_mask = (pred_grid < 10) | (scav_grid < 10) | (apex_grid < 10) | (turtle_grid < 10)
    prey_dom_mask = (prey_grid > pred_grid * 10) & ~collapse_mask
    pred_dom_mask = (pred_grid > prey_grid) & ~collapse_mask
    
    dominance_grid[prey_dom_mask] = 1
    dominance_grid[pred_dom_mask] = 2
    dominance_grid[~(collapse_mask | prey_dom_mask | pred_dom_mask)] = 3
    return dominance_grid


def plot_core_dashboard(results, map_config):
    """Generates a 2x2 dashboard of the most important result heatmaps."""
    filename = f"core_dashboard_{map_config['basename']}.png"
    print(f"\n--- Generating Core Dashboard: {filename} ---")
    
    fig, axes = plt.subplots(2, 2, figsize=(18, 15))
    fig.suptitle(f"Core Results Dashboard: {map_config['title']}", fontsize=16)

    def format_heatmap_axes(ax, title):
        x_labels = [f'{val:.2f}' for val in map_config["x_range"]]
        y_labels = [f'{val:.2f}' for val in map_config["y_range"]]
        ax.set_xticks(np.arange(len(x_labels)))
        ax.set_xticklabels(x_labels, rotation=45, ha="right")
        ax.set_yticks(np.arange(len(y_labels)))
        ax.set_yticklabels(y_labels)
        ax.set_title(title)
        ax.set_xlabel(map_config["x_param"])
        ax.set_ylabel(map_config["y_param"])

    # 1. Overall Fitness Score Heatmap
    ax = axes[0, 0]
    fitness_data = results['fitness']
    positive_fitness = fitness_data[fitness_data > 0]
    
    norm = None
    if positive_fitness.size > 0:
        vmin = positive_fitness.min()
        vmax = fitness_data.max()
        if vmin < vmax:
            norm = plt.matplotlib.colors.LogNorm(vmin=vmin, vmax=vmax)

    im = ax.imshow(fitness_data, cmap='viridis', origin='lower', aspect='auto', norm=norm)
    try:
        fig.colorbar(im, ax=ax, label='Fitness Score (Log Scale)')
    except Exception as e:
        print(f"Could not create colorbar for fitness scores: {e}") 
    format_heatmap_axes(ax, 'Overall Fitness Score')

    # 2. Ecosystem Outcome Heatmap
    dominance_grid = create_dominance_grid(results['prey_pop'], results['pred_pop'], results['scav_pop'], results['apex_pop'], results['turtle_pop'])
    cmap = ListedColormap(['#440154', '#21908d', '#fde725', '#31688e'])
    bounds = [-0.5, 0.5, 1.5, 2.5, 3.5]
    norm = BoundaryNorm(bounds, cmap.N)
    im = axes[0, 1].imshow(dominance_grid, cmap=cmap, norm=norm, origin='lower', aspect='auto')
    cbar = fig.colorbar(im, ax=axes[0, 1], ticks=[0, 1, 2, 3])
    cbar.set_ticklabels(['Collapse', 'Prey Dom.', 'Pred Dom.', 'Balanced'])
    format_heatmap_axes(axes[0, 1], 'Ecosystem Outcome')

    # 3. Prey Population Heatmap
    im = axes[1, 0].imshow(results['prey_pop'], cmap='Blues', origin='lower', aspect='auto')
    fig.colorbar(im, ax=axes[1, 0], label='Final Prey Population (Zooplankton)')
    format_heatmap_axes(axes[1, 0], 'Prey (Zooplankton) Population')

    # 4. Apex Predator Population Heatmap
    im = axes[1, 1].imshow(results['apex_pop'], cmap='Oranges', origin='lower', aspect='auto')
    fig.colorbar(im, ax=axes[1, 1], label='Final Apex Population (Seal)')
    format_heatmap_axes(axes[1, 1], 'Apex (Seal) Population')
    
    plt.tight_layout(rect=[0, 0.03, 1, 0.96])
    output_dir = 'results'
    os.makedirs(output_dir, exist_ok=True)
    output_filename = os.path.join(output_dir, filename)
    plt.savefig(output_filename)
    print(f"Dashboard saved as {output_filename}")
    plt.close()

def plot_analytics_dashboard(results, map_config):
    """Generates a 2x2 advanced dashboard with different graph types."""
    filename = f"analytics_dashboard_{map_config['basename']}.png"
    print(f"\n--- Generating Analytics Dashboard: {filename} ---")
    
    fig, axes = plt.subplots(2, 2, figsize=(20, 18))
    fig.suptitle(f"Ecosystem Analytics Dashboard: {map_config['title']}", fontsize=20)
    x_range, y_range = map_config["x_range"], map_config["y_range"]
    X, Y = np.meshgrid(x_range, y_range)

    # 1. Population Trends Line Plot
    ax = axes[0, 0]
    ax2 = ax.twinx()
    mean_prey = np.mean(results['prey_pop'], axis=0)
    mean_pred = np.mean(results['pred_pop'], axis=0)
    mean_apex = np.mean(results['apex_pop'], axis=0)
    p1, = ax.plot(x_range, mean_prey, 'b-', label='Avg Prey (Zoo)')
    p2, = ax2.plot(x_range, mean_pred, 'r-', label='Avg Pred (Fish)')
    p3, = ax2.plot(x_range, mean_apex, 'g--', label='Avg Apex (Seal)')
    ax.set_ylabel('Avg Prey Population', color='b')
    ax2.set_ylabel('Avg Pred/Apex Population', color='r')
    ax.set_title('Population Trends vs. X-Param')
    ax.set_xlabel(map_config["x_param"])
    ax.legend(handles=[p1, p2, p3])
    ax.grid(True, linestyle=':')

    # 2. Parameter Relationship Scatter Plot
    ax = axes[0, 1]
    sizes = np.clip(results['apex_pop'].flatten(), 5, 500)
    fitness_data = results['fitness'].flatten()
    positive_fitness = fitness_data[fitness_data > 0]
    norm = None
    if positive_fitness.size > 0:
        vmin = positive_fitness.min()
        vmax = fitness_data.max()
        if vmin < vmax:
            norm = plt.matplotlib.colors.LogNorm(vmin=vmin, vmax=vmax)

    scatter = ax.scatter(X.flatten(), Y.flatten(), c=fitness_data, s=sizes, cmap='viridis', norm=norm, alpha=0.7)
    try:
        fig.colorbar(scatter, ax=ax, label='Fitness Score (Log Scale)')
    except Exception as e:
        print(f"Could not create colorbar for scatter plot: {e}")

    ax.set_title('Fitness (Color) & Apex Pop (Size)')
    ax.set_xlabel(map_config["x_param"])
    ax.set_ylabel(map_config["y_param"])
    ax.grid(True, linestyle=':')
    
    # 3. Time to Collapse Heatmap
    ax = axes[1, 0]
    cmap = plt.get_cmap('plasma'); cmap.set_bad(color='whitesmoke')
    im = ax.imshow(results['time_to_collapse'], cmap=cmap, origin='lower', aspect='auto')
    fig.colorbar(im, ax=ax, label='Ticks until Collapse')
    ax.set_title('Time to Collapse (Gray = Stable)')
    ax.set_xlabel(map_config["x_param"])
    ax.set_ylabel(map_config["y_param"])

    # 4. Ecosystem Outcome Pie Chart
    ax = axes[1, 1]
    dominance_grid = create_dominance_grid(results['prey_pop'], results['pred_pop'], results['scav_pop'], results['apex_pop'], results['turtle_pop'])
    outcomes, counts = np.unique(dominance_grid, return_counts=True)
    labels = {0: 'Collapse', 1: 'Prey Dom.', 2: 'Pred Dom.', 3: 'Balanced'}
    colors = {0: '#440154', 1: '#21908d', 2: '#fde725', 3: '#31688e'}
    pie_labels = [labels.get(oc, 'Unknown') for oc in outcomes]
    pie_colors = [colors.get(oc, 'gray') for oc in outcomes]
    if len(counts) > 0:
        ax.pie(counts, labels=pie_labels, autopct='%1.1f%%', startangle=90, colors=pie_colors, wedgeprops={'edgecolor': 'white'})
    ax.set_title('Distribution of Outcomes')
    ax.axis('equal')
        
    plt.tight_layout(rect=[0, 0.03, 1, 0.96])
    
    output_dir = 'results'
    os.makedirs(output_dir, exist_ok=True)
    output_filename = os.path.join(output_dir, filename)
    plt.savefig(output_filename)
    print(f"Dashboard saved as {output_filename}")
    plt.close()

def main():
    """Main execution block."""
    for config in MAP_CONFIGS:
        res = run_stability_map(config)
        if res is not None:
            plot_core_dashboard(res, config)
            plot_analytics_dashboard(res, config)

if __name__ == "__main__":
    main()
# src/simulation/runner.py

import random
import math
import sys
import os

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from src.environment import Environment
from src.agents.zooplankton import Zooplankton
from src.agents.small_fish import SmallFish
# --- HIGHLIGHT: Import the new Crab agent ---
from src.agents.crab import Crab
from src.agents.behaviors import RandomWalk, ChasePrey
from src.utils.config_loader import load_fauna_config, load_sim_config
from src.simulation.simulation_manager import SimulationManager

def setup_simulation(sim_config, fauna_configs):
    """
    Initializes the environment and creates the initial list of agent objects.
    """
    agents = []
    
    # --- HIGHLIGHT: Safely get configs and handle missing species ---
    env = Environment(
        sim_config["grid_width"], 
        sim_config["grid_height"], 
        sim_config["grid_depth"],
        sim_config
    )
    
    if "Zooplankton" in fauna_configs:
        for _ in range(int(sim_config.get("initial_zooplankton_count", 0))):
            agents.append(Zooplankton(env, fauna_configs["Zooplankton"].copy(), RandomWalk()))
    
    if "SmallFish" in fauna_configs:
        for _ in range(int(sim_config.get("initial_small_fish_count", 0))):
            agents.append(SmallFish(env, fauna_configs["SmallFish"].copy(), ChasePrey()))

    if "Crab" in fauna_configs:
        for _ in range(int(sim_config.get("initial_crab_count", 0))):
            agents.append(Crab(env, fauna_configs["Crab"].copy()))

    return env, agents

def run_simulation(sim_config, fauna_configs, verbose=True):
    """
    The main simulation loop, driven by the highly optimized SimulationManager.
    """
    env, initial_agents = setup_simulation(sim_config, fauna_configs)
    sim_manager = SimulationManager(env, initial_agents, fauna_configs)
    
    if verbose:
        print(f"Environment and Simulation Manager created. Spawned {len(initial_agents)} agents.")
        print("------------------------------------")

    for tick in range(sim_config["simulation_ticks"]):
        env.update()
        sim_manager.update()
        sim_manager.cleanup()
        
        if verbose and (tick + 1) % 10 == 0:
            # --- HIGHLIGHT: Unpack all three population counts ---
            zoo_pop, fish_pop, crab_pop = sim_manager.get_population_counts()
            print(f"Tick: {tick + 1:3} | Zooplankton: {zoo_pop:5} | Small Fish: {fish_pop:4} | Crabs: {crab_pop:4} | "
                  f"Plankton: {env.plankton.sum():8.2f} | Snow: {env.marine_snow.sum():8.2f}")

    if verbose:
        print("--- Simulation Finished ---")
        
    return sim_manager.get_population_counts()

def run_headless_simulation(sim_config, fauna_configs):
    """
    A wrapper for the parameter sweep that runs without console output and
    returns the full history for analysis.
    """
    history = []
    env, initial_agents = setup_simulation(sim_config, fauna_configs)
    sim_manager = SimulationManager(env, initial_agents, fauna_configs)

    for tick in range(sim_config["simulation_ticks"]):
        env.update()
        sim_manager.update()
        sim_manager.cleanup()
        
        # --- HIGHLIGHT: Unpack three values and update history object ---
        zoo_pop, fish_pop, crab_pop = sim_manager.get_population_counts()
        history.append({
            "tick": tick + 1,
            "zooplankton": zoo_pop,
            "small_fish": fish_pop,
            "crab": crab_pop
        })
        # --- HIGHLIGHT: Stop early if any species has collapsed ---
        if zoo_pop == 0 or fish_pop == 0 or crab_pop == 0:
            # Add a final entry with zeroed populations to ensure crash is logged
            history.append({
                "tick": tick + 2,
                "zooplankton": 0,
                "small_fish": 0,
                "crab": 0
            })
            break
            
    return history
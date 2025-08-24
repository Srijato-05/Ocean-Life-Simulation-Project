# src/simulation/runner.py

import random
import math
import sys
import os

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from src.environment import Environment
from src.utils.config_loader import load_fauna_config, load_sim_config
from src.simulation.simulation_manager import SimulationManager
from src.agents.zooplankton import Zooplankton
from src.agents.small_fish import SmallFish
from src.agents.crab import Crab
from src.agents.seal import Seal
from src.agents.sea_turtle import SeaTurtle

def setup_simulation(sim_config, fauna_configs):
    """Initializes the environment and agent objects."""
    env = Environment(
        sim_config["grid_width"], 
        sim_config["grid_height"], 
        sim_config["grid_depth"],
        sim_config
    )
    agents = []
    agent_map = {"Zooplankton": Zooplankton, "SmallFish": SmallFish, "Crab": Crab, "Seal": Seal, "SeaTurtle": SeaTurtle}
    
    for species_name, agent_class in agent_map.items():
        if species_name in fauna_configs:
            count_key = f"initial_{species_name.lower()}_count"
            count = int(sim_config.get(count_key, 0))
            for _ in range(count):
                agents.append(agent_class(env, fauna_configs[species_name].copy()))
    return env, agents

def run_simulation(sim_config, fauna_configs, verbose=True):
    """Main simulation loop for visualization and standard runs."""
    env, initial_agents = setup_simulation(sim_config, fauna_configs)
    sim_manager = SimulationManager(env, initial_agents, fauna_configs)
    
    if verbose:
        print(f"Environment and Simulation Manager created. Spawned {len(initial_agents)} agents.")
        print("------------------------------------")

    for tick in range(sim_config["simulation_ticks"]):
        env.update()
        sim_manager.update()
        
        if verbose and (tick + 1) % 10 == 0:
            counts = sim_manager.get_population_counts()
            print(f"Tick: {tick + 1:3} | Zoo: {counts.get('zooplankton', 0):4} | "
                  f"Fish: {counts.get('smallfish', 0):3} | Crab: {counts.get('crab', 0):3} | "
                  f"Seal: {counts.get('seal', 0):3} | Turtle: {counts.get('seaturtle', 0):3}")
            
            # Stop if the core ecosystem has clearly collapsed
            if tick > sim_manager.bootstrap_period and counts.get('smallfish', 0) == 0 and counts.get('zooplankton', 0) == 0:
                print("Core prey-predator web collapsed. Ending simulation early.")
                break

    if verbose:
        print("--- Simulation Finished ---")
    return sim_manager.get_population_counts()

def run_headless_simulation(sim_config, fauna_configs):
    """
    A wrapper for the optimizer that runs without console output and
    returns the full history, with a more nuanced early-exit condition.
    """
    history = []
    env, initial_agents = setup_simulation(sim_config, fauna_configs)
    sim_manager = SimulationManager(env, initial_agents, fauna_configs)

    for tick in range(sim_config["simulation_ticks"]):
        env.update()
        sim_manager.update()
        
        counts = sim_manager.get_population_counts()
        history.append({"tick": tick + 1, **counts})

        # --- LOGIC FIX: Exit only if the core prey-predator web collapses ---
        is_post_bootstrap = tick > sim_manager.bootstrap_period
        prey_extinct = counts.get("zooplankton", 0) == 0
        predator_extinct = counts.get("smallfish", 0) == 0

        if is_post_bootstrap and prey_extinct and predator_extinct:
            break
            
    return history
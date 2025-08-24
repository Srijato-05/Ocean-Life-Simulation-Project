# phase2_simulation.py

import random
import time
import sys
import os

# Add the project root to the Python path to allow imports from src
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '.'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from src.environment import Environment
from src.agents.zooplankton import Zooplankton

# --- Simulation Configuration ---
SIM_CONFIG = {
    "grid_width": 30,
    "grid_height": 30,
    "grid_depth": 10,
    "initial_food_density": 0.8, # How much of the grid starts with food
    "simulation_ticks": 100,
    "initial_zooplankton_count": 25,
}

# --- Species Configuration for Zooplankton ---
# This dictionary defines all the specific parameters for our agent.
# Later, this can be loaded from a JSON file.
ZOOPLANKTON_CONFIG = {
    "species_name": "Zooplankton",
    "initial_energy": 10.0,
    "metabolic_rate": 0.1,      # Energy lost per tick just by existing
    "movement_cost": 0.2,       # Energy lost per move
    "size": 1.0,                # Amount of mass contributed to marine snow on death
    "eating_rate": 0.5,         # Max food consumed per tick
    "energy_conversion_factor": 5.0, # Energy gained per unit of food
    "reproduction_threshold": 15.0,  # Energy level required to reproduce
}

def setup_simulation():
    """
    Initializes the environment and populates it with agents and resources.
    """
    print("--- Setting up Phase 2 Simulation ---")
    # 1. Initialize the Environment
    env = Environment(
        width=SIM_CONFIG["grid_width"],
        height=SIM_CONFIG["grid_height"],
        depth=SIM_CONFIG["grid_depth"]
    )

    # 2. Populate the environment with initial food
    for _ in range(int(env.width * env.height * env.depth * SIM_CONFIG["initial_food_density"])):
        x = random.randint(0, env.width - 1)
        y = random.randint(0, env.height - 1)
        z = random.randint(0, env.depth - 1)
        env.add_food(x, y, z, random.uniform(1.0, 3.0))

    # 3. Create the initial population of Zooplankton
    agents = []
    for _ in range(SIM_CONFIG["initial_zooplankton_count"]):
        agent = Zooplankton(env, ZOOPLANKTON_CONFIG)
        agents.append(agent)
    
    print(f"Environment created with size ({env.width}, {env.height}, {env.depth}).")
    print(f"Spawned {len(agents)} Zooplankton agents.")
    print("------------------------------------")
    
    return env, agents

def run_simulation():
    """
    The main entry point for the Phase 2 simulation.
    """
    env, agents = setup_simulation()
    
    for tick in range(SIM_CONFIG["simulation_ticks"]):
        # --- Main Simulation Loop ---
        
        # 1. Update the environment (e.g., marine snow sinking)
        env.update()

        newly_born_agents = []
        
        # 2. Update each agent
        for agent in agents:
            agent.update()
            
            # Check for reproduction
            if agent.energy > agent.config.get("reproduction_threshold", 15.0):
                # Halve parent's energy
                agent.energy /= 2
                # Create offspring at the same location with the same config
                offspring = Zooplankton(env, agent.config, initial_position=(agent.x, agent.y, agent.z))
                offspring.energy = agent.energy # Offspring gets half the energy
                newly_born_agents.append(offspring)

        # 3. Clean up the agent list
        # Remove dead agents
        surviving_agents = [agent for agent in agents if agent.alive]
        
        # Add the newly born agents
        agents = surviving_agents + newly_born_agents

        # 4. Log the state of the simulation for this tick
        if (tick + 1) % 10 == 0:
            total_energy = sum(agent.energy for agent in agents)
            avg_energy = total_energy / len(agents) if agents else 0
            print(f"Tick: {tick + 1:3} | Population: {len(agents):3} | "
                  f"Avg. Energy: {avg_energy:6.2f} | "
                  f"Total Food: {env.food.sum():7.2f} | "
                  f"Total Snow: {env.marine_snow.sum():7.2f}")

    print("--- Simulation Finished ---")
    print(f"Final Population: {len(agents)}")
    print("---------------------------")


# --- Run the simulation ---
if __name__ == "__main__":
    run_simulation()
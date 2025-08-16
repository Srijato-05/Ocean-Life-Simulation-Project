# tests/phase3_simulation.py

import sys
import os

# Add the project root to the Python path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from src.simulation.runner import run_simulation
from src.utils.config_loader import load_fauna_config, load_sim_config

def main():
    """
    Main entry point for running the final, stable Phase 3 simulation.
    """
    print("--- Running Final Phase 3 Simulation ---")
    
    # Load the simulation and fauna configurations
    sim_config = load_sim_config()
    fauna_configs = load_fauna_config()
    
    if not sim_config or not fauna_configs:
        print("Exiting: Could not load configuration files.")
        sys.exit(1)
    
    # Run the simulation with verbose output
    run_simulation(sim_config, fauna_configs, verbose=True)

if __name__ == "__main__":
    main()

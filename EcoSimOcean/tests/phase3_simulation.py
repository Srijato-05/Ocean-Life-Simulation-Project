# tests/phase3_simulation.py

import cProfile
import pstats
import sys
import os

# --- HIGHLIGHT: Fix for the ModuleNotFoundError ---
# This block adds the project's main folder to Python's search path,
# allowing it to correctly find and import the 'src' directory.
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)
# --- END HIGHLIGHT ---

from src.simulation.runner import run_simulation
from src.utils.config_loader import load_fauna_config, load_sim_config

def main():
    """
    Main entry point for running the final, stable Phase 3 simulation
    and profiling its performance.
    """
    print("--- Running Final Phase 3 Simulation ---")
    
    sim_config = load_sim_config()
    fauna_configs = load_fauna_config()
    
    if not sim_config or not fauna_configs:
        print("Exiting: Could not load configuration files.")
        sys.exit(1)
    
    # --- HIGHLIGHT: Profiling Logic ---
    # This will run the simulation and then report on which
    # functions are taking the most time to execute.
    profiler = cProfile.Profile()
    
    profiler.enable()
    run_simulation(sim_config, fauna_configs, verbose=True)
    profiler.disable()
    
    print("\n--- PROFILING RESULTS ---")
    stats = pstats.Stats(profiler).sort_stats('cumulative')
    stats.print_stats(20) # Print the top 20 slowest functions
    # --- END HIGHLIGHT ---

if __name__ == "__main__":
    main()
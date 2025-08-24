# tests/phase1_simulation.py

import sys
import os

# Add the project root to the Python path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# CORRECTED: Import from the newly renamed file
from src.phase1_logic import run_phase1_simulation

def main():
    """
    Runs the simulation with only the environment, as defined in Phase 1.
    """
    print("--- Running Phase 1: Environment Only Simulation ---")
    run_phase1_simulation(ticks=100)
    print("--- Phase 1 Simulation Complete ---")

if __name__ == "__main__":
    main()

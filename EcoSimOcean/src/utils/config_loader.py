# src/utils/config_loader.py

import json
import os

def load_fauna_config():
    """
    Loads the species configuration data from the fauna_config.json file.
    """
    config_path = os.path.join(os.path.dirname(__file__), '..', 'config', 'fauna_config.json')
    
    try:
        with open(config_path, 'r') as f:
            config = json.load(f)
        return config
    except FileNotFoundError:
        print(f"Error: Configuration file not found at {config_path}")
        return None
    except json.JSONDecodeError:
        print(f"Error: Could not decode JSON from {config_path}")
        return None

def load_sim_config():
    """
    Loads the main simulation configuration from the sim_config.json file.
    """
    config_path = os.path.join(os.path.dirname(__file__), '..', 'config', 'sim_config.json')
    
    try:
        with open(config_path, 'r') as f:
            config = json.load(f)
        return config
    except FileNotFoundError:
        print(f"Error: Simulation configuration file not found at {config_path}")
        return None
    except json.JSONDecodeError:
        print(f"Error: Could not decode JSON from {config_path}")
        return None

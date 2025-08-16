# src/optimizer/logging.py

"""
This file contains helper functions for logging the performance of
parameter sets during the optimization process. This version provides
comprehensive, multi-level, and dynamic logging for detailed analysis.
"""
import json

def log_particle_performance(particle_index, score, history, params, param_bounds):
    """
    Prints a detailed, transparent, and dynamically grouped log for a single
    particle's test run.
    """
    final_prey = history[-1]['zooplankton'] if history else 0
    final_predators = history[-1]['small_fish'] if history else 0
    
    # --- HIGHLIGHT: Parameters are now grouped for clarity ---
    log_str = f"  P{particle_index+1:2}: Score={score:<7.0f} (Pry:{final_prey:4}, Prd:{final_predators:4}) "
    
    env_logs, prey_logs, pred_logs = [], [], []

    # Dynamically iterate and group all tunable parameters
    for key in param_bounds.keys():
        sim_key = key.replace('_prey', '').replace('_predator', '')
        
        # Find the correct value from the nested config dictionaries
        if sim_key in params:
             value = params[sim_key]
        elif sim_key in params.get("Zooplankton", {}):
             value = params["Zooplankton"][sim_key]
        elif sim_key in params.get("SmallFish", {}):
             value = params["SmallFish"][sim_key]
        else:
             value = 0
             
        key_abbr = "".join([s[0].upper() for s in sim_key.split('_')])
        
        # Format integers and floats differently
        if any(k in key for k in ["period", "count", "threshold", "age"]):
             log_entry = f"{key_abbr}:{value:.0f}"
        else:
             log_entry = f"{key_abbr}:{value:.2f}"

        # Assign the formatted entry to the correct group
        if key.endswith('_prey') or key in ["carrying_capacity_threshold", "starvation_chance", "disease_threshold", "disease_chance", "eating_rate", "energy_conversion_factor"]:
            prey_logs.append(log_entry)
        elif key.endswith('_predator') or key not in ["food_growth_rate", "initial_zooplankton_count", "initial_small_fish_count"]:
            pred_logs.append(log_entry)
        else:
            env_logs.append(log_entry)
            
    # Assemble the final, structured log string
    if env_logs: log_str += f"| Env: {', '.join(env_logs)} "
    if prey_logs: log_str += f"| Prey: {', '.join(prey_logs)} "
    if pred_logs: log_str += f"| Pred: {', '.join(pred_logs)}"
    print(log_str)


def log_iteration_summary(iteration, best_score, best_sim_config, best_fauna_config):
    """
    Prints a detailed summary of the best-performing particle at the end of an iteration.
    """
    print(f"  --- End of Iteration {iteration}: Best score so far: {best_score:.0f} ---")
    if best_sim_config and best_fauna_config:
        print("    Current Best Simulation Config:")
        for key, value in best_sim_config.items():
            if isinstance(value, float):
                print(f"      {key}: {value:.4f}")
            else:
                print(f"      {key}: {value}")

        print("\n    Current Best Fauna Config:")
        for species, params in best_fauna_config.items():
            print(f"      {species}:")
            for key, value in params.items():
                if isinstance(value, float):
                    print(f"        {key}: {value:.4f}")
                else:
                    print(f"        {key}: {value}")
    print("  ----------------------------------------------------")


def log_final_results(best_score, best_sim_config, best_fauna_config):
    """
    Prints the final report after the entire optimization process is complete,
    showing the absolute best parameters found in a copy-paste friendly format.
    """
    print("\n\n--- Particle Swarm Optimization Complete ---")
    if best_score > 100000:
        print(f"SUCCESS: Found a stable configuration with score: {best_score:.0f}!")
        print("\n========================================")
        print("   FINAL OPTIMAL SIMULATION CONFIG")
        print("========================================")
        print(json.dumps(best_sim_config, indent=4))
        print("\n========================================")
        print("    FINAL OPTIMAL FAUNA CONFIG")
        print("========================================")
        print(json.dumps(best_fauna_config, indent=4))
    else:
        print(f"FAILURE: No stable configuration was found.")
        print(f"The best result was a collapse with score {best_score:.0f}.")
        if best_sim_config and best_fauna_config:
            print("\nBest (but failing) Simulation Config Found:")
            print(json.dumps(best_sim_config, indent=4))
            print("\nBest (but failing) Fauna Config Found:")
            print(json.dumps(best_fauna_config, indent=4))
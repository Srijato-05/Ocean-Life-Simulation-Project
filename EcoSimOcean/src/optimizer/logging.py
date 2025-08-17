# src/optimizer/logging.py

"""
This file contains helper functions for logging the performance of
parameter sets. This version provides a clean, structured, and colored
multi-line output for detailed analysis.
"""
import json
from colorama import Fore, Style, init

# Initialize colorama to work on all platforms and autoreset styles
init(autoreset=True)

def log_particle_performance(particle_index, score, history, params, param_bounds):
    """
    Prints a clean, multi-line, colored log for a single particle's run.
    """
    final_prey = history[-1]['zooplankton'] if history else 0
    final_predators = history[-1]['small_fish'] if history else 0
    final_scavengers = history[-1].get('crab', 0) if history else 0
    
    # --- HIGHLIGHT: Color-coded score for at-a-glance analysis ---
    if score > 100000:
        score_color = Fore.GREEN
    elif score > 500: # A long-lasting collapse
        score_color = Fore.YELLOW
    else: # A quick collapse
        score_color = Fore.RED

    score_str = f"{score_color}{Style.BRIGHT}Score={score:<8.0f}{Style.RESET_ALL}"
    pop_str = f"(Pry:{final_prey:4}, Prd:{final_predators:4}, Crb:{final_scavengers:4})"
    print(f"  P{particle_index+1:2}: {score_str} | {pop_str}")

    env_logs, prey_logs, pred_logs, scav_logs = [], [], [], []

    for key in sorted(param_bounds.keys()):
        sim_key = key.replace('_prey', '').replace('_predator', '').replace('_scav', '')
        value = params.get(sim_key)
        if value is None:
            for species_params in params.values():
                if isinstance(species_params, dict) and sim_key in species_params:
                    value = species_params[sim_key]
                    break
        if value is None: value = 0
        key_abbr = "".join([s[0].upper() for s in sim_key.split('_')])
        log_entry = f"{key_abbr}:{value:.2f}" if isinstance(value, float) else f"{key_abbr}:{value}"

        if key.endswith('_scav'): scav_logs.append(log_entry)
        elif key.endswith('_prey'): prey_logs.append(log_entry)
        elif key.endswith('_predator'): pred_logs.append(log_entry)
        else: env_logs.append(log_entry)

    # --- HIGHLIGHT: Colored parameter groups ---
    log_groups = [
        (f"{Fore.CYAN}Env: {Style.RESET_ALL}{', '.join(env_logs)}", bool(env_logs)),
        (f"{Fore.GREEN}Prey: {Style.RESET_ALL}{', '.join(prey_logs)}", bool(prey_logs)),
        (f"{Fore.RED}Pred: {Style.RESET_ALL}{', '.join(pred_logs)}", bool(pred_logs)),
        (f"{Fore.MAGENTA}Scav: {Style.RESET_ALL}{', '.join(scav_logs)}", bool(scav_logs))
    ]
    
    active_groups = [text for text, is_active in log_groups if is_active]
    for i, log_line in enumerate(active_groups):
        prefix = f"{Style.DIM}{'├─' if i < len(active_groups) - 1 else '└─'}{Style.RESET_ALL}"
        print(f"{prefix} {log_line}")


def log_iteration_summary(iteration, best_score, best_sim_config, best_fauna_config, best_history):
    """
    Prints a detailed summary of the best-performing particle at the end of an iteration.
    """
    pop_str = ""
    if best_history:
        final_prey = best_history[-1].get('zooplankton', 'N/A')
        final_pred = best_history[-1].get('small_fish', 'N/A')
        final_scav = best_history[-1].get('crab', 'N/A')
        pop_str = f" (Pry:{final_prey}, Prd:{final_pred}, Crb:{final_scav})"

    print(f"{Style.BRIGHT}  --- End of Iteration {iteration}: Best score so far: {best_score:.0f}{pop_str} ---{Style.NORMAL}")
    
    if best_sim_config and best_fauna_config:
        print(f"{Fore.CYAN}    Current Best Simulation Config:{Style.RESET_ALL}")
        for key, value in best_sim_config.items():
            print(f"      - {key}: {value:.4f}" if isinstance(value, float) else f"      - {key}: {value}")

        print(f"\n{Fore.YELLOW}    Current Best Fauna Config:{Style.RESET_ALL}")
        for species, params in best_fauna_config.items():
            print(f"      > {species}:")
            for key, value in params.items():
                print(f"        - {key}: {value:.4f}" if isinstance(value, float) else f"        - {key}: {value}")
    print("  ----------------------------------------------------")


def log_final_results(best_score, best_sim_config, best_fauna_config):
    """
    Prints the final report after the entire optimization process is complete.
    """
    print("\n\n--- Particle Swarm Optimization Complete ---")
    if best_score > 100000:
        print(f"{Fore.GREEN}{Style.BRIGHT}SUCCESS: Found a stable configuration with score: {best_score:.0f}!{Style.RESET_ALL}")
        print("\n========================================")
        print("   FINAL OPTIMAL SIMULATION CONFIG")
        print("========================================")
        print(json.dumps(best_sim_config, indent=4))
        print("\n========================================")
        print("    FINAL OPTIMAL FAUNA CONFIG")
        print("========================================")
        print(json.dumps(best_fauna_config, indent=4))
    else:
        print(f"{Fore.RED}FAILURE: No stable configuration was found.{Style.RESET_ALL}")
        print(f"The best result was a collapse with score {best_score:.0f}.")
        if best_sim_config and best_fauna_config:
            print("\nBest (but failing) Simulation Config Found:")
            print(json.dumps(best_sim_config, indent=4))
            print("\nBest (but failing) Fauna Config Found:")
            print(json.dumps(best_fauna_config, indent=4))
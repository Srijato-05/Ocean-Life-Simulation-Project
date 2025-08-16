# src/optimizer/scoring.py

"""
This file contains an expert fitness function for evaluating simulation outcomes.
This final version is heavily focused on rewarding ecological balance and actively
penalizing unsustainable, imbalanced populations using a balance factor.
"""
import numpy as np
from scipy.signal import find_peaks

# --- Fitness Function Weights ---
WEIGHTS = {
    "survival_bonus": 100000,
    "resilience_prey": 0.5,
    "resilience_predator": 2.0,
    "oscillation_peak_bonus": 500,
    "stability_bonus": 1000
}

# --- Target Values for a "Good" Ecosystem ---
IDEAL_PREDATOR_PREY_RATIO = 0.1 # Ideal: 1 predator for every 10 prey
IDEAL_COEFFICIENT_OF_VARIATION = 0.4 # Target volatility for stability score

def fitness(history):
    """
    Calculates a 'fitness' score for a simulation run by analyzing its
    entire history with a strong focus on balance.
    """
    if not history:
        return 0

    prey_pop = np.array([h['zooplankton'] for h in history])
    pred_pop = np.array([h['small_fish'] for h in history])

    # 1. Primary Goal: Survival
    if np.any(prey_pop == 0) or np.any(pred_pop == 0):
        survival_ticks = 0
        for p, pr in zip(prey_pop, pred_pop):
            if p > 0 and pr > 0:
                survival_ticks += 1
            else:
                break
        return survival_ticks

    # If the simulation is stable, calculate a more nuanced score.
    score = WEIGHTS["survival_bonus"]

    # 2. Add bonuses for resilience, oscillation, and stability
    score += np.min(prey_pop) * WEIGHTS["resilience_prey"]
    score += np.min(pred_pop) * WEIGHTS["resilience_predator"]
    
    peaks, _ = find_peaks(prey_pop, prominence=np.std(prey_pop) * 0.5)
    score += len(peaks) * WEIGHTS["oscillation_peak_bonus"]
    
    prey_cv = np.std(prey_pop) / np.mean(prey_pop)
    stability_factor = np.exp(-((prey_cv - IDEAL_COEFFICIENT_OF_VARIATION)**2) / (2 * (IDEAL_COEFFICIENT_OF_VARIATION**2)))
    score += stability_factor * WEIGHTS["stability_bonus"]

    # --- HIGHLIGHT: New, Two-Sided Balance Score ---
    # This replaces the one-sided crash penalty.
    
    # Calculate the average predator-to-prey ratio over the whole run.
    avg_ratio = np.mean(pred_pop / (prey_pop + 1e-6))
    
    # The balance score is a bell curve centered on the ideal ratio.
    # The closer the actual ratio is to the ideal, the higher the score (max 1.0).
    # This penalizes BOTH too many predators and too few predators.
    balance_factor = np.exp(-((avg_ratio - IDEAL_PREDATOR_PREY_RATIO)**2) / (2 * (IDEAL_PREDATOR_PREY_RATIO**2)))
    
    # The final score is scaled by this balance factor.
    # An imbalanced ecosystem, even if it survives, will receive a much lower score.
    return score * balance_factor
# src/optimizer/scoring.py

"""
This file contains a very advanced expert fitness function. It includes a powerful
"Extinction Risk Penalty" to ensure all species maintain healthy populations.
"""
import numpy as np
from scipy.signal import find_peaks

# --- HIGHLIGHT: Renamed to SCORING_CONFIG and added Extinction Risk rules ---
SCORING_CONFIG = {
    "survival_bonus": 100000,
    "resilience_prey": 0.5,
    "resilience_predator": 50.0, # Increased predator importance
    "resilience_scavenger": 1.0,
    "oscillation_peak_bonus": 500,
    "stability_bonus": 1000,
    
    # New penalty rules
    "EXTINCTION_RISK_THRESHOLD": 50, # Population level that triggers the penalty
    "EXTINCTION_RISK_PENALTY": 0.1    # Multiplies score by this if threshold is breached
}

# --- Target Values for a "Good" Ecosystem ---
IDEAL_PREDATOR_PREY_RATIO = 0.2
IDEAL_SCAVENGER_PREY_RATIO = 0.05
IDEAL_PREDATOR_SCAVENGER_RATIO = 2.0

def fitness(history):
    """
    Calculates a 'fitness' score with heavy penalties for near-extinction events.
    """
    if not history or len(history) < 2:
        return 0

    prey_pop = np.array([h['zooplankton'] for h in history])
    pred_pop = np.array([h['small_fish'] for h in history])
    scav_pop = np.array([h.get('crab', 0) for h in history])

    # 1. Primary Goal: Survival of ALL species
    if np.any(prey_pop == 0) or np.any(pred_pop == 0) or np.any(scav_pop == 0):
        survival_ticks = 0
        for p, pr, s in zip(prey_pop, pred_pop, scav_pop):
            if p > 0 and pr > 0 and s > 0:
                survival_ticks += 1
            else:
                break
        return survival_ticks

    score = SCORING_CONFIG["survival_bonus"]

    # 2. Add bonuses for resilience, oscillation, and stability
    score += np.min(prey_pop) * SCORING_CONFIG["resilience_prey"]
    score += np.min(pred_pop) * SCORING_CONFIG["resilience_predator"]
    score += np.min(scav_pop) * SCORING_CONFIG["resilience_scavenger"]
    
    peaks, _ = find_peaks(prey_pop, prominence=np.std(prey_pop) * 0.5)
    score += len(peaks) * SCORING_CONFIG["oscillation_peak_bonus"]
    
    prey_cv = np.std(prey_pop) / np.mean(prey_pop)
    stability_factor = 1 / (1 + prey_cv)
    score += stability_factor * SCORING_CONFIG["stability_bonus"]

    # 3. Multi-Factor Balancing
    avg_pred_prey_ratio = np.mean(pred_pop / (prey_pop + 1e-6))
    avg_scav_prey_ratio = np.mean(scav_pop / (prey_pop + 1e-6))
    avg_pred_scav_ratio = np.mean(pred_pop / (scav_pop + 1e-6))
    
    pred_prey_factor = np.exp(-((avg_pred_prey_ratio - IDEAL_PREDATOR_PREY_RATIO)**2) / (2 * (IDEAL_PREDATOR_PREY_RATIO**2)))
    scav_prey_factor = np.exp(-((avg_scav_prey_ratio - IDEAL_SCAVENGER_PREY_RATIO)**2) / (2 * (IDEAL_SCAVENGER_PREY_RATIO**2)))
    pred_scav_factor = np.exp(-((avg_pred_scav_ratio - IDEAL_PREDATOR_SCAVENGER_RATIO)**2) / (2 * (IDEAL_PREDATOR_SCAVENGER_RATIO**2)))
    
    final_score = score * pred_prey_factor * scav_prey_factor * pred_scav_factor
    
    # --- HIGHLIGHT: New Extinction Risk Penalty ---
    # If any species dips below the threshold, apply a massive penalty.
    threshold = SCORING_CONFIG["EXTINCTION_RISK_THRESHOLD"]
    if np.min(pred_pop) < threshold or np.min(scav_pop) < threshold:
        final_score *= SCORING_CONFIG["EXTINCTION_RISK_PENALTY"]
        
    return final_score
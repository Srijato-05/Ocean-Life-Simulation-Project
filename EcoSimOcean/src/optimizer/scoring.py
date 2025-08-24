# src/optimizer/scoring.py

import numpy as np
from scipy.signal import find_peaks

def fitness(history, sim_config):
    """
    Calculates a fitness score for an ecosystem.
    
    This version includes a critical change: any simulation that results in the
    extinction of a starting species is heavily penalized, ensuring the optimizer
    prioritizes biodiversity and true stability.
    """
    if not history:
        return 0

    scoring_params = sim_config.get("scoring", {})
    simulation_ticks = sim_config.get("simulation_ticks", 500)
    survival_ticks = len(history)
    final_state = history[-1]

    # Identify which species started the simulation
    active_species = [s for s in ['zooplankton', 'smallfish', 'crab', 'seal', 'seaturtle'] if sim_config.get(f"initial_{s}_count", 0) > 0]
    
    # --- LOGIC FIX: Heavy penalty for any extinction ---
    # Check if any active species went extinct, even if the simulation ran to completion.
    is_extinction_event = any(final_state.get(s, 0) == 0 for s in active_species)
    
    if survival_ticks < simulation_ticks or is_extinction_event:
        # If the simulation collapsed OR a species went extinct, the score is just its survival time.
        # This ensures these outcomes are always scored lower than a fully stable run.
        return survival_ticks

    # --- SCORING FOR STABLE, DIVERSE ECOSYSTEMS ONLY ---
    prey_pop = np.array([h.get('zooplankton', 0) for h in history])
    pred_pop = np.array([h.get('smallfish', 0) for h in history])
    scav_pop = np.array([h.get('crab', 0) for h in history])
    apex_pop = np.array([h.get('seal', 0) for h in history])
    competitor_pop = np.array([h.get('seaturtle', 0) for h in history])

    # Base score calculated from resilience and stability bonuses
    base_score = scoring_params.get("survival_bonus", 100000) # Start with a large bonus for full survival
    
    if np.mean(prey_pop) > 0: base_score += np.min(prey_pop) * scoring_params.get("resilience_prey", 25.0)
    if np.mean(pred_pop) > 0: base_score += np.min(pred_pop) * scoring_params.get("resilience_predator", 10.0)
    if np.mean(scav_pop) > 0: base_score += np.min(scav_pop) * scoring_params.get("resilience_scavenger", 1.0)
    if np.mean(apex_pop) > 0: base_score += np.min(apex_pop) * scoring_params.get("resilience_apex", 75.0)
    if np.mean(competitor_pop) > 0: base_score += np.min(competitor_pop) * scoring_params.get("resilience_competitor", 12.0)
    
    peaks, _ = find_peaks(prey_pop, prominence=np.std(prey_pop) * 0.5)
    base_score += len(peaks) * scoring_params.get("oscillation_peak_bonus", 500)
    
    prey_cv = np.std(prey_pop) / (np.mean(prey_pop) + 1e-6)
    stability_factor = 1 / (1 + prey_cv)
    base_score += stability_factor * scoring_params.get("stability_bonus", 1000)

    # Ratio-based scoring factors
    mean_prey_pop = np.mean(prey_pop); mean_pred_pop = np.mean(pred_pop)
    mean_scav_pop = np.mean(scav_pop); mean_apex_pop = np.mean(apex_pop)
    mean_competitor_pop = np.mean(competitor_pop)
    
    ideal_pred_prey = scoring_params.get("ideal_predator_prey_ratio", 0.1)
    ideal_scav_prey = scoring_params.get("ideal_scavenger_prey_ratio", 0.05)
    ideal_apex_pred = scoring_params.get("ideal_apex_predator_ratio", 0.08)
    ideal_comp_prey = scoring_params.get("ideal_competitor_prey_ratio", 0.15)
    
    ratio_score = 1.0
    if mean_pred_pop > 0 and mean_prey_pop > 0:
        ratio = mean_pred_pop / (mean_prey_pop + 1e-6)
        ratio_score *= np.exp(-((ratio - ideal_pred_prey)**2) / (2 * (ideal_pred_prey**2)))
    if mean_scav_pop > 0 and mean_prey_pop > 0:
        ratio = mean_scav_pop / (mean_prey_pop + 1e-6)
        ratio_score *= np.exp(-((ratio - ideal_scav_prey)**2) / (2 * (ideal_scav_prey**2)))
    if mean_apex_pop > 0 and mean_pred_pop > 0:
        ratio = mean_apex_pop / (mean_pred_pop + 1e-6)
        ratio_score *= np.exp(-((ratio - ideal_apex_pred)**2) / (2 * (ideal_apex_pred**2)))
    if mean_competitor_pop > 0 and mean_prey_pop > 0:
        ratio = mean_competitor_pop / (mean_prey_pop + 1e-6)
        ratio_score *= np.exp(-((ratio - ideal_comp_prey)**2) / (2 * (ideal_comp_prey**2)))
    
    final_score = base_score * ratio_score
    
    # Apply penalty for near-extinction events
    threshold = scoring_params.get("extinction_risk_threshold", 10)
    min_penalty = scoring_params.get("extinction_risk_penalty", 0.1)
    
    at_risk_populations = []
    if mean_pred_pop > 0: at_risk_populations.append(np.min(pred_pop))
    if mean_scav_pop > 0: at_risk_populations.append(np.min(scav_pop))
    if mean_apex_pop > 0: at_risk_populations.append(np.min(apex_pop))
    if mean_competitor_pop > 0: at_risk_populations.append(np.min(competitor_pop))
        
    if at_risk_populations:
        min_pop_observed = min(at_risk_populations)
        if min_pop_observed < threshold:
            t = min_pop_observed / threshold
            penalty_factor = min_penalty + (1.0 - min_penalty) * t
            final_score *= penalty_factor
            
    return final_score
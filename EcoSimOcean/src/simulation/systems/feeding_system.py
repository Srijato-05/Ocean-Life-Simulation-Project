# src/simulation/systems/feeding_system.py

import numpy as np
from scipy.spatial import KDTree

def handle_feeding(manager):
    """
    Manages all feeding behaviors for the current tick, including eating
    plankton, predation, and scavenging.
    """
    _eat_plankton(manager)
    _handle_scavenging(manager)
    _handle_all_predation(manager)


def _eat_plankton(manager):
    """Handles plankton consumption for all relevant species."""
    unsatiated_mask = manager.satiation_timers == 0
    plankton_eater_mask = ((manager.species_ids == manager.SPECIES_ID["Zooplankton"]) | 
                           (manager.species_ids == manager.SPECIES_ID["SeaTurtle"]) |
                           (manager.species_ids == manager.SPECIES_ID["SmallFish"])) & manager.alive_mask & unsatiated_mask
    
    if not np.any(plankton_eater_mask): return

    eater_indices = np.where(plankton_eater_mask)[0]
    eater_species_ids = manager.species_ids[plankton_eater_mask]
    
    positions_int = manager.positions[plankton_eater_mask].astype(int)
    unique_cells, cell_map = np.unique(positions_int, axis=0, return_inverse=True)

    eating_rates = np.zeros(len(eater_indices), dtype=float)
    conversion_factors = np.zeros(len(eater_indices), dtype=float)
    satiation_periods = np.zeros(len(eater_indices), dtype=int)
    
    # Standard herbivores
    for species_name in ["Zooplankton", "SeaTurtle"]:
        species_id = manager.SPECIES_ID[species_name]
        mask = eater_species_ids == species_id
        if np.any(mask):
            config = manager.fauna_configs[species_name]
            eating_rates[mask] = config.get("eating_rate", 0.1)
            conversion_factors[mask] = config.get("energy_conversion_factor", 1.0)
            satiation_periods[mask] = config.get("plankton_satiation_period", 5)

    # Special logic for SmallFish
    fish_config = manager.fauna_configs["SmallFish"]
    fish_mask_local = eater_species_ids == manager.SPECIES_ID["SmallFish"]
    
    if np.any(fish_mask_local):
        fish_indices_local = np.where(fish_mask_local)[0]
        fish_indices_global = eater_indices[fish_indices_local]

        maturity_age = fish_config.get("maturity_age", 0)
        fish_ages = manager.ages[fish_indices_global]
        is_juvenile_mask = fish_ages < maturity_age
        
        # Juvenile fish always eat plankton
        if np.any(is_juvenile_mask):
            juvenile_indices_local = fish_indices_local[is_juvenile_mask]
            eating_rates[juvenile_indices_local] = fish_config.get("eating_rate", 0.1)
            conversion_factors[juvenile_indices_local] = fish_config.get("energy_conversion_factor", 1.0)
            satiation_periods[juvenile_indices_local] = fish_config.get("plankton_satiation_period", 5)

        # --- LOGIC FIX: Adult fish eat plankton based on LOCAL prey scarcity ---
        is_adult_mask = ~is_juvenile_mask
        if np.any(is_adult_mask):
            adult_indices_global = fish_indices_global[is_adult_mask]
            adult_positions = manager.positions[adult_indices_global]
            
            prey_scarcity_threshold = fish_config.get("prey_scarcity_threshold", 5)
            vision_radius = fish_config.get("vision_radius", 20)

            # Build a KD-Tree of zooplankton to check local density
            zoo_mask = (manager.species_ids == manager.SPECIES_ID["Zooplankton"]) & manager.alive_mask
            if np.any(zoo_mask):
                zoo_positions = manager.positions[zoo_mask]
                zoo_tree = KDTree(zoo_positions)
                
                # Find number of prey within vision radius for each adult fish
                nearby_prey_counts = zoo_tree.query_ball_point(adult_positions, r=vision_radius, return_length=True)
                
                # Identify fish experiencing prey scarcity
                is_scarce_mask = nearby_prey_counts < prey_scarcity_threshold
                if np.any(is_scarce_mask):
                    scarce_fish_indices_local = np.where(is_adult_mask)[0][is_scarce_mask]
                    eating_rates[scarce_fish_indices_local] = fish_config.get("eating_rate", 0.1)
                    conversion_factors[scarce_fish_indices_local] = fish_config.get("energy_conversion_factor", 1.0)
                    satiation_periods[scarce_fish_indices_local] = fish_config.get("plankton_satiation_period", 5)

    # ... (rest of the plankton eating logic remains the same) ...
    plankton_in_cells = manager.env.plankton[unique_cells[:, 0], unique_cells[:, 1], unique_cells[:, 2]]
    
    low_plankton_threshold = manager.env.config.get("low_plankton_threshold", 0.1)
    scarce_mask = plankton_in_cells < low_plankton_threshold
    if np.any(scarce_mask):
        scaling_factor = plankton_in_cells[scarce_mask] / low_plankton_threshold
        cells_to_scale = np.where(scarce_mask)[0]
        agent_mask_to_scale = np.isin(cell_map, cells_to_scale)
        eating_rates[agent_mask_to_scale] *= scaling_factor[np.searchsorted(cells_to_scale, cell_map[agent_mask_to_scale])]

    total_demand_per_cell = np.bincount(cell_map, weights=eating_rates)
    eaten_per_cell = np.minimum(total_demand_per_cell, plankton_in_cells)
    scale_factor = np.divide(eaten_per_cell, total_demand_per_cell, out=np.zeros_like(eaten_per_cell), where=total_demand_per_cell!=0)
    amount_to_eat = eating_rates * scale_factor[cell_map]
    
    np.subtract.at(manager.env.plankton, tuple(positions_int.T), amount_to_eat)

    baseline_energy_gain = 0.4
    energy_gain = (amount_to_eat * conversion_factors) + baseline_energy_gain
    manager.energies[eater_indices] += energy_gain
    
    consumed_mask = amount_to_eat > 0
    if np.any(consumed_mask):
        satiated_agent_indices = eater_indices[consumed_mask]
        satiation_durations = satiation_periods[consumed_mask]
        manager.satiation_timers[satiated_agent_indices] = satiation_durations

# ... (rest of the file remains the same) ...
def _handle_scavenging(manager):
    """Handles marine snow consumption for crabs."""
    crab_mask = (manager.species_ids == manager.SPECIES_ID["Crab"]) & manager.alive_mask
    if not np.any(crab_mask): return
    crab_indices = np.where(crab_mask)[0]
    crab_positions = manager.positions[crab_mask]
    
    not_on_bottom_mask = crab_positions[:, 2] < manager.env.depth - 1
    manager.positions[crab_indices[not_on_bottom_mask], 2] += 1
    
    on_bottom_mask = ~not_on_bottom_mask
    if np.any(on_bottom_mask):
        bottom_crab_indices = crab_indices[on_bottom_mask]
        bottom_crab_positions = manager.positions[bottom_crab_indices].astype(int)
        offsets = np.array([[dx, dy] for dx in [-1, 0, 1] for dy in [-1, 0, 1]])
        neighbor_coords = bottom_crab_positions[:, np.newaxis, :2] + offsets
        neighbor_coords[:, :, 0] = np.clip(neighbor_coords[:, :, 0], 0, manager.env.width - 1)
        neighbor_coords[:, :, 1] = np.clip(neighbor_coords[:, :, 1], 0, manager.env.height - 1)
        snow_values = manager.env.marine_snow[neighbor_coords[:, :, 0], neighbor_coords[:, :, 1], bottom_crab_positions[:, 2, np.newaxis]]
        best_neighbor_indices = np.argmax(snow_values, axis=1)
        best_offsets = offsets[best_neighbor_indices]
        manager.positions[bottom_crab_indices, :2] += best_offsets
        manager.positions[bottom_crab_indices, 0] %= manager.env.width
        manager.positions[bottom_crab_indices, 1] %= manager.env.height

    final_pos_int = manager.positions[crab_mask].astype(int)
    px, py, pz = np.clip(final_pos_int[:, 0], 0, manager.env.width-1), \
                 np.clip(final_pos_int[:, 1], 0, manager.env.height-1), \
                 np.clip(final_pos_int[:, 2], 0, manager.env.depth-1)
    snow_available = manager.env.marine_snow[px, py, pz]
    eating_rate = manager.fauna_configs["Crab"]["eating_rate"]
    amount_to_eat = np.minimum(snow_available, eating_rate)
    manager.env.marine_snow[px, py, pz] -= amount_to_eat
    energy_gain = amount_to_eat * manager.fauna_configs["Crab"]["energy_conversion_factor"]
    manager.energies[crab_mask] += energy_gain


def _handle_all_predation(manager):
    """Manages predation logic for all predator-prey relationships."""
    if manager.is_bootstrap:
        manager.targets.fill(-1)
        return

    id_to_size_map = np.array([0] * (len(manager.SPECIES_ID) + 1))
    for name, species_id in manager.SPECIES_ID.items():
        id_to_size_map[species_id] = manager.fauna_configs[name]["size"]

    for predator_name, prey_names in manager.diet_config.items():
        predator_id = manager.SPECIES_ID[predator_name]
        predator_mask = (manager.species_ids == predator_id) & manager.alive_mask & (manager.satiation_timers == 0)
        if not np.any(predator_mask): continue

        prey_ids = [manager.SPECIES_ID[name] for name in prey_names]
        prey_mask = np.isin(manager.species_ids, prey_ids) & manager.alive_mask
        
        prey_indices = np.where(prey_mask)[0]
        if prey_indices.size > 0:
            is_adult_prey_mask = np.ones(len(prey_indices), dtype=bool)
            
            for prey_name in prey_names:
                prey_config = manager.fauna_configs.get(prey_name, {})
                maturity_age = prey_config.get("maturity_age", 0)
                if maturity_age > 0:
                    prey_id = manager.SPECIES_ID[prey_name]
                    species_specific_prey_mask = manager.species_ids[prey_indices] == prey_id
                    
                    if np.any(species_specific_prey_mask):
                        ages_of_prey = manager.ages[prey_indices[species_specific_prey_mask]]
                        is_adult_prey_mask[species_specific_prey_mask] = (ages_of_prey >= maturity_age)

            adult_prey_indices = prey_indices[is_adult_prey_mask]
            prey_mask.fill(False)
            if adult_prey_indices.size > 0:
                prey_mask[adult_prey_indices] = True
        
        if not np.any(prey_mask): continue

        predator_indices = np.where(predator_mask)[0]
        prey_indices = np.where(prey_mask)[0] 
        if len(prey_indices) == 0: continue

        prey_tree = KDTree(manager.positions[prey_indices])
        distances, target_local_indices = prey_tree.query(manager.positions[predator_indices])
        
        config = manager.fauna_configs[predator_name]
        vision_radius = config["vision_radius"]
        predation_range = config["predation_range"]
        hunt_chance = config.get("hunt_success_chance", 1.0)
        satiation_period = config["satiation_period"]
        
        can_see_prey_mask = distances < vision_radius
        predators_that_can_see = predator_indices[can_see_prey_mask]
        targets_for_those_predators = prey_indices[target_local_indices[can_see_prey_mask]]
        manager.targets[predators_that_can_see] = targets_for_those_predators
        
        final_hunt_chances = np.full(len(predator_indices), hunt_chance)
        maturity_age = config.get("maturity_age", 0)
        if maturity_age > 0:
            ages_of_hunters = manager.ages[predator_indices]
            is_juvenile_mask = ages_of_hunters < maturity_age
            if np.any(is_juvenile_mask):
                juvenile_modifier = config.get("juvenile_hunt_modifier", 0.5)
                final_hunt_chances[is_juvenile_mask] *= juvenile_modifier
        
        potential_targets_indices = prey_indices[target_local_indices]
        target_positions = manager.positions[potential_targets_indices].astype(int)
        
        in_refuge_mask = manager.env.refuge_map[target_positions[:, 0], target_positions[:, 1], target_positions[:, 2]]
        if np.any(in_refuge_mask):
            refuge_debuff = manager.env.config.get("refuge_hunt_debuff", 0.5)
            final_hunt_chances[in_refuge_mask] *= refuge_debuff

        random_rolls = np.random.random(len(predator_indices))
        final_success_mask = (distances < predation_range) & (random_rolls < final_hunt_chances)
        
        if not np.any(final_success_mask): continue

        potential_hunter_indices = predator_indices[final_success_mask]
        potential_killed_indices = prey_indices[target_local_indices[final_success_mask]]

        unique_killed_prey, first_occurrence_indices = np.unique(potential_killed_indices, return_index=True)
        
        truly_successful_hunter_indices = potential_hunter_indices[first_occurrence_indices]
        killed_prey_indices = unique_killed_prey
        
        if len(killed_prey_indices) > 0:
            manager.alive_mask[killed_prey_indices] = False
            
            max_efficiency = config.get("max_energy_transfer_efficiency", 0.8)
            optimal_size = config.get("optimal_prey_size", 5.0)
            tolerance = config.get("prey_size_tolerance", 5.0)

            killed_prey_species_ids = manager.species_ids[killed_prey_indices]
            prey_sizes = id_to_size_map[killed_prey_species_ids]

            size_diff_sq = (prey_sizes - optimal_size)**2
            dynamic_efficiency = max_efficiency * np.exp(-size_diff_sq / (2 * tolerance**2))
            
            energy_transfer = prey_sizes * dynamic_efficiency
            
            manager.energies[truly_successful_hunter_indices] += energy_transfer
            manager.satiation_timers[truly_successful_hunter_indices] = satiation_period
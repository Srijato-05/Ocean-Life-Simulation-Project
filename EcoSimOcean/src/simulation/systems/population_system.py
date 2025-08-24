# src/simulation/systems/population_system.py

import numpy as np

def update_population_dynamics(manager):
    """
    Handles all non-movement, non-feeding agent state changes for one tick.
    """
    _apply_metabolism_and_aging(manager)
    _handle_overcrowding(manager)
    _handle_disease(manager)
    _handle_deaths(manager)
    _handle_reproduction(manager)


def _apply_metabolism_and_aging(manager):
    """Applies energy loss from metabolism and increments agent ages."""
    active_mask = manager.alive_mask
    if not np.any(active_mask): return
    
    alive_indices = np.where(active_mask)[0]
    alive_positions = manager.positions[active_mask].astype(int)
    alive_species_ids = manager.species_ids[active_mask]
    px, py, pz = np.clip(alive_positions[:, 0], 0, manager.env.width-1), \
                 np.clip(alive_positions[:, 1], 0, manager.env.height-1), \
                 np.clip(alive_positions[:, 2], 0, manager.env.depth-1)
    metabolic_mods = manager.env.metabolic_map[px, py, pz]
    
    for species_name, species_id in manager.SPECIES_ID.items():
        config = manager.fauna_configs[species_name]
        species_mask_local = alive_species_ids == species_id
        if np.any(species_mask_local):
            species_global_indices = alive_indices[species_mask_local]
            base_rate = config["metabolic_rate"]
            
            if manager.is_bootstrap:
                base_rate *= manager.env.config.get("bootstrap_metabolic_modifier", 0.5)

            maturity_age = config.get("maturity_age", 0)
            if maturity_age > 0 and not manager.is_bootstrap:
                agent_ages = manager.ages[species_global_indices]
                is_juvenile_mask = agent_ages < maturity_age
                if np.any(is_juvenile_mask):
                    modifier = config.get("juvenile_metabolic_modifier", 1.0)
                    rates = np.full(len(species_global_indices), base_rate)
                    rates[is_juvenile_mask] *= modifier
                    manager.energies[species_global_indices] -= rates * metabolic_mods[species_mask_local]
                else:
                    manager.energies[species_global_indices] -= base_rate * metabolic_mods[species_mask_local]
            else:
                manager.energies[species_global_indices] -= base_rate * metabolic_mods[species_mask_local]

    predator_mask = (manager.species_ids == manager.SPECIES_ID["SmallFish"]) | (manager.species_ids == manager.SPECIES_ID["Seal"])
    manager.cooldowns[predator_mask] = np.maximum(0, manager.cooldowns[predator_mask] - 1)
    manager.satiation_timers[manager.alive_mask] = np.maximum(0, manager.satiation_timers[manager.alive_mask] - 1)
    
    if not manager.is_bootstrap:
        manager.ages[manager.alive_mask] += 1


def _handle_disease(manager):
    """Handles agent deaths from disease based on population density and biome."""
    for species_name, species_id in manager.SPECIES_ID.items():
        config = manager.fauna_configs[species_name]
        base_chance = config.get("disease_chance", 0.0)
        if base_chance == 0: continue
        species_mask = (manager.species_ids == species_id) & manager.alive_mask
        current_pop = np.sum(species_mask)
        pop_density_threshold = config.get("disease_threshold", 99999)
        if current_pop <= pop_density_threshold: continue
        species_indices = np.where(species_mask)[0]
        agent_positions = manager.positions[species_indices].astype(int)
        px, py, pz = agent_positions.T
        env_risk_factors = manager.env.disease_risk_map[px, py, pz]
        final_chances = base_chance * env_risk_factors
        random_rolls = np.random.random(size=current_pop)
        disease_mask = random_rolls < final_chances
        if np.any(disease_mask):
            agents_to_die_indices = species_indices[disease_mask]
            manager.alive_mask[agents_to_die_indices] = False


def _handle_overcrowding(manager):
    """
    Applies a chance of death to agents in overcrowded grid cells.
    """
    for species_name, species_id in manager.SPECIES_ID.items():
        config = manager.fauna_configs[species_name]
        threshold = config.get("carrying_capacity_threshold", 99)
        starvation_chance = config.get("starvation_chance", 0.0)
        
        if starvation_chance == 0: continue
        species_mask = (manager.species_ids == species_id) & manager.alive_mask
        if not np.any(species_mask): continue
            
        species_indices_global = np.where(species_mask)[0]
        species_positions_int = manager.positions[species_mask].astype(int)
        
        unique_cells, cell_indices, counts = np.unique(species_positions_int, axis=0, return_inverse=True, return_counts=True)
        
        overcrowded_cell_mask = counts > threshold
        if np.any(overcrowded_cell_mask):
            agents_in_overcrowded_cells_mask = np.isin(cell_indices, np.where(overcrowded_cell_mask)[0])
            num_to_roll = np.sum(agents_in_overcrowded_cells_mask)
            
            random_rolls = np.random.random(size=num_to_roll)
            starvation_mask = random_rolls < starvation_chance
            
            if np.any(starvation_mask):
                agents_to_die_indices = species_indices_global[agents_in_overcrowded_cells_mask][starvation_mask]
                manager.alive_mask[agents_to_die_indices] = False


def _handle_deaths(manager):
    """Handles agent deaths from starvation (energy <= 0) or old age."""
    active_mask = manager.alive_mask
    starvation_dead = (manager.energies <= 0) & active_mask
    
    is_old_age = np.zeros_like(active_mask, dtype=bool)
    for species_name, species_id in manager.SPECIES_ID.items():
        lifespan = manager.fauna_configs[species_name].get("max_lifespan", 99999)
        species_mask = (manager.species_ids == species_id) & active_mask
        if np.any(species_mask):
            is_old_age[species_mask] = manager.ages[species_mask] >= lifespan
        
    newly_dead_mask = starvation_dead | is_old_age
    if np.any(newly_dead_mask):
        manager.alive_mask[newly_dead_mask] = False

def _handle_reproduction(manager):
    """
    Handles reproduction, now with a hard cap based on local cell density
    to prevent runaway population explosions.
    """
    threatened_mask = manager.threatened_mask
    
    repro_mask = np.zeros(manager.capacity, dtype=bool)
    for species_name, species_id in manager.SPECIES_ID.items():
        config = manager.fauna_configs[species_name]
        threshold = config.get("reproduction_threshold", 9999)
        
        # Initial mask for agents that have enough energy and are alive
        species_mask = (manager.species_ids == species_id) & (manager.energies > threshold) & manager.alive_mask
        if not np.any(species_mask): continue
        
        # --- LOGIC FIX: Add a hard cap on reproduction based on local density ---
        capacity_threshold = config.get("carrying_capacity_threshold", 99)
        
        # Get positions and counts for the current species
        current_species_indices = np.where((manager.species_ids == species_id) & manager.alive_mask)[0]
        if current_species_indices.size == 0: continue
        
        positions_int = manager.positions[current_species_indices].astype(int)
        unique_cells, cell_map, counts = np.unique(positions_int, axis=0, return_inverse=True, return_counts=True)
        
        # Identify cells that are at or over capacity
        full_cell_indices = np.where(counts >= capacity_threshold)[0]
        if full_cell_indices.size > 0:
            # Mask out agents that are in those full cells
            is_in_full_cell_mask = np.isin(cell_map, full_cell_indices)
            cannot_reproduce_indices = current_species_indices[is_in_full_cell_mask]
            species_mask[cannot_reproduce_indices] = False
        
        if not np.any(species_mask): continue

        maturity_age = config.get("maturity_age", 0)
        if maturity_age > 0 and not manager.is_bootstrap:
            true_indices = np.where(species_mask)[0]
            if true_indices.size > 0:
                agent_ages = manager.ages[true_indices]
                is_adult_mask = agent_ages >= maturity_age
                species_mask[true_indices[~is_adult_mask]] = False

        repro_debuff = config.get("reproduction_fear_debuff", 1.0)
        if repro_debuff < 1.0:
            is_threatened_species = threatened_mask & species_mask
            if np.any(is_threatened_species):
                num_threatened = np.sum(is_threatened_species)
                rand_rolls = np.random.random(num_threatened)
                failed_repro_mask = rand_rolls < (1.0 - repro_debuff)
                threatened_indices = np.where(is_threatened_species)[0]
                cannot_reproduce_indices = threatened_indices[failed_repro_mask]
                species_mask[cannot_reproduce_indices] = False

        if "reproduction_cooldown_period" in config:
            species_mask &= (manager.cooldowns == 0)

        repro_mask |= species_mask

    if not np.any(repro_mask): return
    
    reproducing_indices = np.where(repro_mask)[0]
    num_offspring = len(reproducing_indices)

    empty_slots_indices = np.where(~manager.alive_mask)[0]
    available_slots = len(empty_slots_indices)

    if num_offspring > available_slots:
        new_capacity = int(manager.capacity * 1.5) + num_offspring
        manager._resize_arrays(new_capacity)
    
    num_to_birth = min(num_offspring, available_slots)
    if num_to_birth == 0: return

    reproducing_indices_to_birth = reproducing_indices[:num_to_birth]
    slots_to_fill = empty_slots_indices[:num_to_birth]

    manager.energies[reproducing_indices_to_birth] /= 2
    for species_name, species_id in manager.SPECIES_ID.items():
        config = manager.fauna_configs[species_name]
        if "reproduction_cooldown_period" in config:
            species_mask_local = (manager.species_ids[reproducing_indices_to_birth] == species_id)
            if np.any(species_mask_local):
                cooldown = config["reproduction_cooldown_period"]
                manager.cooldowns[reproducing_indices_to_birth[species_mask_local]] = cooldown

    manager.alive_mask[slots_to_fill] = True
    manager.positions[slots_to_fill] = manager.positions[reproducing_indices_to_birth]
    manager.energies[slots_to_fill] = manager.energies[reproducing_indices_to_birth]
    manager.species_ids[slots_to_fill] = manager.species_ids[reproducing_indices_to_birth]
    manager.ages[slots_to_fill] = 0
    manager.cooldowns[slots_to_fill] = 0
    manager.satiation_timers[slots_to_fill] = 0
    manager.targets[slots_to_fill] = -1
    manager.search_vectors[slots_to_fill] = np.random.randint(-1, 2, size=(num_to_birth, 3))
    
    manager.num_agents += num_to_birth
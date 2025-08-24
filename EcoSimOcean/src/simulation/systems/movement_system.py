# src/simulation/systems/movement_system.py

import numpy as np

def update_positions(manager):
    """
    Calculates and applies movement deltas for all agents based on their
    current state. It now reads the pre-calculated threat mask from the manager.
    """
    movement_deltas = np.zeros_like(manager.positions, dtype=float)
    
    # --- Fleeing Behavior ---
    threatened_mask = manager.threatened_mask
    threatened_prey_indices = np.where(threatened_mask)[0]
    if threatened_prey_indices.size > 0:
        # --- FIX: Select only the flee vectors for the threatened agents ---
        movement_deltas[threatened_prey_indices] = manager.flee_vectors[threatened_prey_indices]

    # --- Chasing Behavior ---
    predator_species_ids = [manager.SPECIES_ID[name] for name in manager.diet_config.keys()]
    predator_mask = np.isin(manager.species_ids, predator_species_ids) & manager.alive_mask
    has_target_mask = (manager.targets != -1) & predator_mask
    chasing_indices = np.where(has_target_mask)[0]
    if chasing_indices.size > 0:
        target_indices = manager.targets[chasing_indices]
        valid_targets_mask = target_indices < len(manager.positions)
        chasing_indices = chasing_indices[valid_targets_mask]
        target_indices = target_indices[valid_targets_mask]
        
        if chasing_indices.size > 0:
            delta_chase = manager.positions[target_indices] - manager.positions[chasing_indices]
            movement_deltas[chasing_indices] = np.sign(delta_chase)

    # --- Searching/Wandering Behavior ---
    is_hungry_mask = np.zeros_like(manager.alive_mask, dtype=bool)
    for name in manager.diet_config.keys():
        config = manager.fauna_configs[name]
        hunger_threshold = config.get("hunger_threshold", config["reproduction_threshold"] / 2)
        species_id = manager.SPECIES_ID[name]
        is_hungry_mask |= (manager.species_ids == species_id) & (manager.energies < hunger_threshold)
    
    searching_mask = is_hungry_mask & ~has_target_mask & manager.alive_mask
    searching_indices = np.where(searching_mask)[0]
    if searching_indices.size > 0:
        change_dir_mask = np.random.random(len(searching_indices)) < 0.1
        new_vectors = np.random.randint(-1, 2, size=(np.sum(change_dir_mask), 3))
        manager.search_vectors[searching_indices[change_dir_mask]] = new_vectors
        movement_deltas[searching_indices] = manager.search_vectors[searching_indices]
    
    # --- Random Movement ---
    random_mask = ~(threatened_mask | has_target_mask | searching_mask) & manager.alive_mask
    num_random = np.sum(random_mask)
    if num_random > 0:
        movement_deltas[random_mask] = np.random.randint(-1, 2, size=(num_random, 3))
    
    # Apply movement and handle world boundaries
    manager.positions += movement_deltas
    manager.positions[:, 0] %= manager.env.width
    manager.positions[:, 1] %= manager.env.height
    manager.positions[:, 2] = np.clip(manager.positions[:, 2], 0, manager.env.depth - 1)
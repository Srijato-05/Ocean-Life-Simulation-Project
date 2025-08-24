# src/simulation/simulation_manager.py

import numpy as np
from scipy.spatial import KDTree
# --- REVERTED: Numba is no longer needed and has been removed ---

from src.utils.config_loader import load_diet_config
from src.simulation.systems import movement_system, feeding_system, population_system

PROCESSED_DEAD_ENERGY = -999.0

class SimulationManager:
    """
    Manages agent state using pre-allocated NumPy arrays for high performance.
    This version enforces a hard cap on the total agent population.
    """
    def __init__(self, env, initial_agents, fauna_configs):
        self.env = env
        self.fauna_configs = fauna_configs
        self.diet_config = load_diet_config()
        self.SPECIES_ID = {"Zooplankton": 1, "SmallFish": 2, "Crab": 3, "Seal": 4, "SeaTurtle": 5}
        
        self.tick = 0
        self.bootstrap_period = self.env.config.get("bootstrap_period", 0)
        self.is_bootstrap = True
        self.cleanup_interval = self.env.config.get("cleanup_interval", 10)
        self.threat_update_interval = self.env.config.get("threat_update_interval", 5) 

        self.num_agents = len(initial_agents)
        self.capacity = self.env.config.get("initial_agent_capacity", 20000)
        self.absolute_max_agents = self.env.config.get("absolute_max_agents", 75000)
        if self.num_agents > self.capacity:
            self.capacity = self.num_agents * 2

        self.positions = np.zeros((self.capacity, 3), dtype=float)
        self.energies = np.zeros(self.capacity, dtype=float)
        self.species_ids = np.zeros(self.capacity, dtype=int)
        self.alive_mask = np.zeros(self.capacity, dtype=bool)
        self.cooldowns = np.zeros(self.capacity, dtype=int)
        self.ages = np.zeros(self.capacity, dtype=int)
        self.satiation_timers = np.zeros(self.capacity, dtype=int)
        self.targets = np.full(self.capacity, -1, dtype=int)
        self.search_vectors = np.zeros((self.capacity, 3), dtype=int)
        
        if self.num_agents > 0:
            self.positions[:self.num_agents] = [[a.x, a.y, a.z] for a in initial_agents]
            self.energies[:self.num_agents] = [a.energy for a in initial_agents]
            self.species_ids[:self.num_agents] = [self.SPECIES_ID[a.species] for a in initial_agents]
            self.alive_mask[:self.num_agents] = True
            self.search_vectors[:self.num_agents] = np.random.randint(-1, 2, size=(self.num_agents, 3))
            
        self.threatened_mask = np.zeros(self.capacity, dtype=bool)
        self.flee_vectors = np.zeros((self.capacity, 3), dtype=float)

    def _resize_arrays(self, requested_capacity):
        """Dynamically resizes arrays, capped at the absolute maximum."""
        if self.capacity >= self.absolute_max_agents: return
        new_capacity = min(requested_capacity, self.absolute_max_agents)
        if new_capacity <= self.capacity: return
        
        self.positions = np.resize(self.positions, (new_capacity, 3))
        self.energies = np.resize(self.energies, new_capacity)
        self.species_ids = np.resize(self.species_ids, new_capacity)
        self.alive_mask = np.resize(self.alive_mask, new_capacity)
        self.cooldowns = np.resize(self.cooldowns, new_capacity)
        self.ages = np.resize(self.ages, new_capacity)
        self.satiation_timers = np.resize(self.satiation_timers, new_capacity)
        self.targets = np.resize(self.targets, new_capacity)
        self.search_vectors = np.resize(self.search_vectors, (new_capacity, 3))
        self.threatened_mask = np.resize(self.threatened_mask, new_capacity)
        self.flee_vectors = np.resize(self.flee_vectors, (new_capacity, 3))
        
        self.capacity = new_capacity

    def update(self):
        """The main update loop."""
        if self.num_agents == 0: return

        self.is_bootstrap = self.tick < self.bootstrap_period
        self.tick += 1

        if self.tick % self.threat_update_interval == 0:
            self._update_threat_mask()

        population_system.update_population_dynamics(self)
        feeding_system.handle_feeding(self)
        movement_system.update_positions(self)
        self.cleanup()

    def _update_threat_mask(self):
        """
        --- REVERTED: Use the faster and more stable KDTree implementation ---
        Calculates and stores the threat mask and flee vectors for the current tick.
        """
        predator_mask = np.isin(self.species_ids, [self.SPECIES_ID["SmallFish"], self.SPECIES_ID["Seal"]]) & self.alive_mask
        prey_mask = np.isin(self.species_ids, [self.SPECIES_ID["Zooplankton"], self.SPECIES_ID["SmallFish"], self.SPECIES_ID["Crab"], self.SPECIES_ID["SeaTurtle"]]) & self.alive_mask
        
        self.threatened_mask.fill(False)
        self.flee_vectors.fill(0)

        if not np.any(predator_mask) or not np.any(prey_mask): return

        predator_indices = np.where(predator_mask)[0]
        prey_indices = np.where(prey_mask)[0]
        
        if predator_indices.size == 0 or prey_indices.size == 0: return

        predator_positions = self.positions[predator_indices]
        prey_positions = self.positions[prey_indices]
        predator_tree = KDTree(predator_positions)
        
        # Find all predators within a fixed radius for each prey
        nearby_predator_indices_list = predator_tree.query_ball_point(prey_positions, r=15.0)

        for i, nearby_indices in enumerate(nearby_predator_indices_list):
            if nearby_indices:
                prey_global_index = prey_indices[i]
                current_prey_pos = self.positions[prey_global_index]
                
                # Flee from the average position of all nearby threats
                avg_flee_vector = np.zeros(3)
                for local_pred_idx in nearby_indices:
                    pred_global_index = predator_indices[local_pred_idx]
                    avg_flee_vector += (current_prey_pos - self.positions[pred_global_index])
                
                norm = np.linalg.norm(avg_flee_vector)
                if norm > 0:
                    self.flee_vectors[prey_global_index] = np.round(avg_flee_vector / norm)
                    self.threatened_mask[prey_global_index] = True

    def get_population_counts(self):
        """Returns a dictionary with the current count of each species."""
        counts = {}
        for species_name, species_id in self.SPECIES_ID.items():
            count = np.sum((self.species_ids == species_id) & self.alive_mask)
            counts[species_name.lower()] = count
        return counts

    def cleanup(self):
        """Handles marine snow deposition and periodic array compaction."""
        newly_dead_mask = ~self.alive_mask & (self.energies != PROCESSED_DEAD_ENERGY)
        if np.any(newly_dead_mask):
            dead_positions = self.positions[newly_dead_mask].astype(int)
            sizes = np.array([0] * (len(self.SPECIES_ID) + 1))
            for name, species_id in self.SPECIES_ID.items():
                sizes[species_id] = self.fauna_configs[name]["size"]
            dead_species_ids = self.species_ids[newly_dead_mask]
            dead_sizes = sizes[dead_species_ids]
            np.add.at(self.env.marine_snow, tuple(dead_positions.T), dead_sizes)
            self.energies[newly_dead_mask] = PROCESSED_DEAD_ENERGY

        if self.tick % self.cleanup_interval != 0: return

        active_indices = np.where(self.alive_mask)[0]
        num_active = len(active_indices)
        
        if num_active == self.num_agents: return

        if num_active < self.num_agents and self.num_agents > 0:
            index_map = np.full(self.capacity, -1, dtype=int)
            index_map[active_indices] = np.arange(num_active)
            
            old_targets = self.targets[active_indices]
            valid_target_mask = old_targets != -1
            remapped_targets = np.full(num_active, -1, dtype=int)
            if np.any(valid_target_mask):
                valid_old_targets = old_targets[valid_target_mask]
                remapped_values = index_map[valid_old_targets]
                remapped_targets[valid_target_mask] = np.where(remapped_values != -1, remapped_values, -1)
            
            self.positions[:num_active] = self.positions[active_indices]
            self.energies[:num_active] = self.energies[active_indices]
            self.species_ids[:num_active] = self.species_ids[active_indices]
            self.cooldowns[:num_active] = self.cooldowns[active_indices]
            self.ages[:num_active] = self.ages[active_indices]
            self.satiation_timers[:num_active] = self.satiation_timers[active_indices]
            self.search_vectors[:num_active] = self.search_vectors[active_indices]
            self.threatened_mask[:num_active] = self.threatened_mask[active_indices]
            self.flee_vectors[:num_active] = self.flee_vectors[active_indices]
            self.targets[:num_active] = remapped_targets
            
            self.alive_mask[:num_active] = True
            self.alive_mask[num_active:] = False
            self.targets[num_active:] = -1
            self.energies[num_active:] = 0
            
            self.num_agents = num_active
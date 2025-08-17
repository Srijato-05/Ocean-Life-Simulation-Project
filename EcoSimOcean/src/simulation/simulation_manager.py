# src/simulation/simulation_manager.py

import numpy as np
import random
from scipy.spatial import KDTree
from src.utils.spatial_hash import SpatialHash
from src.biome import BIOME_DATA

class SimulationManager:
    """
    Manages the state and vectorized updates for all agents in the simulation.
    This version includes all advanced mechanics and is fully optimized.
    """
    def __init__(self, env, initial_agents, fauna_configs):
        self.env = env
        self.fauna_configs = fauna_configs
        
        self.SPECIES_ID = {"Zooplankton": 1, "SmallFish": 2, "Crab": 3}
        
        self.agent_hash = SpatialHash(env.width, env.height, env.depth)
        
        num_agents = len(initial_agents)
        self.positions = np.array([[a.x, a.y, a.z] for a in initial_agents], dtype=float)
        self.energies = np.array([a.energy for a in initial_agents], dtype=float)
        self.species_ids = np.array([self.SPECIES_ID[a.species] for a in initial_agents], dtype=int)
        self.alive_mask = np.ones(num_agents, dtype=bool)
        self.cooldowns = np.zeros(num_agents, dtype=int)
        self.ages = np.zeros(num_agents, dtype=int)
        self.satiation_timers = np.zeros(num_agents, dtype=int)
        self.targets = np.full(num_agents, -1, dtype=int)

    def update(self):
        """The main update loop for all agents, called once per tick."""
        if not self.alive_mask.any(): return

        self._apply_metabolism_and_aging()
        self._eat_plankton()
        self._handle_scavenging()
        self._move_predators()
        self._move_agents()
        self._handle_predation()
        
        # Population control mechanics
        self._handle_carrying_capacity()
        self._handle_disease()
        self._handle_scavenger_overcrowding()
        
        self._handle_deaths()
        self._handle_reproduction()

    def _apply_metabolism_and_aging(self):
        """Applies biome-aware energy costs, aging, and cooldowns."""
        if not self.alive_mask.any(): return
        
        alive_indices = np.where(self.alive_mask)[0]
        alive_positions = self.positions[self.alive_mask].astype(int)
        alive_species_ids = self.species_ids[self.alive_mask]
        
        # Get metabolic modifiers based on agent positions in different biomes
        px, py, pz = alive_positions[:, 0], alive_positions[:, 1], alive_positions[:, 2]
        metabolic_mods = self.env.metabolic_map[px, py, pz]

        # Create local masks for living agents only
        zoo_mask_local = alive_species_ids == self.SPECIES_ID["Zooplankton"]
        fish_mask_local = alive_species_ids == self.SPECIES_ID["SmallFish"]
        crab_mask_local = alive_species_ids == self.SPECIES_ID["Crab"]

        # Apply biome-aware metabolism using boolean masks
        self.energies[alive_indices[zoo_mask_local]] -= self.fauna_configs["Zooplankton"]["metabolic_rate"] * metabolic_mods[zoo_mask_local]
        self.energies[alive_indices[fish_mask_local]] -= self.fauna_configs["SmallFish"]["metabolic_rate"] * metabolic_mods[fish_mask_local]
        self.energies[alive_indices[crab_mask_local]] -= self.fauna_configs["Crab"]["metabolic_rate"] * metabolic_mods[crab_mask_local]
        
        # Vectorized Predator Overcrowding Penalty
        penalty = self.fauna_configs["SmallFish"]["overcrowding_penalty"]
        if penalty > 0 and np.any(fish_mask_local):
            fish_indices_global = alive_indices[fish_mask_local]
            fish_positions_int = alive_positions[fish_mask_local]
            
            unique_cells, cell_indices, counts = np.unique(fish_positions_int, axis=0, return_inverse=True, return_counts=True)
            
            overcrowded_cell_mask = counts > 1
            if np.any(overcrowded_cell_mask):
                overcrowding_counts = counts[cell_indices]
                penalties = (overcrowding_counts - 1) * penalty
                fish_in_overcrowded_cells_mask = np.isin(cell_indices, np.where(overcrowded_cell_mask)[0])
                self.energies[fish_indices_global[fish_in_overcrowded_cells_mask]] -= penalties[fish_in_overcrowded_cells_mask]

        # Decrement timers
        fish_only_mask = self.species_ids == self.SPECIES_ID["SmallFish"]
        self.cooldowns[fish_only_mask] = np.maximum(0, self.cooldowns[fish_only_mask] - 1)
        self.satiation_timers[fish_only_mask] = np.maximum(0, self.satiation_timers[fish_only_mask] - 1)
        self.ages[self.alive_mask] += 1

    def _eat_plankton(self):
        """Handles Zooplankton eating plankton from the environment."""
        zoo_mask = (self.species_ids == self.SPECIES_ID["Zooplankton"]) & self.alive_mask
        if not zoo_mask.any(): return
        positions_int = self.positions[zoo_mask].astype(int)
        px, py, pz = positions_int[:, 0], positions_int[:, 1], positions_int[:, 2]
        plankton_available = self.env.plankton[px, py, pz]
        eating_rate = self.fauna_configs["Zooplankton"]["eating_rate"]
        amount_to_eat = np.minimum(plankton_available, eating_rate)
        self.env.plankton[px, py, pz] -= amount_to_eat
        energy_gain = amount_to_eat * self.fauna_configs["Zooplankton"]["energy_conversion_factor"]
        self.energies[zoo_mask] += energy_gain
    
    def _handle_scavenging(self):
        """Handles Crabs moving towards and eating marine snow on the seafloor."""
        crab_mask = (self.species_ids == self.SPECIES_ID["Crab"]) & self.alive_mask
        if not np.any(crab_mask): return

        crab_indices = np.where(crab_mask)[0]
        crab_positions = self.positions[crab_mask]

        # 1. Vertical movement (vectorized)
        not_on_bottom_mask = crab_positions[:, 2] < self.env.depth - 1
        self.positions[crab_indices[not_on_bottom_mask], 2] += 1

        # 2. Horizontal movement for bottom-dwelling crabs (vectorized)
        on_bottom_mask = ~not_on_bottom_mask
        if np.any(on_bottom_mask):
            bottom_crab_indices = crab_indices[on_bottom_mask]
            bottom_crab_positions = self.positions[bottom_crab_indices].astype(int)
            offsets = np.array([[dx, dy] for dx in [-1, 0, 1] for dy in [-1, 0, 1]])
            neighbor_coords = bottom_crab_positions[:, np.newaxis, :2] + offsets
            neighbor_coords[:, :, 0] = np.clip(neighbor_coords[:, :, 0], 0, self.env.width - 1)
            neighbor_coords[:, :, 1] = np.clip(neighbor_coords[:, :, 1], 0, self.env.height - 1)
            snow_values = self.env.marine_snow[neighbor_coords[:, :, 0], neighbor_coords[:, :, 1], bottom_crab_positions[:, 2, np.newaxis]]
            best_neighbor_indices = np.argmax(snow_values, axis=1)
            best_offsets = offsets[best_neighbor_indices]
            self.positions[bottom_crab_indices, :2] += best_offsets
            self.positions[bottom_crab_indices, 0] %= self.env.width
            self.positions[bottom_crab_indices, 1] %= self.env.height

        # 3. Eating (vectorized)
        final_pos_int = self.positions[crab_mask].astype(int)
        px, py, pz = final_pos_int[:, 0], final_pos_int[:, 1], final_pos_int[:, 2]
        snow_available = self.env.marine_snow[px, py, pz]
        eating_rate = self.fauna_configs["Crab"]["eating_rate"]
        amount_to_eat = np.minimum(snow_available, eating_rate)
        self.env.marine_snow[px, py, pz] -= amount_to_eat
        energy_gain = amount_to_eat * self.fauna_configs["Crab"]["energy_conversion_factor"]
        self.energies[crab_mask] += energy_gain

    def _move_predators(self):
        """Handles predator movement: chase target or wander randomly."""
        fish_mask = (self.species_ids == self.SPECIES_ID["SmallFish"]) & self.alive_mask
        if not fish_mask.any(): return
        
        fish_indices = np.where(fish_mask)[0]
        targets = self.targets[fish_indices]
        
        has_target_mask = targets != -1
        valid_chase_mask = np.zeros_like(has_target_mask, dtype=bool)

        if np.any(has_target_mask):
            chasing_fish_indices = fish_indices[has_target_mask]
            target_indices = targets[has_target_mask]
            
            valid_indices_mask = target_indices < len(self.alive_mask)
            valid_target_indices = target_indices[valid_indices_mask]
            
            if len(valid_target_indices) > 0:
                targets_are_alive = self.alive_mask[valid_target_indices]
                original_indices_of_valid_targets = np.where(has_target_mask)[0][valid_indices_mask]
                valid_chase_mask[original_indices_of_valid_targets] = targets_are_alive
                fish_with_dead_targets = chasing_fish_indices[valid_indices_mask][~targets_are_alive]
                if len(fish_with_dead_targets) > 0:
                    self.targets[fish_with_dead_targets] = -1

            invalid_target_indices_mask = ~valid_indices_mask
            if np.any(invalid_target_indices_mask):
                 self.targets[chasing_fish_indices[invalid_target_indices_mask]] = -1
        
        if np.any(valid_chase_mask):
            chasing_indices = fish_indices[valid_chase_mask]
            target_indices = self.targets[chasing_indices]
            
            delta = self.positions[target_indices] - self.positions[chasing_indices]
            self.positions[chasing_indices] += np.sign(delta)

        no_valid_target_mask = ~valid_chase_mask
        if np.any(no_valid_target_mask):
            wandering_indices = fish_indices[no_valid_target_mask]
            deltas = np.random.randint(-1, 2, size=(len(wandering_indices), 3))
            self.positions[wandering_indices] += deltas

    def _move_agents(self):
        """Handles random movement for all non-predator, non-crab agents."""
        move_mask = (self.species_ids == self.SPECIES_ID["Zooplankton"]) & self.alive_mask
        num_to_move = np.sum(move_mask)
        if num_to_move == 0: return
        deltas = np.random.randint(-1, 2, size=(num_to_move, 3))
        self.positions[move_mask] += deltas
        self.positions[:, 0] %= self.env.width
        self.positions[:, 1] %= self.env.height
        self.positions[:, 2] = np.clip(self.positions[:, 2], 0, self.env.depth - 1)

    def _handle_predation(self):
        """Handles target acquisition and hunting, with robust target validation."""
        fish_mask = (self.species_ids == self.SPECIES_ID["SmallFish"]) & self.alive_mask
        zoo_mask = (self.species_ids == self.SPECIES_ID["Zooplankton"]) & self.alive_mask
        if not fish_mask.any() or not zoo_mask.any(): return

        fish_indices = np.where(fish_mask)[0]

        existing_targets = self.targets[fish_indices]
        has_target_mask = existing_targets != -1
        if np.any(has_target_mask):
            valid_indices_mask = existing_targets[has_target_mask] < len(self.alive_mask)
            if np.any(valid_indices_mask):
                targets_alive = self.alive_mask[existing_targets[has_target_mask][valid_indices_mask]]
                fish_with_dead_targets_mask = ~targets_alive
                fish_with_dead_targets = fish_indices[has_target_mask][valid_indices_mask][fish_with_dead_targets_mask]
                if len(fish_with_dead_targets) > 0:
                    self.targets[fish_with_dead_targets] = -1
        
        needs_target_mask = (self.targets[fish_indices] == -1) & (self.satiation_timers[fish_indices] == 0)
        if np.any(needs_target_mask):
            searching_fish_indices = fish_indices[needs_target_mask]
            self._acquire_targets(searching_fish_indices, zoo_mask)

        final_target_mask = self.targets[fish_indices] != -1
        if not np.any(final_target_mask): return
        hunting_fish_indices = fish_indices[final_target_mask]
        target_indices = self.targets[hunting_fish_indices]
        distances = np.linalg.norm(self.positions[target_indices] - self.positions[hunting_fish_indices], axis=1)
        in_range_mask = distances < self.fauna_configs["SmallFish"]["predation_range"]
        if np.any(in_range_mask):
            self._execute_hunts(hunting_fish_indices[in_range_mask])

    def _acquire_targets(self, searching_fish_indices, zoo_mask):
        """Expensive search for new targets, called only when necessary."""
        zoo_positions = self.positions[zoo_mask]
        zoo_indices = np.where(zoo_mask)[0]
        if len(zoo_indices) == 0: return
        zoo_tree = KDTree(zoo_positions)
        base_vision = self.fauna_configs["SmallFish"]["vision_radius"]
        refuge_vision_mod = self.fauna_configs["SmallFish"]["refuge_vision_modifier"]
        for hunter_idx in searching_fish_indices:
            hunter_pos = self.positions[hunter_idx]
            hunter_pos_int = hunter_pos.astype(int)
            vision_radius = base_vision
            if self.env.refuge_map[hunter_pos_int[0], hunter_pos_int[1], hunter_pos_int[2]] == 1.0:
                vision_radius *= refuge_vision_mod
            biome_id = self.env.biome_map[hunter_pos_int[0], hunter_pos_int[1], hunter_pos_int[2]]
            vision_radius *= BIOME_DATA[biome_id]["vision_modifier"]
            prey_indices_in_vision = zoo_tree.query_ball_point(hunter_pos, r=vision_radius)
            if prey_indices_in_vision:
                distances = np.linalg.norm(zoo_positions[prey_indices_in_vision] - hunter_pos, axis=1)
                closest_prey_local_idx = np.argmin(distances)
                target_global_idx = zoo_indices[prey_indices_in_vision[closest_prey_local_idx]]
                self.targets[hunter_idx] = target_global_idx

    def _execute_hunts(self, hunting_fish_indices):
        """Applies success chance and energy transfer for successful hunts."""
        hunt_success_chance = self.fauna_configs["SmallFish"]["hunt_success_chance"]
        maturity_age = self.fauna_configs["SmallFish"]["maturity_age"]
        juvenile_modifier = self.fauna_configs["SmallFish"]["juvenile_hunt_modifier"]
        successful_hunters = []
        eaten_prey = []
        for hunter_idx in hunting_fish_indices:
            chance = hunt_success_chance
            if self.ages[hunter_idx] < maturity_age:
                chance *= juvenile_modifier
            if random.random() < chance:
                prey_idx = self.targets[hunter_idx]
                if prey_idx not in eaten_prey:
                    successful_hunters.append(hunter_idx)
                    eaten_prey.append(prey_idx)
        if not successful_hunters: return
        final_eaten_prey = np.array(eaten_prey)
        final_hunting_fish = np.array(successful_hunters)
        self.alive_mask[final_eaten_prey] = False
        energy_transfer = self.energies[final_eaten_prey] * self.fauna_configs["SmallFish"]["energy_transfer_efficiency"]
        np.add.at(self.energies, final_hunting_fish, energy_transfer)
        satiation_period = self.fauna_configs["SmallFish"]["satiation_period"]
        self.satiation_timers[final_hunting_fish] = satiation_period
        self.targets[final_hunting_fish] = -1
    
    def _handle_carrying_capacity(self):
        """Applies a chance of death to Zooplankton in overcrowded cells."""
        threshold = self.fauna_configs["Zooplankton"]["carrying_capacity_threshold"]
        starvation_chance = self.fauna_configs["Zooplankton"]["starvation_chance"]
        zoo_mask = (self.species_ids == self.SPECIES_ID["Zooplankton"]) & self.alive_mask
        if not np.any(zoo_mask): return
        zoo_indices_global = np.where(zoo_mask)[0]
        zoo_positions_int = self.positions[zoo_mask].astype(int)
        unique_cells, cell_indices, counts = np.unique(zoo_positions_int, axis=0, return_inverse=True, return_counts=True)
        overcrowded_cell_mask = counts > threshold
        if np.any(overcrowded_cell_mask):
            zoo_in_overcrowded_cells_mask = np.isin(cell_indices, np.where(overcrowded_cell_mask)[0])
            num_to_roll = np.sum(zoo_in_overcrowded_cells_mask)
            random_rolls = np.random.random(size=num_to_roll)
            starvation_mask = random_rolls < starvation_chance
            if np.any(starvation_mask):
                agents_to_die_indices = zoo_indices_global[zoo_in_overcrowded_cells_mask][starvation_mask]
                self.alive_mask[agents_to_die_indices] = False

    def _handle_disease(self):
        """Applies a chance of death to all Zooplankton if the population exceeds the threshold."""
        zoo_config = self.fauna_configs["Zooplankton"]
        threshold = zoo_config.get("disease_threshold", 99999)
        chance = zoo_config.get("disease_chance", 0.0)
        zoo_mask = (self.species_ids == self.SPECIES_ID["Zooplankton"]) & self.alive_mask
        current_zoo_pop = np.sum(zoo_mask)
        if current_zoo_pop > threshold:
            random_rolls = np.random.random(size=current_zoo_pop)
            disease_mask = random_rolls < chance
            zoo_indices = np.where(zoo_mask)[0]
            agents_to_die_indices = zoo_indices[disease_mask]
            if len(agents_to_die_indices) > 0:
                self.alive_mask[agents_to_die_indices] = False

    def _handle_scavenger_overcrowding(self):
        """Applies a chance of death to Crabs in overcrowded cells."""
        crab_config = self.fauna_configs["Crab"]
        threshold = crab_config.get("carrying_capacity_threshold", 99)
        starvation_chance = crab_config.get("starvation_chance", 0.0)

        crab_mask = (self.species_ids == self.SPECIES_ID["Crab"]) & self.alive_mask
        if not np.any(crab_mask): return
        
        crab_indices_global = np.where(crab_mask)[0]
        crab_positions_int = self.positions[crab_mask].astype(int)
        
        unique_cells, cell_indices, counts = np.unique(crab_positions_int, axis=0, return_inverse=True, return_counts=True)
        
        overcrowded_cell_mask = counts > threshold
        if np.any(overcrowded_cell_mask):
            crabs_in_overcrowded_cells_mask = np.isin(cell_indices, np.where(overcrowded_cell_mask)[0])
            
            num_to_roll = np.sum(crabs_in_overcrowded_cells_mask)
            random_rolls = np.random.random(size=num_to_roll)
            starvation_mask = random_rolls < starvation_chance
            
            if np.any(starvation_mask):
                agents_to_die_indices = crab_indices_global[crabs_in_overcrowded_cells_mask][starvation_mask]
                self.alive_mask[agents_to_die_indices] = False

    def _handle_deaths(self):
        """Handles deaths from starvation or old age."""
        starvation_dead = (self.energies <= 0)
        zoo_lifespan = self.fauna_configs["Zooplankton"].get("max_lifespan", 99999)
        fish_lifespan = self.fauna_configs["SmallFish"].get("max_lifespan", 99999)
        crab_lifespan = self.fauna_configs["Crab"].get("max_lifespan", 99999)
        is_zoo = self.species_ids == self.SPECIES_ID["Zooplankton"]
        is_fish = self.species_ids == self.SPECIES_ID["SmallFish"]
        is_crab = self.species_ids == self.SPECIES_ID["Crab"]
        old_age_dead = (is_zoo & (self.ages >= zoo_lifespan)) | (is_fish & (self.ages >= fish_lifespan)) | (is_crab & (self.ages >= crab_lifespan))
        newly_dead_mask = (starvation_dead | old_age_dead) & self.alive_mask
        if newly_dead_mask.any():
            self.alive_mask[newly_dead_mask] = False

    def _handle_reproduction(self):
        """Handles reproduction for all eligible agents."""
        zoo_repro_mask = (self.species_ids == self.SPECIES_ID["Zooplankton"]) & (self.energies > self.fauna_configs["Zooplankton"]["reproduction_threshold"]) & self.alive_mask
        fish_repro_mask = (self.species_ids == self.SPECIES_ID["SmallFish"]) & (self.energies > self.fauna_configs["SmallFish"]["reproduction_threshold"]) & (self.cooldowns == 0) & self.alive_mask
        crab_repro_mask = (self.species_ids == self.SPECIES_ID["Crab"]) & (self.energies > self.fauna_configs["Crab"]["reproduction_threshold"]) & self.alive_mask
        repro_mask = zoo_repro_mask | fish_repro_mask | crab_repro_mask
        if not repro_mask.any(): return
        self.energies[repro_mask] /= 2
        offspring_positions = self.positions[repro_mask]
        offspring_energies = self.energies[repro_mask]
        offspring_species = self.species_ids[repro_mask]
        fish_parents_mask = repro_mask & (self.species_ids == self.SPECIES_ID["SmallFish"])
        self.cooldowns[fish_parents_mask] = self.fauna_configs["SmallFish"]["reproduction_cooldown_period"]
        self.positions = np.vstack([self.positions, offspring_positions])
        self.energies = np.concatenate([self.energies, offspring_energies])
        self.species_ids = np.concatenate([self.species_ids, offspring_species])
        self.cooldowns = np.concatenate([self.cooldowns, np.zeros(len(offspring_positions), dtype=int)])
        self.alive_mask = np.concatenate([self.alive_mask, np.ones(len(offspring_positions), dtype=bool)])
        self.ages = np.concatenate([self.ages, np.zeros(len(offspring_positions), dtype=int)])
        self.satiation_timers = np.concatenate([self.satiation_timers, np.zeros(len(offspring_positions), dtype=int)])
        self.targets = np.concatenate([self.targets, np.full(len(offspring_positions), -1, dtype=int)])

    def get_population_counts(self):
        """Returns the current population count for each species."""
        zoo_pop = np.sum((self.species_ids == self.SPECIES_ID["Zooplankton"]) & self.alive_mask)
        fish_pop = np.sum((self.species_ids == self.SPECIES_ID["SmallFish"]) & self.alive_mask)
        crab_pop = np.sum((self.species_ids == self.SPECIES_ID["Crab"]) & self.alive_mask)
        return zoo_pop, fish_pop, crab_pop

    def cleanup(self):
        """Removes dead agents from the simulation arrays."""
        dead_this_tick_mask = ~self.alive_mask
        if dead_this_tick_mask.any():
            dead_positions = self.positions[dead_this_tick_mask].astype(int)
            sizes = np.array([0, self.fauna_configs["Zooplankton"]["size"], self.fauna_configs["SmallFish"]["size"], self.fauna_configs["Crab"]["size"]])
            dead_species_ids = self.species_ids[dead_this_tick_mask]
            dead_sizes = sizes[dead_species_ids]
            for i in range(len(dead_positions)):
                pos = dead_positions[i]
                self.env.deposit_marine_snow(pos[0], pos[1], pos[2], dead_sizes[i])
        self.positions = self.positions[self.alive_mask]
        self.energies = self.energies[self.alive_mask]
        self.species_ids = self.species_ids[self.alive_mask]
        self.cooldowns = self.cooldowns[self.alive_mask]
        self.ages = self.ages[self.alive_mask]
        self.satiation_timers = self.satiation_timers[self.alive_mask]
        self.targets = self.targets[self.alive_mask]
        self.alive_mask = np.ones(len(self.positions), dtype=bool)
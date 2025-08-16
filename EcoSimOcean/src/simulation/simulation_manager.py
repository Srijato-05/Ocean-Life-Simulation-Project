# src/simulation/simulation_manager.py

import numpy as np
import random
from scipy.spatial import KDTree
from src.utils.spatial_hash import SpatialHash

class SimulationManager:
    """
    Manages the state and vectorized updates for all agents in the simulation.
    This version includes new mechanics for satiation, juvenile predators,
    overcrowding, density-dependent starvation, and disease.
    """
    def __init__(self, env, initial_agents, fauna_configs):
        self.env = env
        self.fauna_configs = fauna_configs
        
        self.SPECIES_ID = {"Zooplankton": 1, "SmallFish": 2}
        
        self.agent_hash = SpatialHash(env.width, env.height, env.depth)
        
        num_agents = len(initial_agents)
        self.positions = np.array([[a.x, a.y, a.z] for a in initial_agents], dtype=float)
        self.energies = np.array([a.energy for a in initial_agents], dtype=float)
        self.species_ids = np.array([self.SPECIES_ID[a.species] for a in initial_agents], dtype=int)
        self.alive_mask = np.ones(num_agents, dtype=bool)
        self.cooldowns = np.zeros(num_agents, dtype=int)
        self.ages = np.zeros(num_agents, dtype=int)
        self.satiation_timers = np.zeros(num_agents, dtype=int)

    def update(self):
        """The main update loop for all agents, called once per tick."""
        if not self.alive_mask.any(): return
        
        self.agent_hash.clear()
        alive_indices = np.where(self.alive_mask)[0]
        for i in alive_indices:
            self.agent_hash.insert(i, self.positions[i])

        self._apply_metabolism_and_aging()
        self._eat_plankton()
        self._move_agents()
        self._handle_predation()
        
        # Population control mechanics
        self._handle_carrying_capacity() # Prey density control
        self._handle_disease()           # Global prey population control
        
        self._handle_deaths()
        self._handle_reproduction()

    def _apply_metabolism_and_aging(self):
        """Applies base energy costs, aging, and cooldowns."""
        zoo_mask = self.species_ids == self.SPECIES_ID["Zooplankton"]
        fish_mask = self.species_ids == self.SPECIES_ID["SmallFish"]
        
        # Base metabolic rates
        self.energies[zoo_mask] -= self.fauna_configs["Zooplankton"]["metabolic_rate"]
        self.energies[fish_mask] -= self.fauna_configs["SmallFish"]["metabolic_rate"]
        
        # Predator Overcrowding Penalty
        penalty = self.fauna_configs["SmallFish"]["overcrowding_penalty"]
        if penalty > 0:
            for cell_key, agent_indices in self.agent_hash.hash.items():
                fish_in_cell_indices = [i for i in agent_indices if self.species_ids[i] == self.SPECIES_ID["SmallFish"]]
                if len(fish_in_cell_indices) > 1:
                    overcrowding_loss = (len(fish_in_cell_indices) - 1) * penalty
                    self.energies[fish_in_cell_indices] -= overcrowding_loss

        # Decrement timers
        self.cooldowns[fish_mask] = np.maximum(0, self.cooldowns[fish_mask] - 1)
        self.satiation_timers[fish_mask] = np.maximum(0, self.satiation_timers[fish_mask] - 1)
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

    def _move_agents(self):
        """Handles random movement for all agents."""
        num_alive = np.sum(self.alive_mask)
        deltas = np.random.randint(-1, 2, size=(num_alive, 3))
        self.positions[self.alive_mask] += deltas
        
        self.positions[:, 0] %= self.env.width
        self.positions[:, 1] %= self.env.height
        self.positions[:, 2] = np.clip(self.positions[:, 2], 0, self.env.depth - 1)

    def _handle_predation(self):
        """Handles SmallFish hunting Zooplankton, including all advanced mechanics."""
        fish_mask = (self.species_ids == self.SPECIES_ID["SmallFish"]) & self.alive_mask & (self.satiation_timers == 0)
        zoo_mask = (self.species_ids == self.SPECIES_ID["Zooplankton"]) & self.alive_mask
        if not fish_mask.any() or not zoo_mask.any(): return

        fish_positions = self.positions[fish_mask]
        zoo_positions = self.positions[zoo_mask]
        zoo_tree = KDTree(zoo_positions)
        
        base_vision = self.fauna_configs["SmallFish"]["vision_radius"]
        vision_modifier = self.fauna_configs["SmallFish"]["refuge_vision_modifier"]
        hunt_success_chance = self.fauna_configs["SmallFish"]["hunt_success_chance"]
        maturity_age = self.fauna_configs["SmallFish"]["maturity_age"]
        juvenile_modifier = self.fauna_configs["SmallFish"]["juvenile_hunt_modifier"]

        successful_hunters = []
        eaten_prey = []

        fish_indices = np.where(fish_mask)[0]
        zoo_indices = np.where(zoo_mask)[0]

        for i, hunter_idx in enumerate(fish_indices):
            hunter_pos = self.positions[hunter_idx]
            
            # Check for refuge zone vision impairment
            vision_radius = base_vision
            hunter_pos_int = hunter_pos.astype(int)
            if self.env.refuge_map[hunter_pos_int[0], hunter_pos_int[1], hunter_pos_int[2]] == 1.0:
                vision_radius *= vision_modifier
            
            prey_indices_in_vision = zoo_tree.query_ball_point(hunter_pos, r=vision_radius)
            if not prey_indices_in_vision:
                continue

            distances = np.linalg.norm(zoo_positions[prey_indices_in_vision] - hunter_pos, axis=1)
            closest_prey_local_idx = np.argmin(distances)
            
            if distances[closest_prey_local_idx] < self.fauna_configs["SmallFish"]["predation_range"]:
                chance = hunt_success_chance
                if self.ages[hunter_idx] < maturity_age:
                    chance *= juvenile_modifier
                
                if random.random() < chance:
                    prey_global_idx = zoo_indices[prey_indices_in_vision[closest_prey_local_idx]]
                    successful_hunters.append(hunter_idx)
                    eaten_prey.append(prey_global_idx)

        if not successful_hunters:
            return

        unique_prey, unique_indices = np.unique(eaten_prey, return_index=True)
        final_hunting_fish = np.array(successful_hunters)[unique_indices]
        final_eaten_prey = unique_prey

        self.alive_mask[final_eaten_prey] = False
        energy_transfer = self.energies[final_eaten_prey] * self.fauna_configs["SmallFish"]["energy_transfer_efficiency"]
        np.add.at(self.energies, final_hunting_fish, energy_transfer)
        
        satiation_period = self.fauna_configs["SmallFish"]["satiation_period"]
        self.satiation_timers[final_hunting_fish] = satiation_period

    def _handle_carrying_capacity(self):
        """Applies a chance of death to Zooplankton in overcrowded cells."""
        threshold = self.fauna_configs["Zooplankton"]["carrying_capacity_threshold"]
        starvation_chance = self.fauna_configs["Zooplankton"]["starvation_chance"]
        
        for cell_key, agent_indices in self.agent_hash.hash.items():
            zooplankton_in_cell_indices = [i for i in agent_indices if self.species_ids[i] == self.SPECIES_ID["Zooplankton"]]
            
            if len(zooplankton_in_cell_indices) > threshold:
                random_rolls = np.random.random(size=len(zooplankton_in_cell_indices))
                starvation_mask = random_rolls < starvation_chance
                
                agents_to_die_indices = np.array(zooplankton_in_cell_indices)[starvation_mask]
                
                if len(agents_to_die_indices) > 0:
                    self.alive_mask[agents_to_die_indices] = False

    # --- HIGHLIGHT: New method for handling density-dependent disease ---
    def _handle_disease(self):
        """
        Applies a chance of death to all Zooplankton if the total population
        exceeds the disease threshold.
        """
        zoo_config = self.fauna_configs["Zooplankton"]
        threshold = zoo_config.get("disease_threshold", 9999)
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

    def _handle_deaths(self):
        """Handles deaths from starvation (energy <= 0)."""
        newly_dead = (self.energies <= 0) & self.alive_mask
        if newly_dead.any():
            self.alive_mask[newly_dead] = False

    def _handle_reproduction(self):
        """Handles reproduction for all eligible agents."""
        zoo_repro_mask = (self.species_ids == self.SPECIES_ID["Zooplankton"]) & (self.energies > self.fauna_configs["Zooplankton"]["reproduction_threshold"]) & self.alive_mask
        fish_repro_mask = (self.species_ids == self.SPECIES_ID["SmallFish"]) & (self.energies > self.fauna_configs["SmallFish"]["reproduction_threshold"]) & (self.cooldowns == 0) & self.alive_mask
        repro_mask = zoo_repro_mask | fish_repro_mask
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

    def get_population_counts(self):
        """Returns the current population count for each species."""
        zoo_pop = np.sum((self.species_ids == self.SPECIES_ID["Zooplankton"]) & self.alive_mask)
        fish_pop = np.sum((self.species_ids == self.SPECIES_ID["SmallFish"]) & self.alive_mask)
        return zoo_pop, fish_pop

    def cleanup(self):
        """Removes dead agents from the simulation arrays."""
        dead_this_tick_mask = ~self.alive_mask
        if dead_this_tick_mask.any():
            dead_positions = self.positions[dead_this_tick_mask].astype(int)
            zoo_size = self.fauna_configs["Zooplankton"]["size"]
            fish_size = self.fauna_configs["SmallFish"]["size"]
            for i in range(len(dead_positions)):
                pos = dead_positions[i]
                species_id = self.species_ids[dead_this_tick_mask][i]
                size = zoo_size if species_id == self.SPECIES_ID["Zooplankton"] else fish_size
                self.env.deposit_marine_snow(pos[0], pos[1], pos[2], size)

        self.positions = self.positions[self.alive_mask]
        self.energies = self.energies[self.alive_mask]
        self.species_ids = self.species_ids[self.alive_mask]
        self.cooldowns = self.cooldowns[self.alive_mask]
        self.ages = self.ages[self.alive_mask]
        self.satiation_timers = self.satiation_timers[self.alive_mask]
        
        self.alive_mask = np.ones(len(self.positions), dtype=bool)
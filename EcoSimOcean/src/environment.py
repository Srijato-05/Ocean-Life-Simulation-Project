# src/environment.py

import numpy as np
from scipy.ndimage import convolve
from src.biome import BIOME_DATA

class Environment:
    """
    Manages the 3D grid environment. This version uses configurable parameters
    for world generation, making it more flexible and scalable.
    """
    def __init__(self, width, height, depth, config):
        self.width = width
        self.height = height
        self.depth = depth
        self.config = config
        self.env_gen_config = config.get("environment_generation", {})

        self.plankton = np.full((width, height, depth), config.get("initial_food_density", 0.8))
        self.marine_snow = np.zeros((width, height, depth))
        
        self.snow_decay = self.config.get("marine_snow_decay_rate", 0.99)
        self.snow_sinking_factor = self.config.get("marine_snow_sinking_factor", 0.9)
        self.snow_to_plankton = self.config.get("snow_to_plankton_conversion", 0.01)

        self.biome_map = self._create_biome_map()
        self.base_nutrient_map = self._create_modifier_map("nutrient_factor")
        self.nutrient_map = self.base_nutrient_map.copy()
        self.metabolic_map = self._create_modifier_map("metabolic_modifier")
        
        self.refuge_map = self._create_refuge_map()
        self.sunlight = self._create_sunlight_gradient()
        
        self.disease_risk_map = np.ones((width, height, depth))
        self.current_event = "none"
        self.event_timer = 0
        
        self.diffusion_kernel = np.array([[[0, 0, 0], [0, 1, 0], [0, 0, 0]],
                                          [[0, 1, 0], [1, -6, 1], [0, 1, 0]],
                                          [[0, 0, 0], [0, 1, 0], [0, 0, 0]]]) * self.config.get("plankton_diffusion_rate", 0.05)

    def _create_biome_map(self):
        biome_map = np.zeros((self.width, self.height, self.depth), dtype=int)
        biome_map.fill(0) # Default to OpenOcean

        # --- UPDATED: Use config values ---
        deep_sea_depth = int(self.depth * self.env_gen_config.get("deep_sea_depth_fraction", 0.66))
        polar_width = int(self.width * self.env_gen_config.get("polar_zone_width_fraction", 0.25))
        num_reefs = self.env_gen_config.get("num_coral_reefs", 3)
        reef_depth = int(self.depth * self.env_gen_config.get("reef_max_depth_fraction", 0.2))

        biome_map[:, :, deep_sea_depth:] = 1 # DeepSea
        biome_map[0:polar_width, :, :] = 2 # PolarSea
        
        for _ in range(num_reefs):
            reef_x = np.random.randint(polar_width + 10, self.width - 10)
            reef_y = np.random.randint(0, self.height - 10)
            biome_map[reef_x-5:reef_x+5, reef_y-5:reef_y+5, 0:reef_depth] = 3 # CoralReef
            
        return biome_map

    def _create_modifier_map(self, factor_name):
        modifier_map = np.ones((self.width, self.height, self.depth))
        for biome_id, properties in BIOME_DATA.items():
            modifier_map[self.biome_map == biome_id] = properties[factor_name]
        return modifier_map

    def _create_refuge_map(self):
        refuge_map = np.zeros((self.width, self.height, self.depth), dtype=bool)
        # --- UPDATED: Use config values ---
        num_refuges = self.env_gen_config.get("num_refuges", 20)
        refuge_size = self.env_gen_config.get("refuge_size", 2)

        refuge_xs = np.random.randint(0, self.width, num_refuges)
        refuge_ys = np.random.randint(0, self.height, num_refuges)
        for x, y in zip(refuge_xs, refuge_ys):
            x_start, x_end = max(0, x - refuge_size), min(self.width, x + refuge_size)
            y_start, y_end = max(0, y - refuge_size), min(self.height, y + refuge_size)
            refuge_map[x_start:x_end, y_start:y_end, :] = True
            
        return refuge_map

    def _create_sunlight_gradient(self):
        z_sunlight = np.exp(-np.arange(self.depth) * 0.5)
        sunlight = np.zeros((self.width, self.height, self.depth))
        sunlight[:, :, :] = z_sunlight
        return sunlight

    def update(self):
        self._update_dynamic_events()
        self._update_plankton_dynamics()
        self._update_marine_snow()

    def _update_dynamic_events(self):
        if self.event_timer > 0:
            self.event_timer -= 1
            if self.event_timer == 0:
                self.nutrient_map = self.base_nutrient_map.copy()
                self.disease_risk_map.fill(1.0)
                self.current_event = "none"
        
        if self.current_event == "none" and np.random.random() < self.config.get("event_chance", 0.01):
            event_type = np.random.choice(["bloom", "disease"])
            self.event_timer = self.config.get("event_duration", 50)
            
            if event_type == "bloom":
                self.current_event = "Plankton Bloom"
                bloom_modifier = self.config.get("plankton_bloom_modifier", 2.0)
                self.nutrient_map[self.biome_map == 0] *= bloom_modifier
            
            elif event_type == "disease":
                self.current_event = "Disease Zone"
                disease_modifier = self.config.get("disease_zone_modifier", 1.5)
                self.disease_risk_map[self.biome_map == 3] *= disease_modifier

    def _update_plankton_dynamics(self):
        diffusion = convolve(self.plankton, self.diffusion_kernel, mode='wrap')
        self.plankton += diffusion
        max_growth = self.config.get("plankton_max_growth_rate", 0.1)
        growth = self.plankton * (1 - self.plankton) * self.sunlight * max_growth * self.nutrient_map
        self.plankton += growth
        np.clip(self.plankton, 0, 1, out=self.plankton)

    def _update_marine_snow(self):
        sinking_snow = np.roll(self.marine_snow, 1, axis=2) * self.snow_sinking_factor
        sinking_snow[:, :, 0] = 0
        self.marine_snow = sinking_snow
        self.plankton += self.marine_snow * self.snow_to_plankton
        self.marine_snow *= self.snow_decay

    def deposit_marine_snow(self, x, y, z, amount):
        if 0 <= x < self.width and 0 <= y < self.height and 0 <= z < self.depth:
            self.marine_snow[int(x), int(y), int(z)] += amount
# src/environment.py

import numpy as np
from scipy.ndimage import convolve
from src.biome import BIOME_DATA

class Environment:
    """
    Manages the 3D grid environment, including plankton dynamics, static refuge
    zones, and a diverse map of biomes.
    """
    def __init__(self, width, height, depth, config):
        self.width = width
        self.height = height
        self.depth = depth
        self.config = config

        self.plankton = np.random.rand(width, height, depth) * config.get("initial_food_density", 0.5)
        self.marine_snow = np.zeros((width, height, depth))

        # --- HIGHLIGHT: Generate biome, nutrient, and metabolic maps ---
        self.biome_map = self._create_biome_map()
        self.nutrient_map = self._create_modifier_map("nutrient_factor")
        self.metabolic_map = self._create_modifier_map("metabolic_modifier")

        self.refuge_map = self._create_refuge_map()
        self.sunlight = self._create_sunlight_gradient()

        self.diffusion_kernel = np.array([[[0, 0, 0], [0, 1, 0], [0, 0, 0]],
                                          [[0, 1, 0], [1, -6, 1], [0, 1, 0]],
                                          [[0, 0, 0], [0, 1, 0], [0, 0, 0]]]) * self.config.get("plankton_diffusion_rate", 0.05)

    # --- HIGHLIGHT: New method to create a diverse biome map ---
    def _create_biome_map(self):
        """Creates a 3D map of various biomes."""
        biome_map = np.zeros((self.width, self.height, self.depth), dtype=int)

        # Default to Open Ocean (ID 0)
        biome_map.fill(0)

        # Deep Sea (ID 1) below a certain depth
        deep_sea_depth = self.depth * 2 // 3
        biome_map[:, :, deep_sea_depth:] = 1

        # Polar Sea (ID 2) along one side of the world
        polar_width = self.width // 4
        biome_map[0:polar_width, :, :] = 2

        # Coral Reefs (ID 3) in a few shallow, non-polar coastal areas
        reef_depth = self.depth // 5
        for _ in range(3): # Create 3 reefs
            reef_x = np.random.randint(polar_width + 10, self.width - 10)
            reef_y = np.random.randint(0, self.height - 10)
            biome_map[reef_x-5:reef_x+5, reef_y-5:reef_y+5, 0:reef_depth] = 3

        return biome_map

    # --- HIGHLIGHT: Generic method to create modifier maps ---
    def _create_modifier_map(self, factor_name):
        """Creates a 3D map of a given modifier from the biome map."""
        modifier_map = np.ones((self.width, self.height, self.depth))
        for biome_id, properties in BIOME_DATA.items():
            modifier_map[self.biome_map == biome_id] = properties[factor_name]
        return modifier_map

    def _create_refuge_map(self):
        """Creates a 3D map of refuge zones."""
        refuge_map = np.zeros((self.width, self.height, self.depth))
        num_refuges = 15
        refuge_xs = np.random.randint(0, self.width, num_refuges)
        refuge_ys = np.random.randint(0, self.height, num_refuges)
        refuge_map[refuge_xs, refuge_ys, :] = 1.0
        return refuge_map

    def _create_sunlight_gradient(self):
        """Creates a sunlight array that decreases exponentially with depth."""
        z_sunlight = np.exp(-np.arange(self.depth) * 0.5)
        sunlight = np.zeros((self.width, self.height, self.depth))
        sunlight[:, :, :] = z_sunlight
        return sunlight

    def update(self):
        """Updates the state of the environment for one time step."""
        self._update_plankton_dynamics()
        self._update_marine_snow()

    def _update_plankton_dynamics(self):
        """Updates plankton distribution, now affected by biome nutrients."""
        diffusion = convolve(self.plankton, self.diffusion_kernel, mode='wrap')
        self.plankton += diffusion

        max_growth = self.config.get("plankton_max_growth_rate", 0.1)
        growth = self.plankton * (1 - self.plankton) * self.sunlight * max_growth * self.nutrient_map
        self.plankton += growth

        np.clip(self.plankton, 0, 1, out=self.plankton)

    def _update_marine_snow(self):
        """Simulates the sinking and decay of marine snow."""
        sinking_snow = np.roll(self.marine_snow, 1, axis=2) * 0.9
        sinking_snow[:, :, 0] = 0
        self.marine_snow = sinking_snow

        self.plankton += self.marine_snow * 0.01
        self.marine_snow *= 0.99

    def deposit_marine_snow(self, x, y, z, amount):
        """Deposits marine snow at a specific location."""
        if 0 <= x < self.width and 0 <= y < self.height and 0 <= z < self.depth:
            self.marine_snow[x, y, z] += amount
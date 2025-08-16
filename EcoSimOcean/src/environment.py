# src/environment.py

import numpy as np
from scipy.ndimage import convolve

class Environment:
    """
    Manages the 3D grid environment, including plankton dynamics based on
    Cellular Automata rules and static refuge zones.
    """
    def __init__(self, width, height, depth, config):
        """
        Initializes the environment.

        Args:
            width (int): The width of the grid.
            height (int): The height of the grid.
            depth (int): The depth of the grid.
            config (dict): The simulation configuration dictionary.
        """
        self.width = width
        self.height = height
        self.depth = depth
        self.config = config
        
        self.plankton = np.random.rand(width, height, depth) * config.get("initial_food_density", 0.5)
        self.marine_snow = np.zeros((width, height, depth))
        
        # --- HIGHLIGHT: Create the refuge map ---
        self.refuge_map = self._create_refuge_map()
        
        self.sunlight = self._create_sunlight_gradient()
        
        self.diffusion_kernel = np.array([[[0, 0, 0],
                                           [0, 1, 0],
                                           [0, 0, 0]],
                                          [[0, 1, 0],
                                           [1, -6, 1],
                                           [0, 1, 0]],
                                          [[0, 0, 0],
                                           [0, 1, 0],
                                           [0, 0, 0]]]) * self.config.get("plankton_diffusion_rate", 0.05)

    # --- HIGHLIGHT: New method to create refuge zones ---
    def _create_refuge_map(self):
        """Creates a 3D map of refuge zones where predator vision is impaired."""
        refuge_map = np.zeros((self.width, self.height, self.depth))
        # Create 15 vertical columns of refuge as an example
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
        """
        Updates the plankton distribution based on Cellular Automata rules.
        """
        # 1. Diffusion Step
        diffusion = convolve(self.plankton, self.diffusion_kernel, mode='wrap')
        self.plankton += diffusion
        
        # 2. Logistic Growth Step
        max_growth = self.config.get("plankton_max_growth_rate", 0.1)
        growth = self.plankton * (1 - self.plankton) * self.sunlight * max_growth
        self.plankton += growth
        
        np.clip(self.plankton, 0, 1, out=self.plankton)

    def _update_marine_snow(self):
        """Simulates the sinking and decay of marine snow."""
        sinking_snow = np.roll(self.marine_snow, 1, axis=2) * 0.9
        sinking_snow[:, :, 0] = 0 # No snow falls from above the surface
        self.marine_snow = sinking_snow
        
        self.plankton += self.marine_snow * 0.01
        self.marine_snow *= 0.99

    def deposit_marine_snow(self, x, y, z, amount):
        """Deposits marine snow at a specific location."""
        if 0 <= x < self.width and 0 <= y < self.height and 0 <= z < self.depth:
            self.marine_snow[x, y, z] += amount
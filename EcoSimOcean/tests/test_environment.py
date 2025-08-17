# tests/test_environment.py

import unittest
import numpy as np
import os
import sys

# Add the project root to the Python path to allow imports from src
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from src.environment import Environment

class TestEnvironment(unittest.TestCase):
    """
    Unit tests for the Environment class to verify core Phase 1 functionality.
    """

    def setUp(self):
        """
        This method is called before each test function.
        It creates a standard environment instance for testing.
        """
        self.width = 10
        self.height = 10
        self.depth = 5
        self.env = Environment(width=self.width, height=self.height, depth=self.depth)
        print(f"\n--- Running test: {self._testMethodName} ---")

    def test_initialization(self):
        """
        Test 1: Verify that the environment and its layers are created with the correct dimensions.
        """
        self.assertEqual(self.env.width, self.width)
        self.assertEqual(self.env.height, self.height)
        self.assertEqual(self.env.depth, self.depth)
        
        expected_shape = (self.width, self.height, self.depth)
        self.assertEqual(self.env.food.shape, expected_shape)
        self.assertEqual(self.env.co2.shape, expected_shape)
        self.assertEqual(self.env.marine_snow.shape, expected_shape)
        self.assertEqual(self.env.sunlight.shape, expected_shape)
        self.assertEqual(self.env.temperature.shape, expected_shape)
        self.assertEqual(self.env.pressure.shape, expected_shape)
        self.assertEqual(self.env.biome.shape, expected_shape)
        print("✅ Initialization dimensions are correct.")

    def test_environmental_gradients(self):
        """
        Test 2: Check if environmental layers correctly change with depth (z-axis).
        """
        # Sunlight should decrease with depth
        sunlight_surface = self.env.sunlight[0, 0, 0]
        sunlight_deep = self.env.sunlight[0, 0, self.depth - 1]
        self.assertGreater(sunlight_surface, sunlight_deep)
        print(f"✅ Sunlight decreases with depth ({sunlight_surface:.2f} > {sunlight_deep:.2f}).")

        # Temperature should decrease with depth
        temp_surface = self.env.temperature[0, 0, 0]
        temp_deep = self.env.temperature[0, 0, self.depth - 1]
        self.assertGreater(temp_surface, temp_deep)
        print(f"✅ Temperature decreases with depth ({temp_surface:.2f} > {temp_deep:.2f}).")

        # Pressure should increase with depth
        pressure_surface = self.env.pressure[0, 0, 0]
        pressure_deep = self.env.pressure[0, 0, self.depth - 1]
        self.assertLess(pressure_surface, pressure_deep)
        print(f"✅ Pressure increases with depth ({pressure_surface:.2f} < {pressure_deep:.2f}).")

    def test_biome_classification(self):
        """
        Test 3: Verify that the biome classification logic is working.
        """
        # At z=0, the biome should be 'surface'
        surface_biome = self.env.get_biome_at(0, 0, 0)
        self.assertEqual(surface_biome, "surface")
        print(f"✅ Biome at z=0 is correctly identified as '{surface_biome}'.")

        # At max depth, the biome should be 'deep_sea'
        deep_biome = self.env.get_biome_at(0, 0, self.depth - 1)
        self.assertEqual(deep_biome, "deep_sea")
        print(f"✅ Biome at z={self.depth - 1} is correctly identified as '{deep_biome}'.")

    def test_marine_snow_sinking(self):
        """
        Test 4: Ensure that marine snow correctly sinks from one layer to the one below it.
        """
        # Deposit snow at a specific location in a middle layer
        x, y, z = 5, 5, 2
        initial_amount = 10.0
        self.env.deposit_marine_snow(x, y, z, initial_amount)
        self.assertEqual(self.env.marine_snow[x, y, z], initial_amount)

        # Run one environment update
        self.env.update()

        # After the update, half the snow should have moved to the layer below (z+1)
        # and half should remain.
        expected_remaining_at_z = initial_amount * 0.5
        expected_sunk_to_z_plus_1 = initial_amount * 0.5

        self.assertAlmostEqual(self.env.marine_snow[x, y, z], expected_remaining_at_z)
        self.assertAlmostEqual(self.env.marine_snow[x, y, z + 1], expected_sunk_to_z_plus_1)
        print("✅ Marine snow sinking mechanism is working correctly.")

# This allows running the tests directly from the command line
if __name__ == '__main__':
    unittest.main()
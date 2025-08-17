# src/utils/visualizer.py

import matplotlib.pyplot as plt
from src.config import BIOME_TYPES

biome_colors = {
    'sunlit': 0,
    'twilight': 1,
    'abyssal': 2,
    'trench': 3,
    'open_ocean': 4,
    'deep_ocean': 5
}

def visualize_biome_slice(env, z):
    biome_grid = [[biome_colors.get(env.get_biome_at(x, y, z), -1)
                   for x in range(env.width)]
                   for y in range(env.height)]
    plt.imshow(biome_grid, cmap='tab10')
    plt.title(f"Biome map at depth z={z}")
    plt.colorbar(ticks=range(len(biome_colors)), label="Biome Index")
    plt.clim(-1, len(biome_colors) - 1)
    plt.show()

def visualize_layer_slice(env, layer_name, z):
    layer = getattr(env, layer_name)
    if layer is None:
        raise ValueError(f"Layer '{layer_name}' does not exist in the environment")
    layer_grid = [[layer[x][y][z] for x in range(env.width)]
                  for y in range(env.height)]
    plt.imshow(layer_grid, cmap='viridis')
    plt.title(f"{layer_name.capitalize()} at depth z={z}")
    plt.colorbar(label=layer_name)
    plt.show()

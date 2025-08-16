# src/config.py

# 3D ocean grid dimensions
GRID_WIDTH = 50
GRID_HEIGHT = 50
GRID_DEPTH = 10  # Supports vertical layers (z-axis)

# Default biome for environment initialization
DEFAULT_BIOME = "open_ocean"

# Environmental profiles by biome type
BIOME_TYPES = {
    "open_ocean": {
        "sunlight": lambda z: max(0, 1.0 - z / 10),
        "temperature": lambda z: 25 - z * 1.5,
        "pressure": lambda z: 1 + z * 0.5,
        "co2": lambda z: 0.03 + z * 0.01  # mol fraction or % depending on scale
    },
    "deep_sea": {
        "sunlight": lambda z: 0.0,
        "temperature": lambda z: 4,
        "pressure": lambda z: 10 + z * 1.0,
        "co2": lambda z: 0.06 + z * 0.01
    },
    "coral_reef": {
        "sunlight": lambda z: max(0.2, 1.0 - z / 5),
        "temperature": lambda z: 26 - z,
        "pressure": lambda z: 1 + z * 0.3,
        "co2": lambda z: 0.025 + z * 0.005
    }
}

# Optional 3D biome map (e.g., if you want per-cell biome types in future)
# For now, this remains unused until Phase 2 or 3
BIOME_MAP = None

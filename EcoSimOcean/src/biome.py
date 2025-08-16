# src/biome.py

# Biome properties and effects can be expanded here

BIOME_RULES = {
    "coastal": {
        "max_agents": 100,
        "nutrient_factor": 1.2
    },
    "open_ocean": {
        "max_agents": 70,
        "nutrient_factor": 0.9
    },
    "deep_ocean": {
        "max_agents": 50,
        "nutrient_factor": 0.4
    },
    "unknown": {
        "max_agents": 10,
        "nutrient_factor": 0.1
    }
}

def get_biome_rules(biome_name):
    return BIOME_RULES.get(biome_name, BIOME_RULES["unknown"])

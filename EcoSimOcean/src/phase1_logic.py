import random
import csv
import matplotlib.pyplot as plt
import numpy as np
import json
import os

# Assuming this script is run from a 'scripts' directory, adjust path to import from 'src'
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from src.environment import Environment
from src.utils.config_loader import load_sim_config

def print_environment_slice(env, z):
    """Prints a text-based slice of the biome map."""
    print(f"Slice of Biome Map at depth z={z}: (0:OpenOcean, 1:DeepSea, 2:Polar, 3:Reef)")
    for y in range(env.height):
        row = " ".join([str(env.biome_map[x, y, z]) for x in range(env.width)])
        print(row)

def run_phase1_simulation(ticks=10):
    """Runs a simple simulation to observe environment dynamics without agents."""
    sim_config = load_sim_config()
    if not sim_config:
        print("Failed to load simulation config. Aborting.")
        return

    env = Environment(
        sim_config['grid_width'],
        sim_config['grid_height'],
        sim_config['grid_depth'],
        sim_config
    )
    
    print_environment_slice(env, z=0)

    summary = []

    for tick in range(ticks):
        print(f"\n🔄 Tick {tick+1}/{ticks}")
        env.update()

        # Randomly deposit some marine snow to observe sinking
        for _ in range(5):
            x = random.randint(0, env.width - 1)
            y = random.randint(0, env.height - 1)
            z = random.randint(0, env.depth - 1)
            env.deposit_marine_snow(x, y, z, random.uniform(0.05, 0.5))

        # Inspect at one random coordinate per tick
        rx, ry, rz = random.randint(0, env.width - 1), random.randint(0, env.height - 1), random.randint(0, env.depth - 1)
        biome_id = env.biome_map[rx, ry, rz]
        print(f"📍 Inspecting ({rx}, {ry}, {rz}) - Biome ID: {biome_id}")
        print(f"  Plankton: {env.plankton[rx, ry, rz]:.2f}, Snow: {env.marine_snow[rx, ry, rz]:.2f}")
        print(f"  Sunlight: {env.sunlight[rx, ry, rz]:.2f}, Nutrient Mod: {env.nutrient_map[rx, ry, rz]:.2f}")

        total_plankton = env.plankton.sum()
        total_snow = env.marine_snow.sum()

        print(f"📊 Totals -> Plankton: {total_plankton:.2f}, Marine Snow: {total_snow:.2f}")

        summary.append({
            "tick": tick + 1,
            "total_plankton": round(total_plankton, 2),
            "total_marine_snow": round(total_snow, 2),
        })
    
    export_summary_to_csv(summary)
    plot_summary(summary)

    print("\n✅ Simulation finished. Exported to:")
    print("- environment_tick_summary.csv")
    print("- plots saved as PNGs")

def export_summary_to_csv(summary):
    """Exports the simulation summary to a CSV file."""
    filename = "environment_tick_summary.csv"
    fieldnames = ["tick", "total_plankton", "total_marine_snow"]

    with open(filename, mode="w", newline="") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(summary)

def plot_summary(summary):
    """Plots the total plankton and marine snow over time."""
    ticks = [entry["tick"] for entry in summary]
    plankton = [entry["total_plankton"] for entry in summary]
    snow = [entry["total_marine_snow"] for entry in summary]

    # Plot Plankton
    plt.figure()
    plt.plot(ticks, plankton, marker="o")
    plt.title("Total Plankton Over Time")
    plt.xlabel("Tick")
    plt.ylabel("Total Plankton")
    plt.grid(True)
    plt.savefig("plot_total_plankton.png")
    plt.close()

    # Plot Marine Snow
    plt.figure()
    plt.plot(ticks, snow, marker="o", color="orange")
    plt.title("Marine Snow Over Time")
    plt.xlabel("Tick")
    plt.ylabel("Total Marine Snow")
    plt.grid(True)
    plt.savefig("plot_marine_snow.png")
    plt.close()
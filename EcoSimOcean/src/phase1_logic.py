import random
import csv
import matplotlib.pyplot as plt
from src.environment import Environment

def print_environment_slice(env, z):
    print(f"Slice at depth z={z}:")
    for y in range(env.height):
        row = ""
        for x in range(env.width):
            biome = env.get_biome_at(x, y, z)
            if biome == "coastal":
                row += "C "
            elif biome == "open_ocean":
                row += "O "
            elif biome == "deep_ocean":
                row += "D "
            else:
                row += ". "
        print(row)

def run_phase1_simulation(ticks=10):
    env = Environment()
    z = 0
    print_environment_slice(env, z)

    summary = []

    for tick in range(ticks):
        print(f"\n🔄 Tick {tick+1}/{ticks}")
        env.update()

        for _ in range(5):
            x = random.randint(0, env.width - 1)
            y = random.randint(0, env.height - 1)
            z = random.randint(0, env.depth - 1)
            env.add_food(x, y, z, random.uniform(0.1, 1.0))
            env.deposit_marine_snow(x, y, z, random.uniform(0.05, 0.5))

        # Inspect at one random coordinate per tick
        rx, ry, rz = random.randint(0, env.width - 1), random.randint(0, env.height - 1), random.randint(0, env.depth - 1)
        print(f"📍 Inspecting ({rx}, {ry}, {rz}) - Biome: {env.get_biome_at(rx, ry, rz)}")
        print(f"  Food: {env.food[rx, ry, rz]:.2f}, Snow: {env.marine_snow[rx, ry, rz]:.2f}, CO₂: {env.co2[rx, ry, rz]:.2f}")
        print(f"  Sunlight: {env.sunlight[rx, ry, rz]:.2f}, Temp: {env.temperature[rx, ry, rz]:.2f}, Pressure: {env.pressure[rx, ry, rz]:.2f}")

        total_food = env.food.sum()
        total_snow = env.marine_snow.sum()
        total_co2 = env.get_total_co2()

        print(f"📊 Totals -> Food: {total_food:.2f}, Marine Snow: {total_snow:.2f}, CO₂: {total_co2:.2f}")

        summary.append({
            "tick": tick + 1,
            "total_food": round(total_food, 2),
            "total_marine_snow": round(total_snow, 2),
            "total_co2": round(total_co2, 2)
        })

    env.export_state()
    export_summary_to_csv(summary)
    plot_summary(summary)

    print("\n✅ Simulation finished. Exported to:")
    print("- environment_export.json")
    print("- environment_tick_summary.csv")
    print("- plots saved as PNGs")

def export_summary_to_csv(summary):
    filename = "environment_tick_summary.csv"
    fieldnames = ["tick", "total_food", "total_marine_snow", "total_co2"]

    with open(filename, mode="w", newline="") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(summary)

def plot_summary(summary):
    ticks = [entry["tick"] for entry in summary]
    food = [entry["total_food"] for entry in summary]
    snow = [entry["total_marine_snow"] for entry in summary]
    co2 = [entry["total_co2"] for entry in summary]

    # Plot Food
    plt.figure()
    plt.plot(ticks, food, marker="o")
    plt.title("Total Food Over Time")
    plt.xlabel("Tick")
    plt.ylabel("Total Food")
    plt.grid(True)
    plt.savefig("plot_total_food.png")
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

    # Plot CO₂
    plt.figure()
    plt.plot(ticks, co2, marker="o", color="green")
    plt.title("CO₂ Levels Over Time")
    plt.xlabel("Tick")
    plt.ylabel("Total CO₂")
    plt.grid(True)
    plt.savefig("plot_total_co2.png")
    plt.close()

# 📦 Test Function (Optional)
def run_phase1_test():
    env = Environment(width=10, height=10, depth=5)

    # Add some food and marine snow at specific locations
    env.add_food(2, 2, 0, 10.0)
    env.deposit_marine_snow(2, 2, 1, 5.0)

    print("\n--- Environment Initialized ---")
    x, y, z = random.randint(0, 9), random.randint(0, 9), random.randint(0, 4)
    print(f"\nInspecting environment at random position ({x}, {y}, {z}):")
    print("Biome:", env.get_biome_at(x, y, z))
    print("Sunlight:", round(env.sunlight[x, y, z], 2))
    print("Temperature:", round(env.temperature[x, y, z], 2))
    print("Pressure:", round(env.pressure[x, y, z], 2))
    print("Food:", round(env.food[x, y, z], 2))
    print("CO2:", round(env.co2[x, y, z], 2))
    print("Marine Snow:", round(env.marine_snow[x, y, z], 2))

    # Simulate a few steps
    for step in range(3):
        env.update()
        print(f"\nAfter step {step + 1}: Total CO2 = {round(env.get_total_co2(), 2)}")

    # Export the final environment state to JSON
    env.export_state()
    print("\nEnvironment state exported to environment_export.json")

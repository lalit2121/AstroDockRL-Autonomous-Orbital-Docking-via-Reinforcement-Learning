import os
import sys
import json
import numpy as np
import matplotlib.pyplot as plt
from stable_baselines3 import PPO

# Ensure Python can find your custom environment files
sys.path.append(r"C:\Users\Lkd\Desktop\proj\ML")
from docking_env import OrbitalDockingEnv
from hyperparameters import ENV_CONFIG

def plot_everything():
    # ─── 1. PLOT TRAINING METRICS ────────────────────────────────────────
    file_path = r"C:\Users\Lkd\Desktop\proj\ML\logs\docking_metrics.json"

    with open(file_path, "r") as f:
        data = json.load(f)

    metrics = [
        ("success_rates", "Success Rate", "blue"),
        ("avg_fuels", "Avg Fuel Used", "green"),
        ("avg_distances", "Avg Final Distance", "red"),
        ("avg_lengths", "Avg Episode Length", "purple")
    ]

    fig, axs = plt.subplots(len(metrics), 1, figsize=(10, 3 * len(metrics)), sharex=True)
    if len(metrics) == 1:
        axs = [axs]

    for i, (key, label, color) in enumerate(metrics):
        axs[i].plot(
            data["timesteps"][:len(data[key])], 
            data[key],
            color=color, marker='o', markersize=3, linewidth=1.5
        )
        axs[i].set_ylabel(label, fontsize=10, fontweight='bold')
        axs[i].grid(True, linestyle='--', alpha=0.6)

    axs[-1].set_xlabel("Timesteps", fontsize=12, fontweight='bold')
    plt.suptitle("PPO Orbital Docking Training Metrics", fontsize=14, fontweight='bold')
    plt.tight_layout(rect=[0, 0.03, 1, 0.97])
    
    # ─── 2. GENERATE AND PLOT 3D/2D TRAJECTORY ───────────────────────────
    # Load the best model from your models folder
    # Note: Omit the .zip extension, SB3 handles it automatically
    model_path = r"C:\Users\Lkd\Desktop\proj\ML\models\best_model" 
    
    if not os.path.exists(model_path + ".zip"):
        model_path = r"C:\Users\Lkd\Desktop\proj\ML\models\ppo_orbital_docking_final"

    print(f"Loading model from {model_path} for trajectory simulation...")
    model = PPO.load(model_path, device="cpu")
    
    # Initialize environment and run one deterministic episode
    env = OrbitalDockingEnv(**ENV_CONFIG)
    obs, info = env.reset(seed=42) # Fixed seed for reproducible trajectory
    done = False
    
    while not done:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated

    # Extract X, Y, Z coordinates from the environment's trajectory
    trajectory = env.get_trajectory()
    x = trajectory[:, 0]  # In-track
    y = trajectory[:, 1]  # Radial
    z = trajectory[:, 2]  # Cross-track

    # Create Trajectory Figure
    fig_traj = plt.figure(figsize=(14, 6))

    # --- 3D Plot ---
    ax_3d = fig_traj.add_subplot(121, projection='3d')
    ax_3d.plot(x, y, z, label='Chaser Flight Path', color='dodgerblue', linewidth=2)
    ax_3d.scatter(0, 0, 0, color='red', s=100, label='Target (ISS)', edgecolors='black', zorder=5)
    ax_3d.scatter(x[0], y[0], z[0], color='green', s=50, label='Start Position')
    
    ax_3d.set_xlabel('In-track (m)')
    ax_3d.set_ylabel('Radial (m)')
    ax_3d.set_zlabel('Cross-track (m)')
    ax_3d.set_title("3D Orbital Docking Trajectory", fontweight='bold')
    ax_3d.legend()

    # --- 2D Projection (X-Y Orbital Plane) ---
    ax_2d = fig_traj.add_subplot(122)
    ax_2d.plot(x, y, label='Chaser Flight Path', color='dodgerblue', linewidth=2)
    ax_2d.scatter(0, 0, color='red', s=150, marker='*', label='Target (ISS)', zorder=5)
    ax_2d.scatter(x[0], y[0], color='green', s=50, label='Start Position')
    
    ax_2d.set_xlabel('In-track X (m)')
    ax_2d.set_ylabel('Radial Y (m)')
    ax_2d.set_title("2D Projection (In-track vs Radial)", fontweight='bold')
    ax_2d.grid(True, linestyle='--', alpha=0.7)
    ax_2d.legend()
    ax_2d.axhline(0, color='black', linewidth=0.5, alpha=0.5)
    ax_2d.axvline(0, color='black', linewidth=0.5, alpha=0.5)

    plt.tight_layout()

    # ─── 3. GENERATE SCATTER PLOTS (RELATIONSHIPS) ───────────────────────
    # Ensure arrays are the exact same length to prevent plotting errors
    min_len = min(len(data["avg_distances"]), len(data["success_rates"]), len(data["avg_fuels"]))
    dist = data["avg_distances"][:min_len]
    succ = data["success_rates"][:min_len]
    fuel = data["avg_fuels"][:min_len]

    fig_scatter, axs_scatter = plt.subplots(1, 3, figsize=(16, 5))

    # Distance vs Success
    axs_scatter[0].scatter(dist, succ, alpha=0.7, color='blue', edgecolors='black', linewidths=0.5)
    axs_scatter[0].set_title("Distance vs Success Rate", fontweight='bold')
    axs_scatter[0].set_xlabel("Avg Final Distance (m)")
    axs_scatter[0].set_ylabel("Success Rate")
    axs_scatter[0].grid(True, linestyle='--', alpha=0.6)

    # Distance vs Fuel Consumed
    axs_scatter[1].scatter(dist, fuel, alpha=0.7, color='green', edgecolors='black', linewidths=0.5)
    axs_scatter[1].set_title("Distance vs Fuel Consumed", fontweight='bold')
    axs_scatter[1].set_xlabel("Avg Final Distance (m)")
    axs_scatter[1].set_ylabel("Avg Fuel Used")
    axs_scatter[1].grid(True, linestyle='--', alpha=0.6)

    # Fuel vs Success
    axs_scatter[2].scatter(fuel, succ, alpha=0.7, color='purple', edgecolors='black', linewidths=0.5)
    axs_scatter[2].set_title("Fuel Consumed vs Success Rate", fontweight='bold')
    axs_scatter[2].set_xlabel("Avg Fuel Used")
    axs_scatter[2].set_ylabel("Success Rate")
    axs_scatter[2].grid(True, linestyle='--', alpha=0.6)

    fig_scatter.suptitle("Correlations: Distance, Fuel, and Success", fontsize=14, fontweight='bold')
    fig_scatter.tight_layout()

    # Display all 3 pages
    plt.show()

if __name__ == "__main__":
    plot_everything()
"""
===========================================================================
  HYPERPARAMETER CONFIGURATION
  PPO for Autonomous Orbital Docking
===========================================================================

BEGINNER GUIDE TO PPO HYPERPARAMETERS:
---------------------------------------
PPO (Proximal Policy Optimization) has several key hyperparameters.
Each one is explained below with the intuition behind it.

PPO works by:
  1. Running the current policy to collect experience (rollout)
  2. Computing advantages (how much better/worse than expected)
  3. Updating the policy using clipped gradient descent
  4. Ensuring the new policy doesn't change too much (the "proximal" part)

TRAINING SPEED EXPECTATIONS (CPU, Intel i7):
  - ~500-1000 environment steps/second
  - ~200-400 episodes to start seeing improvement
  - ~1M steps for good performance (~30-60 min)
  - ~2M steps for convergent performance (~1-2 hours)
"""

# ─────────────────────────────────────────────────────────────────────────────
#  ENVIRONMENT CONFIGURATION
# ─────────────────────────────────────────────────────────────────────────────
ENV_CONFIG = {
    # ISS-like orbit altitude
    "orbit_altitude_km": 400.0,

    # Simulation timestep (seconds)
    # Smaller = more accurate but slower training
    # 1.0 second is a good balance for this problem
    "dt": 1.0,

    # Maximum steps per episode
    # At dt=1s, this gives 1000 seconds per episode (~1/5 orbit)
    "max_episode_steps": 1000,

    # Maximum thrust acceleration [m/s²]
    # Real spacecraft: ~0.01–0.1 m/s² for RCS thrusters
    # Larger = faster but less fuel-efficient agent
    "max_thrust": 0.05,

    # Docking success zone radius [meters]
    "docking_radius": 0.5,

    # Maximum speed for safe docking [m/s]
    "max_docking_speed": 0.05,

    # Initial position range [meters from target]
    # Start: 50–200m away (realistic V-bar approach)
    "initial_distance_range": (50.0, 200.0),

    # Fuel penalty weight (higher = more fuel-efficient agent)
    "fuel_penalty_weight": 0.03,

    # Velocity safety penalty weight
    "velocity_penalty_weight": 0.1,

    # Use 3D simulation (True) or 2D (False)
    # 3D is harder but more realistic
    "use_3d": True,

    # ── Perturbation scales ────────────────────────────────────────────────
    # 0.0 = disabled (pure CW), 1.0 = full physical magnitude.
    # Increase gradually once the agent is already stable on clean CW.
    # Suggested schedule:
    #   0.00 → baseline training (pure CW)
    #   0.10 → gentle intro (~10% of real perturbation)
    #   0.40 → moderate realism
    #   0.70 → high realism
    #   1.00 → full physics
    "j2_perturbation_scale": 1,    # J2 oblateness differential acceleration
    "drag_perturbation_scale": 1.0,  # aerodynamic drag differential acceleration

    # ── New perturbations (set to 0.0 for baseline; increase via curriculum) ─
    # srp_perturbation_scale:
    #   Solar Radiation Pressure differential accel (~3e-8 m/s² at full scale).
    #   Randomises sun direction each episode; toggles off during eclipse.
    "srp_perturbation_scale": 1,

    # thruster_noise_scale:
    #   RCS execution errors: ±5% magnitude noise, ±1° misalignment, 1% dead-band.
    #   Most impactful perturbation for agent robustness — add early.
    "thruster_noise_scale": 1,

    # sensor_noise_scale:
    #   GPS/IMU observation noise: σ_pos=0.3 m, σ_vel=0.003 m/s at full scale.
    #   Agent never sees true state — must learn to act under measurement uncertainty.
    "sensor_noise_scale": 1,
}

# ─────────────────────────────────────────────────────────────────────────────
#  PPO HYPERPARAMETERS
# ─────────────────────────────────────────────────────────────────────────────
PPO_CONFIG = {
    # ── Learning rate ──────────────────────────────────────────────────────
    # How large each gradient step is.
    # Too large → unstable, oscillating training
    # Too small → very slow learning
    # 3e-4 is a reliable default for PPO
    "learning_rate": 3e-4, # 3e-4 is a reliable default for PPO

    # ── Number of parallel environments ───────────────────────────────────
    # Running multiple envs simultaneously gives more diverse experience.
    # With CPU, 4-8 envs is a good balance (no GPU bottleneck).
    "n_envs": 4,

    # ── Steps per rollout (per environment) ───────────────────────────────
    # Number of steps collected before each policy update.
    # Total steps per update = n_steps * n_envs
    # Larger = more stable but slower; smaller = faster but noisier
    "n_steps": 2048,

    # ── Batch size ────────────────────────────────────────────────────────
    # Mini-batch size for gradient updates.
    # Must divide n_steps * n_envs evenly.
    # Larger = more stable gradients
    "batch_size": 256,

    # ── Number of optimization epochs ─────────────────────────────────────
    # How many times to reuse each batch of experience.
    # PPO allows multiple epochs because of the clipping constraint.
    # Too many → policy diverges; too few → sample inefficient
    "n_epochs": 10,

    # ── Discount factor (gamma) ───────────────────────────────────────────
    # How much to value future rewards vs. immediate rewards.
    # 0.0 → only immediate reward; 1.0 → equal weight to all future
    # 0.99 is standard: values rewards ~100 steps in the future
    "gamma": 0.99,

    # ── GAE lambda ────────────────────────────────────────────────────────
    # Generalized Advantage Estimation parameter.
    # Controls bias-variance tradeoff in advantage estimation.
    # 1.0 → low bias, high variance; 0.0 → high bias, low variance
    # 0.95 is the Schulman et al. recommended value
    "gae_lambda": 0.95,

    # ── Clipping parameter (epsilon) ──────────────────────────────────────
    # The KEY parameter that makes PPO stable.
    # Prevents the policy from changing too much in one update.
    # Mathematically: clip(ratio, 1-clip_range, 1+clip_range)
    # 0.2 means: new policy can't be more than 20% different from old
    "clip_range": 0.2,

    # ── Value function clipping ───────────────────────────────────────────
    # Whether to also clip the value function loss.
    # None = no clipping
    "clip_range_vf": None,

    # ── Entropy coefficient ───────────────────────────────────────────────
    # Encourages exploration by penalizing overconfident policies.
    # Higher = more random exploration; lower = more exploitation
    # 0.01 is typical; increase if agent gets stuck in local optima
    "ent_coef": 0.01,

    # ── Value function loss weight ─────────────────────────────────────────
    # Scales the value function (critic) loss relative to policy (actor) loss.
    "vf_coef": 0.5,

    # ── Max gradient norm ──────────────────────────────────────────────────
    # Clips gradients to prevent catastrophic updates.
    # 0.5 is a safe default.
    "max_grad_norm": 0.5,

    # ── Total training steps ───────────────────────────────────────────────
    # Total environment steps for training.
    # 1M steps → ~30-60 min on CPU i7
    # 2M steps → ~1-2 hours, better performance
    "total_timesteps": 2_000_000,

    # ── Network architecture ──────────────────────────────────────────────
    # Two hidden layers, each with 256 neurons.
    # This is large enough for the problem complexity but not too slow.
    # Both actor (policy) and critic (value function) share this architecture.
    "policy_kwargs": {
        "net_arch": [256, 256],
    },

    # ── Normalize advantages ───────────────────────────────────────────────
    # Normalize advantages to mean 0, std 1 within each minibatch.
    # Helps with numerical stability of gradient updates.
    "normalize_advantage": True,

    # ── Verbose level ─────────────────────────────────────────────────────
    # 0 = no output, 1 = training progress, 2 = verbose debug
    "verbose": 1,
}

# ─────────────────────────────────────────────────────────────────────────────
#  TRAINING CONFIGURATION
# ─────────────────────────────────────────────────────────────────────────────
TRAINING_CONFIG = {
     "model_save_dir": r"C:\Users\Lkd\Desktop\proj\ML\models",
    "log_dir": r"C:\Users\Lkd\Desktop\proj\ML\logs",

    # Save checkpoint every N steps
    "save_freq": 100_000,

    # Evaluate (no exploration) every N steps
    "eval_freq": 25_000,

    # Number of evaluation episodes
    "n_eval_episodes": 20,

    # Random seed for reproducibility
    "seed": 42,

    # Model name prefix
    "model_name": "ppo_orbital_docking",
}

# ─────────────────────────────────────────────────────────────────────────────
#  CURRICULUM LEARNING CONFIG (advanced, optional)
# ─────────────────────────────────────────────────────────────────────────────
# Curriculum learning: start easy, gradually increase difficulty.
# This can significantly speed up learning.
CURRICULUM_CONFIG = {
    "enabled": True,

    # Curriculum stages: (min_timesteps, config_override)
    "stages": [
        # Stage 1: Easy — start close, 2D only, large docking zone
        {
            "timesteps": 500_000,
            "env_overrides": {
                "initial_distance_range": (20.0, 80.0),
                "use_3d": False,
                "docking_radius": 2.0,
                "max_docking_speed": 0.2,
            },
        },
        # Stage 2: Medium — wider range, 3D, standard constraints
        {
            "timesteps": 1_000_000,
            "env_overrides": {
                "initial_distance_range": (30.0, 150.0),
                "use_3d": True,
                "docking_radius": 1.0,
                "max_docking_speed": 0.1,
            },
        },
        # Stage 3: Hard — full range, strict constraints
        {
            "timesteps": 2_000_000,
            "env_overrides": {
                "initial_distance_range": (50.0, 200.0),
                "use_3d": True,
                "docking_radius": 0.5,
                "max_docking_speed": 0.05,
            },
        },
    ],
}

# ─────────────────────────────────────────────────────────────────────────────
#  PERTURBATION CURRICULUM CONFIG
# ─────────────────────────────────────────────────────────────────────────────
# Use this AFTER the agent has already converged on clean CW (~2M steps).
# Five perturbation axes, each a scalar 0.0 → 1.0:
#
#   j2_perturbation_scale    — Earth oblateness differential accel
#   drag_perturbation_scale  — aerodynamic drag differential decel
#   srp_perturbation_scale   — solar radiation pressure differential accel
#   thruster_noise_scale     — RCS magnitude error + misalignment + dead-band
#   sensor_noise_scale       — GPS/IMU position & velocity measurement noise
#
# RECOMMENDED WORKFLOW:
#   Phase 0: Train 2M steps at all scales = 0.0 (clean CW baseline).
#   Phase 1: Load best_model, resume with P1 overrides below.
#   Phase 2→5: Advance when success rate stabilises above ~80%.
#
# ORDERING RATIONALE:
#   Introduce thruster + sensor noise first — they have the largest
#   observable effect on the agent and are most common in Monte Carlo
#   validation. SRP and J2/drag are added later as they're subtler.
#
PERTURBATION_CURRICULUM = {
    "stages": [
        # Stage P1: Actuation uncertainty — thruster + sensor noise at 30%
        # Impact is immediate and large; agent must learn robustness fast.
        {
            "timesteps": 500_000,
            "env_overrides": {
                "thruster_noise_scale": 0.30,
                "sensor_noise_scale":   0.30,
                "j2_perturbation_scale":   0.0,
                "drag_perturbation_scale": 0.0,
                "srp_perturbation_scale":  0.0,
            },
            "note": "30% thruster + sensor noise. First robustness test.",
        },
        # Stage P2: Full actuation uncertainty + gentle physics
        {
            "timesteps": 500_000,
            "env_overrides": {
                "thruster_noise_scale": 1.00,
                "sensor_noise_scale":   1.00,
                "srp_perturbation_scale":  0.30,
                "j2_perturbation_scale":   0.10,
                "drag_perturbation_scale": 0.10,
            },
            "note": "Full noise. SRP + J2/drag gently introduced.",
        },
        # Stage P3: Full noise + moderate physics
        {
            "timesteps": 500_000,
            "env_overrides": {
                "thruster_noise_scale": 1.00,
                "sensor_noise_scale":   1.00,
                "srp_perturbation_scale":  0.70,
                "j2_perturbation_scale":   0.40,
                "drag_perturbation_scale": 0.40,
            },
            "note": "Near-realistic. Agent must compensate all perturbations.",
        },
        # Stage P4: Full everything — maximum realism
        {
            "timesteps": 500_000,
            "env_overrides": {
                "thruster_noise_scale": 1.00,
                "sensor_noise_scale":   1.00,
                "srp_perturbation_scale":  1.00,
                "j2_perturbation_scale":   1.00,
                "drag_perturbation_scale": 1.00,
            },
            "note": "All perturbations at full physical magnitude.",
        },
    ],
}
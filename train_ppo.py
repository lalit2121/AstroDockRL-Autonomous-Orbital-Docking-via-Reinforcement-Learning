"""
===========================================================================
  PPO TRAINING SCRIPT
  Autonomous Orbital Docking with Stable-Baselines3 + PyTorch
===========================================================================

REINFORCEMENT LEARNING OVERVIEW (beginner-friendly):
------------------------------------------------------
In Reinforcement Learning (RL), an AGENT learns by interacting with an
ENVIRONMENT through trial and error.

At each step:
  1. Agent observes the CURRENT STATE  s_t  (position, velocity, ...)
  2. Agent takes an ACTION  a_t        (fire thrusters in x/y/z direction)
  3. Environment returns:
       - NEW STATE  s_{t+1}            (updated position, velocity)
       - REWARD  r_t                   (how good was that action?)
  4. Agent updates its POLICY to prefer actions that led to higher rewards.

PPO (Proximal Policy Optimization) is the algorithm we use to update the
policy. It has two components:
  - ACTOR  (policy network): decides WHAT ACTION to take
  - CRITIC (value network):  estimates HOW GOOD the current state is

The actor and critic are neural networks with the architecture:
  Input (9) → Dense(256) → ReLU → Dense(256) → ReLU → Output

WHY PPO?
  - Very stable training (few hyperparameters to tune)
  - Works well on continuous action spaces (like thrust vectors)
  - Efficient sample usage (uses experience multiple times via clipping)
  - Industry-proven: used in robotics, game AI, spaceflight research
  - CPU-friendly: not as compute-heavy as SAC or TD3 alternatives
"""

import os
import sys
import time
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.callbacks import (
    EvalCallback,
    CheckpointCallback,
    BaseCallback,
    CallbackList,
)
from stable_baselines3.common.vec_env import SubprocVecEnv
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.utils import set_random_seed

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from docking_env import OrbitalDockingEnv
from hyperparameters import ENV_CONFIG, PPO_CONFIG, TRAINING_CONFIG, CURRICULUM_CONFIG, PERTURBATION_CURRICULUM


# ─────────────────────────────────────────────────────────────────────────────
#  CUSTOM CALLBACK: Track episode statistics during training
# ─────────────────────────────────────────────────────────────────────────────
class DockingMetricsCallback(BaseCallback):
    """
    Custom callback to track docking-specific metrics during training.

    At regular intervals, this callback reports:
      - Success rate (fraction of episodes that docked successfully)
      - Average fuel used per episode
      - Average distance at episode end
      - Average episode length

    These metrics help us understand if the agent is actually learning
    to dock, not just optimizing reward in an unexpected way.
    """

    def __init__(self, eval_env, eval_freq: int = 10_000, n_eval_ep: int = 20, verbose: int = 1):
        super().__init__(verbose)
        self.eval_env = eval_env
        self.eval_freq = eval_freq
        self.n_eval_ep = n_eval_ep

        # Storage for metrics over time
        self.success_rates = []
        self.avg_fuels = []
        self.avg_distances = []
        self.avg_lengths = []
        self.timesteps_list = []

    def _on_step(self) -> bool:
        """Called after every environment step. Return False to stop training."""
        if self.n_calls % self.eval_freq == 0:
            self._evaluate_policy()
        return True  # Continue training

    def _evaluate_policy(self):
        """Run deterministic episodes and collect docking statistics."""
        successes = []
        fuels = []
        final_distances = []
        ep_lengths = []

        for ep in range(self.n_eval_ep):
            obs, info = self.eval_env.reset()
            done = False
            ep_len = 0

            while not done:
                # Use deterministic action (no exploration noise)
                action, _ = self.model.predict(obs, deterministic=True)
                obs, reward, terminated, truncated, info = self.eval_env.step(action)
                done = terminated or truncated
                ep_len += 1

            successes.append(float(info["success"]))
            fuels.append(info["fuel_used"])
            final_distances.append(info["distance"])
            ep_lengths.append(ep_len)

        success_rate = np.mean(successes)
        avg_fuel = np.mean(fuels)
        avg_dist = np.mean(final_distances)
        avg_len = np.mean(ep_lengths)

        self.success_rates.append(success_rate)
        self.avg_fuels.append(avg_fuel)
        self.avg_distances.append(avg_dist)
        self.avg_lengths.append(avg_len)
        self.timesteps_list.append(self.num_timesteps)
        

        if self.verbose >= 1:
            print(
                f"\n[Eval @ {self.num_timesteps:,} steps] "
                f"Success: {success_rate*100:.1f}% | "
                f"Fuel: {avg_fuel:.3f} | "
                f"Dist: {avg_dist:.2f}m | "
                f"Len: {avg_len:.0f} steps"
            )

    def get_metrics(self):
        return {
            "timesteps": self.timesteps_list,
            "success_rates": self.success_rates,
            "avg_fuels": self.avg_fuels,
            "avg_distances": self.avg_distances,
            "avg_lengths": self.avg_lengths,
        }


# ─────────────────────────────────────────────────────────────────────────────
#  ENVIRONMENT FACTORY
# ─────────────────────────────────────────────────────────────────────────────
def make_docking_env(env_config=None, seed=0):
    """Factory function to create a monitored docking environment."""
    if env_config is None:
        env_config = ENV_CONFIG.copy()

    def _make():
        env = OrbitalDockingEnv(**env_config)
        env = Monitor(env)
        return env

    return _make


def create_vec_env(n_envs, env_config=None, seed=42):
    """
    Create vectorized environments for parallel training.

    WHY VECTORIZED ENVS?
    Running N environments in parallel collects N times as much
    experience per unit of wall-clock time. This is free speedup!

    DummyVecEnv: runs all envs in the same process (simpler, good for CPU)
    SubprocVecEnv: uses subprocess for each env (better for heavy envs)
    """
    if env_config is None:
        env_config = ENV_CONFIG.copy()

    env = make_vec_env(
        make_docking_env(env_config, seed),
        n_envs=n_envs,
        seed=seed,
        vec_env_cls=SubprocVecEnv,  # Uses DummyVecEnv (best for lightweight CPU envs)
    )
    return env


# ─────────────────────────────────────────────────────────────────────────────
#  PPO MODEL CREATION
# ─────────────────────────────────────────────────────────────────────────────
def create_ppo_model(env, ppo_config=None, device="cpu"):
    """
    Create a PPO model with the given configuration.

    NETWORK ARCHITECTURE:
    ┌─────────────────────────────────────────────────────┐
    │  Shared layers (actor + critic both use these):      │
    │  Input(9) → Linear(256) → Tanh → Linear(256) → Tanh │
    │                                                       │
    │  Actor head:  → Linear(256) → Linear(3)  [μ of action distribution]
    │  Critic head: → Linear(256) → Linear(1)  [value estimate V(s)]
    └─────────────────────────────────────────────────────┘

    The ACTOR outputs the mean of a Gaussian distribution over actions.
    During training: actions are sampled from this distribution (exploration).
    During evaluation: the mean action is used (exploitation).

    The CRITIC estimates V(s): the expected total future reward from state s.
    This is used to compute ADVANTAGES: A(s,a) = Q(s,a) - V(s)
    Advantage tells us: "was this action better or worse than average?"
    """
    if ppo_config is None:
        ppo_config = PPO_CONFIG.copy()

    model = PPO(
        policy="MlpPolicy",           # Multi-Layer Perceptron policy
        env=env,
        learning_rate=ppo_config["learning_rate"],
        n_steps=ppo_config["n_steps"],
        batch_size=ppo_config["batch_size"],
        n_epochs=ppo_config["n_epochs"],
        gamma=ppo_config["gamma"],
        gae_lambda=ppo_config["gae_lambda"],
        clip_range=ppo_config["clip_range"],
        clip_range_vf=ppo_config["clip_range_vf"],
        ent_coef=ppo_config["ent_coef"],
        vf_coef=ppo_config["vf_coef"],
        max_grad_norm=ppo_config["max_grad_norm"],
        normalize_advantage=ppo_config["normalize_advantage"],
        policy_kwargs=ppo_config["policy_kwargs"],
        verbose=ppo_config["verbose"],
        device=device,
        tensorboard_log=TRAINING_CONFIG["log_dir"],
    )
    return model


# ─────────────────────────────────────────────────────────────────────────────
#  MAIN TRAINING FUNCTION
# ─────────────────────────────────────────────────────────────────────────────
def train(
    use_curriculum: bool = False,
    resume_from: str = None,
    total_timesteps: int = None,
    device: str = "cpu",
):
    """
    Main training pipeline for the orbital docking PPO agent.

    Args:
        use_curriculum: Whether to use curriculum learning (start easy)
        resume_from:    Path to a model to resume training from
        total_timesteps: Override the total timesteps from config
        device:         'cpu' or 'cuda' (use 'cpu' for this project)
    """

    # ── Setup directories ─────────────────────────────────────────────────
    os.makedirs(TRAINING_CONFIG["model_save_dir"], exist_ok=True)
    os.makedirs(TRAINING_CONFIG["log_dir"], exist_ok=True)
    os.makedirs("results/plots", exist_ok=True)

    print("=" * 65)
    print("  AUTONOMOUS ORBITAL DOCKING — PPO TRAINING")
    print("=" * 65)
    print(f"  Device:           {device}")
    print(f"  Parallel envs:    {PPO_CONFIG['n_envs']}")
    print(f"  Total timesteps:  {total_timesteps or PPO_CONFIG['total_timesteps']:,}")
    print(f"  Curriculum:       {'ON' if use_curriculum else 'OFF'}")
    print(f"  Orbit altitude:   {ENV_CONFIG['orbit_altitude_km']} km")
    print(f"  Mean motion n:    {np.sqrt(3.986004418e14/(6371e3+400e3*1e3)**3)*1e-3:.6f} mrad/s")
    print("=" * 65)

    # Set random seed for reproducibility
    set_random_seed(TRAINING_CONFIG["seed"])

    # ── Create environments ───────────────────────────────────────────────
    n_envs = PPO_CONFIG["n_envs"]

    if use_curriculum:
        # Start with the easiest stage
        env_config = {**ENV_CONFIG, **CURRICULUM_CONFIG["stages"][0]["env_overrides"]}
    else:
        env_config = ENV_CONFIG.copy()

    print("\nCreating training environments...")
    train_env = create_vec_env(n_envs, env_config, seed=TRAINING_CONFIG["seed"])

    # Single eval environment (no parallelism needed for evaluation)
    eval_env = OrbitalDockingEnv(**ENV_CONFIG)  

    # ── Create PPO model ──────────────────────────────────────────────────
    if resume_from and os.path.exists(resume_from):
        print(f"\nResuming from: {resume_from}")
        model = PPO.load(resume_from, env=train_env, device=device)
    else:
        print("\nCreating new PPO model...")
        model = create_ppo_model(train_env, device=device)

    # Print model architecture
    print("\nNeural Network Architecture:")
    print(model.policy)
    total_params = sum(p.numel() for p in model.policy.parameters())
    print(f"Total parameters: {total_params:,}\n")

    # ── Setup callbacks ───────────────────────────────────────────────────
    # 1. Checkpoint: save model periodically
    checkpoint_cb = CheckpointCallback(
        save_freq=TRAINING_CONFIG["save_freq"] // n_envs,
        save_path=TRAINING_CONFIG["model_save_dir"],
        name_prefix=TRAINING_CONFIG["model_name"],
        verbose=1,
    )

    # 2. Evaluation: track best model
    eval_cb = EvalCallback(
        eval_env,
        best_model_save_path=TRAINING_CONFIG["model_save_dir"],
        log_path=TRAINING_CONFIG["log_dir"],
        eval_freq=TRAINING_CONFIG["eval_freq"] // n_envs,
        n_eval_episodes=TRAINING_CONFIG["n_eval_episodes"],
        deterministic=True,
        render=False,
        verbose=1,
    )

    # 3. Custom metrics: track docking success rate
    metrics_cb = DockingMetricsCallback(
        eval_env=eval_env,
        eval_freq=TRAINING_CONFIG["eval_freq"] // n_envs,
        n_eval_ep=TRAINING_CONFIG["n_eval_episodes"],
        verbose=1,
    )

    callback_list = CallbackList([checkpoint_cb, eval_cb, metrics_cb])

    # ── TRAINING LOOP ─────────────────────────────────────────────────────
    total_steps = total_timesteps or PPO_CONFIG["total_timesteps"]

    if use_curriculum:
        # Curriculum: train in stages of increasing difficulty
        print("Starting CURRICULUM training...\n")
        stages = CURRICULUM_CONFIG["stages"]

        for stage_idx, stage in enumerate(stages):
            stage_steps = stage["timesteps"]
            stage_config = {**ENV_CONFIG, **stage["env_overrides"]}

            print(f"\n{'='*50}")
            print(f"  CURRICULUM STAGE {stage_idx + 1}/{len(stages)}")
            print(f"  Initial range: {stage_config['initial_distance_range']} m")
            print(f"  3D mode: {stage_config['use_3d']}")
            print(f"  Docking radius: {stage_config['docking_radius']} m")
            print(f"  Steps: {stage_steps:,}")
            print(f"{'='*50}\n")

            # Recreate env for this stage
            stage_env = create_vec_env(n_envs, stage_config, seed=TRAINING_CONFIG["seed"])
            model.set_env(stage_env)
            metrics_cb.model = model

            model.learn(
                total_timesteps=stage_steps,
                callback=callback_list,
                reset_num_timesteps=(stage_idx == 0),
                tb_log_name=f"PPO_stage{stage_idx+1}",
                progress_bar=True,
            )

    else:
        # Standard training (no curriculum)
        print("\nStarting training...\n")
        t_start = time.time()

        model.learn(
            total_timesteps=total_steps,
            callback=callback_list,
            reset_num_timesteps=True,
            tb_log_name="PPO_docking",
            progress_bar=True,
        )

        t_end = time.time()
        elapsed = t_end - t_start
        print(f"\nTraining complete! Time: {elapsed/60:.1f} min")
        print(f"Speed: {total_steps/elapsed:.0f} steps/sec")

    # ── Save final model ──────────────────────────────────────────────────
    final_path = os.path.join(
        TRAINING_CONFIG["model_save_dir"],
        f"{TRAINING_CONFIG['model_name']}_final"
    )
    model.save(final_path)
    print(f"\nFinal model saved to: {final_path}.zip")

    # ── Save metrics for plotting ─────────────────────────────────────────
    import json
    metrics = metrics_cb.get_metrics()
    metrics_path = os.path.join(TRAINING_CONFIG["log_dir"], "docking_metrics.json")
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"Metrics saved to: {metrics_path}")

    train_env.close()
    eval_env.close()

    return model, metrics_cb.get_metrics()


# ─────────────────────────────────────────────────────────────────────────────
#  INFERENCE / TESTING FUNCTION
# ─────────────────────────────────────────────────────────────────────────────
def run_inference(model_path: str, n_episodes: int = 5, render: bool = True):
    """
    Load a trained model and run it deterministically.

    This lets us visualize the learned docking behavior without
    any exploration noise — the agent acts purely based on what
    it has learned.

    Args:
        model_path: Path to the saved .zip model file
        n_episodes: Number of test episodes to run
        render:     Whether to print step-by-step info

    Returns:
        List of episode results (trajectory, info, success)
    """
    print(f"\nLoading model from: {model_path}")
    model = PPO.load(model_path, device="cpu")

    env = OrbitalDockingEnv(**ENV_CONFIG, render_mode="human" if render else None)

    results = []

    for ep in range(n_episodes):
        print(f"\n{'─'*50}")
        print(f"  EPISODE {ep+1}/{n_episodes}")
        print(f"{'─'*50}")

        obs, info = env.reset(seed=ep * 100)
        done = False
        total_reward = 0

        while not done:
            # DETERMINISTIC action: use policy mean (no sampling noise)
            action, _states = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            total_reward += reward
            done = terminated or truncated

            if render and env.step_count % 50 == 0:
                env.render()

        trajectory = env.get_trajectory()

        print(f"\n  Final result:")
        print(f"    Success:    {info['success']}")
        print(f"    Crash:      {info['crash']}")
        print(f"    Distance:   {info['distance']:.3f} m")
        print(f"    Speed:      {info['speed']:.4f} m/s")
        print(f"    Fuel used:  {info['fuel_used']:.4f}")
        print(f"    Steps:      {info['step']}")
        print(f"    Reward:     {total_reward:.2f}")

        results.append({
            "episode": ep + 1,
            "trajectory": trajectory,
            "success": info["success"],
            "crash": info["crash"],
            "distance": info["distance"],
            "speed": info["speed"],
            "fuel_used": info["fuel_used"],
            "steps": info["step"],
            "total_reward": total_reward,
        })

    # Summary
    successes = sum(r["success"] for r in results)
    print(f"\n{'='*50}")
    print(f"  SUMMARY: {successes}/{n_episodes} successful dockings")
    print(f"  Success rate: {successes/n_episodes*100:.0f}%")
    print(f"{'='*50}")

    env.close()
    return results


# ─────────────────────────────────────────────────────────────────────────────
#  PERTURBATION CURRICULUM TRAINING
# ─────────────────────────────────────────────────────────────────────────────
def train_perturbation_curriculum(resume_from: str, device: str = "cpu"):
    """
    Continue training a converged model through progressively stronger
    J2 + drag perturbations (defined in PERTURBATION_CURRICULUM).

    Must be called with a pre-trained model (resume_from is required).
    Intended to run AFTER standard training has reached ~80%+ success.

    Args:
        resume_from: path to a converged model (.zip)
        device:      'cpu' or 'cuda'
    """
    assert resume_from and os.path.exists(resume_from + ".zip"), \
        f"Model not found: {resume_from}.zip — perturbation curriculum requires a pre-trained model."

    os.makedirs(TRAINING_CONFIG["model_save_dir"], exist_ok=True)
    os.makedirs(TRAINING_CONFIG["log_dir"], exist_ok=True)

    set_random_seed(TRAINING_CONFIG["seed"])
    n_envs = PPO_CONFIG["n_envs"]
    stages = PERTURBATION_CURRICULUM["stages"]

    print("=" * 65)
    print("  PERTURBATION CURRICULUM TRAINING")
    print(f"  Resuming from: {resume_from}")
    print(f"  Stages: {len(stages)}")
    print("=" * 65)

    # Eval env always runs at current stage's perturbation level
    eval_env = OrbitalDockingEnv(**ENV_CONFIG)

    # Load model once; we'll swap envs per stage
    dummy_env = create_vec_env(n_envs, ENV_CONFIG.copy(), seed=TRAINING_CONFIG["seed"])
    model = PPO.load(resume_from, env=dummy_env, device=device)

    for stage_idx, stage in enumerate(stages):
        stage_config = {**ENV_CONFIG, **stage["env_overrides"]}
        stage_steps = stage["timesteps"]
        note = stage.get("note", "")

        j2   = stage["env_overrides"]["j2_perturbation_scale"]
        drag = stage["env_overrides"]["drag_perturbation_scale"]

        print(f"\n{'='*50}")
        print(f"  PERTURBATION STAGE {stage_idx+1}/{len(stages)}")
        print(f"  J2 scale: {j2}  |  Drag scale: {drag}")
        print(f"  Steps: {stage_steps:,}  |  {note}")
        print(f"{'='*50}\n")

        # Recreate envs with updated perturbation scales
               
        stage_env = create_vec_env(n_envs, stage_config, seed=TRAINING_CONFIG["seed"])

       
        eval_env.j2_perturbation_scale    = stage_config["j2_perturbation_scale"]
        eval_env.drag_perturbation_scale  = stage_config["drag_perturbation_scale"]
        eval_env.srp_perturbation_scale   = stage_config["srp_perturbation_scale"]
        eval_env.thruster_noise_scale     = stage_config["thruster_noise_scale"]
        eval_env.sensor_noise_scale       = stage_config["sensor_noise_scale"]

        model.set_env(stage_env)

        checkpoint_cb = CheckpointCallback(
            save_freq=TRAINING_CONFIG["save_freq"] // n_envs,
            save_path=TRAINING_CONFIG["model_save_dir"],
            name_prefix=f"{TRAINING_CONFIG['model_name']}_perturb_s{stage_idx+1}",
            verbose=1,
        )
        eval_cb = EvalCallback(
            eval_env,
            best_model_save_path=TRAINING_CONFIG["model_save_dir"],
            log_path=TRAINING_CONFIG["log_dir"],
            eval_freq=TRAINING_CONFIG["eval_freq"] // n_envs,
            n_eval_episodes=TRAINING_CONFIG["n_eval_episodes"],
            deterministic=True,
            render=False,
            verbose=1,
        )
        metrics_cb = DockingMetricsCallback(
            eval_env=eval_env,
            eval_freq=TRAINING_CONFIG["eval_freq"] // n_envs,
            n_eval_ep=TRAINING_CONFIG["n_eval_episodes"],
            verbose=1,
        )

        model.learn(
            total_timesteps=stage_steps,
            callback=CallbackList([checkpoint_cb, eval_cb, metrics_cb]),
            reset_num_timesteps=False,  # keep global step counter
            tb_log_name=f"PPO_perturb_s{stage_idx+1}",
            progress_bar=True,
        )

        # Save checkpoint after each stage
        ckpt_path = os.path.join(
            TRAINING_CONFIG["model_save_dir"],
            f"{TRAINING_CONFIG['model_name']}_perturb_s{stage_idx+1}_final"
        )
        model.save(ckpt_path)
        print(f"Stage {stage_idx+1} model saved → {ckpt_path}.zip")
        stage_env.close()

    eval_env.close()
    print("\nPerturbation curriculum complete.")
    return model


# ─────────────────────────────────────────────────────────────────────────────
#  ENTRY POINT
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Train or test orbital docking PPO agent")
    parser.add_argument("--mode", choices=["train", "test", "both"], default="train",
                        help="Mode: train, test, or both")
    parser.add_argument("--timesteps", type=int, default=None,
                        help="Override total training timesteps")
    parser.add_argument("--curriculum", action="store_true",
                        help="Use curriculum learning")
    parser.add_argument("--perturb", action="store_true",
                        help="Run perturbation curriculum (requires --model)")
    parser.add_argument("--model", type=str, default=None,
                        help="Model path for testing (or resuming)")
    parser.add_argument("--episodes", type=int, default=5,
                        help="Number of test episodes")
    parser.add_argument("--device", type=str, default="cpu",
                        help="Device: cpu or cuda")
    args = parser.parse_args()

    if args.mode in ("train", "both"):
        model, metrics = train(
            use_curriculum=args.curriculum,
            resume_from=args.model,
            total_timesteps=args.timesteps,
            device=args.device,
        )
        model_path = os.path.join(
            TRAINING_CONFIG["model_save_dir"],
            f"{TRAINING_CONFIG['model_name']}_final"
        )

    if args.perturb:
        perturb_model_path = args.model or os.path.join(
            TRAINING_CONFIG["model_save_dir"], "best_model"
        )
        train_perturbation_curriculum(resume_from=perturb_model_path, device=args.device)

    if args.mode in ("test", "both"):
        model_path = args.model or os.path.join(
            TRAINING_CONFIG["model_save_dir"],
            "best_model"
        )
        results = run_inference(model_path, n_episodes=args.episodes)
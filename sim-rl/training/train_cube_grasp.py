"""PPO training script for cube grasping with the YAM arm.

All hyperparameters are read from config.yaml — edit that file to tune.

Usage:
  python train_cube_grasp.py                       # uses config.yaml
  python train_cube_grasp.py --config other.yaml   # custom config
"""

from __future__ import annotations

import argparse
import os
import sys
import time

import numpy as np
import torch
import yaml

from rsl_rl.storage import RolloutStorage
from rsl_rl.algorithms import PPO

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "envs"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "models"))
from cube_grasp_env import CubeGraspVecEnv
from networks import SimpleActor, SimpleCritic


def load_config(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def save_training_curves(iters, returns, successes, dists, save_dir):
    """Save training curves plot."""
    if not iters:
        return
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(3, 1, figsize=(10, 10), sharex=True)

    axes[0].plot(iters, returns, alpha=0.4, color="tab:blue")
    # Smoothed (rolling mean)
    window = max(1, len(iters) // 50)
    if window > 1:
        smooth = np.convolve(returns, np.ones(window)/window, mode="valid")
        axes[0].plot(iters[window-1:], smooth, color="tab:blue", linewidth=2)
    axes[0].set_ylabel("Mean Return")
    axes[0].set_title("Training Curves")
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(iters, [s * 100 for s in successes], alpha=0.4, color="tab:green")
    if window > 1:
        smooth_s = np.convolve([s*100 for s in successes], np.ones(window)/window, mode="valid")
        axes[1].plot(iters[window-1:], smooth_s, color="tab:green", linewidth=2)
    axes[1].set_ylabel("Success Rate (%)")
    axes[1].set_ylim(-5, 105)
    axes[1].grid(True, alpha=0.3)

    axes[2].plot(iters, dists, alpha=0.4, color="tab:orange")
    if window > 1:
        smooth_d = np.convolve(dists, np.ones(window)/window, mode="valid")
        axes[2].plot(iters[window-1:], smooth_d, color="tab:orange", linewidth=2)
    axes[2].set_ylabel("Min Distance (m)")
    axes[2].set_xlabel("Iteration")
    axes[2].grid(True, alpha=0.3)

    plt.tight_layout()
    plot_path = os.path.join(save_dir, "training_curves.png")
    plt.savefig(plot_path, dpi=150)
    plt.close()
    print(f"Training curves saved: {plot_path}")


# ---------- Main ----------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str,
                        default=os.path.join(os.path.dirname(__file__), "config.yaml"))
    args = parser.parse_args()

    cfg = load_config(args.config)
    device = cfg.get("device", "cpu")
    seed = cfg.get("seed", 42)
    torch.manual_seed(seed)

    env_cfg = cfg.get("env", {})
    rew_cfg = cfg.get("reward", {})
    ppo_cfg = cfg.get("ppo", {})
    net_cfg = cfg.get("network", {})
    run_cfg = cfg.get("runner", {})

    # --- Environment ---
    num_envs = env_cfg.get("num_envs", 64)
    print(f"Creating {num_envs} environments...")
    env = CubeGraspVecEnv(num_envs=num_envs, device=device, cfg=cfg)

    obs = env.get_observations()
    obs_dim = obs["obs"].shape[-1]
    act_dim = env.num_actions
    print(f"Obs dim: {obs_dim}, Act dim: {act_dim}")

    # --- Networks ---
    activation = net_cfg.get("activation", "elu")
    actor = SimpleActor(
        obs_dim, act_dim,
        hidden_dims=net_cfg.get("actor_hidden_dims", [256, 256, 128]),
        activation=activation,
    ).to(device)
    critic = SimpleCritic(
        obs_dim,
        hidden_dims=net_cfg.get("critic_hidden_dims", [256, 256, 128]),
        activation=activation,
    ).to(device)
    print(f"Actor params: {sum(p.numel() for p in actor.parameters()):,}")
    print(f"Critic params: {sum(p.numel() for p in critic.parameters()):,}")

    # --- Storage + PPO ---
    num_steps = run_cfg.get("num_steps_per_env", 24)
    storage = RolloutStorage("rl", num_envs, num_steps, obs, [act_dim], device)

    alg = PPO(
        actor=actor, critic=critic, storage=storage,
        num_learning_epochs=ppo_cfg.get("num_learning_epochs", 5),
        num_mini_batches=ppo_cfg.get("num_mini_batches", 4),
        clip_param=ppo_cfg.get("clip_param", 0.2),
        gamma=ppo_cfg.get("gamma", 0.99),
        lam=ppo_cfg.get("lam", 0.95),
        value_loss_coef=ppo_cfg.get("value_loss_coef", 1.0),
        entropy_coef=ppo_cfg.get("entropy_coef", 0.005),
        learning_rate=ppo_cfg.get("learning_rate", 3e-4),
        max_grad_norm=ppo_cfg.get("max_grad_norm", 1.0),
        use_clipped_value_loss=ppo_cfg.get("use_clipped_value_loss", True),
        schedule=ppo_cfg.get("schedule", "adaptive"),
        desired_kl=ppo_cfg.get("desired_kl", 0.01),
        device=device,
    )

    # --- Resume ---
    start_iter = 0
    ckpt_path = cfg.get("checkpoint")
    if ckpt_path and os.path.exists(ckpt_path):
        print(f"Loading checkpoint: {ckpt_path}")
        ckpt = torch.load(ckpt_path, map_location=device)
        alg.load(ckpt, None, strict=True)
        start_iter = ckpt.get("iteration", 0)
        print(f"Resuming from iteration {start_iter}")

    save_dir = os.path.join(os.path.dirname(__file__), "..", "checkpoints")
    os.makedirs(save_dir, exist_ok=True)

    max_iters = run_cfg.get("max_iterations", 5000)
    save_interval = run_cfg.get("save_interval", 500)
    log_interval = run_cfg.get("log_interval", 10)

    # --- Train ---
    print(f"\nTraining for {max_iters} iterations")
    print(f"  {num_envs} envs x {num_steps} steps = {num_envs * num_steps} samples/iter\n")

    def save_checkpoint(iteration, label=""):
        path = os.path.join(save_dir, f"ckpt_{label}{iteration:06d}.pt")
        ckpt = alg.save()
        ckpt["iteration"] = iteration
        torch.save(ckpt, path)
        return path

    obs = env.get_observations()
    total_steps = 0

    # Training curves
    curve_iters = []
    curve_ret = []
    curve_succ = []
    curve_dist = []

    try:
        for it in range(start_iter, max_iters):
            t0 = time.time()
            alg.train_mode()

            for _ in range(num_steps):
                actions = alg.act(obs)
                obs, rewards, dones, extras = env.step(actions)
                alg.process_env_step(obs, rewards, dones, extras)
                total_steps += num_envs

            alg.compute_returns(obs)
            losses = alg.update()
            dt = time.time() - t0

            if it % log_interval == 0:
                fps = (num_envs * num_steps) / dt
                metrics = env.get_metrics_and_reset()
                metric_str = ""
                if metrics:
                    curve_iters.append(it)
                    curve_ret.append(metrics['mean_return'])
                    curve_succ.append(metrics['success_rate'])
                    curve_dist.append(metrics['mean_min_dist'])
                    metric_str = (
                        f"  succ={metrics['success_rate']:.0%}"
                        f"  ret={metrics['mean_return']:.1f}"
                        f"  dist={metrics['mean_min_dist']:.3f}"
                        f"  eps={metrics['episodes']}"
                    )
                # Gradient norms
                actor_grad = torch.nn.utils.clip_grad_norm_(actor.parameters(), float("inf"))
                critic_grad = torch.nn.utils.clip_grad_norm_(critic.parameters(), float("inf"))
                print(
                    f"[{it:5d}] "
                    f"sur={losses['surrogate']:.4f}  "
                    f"val={losses['value']:.4f}  "
                    f"ent={losses['entropy']:.4f}  "
                    f"grad_a={actor_grad:.4f}  "
                    f"grad_c={critic_grad:.4f}  "
                    f"lr={alg.learning_rate:.2e}  "
                    f"fps={fps:.0f}"
                    f"{metric_str}"
                )

            if (it + 1) % save_interval == 0:
                path = save_checkpoint(it + 1)
                print(f"  Saved: {path}")

    except KeyboardInterrupt:
        print(f"\n\nInterrupted at iteration {it}.")
        path = save_checkpoint(it, label="interrupted_")
        print(f"Saved: {path}")
        save_training_curves(curve_iters, curve_ret, curve_succ, curve_dist, save_dir)
        return

    final = os.path.join(save_dir, "ckpt_final.pt")
    ckpt = alg.save()
    ckpt["iteration"] = max_iters
    torch.save(ckpt, final)
    print(f"\nDone. Final checkpoint: {final}")
    save_training_curves(curve_iters, curve_ret, curve_succ, curve_dist, save_dir)


if __name__ == "__main__":
    main()

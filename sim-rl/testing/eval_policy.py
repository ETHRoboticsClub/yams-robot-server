#!/usr/bin/env python
"""Evaluate a trained policy: run episodes, render in MuJoCo viewer, and produce plots.

Usage:
  python eval_policy.py --checkpoint ../checkpoints/ckpt_final.pt
  python eval_policy.py --checkpoint ../checkpoints/ckpt_final.pt --no-render --episodes 100

Outputs (saved to --out_dir, default ../results/):
  - episode_rewards.png    — per-episode return histogram + over-time curve
  - reward_components.png  — breakdown of reach/align/grip/lift per step (averaged)
  - value_vs_return.png    — critic prediction vs actual return (calibration)
  - action_distribution.png — action magnitude distribution per joint
  - summary.txt            — success rate, mean return, etc.
"""

from __future__ import annotations

import argparse
import os
import sys
import platform

# Auto-relaunch with mjpython on macOS (only when rendering)
_RENDER = "--no-render" not in sys.argv
if _RENDER and platform.system() == "Darwin" and "MJPYTHON" not in os.environ:
    import shutil
    mjpython = shutil.which("mjpython")
    if mjpython:
        os.environ["MJPYTHON"] = "1"
        os.execv(mjpython, [mjpython] + sys.argv)

import time
import numpy as np
import torch
import yaml
import mujoco
import matplotlib
matplotlib.use("Agg")  # always non-interactive — saves to files, avoids NSWindow crash with mjpython
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "envs"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "models"))
from cube_grasp_env import CubeGraspVecEnv
from networks import SimpleActor, SimpleCritic


def load_config(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def build_models(obs_dim, act_dim, net_cfg, device):
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
    return actor, critic


def run_eval_episodes(env, actor, critic, num_episodes, device):
    """Run episodes collecting per-step data."""
    episode_returns = []
    episode_lengths = []
    episode_successes = []

    # Per-step accumulators (for averaging)
    all_values = []
    all_returns_per_step = []
    all_actions = []
    all_reach_dists = []

    # Obs layout (45-dim):
    #   arm1: [0:6] joint_pos, [6:12] joint_vel, [12] grip, [13:16] rel
    #   arm2: [16:22] joint_pos, [22:28] joint_vel, [28] grip, [29:32] rel
    #   shared: [32:35] cube_pos, [35:39] cube_quat, [39:42] grip_z_arm1, [42:45] grip_z_arm2
    ARM1_REL = slice(13, 16)
    ARM2_REL = slice(29, 32)
    CUBE_Z_IDX = 34  # cube_pos z component

    # Cube rest height — success = cube lifted above this + lift_target
    cube_rest_z = 0.015
    lift_target = 0.08

    ep_count = 0
    obs = env.get_observations()
    ep_reward_accum = np.zeros(env.num_envs)
    ep_step_count = np.zeros(env.num_envs, dtype=int)
    ep_max_cube_z = np.full(env.num_envs, -np.inf)
    ep_actions_buf = [[] for _ in range(env.num_envs)]
    ep_values_buf = [[] for _ in range(env.num_envs)]
    ep_rewards_buf = [[] for _ in range(env.num_envs)]
    ep_dists_buf = [[] for _ in range(env.num_envs)]

    while ep_count < num_episodes:
        with torch.no_grad():
            actions = actor(obs, stochastic_output=False)  # deterministic
            values = critic(obs).squeeze(-1)

        obs_np = obs["obs"].cpu().numpy()
        actions_np = actions.cpu().numpy()
        values_np = values.cpu().numpy()

        for i in range(env.num_envs):
            # Best (closest) arm distance to cube
            d1 = np.linalg.norm(obs_np[i, ARM1_REL])
            d2 = np.linalg.norm(obs_np[i, ARM2_REL])
            ep_dists_buf[i].append(min(d1, d2))
            ep_actions_buf[i].append(actions_np[i].copy())
            ep_values_buf[i].append(values_np[i])
            # Track max cube height
            ep_max_cube_z[i] = max(ep_max_cube_z[i], obs_np[i, CUBE_Z_IDX])

        obs, rewards, dones, extras = env.step(actions)
        rewards_np = rewards.cpu().numpy()
        dones_np = dones.cpu().numpy()

        for i in range(env.num_envs):
            ep_reward_accum[i] += rewards_np[i]
            ep_rewards_buf[i].append(rewards_np[i])
            ep_step_count[i] += 1

            if dones_np[i] > 0.5 and ep_count < num_episodes:
                episode_returns.append(ep_reward_accum[i])
                episode_lengths.append(ep_step_count[i])

                # Success = cube was lifted above rest + lift_target at some point
                episode_successes.append(ep_max_cube_z[i] > cube_rest_z + lift_target)

                all_actions.extend(ep_actions_buf[i])
                all_values.extend(ep_values_buf[i])
                all_returns_per_step.extend(compute_discounted_returns(ep_rewards_buf[i], gamma=0.99))
                all_reach_dists.extend(ep_dists_buf[i])

                ep_reward_accum[i] = 0
                ep_step_count[i] = 0
                ep_max_cube_z[i] = -np.inf
                ep_actions_buf[i] = []
                ep_values_buf[i] = []
                ep_rewards_buf[i] = []
                ep_dists_buf[i] = []
                ep_count += 1

    return {
        "episode_returns": np.array(episode_returns[:num_episodes]),
        "episode_lengths": np.array(episode_lengths[:num_episodes]),
        "episode_successes": np.array(episode_successes[:num_episodes]),
        "all_actions": np.array(all_actions),
        "all_values": np.array(all_values),
        "all_returns": np.array(all_returns_per_step),
        "all_reach_dists": np.array(all_reach_dists),
    }


def compute_discounted_returns(rewards, gamma=0.99):
    returns = []
    G = 0
    for r in reversed(rewards):
        G = r + gamma * G
        returns.insert(0, G)
    return returns


def render_episodes_to_gif(env, actor, device, num_episodes, max_steps, out_dir,
                           tile_width=320, tile_height=240, fps=20):
    """Run all episodes in parallel, render offscreen, combine into one grid GIF."""
    from PIL import Image
    import math

    model = env.model
    num_envs = env.num_envs
    os.makedirs(out_dir, exist_ok=True)

    # Grid layout
    cols = math.ceil(math.sqrt(num_envs))
    rows = math.ceil(num_envs / cols)
    grid_w = cols * tile_width
    grid_h = rows * tile_height

    # Offscreen renderer
    renderer = mujoco.Renderer(model, height=tile_height, width=tile_width)

    # Camera
    cam = mujoco.MjvCamera()
    cam.type = mujoco.mjtCamera.mjCAMERA_FREE
    cam.distance = 1.2
    cam.azimuth = 135
    cam.elevation = -25
    cam.lookat[:] = [0.15, -0.28, 0.1]

    # Reset all envs
    env._reset_all()
    obs = env.get_observations()

    grid_frames = []
    frame_skip = max(1, round(1.0 / (fps * env.ctrl_dt)))

    print(f"Recording {num_envs} episodes as {cols}x{rows} grid GIF ({grid_w}x{grid_h}, ~{fps} fps)...")

    for step in range(max_steps):
        with torch.no_grad():
            actions = actor(obs, stochastic_output=False)

        obs, rewards, dones, extras = env.step(actions)

        if step % frame_skip == 0:
            # Render each env tile and compose into grid
            grid = np.zeros((grid_h, grid_w, 3), dtype=np.uint8)
            for i in range(num_envs):
                renderer.update_scene(env.data_list[i], camera=cam)
                tile = renderer.render()
                r, c = divmod(i, cols)
                y0, x0 = r * tile_height, c * tile_width
                grid[y0: y0 + tile_height, x0: x0 + tile_width] = tile
            grid_frames.append(grid.copy())

    renderer.close()

    # Save single grid GIF
    if grid_frames:
        images = [Image.fromarray(f) for f in grid_frames]
        gif_path = os.path.join(out_dir, "eval_episodes.gif")
        frame_duration = max(1, int(1000 / fps))
        images[0].save(gif_path, save_all=True, append_images=images[1:],
                       duration=frame_duration, loop=0)
        print(f"Saved grid GIF: {gif_path} ({len(grid_frames)} frames)")
    else:
        print("No frames captured.")


def render_policy(env, actor, device, max_steps=500):
    """Render a single episode in the MuJoCo viewer."""
    from tensordict import TensorDict
    import mujoco.viewer

    model = env.model
    data = env.data_list[0]
    env._reset_env(0)
    obs = env.get_observations()

    with mujoco.viewer.launch_passive(model, data) as viewer:
        step = 0
        while viewer.is_running() and step < max_steps:
            with torch.no_grad():
                actions = actor(obs, stochastic_output=False)

            act = actions[0].cpu().numpy()

            # Apply actions to both arms (same layout as env.step)
            offset = 0
            for prefix in ["arm1", "arm2"]:
                arm = env.arms[prefix]
                joint_deltas = act[offset: offset + 6] * env.action_scale
                grip_target = np.clip(act[offset + 6], 0, 1) * 0.0475

                for j_idx, aid in enumerate(arm["act_ids"]):
                    data.ctrl[aid] = np.clip(
                        data.ctrl[aid] + joint_deltas[j_idx],
                        arm["joint_lo"][j_idx], arm["joint_hi"][j_idx],
                    )
                data.ctrl[arm["grip_act_id"]] = grip_target
                offset += 7

            for _ in range(env.sim_steps_per_ctrl):
                mujoco.mj_step(model, data)

            viewer.sync()
            time.sleep(env.ctrl_dt)

            # Refresh obs from env 0
            obs_np = env._get_obs_single(data)
            obs_t = torch.from_numpy(obs_np).float().unsqueeze(0).to(device)
            obs = TensorDict({"obs": obs_t}, batch_size=[1])

            step += 1

    print(f"Rendered {step} steps")


def plot_results(data: dict, out_dir: str):
    os.makedirs(out_dir, exist_ok=True)

    returns = data["episode_returns"]
    lengths = data["episode_lengths"]
    successes = data["episode_successes"]
    actions = data["all_actions"]
    values = data["all_values"]
    gt_returns = data["all_returns"]
    dists = data["all_reach_dists"]

    # 1. Episode rewards
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes[0].plot(returns, alpha=0.7)
    axes[0].set_xlabel("Episode")
    axes[0].set_ylabel("Return")
    axes[0].set_title("Episode Returns")
    axes[0].axhline(np.mean(returns), color="r", ls="--", label=f"mean={np.mean(returns):.2f}")
    axes[0].legend()
    axes[1].hist(returns, bins=30, edgecolor="black", alpha=0.7)
    axes[1].set_xlabel("Return")
    axes[1].set_ylabel("Count")
    axes[1].set_title("Return Distribution")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "episode_rewards.png"), dpi=150)
    plt.close()

    # 2. Value vs actual return (calibration)
    n = min(len(values), len(gt_returns))
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.scatter(gt_returns[:n], values[:n], alpha=0.05, s=2)
    lims = [min(np.min(gt_returns[:n]), np.min(values[:n])),
            max(np.max(gt_returns[:n]), np.max(values[:n]))]
    ax.plot(lims, lims, "r--", label="perfect calibration")
    ax.set_xlabel("Actual Return (MC)")
    ax.set_ylabel("Critic Prediction")
    ax.set_title("Value Function Calibration")
    ax.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "value_vs_return.png"), dpi=150)
    plt.close()

    # 3. Action distributions per dim
    joint_labels = ["j1", "j2", "j3", "j4", "j5", "j6", "grip"]
    fig, axes = plt.subplots(1, 7, figsize=(21, 3), sharey=True)
    for j in range(7):
        axes[j].hist(actions[:, j], bins=50, alpha=0.7, edgecolor="black")
        axes[j].set_title(joint_labels[j])
        axes[j].set_xlabel("Action value")
    axes[0].set_ylabel("Count")
    plt.suptitle("Action Distributions")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "action_distribution.png"), dpi=150)
    plt.close()

    # 4. Reach distance over time (averaged across episodes)
    max_len = int(np.max(lengths))
    dist_matrix = np.full((len(returns), max_len), np.nan)
    idx = 0
    for ep, ep_len in enumerate(lengths):
        dist_matrix[ep, :ep_len] = dists[idx:idx + ep_len]
        idx += ep_len
    mean_dist = np.nanmean(dist_matrix, axis=0)
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(mean_dist, label="mean reach distance")
    ax.set_xlabel("Step within episode")
    ax.set_ylabel("Grasp-to-cube distance (m)")
    ax.set_title("Reaching Progress (averaged)")
    ax.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "reach_distance.png"), dpi=150)
    plt.close()

    # 5. Value over time within episodes
    val_matrix = np.full((len(returns), max_len), np.nan)
    idx = 0
    for ep, ep_len in enumerate(lengths):
        val_matrix[ep, :ep_len] = values[idx:idx + ep_len]
        idx += ep_len
    mean_val = np.nanmean(val_matrix, axis=0)
    std_val = np.nanstd(val_matrix, axis=0)

    fig, ax = plt.subplots(figsize=(8, 4))
    steps_x = np.arange(max_len)
    ax.plot(steps_x, mean_val, label="mean V(s)", color="tab:blue")
    ax.fill_between(steps_x, mean_val - std_val, mean_val + std_val, alpha=0.2, color="tab:blue")
    ax.set_xlabel("Step within episode")
    ax.set_ylabel("Value V(s)")
    ax.set_title("Critic Value Over Episode")
    ax.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "value_over_time.png"), dpi=150)
    plt.close()

    # 6. Summary text
    success_rate = np.mean(successes) * 100
    summary = (
        f"Episodes:      {len(returns)}\n"
        f"Mean return:   {np.mean(returns):.2f} +/- {np.std(returns):.2f}\n"
        f"Median return: {np.median(returns):.2f}\n"
        f"Min / Max:     {np.min(returns):.2f} / {np.max(returns):.2f}\n"
        f"Mean length:   {np.mean(lengths):.1f}\n"
        f"Success rate:  {success_rate:.1f}%\n"
    )
    with open(os.path.join(out_dir, "summary.txt"), "w") as f:
        f.write(summary)
    print("\n" + summary)

    print(f"Plots saved to {out_dir}/")


def plot_value_heatmap(env, critic, device, out_dir, grid_res=30):
    """Sweep cube (x,y) positions on a grid, query critic at home pose, plot heatmap.

    Shows where the value function thinks the cube is easiest/hardest to grasp.
    """
    from tensordict import TensorDict

    model = env.model
    d = env.data_list[0]

    x_range = env.cube_x_range
    y_range = env.cube_y_range
    xs = np.linspace(x_range[0], x_range[1], grid_res)
    ys = np.linspace(y_range[0], y_range[1], grid_res)

    value_grid = np.zeros((grid_res, grid_res))

    for ix, cx in enumerate(xs):
        for iy, cy in enumerate(ys):
            # Reset to home pose with cube at (cx, cy)
            mujoco.mj_resetData(model, d)
            ba = env.block_qpos_addr
            d.qpos[ba: ba + 3] = [cx, cy, env.cube_z]
            d.qpos[ba + 3: ba + 7] = [1, 0, 0, 0]
            for arm in env.arms.values():
                d.ctrl[arm["grip_act_id"]] = 0.0475
            mujoco.mj_forward(model, d)

            obs_np = env._get_obs_single(d)
            obs_t = torch.from_numpy(obs_np).float().unsqueeze(0).to(device)
            obs_td = TensorDict({"obs": obs_t}, batch_size=[1])

            with torch.no_grad():
                v = critic(obs_td).item()
            value_grid[iy, ix] = v

    fig, ax = plt.subplots(figsize=(8, 6))
    im = ax.imshow(value_grid, origin="lower", aspect="auto",
                   extent=[x_range[0], x_range[1], y_range[0], y_range[1]],
                   cmap="RdYlGn")
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label("V(s) at home pose")

    # Mark arm bases
    ax.plot(0, 0, "k^", markersize=12, label="arm1 base")
    ax.plot(0, -0.57, "ks", markersize=12, label="arm2 base")
    ax.set_xlabel("Cube x (m)")
    ax.set_ylabel("Cube y (m)")
    ax.set_title("Value Heatmap: Where does the critic think grasping is easy?")
    ax.legend(loc="upper right")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "value_heatmap.png"), dpi=150)
    plt.close()
    print(f"Value heatmap saved to {out_dir}/value_heatmap.png")


def main():
    parser = argparse.ArgumentParser(description="Evaluate a trained cube-grasp policy")
    parser.add_argument("--config", type=str,
                        default=os.path.join(os.path.dirname(__file__), "config.yaml"),
                        help="Path to testing config.yaml")
    args = parser.parse_args()

    # Load testing config
    eval_cfg = load_config(args.config)
    device = eval_cfg.get("device", "cpu")
    checkpoint = eval_cfg["checkpoint"]
    episodes = eval_cfg.get("episodes", 50)
    do_render = eval_cfg.get("render", True) and _RENDER
    out_dir = eval_cfg.get("out_dir",
                           os.path.join(os.path.dirname(__file__), "..", "results"))

    # Resolve paths relative to testing config dir
    cfg_dir = os.path.dirname(os.path.abspath(args.config))
    if not os.path.isabs(checkpoint):
        checkpoint = os.path.join(cfg_dir, checkpoint)
    training_config = eval_cfg.get("training_config",
                                   os.path.join(cfg_dir, "..", "training", "config.yaml"))
    if not os.path.isabs(training_config):
        training_config = os.path.join(cfg_dir, training_config)
    if not os.path.isabs(out_dir):
        out_dir = os.path.join(cfg_dir, out_dir)

    # Load training config for env + network params
    train_cfg = load_config(training_config)
    net_cfg = train_cfg.get("network", {})

    # Create env
    eval_envs = min(episodes, 32)
    env = CubeGraspVecEnv(num_envs=eval_envs, device=device, cfg=train_cfg)

    obs = env.get_observations()
    obs_dim = obs["obs"].shape[-1]
    act_dim = env.num_actions

    # Build and load models
    actor, critic = build_models(obs_dim, act_dim, net_cfg, device)

    ckpt = torch.load(checkpoint, map_location=device, weights_only=True)
    actor.load_state_dict(ckpt["actor_state_dict"])
    critic.load_state_dict(ckpt["critic_state_dict"])
    actor.eval()
    critic.eval()

    iteration = ckpt.get("iteration", "?")
    print(f"Loaded checkpoint: {checkpoint} (iteration {iteration})")
    print(f"Obs dim: {obs_dim}, Act dim: {act_dim}")
    print(f"Actor params: {sum(p.numel() for p in actor.parameters()):,}")

    # Run evaluation episodes
    print(f"\nRunning {episodes} eval episodes ({eval_envs} parallel)...")
    results = run_eval_episodes(env, actor, critic, episodes, device)

    # Generate plots
    plot_results(results, out_dir)

    # Value heatmap — what does the critic think about different cube positions?
    print("\nGenerating value heatmap...")
    heatmap_env = CubeGraspVecEnv(num_envs=1, device=device, cfg=train_cfg)
    plot_value_heatmap(heatmap_env, critic, device, out_dir)

    # Render all episodes as a grid GIF
    max_steps = train_cfg["env"].get("max_episode_steps", 200)
    print(f"\nRecording {eval_envs} episodes to GIF...")
    gif_env = CubeGraspVecEnv(num_envs=eval_envs, device=device, cfg=train_cfg)
    render_episodes_to_gif(gif_env, actor, device, eval_envs, max_steps, out_dir)

    # Render one episode in viewer
    if do_render:
        print("\nRendering one episode in MuJoCo viewer...")
        render_env = CubeGraspVecEnv(num_envs=1, device=device, cfg=train_cfg)
        render_policy(render_env, actor, device, max_steps=max_steps)


if __name__ == "__main__":
    main()

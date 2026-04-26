"""Vectorized MuJoCo environment for cube grasping with dual YAM arms.

Both arms are symmetric — the policy can grab with either or both.

Observation (45-dim):
  Per arm (x2 = 32):
    - 6 joint positions
    - 6 joint velocities
    - 1 gripper opening
    - 3 cube-to-grasp relative vector
  Shared (13):
    - 3 cube position (world)
    - 4 cube orientation (quat)
    - 3 gripper z-axis arm1 (world)
    - 3 gripper z-axis arm2 (world)

Action (14-dim):
  - 6 joint position deltas (arm1)
  - 1 gripper target arm1 (0=closed, 1=open)
  - 6 joint position deltas (arm2)
  - 1 gripper target arm2 (0=closed, 1=open)

Reward stages (each gated on the previous):
  1. Reach       — always active, get grasp site near the cube
  2. Align       — always active, point gripper downward
  3. Close grip  — gated on dist < threshold, reward closing the gripper
  4. Contact     — gated on gripper closing, reward any tip touching the cube
  5. Grasp       — gated on both tips contacting AND cube between tips
  6. Lift        — gated on grasp, reward lifting the cube
"""

from __future__ import annotations

import os
import numpy as np
import torch
import mujoco
from tensordict import TensorDict

MJCF_PATH = os.path.join(os.path.dirname(__file__), "..", "models", "dual_yam.xml")

ARM_PREFIXES = ["arm1", "arm2"]
GRIPPER_MAX_OPENING = 0.0475


class CubeGraspVecEnv:
    """Vectorized cube grasping environment for PPO training."""

    def __init__(self, num_envs: int, device: str = "cpu", cfg: dict | None = None):
        self.device = device
        self._num_envs = num_envs
        self._num_actions = 14  # (6 joints + 1 gripper) x 2 arms

        # Unpack config
        env_cfg = (cfg or {}).get("env", {})
        rew_cfg = (cfg or {}).get("reward", {})

        self.max_episode_length = env_cfg.get("max_episode_steps", 200)
        self.sim_dt = env_cfg.get("sim_dt", 0.002)
        self.ctrl_dt = env_cfg.get("ctrl_dt", 0.02)
        self.sim_steps_per_ctrl = max(1, round(self.ctrl_dt / self.sim_dt))
        self.action_scale = env_cfg.get("action_scale", 0.05)
        self.cube_x_range = tuple(env_cfg.get("cube_x_range", [0.10, 0.35]))
        self.cube_y_range = tuple(env_cfg.get("cube_y_range", [-0.57, 0.0]))
        self.cube_z = 0.01

        # Reward weights (staged)
        self.w_reach = rew_cfg.get("w_reach", 1.0)
        self.w_align = rew_cfg.get("w_align", 0.3)
        self.w_close = rew_cfg.get("w_close", 0.5)
        self.w_contact = rew_cfg.get("w_contact", 1.0)
        self.w_grasp = rew_cfg.get("w_grasp", 3.0)
        self.w_lift = rew_cfg.get("w_lift", 4.0)
        self.c_action = rew_cfg.get("c_action", 0.01)
        self.c_time = rew_cfg.get("c_time", 0.005)
        self.bonus_lift = rew_cfg.get("bonus_lift", 10.0)
        self.reach_tanh_scale = rew_cfg.get("reach_tanh_scale", 20.0)
        self.close_threshold = rew_cfg.get("close_threshold", 0.08)
        self.lift_target = rew_cfg.get("lift_target", 0.08)
        self.c_off_arm = rew_cfg.get("c_off_arm", 0.05)

        # Load model
        self.model = mujoco.MjModel.from_xml_path(MJCF_PATH)
        self.model.opt.timestep = self.sim_dt
        self.data_list = [mujoco.MjData(self.model) for _ in range(num_envs)]

        # Resolve IDs per arm
        self.arms = {}
        for prefix in ARM_PREFIXES:
            joint_ids = [mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, f"{prefix}_joint{i}") for i in range(1, 7)]
            self.arms[prefix] = {
                "joint_ids": joint_ids,
                "act_ids": [mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, f"{prefix}_act{i}") for i in range(1, 7)],
                "grip_act_id": mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, f"{prefix}_grip"),
                "grasp_site_id": mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, f"{prefix}_grasp_site"),
                "qpos_addrs": [self.model.jnt_qposadr[j] for j in joint_ids],
                "dof_addrs": [self.model.jnt_dofadr[j] for j in joint_ids],
                "grip_qpos_addr": self.model.jnt_qposadr[
                    mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, f"{prefix}_gripper_left")
                ],
                "joint_lo": np.array([self.model.jnt_range[j, 0] for j in joint_ids]),
                "joint_hi": np.array([self.model.jnt_range[j, 1] for j in joint_ids]),
                "tip_left_geom": mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, f"{prefix}_finger_left"),
                "tip_right_geom": mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, f"{prefix}_finger_right"),
                "tip_left_body": mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, f"{prefix}_tip_left"),
                "tip_right_body": mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, f"{prefix}_tip_right"),
            }

        # Block IDs
        self.block_joint_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, "block_joint")
        self.block_qpos_addr = self.model.jnt_qposadr[self.block_joint_id]
        self.block_geom = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, "block_geom")
        self.block_body = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "block")

        # Episode tracking
        self.episode_length_buf = torch.zeros(num_envs, dtype=torch.long, device=device)
        self.ep_max_cube_z = np.full(num_envs, -np.inf)
        self.ep_min_dist = np.full(num_envs, np.inf)
        self.ep_reward_sum = np.zeros(num_envs)

        # Rolling metrics
        self._completed_eps = 0
        self._completed_successes = 0
        self._completed_returns_sum = 0.0
        self._completed_min_dists_sum = 0.0

        self._reset_all()

    @property
    def num_envs(self) -> int:
        return self._num_envs

    @property
    def num_actions(self) -> int:
        return self._num_actions

    def _reset_env(self, env_idx: int):
        d = self.data_list[env_idx]
        mujoco.mj_resetData(self.model, d)

        cx = np.random.uniform(*self.cube_x_range)
        cy = np.random.uniform(*self.cube_y_range)
        ba = self.block_qpos_addr
        d.qpos[ba: ba + 3] = [cx, cy, self.cube_z]
        d.qpos[ba + 3: ba + 7] = [1, 0, 0, 0]

        for arm in self.arms.values():
            d.ctrl[arm["grip_act_id"]] = GRIPPER_MAX_OPENING

        mujoco.mj_forward(self.model, d)
        self.episode_length_buf[env_idx] = 0
        self.ep_max_cube_z[env_idx] = -np.inf
        self.ep_min_dist[env_idx] = np.inf
        self.ep_reward_sum[env_idx] = 0.0

    def _reset_all(self):
        for i in range(self._num_envs):
            self._reset_env(i)

    def _get_obs_single(self, d: mujoco.MjData) -> np.ndarray:
        parts = []
        ba = self.block_qpos_addr
        cube_pos = d.qpos[ba: ba + 3].copy()

        for prefix in ARM_PREFIXES:
            arm = self.arms[prefix]
            parts.append(np.array([d.qpos[a] for a in arm["qpos_addrs"]]))   # 6
            parts.append(np.array([d.qvel[a] for a in arm["dof_addrs"]]))    # 6
            parts.append(np.array([d.qpos[arm["grip_qpos_addr"]]]))         # 1
            grasp_pos = d.site_xpos[arm["grasp_site_id"]].copy()
            parts.append(cube_pos - grasp_pos)                               # 3

        parts.append(cube_pos)                                               # 3
        parts.append(d.qpos[ba + 3: ba + 7].copy())                         # 4
        for prefix in ARM_PREFIXES:
            arm = self.arms[prefix]
            grip_rot = d.site_xmat[arm["grasp_site_id"]].reshape(3, 3)
            parts.append(grip_rot[:, 2].copy())                              # 3

        return np.concatenate(parts)  # 45-dim

    def get_observations(self) -> TensorDict:
        obs_np = np.stack([self._get_obs_single(d) for d in self.data_list])
        obs_tensor = torch.from_numpy(obs_np).float().to(self.device)
        return TensorDict({"obs": obs_tensor}, batch_size=[self._num_envs])

    def _check_tip_contacts(self, d: mujoco.MjData, arm: dict) -> tuple[bool, bool]:
        """Check if left/right gripper tips are in contact with the block."""
        left_c = False
        right_c = False
        for ci in range(d.ncon):
            c = d.contact[ci]
            g1, g2 = c.geom1, c.geom2
            if (g1 == arm["tip_left_geom"] and g2 == self.block_geom) or \
               (g2 == arm["tip_left_geom"] and g1 == self.block_geom):
                left_c = True
            if (g1 == arm["tip_right_geom"] and g2 == self.block_geom) or \
               (g2 == arm["tip_right_geom"] and g1 == self.block_geom):
                right_c = True
        return left_c, right_c

    def _is_cube_between_tips(self, d: mujoco.MjData, arm: dict) -> bool:
        """Check if the cube center is between the two gripper tips.

        Projects the cube position onto the line connecting the two tips.
        If the projection falls between them (0 < t < 1), the cube is "inside" the grip.
        """
        tip_l_pos = d.xpos[arm["tip_left_body"]]
        tip_r_pos = d.xpos[arm["tip_right_body"]]
        cube_pos = d.xpos[self.block_body]

        # Vector from left tip to right tip
        lr = tip_r_pos - tip_l_pos
        lr_len_sq = np.dot(lr, lr)
        if lr_len_sq < 1e-10:
            return False

        # Project cube onto the line
        t = np.dot(cube_pos - tip_l_pos, lr) / lr_len_sq
        return 0.0 < t < 1.0

    def _compute_arm_reward(self, d: mujoco.MjData, arm: dict, cube_pos: np.ndarray, cube_z: float) -> tuple[float, float]:
        """Compute staged reward for one arm. Returns (reward, dist)."""

        grasp_pos = d.site_xpos[arm["grasp_site_id"]]
        dist = np.linalg.norm(grasp_pos - cube_pos)
        grip_opening = d.qpos[arm["grip_qpos_addr"]]
        grip_closed_frac = 1.0 - (grip_opening / GRIPPER_MAX_OPENING)  # 0=open, 1=closed

        # Stage 1: Reach — always active
        r_reach = 1.0 - np.tanh(self.reach_tanh_scale * dist)

        # Stage 2: Align — always active, gripper z-axis should point down
        grip_rot = d.site_xmat[arm["grasp_site_id"]].reshape(3, 3)
        grip_z = grip_rot[:, 2]
        align_score = max(0.0, -grip_z[2])  # 1.0 when pointing straight down
        r_align = align_score

        # Stage 3: Close grip — gated on proximity AND alignment AND cube below grasp
        r_close = 0.0
        if dist < self.close_threshold and align_score > 0.5:
            # Check cube is below the grasp site (not to the side)
            cube_below = grasp_pos[2] - cube_pos[2]  # positive when grasp is above cube
            if cube_below > 0:
                proximity = 1.0 - (dist / self.close_threshold)
                r_close = grip_closed_frac * proximity * align_score

        # Stage 4: Contact — gated on gripper partially closed (> 20% closed)
        left_c, right_c = self._check_tip_contacts(d, arm)
        r_contact = 0.0
        if grip_closed_frac > 0.2:
            if left_c and right_c:
                r_contact = 1.0
            elif left_c or right_c:
                r_contact = 0.3

        # Stage 5: Grasp — gated on both tips contacting AND cube between tips
        r_grasp = 0.0
        is_grasped = False
        if left_c and right_c and self._is_cube_between_tips(d, arm):
            r_grasp = 1.0
            is_grasped = True

        # Stage 6: Lift — gated on grasp
        r_lift = 0.0
        bonus = 0.0
        if is_grasped:
            lift = (cube_z - self.cube_z) / self.lift_target
            r_lift = np.clip(lift, 0.0, 1.0)
            if cube_z > self.cube_z + self.lift_target:
                bonus = self.bonus_lift

        reward = (
            self.w_reach * r_reach
            + self.w_align * r_align
            + self.w_close * r_close
            + self.w_contact * r_contact
            + self.w_grasp * r_grasp
            + self.w_lift * r_lift
            + bonus
        )
        return reward, dist

    def step(self, actions: torch.Tensor):
        """Step all envs. Actions: (num_envs, 14)."""
        actions_np = actions.cpu().numpy()
        rewards = np.zeros(self._num_envs)
        dones = np.zeros(self._num_envs, dtype=bool)
        timeouts = np.zeros(self._num_envs, dtype=bool)

        for i, d in enumerate(self.data_list):
            act = actions_np[i]

            # Apply actions to both arms
            offset = 0
            for prefix in ARM_PREFIXES:
                arm = self.arms[prefix]
                joint_deltas = act[offset: offset + 6] * self.action_scale
                grip_target = np.clip(act[offset + 6], 0, 1) * GRIPPER_MAX_OPENING

                for j_idx, aid in enumerate(arm["act_ids"]):
                    d.ctrl[aid] = np.clip(
                        d.ctrl[aid] + joint_deltas[j_idx],
                        arm["joint_lo"][j_idx], arm["joint_hi"][j_idx],
                    )
                d.ctrl[arm["grip_act_id"]] = grip_target
                offset += 7

            # Step simulation
            for _ in range(self.sim_steps_per_ctrl):
                mujoco.mj_step(self.model, d)

            # --- Staged reward (best of either arm) ---
            ba = self.block_qpos_addr
            cube_pos = d.qpos[ba: ba + 3]
            cube_z = cube_pos[2]

            r1, d1 = self._compute_arm_reward(d, self.arms["arm1"], cube_pos, cube_z)
            r2, d2 = self._compute_arm_reward(d, self.arms["arm2"], cube_pos, cube_z)

            if r1 >= r2:
                best_arm_reward = r1
                off_arm_prefix = "arm2"
                off_act_slice = slice(7, 13)  # arm2 joint actions
            else:
                best_arm_reward = r2
                off_arm_prefix = "arm1"
                off_act_slice = slice(0, 6)   # arm1 joint actions

            # Action penalty on active arm only
            active_act_slice = slice(0, 6) if r1 >= r2 else slice(7, 13)
            action_penalty = self.c_action * np.sum(act[active_act_slice] ** 2)

            # Off-arm regularization: penalize deviation from home pose (all zeros)
            off_arm = self.arms[off_arm_prefix]
            off_joint_pos = np.array([d.qpos[a] for a in off_arm["qpos_addrs"]])
            off_arm_penalty = self.c_off_arm * np.sum(off_joint_pos ** 2)

            # Also penalize off-arm actions (should output zero)
            off_arm_penalty += self.c_off_arm * np.sum(act[off_act_slice] ** 2)

            rewards[i] = best_arm_reward - action_penalty - off_arm_penalty - self.c_time

            # Track per-episode metrics
            self.ep_max_cube_z[i] = max(self.ep_max_cube_z[i], cube_z)
            best_dist = min(d1, d2)
            self.ep_min_dist[i] = min(self.ep_min_dist[i], best_dist)
            self.ep_reward_sum[i] += rewards[i]

            self.episode_length_buf[i] += 1
            if self.episode_length_buf[i] >= self.max_episode_length:
                dones[i] = True
                timeouts[i] = True
            if cube_z < -0.1:
                dones[i] = True

        # Accumulate completed episode stats before resetting
        for i in range(self._num_envs):
            if dones[i]:
                self._completed_eps += 1
                self._completed_returns_sum += self.ep_reward_sum[i]
                self._completed_min_dists_sum += self.ep_min_dist[i]
                if self.ep_max_cube_z[i] > self.cube_z + self.lift_target:
                    self._completed_successes += 1
                self._reset_env(i)

        obs = self.get_observations()
        rewards_t = torch.from_numpy(rewards).float().to(self.device)
        dones_t = torch.from_numpy(dones).float().to(self.device)
        extras = {
            "time_outs": torch.from_numpy(timeouts).float().to(self.device),
            "log": {},
        }
        return obs, rewards_t, dones_t, extras

    def get_metrics_and_reset(self) -> dict:
        """Return rolling episode metrics since last call, then reset counters."""
        n = self._completed_eps
        if n == 0:
            return {}
        metrics = {
            "episodes": n,
            "success_rate": self._completed_successes / n,
            "mean_return": self._completed_returns_sum / n,
            "mean_min_dist": self._completed_min_dists_sum / n,
        }
        self._completed_eps = 0
        self._completed_successes = 0
        self._completed_returns_sum = 0.0
        self._completed_min_dists_sum = 0.0
        return metrics

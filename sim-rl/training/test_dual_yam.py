#!/usr/bin/env python
"""
Interactive MuJoCo viewer for the dual YAM bimanual setup.
End-effector (Cartesian) control via Jacobian IK + linear_4310 gripper.

Controls:
  W/S:   TCP forward / backward  (x)
  A/D:   TCP left / right        (y)
  Q/E:   TCP up / down           (z)
  Tab:   switch between arm1 and arm2
  Space: toggle gripper open/close
  R:     reset all to home

Launch:
  python sim-rl/test_dual_yam.py
"""

import sys
import os
import platform

# Auto-relaunch with mjpython on macOS
if platform.system() == "Darwin" and "MJPYTHON" not in os.environ:
    import shutil
    mjpython = shutil.which("mjpython")
    if mjpython is None:
        print("ERROR: mjpython not found. Install with: pip install mujoco")
        sys.exit(1)
    os.environ["MJPYTHON"] = "1"
    os.execv(mjpython, [mjpython] + sys.argv)

import mujoco
import mujoco.viewer
import numpy as np
import time

MJCF_PATH = os.path.join(os.path.dirname(__file__), "..", "models", "dual_yam.xml")

CART_STEP = 0.01
DAMPING = 1e-4
GRIPPER_OPEN = 0.0475   # fully open (max slide)
GRIPPER_CLOSED = 0.0    # fully closed


def solve_ik(model, data, site_id, delta_pos, arm_joint_ids):
    """Compute joint delta via damped least-squares IK for position only."""
    jacp = np.zeros((3, model.nv))
    mujoco.mj_jacSite(model, data, jacp, None, site_id)
    dof_ids = [model.jnt_dofadr[j] for j in arm_joint_ids]
    J = jacp[:, dof_ids]
    JJT = J @ J.T + DAMPING * np.eye(3)
    dq = J.T @ np.linalg.solve(JJT, delta_pos)
    return dq


def main():
    model = mujoco.MjModel.from_xml_path(MJCF_PATH)
    data = mujoco.MjData(model)

    arms = {}
    for prefix in ["arm1", "arm2"]:
        joint_ids = [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, f"{prefix}_joint{i}") for i in range(1, 7)]
        act_ids = [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, f"{prefix}_act{i}") for i in range(1, 7)]
        tcp_site = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, f"{prefix}_tcp_site")
        grip_act = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, f"{prefix}_grip")
        arms[prefix] = {
            "joint_ids": joint_ids,
            "act_ids": act_ids,
            "tcp_site": tcp_site,
            "grip_act": grip_act,
            "gripper_open": True,
        }

    active_arm = "arm1"
    pending_delta = np.zeros(3)

    # Start grippers open
    for arm in arms.values():
        data.ctrl[arm["grip_act"]] = GRIPPER_OPEN

    def get_tcp_pos():
        return data.site_xpos[arms[active_arm]["tcp_site"]].copy()

    def print_status():
        pos = get_tcp_pos()
        grip = "OPEN" if arms[active_arm]["gripper_open"] else "CLOSED"
        print(f"  [{active_arm}] TCP: x={pos[0]:.3f} y={pos[1]:.3f} z={pos[2]:.3f}  gripper={grip}")

    def key_callback(keycode):
        nonlocal active_arm, pending_delta

        if keycode == 87:    pending_delta[0] += CART_STEP   # W
        elif keycode == 83:  pending_delta[0] -= CART_STEP   # S
        elif keycode == 65:  pending_delta[1] += CART_STEP   # A
        elif keycode == 68:  pending_delta[1] -= CART_STEP   # D
        elif keycode == 81:  pending_delta[2] += CART_STEP   # Q
        elif keycode == 69:  pending_delta[2] -= CART_STEP   # E

        elif keycode == 258:  # Tab
            active_arm = "arm2" if active_arm == "arm1" else "arm1"
            print_status()

        elif keycode == 32:  # Space — toggle gripper
            arm = arms[active_arm]
            arm["gripper_open"] = not arm["gripper_open"]
            data.ctrl[arm["grip_act"]] = GRIPPER_OPEN if arm["gripper_open"] else GRIPPER_CLOSED
            print_status()

        elif keycode == 82:  # R — reset
            data.qpos[:] = 0
            data.qvel[:] = 0
            data.ctrl[:] = 0
            pending_delta[:] = 0
            for a in arms.values():
                a["gripper_open"] = True
                data.ctrl[a["grip_act"]] = GRIPPER_OPEN
            mujoco.mj_forward(model, data)
            print("  Reset to home")

    print("Dual YAM — End-Effector Control + Gripper")
    print("  W/S: x | A/D: y | Q/E: z | Tab: switch arm | Space: gripper | R: reset")

    mujoco.mj_forward(model, data)
    print_status()

    with mujoco.viewer.launch_passive(model, data, key_callback=key_callback) as viewer:
        while viewer.is_running():
            if np.any(pending_delta != 0):
                arm = arms[active_arm]
                dq = solve_ik(model, data, arm["tcp_site"], pending_delta, arm["joint_ids"])
                for i, aid in enumerate(arm["act_ids"]):
                    jid = arm["joint_ids"][i]
                    lo = model.jnt_range[jid, 0]
                    hi = model.jnt_range[jid, 1]
                    data.ctrl[aid] = np.clip(data.ctrl[aid] + dq[i], lo, hi)
                pending_delta[:] = 0
                print_status()

            mujoco.mj_step(model, data)
            viewer.sync()
            time.sleep(model.opt.timestep)


if __name__ == "__main__":
    main()

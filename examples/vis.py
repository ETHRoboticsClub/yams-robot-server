"""Lightweight MuJoCo visualizer for a single YAM follower arm.

Usage:
    uv run python vis.py left          # connect to left arm server
    uv run python vis.py right         # connect to right arm server
    uv run python vis.py left --dummy  # offline / no arm connected
    uv run python vis.py left --angles 0.5 1.2 0.8 0 0 0
"""

import argparse
import time
from pathlib import Path

import mujoco
import mujoco.viewer
import numpy as np
import yaml

from i2rt.robots.utils import GripperType
from lerobot_robot_yams.forward_kinematics import check_action

PORTS = {"left": 11333, "right": 11334}
DT = 0.02  # 50 Hz

_ARMS_CONFIG = Path(__file__).resolve().parents[1] / "configs" / "arms.yaml"

# RGBA: safe=green, collision=red
_SAFE  = np.array([0.2, 0.8, 0.2, 1.0])
_CLASH = np.array([1.0, 0.2, 0.2, 1.0])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("side", choices=["left", "right"])
    parser.add_argument("--dummy", action="store_true", help="Run without a connected arm")
    parser.add_argument("--angles", type=float, nargs="+", help="Fixed joint angles in radians")
    args = parser.parse_args()

    collision_cfg = yaml.safe_load(_ARMS_CONFIG.read_text()).get("collision", {})
    ground_z = collision_cfg.get("ground_z", 0.05)
    end_effector_length = collision_cfg.get("end_effector_length", 0.15)
    max_joint_step = np.array(collision_cfg.get("max_joint_step", [0.2] * 6))

    xml = GripperType.LINEAR_3507.get_xml_path()
    model = mujoco.MjModel.from_xml_path(xml)
    data = mujoco.MjData(model)

    if args.angles:
        joints = np.zeros(model.nq)
        joints[: len(args.angles)] = args.angles
        get_joints = lambda: joints
    elif args.dummy:
        get_joints = lambda: np.zeros(model.nq)
    else:
        import portal
        client = portal.Client(PORTS[args.side])
        get_joints = client.get_joint_pos

    with mujoco.viewer.launch_passive(
        model=model, data=data, show_left_ui=False, show_right_ui=False
    ) as viewer:
        mujoco.mjv_defaultFreeCamera(model, viewer.cam)

        # Add a ground plane to the user scene so we can see ground_z
        def add_ground(scn: mujoco.MjvScene) -> None:
            if scn.ngeom >= scn.maxgeom:
                return
            g = scn.geoms[scn.ngeom]
            mujoco.mjv_initGeom(g, mujoco.mjtGeom.mjGEOM_PLANE, np.zeros(3), np.zeros(3), np.eye(3).flatten(), np.array([0.5, 0.5, 0.5, 0.3]))
            g.size[:] = [0.5, 0.5, 0.001]
            g.pos[:] = [0, 0, ground_z]
            g.mat[:] = np.eye(3)
            scn.ngeom += 1

        last_collision = None
        while viewer.is_running():
            t = time.monotonic()
            joints = get_joints()
            data.qpos[: model.nq] = joints[: model.nq]
            mujoco.mj_kinematics(model, data)

            collision = check_action(joints[:6], None, ground_z, end_effector_length, max_joint_step)
            color = _CLASH if collision else _SAFE
            model.geom_rgba[:] = color  # tint all geoms

            if collision != last_collision:
                print("GROUND COLLISION" if collision else "clear")
                last_collision = collision

            with viewer.lock():
                viewer.user_scn.ngeom = 0
                add_ground(viewer.user_scn)

            viewer.sync()
            elapsed = time.monotonic() - t
            if elapsed < DT:
                time.sleep(DT - elapsed)


if __name__ == "__main__":
    main()

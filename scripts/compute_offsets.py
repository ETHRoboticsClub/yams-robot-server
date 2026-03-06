import argparse
from pathlib import Path

import yaml

from lerobot_teleoperator_gello.leader import YamsLeader, YamsLeaderConfig

ARMS_CONFIG_PATH = Path(__file__).resolve().parents[1] / "configs" / "arms.yaml"


def load_scales(arm: str) -> dict:
    with open(ARMS_CONFIG_PATH, "r") as f:
        motors = yaml.safe_load(f)["leader"][f"{arm}_arm"]["motors"]
    return {
        name: cfg["calibration_scale"]
        for name, cfg in motors.items()
        if name != "gripper"
    }


def compute_offsets(
    leader: YamsLeader,
    arm: str,
) -> dict:
    """
    Compute offsets from current position to neutral position.

    Args:
        leader: Connected YamsLeader instance
        neutral_position: Target neutral position for each joint (in raw encoder values).
                         If None, uses 2048 (center position) for all joints.

    Returns:
        Dictionary containing offsets and scales for each joint
    """
    neutral_position = {
        "joint_1": 2048,
        "joint_2": 2048,
        "joint_3": 2048,
        "joint_4": 2048,
        "joint_5": 2048,
        "joint_6": 2048,
    }

    current_position = leader.bus.sync_read(
        normalize=False, data_name="Present_Position"
    )

    # Compute offsets
    offsets = {}
    for joint, target_pos in neutral_position.items():
        if joint in current_position:
            offset = target_pos - current_position[joint]
            offsets[joint] = offset
            print(f"\n{joint}:")
            print(f"  Current: {current_position[joint]}")
            print(f"  Target:  {target_pos}")
            print(f"  Offset:  {offset}")

    # Compile calibration data
    calibration = {
        "offsets": offsets,
        "scales": load_scales(arm),
    }

    return calibration


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--arm",
        choices=["left", "right"],
        help="Arm to calibrate. If omitted, calibrates both arms.",
    )
    args = parser.parse_args()

    port_by_arm = {"left": "/dev/ttyUSB0", "right": "/dev/ttyUSB1"}
    arms = [args.arm] if args.arm else ["left", "right"]

    for arm in arms:
        leader_config = YamsLeaderConfig(port=port_by_arm[arm], side=arm)
        leader = YamsLeader(leader_config)
        leader.connect()
        output_path = Path(
            f"src/lerobot_teleoperator_gello/calibration/leader_calibration_{arm}.yaml"
        )
        output_path.parent.mkdir(parents=True, exist_ok=True)
        print(f"\nSaving calibration to {output_path}...")
        with open(output_path, "w") as f:
            yaml.dump(
                compute_offsets(leader, arm), f, default_flow_style=False, sort_keys=False
            )
        leader.disconnect()


if __name__ == "__main__":
    main()

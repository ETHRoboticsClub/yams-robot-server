from pathlib import Path

import yaml

from lerobot_teleoperator_gello.leader import YamsLeader, YamsLeaderConfig


def compute_offsets(
    leader: YamsLeader,
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
        "scales": {
            "joint_1": 1.0,
            "joint_2": -1.0,
            "joint_3": 1.0,
            "joint_4": 1.0,
            "joint_5": 1.0,
            "joint_6": 1.0,
        },
    }

    return calibration


def main():
    leader_config = YamsLeaderConfig(port="/dev/ttyACM1", side="right")
    leader = YamsLeader(leader_config)
    leader.connect()

    # Save to YAML file
    output_path = Path(
        f"src/lerobot_teleoperator_gello/calibration/leader_calibration_{leader_config.side}.yaml"
    )
    print(f"\nSaving calibration to {output_path}...")
    with open(output_path, "w") as f:
        yaml.dump(compute_offsets(leader), f, default_flow_style=False, sort_keys=False)
    leader.disconnect()


if __name__ == "__main__":
    main()

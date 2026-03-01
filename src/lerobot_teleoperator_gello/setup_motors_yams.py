import sys

try:
    from lerobot_teleoperator_gello import BiYamsLeader, YamsLeader
    from lerobot_robot_yams import BiYamsFollower, YamsFollower

    print("[INFO] Custom YAMS plugins loaded and registered", file=sys.stderr)
except ImportError as e:
    print(f"[WARNING] Could not load custom YAMS plugins: {e}", file=sys.stderr)


def main() -> None:
    from lerobot.scripts import lerobot_setup_motors

    for name in ["yams_leader", "bi_yams_leader", "yams_follower", "bi_yams_follower"]:
        if name not in lerobot_setup_motors.COMPATIBLE_DEVICES:
            lerobot_setup_motors.COMPATIBLE_DEVICES.append(name)

    lerobot_setup_motors.main()

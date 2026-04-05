import time
from pathlib import Path

import numpy as np
import portal
import yaml


def slow_move(client: portal.Client, duration: float = 2.0, freq: float = 200.0) -> None:
    obs = client.get_observations().result()
    current = np.concatenate([obs["joint_pos"], obs.get("gripper_pos", np.array([]))])
    target = np.zeros_like(current)
    steps = max(int(duration * freq), 1)
    for i in range(steps + 1):
        alpha = i / steps
        client.command_joint_pos((1 - alpha) * current + alpha * target)
        time.sleep(1 / freq)


def main():
    config = yaml.safe_load((Path(__file__).resolve().parents[2] / "configs" / "arms.yaml").read_text())
    ports = [
        int(config["follower"]["left_arm"]["server_port"]),
        int(config["follower"]["right_arm"]["server_port"]),
    ]
    clients = [portal.Client(f"localhost:{port}") for port in ports]
    for client in clients:
        client.get_robot_info().result()
    for client in clients:
        slow_move(client)
    for client in clients:
        client.close()


if __name__ == "__main__":
    main()

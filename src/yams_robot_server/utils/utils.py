import time

import numpy as np

from yams_robot_server.follower import YamsFollower


def slow_move(
    follower: YamsFollower,
    leader_joint_pos: dict[str, float],
    duration: float = 1.0,
    freq: float = 200.0,
) -> None:
    obs = follower.get_observation()
    follower_joint_names = follower.config.joint_names

    current_pos = np.array([obs[f"{name}.pos"] for name in follower_joint_names])
    target_pos = np.array(
        [leader_joint_pos[f"{name}.pos"] for name in follower_joint_names]
    )

    n_steps = max(int(duration * freq), 1)
    for t in range(n_steps + 1):
        alpha = t / n_steps
        interp_pos = (1 - alpha) * current_pos + alpha * target_pos
        follower.send_action(
            {
                f"{name}.pos": float(pos)
                for name, pos in zip(follower_joint_names, interp_pos)
            }
        )
        time.sleep(1 / freq)

from yams_robot_server.bi_leader import BiYamsLeaderConfig, BiYamsLeader
import time

bi_leader_config = BiYamsLeaderConfig(
    left_arm_port="/dev/ttyACM0",
    right_arm_port="/dev/ttyACM1",
)

bi_leader = BiYamsLeader(bi_leader_config)
bi_leader.connect()

freq = 200  # Hz

try:
    while True:
        bi_leader_action = bi_leader.get_action()
        print({key: f"{value:.2f}" for key, value in bi_leader_action.items()})
        time.sleep(1 / freq)
except KeyboardInterrupt:
    print("\nStopping reading...")
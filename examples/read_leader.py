import time

from yams_robot_server.leader import YamsLeader, YamsLeaderConfig

leader_config = YamsLeaderConfig(port="/dev/ttyACM1", side="left")

leader = YamsLeader(leader_config)
leader.connect()


while True:
    action = leader.get_action()
    print({key: f"{value:.2f}" for key, value in action.items()})
    time.sleep(0.02)

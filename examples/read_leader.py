import gc
import time

from lerobot_teleoperator_gello.leader import YamsLeader, YamsLeaderConfig

gc.disable()

leader_config = YamsLeaderConfig(port="/dev/ttyACM1", side="left")

leader = YamsLeader(leader_config)
leader.connect()


while True:
    action = leader.get_action()
    print({key: f"{value:.2f}" for key, value in action.items()})
    time.sleep(0.02)

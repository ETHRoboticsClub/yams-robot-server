import time

from lerobot_teleoperator_gello.bi_leader import BiYamsLeader, BiYamsLeaderConfig

bi_leader_config = BiYamsLeaderConfig(
    left_arm_port="/dev/ttyUSB0",
    right_arm_port="/dev/ttyUSB1",
)

bi_leader = BiYamsLeader(bi_leader_config)
bi_leader.connect()


start_time = time.time()
count = 0
try:
    while True:
        count += 1
        bi_leader_action = bi_leader.get_action()
        if count % 400 == 0:
            time_elapsed = (time.time() - start_time)
            print(f"Hz {time_elapsed/400*1000:.6f} HZ")
            start_time = time.time()
            count = 0
except KeyboardInterrupt:
    print("\nStopping reading...")

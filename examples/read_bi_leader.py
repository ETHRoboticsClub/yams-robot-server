import time

from lerobot_teleoperator_gello.bi_leader import BiYamsLeader, BiYamsLeaderConfig

bi_leader_config = BiYamsLeaderConfig(
    left_arm_port="/dev/ttyACM0",
    right_arm_port="/dev/ttyACM1",
)

bi_leader = BiYamsLeader(bi_leader_config)
bi_leader.connect()

freq = 200  # Hz

start_time = time.time()
count = 0
try:
    while True:
        count += 1
        bi_leader_action = bi_leader.get_action()
        time.sleep(1 / freq)
        time_elapsed = time.time() - start_time
        if count % 400 == 0:
            print(f"elapsed time iterations: {time_elapsed:.6f} seconds")
        if time_elapsed >= 0.1:
            print(f"Max elapsed time larger then 100ms: {time_elapsed:.2f} seconds")
        start_time = time.time()
except KeyboardInterrupt:
    print("\nStopping reading...")

import time

from yams_robot_server.follower import YamsFollower, YamsFollowerConfig

follower_config = YamsFollowerConfig(
    port="can0",
)
follower = YamsFollower(follower_config)

follower.connect()

hz = 100  # Hz

try:
    while True:
        print(
            {key: f"{value:.2f}" for key, value in follower.get_observation().items()}
        )
        time.sleep(1 / hz)

except KeyboardInterrupt:
    print("\nStopping read position...")
    follower.disconnect()

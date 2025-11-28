import time

from yams_robot_server.follower import YamsFollower, YamsFollowerConfig

follower_config = YamsFollowerConfig(
    port="can0",
)
follower = YamsFollower(follower_config)

follower.connect()

try:
    while True:
        print(
            {key: f"{value:.2f}" for key, value in follower.get_observation().items()}
        )
        time.sleep(0.01)

except KeyboardInterrupt:
    print("\nStopping read position...")
    follower.disconnect()

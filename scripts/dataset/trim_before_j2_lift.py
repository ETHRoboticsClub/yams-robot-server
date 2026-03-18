import numpy as np
from lerobot.datasets.lerobot_dataset import LeRobotDataset

repo_id = "ETHRC/towelspring26"
seconds_before = 0.5
threshold = 0.3

ds = LeRobotDataset(repo_id)
fps = ds.fps
frames_before = round(seconds_before * fps)

print(f"num_episodes={ds.num_episodes}")

for ep in range(ds.num_episodes):
    ep_start = ds.meta.episodes["dataset_from_index"][ep]
    ep_end = ds.meta.episodes["dataset_to_index"][ep]

    rows = [ds[i] for i in range(ep_start, ep_end)]

    joint2 = np.array([row["observation.state"][1].item() for row in rows])
    hit = np.flatnonzero(joint2 > threshold)

    new_start = ep_start if len(hit) == 0 else max(ep_start, ep_start + int(hit[0]) - frames_before)
    print(f"ep {ep + 1}/{ds.num_episodes}: {ep} {ep_start} {ep_end} -> {new_start} {ep_end}")

print("done")

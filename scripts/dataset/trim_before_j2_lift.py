import numpy as np
from lerobot.datasets.lerobot_dataset import LeRobotDataset

repo_id = "ETHRC/towelspring26"
seconds_before = 0.5
threshold = 0.3

ds = LeRobotDataset(repo_id)
fps = ds.fps
frames_before = round(seconds_before * fps)


def format_ts(frame_idx: int) -> str:
    seconds = frame_idx / fps
    whole = int(seconds)
    millis = round((seconds - whole) * 1000)
    if millis == 1000:
        whole += 1
        millis = 0
    return f"{whole}.{millis:03d}s"

print(f"num_episodes={ds.num_episodes}", flush=True)

for ep in range(ds.num_episodes):
    ep_start = ds.meta.episodes["dataset_from_index"][ep]
    ep_end = ds.meta.episodes["dataset_to_index"][ep]

    hit = np.flatnonzero([
        ds[i]["observation.state"][1].item() > threshold
        for i in range(ep_start, ep_end)
    ])

    new_start = ep_start if len(hit) == 0 else max(ep_start, ep_start + int(hit[0]) - frames_before)
    local_start = 0
    local_end = ep_end - ep_start
    local_new_start = new_start - ep_start
    print(
        f"ep {ep + 1}/{ds.num_episodes}: {ep} {local_start} {local_end} -> "
        f"{local_new_start} {local_end} ({format_ts(local_new_start)})",
        flush=True,
    )

print("done", flush=True)

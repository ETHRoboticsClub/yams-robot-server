from lerobot.datasets.lerobot_dataset import LeRobotDataset

dataset = LeRobotDataset(
    repo_id="ETHRC/towel_folding_baseline",
    root="/home/ethrc/.cache/huggingface/lerobot/ETHRC/towel_folding_baseline_debug",
)


for episode in dataset:
    print(episode)
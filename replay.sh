

YAML=configs/arms.yaml
REPO=ETHRC/fake1
LEFT_PORT=$(yq '.leader.left_arm.port' "$YAML")
RIGHT_PORT=$(yq '.leader.right_arm.port' "$YAML")
cameras=$(yq -c '.cameras.configs' "$YAML")

lerobot-dataset-viz --dataset.root="$HOME/.cache/huggingface/lerobot/$REPO"  --episode-index 0 --mode distant --display-compressed-images true



    # --dataset.streaming_encoding=true \
    # --dataset.encoder_threads=2



YAML=configs/arms.yaml
REPO=ETHRC/fake4
LEFT_PORT=$(yq '.leader.left_arm.port' "$YAML")
RIGHT_PORT=$(yq '.leader.right_arm.port' "$YAML")
cameras=$(yq -c '.cameras.configs' "$YAML")

lerobot-dataset-viz --repo-id "$REPO" --episode-index 0 --mode local



    # --dataset.streaming_encoding=true \
    # --dataset.encoder_threads=2

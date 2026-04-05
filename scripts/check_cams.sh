#!/usr/bin/env bash
set -euo pipefail

shopt -s nullglob
CAMERAS=(/dev/v4l/by-path/*-video-index0)
[ ${#CAMERAS[@]} -eq 0 ] && CAMERAS=(/dev/video*)

for camera in "${CAMERAS[@]}"; do
    camera=$(readlink -f "$camera")
    ctrls=$(v4l2-ctl -d "$camera" --list-ctrls 2>/dev/null || true)
    if grep -q 'brightness' <<<"$ctrls" && grep -q 'exposure_time_absolute' <<<"$ctrls"; then
        ./scripts/set_camera_profile.sh "$camera"
    else
        echo "Skipping $camera: unsupported controls"
    fi
done

rm -rf outputs/captured_images
uv run lerobot-find-cameras

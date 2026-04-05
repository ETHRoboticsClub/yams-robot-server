#!/usr/bin/env bash
set -euo pipefail

if [ "$#" -lt 1 ]; then
  echo "Usage: $0 /dev/videoX"
  exit 1
fi

DEV="$1"

v4l2-ctl -d "$DEV" --set-ctrl=brightness=-20
v4l2-ctl -d "$DEV" --set-ctrl=contrast=28
v4l2-ctl -d "$DEV" --set-ctrl=saturation=64
v4l2-ctl -d "$DEV" --set-ctrl=hue=0
v4l2-ctl -d "$DEV" --set-ctrl=white_balance_automatic=1
v4l2-ctl -d "$DEV" --set-ctrl=gamma=100
v4l2-ctl -d "$DEV" --set-ctrl=gain=0
v4l2-ctl -d "$DEV" --set-ctrl=power_line_frequency=1
v4l2-ctl -d "$DEV" --set-ctrl=sharpness=3
v4l2-ctl -d "$DEV" --set-ctrl=backlight_compensation=0

v4l2-ctl -d "$DEV" --set-ctrl=auto_exposure=1
v4l2-ctl -d "$DEV" --set-ctrl=exposure_dynamic_framerate=1
v4l2-ctl -d "$DEV" --set-ctrl=exposure_time_absolute=100

echo "Applied camera profile to $DEV"
v4l2-ctl -d "$DEV" --list-ctrls
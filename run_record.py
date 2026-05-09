"""Wrapper around lerobot-record that registers the mimic_video policy first.

lerobot 0.5.0 doesn't auto-discover plugins via entry_points, so the adapter
must be imported before draccus parses argv.
"""

import mimic_adapter  # noqa: F401 -- registers MimicVideoConfig before argparse runs.

from lerobot.scripts.lerobot_record import main

if __name__ == "__main__":
    main()

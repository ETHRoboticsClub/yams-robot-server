import sys
sys.path.insert(0, "/home/ethrc/Desktop/mimic-video-repo/model")

import mimic_adapter  # registers MimicVideoConfig before argparse runs

from lerobot.scripts.lerobot_eval import main

if __name__ == "__main__":
    main()

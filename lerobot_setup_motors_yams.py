#!/usr/bin/env python
"""
Wrapper for lerobot-setup-motors that ensures custom YAMS plugins are discovered.
"""

import sys

# CRITICAL: Import custom modules BEFORE lerobot initializes to register the configs
try:
    # Import packages to trigger @register_subclass decorators
    from lerobot_teleoperator_gello import BiYamsLeader, YamsLeader
    from lerobot_robot_yams import BiYamsFollower, YamsFollower
    print("[INFO] Custom YAMS plugins loaded and registered", file=sys.stderr)
except ImportError as e:
    print(f"[WARNING] Could not load custom YAMS plugins: {e}", file=sys.stderr)

# Now run the original lerobot command
if __name__ == "__main__":
    from lerobot.scripts.lerobot_setup_motors import main
    main()

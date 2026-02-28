#!/usr/bin/env python
"""Test if lerobot can discover bi_yams_leader"""

import importlib
import importlib.metadata

# Load entry points and import modules
eps = importlib.metadata.entry_points()
if hasattr(eps, 'select'):
    teleop_eps = eps.select(group='lerobot.teleoperators')
else:
    teleop_eps = eps.get('lerobot.teleoperators', [])

print("Loading teleoperators from entry points:")
for ep in teleop_eps:
    try:
        module = importlib.import_module(ep.value)
        print(f"  ✓ Loaded {ep.name} from {ep.value}")
    except Exception as e:
        print(f"  ✗ Failed to load {ep.name}: {e}")

# Now check registry
from lerobot.teleoperators.teleoperator import TeleoperatorConfig
print(f"\nRegistered TeleoperatorConfig subclasses:")
for cls in TeleoperatorConfig.__subclasses__():
    print(f"  - {cls.__name__}")

# Try to get bi_yams_leader
try:
    BiYamsLeaderConfig = TeleoperatorConfig.get_subclass_by_name("bi_yams_leader")
    print(f"\n✓ Successfully retrieved bi_yams_leader config: {BiYamsLeaderConfig}")
except Exception as e:
    print(f"\n✗ Failed to get bi_yams_leader: {e}")

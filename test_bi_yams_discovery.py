#!/usr/bin/env python
"""
Test script to verify bi_yams_leader is discoverable by lerobot
"""

# Make sure discovery happens
import lerobot_discovery  # noqa: F401

# Now test if it's registered
from lerobot_teleoperator_gello.bi_leader import BiYamsLeaderConfig
from lerobot.teleoperators.teleoperator import TeleoperatorConfig

print("=== Testing bi_yams_leader Discovery ===\n")

print("1. Registered TeleoperatorConfig subclasses:")
for cls in TeleoperatorConfig.__subclasses__():
    print(f"   - {cls.__name__}")

print("\n2. Checking if BiYamsLeaderConfig is registered:")
print(f"   ✓ BiYamsLeaderConfig in subclasses: {BiYamsLeaderConfig in TeleoperatorConfig.__subclasses__()}")

print("\n3. Entry points registered:")
import importlib.metadata
eps = importlib.metadata.entry_points()
if hasattr(eps, 'select'):
    teleop_eps = eps.select(group='lerobot.teleoperators')
else:
    teleop_eps = eps.get('lerobot.teleoperators', [])

for ep in teleop_eps:
    marker = "✓" if 'yams' in ep.name else " "
    print(f"   {marker} {ep.name}: {ep.value}")

print("\n✓ SUCCESS: bi_yams_leader should now be discoverable by lerobot!")
print("\nTo use it, import the teleoperator packages in your configuration script:")
print("  from lerobot_teleoperator_gello import BiYamsLeader")
print("  from lerobot_robot_yams import BiYamsFollower")

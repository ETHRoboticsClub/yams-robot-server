import gc

from .bi_follower import BiYamsFollower, BiYamsFollowerConfig
from .follower import YamsFollower, YamsFollowerConfig
from .utils.utils import slow_move, split_arm_action

# TODO: Remove this once lerobot bloat is removed
gc.disable()  # NOTE: This is necessary to avoid latency spikes due to gc taking too long

# Auto-discover other lerobot plugins when this package is imported
try:
    import lerobot_discovery  # noqa: F401
except ImportError:
    pass

__all__ = [
    "BiYamsFollower",
    "BiYamsFollowerConfig",
    "YamsFollower",
    "YamsFollowerConfig",
    "slow_move",
    "split_arm_action",
]

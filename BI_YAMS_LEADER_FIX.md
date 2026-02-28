# bi_yams_leader Discovery Fix

## Problem

When trying to configure arms using lerobot, the `bi_yams_leader` class wasn't appearing in the list of available teleoperators. Only `koch_leader`, `sol01`, and `so100` were visible.

## Root Causes

1. **Packages Not Installed**: The `lerobot_teleoperator_gello` and `lerobot_robot_yams` packages were defined in the source tree (`src/`) but not installed in the Python environment. Even though the code had the correct `@TeleoperatorConfig.register_subclass("bi_yams_leader")` decorator, the registration only happens when the module is imported.

2. **Python Version Mismatch**: The project required Python 3.10 exactly (`==3.10.*`), but the environment was using Python 3.12.

3. **No Entry Points Discovery**: Lerobot plugins need to be registered with entry points in `pyproject.toml` so they can be automatically discovered and imported when needed.

4. **No Auto-Import Mechanism**: Even with entry points registered, there was no automatic mechanism to load them when lerobot initialized.

## Solution

### Step 1: Fix Python Version Constraint
Updated `pyproject.toml` to allow Python 3.10+:
```toml
requires-python = ">=3.10"  # Changed from "==3.10.*"
```

### Step 2: Register Entry Points
Added entry points to `pyproject.toml` so lerobot can discover these plugins:
```toml
[project.entry-points."lerobot.teleoperators"]
bi_yams_leader = "lerobot_teleoperator_gello.bi_leader"
yams_leader = "lerobot_teleoperator_gello.leader"

[project.entry-points."lerobot.robots"]
yams_follower = "lerobot_robot_yams.follower"
bi_yams_follower = "lerobot_robot_yams.bi_follower"

[project.entry-points."lerobot.cameras"]
zed_camera = "lerobot_camera_zed"
```

### Step 3: Create Auto-Discovery Module
Created `lerobot_discovery.py` that automatically loads all registered entry points:
- Ensures that `bi_yams_leader` and other custom classes are imported
- Triggers the `@TeleoperatorConfig.register_subclass()` decorators

### Step 4: Auto-Import Discovery Module
Updated package `__init__.py` files to import the discovery module:
- `src/lerobot_teleoperator_gello/__init__.py`
- `src/lerobot_robot_yams/__init__.py`

### Step 5: Install Package
Reinstalled the package in editable mode:
```bash
pip install -e . --force-reinstall --no-deps
```

## Verification

Run the test script to verify everything is working:
```bash
python test_bi_yams_discovery.py
```

Expected output shows:
- ✓ BiYamsLeaderConfig in registered subclasses
- ✓ bi_yams_leader entry point registered
- ✓ yams_leader entry point registered

## Usage

Now when configuring arms in lerobot, `bi_yams_leader` should appear alongside the built-in options. When importing the packages:

```python
from lerobot_teleoperator_gello import BiYamsLeader, BiYamsLeaderConfig
from lerobot_robot_yams import BiYamsFollower, BiYamsFollowerConfig
```

The discovery module will automatically load and register all available plugins.

## Files Modified

1. `pyproject.toml` - Added entry points and fixed Python version constraint
2. `src/lerobot_teleoperator_gello/__init__.py` - Added discovery import
3. `src/lerobot_robot_yams/__init__.py` - Added discovery import
4. `lerobot_discovery.py` - New auto-discovery module

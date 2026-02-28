"""
Auto-discovery module for lerobot plugins.
This module ensures that custom teleoperators and robots are imported when lerobot needs them.
"""

import importlib
import importlib.metadata


def load_entry_points(group: str) -> None:
    """Load all entry points for a given group.
    
    Args:
        group: Entry point group name (e.g., 'lerobot.teleoperators')
    """
    try:
        eps = importlib.metadata.entry_points()
        if hasattr(eps, 'select'):
            # Python 3.10+
            group_eps = eps.select(group=group)
        else:
            # Python 3.9 and earlier
            group_eps = eps.get(group, [])
        
        for ep in group_eps:
            try:
                importlib.import_module(ep.value)
            except Exception as e:
                print(f"Warning: Failed to load {group} entry point '{ep.name}': {e}")
    except Exception as e:
        print(f"Warning: Failed to load entry points for group '{group}': {e}")


def discover_teleoperators() -> None:
    """Auto-discover and load all registered teleoperators."""
    load_entry_points('lerobot.teleoperators')


def discover_robots() -> None:
    """Auto-discover and load all registered robots."""
    load_entry_points('lerobot.robots')


def discover_cameras() -> None:
    """Auto-discover and load all registered cameras."""
    load_entry_points('lerobot.cameras')


# Auto-load on import
discover_teleoperators()
discover_robots()
discover_cameras()

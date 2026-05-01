"""lerobot-record: r=record  space=save  d=discard  esc=exit.
Robot connects once at startup — no disconnect between idle and recording.
"""
import logging
import sys

logging.disable(logging.WARNING)

import lerobot.utils.control_utils as _cu  # noqa: E402
import lerobot.scripts.lerobot_record as _lr  # noqa: E402

_esc_pressed = False
_in_idle = True   # changes between idle and recording phases


def _keyboard_listener():
    global _esc_pressed
    events = {"exit_early": False, "rerecord_episode": False, "stop_recording": False}
    try:
        from pynput import keyboard

        def on_press(key):
            global _esc_pressed
            try:
                ch = getattr(key, "char", None)
                if _in_idle and ch == "r":
                    print("\n'r' → recording")
                    events["exit_early"] = True
                elif not _in_idle and key == keyboard.Key.space:
                    print("\nSpace → saving episode")
                    events["exit_early"] = True
                elif not _in_idle and ch == "d":
                    print("\n'd' → discarding")
                    events["rerecord_episode"] = True
                    events["exit_early"] = True
                elif key == keyboard.Key.esc:
                    print("\nEsc → exiting")
                    _esc_pressed = True
                    events["stop_recording"] = True
                    events["exit_early"] = True
            except Exception:
                pass

        listener = keyboard.Listener(on_press=on_press)
        listener.start()
        return listener, events
    except Exception as e:
        print(f"Keyboard listener unavailable: {e}")
        return None, events


_cu.init_keyboard_listener = _keyboard_listener

# Patch record_loop so it shows idle/recording banners and the first call
# gets an idle phase before recording (avoiding a fresh reconnect each time).
_orig_record_loop = _lr.record_loop
_first_call = True


def _patched_record_loop(**kwargs):
    global _in_idle, _first_call

    is_recording = kwargs.get("dataset") is not None

    if is_recording and _first_call:
        # Inject idle phase before the very first recording.
        _first_call = False
        _in_idle = True
        print("\n=== IDLE: press 'r' to record — esc=exit ===")
        _orig_record_loop(**{**kwargs, "dataset": None, "control_time_s": 86400})
        _in_idle = False
        kwargs["events"]["exit_early"] = False
        if kwargs["events"]["stop_recording"]:
            return
        print("\n=== RECORDING: space=save  d=discard  esc=exit ===")
        return _orig_record_loop(**kwargs)

    if not is_recording:
        # lerobot's reset phase → treat as idle between episodes.
        _in_idle = True
        print("\n=== IDLE: press 'r' to record — esc=exit ===")
        _orig_record_loop(**kwargs)
        _in_idle = False
        kwargs["events"]["exit_early"] = False
        if not kwargs["events"]["stop_recording"]:
            print("\n=== RECORDING: space=save  d=discard  esc=exit ===")
        return

    # Normal recording call (episode 2+).
    return _orig_record_loop(**kwargs)


_lr.record_loop = _patched_record_loop

from lerobot.scripts.lerobot_record import main  # noqa: E402

if __name__ == "__main__":
    result = None
    try:
        result = main()
    except KeyboardInterrupt:
        pass  # silent abort — no save
    sys.exit(42 if _esc_pressed else (result or 0))

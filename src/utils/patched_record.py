"""lerobot-record with custom key bindings: space=save, d=discard, esc=quit."""
import sys
import lerobot.utils.control_utils as _cu


def _keyboard_listener():
    events = {"exit_early": False, "rerecord_episode": False, "stop_recording": False}
    try:
        from pynput import keyboard

        def on_press(key):
            try:
                if key == keyboard.Key.space:
                    print("\nSpace → saving episode")
                    events["exit_early"] = True
                elif getattr(key, "char", None) == "d":
                    print("\n'd' → discarding, re-recording")
                    events["rerecord_episode"] = True
                    events["exit_early"] = True
                elif key == keyboard.Key.esc:
                    print("\nEsc → stopping")
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

from lerobot.scripts.lerobot_record import main  # noqa: E402

sys.exit(main())

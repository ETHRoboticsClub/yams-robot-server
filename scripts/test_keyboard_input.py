#!/usr/bin/env python3
import argparse
import os
import select
import shutil
import subprocess
import sys
import termios
import tty


def print_env():
    for name in [
        "DISPLAY",
        "WAYLAND_DISPLAY",
        "XDG_SESSION_TYPE",
        "TERM",
        "PYNPUT_BACKEND",
    ]:
        print(f"{name}={os.environ.get(name)}")


def import_test(backend: str | None):
    env = os.environ.copy()
    label = backend or "default"
    if backend is None:
        env.pop("PYNPUT_BACKEND", None)
    else:
        env["PYNPUT_BACKEND"] = backend
    code = "from pynput import keyboard; print(keyboard.Listener)"
    result = subprocess.run(
        [sys.executable, "-c", code],
        env=env,
        capture_output=True,
        text=True,
    )
    print(f"\n=== import test: {label} ===")
    print(f"exit={result.returncode}")
    if result.stdout.strip():
        print(result.stdout.strip())
    if result.stderr.strip():
        print(result.stderr.strip())


def import_keyboard():
    try:
        from pynput import keyboard
    except Exception as e:
        print(f"live listener import failed: {e}")
        return None
    return keyboard


def live_test():
    keyboard = import_keyboard()
    print("\n=== live test ===")
    print("Press arrows, letters, or Esc. Press q or Ctrl-C to quit.")

    listener = None
    if keyboard is not None:
        try:
            listener = keyboard.Listener(
                on_press=lambda key: print(f"pynput press   {key!r}", flush=True),
                on_release=lambda key: print(f"pynput release {key!r}", flush=True),
            )
            listener.start()
            print("listener started")
        except Exception as e:
            print(f"listener start failed: {e}")

    fd = sys.stdin.fileno()
    old = termios.tcgetattr(fd)
    try:
        tty.setraw(fd)
        while True:
            if not select.select([sys.stdin], [], [], 0.1)[0]:
                continue
            data = os.read(fd, 8)
            print(f"stdin bytes    {data!r}", flush=True)
            if data in (b"q", b"\x03"):
                break
    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, old)
        if listener is not None:
            listener.stop()


def print_suggestions():
    print("\n=== suggested commands ===")
    print(f"{sys.executable} {sys.argv[0]} --backend xorg")
    print(f"sudo -E {sys.executable} {sys.argv[0]} --backend uinput")

    gnome_terminal = shutil.which("gnome-terminal")
    kgx = shutil.which("kgx")
    if gnome_terminal:
        print(f"GDK_BACKEND=x11 {gnome_terminal} -- {sys.executable} {sys.argv[0]} --backend xorg")
    if kgx:
        print(f"GDK_BACKEND=x11 {kgx} {sys.executable} {sys.argv[0]} --backend xorg")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--backend", choices=["xorg", "uinput"])
    parser.add_argument("--no-live", action="store_true")
    args = parser.parse_args()

    if args.backend:
        os.environ["PYNPUT_BACKEND"] = args.backend

    print_env()
    import_test(None)
    import_test("xorg")
    import_test("uinput")
    print_suggestions()

    if not args.no_live:
        live_test()


if __name__ == "__main__":
    main()

import sys
import time


class TerminalStatus:
    def __init__(self, interval: float = 0.2):
        self.interval = interval
        self.stream = sys.stderr
        self.is_tty = self.stream.isatty()
        self._last_time = 0.0
        self._last_width = 0

    def update(self, text: str) -> None:
        now = time.monotonic()
        if now - self._last_time < self.interval:
            return
        self._last_time = now
        if not self.is_tty:
            print(text, file=self.stream, flush=True)
            return
        width = max(self._last_width, len(text))
        self.stream.write("\r\033[2K" + text.ljust(width))
        self.stream.flush()
        self._last_width = len(text)

    def close(self) -> None:
        if self.is_tty and self._last_width:
            self.stream.write("\n")
            self.stream.flush()

"""Timing helpers: processing time and smoothed FPS."""
import time
from collections import deque


class FPSMeter:
    """Computes FPS averaged over the last N frames (so the number is stable)."""

    def __init__(self, window: int = 30):
        self._stamps = deque(maxlen=window)

    def tick(self) -> float:
        """Call once per finished frame. Returns the current FPS."""
        self._stamps.append(time.perf_counter())
        if len(self._stamps) < 2:
            return 0.0
        elapsed = self._stamps[-1] - self._stamps[0]
        return (len(self._stamps) - 1) / elapsed if elapsed > 0 else 0.0

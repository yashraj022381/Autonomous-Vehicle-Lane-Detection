"""Small helpers for the web app: rolling speed stats, resizing, labels."""
from collections import deque

import cv2
import numpy as np


class RollingSpeed:
    """Average processing time over the last N frames -> stable FPS number."""

    def __init__(self, window: int = 30):
        self._ms = deque(maxlen=window)

    def add(self, ms: float) -> None:
        self._ms.append(ms)

    @property
    def avg_ms(self) -> float:
        return sum(self._ms) / len(self._ms) if self._ms else 0.0

    @property
    def fps(self) -> float:
        return 1000.0 / self.avg_ms if self.avg_ms > 0 else 0.0


def resize_to_width(frame: np.ndarray, width: int) -> np.ndarray:
    if frame.shape[1] <= width:
        return frame
    scale = width / frame.shape[1]
    return cv2.resize(frame, (width, int(frame.shape[0] * scale)),
                      interpolation=cv2.INTER_AREA)


def add_label(frame: np.ndarray, text: str) -> np.ndarray:
    """Return a copy with a readable title in the top-left corner."""
    out = frame.copy()
    for color, thick in (((0, 0, 0), 4), ((255, 255, 255), 2)):
        cv2.putText(out, text, (15, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.9,
                    color, thick, cv2.LINE_AA)
    return out


def side_by_side(*frames: np.ndarray) -> np.ndarray:
    """Glue frames left-to-right (used for the Comparison video)."""
    return np.hstack(frames)

from typing import NamedTuple, Optional, Tuple
import numpy as np
from ..config import ClassicalConfig


class Lane(NamedTuple):
    x1: float   # bottom point
    y1: float
    x2: float   # top point
    y2: float


def _average_to_lane(group: list, y_bottom: int, y_top: int) -> Optional[Lane]:
    if not group:
        return None
    arr = np.array(group)                       # columns: slope, intercept, length
    slope = np.average(arr[:, 0], weights=arr[:, 2])
    intercept = np.average(arr[:, 1], weights=arr[:, 2])
    # Line equation: y = slope*x + intercept  ->  x = (y - intercept) / slope
    return Lane((y_bottom - intercept) / slope, y_bottom,
                (y_top - intercept) / slope, y_top)


def is_plausible_pair(left: Lane, right: Lane) -> bool:
    """A real lane gets NARROWER as it goes away from the car.

    Three checks, all must pass:
      1. at the bottom the left line is to the left of the right line,
      2. at the top the left line is still to the left of the right line
         (if they swapped, the two lines CROSS - that makes an "X"),
      3. the gap at the top is smaller than the gap at the bottom.
    """
    bottom_gap = right.x1 - left.x1
    top_gap = right.x2 - left.x2
    return bottom_gap > 0 and top_gap > 0 and top_gap < bottom_gap


def fit_lanes(segments: np.ndarray, shape: tuple,
              cfg: ClassicalConfig) -> Tuple[Optional[Lane], Optional[Lane]]:
    h = shape[0]
    left, right = [], []

    for x1, y1, x2, y2 in segments:
        if x1 == x2:                            # perfectly vertical: skip
            continue
        slope = (y2 - y1) / (x2 - x1)
        if not (cfg.min_abs_slope < abs(slope) < cfg.max_abs_slope):
            continue                            # too flat or too steep = not a lane
        intercept = y1 - slope * x1
        length = float(np.hypot(x2 - x1, y2 - y1))
        # In images, y grows DOWNWARD, so the left lane leans "\" (negative slope)
        (left if slope < 0 else right).append((slope, intercept, length))

    y_bottom, y_top = h, int(h * cfg.lane_top_y)
    left_lane = _average_to_lane(left, y_bottom, y_top)
    right_lane = _average_to_lane(right, y_bottom, y_top)

    # Both found but they don't look like one road (crossed / widening):
    # better to draw nothing than a wrong shape.
    if left_lane is not None and right_lane is not None \
            and not is_plausible_pair(left_lane, right_lane):
        return None, None
    return left_lane, right_lane


class LaneSmoother:
    """Blends each new lane with the previous one so the video doesn't flicker."""

    def __init__(self, alpha: float, max_missed: int):
        self.alpha, self.max_missed = alpha, max_missed
        self.reset()

    def reset(self) -> None:
        self._lanes = {"left": None, "right": None}
        self._missed = {"left": 0, "right": 0}

    def update(self, side: str, lane: Optional[Lane]) -> Optional[Lane]:
        if lane is None:                        # lane lost this frame
            self._missed[side] += 1
            if self._missed[side] > self.max_missed:
                self._lanes[side] = None        # lost for too long: forget it
            return self._lanes[side]            # otherwise reuse the old one

        self._missed[side] = 0
        prev = self._lanes[side]
        if prev is not None:
            blended = self.alpha * np.array(lane) + (1 - self.alpha) * np.array(prev)
            lane = Lane(*blended)
        self._lanes[side] = lane
        return lane

"""Step 6b: draw the detected lane on the original frame."""
from typing import Optional
import cv2
import numpy as np
from .classical.lane_fitting import Lane
from .config import ClassicalConfig


def _pt(x: float, y: float) -> tuple:
    return int(round(x)), int(round(y))


def draw_lanes(frame: np.ndarray, left: Optional[Lane], right: Optional[Lane],
               cfg: ClassicalConfig) -> np.ndarray:
    result = frame.copy()

    # Translucent green "carpet" between the two lines
    if left is not None and right is not None:
        carpet = np.zeros_like(frame)
        poly = np.array([_pt(left.x1, left.y1), _pt(left.x2, left.y2),
                         _pt(right.x2, right.y2), _pt(right.x1, right.y1)],
                        dtype=np.int32)
        cv2.fillPoly(carpet, [poly], cfg.fill_color)
        result = cv2.addWeighted(result, 1.0, carpet, cfg.fill_opacity, 0)

    # Solid lines on top
    for lane in (left, right):
        if lane is not None:
            cv2.line(result, _pt(lane.x1, lane.y1), _pt(lane.x2, lane.y2),
                     cfg.line_color, cfg.line_thickness, cv2.LINE_AA)
    return result

import cv2
import numpy as np
from dataclasses import dataclass

from .deep_learning.postprocess import ego_lane, lane_area


@dataclass
class AIInsights:
    confidence: float
    coverage: float
    offset_pct: float
    status: str


def _no_lane(confidence: float, coverage: float) -> "AIInsights":
    return AIInsights(confidence, coverage, float("nan"), "No lane found")


def compute_insights(prob_u8: np.ndarray, threshold: float = 0.5,
                     ego_only: bool = True) -> AIInsights:
    """Confidence, coverage and car position.

    The car position is measured on the SAME lane that is drawn on the video
    (the "my lane" area), so the panel can never say "No lane found" while a
    lane is on screen.
    """
    p = prob_u8.astype(np.float32) / 255.0
    mask = p > threshold
    count = int(mask.sum())
    confidence = float(p[mask].mean()) if count else 0.0
    coverage = 100.0 * count / mask.size

    h, w = mask.shape
    if count == 0:
        return _no_lane(confidence, coverage)

    # The lane area between the two lines = exactly what the overlay paints green.
    if ego_only:
        _, area = ego_lane(mask)
    else:
        area = lane_area(mask)
    rows = np.flatnonzero(area.any(axis=1))
    if rows.size < 3:
        return _no_lane(confidence, coverage)

    # The rows of that lane that are closest to the car (the lowest ones).
    near = rows[-max(3, int(h * 0.10)):]
    lefts, rights = [], []
    for y in near:
        xs = np.flatnonzero(area[y])
        lefts.append(xs[0])
        rights.append(xs[-1])
    left_x, right_x = float(np.median(lefts)), float(np.median(rights))
    lane_width = right_x - left_x
    if lane_width < max(4.0, 0.05 * w):
        return _no_lane(confidence, coverage)

    # Car = middle of the image (assumes a centered dashcam).
    mid = w / 2
    offset = (mid - (left_x + right_x) / 2) / lane_width * 100
    if abs(offset) < 10:
        status = "Centered"
    else:
        status = "Right of center" if offset > 0 else "Left of center"
    if abs(offset) > 35:
        status += " ⚠️"
    return AIInsights(confidence, coverage, offset, status)


def probability_heatmap(frame_bgr: np.ndarray, prob_u8: np.ndarray,
                        opacity: float = 0.6) -> np.ndarray:
    """Paint the model's certainty on the frame: red = very sure, blue = unsure."""
    h, w = frame_bgr.shape[:2]
    big = cv2.resize(prob_u8, (w, h), interpolation=cv2.INTER_LINEAR)
    color = cv2.applyColorMap(big, cv2.COLORMAP_JET)
    blended = cv2.addWeighted(frame_bgr, 1 - opacity, color, opacity, 0)
    out = frame_bgr.copy()
    seen = big > 25                  # only paint where the model has an opinion
    out[seen] = blended[seen]
    return out

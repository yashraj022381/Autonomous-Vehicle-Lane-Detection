"""Turn the model's raw probability map into AI results people can understand."""
import cv2

import numpy as np

from dataclasses import dataclass



@dataclass
class AIInsights:
    confidence: float
    coverage: float
    offset_pct: float
    status: str


def compute_insights(prob_u8: np.ndarray, threshold: float = 0.5) -> AIInsights:
    p = prob_u8.astype(np.float32) / 255.0
    mask = p > threshold
    count = int(mask.sum())
    confidence = float(p[mask].mean()) if count else 0.0
    coverage = 100.0 * count / mask.size


    # Look at rows near the bottom (closest to the car) for the lane edges.
    h, w = mask.shape
    mid = w // 2
    lefts, rights = [], []
    for y in range(int(h * 0.80), int(h * 0.95)):
        xs = np.flatnonzero(mask[y])
        left, right = xs[xs < mid], xs[xs >= mid]
        if left.size and right.size:
            lefts.append(left.max())
            rights.append(right.min())

    if len(lefts) < 3:
        return AIInsights(confidence, coverage, float("nan"), "No lane found")
    left_x, right_x = float(np.median(lefts)), float(np.median(rights))
    lane_width = right_x - left_x
    if lane_width < 10:
        return AIInsights(confidence, coverage, float("nan"), "No lane found")

    # Car = middle of the image (assumes a centered dashcam).
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
        

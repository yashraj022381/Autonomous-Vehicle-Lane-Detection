"""Step 1-3: gray -> blur -> Canny edges -> (optional) color filter."""
import cv2
import numpy as np
from ..config import ClassicalConfig


def lane_color_mask(frame_bgr: np.ndarray) -> np.ndarray:
    """White-or-yellow mask. Lane paint is almost always one of those."""
    hls = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2HLS)
    white = cv2.inRange(hls, np.array([0, 200, 0]), np.array([255, 255, 255]))
    yellow = cv2.inRange(hls, np.array([15, 60, 100]), np.array([35, 255, 255]))
    return cv2.bitwise_or(white, yellow)


def get_edges(frame_bgr: np.ndarray, cfg: ClassicalConfig) -> np.ndarray:
    """Return a black image with white outlines where lanes might be."""
    gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
    k = cfg.blur_kernel
    blurred = cv2.GaussianBlur(gray, (k, k), 0)
    edges = cv2.Canny(blurred, cfg.canny_low, cfg.canny_high)

    if cfg.use_color_mask:
        mask = lane_color_mask(frame_bgr)
        # Grow the mask a bit so edges on the *border* of paint survive.
        mask = cv2.dilate(mask, np.ones((15, 15), np.uint8))
        edges = cv2.bitwise_and(edges, mask)
    return edges

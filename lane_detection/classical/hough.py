"""Step 5: probabilistic Hough transform -> straight line segments."""
import cv2
import numpy as np
from ..config import ClassicalConfig


def detect_segments(masked_edges: np.ndarray, cfg: ClassicalConfig) -> np.ndarray:
    """Return an (N, 4) array of [x1, y1, x2, y2]; empty (0, 4) if none."""
    lines = cv2.HoughLinesP(
        masked_edges,
        rho=cfg.hough_rho,
        theta=cfg.hough_theta,
        threshold=cfg.hough_threshold,
        minLineLength=cfg.min_line_length,
        maxLineGap=cfg.max_line_gap,
    )
    if lines is None:
        return np.empty((0, 4), dtype=np.int32)
    return lines.reshape(-1, 4)

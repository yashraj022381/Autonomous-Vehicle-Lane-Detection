"""Step 4: keep only the road area in front of the car."""
import cv2
import numpy as np
from ..config import ClassicalConfig


def build_roi_polygon(shape: tuple, cfg: ClassicalConfig) -> np.ndarray:
    h, w = shape[:2]
    return np.array([[
        (int(w * cfg.roi_bottom_left_x), h),
        (int(w * cfg.roi_top_left_x), int(h * cfg.roi_top_y)),
        (int(w * cfg.roi_top_right_x), int(h * cfg.roi_top_y)),
        (int(w * cfg.roi_bottom_right_x), h),
    ]], dtype=np.int32)


def apply_roi(edges: np.ndarray, polygon: np.ndarray) -> np.ndarray:
    mask = np.zeros_like(edges)
    cv2.fillPoly(mask, polygon, 255)
    return cv2.bitwise_and(edges, mask)

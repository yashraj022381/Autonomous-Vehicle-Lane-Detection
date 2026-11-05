"""The full classical pipeline behind one simple method: detector.process(frame)."""
from dataclasses import dataclass, field
from time import perf_counter
from typing import Dict, Optional
import numpy as np

from ..config import ClassicalConfig
from ..overlay import draw_lanes
from .hough import detect_segments
from .lane_fitting import Lane, LaneSmoother, fit_lanes
from .preprocessing import get_edges
from .roi import apply_roi, build_roi_polygon


@dataclass
class DetectionResult:
    overlay: np.ndarray                       # frame with lanes drawn
    left_lane: Optional[Lane]
    right_lane: Optional[Lane]
    processing_ms: float
    debug: Dict[str, np.ndarray] = field(default_factory=dict)  # edges, masked edges


class ClassicalLaneDetector:
    def __init__(self, cfg: Optional[ClassicalConfig] = None):
        self.cfg = cfg or ClassicalConfig()
        self.smoother = LaneSmoother(self.cfg.smoothing_alpha,
                                     self.cfg.max_missed_frames)

    def reset(self) -> None:
        """Call when starting a NEW video so old lanes don't carry over."""
        self.smoother.reset()

    def process(self, frame: np.ndarray, smooth: bool = True) -> DetectionResult:
        start = perf_counter()

        edges = get_edges(frame, self.cfg)
        roi = build_roi_polygon(frame.shape, self.cfg)
        masked = apply_roi(edges, roi)
        segments = detect_segments(masked, self.cfg)
        left, right = fit_lanes(segments, frame.shape, self.cfg)

        if smooth:
            left = self.smoother.update("left", left)
            right = self.smoother.update("right", right)

        overlay = draw_lanes(frame, left, right, self.cfg)
        ms = (perf_counter() - start) * 1000
        return DetectionResult(overlay, left, right, ms,
                               {"edges": edges, "masked_edges": masked})

"""Message types (like ROS .msg files)."""
from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass
class ImageMsg:
    seq: int
    stamp: float           # when the camera captured it (perf_counter seconds)
    frame: np.ndarray



@dataclass
class LaneMsg:
    seq: int
    stamp: float           # copied from the camera image, so we can measure lag
    source: str            # "classical" or "deep"
    overlay: np.ndarray
    processing_ms: float
    left: Any = None
    right: Any = None

"""Central configuration. Tweak numbers here, not deep inside the code."""
from dataclasses import dataclass
import numpy as np


@dataclass(frozen=True)
class ClassicalConfig:
    # --- Preprocessing ---
    blur_kernel: int = 5            # must be odd; bigger = blurrier
    canny_low: int = 50             # weak edge threshold
    canny_high: int = 150           # strong edge threshold
    use_color_mask: bool = True     # keep only white/yellow-ish areas

    # --- Region of interest (fractions of image width/height) ---
    roi_top_y: float = 0.60         # where the trapezoid's top edge sits
    roi_top_left_x: float = 0.42
    roi_top_right_x: float = 0.58
    roi_bottom_left_x: float = 0.05
    roi_bottom_right_x: float = 0.95

    # --- Hough transform ---
    hough_rho: float = 1.0
    hough_theta: float = np.pi / 180
    hough_threshold: int = 30       # votes needed to call something a line
    min_line_length: int = 40       # ignore tiny lines
    max_line_gap: int = 100         # connect dashes closer than this

    # --- Lane fitting ---
    min_abs_slope: float = 0.4      # ignore almost-horizontal lines
    max_abs_slope: float = 3.0      # ignore almost-vertical lines
    lane_top_y: float = 0.63        # where drawn lane lines end (fraction)

    # --- Smoothing ---
    smoothing_alpha: float = 0.25   # 0 = never change, 1 = no smoothing
    max_missed_frames: int = 8      # keep old lane this long if lost

    # --- Drawing (BGR colors) ---
    line_color: tuple = (0, 0, 255)
    fill_color: tuple = (0, 255, 0)
    line_thickness: int = 8
    fill_opacity: float = 0.35

@dataclass(frozen=True)
class DeepConfig:
    input_height: int = 256          # model input size (must divide by 16)
    input_width: int = 512           
    base_channels: int = 16          # model width: bigger = smarter but slower
    threshold: float = 0.5           # probability above this = "lane pixel"
    prob_smoothing: float = 0.5      # blend with previous frame (1 = no smoothing)
    weights_path: str = "weighs/unet_lane.pt"
    device: str = "auto"             # "auto", "cpu" or "cuda"

    line_color: tuple = (0, 0, 255)  # BGR
    fill_color: tuple = (0, 255, 0)
    fill_opacity: float = 0.35
    ego_only: bool = True            # hide neighbouring lanes and stray blobs

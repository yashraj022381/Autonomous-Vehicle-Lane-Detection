"""Quiz for the classical pipeline. Every test has a known right answer."""
import cv2
import pytest
import numpy as np

from lane_detection.classical.hough import detect_segments
from lane_detection.classical.lane_fitting import Lane, LaneSmoother, fit_lanes
from lane_detection.classical.pipeline import ClassicalLaneDetector
from lane_detection.classical.roi import apply_roi, build_roi_polygon
from lane_detection.config import ClassicalConfig
from lane_detection.overlay import draw_lanes

SHAPE = (480, 640, 3)
CFG = ClassicalConfig()


def synthetic_road() -> np.ndarray:
    """Grey road with two thick white lanes that lean toward each other."""
    frame = np.full(SHAPE, 60, np.uint8)
    cv2.line(frame, (160, 479), (290, 300), (255, 255, 255), 8)
    cv2.line(frame, (480, 479), (350, 300), (255, 255, 255), 8)
    return frame


# ---------------------------------------------------------------- ROI
def test_roi_polygon_has_four_corners_and_touches_bottom():
    poly = build_roi_polygon(SHAPE, CFG)
    assert poly.shape == (1, 4, 2)
    assert poly[0][0][1] == 480 and poly[0][3][1] == 480


def test_roi_keeps_the_road_and_removes_the_sky():
    edges = np.full(SHAPE[:2], 255, np.uint8)
    out = apply_roi(edges, build_roi_polygon(SHAPE, CFG))
    assert out[470, 320] == 255      # road in front of the car: kept
    assert out[10, 10] == 0          # sky: removed


# -------------------------------------------------------------- Hough
def test_hough_finds_a_drawn_line():
    img = np.zeros(SHAPE[:2], np.uint8)
    cv2.line(img, (100, 400), (300, 200), 255, 1)
    segments = detect_segments(img, CFG)
    assert segments.ndim == 2 and segments.shape[1] == 4
    assert len(segments) >= 1


def test_hough_on_empty_image_returns_empty_array():
    segments = detect_segments(np.zeros(SHAPE[:2], np.uint8), CFG)
    assert segments.shape == (0, 4)


# ------------------------------------------------------- lane fitting
def test_fit_lanes_sorts_left_and_right_correctly():
    segs = np.array([[100, 400, 200, 300],     # leans "\"  -> left
                     [400, 300, 500, 400]],    # leans "/"  -> right
                    dtype=np.int32)
    left, right = fit_lanes(segs, SHAPE, CFG)
    #assert left.x1 == 20 and left.y1 == 480
    #assert abs(left.x2 - 198) < 1e-6
    #assert right.x1 == 580
    #assert abs(right.x2 - 402) < 1e-6
    assert left.x1 == pytest.approx(20) and left.y1 == 480
    assert left.x2 == pytest.approx(198)
    assert right.x1 == pytest.approx(580)
    assert right.x2 == pytest.approx(402)


def test_fit_lanes_ignores_flat_and_vertical_segments():
    segs = np.array([[0, 100, 200, 110],       # almost flat
                     [50, 50, 50, 300]],       # perfectly vertical
                    dtype=np.int32)
    assert fit_lanes(segs, SHAPE, CFG) == (None, None)


def test_fit_lanes_with_no_segments():
    assert fit_lanes(np.empty((0, 4), np.int32), SHAPE, CFG) == (None, None)


# ----------------------------------------------------------- smoother
def test_smoother_blends_old_and_new():
    s = LaneSmoother(alpha=0.5, max_missed=2)
    s.update("left", Lane(0, 0, 0, 0))
    blended = s.update("left", Lane(10, 10, 10, 10))
    assert blended.x1 == 5


def test_smoother_remembers_then_forgets():
    s = LaneSmoother(alpha=0.5, max_missed=2)
    s.update("left", Lane(10, 0, 10, 0))
    assert s.update("left", None) is not None      # missed 1: keep old
    assert s.update("left", None) is not None      # missed 2: keep old
    assert s.update("left", None) is None          # missed 3: forget


def test_smoother_reset_clears_memory():
    s = LaneSmoother(alpha=0.5, max_missed=5)
    s.update("right", Lane(1, 1, 1, 1))
    s.reset()
    assert s.update("right", None) is None


# ------------------------------------------------------------ overlay
def test_overlay_without_lanes_returns_an_unchanged_copy():
    frame = synthetic_road()
    out = draw_lanes(frame, None, None, CFG)
    assert np.array_equal(out, frame) and out is not frame


def test_overlay_with_lanes_changes_the_picture_but_not_the_original():
    frame = np.zeros(SHAPE, np.uint8)
    out = draw_lanes(frame, Lane(100, 480, 250, 300), Lane(540, 480, 390, 300), CFG)
    assert out.shape == frame.shape
    assert out.any()                 # something was drawn
    assert not frame.any()           # original untouched


# ----------------------------------------------------------- pipeline
def test_pipeline_finds_both_lanes_on_a_synthetic_road():
    frame = synthetic_road()
    res = ClassicalLaneDetector().process(frame, smooth=False)
    assert res.left_lane is not None and res.right_lane is not None
    assert abs(res.left_lane.x1 - 160) < 25
    assert abs(res.right_lane.x1 - 480) < 25
    assert res.left_lane.x1 < res.right_lane.x1
    assert res.overlay.shape == frame.shape
    assert not np.array_equal(res.overlay, frame)
    assert res.processing_ms > 0
    assert {"edges", "masked_edges"} <= set(res.debug)


def test_pipeline_on_a_blank_frame_finds_nothing_and_does_not_crash():
    frame = np.zeros(SHAPE, np.uint8)
    res = ClassicalLaneDetector().process(frame)
    assert res.left_lane is None and res.right_lane is None
    assert np.array_equal(res.overlay, frame)

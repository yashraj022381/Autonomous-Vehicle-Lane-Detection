"""Tests for: (1) the AI panel agreeing with the drawn lane, (2) crossed classical lines."""
import cv2
import numpy as np

from lane_detection.classical.lane_fitting import Lane, fit_lanes, is_plausible_pair
from lane_detection.config import ClassicalConfig
from lane_detection.deep_learning.postprocess import ego_lane
from lane_detection.insights import compute_insights

CFG = ClassicalConfig()
H, W = 128, 256


def _prob(lines):
    """Make a fake probability map (uint8) from lists of (x_top, x_bottom)."""
    img = np.zeros((H, W), np.uint8)
    for x_top, x_bot in lines:
        cv2.line(img, (x_top, 40), (x_bot, H - 1), 255, 4)
    return img


# ------------------------------------------------------------ insights
def test_centered_car():
    ins = compute_insights(_prob([(110, 70), (146, 186)]))
    assert ins.status == "Centered"
    assert abs(ins.offset_pct) < 10


def test_car_right_of_center_when_lane_is_left():
    ins = compute_insights(_prob([(100, 60), (136, 150)]))
    assert ins.status.startswith("Right of center")
    assert ins.offset_pct > 0


def test_car_left_of_center_when_lane_is_right():
    ins = compute_insights(_prob([(120, 106), (156, 196)]))
    assert ins.status.startswith("Left of center")
    assert ins.offset_pct < 0


def test_no_lane_on_empty_map():
    ins = compute_insights(np.zeros((H, W), np.uint8))
    assert ins.status == "No lane found"
    assert np.isnan(ins.offset_pct)


def test_panel_agrees_with_overlay_whenever_a_lane_is_drawn():
    """If the overlay logic finds a lane, the panel must not say 'No lane found'."""
    prob = _prob([(110, 50), (146, 206)])
    _, area = ego_lane(prob > 127)
    assert area.any()
    assert compute_insights(prob).status != "No lane found"


def test_extra_neighbour_lane_does_not_change_position():
    ego = compute_insights(_prob([(110, 70), (146, 186)]))
    busy = compute_insights(_prob([(60, 0), (110, 70), (146, 186), (190, 250)]))
    assert busy.status == ego.status
    assert abs(busy.offset_pct - ego.offset_pct) < 5


# ------------------------------------------------- classical sanity check
def test_crossed_lines_are_rejected():
    # left line ends right of the right line -> an "X"
    left = Lane(100, 540, 700, 340)
    right = Lane(800, 540, 300, 340)
    assert not is_plausible_pair(left, right)


def test_normal_pair_is_accepted():
    left = Lane(150, 540, 438, 340)
    right = Lane(810, 540, 522, 340)
    assert is_plausible_pair(left, right)


def test_widening_pair_is_rejected():
    left = Lane(300, 540, 100, 340)      # gap grows going up: not a road
    right = Lane(500, 540, 700, 340)
    assert not is_plausible_pair(left, right)


def test_fit_lanes_returns_nothing_for_crossing_segments():
    # the "\" segment sits on the right and the "/" segment on the left, so they cross
    segments = np.array([[400, 540, 700, 330],
                         [800, 540, 500, 330]])
    left, right = fit_lanes(segments, (540, 960, 3), CFG)
    assert (left, right) == (None, None)


def test_fit_lanes_keeps_good_pair():
    segments = np.array([[150, 540, 430, 330], [810, 540, 530, 330]])
    left, right = fit_lanes(segments, (540, 960, 3), CFG)
    assert left is not None and right is not None

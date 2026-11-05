"""Quiz for the stopwatch, the app helpers and the AI insights."""
from types import SimpleNamespace

import numpy as np
import pytest

from lane_detection import metrics
from lane_detection.app_helpers import (RollingSpeed, add_label,
                                         resize_to_width, side_by_side)
from lane_detection.insights import compute_insights, probability_heatmap


# ------------------------------------------------------------- metrics
def test_fps_meter_is_zero_until_it_has_two_ticks():
    assert metrics.FPSMeter().tick() == 0.0


def test_fps_meter_measures_ten_fps(monkeypatch):
    times = iter([i * 0.1 for i in range(11)])       # a fake clock: 0.1 s per frame
    monkeypatch.setattr(metrics, "time", SimpleNamespace(perf_counter=lambda: next(times)))
    meter = metrics.FPSMeter()
    fps = [meter.tick() for _ in range(11)]
    assert fps[-1] == pytest.approx(10.0)


# --------------------------------------------------------- app helpers
def test_rolling_speed_average_and_fps():
    s = RollingSpeed()
    s.add(20)
    s.add(40)
    assert s.avg_ms == 30
    assert s.fps == pytest.approx(1000 / 30)


def test_rolling_speed_empty_is_zero():
    s = RollingSpeed()
    assert s.avg_ms == 0.0 and s.fps == 0.0


def test_resize_shrinks_wide_frames_and_keeps_small_ones():
    big = np.zeros((480, 1280, 3), np.uint8)
    assert resize_to_width(big, 640).shape == (240, 640, 3)
    small = np.zeros((100, 200, 3), np.uint8)
    assert resize_to_width(small, 640) is small


def test_add_label_does_not_change_the_original():
    frame = np.zeros((100, 300, 3), np.uint8)
    out = add_label(frame, "Classical")
    assert not frame.any() and out.any()


def test_side_by_side_doubles_the_width():
    a = np.zeros((10, 20, 3), np.uint8)
    assert side_by_side(a, a).shape == (10, 40, 3)


# ------------------------------------------------------------ insights
def lane_probability(left_x, right_x):
    prob = np.zeros((256, 512), np.uint8)
    prob[:, left_x] = 255
    prob[:, right_x] = 255
    return prob


def test_insights_centered_car():
    ins = compute_insights(lane_probability(150, 350))     # lane center = 250, image center = 256
    assert ins.status == "Centered"
    assert ins.offset_pct == pytest.approx(3.0)
    assert ins.confidence == pytest.approx(1.0)
    assert ins.coverage == pytest.approx(100 * 2 / 512)


def test_insights_car_far_to_the_right_gets_a_warning():
    ins = compute_insights(lane_probability(60, 260))      # lane center = 160, image center = 256
    assert ins.status == "Right of center ⚠️"
    assert ins.offset_pct > 35


def test_insights_with_no_lane():
    ins = compute_insights(np.zeros((256, 512), np.uint8))
    assert ins.status == "No lane found"
    assert ins.confidence == 0.0 and np.isnan(ins.offset_pct)


def test_heatmap_keeps_size_and_leaves_unsure_areas_alone():
    frame = np.full((120, 200, 3), 90, np.uint8)
    out = probability_heatmap(frame, np.zeros((64, 128), np.uint8))
    assert out.shape == frame.shape
    assert np.array_equal(out, frame)

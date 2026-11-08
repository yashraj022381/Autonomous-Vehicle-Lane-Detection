"""Tests for automatic model selection."""
from types import SimpleNamespace

import cv2
import numpy as np
import pytest

from lane_detection.auto_select import (
    Candidate, candidates_from_choices, choose, eligible, run_auto, sample_video_frames,
)

H, W = 128, 256
FAST_V2 = Candidate("Fast v2", "w/fast_v2.pt", fast=True, newer=True)
FAST = Candidate("Fast", "w/fast.pt", fast=True, newer=False)
ACC_V2 = Candidate("Accurate v2", "w/acc_v2.pt", fast=False, newer=True)
ACC = Candidate("Accurate", "w/acc.pt", fast=False, newer=False)
ALL = [FAST_V2, FAST, ACC_V2, ACC]


def lane_prob(value=255):
    """A clean pair of lane lines; `value` sets how sure the model is (255 = 100%)."""
    img = np.zeros((H, W), np.uint8)
    for x_top, x_bot in ((110, 70), (146, 186)):
        cv2.line(img, (x_top, 40), (x_bot, H - 1), int(value), 4)
    return img


class FakeDetector:
    def __init__(self, prob):
        self.prob, self.resets, self.calls = prob, 0, 0

    def reset(self):
        self.resets += 1

    def process(self, frame, smooth=True):
        assert smooth is False, "the audition must not use temporal smoothing"
        self.calls += 1
        return SimpleNamespace(debug={"probability": self.prob})


def run(probs, is_video=False, candidates=ALL, frames=3):
    dets = {c.weights: FakeDetector(probs[c.label]) for c in candidates if c.label in probs}
    choice = run_auto([np.zeros((10, 10, 3), np.uint8)] * frames, candidates,
                      lambda w: dets[w], is_video)
    return choice, dets


# ---------------------------------------------------------------- choosing
def test_prefers_model_that_finds_a_lane():
    probs = {"Fast v2": np.zeros((H, W), np.uint8), "Fast": lane_prob(),
             "Accurate v2": np.zeros((H, W), np.uint8), "Accurate": np.zeros((H, W), np.uint8)}
    choice, _ = run(probs)
    assert choice.label == "Fast"


def test_higher_confidence_wins_when_the_gap_is_large():
    probs = {"Fast v2": lane_prob(255), "Fast": lane_prob(255),
             "Accurate v2": lane_prob(190), "Accurate": lane_prob(190)}   # about 75% sure
    choice, _ = run(probs)
    assert choice.label in {"Fast v2", "Fast"}
    assert choice.label == "Fast v2"            # tie between the two: newer first


def test_close_call_prefers_accurate_for_images():
    probs = {c.label: lane_prob(250) for c in ALL}
    choice, _ = run(probs, is_video=False)
    assert choice.label == "Accurate v2"


def test_close_call_prefers_v2_among_same_speed():
    probs = {c.label: lane_prob(250) for c in (FAST_V2, FAST)}
    choice, _ = run(probs, candidates=[FAST, FAST_V2])
    assert choice.label == "Fast v2"


def test_video_only_tries_fast_models():
    probs = {c.label: lane_prob(250) for c in ALL}
    choice, dets = run(probs, is_video=True)
    assert choice.label == "Fast v2"
    assert dets["w/acc.pt"].calls == 0 and dets["w/acc_v2.pt"].calls == 0
    assert {c.label for c in eligible(ALL, True)} == {"Fast v2", "Fast"}


def test_video_with_no_fast_models_uses_what_exists():
    assert eligible([ACC, ACC_V2], True) == [ACC, ACC_V2]


def test_nobody_finds_a_lane_falls_back_to_default():
    probs = {c.label: np.zeros((H, W), np.uint8) for c in ALL}
    image_choice, _ = run(probs, is_video=False)
    video_choice, _ = run(probs, is_video=True)
    assert image_choice.label == "Accurate v2"
    assert video_choice.label == "Fast v2"
    assert "No model found" in image_choice.reason


def test_reason_and_scores_are_reported():
    probs = {c.label: lane_prob(255) for c in ALL}
    choice, _ = run(probs)
    assert "3 of 3" in choice.reason
    assert set(choice.scores) == {c.label for c in ALL}
    assert choice.frames_used == 3
    assert choice.weights == "w/acc_v2.pt"


def test_choose_rejects_empty_list():
    with pytest.raises(ValueError):
        choose([], {}, is_video=False)


# ------------------------------------------------------- shared detectors
def test_detectors_are_reset_before_and_after_the_audition():
    probs = {c.label: lane_prob() for c in ALL}
    _, dets = run(probs)
    assert all(d.resets == 2 for d in dets.values())


# -------------------------------------------------------- app integration
def test_candidates_come_from_the_app_labels():
    cands = candidates_from_choices({
        "Fast v2 + CULane (256x128)": "weights/a.pt",
        "Fast (256x128)": "weights/b.pt",
        "Accurate v2 + CULane (512x256)": "weights/c.pt",
        "Accurate (512x256)": "weights/d.pt"})
    flags = {c.weights: (c.fast, c.newer) for c in cands}
    assert flags == {"weights/a.pt": (True, True), "weights/b.pt": (True, False),
                     "weights/c.pt": (False, True), "weights/d.pt": (False, False)}


# ----------------------------------------------------------- video frames
def test_sample_video_frames(tmp_path):
    path = str(tmp_path / "clip.mp4")
    writer = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*"mp4v"), 10, (320, 180))
    for i in range(40):
        writer.write(np.full((180, 320, 3), i * 5, np.uint8))
    writer.release()
    frames = sample_video_frames(path, n=5, width=160)
    assert len(frames) == 5
    assert all(f.shape == (90, 160, 3) for f in frames)


def test_sample_video_frames_bad_file_gives_empty_list(tmp_path):
    assert sample_video_frames(str(tmp_path / "missing.mp4")) == []

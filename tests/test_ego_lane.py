"""Quiz for 'show only my lane'. No PyTorch needed: we feed in drawn lane masks."""
from dataclasses import replace

import cv2
import numpy as np

from lane_detection.config import DeepConfig
from lane_detection.deep_learning.postprocess import build_masks, ego_lane, lane_area

H, W = 256, 512


def canvas():
    return np.zeros((H, W), np.uint8)


def line(img, p1, p2, thick=5):
    cv2.line(img, p1, p2, 255, thick)


def ego_pair(img):
    """The two lines around the car, leaning toward each other."""
    line(img, (130, 255), (245, 140))      # left
    line(img, (390, 255), (275, 140))      # right


def messy_scene():
    """Ego lane + extra left lanes (one touching the ego line) + a bright bar on the
    right edge + stray blobs: like the photographed screen in the screenshots."""
    img = canvas()
    ego_pair(img)
    line(img, (10, 235), (235, 150), 4)    # neighbour lane, runs into the ego line
    line(img, (60, 205), (225, 160), 4)    # another neighbour lane
    cv2.rectangle(img, (480, 60), (488, 250), 255, -1)   # vertical bar, right edge
    cv2.rectangle(img, (60, 100), (80, 110), 255, -1)    # stray blob, left
    return img > 0


def test_clean_two_lines_are_kept_completely():
    img = canvas()
    ego_pair(img)
    mask = img > 0
    lines, area = ego_lane(mask)
    assert not (lines & ~mask).any()                    # nothing invented
    assert lines.sum() >= 0.95 * mask.sum()             # nothing lost
    assert area.sum() > 0


def test_old_method_makes_horizontal_stripes_but_new_does_not():
    mask = messy_scene()
    old_area = lane_area(mask)
    assert old_area[100:110].any()                      # the bug you saw: stripes above the road
    _, area = ego_lane(mask)
    assert not area[:135].any()                         # no stripes above the lane top


def test_extra_lanes_and_edge_bar_are_not_drawn():
    lines, area = ego_lane(messy_scene())
    assert not lines[:, :115].any()                     # neighbour lanes on the left
    assert not lines[:, 420:].any()                     # bar on the right edge
    assert not lines[95:115, :].any()                   # stray blob
    assert lines[250:, :].any()                         # the ego lines ARE there
    assert area[200:250].any()                          # the road between them is filled


def test_green_area_sits_between_the_two_ego_lines():
    _, area = ego_lane(messy_scene())
    ys, xs = np.nonzero(area)
    assert xs.min() > 130 and xs.max() < 395


def test_dashed_line_gaps_do_not_make_holes():
    img = canvas()
    ego_pair(img)
    img[180:186, 100:260] = 0                           # the left line vanishes for 6 rows
    lines, area = ego_lane(img > 0)
    assert all(area[y].any() for y in range(150, 250))  # no empty rows inside the lane


def test_curved_lane_is_followed_to_the_top():
    img = canvas()
    left = np.array([(130, 255), (140, 210), (165, 175), (200, 150), (240, 135)], np.int32)
    right = np.array([(390, 255), (385, 210), (360, 175), (330, 150), (290, 135)], np.int32)
    cv2.polylines(img, [left], False, 255, 5)
    cv2.polylines(img, [right], False, 255, 5)
    lines, area = ego_lane(img > 0)
    assert lines[140:150].any()                         # still tracking near the top


def test_no_lane_found_gives_empty_masks():
    for mask in (np.zeros((H, W), bool), messy_scene() & False):
        lines, area = ego_lane(mask)
        assert not lines.any() and not area.any()


def test_only_one_side_visible_gives_empty_masks():
    img = canvas()
    line(img, (130, 255), (245, 140))
    lines, area = ego_lane(img > 0)
    assert not lines.any() and not area.any()


def test_only_stray_blobs_give_empty_masks():
    img = canvas()
    cv2.rectangle(img, (480, 60), (488, 250), 255, -1)
    cv2.rectangle(img, (60, 100), (80, 110), 255, -1)
    lines, area = ego_lane(img > 0)
    assert not lines.any() and not area.any()


def test_very_short_fragments_are_ignored():
    img = canvas()
    line(img, (130, 255), (140, 250))
    line(img, (390, 255), (380, 250))
    lines, area = ego_lane(img > 0)
    assert not lines.any() and not area.any()


def test_build_masks_returns_frame_sized_masks_and_respects_the_switch():
    prob = messy_scene().astype(np.float32) * 0.9
    cfg_on = DeepConfig()
    cfg_off = replace(DeepConfig(), ego_only=False)
    lines_on, area_on = build_masks(prob, (480, 640), cfg_on)
    lines_off, area_off = build_masks(prob, (480, 640), cfg_off)
    assert lines_on.shape == area_on.shape == (480, 640)
    assert lines_on.dtype == bool and area_on.dtype == bool
    assert lines_off.sum() > lines_on.sum()             # the switch brings the extras back
    assert area_off[100 * 480 // 256: 110 * 480 // 256].any()   # old-style stripes
    assert not area_on[: 130 * 480 // 256].any()
    assert lines_on[470:].any()                         # ego lines drawn at the bottom


def test_neighbour_lane_merging_into_the_ego_line_leaves_no_stub():
    lines, _ = ego_lane(messy_scene())
    assert not lines[140:175, :200].any()               # nothing sticks out to the left

"""Turn the model's lane-line pixels into a drawn result."""
import cv2
import numpy as np
from ..config import DeepConfig


def lane_area(mask: np.ndarray) -> np.ndarray:
    """For each row, fill between the nearest lane line left and right of center."""
    h, w = mask.shape
    mid, area = w // 2, np.zeros((h, w), dtype=bool)
    for y in range(h):
        xs = np.flatnonzero(mask[y])
        left, right = xs[xs < mid], xs[xs >= mid]
        if left.size and right.size:
            area[y, left.max(): right.min() + 1] = True
    return area


def draw_mask_overlay(frame, line_mask, area_mask, cfg: DeepConfig) -> np.ndarray:
    tinted = frame.copy()
    tinted[area_mask] = cfg.fill_color                      # green carpet
    out = cv2.addWeighted(tinted, cfg.fill_opacity, frame, 1 - cfg.fill_opacity, 0)
    out[line_mask] = cfg.line_color                         # red lane lines
    return out


# ---------------------------------------------------------------------------
# NEW: keep only the lane the car is driving in
# ---------------------------------------------------------------------------
def _runs(row: np.ndarray) -> list:
    """Split one image row into runs of neighbouring lane pixels: [(first_x, last_x), ...]."""
    xs = np.flatnonzero(row)
    if xs.size == 0:
        return []
    cuts = np.flatnonzero(np.diff(xs) > 1) + 1
    return [(int(g[0]), int(g[-1])) for g in np.split(xs, cuts)]


def _center(run: tuple) -> float:
    return (run[0] + run[1]) / 2.0


def _closest(runs: list, target: float, tol: float):
    """The run whose center is nearest to `target`, but only if it is within `tol` pixels."""
    best, best_dist = None, tol
    for run in runs:
        dist = abs(_center(run) - target)
        if dist <= best_dist:
            best, best_dist = run, dist
    return best


def _cap(widths: list, default_cap: int) -> int:
    """Widest a line may be: twice its usual width (or the default at the start)."""
    if len(widths) < 5:
        return default_cap
    return max(4, int(2 * float(np.median(widths))) + 2)


def ego_lane(mask: np.ndarray, max_shift: float = 0.02, max_gap: float = 0.06,
             min_width: float = 0.08, min_length: float = 0.05,
             search_from: float = 0.55, max_run: float = 0.10):
    """Find ONLY the two lines around the car and the road between them.

    Idea: start at the bottom of the picture (right in front of the car), pick the
    nearest line on the left and on the right of the image center, then walk UP the
    picture row by row, following each line. A line may only move a few pixels from
    one row to the next, so flat lines of other lanes and stray blobs are ignored.

    Returns (lines, area): two boolean arrays with the same shape as `mask`.
    """
    h, w = mask.shape
    lines = np.zeros((h, w), dtype=bool)
    area = np.zeros((h, w), dtype=bool)
    mid = w / 2.0
    tol = max(2.0, max_shift * w)              # how far a line may shift between two rows
    gap_limit = max(2, int(max_gap * h))       # how many rows a line may vanish (dashes)
    run_cap = max(3, int(max_run * w))         # a line is never wider than this

    # 1) Seed: the lowest row (near the car) that has a line on both sides of the center.
    seed_row, left_run, right_run = None, None, None
    for y in range(h - 1, int(h * search_from) - 1, -1):
        runs = _runs(mask[y])
        left = [r for r in runs if _center(r) < mid]
        right = [r for r in runs if _center(r) >= mid]
        if left and right and _center(right[0]) - _center(left[-1]) >= min_width * w:
            seed_row, left_run, right_run = y, left[-1], right[0]
            break
    if seed_row is None:
        return lines, area

    # 2) Walk up the picture, following both lines.
    prev_l, prev_r = _center(left_run), _center(right_run)
    l_end, r_start = left_run[1], right_run[0]
    miss_l = miss_r = 0
    last_good = seed_row                       # highest row where BOTH lines were seen
    width_l, width_r = [], []                  # how thick each line has been so far
    for y in range(seed_row, -1, -1):
        runs = _runs(mask[y])
        cl = _closest(runs, prev_l, tol * (miss_l + 1))
        cr = _closest(runs, prev_r, tol * (miss_r + 1))
        if cl is not None and (cl == cr or (cr is not None and _center(cl) >= _center(cr))):
            break                              # the two lines met: top of the lane

        if cl is not None:
            # A line is never much thicker than it has been so far. If it suddenly is
            # (a neighbour lane merged into it), keep only the part nearest the car.
            cap = _cap(width_l, run_cap)
            a = max(cl[0], cl[1] - cap + 1)
            width_l.append(cl[1] - a + 1)
            lines[y, a:cl[1] + 1] = True
            prev_l, l_end, miss_l = _center((a, cl[1])), cl[1], 0
        else:
            miss_l += 1
        if cr is not None:
            cap = _cap(width_r, run_cap)
            b = min(cr[1], cr[0] + cap - 1)
            width_r.append(b - cr[0] + 1)
            lines[y, cr[0]:b + 1] = True
            prev_r, r_start, miss_r = _center((cr[0], b)), cr[0], 0
        else:
            miss_r += 1

        if miss_l > gap_limit or miss_r > gap_limit:
            break                              # lost a line for too long: stop
        if l_end < r_start:
            area[y, l_end:r_start + 1] = True  # missing rows reuse the last known edge
        if cl is not None and cr is not None:
            last_good = y

    lines[:last_good] = False                  # forget rows above the real lane top
    area[:last_good] = False
    if seed_row - last_good + 1 < max(4, int(min_length * h)):   # too short to be a lane
        return np.zeros_like(lines), np.zeros_like(area)
    return lines, area


def build_masks(prob: np.ndarray, frame_hw: tuple, cfg: DeepConfig):
    """Probability map (model size) -> (lines, area) boolean masks at frame size."""
    h, w = frame_hw
    small = prob > cfg.threshold
    prob_up = cv2.resize(prob, (w, h), interpolation=cv2.INTER_LINEAR)

    if cfg.ego_only:
        ego_lines, area_small = ego_lane(small)
        keep = cv2.dilate(ego_lines.astype(np.uint8), np.ones((3, 3), np.uint8))
        keep = cv2.resize(keep, (w, h), interpolation=cv2.INTER_NEAREST).astype(bool)
        lines = (prob_up > cfg.threshold) & keep          # smooth edges, but only ego lines
    else:
        area_small = lane_area(small)
        lines = prob_up > cfg.threshold

    area = cv2.resize(area_small.astype(np.uint8), (w, h),
                      interpolation=cv2.INTER_NEAREST).astype(bool)
    return lines, area

"""Auto model selection: pick a suitable deep-learning model for the uploaded file.

How it decides (simple on purpose, so it can be explained in one breath):

1. SPEED BUDGET  - a video must run frame after frame, so only the FAST models are
                   considered. A single image can afford the slower, more detailed ones.
2. QUICK AUDITION - every eligible model looks at up to 5 sample frames. On a frame
                   where it finds no lane it scores 0, otherwise its confidence.
3. WINNER        - the model with the best average score. If the top models are within
                   3 points of each other (a close call) a preference order decides:
                   accurate before fast (images only), then v2 (trained on more varied
                   data) before the original.

Confidence is NOT the same as correctness. Auto is a sensible default that avoids a model
that fails on your file; it is not a guarantee that the chosen model is the most accurate.
"""
from dataclasses import dataclass
from typing import Callable, Dict, List, Sequence

import cv2
import numpy as np

from .insights import compute_insights

N_SAMPLES = 5          # frames each model looks at
CLOSE_CALL = 0.03      # scores closer than this count as a tie


@dataclass(frozen=True)
class Candidate:
    label: str         # name shown in the app
    weights: str       # path to the .pt file
    fast: bool         # small/fast model (256x128) vs accurate (512x256)
    newer: bool        # "v2": trained on a more varied dataset


@dataclass
class AutoChoice:
    label: str
    weights: str
    scores: Dict[str, float]   # average score per model that was tested (0..1)
    reason: str                # one sentence for the user
    frames_used: int


def candidates_from_choices(weight_choices: Dict[str, str]) -> List[Candidate]:
    """Build candidates from the app's {label: path} dictionary.
    A label containing 'fast' is a fast model; one containing 'v2' is a newer model."""
    out = []
    for label, path in weight_choices.items():
        low = label.lower()
        out.append(Candidate(label, str(path), fast="fast" in low, newer="v2" in low))
    return out


def score_probability(prob_u8: np.ndarray, threshold: float = 0.5,
                      ego_only: bool = True) -> float:
    """0 if no lane is found, otherwise the model's confidence (0..1)."""
    ins = compute_insights(prob_u8, threshold, ego_only)
    return 0.0 if ins.status.startswith("No lane") else float(ins.confidence)


def eligible(candidates: Sequence[Candidate], is_video: bool) -> List[Candidate]:
    """Videos only audition the fast models (if there are any)."""
    if is_video:
        fast = [c for c in candidates if c.fast]
        if fast:
            return fast
    return list(candidates)


def _preference_key(c: Candidate, is_video: bool):
    """Lower sorts first: images prefer accurate models, video prefers fast; v2 first."""
    speed_rank = 0 if c.fast == is_video else 1
    return (speed_rank, 0 if c.newer else 1)


def choose(candidates: Sequence[Candidate], frame_scores: Dict[str, List[float]],
           is_video: bool, close_call: float = CLOSE_CALL) -> AutoChoice:
    pool = eligible(candidates, is_video)
    if not pool:
        raise ValueError("No candidate models to choose from")

    means = {c.label: (float(np.mean(frame_scores[c.label])) if frame_scores.get(c.label) else 0.0)
             for c in pool}
    n_frames = max((len(v) for v in frame_scores.values()), default=0)
    best = max(means.values())

    def key(c):
        return _preference_key(c, is_video)

    if best <= 0.0:
        pick = min(pool, key=key)
        reason = ("No model found a clear lane in the sample frames, so the default for "
                  f"{'video' if is_video else 'images'} was used.")
    else:
        contenders = [c for c in pool if means[c.label] >= best - close_call]
        pick = min(contenders, key=key)
        scores = frame_scores.get(pick.label, [])
        found = [s for s in scores if s > 0]
        reason = (f"Found the lane in {len(found)} of {len(scores)} sample frame(s) "
                  f"with {np.mean(found):.0%} average confidence.")
        if len(contenders) > 1:
            reason += (" Several models were within a few points of each other, so the "
                       f"preferred one for {'video (fast, newer data)' if is_video else 'images (accurate, newer data)'}"
                       " was chosen.")
    return AutoChoice(pick.label, pick.weights, {k: round(v, 3) for k, v in means.items()},
                      reason, n_frames)


def run_auto(frames: Sequence[np.ndarray], candidates: Sequence[Candidate],
             load_detector: Callable, is_video: bool, threshold: float = 0.5,
             ego_only: bool = True) -> AutoChoice:
    """Let each eligible model look at the sample frames, then choose.

    `load_detector(weights_path)` must return a DeepLaneDetector (the app's cached loader).
    The detectors are shared with the real run, so their smoothing memory is reset
    before AND after the audition.
    """
    frame_scores: Dict[str, List[float]] = {}
    for c in eligible(candidates, is_video):
        det = load_detector(c.weights)
        det.reset()
        scores = []
        for frame in frames:
            result = det.process(frame, smooth=False)
            scores.append(score_probability(result.debug["probability"], threshold, ego_only))
        det.reset()
        frame_scores[c.label] = scores
    return choose(candidates, frame_scores, is_video)


def _resize_to_width(frame: np.ndarray, width: int) -> np.ndarray:
    h, w = frame.shape[:2]
    if w == width:
        return frame
    return cv2.resize(frame, (width, max(1, round(h * width / w))), interpolation=cv2.INTER_AREA)


def sample_video_frames(path: str, n: int = N_SAMPLES, width: int = 640) -> List[np.ndarray]:
    """Up to n frames spread between 10% and 90% of the video, resized to `width`."""
    cap = cv2.VideoCapture(str(path))
    frames: List[np.ndarray] = []
    try:
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if total > 0:
            positions = (np.linspace(total * 0.1, total * 0.9, n).astype(int)
                         if total >= n else range(total))
            for pos in positions:
                cap.set(cv2.CAP_PROP_POS_FRAMES, int(pos))
                ok, frame = cap.read()
                if ok:
                    frames.append(_resize_to_width(frame, width))
        if not frames:                                   # seeking not supported: read the start
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            for _ in range(n):
                ok, frame = cap.read()
                if not ok:
                    break
                frames.append(_resize_to_width(frame, width))
    finally:
        cap.release()
    return frames

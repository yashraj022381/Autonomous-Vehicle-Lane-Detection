"""DeepLaneDetector: same .process(frame) interface as the classical detector."""
from pathlib import Path
from time import perf_counter
from typing import Optional

import cv2
import numpy as np
import torch

from ..classical.pipeline import DetectionResult
from ..config import DeepConfig
from .dataset import preprocess_bgr
from .model import UNet
from .postprocess import build_masks, draw_mask_overlay


class DeepLaneDetector:
    def __init__(self, weights: Optional[str] = None, cfg: Optional[DeepConfig] = None):
        self.cfg = cfg or DeepConfig()
        path = Path(weights or self.cfg.weights_path)
        if not path.exists():
            raise FileNotFoundError(
                f"Weights not found: {path}. Train first: python -m scripts.train --data ...")

        if self.cfg.device == "auto":
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(self.cfg.device)

        ckpt = torch.load(path, map_location=self.device)
        self.size = tuple(ckpt["size"])                      # (H, W) used in training
        self.model = UNet(base=ckpt["base"]).to(self.device)
        self.model.load_state_dict(ckpt["model"])
        self.model.eval()                                    # "exam mode"
        self._prev_prob = None

    def reset(self) -> None:
        self._prev_prob = None

    @torch.inference_mode()
    def process(self, frame: np.ndarray, smooth: bool = True) -> DetectionResult:
        start = perf_counter()
        h, w = frame.shape[:2]

        x = preprocess_bgr(frame, self.size).unsqueeze(0).to(self.device)
        prob = torch.sigmoid(self.model(x))[0, 0].cpu().numpy()   # (256, 512) in 0..1

        if smooth and self._prev_prob is not None:           # calm down the flicker
            a = self.cfg.prob_smoothing
            prob = a * prob + (1 - a) * self._prev_prob
        self._prev_prob = prob
        lines, area = build_masks(prob, (h, w), self.cfg)
        #small_mask = prob > self.cfg.threshold
        #area = cv2.resize(lane_area(small_mask).astype(np.uint8), (w, h),
        #                  interpolation=cv2.INTER_NEAREST).astype(bool)
        #lines = cv2.resize(prob, (w, h), interpolation=cv2.INTER_LINEAR) > self.cfg.threshold

        overlay = draw_mask_overlay(frame, lines, area, self.cfg)
        ms = (perf_counter() - start) * 1000
        return DetectionResult(overlay, None, None, ms,
                               {"probability": (prob * 255).astype(np.uint8)})

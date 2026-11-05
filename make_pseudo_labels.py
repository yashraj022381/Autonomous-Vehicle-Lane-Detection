"""Make training pairs from dashcam videos using the classical detector as teacher.

    python -m scripts.make_pseudo_labels --video data/solidWhiteRight.mp4 --out data/pseudo
"""
import argparse
from pathlib import Path
import cv2
import numpy as np
from lane_detection.classical.pipeline import ClassicalLaneDetector


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--video", required=True, type=Path)
    p.add_argument("--out", type=Path, default=Path("data/pseudo"))
    p.add_argument("--every", type=int, default=3, help="keep every Nth frame")
    args = p.parse_args()

    (args.out / "images").mkdir(parents=True, exist_ok=True)
    (args.out / "masks").mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(str(args.video))
    detector = ClassicalLaneDetector()
    idx = saved = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        idx += 1
        if idx % args.every:
            continue
        res = detector.process(frame, smooth=False)
        if res.left_lane is None or res.right_lane is None:
            continue                                  # skip uncertain frames

        h, w = frame.shape[:2]
        mask = np.zeros((h, w), np.uint8)
        for lane in (res.left_lane, res.right_lane):
            cv2.line(mask, (int(lane.x1), int(lane.y1)), (int(lane.x2), int(lane.y2)),
                     255, max(6, w // 80))

        name = f"{args.video.stem}_{idx:05d}"
        cv2.imwrite(str(args.out / "images" / f"{name}.jpg"),
                    cv2.resize(frame, (512, 256), interpolation=cv2.INTER_AREA))
        small = cv2.resize(mask, (512, 256), interpolation=cv2.INTER_AREA)
        cv2.imwrite(str(args.out / "masks" / f"{name}.png"),
                    ((small > 64) * 255).astype(np.uint8))
        saved += 1
    print(f"Saved {saved} training pairs -> {args.out}")


if __name__ == "__main__":
    main()

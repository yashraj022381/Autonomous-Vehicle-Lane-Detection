"""Convert TuSimple labels into (image, mask) pairs.

    python -m scripts.tusimple_to_masks --root path/to/train_set --out data/tusimple_small --limit 3000
"""
import argparse
import json
from pathlib import Path
import cv2
import numpy as np


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--root", required=True, type=Path)
    p.add_argument("--out", required=True, type=Path)
    p.add_argument("--limit", type=int, default=3000, help="max images (CPU-friendly)")
    p.add_argument("--thickness", type=int, default=12)
    args = p.parse_args()

    (args.out / "images").mkdir(parents=True, exist_ok=True)
    (args.out / "masks").mkdir(parents=True, exist_ok=True)

    items = []
    for label_file in sorted(args.root.glob("label_data_*.json")):
        with open(label_file) as f:
            items += [json.loads(line) for line in f]
    step = max(1, len(items) // args.limit)       # spread picks across all clips
    items = items[::step][: args.limit]
    if not items:
        raise SystemExit(f"No label_data_*.json found in {args.root}. Check --root.")

    saved = skipped = 0
    for i, item in enumerate(items):
        img = cv2.imread(str(args.root / item["raw_file"]))
        if img is None:
            skipped += 1
            if skipped <= 3:
                print("Could not read:", args.root / item["raw_file"])
            continue
        h, w = img.shape[:2]
        mask = np.zeros((h, w), np.uint8)
        for lane in item["lanes"]:
            pts = [(x, y) for x, y in zip(lane, item["h_samples"]) if x >= 0]
            if len(pts) >= 2:
                cv2.polylines(mask, [np.array(pts, np.int32)], False, 255, args.thickness)

        name = item["raw_file"].replace("/", "_").rsplit(".", 1)[0]
        cv2.imwrite(str(args.out / "images" / f"{name}.jpg"),
                    cv2.resize(img, (512, 256), interpolation=cv2.INTER_AREA))
        small = cv2.resize(mask, (512, 256), interpolation=cv2.INTER_AREA)
        cv2.imwrite(str(args.out / "masks" / f"{name}.png"),
                    ((small > 64) * 255).astype(np.uint8))
        saved += 1

    print(f"Label entries: {len(items)} | saved: {saved} | unreadable: {skipped}")
    print("Done ->", args.out)


if __name__ == "__main__":
    main()

"""Run classical lane detection on an image or a video from the terminal.

Examples:
    python -m scripts.run_classical --input data/road.mp4
    python -m scripts.run_classical --input data/road.jpg --output outputs/out.jpg
"""
import argparse
from pathlib import Path
import cv2

from lane_detection.classical.pipeline import ClassicalLaneDetector
from lane_detection.metrics import FPSMeter

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp"}


def resize_to_width(frame, width):
    if width is None or frame.shape[1] <= width:
        return frame
    scale = width / frame.shape[1]
    return cv2.resize(frame, (width, int(frame.shape[0] * scale)))


def put_stats(frame, fps, ms):
    text = f"FPS: {fps:5.1f} | {ms:5.1f} ms"
    cv2.putText(frame, text, (15, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.9,
                (0, 0, 0), 4, cv2.LINE_AA)
    cv2.putText(frame, text, (15, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.9,
                (255, 255, 255), 2, cv2.LINE_AA)


def run_image(src: Path, dst: Path, width):
    frame = cv2.imread(str(src))
    if frame is None:
        raise FileNotFoundError(f"Could not read image: {src}")
    frame = resize_to_width(frame, width)
    result = ClassicalLaneDetector().process(frame, smooth=False)
    cv2.imwrite(str(dst), result.overlay)
    print(f"Saved {dst}  ({result.processing_ms:.1f} ms)")


def run_video(src: Path, dst: Path, width, show):
    cap = cv2.VideoCapture(str(src))
    if not cap.isOpened():
        raise FileNotFoundError(f"Could not open video: {src}")

    src_fps = cap.get(cv2.CAP_PROP_FPS) or 30
    detector, meter, writer = ClassicalLaneDetector(), FPSMeter(), None
    n = 0

    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frame = resize_to_width(frame, width)
        result = detector.process(frame)
        fps = meter.tick()
        put_stats(result.overlay, fps, result.processing_ms)

        if writer is None:
            h, w = result.overlay.shape[:2]
            writer = cv2.VideoWriter(str(dst), cv2.VideoWriter_fourcc(*"mp4v"),
                                     src_fps, (w, h))
        writer.write(result.overlay)
        n += 1

        if show:
            cv2.imshow("Classical Lane Detection (press q to quit)", result.overlay)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    cap.release()
    if writer:
        writer.release()
    cv2.destroyAllWindows()
    print(f"Done: {n} frames -> {dst}  (avg {meter.tick():.1f} FPS)")


def main():
    p = argparse.ArgumentParser(description="Classical lane detection")
    p.add_argument("--input", required=True, type=Path)
    p.add_argument("--output", type=Path, default=None)
    p.add_argument("--width", type=int, default=960, help="resize for speed")
    p.add_argument("--show", action="store_true", help="live preview window")
    args = p.parse_args()

    Path("outputs").mkdir(exist_ok=True)
    is_image = args.input.suffix.lower() in IMAGE_EXTS
    out = args.output or Path("outputs") / f"{args.input.stem}_classical" \
        f"{args.input.suffix if is_image else '.mp4'}"

    (run_image if is_image else run_video)(*(
        (args.input, out, args.width) if is_image
        else (args.input, out, args.width, args.show)))


if __name__ == "__main__":
    main()

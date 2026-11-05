"""Run lane detection (classical or deep) on an image or a video.

    python -m scripts.run_lane --method classical --input data/solidWhiteRight.mp4 --show
    python -m scripts.run_lane --method deep --input data/solidWhiteRight.mp4 --show
"""
import argparse
from pathlib import Path
import cv2
from lane_detection.metrics import FPSMeter

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp"}


def build_detector(method: str, weights: str):
    if method == "classical":
        from lane_detection.classical.pipeline import ClassicalLaneDetector
        return ClassicalLaneDetector()
    from lane_detection.deep_learning.inference import DeepLaneDetector  # loads torch only here
    return DeepLaneDetector(weights)


def resize_to_width(frame, width):
    if width is None or frame.shape[1] <= width:
        return frame
    return cv2.resize(frame, (width, int(frame.shape[0] * width / frame.shape[1])))


def put_stats(frame, label, fps, ms):
    text = f"{label} | FPS {fps:5.1f} | {ms:5.1f} ms"
    for color, thick in (((0, 0, 0), 4), ((255, 255, 255), 2)):
        cv2.putText(frame, text, (15, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.8,
                    color, thick, cv2.LINE_AA)


def run_image(detector, src, dst, width):
    frame = cv2.imread(str(src))
    if frame is None:
        raise FileNotFoundError(f"Could not read image: {src}")
    result = detector.process(resize_to_width(frame, width), smooth=False)
    cv2.imwrite(str(dst), result.overlay)
    print(f"Saved {dst} ({result.processing_ms:.1f} ms)")


def run_video(detector, src, dst, width, show, label):
    cap = cv2.VideoCapture(str(src))
    if not cap.isOpened():
        raise FileNotFoundError(f"Could not open video: {src}")
    src_fps = cap.get(cv2.CAP_PROP_FPS) or 30
    meter, writer, n, fps = FPSMeter(), None, 0, 0.0

    while True:
        ok, frame = cap.read()
        if not ok:
            break
        result = detector.process(resize_to_width(frame, width))
        fps = meter.tick()
        put_stats(result.overlay, label, fps, result.processing_ms)
        if writer is None:
            h, w = result.overlay.shape[:2]
            writer = cv2.VideoWriter(str(dst), cv2.VideoWriter_fourcc(*"mp4v"),
                                     src_fps, (w, h))
        writer.write(result.overlay)
        n += 1
        if show:
            cv2.imshow(f"{label} lane detection (press q to quit)", result.overlay)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    cap.release()
    if writer:
        writer.release()
    cv2.destroyAllWindows()
    print(f"Done: {n} frames -> {dst} (last FPS {fps:.1f})")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--method", choices=["classical", "deep"], default="classical")
    p.add_argument("--input", required=True, type=Path)
    p.add_argument("--output", type=Path, default=None)
    p.add_argument("--weights", default="weights/unet_lane.pt")
    p.add_argument("--width", type=int, default=960)
    p.add_argument("--show", action="store_true")
    args = p.parse_args()

    Path("outputs").mkdir(exist_ok=True)
    is_image = args.input.suffix.lower() in IMAGE_EXTS
    ext = args.input.suffix if is_image else ".mp4"
    out = args.output or Path("outputs") / f"{args.input.stem}_{args.method}{ext}"

    detector = build_detector(args.method, args.weights)
    if is_image:
        run_image(detector, args.input, out, args.width)
    else:
        run_video(detector, args.input, out, args.width, args.show, args.method.upper())


if __name__ == "__main__":
    main()

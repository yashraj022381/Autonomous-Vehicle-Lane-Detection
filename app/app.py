"""Lane Detection Studio.   Run:  streamlit run app/app.py"""
import os
import sys
import tempfile
import time
from dataclasses import dataclass, field, replace
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]   # the repo folder (one level above app/)
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


import cv2
import numpy as np
import streamlit as st

from lane_detection.app_helpers import (RollingSpeed, add_label,
                                         resize_to_width, side_by_side)
from lane_detection.auto_select import (candidates_from_choices, run_auto,
                                        sample_video_frames)
from lane_detection.classical.pipeline import ClassicalLaneDetector
from lane_detection.config import ClassicalConfig, DeepConfig
from lane_detection.insights import compute_insights, probability_heatmap

WEIGHT_CHOICES = {
    "Fast v2 + CULane (256x128)": str(ROOT / "weights" / "unet_lane_fast_v2.pt"),
    "Fast (256x128)": str(ROOT / "weights" / "unet_lane_fast.pt"),
    "Accurate v2 + CULane (512x256)": str(ROOT / "weights" / "unet_lane_v2.pt"),
    "Accurate (512x256)": str(ROOT / "weights" / "unet_lane.pt"),
}
AUTO_LABEL = "🤖 Auto (picks the best for your file)"
MODES = ["Classical", "Deep Learning", "Comparison"]

st.set_page_config(page_title="Lane Detection Studio", page_icon="🛣️", layout="wide")
UI_REFRESH_SECONDS = 0.1       # redraw the pictures at most 10 times per second


# ---------------------------------------------------------------- settings
@dataclass
class Settings:
    mode: str
    width: int
    smooth: bool
    classical_cfg: ClassicalConfig
    deep_cfg: DeepConfig
    every: int
    max_frames: int
    weights: str
    auto: bool = False                               # True = let the app pick the model
    available: dict = field(default_factory=dict)    # {label: path} of models that exist


def sidebar() -> Settings:
    st.sidebar.title("⚙️ Settings")
    mode = st.sidebar.radio("Mode", MODES)
    width = st.sidebar.select_slider("Processing width (px)",
                                     [480, 640, 800, 960, 1280], value=640,
                                     help="Smaller = faster")
    smooth = st.sidebar.checkbox("Temporal smoothing (video)", True)

    c_cfg, d_cfg = ClassicalConfig(), DeepConfig()
    if mode in ("Classical", "Comparison"):
        st.sidebar.subheader("Classical")
        low = st.sidebar.slider("Canny low", 10, 200, c_cfg.canny_low)
        high = st.sidebar.slider("Canny high", 50, 300, c_cfg.canny_high)
        if high <= low:
            st.sidebar.warning("Canny high must be above low; adjusted.")
            high = low + 1
        hough = st.sidebar.slider("Hough votes", 5, 100, c_cfg.hough_threshold)
        color = st.sidebar.checkbox("White/yellow color filter", c_cfg.use_color_mask)
        c_cfg = replace(c_cfg, canny_low=low, canny_high=high,
                        hough_threshold=hough, use_color_mask=color)

    weights = next(iter(WEIGHT_CHOICES.values()))
    available, auto = {}, False
    if mode in ("Deep Learning", "Comparison"):
        st.sidebar.subheader("Deep Learning")
        available = {k: v for k, v in WEIGHT_CHOICES.items() if Path(v).exists()}
        if available:
            pick = st.sidebar.selectbox("Model", [AUTO_LABEL] + list(available))
            auto = pick == AUTO_LABEL
            if auto:
                weights = next(iter(available.values()))     # used until Auto has chosen
                st.sidebar.caption("Auto tries the models on your file and picks one. "
                                   "Videos use the fast models only.")
            else:
                weights = available[pick]
        thr = st.sidebar.slider("Lane threshold", 0.1, 0.9, d_cfg.threshold, 0.05)
        ego = st.sidebar.checkbox("Show only my lane", d_cfg.ego_only,
                                  help="Hide neighbouring lanes and stray blobs")
        d_cfg = replace(d_cfg, threshold=thr, ego_only=ego)

    st.sidebar.subheader("Video options")
    every = st.sidebar.number_input("Process every Nth frame", 1, 10, 1)
    max_frames = st.sidebar.number_input("Max frames", 10, 5000, 300, step=50)
    return Settings(mode, width, smooth, c_cfg, d_cfg, int(every), int(max_frames),
                    weights, auto, available)


# --------------------------------------------------------------- detectors
@st.cache_resource(show_spinner="Loading U-Net weights...")
def load_deep(weights: str):
    from lane_detection.deep_learning.inference import DeepLaneDetector
    return DeepLaneDetector(weights)          # loaded ONCE, reused on every rerun


def build_detectors(s: Settings) -> dict:
    dets = {}
    if s.mode in ("Classical", "Comparison"):
        dets["Classical"] = ClassicalLaneDetector(s.classical_cfg)
    if s.mode in ("Deep Learning", "Comparison"):
        try:
            deep = load_deep(s.weights)
        except FileNotFoundError:
            st.error(f"No trained model at `{s.weights}`. Train first: "
                     "`python -m train --data data/pseudo`")
            st.stop()
        deep.cfg = s.deep_cfg
        deep.reset()
        dets["Deep Learning"] = deep
    return dets


# -------------------------------------------------------------------- auto
def choose_model_automatically(s: Settings, upload, kind: str):
    """Let the models audition on the uploaded file and return the winner (or None).

    The result is remembered for this file and these settings, so clicking buttons
    does not repeat the audition."""
    key = (upload.name, upload.size, kind, s.width, s.deep_cfg.threshold,
           s.deep_cfg.ego_only, tuple(s.available))
    saved = st.session_state.get("auto_choice")
    if saved and saved[0] == key:
        return saved[1]

    is_video = kind == "Video"
    with st.spinner("🤖 Auto is trying the models on your file..."):
        if is_video:
            suffix = Path(upload.name).suffix or ".mp4"
            with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
                tmp.write(upload.getbuffer())
                path = tmp.name
            try:
                frames = sample_video_frames(path, width=s.width)
            finally:
                os.remove(path)
        else:
            frame = cv2.imdecode(np.frombuffer(upload.getvalue(), np.uint8), cv2.IMREAD_COLOR)
            frames = [] if frame is None else [resize_to_width(frame, s.width)]
        if not frames:
            return None

        def loader(weights_path: str):
            det = load_deep(weights_path)
            det.cfg = s.deep_cfg
            return det

        choice = run_auto(frames, candidates_from_choices(s.available), loader,
                          is_video, s.deep_cfg.threshold, s.deep_cfg.ego_only)
    st.session_state["auto_choice"] = (key, choice)
    return choice


def explain_auto(choice) -> None:
    st.info(f"🤖 **Auto picked {choice.label}.** {choice.reason}")
    with st.expander("How did Auto decide?"):
        for label, score in sorted(choice.scores.items(), key=lambda kv: -kv[1]):
            st.write(f"- {label}: **{score:.0%}**")
        st.caption(f"Score = average confidence over {choice.frames_used} sample frame(s); "
                   "0% means no lane was found. Models not listed were not tried "
                   "(video uses the fast models only). Confidence is not the same as accuracy.")


# ------------------------------------------------------------------- panel
def show_image(placeholder, frame, **kw):
    try:                                       # newer Streamlit
        placeholder.image(frame, width="stretch", **kw)
    except Exception:                          # older Streamlit
        placeholder.image(frame, use_container_width=True, **kw)


class Panel:
    """One column: a title, a picture, and an FPS + time stopwatch."""

    def __init__(self, container, title: str):
        container.subheader(title)
        self.image = container.empty()
        c1, c2 = container.columns(2)
        self.fps_box, self.ms_box = c1.empty(), c2.empty()
        self.speed = RollingSpeed()

    def record(self, ms: float):
        """Count one frame in the stopwatch. Cheap: nothing is sent to the browser."""
        self.speed.add(ms)

    def show(self, frame_bgr):
        """Draw the picture and the numbers. Slow: this talks to the browser."""
        show_image(self.image, frame_bgr, channels="BGR")
        self.fps_box.metric("FPS", f"{self.speed.fps:.1f}")
        self.ms_box.metric("Processing time", f"{self.speed.avg_ms:.1f} ms")

    def update(self, frame_bgr, ms: float):
        self.record(ms)
        self.show(frame_bgr)


def ai_panel(frame, res, threshold: float, ego_only: bool = True):
    prob = res.debug["probability"]
    ins = compute_insights(prob, threshold, ego_only)
    st.markdown(f"**Confidence:** {ins.confidence:.0%} &nbsp;|&nbsp; "
                f"**Lane coverage:** {ins.coverage:.1f}% &nbsp;|&nbsp; "
                f"**Car position:** {ins.status}")
    show_image(st, probability_heatmap(frame, prob), channels="BGR",
               caption="Heatmap: red = model is very sure this is lane")


# ------------------------------------------------------------------- image
def run_image(dets: dict, data: bytes, s: Settings):
    frame = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
    if frame is None:
        st.error("Could not read that image.")
        return
    frame = resize_to_width(frame, s.width)

    cols = st.columns(len(dets))
    for col, (name, det) in zip(cols, dets.items()):
        det.process(frame, smooth=False)                 # warm-up (first run is slow)
        res = det.process(frame, smooth=False)
        panel = Panel(col, name)
        panel.update(res.overlay, res.processing_ms)
        if name == "Classical":
            with col.expander("What the computer sees (edges after ROI)"):
                st.image(res.debug["masked_edges"])
        else:
            with col.expander("🧠 AI results (what the model thinks)", expanded=True):
                ai_panel(frame, res, s.deep_cfg.threshold, s.deep_cfg.ego_only)


# ------------------------------------------------------------------- video
def run_video(dets: dict, upload, s: Settings):
    suffix = Path(upload.name).suffix or ".mp4"
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(upload.getbuffer())
        src = tmp.name

    cap = cv2.VideoCapture(src)
    if not cap.isOpened():
        cap.release()
        os.remove(src)
        st.error("Could not open that video.")
        return
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 0
    out_fps = (cap.get(cv2.CAP_PROP_FPS) or 30) / s.every

    for det in dets.values():
        det.reset()                                      # forget the previous video
    panels = {n: Panel(c, n) for c, n in zip(st.columns(len(dets)), dets)}
    progress = st.progress(0.0, text="Processing...")
    ai_box = st.empty()

    out_path = Path(tempfile.gettempdir()) / f"lane_result_{int(time.time())}.mp4"
    writer, idx, done = None, 0, 0
    latest, last_ui = {}, 0.0

    def refresh():
        """Send the newest pictures and numbers to the browser."""
        for name, res in latest.items():
            panels[name].show(res.overlay)
        if "Deep Learning" in latest:
            ins = compute_insights(latest["Deep Learning"].debug["probability"],
                                   s.deep_cfg.threshold, s.deep_cfg.ego_only)
            ai_box.markdown(f"🧠 **AI results** — confidence **{ins.confidence:.0%}** · "
                            f"coverage **{ins.coverage:.1f}%** · car: **{ins.status}**")
        if total:
            progress.progress(min(1.0, idx / total), text=f"Frame {idx}/{total}")

    try:
        while done < s.max_frames:
            ok, frame = cap.read()
            if not ok:
                break
            idx += 1
            if (idx - 1) % s.every:
                continue
            frame = resize_to_width(frame, s.width)

            labelled = []
            for name, det in dets.items():
                res = det.process(frame, smooth=s.smooth)
                panels[name].record(res.processing_ms)   # time EVERY frame...
                latest[name] = res
                labelled.append(add_label(res.overlay, name))

            combined = labelled[0] if len(labelled) == 1 else side_by_side(*labelled)
            if writer is None:
                h, w = combined.shape[:2]
                writer = cv2.VideoWriter(str(out_path), cv2.VideoWriter_fourcc(*"mp4v"),
                                         out_fps, (w, h))
            writer.write(combined)                       # ...and save EVERY frame
            done += 1

            now = time.perf_counter()
            if now - last_ui >= UI_REFRESH_SECONDS:      # ...but redraw the page only
                last_ui = now                            # 10 times per second
                refresh()
        if latest:
            refresh()

    finally:
        cap.release()
        if writer:
            writer.release()
        os.remove(src)
    progress.progress(1.0, text="Done ✅")

    # Save results so the page can still show them after a button click.
    st.session_state["result"] = {
        "video": out_path.read_bytes() if out_path.exists() else None,
        "frames": done,
        "summary": {n: (p.speed.fps, p.speed.avg_ms) for n, p in panels.items()},
    }


def show_saved_result():
    res = st.session_state.get("result")
    if not res:
        return
    st.divider()
    st.subheader("📊 Summary of the last video run")
    st.caption(f"{res['frames']} frames processed")
    cols = st.columns(max(1, len(res["summary"])))
    for col, (name, (fps, ms)) in zip(cols, res["summary"].items()):
        col.metric(f"{name}: average FPS", f"{fps:.1f}", f"{ms:.1f} ms per frame",
                   delta_color="off")
    if res["video"]:
        st.download_button("⬇️ Download result video (.mp4)", res["video"],
                           file_name="lane_result.mp4", mime="video/mp4")
        st.caption("Tip: if your browser can't play it, open it with VLC or Windows Media Player.")


# -------------------------------------------------------------------- main
def main():
    st.title("🛣️ Lane Detection Studio")
    st.caption("Classical computer vision (Canny + Hough) vs Deep Learning (U-Net)")
    s = sidebar()

    kind = st.radio("Input type", ["Image", "Video"], horizontal=True)
    types = ["jpg", "jpeg", "png"] if kind == "Image" else ["mp4", "avi", "mov", "mkv"]
    upload = st.file_uploader(f"Upload a dashcam {kind.lower()}", type=types)

    if upload is None:
        st.info("👆 Upload a dashcam image or video to begin.")
        show_saved_result()
        return

    if s.auto and s.mode != "Classical" and s.available:
        choice = choose_model_automatically(s, upload, kind)
        if choice is None:
            st.warning("Auto could not read your file, so the default model is used.")
        else:
            s = replace(s, weights=choice.weights)
            explain_auto(choice)

    dets = build_detectors(s)
    if kind == "Image":
        run_image(dets, upload.getvalue(), s)
    else:
        if st.button("▶ Run detection", type="primary"):
            run_video(dets, upload, s)
        show_saved_result()


main()

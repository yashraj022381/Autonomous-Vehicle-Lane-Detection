"""ROS-style nodes. Each node = one small job."""
import cv2
import time
import threading
import queue

import numpy as np

from time import perf_counter

from ..app_helpers import resize_to_width
from ..metrics import FPSMeter
from .bus import MessageBus
from .messages import ImageMsg, LaneMsg


class Node:
    def __init__(self, name: str, bus: MessageBus):
        self.name, self.bus = name, bus
        self._thread = None

    def start(self) -> None:
        self._thread = threading.Thread(target=self._run, name=self.name, daemon=True)
        self._thread.start()

    def _run(self) -> None:
        try:
            self.spin()
        except Exception as exc:
            print(f"[{self.name}] crashed: {exc!r}")
            self.bus.shutdown.set()

    def join(self) -> None:
        if self._thread:
            self._thread.join(timeout=2)

    def spin(self) -> None:
        raise NotImplementedError

class CameraNode(Node):
    """Reads a video and publishes frames, paced like a real camera."""


    def __init__(self, name, bus, video, width=640, topic="/camera/image_raw"):
        super().__init__(name, bus)
        self.video, self.width, self.topic = video, width, topic


    def spin(self) -> None:
        cap = cv2.VideoCapture(str(self.video))
        if not cap.isOpened():
            raise FileNotFoundError(f"Could not open video: {self.video}")
        period = 1.0 / (cap.get(cv2.CAP_PROP_FPS) or 30)
        next_time, seq = perf_counter(), 0
        while not self.bus.shutdown.is_set():
            ok, frame = cap.read()
            if not ok:
                break
            seq += 1
            self.bus.publish(self.topic, ImageMsg(seq, perf_counter(),
                                                  resize_to_width(frame, self.width)))
            next_time += period                   # wait so we publish at camera speed
            pause = next_time - perf_counter()
            if pause > 0:
                time.sleep(pause)
        cap.release()
        time.sleep(1.0)                           # let the last frames finish
        self.bus.shutdown.set()
    
class LaneNode(Node):
    """Subscribes to images, runs ONE detector, publishes the result."""

    def __init__(self, name, bus, detector, source, in_topic, out_topic):
        super().__init__(name, bus)
        self.detector, self.source, self.out_topic = detector, source, out_topic
        self.inbox = bus.subscribe(in_topic, queue_size=1)

    def spin(self) -> None:
        while not self.bus.shutdown.is_set():
            try:
                msg = self.inbox.get(timeout=0.1)
            except queue.Empty:
                continue
            res = self.detector.process(msg.frame)
            self.bus.publish(self.out_topic, LaneMsg(
                msg.seq, msg.stamp, self.source, res.overlay,
                res.processing_ms, res.left_lane, res.right_lane))

class DisplayNode(Node):


    def __init__(self, bus, topics, show=True):
        super().__init__("display_node", bus)
        self.topics, self.show = list(topics), show
        self.inboxes = {t: bus.subscribe(t, queue_size=1) for t in self.topics}
        self.meters = {t: FPSMeter() for t in self.topics}

    @staticmethod
    def _label(frame, text):
        for color, thick in (((0, 0, 0), 3), ((255, 255, 255), 1)):
            cv2.putText(frame, text, (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                        color, thick, cv2.LINE_AA)

    def spin(self) -> None:
        latest = {}
        try:
            while not self.bus.shutdown.is_set():
                for topic, inbox in self.inboxes.items():
                    try:
                        msg = inbox.get_nowait()
                    except queue.Empty:
                        continue
                    hz = self.meters[topic].tick()
                    lag_ms = (perf_counter() - msg.stamp) * 1000   # camera -> screen
                    frame = msg.overlay.copy()
                    self._label(frame, f"{msg.source} | {hz:.1f} Hz | lag {lag_ms:.0f} ms")
                    latest[topic] = frame
                if self.show and latest:
                    ordered = [latest[t] for t in self.topics if t in latest]
                    cv2.imshow("ros_style view (press q to quit)", np.hstack(ordered))
                    if cv2.waitKey(1) & 0xFF == ord("q"):
                        self.bus.shutdown.set()
                else:
                    time.sleep(0.01)
        finally:
            cv2.destroyAllWindows()

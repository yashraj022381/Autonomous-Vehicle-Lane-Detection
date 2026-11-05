"""Start the nodes.

    python -m ros_demo --video data/solidWhiteRight.mp4
    python -m ros_demo --video data/solidWhiteRight.mp4 --detectors classical
"""
import argparse

from lane_detection.ros_style.bus import MessageBus
from lane_detection.ros_style.nodes import CameraNode, DisplayNode, LaneNode

CAMERA_TOPIC = "/camera/image_raw"


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--video", required=True)
    p.add_argument("--width", type=int, default=640)
    p.add_argument("--detectors", nargs="+", choices=["classical", "deep"],
                   default=["classical", "deep"])
    p.add_argument("--weights", default="weights/unet_lane.pt")
    p.add_argument("--no-show", action="store_true")
    args = p.parse_args()

    bus = MessageBus()
    nodes = [CameraNode("camera_node", bus, args.video, args.width, CAMERA_TOPIC)]
    out_topics = []
    for kind in args.detectors:
        if kind == "classical":
            from lane_detection.classical.pipeline import ClassicalLaneDetector
            detector = ClassicalLaneDetector()
        else:
            from lane_detection.deep_learning.inference import DeepLaneDetector
            detector = DeepLaneDetector(args.weights)
        topic = f"/lane/{kind}"
        nodes.append(LaneNode(f"{kind}_node", bus, detector, kind, CAMERA_TOPIC, topic))
        out_topics.append(topic)

    display = DisplayNode(bus, out_topics, show=not args.no_show)
    print("Nodes running:", ", ".join(n.name for n in nodes), "+ display_node")
    for n in nodes:                      # all subscriptions exist before the camera starts
        n.start()
    try:
        display.spin()                   # main thread
    finally:
        bus.shutdown.set()
        for n in nodes:
            n.join()
        counts = bus.counts
        print("\nTopic summary (like `ros2 topic list`):")
        for topic in sorted(counts):
            print(f"  {topic:<22}{counts[topic]:>6} messages")
        sent = counts.get(CAMERA_TOPIC, 0)
        for topic in out_topics:
            print(f"  {topic} dropped {sent - counts.get(topic, 0)} of {sent} frames "
                  "(slow node = skips old frames, just like ROS)")


if __name__ == "__main__":
    main()

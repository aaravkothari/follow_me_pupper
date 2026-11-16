"""Print red_cap detections from /detections. Never moves the robot.

    python3 robot/watch_red_cap.py

Needs the stack running (ros2 launch robot/follow_me.launch.py) and NOT
lab/follow_me.py if you want Pupper to stay put. Each line shows where the cap
is (offset -0.5 left .. +0.5 right), its size in px (the distance signal
CAP_STOP_SIZE in lab/target_lock.py is compared against), and its mean color.
"""

import time

import rclpy
from rclpy.node import Node
from vision_msgs.msg import Detection2DArray

RED_CAP_CLASS_ID = "80"
IMAGE_WIDTH = 700


class Watcher(Node):
    def __init__(self):
        super().__init__("red_cap_watcher")
        self.create_subscription(Detection2DArray, "/detections", self.cb, 10)
        self.last_seen = None
        self.create_timer(1.0, self.heartbeat)

    def cb(self, msg):
        caps = [d for d in msg.detections
                if d.results and d.results[0].hypothesis.class_id == RED_CAP_CLASS_ID]
        for i, d in enumerate(sorted(caps, key=lambda d: -d.bbox.size_x * d.bbox.size_y)):
            off = (d.bbox.center.position.x - IMAGE_WIDTH / 2) / IMAGE_WIDTH
            size = max(d.bbox.size_x, d.bbox.size_y)
            print(f"red_cap #{i}  offset {off:+.2f}  size {size:4.0f}px  "
                  f"box {d.bbox.size_x:.0f}x{d.bbox.size_y:.0f}  rgb {d.id}", flush=True)
        if caps:
            self.last_seen = time.time()

    def heartbeat(self):
        if self.last_seen is None or time.time() - self.last_seen > 1.0:
            print("... no red cap", flush=True)


def main():
    rclpy.init()
    node = Watcher()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.try_shutdown()


if __name__ == "__main__":
    main()

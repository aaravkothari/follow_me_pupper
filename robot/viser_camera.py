#!/usr/bin/env python3
"""Stream the Pupper's fisheye camera to a viser web viewer, with detection.

YOLOv8n-seg runs on the Hailo accelerator alongside the stream, so the viewer
shows bounding boxes *and* per-object segmentation masks. Toggle either one from
the "Detection" folder in the browser GUI. When frames come from ROS the same
detections are published on /detections, which is what follow_me.py tracks.

The model always runs on the *undistorted* (equirectangular) image, never on the
raw fisheye -- the lens bends people into shapes YOLO has never seen, and scores
drop badly. That happens regardless of which view the GUI is showing, so
/detections keeps flowing either way; the overlays themselves are drawn in the
equirectangular view, since that is the frame their pixels belong to.

Two ways to get frames:

  picam  (default) -- grab straight from the camera with picamera2. Use this when
                      nothing else has the camera open.
  ros              -- subscribe to /camera/image_raw/compressed. Use this when
                      follow_me.launch.py (and its camera_ros node) is already
                      running, since libcamera only lets one process open the
                      sensor at a time.

Usage:
    python3 robot/viser_camera.py                 # picamera2 source
    python3 robot/viser_camera.py --source ros    # alongside follow_me.launch.py

It also comes up automatically with the lab stack:

    ros2 launch robot/follow_me.launch.py             # add viser:=false to skip it
                                            # add detect:=false to skip the model

Viewing it:
  * Same network as the Pupper -- open http://<pupper-ip>:8080 in your browser.
  * Over SSH -- forward the port from your laptop, then browse to localhost:8080:

        ssh -N -L 8080:localhost:8080 pi@<pupper-ip>

    The script prints the exact command (with this Pi's IP filled in) at startup.
"""

import argparse
import os
import socket
import subprocess
import threading
import time

import cv2
import numpy as np
import viser
import yaml

from fisheye_converter import equirect_height, fisheye_to_equirectangular, load_camera_model
from red_cap import RedCapDetector
from yolo_seg import Detection, HailoYoloSeg, draw_detections

# follow_me.py measures against this reference frame, so publish in it no matter
# what resolution the viewer happens to be showing. The height is pinned too --
# obstacle avoidance thresholds boxes against the bottom of the image, so a
# frame that changes height with the FOV slider would move the goalposts. Since
# the equirectangular projection always spans 180 deg vertically, a fixed height
# means a given y is always the same elevation angle.
DETECTION_PUBLISH_WIDTH = 700
DETECTION_PUBLISH_HEIGHT = 572


class PiCamSource:
    """Frames straight off the imx296 via picamera2."""

    def __init__(self, width, height, white_balance=None):
        from picamera2 import Picamera2

        self.picam2 = Picamera2()
        config = self.picam2.create_preview_configuration(
            main={"format": "RGB888", "size": (width, height)}
        )
        self.picam2.configure(config)
        wb = white_balance or {}
        if not wb.get("awb_enable", True):
            self.picam2.set_controls(
                {"AwbEnable": False, "ColourGains": tuple(wb["colour_gains"])}
            )
        self.picam2.start()
        time.sleep(1.0)  # let AE/AWB settle

    def ok(self):
        return True

    def read(self):
        # picamera2's "RGB888" hands back BGR-ordered arrays.
        return cv2.cvtColor(self.picam2.capture_array(), cv2.COLOR_BGR2RGB)

    def close(self):
        self.picam2.stop()
        self.picam2.close()


class RosSource:
    """Frames from the camera_ros node on /camera/image_raw/compressed."""

    def __init__(self, topic="/camera/image_raw/compressed"):
        import rclpy
        from rclpy.node import Node
        from sensor_msgs.msg import CompressedImage

        rclpy.init()
        self.rclpy = rclpy
        self.node = Node("viser_camera")
        self.lock = threading.Lock()
        self.frame = None

        def callback(msg):
            arr = np.frombuffer(msg.data, np.uint8)
            bgr = cv2.imdecode(arr, cv2.IMREAD_COLOR)
            if bgr is not None:
                with self.lock:
                    self.frame = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

        self.node.create_subscription(CompressedImage, topic, callback, 10)
        self.thread = threading.Thread(target=self._spin, daemon=True)
        self.thread.start()

        print(f"Waiting for frames on {topic} ...")
        deadline = time.time() + 15.0
        while time.time() < deadline:
            with self.lock:
                if self.frame is not None:
                    return
            time.sleep(0.1)
        raise RuntimeError(
            f"No images on {topic} after 15s. Is follow_me.launch.py running? "
            "If not, use --source picam instead."
        )

    def _spin(self):
        # ros2 launch shuts us down with SIGINT/SIGTERM, which rclpy turns into
        # an ExternalShutdownException here. That's a normal exit, not a crash.
        from rclpy.executors import ExternalShutdownException

        try:
            self.rclpy.spin(self.node)
        except (KeyboardInterrupt, ExternalShutdownException):
            pass

    def ok(self):
        return self.rclpy.ok()

    def read(self):
        with self.lock:
            return None if self.frame is None else self.frame.copy()

    def close(self):
        # Order matters: shut down rclpy so spin() returns and the thread exits,
        # and only then destroy the node. Destroying a node that a live spin
        # thread still holds aborts the process during C++ teardown.
        if self.rclpy.ok():
            self.rclpy.shutdown()
        self.thread.join(timeout=2.0)
        self.node.destroy_node()


class DetectionPublisher:
    """Puts the YOLO detections on /detections for the rest of the lab stack.

    Every class goes out, not just the tracked one: follow_me.py picks its target out
    of the stream itself, and it needs to see the *other* objects to notice when
    one of them is standing between the robot and the person it is following.
    /tracking_control is still watched, but only so the GUI can say what follow_me.py
    is chasing.

    Each Detection2D also carries the mean color of its segmentation mask in
    the (otherwise unused) ``id`` field, as "r,g,b" -- for a person this is
    mostly the color of their clothes, which is what the optional color
    re-identification part of the lab matches on. Boxes without a mask get the
    mean of their box crop instead; an empty id means no color was available.
    """

    def __init__(self, node, labels):
        from std_msgs.msg import String
        from vision_msgs.msg import Detection2DArray

        self.node = node
        self.labels = labels
        self.class_name_to_id = {name: i for i, name in enumerate(labels)}
        self.tracking_enabled = False
        self.tracking_object = None
        self.tracking_class_id = None

        self.publisher = node.create_publisher(Detection2DArray, "/detections", 10)
        node.create_subscription(String, "/tracking_control", self._on_control, 10)

    def _on_control(self, msg):
        command = msg.data
        if command.startswith("start:"):
            name = command.split(":", 1)[1]
            if name in self.class_name_to_id:
                self.tracking_enabled = True
                self.tracking_object = name
                self.tracking_class_id = self.class_name_to_id[name]
                self.node.get_logger().info(
                    f"Tracking {name} (class_id={self.tracking_class_id})"
                )
            else:
                self.node.get_logger().warning(f"Unknown object class: {name}")
                self.tracking_enabled = False
                self.tracking_class_id = None
        elif command == "stop":
            self.tracking_enabled = False
            self.tracking_object = None
            self.tracking_class_id = None
            self.node.get_logger().info("Tracking stopped")

    def status(self):
        target = self.tracking_object if self.tracking_enabled else "person"
        return f"all classes (follow_me chases '{target}')"

    @staticmethod
    def mean_color(det, frame):
        """Mean RGB inside the detection's mask (or its box), as "r,g,b" or ""."""
        h, w = frame.shape[:2]
        if det.mask is not None and det.mask.any():
            pixels = frame[det.mask]
        else:
            x1, y1, x2, y2 = (int(round(v)) for v in det.box)
            x1, x2 = max(x1, 0), min(x2, w)
            y1, y2 = max(y1, 0), min(y2, h)
            if x2 <= x1 or y2 <= y1:
                return ""
            pixels = frame[y1:y2, x1:x2].reshape(-1, 3)
        if pixels.size == 0:
            return ""
        r, g, b = pixels.mean(axis=0)
        return f"{r:.0f},{g:.0f},{b:.0f}"

    def publish(self, detections, frame):
        from vision_msgs.msg import Detection2D, Detection2DArray, ObjectHypothesisWithPose

        msg = Detection2DArray()
        msg.header.stamp = self.node.get_clock().now().to_msg()
        msg.header.frame_id = "camera"

        frame_shape = frame.shape[:2]
        sx = DETECTION_PUBLISH_WIDTH / max(frame_shape[1], 1)
        sy = DETECTION_PUBLISH_HEIGHT / max(frame_shape[0], 1)

        for det in detections:
            x1, y1, x2, y2 = det.box
            x1, x2 = x1 * sx, x2 * sx
            y1, y2 = y1 * sy, y2 * sy

            det_msg = Detection2D()
            det_msg.header = msg.header
            det_msg.id = self.mean_color(det, frame)
            det_msg.bbox.center.position.x = float((x1 + x2) / 2)
            det_msg.bbox.center.position.y = float((y1 + y2) / 2)
            det_msg.bbox.size_x = float(x2 - x1)
            det_msg.bbox.size_y = float(y2 - y1)

            hyp = ObjectHypothesisWithPose()
            hyp.hypothesis.class_id = str(det.class_id)
            hyp.hypothesis.score = float(det.score)
            det_msg.results.append(hyp)

            msg.detections.append(det_msg)

        self.publisher.publish(msg)
        return len(msg.detections)


class DetectionWorker:
    """Runs the segmentation model on a background thread.

    The viewer should never stall waiting on the accelerator, so the capture loop
    just hands over the newest frame and draws whatever result is ready. Frames
    that arrive while an inference is in flight are dropped, which is what keeps
    the stream smooth when the model is slower than the camera.
    """

    def __init__(self, model, on_result=None, red_cap=None):
        self.model = model
        self.on_result = on_result
        # Color-threshold detector for the red pen cap (robot/red_cap.py); its
        # hits are appended to YOLO's, as class red_cap.
        self.red_cap = red_cap
        self.enabled = True

        self._lock = threading.Lock()
        self._pending = None
        self._wake = threading.Event()
        self._stop = False
        self._result = ([], (0, 0))
        self._scaled_cache = {}

        self.inference_ms = 0.0
        self.fps = 0.0

        self.thread = threading.Thread(target=self._run, daemon=True)
        self.thread.start()

    def submit(self, frame):
        with self._lock:
            self._pending = frame
        self._wake.set()

    def raw(self):
        """The last detections in the coordinates they were computed in."""
        with self._lock:
            return self._result[0]

    def latest(self, target_shape):
        """Detections scaled to target_shape (h, w), reusing the last scaling."""
        with self._lock:
            detections, shape = self._result
        if not detections:
            return []
        if shape == target_shape:
            return detections
        cached = self._scaled_cache.get(target_shape)
        if cached is not None and cached[0] is detections:
            return cached[1]
        scaled = _scale_detections(detections, shape, target_shape)
        self._scaled_cache = {target_shape: (detections, scaled)}
        return scaled

    def _run(self):
        last_done = None
        while not self._stop:
            self._wake.wait(timeout=0.5)
            self._wake.clear()
            with self._lock:
                frame, self._pending = self._pending, None
            if frame is None or self._stop:
                continue

            try:
                detections = self.model.infer(frame)
            except Exception as exc:  # keep streaming even if the chip hiccups
                print(f"[detection] inference failed: {exc}")
                time.sleep(0.5)
                continue
            if self.red_cap is not None:
                try:
                    detections = list(detections) + self.red_cap.detect(frame)
                except Exception as exc:
                    print(f"[red_cap] failed: {exc}")

            with self._lock:
                self._result = (detections, frame.shape[:2])
            self.inference_ms = self.model.last_inference_ms

            now = time.time()
            if last_done is not None:
                instant = 1.0 / max(now - last_done, 1e-6)
                self.fps = instant if self.fps == 0 else 0.7 * self.fps + 0.3 * instant
            last_done = now

            if self.on_result is not None:
                self.on_result(detections, frame)

    def close(self):
        self._stop = True
        self._wake.set()
        self.thread.join(timeout=2.0)


def _scale_detections(detections, src_shape, dst_shape):
    """Re-express detections computed at src_shape (h, w) in dst_shape pixels."""
    sy = dst_shape[0] / max(src_shape[0], 1)
    sx = dst_shape[1] / max(src_shape[1], 1)
    scaled = []
    for det in detections:
        x1, y1, x2, y2 = det.box
        mask = det.mask
        if mask is not None:
            mask = (
                cv2.resize(
                    mask.astype(np.uint8),
                    (dst_shape[1], dst_shape[0]),
                    interpolation=cv2.INTER_NEAREST,
                ).astype(bool)
            )
        scaled.append(
            Detection(
                det.class_id,
                det.class_name,
                det.score,
                (x1 * sx, y1 * sy, x2 * sx, y2 * sy),
                mask,
            )
        )
    return scaled


def focus_score(frame):
    """Sharpness of the middle of the fisheye image circle.

    Variance of the Laplacian: near zero for a blurred image, and it climbs as
    detail comes in, so you can turn the lens and watch for the peak. Measured
    on the centre crop because that's where the lens is sharpest and it keeps
    the black surround outside the image circle from diluting the number.
    """
    h, w = frame.shape[:2]
    crop = frame[h // 3 : 2 * h // 3, w // 3 : 2 * w // 3]
    small = cv2.resize(crop, (320, 240), interpolation=cv2.INTER_AREA)
    gray = cv2.cvtColor(small, cv2.COLOR_RGB2GRAY)
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


def focus_verdict(score):
    if score < 5:
        return "way out of focus"
    if score < 20:
        return "badly out of focus"
    if score < 60:
        return "soft - keep turning"
    if score < 150:
        return "close"
    return "sharp"


def letterbox(frame, aspect):
    """Pad `frame` with black bars until its shape matches `aspect` (width/height).

    viser stretches a background image across the entire canvas, so a frame whose
    shape does not match the browser window gets squashed to fit. At the default
    220 deg FOV the equirectangular frame is 1.22:1 while a typical window is
    1.78:1, which stretches everything sideways by ~45%. Padding first means the
    picture keeps the true proportions fisheye_to_equirectangular gave it, and the
    leftover space shows up as bars instead of distortion.
    """
    h, w = frame.shape[:2]
    if aspect <= 0:
        return frame

    # Grow the short axis only -- never crop, never rescale the picture itself.
    target_w = max(w, int(round(h * aspect)))
    target_h = max(h, int(round(target_w / aspect)))
    if (target_w, target_h) == (w, h):
        return frame

    top = (target_h - h) // 2
    left = (target_w - w) // 2
    return cv2.copyMakeBorder(
        frame,
        top,
        target_h - h - top,
        left,
        target_w - w - left,
        cv2.BORDER_CONSTANT,
        value=(0, 0, 0),
    )


class StreamPacer:
    """Skip encoding frames a client's link cannot carry.

    viser already guarantees the newest frame wins: pushing a background image
    evicts any earlier one still waiting in that client's buffer, so the viewer
    never plays back a queue of stale video. What it cannot do is give back the
    CPU already spent encoding the frame that got evicted -- about 3 ms of Pi
    time per frame that nobody will ever see.

    Per-client message buffers pop a message the moment the socket accepts it
    (the broadcast buffer instead keeps messages around for late joiners, so it
    cannot tell us anything). A message id still sitting in a client's buffer
    therefore means that client is genuinely behind, and we skip the frame
    outright: no padding, no JPEG, no queue growth. The send rate then follows
    the link instead of fighting it, which is what keeps latency flat when the
    wifi degrades.
    """

    def __init__(self, stall_timeout=2.0):
        # Ceiling on how long we will keep skipping before forcing a frame
        # through anyway. viser's buffer internals are private, so if a future
        # version stops popping sent messages this timeout is what keeps the
        # stream alive instead of frozen.
        self.stall_timeout = stall_timeout
        self._pending = {}
        self._blocked_since = {}
        self.dropped = 0

    @staticmethod
    def _buffer(client):
        try:
            return client._websock_connection.get_message_buffer()
        except Exception:
            return None  # private API moved: pace nothing, stream normally

    def ready(self, client_id, client, now):
        """True when this client has drained the frame we sent it last."""
        pending = self._pending.get(client_id)
        if pending is None:
            return True
        buffer = self._buffer(client)
        if buffer is None:
            return True

        queued = buffer.message_from_id.get(pending)
        # Another thread may have taken the id we predicted; anything that is
        # not a background image means our frame already went out.
        if queued is None or type(queued).__name__ != "BackgroundImageMessage":
            self._blocked_since.pop(client_id, None)
            return True

        since = self._blocked_since.setdefault(client_id, now)
        if now - since >= self.stall_timeout:
            self._blocked_since[client_id] = now
            return True

        self.dropped += 1
        return False

    def claim(self, client):
        """Id the next push will land on, read before pushing."""
        buffer = self._buffer(client)
        return None if buffer is None else buffer.message_counter

    def note_sent(self, client_id, message_id):
        self._pending[client_id] = message_id

    def forget_disconnected(self, live_ids):
        for client_id in [c for c in self._pending if c not in live_ids]:
            self._pending.pop(client_id, None)
            self._blocked_since.pop(client_id, None)


def canvas_aspects(server):
    """Canvas width/height for each client that has reported its camera yet.

    viser raises rather than guessing when a client has connected but not sent
    camera state, so those clients are skipped until their first update.
    """
    aspects = {}
    for client_id, client in server.get_clients().items():
        try:
            aspect = client.camera.aspect
        except Exception:
            continue
        if aspect and aspect > 0:
            aspects[client_id] = aspect
    return aspects


def push_background(server, frame, quality, lock_aspect, pacer=None):
    """Send `frame` as the scene background. Returns how many clients it went to.

    With no pacer this broadcasts once, which is the cheapest way to feed several
    identical windows. With one, each client is handled separately: that costs an
    encode per viewer, but it is the only way to tell who is behind, and a client
    on bad wifi then simply misses frames instead of accumulating them.
    """
    clients = server.get_clients()
    if not clients:
        # Nobody is watching. Encoding here would be pure waste -- and viser
        # keeps the last broadcast frame for whoever connects next anyway.
        return 0

    aspects = canvas_aspects(server) if lock_aspect else {}

    if pacer is None:
        # One encode for everyone. Only safe when every window has the same
        # shape; otherwise fall through to the per-client path below.
        unique = {round(a, 2) for a in aspects.values()}
        if len(unique) <= 1:
            padded = letterbox(frame, unique.pop()) if unique else frame
            server.scene.set_background_image(
                padded, format="jpeg", jpeg_quality=quality
            )
            return len(clients)

    pacer_now = time.time()
    if pacer is not None:
        pacer.forget_disconnected(clients.keys())

    sent = 0
    for client_id, client in clients.items():
        if pacer is not None and not pacer.ready(client_id, client, pacer_now):
            continue
        aspect = aspects.get(client_id)
        padded = letterbox(frame, aspect) if aspect else frame
        claimed = pacer.claim(client) if pacer is not None else None
        client.scene.set_background_image(padded, format="jpeg", jpeg_quality=quality)
        if pacer is not None:
            pacer.note_sent(client_id, claimed)
        sent += 1
    return sent


def local_ips():
    """Every non-loopback IPv4 address this Pi is reachable at."""
    try:
        out = subprocess.check_output(["hostname", "-I"], text=True, timeout=5)
    except Exception:
        return []
    return [ip for ip in out.split() if ":" not in ip and not ip.startswith("127.")]


def print_access_banner(port):
    ips = local_ips()
    user = subprocess.getoutput("whoami").strip() or "pi"
    host = socket.gethostname()

    print("\n" + "=" * 68)
    print(f"  Viser camera stream is live on port {port}")
    print("=" * 68)
    if ips:
        print("\n  Same Wi-Fi as the Pupper? Open this in your laptop's browser:")
        for ip in ips:
            print(f"      http://{ip}:{port}")
    print("\n  Viewing over SSH? Run this on your LAPTOP (new terminal),")
    print("  then open http://localhost:%d there:" % port)
    target = ips[0] if ips else f"{host}.local"
    print(f"\n      ssh -N -L {port}:localhost:{port} {user}@{target}")
    print("\n  (-N just holds the tunnel open; leave that terminal running.)")
    print("=" * 68 + "\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", choices=["picam", "ros"], default="picam")
    parser.add_argument("--port", type=int, default=8080)
    parser.add_argument("--width", type=int, default=1400, help="capture width (picam)")
    parser.add_argument("--height", type=int, default=1050, help="capture height (picam)")
    parser.add_argument(
        "--camera-params",
        default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "camera_params.yaml"),
    )
    parser.add_argument(
        "--swap-rb",
        action="store_true",
        help="start with red and blue swapped (also toggleable in the GUI)",
    )
    parser.add_argument(
        "--detect",
        dest="detect",
        action="store_true",
        default=True,
        help="run YOLOv8n-seg (boxes + masks) on the stream (default)",
    )
    parser.add_argument("--no-detect", dest="detect", action="store_false")
    parser.add_argument(
        "--model",
        default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "yolov8n_seg.hef"),
    )
    parser.add_argument(
        "--detect-width",
        type=int,
        default=700,
        help="frames are shrunk to this width before inference",
    )
    parser.add_argument(
        "--publish-detections",
        dest="publish_detections",
        action="store_true",
        default=True,
        help="publish /detections for follow_me.py (ROS source only, default)",
    )
    parser.add_argument(
        "--no-publish-detections", dest="publish_detections", action="store_false"
    )
    args = parser.parse_args()

    if args.source == "picam":
        with open(args.camera_params) as f:
            white_balance = yaml.safe_load(f).get("white_balance")
        source = PiCamSource(args.width, args.height, white_balance)
    else:
        source = RosSource()

    server = viser.ViserServer(port=args.port, label="Pupper Camera")

    with server.gui.add_folder("Camera"):
        # Equirectangular by default: it is the view the detector runs on, so it
        # is the only one that can show the boxes and masks.
        gui_view = server.gui.add_dropdown(
            "View", ("fisheye", "equirectangular"), initial_value="equirectangular"
        )
        gui_fov = server.gui.add_slider(
            "Equirect FOV (deg)", min=90, max=280, step=5, initial_value=220
        )
        gui_rotate = server.gui.add_checkbox("Rotate 180", False)
        # Off, the picture is stretched to fill the window and proportions go
        # wrong; on, it keeps its true shape and pads with bars instead.
        gui_lock_aspect = server.gui.add_checkbox("Lock aspect ratio", True)
        # The camera stack occasionally hands frames back with red and blue the
        # wrong way round (skin goes blue, the floor goes purple). Flip it here
        # and both the view and the detector see the corrected colours.
        gui_swap_rb = server.gui.add_checkbox("Swap R/B", args.swap_rb)
        gui_scale = server.gui.add_slider(
            "Display scale", min=0.25, max=1.0, step=0.05, initial_value=0.5
        )
        gui_quality = server.gui.add_slider(
            "JPEG quality", min=20, max=95, step=5, initial_value=70
        )
        gui_fps_target = server.gui.add_slider(
            "Target FPS", min=1, max=30, step=1, initial_value=15
        )
        # On a weak link, sending every frame just grows a queue the viewer has
        # to chew through, so the picture drifts further behind real time the
        # longer you watch. Skipping frames a client has not kept up with holds
        # latency flat: you lose smoothness, never currency.
        gui_pace = server.gui.add_checkbox("Drop frames when behind", True)

    # YOLOv8n-seg on the Hailo chip: one model, two outputs -- boxes and a mask
    # per object. If the accelerator is busy (another process holds it, say)
    # we say so and keep streaming video without detection.
    detector = None
    publisher = None
    if args.detect:
        try:
            detector = HailoYoloSeg(model_path=args.model)
            print(f"Loaded {os.path.basename(args.model)} ({len(detector.labels)} classes).")
        except Exception as exc:
            print(f"[detection] disabled: {exc}")

    if detector is not None and args.publish_detections and isinstance(source, RosSource):
        publisher = DetectionPublisher(source.node, detector.labels)
        print("Publishing /detections (follow_me.py can track off this).")

    with server.gui.add_folder("Detection"):
        gui_detect = server.gui.add_checkbox(
            "Run YOLOv8n-seg", detector is not None, disabled=detector is None
        )
        gui_boxes = server.gui.add_checkbox("Bounding boxes", True)
        gui_masks = server.gui.add_checkbox("Segmentation masks", True)
        gui_conf = server.gui.add_slider(
            "Confidence", min=0.1, max=0.9, step=0.05, initial_value=0.4
        )
        gui_mask_alpha = server.gui.add_slider(
            "Mask opacity", min=0.1, max=0.9, step=0.05, initial_value=0.45
        )
        gui_detect_fps_target = server.gui.add_slider(
            "Detect FPS", min=1, max=20, step=1, initial_value=10
        )
        gui_detect_input = server.gui.add_text(
            "Runs on", initial_value="equirectangular image", disabled=True
        )
        gui_objects = server.gui.add_text(
            "Objects", initial_value="none" if detector else "model not loaded", disabled=True
        )
        gui_detect_stats = server.gui.add_text("Detector", initial_value="--", disabled=True)
        gui_publish = server.gui.add_text(
            "/detections",
            initial_value="off" if publisher is None else publisher.status(),
            disabled=True,
        )

    # Red pen cap: HSV color threshold (robot/red_cap.py), published as class
    # red_cap alongside YOLO's detections. Tune until only the cap is boxed.
    red_cap = RedCapDetector()
    with server.gui.add_folder("Red cap"):
        gui_rc_enable = server.gui.add_checkbox("Detect red cap", True)
        gui_rc_hue = server.gui.add_slider("Hue width", min=2, max=25, step=1, initial_value=red_cap.hue_width)
        gui_rc_sat = server.gui.add_slider("Min saturation", min=0, max=255, step=5, initial_value=red_cap.min_saturation)
        gui_rc_val = server.gui.add_slider("Min value", min=0, max=255, step=5, initial_value=red_cap.min_value)
        gui_rc_area = server.gui.add_slider("Min area (px)", min=1, max=500, step=1, initial_value=red_cap.min_area)

    # Focus aid: the lens is manual-focus, so give a number to turn it against.
    # Laplacian variance on the middle of the image circle -- peaks at best focus.
    with server.gui.add_folder("Focus meter"):
        gui_focus = server.gui.add_text("Focus score", initial_value="--", disabled=True)
        gui_focus_best = server.gui.add_text("Best so far", initial_value="--", disabled=True)
        gui_focus_verdict = server.gui.add_text("Verdict", initial_value="--", disabled=True)
        gui_focus_reset = server.gui.add_button("Reset peak")

    best_focus = [0.0]

    @gui_focus_reset.on_click
    def _(_) -> None:
        best_focus[0] = 0.0

    with server.gui.add_folder("Status"):
        gui_fps = server.gui.add_text("Measured FPS", initial_value="--", disabled=True)
        gui_res = server.gui.add_text("Sent resolution", initial_value="--", disabled=True)
        gui_clients = server.gui.add_text("Clients", initial_value="0", disabled=True)
        gui_dropped = server.gui.add_text(
            "Dropped (behind)", initial_value="0", disabled=True
        )

    gui_preview = server.gui.add_image(np.zeros((2, 2, 3), np.uint8), label="Live feed")

    worker = None
    if detector is not None:
        on_result = None
        if publisher is not None:
            on_result = lambda dets, frame: publisher.publish(dets, frame)
        worker = DetectionWorker(detector, on_result=on_result, red_cap=red_cap)

    pacer = StreamPacer()
    fisheye_model = None
    model_shape = None
    last_report = time.time()
    last_submit = 0.0
    frames = 0

    print_access_banner(args.port)
    print(f"Frame source: {args.source}.  Ctrl-C to stop.\n")

    try:
        while source.ok():
            loop_start = time.time()

            frame = source.read()
            if frame is None:
                time.sleep(0.01)
                continue

            if gui_swap_rb.value:
                frame = frame[:, :, ::-1].copy()

            if gui_rotate.value:
                frame = cv2.rotate(frame, cv2.ROTATE_180)

            focus = focus_score(frame)
            best_focus[0] = max(best_focus[0], focus)

            # The model always sees the undistorted (equirectangular) image, even
            # when the browser is showing the raw fisheye: straight lines stay
            # straight there and people keep their usual proportions, which is
            # what YOLO was trained on. It also means /detections keeps flowing
            # no matter which view the GUI happens to be set to.
            showing_equirect = gui_view.value == "equirectangular"
            detecting = worker is not None and gui_detect.value
            submit_due = detecting and (
                time.time() - last_submit >= 1.0 / gui_detect_fps_target.value
            )

            equirect = None
            if showing_equirect or submit_due:
                h, w = frame.shape[:2]
                if model_shape != (w, h):
                    fisheye_model = load_camera_model(args.camera_params, w, h)
                    model_shape = (w, h)
                # Displayed at the requested scale; when it is only for the
                # detector, build it small right away rather than warping big
                # and shrinking afterwards.
                out_w = int(w * gui_scale.value) // 2 * 2 if showing_equirect else args.detect_width
                equirect = fisheye_to_equirectangular(
                    frame,
                    fisheye_model,
                    out_w,
                    equirect_height(out_w, gui_fov.value),
                    h_fov_deg=gui_fov.value,
                )

            if showing_equirect:
                frame = equirect
            elif gui_scale.value < 1.0:
                frame = cv2.resize(
                    frame,
                    None,
                    fx=gui_scale.value,
                    fy=gui_scale.value,
                    interpolation=cv2.INTER_AREA,
                )

            detections = []
            if detecting:
                detector.score_threshold = float(gui_conf.value)
                red_cap.enabled = bool(gui_rc_enable.value)
                red_cap.hue_width = int(gui_rc_hue.value)
                red_cap.min_saturation = int(gui_rc_sat.value)
                red_cap.min_value = int(gui_rc_val.value)
                red_cap.min_area = int(gui_rc_area.value)
                detector.want_masks = bool(gui_masks.value)

                # Hand the model a frame no wider than --detect-width; masks and
                # boxes get scaled back up for display.
                if submit_due:
                    detect_frame = equirect
                    if detect_frame.shape[1] > args.detect_width:
                        ratio = args.detect_width / detect_frame.shape[1]
                        detect_frame = cv2.resize(
                            detect_frame, None, fx=ratio, fy=ratio, interpolation=cv2.INTER_AREA
                        )
                    worker.submit(detect_frame)
                    last_submit = time.time()

                # Boxes and masks live in equirectangular pixels, so they can
                # only be drawn on that view -- the fisheye frame is a different
                # geometry. The counts and /detections stay live either way.
                if showing_equirect:
                    detections = worker.latest(frame.shape[:2])
                    if detections and (gui_boxes.value or gui_masks.value):
                        frame = draw_detections(
                            frame,
                            detections,
                            draw_boxes=gui_boxes.value,
                            draw_masks=gui_masks.value,
                            mask_alpha=float(gui_mask_alpha.value),
                        )
                else:
                    detections = worker.raw()

            quality = int(gui_quality.value)
            push_background(
                server,
                frame,
                quality,
                gui_lock_aspect.value,
                pacer if gui_pace.value else None,
            )
            # The GUI panel scales images to fit on its own, so it gets the
            # unpadded frame -- bars there would just waste sidebar space.
            gui_preview.image = frame

            frames += 1
            now = time.time()
            if now - last_report >= 1.0:
                gui_fps.value = f"{frames / (now - last_report):.1f}"
                gui_res.value = f"{frame.shape[1]}x{frame.shape[0]}"
                gui_clients.value = str(len(server.get_clients()))
                gui_dropped.value = str(pacer.dropped)
                gui_focus.value = f"{focus:.1f}"
                gui_focus_best.value = f"{best_focus[0]:.1f}"
                gui_focus_verdict.value = focus_verdict(focus)
                if worker is not None:
                    if gui_detect.value:
                        counts = {}
                        for det in detections:
                            counts[det.class_name] = counts.get(det.class_name, 0) + 1
                        gui_objects.value = (
                            ", ".join(f"{n}x {name}" for name, n in counts.items()) or "none"
                        )
                        gui_detect_stats.value = (
                            f"{worker.fps:.1f} fps, {worker.inference_ms:.0f} ms/frame"
                        )
                        hint = "" if showing_equirect else " - switch View to see overlays"
                        gui_detect_input.value = (
                            f"equirectangular ({int(gui_fov.value)} deg){hint}"
                        )
                    else:
                        gui_objects.value = "paused"
                    if publisher is not None:
                        gui_publish.value = publisher.status()
                frames = 0
                last_report = now

            budget = 1.0 / gui_fps_target.value
            elapsed = time.time() - loop_start
            if elapsed < budget:
                time.sleep(budget - elapsed)
    except KeyboardInterrupt:
        print("\nStopping.")
    finally:
        if worker is not None:
            worker.close()
        if detector is not None:
            detector.close()
        source.close()


if __name__ == "__main__":
    main()

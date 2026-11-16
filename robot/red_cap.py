"""Red pen cap detector: plain HSV color thresholding, no neural net.

YOLO only knows the 80 COCO classes, and a pen cap is not one of them. A
small, saturated red object is easy to find by color instead, so this runs
next to YOLO on the same undistorted (equirectangular) frame and adds its
hits to the same detection list as class ``red_cap`` (id 80, the 81st line of
coco.txt). From there everything downstream (the /detections publisher, the
viewer overlay, follow_me.py) treats it like any other class.

Tune it live in the viewer's "Red cap" folder: hold the cap in front of the
camera and raise "Min saturation" / "Min value" until only the cap lights
up. Typical false positives are skin (low saturation) and dark red things
(low value).

Run it by hand on a still image to check thresholds:

    python3 robot/red_cap.py some_frame.jpg      # writes some_frame_redcap.jpg
"""

import cv2
import numpy as np

from yolo_seg import Detection

RED_CAP_CLASS_ID = 80
RED_CAP_NAME = "red_cap"


class RedCapDetector:
    """Finds red blobs in an RGB frame. Settings are plain attributes so the GUI can change them live."""

    def __init__(self):
        self.enabled = True
        # OpenCV hue runs 0..179, and red wraps around 0: accept
        # [0, hue_width] and [180 - hue_width, 179].
        self.hue_width = 10
        self.min_saturation = 140   # 0..255; skin and wood sit below ~120
        self.min_value = 110        # 0..255; rejects dark reds (maroon) and shadows
        self.min_area = 80          # px of area; rejects distant red clothing (a hoodie across the room is ~20-40)
        self.max_blobs = 3          # report at most this many, biggest first
        self.last_mask = None       # the thresholded mask, for debugging

    def threshold(self, frame_rgb):
        """Boolean mask of 'red enough' pixels."""
        hsv = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2HSV)
        s_lo, v_lo, hw = int(self.min_saturation), int(self.min_value), int(self.hue_width)
        low = cv2.inRange(hsv, (0, s_lo, v_lo), (hw, 255, 255))
        high = cv2.inRange(hsv, (180 - hw, s_lo, v_lo), (179, 255, 255))
        mask = cv2.bitwise_or(low, high)
        # Open removes speckle; close fills the glare stripe a glossy cap often has.
        kernel = np.ones((3, 3), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
        return mask

    def detect(self, frame_rgb):
        """List of yolo_seg.Detection for the red blobs in frame_rgb (frame pixels)."""
        if not self.enabled or frame_rgb is None:
            return []
        mask = self.threshold(frame_rgb)
        self.last_mask = mask
        n, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
        blobs = []
        for i in range(1, n):  # 0 is the background
            x, y, w, h, area = stats[i]
            if area < self.min_area:
                continue
            blobs.append((area, i, x, y, w, h))
        blobs.sort(reverse=True)

        detections = []
        for area, i, x, y, w, h in blobs[: self.max_blobs]:
            # Score: how solidly the blob fills its box, so a compact cap
            # beats a stringy red reflection. Floor it so it always clears the
            # viewer's YOLO confidence slider.
            fill = area / float(max(w * h, 1))
            score = float(min(0.99, 0.5 + 0.5 * fill))
            detections.append(
                Detection(
                    RED_CAP_CLASS_ID,
                    RED_CAP_NAME,
                    score,
                    (float(x), float(y), float(x + w), float(y + h)),
                    labels == i,
                )
            )
        return detections


def main():
    import sys

    if len(sys.argv) != 2:
        print(__doc__)
        sys.exit(1)
    path = sys.argv[1]
    bgr = cv2.imread(path)
    if bgr is None:
        sys.exit(f"could not read {path}")
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    det = RedCapDetector()
    found = det.detect(rgb)
    for d in found:
        print(d)
    out = bgr.copy()
    for d in found:
        x1, y1, x2, y2 = (int(v) for v in d.box)
        cv2.rectangle(out, (x1, y1), (x2, y2), (0, 255, 0), 2)
    stem = path.rsplit(".", 1)[0]
    cv2.imwrite(f"{stem}_redcap.jpg", out)
    cv2.imwrite(f"{stem}_redmask.png", det.last_mask)
    print(f"{len(found)} blob(s); wrote {stem}_redcap.jpg and {stem}_redmask.png")


if __name__ == "__main__":
    main()

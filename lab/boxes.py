"""Reading a detection: which class it is, where it is, how near it is.

Nothing in here talks to ROS: detections in, numbers out, so
``python3 lab/test.py`` checks it at your desk.

What a detection looks like
---------------------------
/detections is a vision_msgs/Detection2DArray; msg.detections is a list of
vision_msgs/Detection2D, one per object YOLO found (the desk tests pass a
FakeDetection with the same fields). The fields this lab uses:

    detection.bbox.center.position.x   box centre x, px: 0 at the left edge, grows to the right
    detection.bbox.center.position.y   box centre y, px: 0 at the TOP edge, grows DOWNWARD
    detection.bbox.size_x              box width, px
    detection.bbox.size_y              box height, px
    detection.results[0].hypothesis.class_id   COCO class id as a string ("0" = person; robot/coco.txt)
    detection.results[0].hypothesis.score      confidence, 0..1
    detection.id                       mean mask color "r,g,b", or "" (Optional Step 5)

The box gives you its centre and size, not its corners: the edges are
centre -/+ size / 2.

Detections arrive in a fixed reference frame -- the rectified
(equirectangular) image, rescaled to IMAGE_WIDTH x IMAGE_HEIGHT by the
detector -- so pixel thresholds mean the same thing whatever resolution the
camera is running.
"""

import numpy as np

# The fixed frame /detections is published in (see robot/viser_camera.py).
IMAGE_WIDTH = 700
IMAGE_HEIGHT = 572


def detection_class_id(detection):
    """COCO class id of a detection as an int (0 = person), or None when it cannot be read."""
    if not detection.results:
        return None
    try:
        return int(detection.results[0].hypothesis.class_id)
    except (TypeError, ValueError):
        return None


def detection_color(detection):
    """Mean mask color of a detection as a numpy array (r, g, b), or None.

    robot/viser_camera.py stamps it into detection.id as "r,g,b"; an empty id
    means no color was available for this box.
    """
    if not detection.id:
        return None
    try:
        parts = [float(p) for p in detection.id.split(',')]
    except ValueError:
        return None
    return np.array(parts) if len(parts) == 3 else None


def box_offset(detection):
    """How far off-centre a detection is: -0.5 hard left, +0.5 hard right, 0 centred.

    Step 2 (follow) ------------------------------------------------------------
    detection:  one Detection2D (see "What a detection looks like" above).
    Returns:    float, a fraction of the image width (not pixels), from -0.5
                (left edge) through 0.0 (centred) to +0.5 (right edge).
                Centre x = 350 of 700 -> 0.0; centre x = 690 -> 0.486
    Used by:    follow_me.py, to pick the most centred target and to steer on.
                Mind the signs there: a positive offset (target on the right)
                needs a negative yaw (a turn to the right).
    Test:       python3 lab/test.py step2
    """
    # ========== YOUR CODE HERE: Step 2: box_offset ==========
    # Normalize the box's centre x by IMAGE_WIDTH so that the middle of
    # the image comes out as 0.
    # raise NotImplementedError
    return (detection.bbox.center.position.x - IMAGE_WIDTH/2) / IMAGE_WIDTH
    # ========================================================


def box_bottom(detection):
    """y of the bottom edge of a detection's box: the robot's distance sensor.

    The camera has no depth, but everything stands on the same floor, and the
    floor recedes *upward* in the image: the further away something is, the
    higher its point of contact with the ground appears. So the lower a box's
    bottom edge sits in the image, the closer the object is. (It lies for
    anything not standing on the floor, and anything cut off by the frame.)

    Step 4 (polite distance) ---------------------------------------------------
    detection:  one Detection2D.
    Returns:    float, px: 0 at the top of the image, IMAGE_HEIGHT (572) at
                the bottom. Bigger = closer. A box from y=200 to y=500
                (centre 350, height 300) -> 500.0
    Used by:    target_lock.approach_speed (the stop distance), and
                avoidance.py in Optional Step 6.
    Test:       python3 lab/test.py step4
    """
    # ========== YOUR CODE HERE: Step 4: box_bottom ==========
    # The box gives you its centre y and its height. Return the y of its
    # bottom edge (remember: y grows downward).
    # return detection.bbox.center.position.y +  detection.bbox.size_y / 2
    raise NotImplementedError
    # ========================================================

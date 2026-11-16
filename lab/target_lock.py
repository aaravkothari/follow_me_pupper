"""Follow *me*: which person is mine, and how close to stand. Pure functions, no ROS.

Detections in, decisions out, so ``python3 lab/test.py`` checks all of it at
your desk.

Three jobs, one per step
------------------------
Step 3. **Which person is mine?** To YOLO every person is just class
   ``person``, so "most centred" hops between people the moment two are in
   frame. But detections arrive about ten times a second, and between two
   frames (~0.1 s) a person barely moves in the image -- the same small-motion
   assumption optical flow is built on. So the person nearest to where the
   target just was is almost certainly the target. Nobody near that spot
   means the target is gone, not that a bystander should take over.

Step 4. **How close?** The bottom edge of the box is a distance sensor
   (boxes.box_bottom). Pick the bottom edge you want -- STOP_BOTTOM, measured
   with a person standing at a comfortable distance -- and drive the error to
   zero: walk in when they are far, stop when they are there, step back when
   they come too close.

Optional Step 5. **Which person is mine, even after losing them?** Position
   alone forgets the target the moment the robot has to search. Each
   detection also carries the mean color of its segmentation mask
   (``detection.id``, ``"r,g,b"``) -- mostly the color of their clothes. A
   person whose color is far from the remembered one is a stranger, wherever
   they stand; and when the target vanished and reappears somewhere else, the
   color finds them again. Switch it on with USE_COLOR.
"""

import numpy as np

from boxes import IMAGE_HEIGHT, box_offset, detection_color

# --- Step 3: keeping the lock -----------------------------------------------
# How far (in the units of position_distance: fractions of the image) a
# person may be from where the target was last frame and still count as
# "nearby". Generous enough for the robot's own turning, tight enough that a
# person across the room is not a candidate.
LOCK_GATE = 0.12

# --- Step 4: keeping the distance -------------------------------------------
# Box bottom edge (reference-frame px) with a person standing at the
# distance we want to keep. Measure it on the robot: stand where you would
# like Pupper to stop, save the viewer's "Live feed" image, and scale the
# y of the box's bottom edge to the 572 px frame:
#     STOP_BOTTOM = y_bottom / screenshot_height * IMAGE_HEIGHT
# ========== YOUR CODE HERE: Step 4: measure the stop distance ==========
STOP_BOTTOM = 372  # px -- measure it (see above)
# =======================================================================
# Closer to STOP_BOTTOM than this and we stand still: the box jitters by a
# few pixels every frame, and a robot that shuffles back and forth to chase
# that jitter looks nervous.
STOP_TOLERANCE = 15.0
# How far short of STOP_BOTTOM (px) the robot starts slowing down. Further
# than this it walks at full speed (follow_me.py's TRACK_FORWARD_VEL); closer,
# its speed falls in proportion to the error, reaching 0 at STOP_BOTTOM. So
# raising TRACK_FORWARD_VEL makes the whole approach faster, not just the
# far end of it. If a fast robot overshoots and bumps into you, raise this.
SLOWDOWN_PX = 120.0
# Fastest the robot backs away from someone too close (m/s). Slower than
# walking forward: it cannot see where it is backing into.
MAX_BACKUP_VEL = 0.2

# --- Red pen cap: distance from apparent size -------------------------------
# The cap's longer box side (reference-frame px) at the distance we want to
# keep. Measure it: hold the cap where Pupper should stop and read "size" in
# follow_me.py's 🎯 log lines. Roughly 45 px at ~20 cm, 10 px at ~1 m.
CAP_STOP_SIZE = 45.0
CAP_SIZE_TOLERANCE = 5.0     # px: closer than this to CAP_STOP_SIZE, stand still
CAP_SLOWDOWN_PX = 25.0       # px short of CAP_STOP_SIZE at which we're at full speed


def cap_approach_speed(size, max_forward):
    """Forward speed (m/s) that holds the red cap at CAP_STOP_SIZE.

    Like approach_speed, but bigger box = closer. Positive walks in, negative
    backs off (capped at MAX_BACKUP_VEL), 0.0 inside CAP_SIZE_TOLERANCE.
    """
    error = CAP_STOP_SIZE - size  # positive = too far (looks too small)
    if abs(error) <= CAP_SIZE_TOLERANCE:
        return 0.0
    speed = max_forward * error / CAP_SLOWDOWN_PX
    return float(min(max_forward, max(-MAX_BACKUP_VEL, speed)))


# --- Optional Step 5: telling people apart by color -------------------------
# Off unless you do Optional Step 5: with it off, the lock is position only.
USE_COLOR = False  # Optional Step 5: True
# Mask colors further apart than this (RGB distance, 0..441) are different
# people. Under the camera's fixed white balance the same shirt stays within
# a few tens of this from frame to frame.
REID_MAX_DIST = 90.0
# Per-frame blend of the remembered color toward the target's current color,
# so the memory follows a slow change in lighting instead of being shaken off
# by it.
REID_BLEND = 0.1


class Lock:
    """What we remember about the target between frames."""

    def __init__(self):
        self.position = None  # (box_offset, centre y / IMAGE_HEIGHT) last frame, or None
        self.color = None     # Optional Step 5: remembered mask color, a numpy array (r, g, b), or None

    def reset(self):
        """A new chase: forget everything."""
        self.position = None
        self.color = None


def box_position(detection):
    """Where a detection is, as (offset, centre y / IMAGE_HEIGHT).

    Both are fractions of the image: offset -0.5..+0.5, centre y 0..1.
    """
    return (box_offset(detection), detection.bbox.center.position.y / IMAGE_HEIGHT)


# ======================================================== Step 3: which person
def position_distance(detection, position):
    """How far a detection is from a remembered position.

    Step 3 (follow me, specifically) -------------------------------------------
    detection:  one Detection2D.
    position:   an (offset, centre y fraction) tuple, as box_position returns
                and lock.position stores.
    Returns:    float >= 0, in fractions of the image (not pixels): 0.0 for
                the detection's own position; 70 px sideways -> 0.1
    Used by:    pick_target, against LOCK_GATE.
    Test:       python3 lab/test.py step3
    """
    # ========== YOUR CODE HERE: Step 3: position_distance ==========
    # The straight-line (Euclidean) distance between box_position(detection)
    # and position. Both are fractions of the image, so compare them directly.
    
    # ===============================================================

    difference = np.square(box_position(detection)[0] - position[0]) + np.square(box_position(detection)[1] - position[1])
    return np.sqrt(difference)
    


def pick_target(targets, lock):
    """Which of this frame's people is *our* target, or None.

    Step 3 (follow me, specifically) -------------------------------------------
    targets:    list of Detection2D, every detection of the tracked class in
                this frame (at least one).
    lock:       a Lock, what we remember about our target:
                  lock.position  (offset, centre y fraction) last frame, or
                                 None (a new chase, or the robot just searched)
                Only read it here: remember updates it afterwards.
    Returns:    one of `targets`, or None when nobody is within LOCK_GATE of
                where the target was (it is gone; a bystander does not take
                over).
    Used by:    follow_me.py's process_detections, in place of "most centred".
    Test:       python3 lab/test.py step3
    """
    if USE_COLOR:
        return pick_target_by_color(targets, lock)  # Optional Step 5
    # ========== YOUR CODE HERE: Step 3: pick_target ==========
    # Two rules, in order:
    #   1. No position remembered (lock.position is None): the most
    #      centred person, as in Step 2.
    #   2. Otherwise: of the people within LOCK_GATE of lock.position,
    #      the nearest one. Nobody that close? None.
    # Tip: min(people, key=lambda d: position_distance(d, lock.position))
    # is the nearest of a non-empty list.
    if lock.position is None:
        filtered_detections = targets
        offset_min = abs(box_offset(filtered_detections[0]))
        offset_idx = 0
        for i in range(len(filtered_detections)):
            detection = filtered_detections[i]
            if offset_min > abs(box_offset(detection)):
                offset_min = abs(box_offset(detection))
                offset_idx = i
        target = filtered_detections[offset_idx]
        return target
    else:
        filtered_detections = []
        detections = targets
        for i in range(len(detections)):
            d = detections[i]
            if position_distance(d, lock.position) < LOCK_GATE:
                filtered_detections.append(d)
        if len(filtered_detections) > 0:
            
            target = min(filtered_detections, key=lambda d: position_distance(d, lock.position))
            
            # self.target_pos = box_offset(target)
            # self.last_detection_time = self.get_clock().now()
            # self.target_bottom = box_bottom(target)
            return target
        else:
            return None
    
    # =========================================================


def remember(lock, detection):
    """Update the lock after choosing ``detection`` as the target.

    Step 3 (follow me, specifically) -------------------------------------------
    lock:       the Lock to update, in place.
    detection:  the Detection2D pick_target just chose (never None here).
    Returns:    nothing. Afterwards lock.position == box_position(detection).
    Test:       python3 lab/test.py step3
    """
    # ========== YOUR CODE HERE: Step 3: remember ==========
    # lock.position = where the detection is (box_position).
    
    # ======================================================
    if USE_COLOR:
        remember_color(lock, detection)  # Optional Step 5

    lock.position = box_position(detection)


# ======================================================== Step 4: how close
def approach_speed(target_bottom, max_forward):
    """Forward speed (m/s) that holds the target at STOP_BOTTOM.

    Step 4 (polite distance) ---------------------------------------------------
    target_bottom:  float, px: box_bottom of the target. Bigger = closer.
    max_forward:    float, m/s: the top speed (follow_me.py passes
                    TRACK_FORWARD_VEL).
    Returns:        float, m/s: positive walks in, negative backs away, 0.0
                    inside STOP_TOLERANCE. Never faster than max_forward
                    forward or MAX_BACKUP_VEL backward. With max_forward 0.3:
                    at STOP_BOTTOM -> 0.0; 40 px short -> 0.1;
                    200 px short -> 0.3 (capped); bottom 572 -> -0.2
    Used by:        the TRACK state's forward speed in follow_me.py.
    Test:           python3 lab/test.py step4
    """
    # ========== YOUR CODE HERE: Step 4: approach_speed ==========
    # error = STOP_BOTTOM - target_bottom (positive = target is too far:
    # its bottom edge is above where we want it). Inside STOP_TOLERANCE,
    # stand still. Otherwise a proportional command that reaches
    # max_forward at SLOWDOWN_PX of error, max_forward * error / SLOWDOWN_PX,
    # clamped to [-MAX_BACKUP_VEL, max_forward].
    
    # ============================================================

    error = STOP_BOTTOM - target_bottom

    

    if error > STOP_TOLERANCE:
        return  min(max_forward, max_forward * error / SLOWDOWN_PX)
    elif error < -1* STOP_TOLERANCE:
        return  max(-MAX_BACKUP_VEL, MAX_BACKUP_VEL * error / SLOWDOWN_PX)
    else:
        return 0



# ======================================================== Optional Step 5: color
def color_distance(detection, color):
    """How different a detection's mask color is from a remembered one.

    Optional Step 5 (follow me, very specifically) ------------------------------
    detection:  one Detection2D; boxes.detection_color(detection) gives its
                color as an RGB array, or None when it has none.
    color:      the remembered color (lock.color): an RGB array, or None.
    Returns:    float >= 0, the RGB (Euclidean) distance: 0.0 for the same
                shirt. None when either color is missing: that is "no
                evidence", which is not the same thing as "a match".
    Used by:    pick_target_by_color, against REID_MAX_DIST.
    Test:       python3 lab/test.py step5
    """
    # ========== OPTIONAL — YOUR CODE HERE: Step 5: color_distance ==========
    # Return None if either color is missing, else the distance between them
    # (np.linalg.norm of the difference).
    raise NotImplementedError
    # =======================================================================


def pick_target_by_color(targets, lock):
    """pick_target, with color: which of this frame's people is ours, or None.

    Optional Step 5 (follow me, very specifically) ------------------------------
    targets:    list of Detection2D, every detection of the tracked class in
                this frame (at least one).
    lock:       a Lock: lock.position as in Step 3, and
                  lock.color  the target's mask color, an RGB array, or None
    Returns:    one of `targets`, or None when they are all strangers.
                A stranger is a person whose color_distance to lock.color is
                more than REID_MAX_DIST. (A missing color is no evidence:
                that person is not a stranger.)
    Used by:    pick_target, when USE_COLOR is on.
    Test:       python3 lab/test.py step5
    """
    # ========== OPTIONAL — YOUR CODE HERE: Step 5: pick_target_by_color ==========
    # Three rules, in order:
    #   1. Nothing remembered at all (a new chase): the most centred person.
    #   2. Where the target was is known (lock.position): of the people
    #      within LOCK_GATE of it who are not strangers, the nearest one.
    #   3. Otherwise, the person whose color is closest to lock.color, if
    #      it is within REID_MAX_DIST; else None (only strangers).
    raise NotImplementedError
    # =============================================================================


def remember_color(lock, detection):
    """Blend the chosen detection's mask color into the lock's memory.

    Optional Step 5 (follow me, very specifically) ------------------------------
    lock:       the Lock to update, in place.
    detection:  the Detection2D just chosen as the target.
    Returns:    nothing. Afterwards lock.color is the detection's color on
                first sight, then (1 - REID_BLEND) * old + REID_BLEND * new
                after that, so it follows slow lighting changes. A detection
                with no color leaves lock.color as it was.
                red [200, 40, 40], then reddish [185, 55, 50] -> [198.5, 41.5, 41]
    Used by:    remember, when USE_COLOR is on.
    Test:       python3 lab/test.py step5
    """
    # ========== OPTIONAL — YOUR CODE HERE: Step 5: remember_color ==========
    # 1. color = detection_color(detection). If it is None, stop here.
    # 2. First sight (lock.color is None): store it. Otherwise blend:
    #    lock.color = (1 - REID_BLEND) * lock.color + REID_BLEND * color
    raise NotImplementedError
    # =======================================================================

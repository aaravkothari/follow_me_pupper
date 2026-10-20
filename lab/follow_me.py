"""Follow Me Pupper: the tracking state machine.

The Pupper picks a target class out of /detections, walks up to it, and
steers to keep it centred. You build it in steps, and each one adds to the
same machine:

    Step 2  Follow:         IDLE -> SEARCH <-> TRACK, the most centred target,
                            a constant forward speed.
    Step 3  Follow *me*, specifically: stay on the same person
                            (target_lock.pick_target).
    Step 4  At a polite distance: stop short of them (approach_speed).
    Optional Step 5  Follow me, very specifically: tell people apart by
                            color too (all in target_lock.py).
    Optional Step 6  Something in the way: detour around it (AVOID_TURN ->
                            AVOID_PASS -> AVOID_RETURN, provided) -- you decide
                            when one starts (ready_to_avoid).

What you write here are the methods marked YOUR CODE, plus the constants.
The rest (ROS plumbing, the detour, logging) is provided: read it, don't
rewrite it.

Run it (see README.md for the full three-terminal setup):

    python3 lab/follow_me.py                            # waits for /tracking_control
    python3 lab/follow_me.py --ros-args -p target:=person   # start tracking at once
"""

from enum import Enum
import os

import rclpy
from geometry_msgs.msg import Twist
from rclpy.node import Node
from rclpy.signals import SignalHandlerOptions
from std_msgs.msg import String
from vision_msgs.msg import Detection2DArray

import avoidance
import target_lock
from boxes import box_bottom, box_offset, detection_class_id

# --- Step 2: control constants ----------------------------------------------
# TRACK_FORWARD_VEL is how fast Pupper walks. At about 0.5 m/s and above, the
# walking policy switches from a trot to a gallop: that is when it runs.
# ========== YOUR CODE HERE: Step 2: control constants ==========
TIMEOUT = 1.0 #None            # s without a detection before we go searching
SEARCH_YAW_VEL = 1.0 #None     # rad/s rotation speed while searching
TRACK_FORWARD_VEL = 0.3 #None  # m/s forward speed while tracking (Step 4: the top speed)
KP = 5 #None                 # rad/s of yaw per unit of box_offset
# ===============================================================

# --- Optional Step 6: obstacle avoidance ------------------------------------
# Off unless you do Optional Step 6: with it off, the robot walks straight at
# the target, ignoring anything in between.
ENABLE_OBSTACLE_AVOIDANCE = False  # Optional Step 6: True

# The detour (provided): turn away, walk past, turn back. The two turns are
# the same speed for the same time in opposite directions, so the robot comes
# out of it pointing the way it went in -- displaced sideways by the walk,
# which is the whole point. Tune them if your detours are too wide or narrow.
AVOID_YAW_VEL = 1.0      # rad/s during the two turns
AVOID_TURN_TIME = 0.8    # s turning away, and again turning back
AVOID_FORWARD_VEL = 0.3  # m/s while passing the obstacle
AVOID_PASS_TIME = 3.0    # s walking forward alongside the obstacle
AVOID_COOLDOWN = 1.0     # s after a detour before another may start


class State(Enum):
    IDLE = 0          # Stay in place, no tracking
    SEARCH = 1        # Rotate to search for target
    TRACK = 2         # Follow the target
    AVOID_TURN = 3    # Turn away from an object in the way
    AVOID_PASS = 4    # Walk forward past it
    AVOID_RETURN = 5  # Turn back onto the original heading


AVOID_STATES = (State.AVOID_TURN, State.AVOID_PASS, State.AVOID_RETURN)


def load_class_names():
    """COCO class names, in the id order the detectors publish."""
    path = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'robot', 'coco.txt')
    with open(path) as f:
        return [line.strip() for line in f if line.strip()]


class StateMachineNode(Node):
    def __init__(self):
        super().__init__('state_machine_node')

        self.detection_subscription = self.create_subscription(
            Detection2DArray, '/detections', self.detection_callback, 10)
        self.command_publisher = self.create_publisher(Twist, 'cmd_vel', 10)
        # Tracking starts and stops on /tracking_control.
        self.tracking_control_subscription = self.create_subscription(
            String, '/tracking_control', self.tracking_control_callback, 10)
        self.timer = self.create_timer(0.1, self.timer_callback)

        # Start in IDLE mode (no tracking until commanded)
        self.state = State.IDLE
        self.tracking_enabled = False

        # /detections carries every class, so pick the target out of it here.
        self.class_names = load_class_names()
        self.class_name_to_id = {name: i for i, name in enumerate(self.class_names)}
        self.target_class_id = 0  # person
        self.target_object = 'person'

        # What the target looked like last frame (process_detections sets these).
        self.target_pos = 0.0               # its box_offset, -0.5..+0.5
        self.target_bottom = 0.0            # its box_bottom, px (Step 4)
        self.last_detection_time = None     # when we last saw it; None = not yet

        # Step 3: where our target was (Optional Step 5: and what it looks like).
        self.lock = target_lock.Lock()

        # Optional Step 6: the object blocking our way (its box_offset), or
        # None when the path is clear; how much stuff the last frame had in
        # each half of it, (left, right); and the detour's direction and clocks.
        self.obstacle_pos = None
        self.crowding = (0.0, 0.0)
        self.avoid_dir = 1.0
        self.phase_start = self.get_clock().now()
        self.avoid_done_time = None

        # Tracking normally starts when something publishes to /tracking_control
        # -- KarelPupper.begin_tracking(), or the robot-foundation-model lab's
        # commander later in the quarter. Pass a target to skip that and start
        # straight away:
        #
        #     python3 lab/follow_me.py --ros-args -p target:=person
        self.declare_parameter('target', '')
        initial_target = self.get_parameter('target').value.strip()

        self.get_logger().info('State Machine Node initialized in IDLE state.')
        if initial_target:
            self.start_tracking(initial_target)
        else:
            self.get_logger().info(
                'Use begin_tracking(object), or -p target:=<class>, to enable tracking.'
            )

    def start_tracking(self, obj_name):
        """Enter tracking mode for a COCO class, from the parameter or the topic."""
        self.tracking_enabled = True
        self.lock.reset()  # a new chase locks onto a new target
        self.last_detection_time = None
        if obj_name in self.class_name_to_id:
            self.target_object = obj_name
            self.target_class_id = self.class_name_to_id[obj_name]
        else:
            self.get_logger().warning(
                f'⚠️  Unknown object class "{obj_name}" - still chasing {self.target_object}'
            )
        self.get_logger().info(
            f'✅ Tracking enabled for: {self.target_object} '
            f'(class_id={self.target_class_id})'
        )
        self.get_logger().info(f'   State transition: {self.state.name} → SEARCH')
        self.state = State.SEARCH

    def stop_tracking(self):
        """Leave tracking mode and stand still."""
        self.tracking_enabled = False
        self.obstacle_pos = None
        self.lock.reset()
        self.get_logger().info('⏸️  Tracking disabled - returning to IDLE')
        self.get_logger().info(f'   State transition: {self.state.name} → IDLE')
        self.state = State.IDLE
        self.command_publisher.publish(Twist())

    def tracking_control_callback(self, msg):
        """Handle tracking control commands: "start:<class>" or "stop"."""
        command = msg.data
        self.get_logger().info(f'📥 Received tracking control: "{command}"')
        if command.startswith("start:"):
            self.start_tracking(command.split(":", 1)[1])
        elif command == "stop":
            self.stop_tracking()

    # ======================================================== detections in
    def detection_callback(self, msg):
        """One frame of detections (provided). ROS calls it about 10 times a second.

        Your process_detections finds the target; the rest is Optional Step
        6's bookkeeping: how crowded each half of the frame is, and whether
        something blocks the way to the target.
        """
        target = self.process_detections(msg.detections)

        if ENABLE_OBSTACLE_AVOIDANCE:
            others = [d for d in msg.detections
                      if detection_class_id(d) not in (None, self.target_class_id)
                      and d.results[0].hypothesis.score >= avoidance.OBSTACLE_MIN_SCORE]
            # Measured even when the target is not in this frame: it decides
            # which way a detour goes, so it should describe what we see now.
            self.crowding = avoidance.measure_crowding(others)
            if target is not None:
                self.obstacle_pos = avoidance.find_blocking_obstacle(target, others)
                if self.obstacle_pos is not None:
                    self.get_logger().info(f'🚧 Something in the way at offset {self.obstacle_pos:+.2f}')
        if target is not None and self.tracking_enabled:
            self.get_logger().info(
                f'🎯 Target at {self.target_pos:+.3f} | '
                f'centre x {target.bbox.center.position.x:.1f} | bottom {self.target_bottom:.0f}'
            )

    def process_detections(self, detections):
        """Find our target in this frame, and remember where it is.

        Step 2 (follow), then Steps 3 and 4 ---------------------------------------
        detections:  list of Detection2D, every object YOLO found in this frame,
                     of every class (see "What a detection looks like" in
                     boxes.py). May be empty.
        Returns:     the target's Detection2D, or None when this frame has no
                     target (Step 3: or nobody near where the target was).
        Sets:        when there is a target --
                       self.target_pos           box_offset(target)
                       self.last_detection_time  self.get_clock().now()
                       self.target_bottom        box_bottom(target)  (Step 4)
        Test:        on the robot. Put breakpoint() here and run
                     python3 lab/follow_me.py to see what arrives.
        """
        # ========== YOUR CODE HERE: Steps 2, 3 and 4: process_detections ==========
        # Step 2:
        #   1. targets = the detections whose detection_class_id(d) is
        #      self.target_class_id. None of them? Return None.
        #   2. target = the most centred one (smallest abs(box_offset(d))).
        #   3. Set self.target_pos and self.last_detection_time, and return target.
        # Step 3: replace 2 with
        #      target = target_lock.pick_target(targets, self.lock)
        #   return None if it is None (nobody near where the target was),
        #   else target_lock.remember(self.lock, target).
        # Step 4: in 3, also set self.target_bottom = box_bottom(target).
        filtered_detections = []
        for i in range(len(detections)):
            d = detections[i]
            if detection_class_id(d) == self.target_class_id:
                filtered_detections.append(d)
        print(filtered_detections)
        if len(filtered_detections) > 0:
            offset_min = abs(box_offset(filtered_detections[0]))
            offset_idx = 0
            for i in range(len(filtered_detections)):
                detection = filtered_detections[i]
                if offset_min > abs(box_offset(detection)):
                    offset_min = abs(box_offset(detection))
                    offset_idx = i
            target = filtered_detections[offset_idx]
            self.target_pos = box_offset(target)
            self.last_detection_time = self.get_clock().now()
            self.target_bottom = box_bottom(target)
            return target
        else:
            return None
        # raise NotImplementedError
        # ==========================================================================

    # ======================================================== motion out
    def timer_callback(self):
        """One tick of the state machine (provided). A ROS timer calls it every 0.1 s.

        Your choose_state picks SEARCH or TRACK, and your motion_command says
        how to move in it. A detour, once started (Optional Step 6), runs on
        its own clock until it hands back to TRACK.
        """
        if not self.tracking_enabled:
            # Not tracking: stay idle and DON'T publish, so Karel commands
            # can still drive the robot.
            self.state = State.IDLE
            return
        now = self.get_clock().now()

        if self.state in AVOID_STATES:
            yaw, forward = self.detour_command(now)
        else:
            if self.last_detection_time is None:
                seconds = float('inf')
            else:
                seconds = (now - self.last_detection_time).nanoseconds / 1e9
            self.state = self.choose_state(seconds)
            if self.state == State.SEARCH:
                # The robot is about to spin: where the target was means
                # nothing now. What it looks like still does (Optional Step 5).
                self.lock.position = None
            if self.state == State.TRACK and ENABLE_OBSTACLE_AVOIDANCE and self.ready_to_avoid(now):
                self.start_avoiding(now)
                yaw, forward = self.detour_command(now)
            else:
                yaw, forward = self.motion_command()

        cmd = Twist()
        cmd.angular.z = float(yaw)
        cmd.linear.x = float(forward)
        self.command_publisher.publish(cmd)
        self.get_logger().info(
            f'🤖 State: {self.state.name} | Yaw: {yaw:.2f} | Fwd: {forward:.2f} | Target: {self.target_pos:+.3f}'
        )

    def choose_state(self, seconds_since_detection):
        """SEARCH or TRACK, from how long ago we last saw the target.

        Step 2 (follow) -------------------------------------------------------
        seconds_since_detection:  float, s since process_detections last found
                                  the target; float('inf') if it never has.
        Returns:     State.SEARCH or State.TRACK.
        """
        # ========== YOUR CODE HERE: Step 2: choose_state ==========
        # Seen within the last TIMEOUT seconds: TRACK. Otherwise: SEARCH.
        # raise NotImplementedError
        if seconds_since_detection < TIMEOUT:
            return State.TRACK
        else:
            return State.SEARCH
        # ==========================================================

    def motion_command(self):
        """How to move in the current state: (yaw, forward).

        Step 2 (follow), then Step 4 (polite distance) --------------------------
        Reads:       self.state (State.SEARCH or State.TRACK), self.target_pos,
                     and in Step 4 self.target_bottom.
        Returns:     (yaw, forward), two floats:
                       yaw      rad/s, + turns LEFT (counter-clockwise)
                       forward  m/s, + walks forward
                     Mind the signs: target_pos is + when the target is on the
                     RIGHT, and turning right toward it needs a - yaw.
        """
        # ========== YOUR CODE HERE: Steps 2 and 4: motion_command ==========
        # Step 2:
        #   SEARCH: turn in place at SEARCH_YAW_VEL, toward the side the
        #           target was last seen on: right if self.target_pos > 0,
        #           otherwise left (also when it was never seen: 0.0).
        #   TRACK:  steer against the offset (yaw = -KP * self.target_pos)
        #           and walk at TRACK_FORWARD_VEL.
        # Step 4: in TRACK, walk at
        #   target_lock.approach_speed(self.target_bottom, TRACK_FORWARD_VEL)
        #   instead: full speed when far, slowing to a stop at STOP_BOTTOM,
        #   backing off when too close.

        if self.state == State.SEARCH:
            if self.target_pos > 0:
                return (-SEARCH_YAW_VEL, 0)
            else:
                return (SEARCH_YAW_VEL, 0)
        elif self.state == State.TRACK:
            yaw = -KP * self.target_pos
            return (yaw, TRACK_FORWARD_VEL)
            
        # ===================================================================

    # ======================================================== Optional Step 6: the detour
    def ready_to_avoid(self, now):
        """Should TRACK hand over to a detour right now?

        Optional Step 6 (something in the way) ---------------------------------
        now:         an rclpy Time, self.get_clock().now(). Seconds since the
                     last detour ended: (now - self.avoid_done_time).nanoseconds / 1e9
        Reads:       self.obstacle_pos (find_blocking_obstacle's answer),
                     self.avoid_done_time (None until the first detour ends),
                     self.target_bottom.
        Returns:     bool: True to start a detour now, False to keep tracking.
        """
        # ========== OPTIONAL — YOUR CODE HERE: Step 6: ready_to_avoid ==========
        # True only if all three hold:
        #   1. something blocks (self.obstacle_pos is not None);
        #   2. we are walking toward the target: approach_speed(...) > 0. (At
        #      the stop distance, a detour would walk into the person.)
        #   3. no detour ended in the last AVOID_COOLDOWN seconds. (Right after
        #      one, the obstacle is often still half in frame.)
        raise NotImplementedError
        # =======================================================================

    def start_avoiding(self, now):
        """Commit to a side and begin the detour (provided)."""
        self.avoid_dir, why = avoidance.choose_side(self.crowding, self.obstacle_pos, self.target_pos)
        self.state = State.AVOID_TURN
        self.phase_start = now
        side = 'left' if self.avoid_dir > 0 else 'right'
        self.get_logger().info(f'↪️  Obstacle ahead - going around it on the {side} ({why})')

    def detour_command(self, now):
        """Step the timed turn / pass / turn-back sequence along, and say how to move (provided).

        Detections keep arriving during a detour and updating target_pos, so
        the chase picks straight back up when it ends.
        """
        elapsed = (now - self.phase_start).nanoseconds / 1e9
        if self.state == State.AVOID_TURN and elapsed >= AVOID_TURN_TIME:
            self.state, self.phase_start = State.AVOID_PASS, now
        elif self.state == State.AVOID_PASS and elapsed >= AVOID_PASS_TIME:
            self.state, self.phase_start = State.AVOID_RETURN, now
        elif self.state == State.AVOID_RETURN and elapsed >= AVOID_TURN_TIME:
            self.state = State.TRACK
            self.avoid_done_time = now
            self.obstacle_pos = None
            self.get_logger().info('↩️  Back on the original heading - resuming chase')
            return self.motion_command()

        if self.state == State.AVOID_TURN:
            return AVOID_YAW_VEL * self.avoid_dir, 0.0
        if self.state == State.AVOID_PASS:
            return 0.0, AVOID_FORWARD_VEL
        return -AVOID_YAW_VEL * self.avoid_dir, 0.0


def main():
    # Handle Ctrl+C here rather than in rclpy, so the context is still alive
    # to send the final stand-still command.
    rclpy.init(signal_handler_options=SignalHandlerOptions.NO)
    node = StateMachineNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        print("Program terminated by user")
    finally:
        node.command_publisher.publish(Twist())
        node.destroy_node()
        rclpy.try_shutdown()


if __name__ == '__main__':
    main()

#!/usr/bin/env python3
"""
wall_follower_node.py  (v3 -- adds obstacle bypass + reconnect)

LOGIC OVERVIEW:
----------------
This version adds a 3-state machine on top of the v2 two-ray PD
wall-follower:

  STATE 1: FOLLOWING
      Normal behaviour from v2. Uses two LiDAR rays to estimate the
      wall's angle and keep a smooth PD-controlled distance.
      While in this state, every scan is checked for TWO different
      close-range conditions:

        (a) CORNER:   front is close AND side is ALSO close.
                       -> the perimeter wall itself is turning.
                       -> rotate in place away from the wall, like v1/v2.

        (b) OBSTACLE: front is close BUT side is still roughly
                       at the normal following distance.
                       -> this means the actual wall is still far away,
                          so the thing blocking us is a SEPARATE object
                          sitting in our path, not the wall itself.
                       -> switch to STATE 2 (AVOIDING).

  STATE 2: AVOIDING
      Steer AWAY from the follow side (towards open space / arena
      centre) while still moving forward, to go around the obstacle.
      We stay in this state until the front is clear again AND we've
      travelled a minimum distance (so we don't flip back too early
      and clip the corner of the obstacle).

  STATE 3: RECONNECTING
      Steer back TOWARD the follow side until the side-distance
      measurement returns close to desired_distance again (meaning
      we have rejoined our normal wall-following offset). Then we
      switch back to STATE 1 (FOLLOWING) and resume normal PD control.

This gives the classic "bypass track that reconnects to the original
path" behaviour required by the CEP, using only the single 360-degree
LiDAR already on the robot.
"""

import math

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import LaserScan
from geometry_msgs.msg import Twist


class WallFollower(Node):
    def __init__(self):
        super().__init__('wall_follower_node')

        # ---- Core wall-following parameters (same as v2) ----
        self.declare_parameter('desired_distance', 0.5)
        self.declare_parameter('forward_speed', 0.15)
        self.declare_parameter('kp', 2.0)
        self.declare_parameter('kd', 0.4)
        self.declare_parameter('follow_side', 'right')
        self.declare_parameter('front_stop_distance', 0.6)
        self.declare_parameter('max_angular_speed', 0.8)
        self.declare_parameter('lookahead_distance', 0.4)
        self.declare_parameter('ray_angle_gap_deg', 40.0)

        # ---- NEW: obstacle bypass parameters ----
        self.declare_parameter('obstacle_front_distance', 0.8)   # trigger range
        self.declare_parameter('corner_side_multiplier', 1.5)    # side*this = "still close"
        self.declare_parameter('avoid_turn_speed', 0.5)          # rad/s while avoiding
        self.declare_parameter('avoid_forward_speed', 0.12)      # m/s while avoiding
        self.declare_parameter('min_avoid_distance', 0.6)        # metres travelled before re-checking
        self.declare_parameter('reconnect_tolerance', 0.08)      # metres, "close enough" to resume

        self._load_params()

        self.prev_error = 0.0
        self.last_time = self.get_clock().now()

        # ---- State machine ----
        # One of: 'FOLLOWING', 'AVOIDING', 'RECONNECTING'
        self.state = 'FOLLOWING'
        self.avoid_start_x = None
        self.avoid_start_y = None
        self.last_odom_pose = (0.0, 0.0)  # updated via /odom if available

        self.cmd_pub = self.create_publisher(Twist, '/cmd_vel', 10)
        self.scan_sub = self.create_subscription(
            LaserScan, '/scan', self.on_scan, 10
        )

        self.get_logger().info(
            f'WallFollower v3 (obstacle bypass) started: '
            f'side={self.follow_side}, desired_distance={self.desired_distance} m'
        )

    def _load_params(self):
        self.desired_distance = self.get_parameter('desired_distance').value
        self.forward_speed = self.get_parameter('forward_speed').value
        self.kp = self.get_parameter('kp').value
        self.kd = self.get_parameter('kd').value
        self.follow_side = self.get_parameter('follow_side').value
        self.front_stop_distance = self.get_parameter('front_stop_distance').value
        self.max_angular_speed = self.get_parameter('max_angular_speed').value
        self.lookahead_distance = self.get_parameter('lookahead_distance').value
        self.ray_gap = math.radians(self.get_parameter('ray_angle_gap_deg').value)

        self.obstacle_front_distance = self.get_parameter('obstacle_front_distance').value
        self.corner_side_multiplier = self.get_parameter('corner_side_multiplier').value
        self.avoid_turn_speed = self.get_parameter('avoid_turn_speed').value
        self.avoid_forward_speed = self.get_parameter('avoid_forward_speed').value
        self.min_avoid_distance = self.get_parameter('min_avoid_distance').value
        self.reconnect_tolerance = self.get_parameter('reconnect_tolerance').value

    def get_range_at_angle(self, scan: LaserScan, angle_rad, cone_deg=6):
        if scan.angle_increment == 0.0:
            return float('inf')

        center_index = int(round((angle_rad - scan.angle_min) / scan.angle_increment))
        cone_half_width = int(round(math.radians(cone_deg) / scan.angle_increment))

        lo = max(0, center_index - cone_half_width)
        hi = min(len(scan.ranges) - 1, center_index + cone_half_width)

        valid = [
            r for r in scan.ranges[lo:hi + 1]
            if not math.isinf(r) and not math.isnan(r) and r > 0.0
        ]
        if not valid:
            return float('inf')
        return min(valid)

    def on_scan(self, scan: LaserScan):
        front = self.get_range_at_angle(scan, 0.0, cone_deg=15)

        if self.follow_side == 'right':
            side_angle = -math.pi / 2
            fwd_angle = side_angle + self.ray_gap
            side_sign = -1.0
        else:
            side_angle = math.pi / 2
            fwd_angle = side_angle - self.ray_gap
            side_sign = 1.0

        dist_b = self.get_range_at_angle(scan, side_angle, cone_deg=6)
        dist_a = self.get_range_at_angle(scan, fwd_angle, cone_deg=6)

        cmd = Twist()
        now = self.get_clock().now()
        dt = (now - self.last_time).nanoseconds / 1e9
        if dt <= 0.0:
            dt = 0.1
        self.last_time = now

        # =====================================================
        # STATE 1: FOLLOWING
        # =====================================================
        if self.state == 'FOLLOWING':

            is_front_close = front < self.front_stop_distance
            is_side_also_close = dist_b < (self.desired_distance * self.corner_side_multiplier)

            # --- CORNER: wall itself is turning -> rotate in place ---
            if is_front_close and is_side_also_close:
                cmd.linear.x = 0.0
                cmd.angular.z = -side_sign * self.max_angular_speed
                self.cmd_pub.publish(cmd)
                return

            # --- OBSTACLE: something close ahead, but real wall still far ---
            if front < self.obstacle_front_distance and not is_side_also_close:
                self.get_logger().info('Obstacle detected -> switching to AVOIDING')
                self.state = 'AVOIDING'
                self.avoid_travelled = 0.0
                cmd.linear.x = self.avoid_forward_speed
                # Steer AWAY from the follow side (into open space)
                cmd.angular.z = -side_sign * self.avoid_turn_speed
                self.cmd_pub.publish(cmd)
                return

            # --- Wall lost on following side -> gentle search ---
            if math.isinf(dist_a) or math.isinf(dist_b):
                cmd.linear.x = self.forward_speed * 0.5
                cmd.angular.z = side_sign * 0.3
                self.cmd_pub.publish(cmd)
                return

            # --- Normal two-ray PD wall following (v2 logic) ---
            theta = self.ray_gap
            alpha = math.atan2(
                dist_a * math.cos(theta) - dist_b,
                dist_a * math.sin(theta)
            )
            projected_distance = (
                dist_b * math.cos(alpha) + self.lookahead_distance * math.sin(alpha)
            )

            error = projected_distance - self.desired_distance
            d_error = (error - self.prev_error) / dt
            self.prev_error = error

            steering = side_sign * (self.kp * error + self.kd * d_error)
            steering = max(-self.max_angular_speed, min(self.max_angular_speed, steering))

            cmd.linear.x = self.forward_speed
            cmd.angular.z = steering
            self.cmd_pub.publish(cmd)
            return

        # =====================================================
        # STATE 2: AVOIDING
        # =====================================================
        if self.state == 'AVOIDING':
            # Keep moving forward while steering away from obstacle side.
            # Track rough distance travelled using dt * speed (simple
            # open-loop estimate -- good enough for a short bypass manoeuvre).
            self.avoid_travelled += self.avoid_forward_speed * dt

            cmd.linear.x = self.avoid_forward_speed
            cmd.angular.z = -side_sign * self.avoid_turn_speed

            front_clear = front > self.obstacle_front_distance
            travelled_enough = self.avoid_travelled > self.min_avoid_distance

            if front_clear and travelled_enough:
                self.get_logger().info('Front clear -> switching to RECONNECTING')
                self.state = 'RECONNECTING'

            self.cmd_pub.publish(cmd)
            return

        # =====================================================
        # STATE 3: RECONNECTING
        # =====================================================
        if self.state == 'RECONNECTING':
            # Steer back TOWARD the follow side until side distance
            # returns close to desired_distance (i.e. we've rejoined
            # our normal offset path next to the perimeter wall).
            cmd.linear.x = self.forward_speed
            cmd.angular.z = side_sign * self.avoid_turn_speed * 0.6

            if not math.isinf(dist_b):
                error = abs(dist_b - self.desired_distance)
                if error < self.reconnect_tolerance:
                    self.get_logger().info('Reconnected to wall -> resuming FOLLOWING')
                    self.state = 'FOLLOWING'
                    self.prev_error = 0.0

            self.cmd_pub.publish(cmd)
            return


def main(args=None):
    rclpy.init(args=args)
    node = WallFollower()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.cmd_pub.publish(Twist())
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
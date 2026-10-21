#!/usr/bin/env python3
"""
wall_follower_node.py  (v5 -- clamped error + slew-rate limiting)

WHAT CHANGED FROM v4, AND WHY
------------------------------
After bypassing an obstacle and reconnecting, the robot was far from the
wall (e.g. 0.7 m instead of the normal small drift). The PD controller
saw a LARGE error, multiplied it by kp, and produced a steering command
that saturated at max_angular_speed -- a hard, fast turn back toward the
wall. This caused overshoot past the wall, the error flipped sign, and
the controller swung back the other way -- a zigzag.

TWO FIXES ADDED:
1. max_error: the distance error fed into the PD formula is clamped, so
   the controller never reacts as violently as a raw 0.7 m error would
   cause -- it treats anything beyond max_error as just "far", not
   "extremely far".
2. angular_slew_rate: limits how much angular.z can change per second,
   regardless of what the PD math requests. This physically prevents
   sudden steering jerks, smoothing motion everywhere, not just after
   a bypass.
"""

import math

import rclpy
from rclpy.node import Node
from rclpy.parameter import Parameter
from sensor_msgs.msg import LaserScan
from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry


def wrap(angle):
    """Wrap an angle to [-pi, pi]."""
    return math.atan2(math.sin(angle), math.cos(angle))


class WallFollower(Node):
    def __init__(self):
        # Run on simulation time so all timing matches Gazebo.
        super().__init__(
            'wall_follower_node',
            parameter_overrides=[Parameter('use_sim_time', value=True)],
        )

        # ---- Wall following parameters ----
        self.declare_parameter('desired_distance', 0.5)
        self.declare_parameter('forward_speed', 0.20)
        self.declare_parameter('kp', 2.0)
        self.declare_parameter('kd', 0.4)
        self.declare_parameter('d_filter_alpha', 0.3)
        self.declare_parameter('follow_side', 'right')
        self.declare_parameter('front_stop_distance', 0.6)
        self.declare_parameter('max_angular_speed', 0.8)
        self.declare_parameter('lookahead_distance', 0.4)
        self.declare_parameter('ray_angle_gap_deg', 40.0)

        # ---- NEW: anti-zigzag parameters ----
        self.declare_parameter('max_error', 0.3)
        self.declare_parameter('angular_slew_rate', 1.5)

        # ---- Obstacle bypass parameters ----
        self.declare_parameter('obstacle_front_distance', 0.8)
        self.declare_parameter('corner_side_multiplier', 1.5)
        self.declare_parameter('avoid_turn_angle_deg', 35.0)
        self.declare_parameter('avoid_clear_distance', 1.2)
        self.declare_parameter('avoid_forward_speed', 0.15)
        self.declare_parameter('avoid_turn_speed', 0.5)
        self.declare_parameter('reconnect_turn_speed', 0.5)

        self._load_params()

        # ---- Runtime state ----
        # FOLLOWING -> AVOID_TURN -> AVOID_CLEAR -> RECONNECTING -> FOLLOWING
        self.state = 'FOLLOWING'
        self.prev_error = None
        self.prev_angular = 0.0
        self.d_filt = 0.0
        self.last_t = None
        self.have_odom = False
        self.x = 0.0
        self.y = 0.0
        self.yaw = 0.0
        self.avoid_dir = 1.0          # +1 = turn left, -1 = turn right
        self.avoid_target_yaw = 0.0
        self.start_yaw = 0.0
        self.clear_x = 0.0
        self.clear_y = 0.0

        self.cmd_pub = self.create_publisher(Twist, '/cmd_vel', 10)
        self.create_subscription(LaserScan, '/scan', self.on_scan, 10)
        self.create_subscription(Odometry, '/odom', self.on_odom, 10)

        self.get_logger().info(
            f'WallFollower v5 started: side={self.follow_side}, '
            f'desired_distance={self.desired_distance} m'
        )

    def _load_params(self):
        g = self.get_parameter
        self.desired_distance = g('desired_distance').value
        self.forward_speed = g('forward_speed').value
        self.kp = g('kp').value
        self.kd = g('kd').value
        self.d_filter_alpha = g('d_filter_alpha').value
        self.follow_side = g('follow_side').value
        self.front_stop_distance = g('front_stop_distance').value
        self.max_angular_speed = g('max_angular_speed').value
        self.lookahead_distance = g('lookahead_distance').value
        self.ray_gap = math.radians(g('ray_angle_gap_deg').value)

        self.max_error = g('max_error').value
        self.angular_slew_rate = g('angular_slew_rate').value

        self.obstacle_front_distance = g('obstacle_front_distance').value
        self.corner_side_multiplier = g('corner_side_multiplier').value
        self.avoid_turn_angle = math.radians(g('avoid_turn_angle_deg').value)
        self.avoid_clear_distance = g('avoid_clear_distance').value
        self.avoid_forward_speed = g('avoid_forward_speed').value
        self.avoid_turn_speed = g('avoid_turn_speed').value
        self.reconnect_turn_speed = g('reconnect_turn_speed').value

    # ------------------------------------------------------------------
    # Odometry: store the robot's x, y and heading (yaw).
    # ------------------------------------------------------------------
    def on_odom(self, msg: Odometry):
        p = msg.pose.pose.position
        q = msg.pose.pose.orientation
        self.x = p.x
        self.y = p.y
        self.yaw = math.atan2(
            2.0 * (q.w * q.z + q.x * q.y),
            1.0 - 2.0 * (q.y * q.y + q.z * q.z),
        )
        self.have_odom = True

    # ------------------------------------------------------------------
    # Look up the closest valid LiDAR reading near a given angle.
    # Taking the minimum over a small cone ignores single noisy beams.
    # ------------------------------------------------------------------
    def range_at(self, scan: LaserScan, angle_rad, cone_deg=6):
        if scan.angle_increment == 0.0:
            return float('inf')
        center = int(round((angle_rad - scan.angle_min) / scan.angle_increment))
        half = int(round(math.radians(cone_deg) / scan.angle_increment))
        lo = max(0, center - half)
        hi = min(len(scan.ranges) - 1, center + half)
        valid = [
            r for r in scan.ranges[lo:hi + 1]
            if not math.isinf(r) and not math.isnan(r) and r > 0.0
        ]
        return min(valid) if valid else float('inf')

    # ------------------------------------------------------------------
    # Helper: proportional heading control toward a target yaw.
    # ------------------------------------------------------------------
    def _turn_to(self, target_yaw, max_speed):
        err = wrap(target_yaw - self.yaw)
        return max(-max_speed, min(max_speed, 2.0 * err))

    def _reset_pd(self):
        self.prev_error = None
        self.d_filt = 0.0

    # ------------------------------------------------------------------
    # Two-ray PD wall following, with clamped error and slew-rate
    # limited output (this is what removes the post-bypass zigzag).
    # ------------------------------------------------------------------
    def _pd_steer(self, dist_a, dist_b, side_sign, dt):
        theta = self.ray_gap
        alpha = math.atan2(
            dist_a * math.cos(theta) - dist_b,
            dist_a * math.sin(theta),
        )
        projected = dist_b * math.cos(alpha) + self.lookahead_distance * math.sin(alpha)

        error = projected - self.desired_distance
        # Clamp the error so a very large offset (e.g. right after
        # reconnecting from a bypass) doesn't cause a violent reaction.
        error = max(-self.max_error, min(self.max_error, error))

        raw_d = 0.0 if self.prev_error is None else (error - self.prev_error) / dt
        self.d_filt = self.d_filter_alpha * raw_d + (1.0 - self.d_filter_alpha) * self.d_filt
        self.prev_error = error

        steer = side_sign * (self.kp * error + self.kd * self.d_filt)
        steer = max(-self.max_angular_speed, min(self.max_angular_speed, steer))

        # Slew-rate limit: cap how much angular.z can change this cycle,
        # regardless of what the PD formula above requested.
        max_delta = self.angular_slew_rate * dt
        delta = steer - self.prev_angular
        delta = max(-max_delta, min(max_delta, delta))
        steer = self.prev_angular + delta
        self.prev_angular = steer

        return steer

    # ------------------------------------------------------------------
    # Main loop: runs on every LiDAR scan.
    # ------------------------------------------------------------------
    def on_scan(self, scan: LaserScan):
        # dt from the scan's own timestamp (simulation time).
        t = scan.header.stamp.sec + scan.header.stamp.nanosec * 1e-9
        dt = (t - self.last_t) if self.last_t is not None else 0.1
        self.last_t = t
        dt = min(max(dt, 0.01), 0.5)

        cmd = Twist()

        # Do nothing until odometry is available.
        if not self.have_odom:
            self.cmd_pub.publish(cmd)
            return

        # Choose the side rays based on which wall we follow.
        if self.follow_side == 'right':
            side_angle = -math.pi / 2
            fwd_angle = side_angle + self.ray_gap
            side_sign = -1.0
        else:
            side_angle = math.pi / 2
            fwd_angle = side_angle - self.ray_gap
            side_sign = 1.0

        front = self.range_at(scan, 0.0, cone_deg=15)
        dist_b = self.range_at(scan, side_angle, cone_deg=6)
        dist_a = self.range_at(scan, fwd_angle, cone_deg=6)
        side_close = dist_b < self.desired_distance * self.corner_side_multiplier

        # =========================================================
        # FOLLOWING
        # =========================================================
        if self.state == 'FOLLOWING':
            # CORNER: wall ahead AND wall beside -> rotate in place away from wall.
            if front < self.front_stop_distance and side_close:
                cmd.angular.z = -side_sign * self.max_angular_speed
                self.cmd_pub.publish(cmd)
                return

            # OBSTACLE: something ahead but the wall is still far.
            if front < self.obstacle_front_distance and not side_close:
                self.avoid_dir = -side_sign
                self.start_yaw = self.yaw
                self.avoid_target_yaw = wrap(
                    self.yaw + self.avoid_dir * self.avoid_turn_angle
                )
                self.state = 'AVOID_TURN'
                self._reset_pd()
                self.get_logger().info('Obstacle ahead -> AVOID_TURN')

            # WALL LOST: gentle search turn.
            elif math.isinf(dist_a) or math.isinf(dist_b):
                cmd.linear.x = self.forward_speed * 0.5
                cmd.angular.z = side_sign * 0.3
                self.cmd_pub.publish(cmd)
                return

            # NORMAL: PD wall following.
            else:
                cmd.linear.x = self.forward_speed
                cmd.angular.z = self._pd_steer(dist_a, dist_b, side_sign, dt)
                self.cmd_pub.publish(cmd)
                return

        # =========================================================
        # AVOID_TURN: rotate a fixed angle away from the wall.
        # =========================================================
        if self.state == 'AVOID_TURN':
            cmd.angular.z = self._turn_to(self.avoid_target_yaw, self.avoid_turn_speed)
            if abs(wrap(self.avoid_target_yaw - self.yaw)) < 0.05:
                self.clear_x = self.x
                self.clear_y = self.y
                self.state = 'AVOID_CLEAR'
                self.get_logger().info('Turned away -> AVOID_CLEAR')
            self.cmd_pub.publish(cmd)
            return

        # =========================================================
        # AVOID_CLEAR: drive forward past the obstacle, holding heading.
        # =========================================================
        if self.state == 'AVOID_CLEAR':
            cmd.linear.x = self.avoid_forward_speed
            cmd.angular.z = self._turn_to(self.avoid_target_yaw, 0.5)
            travelled = math.hypot(self.x - self.clear_x, self.y - self.clear_y)
            if travelled >= self.avoid_clear_distance:
                self.state = 'RECONNECTING'
                self.get_logger().info('Passed obstacle -> RECONNECTING')
            self.cmd_pub.publish(cmd)
            return

        # =========================================================
        # RECONNECTING: rotate back to the original heading so the robot
        # is parallel to the wall again. PD then closes the distance.
        # =========================================================
        if self.state == 'RECONNECTING':
            cmd.angular.z = self._turn_to(self.start_yaw, self.reconnect_turn_speed)
            if abs(wrap(self.start_yaw - self.yaw)) < 0.05:
                self.state = 'FOLLOWING'
                self._reset_pd()
                self.prev_angular = 0.0
                self.get_logger().info('Heading restored -> FOLLOWING')
            self.cmd_pub.publish(cmd)
            return

        # Safety fallback: publish a stop command.
        self.cmd_pub.publish(cmd)


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
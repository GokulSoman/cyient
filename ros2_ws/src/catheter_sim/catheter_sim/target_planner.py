#!/usr/bin/python3
"""
ROS2 node: target_planner

Subscribes to /vessel_detection (catheter_sim/VesselDetection) and converts
the pixel-space centroid to a 3D world-frame PoseStamped.

Coordinate transform (simplified fixed-FOV pin-hole model):
    world_x = 0.6 + (center_x / img_w - 0.5) * 0.6   → [0.3, 0.9] m
    world_y = 0.0 + (center_y / img_h - 0.5) * 0.4   → [-0.2, 0.2] m
    world_z = 0.25                                      (above phantom surface)

Publishes:
  /target_pose   (geometry_msgs/PoseStamped)         — 3D target for arm_controller
  /target_marker (visualization_msgs/Marker)         — red sphere for RViz

Clinical note:
  In a real system this transform requires camera intrinsic calibration,
  a tracked ultrasound probe, and per-patient depth estimation.
  Here we assume a fixed overhead camera with known FOV.
"""

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import PoseStamped
from visualization_msgs.msg import Marker
from std_msgs.msg import ColorRGBA


class TargetPlannerNode(Node):

    # Default image dimensions (overridden at runtime from incoming msg)
    _default_img_w = 640
    _default_img_h = 480

    def __init__(self):
        super().__init__('target_planner')

        self.declare_parameter('world_frame', 'world')

        # Publishers
        self.pub_pose = self.create_publisher(PoseStamped, '/target_pose', 10)
        self.pub_marker = self.create_publisher(Marker, '/target_marker', 10)

        # Subscriber (lazy import of custom message)
        self._subscribe()

        self.get_logger().info('TargetPlanner node started')

    def _subscribe(self):
        from catheter_sim.msg import VesselDetection
        self.sub = self.create_subscription(
            VesselDetection, '/vessel_detection', self.detection_callback, 10
        )

    # ------------------------------------------------------------------
    # Transform
    # ------------------------------------------------------------------

    def pixel_to_world(
        self,
        center_x: float,
        center_y: float,
        img_w: int,
        img_h: int,
    ):
        """
        Convert pixel centroid to 3D world coordinates.

        Assumptions
        -----------
        - Camera is mounted top-down above the patient phantom.
        - Phantom centre is at world position [0.6, 0.0, 0.15].
        - Camera FOV covers ±0.3 m in X and ±0.2 m in Y.
        - Fixed depth: needle approaches from z = 0.25 m.
        """
        nx = center_x / img_w - 0.5   # [-0.5, +0.5]
        ny = center_y / img_h - 0.5   # [-0.5, +0.5]

        world_x = 0.6 + nx * 0.6     # [0.3, 0.9] m
        world_y = 0.0 + ny * 0.4     # [-0.2, 0.2] m
        world_z = 0.25               # 10 cm above phantom surface

        return world_x, world_y, world_z

    # ------------------------------------------------------------------
    # Callback
    # ------------------------------------------------------------------

    def detection_callback(self, msg):
        # Read image size from header (or use defaults)
        # The VesselDetection message does not carry image size directly;
        # we use the Image message width/height which are NOT in this msg.
        # Fall back to defaults; full_system pipeline can override via params.
        img_w = self._default_img_w
        img_h = self._default_img_h

        wx, wy, wz = self.pixel_to_world(
            msg.center_x, msg.center_y, img_w, img_h
        )

        frame = self.get_parameter('world_frame').value
        now = self.get_clock().now().to_msg()

        # --- PoseStamped ---
        pose_msg = PoseStamped()
        pose_msg.header.stamp = now
        pose_msg.header.frame_id = frame
        pose_msg.pose.position.x = wx
        pose_msg.pose.position.y = wy
        pose_msg.pose.position.z = wz
        # Orientation: needle pointing downward (rotate 180° about X)
        pose_msg.pose.orientation.x = 1.0
        pose_msg.pose.orientation.y = 0.0
        pose_msg.pose.orientation.z = 0.0
        pose_msg.pose.orientation.w = 0.0
        self.pub_pose.publish(pose_msg)

        # --- Marker (red sphere) ---
        marker = Marker()
        marker.header.stamp = now
        marker.header.frame_id = frame
        marker.ns = 'target'
        marker.id = 0
        marker.type = Marker.SPHERE
        marker.action = Marker.ADD
        marker.pose = pose_msg.pose
        marker.scale.x = 0.02
        marker.scale.y = 0.02
        marker.scale.z = 0.02
        marker.color = ColorRGBA(r=1.0, g=0.0, b=0.0, a=1.0)
        self.pub_marker.publish(marker)

        self.get_logger().debug(
            f'Target: world=({wx:.3f}, {wy:.3f}, {wz:.3f})'
            f' pixel=({msg.center_x:.0f}, {msg.center_y:.0f})'
        )


def main(args=None):
    rclpy.init(args=args)
    node = TargetPlannerNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()

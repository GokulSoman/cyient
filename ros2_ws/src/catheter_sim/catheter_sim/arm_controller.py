#!/usr/bin/python3
"""
ROS2 node: arm_controller

Subscribes to /target_pose (geometry_msgs/PoseStamped) and moves the
catheter arm toward the detected vessel target.

Strategy
--------
1. Debounce: ignore targets that are < 0.02 m from the last accepted target.
2. Try MoveIt2 (MoveGroupInterface) for proper IK-based planning.
3. Fall back to direct JointTrajectory publishing with a simple atan2/clip
   heuristic when MoveIt2 is unavailable.

Publishes /arm_status (std_msgs/String):
    PLANNING / MOVING / MOVING_FALLBACK / REACHED / PLAN_FAILED / EXEC_FAILED / ERROR

Clinical note:
  In a real system fallback heuristic would be replaced with a certified IK
  solver and force/torque safety monitoring.
"""

import numpy as np

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import PoseStamped
from std_msgs.msg import String

# MoveIt2 Python bindings (ROS2 Jazzy)
try:
    from moveit.python_bindings import MoveGroupInterface  # type: ignore
    MOVEIT_AVAILABLE = True
except ImportError:
    MOVEIT_AVAILABLE = False


class ArmControllerNode(Node):

    def __init__(self):
        super().__init__('arm_controller')

        self.declare_parameter('move_group_name', 'arm')
        self.declare_parameter('position_tolerance', 0.02)   # metres
        self.declare_parameter('use_moveit', MOVEIT_AVAILABLE)
        self.declare_parameter('planning_time', 5.0)
        self.declare_parameter('velocity_scale', 0.3)

        self.last_target: np.ndarray | None = None
        self.tolerance = self.get_parameter('position_tolerance').value

        # Initialize MoveGroup interface
        use_moveit = self.get_parameter('use_moveit').value and MOVEIT_AVAILABLE
        if use_moveit:
            group_name = self.get_parameter('move_group_name').value
            try:
                self.move_group = MoveGroupInterface(self, group_name)
                self.get_logger().info(
                    f'MoveIt2 initialized for group: {group_name}'
                )
            except Exception as exc:
                self.get_logger().warn(
                    f'MoveIt2 init failed ({exc}) — using fallback control'
                )
                self.move_group = None
        else:
            self.move_group = None
            self.get_logger().warn(
                'MoveIt2 not available — using fallback joint trajectory control'
            )

        # Subscriber
        self.sub = self.create_subscription(
            PoseStamped, '/target_pose', self.target_callback, 10
        )

        # Status publisher
        self.pub_status = self.create_publisher(String, '/arm_status', 10)

        # Fallback trajectory publisher (created on-demand)
        self.traj_pub = None

        self.get_logger().info('ArmController node started')

    # ------------------------------------------------------------------
    # Target callback
    # ------------------------------------------------------------------

    def target_callback(self, msg: PoseStamped):
        new_pos = np.array([
            msg.pose.position.x,
            msg.pose.position.y,
            msg.pose.position.z,
        ])

        # Debounce
        if self.last_target is not None:
            dist = float(np.linalg.norm(new_pos - self.last_target))
            if dist < self.tolerance:
                return

        self.last_target = new_pos.copy()
        self.get_logger().info(
            f'New target: ({new_pos[0]:.3f}, {new_pos[1]:.3f}, {new_pos[2]:.3f})'
        )
        self._publish_status('PLANNING')

        if self.move_group is not None:
            self._moveit_move(msg)
        else:
            self._fallback_move(msg)

    # ------------------------------------------------------------------
    # MoveIt2 path
    # ------------------------------------------------------------------

    def _moveit_move(self, target_pose: PoseStamped):
        try:
            self.move_group.set_pose_target(target_pose.pose)
            self.move_group.set_planning_time(
                self.get_parameter('planning_time').value
            )
            self.move_group.set_num_planning_attempts(5)
            self.move_group.set_max_velocity_scaling_factor(
                self.get_parameter('velocity_scale').value
            )
            self.move_group.set_max_acceleration_scaling_factor(0.2)

            success, plan, _, _ = self.move_group.plan()

            if not success:
                self.get_logger().warn('MoveIt2 planning failed')
                self._publish_status('PLAN_FAILED')
                return

            self._publish_status('MOVING')
            result = self.move_group.execute(plan, wait=True)

            if result:
                self.get_logger().info('Motion executed successfully')
                self._publish_status('REACHED')
            else:
                self.get_logger().warn('Motion execution failed')
                self._publish_status('EXEC_FAILED')

        except Exception as exc:
            self.get_logger().error(f'MoveIt2 error: {exc}')
            self._publish_status('ERROR')
            # Attempt fallback on error
            self._fallback_move(target_pose)

    # ------------------------------------------------------------------
    # Fallback: direct JointTrajectory
    # ------------------------------------------------------------------

    def _fallback_move(self, target_pose: PoseStamped):
        """
        Naive heuristic: map (x, y) target to shoulder + upper_arm angles.
        This is NOT real IK — it demonstrates the ROS2 control pipeline.
        """
        from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint

        if self.traj_pub is None:
            self.traj_pub = self.create_publisher(
                JointTrajectory,
                '/joint_trajectory_controller/joint_trajectory',
                10,
            )

        x = target_pose.pose.position.x
        y = target_pose.pose.position.y

        # Heuristic: shoulder rotates to face target in XY plane
        shoulder_angle = float(np.arctan2(y, x - 0.0))  # relative to base
        # upper_arm pitches based on distance
        dist_xy = float(np.sqrt(x ** 2 + y ** 2))
        upper_arm_angle = float(np.clip(0.8 - dist_xy * 0.5, -1.0, 1.0))

        traj = JointTrajectory()
        traj.joint_names = [
            'shoulder_joint',
            'upper_arm_joint',
            'forearm_joint',
            'wrist1_joint',
            'wrist2_joint',
        ]

        pt = JointTrajectoryPoint()
        pt.positions = [
            shoulder_angle,
            upper_arm_angle,
            -0.5,   # forearm slightly bent
            0.0,
            0.0,
        ]
        pt.time_from_start.sec = 3
        traj.points = [pt]

        self.traj_pub.publish(traj)
        self._publish_status('MOVING_FALLBACK')
        self.get_logger().info(
            f'Fallback: shoulder={shoulder_angle:.2f} '
            f'upper_arm={upper_arm_angle:.2f}'
        )

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _publish_status(self, status: str):
        msg = String()
        msg.data = status
        self.pub_status.publish(msg)


def main(args=None):
    rclpy.init(args=args)
    node = ArmControllerNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()

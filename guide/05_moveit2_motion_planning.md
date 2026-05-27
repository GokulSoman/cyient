# Guide 05 — MoveIt2 Motion Planning (ROS2 Jazzy)

## Overview

MoveIt2 is used to plan and execute motions of the catheter arm toward the detected vessel target. We use the Python `moveit` bindings available in ROS2 Jazzy.

**Install:**
```bash
sudo apt install -y ros-jazzy-moveit
```

---

## Required Config Files

MoveIt2 needs three config files per robot:

| File | Purpose |
|------|---------|
| `catheter_arm.srdf` | Semantic description: planning groups, end-effectors, self-collision disabling |
| `kinematics.yaml` | IK solver selection and parameters |
| `joint_limits.yaml` | Per-joint velocity/acceleration override |

---

## SRDF: `config/moveit_config/catheter_arm.srdf`

```xml
<?xml version="1.0"?>
<robot name="catheter_arm">

  <!-- Define the planning group for the full arm -->
  <group name="arm">
    <chain base_link="base_link" tip_link="needle_link"/>
  </group>

  <!-- End-effector definition -->
  <end_effector name="needle_ee" parent_link="needle_link" group="arm"/>

  <!-- Virtual joint attaching robot to world frame -->
  <virtual_joint name="virtual_joint" type="fixed"
                 parent_frame="world" child_link="base_link"/>

  <!-- Disable self-collision between adjacent links (always safe) -->
  <disable_collisions link1="base_link"    link2="shoulder_link"  reason="Adjacent"/>
  <disable_collisions link1="shoulder_link" link2="upper_arm_link" reason="Adjacent"/>
  <disable_collisions link1="upper_arm_link" link2="forearm_link"  reason="Adjacent"/>
  <disable_collisions link1="forearm_link"  link2="wrist1_link"   reason="Adjacent"/>
  <disable_collisions link1="wrist1_link"   link2="wrist2_link"   reason="Adjacent"/>
  <disable_collisions link1="wrist2_link"   link2="needle_link"   reason="Adjacent"/>

  <!-- Disable base-link/world collision (static) -->
  <disable_collisions link1="base_link" link2="world" reason="Never"/>

</robot>
```

---

## Kinematics Config: `config/moveit_config/kinematics.yaml`

```yaml
arm:
  kinematics_solver: kdl_kinematics_plugin/KDLKinematicsPlugin
  kinematics_solver_search_resolution: 0.005
  kinematics_solver_timeout: 0.05
  kinematics_solver_attempts: 3
```

> **KDL** (Kinematics and Dynamics Library) is the default solver — works well for simple serial arms.  
> For better performance, consider `bio_ik` or `trac_ik` plugins (requires separate install).

---

## Joint Limits: `config/moveit_config/joint_limits.yaml`

```yaml
joint_limits:
  shoulder_joint:
    has_velocity_limits: true
    max_velocity: 1.0
    has_acceleration_limits: true
    max_acceleration: 2.0
  upper_arm_joint:
    has_velocity_limits: true
    max_velocity: 1.0
    has_acceleration_limits: true
    max_acceleration: 2.0
  forearm_joint:
    has_velocity_limits: true
    max_velocity: 1.5
    has_acceleration_limits: true
    max_acceleration: 3.0
  wrist1_joint:
    has_velocity_limits: true
    max_velocity: 2.0
    has_acceleration_limits: true
    max_acceleration: 4.0
  wrist2_joint:
    has_velocity_limits: true
    max_velocity: 2.0
    has_acceleration_limits: true
    max_acceleration: 4.0
```

---

## Launch: `move_group` Node

Add to `full_system.launch.py`:

```python
from moveit_configs_utils import MoveItConfigsBuilder
from launch_ros.actions import Node

def get_moveit_config():
    moveit_config = (
        MoveItConfigsBuilder("catheter_arm", package_name="catheter_sim")
        .robot_description(file_path="urdf/catheter_arm.urdf.xacro")
        .robot_description_semantic(file_path="config/moveit_config/catheter_arm.srdf")
        .robot_description_kinematics(file_path="config/moveit_config/kinematics.yaml")
        .joint_limits(file_path="config/moveit_config/joint_limits.yaml")
        .to_moveit_configs()
    )
    return moveit_config

# In generate_launch_description():
moveit_config = get_moveit_config()

move_group_node = Node(
    package='moveit_ros_move_group',
    executable='move_group',
    output='screen',
    parameters=[
        moveit_config.to_dict(),
        {'use_sim_time': True},
    ],
)
```

---

## Arm Controller Node: `arm_controller.py`

```python
#!/usr/bin/env python3
"""
ROS2 node: subscribes to /target_pose, plans with MoveIt2 and executes arm motion.
Falls back to direct joint trajectory if MoveIt2 is unavailable.
"""

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import PoseStamped
from std_msgs.msg import String

import numpy as np

# MoveIt2 Python bindings (Jazzy)
try:
    from moveit.python_bindings import MoveGroupInterface
    MOVEIT_AVAILABLE = True
except ImportError:
    MOVEIT_AVAILABLE = False


class ArmControllerNode(Node):

    def __init__(self):
        super().__init__('arm_controller')
        
        self.declare_parameter('move_group_name', 'arm')
        self.declare_parameter('position_tolerance', 0.02)  # 2 cm
        self.declare_parameter('use_moveit', MOVEIT_AVAILABLE)
        
        self.last_target = None
        self.tolerance = self.get_parameter('position_tolerance').value
        
        # Initialize MoveGroup interface
        if self.get_parameter('use_moveit').value and MOVEIT_AVAILABLE:
            group_name = self.get_parameter('move_group_name').value
            self.move_group = MoveGroupInterface(self, group_name)
            self.get_logger().info(f'MoveIt2 initialized for group: {group_name}')
        else:
            self.move_group = None
            self.get_logger().warn('MoveIt2 not available — using fallback control')
        
        # Subscribers
        self.sub = self.create_subscription(
            PoseStamped, '/target_pose', self.target_callback, 10)
        
        # Publishers
        self.pub_status = self.create_publisher(String, '/arm_status', 10)
        
        self.get_logger().info('ArmController node started')

    def target_callback(self, msg: PoseStamped):
        """Called when a new target pose is received."""
        
        new_pos = np.array([
            msg.pose.position.x,
            msg.pose.position.y,
            msg.pose.position.z,
        ])
        
        # Debounce: only move if target changed significantly
        if self.last_target is not None:
            if np.linalg.norm(new_pos - self.last_target) < self.tolerance:
                return
        
        self.last_target = new_pos
        self.get_logger().info(
            f'New target: ({new_pos[0]:.3f}, {new_pos[1]:.3f}, {new_pos[2]:.3f})')
        
        self._publish_status('PLANNING')
        
        if self.move_group is not None:
            self._moveit_move(msg)
        else:
            self._fallback_move(msg)

    def _moveit_move(self, target_pose: PoseStamped):
        """Plan and execute via MoveIt2."""
        try:
            # Set target pose
            self.move_group.set_pose_target(target_pose.pose)
            self.move_group.set_planning_time(5.0)
            self.move_group.set_num_planning_attempts(5)
            self.move_group.set_max_velocity_scaling_factor(0.3)   # 30% of max vel
            self.move_group.set_max_acceleration_scaling_factor(0.2)
            
            # Plan
            success, plan, _, _ = self.move_group.plan()
            
            if not success:
                self.get_logger().warn('MoveIt2 planning failed')
                self._publish_status('PLAN_FAILED')
                return
            
            # Execute
            result = self.move_group.execute(plan, wait=True)
            
            if result:
                self.get_logger().info('Motion executed successfully')
                self._publish_status('REACHED')
            else:
                self.get_logger().warn('Motion execution failed')
                self._publish_status('EXEC_FAILED')
                
        except Exception as e:
            self.get_logger().error(f'MoveIt2 error: {e}')
            self._publish_status('ERROR')

    def _fallback_move(self, target_pose: PoseStamped):
        """
        Fallback: publish a simple JointTrajectory.
        This is a naive home position approach — just demonstrates the pipeline.
        In a real system, this would use an IK solver.
        """
        from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
        
        # Create publisher on-demand
        if not hasattr(self, 'traj_pub'):
            self.traj_pub = self.create_publisher(
                JointTrajectory,
                '/joint_trajectory_controller/joint_trajectory',
                10
            )
        
        msg = JointTrajectory()
        msg.joint_names = [
            'shoulder_joint', 'upper_arm_joint', 'forearm_joint',
            'wrist1_joint', 'wrist2_joint'
        ]
        
        # Simple heuristic: map target x,y → shoulder_joint, upper_arm_joint
        # (This is NOT real IK — just for demo)
        x = target_pose.pose.position.x
        y = target_pose.pose.position.y
        
        shoulder_angle = float(np.arctan2(y, x))
        upper_arm_angle = float(np.clip(0.8 - x * 0.5, -1.0, 1.0))
        
        point = JointTrajectoryPoint()
        point.positions = [shoulder_angle, upper_arm_angle, -0.5, 0.0, 0.0]
        point.time_from_start.sec = 3
        msg.points = [point]
        
        self.traj_pub.publish(msg)
        self._publish_status('MOVING_FALLBACK')
        self.get_logger().info(f'Fallback move: shoulder={shoulder_angle:.2f}, upper_arm={upper_arm_angle:.2f}')

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
```

---

## MoveIt2 Setup Assistant (Alternative)

If you prefer to use the MoveIt2 Setup Assistant to auto-generate config:

```bash
# Launch setup assistant (GUI tool)
ros2 launch moveit_setup_assistant setup_assistant.launch.py

# In GUI:
# 1. Load robot from: urdf/catheter_arm.urdf.xacro
# 2. Generate Self-Collision Matrix
# 3. Add Planning Group "arm" (chain: base_link → needle_link)
# 4. Add End Effector
# 5. Export config to: config/moveit_config/
```

---

## Testing MoveIt2

```bash
# Terminal 1: Start simulation + MoveGroup
ros2 launch catheter_sim full_system.launch.py

# Terminal 2: Manual test via Python
python3 - << 'EOF'
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import PoseStamped
from moveit.python_bindings import MoveGroupInterface

rclpy.init()
node = Node('test_moveit')
mg = MoveGroupInterface(node, 'arm')

target = PoseStamped()
target.header.frame_id = 'world'
target.pose.position.x = 0.5
target.pose.position.y = 0.0
target.pose.position.z = 0.3
target.pose.orientation.w = 1.0

mg.set_pose_target(target.pose)
success, plan, _, _ = mg.plan()
print(f'Planning {"succeeded" if success else "FAILED"}')
if success:
    mg.execute(plan, wait=True)
    print('Done')

rclpy.shutdown()
EOF
```

---

## Fallback: No MoveIt2 (Direct Joint Control)

If MoveIt2 setup is too time-consuming, use the direct joint trajectory approach:
- The fallback `_fallback_move()` in `arm_controller.py` handles this
- Set `use_moveit: false` in the node parameters
- Demonstrates the pipeline even without full motion planning

This is acceptable for the assignment — document it as a known simplification.

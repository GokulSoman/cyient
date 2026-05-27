# Guide 04 — Robot Arm Model (URDF/XACRO)

## Design: Custom 6-DOF Catheterization Arm

We build a **simple custom 6-DOF arm** in XACRO rather than using a real robot model (e.g., UR3e, Panda).

**Why custom:**
- No licensing overhead or complex setup
- Geometry matches the task (vertical needle approach from above)
- Faster to build and iterate

### Kinematic Chain

```
base_link
    └── shoulder_joint (revolute, Z-axis)
            └── shoulder_link
                    └── upper_arm_joint (revolute, Y-axis)
                            └── upper_arm_link
                                    └── forearm_joint (revolute, Y-axis)
                                            └── forearm_link
                                                    └── wrist1_joint (revolute, Y-axis)
                                                            └── wrist1_link
                                                                    └── wrist2_joint (revolute, X-axis)
                                                                            └── wrist2_link
                                                                                    └── needle_joint (fixed)
                                                                                            └── needle_link (end-effector)
```

### Link Dimensions

| Link | Geometry | Dimensions |
|------|----------|-----------|
| base | cylinder | r=0.08m, h=0.05m |
| shoulder | cylinder | r=0.06m, h=0.05m |
| upper_arm | cylinder | r=0.04m, h=0.3m |
| forearm | cylinder | r=0.035m, h=0.25m |
| wrist1 | cylinder | r=0.03m, h=0.05m |
| wrist2 | cylinder | r=0.03m, h=0.05m |
| needle | cylinder | r=0.005m, h=0.08m |

---

## XACRO Template: `catheter_arm.urdf.xacro`

```xml
<?xml version="1.0"?>
<robot name="catheter_arm" xmlns:xacro="http://ros.org/wiki/xacro">

  <!-- ==================== MATERIALS ==================== -->
  <material name="grey">
    <color rgba="0.5 0.5 0.5 1.0"/>
  </material>
  <material name="blue">
    <color rgba="0.0 0.4 0.8 1.0"/>
  </material>
  <material name="silver">
    <color rgba="0.8 0.8 0.9 1.0"/>
  </material>

  <!-- ==================== MACROS ==================== -->
  <!-- Cylinder inertia helper -->
  <xacro:macro name="cylinder_inertia" params="mass radius length">
    <inertial>
      <mass value="${mass}"/>
      <inertia
        ixx="${mass * (3 * radius**2 + length**2) / 12.0}"
        ixy="0" ixz="0"
        iyy="${mass * (3 * radius**2 + length**2) / 12.0}"
        iyz="0"
        izz="${mass * radius**2 / 2.0}"/>
    </inertial>
  </xacro:macro>

  <!-- ==================== BASE LINK ==================== -->
  <link name="world"/>

  <joint name="world_to_base" type="fixed">
    <parent link="world"/>
    <child link="base_link"/>
    <origin xyz="0 0 0" rpy="0 0 0"/>
  </joint>

  <link name="base_link">
    <xacro:cylinder_inertia mass="5.0" radius="0.08" length="0.05"/>
    <visual>
      <origin xyz="0 0 0.025" rpy="0 0 0"/>
      <geometry><cylinder radius="0.08" length="0.05"/></geometry>
      <material name="grey"/>
    </visual>
    <collision>
      <origin xyz="0 0 0.025" rpy="0 0 0"/>
      <geometry><cylinder radius="0.08" length="0.05"/></geometry>
    </collision>
  </link>

  <!-- ==================== SHOULDER ==================== -->
  <joint name="shoulder_joint" type="revolute">
    <parent link="base_link"/>
    <child link="shoulder_link"/>
    <origin xyz="0 0 0.05" rpy="0 0 0"/>
    <axis xyz="0 0 1"/>
    <limit lower="-3.14159" upper="3.14159" effort="100" velocity="1.0"/>
    <dynamics damping="0.5" friction="0.0"/>
  </joint>

  <link name="shoulder_link">
    <xacro:cylinder_inertia mass="1.0" radius="0.06" length="0.05"/>
    <visual>
      <origin xyz="0 0 0.025" rpy="0 0 0"/>
      <geometry><cylinder radius="0.06" length="0.05"/></geometry>
      <material name="blue"/>
    </visual>
    <collision>
      <origin xyz="0 0 0.025" rpy="0 0 0"/>
      <geometry><cylinder radius="0.06" length="0.05"/></geometry>
    </collision>
  </link>

  <!-- ==================== UPPER ARM ==================== -->
  <joint name="upper_arm_joint" type="revolute">
    <parent link="shoulder_link"/>
    <child link="upper_arm_link"/>
    <origin xyz="0 0 0.05" rpy="0 0 0"/>
    <axis xyz="0 1 0"/>
    <limit lower="-1.5708" upper="1.5708" effort="100" velocity="1.0"/>
    <dynamics damping="0.5" friction="0.0"/>
  </joint>

  <link name="upper_arm_link">
    <xacro:cylinder_inertia mass="1.5" radius="0.04" length="0.3"/>
    <visual>
      <origin xyz="0 0 0.15" rpy="0 0 0"/>
      <geometry><cylinder radius="0.04" length="0.3"/></geometry>
      <material name="grey"/>
    </visual>
    <collision>
      <origin xyz="0 0 0.15" rpy="0 0 0"/>
      <geometry><cylinder radius="0.04" length="0.3"/></geometry>
    </collision>
  </link>

  <!-- ==================== FOREARM ==================== -->
  <joint name="forearm_joint" type="revolute">
    <parent link="upper_arm_link"/>
    <child link="forearm_link"/>
    <origin xyz="0 0 0.3" rpy="0 0 0"/>
    <axis xyz="0 1 0"/>
    <limit lower="-3.14159" upper="0.0" effort="100" velocity="1.0"/>
    <dynamics damping="0.5" friction="0.0"/>
  </joint>

  <link name="forearm_link">
    <xacro:cylinder_inertia mass="1.0" radius="0.035" length="0.25"/>
    <visual>
      <origin xyz="0 0 0.125" rpy="0 0 0"/>
      <geometry><cylinder radius="0.035" length="0.25"/></geometry>
      <material name="blue"/>
    </visual>
    <collision>
      <origin xyz="0 0 0.125" rpy="0 0 0"/>
      <geometry><cylinder radius="0.035" length="0.25"/></geometry>
    </collision>
  </link>

  <!-- ==================== WRIST 1 ==================== -->
  <joint name="wrist1_joint" type="revolute">
    <parent link="forearm_link"/>
    <child link="wrist1_link"/>
    <origin xyz="0 0 0.25" rpy="0 0 0"/>
    <axis xyz="0 1 0"/>
    <limit lower="-3.14159" upper="3.14159" effort="50" velocity="2.0"/>
    <dynamics damping="0.1" friction="0.0"/>
  </joint>

  <link name="wrist1_link">
    <xacro:cylinder_inertia mass="0.3" radius="0.03" length="0.05"/>
    <visual>
      <origin xyz="0 0 0.025" rpy="0 0 0"/>
      <geometry><cylinder radius="0.03" length="0.05"/></geometry>
      <material name="grey"/>
    </visual>
    <collision>
      <origin xyz="0 0 0.025" rpy="0 0 0"/>
      <geometry><cylinder radius="0.03" length="0.05"/></geometry>
    </collision>
  </link>

  <!-- ==================== WRIST 2 ==================== -->
  <joint name="wrist2_joint" type="revolute">
    <parent link="wrist1_link"/>
    <child link="wrist2_link"/>
    <origin xyz="0 0 0.05" rpy="0 0 0"/>
    <axis xyz="1 0 0"/>
    <limit lower="-3.14159" upper="3.14159" effort="50" velocity="2.0"/>
    <dynamics damping="0.1" friction="0.0"/>
  </joint>

  <link name="wrist2_link">
    <xacro:cylinder_inertia mass="0.3" radius="0.03" length="0.05"/>
    <visual>
      <origin xyz="0 0 0.025" rpy="0 0 0"/>
      <geometry><cylinder radius="0.03" length="0.05"/></geometry>
      <material name="silver"/>
    </visual>
    <collision>
      <origin xyz="0 0 0.025" rpy="0 0 0"/>
      <geometry><cylinder radius="0.03" length="0.05"/></geometry>
    </collision>
  </link>

  <!-- ==================== NEEDLE END-EFFECTOR ==================== -->
  <joint name="needle_joint" type="fixed">
    <parent link="wrist2_link"/>
    <child link="needle_link"/>
    <origin xyz="0 0 0.05" rpy="0 0 0"/>
  </joint>

  <link name="needle_link">
    <inertial>
      <mass value="0.05"/>
      <inertia ixx="0.0001" ixy="0" ixz="0" iyy="0.0001" iyz="0" izz="0.000001"/>
    </inertial>
    <visual>
      <origin xyz="0 0 0.04" rpy="0 0 0"/>
      <geometry><cylinder radius="0.005" length="0.08"/></geometry>
      <material name="silver"/>
    </visual>
    <collision>
      <origin xyz="0 0 0.04" rpy="0 0 0"/>
      <geometry><cylinder radius="0.005" length="0.08"/></geometry>
    </collision>
  </link>

  <!-- ==================== ROS2 CONTROL ==================== -->
  <ros2_control name="GazeboSystem" type="system">
    <hardware>
      <plugin>gz_ros2_control/GazeboSimSystem</plugin>
    </hardware>
    <joint name="shoulder_joint">
      <command_interface name="position"/>
      <state_interface name="position"/>
      <state_interface name="velocity"/>
    </joint>
    <joint name="upper_arm_joint">
      <command_interface name="position"/>
      <state_interface name="position"/>
      <state_interface name="velocity"/>
    </joint>
    <joint name="forearm_joint">
      <command_interface name="position"/>
      <state_interface name="position"/>
      <state_interface name="velocity"/>
    </joint>
    <joint name="wrist1_joint">
      <command_interface name="position"/>
      <state_interface name="position"/>
      <state_interface name="velocity"/>
    </joint>
    <joint name="wrist2_joint">
      <command_interface name="position"/>
      <state_interface name="position"/>
      <state_interface name="velocity"/>
    </joint>
  </ros2_control>

  <!-- ==================== GZ SIM PLUGIN ==================== -->
  <gazebo>
    <plugin filename="gz_ros2_control-system"
            name="gz_ros2_control::GazeboSimROS2ControlPlugin">
      <parameters>$(find catheter_sim)/config/controllers.yaml</parameters>
      <ros>
        <remapping>~/robot_description:=robot_description</remapping>
      </ros>
    </plugin>
  </gazebo>

</robot>
```

---

## Controllers Config: `config/controllers.yaml`

```yaml
controller_manager:
  ros__parameters:
    update_rate: 100  # Hz

    joint_state_broadcaster:
      type: joint_state_broadcaster/JointStateBroadcaster

    joint_trajectory_controller:
      type: joint_trajectory_controller/JointTrajectoryController

joint_trajectory_controller:
  ros__parameters:
    joints:
      - shoulder_joint
      - upper_arm_joint
      - forearm_joint
      - wrist1_joint
      - wrist2_joint
    command_interfaces:
      - position
    state_interfaces:
      - position
      - velocity
    state_publish_rate: 50.0
    action_monitor_rate: 20.0
    allow_partial_joints_goal: false
    constraints:
      stopped_velocity_tolerance: 0.01
      goal_time: 5.0
```

---

## Gazebo World: `worlds/catheter_world.sdf`

```xml
<?xml version="1.0"?>
<sdf version="1.9">
  <world name="catheter_world">

    <!-- Physics -->
    <plugin filename="gz-sim-physics-system"
            name="gz::sim::systems::Physics"/>
    <plugin filename="gz-sim-scene-broadcaster-system"
            name="gz::sim::systems::SceneBroadcaster"/>
    <plugin filename="gz-sim-user-commands-system"
            name="gz::sim::systems::UserCommands"/>

    <!-- Lighting -->
    <light type="directional" name="sun">
      <cast_shadows>true</cast_shadows>
      <pose>0 0 10 0 0 0</pose>
      <diffuse>0.8 0.8 0.8 1</diffuse>
      <specular>0.2 0.2 0.2 1</specular>
      <direction>-0.5 0.1 -0.9</direction>
    </light>

    <!-- Ground plane -->
    <model name="ground_plane">
      <static>true</static>
      <link name="link">
        <collision name="collision">
          <geometry><plane><normal>0 0 1</normal></plane></geometry>
        </collision>
        <visual name="visual">
          <geometry><plane><normal>0 0 1</normal><size>10 10</size></plane></geometry>
          <material>
            <ambient>0.8 0.8 0.8 1</ambient>
            <diffuse>0.8 0.8 0.8 1</diffuse>
          </material>
        </visual>
      </link>
    </model>

    <!-- Patient phantom (simplified body region) -->
    <model name="patient_phantom">
      <static>true</static>
      <pose>0.6 0 0.075 0 0 0</pose>
      <link name="link">
        <collision name="collision">
          <geometry><box><size>0.6 0.3 0.15</size></geometry>
        </collision>
        <visual name="visual">
          <geometry><box><size>0.6 0.3 0.15</size></geometry>
          <material>
            <ambient>0.9 0.8 0.7 1</ambient>
            <diffuse>0.9 0.8 0.7 1</diffuse>
          </material>
        </visual>
      </link>
    </model>

    <!-- Target marker (small sphere to show planned access point) -->
    <model name="target_marker">
      <static>true</static>
      <pose>0.6 0 0.155 0 0 0</pose>
      <link name="link">
        <visual name="visual">
          <geometry><sphere><radius>0.01</radius></sphere></geometry>
          <material>
            <ambient>1 0 0 1</ambient>
            <diffuse>1 0 0 1</diffuse>
          </material>
        </visual>
      </link>
    </model>

  </world>
</sdf>
```

---

## Verification Commands

```bash
# 1. Check URDF is valid XML
xacro urdf/catheter_arm.urdf.xacro > /tmp/catheter_arm.urdf
check_urdf /tmp/catheter_arm.urdf

# Expected output:
# robot name is: catheter_arm
# ---------- Successfully Parsed XML ---------------
# root Link: world has 1 child(ren)
#     child(1): base_link
# ...

# 2. View in RViz (joints only, no Gazebo)
ros2 launch urdf_launch display.launch.py \
  urdf_package:=catheter_sim \
  urdf_package_path:=urdf/catheter_arm.urdf.xacro

# 3. Launch in Gazebo
ros2 launch catheter_sim sim.launch.py

# 4. Test joint control manually
ros2 topic pub /joint_trajectory_controller/joint_trajectory \
  trajectory_msgs/msg/JointTrajectory \
  '{joint_names: ["shoulder_joint", "upper_arm_joint"], 
    points: [{positions: [0.5, 0.3], time_from_start: {sec: 2}}]}'
```

---

## Clinical Representation Notes

The simplified arm represents an over-bed robotic arm system positioned above the inguinal region. In a real catheterization lab:
- The robot would be positioned at the patient's right groin
- The end-effector would hold an ultrasound probe (for vessel imaging) or a needle guide (for puncture)
- Force/torque sensors would be mandatory on the needle axis
- The arm would have 7+ DOF for flexibility in constrained OR environments

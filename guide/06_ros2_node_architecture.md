# Guide 06 — ROS2 Node Architecture

## Full System Overview

```
┌────────────────────────────────────────────────────────────────────────────┐
│                     CATHETER SIM — ROS2 Node Graph                         │
│                                                                            │
│   ┌──────────────────┐   /image_raw          ┌──────────────────────┐      │
│   │  image_publisher │ ────────────────────► │  vessel_detector     │      │
│   │  (static images  │  sensor_msgs/Image    │  (OpenCV pipeline)   │      │
│   │   from Mus-V)    │                       └──────────┬───────────┘      │
│   └──────────────────┘                                  │                  │
│                                              /vessel_detection             │
│                                              /vessel_image (debug)         │
│                                                          │                  │
│                                                          ▼                  │
│                                               ┌──────────────────────┐     │
│                                               │   target_planner     │     │
│                                               │  (pixel → 3D pose)   │     │
│                                               └──────────┬───────────┘     │
│                                                          │                  │
│                                              /target_pose                  │
│                                              /target_marker                │
│                                                          │                  │
│                                                          ▼                  │
│                                               ┌──────────────────────┐     │
│                                               │   arm_controller     │     │
│                                               │  (MoveIt2 / joint    │     │
│                                               │   trajectory)        │     │
│                                               └──────────┬───────────┘     │
│                                                          │                  │
│                                    ┌─────────────────────┘                 │
│                          /arm_status (std_msgs/String)                     │
│                          /joint_trajectory_controller/joint_trajectory     │
│                                                          │                  │
│   ┌──────────────────────────────────────────────────────┘                 │
│   │                                                                        │
│   ▼                                                                        │
│   ┌────────────────────────────────────────────────────────────────────┐  │
│   │              GZ Sim 8 (catheter_world)                              │  │
│   │  • catheter_arm (6-DOF + needle)                                   │  │
│   │  • patient_phantom (static box)                                    │  │
│   │  • gz_ros2_control plugin                                          │  │
│   └────────────────────────────────────────────────────────────────────┘  │
│                                                                            │
│   ┌──────────────────────────────────────────────────────────────────┐    │
│   │              Supporting Infrastructure                             │    │
│   │  • robot_state_publisher  (URDF + TF tree)                       │    │
│   │  • joint_state_broadcaster (→ /joint_states)                     │    │
│   │  • controller_manager     (manages ros2_control)                 │    │
│   │  • move_group             (MoveIt2 planning server)              │    │
│   │  • ros_gz_bridge          (ROS2 ↔ GZ Sim topic bridge)          │    │
│   │  • rviz2                  (visualization)                        │    │
│   └──────────────────────────────────────────────────────────────────┘    │
└────────────────────────────────────────────────────────────────────────────┘
```

---

## Topic Reference

| Topic | Message Type | Publisher | Subscribers | Description |
|-------|-------------|-----------|-------------|-------------|
| `/image_raw` | `sensor_msgs/Image` | image_publisher | vessel_detector | Mus-V ultrasound images at 1 Hz |
| `/vessel_detection` | `catheter_sim/VesselDetection` | vessel_detector | target_planner | Bounding box + centroid of detected vessel |
| `/vessel_image` | `sensor_msgs/Image` | vessel_detector | RViz | Annotated debug image with detection overlay |
| `/target_pose` | `geometry_msgs/PoseStamped` | target_planner | arm_controller | 3D target pose for arm end-effector |
| `/target_marker` | `visualization_msgs/Marker` | target_planner | RViz | Red sphere marker at target in 3D space |
| `/arm_status` | `std_msgs/String` | arm_controller | — | PLANNING / MOVING / REACHED / FAILED |
| `/joint_states` | `sensor_msgs/JointState` | joint_state_broadcaster | robot_state_publisher, MoveIt2 | Current joint positions/velocities |
| `/tf` | `tf2_msgs/TFMessage` | robot_state_publisher | RViz, arm_controller | Transform tree |
| `/tf_static` | `tf2_msgs/TFMessage` | robot_state_publisher | RViz | Static transforms |
| `/joint_trajectory_controller/joint_trajectory` | `trajectory_msgs/JointTrajectory` | arm_controller | controller_manager | Joint motion commands |

---

## Custom Message: `msg/VesselDetection.msg`

```
# VesselDetection.msg
# Detected vessel region in an ultrasound image frame

std_msgs/Header header    # timestamp + frame_id (camera_frame)

float32 center_x          # centroid pixel x
float32 center_y          # centroid pixel y

float32 bbox_x            # bounding box top-left x (pixels)
float32 bbox_y            # bounding box top-left y (pixels)
float32 bbox_w            # bounding box width (pixels)
float32 bbox_h            # bounding box height (pixels)

float32 confidence        # detection confidence [0.0, 1.0]
string label              # e.g. "femoral_artery_proxy"
```

Build the message by adding to `CMakeLists.txt`:
```cmake
find_package(rosidl_default_generators REQUIRED)
rosidl_generate_interfaces(${PROJECT_NAME}
  "msg/VesselDetection.msg"
  DEPENDENCIES std_msgs
)
```

And in `package.xml`:
```xml
<build_depend>rosidl_default_generators</build_depend>
<exec_depend>rosidl_default_runtime</exec_depend>
<member_of_group>rosidl_interface_packages</member_of_group>
```

---

## Package Structure: `catheter_sim`

```
catheter_sim/
├── catheter_sim/               # Python package (nodes)
│   ├── __init__.py
│   ├── vessel_detector.py      # Node: subscribes /image_raw, publishes /vessel_detection
│   ├── image_publisher.py      # Node: loads Mus-V images, publishes /image_raw
│   ├── target_planner.py       # Node: converts pixel coord → PoseStamped
│   └── arm_controller.py       # Node: MoveIt2 or joint trajectory control
│
├── launch/
│   ├── sim.launch.py           # GZ Sim + robot only
│   └── full_system.launch.py   # Everything: sim + nodes + rviz2
│
├── urdf/
│   └── catheter_arm.urdf.xacro
│
├── config/
│   ├── controllers.yaml            # ros2_control config
│   ├── catheter_sim.rviz           # RViz layout config
│   └── moveit_config/
│       ├── catheter_arm.srdf
│       ├── kinematics.yaml
│       └── joint_limits.yaml
│
├── worlds/
│   └── catheter_world.sdf
│
├── msg/
│   └── VesselDetection.msg
│
├── package.xml
├── setup.py
└── CMakeLists.txt
```

---

## `package.xml`

```xml
<?xml version="1.0"?>
<package format="3">
  <name>catheter_sim</name>
  <version>0.1.0</version>
  <description>Robotic arm simulation for femoral artery catheterization guidance</description>
  <maintainer email="gokul@example.com">Gokul Soman</maintainer>
  <license>MIT</license>

  <!-- Build -->
  <buildtool_depend>ament_cmake</buildtool_depend>
  <buildtool_depend>ament_cmake_python</buildtool_depend>

  <!-- Message generation -->
  <build_depend>rosidl_default_generators</build_depend>
  <exec_depend>rosidl_default_runtime</exec_depend>
  <member_of_group>rosidl_interface_packages</member_of_group>

  <!-- ROS2 runtime -->
  <depend>rclpy</depend>
  <depend>std_msgs</depend>
  <depend>sensor_msgs</depend>
  <depend>geometry_msgs</depend>
  <depend>visualization_msgs</depend>
  <depend>trajectory_msgs</depend>
  <depend>cv_bridge</depend>
  <depend>image_transport</depend>
  <depend>tf2_ros</depend>
  <depend>tf2_geometry_msgs</depend>

  <!-- Simulation -->
  <depend>ros_gz_sim</depend>
  <depend>ros_gz_bridge</depend>

  <!-- Control -->
  <depend>ros2_control</depend>
  <depend>ros2_controllers</depend>
  <depend>gz_ros2_control</depend>

  <!-- Motion planning -->
  <depend>moveit_ros_planning_interface</depend>
  <depend>moveit_ros_move_group</depend>

  <!-- Robot description -->
  <depend>robot_state_publisher</depend>
  <depend>xacro</depend>

  <export>
    <build_type>ament_cmake</build_type>
  </export>
</package>
```

---

## `CMakeLists.txt`

```cmake
cmake_minimum_required(VERSION 3.8)
project(catheter_sim)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

find_package(ament_cmake REQUIRED)
find_package(ament_cmake_python REQUIRED)
find_package(rosidl_default_generators REQUIRED)

# Generate custom message
rosidl_generate_interfaces(${PROJECT_NAME}
  "msg/VesselDetection.msg"
  DEPENDENCIES std_msgs
)

# Install Python package
ament_python_install_package(${PROJECT_NAME})

# Install scripts as executables
install(PROGRAMS
  catheter_sim/vessel_detector.py
  catheter_sim/image_publisher.py
  catheter_sim/target_planner.py
  catheter_sim/arm_controller.py
  DESTINATION lib/${PROJECT_NAME}
)

# Install resource files
install(DIRECTORY
  launch urdf config worlds
  DESTINATION share/${PROJECT_NAME}
)

ament_package()
```

---

## `setup.py`

```python
from setuptools import setup

package_name = 'catheter_sim'

setup(
    name=package_name,
    version='0.1.0',
    packages=[package_name],
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='Gokul Soman',
    maintainer_email='gokul@example.com',
    description='Robotic arm simulation for femoral artery catheterization guidance',
    license='MIT',
    entry_points={
        'console_scripts': [
            'vessel_detector  = catheter_sim.vessel_detector:main',
            'image_publisher  = catheter_sim.image_publisher:main',
            'target_planner   = catheter_sim.target_planner:main',
            'arm_controller   = catheter_sim.arm_controller:main',
        ],
    },
)
```

---

## Launch File: `full_system.launch.py`

```python
#!/usr/bin/env python3
"""Full system launch: Gazebo + robot + perception + planning + control + RViz."""

import os
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import IncludeLaunchDescription, TimerAction
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch_ros.actions import Node
import xacro


def generate_launch_description():
    pkg = get_package_share_directory('catheter_sim')

    # Process XACRO
    xacro_file = os.path.join(pkg, 'urdf', 'catheter_arm.urdf.xacro')
    robot_description = xacro.process_file(xacro_file).toxml()

    # 1. Simulation
    sim_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(pkg, 'launch', 'sim.launch.py')
        )
    )

    # 2. Perception nodes (delay 3s to let sim start)
    image_publisher = TimerAction(period=3.0, actions=[
        Node(package='catheter_sim', executable='image_publisher',
             name='image_publisher', output='screen',
             parameters=[{
                 'image_dir': os.path.join(
                     os.path.expanduser('~'),
                     'github_repos/cyient/data/sample_images/images'),
                 'publish_rate': 1.0,
             }])
    ])

    vessel_detector = TimerAction(period=3.0, actions=[
        Node(package='catheter_sim', executable='vessel_detector',
             name='vessel_detector', output='screen',
             parameters=[{'debug_publish': True}])
    ])

    # 3. Planning node
    target_planner = TimerAction(period=3.5, actions=[
        Node(package='catheter_sim', executable='target_planner',
             name='target_planner', output='screen')
    ])

    # 4. Controller node
    arm_controller = TimerAction(period=5.0, actions=[
        Node(package='catheter_sim', executable='arm_controller',
             name='arm_controller', output='screen',
             parameters=[{
                 'move_group_name': 'arm',
                 'position_tolerance': 0.02,
             }])
    ])

    # 5. RViz
    rviz_config = os.path.join(pkg, 'config', 'catheter_sim.rviz')
    rviz = TimerAction(period=4.0, actions=[
        Node(package='rviz2', executable='rviz2',
             name='rviz2', output='screen',
             arguments=['-d', rviz_config] if os.path.exists(rviz_config) else [])
    ])

    return LaunchDescription([
        sim_launch,
        image_publisher,
        vessel_detector,
        target_planner,
        arm_controller,
        rviz,
    ])
```

---

## Coordinate Transform: Target Planner Logic

```python
# target_planner.py — key transform function

def pixel_to_world(center_x: float, center_y: float,
                   img_w: int, img_h: int) -> tuple[float, float, float]:
    """
    Convert pixel centroid to 3D world coordinates.
    
    Assumption: camera is mounted top-down above the patient phantom.
    Phantom is at world position [0.6, 0.0, 0.15].
    Camera FOV covers ±0.3m in x and ±0.2m in y.
    
    This is a simplified fixed-depth, fixed-FOV model.
    In a real system, this would require:
    - Camera intrinsic calibration (K matrix)
    - Camera-to-robot extrinsic calibration
    - Depth estimation (stereo, structured light, or 3D US)
    """
    # Normalize to [-0.5, 0.5]
    nx = center_x / img_w - 0.5   # -0.5 (left) to +0.5 (right)
    ny = center_y / img_h - 0.5   # -0.5 (top)  to +0.5 (bottom)
    
    # Scale to workspace around phantom center [0.6, 0.0]
    world_x = 0.6 + nx * 0.6   # [0.3, 0.9] m
    world_y = 0.0 + ny * 0.4   # [-0.2, 0.2] m
    world_z = 0.25              # 10 cm above phantom surface (at z=0.15)
    
    return world_x, world_y, world_z
```

---

## RViz Configuration Tips

Open RViz2 and add these displays:
1. **RobotModel** — `Description Topic: /robot_description`
2. **Image** — `Image Topic: /vessel_image` (shows detection overlay)
3. **Marker** — `Marker Topic: /target_marker` (shows target sphere)
4. **TF** — shows coordinate frames
5. **Grid** — for spatial reference

Save as `config/catheter_sim.rviz` after configuring.

---

## Building and Testing

```bash
# Always in this order
conda activate cyient
source /opt/ros/jazzy/setup.bash
cd /home/gokul/github_repos/cyient/ros2_ws

# Build
colcon build --packages-select catheter_sim --symlink-install

# Source
source install/setup.bash

# Run full system
ros2 launch catheter_sim full_system.launch.py

# Monitor topics
ros2 topic hz /image_raw           # should be ~1 Hz
ros2 topic hz /vessel_detection    # should be ~1 Hz
ros2 topic echo /target_pose       # should show 3D coordinates
ros2 topic echo /arm_status        # PLANNING / REACHED
```

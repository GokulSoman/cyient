# Guide 01 — ROS2 Jazzy + GZ Sim 8 Setup

## Environment Summary

| Tool | Version | Status |
|------|---------|--------|
| ROS2 | **Jazzy** | ✅ installed at `/opt/ros/jazzy` |
| Gazebo | **GZ Sim 8.11.0 (Harmonic)** | ✅ installed |
| ros-gz bridge/sim/image | ✅ | installed |
| MoveIt2 | ❌ | needs `sudo apt install ros-jazzy-moveit` |
| ros2_control | ❌ | needs install |
| gz_ros2_control | ❌ | needs install |

> ⚠️ **Important**: This system uses **ROS2 Jazzy** (Ubuntu 24.04 Noble), NOT Humble.  
> Gazebo used is **GZ Sim 8 (Harmonic)** — the command is `gz sim`, not `gazebo`.

---

## Step 1: Create Conda Environment (Do This First)

All Python packages for this project go inside a dedicated conda environment called **`cyient`**. This keeps the project isolated from system Python (3.13) and avoids dependency conflicts.

```bash
# Create environment with Python 3.11 (compatible with ROS2 Jazzy Python bindings)
conda create -n cyient python=3.11 -y

# Activate it — do this in every terminal session before working on this project
conda activate cyient

# Install all Python packages
conda install -c conda-forge -y \
  opencv \
  scikit-learn \
  scipy \
  pandas \
  matplotlib \
  numpy \
  pillow

# PyTorch (CPU build; switch to cu121 if you have CUDA)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu

# Kaggle CLI (for dataset download)
pip install kaggle

# Verify
python -c "import cv2, sklearn, torch, scipy; print('All packages OK')"
```

> **Session workflow in every terminal:**
> ```bash
> conda activate cyient
> source /opt/ros/jazzy/setup.bash
> source /home/gokul/github_repos/cyient/ros2_ws/install/setup.bash
> ```

---

## Step 2: Install Missing ROS2 Packages

```bash
sudo apt update

# MoveIt2 (motion planning)
sudo apt install -y ros-jazzy-moveit

# ros2_control framework
sudo apt install -y \
  ros-jazzy-ros2-control \
  ros-jazzy-ros2-controllers \
  ros-jazzy-controller-manager

# Gazebo-ROS2 control bridge
sudo apt install -y \
  ros-jazzy-gz-ros2-control \
  ros-jazzy-gz-ros2-control-demos

# Additional useful packages
sudo apt install -y \
  ros-jazzy-joint-state-publisher-gui \
  ros-jazzy-rqt-image-view
```

---

## Step 2: Install Python Dependencies

```bash
pip install \
  opencv-python \
  scikit-learn \
  torch torchvision \
  scipy pandas \
  kaggle
```

> **Note on PyTorch**: If CUDA is available, install the CUDA version:
> ```bash
> pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
> ```

---

## Step 3: Source ROS2 in Every Terminal

Add to your `~/.bashrc` to make this automatic:

```bash
echo "source /opt/ros/jazzy/setup.bash" >> ~/.bashrc
source ~/.bashrc
```

Or source manually each session:

```bash
source /opt/ros/jazzy/setup.bash
```

---

## Step 4: Create the ROS2 Workspace

```bash
cd /home/gokul/github_repos/cyient
mkdir -p ros2_ws/src
cd ros2_ws

# Initial build (empty workspace)
colcon build

# Source the workspace
source install/setup.bash
```

Add workspace source to bashrc (after the ROS2 source line):

```bash
echo "source /home/gokul/github_repos/cyient/ros2_ws/install/setup.bash" >> ~/.bashrc
```

---

## Step 5: Verify GZ Sim 8

```bash
# Should open an empty world
gz sim

# Check version
gz sim --version
# Expected: Gazebo Sim, version 8.x.x
```

Key differences from Gazebo Classic:
- Command is `gz sim` (not `gazebo`)
- World format: SDF (same), but plugin syntax changed
- ROS2 bridge: `ros_gz_bridge` package (not `gazebo_ros`)
- Topic bridge format: `ros_topic:=gz_topic@ros_msg_type[gz_msg_type`

---

## Step 6: Verify ROS-GZ Bridge

```bash
# The ros_gz_bridge converts between ROS2 and GZ topics
# Example: bridge /clock from GZ to ROS2
ros2 run ros_gz_bridge parameter_bridge /clock@rosgraph_msgs/msg/Clock[gz.msgs.Clock

# Verify bridge package
ros2 pkg list | grep ros_gz
# Should show: ros_gz_bridge, ros_gz_image, ros_gz_interfaces, ros_gz_sim
```

---

## Step 7: GZ Sim 8 — Key Plugin Names (Jazzy)

When writing SDF world files and URDF/XACRO for GZ Sim 8, use these plugin names:

```xml
<!-- In SDF world file -->
<plugin filename="gz-sim-physics-system"
        name="gz::sim::systems::Physics"/>
<plugin filename="gz-sim-scene-broadcaster-system"
        name="gz::sim::systems::SceneBroadcaster"/>
<plugin filename="gz-sim-sensors-system"
        name="gz::sim::systems::Sensors">
  <render_engine>ogre2</render_engine>
</plugin>
<plugin filename="gz-sim-user-commands-system"
        name="gz::sim::systems::UserCommands"/>

<!-- In URDF/XACRO for ros2_control -->
<plugin filename="gz_ros2_control-system"
        name="gz_ros2_control::GazeboSimROS2ControlPlugin">
  <parameters>$(find catheter_sim)/config/controllers.yaml</parameters>
</plugin>
```

---

## Step 8: Workspace Build Commands

```bash
cd /home/gokul/github_repos/cyient/ros2_ws

# Full build
colcon build

# Build specific package
colcon build --packages-select catheter_sim

# Build with symlinks (faster iteration, Python packages)
colcon build --symlink-install

# After each build, source
source install/setup.bash

# Check for build errors
colcon build 2>&1 | grep -E "error:|warning:" | head -20
```

---

## Common Issues & Fixes

### Issue: `ros2` command not found
```bash
source /opt/ros/jazzy/setup.bash
```

### Issue: `gz sim` opens but no ROS2 topics
```bash
# Make sure ros_gz_bridge is running or started in launch file
ros2 run ros_gz_bridge parameter_bridge /world/my_world/pose/info@tf2_msgs/msg/TFMessage[gz.msgs.Pose_V
```

### Issue: Package not found after `colcon build`
```bash
source /home/gokul/github_repos/cyient/ros2_ws/install/setup.bash
```

### Issue: MoveIt2 `move_group` can't find robot
- Ensure `robot_description` parameter is set (via `robot_state_publisher`)
- Check URDF loads: `ros2 param get /robot_state_publisher robot_description`

### Issue: `xacro` file fails to process
```bash
# Test XACRO processing separately
ros2 run xacro xacro urdf/catheter_arm.urdf.xacro
# OR
xacro urdf/catheter_arm.urdf.xacro > /tmp/robot.urdf
check_urdf /tmp/robot.urdf
```

---

## Useful ROS2 Commands for This Project

```bash
# List all running nodes
ros2 node list

# Monitor topics
ros2 topic list
ros2 topic echo /vessel_detection
ros2 topic hz /image_raw

# Check TF tree
ros2 run tf2_tools view_frames

# RViz
rviz2

# RQT image viewer
ros2 run rqt_image_view rqt_image_view

# GZ Sim topic bridge (example)
ros2 run ros_gz_bridge parameter_bridge \
  /joint_states@sensor_msgs/msg/JointState[gz.msgs.Model
```

---

## Reference Links

- [ROS2 Jazzy Docs](https://docs.ros.org/en/jazzy/)
- [GZ Sim 8 / Harmonic Docs](https://gazebosim.org/docs/harmonic/)
- [ros_gz (ROS2-Gazebo bridge)](https://github.com/gazebosim/ros_gz)
- [gz_ros2_control](https://github.com/ros-controls/gz_ros2_control)
- [MoveIt2 Jazzy](https://moveit.picknik.ai/main/index.html)
- [ROS2 Control Docs](https://control.ros.org/jazzy/)

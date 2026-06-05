# Medical-Industry Cobot Manipulators (ROS descriptions)

Candidate collaborative manipulators for the catheterization simulation, selected for their
adoption in medical / surgical robotics. Each links to its **official ROS 2 description package**
(URDF/xacro + meshes + configs).

The full packages are large (~300 MB, almost entirely `.dae`/`.stl` meshes), so they are **not
vendored in this repo** — everything under `manipulators/` except this file is gitignored.
Clone them from upstream as needed (commands below).

| Cobot | Why it's relevant to medtech | ROS 2 description repo | Key models | License |
|-------|------------------------------|------------------------|------------|---------|
| **KUKA LBR Med / iiwa** | The **LBR Med** is a medically‑certified 7‑DOF arm used as the robotic base in several surgical systems | https://github.com/lbr-stack/lbr_fri_ros2_stack (`lbr_description/`) | `med7`, `med14`, `iiwa7`, `iiwa14` | Apache-2.0 |
| **Universal Robots e‑Series** | Widely used for robotic ultrasound scanning, medical‑device assembly and rehab/assistive setups | https://github.com/UniversalRobots/Universal_Robots_ROS2_Description | `ur3e`, `ur5e`, `ur10e`, `ur16e` (+ more) | BSD-3-Clause |
| **Franka Emika FR3** | Dominant 7‑DOF platform in surgical / medical‑robotics research (robotic US, minimally‑invasive surgery) | https://github.com/frankaemika/franka_description | `fr3`, `fp3`, `fer` | Apache-2.0 |

## Fetch

```bash
cd manipulators

# KUKA LBR (sparse checkout of just the description package)
git clone --depth 1 --filter=blob:none --sparse https://github.com/lbr-stack/lbr_fri_ros2_stack.git
( cd lbr_fri_ros2_stack && git sparse-checkout set lbr_description )

# Universal Robots
git clone --depth 1 https://github.com/UniversalRobots/Universal_Robots_ROS2_Description.git

# Franka Emika
git clone --depth 1 https://github.com/frankaemika/franka_description.git
```

## Flatten a xacro into a plain URDF (examples)

```bash
# Universal Robots UR5e
xacro Universal_Robots_ROS2_Description/urdf/ur.urdf.xacro ur_type:=ur5e name:=ur5e > ur5e.urdf

# KUKA LBR Med7  (medically-certified arm)
xacro lbr_fri_ros2_stack/lbr_description/urdf/med7/med7.urdf.xacro > med7.urdf

# Franka Emika FR3
xacro franka_description/robots/fr3/fr3.urdf.xacro > fr3.urdf
```

## Notes
- All three ship as ROS 2 packages (`package.xml` format 3): `lbr_description`, `ur_description`,
  `franka_description`. Install the matching `ros-<distro>-*-description` via apt, or build from
  source, for full MoveIt 2 / Gazebo support.
- Some xacros require arguments (e.g. UR needs `ur_type`/`name`); check each package's `urdf/` for
  the available macros and args.
- Alternative KUKA iiwa stack: https://github.com/ICube-Robotics/iiwa_ros2

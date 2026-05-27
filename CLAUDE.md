# Cyient Technical Assignment — Healthcare / MedTech Robotics
**Candidate:** Gokul Soman | **Deadline:** 2 days | **Date started:** 2026-05-27

---

## Objectives

### Option 1 (Primary): Robotic Arm Based Femoral Artery Catheterization Simulation
Build a simplified ROS2/GZ Sim simulation where:
1. A vascular ultrasound image is processed to detect a vessel region
2. The detected centroid is converted to a 3D target pose in simulation
3. A robotic arm end-effector moves toward the planned access point

**Success criteria:** Single launch command brings up Gazebo with arm, runs vessel detection, and drives the arm toward the detected target.

### Option 2 (Extension): Exoskeleton Gait Intent Recognition and Assistive Control
Build an ML pipeline that:
1. Loads and preprocesses IMU+EMG wearable sensor data
2. Classifies locomotion mode (walking, stairs, ramps, standing, transitions)
3. Maps predictions to a finite-state-machine (FSM) assistive control logic
4. (Optional) Wraps in a ROS2 node

**Success criteria:** ≥80% test accuracy, confusion matrix saved, FSM transitions demonstrated.

---

## Evaluation Weights

| Area | Option 1 Weight | Option 2 Weight |
|------|----------------|----------------|
| ROS2/Gazebo integration | **25%** | — |
| Perception / ML modeling | **25%** | **25%** |
| Motion planning / Control logic | **20%** | **20%** |
| Healthcare/clinical reasoning | **15%** | **15%** |
| Code quality, README, reproducibility | **15%** | **15%** |
| Data preprocessing / signal handling | — | **25%** |

---

## Confirmed Datasets

| Option | Dataset | Source | Size |
|--------|---------|--------|------|
| 1 | **Mus-V** — Multimodal Ultrasound Vascular Segmentation (MICCAI 2024) | Kaggle: `among22/multimodal-ultrasound-vascular-segmentation` | 3,114 images, ~200 MB |
| 2 | **HDsEMG+IMU** — High-density EMG, IMU, kinetics (Nature Sci Data 2023) | Figshare: `doi:10.6084/m9.figshare.22227337` | 10 subjects, 8+ activities, ~2–5 GB |

---

## Environment

| Tool | Version | Notes |
|------|---------|-------|
| ROS2 | **Jazzy** | Source: `/opt/ros/jazzy/setup.bash` |
| Gazebo | **GZ Sim 8.11.0 (Harmonic)** | Command: `gz sim` |
| Conda env | **cyient** | All Python packages installed here |
| Python | 3.11 (via conda) | System Python is 3.13 — use conda env |
| numpy, matplotlib, Pillow | ✅ (install in conda) | |
| MoveIt2 | ❌ apt install needed | `sudo apt install ros-jazzy-moveit` |
| ros2_control | ❌ apt install needed | `sudo apt install ros-jazzy-ros2-control ros-jazzy-ros2-controllers` |
| gz_ros2_control | ❌ apt install needed | `sudo apt install ros-jazzy-gz-ros2-control` |
| OpenCV (Python) | ❌ conda install needed | `conda install -c conda-forge opencv` |
| scikit-learn | ❌ install needed | `pip install scikit-learn` |
| PyTorch | ❌ install needed | `pip install torch` |

### First-time setup

```bash
# ── Step 1: Create conda environment (ALL Python packages go here) ──────────
conda create -n cyient python=3.11 -y
conda activate cyient

# ── Step 2: Install Python packages inside conda env ────────────────────────
conda install -c conda-forge -y \
  opencv \
  scikit-learn \
  scipy \
  pandas \
  matplotlib \
  numpy \
  pillow

# PyTorch (CPU — change to cu121 if CUDA available)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu

# Kaggle CLI for dataset download
pip install kaggle

# ── Step 3: Install ROS2 apt packages (system-wide, not conda) ──────────────
sudo apt install -y \
  ros-jazzy-moveit \
  ros-jazzy-ros2-control \
  ros-jazzy-ros2-controllers \
  ros-jazzy-gz-ros2-control \
  ros-jazzy-gz-ros2-control-demos

# ── Step 4: Source ROS2 inside conda env sessions ───────────────────────────
# Always run in this order:
#   conda activate cyient
#   source /opt/ros/jazzy/setup.bash
#   source ros2_ws/install/setup.bash  (after first build)

# ── Step 5: Dataset download ─────────────────────────────────────────────────
# Option 1 — Mus-V (configure ~/.kaggle/kaggle.json first)
kaggle datasets download among22/multimodal-ultrasound-vascular-segmentation \
  -p data/sample_images/ --unzip

# Option 2 — HDsEMG+IMU
# Visit https://figshare.com/articles/dataset/22227337 and download manually to data/gait/
```

---

## Repository Structure

```
cyient/
├── CLAUDE.md                          ← this file
├── README.md                          ← submission README
├── Assignment.pdf
├── guide/                             ← reference guides
│   ├── 01_ros2_gazebo_setup.md
│   ├── 02_medical_datasets.md
│   ├── 03_perception_pipeline.md
│   ├── 04_robot_arm_model.md
│   ├── 05_moveit2_motion_planning.md
│   ├── 06_ros2_node_architecture.md
│   └── 07_exoskeleton_extension.md
├── ros2_ws/
│   └── src/
│       ├── catheter_sim/              ← Option 1 ROS2 package
│       └── gait_intent/               ← Option 2 ROS2 package
├── data/
│   ├── sample_images/                 ← Mus-V vascular images
│   └── gait/                          ← HDsEMG+IMU dataset
├── results/
│   ├── screenshots/
│   └── plots/
├── docs/
│   └── architecture_diagram.png
└── docker/
    └── Dockerfile
```

---

## System Architecture (Option 1)

```
┌─────────────────────────────────────────────────────────────────┐
│                        ROS2 Node Graph                          │
│                                                                 │
│  [image_publisher]  ─/image_raw──►  [vessel_detector]          │
│                                           │                     │
│                              /vessel_detection                  │
│                              /vessel_image (debug)              │
│                                           │                     │
│                                           ▼                     │
│                                   [target_planner]              │
│                                           │                     │
│                                   /target_pose                  │
│                                   /target_marker                │
│                                           │                     │
│                                           ▼                     │
│                                   [arm_controller]              │
│                                           │                     │
│                                    MoveIt2 / joint control      │
│                                           │                     │
│                                           ▼                     │
│                              [GZ Sim 8 — catheter_arm]          │
│                           (6-DOF arm + patient phantom)         │
└─────────────────────────────────────────────────────────────────┘
```

### Custom Message: `VesselDetection.msg`
```
std_msgs/Header header
float32 center_x      # pixel centroid x
float32 center_y      # pixel centroid y
float32 bbox_x        # bounding box top-left x
float32 bbox_y        # bounding box top-left y
float32 bbox_w        # bounding box width
float32 bbox_h        # bounding box height
float32 confidence    # detection confidence 0–1
string label          # e.g. "femoral_artery_proxy"
```

### Topic Summary
| Topic | Type | Publisher | Subscriber |
|-------|------|-----------|------------|
| `/image_raw` | `sensor_msgs/Image` | image_publisher | vessel_detector |
| `/vessel_detection` | `catheter_sim/VesselDetection` | vessel_detector | target_planner |
| `/vessel_image` | `sensor_msgs/Image` | vessel_detector | RViz (debug) |
| `/target_pose` | `geometry_msgs/PoseStamped` | target_planner | arm_controller |
| `/target_marker` | `visualization_msgs/Marker` | target_planner | RViz |
| `/arm_status` | `std_msgs/String` | arm_controller | — |

---

## Implementation Plan

### ─── OPTION 1: catheter_sim ───

#### Phase 0: Dataset + Environment Setup
- [ ] Confirm Mus-V download at `data/sample_images/`
- [ ] Install all missing packages (see setup above)
- [ ] Create `ros2_ws/src/catheter_sim/` package skeleton
- [ ] Verify `colcon build` succeeds

**Deliverable:** `source ros2_ws/install/setup.bash && ros2 pkg list | grep catheter` shows package

---

#### Phase 1: Robot URDF + Gazebo World
**File:** `ros2_ws/src/catheter_sim/urdf/catheter_arm.urdf.xacro`

Links: `base_link → shoulder → upper_arm → forearm → wrist1 → wrist2 → needle_link`
- Cylinders for arm segments; box for base; small cylinder for needle
- Inertia tensors calculated from geometry
- `<ros2_control>` hardware interface tag (`gz_ros2_control/GazeboSimSystem`)

**File:** `ros2_ws/src/catheter_sim/worlds/catheter_world.sdf`
- Ground plane, ambient lighting
- Patient phantom: box 0.6×0.3×0.1 m at pose [0.6, 0, 0.15]

**File:** `ros2_ws/src/catheter_sim/launch/sim.launch.py`
- GZ Sim launch with world file
- `robot_state_publisher` with URDF
- `ros_gz_bridge` for joint states / TF
- `spawn_entity` for robot model
- `joint_state_broadcaster` + `joint_trajectory_controller`

- [ ] URDF created and validated (`check_urdf`)
- [ ] World file renders correctly in GZ Sim
- [ ] Launch file brings up arm in simulation
- [ ] `ros2 topic echo /joint_states` shows joint positions

**Deliverable:** Robot visible in Gazebo, joints stream on `/joint_states`

---

#### Phase 2: Perception Node
**File:** `ros2_ws/src/catheter_sim/catheter_sim/vessel_detector.py`

Algorithm:
1. Subscribe `/image_raw` → convert to BGR
2. Grayscale → Gaussian blur (5×5, σ=1.5)
3. Adaptive threshold (blockSize=15, C=4)
4. Morphological close (ellipse kernel 7×7)
5. `findContours` → filter by area (min 200 px²)
6. Largest contour → `boundingRect` + centroid via moments
7. Publish `VesselDetection` + annotated image

**File:** `ros2_ws/src/catheter_sim/catheter_sim/image_publisher.py`
- Loads images from `data/sample_images/`
- Cycles through images, publishes at 1 Hz

- [ ] `VesselDetection.msg` defined and built
- [ ] Detection node runs standalone: `ros2 run catheter_sim vessel_detector`
- [ ] Debug image visible in `rqt_image_view /vessel_image`
- [ ] Screenshot of detection overlay saved to `results/screenshots/`

**Deliverable:** Bounding box + centroid drawn on ultrasound image; screenshot saved

---

#### Phase 3: Target Planner Node
**File:** `ros2_ws/src/catheter_sim/catheter_sim/target_planner.py`

Coordinate transform (simplified pin-hole model):
```python
# Assume 640×480 input; workspace: x∈[0.3, 0.9], y∈[-0.3, 0.3]
target_x = 0.6 + (center_x / img_w - 0.5) * 0.6   # [0.3, 0.9] m
target_y = (center_y / img_h - 0.5) * 0.6           # [-0.3, 0.3] m
target_z = 0.25                                       # above phantom surface
```

- Publish `PoseStamped` in `world` frame, orientation pointing downward (needle axis)
- Publish `visualization_msgs/Marker` (sphere, red) for RViz

- [ ] `ros2 topic echo /target_pose` prints valid 3D coords
- [ ] Marker visible in RViz above phantom
- [ ] Screenshot saved to `results/screenshots/`

**Deliverable:** 3D target pose streams; red sphere marker in RViz

---

#### Phase 4: Arm Controller Node
**File:** `ros2_ws/src/catheter_sim/catheter_sim/arm_controller.py`

```python
from moveit.python_bindings import MoveGroupInterface  # Jazzy API
# OR direct JointTrajectory publisher as fallback
```

Logic:
1. Subscribe `/target_pose`
2. Debounce: skip if distance < 0.02 m from last target
3. `move_group.set_pose_target(target_pose)` → plan → execute
4. Publish `/arm_status` ("MOVING" / "REACHED" / "FAILED")

**Config files needed:**
- `config/moveit_config/kinematics.yaml` — KDL solver for custom arm
- `config/moveit_config/catheter_arm.srdf` — planning group definition
- `config/moveit_config/joint_limits.yaml`

- [ ] MoveIt2 config generates correctly
- [ ] `ros2 run catheter_sim arm_controller` connects to move_group
- [ ] Arm moves toward target in Gazebo simulation
- [ ] Video/screenshot captured

**Deliverable:** Arm end-effector moves toward detected vessel target in Gazebo

---

#### Phase 5: Full System Launch
**File:** `ros2_ws/src/catheter_sim/launch/full_system.launch.py`

Launches in sequence:
1. `sim.launch.py` (Gazebo + robot + controllers)
2. `vessel_detector` node
3. `image_publisher` node  
4. `target_planner` node
5. `arm_controller` node
6. RViz2 with `catheter_sim.rviz` config

- [ ] Single command: `ros2 launch catheter_sim full_system.launch.py`
- [ ] All nodes start without errors
- [ ] End-to-end pipeline runs: image → detection → target → arm motion

**Deliverable:** Full demo runnable with one command

---

#### Phase 6: Clinical Documentation
Address in README / code comments:

**Simplifications vs. real system:**
- Fixed camera-to-world transform (no spatial calibration)
- 2D image → 3D target assumes fixed depth (no stereo/3D ultrasound)
- No real-time ultrasound feed (static images)
- No force/torque sensing or tissue interaction

**Clinical & engineering risks:**
| Risk | Description | Mitigation in real system |
|------|-------------|--------------------------|
| Ultrasound calibration | Spatial registration error → wrong target | Tracked probe + calibration phantom |
| Patient variability | Vessel depth varies ±2 cm across patients | Per-patient depth estimation |
| Vessel movement | Breathing/heartbeat moves vessel | Real-time tracking loop |
| Sterility | Robot must not contaminate sterile field | Sterile draping, instrument design |
| Force control | Tissue damage if force not limited | Impedance/admittance control |
| Safety interlocks | Runaway motion → injury | Hardware e-stop, workspace limits |
| False detection | Wrong structure targeted | Confirmation step, human-in-the-loop |

---

### ─── OPTION 2: gait_intent ───

#### Phase 0: Dataset + Environment Setup
- [ ] Download HDsEMG+IMU dataset from Figshare `doi:10.6084/m9.figshare.22227337`
- [ ] Install Python packages: `pip install torch scikit-learn scipy pandas`
- [ ] Explore dataset structure: print file list, read one subject's data
- [ ] Create `ros2_ws/src/gait_intent/` package skeleton

**Deliverable:** Dataset loaded in Python, print summary of activities and sensor channels

---

#### Phase 1: Data Loading
**File:** `ros2_ws/src/gait_intent/gait_intent/data_loader.py`

- Load all subjects' data (`.mat` or `.csv`)
- Extract: IMU channels (accel, gyro x/y/z per sensor) + subset of EMG channels
- Map activity labels → 7 classes: `WALKING, STAIR_ASCENT, STAIR_DESCENT, RAMP_ASCENT, RAMP_DESCENT, STANDING, SIT_TO_STAND`
- Plot class distribution → save to `results/plots/class_distribution.png`

**Deliverable:** `python3 data_loader.py` prints (N_samples, n_channels) per class

---

#### Phase 2: Signal Preprocessing
**File:** `ros2_ws/src/gait_intent/gait_intent/preprocess.py`

Steps:
1. **Bandpass filter:** `scipy.signal.butter(4, [0.5, 40], 'bandpass', fs=1000)` for IMU
2. **Sliding window:** 200 samples (200 ms), 100-sample hop (50% overlap)
3. **Normalization:** Z-score per channel (compute mean/std on train set only)
4. **Split:** Subject-independent 70/15/15 (subjects 1–7 train, 8–9 val, 10 test)
5. **RF features:** Per window, per channel: [mean, std, RMS, zero-crossing-rate, dominant_freq]

- [ ] Filtered signals plotted vs. raw
- [ ] Window shapes: X_train=(N, 200, C), y_train=(N,)
- [ ] Normalization stats saved to `config/norm_stats.json`

**Deliverable:** `python3 preprocess.py` saves windowed numpy arrays to `data/gait/processed/`

---

#### Phase 3: Classifier
**File:** `ros2_ws/src/gait_intent/gait_intent/classifier.py` (model definition)
**File:** `ros2_ws/src/gait_intent/gait_intent/train.py`

**Random Forest baseline:**
```python
from sklearn.ensemble import RandomForestClassifier
rf = RandomForestClassifier(n_estimators=200, max_depth=15, n_jobs=-1)
```

**1D CNN (primary):**
```
Input (200, n_channels)
→ Conv1D(32, k=5, padding=same) → BatchNorm → ReLU → MaxPool(2)
→ Conv1D(64, k=3, padding=same) → BatchNorm → ReLU → MaxPool(2)
→ Conv1D(128, k=3, padding=same) → BatchNorm → ReLU
→ GlobalAveragePool1D
→ Dense(128) → Dropout(0.3) → ReLU
→ Dense(7) → Softmax
```

Training: Adam lr=1e-3, batch=64, epochs=50, ReduceLROnPlateau, EarlyStopping(patience=10)

- [ ] RF baseline accuracy reported
- [ ] CNN trained, best weights saved to `results/gait_cnn.pt`
- [ ] Training curves saved to `results/plots/training_curves.png`

**Deliverable:** `python3 train.py` completes; model weights saved

---

#### Phase 4: Evaluation
**File:** `ros2_ws/src/gait_intent/gait_intent/evaluate.py`

- Load test set, run inference
- Print: accuracy, macro-F1, per-class precision/recall
- Save: confusion matrix → `results/plots/confusion_matrix.png`
- Save: sample prediction time-series → `results/plots/sample_predictions.png`
- Latency: time 1000 inferences, report mean ± std ms

**Deliverable:** Test accuracy ≥ 80%; confusion matrix image saved

---

#### Phase 5: Assistive Control FSM
**File:** `ros2_ws/src/gait_intent/gait_intent/control_fsm.py`

States and torque scales:
```
IDLE          → torque = 0.0  (system off)
STANDING      → torque = 0.0  (weight support only)
LEVEL_WALKING → torque = 1.0  (normal gait assistance)
STAIR_ASCENT  → torque = 1.5  (increased knee flexion assist)
STAIR_DESCENT → torque = 0.8  (eccentric brake mode)
RAMP_ASCENT   → torque = 1.2  (moderate assist)
RAMP_DESCENT  → torque = 0.8  (brake mode)
TRANSITION    → torque = 0.5  (intermediate, safety hold)
```

Transition rules: triggered by classifier output; hysteresis = 3 consecutive same-class windows before switching.

ASCII FSM diagram included in docstring.

- [ ] `python3 control_fsm.py` prints demo: 20-step sequence with states + torque values

**Deliverable:** FSM transitions visible in output; ASCII diagram in code

---

#### Phase 6: Optional ROS2 Node
**File:** `ros2_ws/src/gait_intent/gait_intent/gait_ros2_node.py`

```
Subscribes: /imu_raw (sensor_msgs/Imu)
Publishes:  /gait_mode     (std_msgs/String)
            /assistive_state (std_msgs/Float32)  ← torque scale
```

- [ ] Node runs: `ros2 run gait_intent gait_ros2_node`
- [ ] ONNX export: `results/gait_cnn.onnx`

**Deliverable:** Node publishes gait mode from simulated IMU stream

---

## Documentation Requirements (README.md)

### Option 1 sections:
- Setup instructions (with install commands)
- System architecture diagram
- Dataset used and why (Mus-V)
- Perception approach: OpenCV classical pipeline + why chosen
- Simplifications vs. real system
- Clinical risks and mitigations
- Limitations and future improvements
- How to run

### Option 2 sections:
- Dataset source and description
- Preprocessing pipeline
- Model architecture and training details
- Results (accuracy, F1, confusion matrix image)
- FSM design and control logic
- Latency analysis
- Safety and deployment considerations
- Limitations

---

## Two-Agent Execution Plan

When implementation begins, spawn two parallel agents:

**Agent 1 — Option 1 (`catheter_sim`)**
Scope: Phases 1–6 of Option 1
Workspace: `ros2_ws/src/catheter_sim/`
Key files: URDF, world SDF, vessel_detector.py, target_planner.py, arm_controller.py, launch files

**Agent 2 — Option 2 (`gait_intent`)**
Scope: Phases 1–6 of Option 2
Workspace: `ros2_ws/src/gait_intent/`
Key files: data_loader.py, preprocess.py, classifier.py, train.py, evaluate.py, control_fsm.py

Both agents write to their own package directories; no file conflicts expected.
README.md consolidated at the end.

---

## Progress Tracker

### Option 1 — catheter_sim
- [ ] Phase 0: Environment setup
- [ ] Phase 1: URDF + Gazebo world
- [ ] Phase 2: Perception node
- [ ] Phase 3: Target planner
- [ ] Phase 4: Arm controller (MoveIt2)
- [ ] Phase 5: Full launch file
- [ ] Phase 6: Screenshots + clinical docs

### Option 2 — gait_intent
- [ ] Phase 0: Dataset download
- [ ] Phase 1: Data loading
- [ ] Phase 2: Preprocessing
- [ ] Phase 3: Classifier training
- [ ] Phase 4: Evaluation
- [ ] Phase 5: FSM control
- [ ] Phase 6: Optional ROS2 node

### Final
- [ ] README.md
- [ ] Dockerfile
- [ ] Architecture diagram
- [ ] Results folder populated

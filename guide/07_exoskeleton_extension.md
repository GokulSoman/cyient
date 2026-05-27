# Guide 07 — Exoskeleton Gait Intent Recognition (Option 2)

## Overview

**Package:** `gait_intent` (in `ros2_ws/src/gait_intent/`)  
**Dataset:** HDsEMG+IMU — Nature Scientific Data 2023 (Figshare: `doi:10.6084/m9.figshare.22227337`)  
**Goal:** Classify locomotion mode from wearable sensors → drive an assistive FSM

---

## Package Structure

```
gait_intent/
├── gait_intent/                # Python package
│   ├── __init__.py
│   ├── data_loader.py          # Load Figshare dataset
│   ├── preprocess.py           # Filter, window, normalize, split
│   ├── classifier.py           # RF baseline + 1D CNN definitions
│   ├── train.py                # Training loop
│   ├── evaluate.py             # Metrics, confusion matrix, latency
│   ├── control_fsm.py          # Finite-state machine control logic
│   └── gait_ros2_node.py       # Optional ROS2 node
├── scripts/
│   └── run_pipeline.py         # Single-script end-to-end demo
├── config/
│   └── params.yaml
├── package.xml
├── setup.py
└── CMakeLists.txt
```

---

## Dataset: HDsEMG+IMU

**Citation:** Brod et al. (2023). "High-density EMG, IMU, kinetic, and kinematic open-source data for comprehensive locomotion activities." *Scientific Data*, 10, 709.

```bash
# Download from Figshare (CC0 license)
# Manual: https://figshare.com/articles/dataset/22227337
# OR via API:

python3 - << 'EOF'
import requests, os

article_id = "22227337"
url = f"https://api.figshare.com/v2/articles/{article_id}/files"
r = requests.get(url)

os.makedirs("data/gait", exist_ok=True)
for f in r.json():
    print(f"{f['name']}: {f['size']//1024//1024} MB")
    # Uncomment to download:
    # resp = requests.get(f['download_url'], stream=True)
    # with open(f"data/gait/{f['name']}", 'wb') as fp:
    #     for chunk in resp.iter_content(8192): fp.write(chunk)
EOF
```

---

## Activity Labels and FSM Mapping

| Dataset Activity | Class Index | FSM State | Torque Scale |
|-----------------|-------------|-----------|-------------|
| level_walking (slow/normal/fast) | 0 | LEVEL_WALKING | 1.0× |
| stair_ascent | 1 | STAIR_ASCENT | 1.5× |
| stair_descent | 2 | STAIR_DESCENT | 0.8× (brake) |
| ramp_ascent | 3 | RAMP_ASCENT | 1.2× |
| ramp_descent | 4 | RAMP_DESCENT | 0.8× (brake) |
| standing / static | 5 | STANDING | 0.0 |
| sit_to_stand | 6 | SIT_TO_STAND | burst 2.0× → 1.0× |

---

## `data_loader.py`

```python
"""Load HDsEMG+IMU dataset from Figshare files."""

import os
import numpy as np
import scipy.io
from pathlib import Path


# Map from filename keyword → class index
ACTIVITY_MAP = {
    'level_walking': 0,
    'stair_ascent':  1,
    'stair_descent': 2,
    'ramp_ascent':   3,
    'ramp_descent':  4,
    'standing':      5,
    'sit_to_stand':  6,
}

CLASS_NAMES = [
    'level_walking', 'stair_ascent', 'stair_descent',
    'ramp_ascent', 'ramp_descent', 'standing', 'sit_to_stand'
]


def load_dataset(data_dir: str = 'data/gait') -> tuple:
    """
    Load all subjects from the dataset.
    
    Returns:
        X: list of (n_samples, n_channels) arrays per segment
        y: list of int labels per segment
        subjects: list of subject IDs per segment
    """
    X_all, y_all, subjects = [], [], []
    
    data_dir = Path(data_dir)
    
    # Explore structure
    mat_files = list(data_dir.rglob('*.mat'))
    if not mat_files:
        raise FileNotFoundError(f"No .mat files found in {data_dir}")
    
    print(f"Found {len(mat_files)} .mat files")
    
    for f in sorted(mat_files):
        # Determine activity from filename
        label = None
        for key, idx in ACTIVITY_MAP.items():
            if key in f.stem.lower():
                label = idx
                break
        
        if label is None:
            print(f"  Skipping (unknown activity): {f.name}")
            continue
        
        # Determine subject from path
        subject_id = _get_subject_id(f)
        
        # Load .mat file
        try:
            data = scipy.io.loadmat(str(f))
        except Exception as e:
            print(f"  Error loading {f.name}: {e}")
            continue
        
        # Extract sensor channels (adapt key names to actual dataset structure)
        signals = _extract_signals(data)
        if signals is None:
            continue
        
        X_all.append(signals)  # (n_samples, n_channels)
        y_all.append(label)
        subjects.append(subject_id)
        
        print(f"  {f.name}: shape={signals.shape}, label={CLASS_NAMES[label]}, subject={subject_id}")
    
    return X_all, y_all, subjects


def _extract_signals(data: dict) -> np.ndarray | None:
    """
    Extract IMU + selected EMG channels from a loaded .mat dict.
    Adapt key names after exploring actual dataset structure.
    """
    # Try common key names in biomechanics datasets
    imu_keys = ['IMU', 'imu', 'Acc', 'acc', 'Gyro', 'gyro']
    
    signals_list = []
    
    for key in data.keys():
        if key.startswith('__'):
            continue
        val = data[key]
        if isinstance(val, np.ndarray) and val.ndim == 2:
            # Likely a signal matrix (n_samples, n_channels) or (n_channels, n_samples)
            if val.shape[0] > val.shape[1]:
                signals_list.append(val)   # (n_samples, n_channels) format
            else:
                signals_list.append(val.T)  # transpose to (n_samples, n_channels)
    
    if not signals_list:
        return None
    
    # Use first valid signal matrix found
    # (Refine this after exploring actual dataset structure)
    return signals_list[0].astype(np.float32)


def _get_subject_id(file_path: Path) -> int:
    """Extract subject ID from file path."""
    parts = file_path.parts
    for part in parts:
        if part.lower().startswith('subject') or part.lower().startswith('sub'):
            try:
                return int(''.join(filter(str.isdigit, part)))
            except ValueError:
                pass
    return 0


def explore_dataset(data_dir: str = 'data/gait'):
    """Quick exploration tool — run this first to understand data structure."""
    data_dir = Path(data_dir)
    mat_files = list(data_dir.rglob('*.mat'))
    
    if not mat_files:
        print("No .mat files found. Check data/gait/ directory.")
        return
    
    print(f"\n=== Dataset Explorer ===")
    print(f"Found {len(mat_files)} files in {data_dir}")
    
    # Load first file and print structure
    sample = scipy.io.loadmat(str(mat_files[0]))
    print(f"\nSample file: {mat_files[0].name}")
    print("Keys and shapes:")
    for k, v in sample.items():
        if not k.startswith('__'):
            if hasattr(v, 'shape'):
                print(f"  '{k}': shape={v.shape}, dtype={v.dtype}")
            else:
                print(f"  '{k}': {type(v).__name__} = {v}")


if __name__ == '__main__':
    explore_dataset()
```

---

## `preprocess.py`

```python
"""Signal preprocessing: filter, window, normalize, split."""

import numpy as np
from scipy.signal import butter, sosfilt
from sklearn.preprocessing import StandardScaler
import pickle
import os


def bandpass_filter(signal: np.ndarray, lowcut: float = 0.5, highcut: float = 40.0,
                    fs: float = 1000.0, order: int = 4) -> np.ndarray:
    """Apply 4th-order Butterworth bandpass filter to each channel."""
    nyq = fs / 2.0
    sos = butter(order, [lowcut / nyq, highcut / nyq], btype='band', output='sos')
    return sosfilt(sos, signal, axis=0)


def sliding_window(signal: np.ndarray, window_size: int = 200, step: int = 100) -> np.ndarray:
    """
    Create overlapping windows from a signal.
    
    Args:
        signal: (n_samples, n_channels)
        window_size: samples per window (200 samples @ 1kHz = 200 ms)
        step: hop size (100 samples = 50% overlap)
    
    Returns:
        windows: (n_windows, window_size, n_channels)
    """
    n_samples, n_channels = signal.shape
    windows = []
    
    for start in range(0, n_samples - window_size + 1, step):
        end = start + window_size
        windows.append(signal[start:end, :])
    
    return np.stack(windows, axis=0) if windows else np.empty((0, window_size, n_channels))


def extract_features(windows: np.ndarray) -> np.ndarray:
    """
    Extract handcrafted features per window for Random Forest.
    
    Args:
        windows: (n_windows, window_size, n_channels)
    
    Returns:
        features: (n_windows, n_features)
        Features per channel: [mean, std, RMS, zero_crossing_rate, dominant_freq_idx]
    """
    n_windows, window_size, n_channels = windows.shape
    features_list = []
    
    for w in windows:  # w: (window_size, n_channels)
        feats = []
        for ch in range(n_channels):
            x = w[:, ch]
            mean = np.mean(x)
            std = np.std(x)
            rms = np.sqrt(np.mean(x ** 2))
            zcr = np.sum(np.diff(np.sign(x)) != 0) / window_size
            fft_mag = np.abs(np.fft.rfft(x))
            dom_freq = np.argmax(fft_mag[1:]) + 1  # skip DC
            feats.extend([mean, std, rms, zcr, dom_freq])
        features_list.append(feats)
    
    return np.array(features_list, dtype=np.float32)


def train_test_split_by_subject(X_list, y_list, subjects,
                                 train_subj=None, val_subj=None, test_subj=None,
                                 window_size=200, step=100, fs=1000.0):
    """
    Subject-independent split: different subjects in train/val/test.
    Default: subjects 1–7 train, 8–9 val, 10 test.
    """
    if train_subj is None:
        train_subj = list(range(1, 8))
    if val_subj is None:
        val_subj = [8, 9]
    if test_subj is None:
        test_subj = [10]
    
    def process_subset(subset_subjects):
        X_wins, y_wins = [], []
        for sig, label, subj in zip(X_list, y_list, subjects):
            if subj not in subset_subjects:
                continue
            # Filter signal
            filtered = bandpass_filter(sig, fs=fs)
            # Window
            wins = sliding_window(filtered, window_size, step)
            if len(wins) == 0:
                continue
            X_wins.append(wins)
            y_wins.extend([label] * len(wins))
        
        if not X_wins:
            return np.empty((0, window_size, 1)), np.array([])
        
        X_combined = np.concatenate(X_wins, axis=0)
        y_combined = np.array(y_wins, dtype=np.int64)
        return X_combined, y_combined
    
    X_train, y_train = process_subset(train_subj)
    X_val,   y_val   = process_subset(val_subj)
    X_test,  y_test  = process_subset(test_subj)
    
    # Normalize using train set statistics
    n_train, win, n_ch = X_train.shape
    scaler_mean = X_train.reshape(-1, n_ch).mean(axis=0)
    scaler_std  = X_train.reshape(-1, n_ch).std(axis=0) + 1e-8
    
    def normalize(X):
        sh = X.shape
        X_flat = X.reshape(-1, n_ch)
        X_norm = (X_flat - scaler_mean) / scaler_std
        return X_norm.reshape(sh)
    
    X_train = normalize(X_train)
    X_val   = normalize(X_val)
    X_test  = normalize(X_test)
    
    # Save scaler stats
    os.makedirs('config', exist_ok=True)
    np.save('config/norm_mean.npy', scaler_mean)
    np.save('config/norm_std.npy', scaler_std)
    
    print(f"Train: {X_train.shape}, Val: {X_val.shape}, Test: {X_test.shape}")
    return X_train, y_train, X_val, y_val, X_test, y_test
```

---

## `classifier.py`

```python
"""Random Forest baseline and 1D CNN for gait mode classification."""

import torch
import torch.nn as nn
from sklearn.ensemble import RandomForestClassifier


class GaitCNN1D(nn.Module):
    """1D CNN for locomotion mode classification from wearable sensor windows."""
    
    def __init__(self, n_channels: int, n_classes: int = 7, window_size: int = 200):
        super().__init__()
        
        self.features = nn.Sequential(
            # Block 1
            nn.Conv1d(n_channels, 32, kernel_size=5, padding=2),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.MaxPool1d(2),               # → 100 timesteps
            
            # Block 2
            nn.Conv1d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(2),               # → 50 timesteps
            
            # Block 3
            nn.Conv1d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm1d(128),
            nn.ReLU(),
        )
        
        # Global average pool collapses time dimension → (batch, 128)
        self.global_pool = nn.AdaptiveAvgPool1d(1)
        
        self.classifier = nn.Sequential(
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, n_classes),
        )
    
    def forward(self, x):
        # x: (batch, window_size, n_channels) → need (batch, n_channels, window_size)
        x = x.permute(0, 2, 1)
        x = self.features(x)                # (batch, 128, T')
        x = self.global_pool(x).squeeze(-1) # (batch, 128)
        x = self.classifier(x)              # (batch, n_classes)
        return x


def build_random_forest():
    return RandomForestClassifier(
        n_estimators=200,
        max_depth=20,
        min_samples_split=5,
        n_jobs=-1,
        random_state=42,
        class_weight='balanced',
    )
```

---

## `control_fsm.py`

```python
"""
Assistive exoskeleton finite-state machine.

States and transitions driven by gait mode classifier output.
Hysteresis: require 3 consecutive same predictions before switching state.

ASCII State Diagram:
───────────────────────────────────────────────────────────────
                   ┌──────────────┐
              ┌───►│     IDLE     │◄───────────────┐
              │    └──────┬───────┘                │
              │           │ pred≠idle              │ error/stop
              │           ▼                        │
              │    ┌──────────────┐                │
              │    │  STANDING    │ ←──────────┐   │
              │    │  torque=0.0  │            │   │
              │    └──┬─────┬─────┘            │   │
              │       │     │ sitting           │   │
              │walking│     ▼                  │   │
              │       │  ┌──────────────┐      │   │
              │       │  │ SIT_TO_STAND │──────►   │
              │       │  │ torque=2.0→1 │ standing  │
              │       │  └──────────────┘           │
              │       │                             │
              │       ▼                             │
              │    ┌──────────────┐                 │
              │    │LEVEL_WALKING │                 │
              │    │ torque=1.0   │                 │
              │    └──┬──┬──┬────┘                 │
              │  stair│  │  │ramp                  │
              │  asc  │  │  │                      │
              │  ┌────┘  │  └────────┐             │
              │  ▼        ▼           ▼             │
              │ ┌──────────┐  ┌──────────────┐     │
              │ │STAIR_ASC │  │ RAMP_ASC     │     │
              │ │torque=1.5│  │ torque=1.2   │     │
              │ └──────────┘  └──────────────┘     │
              │                                     │
              └─────────────────────────────────────┘
───────────────────────────────────────────────────────────────
"""

from dataclasses import dataclass
from typing import Optional
from collections import deque


@dataclass
class FSMState:
    name: str
    torque_scale: float
    description: str


# Define all states
STATES = {
    'IDLE':          FSMState('IDLE',          0.0, 'System off / initializing'),
    'STANDING':      FSMState('STANDING',      0.0, 'Static standing, no assist needed'),
    'LEVEL_WALKING': FSMState('LEVEL_WALKING', 1.0, 'Normal gait assistance'),
    'STAIR_ASCENT':  FSMState('STAIR_ASCENT',  1.5, 'Increased knee flexion assist'),
    'STAIR_DESCENT': FSMState('STAIR_DESCENT', 0.8, 'Eccentric brake mode'),
    'RAMP_ASCENT':   FSMState('RAMP_ASCENT',   1.2, 'Moderate uphill assist'),
    'RAMP_DESCENT':  FSMState('RAMP_DESCENT',  0.8, 'Downhill brake mode'),
    'SIT_TO_STAND':  FSMState('SIT_TO_STAND',  2.0, 'Hip/knee extension burst'),
    'TRANSITION':    FSMState('TRANSITION',    0.5, 'Mode switching — safety hold'),
}

# Map from class index → FSM state name
PREDICTION_TO_STATE = {
    0: 'LEVEL_WALKING',
    1: 'STAIR_ASCENT',
    2: 'STAIR_DESCENT',
    3: 'RAMP_ASCENT',
    4: 'RAMP_DESCENT',
    5: 'STANDING',
    6: 'SIT_TO_STAND',
}


class ExoskeletonFSM:
    """
    Finite-state machine for exoskeleton assistive control.
    
    Uses hysteresis (N consecutive same predictions) before
    switching states to avoid chattering on ambiguous transitions.
    """
    
    def __init__(self, hysteresis: int = 3):
        self.current_state = STATES['IDLE']
        self.hysteresis = hysteresis
        self.prediction_buffer: deque = deque(maxlen=hysteresis)
        self.step_count = 0
        self.state_history = []
    
    def update(self, prediction: int) -> FSMState:
        """
        Update FSM with a new classifier prediction.
        
        Args:
            prediction: class index from classifier
        
        Returns:
            Current FSMState (may or may not have changed)
        """
        self.step_count += 1
        self.prediction_buffer.append(prediction)
        
        # Only switch if hysteresis buffer is full and all same
        if len(self.prediction_buffer) == self.hysteresis:
            if len(set(self.prediction_buffer)) == 1:  # all same
                new_state_name = PREDICTION_TO_STATE.get(prediction, 'STANDING')
                new_state = STATES[new_state_name]
                
                if new_state.name != self.current_state.name:
                    # Log transition
                    print(f"  [FSM] {self.current_state.name} → {new_state.name}"
                          f"  (torque: {self.current_state.torque_scale:.1f} → {new_state.torque_scale:.1f})")
                    self.state_history.append((self.step_count, self.current_state.name, new_state.name))
                    self.current_state = new_state
        
        return self.current_state
    
    def get_torque_command(self) -> float:
        """Return current torque scale factor for actuator."""
        return self.current_state.torque_scale
    
    def print_transition_table(self):
        print("\n=== Transition History ===")
        print(f"{'Step':>6} | {'From':>14} | {'To':>14} | {'Torque':>8}")
        print("-" * 48)
        for step, from_s, to_s in self.state_history:
            torque = STATES[to_s].torque_scale
            print(f"{step:>6} | {from_s:>14} | {to_s:>14} | {torque:>8.1f}")


def demo():
    """Run a demo of the FSM with a synthetic prediction sequence."""
    print("=== Exoskeleton FSM Demo ===\n")
    
    fsm = ExoskeletonFSM(hysteresis=3)
    
    # Simulate: standing → walking → stair ascent → descent → walking → standing
    sequence = (
        [5] * 5 +     # STANDING
        [0] * 6 +     # LEVEL_WALKING
        [1] * 5 +     # STAIR_ASCENT
        [2] * 5 +     # STAIR_DESCENT
        [0] * 5 +     # LEVEL_WALKING
        [5] * 4       # STANDING
    )
    
    print(f"{'Step':>4} | {'Prediction':>15} | {'State':>14} | {'Torque':>8}")
    print("-" * 50)
    
    for i, pred in enumerate(sequence):
        state = fsm.update(pred)
        pred_name = PREDICTION_TO_STATE.get(pred, '?')
        print(f"{i+1:>4} | {pred_name:>15} | {state.name:>14} | {state.torque_scale:>8.1f}")
    
    fsm.print_transition_table()


if __name__ == '__main__':
    demo()
```

---

## ROS2 Node (Optional): `gait_ros2_node.py`

```python
#!/usr/bin/env python3
"""ROS2 node wrapping the gait intent classifier and FSM."""

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Imu
from std_msgs.msg import String, Float32

import numpy as np
import torch
from collections import deque

from gait_intent.classifier import GaitCNN1D
from gait_intent.control_fsm import ExoskeletonFSM, PREDICTION_TO_STATE


class GaitIntentNode(Node):
    
    def __init__(self):
        super().__init__('gait_intent')
        
        self.declare_parameter('model_path', 'results/gait_cnn.pt')
        self.declare_parameter('window_size', 200)
        self.declare_parameter('n_channels', 6)
        self.declare_parameter('n_classes', 7)
        
        win  = self.get_parameter('window_size').value
        n_ch = self.get_parameter('n_channels').value
        n_cl = self.get_parameter('n_classes').value
        
        # Load model
        self.model = GaitCNN1D(n_channels=n_ch, n_classes=n_cl, window_size=win)
        model_path = self.get_parameter('model_path').value
        if os.path.exists(model_path):
            self.model.load_state_dict(torch.load(model_path, map_location='cpu'))
        self.model.eval()
        
        # Load normalization stats
        self.norm_mean = np.load('config/norm_mean.npy')
        self.norm_std  = np.load('config/norm_std.npy')
        
        # Sliding window buffer
        self.buffer = deque(maxlen=win)
        self.fsm = ExoskeletonFSM(hysteresis=3)
        
        # Subscribers
        self.create_subscription(Imu, '/imu_raw', self.imu_callback, 10)
        
        # Publishers
        self.pub_mode  = self.create_publisher(String, '/gait_mode', 10)
        self.pub_torque = self.create_publisher(Float32, '/assistive_state', 10)
        
        self.get_logger().info('GaitIntent node started')
    
    def imu_callback(self, msg: Imu):
        # Extract 6-channel IMU: accel xyz + gyro xyz
        sample = np.array([
            msg.linear_acceleration.x,
            msg.linear_acceleration.y,
            msg.linear_acceleration.z,
            msg.angular_velocity.x,
            msg.angular_velocity.y,
            msg.angular_velocity.z,
        ], dtype=np.float32)
        
        self.buffer.append(sample)
        
        if len(self.buffer) < self.get_parameter('window_size').value:
            return
        
        # Build window
        window = np.stack(list(self.buffer), axis=0)  # (200, 6)
        window = (window - self.norm_mean) / self.norm_std
        
        # Inference
        x = torch.from_numpy(window).unsqueeze(0)  # (1, 200, 6)
        with torch.no_grad():
            logits = self.model(x)
            pred = int(logits.argmax(dim=1).item())
        
        # FSM update
        state = self.fsm.update(pred)
        
        # Publish
        mode_msg = String(); mode_msg.data = state.name
        torque_msg = Float32(); torque_msg.data = state.torque_scale
        
        self.pub_mode.publish(mode_msg)
        self.pub_torque.publish(torque_msg)


def main(args=None):
    import os
    rclpy.init(args=args)
    node = GaitIntentNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()
```

---

## Running the Full Option 2 Pipeline

```bash
# Activate conda env
conda activate cyient
source /opt/ros/jazzy/setup.bash
source ros2_ws/install/setup.bash

# 1. Explore dataset structure
python3 ros2_ws/src/gait_intent/gait_intent/data_loader.py

# 2. Run full ML pipeline
python3 ros2_ws/src/gait_intent/scripts/run_pipeline.py

# 3. Evaluate model
python3 ros2_ws/src/gait_intent/gait_intent/evaluate.py
# → Prints: accuracy, F1 per class, confusion matrix
# → Saves: results/plots/confusion_matrix.png

# 4. Demo FSM
python3 ros2_ws/src/gait_intent/gait_intent/control_fsm.py
# → Prints FSM state transitions for demo sequence

# 5. Optional: ROS2 node
ros2 run gait_intent gait_intent_node
```

---

## Expected Results

| Metric | Target |
|--------|--------|
| Test accuracy | ≥ 80% |
| Macro F1 | ≥ 0.78 |
| Inference latency | < 20 ms per window |
| Training time | < 30 min (CPU) |

---

## Clinical Safety Reasoning (For README)

| Risk | Description | Mitigation |
|------|-------------|-----------|
| Misclassification | Wrong mode → wrong torque (e.g., stair torque during walking) | Hysteresis, confidence threshold, gradual torque ramp |
| Latency | ~200 ms window + inference > 20 ms could cause control lag | Optimize with ONNX/TFLite; use shorter windows |
| Patient variability | Sensor placement varies between sessions | Per-session calibration; normalize to rest baseline |
| Sensor noise | EMG affected by electrode impedance | Bandpass filter; artifact rejection |
| Battery failure | Embedded system crash → exo locks | Fail-safe: spring-return mechanism; hardware watchdog |
| False transitions | Chattering between states | Hysteresis N=3; add transition state with reduced torque |

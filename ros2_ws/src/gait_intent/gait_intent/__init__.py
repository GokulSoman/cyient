"""
gait_intent — Exoskeleton Gait Intent Recognition Package

Provides:
  - data_loader: Load HDsEMG+IMU dataset (with synthetic fallback)
  - preprocess:  Filter, window, normalize, split
  - classifier:  RF baseline + 1D CNN model
  - train:       Training loop
  - evaluate:    Metrics, confusion matrix, latency
  - control_fsm: 8-state FSM with hysteresis
  - gait_ros2_node: Optional ROS2 node
"""

__version__ = "0.1.0"

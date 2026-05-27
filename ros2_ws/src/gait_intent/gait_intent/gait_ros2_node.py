"""
gait_ros2_node.py — ROS2 node for real-time gait intent recognition.

Subscribes:
  /imu_raw  (sensor_msgs/Imu)  — 6-axis IMU: linear_acceleration + angular_velocity

Publishes:
  /gait_mode       (std_msgs/String)  — current locomotion mode name
  /assistive_state (std_msgs/Float32) — torque scale for exoskeleton actuators

The node maintains a sliding window buffer (deque).  When a full window
accumulates, it runs the CNN (or RF as fallback) and updates the FSM.

If no trained model is found, the node falls back to random predictions for
demonstration purposes (useful for testing the ROS2 interface without training).
"""

import os
import sys
import numpy as np
from collections import deque
from pathlib import Path

# ── Allow imports when not installed as ROS2 package ─────────────────────────
_PKG = Path(__file__).resolve().parents[2]
if str(_PKG) not in sys.path:
    sys.path.insert(0, str(_PKG))

from gait_intent.control_fsm import ExoskeletonFSM, PREDICTION_TO_STATE  # noqa: F401

try:
    from gait_intent.classifier import TORCH_OK
except Exception:
    TORCH_OK = False

# ROS2 imports — graceful if not in ROS2 environment
try:
    import rclpy
    from rclpy.node import Node
    from sensor_msgs.msg import Imu
    from std_msgs.msg import String, Float32
    ROS2_OK = True
except ImportError:
    ROS2_OK = False
    print("[gait_ros2_node] rclpy not available — ROS2 node cannot run.")


REPO_ROOT   = Path(__file__).resolve().parents[4]
RESULTS_DIR = REPO_ROOT / 'results'
CONFIG_DIR  = REPO_ROOT / 'ros2_ws' / 'src' / 'gait_intent' / 'config'


class GaitIntentNode(Node if ROS2_OK else object):  # type: ignore
    """
    ROS2 node that classifies gait mode from IMU data and drives an FSM.

    Parameters (ROS2 params):
      model_path   : path to gait_cnn.pt   (default: results/gait_cnn.pt)
      rf_path      : path to gait_rf.pkl   (default: results/gait_rf.pkl)
      window_size  : samples per window    (default: 200)
      n_channels   : IMU channels          (default: 6)
      n_classes    : number of activities  (default: 7)
      hysteresis   : FSM hysteresis count  (default: 3)
      publish_rate : Hz for timer-based    (default: 0 = callback-only)
    """

    def __init__(self):
        if not ROS2_OK:
            raise RuntimeError("rclpy is not available")
        super().__init__('gait_intent')

        # ── Declare parameters ────────────────────────────────────────────
        self.declare_parameter('model_path',  str(RESULTS_DIR / 'gait_cnn.pt'))
        self.declare_parameter('rf_path',     str(RESULTS_DIR / 'gait_rf.pkl'))
        self.declare_parameter('window_size', 200)
        self.declare_parameter('n_channels',  6)
        self.declare_parameter('n_classes',   7)
        self.declare_parameter('hysteresis',  3)

        win    = self.get_parameter('window_size').value
        n_ch   = self.get_parameter('n_channels').value
        n_cl   = self.get_parameter('n_classes').value
        hyst   = self.get_parameter('hysteresis').value

        # ── Load normalization stats ──────────────────────────────────────
        self.norm_mean = np.zeros(n_ch, dtype=np.float32)
        self.norm_std  = np.ones(n_ch,  dtype=np.float32)
        mean_path = CONFIG_DIR / 'norm_mean.npy'
        std_path  = CONFIG_DIR / 'norm_std.npy'
        if mean_path.exists() and std_path.exists():
            self.norm_mean = np.load(str(mean_path)).astype(np.float32)
            self.norm_std  = np.load(str(std_path)).astype(np.float32)
            self.get_logger().info('Loaded normalization statistics')
        else:
            self.get_logger().warn(
                f'Norm stats not found ({mean_path}) — using identity normalization')

        # ── Load CNN model ────────────────────────────────────────────────
        self.model = None
        self.use_cnn = False
        model_path = self.get_parameter('model_path').value
        if TORCH_OK and os.path.exists(model_path):
            try:
                import torch
                from gait_intent.classifier import GaitCNN1D
                m = GaitCNN1D(n_channels=n_ch, n_classes=n_cl, window_size=win)
                m.load_state_dict(torch.load(model_path, map_location='cpu',
                                              weights_only=True))
                m.eval()
                self.model = m
                self.use_cnn = True
                self.get_logger().info(f'Loaded CNN model from {model_path}')
            except Exception as e:
                self.get_logger().warn(f'Could not load CNN: {e}')

        # ── Load RF model (fallback) ──────────────────────────────────────
        self.rf = None
        rf_path = self.get_parameter('rf_path').value
        if not self.use_cnn and os.path.exists(rf_path):
            try:
                import pickle
                from gait_intent.preprocess import extract_features
                with open(rf_path, 'rb') as fh:
                    self.rf = pickle.load(fh)
                self.extract_features = extract_features
                self.get_logger().info(f'Loaded RF model from {rf_path}')
            except Exception as e:
                self.get_logger().warn(f'Could not load RF: {e}')

        if not self.use_cnn and self.rf is None:
            self.get_logger().warn(
                'No model found — using RANDOM predictions (demo only!)')

        # ── Sliding window buffer ─────────────────────────────────────────
        self.window_size = win
        self.n_channels  = n_ch
        self.buffer: deque = deque(maxlen=win)

        # ── FSM ───────────────────────────────────────────────────────────
        self.fsm = ExoskeletonFSM(hysteresis=hyst)

        # ── ROS2 pub/sub ──────────────────────────────────────────────────
        self.sub_imu = self.create_subscription(
            Imu, '/imu_raw', self._imu_callback, 10)
        self.pub_mode   = self.create_publisher(String,  '/gait_mode',       10)
        self.pub_torque = self.create_publisher(Float32, '/assistive_state',  10)

        self.get_logger().info(
            f'GaitIntent node ready  (window={win}, channels={n_ch}, '
            f'hysteresis={hyst}, model={"CNN" if self.use_cnn else ("RF" if self.rf else "RANDOM")})'
        )

    # ── IMU callback ─────────────────────────────────────────────────────────

    def _imu_callback(self, msg: 'Imu') -> None:
        """Process one IMU sample; run inference when buffer is full."""
        sample = np.array([
            msg.linear_acceleration.x,
            msg.linear_acceleration.y,
            msg.linear_acceleration.z,
            msg.angular_velocity.x,
            msg.angular_velocity.y,
            msg.angular_velocity.z,
        ], dtype=np.float32)

        # Pad or trim if channel count differs
        if len(sample) < self.n_channels:
            sample = np.concatenate([
                sample,
                np.zeros(self.n_channels - len(sample), dtype=np.float32)
            ])
        else:
            sample = sample[:self.n_channels]

        self.buffer.append(sample)

        if len(self.buffer) < self.window_size:
            return  # not enough data yet

        # Build window and normalize
        window = np.stack(list(self.buffer), axis=0)  # (win, n_ch)
        window = (window - self.norm_mean) / self.norm_std

        prediction = self._infer(window)
        state = self.fsm.update(prediction)

        # Publish
        mode_msg = String()
        mode_msg.data = state.name
        self.pub_mode.publish(mode_msg)

        torque_msg = Float32()
        torque_msg.data = float(state.torque_scale)
        self.pub_torque.publish(torque_msg)

    def _infer(self, window: np.ndarray) -> int:
        """Run model inference; return predicted class index."""
        if self.use_cnn:
            try:
                import torch
                with torch.no_grad():
                    x = torch.from_numpy(window).unsqueeze(0)  # (1, win, ch)
                    logits = self.model(x)
                    return int(logits.argmax(dim=1).item())
            except Exception as e:
                self.get_logger().warn(f'CNN inference error: {e}', throttle_duration_sec=5)

        if self.rf is not None:
            try:
                from gait_intent.preprocess import sliding_window, extract_features
                feat = extract_features(window[np.newaxis, :, :])  # (1, win, ch)
                return int(self.rf.predict(feat)[0])
            except Exception as e:
                self.get_logger().warn(f'RF inference error: {e}', throttle_duration_sec=5)

        # Random fallback for demo
        return int(np.random.randint(0, 7))


# ── Entry point ───────────────────────────────────────────────────────────────

def main(args=None):
    if not ROS2_OK:
        print("[gait_ros2_node] rclpy not available. Cannot start ROS2 node.")
        print("Install ROS2 Jazzy and source /opt/ros/jazzy/setup.bash")
        return

    rclpy.init(args=args)
    node = GaitIntentNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()

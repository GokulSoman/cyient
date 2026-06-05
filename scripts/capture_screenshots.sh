#!/usr/bin/env bash
# capture_screenshots.sh — takes timed screenshots of the running simulation
#
# Usage: bash scripts/capture_screenshots.sh [delay_seconds]
#   delay_seconds: how long to wait after calling this script before shooting (default 5)
#
# Requires: DISPLAY set, ImageMagick (import), xwd
# Run AFTER: ros2 launch catheter_sim full_system.launch.py
#
# Captures:
#   1. Full desktop (overview)
#   2. GZ Sim window
#   3. RViz window
#   4. Vessel detection image (via ros2 topic + cv2)
#   5. /target_pose and /vessel_detection topic echoes

set -euo pipefail

REPO=/home/gokul/github_repos/cyient
OUT=$REPO/results/screenshots
mkdir -p "$OUT"

DELAY=${1:-5}
echo "[capture] Waiting ${DELAY}s for windows to settle…"
sleep "$DELAY"

DISPLAY="${DISPLAY:-:1}"
export DISPLAY

# ── Helper: screenshot a window by title substring ────────────────────────────
screenshot_window() {
  local title_grep="$1"
  local outfile="$2"
  local wid
  # Try xdotool first (precise), fall back to import root crop
  if command -v xdotool &>/dev/null; then
    wid=$(xdotool search --name "$title_grep" 2>/dev/null | head -1)
    if [[ -n "$wid" ]]; then
      import -window "$wid" "$outfile" 2>/dev/null && \
        echo "[capture] ✅  $outfile  (window: $wid)" && return
    fi
  fi
  # Fallback: full desktop screenshot
  import -window root "$outfile" 2>/dev/null && \
    echo "[capture] ✅  $outfile  (full desktop fallback)"
}

# ── 1. Full desktop overview ──────────────────────────────────────────────────
echo "[capture] 1/5 Full desktop…"
import -window root "$OUT/01_full_desktop.png" 2>/dev/null && \
  echo "[capture] ✅  01_full_desktop.png"

sleep 2

# ── 2. GZ Sim window ─────────────────────────────────────────────────────────
echo "[capture] 2/5 GZ Sim window…"
screenshot_window "Gazebo" "$OUT/02_gazebo_simulation.png" || \
screenshot_window "gz"     "$OUT/02_gazebo_simulation.png" || \
import -window root "$OUT/02_gazebo_simulation.png"

sleep 1

# ── 3. RViz window ───────────────────────────────────────────────────────────
echo "[capture] 3/5 RViz window…"
screenshot_window "RViz"  "$OUT/03_rviz_overview.png" || \
screenshot_window "rviz2" "$OUT/03_rviz_overview.png" || \
import -window root "$OUT/03_rviz_overview.png"

sleep 1

# ── 4. Capture vessel detection image from ROS2 topic ────────────────────────
echo "[capture] 4/5 Vessel detection image from /vessel_image topic…"
source /opt/ros/jazzy/setup.bash 2>/dev/null
source "$REPO/ros2_ws/install/setup.bash" 2>/dev/null

python3 - << 'PYEOF'
import sys, os
sys.path.insert(0, '/opt/ros/jazzy/lib/python3.12/dist-packages')
repo = '/home/gokul/github_repos/cyient'
out  = f'{repo}/results/screenshots'

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2, time

class OneShot(Node):
    def __init__(self):
        super().__init__('screenshot_node')
        self.bridge = CvBridge()
        self.saved  = False
        self.create_subscription(Image, '/vessel_image', self.cb, 10)
        self.create_subscription(Image, '/image_raw',    self.cb_raw, 10)
    def cb(self, msg):
        if self.saved: return
        img = self.bridge.imgmsg_to_cv2(msg, 'bgr8')
        path = f'{out}/04_vessel_detection.png'
        cv2.imwrite(path, img)
        print(f'[capture] ✅  {path}')
        self.saved = True
    def cb_raw(self, msg):
        if self.saved: return
        img = self.bridge.imgmsg_to_cv2(msg, 'bgr8')
        path = f'{out}/04_raw_image.png'
        cv2.imwrite(path, img)
        print(f'[capture] ✅  {path}  (raw, no detection yet)')

rclpy.init()
node = OneShot()
deadline = time.time() + 10
while rclpy.ok() and time.time() < deadline and not node.saved:
    rclpy.spin_once(node, timeout_sec=0.5)
node.destroy_node()
rclpy.shutdown()
PYEOF

# ── 5. Capture topic data as text ─────────────────────────────────────────────
echo "[capture] 5/5 Topic data snapshots…"
source /opt/ros/jazzy/setup.bash 2>/dev/null
source "$REPO/ros2_ws/install/setup.bash" 2>/dev/null

{
  echo "=== /vessel_detection ==="
  timeout 5 ros2 topic echo /vessel_detection --once 2>/dev/null || echo "(no message yet)"
  echo ""
  echo "=== /target_pose ==="
  timeout 5 ros2 topic echo /target_pose --once 2>/dev/null || echo "(no message yet)"
  echo ""
  echo "=== /arm_status ==="
  timeout 5 ros2 topic echo /arm_status --once 2>/dev/null || echo "(no message yet)"
  echo ""
  echo "=== Active nodes ==="
  ros2 node list 2>/dev/null || echo "(ros2 not ready)"
} | tee "$OUT/05_topic_snapshot.txt"

echo ""
echo "[capture] Done. Screenshots saved to $OUT/"
ls -lh "$OUT/"

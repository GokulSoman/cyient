# Guide 03 — Vessel Perception Pipeline

## Overview

The perception pipeline detects the vessel (artery) in an ultrasound image and outputs the bounding box and centroid for downstream target planning.

**Approach chosen:** Classical OpenCV pipeline (no training required)  
**Why:** Fast to implement, fully explainable, works well on ultrasound images where vessels appear as dark circular/oval regions surrounded by brighter tissue.

---

## Algorithm: Classical CV Vessel Detection

```
Input image (grayscale ultrasound)
    │
    ▼
1. Grayscale conversion        (if input is BGR/RGB)
    │
    ▼
2. Gaussian blur               (removes speckle noise)
    │
    ▼
3. Adaptive thresholding       (handles non-uniform illumination)
    │
    ▼
4. Morphological close         (fills holes in vessel cross-section)
    │
    ▼
5. Contour detection           (findContours)
    │
    ▼
6. Filter contours             (area, circularity)
    │
    ▼
7. Select best candidate       (largest area OR most circular)
    │
    ▼
Output: bounding box (x,y,w,h), centroid (cx, cy), confidence
```

---

## Implementation Reference

```python
import cv2
import numpy as np

def detect_vessel(image: np.ndarray) -> dict | None:
    """
    Detect vessel in an ultrasound image using classical CV.
    
    Args:
        image: BGR or grayscale numpy array
        
    Returns:
        dict with keys: bbox (x,y,w,h), centroid (cx,cy), confidence, contour
        None if no vessel detected
    """
    # Step 1: Convert to grayscale
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image.copy()
    
    # Step 2: Gaussian blur to reduce speckle noise
    blurred = cv2.GaussianBlur(gray, ksize=(5, 5), sigmaX=1.5)
    
    # Step 3: Adaptive thresholding
    # ADAPTIVE_THRESH_GAUSSIAN_C works better for ultrasound than global threshold
    thresh = cv2.adaptiveThreshold(
        blurred,
        maxValue=255,
        adaptiveMethod=cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        thresholdType=cv2.THRESH_BINARY_INV,  # INV: vessel (dark) → white
        blockSize=15,
        C=4
    )
    
    # Step 4: Morphological close — fills holes, connects vessel cross-section
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    closed = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
    
    # Optional: erode to remove tiny noise blobs
    erode_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    cleaned = cv2.erode(closed, erode_kernel, iterations=1)
    
    # Step 5: Find contours
    contours, _ = cv2.findContours(
        cleaned,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )
    
    if not contours:
        return None
    
    # Step 6: Filter by area (remove tiny blobs)
    min_area = 200   # pixels²
    max_area = gray.shape[0] * gray.shape[1] * 0.5  # not more than 50% of image
    valid = [c for c in contours if min_area < cv2.contourArea(c) < max_area]
    
    if not valid:
        return None
    
    # Step 7: Score and select best contour
    # Score = area × circularity (vessels are roughly circular in cross-section)
    def score(c):
        area = cv2.contourArea(c)
        perimeter = cv2.arcLength(c, True)
        if perimeter == 0:
            return 0
        circularity = 4 * np.pi * area / (perimeter ** 2)  # 1.0 = perfect circle
        return area * circularity  # favor large + round blobs
    
    best = max(valid, key=score)
    
    # Compute bounding box and centroid
    x, y, w, h = cv2.boundingRect(best)
    M = cv2.moments(best)
    if M["m00"] == 0:
        return None
    cx = int(M["m10"] / M["m00"])
    cy = int(M["m01"] / M["m00"])
    
    # Confidence: circularity of selected contour (0–1)
    area = cv2.contourArea(best)
    perimeter = cv2.arcLength(best, True)
    confidence = min(1.0, 4 * np.pi * area / (perimeter ** 2)) if perimeter > 0 else 0.0
    
    return {
        "bbox": (x, y, w, h),
        "centroid": (cx, cy),
        "confidence": float(confidence),
        "contour": best,
        "area": float(area)
    }


def draw_detection(image: np.ndarray, detection: dict) -> np.ndarray:
    """Draw detection overlay on image for visualization."""
    vis = image.copy()
    if len(vis.shape) == 2:
        vis = cv2.cvtColor(vis, cv2.COLOR_GRAY2BGR)
    
    x, y, w, h = detection["bbox"]
    cx, cy = detection["centroid"]
    
    # Bounding box (green)
    cv2.rectangle(vis, (x, y), (x + w, y + h), (0, 255, 0), 2)
    
    # Centroid (red dot)
    cv2.circle(vis, (cx, cy), 5, (0, 0, 255), -1)
    
    # Contour (cyan)
    cv2.drawContours(vis, [detection["contour"]], -1, (255, 255, 0), 1)
    
    # Label
    label = f"Vessel  conf:{detection['confidence']:.2f}"
    cv2.putText(vis, label, (x, y - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
    
    return vis
```

---

## ROS2 Node: `vessel_detector.py`

```python
#!/usr/bin/env python3
"""ROS2 node: subscribes to /image_raw, publishes /vessel_detection and /vessel_image."""

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge

# Custom message (build catheter_sim package first)
from catheter_sim.msg import VesselDetection

import cv2
import numpy as np
from .detector import detect_vessel, draw_detection  # local module


class VesselDetectorNode(Node):

    def __init__(self):
        super().__init__('vessel_detector')
        
        # Declare parameters
        self.declare_parameter('min_area', 200)
        self.declare_parameter('debug_publish', True)
        
        self.bridge = CvBridge()
        
        # Subscribers
        self.sub = self.create_subscription(
            Image, '/image_raw', self.image_callback, 10)
        
        # Publishers
        self.pub_detection = self.create_publisher(
            VesselDetection, '/vessel_detection', 10)
        self.pub_image = self.create_publisher(
            Image, '/vessel_image', 10)
        
        self.get_logger().info('VesselDetector node started')

    def image_callback(self, msg: Image):
        try:
            cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        except Exception as e:
            self.get_logger().error(f'cv_bridge error: {e}')
            return
        
        result = detect_vessel(cv_image)
        
        if result is None:
            self.get_logger().debug('No vessel detected in frame')
            return
        
        # Publish VesselDetection message
        det_msg = VesselDetection()
        det_msg.header = msg.header
        det_msg.header.stamp = self.get_clock().now().to_msg()
        det_msg.center_x = float(result['centroid'][0])
        det_msg.center_y = float(result['centroid'][1])
        det_msg.bbox_x   = float(result['bbox'][0])
        det_msg.bbox_y   = float(result['bbox'][1])
        det_msg.bbox_w   = float(result['bbox'][2])
        det_msg.bbox_h   = float(result['bbox'][3])
        det_msg.confidence = result['confidence']
        det_msg.label    = 'femoral_artery_proxy'
        
        self.pub_detection.publish(det_msg)
        
        # Publish debug image
        if self.get_parameter('debug_publish').value:
            vis = draw_detection(cv_image, result)
            vis_msg = self.bridge.cv2_to_imgmsg(vis, encoding='bgr8')
            vis_msg.header = msg.header
            self.pub_image.publish(vis_msg)
        
        self.get_logger().debug(
            f'Detection: centroid=({result["centroid"][0]:.0f}, {result["centroid"][1]:.0f})'
            f' conf={result["confidence"]:.2f}'
        )


def main(args=None):
    rclpy.init(args=args)
    node = VesselDetectorNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
```

---

## Image Publisher Node: `image_publisher.py`

```python
#!/usr/bin/env python3
"""Publishes Mus-V sample images as /image_raw for testing the perception pipeline."""

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2
import os
import glob


class ImagePublisherNode(Node):

    def __init__(self):
        super().__init__('image_publisher')
        self.declare_parameter('image_dir', 'data/sample_images/images')
        self.declare_parameter('publish_rate', 1.0)  # Hz
        
        self.bridge = CvBridge()
        self.pub = self.create_publisher(Image, '/image_raw', 10)
        
        img_dir = self.get_parameter('image_dir').value
        self.images = sorted(glob.glob(os.path.join(img_dir, '*.png')) +
                             glob.glob(os.path.join(img_dir, '*.jpg')))
        
        if not self.images:
            self.get_logger().error(f'No images found in {img_dir}')
            return
        
        self.get_logger().info(f'Found {len(self.images)} images in {img_dir}')
        self.idx = 0
        
        rate = self.get_parameter('publish_rate').value
        self.timer = self.create_timer(1.0 / rate, self.timer_callback)

    def timer_callback(self):
        path = self.images[self.idx % len(self.images)]
        self.idx += 1
        
        img = cv2.imread(path)
        if img is None:
            self.get_logger().warn(f'Failed to read {path}')
            return
        
        msg = self.bridge.cv2_to_imgmsg(img, encoding='bgr8')
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = 'camera_frame'
        self.pub.publish(msg)
        self.get_logger().debug(f'Published: {os.path.basename(path)}')


def main(args=None):
    rclpy.init(args=args)
    node = ImagePublisherNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()
```

---

## Testing the Pipeline Standalone

```bash
# Activate conda env and source ROS2
conda activate cyient
source /opt/ros/jazzy/setup.bash
source ros2_ws/install/setup.bash

# Terminal 1: Run image publisher
ros2 run catheter_sim image_publisher

# Terminal 2: Run vessel detector
ros2 run catheter_sim vessel_detector

# Terminal 3: View detections
ros2 topic echo /vessel_detection

# Terminal 4: View annotated image
ros2 run rqt_image_view rqt_image_view /vessel_image
```

---

## Parameter Tuning

If detection is poor on Mus-V images, tune these parameters:

| Parameter | Default | Effect |
|-----------|---------|--------|
| `GaussianBlur ksize` | (5,5) | Larger → more smoothing, less noise |
| `adaptiveThreshold blockSize` | 15 | Larger → more global comparison |
| `adaptiveThreshold C` | 4 | Increase if too much noise passes |
| `morph kernel size` | (7,7) | Larger → fills bigger gaps |
| `min_area` | 200 px² | Increase to ignore small blobs |

For verification, run the standalone test:
```bash
conda activate cyient
python3 -c "
import cv2
from perception.detect_vessels import detect_vessel, draw_detection
img = cv2.imread('data/sample_images/images/0001.png')
result = detect_vessel(img)
print('Detection:', result)
vis = draw_detection(img, result)
cv2.imwrite('results/screenshots/detection_test.png', vis)
print('Saved to results/screenshots/detection_test.png')
"
```

---

## Clinical Context: Why Classical CV for Vessel Detection

For the real system, a deep learning approach (U-Net, YOLO) would be preferred because:
- More robust to anatomical variation across patients
- Can distinguish artery from vein by pulsatility (temporal feature)
- Can handle different probe orientations

However, for this prototype:
- Classical CV is **fully explainable** — important for clinical safety review
- No GPU / training time required
- Demonstrates the **pipeline structure** clearly
- Mus-V masks can validate detection quality quantitatively (IoU)

**Future improvement:** Replace `detect_vessel()` with a trained U-Net that takes the Mus-V masks as ground truth. The ROS2 node interface stays identical.

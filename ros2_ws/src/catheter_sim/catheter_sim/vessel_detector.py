#!/usr/bin/python3
"""
ROS2 node: vessel_detector

Subscribes to /image_raw (sensor_msgs/Image), runs a classical OpenCV pipeline
to detect vessel cross-sections, and publishes:
  - /vessel_detection  (catheter_sim/VesselDetection)  — bounding box + centroid
  - /vessel_image      (sensor_msgs/Image)              — annotated debug image

Algorithm
---------
1. BGR → grayscale
2. Gaussian blur 5×5 (σ=1.5) — reduces speckle noise
3. Adaptive threshold (GAUSSIAN_C, THRESH_BINARY_INV, blockSize=15, C=4)
4. Morphological close with 7×7 ellipse kernel
5. findContours (RETR_EXTERNAL)
6. Filter contours: area > 200 px²
7. Score = area × circularity; pick best candidate
8. Compute bounding rect + centroid via moments
"""

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2
import numpy as np


class VesselDetectorNode(Node):

    def __init__(self):
        super().__init__('vessel_detector')

        self.declare_parameter('min_area', 200)
        self.declare_parameter('debug_publish', True)

        self.bridge = CvBridge()

        # Subscribers
        self.sub = self.create_subscription(
            Image, '/image_raw', self.image_callback, 10
        )

        # Publishers
        self.pub_detection = None  # lazy init after msg type is available
        self.pub_image = self.create_publisher(Image, '/vessel_image', 10)

        self.get_logger().info('VesselDetector node started — waiting for messages')

    # ------------------------------------------------------------------
    # Lazy publisher for custom message (avoids import-time issues)
    # ------------------------------------------------------------------

    def _ensure_detection_publisher(self):
        if self.pub_detection is None:
            from catheter_sim.msg import VesselDetection
            self._VesselDetection = VesselDetection
            self.pub_detection = self.create_publisher(
                VesselDetection, '/vessel_detection', 10
            )

    # ------------------------------------------------------------------
    # OpenCV detection pipeline
    # ------------------------------------------------------------------

    def _detect_vessel(self, bgr_image: np.ndarray):
        """
        Returns dict with bbox, centroid, confidence, contour — or None.
        """
        min_area = self.get_parameter('min_area').value
        h, w = bgr_image.shape[:2]
        max_area = h * w * 0.5  # not more than 50 % of image

        # Step 1: Grayscale
        gray = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2GRAY)

        # Step 2: Gaussian blur
        blurred = cv2.GaussianBlur(gray, (5, 5), sigmaX=1.5)

        # Step 3: Adaptive threshold
        thresh = cv2.adaptiveThreshold(
            blurred,
            maxValue=255,
            adaptiveMethod=cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            thresholdType=cv2.THRESH_BINARY_INV,
            blockSize=15,
            C=4,
        )

        # Step 4: Morphological close
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
        closed = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)

        # Step 5: Find contours
        contours, _ = cv2.findContours(
            closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        if not contours:
            return None

        # Step 6: Filter by area
        valid = [
            c for c in contours
            if min_area < cv2.contourArea(c) < max_area
        ]
        if not valid:
            return None

        # Step 7: Score = area × circularity; pick best
        def _score(c):
            area = cv2.contourArea(c)
            perim = cv2.arcLength(c, True)
            if perim == 0:
                return 0.0
            circ = 4.0 * np.pi * area / (perim ** 2)
            return area * circ

        best = max(valid, key=_score)
        area = cv2.contourArea(best)
        perim = cv2.arcLength(best, True)
        confidence = (
            float(min(1.0, 4.0 * np.pi * area / (perim ** 2)))
            if perim > 0 else 0.0
        )

        # Bounding box + centroid via moments
        x, y, bw, bh = cv2.boundingRect(best)
        M = cv2.moments(best)
        if M['m00'] == 0:
            return None
        cx = int(M['m10'] / M['m00'])
        cy = int(M['m01'] / M['m00'])

        return {
            'bbox': (x, y, bw, bh),
            'centroid': (cx, cy),
            'confidence': confidence,
            'contour': best,
        }

    def _draw_detection(self, bgr_image: np.ndarray, det: dict) -> np.ndarray:
        vis = bgr_image.copy()
        x, y, bw, bh = det['bbox']
        cx, cy = det['centroid']

        # Bounding box — green
        cv2.rectangle(vis, (x, y), (x + bw, y + bh), (0, 255, 0), 2)
        # Centroid — red dot
        cv2.circle(vis, (cx, cy), 5, (0, 0, 255), -1)
        # Contour — cyan
        cv2.drawContours(vis, [det['contour']], -1, (255, 255, 0), 1)
        # Label
        label = f'Vessel conf:{det["confidence"]:.2f}'
        cv2.putText(
            vis, label, (x, max(y - 8, 10)),
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1
        )
        return vis

    # ------------------------------------------------------------------
    # ROS2 callback
    # ------------------------------------------------------------------

    def image_callback(self, msg: Image):
        self._ensure_detection_publisher()

        try:
            cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        except Exception as e:
            self.get_logger().error(f'cv_bridge error: {e}')
            return

        result = self._detect_vessel(cv_image)

        if result is None:
            self.get_logger().debug('No vessel detected in frame')
            return

        # Build and publish VesselDetection message
        det_msg = self._VesselDetection()
        det_msg.header = msg.header
        det_msg.header.stamp = self.get_clock().now().to_msg()
        det_msg.center_x = float(result['centroid'][0])
        det_msg.center_y = float(result['centroid'][1])
        det_msg.bbox_x = float(result['bbox'][0])
        det_msg.bbox_y = float(result['bbox'][1])
        det_msg.bbox_w = float(result['bbox'][2])
        det_msg.bbox_h = float(result['bbox'][3])
        det_msg.confidence = float(result['confidence'])
        det_msg.label = 'femoral_artery_proxy'
        self.pub_detection.publish(det_msg)

        # Publish annotated image
        if self.get_parameter('debug_publish').value:
            vis = self._draw_detection(cv_image, result)
            vis_msg = self.bridge.cv2_to_imgmsg(vis, encoding='bgr8')
            vis_msg.header = msg.header
            self.pub_image.publish(vis_msg)

        self.get_logger().debug(
            f'Detection: centroid=({result["centroid"][0]}, {result["centroid"][1]})'
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

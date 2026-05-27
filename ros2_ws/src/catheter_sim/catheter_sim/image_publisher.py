#!/usr/bin/python3
"""
ROS2 node: loads images from data/sample_images/ and publishes them at
/image_raw (sensor_msgs/Image, encoding bgr8) at a configurable rate.

If no images are found in the directory, a synthetic test image is generated
(white ellipse on dark background) so the perception pipeline can run even
without the Mus-V dataset.

Parameters
----------
image_dir    : str   — directory containing PNG/JPG images
                       default: /home/gokul/github_repos/cyient/data/sample_images/
publish_rate : float — Hz (default 1.0)
"""

import os
import glob

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2
import numpy as np


class ImagePublisherNode(Node):

    def __init__(self):
        super().__init__('image_publisher')

        self.declare_parameter(
            'image_dir',
            '/home/gokul/github_repos/cyient/data/sample_images/'
        )
        self.declare_parameter('publish_rate', 1.0)

        self.bridge = CvBridge()
        self.pub = self.create_publisher(Image, '/image_raw', 10)

        img_dir = self.get_parameter('image_dir').value

        # Collect PNG and JPG images
        self.images = sorted(
            glob.glob(os.path.join(img_dir, '*.png')) +
            glob.glob(os.path.join(img_dir, '*.jpg')) +
            glob.glob(os.path.join(img_dir, '*.jpeg'))
        )

        if not self.images:
            self.get_logger().warn(
                f'No images found in {img_dir} — using synthetic test image.'
            )
            self._synthetic = True
        else:
            self._synthetic = False
            self.get_logger().info(
                f'Found {len(self.images)} images in {img_dir}'
            )

        self.idx = 0

        rate = self.get_parameter('publish_rate').value
        self.timer = self.create_timer(1.0 / rate, self.timer_callback)
        self.get_logger().info('ImagePublisher node started')

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _make_synthetic_image(self) -> np.ndarray:
        """Return a 480×640 BGR synthetic ultrasound-like image."""
        img = np.zeros((480, 640, 3), dtype=np.uint8)
        # Add some grey background texture
        noise = np.random.randint(10, 40, (480, 640), dtype=np.uint8)
        img[:, :, 0] = noise
        img[:, :, 1] = noise
        img[:, :, 2] = noise
        # Draw a bright ellipse to simulate a vessel cross-section
        cv2.ellipse(img, (320, 240), (60, 45), 0, 0, 360, (200, 200, 200), -1)
        # Add a second smaller ellipse (e.g. femoral vein nearby)
        cv2.ellipse(img, (390, 255), (35, 28), 0, 0, 360, (160, 160, 160), -1)
        # Dark lumen inside vessel
        cv2.ellipse(img, (320, 240), (30, 22), 0, 0, 360, (20, 20, 20), -1)
        return img

    # ------------------------------------------------------------------
    # Timer callback
    # ------------------------------------------------------------------

    def timer_callback(self):
        if self._synthetic:
            img = self._make_synthetic_image()
        else:
            path = self.images[self.idx % len(self.images)]
            self.idx += 1
            img = cv2.imread(path)
            if img is None:
                self.get_logger().warn(f'Failed to read {path}')
                return
            self.get_logger().debug(f'Published: {os.path.basename(path)}')

        msg = self.bridge.cv2_to_imgmsg(img, encoding='bgr8')
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = 'camera_frame'
        self.pub.publish(msg)


def main(args=None):
    rclpy.init(args=args)
    node = ImagePublisherNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()

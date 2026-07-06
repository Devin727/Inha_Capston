#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import cv2
import numpy as np
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image, CompressedImage
from cv_bridge import CvBridge
from rclpy.qos import QoSProfile, ReliabilityPolicy

class ImageCompressorNode(Node):
    def __init__(self):
        super().__init__('image_compressor_node')

        self.bridge = CvBridge()



        self.subscription = self.create_subscription(
            Image,
            '/camera/color/image_raw',
            self.image_callback,
            1)

        self.publisher = self.create_publisher(
            CompressedImage,
            '/camera/color/image_raw/compressed',
            10)

        self.get_logger().info('Image Compressor Node 시작')

    def image_callback(self, msg):
        try:
            cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')

            comp_image = self.bridge.cv2_to_compressed_imgmsg(cv_image, dst_format='jpeg')

            self.publisher.publish(comp_image)

        except Exception as e:
            self.get_logger().error(f'오류 발생: {str(e)}')


def main(args=None):
    rclpy.init(args=args)
    node = ImageCompressorNode()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info('노드 종료')
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
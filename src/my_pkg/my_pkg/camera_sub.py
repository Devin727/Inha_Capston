import cv2
import numpy as np
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import CompressedImage
from geometry_msgs.msg import Twist
from cv_bridge import CvBridge

MAX_STEER = 0.42

def clamp_steer(value):
    return max(-MAX_STEER, min(MAX_STEER, value))

class CameraBrightnessStop(Node):
    def __init__(self):
        super().__init__('camera_brightness_stop')
        self.bridge = CvBridge()
        self.cmd_pub = self.create_publisher(Twist, '/cmd_vel', 10)
        self.image_sub = self.create_subscription(CompressedImage, '/camera/color/image_raw/compressed', self.image_callback, 10)

    def image_callback(self, msg):
        frame = self.bridge.compressed_imgmsg_to_cv2(msg)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        brightness = gray.mean()



        cmd = Twist()

        if brightness < 50.0:
            cmd.linear.x = 0.0
            cmd.angular.z = clamp_steer(0.0)
            self.get_logger().warn('Image is too dark. Stop.')
        else:
            cmd.linear.x = 0.12
            cmd.angular.z = clamp_steer(0.0)
            self.get_logger().info('Brightness is OK. Move forward.')

        self.cmd_pub.publish(cmd)
        cv2.putText(gray, f'brightness: {brightness:.1f}', (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, 255, 2)
        cv2.imshow('brightness_check', gray)
        cv2.waitKey(1)

    def destroy_node(self):
        cv2.destroyAllWindows()
        super().destroy_node()

def main(args=None):
    rclpy.init(args=args)
    node = CameraBrightnessStop()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()
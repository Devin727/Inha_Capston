#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import cv2
import numpy as np
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import CompressedImage, Image, LaserScan
from geometry_msgs.msg import Twist
from rclpy.qos import QoSProfile, ReliabilityPolicy
from cv_bridge import CvBridge
from enum import Enum

# ===== 차선 추종 파라미터 =====
MAX_STEER        = 0.42
BASE_SPEED       = 0.6
MIN_SPEED        = 0.18
KP               = 0.015
LOST_BIAS        = 0.25
STEER_THRESHOLD  = 0.25

# ===== 신호등 파라미터 =====
MIN_BLOB_AREA = 50
STABLE_FRAMES = 5


def clamp_steer(value):
    return max(-MAX_STEER, min(MAX_STEER, value))


class Mission(Enum):
    WAIT_SIGNAL   = 0   # 출발 전 신호 대기
    DEFAULT_DRIVE = 1   # 기본 차선 추종 (모든 미션의 복귀 지점)
    DYNAMIC_OBS   = 2   # 동적 장애물 대응


class MissionController(Node):
    def __init__(self):
        super().__init__('mission_controller')

        self.state = Mission.WAIT_SIGNAL

        # ===== 퍼블리셔 =====
        self.cmd_pub    = self.create_publisher(Twist, '/cmd_vel', 10)
        self.bridge     = CvBridge()
        self.bev_pub    = self.create_publisher(Image, '/debug/bev', 10)
        self.binary_pub = self.create_publisher(Image, '/debug/binary', 10)

        # ===== 신호등 상태 변수 =====
        self.green_count = 0

        # ===== 차선 추종 상태 변수 =====
        self.last_waypoint_x = None

        # BEV 워핑 행렬
        src = np.float32([
            [  0, 480],
            [640, 480],
            [500, 250],
            [140, 250],
        ])
        dst = np.float32([
            [100, 480],
            [540, 480],
            [540,   0],
            [100,   0],
        ])
        self.M     = cv2.getPerspectiveTransform(src, dst)
        self.BEV_W = 640
        self.BEV_H = 480

        # ===== 구독 =====
        qos = QoSProfile(depth=1, reliability=ReliabilityPolicy.BEST_EFFORT)
        self.image_sub = self.create_subscription(
            CompressedImage,
            '/camera/color/image_raw/compressed',
            self.image_callback,
            qos)

        self.scan_sub = self.create_subscription(
            LaserScan,
            '/scan',
            self.scan_callback,
            qos)

        self.latest_scan = None

        self.get_logger().info('Mission Controller 시작 — WAIT_SIGNAL')

    def scan_callback(self, msg):
        self.latest_scan = msg

    def image_callback(self, msg):
        np_arr = np.frombuffer(msg.data, np.uint8)
        frame  = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
        if frame is None:
            return

        if self.state == Mission.WAIT_SIGNAL:
            self.handle_wait_signal(frame)
        elif self.state == Mission.DEFAULT_DRIVE:
            self.handle_default_drive(frame)
        elif self.state == Mission.DYNAMIC_OBS:
            self.handle_dynamic_obs(frame)

    # ===================================================
    # 신호등 대기
    # ===================================================
    def handle_wait_signal(self, frame):
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        green_lower = np.array([40, 100, 100])
        green_upper = np.array([80, 255, 255])
        mask_green  = cv2.inRange(hsv, green_lower, green_upper)
        green_area  = cv2.countNonZero(mask_green)

        if green_area > MIN_BLOB_AREA:
            self.green_count += 1
            self.get_logger().info(f'초록불 감지 ({self.green_count}/{STABLE_FRAMES}) area:{green_area}')
        else:
            self.green_count = 0
            self.get_logger().warn('초록불 미검출 — 대기 중')

        if self.green_count >= STABLE_FRAMES:
            self.get_logger().info('초록불 확정 — 기본 주행 시작')
            self.state = Mission.DEFAULT_DRIVE
            return

        self.publish_stop()

    # ===================================================
    # 기본 차선 추종 (모든 미션의 복귀 지점)
    # ===================================================
    def handle_default_drive(self, frame):
        bev = cv2.warpPerspective(frame, self.M, (self.BEV_W, self.BEV_H))

        h, w         = bev.shape[:2]
        roi_top      = int(h * 0.50)
        roi          = bev[roi_top:h, :]
        roi_h, roi_w = roi.shape[:2]

        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        _, binary = cv2.threshold(gray, 180, 255, cv2.THRESH_BINARY)

        left_region  = binary[:, :roi_w // 2]
        right_region = binary[:, roi_w // 2:]

        left_cx  = self._get_centroid_x(left_region)
        right_cx = self._get_centroid_x(right_region)

        cmd  = Twist()
        bias = 0.0

        if left_cx is not None and right_cx is not None:
            right_cx_abs         = right_cx + roi_w // 2
            waypoint_x           = (left_cx + right_cx_abs) / 2.0
            self.last_waypoint_x = waypoint_x

        elif left_cx is None and right_cx is not None:
            right_cx_abs         = right_cx + roi_w // 2
            waypoint_x           = right_cx_abs * 0.3
            self.last_waypoint_x = waypoint_x
            bias = LOST_BIAS
            self.get_logger().warn(f'왼쪽 소실 — 강제 좌편향 wp:{waypoint_x:.1f}')

        elif left_cx is not None and right_cx is None:
            waypoint_x           = left_cx + (roi_w - left_cx) * 0.7
            self.last_waypoint_x = waypoint_x
            bias = -LOST_BIAS
            self.get_logger().warn(f'오른쪽 소실 — 강제 우편향 wp:{waypoint_x:.1f}')

        elif self.last_waypoint_x is not None:
            waypoint_x = self.last_waypoint_x
            self.get_logger().warn('전체 소실 — 마지막 웨이포인트 유지')

        else:
            self.get_logger().warn('차선 미검출 — 정지')
            self.publish_stop()
            self._publish_debug(bev, binary, None, left_cx, right_cx, roi_w, roi_h, roi_top)
            return

        error = waypoint_x - (roi_w / 2.0)
        steer = clamp_steer(-error * KP + bias)

        if abs(steer) < STEER_THRESHOLD:
            speed = BASE_SPEED
        else:
            steer_ratio = (abs(steer) - STEER_THRESHOLD) / (MAX_STEER - STEER_THRESHOLD)
            speed = BASE_SPEED * (1.0 - steer_ratio) ** 2
            speed = max(MIN_SPEED, speed)

        cmd.linear.x  = speed
        cmd.angular.z = steer
        self.cmd_pub.publish(cmd)

        self.get_logger().info(
            f'[DEFAULT_DRIVE] err: {error:.1f} | bias: {bias:.2f} | '
            f'steer: {steer:.3f} | speed: {speed:.3f}')

        self._publish_debug(bev, binary, waypoint_x, left_cx, right_cx, roi_w, roi_h, roi_top)

        # ===== 다음 미션 전환 조건 (TODO) =====
        # if self.detect_dynamic_obs_zone():
        #     self.state = Mission.DYNAMIC_OBS

    # ===================================================
    # 동적 장애물 (추후 구현)
    # ===================================================
    def handle_dynamic_obs(self, frame):
        # TODO: self.latest_scan 기반 장애물 검출 + 정지/재출발
        # 통과 완료 시:
        # self.state = Mission.DEFAULT_DRIVE
        self.publish_stop()

    # ===================================================
    # 유틸
    # ===================================================
    def publish_stop(self):
        cmd = Twist()
        cmd.linear.x  = 0.0
        cmd.angular.z = 0.0
        self.cmd_pub.publish(cmd)

    def _get_centroid_x(self, region):
        coords = cv2.findNonZero(region)
        if coords is None or len(coords) < 50:
            return None
        return float(np.mean(coords[:, 0, 0]))

    def _publish_debug(self, bev, binary, waypoint_x, left_cx, right_cx, roi_w, roi_h, roi_top):
        debug_bev = bev.copy()
        cv2.rectangle(debug_bev, (0, roi_top), (roi_w, roi_top + roi_h), (255, 100, 0), 2)

        if left_cx is not None:
            cv2.circle(debug_bev, (int(left_cx), roi_top + roi_h // 2), 8, (255, 0, 0), -1)
        if right_cx is not None:
            cx = int(right_cx + roi_w // 2)
            cv2.circle(debug_bev, (cx, roi_top + roi_h // 2), 8, (0, 0, 255), -1)
        if waypoint_x is not None:
            cv2.circle(debug_bev, (int(waypoint_x), roi_top + roi_h // 2), 10, (0, 255, 0), -1)
            cv2.line(debug_bev, (int(waypoint_x), roi_top), (int(waypoint_x), roi_top + roi_h), (0, 255, 0), 2)
        cv2.line(debug_bev, (roi_w // 2, roi_top), (roi_w // 2, roi_top + roi_h), (100, 100, 100), 1)

        self.bev_pub.publish(self.bridge.cv2_to_imgmsg(debug_bev, 'bgr8'))

        binary_full = np.zeros((self.BEV_H, self.BEV_W), dtype=np.uint8)
        binary_full[roi_top:roi_top + roi_h, :] = binary
        self.binary_pub.publish(self.bridge.cv2_to_imgmsg(binary_full, 'mono8'))


def main(args=None):
    rclpy.init(args=args)
    node = MissionController()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info('노드 종료')
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
#!/usr/bin/env python3
"""
roundabout_node.py
회전교차로 미션 노드

State:
  ROUNDABOUT_WAIT  : LiDAR로 교차로 내 차량 감지 → 정지 대기
  ROUNDABOUT_ENTER : 차량 없으면 진입, 오른쪽 차선만 추종
                     노란 정지선 감지됐다가 사라지면 → 종료
"""

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
from sensor_msgs.msg import CompressedImage, LaserScan
from geometry_msgs.msg import Twist

import cv2
import numpy as np
from enum import Enum
import math


# ─── 파라미터 ───────────────────────────────────────────────
# LiDAR 감시 구간 (왼쪽~정면: +6도 ~ +100도)
WATCH_IDX_START = 113   # -15도 (정면에서 약간 오른쪽까지 포함)
WATCH_IDX_END   = 228   # +90도
WATCH_DIST      = 1.5   # m 이내 물체 = 차량으로 판단
WATCH_MIN_PTS   = 3     # 최소 감지 포인트 수
CLEAR_FRAMES    = 8     # 연속 N프레임 미감지 시 진입

# BEV 워핑 (combined_node 와 동일)
BEV_SRC = np.float32([(0,480),(640,480),(500,250),(140,250)])
BEV_DST = np.float32([(100,480),(540,480),(540,0),(100,0)])

# 차선 추종 (오른쪽 차선만)
KP            = 0.015
MAX_STEER     = 0.42
BASE_SPEED    = 0.4     # 교차로 안은 살짝 느리게
MIN_SPEED     = 0.20
STEER_THRESH  = 0.25
RIGHT_OFFSET  = 120     # 오른쪽 차선에서 왼쪽으로 얼마나 offset할지 (픽셀)
WHITE_THRESH  = 180

# 노란 정지선 HSV
YELLOW_LOW  = np.array([20, 100, 100])
YELLOW_HIGH = np.array([35, 255, 255])
YELLOW_MIN_AREA   = 3000   # 정지선 최소 면적
YELLOW_MIN_ASPECT = 3.0    # 가로/세로 비율 (가로로 긴 것만)
YELLOW_ROI_TOP    = 0.6    # 화면 하단 40%만 봄


class State(Enum):
    ROUNDABOUT_WAIT  = 0
    ROUNDABOUT_ENTER = 1
    DONE             = 2


class RoundaboutNode(Node):
    def __init__(self):
        super().__init__('roundabout_node')

        qos_sensor = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            depth=1
        )

        # Subscribers
        self.create_subscription(
            CompressedImage,
            '/camera/color/image_raw/compressed',
            self.image_cb,
            qos_sensor
        )
        self.create_subscription(
            LaserScan,
            '/scan',
            self.scan_cb,
            qos_sensor
        )

        # Publisher
        self.cmd_pub = self.create_publisher(Twist, '/cmd_vel', 10)

        # Debug publishers
        self.pub_bev   = self.create_publisher(CompressedImage, '/debug/roundabout/bev',   1)
        self.pub_bin   = self.create_publisher(CompressedImage, '/debug/roundabout/binary', 1)
        self.pub_yellow = self.create_publisher(CompressedImage, '/debug/roundabout/yellow', 1)

        # 상태
        self.state       = State.ROUNDABOUT_WAIT
        self.scan_data   = None
        self.clear_count = 0          # 연속 차량 미감지 프레임

        # 노란선 상태
        self.yellow_detected_ever = False   # 한번이라도 감지됐는지
        self.yellow_gone_count    = 0       # 사라진 프레임 수
        self.YELLOW_GONE_FRAMES   = 5       # N프레임 연속 사라지면 전환

        # BEV 변환 행렬
        self.M = cv2.getPerspectiveTransform(BEV_SRC, BEV_DST)

        self.create_timer(0.05, self.control_loop)
        self.get_logger().info('RoundaboutNode started → ROUNDABOUT_WAIT')

    # ── 콜백 ────────────────────────────────────────────────
    def scan_cb(self, msg: LaserScan):
        self.scan_data = msg

    def image_cb(self, msg: CompressedImage):
        self.latest_image = msg

    # ── LiDAR 차량 감지 ─────────────────────────────────────
    def _detect_vehicle(self) -> bool:
        if self.scan_data is None:
            return False
        ranges = self.scan_data.ranges
        count = 0
        for i in range(WATCH_IDX_START, WATCH_IDX_END + 1):
            if i >= len(ranges):
                break
            r = ranges[i]
            if 0.05 < r < WATCH_DIST:
                count += 1
        return count >= WATCH_MIN_PTS

    # ── 카메라 처리 ─────────────────────────────────────────
    def _process_image(self, msg: CompressedImage):
        """BEV 변환 후 오른쪽 차선 무게중심 + 노란선 감지"""
        np_arr = np.frombuffer(msg.data, np.uint8)
        frame  = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
        if frame is None:
            return None, None, False

        h, w = frame.shape[:2]

        # BEV 변환
        bev = cv2.warpPerspective(frame, self.M, (w, h))

        # ROI: 하단 50%
        roi_top = int(h * 0.50)
        roi = bev[roi_top:, :]
        roi_h, roi_w = roi.shape[:2]

        # 흰색 이진화 (차선)
        gray   = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        _, binary = cv2.threshold(gray, WHITE_THRESH, 255, cv2.THRESH_BINARY)

        # ── 오른쪽 차선 무게중심 ──
        right_cx = None
        right_half = binary[:, roi_w//2:]
        M_right = cv2.moments(right_half)
        if M_right['m00'] > 50:
            right_cx = int(M_right['m10'] / M_right['m00']) + roi_w // 2

        # ── 노란선 감지 (하단 ROI) ──
        yellow_roi_top = int(h * YELLOW_ROI_TOP)
        yellow_roi = frame[yellow_roi_top:, :]
        hsv = cv2.cvtColor(yellow_roi, cv2.COLOR_BGR2HSV)
        yellow_mask = cv2.inRange(hsv, YELLOW_LOW, YELLOW_HIGH)
        contours, _ = cv2.findContours(yellow_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        yellow_found = False
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area < YELLOW_MIN_AREA:
                continue
            x, y, cw, ch = cv2.boundingRect(cnt)
            aspect = cw / max(ch, 1)
            if aspect >= YELLOW_MIN_ASPECT:
                yellow_found = True
                break

        # ── 디버그 이미지 퍼블리시 ──
        self._pub_debug(bev, binary, yellow_mask, roi_top)

        return right_cx, roi_w, yellow_found

    def _pub_debug(self, bev, binary, yellow_mask, roi_top):
        def to_compressed(img):
            msg = CompressedImage()
            msg.format = 'jpeg'
            _, buf = cv2.imencode('.jpg', img)
            msg.data = buf.tobytes()
            return msg

        self.pub_bev.publish(to_compressed(bev))

        bin_color = cv2.cvtColor(binary, cv2.COLOR_GRAY2BGR)
        self.pub_bin.publish(to_compressed(bin_color))

        yellow_color = cv2.cvtColor(yellow_mask, cv2.COLOR_GRAY2BGR)
        self.pub_yellow.publish(to_compressed(yellow_color))

    # ── 메인 제어 루프 ───────────────────────────────────────
    def control_loop(self):
        if self.state == State.DONE:
            return

        twist = Twist()

        if self.state == State.ROUNDABOUT_WAIT:
            # 차량 감지 중이면 정지
            if self._detect_vehicle():
                self.clear_count = 0
                self.get_logger().info('차량 감지 → 대기', throttle_duration_sec=1.0)
            else:
                self.clear_count += 1
                self.get_logger().info(
                    f'차량 없음 ({self.clear_count}/{CLEAR_FRAMES})',
                    throttle_duration_sec=0.5
                )
                if self.clear_count >= CLEAR_FRAMES:
                    self.transition_to(State.ROUNDABOUT_ENTER)
                    return

            # 정지
            self.cmd_pub.publish(twist)

        elif self.state == State.ROUNDABOUT_ENTER:
            if not hasattr(self, 'latest_image') or self.latest_image is None:
                self.cmd_pub.publish(twist)
                return

            right_cx, roi_w, yellow_found = self._process_image(self.latest_image)

            # ── 노란선 상태 업데이트 ──
            if yellow_found:
                self.yellow_detected_ever = True
                self.yellow_gone_count = 0
                self.get_logger().info('노란선 감지!', throttle_duration_sec=0.5)
            else:
                if self.yellow_detected_ever:
                    self.yellow_gone_count += 1
                    self.get_logger().info(
                        f'노란선 사라짐 ({self.yellow_gone_count}/{self.YELLOW_GONE_FRAMES})',
                        throttle_duration_sec=0.3
                    )
                    if self.yellow_gone_count >= self.YELLOW_GONE_FRAMES:
                        self.get_logger().info('노란선 통과 완료 → DONE')
                        self.transition_to(State.DONE)
                        return

            # ── 오른쪽 차선 추종 ──
            if right_cx is not None:
                # 오른쪽 차선에서 RIGHT_OFFSET 만큼 왼쪽을 목표로
                wp = right_cx - RIGHT_OFFSET
                error = wp - roi_w / 2
                steer = float(np.clip(-error * KP, -MAX_STEER, MAX_STEER))
            else:
                # 오른쪽 차선 소실 → 완만하게 오른쪽으로
                steer = -0.15
                self.get_logger().warn('오른쪽 차선 소실', throttle_duration_sec=1.0)

            # 속도 조절
            ratio = abs(steer) / MAX_STEER
            if abs(steer) < STEER_THRESH:
                speed = BASE_SPEED
            else:
                speed = max(BASE_SPEED * (1 - ratio) ** 2, MIN_SPEED)

            twist.linear.x  = speed
            twist.angular.z = steer
            self.cmd_pub.publish(twist)

    # ── 상태 전환 ────────────────────────────────────────────
    def transition_to(self, new_state: State):
        self.get_logger().info(f'[State] {self.state.name} → {new_state.name}')
        self.state = new_state

        if new_state == State.ROUNDABOUT_ENTER:
            self.yellow_detected_ever = False
            self.yellow_gone_count    = 0
            self.clear_count          = 0

        elif new_state == State.DONE:
            # 정지 후 노드 종료
            self.cmd_pub.publish(Twist())
            self.get_logger().info('미션 완료! 노드 종료')
            raise SystemExit


def main(args=None):
    rclpy.init(args=args)
    node = RoundaboutNode()
    try:
        rclpy.spin(node)
    except SystemExit:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
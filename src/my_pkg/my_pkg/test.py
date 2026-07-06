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
import time

# ===== BEV =====
BEV_SRC = np.float32([[0,480],[640,480],[500,250],[140,250]])
BEV_DST = np.float32([[100,480],[540,480],[540,0],[100,0]])

# ===== 차선 추종 =====
ROI_TOP       = 0.60
WHITE_THRESH  = 210
LANE_MIN_PX   = 50
LANE_MIN_DIST = 200
BASE_SPEED    = 0.2
MAX_STEER     = 0.42
KP            = 0.005
LOST_OFFSET   = 0.50

# ===== 신호등 =====
MIN_BLOB_AREA = 50
STABLE_FRAMES = 5

# ===== 노란선 =====
YELLOW_LOW         = np.array([18,  30, 100])
YELLOW_HIGH        = np.array([35, 255, 255])
YELLOW_MIN_AREA    = 1500
YELLOW_MIN_ASPECT  = 5.0
YELLOW_GONE_FRAMES = 5
YELLOW_COOLDOWN    = 8.0  # 노란선 감지 후 쿨다운 (초)

# ===== 동적 장애물 정지 (보행자 등) =====
DYN_WATCH_DIST    = 0.3   # 전방 감지 거리 (m)
DYN_WATCH_SEC     = 5.0   # 감시 시간 (초)
DYN_FRONT_START   = 120   # -3도 idx
DYN_FRONT_END     = 124   # +3도 idx

# ===== LiDAR =====
WATCH_IDX_START = 113
WATCH_IDX_END   = 228
WATCH_DIST      = 1.5
WATCH_MIN_PTS   = 3
CLEAR_FRAMES    = 8

# ===== 라바콘 =====
MIN_CONE_AREA      = 100
CONE_SPEED         = 0.2
CONE_CLEAR_FRAMES  = 10
HOLD_DURATION      = 1.5
RETURN_DURATION    = 10.0
RETURN_MIN_TIME    = 7.0
RETURN_BALANCE_MIN = 0.6


def clamp(val, lo, hi):
    return max(lo, min(hi, val))


class Mission(Enum):
    WAIT_SIGNAL      = 0
    DEFAULT_DRIVE    = 1
    YELLOW_STOP      = 2
    ROUNDABOUT_WAIT  = 3
    ROUNDABOUT_DRIVE = 4
    CONE_AVOIDING    = 5
    CONE_HOLD        = 6
    CONE_RETURN      = 7
    PARKING          = 8


class MissionController(Node):
    def __init__(self):
        super().__init__('mission_controller')

        self.state = Mission.WAIT_SIGNAL

        self.cmd_pub    = self.create_publisher(Twist, '/cmd_vel', 10)
        self.bridge     = CvBridge()
        self.pub_result = self.create_publisher(Image, '/debug/result', 1)
        self.pub_binary = self.create_publisher(Image, '/debug/binary', 1)
        self.pub_cone   = self.create_publisher(Image, '/debug/cone',   1)

        qos = QoSProfile(depth=1, reliability=ReliabilityPolicy.BEST_EFFORT)
        self.create_subscription(CompressedImage, '/camera/color/image_raw/compressed', self.image_cb, qos)
        self.create_subscription(LaserScan, '/scan', self.scan_cb, qos)
        self.latest_scan = None

        # BEV
        self.M     = cv2.getPerspectiveTransform(BEV_SRC, BEV_DST)
        self.BEV_W = 640
        self.BEV_H = 480

        # 차선 추종 상태
        self.prev_waypoint = None
        self.prev_left_cx  = None
        self.prev_right_cx = None

        # 신호등
        self.green_count = 0

        # 노란선
        self.yellow_detected_ever = False
        self.yellow_gone_count    = 0
        self.yellow_line_count    = 0
        self.yellow_cooldown_until = 0.0  # 쿨다운 종료 시각

        # 교차로
        self.clear_count      = 0
        self.state_enter_time = time.time()
        self.stop_clear_time  = None   # 전방 물체 사라진 시각

        # 라바콘
        self.cone_clear_count = 0
        self.avoid_dir        = None

        self.create_timer(1.0, self._log_state)
        self.get_logger().info('MissionController 시작 → WAIT_SIGNAL')

    def _log_state(self):
        self.get_logger().info(f'[현재 모드] {self.state.name}')

    def scan_cb(self, msg):
        self.latest_scan = msg

    def image_cb(self, msg):
        np_arr = np.frombuffer(msg.data, np.uint8)
        frame  = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
        if frame is None:
            return

        if   self.state == Mission.WAIT_SIGNAL:
            self.handle_wait_signal(frame)
        elif self.state == Mission.DEFAULT_DRIVE:
            self.handle_default_drive(frame)
        elif self.state == Mission.YELLOW_STOP:
            self.handle_yellow_stop(frame)
        elif self.state == Mission.ROUNDABOUT_WAIT:
            self.handle_roundabout_wait(frame)
        elif self.state == Mission.ROUNDABOUT_DRIVE:
            self.handle_roundabout_drive(frame)
        elif self.state == Mission.CONE_AVOIDING:
            self.handle_cone_avoiding(frame)
        elif self.state == Mission.CONE_HOLD:
            self.handle_cone_hold(frame)
        elif self.state == Mission.CONE_RETURN:
            self.handle_cone_return(frame)
        elif self.state == Mission.PARKING:
            self.handle_parking(frame)

    # ══════════════════════════════════════════════════════
    # WAIT_SIGNAL
    # ══════════════════════════════════════════════════════
    def handle_wait_signal(self, frame):
        hsv  = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv, np.array([40,100,100]), np.array([80,255,255]))
        area = cv2.countNonZero(mask)
        if area > MIN_BLOB_AREA:
            self.green_count += 1
            self.get_logger().info(f'초록불 감지 area:{area} ({self.green_count}/{STABLE_FRAMES})')
        else:
            self.green_count = 0
            self.get_logger().info(f'초록불 없음 area:{area}', throttle_duration_sec=1.0)

        if self.green_count >= STABLE_FRAMES:
            self.transition_to(Mission.DEFAULT_DRIVE)
        else:
            self.publish_stop()

    # ══════════════════════════════════════════════════════
    # DEFAULT_DRIVE
    # ══════════════════════════════════════════════════════
    def handle_default_drive(self, frame):
        self._do_lane_follow(frame)
        if self.yellow_line_count == 0:
            next_state = Mission.YELLOW_STOP
        elif self.yellow_line_count == 1:
            next_state = Mission.ROUNDABOUT_WAIT
        else:
            next_state = Mission.PARKING
        self._check_yellow(frame, next_state)

    # ══════════════════════════════════════════════════════
    # YELLOW_STOP : 노란선 감지 후 DYN_WATCH_SEC초간 동적 장애물 감시
    # 전방 물체 있으면 정지, 없으면 주행, 시간 후 복귀
    # ══════════════════════════════════════════════════════
    def handle_yellow_stop(self, frame):
        elapsed = time.time() - self.state_enter_time

        if elapsed >= DYN_WATCH_SEC:
            self.get_logger().info(f'[DYN_STOP] {DYN_WATCH_SEC}초 종료 → DEFAULT_DRIVE')
            self.transition_to(Mission.DEFAULT_DRIVE)
            return

        detected = False
        if self.latest_scan is not None:
            ranges = self.latest_scan.ranges
            for i in range(DYN_FRONT_START, min(DYN_FRONT_END+1, len(ranges))):
                r = ranges[i]
                if 0.05 < r < DYN_WATCH_DIST:
                    detected = True
                    break

        if detected:
            self.publish_stop()
            self.get_logger().info(f'[DYN_STOP] 장애물 감지 → 정지 ({elapsed:.1f}/{DYN_WATCH_SEC}s)', throttle_duration_sec=0.3)
        else:
            self._do_lane_follow(frame)
            self.get_logger().info(f'[DYN_STOP] 장애물 없음 → 주행 ({elapsed:.1f}/{DYN_WATCH_SEC}s)', throttle_duration_sec=0.3)

    # ══════════════════════════════════════════════════════
    # ROUNDABOUT_WAIT
    # ══════════════════════════════════════════════════════
    def handle_roundabout_wait(self, frame):
        if self._detect_vehicle():
            self.clear_count = 0
            self.publish_stop()
            self.get_logger().info('교차로 차량 감지 → 대기', throttle_duration_sec=1.0)
        else:
            self.clear_count += 1
            self.get_logger().info(f'교차로 진입 가능 ({self.clear_count}/{CLEAR_FRAMES})', throttle_duration_sec=0.5)
            if self.clear_count >= CLEAR_FRAMES:
                self.transition_to(Mission.ROUNDABOUT_DRIVE)

    def _detect_vehicle(self):
        if self.latest_scan is None:
            return False
        ranges = self.latest_scan.ranges
        count  = sum(1 for i in range(WATCH_IDX_START, min(WATCH_IDX_END+1, len(ranges)))
                     if 0.05 < ranges[i] < WATCH_DIST)
        return count >= WATCH_MIN_PTS

    # ══════════════════════════════════════════════════════
    # ROUNDABOUT_DRIVE
    # ══════════════════════════════════════════════════════
    def handle_roundabout_drive(self, frame):
        self._do_lane_follow(frame)
        self._check_yellow(frame, Mission.CONE_AVOIDING)

    # ══════════════════════════════════════════════════════
    # PARKING : 파킹 미션 (임시 - DEFAULT_DRIVE로 대체)
    # ══════════════════════════════════════════════════════
    def handle_parking(self, frame):
        # TODO: 파킹 로직 구현 필요
        # 임시로 DEFAULT_DRIVE로 동작
        self._do_lane_follow(frame)
        self.get_logger().info('[PARKING] 미구현 → 차선 추종으로 대체', throttle_duration_sec=1.0)

    # ══════════════════════════════════════════════════════
    # CONE_AVOIDING
    # ══════════════════════════════════════════════════════
    def handle_cone_avoiding(self, frame):
        h, w = frame.shape[:2]
        roi_top    = int(h * 0.30)
        roi_bottom = int(h * 0.70)
        roi        = frame[roi_top:roi_bottom, :]
        roi_w      = roi.shape[1]

        hsv   = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
        mask  = cv2.inRange(hsv, np.array([20,100,100]), np.array([35,255,255]))
        total = cv2.countNonZero(mask)

        third       = roi_w // 3
        left_cone   = cv2.countNonZero(mask[:, :third])          > MIN_CONE_AREA
        center_cone = cv2.countNonZero(mask[:, third:2*third])   > MIN_CONE_AREA
        right_cone  = cv2.countNonZero(mask[:, 2*third:])        > MIN_CONE_AREA

        cmd = Twist()
        cmd.linear.x = CONE_SPEED
        if left_cone and not right_cone:
            self.avoid_dir = 'right'; cmd.angular.z = -MAX_STEER
        elif right_cone and not left_cone:
            self.avoid_dir = 'left';  cmd.angular.z =  MAX_STEER
        elif center_cone:
            self.avoid_dir = 'left';  cmd.angular.z =  MAX_STEER
        else:
            cmd.angular.z = 0.0
        self.cmd_pub.publish(cmd)

        # 디버그
        debug = frame.copy()
        cv2.rectangle(debug, (0,roi_top), (w,roi_bottom), (255,100,0), 2)
        cv2.line(debug, (third,roi_top), (third,roi_bottom), (0,255,255), 1)
        cv2.line(debug, (2*third,roi_top), (2*third,roi_bottom), (0,255,255), 1)
        cv2.putText(debug, self.state.name, (10,30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,255), 2)
        self.pub_cone.publish(self.bridge.cv2_to_imgmsg(debug, 'bgr8'))

        if total < MIN_CONE_AREA:
            self.cone_clear_count += 1
            if self.cone_clear_count >= CONE_CLEAR_FRAMES:
                self.transition_to(Mission.CONE_HOLD)
        else:
            self.cone_clear_count = 0

    # ══════════════════════════════════════════════════════
    # CONE_HOLD
    # ══════════════════════════════════════════════════════
    def handle_cone_hold(self, frame):
        cmd = Twist()
        cmd.linear.x = CONE_SPEED
        self.cmd_pub.publish(cmd)
        if time.time() - self.state_enter_time >= HOLD_DURATION:
            self.transition_to(Mission.CONE_RETURN)

    # ══════════════════════════════════════════════════════
    # CONE_RETURN
    # ══════════════════════════════════════════════════════
    def handle_cone_return(self, frame):
        h, w = frame.shape[:2]
        cmd  = Twist()
        cmd.linear.x  = CONE_SPEED
        cmd.angular.z = -MAX_STEER if self.avoid_dir == 'left' else MAX_STEER
        self.cmd_pub.publish(cmd)

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        wroi = gray[int(h*0.6):int(h*0.8), :]
        _, wmask = cv2.threshold(wroi, WHITE_THRESH, 255, cv2.THRESH_BINARY)
        ww = wmask.shape[1]
        lw = cv2.countNonZero(wmask[:, :ww//2])
        rw = cv2.countNonZero(wmask[:, ww//2:])
        ratio    = min(lw, rw) / max(lw, rw, 1)
        centered = lw > 50 and rw > 50 and ratio > RETURN_BALANCE_MIN

        elapsed = time.time() - self.state_enter_time
        if (elapsed >= RETURN_MIN_TIME and centered) or elapsed >= RETURN_DURATION:
            self.transition_to(Mission.DEFAULT_DRIVE)

    # ══════════════════════════════════════════════════════
    # 공통 차선 추종
    # ══════════════════════════════════════════════════════
    def _do_lane_follow(self, frame):
        bev = cv2.warpPerspective(frame, self.M, (self.BEV_W, self.BEV_H))
        bev_h, bev_w = bev.shape[:2]
        roi_top      = int(bev_h * ROI_TOP)
        roi          = bev[roi_top:, :]
        roi_h, roi_w = roi.shape[:2]

        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        _, binary = cv2.threshold(gray, WHITE_THRESH, 255, cv2.THRESH_BINARY)

        hsv_roi     = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
        yellow_mask = cv2.inRange(hsv_roi, YELLOW_LOW, YELLOW_HIGH)
        binary      = cv2.bitwise_and(binary, cv2.bitwise_not(yellow_mask))

        binary_full = np.zeros((self.BEV_H, self.BEV_W), dtype=np.uint8)
        binary_full[roi_top:, :] = binary
        self.pub_binary.publish(self.bridge.cv2_to_imgmsg(binary_full, 'mono8'))

        # 절대좌표 무게중심
        left_cx_rel  = self._get_cx(binary[:, :roi_w // 2])
        right_cx_rel = self._get_cx(binary[:, roi_w // 2:])
        left_cx      = left_cx_rel
        right_cx     = right_cx_rel + roi_w // 2 if right_cx_rel is not None else None

        # 튐 방지
        JUMP = roi_w * 0.25
        if self.prev_left_cx is not None and left_cx is not None:
            if abs(left_cx - self.prev_left_cx) > JUMP:
                left_cx = self.prev_left_cx
        if self.prev_right_cx is not None and right_cx is not None:
            if abs(right_cx - self.prev_right_cx) > JUMP:
                right_cx = self.prev_right_cx

        l_pixels = cv2.countNonZero(binary[:, :roi_w // 2])
        r_pixels = cv2.countNonZero(binary[:, roi_w // 2:])

        # 거리 필터
        if left_cx is not None and right_cx is not None:
            if right_cx - left_cx < LANE_MIN_DIST:
                if l_pixels >= r_pixels: right_cx = None
                else:                    left_cx  = None

        # 웨이포인트
        waypoint_x = None
        if left_cx is not None and right_cx is not None:
            waypoint_x = (left_cx + right_cx) / 2.0
        elif left_cx is None and right_cx is not None:
            waypoint_x = right_cx - roi_w * LOST_OFFSET
        elif left_cx is not None and right_cx is None:
            waypoint_x = left_cx  + roi_w * LOST_OFFSET
        elif self.prev_waypoint is not None:
            waypoint_x = self.prev_waypoint
            self.get_logger().warn('차선 소실 → 이전값 유지', throttle_duration_sec=0.5)

        if waypoint_x is not None: self.prev_waypoint = waypoint_x
        # 실제로 잡힌 값만 prev에 저장, 못 잡으면 prev 리셋
        if left_cx  is not None: self.prev_left_cx  = left_cx
        else:                     self.prev_left_cx  = None
        if right_cx is not None: self.prev_right_cx = right_cx
        else:                     self.prev_right_cx = None

        cmd = Twist()
        if waypoint_x is not None:
            error = waypoint_x - roi_w / 2.0
            steer = clamp(-error * KP, -MAX_STEER, MAX_STEER)
            cmd.linear.x  = BASE_SPEED
            cmd.angular.z = steer
            self.get_logger().info(f'[{self.state.name}] err:{error:.1f} steer:{steer:.3f}', throttle_duration_sec=0.3)
        else:
            self.get_logger().warn('웨이포인트 없음 → 정지', throttle_duration_sec=0.5)
        self.cmd_pub.publish(cmd)

        # 디버그
        result = bev.copy()
        cv2.rectangle(result, (0, roi_top), (bev_w, bev_h), (255,100,0), 2)
        cv2.line(result, (roi_w//2, roi_top), (roi_w//2, bev_h), (100,100,100), 1)
        if left_cx  is not None: cv2.circle(result, (int(left_cx),  roi_top+roi_h//2), 8, (255,0,0), -1)
        if right_cx is not None: cv2.circle(result, (int(right_cx), roi_top+roi_h//2), 8, (0,0,255), -1)
        if waypoint_x is not None:
            cv2.circle(result, (int(waypoint_x), roi_top+roi_h//2), 10, (0,255,0), -1)
            cv2.line(result, (int(waypoint_x), roi_top), (int(waypoint_x), bev_h), (0,255,0), 2)
        cv2.putText(result, f'{self.state.name} L:{"O" if left_cx else "X"} R:{"O" if right_cx else "X"}',
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,255), 2)
        self.pub_result.publish(self.bridge.cv2_to_imgmsg(result, 'bgr8'))

    # ══════════════════════════════════════════════════════
    # 노란선 감지
    # ══════════════════════════════════════════════════════
    def _check_yellow(self, frame, next_state):
        # 쿨다운 중이면 감지 스킵
        if time.time() < self.yellow_cooldown_until:
            remaining = self.yellow_cooldown_until - time.time()
            self.get_logger().info(f'[YELLOW] 쿨다운 중 {remaining:.1f}s', throttle_duration_sec=1.0)
            return

        # 원본 이미지 하단 30%에서 감지 (더 가까운 정지선만 감지)
        h, w = frame.shape[:2]
        roi  = frame[int(h*0.7):, :]
        hsv  = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv, YELLOW_LOW, YELLOW_HIGH)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        yellow_found = any(
            cv2.contourArea(c) >= YELLOW_MIN_AREA and
            (lambda b: b[2] / max(b[3],1) >= YELLOW_MIN_ASPECT)(cv2.boundingRect(c))
            for c in contours
        )
        if yellow_found:
            self.yellow_detected_ever = True
            self.yellow_gone_count    = 0
            self.get_logger().info('[YELLOW] 노란선 감지!', throttle_duration_sec=0.5)
        elif self.yellow_detected_ever:
            self.yellow_gone_count += 1
            if self.yellow_gone_count >= YELLOW_GONE_FRAMES:
                self.yellow_line_count    += 1
                self.yellow_cooldown_until = time.time() + YELLOW_COOLDOWN
                self.get_logger().info(f'노란선 {self.yellow_line_count}번째 통과 → {next_state.name} (쿨다운 {YELLOW_COOLDOWN}s)')
                self.transition_to(next_state)

    # ══════════════════════════════════════════════════════
    # 유틸
    # ══════════════════════════════════════════════════════
    def transition_to(self, new_state):
        self.get_logger().info(f'[State] {self.state.name} → {new_state.name}')
        self.state            = new_state
        self.state_enter_time = time.time()
        self.yellow_detected_ever = False
        self.yellow_gone_count    = 0
        if new_state == Mission.YELLOW_STOP:
            self.stop_clear_time = None
        if new_state == Mission.ROUNDABOUT_WAIT:
            self.clear_count = 0
        if new_state == Mission.CONE_AVOIDING:
            self.cone_clear_count = 0
            self.avoid_dir        = None

    def publish_stop(self):
        self.cmd_pub.publish(Twist())

    def _get_cx(self, region):
        coords = cv2.findNonZero(region)
        if coords is None or len(coords) < LANE_MIN_PX:
            return None
        return float(np.mean(coords[:, 0, 0]))


def main(args=None):
    rclpy.init(args=args)
    node = MissionController()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info('노드 종료')
        node.publish_stop()
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
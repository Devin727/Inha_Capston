#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
yellow_debug_node.py
노란선 HSV 디버그 노드
- 화면 하단 ROI에서 HSV 값 실시간 출력
- 현재 설정된 HSV 범위로 마스크 결과 퍼블리시
- 밝은 픽셀 HSV 평균값 출력으로 범위 튜닝 가능
"""

import cv2
import numpy as np
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import CompressedImage, Image
from rclpy.qos import QoSProfile, ReliabilityPolicy
from cv_bridge import CvBridge

# ===== 튜닝 파라미터 =====
YELLOW_LOW  = np.array([15,  30,  80])
YELLOW_HIGH = np.array([40, 255, 255])
ROI_TOP     = 0.7   # 하단 30%만


class YellowDebugNode(Node):
    def __init__(self):
        super().__init__('yellow_debug_node')

        self.bridge = CvBridge()
        self.pub_mask   = self.create_publisher(Image, '/debug/yellow_mask',   1)
        self.pub_result = self.create_publisher(Image, '/debug/yellow_result', 1)

        qos = QoSProfile(depth=1, reliability=ReliabilityPolicy.BEST_EFFORT)
        self.create_subscription(
            CompressedImage,
            '/camera/color/image_raw/compressed',
            self.image_cb,
            qos
        )

        self.get_logger().info('YellowDebugNode 시작')
        self.get_logger().info(f'현재 HSV 범위: {YELLOW_LOW} ~ {YELLOW_HIGH}')
        self.get_logger().info(f'ROI: 하단 {int((1-ROI_TOP)*100)}%')

    def image_cb(self, msg):
        np_arr = np.frombuffer(msg.data, np.uint8)
        frame  = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
        if frame is None:
            return

        h, w = frame.shape[:2]
        roi  = frame[int(h * ROI_TOP):, :]

        # HSV 변환
        hsv  = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)

        # 현재 설정 범위로 마스크
        mask = cv2.inRange(hsv, YELLOW_LOW, YELLOW_HIGH)

        # 밝은 픽셀(V > 80) HSV 평균 출력
        bright_pixels = hsv[gray > 80]
        if len(bright_pixels) > 0:
            h_mean = bright_pixels[:, 0].mean()
            s_mean = bright_pixels[:, 1].mean()
            v_mean = bright_pixels[:, 2].mean()
            self.get_logger().info(
                f'밝은픽셀 HSV 평균 → H:{h_mean:.0f} S:{s_mean:.0f} V:{v_mean:.0f}',
                throttle_duration_sec=0.5
            )

        # 마스크 내 픽셀 HSV 평균 출력
        yellow_pixels = hsv[mask > 0]
        if len(yellow_pixels) > 0:
            h_y = yellow_pixels[:, 0].mean()
            s_y = yellow_pixels[:, 1].mean()
            v_y = yellow_pixels[:, 2].mean()
            area = len(yellow_pixels)
            self.get_logger().info(
                f'[감지] 노란픽셀 {area}개 → H:{h_y:.0f} S:{s_y:.0f} V:{v_y:.0f}',
                throttle_duration_sec=0.3
            )
        else:
            self.get_logger().info('노란픽셀 없음', throttle_duration_sec=0.5)

        # 마스크 시각화
        mask_color = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
        self.pub_mask.publish(self.bridge.cv2_to_imgmsg(mask_color, 'bgr8'))

        # 결과 이미지 (원본 + ROI 표시 + 마스크 오버레이)
        result = frame.copy()
        cv2.rectangle(result, (0, int(h*ROI_TOP)), (w, h), (255,100,0), 2)
        overlay = result[int(h*ROI_TOP):, :].copy()
        overlay[mask > 0] = [0, 255, 255]  # 감지된 부분 노란색으로 표시
        result[int(h*ROI_TOP):, :] = cv2.addWeighted(result[int(h*ROI_TOP):, :], 0.6, overlay, 0.4, 0)
        cv2.putText(result, f'H:{YELLOW_LOW[0]}~{YELLOW_HIGH[0]} S:{YELLOW_LOW[1]}~{YELLOW_HIGH[1]} V:{YELLOW_LOW[2]}~{YELLOW_HIGH[2]}',
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,255), 2)
        cv2.putText(result, f'감지픽셀: {len(yellow_pixels)}',
                    (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0) if len(yellow_pixels) > 0 else (0,0,255), 2)
        self.pub_result.publish(self.bridge.cv2_to_imgmsg(result, 'bgr8'))


def main(args=None):
    rclpy.init(args=args)
    node = YellowDebugNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
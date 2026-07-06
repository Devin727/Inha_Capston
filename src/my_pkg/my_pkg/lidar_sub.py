import rclpy  # ROS2 Python 라이브러리를 가져온다.
from rclpy.node import Node  # ROS2 노드 클래스를 가져온다.
from sensor_msgs.msg import LaserScan  # /scan 메시지 타입을 가져온다.
from geometry_msgs.msg import Twist  # /cmd_vel 메시지 타입을 가져온다.
from rclpy.qos import qos_profile_sensor_data  # 센서 데이터 QoS 프로파일을 가져온다.

MAX_STEER = 0.42  # LIMO Ackermann 모드에서 사용할 angular.z 제한값이다.


def clamp_steer(value):  # 조향 명령이 제한값을 넘지 않게 만드는 함수이다.
    return max(-MAX_STEER, min(MAX_STEER, value))  # -0.42 ~ 0.42 사이로 값을 제한한다.


class LidarObstacleStop(Node):  # LiDAR 장애물 정지 노드이다.
    def __init__(self):  # 노드가 생성될 때 실행된다.
        super().__init__('lidar_obstacle_stop')  # 노드 이름을 설정한다.
        self.cmd_pub = self.create_publisher(Twist, '/cmd_vel', 10)  # 제어 명령 Publisher를 만든다.
        self.scan_sub = self.create_subscription(LaserScan, '/scan', self.scan_callback, qos_profile_sensor_data)  # /scan을 구독한다.

    def scan_callback(self, msg):  # LiDAR 데이터가 들어올 때마다 실행된다.
        front_index = len(msg.ranges) // 2  # ranges 배열의 가운데를 전방으로 가정한다.
        front_distance = msg.ranges[front_index]  # 전방 거리값을 가져온다.

        cmd = Twist()  # 속도 명령 메시지를 만든다.

        if front_distance < 0.5 and front_distance > 0.0:  # 전방 장애물이 0.5m보다 가까우면 위험하다.
            cmd.linear.x = 0.0  # 차량을 정지시킨다.
            cmd.angular.z = clamp_steer(0.0)  # 조향도 0으로 둔다.
            self.get_logger().warn('Obstacle detected. Stop.')  # 정지 상태를 출력한다.
        else:  # 전방이 비어 있으면 주행 가능하다.
            cmd.linear.x = 0.2  # 천천히 전진한다.
            cmd.angular.z = clamp_steer(0.0)  # 직진 조향을 유지한다.
            self.get_logger().info('Path clear. Move forward.')  # 전진 상태를 출력한다.

        self.cmd_pub.publish(cmd)  # /cmd_vel 명령을 발행한다.


def main(args=None):  # 실행 시작 함수이다.
    rclpy.init(args=args)  # ROS2를 초기화한다.
    node = LidarObstacleStop()  # 노드를 만든다.
    rclpy.spin(node)  # 콜백을 계속 기다린다.
    node.destroy_node()  # 노드를 정리한다.
    rclpy.shutdown()  # ROS2를 종료한다.

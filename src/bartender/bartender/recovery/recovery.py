import rclpy
from rclpy.node import Node
from std_msgs.msg import Bool
from geometry_msgs.msg import PoseStamped
import sys

class FailureRecoveryNode(Node):
    def __init__(self):
        super().__init__('failure_recovery_node')
        
        # ---------------------------------------------------------------------
        # [장점 1: 확장성] 
        # 새로운 데이터(폰번호, 메뉴 등)가 필요하면 키(Key)값만 추가하면 됩니다.
        # 고정된 변수가 아닌 구조화된 데이터를 통해 유지보수가 쉬워집니다.
        # ---------------------------------------------------------------------
        self.order_info = {
            "id": 2,
            "name": "이영희",
            "status": "미 수령",
            "target_pose": {"x": 1.5, "y": 0.5, "w": 1.0}
        }
        
        self.is_terminated = False
        self.goal_pub = self.create_publisher(PoseStamped, '/goal_pose', 10)
        
        # '사람 사라짐' 신호 구독
        self.sub_disappeared = self.create_subscription(
            Bool, '/person_disappeared', self.failure_trigger_callback, 10)

        self.get_logger().info(f"⚠️ [{self.order_info['name']}]님 주문 감시 중...")

    def failure_trigger_callback(self, msg):
        """사람이 사라지는 찰나에 실행되는 콜백"""
        if msg.data is True and not self.is_terminated:
            
            # -----------------------------------------------------------------
            # [장점 2: 실시간 연동] 
            # 외부 DB나 API에서 받은 최신 데이터를 self.order_info에 덮어씌우기만 하면
            # 아래 로직은 코드 수정 없이 최신 정보를 바탕으로 즉시 동작합니다.
            # -----------------------------------------------------------------
            if self.order_info["status"] == "미 수령":
                self.get_logger().error(f"🚨 인식 실패: {self.order_info['id']}번 {self.order_info['name']}님 부재.")
                
                # 딕셔너리 내부 값을 변경하여 현재 진행 상태를 실시간 기록
                self.order_info["status"] = "보관대_A로 이동 중"
                
                # [장점 3: 유동성]
                # 타겟이 바뀌어도 동일한 함수(move_to_shelf)에 딕셔너리 좌표값만 던져주면 끝!
                self.move_to_shelf(self.order_info["target_pose"])
                
                self.get_logger().info(f"📦 {self.order_info['name']}님 물품 이동 명령 전송.")
                self.terminate_node()
            else:
                self.get_logger().info(f"✅ {self.order_info['name']}님 정상 수령 확인.")
                self.terminate_node()

    def move_to_shelf(self, pose_data):
        """딕셔너리 좌표 데이터를 기반으로 실제 로봇 이동 명령 발행"""
        goal_msg = PoseStamped()
        goal_msg.header.stamp = self.get_clock().now().to_msg()
        goal_msg.header.frame_id = 'map'
        
        # 딕셔너리에서 꺼내온 좌표값 적용
        goal_msg.pose.position.x = pose_data["x"]
        goal_msg.pose.position.y = pose_data["y"]
        goal_msg.pose.orientation.w = pose_data["w"]
        
        self.goal_pub.publish(goal_msg)
        self.get_logger().info(f"🚀 [좌표 발행] x: {pose_data['x']}, y: {pose_data['y']}")

    def terminate_node(self):
        """임무 완료 후 스스로 노드 파괴 및 프로세스 종료"""
        self.is_terminated = True
        raise SystemExit 

def main(args=None):
    rclpy.init(args=args)
    node = FailureRecoveryNode()
    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, SystemExit):
        node.get_logger().info('Mission Complete. 노드를 종료합니다.')
    finally:
        if rclpy.ok():
            node.destroy_node()
            rclpy.shutdown()

if __name__ == '__main__':
    main()
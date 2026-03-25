#!/usr/bin/env python3
"""
Recipe Node - Action Server for recipe motion control
"""
import time
import rclpy
from rclpy.node import Node
from rclpy.action import ActionServer
from rclpy.action.server import ServerGoalHandle

from bartender_interfaces.action import Motion


class RecipeController(Node):
    def __init__(self):
        super().__init__("recipe_node")
        self.get_logger().info("Recipe Node initialized")

        # Action Server 생성
        self._action_server = ActionServer(
            self,
            Motion,
            'recipe/motion',  # Action 이름
            self.execute_callback
        )

        self.get_logger().info("Recipe Action Server ready (recipe/motion)")

    def execute_callback(self, goal_handle: ServerGoalHandle):
        """Action 실행 콜백 (Goal 수신 시 호출됨)"""
        motion_name = goal_handle.request.motion_name
        self.get_logger().info(f"Received goal: name={motion_name}")

        # Feedback 메시지 생성
        feedback_msg = Motion.Feedback()
        start_time = time.time()

        # 모션 리스트 정의 (step_name, position)
        # TODO: 실제 movel, movej 좌표로 교체
        motions = [
            ("Move to ingredient position", [0, 0, 0, 0, 0, 0]),
            ("Pour ingredient 1", [50, 50, 50, 0, 0, 0]),
            ("Move to next ingredient", [100, 0, 0, 0, 0, 0]),
            ("Pour ingredient 2", [100, 50, 50, 0, 0, 0]),
            ("Return to home", [0, 0, 0, 0, 0, 0]),
        ]

        total_motions = len(motions)

        for i, (step_name, position) in enumerate(motions):
            # 진행률 계산 (각 모션 완료 시)
            progress = int((i + 1) / total_motions * 100)

            # Feedback 발행 (모션 시작)
            feedback_msg.progress = progress - (100 // total_motions)
            feedback_msg.current_step = f"[{i + 1}/{total_motions}] {step_name}"
            goal_handle.publish_feedback(feedback_msg)
            self.get_logger().info(f"Starting: {feedback_msg.current_step}")

            # 실제 모션 실행
            # TODO: 실제 로봇 제어 코드로 교체
            # self.movel(position) 또는 self.movej(position)
            time.sleep(0.5)  # 시뮬레이션

            # Feedback 발행 (모션 완료)
            feedback_msg.progress = progress
            feedback_msg.current_step = f"[{i + 1}/{total_motions}] {step_name} - Done"
            goal_handle.publish_feedback(feedback_msg)
            self.get_logger().info(f"Completed: {step_name} ({progress}%)")

        # 완료 시간 계산
        elapsed_ms = int((time.time() - start_time) * 1000)

        # Goal 성공 처리
        goal_handle.succeed()

        # Result 반환
        result = Motion.Result()
        result.success = True
        result.message = f"Recipe '{motion_name}' completed successfully"
        result.total_time_ms = elapsed_ms

        self.get_logger().info(f"Recipe completed in {elapsed_ms}ms")

        return result


def main(args=None):
    rclpy.init(args=args)
    node = RecipeController()

    # MultiThreadedExecutor 사용 (콜백 내 서비스 호출 데드락 방지)
    from rclpy.executors import MultiThreadedExecutor
    executor = MultiThreadedExecutor()
    executor.add_node(node)

    try:
        executor.spin()
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()

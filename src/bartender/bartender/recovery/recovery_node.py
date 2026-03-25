#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import time
import rclpy
from rclpy.node import Node
from std_msgs.msg import String
import sys
import io
from bartender.onrobot import RG

# 한글 출력 설정
sys.stdout = io.TextIOWrapper(sys.stdout.detach(), encoding='utf-8')
sys.stderr = io.TextIOWrapper(sys.stderr.detach(), encoding='utf-8')

# 두산 라이브러리 초기화 모듈
import DR_init

# ========================================
# 로봇 설정 파라미터
# ========================================
ROBOT_ID = "dsr01"
ROBOT_MODEL = "m0609"
VELJ = 60
ACCJ = 60
VELX, ACCX = 150, 150
J_READY = [0, 0, 90, 0, 90, 0]
GRIPPER_NAME = "rg2"
TOOLCHARGER_IP = "192.168.1.1"
TOOLCHARGER_PORT = "502"

mini_depth = 25.0

test_dec = {"green_cup": 150, "black_cup": 85, "yellow_cup": 55}

########### Gripper Setup. Do not modify this area ############

gripper = RG(GRIPPER_NAME, TOOLCHARGER_IP, TOOLCHARGER_PORT)

class FailureRecoveryBot(Node):
    def __init__(self):
        super().__init__("failure_recovery_bot", namespace=ROBOT_ID)

        # 미션 상태 관리
        self.last_failed_customer = "미확인 고객"
        self.mission_requested = False
        self.customer_ready = False      # 고객 등록됨 (컵 대기 중)
        self.robot_executing = False     # 로봇 실제 동작 중
        self.cup_name = None

        # 보관대 좌표
        self.storage_posx = [665.0, 8.0, 155.0, 0.0, 180.0, 0.0]

        # 구독자 설정
        self.sub_disappeared = self.create_subscription(
            String, '/disappeared_customer_name', self.disappeared_cb, 10)
        self.sub_manufacturing = self.create_subscription(
            String, '/cup_type', self.start_mission_cb, 10)

        self.get_logger().info('='*50)
        self.get_logger().info(f"M0609 복구 시스템 v5 (spin_once)")
        self.get_logger().info(f"토픽: /disappeared_customer_name, /manufacturing_done")
        self.get_logger().info('='*50)

    def grip(self):
        from DSR_ROBOT2 import set_digital_output
        self.get_logger().info("GRIP ON")
        set_digital_output(1, 1)
        set_digital_output(2, 0)
        time.sleep(0.3)

    def release(self):
        from DSR_ROBOT2 import set_digital_output
        self.get_logger().info("GRIP OFF")
        set_digital_output(1, 0)
        set_digital_output(2, 1)
        time.sleep(0.3)

    def disappeared_cb(self, msg):
        """인식 실패 고객 정보 수신"""
        self.last_failed_customer = msg.data.strip()
        self.get_logger().warn(f"인식 실패 접수: [{self.last_failed_customer}]")
        self.customer_ready = True  # 고객 등록, 컵 대기

    def start_mission_cb(self, msg):
        """제조 완료 신호 수신 - 플래그만 설정"""
        # 로봇이 움직이는 중이면 무시
        if self.robot_executing:
            self.get_logger().warn("로봇 동작 중 - 요청 무시")
            return

        # 고객이 등록되지 않았으면 무시
        if not self.customer_ready:
            self.get_logger().warn("등록된 고객 없음 - 요청 무시")
            return

        msg_data = msg.data.strip()
        self.cup_name = msg_data if msg_data else self.last_failed_customer
        self.mission_requested = True
        self.get_logger().info(f"[미션 요청] {self.cup_name}")

    def recovery_sequence(self):
        """실제 로봇 동작 시퀀스"""
        from DSR_ROBOT2 import (movej, movel, wait, DR_MV_MOD_REL, 
                                release_force, release_compliance_ctrl,
                                task_compliance_ctrl,set_stiffnessx,set_desired_force,DR_FC_MOD_REL,
                                check_force_condition, DR_AXIS_Z,get_tool_force)

        # cup_name 유효성 검사
        if self.cup_name not in test_dec:
            self.get_logger().error(f"알 수 없는 컵: '{self.cup_name}' (유효: {list(test_dec.keys())})")
            self.get_logger().info("올바른 컵 이름으로 다시 요청하세요")
            return  # customer_ready는 유지됨 → 재시도 가능

        self.get_logger().info("1. 홈 위치로 이동")
        movej(J_READY, vel=VELJ, acc=ACCJ)
        wait(0.5)

        self.get_logger().info("2. 그리퍼 ON")
        # self.grip()
        gripper.close_gripper()
        wait(1.0)

        self.get_logger().info("3. 보관대 상공 이동")
        target_up = list(self.storage_posx)
        target_up[2] = test_dec[self.cup_name] + mini_depth
        self.get_logger().info(f"이동 좌표 + 20: {target_up}")
        movel(target_up, vel=VELX, acc=ACCX)
        self.get_logger().info("병 위 이동 완료...")
        task_compliance_ctrl() 
        set_stiffnessx([100, 100, 50, 100, 100, 100])
        set_desired_force([0, 0, -30, 0, 0, 0], [0, 0, 5, 0, 0, 0], mod=DR_FC_MOD_REL)  
        while True:  
            # self.get_logger().info(f"힘 체크 : {get_tool_force()}")
            # self.get_logger().info("하강 중...")
            if not check_force_condition(DR_AXIS_Z, min=10, max=100):
                self.get_logger().info("병 바닥 닿음 감지!")
                break
        release_force(time=0.0)
        release_compliance_ctrl()
        # movel([0, 0, -mini_depth, 0, 0, 0], vel=VELX, acc=ACCX, mod=DR_MV_MOD_REL)
        wait(0.5)

        self.get_logger().info("4. 그리퍼 OFF")
        # self.release()
        gripper.open_gripper()
        wait(1.0)

        self.get_logger().info("5. 복귀")
        movel([0, 0, 100, 0, 0, 0], vel=VELX, acc=ACCX, mod=DR_MV_MOD_REL)
        wait(0.5)
        movej(J_READY, vel=VELJ, acc=ACCJ)

        self.get_logger().info("시퀀스 완료!")


def main(args=None):
    rclpy.init(args=args)

    # 두산 라이브러리 설정
    DR_init.__dsr__id = ROBOT_ID
    DR_init.__dsr__model = ROBOT_MODEL

    node = FailureRecoveryBot()
    DR_init.__dsr__node = node

    from DSR_ROBOT2 import get_tcp, movej, movel, DR_MV_MOD_REL,release_force,release_compliance_ctrl
    node.get_logger().info(f"엔드이펙터 - Gripper : {get_tcp()}")
    release_force(time=0.0)
    release_compliance_ctrl()
    node.get_logger().info("준비 완료. 토픽 대기 중...")

    try:
        # 메인 루프: spin_once + 미션 체크
        while rclpy.ok():
            # 토픽 메시지 처리 (non-blocking)
            rclpy.spin_once(node, timeout_sec=0.1)

            # 미션 요청 확인
            if node.mission_requested:
                node.mission_requested = False
                node.get_logger().info('='*50)
                node.get_logger().info(f"[미션 시작] {node.cup_name}")
                node.get_logger().info('='*50)

                # 유효한 컵 이름인지 먼저 확인
                if node.cup_name not in test_dec:
                    node.recovery_sequence()  # 에러 메시지 출력용
                    continue  # customer_ready 유지 → 재시도 가능

                # 유효한 컵 → 로봇 동작 시작
                node.robot_executing = True
                try:
                    node.recovery_sequence()
                    node.customer_ready = False  # 성공 시에만 고객 초기화
                except Exception as e:
                    node.get_logger().error(f"에러: {e}")
                    import traceback
                    node.get_logger().error(traceback.format_exc())
                finally:
                    node.robot_executing = False  # 로봇 동작 완료

    except KeyboardInterrupt:
        node.get_logger().info("종료됨")
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == "__main__":
    main()
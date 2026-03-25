#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import sys
import os
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image, CameraInfo
from cv_bridge import CvBridge
from bartender_interfaces.srv import DrinkDelivery
from geometry_msgs.msg import Point
from rclpy.action import ActionServer
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.executors import MultiThreadedExecutor
import DR_init
import cv2
import numpy as np
from ultralytics import YOLO
import pyrealsense2 as rs
import threading
import time

# DB 클라이언트 및 Depth Estimation 유틸리티 임포트
from bartender.recipe.depth_estimation import estimate_depth_from_window
from bartender.db.db_client import DBClient
from bartender_interfaces.action import Motion

# ==============================================================================
# [로봇 설정 상수]
# ==============================================================================
ROBOT_ID = "dsr01"
ROBOT_MODEL = "m0609"
ROBOT_TOOL = "Tool Weight"
ROBOT_TCP = "GripperDA_v1"

# ==============================================================================
# [Doosan ROS2 라이브러리 경로 추가]
# ==============================================================================
sys.path.append('/home/rokey/cobot_ws/install/dsr_msgs2/lib/python3.10/site-packages')
sys.path.append('/home/rokey/cobot_ws/install/dsr_msgs2/local/lib/python3.10/dist-packages')
sys.path.append('/home/rokey/cobot_ws/install/dsr_common2/local/lib/python3.10/dist-packages')

try:
    from dsr_msgs2.srv import MoveLine, MoveJoint
    from dsr_msgs2.srv import SetCtrlBoxDigitalOutput
    from dsr_msgs2.srv import SetCurrentTool
    try:
        from dsr_msgs2.srv import GetCurrentPose as GetCurrentPos
    except ImportError:
        from dsr_msgs2.srv import GetCurrentPose as GetCurrentPos
except ImportError as e:
    print(f"ERROR: dsr_msgs2 라이브러리 로드 실패: {e}")
    sys.exit(1)

class ToppingNode(Node):
    """
    토핑 피킹 및 배치를 담당하는 ROS2 노드
    - YOLO 기반 객체 인식
    - RealSense 카메라를 통한 Depth 측정
    - DB에서 메뉴별 토핑 정보 조회
    - Action Server를 통한 Supervisor 연동
    """
    def __init__(self):
        super().__init__("topping_node", namespace=ROBOT_ID)
        
        self.get_logger().info("=== 토핑 노드 초기화 시작 ===")
        
        # ==============================================================================
        # [1. 파일 경로 설정]
        # ==============================================================================
        current_dir = os.path.dirname(os.path.abspath(__file__))
        model_path = os.path.join(current_dir, 'best.pt')  # YOLO 모델 (기존 파일 사용)
        calib_path = os.path.join(current_dir, 'T_gripper2camera.npy')  # 캘리브레이션 매트릭스
        
        # ==============================================================================
        # [2. YOLO 모델 로드]
        # ==============================================================================
        try:
            self.model = YOLO(model_path)
            self.get_logger().info(f"✅ YOLO 모델 로드 성공: {model_path}")
        except Exception as e:
            self.get_logger().error(f"❌ YOLO 모델 로드 실패: {e}")
            sys.exit(1)
        
        # ==============================================================================
        # [3. 캘리브레이션 매트릭스 로드]
        # ==============================================================================
        if os.path.exists(calib_path):
            self.calib_matrix = np.load(calib_path)
            self.get_logger().info("✅ 캘리브레이션 매트릭스 로드 성공")
        else:
            self.calib_matrix = np.eye(4)
            self.get_logger().warn("⚠️ 캘리브레이션 파일 없음. 단위 행렬 사용")
        
        # ==============================================================================
        # [4. RealSense 카메라 초기화]
        # ==============================================================================
        self.pipeline = rs.pipeline()
        config = rs.config()
        config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
        config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
        
        try:
            self.profile = self.pipeline.start(config)
            depth_sensor = self.profile.get_device().first_depth_sensor()
            self.depth_scale = depth_sensor.get_depth_scale()  # depth raw 값을 미터로 변환하는 스케일
            self.align = rs.align(rs.stream.color)  # depth를 color 좌표계로 정렬
            self.get_logger().info("✅ RealSense 카메라 초기화 성공")
        except Exception as e:
            self.get_logger().error(f"❌ RealSense 초기화 실패: {e}")
            sys.exit(1)
        
        # ==============================================================================
        # [5. ROS2 Publisher 및 CvBridge]
        # ==============================================================================
        self.pub_img = self.create_publisher(Image, '/topping/yolo/image', 10)
        self.bridge = CvBridge()
        
        # ==============================================================================
        # [6. ROS2 Service Clients (로봇 제어)]
        # ==============================================================================
        self.move_line_client = self.create_client(MoveLine, f'/{ROBOT_ID}/motion/move_line')
        self.move_joint_client = self.create_client(MoveJoint, f'/{ROBOT_ID}/motion/move_joint')
        self.io_client = self.create_client(SetCtrlBoxDigitalOutput, f'/{ROBOT_ID}/io/set_ctrl_box_digital_output')
        self.get_pos_client = self.create_client(GetCurrentPos, f'/{ROBOT_ID}/system/get_current_pose')
        self.set_tool_client = self.create_client(SetCurrentTool, f'/{ROBOT_ID}/system/set_current_tool')
        
        # 서비스 연결 대기
        if not self.move_line_client.wait_for_service(timeout_sec=2.0):
            self.get_logger().warn("⚠️ MoveLine 서비스 연결 실패")
        if not self.io_client.wait_for_service(timeout_sec=2.0):
            self.get_logger().warn("⚠️ IO 서비스 연결 실패")
        if not self.get_pos_client.wait_for_service(timeout_sec=2.0):
            self.get_logger().warn("⚠️ GetCurrentPose 서비스 연결 실패")
        
        # ==============================================================================
        # [7. DB 클라이언트 초기화]
        # ==============================================================================
        self.db_client = DBClient(self)
        self.db_query_event = threading.Event()  # DB 쿼리 응답 대기용 이벤트
        self.db_query_result = []  # DB 쿼리 결과 저장
        
        # ==============================================================================
        # [8. Action Server 설정 (Supervisor 연동)]
        # ==============================================================================
        self._action_server = ActionServer(
            self,
            Motion,
            'topping/motion',
            self.execute_action_callback,
            callback_group=ReentrantCallbackGroup()
        )
        
        # ==============================================================================
        # [9. 변수 초기화]
        # ==============================================================================
        # 현재 작업 상태
        self.current_recipe = None  # 현재 처리 중인 레시피 정보
        self.target_object = None   # 현재 찾고 있는 객체 (토핑명)
        self.task_step = "idle"     # 작업 단계: "idle", "topping", "drink_delivery"
        self.status_msg = "Waiting..."  # 상태 메시지
        self.is_moving = False      # 로봇 이동 중 플래그
        
        # Vision 관련 저장값
        self.saved_vision_offset = [0.0, 0.0]  # XY 오프셋 저장
        self.saved_approach_dist = 0.0  # 접근 거리 저장
        self.topping_origin_pos = None  # 토핑 픽업 위치 저장
        
        # Action Feedback용 변수
        self.current_goal_handle = None
        self.action_event = threading.Event()
        self.total_action_steps = 0
        self.current_action_step = 0
        
        # ==============================================================================
        # [10. 위치 파라미터 정의]
        # ==============================================================================
        # 토핑 탐색 초기 위치 (카메라가 토핑 트레이를 볼 수 있는 위치)
        self.TOPPING_VIEW_POS = [300.0, 0.0, 400.0, 0.0, 180.0, 0.0]
        
        # 음료 위에 토핑을 올리는 위치
        self.DRINK_TOPPING_POS = [307.16, -12.14, 78.81, 129.37, -177.29, 139.48]
        
        # 홈 위치
        self.HOME_POSITION = [0.0, 0.0, 90.0, 0.0, 90.0, 0.0]
        self.JOINT_HOME_POS = [0.0, 0.0, 90.0, 0.0, 90.0, 0.0]
        
        # 현재 Z 높이 (탐색 위치 기준)
        self.CURRENT_Z_HEIGHT = 400.0
        
        # ==============================================================================
        # [11. 토핑별 파라미터 (객체 인식 시 보정값)]
        # ==============================================================================
        # margin: 접근 시 여유 거리 (작을수록 객체에 가까이 접근)
        # off_x, off_y: XY 보정 오프셋
        self.topping_params = {
            "white_duck": {"off_x": 0.0, "off_y": 0.0, "margin": 120.0},
            "yellow_duck": {"off_x": 0.0, "off_y": 0.0, "margin": 120.0},
            "leaf": {"off_x": 0.0, "off_y": 0.0, "margin": 120.0},
            "default": {"off_x": 0.0, "off_y": 0.0, "margin": 120.0}
        }
        
        # ==============================================================================
        # [12. TCP(Tool Center Point) 설정]
        # ==============================================================================
        self.set_robot_tcp()
        
        # ==============================================================================
        # [13. 타이머 콜백 시작 (Vision Loop)]
        # ==============================================================================
        # 30Hz (0.033초마다) 카메라 이미지 처리 및 객체 인식
        # self.timer = self.create_timer(0.033, self.timer_callback) # 메인 루프에서 직접 호출하도록 변경

        self.get_logger().info("✅ 토핑 노드 초기화 완료")
    
    # ==============================================================================
    # [유틸리티 메서드]
    # ==============================================================================
    
    def set_robot_tcp(self):
        """로봇 TCP(Tool Center Point) 설정"""
        if self.set_tool_client.wait_for_service(timeout_sec=1.0):
            req = SetCurrentTool.Request()
            req.name = ROBOT_TCP
            self.set_tool_client.call_async(req)
            self.get_logger().info(f"✅ TCP 설정: {ROBOT_TCP}")
    
    def set_digital_output(self, index, value):
        """
        디지털 출력 설정 (그리퍼 제어)
        :param index: 출력 포트 번호
        :param value: 0(열기) 또는 1(닫기)
        """
        try:
            req = SetCtrlBoxDigitalOutput.Request()
            req.index = index
            req.value = value
            self.io_client.call_async(req)
            self.get_logger().info(f"IO 설정: Port {index} = {value}")
        except Exception as e:
            self.get_logger().error(f"IO 설정 에러: {e}")
    
    def abort_task(self, reason: str):
        """
        작업 중단 처리
        :param reason: 중단 사유
        """
        self.get_logger().error(f"❌ 작업 중단: {reason}")
        self.status_msg = f"ERROR: {reason}"
        self.is_moving = False
        self.task_step = "idle"
        self.target_object = None
        
        # 액션 실행 중이었다면 중단 처리
        if self.current_goal_handle is not None:
            self.action_event.set()
    
    # ==============================================================================
    # [DB 관련 메서드]
    # ==============================================================================
    
    def fetch_topping_from_db(self, menu_name):
        """
        DB에서 메뉴명에 해당하는 토핑 정보 조회
        :param menu_name: 메뉴 이름
        :return: 토핑 정보 리스트 (예: [{"name": "white_duck"}])
        """
        self.db_query_result = []
        self.db_query_event.clear()
        
        # SQL Injection 방지를 위한 이스케이프 처리
        escaped_menu = menu_name.replace("'", "''")
        
        # 토핑 정보 조회 쿼리
        # 실제 DB 스키마에 맞게 수정 필요
        query = f"""
        SELECT topping_name
        FROM bartender_menu_topping
        WHERE menu_name LIKE '%{escaped_menu}%'
        ORDER BY created_at DESC
        LIMIT 1
        """
        
        self.get_logger().info(f"DB Query: {query.strip()}")
        self.db_client.execute_query_with_response(query, callback=self.on_db_response)
        
        # 응답 대기 (최대 3초)
        if self.db_query_event.wait(timeout=3.0):
            return self.db_query_result
        else:
            self.get_logger().error("DB Query Timeout")
            return []
    
    def on_db_response(self, response):
        """
        DB 쿼리 응답 처리 콜백
        :param response: DB 응답 딕셔너리
        """
        if response.get('success', False):
            self.db_query_result = response.get('result', [])
            self.get_logger().info(f"✅ DB 조회 성공: {len(self.db_query_result)}개 결과")
        else:
            self.get_logger().error(f"❌ DB Error: {response.get('error')}")
        self.db_query_event.set()
    
    # ==============================================================================
    # [Action Server 관련 메서드]
    # ==============================================================================
    
    def execute_action_callback(self, goal_handle):
        """
        Action Server 콜백 - Supervisor로부터 작업 요청 수신
        :param goal_handle: Action Goal Handle
        :return: Action Result
        """
        menu_name = goal_handle.request.motion_name
        self.get_logger().info(f"📋 Action Goal Received: {menu_name}")
        
        # 이미 작업 중인 경우 거부
        if self.is_moving:
            self.get_logger().warn("⚠️ 이미 작업 중입니다. 요청 거부.")
            goal_handle.abort()
            return Motion.Result(success=False, message="Busy")
        
        self.current_goal_handle = goal_handle
        self.action_event.clear()
        
        # 주문 처리 시작
        if not self.process_order(menu_name):
            goal_handle.abort()
            self.current_goal_handle = None
            return Motion.Result(success=False, message="Topping not found in DB")
        
        # 작업 완료 대기
        self.action_event.wait()
        
        self.current_goal_handle = None
        goal_handle.succeed()
        return Motion.Result(success=True, message="Topping task completed", total_time_ms=0)
    
    def report_progress(self, step_desc):
        """
        액션 피드백 발행 (진행 상황 보고)
        :param step_desc: 현재 단계 설명
        """
        if self.current_goal_handle is None:
            return
        
        self.current_action_step += 1
        progress = int((self.current_action_step / self.total_action_steps) * 100)
        progress = min(100, max(0, progress))
        
        feedback = Motion.Feedback()
        feedback.progress = progress
        feedback.current_step = f"[{self.current_action_step}/{self.total_action_steps}] {step_desc}"
        
        self.current_goal_handle.publish_feedback(feedback)
        self.get_logger().info(f"📢 Feedback: {feedback.current_step} ({progress}%)")
    
    # ==============================================================================
    # [주문 처리 메서드]
    # ==============================================================================
    
    def process_order(self, menu_name):
        """
        주문 처리 로직
        :param menu_name: 메뉴 이름
        :return: 성공 여부 (bool)
        """
        # DB에서 토핑 정보 조회
        db_rows = self.fetch_topping_from_db(menu_name)
        
        if not db_rows:
            self.get_logger().error(f"❌ DB에서 '{menu_name}' 토핑 정보를 찾을 수 없습니다.")
            return False
        
        # 첫 번째 결과에서 토핑 이름 추출
        topping_name = db_rows[0].get('topping_name')
        
        if not topping_name:
            self.get_logger().error("❌ 토핑 이름이 비어있습니다.")
            return False
        
        self.current_recipe = {
            "menu_name": menu_name,
            "topping": topping_name
        }
        
        # 액션 시퀀스 생성 (총 단계 수 계산)
        self.total_action_steps = 4  # 1.토핑 탐색, 2.토핑 그립, 3.음료 위 배치, 4.완료
        self.current_action_step = 0
        
        self.get_logger().info(f"📋 작업 시퀀스: 메뉴={menu_name}, 토핑={topping_name}")
        
        # 타겟 객체 설정
        self.target_object = topping_name
        self.task_step = "topping"
        self.status_msg = f"Search: {topping_name}"
        self.is_moving = True
        
        # 그리퍼 열기
        self.set_digital_output(1, 0)
        time.sleep(1.0)
        
        # 초기 위치로 이동
        self.move_to_topping_view()
        return True
    
    # ==============================================================================
    # [Vision Loop - 타이머 콜백]
    # ==============================================================================
    
    def timer_callback(self):
        """
        타이머 콜백 - 카메라 이미지 처리 및 객체 인식
        30Hz로 실행
        """
        annotated_frame = None
        try:
            # 1. RealSense에서 프레임 받기 (타임아웃 2초)
            frames = self.pipeline.wait_for_frames(timeout_ms=2000)
            if not frames:
                self.get_logger().warn("⚠️ RealSense 프레임 없음")
                return
            
            # 2. Depth를 Color에 정렬
            aligned_frames = self.align.process(frames)
            color_frame = aligned_frames.get_color_frame()
            depth_frame = aligned_frames.get_depth_frame()
            
            if not color_frame or not depth_frame:
                return
            
            # 3. 이미지 변환 (ROS -> OpenCV)
            img = np.asanyarray(color_frame.get_data())
            depth_image = np.asanyarray(depth_frame.get_data())
            intrinsics = color_frame.profile.as_video_stream_profile().intrinsics
            
            # 4. YOLO 추론
            results = self.model(img, verbose=False)
            annotated_frame = results[0].plot()
            
            # 5. 상태 메시지 표시 (화면 상단에 오버레이)
            cv2.rectangle(annotated_frame, (0, 0), (640, 60), (0, 0, 0), -1)
            cv2.putText(annotated_frame, self.status_msg, (10, 35),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
            # 6. 객체 탐색 로직 (작업 중이 아니고, 타겟이 설정되어 있을 때만)
            if not self.is_moving and self.task_step == "topping" and self.target_object:
                boxes = results[0].boxes
                
                for box in boxes:
                    cls_id = int(box.cls[0])
                    cls_name = self.model.names[cls_id]
                    
                    # 타겟 객체인지 확인
                    if cls_name == self.target_object:
                        # 바운딩 박스 좌표
                        x1, y1, x2, y2 = map(int, box.xyxy[0])
                        
                        # 중심점 계산 (토핑은 박스 중심 사용)
                        cx = (x1 + x2) // 2
                        cy = (y1 + y2) // 2
                        
                        # Depth 측정 (통계 기반 노이즈 제거)
                        dist, depth_stats = estimate_depth_from_window(
                            depth_image,
                            center_xy=(cx, cy),
                            window_radius=5,  # 중심점 주변 5픽셀 윈도우
                            std_threshold=0.03,  # 표준편차 임계값 (m)
                            k=1.0,
                            min_inliers=8,  # 최소 유효 픽셀 수
                            depth_scale=self.depth_scale,  # z16 raw -> meters
                            min_depth=0.1,  # 최소 거리 (m)
                            max_depth=1.2,  # 최대 거리 (m)
                            reducer="median",  # 중앙값 사용
                            fallback_reducer="median",
                            prefer_near_cluster=True,  # 가까운 클러스터 우선
                        )
                        
                        # 유효한 거리가 측정되었을 때만 표시
                        if dist > 0:
                            cv2.putText(annotated_frame, f"Dist: {dist:.3f}m", (x1, y1-20),
                                      cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                            cv2.circle(annotated_frame, (cx, cy), 5, (0, 0, 255), -1)
                        
                        # 인식 범위 내 들어오면 이동 시작
                        if 0.1 < dist < 1.2:
                            try:
                                self.get_logger().info(
                                    f"📏 Depth stats: "
                                    f"mean={float(depth_stats['mean']):.3f}m, "
                                    f"std={float(depth_stats['std']):.3f}m, "
                                    f"samples={int(depth_stats['num_samples'])}, "
                                    f"inliers={int(depth_stats['num_inliers'])}"
                                )
                            except Exception:
                                pass
                            
                            # 픽셀 좌표 + Depth -> 3D 카메라 좌표
                            cam_point = rs.rs2_deproject_pixel_to_point(intrinsics, [cx, cy], dist)
                            c_x = cam_point[0] * 1000.0  # m -> mm
                            c_y = cam_point[1] * 1000.0
                            c_z = cam_point[2] * 1000.0
                            
                            # 카메라 좌표계 -> 그리퍼 좌표계 변환
                            gripper_pos = np.dot(self.calib_matrix, np.array([c_x, c_y, c_z, 1.0]))
                            gx, gy, gz = gripper_pos[0], gripper_pos[1], gripper_pos[2]
                            
                            self.get_logger().info(
                                f"🎯 Vision(mm): cx,cy=({cx},{cy}) "
                                f"cam=[{c_x:.1f},{c_y:.1f},{c_z:.1f}] -> "
                                f"tool=[{gx:.1f},{gy:.1f},{gz:.1f}]"
                            )
                            
                            # Eye-in-Hand 이동 실행
                            self.execute_eye_in_hand_move(gx, gy, gz)
                            break  # 첫 번째 타겟만 처리
            
            # 7. 이미지 퍼블리시 (ROS Topic으로 전송)
            try:
                msg = self.bridge.cv2_to_imgmsg(annotated_frame, encoding="bgr8")
                self.pub_img.publish(msg)
            except Exception as e:
                pass  # 퍼블리시 에러는 무시
            
            # 8. 화면에 표시
            cv2.imshow("Topping Vision", annotated_frame)
            if cv2.waitKey(1) == 27:  # ESC 키로 종료
                rclpy.shutdown()
        
        except Exception as e:
            self.get_logger().error(f"❌ Vision Loop Error: {e}")
            # 에러 발생 시에도 마지막 프레임은 표시
            if annotated_frame is not None:
                try:
                    self.pub_img.publish(self.bridge.cv2_to_imgmsg(annotated_frame, encoding="bgr8"))
                except:
                    pass
                
                cv2.imshow("Topping Vision", annotated_frame)
                if cv2.waitKey(1) == 27:
                    rclpy.shutdown()
    
    # ==============================================================================
    # [로봇 동작 메서드 - 이동 시퀀스]
    # ==============================================================================
    
    def move_to_topping_view(self):
        """
        토핑 탐색 위치로 이동
        Joint Home -> Topping View Position 순서로 안전하게 이동
        """
        self.get_logger().info("🏠 토핑 탐색 위치로 이동: Joint Home -> Topping View")
        
        # Feedback: 토핑 탐색 시작
        self.report_progress(f"토핑 탐색 ({self.target_object})")
        
        # 먼저 Joint Home으로 이동 (안전한 자세)
        if not self.move_joint_client.wait_for_service(timeout_sec=1.0):
            self.abort_task("MoveJoint 서비스 미연결")
            return
        
        req = MoveJoint.Request()
        req.pos = self.JOINT_HOME_POS
        req.vel = 60.0
        req.acc = 40.0
        
        future = self.move_joint_client.call_async(req)
        future.add_done_callback(self.move_to_topping_view_linear)
    
    def move_to_topping_view_linear(self, future):
        """
        Joint Home 도착 후 Topping View로 직선 이동
        :param future: MoveJoint Future
        """
        try:
            res = future.result()
        except Exception as e:
            self.abort_task(f"Joint Home 이동 실패: {e}")
            return
        
        if getattr(res, "success", False):
            # Topping View 위치로 직선 이동
            req = MoveLine.Request()
            req.pos = self.TOPPING_VIEW_POS
            req.vel = [100.0, 0.0]
            req.acc = [100.0, 0.0]
            req.ref = 0  # Base 좌표계
            req.mode = 0  # Absolute
            req.sync_type = 0
            
            future = self.move_line_client.call_async(req)
            future.add_done_callback(self.ready_to_search_topping)
        else:
            self.abort_task(f"Joint Home 이동 실패: {res}")
    
    def ready_to_search_topping(self, future):
        """
        Topping View 도착 후 탐색 준비
        :param future: MoveLine Future
        """
        try:
            res = future.result()
        except Exception as e:
            self.abort_task(f"Topping View 이동 실패: {e}")
            return
        
        if getattr(res, "success", False):
            self.get_logger().info("✅ 토핑 탐색 위치 도착. 객체 인식 시작.")
            self.status_msg = f"Search: {self.target_object}"
            self.is_moving = False  # Vision Loop가 객체를 찾을 수 있도록 플래그 해제
            self.CURRENT_Z_HEIGHT = float(self.TOPPING_VIEW_POS[2])
        else:
            self.abort_task(f"Topping View 이동 실패: {res}")
    
    def execute_eye_in_hand_move(self, offset_x, offset_y, offset_z):
        """
        Eye-in-Hand 방식으로 객체 정렬
        Vision에서 계산한 오프셋으로 XY 정렬 후 하강
        
        :param offset_x: 그리퍼 기준 X 오프셋 (mm)
        :param offset_y: 그리퍼 기준 Y 오프셋 (mm)
        :param offset_z: 객체까지 거리 (mm)
        """
        self.is_moving = True
        self.saved_approach_dist = offset_z
        self.status_msg = "Eye-in-Hand Aligning..."
        
        # 토핑별 파라미터 로드
        params = self.topping_params.get(self.target_object, self.topping_params["default"])
        
        # XY 오프셋 보정 적용
        offset_x += params["off_x"]
        offset_y += params["off_y"]
        
        # ★ 저장: 나중에 토핑 원위치 복귀 시 사용
        self.saved_vision_offset = [offset_x, offset_y]
        
        self.get_logger().info(
            f"🎯 XY 정렬 시작: X={offset_x:.1f}, Y={offset_y:.1f} "
            f"(보정: X+={params['off_x']}, Y+={params['off_y']})"
        )
        
        # [1단계] XY 정렬 (그리퍼 기준 상대 이동)
        req = MoveLine.Request()
        req.pos = [float(offset_x), float(offset_y), 0.0, 0.0, 0.0, 0.0]
        req.vel = [100.0, 0.0]
        req.acc = [100.0, 0.0]
        req.ref = 1  # Tool 좌표계
        req.mode = 1  # Relative
        req.sync_type = 0
        
        future = self.move_line_client.call_async(req)
        
        # [2단계] 하강 준비 (XY 정렬 완료 후 실행)
        future.add_done_callback(lambda f: self.descend_to_topping(f))
    
    def descend_to_topping(self, future=None):
        """
        토핑 위치로 수직 하강
        :param future: MoveLine Future (XY 정렬 결과)
        """
        # XY 정렬 결과 확인
        if future is not None:
            try:
                if not future.result().success:
                    self.get_logger().warn("⚠️ XY 정렬 실패")
                    self.reset_state()
                    return
            except Exception as e:
                self.get_logger().warn(f"⚠️ XY 정렬 결과 확인 실패: {e}")
                self.reset_state()
                return
        
        self.status_msg = "Descending to Topping..."
        
        # 접근 거리 계산 (마진 적용)
        params = self.topping_params.get(self.target_object, self.topping_params["default"])
        margin = params["margin"]
        approach_dist = self.saved_approach_dist - margin
        
        self.get_logger().info(f"⬇️ 토핑 접근: Tool Z +{approach_dist:.1f}mm (마진: {margin:.1f}mm)")
        
        # Tool Z축 방향으로 전진 (카메라가 앞을 보고 있으므로 Z축이 접근 방향)
        req = MoveLine.Request()
        req.pos = [0.0, 0.0, float(approach_dist), 0.0, 0.0, 0.0]
        req.vel = [50.0, 0.0]
        req.acc = [50.0, 0.0]
        req.ref = 1  # Tool 좌표계
        req.mode = 1  # Relative
        req.sync_type = 0
        
        future = self.move_line_client.call_async(req)
        future.add_done_callback(self.grasp_topping)
    
    def grasp_topping(self, future):
        """
        토핑 그립
        :param future: MoveLine Future (하강 결과)
        """
        if future.result().success:
            self.status_msg = "Grasping Topping..."
            self.get_logger().info("✋ 토핑 그립")
            
            # Feedback: 토핑 그립
            self.report_progress("토핑 그립")
            
            # 현재 위치 저장 (원위치 복귀용)
            self.get_current_pose_and_save()
            
            # 그리퍼 닫기
            self.set_digital_output(1, 1)
            time.sleep(1.0)  # 그립 안정화 대기
            
            # 상승
            self.lift_topping()
        else:
            self.get_logger().warn("❌ 토핑 접근 실패")
            self.reset_state()
    
    def get_current_pose_and_save(self):
        """토핑을 잡은 위치 저장 (나중에 원위치 복귀용)"""
        self.topping_origin_pos = None
        
        if not self.get_pos_client.wait_for_service(timeout_sec=1.0):
            self.get_logger().warn("⚠️ GetCurrentPose 서비스 미연결. 위치 저장 건너뜀")
            return
        
        req = GetCurrentPos.Request()
        req.space_type = 1  # Task Space
        future = self.get_pos_client.call_async(req)
        future.add_done_callback(self.save_topping_pos)
    
    def save_topping_pos(self, future):
        """
        토핑 Pick 위치 저장 콜백
        :param future: GetCurrentPos Future
        """
        try:
            res = future.result()
            if res.success:
                self.topping_origin_pos = res.pos
                self.get_logger().info(
                    f"💾 토핑 Pick 위치 저장: "
                    f"X={res.pos[0]:.1f}, Y={res.pos[1]:.1f}, Z={res.pos[2]:.1f}"
                )
            else:
                self.get_logger().warn("⚠️ 위치 저장 실패 (Result Fail)")
        except Exception as e:
            self.get_logger().warn(f"⚠️ 위치 저장 에러: {e}")
    
    def lift_topping(self):
        """토핑을 들어올림 (Base 기준 상대 상승)"""
        self.status_msg = "Lifting Topping..."
        self.get_logger().info("🚀 토핑 상승: Base Z +200mm")
        
        req = MoveLine.Request()
        req.pos = [0.0, 0.0, 200.0, 0.0, 0.0, 0.0]
        req.vel = [100.0, 0.0]
        req.acc = [100.0, 0.0]
        req.ref = 0  # Base 좌표계
        req.mode = 1  # Relative
        req.sync_type = 0
        
        future = self.move_line_client.call_async(req)
        future.add_done_callback(self.move_to_drink_position)
    
    def move_to_drink_position(self, future):
        """
        음료 위치로 이동 (Joint Home 경유)
        :param future: MoveLine Future (상승 결과)
        """
        if future.result().success:
            self.status_msg = "Moving to Drink..."
            self.get_logger().info("🥤 음료 위치로 이동: Joint Home 경유")
            
            # Feedback: 음료 위 배치
            self.report_progress("음료 위 배치")
            
            # Joint Home으로 이동
            req = MoveJoint.Request()
            req.pos = self.JOINT_HOME_POS
            req.vel = 60.0
            req.acc = 40.0
            
            future = self.move_joint_client.call_async(req)
            future.add_done_callback(self.place_topping_on_drink)
        else:
            self.reset_state()
    
    def place_topping_on_drink(self, future):
        """
        음료 위에 토핑 배치
        :param future: MoveJoint Future
        """
        if future.result().success:
            self.status_msg = "Placing Topping on Drink..."
            self.get_logger().info("🍹 음료 위에 토핑 올리기")
            
            # 음료 위치로 이동 (상공 100mm)
            drink_pos = list(self.DRINK_TOPPING_POS)
            drink_pos[2] += 100.0  # 안전 여유
            
            req = MoveLine.Request()
            req.pos = [float(x) for x in drink_pos]
            req.vel = [50.0, 0.0]
            req.acc = [50.0, 0.0]
            req.ref = 0
            req.mode = 0
            
            future = self.move_line_client.call_async(req)
            future.add_done_callback(self.descend_to_drink)
        else:
            self.reset_state()
    
    def descend_to_drink(self, future):
        """
        음료 위치로 하강
        :param future: MoveLine Future
        """
        if future.result().success:
            # 최종 배치 위치로 하강
            req = MoveLine.Request()
            req.pos = [float(x) for x in self.DRINK_TOPPING_POS]
            req.vel = [30.0, 0.0]
            req.acc = [30.0, 0.0]
            req.ref = 0
            req.mode = 0
            
            future = self.move_line_client.call_async(req)
            future.add_done_callback(self.release_topping)
        else:
            self.reset_state()
    
    def release_topping(self, future):
        """
        토핑 놓기 (그리퍼 열기)
        :param future: MoveLine Future
        """
        if future.result().success:
            self.get_logger().info("✅ 토핑 배치 완료. 그리퍼 해제")
            
            # 그리퍼 열기
            self.set_digital_output(1, 0)
            time.sleep(0.5)
            
            # 안전하게 위로 빠져나오기
            self.status_msg = "Retracting..."
            req = MoveLine.Request()
            req.pos = [0.0, 0.0, 100.0, 0.0, 0.0, 0.0]
            req.vel = [100.0, 0.0]
            req.acc = [100.0, 0.0]
            req.ref = 0
            req.mode = 1
            
            future = self.move_line_client.call_async(req)
            future.add_done_callback(lambda f: threading.Timer(1.0, self.finish_task).start())
        else:
            self.reset_state()
    
    def finish_task(self):
        """모든 작업 완료"""
        self.status_msg = "Task Completed. Returning Home..."
        self.get_logger().info("🏁 모든 작업 완료. 홈 위치로 복귀")
        
        # Feedback: 완료
        self.report_progress("작업 완료")
        
        # 작업 상태 초기화
        self.task_step = "idle"
        self.target_object = None
        
        # Joint Home으로 복귀
        req = MoveJoint.Request()
        req.pos = self.JOINT_HOME_POS
        req.vel = 50.0
        req.acc = 30.0
        
        self.move_joint_client.call_async(req)
        
        # Action 완료 이벤트 설정
        self.action_event.set()
        self.reset_state()
    
    def reset_state(self):
        """상태 초기화"""
        time.sleep(1.0)
        self.is_moving = False
        self.status_msg = "Ready"
        if self.task_step == "idle":
            self.current_recipe = None
    
    # ==============================================================================
    # [노드 종료 처리]
    # ==============================================================================
    
    def destroy_node(self):
        """노드 종료 시 리소스 정리"""
        try:
            self.pipeline.stop()
            self.get_logger().info("✅ RealSense 파이프라인 종료")
        except:
            pass
        
        cv2.destroyAllWindows()
        super().destroy_node()


def main(args=None):
    """메인 함수"""
    rclpy.init(args=args)
    
    import DR_init
    # Doosan 로봇 초기화
    DR_init.__dsr__id = ROBOT_ID
    DR_init.__dsr__model = ROBOT_MODEL
    
    # 노드 생성
    node = ToppingNode()
    DR_init.__dsr__node = node
    
    # TCP 확인
    from DSR_ROBOT2 import get_tcp
    
    if get_tcp() != ROBOT_TCP:
        print(f"❌ TCP 불일치: {get_tcp()} != {ROBOT_TCP}")
        node.destroy_node()
        rclpy.shutdown()
        return
    
    # MultiThreadedExecutor 사용 (Action Server와 Service 동시 처리)
    executor = MultiThreadedExecutor()
    executor.add_node(node)
    
    try:
        # 메인 스레드에서 비전 처리를, 서브 스레드에서 ROS 통신을 처리
        while rclpy.ok():
            # ROS2 콜백 및 서비스 처리 (non-blocking)
            executor.spin_once(timeout_sec=0.001)
            # 비전 처리 (cv2.imshow는 메인 스레드에서 호출되어야 안정적)
            node.timer_callback()
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()

if __name__ == "__main__":
    main()
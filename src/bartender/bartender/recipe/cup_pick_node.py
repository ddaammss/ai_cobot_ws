#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from std_msgs.msg import String
from cv_bridge import CvBridge
from rclpy.action import ActionServer
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.executors import MultiThreadedExecutor
import cv2
import numpy as np
import pyrealsense2 as rs
from ultralytics import YOLO
import threading
import sys
import os
# import json
import time
from bartender.onrobot import RG
from bartender.recipe.depth_estimation import estimate_depth_from_window
from bartender.db.db_client import DBClient
from bartender_interfaces.action import Motion

ROBOT_TCP = "GripperDA_v1"
GRIPPER_NAME = "rg2"
TOOLCHARGER_IP = "192.168.1.1"
TOOLCHARGER_PORT = "502"

gripper = RG(GRIPPER_NAME, TOOLCHARGER_IP, TOOLCHARGER_PORT)

# ==============================================================================
# [라이브러리 경로 추가]
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

class BartenderNode(Node):
    def __init__(self):
        super().__init__('bartender_cup_pick', namespace='dsr01')
        self.get_logger().info("=== Bartender Bot (Fixed Version) ===")

        # 1. 파일 경로 설정
        current_dir = os.path.dirname(os.path.abspath(__file__))
        json_path = os.path.join(current_dir, 'recipe.json')
        model_path = os.path.join(current_dir, 'best.pt')
        calib_path = os.path.join(current_dir, 'T_gripper2camera.npy')

        # 2. 데이터 로드 (JSON 대신 DB 사용 예정이므로 주석 처리하거나 fallback으로 유지)
        # if os.path.exists(json_path):
        #     with open(json_path, 'r', encoding='utf-8') as f:
        #         self.recipe_data = json.load(f)
        # else:
        #     self.get_logger().error("recipe.json 없음"); sys.exit(1)

        # Callback Group 생성 (Action과 DB 응답을 동시 처리)
        self._callback_group = ReentrantCallbackGroup()
        self.get_logger().info(f"🔧 ReentrantCallbackGroup 생성됨: {self._callback_group}")
        # DB 클라이언트 초기화 (callback_group 전달)
        self.db_client = DBClient(self)
        self.get_logger().info("✅ DBClient 초기화 완료")
        self.db_query_event = threading.Event()
        self.db_query_result = []

        self.calib_matrix = np.load(calib_path) if os.path.exists(calib_path) else np.eye(4)
        
        try:
            self.model = YOLO(model_path)
        except Exception:
            self.get_logger().error("YOLO 모델 로드 실패"); sys.exit(1)

        # 3. RealSense (필요할 때만 열고 닫음)
        self.pipeline = None
        self.align = None
        self.depth_scale = None

        # 4. ROS 클라이언트 설정
        self.pub_img = self.create_publisher(Image, '/yolo/image', 10)
        self.br = CvBridge()
        
        self.move_line_client = self.create_client(MoveLine, '/dsr01/motion/move_line')
        self.move_joint_client = self.create_client(MoveJoint, '/dsr01/motion/move_joint')
        self.io_client = self.create_client(SetCtrlBoxDigitalOutput, '/dsr01/io/set_ctrl_box_digital_output')

        # 로봇 서비스 클라이언트들도 ReentrantCallbackGroup 사용 (Action 실행 중 응답 받기 위해)
        self.move_line_client = self.create_client(
        MoveLine, '/dsr01/motion/move_line', callback_group=self._callback_group)
        self.move_joint_client = self.create_client(
        MoveJoint, '/dsr01/motion/move_joint', callback_group=self._callback_group)
        self.io_client = self.create_client(
        SetCtrlBoxDigitalOutput, '/dsr01/io/set_ctrl_box_digital_output', callback_group=self._callback_group)
        # Doosan ROS2 표준 서비스명은 get_current_pose 입니다. (get_current_pos 아님)
        self.get_pos_client = self.create_client(
        GetCurrentPos, '/dsr01/system/get_current_pose', callback_group=self._callback_group)
        self.set_tool_client = self.create_client(
        SetCurrentTool, '/dsr01/system/set_current_tool', callback_group=self._callback_group)

        if not self.move_line_client.wait_for_service(timeout_sec=2.0):
            self.get_logger().warn("⚠️ 로봇 서비스 연결 실패")
        if not self.io_client.wait_for_service(timeout_sec=2.0):
            self.get_logger().warn("⚠️ IO 서비스 연결 실패")
        if not self.get_pos_client.wait_for_service(timeout_sec=2.0):
            self.get_logger().warn("⚠️ get_current_pose 서비스 연결 실패 (현재 Z 기반 상승 보정 불가)")

        # [토픽 구독] 메뉴명 수신 (아직은 주석 처리)
        # self.create_subscription(String, '/customer_name', self.on_customer_name_received, 10)
        # self.get_logger().info("Waiting for /customer_name topic...")

        # [Action Server] Supervisor 연동용
        self._action_server = ActionServer(
            self,
            Motion,
            'recipe/motion',
            self.execute_action_callback,
            callback_group=ReentrantCallbackGroup()
        )

        # 5. 변수 초기화
        self.current_recipe = None
        self.target_object = None   
        self.task_step = "idle"     
        self.liquor_idx = 0         
        self.bottle_origin_pos = None 
        self.saved_vision_offset = [0.0, 0.0]
        self.saved_approach_dist = 0.0

        # Action Feedback용 변수
        self.current_goal_handle = None
        self.action_event = threading.Event()
        self.total_action_steps = 0
        self.current_action_step = 0

        self.status_msg = "Waiting..."
        self.is_moving = False
        self.detected_pose = None
        
        # [위치/높이 파라미터]
        # 컵 탐색 초기 위치 (Z=359.12)
        self.INITIAL_READY_POS = [420.31, 125.52, 377.49, 81.10, -179.79, 80.59]
        self.CURRENT_Z_HEIGHT = 377.49 

        # 컵 놓는 베이스 위치 (X, Y는 고정, Z는 가변)
        self.BASE_HOME_POS = [389.39, 21.52, 55.59, 10.74, -179.71, 10.58]
        self.JOINT_HOME_POS = [0.0, 0.0, 90.0, 0.0, 90.0, 0.0]
        
        # 병 탐색 위치
        self.BOTTLE_VIEW_POS = [-200.0, 600.0, 360.0, 0.0, -90.0, -90.0]

        # [추가] 중간 경유지 (Base 좌표계)
        self.INTERMEDIATE_WAYPOINT_POS_1 = [-142.91, 530.40, 692.69, 138.64, 82.57, 82.81]
        self.INTERMEDIATE_WAYPOINT_POS_2 = [321.62, 318.62, 375.63, 85.71, 149.45, 56.19]

        # [추가] 병별 파라미터 (XY 보정, 접근 여유거리)
        # margin이 작을수록 병 쪽으로 더 많이 전진합니다.
        self.bottle_params = {
            "black_bottle": {"off_x": 0.0, "off_y": 0.0, "margin": 160.0},
            "blue_bottle":  {"off_x": 5.0, "off_y": 0.0, "margin": 150.0},
            "default":      {"off_x": 0.0, "off_y": 0.0, "margin": 160.0}
        }

        # ★ 컵 종류별 놓는 높이 (Z 절대 좌표)
        self.cup_place_target_z = {
            "green_cup": 145.0,
            "black_cup": 80.0,
            "yellow_cup": 50.0
        }

        self.set_robot_tcp()

        self.input_thread = threading.Thread(target=self.user_input_loop, daemon=True)
        self.input_thread.start()
        # self.timer = self.create_timer(0.033, self.timer_callback)

    def abort_task(self, reason: str):
        """Stop current task safely and prevent vision loop from triggering motion."""
        self.get_logger().error(f"❌ 작업 중단: {reason}")
        self.status_msg = f"ERROR: {reason}"
        self.is_moving = False
        self.task_step = "idle"
        self.target_object = None
        self.vision_align_iter = 0
        
        # 액션 실행 중이었다면 중단 처리
        if self.current_goal_handle is not None:
            self.action_event.set()

    def set_robot_tcp(self):
        if self.set_tool_client.wait_for_service(timeout_sec=1.0):
            req = SetCurrentTool.Request()
            req.name = ROBOT_TCP
            self.set_tool_client.call_async(req)

    def set_digital_output(self, index, value):
        try:
            req = SetCtrlBoxDigitalOutput.Request()
            req.index = index; req.value = value
            self.io_client.call_async(req)
        except Exception as e:
            self.get_logger().error(f"IO Error: {e}")

    # --- DB 관련 메서드 ---
    def fetch_recipe_from_db(self, menu_seq_or_name):
        """DB에서 레시피 정보를 조회합니다."""
        self.db_query_result = []
        self.db_query_event.clear()

        escaped_keyword = menu_seq_or_name.replace("'", "''")
        # 요청된 쿼리문
        query = f"""
        SELECT name, pour_time, cup
        FROM bartender_menu_recipe
        WHERE menu_seq LIKE '%{escaped_keyword}%'
        ORDER BY created_at DESC
        """
        
        self.get_logger().info(f"DB Query: {query.strip()}")
        self.db_client.execute_query_with_response(query, callback=self.on_db_response)
        
        # 응답 대기 (최대 5초)
        if self.db_query_event.wait(timeout=5.0):
            self.get_logger().info("✅ DB 응답 수신 완료!")
            return self.db_query_result
        else:
            self.get_logger().error("DB Query Timeout")
            return []

    def on_db_response(self, response):
        if response.get('success', False):
            self.db_query_result = response.get('result', [])
        else:
            self.get_logger().error(f"DB Error: {response.get('error')}")
        self.db_query_event.set()

    def on_customer_name_received(self, msg):
        """토픽으로 메뉴명(또는 고객명에 매핑된 메뉴)을 받았을 때 처리"""
        menu_name = msg.data.strip()
        self.get_logger().info(f"Topic Received: {menu_name}")
        # 여기서 process_order(menu_name) 호출 가능

    def generate_action_sequence(self, recipe):
        """액션 통신을 위해 작업 순서를 생성하여 반환/출력합니다."""
        seq = []
        seq.append(f"컵 픽업 ({recipe.get('cup', 'unknown')})")
        seq.append(f"컵 배치")
        
        liquors = recipe.get('liquors', [])
        for liquor in liquors:
            seq.append(f"병 픽업 ({liquor['name']})")
            seq.append(f"따르기 ({liquor['pour_time']}s)")
            seq.append(f"병 반납")
            
        # seq.append(f"서빙 준비 완료")
        return seq

    def user_input_loop(self):
        time.sleep(1)
        print("\n [System] 메뉴를 입력하세요 (예: blue_sapphire)")
        while rclpy.ok():
            try:
                if self.is_moving:
                    time.sleep(1); continue
                user_input = input("\n메뉴 입력 >> ").strip()
                if not user_input: continue

                self.process_order(user_input)

            except: break

    def start_camera(self):
        """RealSense 카메라 시작"""
        if self.pipeline is not None:
            return
        self.pipeline = rs.pipeline()
        config = rs.config()
        config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
        config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
        profile = self.pipeline.start(config)
        depth_sensor = profile.get_device().first_depth_sensor()
        self.depth_scale = depth_sensor.get_depth_scale()
        self.align = rs.align(rs.stream.color)
        self.get_logger().info("RealSense 시작")

    def stop_camera(self):
        """RealSense 카메라 종료"""
        if self.pipeline:
            try:
                self.pipeline.stop()
            except Exception:
                pass
            self.pipeline = None
            self.align = None
            self.get_logger().info("RealSense 종료")

    def process_order(self, menu_name):
        """주문 처리 로직 (CLI 및 Action 공용)"""
        self.start_camera()
        # DB에서 레시피 조회
        db_rows = self.fetch_recipe_from_db(menu_name)
        
        if not db_rows:
            self.get_logger().error(f"❌ DB에서 '{menu_name}' 레시피를 찾을 수 없습니다.")
            return False

        # DB 결과를 레시피 포맷으로 변환
        liquors = []
        for row in db_rows:
            l_name = row.get('name')
            l_time = float(row.get('pour_time', 2.0))
            liquors.append({"name": l_name, "pour_time": l_time})

        # 컵 정보 매핑 (DB 첫 번째 row 사용)
        cup_type = db_rows[0].get('cup')
        if not cup_type:
            cup_type = self.menu_cup_map.get(menu_name, "black_cup")

        self.current_recipe = {
            "recipe_id": menu_name,
            "display_name": menu_name,
            "cup": cup_type,
            "liquors": liquors
        }

        # 액션 시퀀스 생성 및 단계 수 계산
        action_seq = self.generate_action_sequence(self.current_recipe)
        self.total_action_steps = len(action_seq)
        self.current_action_step = 0
        
        self.get_logger().info(f"📋 작업 시퀀스 ({self.total_action_steps}단계):\n" + "\n".join(action_seq))

        self.target_object = cup_type
        self.task_step = "cup"
        self.liquor_idx = 0
        self.status_msg = "Moving to Start Pos..."
        self.is_moving = True 
        
        gripper.open_gripper()
        time.sleep(1.0)
        
        self.get_logger().info(f"주문 접수: {menu_name}")
        self.move_to_initial_ready()
        return True

    def execute_action_callback(self, goal_handle):
        """Action Server 콜백"""
        motion_name = goal_handle.request.motion_name
        self.get_logger().info(f"Action Goal Received: {motion_name}")

        if self.is_moving:
            self.get_logger().warn("이미 작업 중입니다. 요청 거부.")
            goal_handle.abort()
            return Motion.Result(success=False, message="Busy")

        self.current_goal_handle = goal_handle
        self.action_event.clear()

        # 주문 처리 시작
        if not self.process_order(motion_name):
            goal_handle.abort()
            self.current_goal_handle = None
            return Motion.Result(success=False, message="Recipe not found")

        # 작업 완료 대기 (MultiThreadedExecutor가 콜백 처리)
        completed = self.action_event.wait(timeout=300.0)

        if not completed:
            self.get_logger().error("❌ Action Timeout (5분 초과)")
            goal_handle.abort()
            self.current_goal_handle = None
            return Motion.Result(success=False, message="Timeout")

        self.get_logger().info("✅ 작업 완료!")

        self.current_goal_handle = None
        goal_handle.succeed()
        return Motion.Result(success=True, message="Completed", total_time_ms=0)        

    def report_progress(self, step_desc):
        """액션 피드백 발행"""
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

    def move_to_initial_ready(self):
        # [수정] 초기 이동 시 안전을 위해 Joint Home 경유
        self.get_logger().info("초기 위치 이동: Joint Home -> Ready Pos")
        
        # Feedback: 컵 픽업 시작
        self.report_progress(f"컵 픽업 ({self.target_object})")

        if not self.move_joint_client.wait_for_service(timeout_sec=1.0):
            self.abort_task("move_joint 서비스 미연결")
            return
        req = MoveJoint.Request()
        req.pos = self.JOINT_HOME_POS
        req.vel = 60.0; req.acc = 40.0
        # [수정] sync_type=0 제거 (기본값 사용). MoveJoint 거부 방지.
        future = self.move_joint_client.call_async(req)
        future.add_done_callback(self.move_to_ready_linear)

    def move_to_ready_linear(self, future):
        try:
            res = future.result()
        except Exception as e:
            self.abort_task(f"Joint Home 호출 실패: {e}")
            return

        if getattr(res, "success", False):
            req = MoveLine.Request()
            req.pos = self.INITIAL_READY_POS
            req.vel = [100.0, 0.0]; req.acc = [100.0, 0.0]
            req.ref = 0; req.mode = 0 
            req.sync_type = 0
            future = self.move_line_client.call_async(req)
            future.add_done_callback(self.ready_to_search_cup)
        else:
            self.abort_task(f"Joint Home 이동 실패: {res}")

    def ready_to_search_cup(self, future):
        try:
            res = future.result()
        except Exception as e:
            self.abort_task(f"Ready 위치 이동 실패: {e}")
            return

        if getattr(res, "success", False):
            self.get_logger().info("초기 위치 도착. 탐색 시작.")
            self.status_msg = f"Search: {self.target_object}"
            self.is_moving = False
        else:
            self.abort_task(f"Ready 위치 이동 실패: {res}")

    def timer_callback(self):
        annotated_frame = None
        try:
            # 카메라 파이프라인 확인
            if self.pipeline is None:
                return  # 카메라가 초기화되지 않았으면 조용히 리턴

            # 1. 프레임 받기 (blocking)
            frames = self.pipeline.wait_for_frames(timeout_ms=1000)
            if not frames:
                return

            aligned_frames = self.align.process(frames)
            color_frame = aligned_frames.get_color_frame()
            depth_frame = aligned_frames.get_depth_frame()
            
            if not color_frame or not depth_frame: 
                return

            # 2. 이미지 변환
            img = np.asanyarray(color_frame.get_data())
            depth_image = np.asanyarray(depth_frame.get_data())
            intrinsics = color_frame.profile.as_video_stream_profile().intrinsics

            # 3. YOLO 추론
            if not self.is_moving:
                results = self.model(img, verbose=False)
                annotated_frame = results[0].plot()
            else:
                results = None
                annotated_frame = img.copy()

            # 4. 상태 메시지 표시
            cv2.rectangle(annotated_frame, (0, 0), (640, 60), (0, 0, 0), -1)
            cv2.putText(annotated_frame, self.status_msg, (10, 35), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

            # 5. 객체 탐색 로직
            if results and not self.is_moving and self.task_step in ["cup", "bottle"] and self.target_object:
                boxes = results[0].boxes
                
                # [추가] 디버깅: 탐색 중인데 목표물이 안 보이면 로그 출력 (2초마다)
                if len(boxes) == 0:
                    self.get_logger().info(f"👀 탐색 중... 인식된 객체 없음 (목표: {self.target_object})", throttle_duration_sec=2.0)
                else:
                    detected_names = [self.model.names[int(b.cls[0])] for b in boxes]
                    if self.target_object not in detected_names:
                        self.get_logger().info(f"👀 탐색 중... 다른 객체만 보임: {detected_names} (목표: {self.target_object})", throttle_duration_sec=2.0)

                for box in boxes:
                    cls_id = int(box.cls[0])
                    cls_name = self.model.names[cls_id]
                    
                    if cls_name == self.target_object:
                        x1, y1, x2, y2 = map(int, box.xyxy[0])
                        
                        if self.task_step == "bottle":
                            cx = (x1 + x2) // 2
                            cy = int(y1 + (y2 - y1) * 0.65)
                        else:
                            cx = (x1 + x2) // 2
                            cy = (y1 + y2) // 2 
                        
                        # 거리 측정 (통계 기반 depth 보정)
                        # 튜닝 포인트:
                        # - window_radius: 3~7 픽셀 권장
                        # - std_threshold: 0.02~0.05m 권장 (센서/환경 따라 조정)
                        dist, depth_stats = estimate_depth_from_window(
                            depth_image,
                            center_xy=(cx, cy),
                            window_radius=5,
                            std_threshold=0.03,
                            k=1.0,
                            min_inliers=8,
                            depth_scale=self.depth_scale,  # z16 raw -> meters
                            min_depth=0.1,
                            max_depth=1.2,
                            reducer="median",
                            fallback_reducer="median",
                            prefer_near_cluster=True,
                        )

                        if dist > 0:
                            cv2.putText(annotated_frame, f"Dist: {dist:.3f}m", (x1, y1-20), 
                                      cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                            cv2.circle(annotated_frame, (cx, cy), 5, (0, 0, 255), -1)

                        print(f"Detected object: {self.target_object}, Distance: {dist:.3f}m")

                        # 인식 범위 내 들어오면 이동 시작
                        if 0.1 < dist < 1.2:
                            try:
                                self.get_logger().info(
                                    "📏 Depth stats: "
                                    f"mean={float(depth_stats['mean']):.3f}m, "
                                    f"std={float(depth_stats['std']):.3f}m, "
                                    f"samples={int(depth_stats['num_samples'])}, "
                                    f"inliers={int(depth_stats['num_inliers'])}"
                                )
                            except Exception:
                                pass

                            cam_point = rs.rs2_deproject_pixel_to_point(intrinsics, [cx, cy], dist)
                            c_x = cam_point[0] * 1000.0
                            c_y = cam_point[1] * 1000.0
                            c_z = cam_point[2] * 1000.0
                            
                            gripper_pos = np.dot(self.calib_matrix, np.array([c_x, c_y, c_z, 1.0]))
                            gx, gy, gz = gripper_pos[0], gripper_pos[1], gripper_pos[2]
                            self.get_logger().info(
                                f"🎯 Vision(mm): cx,cy=({cx},{cy}) cam=[{c_x:.1f},{c_y:.1f},{c_z:.1f}] -> "
                                f"tool=[{gx:.1f},{gy:.1f},{gz:.1f}]"
                            )
                            
                            self.detected_pose = (gx, gy, gz)
                            self.is_moving = True
                            break

            # 6. 이미지 퍼블리시 및 출력
            try:
                msg = self.br.cv2_to_imgmsg(annotated_frame, encoding="bgr8")
                self.pub_img.publish(msg)
            except Exception as e:
                pass # 퍼블리시 에러는 무시해도 됨
            
            cv2.imshow("Bartender Vision", annotated_frame)
            if cv2.waitKey(1) == 27: rclpy.shutdown()

        except Exception as e:
            # 프레임이 없거나 일시적 에러는 치명적이지 않음
            self.get_logger().warn(f"Vision Loop Error: {e}")
            if annotated_frame is not None:
                try:
                    self.pub_img.publish(self.br.cv2_to_imgmsg(annotated_frame, encoding="bgr8"))
                except: pass

                cv2.imshow("Bartender Vision", annotated_frame)
                if cv2.waitKey(1) == 27: rclpy.shutdown()

    # --- 동작 로직 ---
    def process_vision_signal(self):
        if self.detected_pose is not None:
            gx, gy, gz = self.detected_pose
            self.detected_pose = None
            self.execute_eye_in_hand_move(gx, gy, gz)

    def execute_eye_in_hand_move(self, offset_x, offset_y, offset_z):
        self.is_moving = True
        self.bottle_approach_dist = offset_z
        self.status_msg = "Eye-in-Hand Aligning..."
        
        # 병별 파라미터 로드
        params = self.bottle_params.get(self.target_object, self.bottle_params["default"])
        
        # [test_bottle.py 로직] 병 작업 시 높이(Z) 유지 (Base Z=360)
        if self.task_step == "bottle":
            self.get_logger().info(f"🍾 병 정렬: 높이 유지(Z=360)를 위해 Y축 이동 제거. 원본 Y={offset_y:.1f}")
            offset_y = 0.0
            
            # XY 오프셋 보정 적용
            offset_x += params["off_x"]
            offset_y += params["off_y"]
            self.get_logger().info(f"🔧 보정 적용({self.target_object}): X+={params['off_x']}, Y+={params['off_y']}")

        # ★ 저장: 나중에 병 놓으러 올 때 사용
        self.saved_vision_offset = [offset_x, offset_y]

        self.get_logger().info(f"🎯 XY 상대 이동: X={offset_x:.1f}, Y={offset_y:.1f}")

        # [1] XY 정렬 (그리퍼 기준 상대 이동)
        req = MoveLine.Request()
        req.pos = [float(offset_x), float(offset_y), 0.0, 0.0, 0.0, 0.0] 
        req.vel = [100.0, 0.0]; req.acc = [100.0, 0.0]
        req.ref = 1; req.mode = 1
        # req.sync_type = 1 # [수정] 동기 모드 명시 (이동 완료 후 콜백 실행)
        req.sync_type = 0 # [수정] 동기 모드 명시 (이동 완료 후 콜백 실행)

        future = self.move_line_client.call_async(req)
        
        # [2] 하강 (Base 기준 절대 높이 차이 계산)
        if self.task_step == "cup":
            target_pick_z = float(self.cup_place_target_z.get(self.target_object, 85.0))
            descend_dist = target_pick_z - float(self.CURRENT_Z_HEIGHT)
            self.get_logger().info(f"🍺 컵 하강 준비 (Diff: {descend_dist:.1f})")
            future.add_done_callback(lambda f: self.descend_vertical(f, descend_dist))
        elif self.task_step == "bottle":
            self.get_logger().info("🍾 병 접근 준비 (XY 정렬 후 전진)")
            future.add_done_callback(self.approach_bottle)

    def descend_vertical(self, future=None, z_diff: float = 0.0):
        # [test_bottle.py 로직] future 결과 확인
        if future is not None:
            try:
                if not future.result().success:
                    self.get_logger().warn("⚠️ XY 정렬 실패 (MoveLine Error)")
                    self.reset_state()
                    return
            except Exception as e:
                self.get_logger().warn(f"⚠️ XY 정렬 결과 확인 실패: {e}")
                self.reset_state()
                return

        self.status_msg = "Descending..."
        self.get_logger().info(f"⬇️ 하강 시작: Base Z {float(z_diff):.1f}mm (relative)")
        
        # Base 기준(ref=0) Relative(mode=1) 이동 -> Z축 수직 하강
        req = MoveLine.Request()
        req.pos = [0.0, 0.0, float(z_diff), 0.0, 0.0, 0.0]
        req.vel = [50.0, 0.0]; req.acc = [50.0, 0.0]
        req.ref = 0; req.mode = 1 
        req.sync_type = 1 # [수정] 동기 모드 명시
        req.sync_type = 0 # [수정] 동기 모드 명시
        
        f = self.move_line_client.call_async(req)
        if self.task_step == "bottle":
            f.add_done_callback(self.approach_bottle)
        else:
            f.add_done_callback(self.after_approach)

    def approach_bottle(self, future=None):
        if future is None or (hasattr(future, 'result') and future.result().success):
            self.status_msg = "Descending..."
            
            if not hasattr(self, 'bottle_approach_dist') or self.bottle_approach_dist is None:
                self.get_logger().error("⚠️ 접근 거리 정보 없음. 병 접근 실패.")
                self.reset_state()
                return

            # 병별 마진 적용하여 이동 거리 계산
            params = self.bottle_params.get(self.target_object, self.bottle_params["default"])
            margin = params["margin"]
            dist = self.bottle_approach_dist - margin
            
            # ★ 저장: 나중에 병 놓으러 올 때 사용
            self.saved_approach_dist = dist
            
            self.get_logger().info(f"🍾 병 접근 전진: {dist:.1f}mm")
            
            req = MoveLine.Request()
            req.pos = [0.0, 0.0, float(dist), 0.0, 0.0, 0.0] # Tool Z축 전진
            req.vel = [50.0, 0.0]; req.acc = [50.0, 0.0]
            req.ref = 1; req.mode = 1 
            req.sync_type = 1 # [수정] 동기 모드 명시
            req.sync_type = 0 # [수정] 동기 모드 명시
            
            f = self.move_line_client.call_async(req)
            f.add_done_callback(self.after_approach)
        else:
            self.reset_state()

    def after_approach(self, future):
        if future.result().success:
            self.status_msg = "Gripping..."
            gripper.close_gripper()
            time.sleep(1.0) # [추가] 그립 안정화 시간
            
            # [수정] 병일 경우 현재 위치(Pick 좌표)를 저장한 뒤 상승
            if self.task_step == "bottle":
                self.get_logger().warn("if: get_current_pose_and_lift 호출")
                self.get_current_pose_and_lift()
            else:
                self.get_logger().warn("else: lift_object 호출")
                self.lift_object()
        else:
            self.get_logger().warn("❌ 접근(Approach) 실패 - 이동 불가")
            self.reset_state()

    def get_current_pose_and_lift(self):
        """병을 잡은 위치를 저장하고 상승합니다."""
        self.bottle_origin_pos = None # 초기화
        if not self.get_pos_client.wait_for_service(timeout_sec=3.0):
            self.get_logger().warn("⚠️ Pose 저장 실패(서비스 미연결). 기존 로직대로 상승합니다.")
            self.lift_object()
            return

        req = GetCurrentPos.Request()
        req.space_type = 1 # Task Space
        future = self.get_pos_client.call_async(req)
        future.add_done_callback(self.save_bottle_pos_and_lift)

    def retract_then_lift_bottle(self, retract_mm: float = 60.0):
        """병 파지 직후, 툴 기준으로 살짝 후진한 뒤 안전 높이로 올립니다."""
        self.status_msg = "Retracting..."
        self.get_logger().info(f"🍾 파지 후 상승: Tool Z -{retract_mm:.1f}mm")

        req = MoveLine.Request()
        req.pos = [0.0, 0.0, -float(retract_mm), 0.0, 0.0, 0.0]
        req.vel = [80.0, 0.0]; req.acc = [80.0, 0.0]
        req.ref = 1; req.mode = 1  # Tool Relative
        f = self.move_line_client.call_async(req)
        f.add_done_callback(self._after_bottle_retract)

    def _after_bottle_retract(self, future):
        # [수정] 1차로 Base 절대 Z=580까지 상승을 시도하되,
        # 작업영역/관절 제한 등으로 실패(또는 무시)될 수 있으므로 실패 시 Z=420으로 fallback 합니다.
        primary_z = 580.0
        fallback_z = 420.0
        self.lift_bottle_to_safe_z(
            primary_z,
            next_cb=lambda lift_fut: self._after_bottle_lift_attempt(lift_fut, primary_z, fallback_z),
        )

    def _after_bottle_lift_attempt(self, future, primary_z: float, fallback_z: float):
        """병 상승(primary_z) 시도 후, 결과를 확인하고 필요 시 fallback_z로 재시도합니다."""
        try:
            ok = future.result().success
        except Exception as e:
            self.get_logger().warn(f"⚠️ 병 상승 결과 확인 실패(Z={primary_z:.1f}). fallback 시도. err={e}")
            ok = False

        if not ok:
            self.get_logger().warn(f"⚠️ 병 상승 실패(Z={primary_z:.1f}). fallback Z={fallback_z:.1f} 시도")
            self.lift_bottle_to_safe_z(fallback_z, next_cb=self.move_to_joint_home_before_pour)
            return

        # success가 true라도 실제 이동이 충분하지 않을 수 있어 pose를 한 번 더 확인합니다.
        if not self.get_pos_client.wait_for_service(timeout_sec=0.2):
            self.get_logger().warn("⚠️ get_current_pose 미연결. 검증 없이 Joint Home으로 진행합니다.")
            self.move_to_joint_home_before_pour(future)
            return

        req = GetCurrentPos.Request()
        req.space_type = 1  # ROBOT_SPACE_TASK
        f = self.get_pos_client.call_async(req)
        f.add_done_callback(lambda pfut: self._verify_bottle_lift(pfut, primary_z, fallback_z, future))

    def _verify_bottle_lift(self, pose_future, primary_z: float, fallback_z: float, lift_future):
        """primary lift 후 현재 Z를 확인하고, 부족하면 fallback lift를 시도합니다."""
        try:
            res = pose_future.result()
        except Exception as e:
            self.get_logger().warn(f"⚠️ 병 상승 검증용 pose 조회 실패. Joint Home으로 진행합니다. err={e}")
            self.move_to_joint_home_before_pour(lift_future)
            return

        if not getattr(res, "success", False):
            self.get_logger().warn("⚠️ 병 상승 검증용 pose 조회 실패(success=false). Joint Home으로 진행합니다.")
            self.move_to_joint_home_before_pour(lift_future)
            return

        current_z = float(res.pos[2])
        # 목표보다 너무 낮으면(>30mm 차이) fallback 수행
        if current_z < (primary_z - 30.0):
            self.get_logger().warn(
                f"⚠️ 병 상승 검증 실패: 현재 Z={current_z:.1f} < {primary_z - 30.0:.1f}. "
                f"fallback Z={fallback_z:.1f}로 재상승 후 진행합니다."
            )
            self.lift_bottle_to_safe_z(fallback_z, next_cb=self.move_to_joint_home_before_pour)
            return

        self.get_logger().info(f"✅ 병 상승 검증 OK: 현재 Z={current_z:.1f} (목표 {primary_z:.1f})")
        self.move_to_joint_home_before_pour(lift_future)

    def save_bottle_pos_and_lift(self, future):
        try:
            res = future.result()
            if res.success:
                self.bottle_origin_pos = res.pos 
                self.get_logger().info(f"💾 병 Pick 위치 저장: X={res.pos[0]:.1f}, Y={res.pos[1]:.1f}, Z={res.pos[2]:.1f}")
            else:
                self.get_logger().warn("⚠️ Pose 저장 실패(Result Fail).")
        except Exception as e: self.get_logger().warn(f"⚠️ Pose 저장 에러: {e}")
        self.lift_object()

    def lift_object(self):
        self.status_msg = "Lifting..."

        # [수정] test_bottle.py와 동일하게 병 작업 시 바로 절대 높이 상승 시도
        if self.task_step == "bottle":
            # [수정] get_current_pose 서비스 호출 지연 방지를 위해 고정 높이(220mm) 상대 상승
            self.get_logger().info("🍾 병 상승: 서비스 호출 없이 바로 상대 상승 (+220mm)")
            self._lift_relative_z(220.0, next_cb=self.move_to_intermediate_waypoint_1_before_pour)
            return

        # 컵 작업 시 (Base Relative)
        req = MoveLine.Request()
        req.pos = [0.0, 0.0, 200.0, 0.0, 0.0, 0.0] # 컵은 너무 높지 않게 200mm만 상승
        req.vel = [100.0, 0.0]; req.acc = [100.0, 0.0]
        req.ref = 0; req.mode = 1 # Base Relative

        self.get_logger().info("🚀 컵 상승: Base Z +200.0mm")

        future = self.move_line_client.call_async(req)
        if self.task_step == "cup":
            future.add_done_callback(self.move_to_joint_waypoint)
        else:
            future.add_done_callback(self.go_to_pour_position)

    def lift_bottle_to_safe_z(self, safe_z: float = 580.0, next_cb=None):
        """병을 잡은 뒤, 현재 자세에서 Base Z를 safe_z(절대)까지 올립니다."""
        if next_cb is None: next_cb = self.go_to_pour_position
        # GetCurrentPose: 0=JOINT, 1=TASK
        if not self.get_pos_client.wait_for_service(timeout_sec=0.5):
            # 서비스가 없으면(또는 일시적으로 못 받으면) "현재가 대략 Bottle View 높이"라는 가정으로 상승
            fallback_base_z = float(self.BOTTLE_VIEW_POS[2])
            fallback_dz = max(0.0, float(safe_z) - fallback_base_z)
            self.get_logger().warn(
                f"⚠️ get_current_pose 서비스 미연결. 병 상승을 Base Z +{fallback_dz:.1f}mm로 대체합니다."
            )
            self._lift_relative_z(fallback_dz, next_cb=next_cb)
            return

        req = GetCurrentPos.Request()
        req.space_type = 1  # ROBOT_SPACE_TASK
        f = self.get_pos_client.call_async(req)
        f.add_done_callback(lambda fut: self._on_bottle_pose_for_lift(fut, float(safe_z), next_cb))

    def _on_bottle_pose_for_lift(self, future, safe_z: float, next_cb):
        try:
            res = future.result()
        except Exception as e:
            self.get_logger().error(f"❌ 현재 pose 조회 실패. 병 상승을 기본값으로 수행합니다. err={e}")
            fallback_dz = max(0.0, safe_z - 360.0)
            self._lift_relative_z(fallback_dz, next_cb=next_cb)
            return

        if not getattr(res, "success", False):
            self.get_logger().error("❌ 현재 pose 조회 실패(success=false). 병 상승을 기본값으로 수행합니다.")
            fallback_dz = max(0.0, safe_z - 360.0)
            self._lift_relative_z(fallback_dz, next_cb=next_cb)
            return

        current_x = float(res.pos[0])
        current_y = float(res.pos[1])
        current_z = float(res.pos[2])
        current_rx = float(res.pos[3])
        current_ry = float(res.pos[4])
        current_rz = float(res.pos[5])
        dz = safe_z - current_z
        if dz < 0.0:
            # 이미 safe_z보다 높은 경우에는 더 올릴 필요가 없으니 0으로 클램프
            dz = 0.0

        self.get_logger().info(
            f"🚀 병 상승: 현재[X,Y,Z]=[{current_x:.1f},{current_y:.1f},{current_z:.1f}] "
            f"R=[{current_rx:.1f},{current_ry:.1f},{current_rz:.1f}] -> 목표 Z={safe_z:.1f} (dZ={dz:.1f}mm)"
        )
        self._lift_relative_z(dz, next_cb=next_cb)

    def _lift_relative_z(self, dz: float, next_cb=None):
        """Base 기준 상대 Z 이동(dz) 후 next_cb로 체인을 이어갑니다."""
        time.sleep(0.5) # [추가] 연속 명령 시 동작 씹힘 방지를 위한 대기
        self.get_logger().info(f"🚀 상승 명령: Base Z +{float(dz):.1f}mm")
        req = MoveLine.Request()
        req.pos = [0.0, 0.0, float(dz), 0.0, 0.0, 0.0]
        req.vel = [100.0, 0.0]; req.acc = [100.0, 0.0]
        req.time = 2.0 # [추가] 속도 대신 시간을 지정하여 동작 생략 방지 (2초 동안 이동)
        req.ref = 0; req.mode = 1  # Base Relative
        req.sync_type = 1 # [추가] 명시적 동기 모드 (동작 완료 후 리턴)

        f = self.move_line_client.call_async(req)
        f.add_done_callback(lambda fut: self._log_move_result(fut, "Lift(Base Z)"))
        if next_cb is not None:
            # [수정] sync_type=0이므로 동작 완료 후 리턴됨. 추가 대기는 짧게 설정.
            f.add_done_callback(lambda fut: self._wait_and_execute(fut, 0.1, next_cb))

    def _wait_and_execute(self, future, wait_time, next_cb):
        """서비스 응답 후 wait_time만큼 대기했다가 next_cb를 실행합니다."""
        try:
            # [수정] 동작 실패 시 체인 중단 (성공 여부 확인)
            if not future.result().success:
                self.get_logger().error("❌ 동작 실패(MoveLine Fail). 다음 단계로 진행하지 않습니다.")
                self.reset_state()
                return

            # 성공 시 시간 지연 후 다음 동작
            threading.Timer(wait_time + 0.1, next_cb, args=[future]).start()
        except Exception as e:
            self.get_logger().error(f"Timer Error: {e}")
            self.reset_state()

    def _log_move_result(self, future, label: str):
        try:
            ok = future.result().success
        except Exception as e:
            self.get_logger().warn(f"⚠️ {label} 결과 확인 실패: {e}")
            return

        if ok:
            self.get_logger().info(f"✅ {label} 완료")
        else:
            self.get_logger().error(f"❌ {label} 실패")

    # --- [수정된 부분] 컵 내려놓기 시퀀스 ---
    def move_to_joint_waypoint(self, future):
        if future.result().success:
            self.status_msg = "Moving to Waypoint..."
            req = MoveJoint.Request()
            req.pos = self.JOINT_HOME_POS 
            req.vel = 50.0; req.acc = 30.0
            
            # ★ 오타 수정: move_joint -> move_joint_client
            f = self.move_joint_client.call_async(req)
            f.add_done_callback(self.go_to_cup_ready_pos)
        else: self.reset_state()

    def go_to_cup_ready_pos(self, future):
        if future.result().success:
            self.status_msg = "Approaching Cup Home..."
            
            # 1. 컵 종류에 따른 목표 Z 높이 가져오기
            target_z = self.cup_place_target_z.get(self.target_object, 78.81)
            
            # 2. 안전 높이 설정 (목표 높이 + 50mm 위)
            safe_z = target_z + 50.0
            
            # 3. 목표 좌표 생성 (X, Y는 BASE_HOME 유지, Z만 변경)
            home_pos = list(self.BASE_HOME_POS)
            home_pos[2] = safe_z
            
            req = MoveLine.Request()
            req.pos = [float(x) for x in home_pos]
            req.vel = [100.0, 0.0]; req.acc = [100.0, 0.0]
            req.ref = 0; req.mode = 0 # Base Absolute
            
            f = self.move_line_client.call_async(req)
            # 도착 후 실제 바닥으로 하강
            f.add_done_callback(lambda f: self.descend_to_place_cup(f, target_z))
        else: self.reset_state()

    def descend_to_place_cup(self, future, target_z):
        if future.result().success:
            self.status_msg = "Placing Cup..."
            self.get_logger().info(f"⬇️ 컵 배치 하강: Z -> {target_z}")
            
            # 절대 좌표 Z로 정밀 하강
            place_pos = list(self.BASE_HOME_POS)
            place_pos[2] = target_z
            
            req = MoveLine.Request()
            req.pos = [float(x) for x in place_pos]
            req.vel = [30.0, 0.0]; req.acc = [30.0, 0.0] # 천천히
            req.ref = 0; req.mode = 0
            
            f = self.move_line_client.call_async(req)
            f.add_done_callback(self.finish_cup_task)
        else: self.reset_state()

    def finish_cup_task(self, future):
        if future.result().success:
            self.get_logger().info("✅ 컵 배치 완료. 그리퍼 해제")
            
            self.report_progress("컵 배치")

            gripper.open_gripper()
            
            # 안전하게 위로 빠져나오기 (Base 기준 +Z 100mm 상승)
            self.status_msg = "Retracting..."
            req = MoveLine.Request()
            req.pos = [0.0, 0.0, 100.0, 0.0, 0.0, 0.0]
            req.vel = [100.0, 0.0]; req.acc = [100.0, 0.0]
            req.ref = 0; req.mode = 1 # Relative
            
            f = self.move_line_client.call_async(req)
            # 상승 후 병 작업 시작
            f.add_done_callback(lambda f: threading.Timer(1.0, self.start_bottle_sequence).start())
        else: self.reset_state()

    # --- Bottle Task Chain ---
    def start_bottle_sequence(self):
        self.task_step = "bottle_transit"
        self.status_msg = "Moving to Joint Home..."
        self.get_logger().info(f"🍾 병 시퀀스 시작 (Index: {self.liquor_idx}): Joint Home 이동")
        req = MoveJoint.Request()
        req.pos = self.JOINT_HOME_POS
        req.vel = 50.0; req.acc = 30.0
        f = self.move_joint_client.call_async(req)
        f.add_done_callback(self.move_to_bottle_view)

    def move_to_bottle_view(self, future=None):
        if future is None or (hasattr(future, 'result') and future.result().success):
            self.status_msg = "Moving to Bottle View..."
            # 요청: 병 탐색 위치로 바로 이동 (상공 Z=580 경유 제거)
            self.get_logger().info("🍾 Bottle View로 바로 이동...")
            req = MoveLine.Request()
            req.pos = [float(x) for x in self.BOTTLE_VIEW_POS]
            req.vel = [100.0, 0.0]; req.acc = [100.0, 0.0]
            req.ref = 0; req.mode = 0
            f = self.move_line_client.call_async(req)
            f.add_done_callback(self.start_bottle_search)
        else:
            self.get_logger().error("❌ 이전 이동 실패")
            self.reset_state()

    def descend_to_bottle_view(self, future):
        if future.result().success:
            self.get_logger().info(f"🍾 Bottle View 위치(Z=360)로 하강...")
            req = MoveLine.Request()
            req.pos = self.BOTTLE_VIEW_POS
            req.vel = [100.0, 0.0]; req.acc = [100.0, 0.0]
            req.ref = 0; req.mode = 0
            f = self.move_line_client.call_async(req)
            f.add_done_callback(self.start_bottle_search)
        else:
            self.reset_state()

    def start_bottle_search(self, future):
        if future.result().success:
            self.CURRENT_Z_HEIGHT = 360.0
            liquors = self.current_recipe.get('liquors', [])
            if self.liquor_idx < len(liquors):
                bottle_name = liquors[self.liquor_idx]['name']
                self.target_object = bottle_name
                
                self.report_progress(f"병 픽업 ({bottle_name})")

                self.task_step = "bottle"
                self.saved_vision_offset = [0.0, 0.0]
                self.saved_approach_dist = 0.0
                self.status_msg = f"Search: {bottle_name}"
                self.is_moving = False
                self.get_logger().info(f"🍾 병 찾기: {bottle_name}")
            else:
                self.get_logger().info("🍾 모든 병 처리 완료")
                self.finish_all_tasks()
        else:
            self.get_logger().error("❌ Bottle View 이동 실패")
            self.reset_state()

    def move_to_joint_home_before_pour(self, future):
        if future.result().success:
            self.status_msg = "Homing..."
            self.get_logger().info("🍾 붓기 전 홈 위치(Joint Home)로 이동")
            req = MoveJoint.Request()
            req.pos = self.JOINT_HOME_POS
            req.vel = 60.0; req.acc = 40.0
            f = self.move_joint_client.call_async(req)
            f.add_done_callback(self.go_to_pour_position)
        else:
            self.reset_state()

    def go_to_pour_position(self, future):
        if future.result().success:
            self.status_msg = "Moving to Pour..."
            
            # 1. 현재 컵 정보 및 Z 높이 확인
            cup_name = "black_cup"
            if self.current_recipe:
                cup_name = self.current_recipe.get("cup", "black_cup")
            cup_z = float(self.cup_place_target_z.get(cup_name, 85.0))

            # 2. 병 종류에 따른 좌표 설정
            # blue_bottle은 크기가 커서 별도 좌표 사용 (yellow_cup Z=50.0 기준)
            if self.target_object == "blue_bottle":
                ref_cup_z = 50.0
                z_diff = cup_z - ref_cup_z
                
                if cup_name == "green_cup":
                    z_diff += 50.0
                elif cup_name == "black_cup":
                    z_diff += 50.0
                
                # yellow_cup 기준 좌표 + Z 보정
                self.pour_start_pos = [397.64, -81.31, 91.66 + z_diff, 30.86, -174.47, 27.07]
                self.pour_end_pos   = [447.88, -51.92, 127.59 + z_diff, 104.87, -148.17, 67.24]
                
                self.get_logger().info(f"🍷 Blue Bottle Special Pour: Cup({cup_name}, Z={cup_z}) -> Z Offset={z_diff:.1f}")
            
            else:
                # 기존 로직 (black_bottle, purple_bottle 등)
                base_ref_cup_z = 85.0
                # [수정] Yellow Cup(Z=50) 실측 보정: 183.68 -> 126.98 (Diff: -56.7)
                # 기존 pour_extra_z(50.0) - 56.7 = -6.7
                pour_extra_z = -6.7
                
                if cup_name == "green_cup":
                    if self.target_object == "purple_bottle":
                        pour_extra_z -= 50.0
                    else:
                        pour_extra_z += 50.0

                pour_start_z = cup_z + (146.83 - base_ref_cup_z) + pour_extra_z
                pour_end_z = cup_z + (168.70 - base_ref_cup_z) + pour_extra_z

                self.pour_start_pos = [400.55, -41.65, float(pour_start_z), 33.90, -174.78, 29.70]
                self.pour_end_pos = [429.46, -18.07, float(pour_end_z), 112.55, -140.10, 67.13]

                self.get_logger().info(
                    f"🍷 Standard Pour: Cup({cup_name}) Z={cup_z:.1f} -> Start Z={pour_start_z:.2f}, End Z={pour_end_z:.2f}"
                )

            self.get_logger().info("🍷 붓기 위치로 이동 시작 (1. 상공 이동)")
            
            # [수정] 안전한 이동을 위해: 상공(Z=580)으로 먼저 수평 이동 후 하강
            high_pour_pos = list(self.pour_start_pos)
            high_pour_pos[2] = 580.0 # 병을 들어올린 높이 유지
            
            req = MoveLine.Request()
            req.pos = [float(x) for x in high_pour_pos]
            req.vel = [100.0, 0.0]; req.acc = [100.0, 0.0]
            req.ref = 0; req.mode = 0
            f = self.move_line_client.call_async(req)
            f.add_done_callback(self.descend_to_pour)
        else:
            self.get_logger().error("❌ 병 상승 실패(또는 이동 불가): Pour 위치로 진행하지 못했습니다.")
            self.reset_state()

    def descend_to_pour(self, future):
        if future.result().success:
            self.get_logger().info("🍷 붓기 높이로 하강 (2. 수직 하강)")
            req = MoveLine.Request()
            req.pos = [float(x) for x in self.pour_start_pos]
            req.vel = [100.0, 0.0]; req.acc = [100.0, 0.0]
            req.ref = 0; req.mode = 0
            f = self.move_line_client.call_async(req)
            f.add_done_callback(self.pour_action)
        else: self.reset_state()

    def pour_action(self, future):
        if future.result().success:
            self.status_msg = "Pouring..."
            self.get_logger().info("🍷 따르기 (기울이기)")
            
            self.report_progress("따르기")
            
            # [추가] pour_time 가져오기
            try:
                pour_time = float(self.current_recipe['liquors'][self.liquor_idx].get('pour_time', 2.0))
            except (IndexError, KeyError, TypeError, ValueError):
                pour_time = 2.0

            # [수정] cup에 맞게 보정된 end pose 사용 (fallback은 test_bottle.py 기준값)
            pour_end_pos = getattr(self, "pour_end_pos", [429.46, -18.07, 168.70, 112.55, -140.10, 67.13])
            req = MoveLine.Request()
            req.pos = pour_end_pos
            req.vel = [60.0, 0.0]; req.acc = [60.0, 0.0]
            req.time = pour_time # [수정] pour_time 동안 이동
            req.ref = 0; req.mode = 0 
            f = self.move_line_client.call_async(req)
            # [수정] 붓는 시간(pour_time)만큼 확실히 대기 후 복귀
            f.add_done_callback(lambda fut: self._wait_and_execute(fut, pour_time, self.wait_and_return))
        else: self.reset_state()

    def wait_and_return(self, future):
        if future.result().success:
            # [수정] 이미 pour_time 동안 이동했으므로 추가 대기는 짧게 설정
            self.get_logger().info(f"⏳ 붓기 완료. 복귀 준비")
            time.sleep(0.5)
            
            self.status_msg = "Returning..."
            # 다시 초기 위치(수평)로 복귀하여 병 세우기
            req = MoveLine.Request()
            req.pos = self.pour_start_pos
            req.vel = [60.0, 0.0]; req.acc = [60.0, 0.0]
            req.ref = 0; req.mode = 0
            f = self.move_line_client.call_async(req)
            f.add_done_callback(self.move_to_intermediate_waypoint_2_after_pour)
        else: self.reset_state()

    def move_to_intermediate_waypoint_1_before_pour(self, future):
        if future.result().success:
            self.status_msg = "Moving to Waypoint 1..."
            self.get_logger().info("🍾 중간 경유지 1로 이동 (Before Pour)")
            req = MoveLine.Request()
            req.pos = self.INTERMEDIATE_WAYPOINT_POS_1
            req.vel = [100.0, 0.0]; req.acc = [100.0, 0.0]
            req.ref = 0; req.mode = 0 # Base Absolute
            req.sync_type = 0
            
            f = self.move_line_client.call_async(req)
            f.add_done_callback(self.move_to_intermediate_waypoint_2_before_pour)
        else:
            self.reset_state()

    def move_to_intermediate_waypoint_2_before_pour(self, future):
        if future.result().success:
            self.status_msg = "Moving to Waypoint 2..."
            self.get_logger().info("🍾 중간 경유지 2로 이동 (Before Pour)")
            req = MoveLine.Request()
            req.pos = self.INTERMEDIATE_WAYPOINT_POS_2
            req.vel = [100.0, 0.0]; req.acc = [100.0, 0.0]
            req.ref = 0; req.mode = 0 # Base Absolute
            req.sync_type = 0
            
            f = self.move_line_client.call_async(req)
            f.add_done_callback(self.go_to_pour_position)
        else:
            self.reset_state()

    def move_to_intermediate_waypoint_2_after_pour(self, future):
        if future.result().success:
            self.status_msg = "Moving to Waypoint 2..."
            self.get_logger().info("🍾 중간 경유지 2로 이동 (After Pour)")
            req = MoveLine.Request()
            req.pos = self.INTERMEDIATE_WAYPOINT_POS_2
            req.vel = [100.0, 0.0]; req.acc = [100.0, 0.0]
            req.ref = 0; req.mode = 0 # Base Absolute
            req.sync_type = 0
            
            f = self.move_line_client.call_async(req)
            f.add_done_callback(self.move_to_intermediate_waypoint_1_after_pour)
        else:
            self.reset_state()

    def move_to_intermediate_waypoint_1_after_pour(self, future):
        if future.result().success:
            self.status_msg = "Moving to Waypoint 1..."
            self.get_logger().info("🍾 중간 경유지 1로 이동 (After Pour)")
            req = MoveLine.Request()
            req.pos = self.INTERMEDIATE_WAYPOINT_POS_1
            req.vel = [100.0, 0.0]; req.acc = [100.0, 0.0]
            req.ref = 0; req.mode = 0 # Base Absolute
            req.sync_type = 0
            
            f = self.move_line_client.call_async(req)
            f.add_done_callback(self.place_bottle_back)
        else:
            self.reset_state()

    def place_bottle_back(self, future):
        if future.result().success:
            self.status_msg = "Returning Bottle..."
            self.get_logger().info("🍾 병 원래 위치로 복귀 시작 (1. 수직 상승)")
            
            self.report_progress("병 반납")

            # [수정] pour Z가 cup마다 달라지므로(+50mm 포함),
            # 상대 +350mm 대신 Base 절대 Z=580까지 올려 상공을 보장합니다.
            self.lift_bottle_to_safe_z(580.0, next_cb=self.move_to_bottle_origin_high)
        else: self.reset_state()

    def move_to_bottle_origin_high(self, future):
        if future.result().success:
            self.get_logger().info("🍾 상공으로 이동 (2. 수평 이동)")
            
            # [수정] 저장된 Pick 위치가 있으면 그 좌표 사용, 없으면 기본 View Pos 사용
            if self.bottle_origin_pos is not None:
                high_pos = list(self.bottle_origin_pos)
                self.get_logger().info(f"📍 저장된 병 좌표 사용: X={high_pos[0]:.1f}, Y={high_pos[1]:.1f}")
            else:
                high_pos = list(self.BOTTLE_VIEW_POS)
            
            # Z는 안전 높이 580 고정
            high_pos[2] = 580.0
            
            req = MoveLine.Request()
            req.pos = [float(x) for x in high_pos]
            req.vel = [100.0, 0.0]; req.acc = [100.0, 0.0]
            req.ref = 0; req.mode = 0
            req.sync_type = 1 # [추가] 동기 모드(1)로 설정하여 이동 완료 대기 시도
            f = self.move_line_client.call_async(req)
            
            # [수정] 이미 정확한 X,Y로 이동했으므로 align/approach 단계 건너뛰고 바로 하강
            if self.bottle_origin_pos is not None:
                f.add_done_callback(self.descend_to_place_bottle)
            else:
                f.add_done_callback(self.place_bottle_align_high)
        else: self.reset_state()

    def place_bottle_align_high(self, future):
        if future.result().success:
            # 3. Vision Offset 적용 (Tool Relative) - 상공에서 수행
            off_x, off_y = self.saved_vision_offset
            self.get_logger().info(f"🍾 상공 위치 보정: X={off_x:.1f}, Y={off_y:.1f}")
            req = MoveLine.Request()
            req.pos = [float(off_x), float(off_y), 0.0, 0.0, 0.0, 0.0]
            req.vel = [100.0, 0.0]; req.acc = [100.0, 0.0]
            req.ref = 1; req.mode = 1
            f = self.move_line_client.call_async(req)
            f.add_done_callback(self.place_bottle_approach_high)
        else: self.reset_state()

    def place_bottle_approach_high(self, future):
        if future.result().success:
            # 4. 상공에서 접근 (Approach) - XY 이동
            dist = self.saved_approach_dist
            self.get_logger().info(f"🍾 상공 접근 (XY 이동): {dist:.1f}mm")
            req = MoveLine.Request()
            req.pos = [0.0, 0.0, float(dist), 0.0, 0.0, 0.0]
            req.vel = [100.0, 0.0]; req.acc = [100.0, 0.0]
            req.ref = 1; req.mode = 1 # Tool Relative
            f = self.move_line_client.call_async(req)
            f.add_done_callback(self.descend_to_place_bottle)
        else: self.reset_state()

    def descend_to_place_bottle(self, future):
        if future.result().success:
            # [수정] 저장된 Pick 높이(Z)로 정확히 하강
            if self.bottle_origin_pos is not None:
                target_z = self.bottle_origin_pos[2]
                self.get_logger().info(f"🍾 수직 하강 (Pick 높이 Z={target_z:.1f}로 복귀)")
                # X,Y,R,P,Y는 현재 유지, Z만 변경 (Absolute)
                target_pos = list(self.bottle_origin_pos)
                req = MoveLine.Request()
                req.pos = [float(x) for x in target_pos]
                req.vel = [50.0, 0.0]; req.acc = [50.0, 0.0]
                req.ref = 0; req.mode = 0 # Absolute
            else:
                self.get_logger().info("🍾 수직 하강 (Z=580 -> 360)")
                req = MoveLine.Request()
                req.pos = [0.0, 0.0, -220.0, 0.0, 0.0, 0.0] # Base Relative Z down
                req.vel = [50.0, 0.0]; req.acc = [50.0, 0.0]
                req.ref = 0; req.mode = 1 # Relative
            
            f = self.move_line_client.call_async(req)
            f.add_done_callback(self.release_bottle)
        else: self.reset_state()

    def release_bottle(self, future):
        if future.result().success:
            self.get_logger().info("🍾 병 놓기 (Release)")
            gripper.open_gripper()
            time.sleep(0.5)
            
            # 4. 후진 (Retract) - 접근했던 거리만큼 뒤로
            dist = -self.saved_approach_dist
            self.get_logger().info(f"🍾 후진: {dist:.1f}mm")
            
            req = MoveLine.Request()
            req.pos = [0.0, 0.0, float(dist), 0.0, 0.0, 0.0]
            req.vel = [100.0, 0.0]; req.acc = [100.0, 0.0]
            req.ref = 1; req.mode = 1
            f = self.move_line_client.call_async(req)
            f.add_done_callback(self.next_bottle)

    def next_bottle(self, future):
        self.liquor_idx += 1
        self.get_logger().info(f"🍾 다음 병 준비 (Index: {self.liquor_idx})")
        self.move_to_bottle_view(future)
        time.sleep(0.2)

    def finish_all_tasks(self):
        self.status_msg = "All Done. Homing..."
        
        # [수정] 작업 종료 시 상태를 초기화하여 비전 루프가 다시 도는 것을 방지 (바닥 충돌 해결)
        self.task_step = "idle"
        self.target_object = None
        
        req = MoveJoint.Request()
        req.pos = self.JOINT_HOME_POS
        req.vel = 50.0; req.acc = 30.0
        self.move_joint_client.call_async(req)
        
        # RealSense 해제 (shake_node에서 사용할 수 있도록)
        self.stop_camera()
        # Action 완료 이벤트 설정
        self.action_event.set()
        self.reset_state()

    def reset_state(self):
        time.sleep(1.0)
        self.is_moving = False
        self.status_msg = "Ready"
        if self.task_step == "idle": self.current_recipe = None

    def destroy_node(self):
        self.stop_camera()
        cv2.destroyAllWindows()
        super().destroy_node()

def main(args=None):
    rclpy.init(args=args)
    import DR_init
    DR_init.__dsr__id = "dsr01"
    DR_init.__dsr__model = "m0609"

    node = BartenderNode()
    DR_init.__dsr__node = node

    from DSR_ROBOT2 import get_tcp

    current_tcp = get_tcp()
    if current_tcp != ROBOT_TCP:
        print(f"⚠️  TCP 불일치 경고: 현재={current_tcp}, 예상={ROBOT_TCP}")
        print(f"⚠️  계속 실행합니다. 문제가 있으면 ROBOT_TCP를 '{current_tcp}'로 수정하세요.")
    else:
        print(f"✅ TCP 확인: {current_tcp}")

    # [수정] Action Server(Blocking Callback)와 Service Callback 동시 처리를 위해 MultiThreadedExecutor 사용
    executor = MultiThreadedExecutor()
    executor.add_node(node)
    try:
        while rclpy.ok():
            executor.spin_once(timeout_sec=0.001)
            node.timer_callback()
            node.process_vision_signal()
    except KeyboardInterrupt: pass
    finally:
        node.destroy_node()
        if rclpy.ok(): rclpy.shutdown()

if __name__ == "__main__":
    main()
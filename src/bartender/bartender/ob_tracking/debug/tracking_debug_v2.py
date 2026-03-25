"""
tracking_debug_v2.py - 디버깅용 사람 추적 코드 (v2)

v1 대비 추가된 기능:
- 화면 3개 구역 분할 (Zone 1, 2, 3)
- 구역별 사람 수 카운트
- ZONE_POSITIONS 로봇 좌표 매핑
- 구역별 색상 구분 시각화
- ROS2 토픽: /zone_status, /active_zone, /zone_robot_pos

실행 방법:
    ros2 run bartender tracking

종료: 'q' 키
"""

import time
import cv2
from ultralytics import YOLO

# ROS2 imports
import rclpy
from rclpy.node import Node
from std_msgs.msg import Bool, Int32, Float32MultiArray, Int32MultiArray


# =============================================================================
# 로봇팔 구역별 좌표 (미리 캘리브레이션된 값)
# -----------------------------------------------------------------------------
# 각 구역에 잔을 놓을 때 로봇팔이 이동할 좌표
# 형식: [x, y, z, rx, ry, rz]
# =============================================================================
ZONE_POSITIONS = {
    1: [436.33, -245.0, 56.0, 29.08, 180.0, 29.02],   # 구역1 (화면 왼쪽)
    2: [318.21, -245.0, 56.0, 29.08, 180.0, 29.02],   # 구역2 (화면 중앙)
    3: [208.26, -245.0, 56.0, 29.08, 180.0, 29.02],   # 구역3 (화면 오른쪽)
}


# =============================================================================
# 구역 판단 함수
# =============================================================================
def get_zone_from_bbox(bbox, frame_width):
    """바운딩 박스의 중심점을 기준으로 구역 판단

    Args:
        bbox: (x1, y1, x2, y2) 바운딩 박스 좌표
        frame_width: 프레임 너비 (픽셀)

    Returns:
        zone: 구역 번호 (1, 2, 3)

    화면 분할:
        |--- 구역1 ---|--- 구역2 ---|--- 구역3 ---|
        0           1/3          2/3          width
    """
    x1, y1, x2, y2 = bbox
    center_x = (x1 + x2) / 2

    zone_width = frame_width / 3

    if center_x < zone_width:
        return 1  # 왼쪽
    elif center_x < zone_width * 2:
        return 2  # 중앙
    else:
        return 3  # 오른쪽


# =============================================================================
# PersonTracker 클래스
# -----------------------------------------------------------------------------
# YOLOv8 + ByteTrack 기반 사람 추적기
#
# 주요 기능:
#   1. 사람 객체 탐지 (YOLOv8n, class 0 = person)
#   2. 객체 추적 및 ID 부여 (ByteTrack)
#   3. 새로운 사람 등장 감지
#   4. 사람 사라짐 감지 (N프레임 이상 미탐지 시)
#   5. 구역별 사람 판단
#
# 사용 예시:
#   tracker = PersonTracker(conf=0.35, lost_threshold=30)
#   tracks, events, zones = tracker.update(frame)
#   # events['new'] = 새로 등장한 ID 리스트
#   # events['lost'] = 사라진 ID 리스트
#   # zones = {track_id: zone_number}
# =============================================================================
class PersonTracker:
    """YOLOv8 + ByteTrack 기반 사람 추적기 (고정 웹캠용)"""

    def __init__(self, model_path='yolov8n.pt', conf=0.35, lost_threshold=30, frame_width=1280):
        """
        Args:
            model_path: YOLOv8 모델 경로 (기본값: yolov8n.pt)
            conf: 탐지 신뢰도 임계값 (0.0 ~ 1.0, 기본값: 0.35)
            lost_threshold: 사라짐 판정 프레임 수 (기본 30프레임 ≈ 1초 @30fps)
            frame_width: 프레임 너비 (구역 판단용)
        """
        self.model = YOLO(model_path)
        self.conf = conf
        self.lost_threshold = lost_threshold
        self.frame_width = frame_width

        # ---------------------------------------------------------------------
        # 추적 상태 관리 변수
        # ---------------------------------------------------------------------
        self.tracked_persons = {}  # {track_id: {'last_seen', 'bbox', 'zone'}}
        self.frame_count = 0
        self.disappeared_persons = []
        self.new_persons = []

    def update(self, frame):
        """프레임을 처리하고 추적 결과를 반환

        Args:
            frame: BGR 이미지 (numpy array, OpenCV 형식)

        Returns:
            tracks: 추적 결과 리스트 [(track_id, (x1,y1,x2,y2), confidence, zone), ...]
            events: 이벤트 딕셔너리 {'new': [새 ID들], 'lost': [사라진 ID들]}
            zone_counts: 구역별 사람 수 [zone1_count, zone2_count, zone3_count]
        """
        self.frame_count += 1
        self.disappeared_persons = []
        self.new_persons = []

        # 프레임 크기 업데이트
        if frame is not None:
            self.frame_width = frame.shape[1]

        # ---------------------------------------------------------------------
        # YOLOv8 + ByteTrack 추적 실행
        # ---------------------------------------------------------------------
        results = self.model.track(
            frame,
            persist=True,
            tracker="bytetrack.yaml",
            conf=self.conf,
            classes=[0],
            verbose=False
        )[0]

        current_ids = set()
        tracks = []
        zone_counts = [0, 0, 0]  # [구역1, 구역2, 구역3]

        # ---------------------------------------------------------------------
        # 탐지 결과 처리
        # ---------------------------------------------------------------------
        if results.boxes is not None and len(results.boxes) > 0:
            boxes = results.boxes

            if boxes.id is not None:
                for i, track_id in enumerate(boxes.id.int().tolist()):
                    bbox = boxes.xyxy[i].int().tolist()
                    conf = boxes.conf[i].item()

                    x1, y1, x2, y2 = bbox
                    current_ids.add(track_id)

                    # 구역 판단
                    zone = get_zone_from_bbox(bbox, self.frame_width)
                    zone_counts[zone - 1] += 1  # 0-indexed

                    tracks.append((track_id, (x1, y1, x2, y2), conf, zone))

                    # 새로운 사람 등장 체크
                    if track_id not in self.tracked_persons:
                        self.new_persons.append(track_id)
                        print(f"[NEW] 새로운 사람 등장: ID {track_id} (구역 {zone})")

                    # 추적 정보 업데이트
                    self.tracked_persons[track_id] = {
                        'last_seen': self.frame_count,
                        'bbox': (x1, y1, x2, y2),
                        'zone': zone
                    }

        # ---------------------------------------------------------------------
        # 사라진 사람 확인
        # ---------------------------------------------------------------------
        for track_id, info in list(self.tracked_persons.items()):
            frames_missing = self.frame_count - info['last_seen']

            if frames_missing > self.lost_threshold:
                self.disappeared_persons.append(track_id)
                print(f"[LOST] 사람 사라짐: ID {track_id} (구역 {info['zone']}, {frames_missing}프레임 미탐지)")
                del self.tracked_persons[track_id]

        events = {
            'new': self.new_persons,
            'lost': self.disappeared_persons
        }

        return tracks, events, zone_counts

    def get_active_count(self):
        """현재 추적 중인 사람 수 반환"""
        return len(self.tracked_persons)

    def get_zone_positions(self, zone_id):
        """구역에 해당하는 로봇 좌표 반환"""
        return ZONE_POSITIONS.get(zone_id, None)


# =============================================================================
# 시각화 함수
# =============================================================================
def draw_results(frame, tracks, events, zone_counts, fps):
    """추적 결과를 프레임에 시각화

    Args:
        frame: BGR 이미지
        tracks: [(track_id, bbox, conf, zone), ...]
        events: {'new': [ids], 'lost': [ids]}
        zone_counts: [z1, z2, z3] 구역별 사람 수
        fps: 현재 FPS

    Returns:
        frame: 시각화가 추가된 프레임
    """
    h, w = frame.shape[:2]
    zone_width = w // 3

    # 구역 경계선 그리기 (세로 점선)
    for i in range(1, 3):
        x = zone_width * i
        for y in range(0, h, 20):
            cv2.line(frame, (x, y), (x, min(y + 10, h)), (128, 128, 128), 2)

    # 구역 라벨 표시
    zone_colors = [(0, 200, 200), (0, 200, 200), (0, 200, 200)]
    for i in range(3):
        zone_label = f"Zone {i+1}: {zone_counts[i]} person(s)"
        x_pos = zone_width * i + 10
        cv2.putText(frame, zone_label, (x_pos, h - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, zone_colors[i], 2)

    # 사람 바운딩 박스 그리기
    for track_id, bbox, conf, zone in tracks:
        x1, y1, x2, y2 = bbox

        # 구역별 색상
        if zone == 1:
            color = (0, 255, 255)  # 노란색
        elif zone == 2:
            color = (0, 255, 0)    # 초록색
        else:
            color = (255, 0, 255)  # 보라색

        # 새로 등장한 사람은 파란색
        if track_id in events['new']:
            color = (255, 100, 0)
            label = f'NEW ID {track_id} Z{zone}'
        else:
            label = f'ID {track_id} Z{zone}'

        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

        # 라벨
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
        cv2.rectangle(frame, (x1, y1 - th - 10), (x1 + tw, y1), color, -1)
        cv2.putText(frame, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

    # 사라진 사람 알림
    if events['lost']:
        lost_text = f"LOST: ID {', '.join(map(str, events['lost']))}"
        cv2.putText(frame, lost_text, (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

    # 상단 정보
    cv2.putText(frame, f'FPS: {fps:.1f}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
    cv2.putText(frame, f'Tracking: {len(tracks)} persons', (10, 110), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)

    return frame


# =============================================================================
# ROS2 노드 클래스
# =============================================================================
class PersonTrackingNode(Node):
    """ROS2 사람 추적 노드

    웹캠에서 사람을 추적하고 등장/사라짐 이벤트 및 구역 정보를 publish합니다.

    Published Topics:
        /person_appeared (std_msgs/Bool): 새 사람 등장 시 True
        /person_disappeared (std_msgs/Bool): 사람 사라짐 시 True
        /person_count (std_msgs/Int32): 현재 추적 중인 사람 수
        /zone_status (std_msgs/Int32MultiArray): 구역별 사람 수 [z1, z2, z3]
        /active_zone (std_msgs/Int32): 사람이 있는 구역 번호 (우선순위: 1>2>3)
        /zone_robot_pos (std_msgs/Float32MultiArray): 해당 구역의 로봇 좌표
    """

    def __init__(self):
        super().__init__('person_tracking_node')

        # ---------------------------------------------------------------------
        # ROS2 Parameters
        # ---------------------------------------------------------------------
        self.declare_parameter('camera_id', 0)
        self.declare_parameter('confidence', 0.35)
        self.declare_parameter('lost_threshold', 30)
        self.declare_parameter('show_window', True)

        self.camera_id = self.get_parameter('camera_id').value
        self.confidence = self.get_parameter('confidence').value
        self.lost_threshold = self.get_parameter('lost_threshold').value
        self.show_window = self.get_parameter('show_window').value

        # ---------------------------------------------------------------------
        # Publishers
        # ---------------------------------------------------------------------
        self.pub_appeared = self.create_publisher(Bool, '/person_appeared', 10)
        self.pub_disappeared = self.create_publisher(Bool, '/person_disappeared', 10)
        self.pub_count = self.create_publisher(Int32, '/person_count', 10)

        # 구역 관련 publisher
        self.pub_zone_status = self.create_publisher(Int32MultiArray, '/zone_status', 10)
        self.pub_active_zone = self.create_publisher(Int32, '/active_zone', 10)
        self.pub_zone_robot_pos = self.create_publisher(Float32MultiArray, '/zone_robot_pos', 10)

        # ---------------------------------------------------------------------
        # 웹캠 초기화
        # ---------------------------------------------------------------------
        self.cap = cv2.VideoCapture(self.camera_id)
        if not self.cap.isOpened():
            self.get_logger().error(f'웹캠을 열 수 없습니다: {self.camera_id}')
            raise RuntimeError(f'웹캠을 열 수 없습니다: {self.camera_id}')

        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        self.cap.set(cv2.CAP_PROP_FPS, 30)

        actual_w = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        actual_h = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.get_logger().info(f'웹캠 해상도: {actual_w}x{actual_h}')

        # ---------------------------------------------------------------------
        # 트래커 초기화
        # ---------------------------------------------------------------------
        self.tracker = PersonTracker(
            conf=self.confidence,
            lost_threshold=self.lost_threshold,
            frame_width=actual_w
        )

        self.fps_time = time.time()
        self.timer = self.create_timer(0.033, self.process_frame)

        self.get_logger().info('='*50)
        self.get_logger().info('사람 추적 노드 시작 (구역 판단 활성화)')
        self.get_logger().info(f'구역 좌표: {ZONE_POSITIONS}')
        self.get_logger().info('='*50)

    def process_frame(self):
        """프레임 처리 및 이벤트 publish"""
        ret, frame = self.cap.read()
        if not ret:
            self.get_logger().warn('프레임을 읽을 수 없습니다.')
            return

        # 추적 수행
        tracks, events, zone_counts = self.tracker.update(frame)

        # FPS 계산
        current_time = time.time()
        fps = 1.0 / (current_time - self.fps_time + 1e-6)
        self.fps_time = current_time

        # -----------------------------------------------------------------
        # ROS2 토픽 Publish
        # -----------------------------------------------------------------
        # 사람 등장/사라짐 이벤트
        if events['new']:
            msg = Bool()
            msg.data = True
            self.pub_appeared.publish(msg)
            self.get_logger().info(f'[PUB] person_appeared = True (ID: {events["new"]})')

        if events['lost']:
            msg = Bool()
            msg.data = True
            self.pub_disappeared.publish(msg)
            self.get_logger().info(f'[PUB] person_disappeared = True (ID: {events["lost"]})')

        # 추적 인원 수
        count_msg = Int32()
        count_msg.data = self.tracker.get_active_count()
        self.pub_count.publish(count_msg)

        # -----------------------------------------------------------------
        # 구역 정보 Publish
        # -----------------------------------------------------------------
        # 구역별 사람 수
        zone_msg = Int32MultiArray()
        zone_msg.data = zone_counts
        self.pub_zone_status.publish(zone_msg)

        # 활성 구역 (사람이 있는 구역, 우선순위: 1 > 2 > 3)
        active_zone = 0
        for i, count in enumerate(zone_counts):
            if count > 0:
                active_zone = i + 1
                break

        active_zone_msg = Int32()
        active_zone_msg.data = active_zone
        self.pub_active_zone.publish(active_zone_msg)

        # 활성 구역의 로봇 좌표
        if active_zone > 0:
            robot_pos = ZONE_POSITIONS[active_zone]
            pos_msg = Float32MultiArray()
            pos_msg.data = [float(x) for x in robot_pos]
            self.pub_zone_robot_pos.publish(pos_msg)
            # self.get_logger().info(f'[PUB] active_zone={active_zone}, pos={robot_pos}')

        # 시각화
        if self.show_window:
            frame = draw_results(frame, tracks, events, zone_counts, fps)
            cv2.imshow('Person Tracker (ROS2)', frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                self.get_logger().info('사용자에 의해 종료됨')
                self.destroy_node()
                rclpy.shutdown()

    def destroy_node(self):
        """노드 종료 시 리소스 정리"""
        self.cap.release()
        cv2.destroyAllWindows()
        super().destroy_node()


# =============================================================================
# 메인 함수
# =============================================================================
def main(args=None):
    rclpy.init(args=args)

    try:
        node = PersonTrackingNode()
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    except Exception as e:
        print(f'에러 발생: {e}')
    finally:
        rclpy.shutdown()


if __name__ == '__main__':
    main()
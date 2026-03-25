"""
tracking_debug_v3.py - 디버깅용 사람 추적 코드 (v3)

v2 대비 추가된 기능:
- 한글 이름 바운딩박스 표시 (PIL 사용)
- 이름 기반 고객 추적
- /customer_name 구독 - 고객 이름 수신
- /make_done 구독 - 제작 완료 신호 수신
- /disappeared_customer_name 발행 - 사라진 고객 이름
- /person_disappeared 제거

실행 방법:
    ros2 run bartender tracking

종료: 'q' 키
"""

import time
import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from ultralytics import YOLO

# ROS2 imports
import rclpy
from rclpy.node import Node
from std_msgs.msg import Bool, Int32, Float32MultiArray, Int32MultiArray, String


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
# 한글 폰트 설정
# =============================================================================
FONT_PATH = '/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc'
try:
    KOREAN_FONT = ImageFont.truetype(FONT_PATH, 20)
    KOREAN_FONT_SMALL = ImageFont.truetype(FONT_PATH, 16)
except:
    KOREAN_FONT = ImageFont.load_default()
    KOREAN_FONT_SMALL = ImageFont.load_default()


def put_korean_text(img, text, position, font=None, color=(255, 255, 255)):
    """OpenCV 이미지에 한글 텍스트 출력

    Args:
        img: OpenCV 이미지 (BGR)
        text: 출력할 텍스트
        position: (x, y) 좌표
        font: PIL 폰트 객체
        color: BGR 색상 튜플

    Returns:
        img: 텍스트가 추가된 이미지
    """
    if font is None:
        font = KOREAN_FONT

    # OpenCV BGR -> PIL RGB
    img_pil = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(img_pil)

    # BGR -> RGB 색상 변환
    rgb_color = (color[2], color[1], color[0])
    draw.text(position, text, font=font, fill=rgb_color)

    # PIL RGB -> OpenCV BGR
    return cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)


def get_text_size(text, font=None):
    """텍스트 크기 계산"""
    if font is None:
        font = KOREAN_FONT
    bbox = font.getbbox(text)
    return bbox[2] - bbox[0], bbox[3] - bbox[1]


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
    """
    x1, y1, x2, y2 = bbox
    center_x = (x1 + x2) / 2
    zone_width = frame_width / 3

    if center_x < zone_width:
        return 1
    elif center_x < zone_width * 2:
        return 2
    else:
        return 3


# =============================================================================
# PersonTracker 클래스
# =============================================================================
class PersonTracker:
    """YOLOv8 + ByteTrack 기반 사람 추적기 (이름 지원)"""

    def __init__(self, model_path='yolov8n.pt', conf=0.35, lost_threshold=30, frame_width=1280):
        self.model = YOLO(model_path)
        self.conf = conf
        self.lost_threshold = lost_threshold
        self.frame_width = frame_width

        # 추적 상태 관리
        # {track_id: {'last_seen', 'bbox', 'zone', 'name'}}
        self.tracked_persons = {}
        self.frame_count = 0
        self.disappeared_persons = []  # [(track_id, name, zone), ...]
        self.new_persons = []

    def assign_name_to_zone(self, zone, name):
        """특정 구역의 사람에게 이름 할당

        Args:
            zone: 구역 번호 (1, 2, 3)
            name: 할당할 이름

        Returns:
            bool: 할당 성공 여부
        """
        for track_id, info in self.tracked_persons.items():
            if info['zone'] == zone and info.get('name') is None:
                self.tracked_persons[track_id]['name'] = name
                print(f"[NAME] 구역 {zone}의 ID {track_id}에게 이름 '{name}' 할당")
                return True
        return False

    def assign_name_to_active(self, name):
        """현재 활성 구역(가장 먼저 탐지된 사람)에게 이름 할당

        Args:
            name: 할당할 이름

        Returns:
            int: 할당된 구역 번호 (0=실패)
        """
        # 이름이 없는 사람 중 가장 먼저 등장한 사람에게 할당
        for track_id, info in self.tracked_persons.items():
            if info.get('name') is None:
                self.tracked_persons[track_id]['name'] = name
                zone = info['zone']
                print(f"[NAME] ID {track_id} (구역 {zone})에게 이름 '{name}' 할당")
                return zone
        return 0

    def get_zone_by_name(self, name):
        """이름으로 구역 찾기

        Args:
            name: 찾을 이름

        Returns:
            int: 구역 번호 (0=없음)
        """
        for track_id, info in self.tracked_persons.items():
            if info.get('name') == name:
                return info['zone']
        return 0

    def get_customer_zone(self):
        """이름이 할당된 고객의 구역 반환 (첫 번째)

        Returns:
            int: 구역 번호 (0=없음)
        """
        for track_id, info in self.tracked_persons.items():
            if info.get('name') is not None:
                return info['zone']
        return 0

    def update(self, frame):
        """프레임 처리"""
        self.frame_count += 1
        self.disappeared_persons = []
        self.new_persons = []

        if frame is not None:
            self.frame_width = frame.shape[1]

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
        zone_counts = [0, 0, 0]

        if results.boxes is not None and len(results.boxes) > 0:
            boxes = results.boxes

            if boxes.id is not None:
                for i, track_id in enumerate(boxes.id.int().tolist()):
                    bbox = boxes.xyxy[i].int().tolist()
                    conf = boxes.conf[i].item()

                    x1, y1, x2, y2 = bbox
                    current_ids.add(track_id)

                    zone = get_zone_from_bbox(bbox, self.frame_width)
                    zone_counts[zone - 1] += 1

                    # 기존 정보 유지 (이름 등)
                    existing_name = None
                    if track_id in self.tracked_persons:
                        existing_name = self.tracked_persons[track_id].get('name')

                    # 새로운 사람 등장
                    if track_id not in self.tracked_persons:
                        self.new_persons.append(track_id)
                        print(f"[NEW] 새로운 사람 등장: ID {track_id} (구역 {zone})")

                    self.tracked_persons[track_id] = {
                        'last_seen': self.frame_count,
                        'bbox': (x1, y1, x2, y2),
                        'zone': zone,
                        'name': existing_name
                    }

                    name = existing_name if existing_name else ""
                    tracks.append((track_id, (x1, y1, x2, y2), conf, zone, name))

        # 사라진 사람 확인
        for track_id, info in list(self.tracked_persons.items()):
            frames_missing = self.frame_count - info['last_seen']

            if frames_missing > self.lost_threshold:
                name = info.get('name', '')
                zone = info['zone']
                self.disappeared_persons.append((track_id, name, zone))

                if name:
                    print(f"[LOST] 고객 '{name}' 사라짐 (ID {track_id}, 구역 {zone})")
                else:
                    print(f"[LOST] 사람 사라짐: ID {track_id} (구역 {zone})")

                del self.tracked_persons[track_id]

        events = {
            'new': self.new_persons,
            'lost': self.disappeared_persons
        }

        return tracks, events, zone_counts

    def get_active_count(self):
        return len(self.tracked_persons)


# =============================================================================
# 시각화 함수
# =============================================================================
def draw_results(frame, tracks, events, zone_counts, fps):
    """추적 결과를 프레임에 시각화"""
    h, w = frame.shape[:2]
    zone_width = w // 3

    # 구역 경계선
    for i in range(1, 3):
        x = zone_width * i
        for y in range(0, h, 20):
            cv2.line(frame, (x, y), (x, min(y + 10, h)), (128, 128, 128), 2)

    # 구역 라벨
    for i in range(3):
        zone_label = f"Zone {i+1}: {zone_counts[i]}"
        x_pos = zone_width * i + 10
        cv2.putText(frame, zone_label, (x_pos, h - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 200, 200), 2)

    # 사람 바운딩 박스
    for track_id, bbox, conf, zone, name in tracks:
        x1, y1, x2, y2 = bbox

        # 이름이 있으면 주황색, 없으면 구역별 색상
        if name:
            color = (0, 165, 255)  # 주황색 (이름 있음)
            label = f'{name} (Z{zone})'
        else:
            if zone == 1:
                color = (0, 255, 255)
            elif zone == 2:
                color = (0, 255, 0)
            else:
                color = (255, 0, 255)
            label = f'ID {track_id} Z{zone}'

        # 새로 등장한 사람
        if track_id in events['new']:
            color = (255, 100, 0)
            label = f'NEW {label}'

        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

        # 라벨 (한글 지원)
        tw, th = get_text_size(label, KOREAN_FONT)
        cv2.rectangle(frame, (x1, y1 - th - 10), (x1 + tw + 10, y1), color, -1)
        frame = put_korean_text(frame, label, (x1 + 2, y1 - th - 8), KOREAN_FONT, (255, 255, 255))

    # 사라진 사람 알림 (한글 지원)
    lost_with_names = [name for (_, name, _) in events['lost'] if name]
    if lost_with_names:
        lost_text = f"LOST: {', '.join(lost_with_names)}"
        frame = put_korean_text(frame, lost_text, (10, 50), KOREAN_FONT, (0, 0, 255))

    # 상단 정보
    cv2.putText(frame, f'FPS: {fps:.1f}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
    cv2.putText(frame, f'Tracking: {len(tracks)} persons', (10, 110), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)

    return frame


# =============================================================================
# ROS2 노드 클래스
# =============================================================================
class PersonTrackingNode(Node):
    """ROS2 사람 추적 노드 (바텐더 연동)

    Subscribe:
        /customer_name (String): 고객 이름 (활성 구역에 할당)
        /make_done (Bool): 제작 완료 신호

    Publish:
        /person_appeared (Bool): 새 사람 등장 시 True
        /person_count (Int32): 추적 인원 수
        /zone_status (Int32MultiArray): 구역별 사람 수
        /active_zone (Int32): 활성 구역 번호
        /zone_robot_pos (Float32MultiArray): 제작완료 시 로봇 좌표
        /disappeared_customer_name (String): 사라진 고객 이름
    """

    def __init__(self):
        super().__init__('person_tracking_node')

        # Parameters
        self.declare_parameter('camera_id', 0)
        self.declare_parameter('confidence', 0.35)
        self.declare_parameter('lost_threshold', 30)
        self.declare_parameter('show_window', True)

        self.camera_id = self.get_parameter('camera_id').value
        self.confidence = self.get_parameter('confidence').value
        self.lost_threshold = self.get_parameter('lost_threshold').value
        self.show_window = self.get_parameter('show_window').value

        # ---------------------------------------------------------------------
        # Subscribers
        # ---------------------------------------------------------------------
        self.sub_customer_name = self.create_subscription(
            String, '/customer_name', self.customer_name_callback, 10)
        self.sub_make_done = self.create_subscription(
            Bool, '/make_done', self.make_done_callback, 10)

        # ---------------------------------------------------------------------
        # Publishers
        # ---------------------------------------------------------------------
        self.pub_appeared = self.create_publisher(Bool, '/person_appeared', 10)
        self.pub_count = self.create_publisher(Int32, '/person_count', 10)
        self.pub_zone_status = self.create_publisher(Int32MultiArray, '/zone_status', 10)
        self.pub_active_zone = self.create_publisher(Int32, '/active_zone', 10)
        self.pub_zone_robot_pos = self.create_publisher(Float32MultiArray, '/zone_robot_pos', 10)
        self.pub_disappeared_name = self.create_publisher(String, '/disappeared_customer_name', 10)

        # ---------------------------------------------------------------------
        # 상태 변수
        # ---------------------------------------------------------------------
        self.pending_customer_name = None  # 아직 할당되지 않은 이름

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

        # 트래커 초기화
        self.tracker = PersonTracker(
            conf=self.confidence,
            lost_threshold=self.lost_threshold,
            frame_width=actual_w
        )

        self.fps_time = time.time()
        self.timer = self.create_timer(0.033, self.process_frame)

        self.get_logger().info('='*50)
        self.get_logger().info('사람 추적 노드 시작 (바텐더 연동)')
        self.get_logger().info(f'구역 좌표: {ZONE_POSITIONS}')
        self.get_logger().info('='*50)

    def customer_name_callback(self, msg):
        """고객 이름 수신 콜백"""
        name = msg.data.strip()
        if not name:
            return

        self.get_logger().info(f'[SUB] 고객 이름 수신: "{name}"')

        # 현재 활성 구역의 사람에게 이름 할당
        zone = self.tracker.assign_name_to_active(name)
        if zone > 0:
            self.get_logger().info(f'[NAME] "{name}" -> 구역 {zone} 할당 완료')
        else:
            # 할당할 사람이 없으면 대기
            self.pending_customer_name = name
            self.get_logger().warn(f'[NAME] 할당할 사람 없음. 대기 중: "{name}"')

    def make_done_callback(self, msg):
        """제작 완료 신호 수신 콜백"""
        self.get_logger().info('test')
        if not msg.data:
            return

        self.get_logger().info('[SUB] 제작 완료 신호 수신!')

        # 이름이 할당된 고객의 구역 찾기
        zone = self.tracker.get_customer_zone()

        if zone > 0:
            robot_pos = ZONE_POSITIONS[zone]
            pos_msg = Float32MultiArray()
            pos_msg.data = [float(x) for x in robot_pos]
            self.pub_zone_robot_pos.publish(pos_msg)
            self.get_logger().info(f'[PUB] 구역 {zone} 로봇 좌표: {robot_pos}')
        else:
            self.get_logger().warn('[WARN] 이름이 할당된 고객이 없습니다.')

    def process_frame(self):
        """프레임 처리"""
        ret, frame = self.cap.read()
        if not ret:
            self.get_logger().warn('프레임을 읽을 수 없습니다.')
            return

        tracks, events, zone_counts = self.tracker.update(frame)

        # FPS 계산
        current_time = time.time()
        fps = 1.0 / (current_time - self.fps_time + 1e-6)
        self.fps_time = current_time

        # 대기 중인 이름이 있고 새 사람이 등장하면 할당
        if self.pending_customer_name and events['new']:
            zone = self.tracker.assign_name_to_active(self.pending_customer_name)
            if zone > 0:
                self.get_logger().info(f'[NAME] 대기 중이던 "{self.pending_customer_name}" -> 구역 {zone} 할당')
                self.pending_customer_name = None

        # -----------------------------------------------------------------
        # Publish
        # -----------------------------------------------------------------
        # 새 사람 등장
        if events['new']:
            msg = Bool()
            msg.data = True
            self.pub_appeared.publish(msg)
            self.get_logger().info(f'[PUB] person_appeared = True (ID: {events["new"]})')

        # 사라진 고객 이름 publish
        for track_id, name, zone in events['lost']:
            if name:
                name_msg = String()
                name_msg.data = name
                self.pub_disappeared_name.publish(name_msg)
                self.get_logger().info(f'[PUB] disappeared_customer_name = "{name}"')

        # 추적 인원 수
        count_msg = Int32()
        count_msg.data = self.tracker.get_active_count()
        self.pub_count.publish(count_msg)

        # 구역별 사람 수
        zone_msg = Int32MultiArray()
        zone_msg.data = zone_counts
        self.pub_zone_status.publish(zone_msg)

        # 활성 구역
        active_zone = 0
        for i, count in enumerate(zone_counts):
            if count > 0:
                active_zone = i + 1
                break

        active_zone_msg = Int32()
        active_zone_msg.data = active_zone
        self.pub_active_zone.publish(active_zone_msg)

        # 시각화
        if self.show_window:
            frame = draw_results(frame, tracks, events, zone_counts, fps)
            cv2.imshow('Person Tracker (Bartender)', frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                self.get_logger().info('사용자에 의해 종료됨')
                self.destroy_node()
                rclpy.shutdown()

    def destroy_node(self):
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
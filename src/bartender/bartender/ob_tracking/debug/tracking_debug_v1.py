"""
tracking_debug_v1.py - 디버깅용 사람 추적 코드 (v1)

이 파일은 ROS2 없이 단독 실행 가능한 디버깅/테스트용 버전입니다.
웹캠에서 사람을 추적하고 화면에 시각화합니다.

실행 방법:
    python3 tracking_debug_v1.py

종료: 'q' 키
"""

import time
import cv2
from ultralytics import YOLO


class PersonTracker:
    """YOLOv8 + ByteTrack 기반 사람 추적기 (고정 웹캠용)

    기능:
    - 사람 객체 탐지 및 추적 (ID 부여)
    - 사람 사라짐 감지 (N프레임 이상 미탐지 시)
    - 새로운 사람 등장 감지
    """

    def __init__(self, model_path='yolov8n.pt', conf=0.35, lost_threshold=30):
        """
        Args:
            model_path: YOLOv8 모델 경로
            conf: 탐지 신뢰도 임계값
            lost_threshold: 사라짐 판정 프레임 수 (기본 30프레임 ≈ 1초)
        """
        self.model = YOLO(model_path)
        self.conf = conf
        self.lost_threshold = lost_threshold

        # 추적 상태 관리
        self.tracked_persons = {}  # {track_id: {'last_seen': frame_count, 'bbox': (x1,y1,x2,y2)}}
        self.frame_count = 0
        self.disappeared_persons = []  # 사라진 사람 ID 리스트 (이번 프레임)
        self.new_persons = []  # 새로 등장한 사람 ID 리스트 (이번 프레임)

    def update(self, frame):
        """프레임을 처리하고 추적 결과를 반환

        Args:
            frame: BGR 이미지 (numpy array)

        Returns:
            tracks: [(track_id, (x1,y1,x2,y2), confidence), ...]
            events: {'new': [ids], 'lost': [ids]}
        """
        self.frame_count += 1
        self.disappeared_persons = []
        self.new_persons = []

        # YOLOv8 + ByteTrack 추적 실행
        results = self.model.track(
            frame,
            persist=True,  # 프레임 간 ID 유지
            tracker="bytetrack.yaml",
            conf=self.conf,
            classes=[0],  # person class only
            verbose=False
        )[0]

        current_ids = set()
        tracks = []

        # 탐지 결과 처리
        if results.boxes is not None and len(results.boxes) > 0:
            boxes = results.boxes

            # track ID가 있는 경우만 처리
            if boxes.id is not None:
                for i, track_id in enumerate(boxes.id.int().tolist()):
                    bbox = boxes.xyxy[i].int().tolist()
                    conf = boxes.conf[i].item()

                    x1, y1, x2, y2 = bbox
                    current_ids.add(track_id)
                    tracks.append((track_id, (x1, y1, x2, y2), conf))

                    # 새로운 사람인지 확인
                    if track_id not in self.tracked_persons:
                        self.new_persons.append(track_id)
                        print(f"[DEBUG] [NEW] 새로운 사람 등장: ID {track_id}")

                    # 추적 정보 업데이트
                    self.tracked_persons[track_id] = {
                        'last_seen': self.frame_count,
                        'bbox': (x1, y1, x2, y2)
                    }

        # 사라진 사람 확인
        for track_id, info in list(self.tracked_persons.items()):
            frames_missing = self.frame_count - info['last_seen']

            if frames_missing > self.lost_threshold:
                self.disappeared_persons.append(track_id)
                print(f"[DEBUG] [LOST] 사람 사라짐: ID {track_id} ({frames_missing}프레임 미탐지)")
                del self.tracked_persons[track_id]

        events = {
            'new': self.new_persons,
            'lost': self.disappeared_persons
        }

        return tracks, events

    def get_active_count(self):
        """현재 추적 중인 사람 수 반환"""
        return len(self.tracked_persons)


def draw_results(frame, tracks, events, fps):
    """추적 결과를 프레임에 시각화

    Args:
        frame: BGR 이미지
        tracks: [(track_id, bbox, conf), ...]
        events: {'new': [ids], 'lost': [ids]}
        fps: 현재 FPS
    """
    # 추적 중인 사람 박스 그리기
    for track_id, bbox, conf in tracks:
        x1, y1, x2, y2 = bbox

        # 새로 등장한 사람은 파란색으로 표시
        if track_id in events['new']:
            color = (255, 100, 0)  # 파란색
            label = f'NEW ID {track_id}'
        else:
            color = (0, 255, 0)  # 초록색
            label = f'ID {track_id} ({conf:.2f})'

        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

        # 배경이 있는 라벨
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
        cv2.rectangle(frame, (x1, y1 - th - 10), (x1 + tw, y1), color, -1)
        cv2.putText(frame, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

    # 사라진 사람 알림 표시
    if events['lost']:
        lost_text = f"LOST: ID {', '.join(map(str, events['lost']))}"
        cv2.putText(frame, lost_text, (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

    # FPS 및 추적 인원 표시
    cv2.putText(frame, f'FPS: {fps:.1f}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
    cv2.putText(frame, f'Tracking: {len(tracks)} persons', (10, 110), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)

    return frame


def run_webcam_tracker(conf=0.35, show=True, src=0, lost_threshold=30):
    """웹캠에서 사람 추적 실행 (디버깅용)

    Args:
        conf: 탐지 신뢰도 임계값
        show: 화면 표시 여부
        src: 카메라 소스 (0=기본 웹캠)
        lost_threshold: 사라짐 판정 프레임 수
    """
    # 웹캠 열기
    cap = cv2.VideoCapture(src)
    if not cap.isOpened():
        raise RuntimeError(f'웹캠을 열 수 없습니다: {src}')

    # C270 HD 웹캠 해상도 설정 (720p)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    cap.set(cv2.CAP_PROP_FPS, 30)

    # 실제 설정된 값 확인
    actual_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    actual_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"[DEBUG] 웹캠 해상도: {actual_w}x{actual_h}")

    # 트래커 초기화
    tracker = PersonTracker(conf=conf, lost_threshold=lost_threshold)

    print("="*50)
    print("[DEBUG MODE] 사람 추적 시작 (종료: 'q' 키)")
    print("="*50)

    fps_time = time.time()

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("[DEBUG] 프레임을 읽을 수 없습니다.")
                break

            # 추적 수행
            tracks, events = tracker.update(frame)

            # FPS 계산
            current_time = time.time()
            fps = 1.0 / (current_time - fps_time + 1e-6)
            fps_time = current_time

            # 이벤트 발생 시 상태 출력
            if events['new']:
                print(f"[DEBUG] >>> person_appeared = 1 (ID: {events['new']})")
            if events['lost']:
                print(f"[DEBUG] >>> person_disappeared = 1 (ID: {events['lost']})")

            # 결과 시각화
            if show:
                frame = draw_results(frame, tracks, events, fps)
                cv2.imshow('[DEBUG] Person Tracker (ByteTrack)', frame)

                if cv2.waitKey(1) & 0xFF == ord('q'):
                    print("\n[DEBUG] 사용자에 의해 종료됨")
                    break

    except KeyboardInterrupt:
        print("\n[DEBUG] 키보드 인터럽트로 종료")

    finally:
        cap.release()
        cv2.destroyAllWindows()
        print("[DEBUG] 추적 종료")


if __name__ == '__main__':
    run_webcam_tracker()
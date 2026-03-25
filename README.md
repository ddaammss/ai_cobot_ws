# DYNAMIC BUSAN - 지능형 바텐더 로봇 시스템

**AI 기반 주문 인식 및 고객 추적을 통한 자율 음료 제조 로봇 시스템**

> ROS2 Humble + YOLOv8 + STT + Doosan M0609 협동로봇

---

## 📋 목차

- [시스템 개요](#-시스템-개요)
- [주요 기능](#-주요-기능)
- [시스템 아키텍처](#-시스템-아키텍처)
- [패키지 구조](#-패키지-구조)
- [설치 방법](#-설치-방법)
- [실행 방법](#-실행-방법)
- [노드 설명](#-노드-설명)
- [인터페이스](#-인터페이스)
- [차별화 포인트](#-차별화-포인트)

---

## 🎯 시스템 개요

DYNAMIC BUSAN은 음성 주문 인식, 실시간 고객 추적, 자동 음료 제조, 예외 상황 복구까지 전 과정을 자동화한 지능형 바텐더 로봇 시스템입니다.

### 핵심 기술 스택

- **ROS2 Humble**: 분산 로봇 제어 프레임워크
- **Doosan M0609**: 6축 협동 로봇팔
- **YOLOv8 + ByteTrack**: 실시간 고객 추적 및 위치 인식
- **OpenAI Whisper**: 고성능 음성-텍스트 변환 (STT)
- **KoNLPy**: 한국어 자연어 처리 (메뉴/이름 분리)
- **MariaDB**: 주문 이력 관리
- **Intel RealSense D435**: 깊이 카메라 (컵/병 인식)

---

## 🚀 주요 기능

### 1. 음성 주문 시스템
- **웨이크업 워드** 감지 ("안녕")
- **이름 + 메뉴** 동시 인식 ("저는 홍길동이고 모히또 주세요")
- **감정 기반 메뉴 추천** (기분 분석 → 맞춤 추천)
- **Fuzzy Matching** 메뉴 유사도 검색

### 2. 실시간 고객 추적
- **고객 이름 기반 개별 추적** (ID + 이름 매핑)
- **3-Zone 자동 분류** (로봇 좌표 자동 매핑)
- **Hysteresis 안정화** (5프레임 다수결 → 떨림 방지)
- **사라진 고객 자동 감지** (이탈 시 Recovery 트리거)

### 3. 자동 음료 제조
- **컵 선택** (RealSense 깊이 인식 + YOLO)
- **병 인식 및 따르기** (칵테일 레시피 기반)
- **쉐이커 동작** (진폭/속도 제어)
- **고객 구역으로 전달** (Tracking 데이터 기반)

### 4. 예외 상황 복구
- **고객 이탈 감지** (음료 미수령)
- **자동 보관대 이동** (작업 공간 확보)
- **로봇 충돌 방지** (중복 실행 차단)
- **컵 타입별 그리퍼 제어**

---

## 🏗️ 시스템 아키텍처

<img width="1020" height="585" alt="스크린샷 2026-02-09 11-54-05" src="https://github.com/user-attachments/assets/9515e3ac-2513-4a37-b46e-52b38f0ad18d" />


### 노드 간 통신 흐름

1. **주문 접수**: `supervisor` (웨이크업 감지 → STT → 이름/메뉴 분리)
2. **고객 등록**: `supervisor` → `/customer_name` → `tracking` (위치 추적 시작)
3. **음료 제조**: `supervisor` → `cup_pick` (Action) → `shake` (Action)
4. **음료 전달**: `shake` → `tracking` (DrinkDelivery Service) → 고객 구역 좌표 획득
5. **예외 처리**: `tracking` → `/disappeared_customer_name` → `recovery` (보관대 이동)

---

## 📦 패키지 구조

### `src/bartender/` - 메인 패키지

```
bartender/
├── supervisor/          # 전체 시퀀스 제어 + STT
│   └── supervisor_node.py
├── stt/                 # 음성 인식 모듈
│   └── wakeup.py
├── ob_tracking/         # 객체 추적 (YOLOv8 + ByteTrack)
│   └── tracking_node.py
├── recipe/              # 레시피 실행 (컵 선택, 따르기)
│   ├── cup_pick_node.py
├── shake/               # 전달 동작
│   └── shake_node.py
├── recovery/            # 예외 상황 복구
│   └── recovery_node.py
├── db/                  # MariaDB 연동
│   ├── mariadb_node.py
│   └── db_client.py
```

### `src/bartender_interfaces/` - 커스텀 인터페이스

```
bartender_interfaces/
├── action/
│   └── Motion.action        # 로봇 동작 액션 (Goal/Feedback/Result)
└── srv/
    └── DrinkDelivery.srv    # 음료 전달 위치 조회 서비스
```

---

## 🛠️ 설치 방법

### 1. 시스템 요구사항

- **OS**: Ubuntu 22.04 LTS
- **ROS2**: Humble Hawksbill
- **Python**: 3.10+
- **Hardware**:
  - Doosan M0609 협동로봇
  - Intel RealSense D435i (깊이 카메라)
  - Logitech C270 웹캠 (고객 추적용)

### 2. ROS2 설치

```bash
# ROS2 Humble 설치
sudo apt update && sudo apt install -y ros-humble-desktop
source /opt/ros/humble/setup.bash
```

### 3. 의존성 설치

```bash
# Python 패키지
pip3 install --user \
    openai \
    sounddevice \
    scipy \
    pyaudio \
    torch \
    transformers \
    ultralytics \
    opencv-python \
    pillow \
    konlpy \
    openpyxl \
    pymysql \
    python-dotenv

# ROS2 패키지
sudo apt install -y \
    ros-humble-cv-bridge \
    ros-humble-image-transport \
    python3-colcon-common-extensions

# Intel RealSense SDK
sudo apt install -y ros-humble-librealsense2* ros-humble-realsense2-*
```

### 4. 프로젝트 빌드

```bash
cd ~/dynamic_busan
colcon build --symlink-install
source install/setup.bash
```

### 5. 환경 변수 설정

`.env` 파일 생성 (`~/dynamic_busan/.env`):

```env
# MariaDB 설정
DB_HOST=localhost
DB_PORT=3306
DB_USER=root
DB_PASSWORD=your_password
DB_NAME=bartender_db

# OpenAI API Key (STT용)
OPENAI_API_KEY=sk-...
```

---

## 🎮 실행 방법

### 기본 실행 (토핑 제외)

```bash
source ~/dynamic_busan/install/setup.bash
ros2 launch bartender bartender.launch.py
```

### 개별 노드 실행

```bash
# Supervisor (STT + 시퀀스 제어)
ros2 run bartender supervisor

# Tracking (고객 추적)
ros2 run bartender tracking

# Cup Pick (컵 선택)
ros2 run bartender cup_pick

# Shake (쉐이커)
ros2 run bartender shake

# Recovery (복구)
ros2 run bartender recovery

# DB (주문 이력)
ros2 run bartender db
```

---

## 📡 노드 설명

### 1. `supervisor_node` - 전체 제어 + STT

**역할**: 음성 주문 → 시퀀스 제어 → 모션 실행 총괄

**Publish**:
- `/customer_name` (String) - 고객 이름 (tracking에 전달)
- `/current_menu` (String) - 현재 제조 중인 메뉴

**Action Client**:
- `/dsr01/recipe/motion` (Motion) - 컵 선택 액션
- `/dsr01/shake/motion` (Motion) - 쉐이크 액션
- `/dsr01/topping/motion` (Motion) - 토핑 액션

**주요 기능**:
- 웨이크업 워드 감지 (`Hello Rokey`)
- STT 기반 이름/메뉴 분리 (KoNLPy + Fuzzy Matching)
- 감정 분석 기반 메뉴 추천
- 모션 시퀀스 실행 (cup_pick → shake)

---

### 2. `tracking_node` - 고객 추적

**역할**: YOLOv8 + ByteTrack으로 실시간 고객 추적 및 위치 관리

**Subscribe**:
- `/customer_name` (String) - 주문한 고객 이름

**Publish**:
- `/person_count` (Int32) - 현재 추적 중인 사람 수
- `/zone_status` (Int32MultiArray) - 구역별 사람 수 [z1, z2, z3]
- `/active_zone` (Int32) - 활성 구역 번호 (1, 2, 3)
- `/disappeared_customer_name` (String) - 사라진 고객 이름

**Service Server**:
- `/drink_delivery` (DrinkDelivery) - 제작 완료 시 음료 전달 위치 제공

**주요 기능**:
- **3-Zone 시스템**: 프레임을 3구역으로 분할 (왼쪽/중앙/오른쪽)
- **Hysteresis**: 5프레임 다수결로 구역 떨림 방지
- **고객 이름 매핑**: track_id ↔ 고객 이름 연동
- **사라진 고객 감지**: 30프레임 이상 미감지 시 Recovery 트리거
- **로봇 좌표 매핑**: 구역별 로봇팔 좌표 자동 변환

---

### 3. `cup_pick_node` - 컵 선택 및 픽업

**역할**: RealSense + YOLO로 컵 인식 및 그리핑

**Action Server**:
- `/dsr01/recipe/motion` (Motion)

**주요 기능**:
- RealSense D435 깊이 데이터 + RGB 이미지
- YOLO 객체 인식 (green_cup, black_cup, yellow_cup)
- 깊이 정보 기반 Z축 좌표 계산
- 그리퍼 제어 (OnRobot RG2)

---

### 4. `shake_node` - 쉐이커 동작

**역할**: 쉐이커를 이용한 칵테일 믹싱

**Subscribe**:
- `/current_menu` (String) - 현재 제조 중인 메뉴 

**Action Server**:
- `/dsr01/shake/motion` (Motion)

**Service Client**:
- `/drink_delivery` (DrinkDelivery) - 고객 위치 조회

**주요 기능**:
- 쉐이커 그립/픽업
- 진폭 및 속도 제어 (칵테일별 파라미터)
- 고객 구역으로 전달 (tracking 데이터 기반)

---

### 5. `recovery_node` - 예외 상황 복구

**역할**: 고객 이탈 시 음료 회수 및 보관

**Subscribe**:
- `/disappeared_customer_name` (String) - 사라진 고객 이름
- `/cup_type` (String) - 컵 타입 정보

**주요 기능**:
- 사라진 고객 감지 시 자동 복구 시작
- 보관대로 음료 이동
- 로봇 충돌 방지 (`robot_executing` 플래그)
- 컵 타입별 그리퍼 제어 (파손 방지)

---

### 6. `mariadb_node` - DB 관리

**역할**: 주문 이력 및 레시피 데이터 관리

**Topics**:
- `/db_query` (String) - DB 쿼리 요청
- `/db_query_response` (String) - 쿼리 결과

**주요 기능**:
- 주문 이력 저장 (`bartender_order_history`)
- 레시피 데이터 조회
- 실시간 쿼리 처리

---

## 🔌 인터페이스

### Action: `Motion.action`

**로봇 동작 실행 인터페이스**

```yaml
# Goal
string motion_name       # 모션 이름 (예: "shake", "모히또")

# Result
bool success             # 성공 여부
string message           # 결과 메시지
int32 total_time_ms      # 총 소요 시간 (ms)

# Feedback
int32 progress           # 진행률 (0-100)
string current_step      # 현재 단계 설명
```

**사용 노드**:
- `cup_pick_node` (Server)
- `shake_node` (Server)
- `supervisor_node` (Client)

---

### Service: `DrinkDelivery.srv`

**음료 전달 위치 조회 인터페이스**

```yaml
# Request
bool finish              # 제조 완료 여부 (항상 true)

# Response
float32[] goal_position  # 목표 로봇 좌표 [x, y, z, rx, ry, rz]
```

**사용 노드**:
- `tracking_node` (Server)
- `shake_node` (Client)

---

## 💡 차별화 포인트

### 1. 객체 추적 (Tracking)

**"고객 이름 기반 개별 추적 + 로봇 좌표 자동 매핑"**

- ✅ **YOLOv8 + ByteTrack**: 실시간 다중 고객 동시 추적 (단순 카운팅 X)
- ✅ **3-Zone 자동 분류**: 고객 위치 → 로봇팔 좌표 자동 변환 → 정확한 음료 전달
- ✅ **Hysteresis (5프레임 다수결)**: 구역 판정 떨림 방지 → 안정적 위치 추적
- ✅ **사라진 고객 자동 감지**: 고객 이탈 시 Recovery 자동 트리거

**비교**: 일반 객체 추적 ("몇 명 있나?") vs **우리 시스템** ("누가 어디 있나?")

---

### 2. Recovery (복구 시스템)

**"예외 상황 자동 처리 + 로봇 충돌 방지"**

- ✅ **고객 이탈 자동 대응**: 음료 미수령 시 보관대로 자동 이동 → 작업 공간 확보
- ✅ **로봇 충돌 방지**: `robot_executing` 플래그 기반 중복 실행 차단
- ✅ **컵 타입별 그리퍼 제어**: 컵 종류에 따라 그립 강도 자동 조절 (파손 방지)
- ✅ **상태 기반 순차 처리**: 고객 등록 → 제조 완료 → 복구 순서 보장

**비교**: 일반 로봇 ("에러 발생" 알림만) vs **우리 시스템** ("자동 복구 완료")

---

### 3. STT + NLP 통합

- ✅ **이름 + 메뉴 동시 인식**: 하나의 문장에서 고객 이름과 메뉴 자동 분리
- ✅ **Fuzzy Matching**: 발음 유사도 기반 메뉴 검색
- ✅ **감정 분석 기반 추천**: 기분에 따른 맞춤 메뉴 제안

---

## 📄 라이선스

이 프로젝트는 교육 및 연구 목적으로 개발되었습니다.

---

## 👥 기여자

- **Maintainer**: dorong (ehdud2312@gmail.com)
- **Developer**: DYNAMIC BUSAN Team

---

## 🔗 참고 자료

- [ROS2 Humble Documentation](https://docs.ros.org/en/humble/)
- [Doosan Robotics](https://www.doosanrobotics.com/)
- [YOLOv8](https://github.com/ultralytics/ultralytics)
- [ByteTrack](https://github.com/ifzhang/ByteTrack)
- [OpenAI Whisper](https://github.com/openai/whisper)

---

**Generated with ❤️ by DYNAMIC BUSAN Team**

#!/usr/bin/env python3
"""
Supervisor Node - STT + 모션 시퀀스 통합 (Topping 없는 버전)

흐름: Wakeup 감지 → STT → DB 저장 → tracking에 고객 이름 전달 → 모션 시퀀스 실행
시퀀스: recipe → shake

연동 토픽:
  - /customer_name (pub): tracking_node에 고객 이름 전달
  - /manufacturing_done (pub): recovery_node에 제작 완료 신호

"""
# Java heap size 증가 설정 (KoNLPy OutOfMemoryError 방지)
import os
os.environ['JAVA_TOOL_OPTIONS'] = '-Xmx2g'  # 2GB heap size

import rclpy
from rclpy.node import Node
from rclpy.action import ActionClient
from rclpy.callback_groups import ReentrantCallbackGroup
from pathlib import Path
from std_msgs.msg import String

from bartender_interfaces.action import Motion
from bartender.db.db_client import DBClient

# 음성 인식
from openai import OpenAI
import sounddevice as sd
import scipy.io.wavfile as wav
import tempfile
import os
from dotenv import load_dotenv
from konlpy.tag import Komoran

from difflib import get_close_matches


# wakeup
from bartender.stt.wakeup import WakeupWord
from bartender.stt import MicController

# 기분 판정 위한 임포트 hugging face
import torch
from transformers import pipeline
import random

# .env 로드
env_path = Path.home() / 'dynamic_busan' / '.env'
load_dotenv(dotenv_path=env_path)
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")


class SupervisorNode(Node):
    def __init__(self, api_key):
        super().__init__("supervisor_node")
        self.get_logger().info("Supervisor Node initialized")

        # Callback Group
        self._cb_group = ReentrantCallbackGroup()

        # ActionClient
        self._action_clients = {
            'recipe': ActionClient(self, Motion, '/dsr01/recipe/motion', callback_group=self._cb_group),
            'shake': ActionClient(self, Motion, '/dsr01/shake/motion', callback_group=self._cb_group),
        }

        # 모션 시퀀스
        self.motion_sequence = [
            {'client': 'recipe', 'name': 'make_drink'},
            {'client': 'shake', 'name': 'shake_it'},
        ]
        self.current_index = 0
        self.is_running = False
        self.current_customer = None
        self.current_menu = None  # 현재 주문 메뉴 (cup_pick에 전달)

        # 유효한 메뉴 목록 (recipe.json의 recipe_id = DB의 menu_seq)
        self.valid_menus = [
            "블루 사파이어", "블루사파이어",
            "테킬라 선라이즈", "테킬라선라이즈",
            "퍼플 레인", "퍼플레인",
            "진 토닉", "진토닉",
            "트로피컬 오션", "트로피컬오션",
            "화이트 마가리타", "화이트마가리타",
            "블루 라군", "블루라군",
        ]


        # 일반 음료 단어 (valid_menus에 없는 일반 음료는 무시)
        self.common_beverage_words = [
            "아메리카노", "라떼", "커피", "에스프레소", "카푸치노", "모카",
            "마키아또", "카라멜", "바닐라", "녹차", "홍차", "밀크티",
            "주스", "콜라", "사이다", "음료", "물", "생수"
        ]

        self.positive_menu=["테킬라 선라이즈","트로피컬 오션","블루 라군"]
        self.negative_menu=["블루 사파이어","퍼플 레인","진 앤 토닉","화이트 마가리타"]


        # DB Client
        self.db_client = DBClient(self)

        # Publishers (tracking, recovery 연동)
        self.pub_customer_name = self.create_publisher(String, '/customer_name', 10)
        self.pub_current_menu = self.create_publisher(String, '/current_menu', 10)
        self.pub_manufacturing_done = self.create_publisher(String, '/manufacturing_done', 10)

        # OpenAI
        self.openai_client = OpenAI(api_key=api_key)
        self.duration = 5
        self.samplerate = 16000

        # 확인 단계 설정 (False로 바꾸면 확인 단계 생략)
        self.enable_confirmation = True
        self.confirmation_duration = 5  # 확인 응답 대기 시간 (초)

        # KoNLPy Komoran 초기화 (재사용을 위해 한 번만 생성)
        self.get_logger().info("Initializing Komoran...")
        self.komoran = Komoran()
        self.get_logger().info("Komoran initialized")

        # hugging face 기존 학습 모델 불러오기
        self.get_logger().info("허깅 페이스 모델 로딩 중...")
        self.sentiment = pipeline( 
            'sentiment-analysis',
            model='sangrimlee/bert-base-multilingual-cased-nsmc')
        self.get_logger().info("모델 로딩 완료")
        print()

        # Wakeup
        self.mic = MicController.MicController()
        self.mic.open_stream()
        self.wakeup = WakeupWord(self.mic.config.buffer_size)
        self.wakeup.set_stream(self.mic.stream)

        # Timer
        self.wakeup_timer = self.create_timer(0.5, self.check_wakeup)
        self.get_logger().info("Ready - Waiting for wakeup word...")

    # positive/negative 기분 감지 함수
    def detect_feel(self, text):
        with torch.no_grad():
            result = self.sentiment(text)
            return result

    def check_wakeup(self):
        """Wakeup 감지"""
        if self.is_running:
            return

        if self.wakeup.is_wakeup():
            self.get_logger().info("Wakeup detected!")
            self.is_running = True
            self.listen_and_process()

    def listen_and_process(self):
        """STT 처리"""
        try:
            self.get_logger().info("5초 동안 말해주세요...")

            audio = sd.rec(
                int(self.duration * self.samplerate),
                samplerate=self.samplerate,
                channels=1,
                dtype="int16",
            )
            sd.wait()
            self.get_logger().info("녹음 완료, STT 처리 중...")

            # 메뉴 힌트 생성 (STT 정확도 향상)
            menu_hint = ", ".join(set([m.replace(" ", "") for m in self.valid_menus]))
            prompt = f"바텐더 음료 주문입니다. 가능한 메뉴: {menu_hint}"

            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temp_wav:
                wav.write(temp_wav.name, self.samplerate, audio)
                with open(temp_wav.name, "rb") as f:
                    transcript = self.openai_client.audio.transcriptions.create(
                        model="gpt-4o-mini-transcribe",
                        file=f,
                        prompt=prompt,
                        language="ko",
                    )

            line = transcript.text
            self.get_logger().info(f"STT 결과: {line}")

            # 명사 추출 (재사용)
            nouns = self.komoran.nouns(line)

            # 필요 없는 말 필터링
            stop_words = ['안녕', '이름', '잔', '기분', '때', '것', '거', '추천', '우울', '축하', '행복']

            # 일반 음료 단어도 제거 (valid_menus에 없는 음료는 무시)
            filtered = [
                n for n in nouns
                if not any(word in n for word in stop_words)
                and n not in self.common_beverage_words
            ]

           # self.get_logger().info(f"명사: {nouns} → 필터: {filtered}")

            if not filtered:
                self.get_logger().warn("이름 인식 실패. 다시 시도해주세요.")
                self.is_running = False
                self.listen_for_menu_only()
                return

            # 메뉴를 먼저 찾고, 그 이전을 이름으로 처리
            name_parts = []
            menu_parts = []

            for noun in filtered:
                # 현재 명사가 메뉴에 포함되는지 확인
                is_menu = False

                # 1. 정확히 일치하는지 확인
                for valid_menu in self.valid_menus:
                    if noun in valid_menu.replace(" ", ""):
                        is_menu = True
                        break

                # 2. Fuzzy Matching (메뉴 단어들과 유사도 체크)
                if not is_menu:
                    menu_words = []
                    for vm in self.valid_menus:
                        menu_words.extend([w for w in vm.split() if w])

                    matches = get_close_matches(noun, menu_words, n=1, cutoff=0.8)
                    if matches:
                        is_menu = True
                        self.get_logger().info(f"🔍 명사 Fuzzy match: '{noun}' → '{matches[0]}'")

                if is_menu:
                    menu_parts.append(noun)
                else:
                    # 메뉴가 아직 안 나왔으면 이름에 추가
                    if not menu_parts:
                        name_parts.append(noun)

            name = "".join(name_parts)  # 공백 없이 결합 (예: "서동" + "찬" = "서동찬")

            # 이름 길이 검증 (한국 이름은 보통 2-4글자)
            if len(name) < 2 or len(name) > 5:
                self.get_logger().warn(f"⚠️  인식된 이름 '{name}'의 길이가 비정상적입니다 (2-5글자 권장).")
                self.get_logger().warn("다시 말씀해주세요.")
                self.is_running = False
                self.listen_for_menu_only()
                return

            #======================== 기분에 따른 메뉴 추천 ============================
            if '추천' in nouns or '기분' in nouns:
                
                result = self.detect_feel(line)
                label = result[0]['label']
                score = result[0]['score']

                if label == 'negative' and score > 0.6 :
                    menu = random.choice(self.negative_menu)
                    self.get_logger().info(f"(n) {menu} 를 추천 드립니다.")
                elif label == 'positive' and score > 0.6 :
                    menu = random.choice(self.positive_menu)
                    self.get_logger().info(f"(p) {menu} 를 추천 드립니다.")
                else:
                    menu = ""
            else:
                # 일반 메뉴 주문 시 메뉴 변수 지정
                menu = " ".join(menu_parts)  # 공백으로 결합 (예: "블루 사파이어")
                self.get_logger().info(f"선택하신 메뉴 : {menu}")
            #======================== 기분에 따른 메뉴 추천 ============================

            # 이름 저장 및 tracking에 전달
            self.current_customer = name
            name_msg = String()
            name_msg.data = name
            self.pub_customer_name.publish(name_msg)
            self.get_logger().info(f"[PUB] /customer_name: {name}")

            # 메뉴가 없으면 메뉴만 다시 받기
            if not menu:
                self.get_logger().warn(f"이름 '{name}'은(는) 확인되었습니다. 메뉴를 말해주세요.")
                self.get_logger().info(f"📋 가능한 메뉴: {', '.join([m for m in self.valid_menus if ' ' in m])}")
                self.listen_for_menu_only()
                return

            # 메뉴 검증
            valid_menu = self.validate_menu(menu)
            if valid_menu:
                # 확인 단계 (enable_confirmation이 True일 때만)
                if self.enable_confirmation:
                    if not self.ask_confirmation(name, valid_menu):
                        self.get_logger().warn("❌ 다시 입력해주세요.")
                        self.listen_for_menu_only()
                        return

                self.current_menu = valid_menu

                # 메뉴 정보 퍼블리시
                menu_msg = String()
                menu_msg.data = valid_menu
                self.pub_current_menu.publish(menu_msg)
                self.get_logger().info(f"[PUB] /current_menu: {valid_menu}")

                self.save_to_database(name, valid_menu)
                self.get_logger().info(f"=== Order: {name}, Menu: {valid_menu} ===")
                self.start_sequence()
            else:
                self.get_logger().warn(f"❌ '{menu}'은(는) 잘못된 메뉴입니다. 다시 말해주세요.")
                self.get_logger().info(f"📋 가능한 메뉴: {', '.join([m for m in self.valid_menus if ' ' in m])}")
                self.listen_for_menu_only()

        except Exception as e:
            self.get_logger().error(f"STT Error: {e}")
            self.is_running = False

    def save_to_database(self, name: str, menu: str):
        """DB 저장"""
        query = f"""
        INSERT INTO bartender_order_history (name, menu)
        VALUES ('{name.replace("'", "''")}', '{menu.replace("'", "''")}')
        """
        self.db_client.execute_query_with_response(query)

    def validate_menu(self, menu: str) -> str:
        """메뉴 유효성 검사. 유효하면 정규화된 메뉴명 반환, 아니면 None"""
        menu_normalized = menu.replace(" ", "")  # 공백 제거하여 비교

        # 1. 정확히 일치하는지 확인
        for valid_menu in self.valid_menus:
            valid_normalized = valid_menu.replace(" ", "")
            if menu_normalized == valid_normalized:
                # 공백 있는 정규 메뉴명 반환 (DB와 일치)
                if " " in valid_menu:
                    return valid_menu
                # 공백 없는 버전이면 공백 있는 버전 찾기
                for vm in self.valid_menus:
                    if vm.replace(" ", "") == valid_normalized and " " in vm:
                        return vm
                return valid_menu

        # 2. Fuzzy Matching (공백 있는 정규 메뉴만 대상)
        valid_menus_spaced = [m for m in self.valid_menus if " " in m]
        matches = get_close_matches(menu, valid_menus_spaced, n=1, cutoff=0.6)
        if matches:
            self.get_logger().info(f"🔍 Fuzzy match: '{menu}' → '{matches[0]}'")
            return matches[0]

        return None

    def ask_confirmation(self, name: str, menu: str) -> bool:
        """주문 확인 (예/아니요 판단)"""
        try:
            self.get_logger().info(f"고객님 성함은 '{name}', 메뉴는 '{menu}' 맞으신가요?")
            self.get_logger().info(f"({self.confirmation_duration}초 안에 대답해주세요)")

            # 음성 녹음
            audio = sd.rec(
                int(self.confirmation_duration * self.samplerate),
                samplerate=self.samplerate,
                channels=1,
                dtype="int16",
            )
            sd.wait()
            self.get_logger().info("확인 응답 처리 중...")

            # STT 처리
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temp_wav:
                wav.write(temp_wav.name, self.samplerate, audio)
                with open(temp_wav.name, "rb") as f:
                    transcript = self.openai_client.audio.transcriptions.create(
                        model="gpt-4o-mini-transcribe",
                        file=f,
                        prompt="예 또는 아니요로 대답해주세요.",
                        language="ko",
                    )

            response = transcript.text.lower()
            self.get_logger().info(f"확인 응답: {response}")

            # 긍정 단어 확인
            positive_words = ["예", "네", "맞", "응", "어", "yes", "ok", "오케이", "확인"]
            is_positive = any(word in response for word in positive_words)

            if is_positive:
                self.get_logger().info("✅ 주문이 확인되었습니다!")
                return True
            else:
                self.get_logger().warn("❌ 주문이 취소되었습니다.")
                self.listen_for_menu_only()
                return False

        except Exception as e:
            self.get_logger().error(f"확인 단계 에러: {e}")
            return False

    def listen_for_menu_only(self):
        """메뉴만 다시 입력받기 (이름은 유지)"""
        try:
            self.get_logger().info("메뉴를 다시 말해주세요 (5초)...")

            audio = sd.rec(
                int(self.duration * self.samplerate),
                samplerate=self.samplerate,
                channels=1,
                dtype="int16",
            )
            sd.wait()
            self.get_logger().info("녹음 완료, STT 처리 중...")

            # 메뉴 힌트 생성 (STT 정확도 향상)
            menu_hint = ", ".join(set([m.replace(" ", "") for m in self.valid_menus]))
            prompt = f"바텐더 음료 주문입니다. 가능한 메뉴: {menu_hint}"

            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temp_wav:
                wav.write(temp_wav.name, self.samplerate, audio)
                with open(temp_wav.name, "rb") as f:
                    transcript = self.openai_client.audio.transcriptions.create(
                        model="gpt-4o-mini-transcribe",
                        file=f,
                        prompt=prompt,
                        language="ko",
                    )

            line = transcript.text
            self.get_logger().info(f"STT 결과: {line}")

            # 명사 추출 (메뉴만, 재사용)
            nouns = self.komoran.nouns(line)
            stop_words = ['안녕', '이름', '잔', '메뉴', '주문']
            filtered = [n for n in nouns if not any(word in n for word in stop_words)]

            #self.get_logger().info(f"명사: {nouns} → 필터: {filtered}")

            if not filtered:
                self.get_logger().warn("메뉴 인식 실패. 처음부터 다시 시도해주세요.")
                self.reset_state()
                return
            
            #======================== 기분에 따른 메뉴 추천 ============================
            if '추천' in filtered or '기분' in filtered:
                
                result = self.detect_feel(line)
                label = result[0]['label']
                score = result[0]['score']

                if label == 'negative' and score > 0.6 :
                    menu = random.choice(self.negative_menu)
                    self.get_logger().info(f"(n) {menu} 를 추천 드립니다.")
                elif label == 'positive' and score > 0.6 :
                    menu = random.choice(self.positive_menu)
                    self.get_logger().info(f"(p) {menu} 를 추천 드립니다.")
                else:
                    menu = ""
            else:
                # 일반 메뉴 주문 시 메뉴 변수 지정
                menu = " ".join(filtered)
                self.get_logger().info(f"선택하신 메뉴 : {menu}")
            #======================== 기분에 따른 메뉴 추천 ============================
            

            # 메뉴 검증
            valid_menu = self.validate_menu(menu)
            if valid_menu:
                # 확인 단계 (enable_confirmation이 True일 때만)
                if self.enable_confirmation:
                    if not self.ask_confirmation(self.current_customer, valid_menu):
                        self.get_logger().warn("❌ 다시 입력해주세요.")
                        self.listen_for_menu_only()
                        return

                self.current_menu = valid_menu
                self.get_logger().info(f"=== 메뉴 확인: {valid_menu} ===")
                self.start_sequence()
            else:
                self.get_logger().warn(f"❌ '{menu}'은(는) 잘못된 메뉴입니다. 다시 말해주세요.")
                self.get_logger().info(f"📋 가능한 메뉴: {', '.join([m for m in self.valid_menus if ' ' in m])}")
                # 재귀적으로 메뉴만 다시 받기
                self.listen_for_menu_only()

        except Exception as e:
            self.get_logger().error(f"STT Error: {e}")
            self.reset_state()

    def start_sequence(self):
        """모션 시퀀스 시작"""
        self.current_index = 0
        self.get_logger().info("Connecting to Action Servers...")

        for name, client in self._action_clients.items():
            if not client.wait_for_server(timeout_sec=5.0):
                self.get_logger().error(f"{name}/motion not available!")
                self.reset_state()
                return
            self.get_logger().info(f"  {name}/motion connected")

        self.get_logger().info("Starting sequence...")
        self.execute_next()

    def execute_next(self):
        """다음 모션 실행"""
        if self.current_index >= len(self.motion_sequence):
            self.get_logger().info(f"=== Completed for {self.current_customer}! ===")

            # recovery_node에 제작 완료 신호 전달
            done_msg = String()
            done_msg.data = self.current_customer if self.current_customer else ""
            self.pub_manufacturing_done.publish(done_msg)
            self.get_logger().info(f"[PUB] /manufacturing_done: {done_msg.data}")

            self.reset_state(auto_restart=True)  # 자동으로 다음 주문 받기
            return

        motion = self.motion_sequence[self.current_index]
        client = self._action_clients[motion['client']]

        # recipe 액션일 때는 실제 메뉴명 전달
        if motion['client'] == 'recipe' and self.current_menu:
            action_name = self.current_menu
        else:
            action_name = motion['name']

        self.get_logger().info(
            f"[{self.current_index + 1}/{len(self.motion_sequence)}] {motion['client']}: {action_name}"
        )

        self.get_logger().info(f"🔍 DEBUG: Goal 생성 및 전송 시작...")
        goal = Motion.Goal()
        goal.motion_name = action_name
        self.get_logger().info(f"📤 send_goal_async 호출...")
        future = client.send_goal_async(goal, feedback_callback=self.on_feedback)
        self.get_logger().info(f"✅ send_goal_async 완료, 콜백 등록 중...")
        future.add_done_callback(self.on_goal_accepted)
        self.get_logger().info(f"✅ 콜백 등록 완료")

    def on_goal_accepted(self, future):
        """Goal 수락"""
        self.get_logger().info("📩 on_goal_accepted 콜백 호출됨")
        goal_handle = future.result()
        self.get_logger().info(f"🔍 goal_handle.accepted = {goal_handle.accepted}")

        if not goal_handle.accepted:
            self.get_logger().error("❌ Goal rejected! Resetting...")
            self.reset_state()
            return

        self.get_logger().info("✅ Goal accepted! get_result_async 호출...")
        goal_handle.get_result_async().add_done_callback(self.on_result)
        self.get_logger().info("✅ Result 콜백 등록 완료")

    def on_feedback(self, feedback_msg):
        """Feedback"""
        fb = feedback_msg.feedback
        self.get_logger().info(f"  {fb.progress}% - {fb.current_step}")

    def on_result(self, future):
        """Result → 다음 실행"""
        self.get_logger().info("🏁 on_result 콜백 호출됨")
        result = future.result().result
        self.get_logger().info(f"  Done: {result.message}")
        self.current_index += 1
        self.get_logger().info(f"📍 다음 인덱스: {self.current_index}")
        self.execute_next()

    def reset_state(self, auto_restart=False):
        """상태 초기화

        Args:
            auto_restart: True면 자동으로 다음 주문 받기 시작
        """
        # 마이크 스트림 버퍼 비우기 (sd.rec 사용 후 PyAudio 버퍼 꼬임 방지)
        try:
            if self.mic.stream and self.mic.stream.is_active():
                available = self.mic.stream.get_read_available()
                if available > 0:
                    self.mic.stream.read(available, exception_on_overflow=False)
        except Exception:
            pass

        self.is_running = False
        self.current_customer = None
        self.current_menu = None
        self.current_index = 0
        self.get_logger().info("Ready for next customer...")

        # 자동 재시작 옵션
        if auto_restart:
            self.get_logger().info("🔄 자동으로 다음 주문 받기 시작...")
            self.is_running = True
            self.listen_and_process()


def main(args=None):
    rclpy.init(args=args)
    node = SupervisorNode(OPENAI_API_KEY)
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()

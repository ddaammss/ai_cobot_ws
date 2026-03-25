#!/usr/bin/env python3
"""
STT Node - Speech to Text with Database Integration
Captures microphone input, converts to text, and saves to database
"""
import rclpy
from rclpy.node import Node
from bartender.db.db_client import DBClient
from std_msgs.msg import String, Bool
from pathlib import Path

# 음성 인식
from openai import OpenAI
import sounddevice as sd
import scipy.io.wavfile as wav
import tempfile
import os
from dotenv import load_dotenv
from konlpy.tag import Komoran

# wakeup
from .wakeup import WakeupWord
from . import MicController

# .env 로드
env_path = Path.home() / 'dynamic_busan' / '.env'
load_dotenv(dotenv_path=env_path)
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

class STTNode(Node):
    def __init__(self, api_key):
        super().__init__("stt_node")
        self.get_logger().info("STT Node initialized")

        # DB Client 초기화
        self.db_client = DBClient(self)

        # 퍼블리시를 위한 변수 추가
        self.last_name = None

        # name 퍼블리셔 생성
        self.name_publisher = self.create_publisher(
            String,
            "/customer_name",
            10
        )

        # OpenAI 클라이언트
        self.client = OpenAI(api_key=api_key)
        self.duration = 5        # 녹음 시간 (초)
        self.samplerate = 16000  # Whisper 권장 샘플레이트

        # 노드 시작 시 쿼리 실행
        #self.query_logs_by_keyword()

        # ---------- wakeup word 초기화 추가 부분 ---------------
        self.mic = MicController.MicController()
        self.mic.open_stream()

        self.wakeup = WakeupWord(self.mic.config.buffer_size)
        self.wakeup.set_stream(self.mic.stream)

        self.waiting_for_wakeup = True
        self.check_in_progress = False  # 중복 실행 방지
        self.wakeup_timer = self.create_timer(0.5, self.check_wakeup)

        # supervisor 상태 구독
        self.supervisor_running = False
        self.create_subscription(
            Bool,
            '/supervisor/state',
            self.on_supervisor_state,
            10
        )

    def on_supervisor_state(self, msg):
        """supervisor 상태 업데이트"""
        self.supervisor_running = msg.data
        print(self.supervisor_running)
    # 트리거 콜백
    def check_wakeup(self):
        try:
            # supervisor 실행 중이면 wakeup 감지 건너뛰기
            if self.supervisor_running:
                return

            if not self.waiting_for_wakeup:
                return

            if self.wakeup.is_wakeup():
                self.get_logger().info("Wakeup detected")
                self.waiting_for_wakeup = False
                self.listen_and_process()
                self.waiting_for_wakeup = True
        finally:
            self.check_in_progress = False

    def listen_and_process(self):
        """마이크로 음성을 듣고 텍스트로 변환한 뒤 DB에 저장"""
        try:
            print("🎙️ 5초 동안 말해주세요...")

            # 1️⃣ 마이크로 음성 녹음
            audio = sd.rec(
                int(self.duration * self.samplerate),
                samplerate=self.samplerate,
                channels=1,
                dtype="int16",
            )
            sd.wait()
            print("✅ 녹음 완료, STT 처리 중...")

            # 2️⃣ 임시 wav 파일 생성
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temp_wav:
                wav.write(temp_wav.name, self.samplerate, audio)

                # 3️⃣ Whisper API로 STT
                with open(temp_wav.name, "rb") as f:
                    transcript = self.client.audio.transcriptions.create(
                        model="whisper-1",
                        file=f,
                    )
            line = transcript.text

            # [4] 문장에서 명사만 추출
            komoran = Komoran()
            nouns = komoran.nouns(line)

            stop_words = ['안녕', '이름', '잔']
            # 필요 없는 말들 제외 리스트

            filtered = [
                n for n in nouns
                if not any(word in n for word in stop_words)
            ]
            # 제외 리스트에 겹치는 말 거르기
            print("원본:", nouns)
            print("필터 후:", filtered)

            # 이름, 메뉴 부분 변수에 지정
            name = filtered[0]
            menu = " ".join(filtered[1:])

            # DB에 저장
            self.save_to_database(name, menu)

            # 이름 퍼블리시
            self.last_name = name
            msg = String()
            msg.data = name
            self.name_publisher.publish(msg)
            self.get_logger().info(f"Published customer name: {name}")

        except Exception as e:
            self.get_logger().error(f"Error in listen_and_process: {e}")

    def save_to_database(self, text: str, text2: str):
        # INSERT 쿼리 작성
        query = f"""
        INSERT INTO bartender_order_history (name, menu )
        VALUES ('{text.replace("'", "''")}','{text2.replace("'", "''")}')
        """

        # DB에 쿼리 전송 및 응답 대기
        request_id = self.db_client.execute_query_with_response(
            query,
          #  callback=self.on_db_response
        )

        self.get_logger().info(f"Sent query to DB (request_id: {request_id})")

    def on_db_response(self, response_data: dict):
        """DB INSERT 쿼리 응답 처리"""
        if response_data.get('success'):
            self.get_logger().info("Successfully saved to database")
            self.get_logger().info(f"Response: {response_data}")

            # INSERT 성공 후 최근 로그 조회 (예시)
            self.query_recent_logs(limit=5)
        else:
            self.get_logger().error(f"Failed to save to database: {response_data.get('error')}")

    # ============ SELECT 쿼리 예시들 ============
    def query_logs_by_keyword(self, keyword: str = '유성'):
        """
        특정 키워드를 포함하는 로그 조회 (LIKE 사용 예시)

        Args:
            keyword: 검색할 키워드
        """
        # SQL Injection 방지를 위해 single quote escape
        escaped_keyword = keyword.replace("'", "''")
        query = f"""
        SELECT name, percent
        FROM bartender_order_history
        WHERE name LIKE '%{escaped_keyword}%'
        ORDER BY created_at DESC
        """

        request_id = self.db_client.execute_query_with_response(
            query,
            callback=self.on_select_response
        )

        self.get_logger().info(f"Querying logs with keyword '{keyword}' (request_id: {request_id})")

    def on_select_response(self, response_data: dict):
        """DB SELECT 쿼리 응답 처리"""
        if response_data.get('success'):
            result = response_data.get('result', [])
            self.get_logger().info(f"Query successful! Retrieved {len(result)} rows")

            # 조회 결과 출력
            for row in result:
                self.get_logger().info(f"  - {row}")
                # 개별 필드 접근 예시
                # name = row.get('name')
                # percent = row.get('percent')
        else:
            self.get_logger().error(f"Query failed: {response_data.get('error')}")


def main(args=None):
    rclpy.init(args=args)
    node = STTNode(OPENAI_API_KEY)

    # MultiThreadedExecutor 사용 (서비스 호출이 콜백 안에서 완료되려면 필요)
    #from rclpy.executors import MultiThreadedExecutor
    #executor = MultiThreadedExecutor()
    #executor.add_node(node)

    try:
     #   executor.spin()
     rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
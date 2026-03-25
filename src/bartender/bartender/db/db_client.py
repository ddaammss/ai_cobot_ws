#!/usr/bin/env python3
"""
DB Client Helper - MariaDB 노드와 통신하기 위한 헬퍼 클래스
"""
import rclpy
from rclpy.node import Node
from std_msgs.msg import String
from std_srvs.srv import Trigger
import json
import uuid
from typing import Optional, Callable


class DBClient:
    """MariaDB 노드와 통신하기 위한 클라이언트 헬퍼 클래스"""

    def __init__(self, node: Node, callback_group=None):
        """
        Args:
            node: ROS2 Node 객체
            callback_group: ROS2 CallbackGroup (옵션)
        """
        self.node = node

        # Publisher: 쿼리 실행 요청 (fire and forget) - 절대경로로 네임스페이스 무관하게 통신
        self.query_pub = self.node.create_publisher(String, '/db_query', 10)

        # Publisher: 쿼리 실행 요청 (with response)
        self.query_request_pub = self.node.create_publisher(String, '/db_query_request', 10)

        # Subscriber: 쿼리 실행 결과 구독
        self.query_response_sub = self.node.create_subscription(
            String,
            '/db_query_response',
            self._query_response_callback,
            10,
            callback_group=callback_group
        )

        # Subscriber: DB 상태 구독
        self.status_sub = self.node.create_subscription(
            String,
            '/db_status',
            self._status_callback,
            10,
            callback_group=callback_group
        )

        # Service Client: DB 연결 상태 확인
        self.check_connection_client = self.node.create_client(
            Trigger,
            '/check_db_connection'
        )

        self.db_status = "unknown"

        # 응답 대기 중인 요청들
        self.pending_requests = {}  # {request_id: callback}

    def _status_callback(self, msg: String):
        """DB 상태 구독 콜백"""
        self.db_status = msg.data

    def _query_response_callback(self, msg: String):
        """쿼리 응답 콜백"""
        try:
            response_data = json.loads(msg.data)
            request_id = response_data.get('request_id')

            if request_id in self.pending_requests:
                callback = self.pending_requests[request_id]
                if callback:
                    callback(response_data)
                del self.pending_requests[request_id]
        except json.JSONDecodeError as e:
            self.node.get_logger().error(f"Failed to parse response JSON: {str(e)}")

    def execute_query(self, query: str):
        """
        쿼리 실행 요청 (fire and forget)

        Args:
            query: 실행할 SQL 쿼리
        """
        msg = String()
        msg.data = query
        self.query_pub.publish(msg)
        self.node.get_logger().info(f"Query published: {query}")

    def execute_query_with_response(self, query: str, callback: Optional[Callable] = None) -> str:
        """
        쿼리 실행 요청 (응답 포함)

        Args:
            query: 실행할 SQL 쿼리
            callback: 응답을 받을 콜백 함수 (response_data dict를 인자로 받음)

        Returns:
            str: request_id
        """
        request_id = str(uuid.uuid4())
        request_data = {
            'request_id': request_id,
            'query': query
        }

        # 콜백 등록
        if callback:
            self.pending_requests[request_id] = callback

        # 요청 발행
        msg = String()
        msg.data = json.dumps(request_data)
        self.query_request_pub.publish(msg)
        #self.node.get_logger().info(f"Query request published [{request_id}]: {query}")

        return request_id

    def check_connection(self):
        """
        DB 연결 상태 확인 (동기식)

        Returns:
            bool: 연결 성공 여부
        """
        if not self.check_connection_client.wait_for_service(timeout_sec=1.0):
            self.node.get_logger().warn("DB connection check service not available")
            return False

        request = Trigger.Request()
        future = self.check_connection_client.call_async(request)
        rclpy.spin_until_future_complete(self.node, future, timeout_sec=2.0)

        if future.result() is not None:
            response = future.result()
            self.node.get_logger().info(f"Connection status: {response.message}")
            return response.success
        else:
            self.node.get_logger().error("Service call failed")
            return False

    def get_status(self) -> str:
        """
        현재 DB 상태 반환

        Returns:
            str: DB 상태 ("connected", "disconnected", "unknown")
        """
        return self.db_status

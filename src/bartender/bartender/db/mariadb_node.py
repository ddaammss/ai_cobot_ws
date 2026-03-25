#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from std_msgs.msg import String
from std_srvs.srv import Trigger
import pymysql
import json
from typing import Optional


class MariaDBNode(Node):
    def __init__(self):
        super().__init__("mariadb_node")

        # MariaDB 연결 파라미터 선언 (launch 파일에서 제공)
        self.declare_parameter('db_host', '')
        self.declare_parameter('db_port', 3306)
        self.declare_parameter('db_user', '')
        self.declare_parameter('db_password', '')
        self.declare_parameter('db_name', '')

        # 파라미터 가져오기
        self.db_host = self.get_parameter('db_host').value
        self.db_port = self.get_parameter('db_port').value
        self.db_user = self.get_parameter('db_user').value
        self.db_password = self.get_parameter('db_password').value
        self.db_name = self.get_parameter('db_name').value

        # MariaDB 연결
        self.connection: Optional[pymysql.connections.Connection] = None

        # Publisher: DB 상태 발행 (절대경로)
        self.status_pub = self.create_publisher(String, '/db_status', 10)

        # Subscriber: 쿼리 실행 요청 받기 (fire and forget)
        self.query_sub = self.create_subscription(
            String,
            '/db_query',
            self.query_callback,
            10
        )

        # Subscriber: 쿼리 실행 요청 받기 (with response)
        self.query_request_sub = self.create_subscription(
            String,
            '/db_query_request',
            self.query_request_callback,
            10
        )

        # Publisher: 쿼리 실행 결과 발행
        self.query_response_pub = self.create_publisher(String, '/db_query_response', 10)

        # Service: DB 연결 상태 확인
        self.connection_check_srv = self.create_service(
            Trigger,
            '/check_db_connection',
            self.check_connection_callback
        )

        # Timer: 주기적으로 연결 상태 확인 (1초마다)
        #self.timer = self.create_timer(60.0, self.check_connection_timer)

        self.get_logger().info(f"MariaDB Node initialized. Connecting to {self.db_host}:{self.db_port}/{self.db_name}")

        # DB 연결 시도
        self.connect_to_database()

    def connect_to_database(self):
        """MariaDB 데이터베이스에 연결"""
        try:
            self.connection = pymysql.connect(
                host=self.db_host,
                port=self.db_port,
                user=self.db_user,
                password=self.db_password,
                database=self.db_name
            )
            self.get_logger().info("Successfully connected to MariaDB database")
            self.publish_status("connected")
            return True
        except Exception as e:
            self.get_logger().error(f"Failed to connect to MariaDB: {str(e)}")
            self.publish_status("disconnected")
            return False

    def query_callback(self, msg: String):
        """쿼리 실행 콜백"""
        query = msg.data
        self.get_logger().info(f"Received query: {query}")

        if not self.is_connected():
            self.get_logger().warn("Not connected to database. Attempting to reconnect...")
            if not self.connect_to_database():
                return

        try:
            cursor = self.connection.cursor()
            cursor.execute(query)

            # SELECT 쿼리인 경우 결과 로깅
            if query.strip().upper().startswith('SELECT'):
                results = cursor.fetchall()
                # Convert results to list of dicts for JSON serialization
                columns = [column[0] for column in cursor.description]
                results_dict = [dict(zip(columns, row)) for row in results]
                self.get_logger().info(f"Query results: {json.dumps(results_dict, indent=2, default=str)}")
            else:
                # INSERT, UPDATE, DELETE 등의 경우 커밋
                self.connection.commit()
                self.get_logger().info(f"Query executed successfully. Rows affected: {cursor.rowcount}")
            cursor.close()
        except Exception as e:
            self.get_logger().error(f"Query execution failed: {str(e)}")

    def check_connection_callback(self, request, response):
        """연결 상태 확인 서비스 콜백"""
        if self.is_connected():
            response.success = True
            response.message = "Database connection is active"
        else:
            response.success = False
            response.message = "Database connection is inactive"
            # 재연결 시도
            self.connect_to_database()

        return response

    def query_request_callback(self, msg: String):
        """쿼리 실행 요청 콜백 (응답 포함)"""
        try:
            # JSON 파싱
            request_data = json.loads(msg.data)
            query = request_data.get('query', '')
            request_id = request_data.get('request_id', '')

            self.get_logger().info(f"Received query request [{request_id}]: {query}")

            # 응답 데이터 초기화
            response_data = {
                'request_id': request_id,
                'success': False,
                'message': '',
                'result': None
            }

            # DB 연결 확인
            if not self.is_connected():
                self.get_logger().warn("Not connected to database. Attempting to reconnect...")
                if not self.connect_to_database():
                    response_data['message'] = "Failed to connect to database"
                    self._publish_response(response_data)
                    return

            try:
                cursor = self.connection.cursor()
                cursor.execute(query)

                # SELECT 쿼리인 경우 결과 반환
                if query.strip().upper().startswith('SELECT'):
                    results = cursor.fetchall()
                    columns = [column[0] for column in cursor.description]
                    results_dict = [dict(zip(columns, row)) for row in results]

                    response_data['success'] = True
                    response_data['message'] = f"Query executed successfully. {len(results)} rows returned."
                    response_data['result'] = results_dict

                    self.get_logger().info(f"Query results [{request_id}]: {len(results)} rows")
                else:
                    # INSERT, UPDATE, DELETE 등의 경우 커밋
                    self.connection.commit()

                    response_data['success'] = True
                    response_data['message'] = f"Query executed successfully. {cursor.rowcount} rows affected."
                    response_data['result'] = {"rows_affected": cursor.rowcount}

                    self.get_logger().info(response_data['message'])

                cursor.close()
            except Exception as e:
                error_msg = f"Query execution failed: {str(e)}"
                self.get_logger().error(error_msg)

                response_data['message'] = error_msg

            # 응답 발행
            self._publish_response(response_data)

        except json.JSONDecodeError as e:
            self.get_logger().error(f"Failed to parse request JSON: {str(e)}")
            error_response = {
                'request_id': 'unknown',
                'success': False,
                'message': f"Invalid JSON format: {str(e)}",
                'result': None
            }
            self._publish_response(error_response)

    def _publish_response(self, response_data: dict):
        """응답 발행"""
        response_msg = String()
        response_msg.data = json.dumps(response_data, default=str)
        self.query_response_pub.publish(response_msg)
        self.get_logger().info(f"Published response for request [{response_data['request_id']}]")

    def check_connection_timer(self):
        """주기적으로 연결 상태 확인"""
        if self.is_connected():
            try:
                cursor = self.connection.cursor()
                cursor.execute("SELECT 1")
                cursor.close()
                self.get_logger().info("✓ Database connection is healthy")
            except:
                self.get_logger().warn("Connection lost. Reconnecting...")
                self.connect_to_database()
        else:
            self.get_logger().warn("Not connected. Attempting to reconnect...")
            self.connect_to_database()

    def is_connected(self) -> bool:
        """연결 상태 확인"""
        if self.connection is None:
            return False
        try:
            self.connection.ping(reconnect=False)
            return True
        except Exception:
            return False

    def publish_status(self, status: str):
        """DB 상태 발행"""
        msg = String()
        msg.data = status
        self.status_pub.publish(msg)

    def destroy_node(self):
        """노드 종료 시 DB 연결 종료"""
        if self.connection:
            try:
                self.connection.close()
                self.get_logger().info("MariaDB connection closed")
            except:
                pass
        super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    node = MariaDBNode()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()

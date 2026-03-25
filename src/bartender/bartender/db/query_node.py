#!/usr/bin/env python3
"""
Query Node - 실제 쿼리를 작성하고 실행하는 노드
"""
import rclpy
from rclpy.node import Node
from bartender.db.db_client import DBClient


class QueryNode(Node):
    def __init__(self):
        super().__init__("query_node")

        # DB 클라이언트 초기화
        self.db_client = DBClient(self)

        # 예제 쿼리들 정의
        self.queries = {
            'create_table': """
                CREATE TABLE IF NOT EXISTS test_table (
                    id INT AUTO_INCREMENT PRIMARY KEY,
                    name VARCHAR(255) NOT NULL,
                    value INT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """,
            'insert_sample': """
                INSERT INTO test_table (name, value)
                VALUES ('sample_data', 100)
            """,
            'select_all': """
                SELECT * FROM test_table
            """,
            'select_recent': """
                SELECT * FROM test_table
                ORDER BY created_at DESC
                LIMIT 10
            """,
            'update_sample': """
                UPDATE test_table
                SET value = value + 1
                WHERE name = 'sample_data'
            """,
            'delete_old': """
                DELETE FROM test_table
                WHERE created_at < DATE_SUB(NOW(), INTERVAL 1 DAY)
            """,
        }

        self.get_logger().info("Query Node initialized")
        self.get_logger().info("Available commands via parameters:")
        self.get_logger().info("  - ros2 run bartender query --ros-args -p query:=create_table")
        self.get_logger().info("  - ros2 run bartender query --ros-args -p query:=insert_sample")
        self.get_logger().info("  - ros2 run bartender query --ros-args -p query:=select_all")

        # 파라미터로 쿼리 받기
        self.declare_parameter('query', '')
        self.declare_parameter('custom_query', '')
        self.declare_parameter('auto_run', False)

        # auto_run이 True면 시작 시 쿼리 자동 실행
        if self.get_parameter('auto_run').value:
            self.execute_query_from_param()

    def execute_query_from_param(self):
        """파라미터로 받은 쿼리 실행"""
        query_name = self.get_parameter('query').value
        custom_query = self.get_parameter('custom_query').value

        if custom_query:
            # 커스텀 쿼리가 있으면 그것을 실행
            self.get_logger().info(f"Executing custom query: {custom_query}")
            self.db_client.execute_query(custom_query)
        elif query_name and query_name in self.queries:
            # 미리 정의된 쿼리 실행
            self.get_logger().info(f"Executing predefined query: {query_name}")
            self.db_client.execute_query(self.queries[query_name])
        elif query_name:
            self.get_logger().warn(f"Query '{query_name}' not found in predefined queries")
        else:
            self.get_logger().info("No query specified. Use -p query:=<name> or -p custom_query:='<sql>'")

    def execute_predefined_query(self, query_name: str):
        """
        미리 정의된 쿼리 실행

        Args:
            query_name: 실행할 쿼리 이름
        """
        if query_name in self.queries:
            query = self.queries[query_name]
            self.get_logger().info(f"Executing query '{query_name}'")
            self.db_client.execute_query(query)
        else:
            self.get_logger().warn(f"Query '{query_name}' not found")
            self.get_logger().info(f"Available queries: {list(self.queries.keys())}")

    def execute_custom_query(self, query: str):
        """
        커스텀 쿼리 실행

        Args:
            query: SQL 쿼리 문자열
        """
        self.get_logger().info(f"Executing custom query: {query}")
        self.db_client.execute_query(query)

    def check_db_connection(self):
        """DB 연결 상태 확인"""
        self.get_logger().info("Checking database connection...")
        is_connected = self.db_client.check_connection()
        if is_connected:
            self.get_logger().info("✓ Database is connected")
        else:
            self.get_logger().warn("✗ Database is not connected")
        return is_connected

    # ========== 실제 사용 예제 메서드들 ==========

    def example_create_and_insert(self):
        """예제: 테이블 생성 및 데이터 삽입"""
        self.execute_predefined_query('create_table')
        # 1초 대기 (토픽 전송 시간)
        self.get_clock().sleep_for(rclpy.duration.Duration(seconds=1))
        self.execute_predefined_query('insert_sample')

    def example_select_data(self):
        """예제: 데이터 조회"""
        self.execute_predefined_query('select_all')

    def example_custom_query(self):
        """예제: 커스텀 쿼리 실행"""
        custom_sql = "SELECT COUNT(*) as total FROM test_table"
        self.execute_custom_query(custom_sql)


def main(args=None):
    rclpy.init(args=args)
    node = QueryNode()

    # 파라미터로 쿼리가 지정되었으면 실행하고 종료
    query_name = node.get_parameter('query').value
    custom_query = node.get_parameter('custom_query').value
    auto_run = node.get_parameter('auto_run').value

    if auto_run or query_name or custom_query:
        node.execute_query_from_param()
        # 쿼리 발행 후 약간 대기
        rclpy.spin_once(node, timeout_sec=0.5)
        node.destroy_node()
        rclpy.shutdown()
    else:
        # 파라미터가 없으면 노드를 계속 실행 (대화형 모드)
        try:
            rclpy.spin(node)
        except KeyboardInterrupt:
            pass
        finally:
            node.destroy_node()
            rclpy.shutdown()


if __name__ == "__main__":
    main()

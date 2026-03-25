#!/usr/bin/env python3
"""
MariaDB Node Launch File
"""
import os
from pathlib import Path
from launch import LaunchDescription
from launch_ros.actions import Node

# .env 파일 로드 (python-dotenv 사용)
try:
    from dotenv import load_dotenv
    env_path = Path(__file__).resolve().parents[3] / '.env'
    load_dotenv(dotenv_path=env_path)
except ImportError:
    print("Warning: python-dotenv not installed. Using system environment variables only.")
except Exception as e:
    print(f"Warning: Could not load .env file: {e}")


def generate_launch_description():
    """MariaDB 노드만 실행하는 launch 파일"""

    # 환경 변수에서 DB 설정 읽기
    db_host = os.environ.get('DB_HOST', 'localhost')
    db_port = int(os.environ.get('DB_PORT', '3306'))
    db_user = os.environ.get('DB_USER', 'root')
    db_password = os.environ.get('DB_PASSWORD', '')
    db_name = os.environ.get('DB_NAME', 'test')

    # MariaDB 노드 설정
    mariadb_node = Node(
        package='bartender',
        executable='db',
        name='mariadb_node',
        output='screen',
        parameters=[{
            'db_host': db_host,
            'db_port': db_port,
            'db_user': db_user,
            'db_password': db_password,
            'db_name': db_name,
        }],
        emulate_tty=True,
    )

    return LaunchDescription([
        mariadb_node,
    ])
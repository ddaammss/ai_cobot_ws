#!/usr/bin/env python3
"""
Bartender Complete Launch File

노드 구성:
  - MariaDB Node (DB 연결)
  - Recipe Node (레시피 액션 서버)
  - Shake Node (쉐이크 액션 서버)
  - Topping Node (토핑 액션 서버, 선택)
  - Tracking Node (사람 추적)
  - Recovery Node (복구 처리)
  - Supervisor Node (전체 흐름 제어)

연동 흐름:
  supervisor --(/customer_name)--> tracking --(/disappeared_customer_name)--> recovery
      |                                |
      +----(Action)---> recipe/shake/topping ----(get_pose srv)----+

사용법:
  기본 (topping 없음): ros2 launch bartender bartender.launch.py
  topping 포함:       ros2 launch bartender bartender.launch.py with_topping:=true
"""
import os
from pathlib import Path
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration, PythonExpression
from launch.conditions import IfCondition
from launch_ros.actions import Node

def generate_launch_description():
    """Bartender 패키지의 모든 노드를 실행하는 launch 파일"""

    # .env 파일 로드 (python-dotenv 사용)
    try:
        from dotenv import load_dotenv
        env_path = Path.home() / 'dynamic_busan' / '.env'
        print(f"[DEBUG] Loading .env from: {env_path}")
        print(f"[DEBUG] .env file exists: {env_path.exists()}")
        result = load_dotenv(dotenv_path=env_path, override=True)
        print(f"[DEBUG] .env file loaded: {result}")
    except ImportError:
        print("Warning: python-dotenv not installed. Using system environment variables only.")
    except Exception as e:
        print(f"Warning: Could not load .env file: {e}")

    # 환경 변수에서 DB 설정 읽기
    db_host = os.environ.get('DB_HOST', 'localhost')
    db_port = int(os.environ.get('DB_PORT', '3306'))
    db_user = os.environ.get('DB_USER', 'root')
    db_password = os.environ.get('DB_PASSWORD', '')
    db_name = os.environ.get('DB_NAME', 'test')

    # 디버깅: 로드된 값 출력
    print("="*50)
    print("[DEBUG] DB Configuration Loaded:")
    print(f"  DB_HOST: {db_host}")
    print(f"  DB_PORT: {db_port}")
    print(f"  DB_USER: {db_user}")
    print(f"  DB_NAME: {db_name}")
    print("="*50)

    # =========================================================================
    # Launch 인자 선언
    # =========================================================================
    db_host_arg = DeclareLaunchArgument(
        'db_host',
        default_value=db_host,
        description='MariaDB host address'
    )

    db_port_arg = DeclareLaunchArgument(
        'db_port',
        default_value=str(db_port),
        description='MariaDB port'
    )

    db_user_arg = DeclareLaunchArgument(
        'db_user',
        default_value=db_user,
        description='MariaDB username'
    )

    db_password_arg = DeclareLaunchArgument(
        'db_password',
        default_value=db_password,
        description='MariaDB password'
    )

    db_name_arg = DeclareLaunchArgument(
        'db_name',
        default_value=db_name,
        description='Database name'
    )

    # Topping 포함 여부
    with_topping_arg = DeclareLaunchArgument(
        'with_topping',
        default_value='false',
        description='Include topping node and use supervisor_full (true/false)'
    )

    # =========================================================================
    # 노드 정의
    # =========================================================================

    # MariaDB 노드
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

    # Recipe 노드 (cup_pick_node 실행 - 실제 로봇 제어)
    recipe_node = Node(
        package='bartender',
        executable='cup_pick',
        name='recipe_node',
        output='screen',
        emulate_tty=True,
    )

    # Shake 노드
    shake_node = Node(
        package='bartender',
        executable='shake',
        name='shake_node',
        output='screen',
        emulate_tty=True,
    )

    # # Topping 노드 (with_topping=true일 때만 실행)
    # topping_node = Node(
    #     package='bartender',
    #     executable='topping_node',
    #     name='topping_node',
    #     output='screen',
    #     emulate_tty=True,
    #     condition=IfCondition(LaunchConfiguration('with_topping')),
    # )

    # Tracking 노드 (사람 추적, 로지텍 C270 웹캠 = video2)
    tracking_node = Node(
        package='bartender',
        executable='tracking',
        name='tracking_node',
        output='screen',
        emulate_tty=True,
        parameters=[{'camera_id': 0}],  # 기본값 0 (C270은 코드에서 자동 탐지)
    )

    # Recovery 노드 (복구 처리)
    recovery_node = Node(
        package='bartender',
        executable='recovery',
        name='recovery_node',
        output='screen',
        emulate_tty=True,
    )

    # Supervisor 노드 (항상 실행)
    supervisor_node = Node(
        package='bartender',
        executable='supervisor',
        name='supervisor_node',
        output='screen',
        emulate_tty=True,
    )

    # =========================================================================
    # Launch Description 반환
    # =========================================================================
    return LaunchDescription([
        # Launch 인자들
        db_host_arg,
        db_port_arg,
        db_user_arg,
        db_password_arg,
        db_name_arg,
        with_topping_arg,

        # 공통 노드들
        mariadb_node,
        recipe_node,
        shake_node,
        tracking_node,
        recovery_node,

        # 조건부 노드들
        # topping_node,         # with_topping=true일 때만
        supervisor_node,
    ])
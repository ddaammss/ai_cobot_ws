#!/bin/bash
# Bartender ROS2 Package 의존성 설치 스크립트

echo "=========================================="
echo "Bartender 의존성 설치 시작"
echo "=========================================="

# 시스템 패키지 설치 (pyaudio 빌드에 필요)
echo "[1/3] 시스템 패키지 설치..."
sudo apt-get update
sudo apt-get install -y \
    portaudio19-dev \
    python3-pip

# pip 패키지 설치
echo "[2/3] Python 패키지 설치..."
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
pip3 install -r "$SCRIPT_DIR/requirements.txt"

# .env 파일 설정
echo "[3/3] .env 파일 설정..."
ENV_FILE="$HOME/dynamic_busan/.env"
ENV_EXAMPLE="$HOME/dynamic_busan/.env.example"

if [ ! -f "$ENV_FILE" ]; then
    echo ".env 파일이 없습니다. .env.example에서 복사합니다..."

    if [ -f "$ENV_EXAMPLE" ]; then
        cp "$ENV_EXAMPLE" "$ENV_FILE"
        echo ".env 파일이 생성되었습니다: $ENV_FILE"
        echo ""
        echo "DB 정보를 입력해주세요:"
        nano "$ENV_FILE"
    else
        echo "ERROR: .env.example 파일도 없습니다!"
        echo "수동으로 .env 파일을 생성해주세요."
    fi
else
    echo ".env 파일이 이미 존재합니다: $ENV_FILE"
fi

echo "=========================================="
echo "설치 완료!"
echo "=========================================="

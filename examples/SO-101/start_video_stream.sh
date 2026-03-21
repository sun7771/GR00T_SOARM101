#!/bin/bash

# Robot Camera Stream Server
# 启动机器人摄像头视频流服务器

echo "=========================================="
echo "  Robot Camera Stream Server"
echo "=========================================="
echo ""

# 默认配置
ROBOT_TYPE=${ROBOT_TYPE:-"so101_follower"}
ROBOT_PORT=${ROBOT_PORT:-"/dev/ttyACM0"}
ROBOT_ID=${ROBOT_ID:-"lil_guy"}

# 摄像头配置（根据你的实际设备调整）
CAMERAS="{wrist: {type: opencv, index_or_path: 0, width: 640, height: 480, fps: 30}, front: {type: opencv, index_or_path: 1, width: 640, height: 480, fps: 30}}"

echo "Starting video stream server with:"
echo "  Robot Type: $ROBOT_TYPE"
echo "  Robot Port: $ROBOT_PORT"
echo "  Robot ID: $ROBOT_ID"
echo ""

# 启动服务器
python video_stream.py \
    --robot.type=$ROBOT_TYPE \
    --robot.port=$ROBOT_PORT \
    --robot.id=$ROBOT_ID \
    --robot.cameras="$CAMERAS"

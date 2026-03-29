#!/bin/bash

# 脚本功能：运行GR00T机器人评估脚本
# 用于启动SO-101机械臂的策略评估

echo "启动GR00T机器人评估脚本..."

export PYTHONPATH=$PYTHONPATH:$(pwd) 
python examples/SO-101/eval_lerobot.py \
     --robot.type=so101_follower \
     --robot.port=/dev/ttyACM0 \
     --robot.id=my_awesome_follower_arm \
     --robot.cameras="{ wrist: {type: opencv, index_or_path: 2, width: 640, height: 480, fps: 30}, side: {type: opencv, index_or_path: 0, width: 640, height: 480, fps: 30}}" \
     --policy_host=192.168.0.200 \
     --lang_instruction="Take the item out of the box and put it on the floor."

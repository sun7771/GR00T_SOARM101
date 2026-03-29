#!/bin/bash

# 脚本功能：列出所有串口设备
# 用于查看系统中的ttyACM设备

echo "列出所有串口设备 (/dev/ttyACM*)..."
ls /dev/ttyACM*
echo "设备列表输出完成！"

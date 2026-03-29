#!/bin/bash

# 脚本功能：设置串口设备权限
# 用于给 /dev/ttyACM* 设备添加读写权限

echo "设置 /dev/ttyACM* 设备权限..."
sudo chmod 666 /dev/ttyACM*
echo "权限设置完成！"

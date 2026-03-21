# 机器人控制 + 视频流集成使用说明

## ✨ 新功能

现在 `eval_lerobot.py` 已经集成了视频流服务器功能，可以**同时运行机器人控制和网页视频流**，无需担心串口冲突！

## 🚀 快速开始

### 启动带视频流的机器人控制

```bash
cd /home/jetson/Lerobot-GR00TN1.5-main/examples/SO-101

python eval_lerobot.py \
    --robot.type=so101_follower \
    --robot.port=/dev/ttyACM0 \
    --robot.id=lil_guy \
    --robot.cameras="{wrist: {type: opencv, index_or_path: 0, width: 640, height: 480, fps: 30}, front: {type: opencv, index_or_path: 1, width: 640, height: 480, fps: 30}}" \
    --policy_host=localhost \
    --enable_video_stream=true \
    --video_stream_port=5000
```

### 访问视频流

启动后在浏览器中打开：

- **本地访问**: http://localhost:5000
- **局域网访问**: http://<你的IP地址>:5000

例如：http://192.168.1.100:5000

## ⚙️ 配置选项

### 视频流配置

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `--enable_video_stream` | 是否启用视频流服务器 | True |
| `--video_stream_port` | 视频流服务器端口 | 5000 |

### 禁用视频流

如果不需要视频流，可以禁用以节省资源：

```bash
python eval_lerobot.py \
    --robot.type=so101_follower \
    --robot.port=/dev/ttyACM0 \
    --robot.id=lil_guy \
    --enable_video_stream=false
```

## 🔒 线程安全机制

程序使用**线程锁**保护串口访问，确保：

1. ✅ 机器人控制线程和视频流线程不会同时访问串口
2. ✅ 避免串口冲突错误（"Port is in use!"）
3. ✅ 保证数据一致性和稳定性

### 线程架构

```
主线程 (asyncio)
├── 机器人控制循环
│   ├── 获取观测数据 (使用 robot_lock)
│   ├── 策略推理
│   └── 发送动作 (使用 robot_lock)
│
└── 视频流服务器线程
    └── 更新摄像头帧 (使用 robot_lock)
```

## 📊 性能影响

启用视频流会带来轻微的性能开销：

- **CPU**: ~5-10% 额外占用
- **内存**: ~100-200MB 额外占用
- **网络**: ~2-5 Mbps（取决于分辨率和帧率）

### 性能优化建议

1. **降低分辨率**（如果网络带宽有限）
   ```bash
   --robot.cameras="{wrist: {type: opencv, index_or_path: 0, width: 320, height: 240, fps: 30}}"
   ```

2. **降低帧率**（如果CPU负载高）
   ```bash
   --robot.cameras="{wrist: {type: opencv, index_or_path: 0, width: 640, height: 480, fps: 15}}"
   ```

3. **禁用视频流**（如果不需要监控）
   ```bash
   --enable_video_stream=false
   ```

## 🌐 网络访问

### 查看本机IP地址

```bash
# Linux
ip addr show | grep inet

# 或
hostname -I
```

### 防火墙配置

如果无法从其他设备访问，需要开放端口：

```bash
# Ubuntu/Debian
sudo ufw allow 5000

# CentOS/RHEL
sudo firewall-cmd --add-port=5000/tcp --permanent
sudo firewall-cmd --reload
```

## 📱 移动设备访问

### 手机访问

1. 确保手机和Jetson在同一WiFi网络
2. 在手机浏览器中打开：http://<Jetson的IP>:5000
3. 横屏查看效果更佳

### 平板访问

平板设备访问方式与手机相同，大屏幕显示效果更好。

## 🐛 故障排除

### 1. 视频流无法访问

```bash
# 检查服务器是否启动
curl http://localhost:5000

# 检查端口占用
lsof -i :5000

# 更换端口
--video_stream_port=8080
```

### 2. 视频卡顿

- 降低分辨率或帧率
- 检查网络带宽
- 关闭其他占用CPU的程序

### 3. 机器人控制变慢

- 禁用视频流：`--enable_video_stream=false`
- 降低视频流质量（修改代码中的JPEG质量参数）

### 4. 串口错误

确保使用了线程锁保护（已集成在代码中），如果仍有问题：

```bash
# 检查串口权限
ls -la /dev/ttyACM0

# 添加权限
sudo chmod 666 /dev/ttyACM0
```

## 📝 完整示例

### 示例1：基本使用

```bash
python eval_lerobot.py \
    --robot.type=so101_follower \
    --robot.port=/dev/ttyACM0 \
    --robot.id=lil_guy \
    --robot.cameras="{wrist: {type: opencv, index_or_path: 0, width: 640, height: 480, fps: 30}}" \
    --policy_host=localhost \
    --lang_instruction="Grab the pen" \
    --enable_video_stream=true
```

### 示例2：高性能模式

```bash
python eval_lerobot.py \
    --robot.type=so101_follower \
    --robot.port=/dev/ttyACM0 \
    --robot.id=lil_guy \
    --robot.cameras="{wrist: {type: opencv, index_or_path: 0, width: 640, height: 480, fps: 30}}" \
    --policy_host=localhost \
    --enable_video_stream=false \
    --ctrl_period=0.001
```

### 示例3：低带宽模式

```bash
python eval_lerobot.py \
    --robot.type=so101_follower \
    --robot.port=/dev/ttyACM0 \
    --robot.id=lil_guy \
    --robot.cameras="{wrist: {type: opencv, index_or_path: 0, width: 320, height: 240, fps: 15}}" \
    --policy_host=localhost \
    --enable_video_stream=true
```

## 🎯 使用场景

### 场景1：开发调试

在开发过程中实时查看摄像头画面，方便调试算法。

### 场景2：远程监控

在另一个房间或远程位置监控机器人运行状态。

### 场景3：演示展示

向客户或同事展示机器人控制效果，无需物理在场。

### 场景4：教学培训

在教学场景中，学生可以通过网页实时查看机器人操作。

## 📚 相关文档

- [VIDEO_STREAM_README.md](./VIDEO_STREAM_README.md) - 独立视频流服务器文档
- [eval_lerobot.py](./eval_lerobot.py) - 主程序代码

## 💡 提示

- 视频流和机器人控制共享同一个机器人连接，使用线程锁保护
- 视频流服务器在后台线程运行，不影响主控制循环
- 可以随时在浏览器中刷新页面查看最新画面
- 支持多个浏览器同时访问

## 🔄 与旧版本对比

| 功能 | 旧版本 | 新版本 |
|------|--------|--------|
| 视频流 | 需要单独运行 `video_stream.py` | 集成在 `eval_lerobot.py` 中 |
| 串口冲突 | 可能发生 | 使用线程锁保护 |
| 资源占用 | 两个进程 | 一个进程 |
| 使用复杂度 | 需要管理两个程序 | 一个命令启动 |
| 性能 | 可能互相影响 | 优化协调 |

## 📞 技术支持

如有问题，请检查：
1. Flask 是否已安装：`pip show flask`
2. 摄像头是否正常工作：`ls /dev/video*`
3. 串口权限是否正确：`ls -la /dev/ttyACM0`
4. 防火墙是否开放端口：`sudo ufw status`

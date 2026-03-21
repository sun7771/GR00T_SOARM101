# Robot Camera Stream Server

实时查看机器人摄像头视频流的网页服务器。

## 功能特性

- 📹 实时视频流（~30 FPS）
- 🌐 网页访问，支持多设备
- 📱 响应式设计，支持手机和电脑
- 🔒 线程安全，不影响机器人控制
- 🎨 美观的网页界面

## 快速开始

### 方法1：使用启动脚本（推荐）

```bash
cd /home/jetson/Lerobot-GR00TN1.5-main/examples/SO-101
chmod +x start_video_stream.sh
./start_video_stream.sh
```

### 方法2：直接运行 Python 脚本

```bash
cd /home/jetson/Lerobot-GR00TN1.5-main/examples/SO-101

python video_stream.py \
    --robot.type=so101_follower \
    --robot.port=/dev/ttyACM0 \
    --robot.id=lil_guy \
    --robot.cameras="{wrist: {type: opencv, index_or_path: 0, width: 640, height: 480, fps: 30}, front: {type: opencv, index_or_path: 1, width: 640, height: 480, fps: 30}}"
```

## 访问视频流

启动服务器后，在浏览器中打开：

- **本地访问**: http://localhost:5000
- **局域网访问**: http://<你的IP地址>:5000

例如：http://192.168.1.100:5000

## 配置说明

### 机器人配置

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `--robot.type` | 机器人类型 | so101_follower |
| `--robot.port` | 串口设备 | /dev/ttyACM0 |
| `--robot.id` | 机器人ID | lil_guy |

### 摄像头配置

```yaml
--robot.cameras="{
  wrist: {
    type: opencv,
    index_or_path: 0,
    width: 640,
    height: 480,
    fps: 30
  },
  front: {
    type: opencv,
    index_or_path: 1,
    width: 640,
    height: 480,
    fps: 30
  }
}"
```

**摄像头索引说明**：
- `index_or_path: 0` - 第一个摄像头设备
- `index_or_path: 1` - 第二个摄像头设备
- `index_or_path: /dev/video0` - 指定设备路径

## 查找摄像头设备

使用以下命令查看可用的摄像头设备：

```bash
# 列出所有视频设备
ls -la /dev/video*

# 使用 v4l2-ctl 查看详细信息
v4l2-ctl --list-devices

# 测试摄像头
python -c "import cv2; cap = cv2.VideoCapture(0); print('Camera 0:', cap.isOpened()); cap.release()"
```

## 故障排除

### 1. 端口被占用

```bash
# 查看端口占用
lsof -i :5000

# 更改端口（修改 video_stream.py 最后一行）
app.run(host='0.0.0.0', port=8080, ...)
```

### 2. 摄像头无法访问

```bash
# 检查摄像头权限
ls -la /dev/video*

# 添加用户到 video 组
sudo usermod -a -G video $USER

# 重新登录后生效
```

### 3. 机器人连接失败

```bash
# 检查串口权限
ls -la /dev/ttyACM0

# 添加权限
sudo chmod 666 /dev/ttyACM0

# 或添加用户到 dialout 组
sudo usermod -a -G dialout $USER
```

### 4. 网页无法访问

```bash
# 检查防火墙
sudo ufw status

# 允许端口 5000
sudo ufw allow 5000

# 检查服务器是否运行
curl http://localhost:5000
```

## 性能优化

### 降低分辨率

```bash
--robot.cameras="{wrist: {type: opencv, index_or_path: 0, width: 320, height: 240, fps: 30}}"
```

### 降低帧率

```bash
--robot.cameras="{wrist: {type: opencv, index_or_path: 0, width: 640, height: 480, fps: 15}}"
```

### 调整 JPEG 质量

修改 `video_stream.py` 中的压缩质量：

```python
ret, buffer = cv2.imencode('.jpg', frame, [int(cv2.IMWRITE_JPEG_QUALITY), 60])  # 60-100
```

## 技术细节

- **框架**: Flask + OpenCV
- **视频格式**: MJPEG
- **线程模型**: 后台线程持续捕获帧，主线程响应 HTTP 请求
- **线程安全**: 使用 `threading.Lock()` 保护共享帧数据
- **延迟**: ~50-100ms（取决于网络和摄像头性能）

## 与主程序同时运行

视频流服务器可以与 `eval_lerobot.py` 同时运行，它们会共享同一个机器人连接：

```bash
# 终端1：启动视频流
./start_video_stream.sh

# 终端2：运行机器人控制
python eval_lerobot.py \
    --robot.type=so101_follower \
    --robot.port=/dev/ttyACM0 \
    --robot.id=lil_guy \
    --robot.cameras="{wrist: {type: opencv, index_or_path: 0, width: 640, height: 480, fps: 30}}"
```

**注意**: 两个程序会竞争串口访问，建议使用线程锁保护（已在 `eval_lerobot.py` 中实现）。

## 许可证

Apache-2.0

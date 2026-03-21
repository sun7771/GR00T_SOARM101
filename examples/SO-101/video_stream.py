#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import cv2
import numpy as np
import threading
import time
from flask import Flask, Response, render_template_string
from lerobot.robots import make_robot_from_config
import draccus

app = Flask(__name__)

robot = None
camera_keys = []
latest_frames = {}
frame_lock = threading.Lock()

HTML_TEMPLATE = """
<!DOCTYPE html>
<html>
<head>
    <title>Robot Camera Stream</title>
    <style>
        body {
            font-family: Arial, sans-serif;
            margin: 20px;
            background-color: #f0f0f0;
        }
        h1 {
            color: #333;
            text-align: center;
        }
        .camera-container {
            display: flex;
            flex-wrap: wrap;
            justify-content: center;
            gap: 20px;
            margin-top: 20px;
        }
        .camera-box {
            background: white;
            padding: 15px;
            border-radius: 10px;
            box-shadow: 0 2px 10px rgba(0,0,0,0.1);
        }
        .camera-box h2 {
            margin-top: 0;
            color: #666;
            font-size: 18px;
        }
        img {
            display: block;
            max-width: 100%;
            height: auto;
            border-radius: 5px;
        }
        .info {
            text-align: center;
            color: #666;
            margin-top: 20px;
        }
    </style>
</head>
<body>
    <h1>🤖 Robot Camera Stream</h1>
    <div class="camera-container">
        {% for camera in camera_keys %}
        <div class="camera-box">
            <h2>{{ camera }}</h2>
            <img src="/video_feed/{{ camera }}" alt="{{ camera }}">
        </div>
        {% endfor %}
    </div>
    <div class="info">
        <p>Refresh rate: ~30 FPS | Resolution: 640x480</p>
    </div>
</body>
</html>
"""


def update_frames():
    """后台线程：持续更新摄像头帧"""
    global robot, camera_keys, latest_frames
    
    while True:
        try:
            observation = robot.get_observation()
            
            with frame_lock:
                for key in camera_keys:
                    if key in observation:
                        frame = observation[key]
                        if isinstance(frame, np.ndarray):
                            latest_frames[key] = frame.copy()
        except Exception as e:
            print(f"Error capturing frame: {e}")
            time.sleep(0.1)
        
        time.sleep(0.033)  # ~30 FPS


def generate_stream(camera_key):
    """生成MJPEG视频流"""
    while True:
        with frame_lock:
            if camera_key in latest_frames:
                frame = latest_frames[camera_key]
                
                ret, buffer = cv2.imencode('.jpg', frame, [int(cv2.IMWRITE_JPEG_QUALITY), 80])
                if ret:
                    frame_bytes = buffer.tobytes()
                    yield (b'--frame\r\n'
                           b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
        
        time.sleep(0.033)


@app.route('/')
def index():
    """主页：显示所有摄像头"""
    return render_template_string(HTML_TEMPLATE, camera_keys=camera_keys)


@app.route('/video_feed/<camera_key>')
def video_feed(camera_key):
    """视频流端点"""
    return Response(generate_stream(camera_key),
                    mimetype='multipart/x-mixed-replace; boundary=frame')


@draccus.wrap()
def main(cfg):
    global robot, camera_keys
    
    print("Initializing robot...")
    robot = make_robot_from_config(cfg.robot)
    robot.connect()
    
    camera_keys = list(cfg.robot.cameras.keys())
    print(f"Available cameras: {camera_keys}")
    
    print("Starting frame update thread...")
    update_thread = threading.Thread(target=update_frames, daemon=True)
    update_thread.start()
    
    print(f"Starting web server on http://0.0.0.0:5000")
    print("Open this URL in your browser to view the camera streams")
    
    app.run(host='0.0.0.0', port=5000, debug=False, threaded=True)


if __name__ == '__main__':
    main()

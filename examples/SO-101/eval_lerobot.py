# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
This is the new Gr00T policy eval script with so100, so101 robot arm. Based on:
https://github.com/huggingface/lerobot/pull/777

Example command:

```shell

python eval_gr00t_so100.py \
    --robot.type=so100_follower \
    --robot.port=/dev/ttyACM0 \
    --robot.id=lil_guy \
    --robot.cameras="{ wrist: {type: opencv, index_or_path: 9, width: 640, height: 480, fps: 30}, front: {type: opencv, index_or_path: 15, width: 640, height: 480, fps: 30}}" \
    --policy_host=10.112.209.136 \
    --lang_instruction="Grab markers and place into pen holder."
```


First replay to ensure the robot is working:
```shell
python -m lerobot.replay \
    --robot.type=so100_follower \
    --robot.port=/dev/ttyACM0 \
    --robot.id=lil_guy \
    --dataset.repo_id=youliangtan/so100-table-cleanup \
    --dataset.episode=2
```
"""

import asyncio
import logging
import time
import threading
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass
from pprint import pformat
import cv2
import draccus
import matplotlib.pyplot as plt
import numpy as np
from scipy.fftpack import dct, idct
from scipy.signal import savgol_filter
from flask import Flask, Response, render_template_string
from lerobot.cameras.opencv.configuration_opencv import (  # noqa: F401
    OpenCVCameraConfig,
)
from lerobot.robots import (  # noqa: F401
    Robot,
    RobotConfig,
    koch_follower,
    make_robot_from_config,
    so101_follower
)
from lerobot.utils.utils import (
    init_logging,
    log_say,
)

# NOTE:
# Sometimes we would like to abstract different env, or run this on a separate machine
# User can just move this single python class method gr00t/eval/service.py
# to their code or do the following line below
# sys.path.append(os.path.expanduser("~/Isaac-GR00T/gr00t/eval/"))
# from service import ExternalRobotInferenceClient

from gr00t.eval.service import ExternalRobotInferenceClient

#################################################################################


class ActionSmoother:
    """高级动作平滑器 - 使用多种平滑算法减少抖动"""
    
    def __init__(self, robot_state_keys, window_size=5, method='ema', dct_keep_ratio=0.5):
        self.robot_state_keys = robot_state_keys
        self.window_size = window_size
        self.method = method
        self.dct_keep_ratio = dct_keep_ratio
        self.history = {key: [] for key in robot_state_keys}
        self.current_action = None
        
    def smooth(self, action_dict, joint_alpha_map):
        """应用平滑算法"""
        smoothed_action = {}
        
        for key in self.robot_state_keys:
            if key in joint_alpha_map:
                if self.method == 'ema':
                    smoothed_action[key] = self._ema_smooth(
                        action_dict[key], 
                        key, 
                        joint_alpha_map[key]
                    )
                elif self.method == 'moving_avg':
                    smoothed_action[key] = self._moving_avg_smooth(
                        action_dict[key], 
                        key
                    )
                elif self.method == 'savgol':
                    smoothed_action[key] = self._savgol_smooth(
                        action_dict[key], 
                        key
                    )
                elif self.method == 'dct':
                    smoothed_action[key] = self._dct_smooth(
                        action_dict[key], 
                        key
                    )
                else:
                    smoothed_action[key] = action_dict[key]
            else:
                smoothed_action[key] = action_dict[key]
        
        self.current_action = smoothed_action.copy()
        return smoothed_action
    
    def _ema_smooth(self, new_value, key, alpha):
        """指数移动平均平滑"""
        if len(self.history[key]) == 0:
            smoothed = new_value
        else:
            smoothed = alpha * new_value + (1 - alpha) * self.history[key][-1]
        
        self.history[key].append(smoothed)
        if len(self.history[key]) > self.window_size:
            self.history[key].pop(0)
        
        return smoothed
    
    def _moving_avg_smooth(self, new_value, key):
        """移动平均平滑"""
        self.history[key].append(new_value)
        if len(self.history[key]) > self.window_size:
            self.history[key].pop(0)
        
        return np.mean(self.history[key])
    
    def _savgol_smooth(self, new_value, key):
        """Savitzky-Golay滤波器平滑"""
        self.history[key].append(new_value)
        if len(self.history[key]) > self.window_size:
            self.history[key].pop(0)
        
        if len(self.history[key]) < 3:
            return np.mean(self.history[key])
        
        return savgol_filter(self.history[key], window_length=min(len(self.history[key]), 5), polyorder=2)[-1]
    
    def _dct_smooth(self, new_value, key):
        """DCT（离散余弦变换）平滑 - 去除高频噪声"""
        self.history[key].append(new_value)
        if len(self.history[key]) > self.window_size:
            self.history[key].pop(0)
        
        if len(self.history[key]) < 3:
            return np.mean(self.history[key])
        
        # 对历史数据进行DCT变换
        signal = np.array(self.history[key])
        dct_coeffs = dct(signal, type=2, norm='ortho')
        
        # 保留低频系数，去除高频噪声
        # 使用配置的保留比例
        keep_ratio = self.dct_keep_ratio
        keep_count = max(1, int(len(dct_coeffs) * keep_ratio))
        dct_coeffs_filtered = dct_coeffs.copy()
        dct_coeffs_filtered[keep_count:] = 0
        
        # 逆DCT变换得到平滑信号
        smoothed_signal = idct(dct_coeffs_filtered, type=2, norm='ortho')
        
        return smoothed_signal[-1]


class ActionInterpolator:
    """动作插值器 - 在动作块内进行平滑插值"""
    
    def __init__(self, robot_state_keys, interpolation_steps=3):
        self.robot_state_keys = robot_state_keys
        self.interpolation_steps = interpolation_steps
        
    def interpolate_action_chunk(self, action_chunk):
        """对动作块进行插值，使过渡更平滑"""
        if len(action_chunk) <= 1:
            return action_chunk
        
        interpolated_chunk = []
        
        for i in range(len(action_chunk) - 1):
            current_action = action_chunk[i]
            next_action = action_chunk[i + 1]
            
            interpolated_chunk.append(current_action)
            
            for step in range(1, self.interpolation_steps + 1):
                t = step / (self.interpolation_steps + 1)
                interpolated_action = {}
                
                for key in self.robot_state_keys:
                    interpolated_action[key] = (
                        (1 - t) * current_action[key] + 
                        t * next_action[key]
                    )
                
                interpolated_chunk.append(interpolated_action)
        
        interpolated_chunk.append(action_chunk[-1])
        return interpolated_chunk


class ObservationPrefetcher:
    """观测数据预取器 - 在后台异步获取观测数据"""
    
    def __init__(self, robot, executor, robot_lock):
        self.robot = robot
        self.executor = executor
        self.robot_lock = robot_lock
        self.current_obs = None
        self.prefetch_task = None
        self.loop = None
        
    async def start(self):
        """启动预取器"""
        self.loop = asyncio.get_event_loop()
        await self._prefetch_next()
        
    async def _prefetch_next(self):
        """在后台预取下一个观测数据"""
        if self.prefetch_task and not self.prefetch_task.done():
            return
            
        self.prefetch_task = asyncio.create_task(self._get_observation_async())
        
    async def _get_observation_async(self):
        """异步获取观测数据（使用线程锁保护串口访问）"""
        loop = asyncio.get_event_loop()
        
        def get_observation_with_lock():
            with self.robot_lock:
                return self.robot.get_observation()
        
        return await loop.run_in_executor(
            self.executor,
            get_observation_with_lock
        )
    
    async def get_observation(self):
        """获取当前观测数据，并立即启动下一次预取"""
        if self.current_obs is None:
            self.current_obs = await self._get_observation_async()
            
        obs = self.current_obs
        
        # 立即启动下一次预取
        await self._prefetch_next()
        
        # 等待预取完成
        if self.prefetch_task:
            self.current_obs = await self.prefetch_task
            
        return obs
    
    async def close(self):
        """关闭预取器"""
        if self.prefetch_task and not self.prefetch_task.done():
            self.prefetch_task.cancel()


class VideoStreamServer:
    """视频流服务器 - 在后台提供网页访问摄像头画面"""
    
    def __init__(self, robot, robot_lock, camera_keys, port=5000):
        self.robot = robot
        self.robot_lock = robot_lock
        self.camera_keys = camera_keys
        self.port = port
        self.latest_frames = {}
        self.frame_lock = threading.Lock()
        self.app = Flask(__name__)
        self.server_thread = None
        self.running = False
        
        self._setup_routes()
    
    def _setup_routes(self):
        """设置Flask路由"""
        
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
        
        @self.app.route('/')
        def index():
            return render_template_string(HTML_TEMPLATE, camera_keys=self.camera_keys)
        
        @self.app.route('/video_feed/<camera_key>')
        def video_feed(camera_key):
            return Response(self._generate_stream(camera_key),
                          mimetype='multipart/x-mixed-replace; boundary=frame')
    
    def _generate_stream(self, camera_key):
        """生成MJPEG视频流"""
        while self.running:
            with self.frame_lock:
                if camera_key in self.latest_frames:
                    frame = self.latest_frames[camera_key]
                    # 将BGR转换为RGB（OpenCV默认使用BGR，网页需要RGB）
                    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    ret, buffer = cv2.imencode('.jpg', frame_rgb, [int(cv2.IMWRITE_JPEG_QUALITY), 80])
                    if ret:
                        frame_bytes = buffer.tobytes()
                        yield (b'--frame\r\n'
                               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
            time.sleep(0.033)
    
    def update_frames(self):
        """后台线程：持续更新摄像头帧"""
        while self.running:
            try:
                with self.robot_lock:
                    observation = self.robot.get_observation()
                
                with self.frame_lock:
                    for key in self.camera_keys:
                        if key in observation:
                            frame = observation[key]
                            if isinstance(frame, np.ndarray):
                                self.latest_frames[key] = frame.copy()
            except Exception as e:
                print(f"Error capturing frame: {e}")
                time.sleep(0.1)
            
            time.sleep(0.033)
    
    def start(self):
        """启动视频流服务器"""
        self.running = True
        
        # 启动帧更新线程
        self.update_thread = threading.Thread(target=self.update_frames, daemon=True)
        self.update_thread.start()
        
        # 启动Flask服务器（在单独线程中）
        def run_flask():
            self.app.run(host='0.0.0.0', port=self.port, debug=False, threaded=True)
        
        self.server_thread = threading.Thread(target=run_flask, daemon=True)
        self.server_thread.start()
        
        print(f"✓ Video stream server started at http://0.0.0.0:{self.port}")
        print(f"  Available cameras: {self.camera_keys}")
    
    def stop(self):
        """停止视频流服务器"""
        self.running = False


class Gr00tRobotInferenceClient:
    """使用的确切键在modality.json中定义

    目前仅支持so100_follower、so101_follower
    根据modality.json修改此代码以支持具有其他键的其他机器人
    """
    #设置默认的 camera_keys 和 robot_state_keys
    def __init__(
        self,
        host="localhost",
        port=5555,
        camera_keys=[],
        robot_state_keys=[],
        show_images=False,
    ):
        self.policy = ExternalRobotInferenceClient(host=host, port=port)
        self.camera_keys = camera_keys
        self.robot_state_keys = robot_state_keys
        self.show_images = show_images
        assert (
            len(robot_state_keys) == 6
        ), f"robot_state_keys should be size 6, but got {len(robot_state_keys)} "
        self.modality_keys = ["single_arm", "gripper"]
        self.executor = ThreadPoolExecutor(max_workers=4)

    async def get_action_async(self, observation_dict, lang: str):
        """异步获取动作块"""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            self.executor,
            self._get_action_sync,
            observation_dict,
            lang
        )

    def _get_action_sync(self, observation_dict, lang: str):
        """同步获取动作块（在线程池中执行）"""
        # 首先添加图像
        obs_dict = {}
        for key in self.camera_keys:
            img = observation_dict[key]
            # 将BGR转换为RGB（策略模型期望RGB格式）
            if len(img.shape) == 3 and img.shape[2] == 3:
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            # 在发送到服务器之前将图像调整为224x224
            if img.shape[:2] != (224, 224):
                img = cv2.resize(img, (224, 224), interpolation=cv2.INTER_LINEAR)
            obs_dict[f"video.{key}"] = img

        # 显示图像
        if self.show_images:
            view_img(obs_dict)

        # 将所有单个浮点值的 dict[str, float] 状态转换为单个数组
        state = np.array([observation_dict[k] for k in self.robot_state_keys])
        obs_dict["state.single_arm"] = state[:5].astype(np.float32)
        obs_dict["state.gripper"] = state[5:6].astype(np.float32)

        # 预包装为列表,主循环会再套一层 -> 最终变为 [[lang]]
        obs_dict["language.annotation.human.task_description"] = [lang]

        # 为所有键添加一个虚拟维度（假设历史长度为1）
        for k in obs_dict:
            if isinstance(obs_dict[k], np.ndarray):
                obs_dict[k] = obs_dict[k][np.newaxis, ...]
                if k.startswith("video.") or k.startswith("state."):
                    obs_dict[k] = obs_dict[k][:, np.newaxis, ...]
            else:
                obs_dict[k] = [obs_dict[k]]  # [lang] -> [[lang]]

        # 通过策略服务器获取动作块
        action_chunk = self.policy.get_action(obs_dict)
        
        # 如果返回的是列表，提取第一个元素（动作数据字典）
        if isinstance(action_chunk, list) and len(action_chunk) > 0:
            action_data = action_chunk[0]  # 获取第一个元素
            
            # 检查是否包含预期的键
            if isinstance(action_data, dict) and 'single_arm' in action_data and 'gripper' in action_data:
                # 构造成原有格式的字典
                reformatted_chunk = {
                    f"action.{self.modality_keys[0]}": action_data['single_arm'][0],  # single_arm
                    f"action.{self.modality_keys[1]}": action_data['gripper'][0],      # gripper
                }
                
                # 使用原有的转换方法
                lerobot_actions = []
                action_horizon = reformatted_chunk[f"action.{self.modality_keys[0]}"].shape[0]
                for i in range(action_horizon):
                    action_dict = self._convert_to_lerobot_action(reformatted_chunk, i)
                    lerobot_actions.append(action_dict)
                
                return lerobot_actions
            else:
                raise ValueError(f"意外的动作数据格式: {action_data}")
        
        # 原有的字典处理逻辑（向后兼容）
        if isinstance(action_chunk, dict):
            lerobot_actions = []
            action_horizon = action_chunk[f"action.{self.modality_keys[0]}"].shape[0]
            for i in range(action_horizon):
                action_dict = self._convert_to_lerobot_action(action_chunk, i)
                lerobot_actions.append(action_dict)
            return lerobot_actions
        
        raise TypeError(f"不支持的 action_chunk 类型: {type(action_chunk)}")

    def get_action(self, observation_dict, lang: str):
        """同步获取动作块（向后兼容）"""
        return self._get_action_sync(observation_dict, lang)

    def _convert_to_lerobot_action(
        self, action_chunk: dict[str, np.array], idx: int
    ) -> dict[str, float]:
        """
        这是一个魔法函数，将动作块转换为 dict[str, float]
        这是因为动作块是 dict[str, np.array]
        我们想要将其转换为 dict[str, float]
        以便可以发送给机器人
        """
        concat_action = np.concatenate(
            [np.atleast_1d(action_chunk[f"action.{key}"][idx]) for key in self.modality_keys],
            axis=0,
        )
        assert len(concat_action) == len(self.robot_state_keys), "this should be size 6"
        # 将动作转换为 dict[str, float]
        action_dict = {key: concat_action[i] for i, key in enumerate(self.robot_state_keys)}
        return action_dict


#################################################################################


#展示图像的函数
def view_img(img, overlay_img=None):
    """
    这是一个matplotlib查看器，因为在lerobot环境中cv2.imshow可能不稳定
    """
    if isinstance(img, dict):
        # 水平堆叠图像
        img = np.concatenate([img[k] for k in img], axis=1)

    plt.imshow(img)
    plt.title("Camera View")
    plt.axis("off")
    plt.pause(0.001)  # 非阻塞显示
    plt.clf()  # 清除图像以显示下一帧


def print_yellow(text):
    print("\033[93m {}\033[00m".format(text))


@dataclass
class EvalConfig:
    robot: RobotConfig  # 要使用的机器人
    policy_host: str = "localhost"  # gr00t服务器的主机地址
    policy_port: int = 5555  # gr00t服务器的端口
    # todo：：调整动作块的长度
    action_horizon: int = 12# 从动作块中执行的动作数量
    lang_instruction: str = "Grab pens and place into pen holder."
    play_sounds: bool = False  # 是否播放声音
    timeout: int = 60  # 超时时间（秒）
    show_images: bool = False  # 是否显示图像
    use_sync: bool = False  # 是否使用同步版本（默认使用异步优化版本）
    enable_video_stream: bool = True  # 是否启用视频流服务器
    video_stream_port: int = 5000  # 视频流服务器端口

    ctrl_period: float =0.001  # 控制周期，单位为秒 0.001s=1000Hz
    
    # 平滑算法配置 
    smoothing_method: str = "savgol"  # 平滑方法: 'ema', 'moving_avg', 'savgol', 'dct'
    smoothing_window_size: int = 10 # 平滑窗口大小
    enable_interpolation: bool = True  # 是否启用动作块内插值
    interpolation_steps: int = 10 # 每个动作之间的插值步数
    
    # DCT平滑配置
    dct_keep_ratio: float = 0.3  # DCT保留低频系数的比例 (0.1-0.9)，越小越平滑
    
    # 速度限制配置（减小以减少抖动）
    max_delta_pos: float = 0.15  # 最大关节角度变化（弧度）
    
    # 为每个关节设置不同的平滑参数
    # 关节顺序: ['shoulder_pan.pos', 'shoulder_lift.pos', 'elbow_flex.pos', 'wrist_flex.pos', 'wrist_roll.pos', 'gripper.pos']
    # 增加平滑系数以减少抖动（alpha越小越平滑）
    shoulder_pan_alpha: float = 0.08    # 肩部转动 - 较大的关节，需要更多平滑
    shoulder_lift_alpha: float = 0.1  # 肩部抬升 - 承重关节，平滑一些
    elbow_flex_alpha: float = 0.08     # 肘部弯曲 - 中等平滑
    wrist_flex_alpha: float = 0.15      # 腕部弯曲 - 精细动作，少一些平滑
    wrist_roll_alpha: float = 0.15     # 腕部旋转 - 快速响应
    gripper_alpha: float = 0.2         # 夹爪 - 需要更多平滑避免抖动


def rad_speed_limit(target_pos, current_pos, max_delta_pos=0.5):

    # if delta_time is None:
    # 计算当前位置与目标位置的差值
    delta_pos = target_pos - current_pos

    # 计算运动缩放比例：最大关节角度变化 / (速度限制 × 控制周期)
    # dp / (vmax * dt)
    # motion_scale = np.max(np.abs(delta_pos)) / (velocity_limit * 0.001)
    motion_scale = np.max(np.abs(delta_pos)) / (max_delta_pos)
    
    # 如果运动幅度超过限制(motion_scale > 1)，则按比例缩放
    # 也就是不能大于 velocity_limit * delta_time
    limited_target_pos = current_pos + delta_pos / max(motion_scale, 1.0)

    return limited_target_pos

# ... (前面的代码保持不变: SPDX, Imports, Gr00tRobotInferenceClient, view_img, rad_speed_limit) ...

@draccus.wrap()
async def eval_async(cfg: EvalConfig):
    init_logging()
    logging.info(pformat(asdict(cfg)))

    # 步骤1：初始化机器人
    robot = make_robot_from_config(cfg.robot)
    robot.connect()

    camera_keys = list(cfg.robot.cameras.keys())
    print("camera_keys: ", camera_keys)

    log_say("Initializing robot", cfg.play_sounds, blocking=True)

    language_instruction = cfg.lang_instruction
    robot_state_keys = list(robot._motors_ft.keys())
    print("robot_state_keys: ", robot_state_keys)

    # 步骤2：初始化策略
    policy = Gr00tRobotInferenceClient(
        host=cfg.policy_host,
        port=cfg.policy_port,
        camera_keys=camera_keys,
        robot_state_keys=robot_state_keys,
    )
    log_say(
        "Initializing policy client with language instruction: " + language_instruction,
        cfg.play_sounds,
        blocking=True,
    )

    # 步骤3：创建线程锁保护机器人串口访问
    robot_lock = threading.Lock()
    
    # 步骤4：初始化视频流服务器（如果启用）
    video_stream_server = None
    if cfg.enable_video_stream:
        video_stream_server = VideoStreamServer(
            robot=robot,
            robot_lock=robot_lock,
            camera_keys=camera_keys,
            port=cfg.video_stream_port
        )
        video_stream_server.start()
    
    # 步骤5：初始化观测预取器
    executor = ThreadPoolExecutor(max_workers=4)
    obs_prefetcher = ObservationPrefetcher(robot, executor, robot_lock)
    await obs_prefetcher.start()
    
    # 步骤6：初始化平滑器和插值器
    action_smoother = ActionSmoother(
        robot_state_keys,
        window_size=cfg.smoothing_window_size,
        method=cfg.smoothing_method,
        dct_keep_ratio=cfg.dct_keep_ratio
    )
    action_interpolator = ActionInterpolator(
        robot_state_keys,
        interpolation_steps=cfg.interpolation_steps
    )
    
    print(f"平滑配置:")
    print(f"  方法: {cfg.smoothing_method}")
    print(f"  窗口大小: {cfg.smoothing_window_size}")
    print(f"  插值启用: {cfg.enable_interpolation}")
    print(f"  插值步数: {cfg.interpolation_steps}")
    print(f"  最大角度变化: {cfg.max_delta_pos} rad")
    if cfg.smoothing_method == 'dct':
        print(f"  DCT保留低频比例: {cfg.dct_keep_ratio}")

    previous_action = None
    
    joint_alpha_map = {
        'shoulder_pan.pos': cfg.shoulder_pan_alpha,
        'shoulder_lift.pos': cfg.shoulder_lift_alpha,
        'elbow_flex.pos': cfg.elbow_flex_alpha,
        'wrist_flex.pos': cfg.wrist_flex_alpha,
        'wrist_roll.pos': cfg.wrist_roll_alpha,
        'gripper.pos': cfg.gripper_alpha,
    }
    
    print("关节平滑参数配置:")
    for joint, alpha in joint_alpha_map.items():
        print(f"  {joint}: {alpha}")
    
    # 初始化平滑器和插值器
    action_smoother = ActionSmoother(
        robot_state_keys,
        window_size=cfg.smoothing_window_size,
        method=cfg.smoothing_method,
        dct_keep_ratio=cfg.dct_keep_ratio
    )
    action_interpolator = ActionInterpolator(
        robot_state_keys,
        interpolation_steps=cfg.interpolation_steps
    )
    
    print(f"平滑配置:")
    print(f"  方法: {cfg.smoothing_method}")
    print(f"  窗口大小: {cfg.smoothing_window_size}")
    print(f"  插值启用: {cfg.enable_interpolation}")
    print(f"  插值步数: {cfg.interpolation_steps}")
    print(f"  最大角度变化: {cfg.max_delta_pos} rad")
    if cfg.smoothing_method == 'dct':
        print(f"  DCT保留低频比例: {cfg.dct_keep_ratio}")

    # --- 频率和延迟统计变量初始化 ---
    last_loop_time = time.time()
    last_action_time = time.time()
    loop_count = 0
    action_count = 0
    print_interval = 1  # 外层循环打印间隔
    action_print_interval = 10  # 内层循环打印间隔（每10个动作打印一次）
    
    # 网络延迟统计
    network_latency_list = []
    max_latency_history = 100  # 保存最近100次延迟记录
    
    # 异步任务统计
    async_obs_time_list = []
    async_policy_time_list = []
    # ------------------------------------

    # 步骤4：运行异步评估循环
    try:
        while True:
            loop_start_time = time.time()
            
            # 异步获取实时观测数据（使用预取器）
            obs_start_time = time.time()
            observation_dict = await obs_prefetcher.get_observation()
            obs_time = time.time() - obs_start_time
            async_obs_time_list.append(obs_time)
            
            # 异步获取动作块（这部分包含网络延迟）
            policy_start_time = time.time()
            action_chunk = await policy.get_action_async(observation_dict, language_instruction)
            policy_time = time.time() - policy_start_time
            async_policy_time_list.append(policy_time)
            
            # 记录网络延迟
            network_latency_list.append(policy_time)
            if len(network_latency_list) > max_latency_history:
                network_latency_list.pop(0)

            # 应用插值（如果启用）
            if cfg.enable_interpolation:
                action_chunk = action_interpolator.interpolate_action_chunk(action_chunk)

            # 执行动作序列（保持同步以确保精确时序）
            for i in range(len(action_chunk)):
                action_dict = action_chunk[i]
                
                # 应用平滑算法
                smoothed_action = action_smoother.smooth(action_dict, joint_alpha_map)
                
                # 应用速度限制
                if previous_action is not None:
                    for key in smoothed_action:
                        smoothed_action[key] = rad_speed_limit(
                            target_pos=smoothed_action[key],
                            current_pos=previous_action[key],
                            max_delta_pos=cfg.max_delta_pos
                        )
                
                previous_action = smoothed_action.copy()
                
                with robot_lock:
                    robot.send_action(smoothed_action)
                time.sleep(cfg.ctrl_period)
                
                # 统计动作执行频率
                action_count += 1
                if action_count % action_print_interval == 0:
                    current_time = time.time()
                    action_dt = current_time - last_action_time
                    action_fps = action_print_interval / action_dt
                    print(f"\r[Action] 执行频率: {action_fps:.2f} Hz", end="")
                    last_action_time = current_time

            # --- 外层循环频率和网络延迟统计 ---
            loop_count += 1
            if loop_count % print_interval == 0:
                current_time = time.time()
                dt = current_time - last_loop_time
                loop_fps = print_interval / dt
                
                # 计算实际执行的动作数量（考虑插值）
                actual_action_count = len(action_chunk) if cfg.enable_interpolation else cfg.action_horizon
                total_action_fps = (print_interval * actual_action_count) / dt
                
                # 计算动作块执行时间和利用率
                action_execution_time = actual_action_count * cfg.ctrl_period
                idle_time = dt - action_execution_time
                action_chunk_utilization = (action_execution_time / dt) * 100 if dt > 0 else 0
                
                # 计算网络延迟统计
                avg_latency = np.mean(network_latency_list) * 1000  # 毫秒
                min_latency = np.min(network_latency_list) * 1000
                max_latency = np.max(network_latency_list) * 1000
                
                # 计算异步操作统计
                avg_obs_time = np.mean(async_obs_time_list[-10:]) * 1000 if async_obs_time_list else 0
                avg_policy_time = np.mean(async_policy_time_list[-10:]) * 1000 if async_policy_time_list else 0
                
                print(f"\n{'='*60}")
                print(f"[Loop {loop_count}] 异步性能统计")
                print(f"{'='*60}")
                print(f"周期频率: {loop_fps:.2f} Hz | 实际指令频率: {total_action_fps:.2f} Hz")
                print(f"异步获取观测耗时: {avg_obs_time:.2f} ms (平均)")
                print(f"异步策略推理耗时: {avg_policy_time:.2f} ms (平均)")
                print(f"总循环耗时: {dt*1000:.2f} ms")
                print(f"\n动作块利用统计:")
                print(f"  动作块长度: {cfg.action_horizon} | 插值后: {actual_action_count}")
                print(f"  动作执行时间: {action_execution_time*1000:.2f} ms")
                print(f"  空闲时间: {idle_time*1000:.2f} ms")
                print(f"  动作块利用率: {action_chunk_utilization:.1f}%")
                print(f"\n网络延迟统计 (最近{len(network_latency_list)}次):")
                print(f"  平均延迟: {avg_latency:.2f} ms")
                print(f"  最小延迟: {min_latency:.2f} ms")
                print(f"  最大延迟: {max_latency:.2f} ms")
                print(f"{'='*60}")
                last_loop_time = current_time
            # ------------------------------------
    finally:
        # 清理资源
        if video_stream_server:
            video_stream_server.stop()
        await obs_prefetcher.close()
        executor.shutdown(wait=True)


@draccus.wrap()
def eval(cfg: EvalConfig):
    """主入口函数，根据配置选择异步或同步版本"""
    if cfg.use_sync:
        print("使用同步版本 (use_sync=True)")
        return eval_sync(cfg)
    else:
        print("使用异步优化版本 (use_sync=False)")
        return asyncio.run(eval_async(cfg))


@draccus.wrap()
def eval_sync(cfg: EvalConfig):
    """原始同步版本（用于性能对比）"""
    init_logging()
    logging.info(pformat(asdict(cfg)))

    # 步骤1：初始化机器人
    robot = make_robot_from_config(cfg.robot)
    robot.connect()

    camera_keys = list(cfg.robot.cameras.keys())
    print("camera_keys: ", camera_keys)

    log_say("Initializing robot", cfg.play_sounds, blocking=True)

    language_instruction = cfg.lang_instruction
    robot_state_keys = list(robot._motors_ft.keys())
    print("robot_state_keys: ", robot_state_keys)

    # 步骤2：初始化策略
    policy = Gr00tRobotInferenceClient(
        host=cfg.policy_host,
        port=cfg.policy_port,
        camera_keys=camera_keys,
        robot_state_keys=robot_state_keys,
    )
    log_say(
        "Initializing policy client with language instruction: " + language_instruction,
        cfg.play_sounds,
        blocking=True,
    )

    previous_action = None
    
    joint_alpha_map = {
        'shoulder_pan.pos': cfg.shoulder_pan_alpha,
        'shoulder_lift.pos': cfg.shoulder_lift_alpha,
        'elbow_flex.pos': cfg.elbow_flex_alpha,
        'wrist_flex.pos': cfg.wrist_flex_alpha,
        'wrist_roll.pos': cfg.wrist_roll_alpha,
        'gripper.pos': cfg.gripper_alpha,
    }
    
    print("关节平滑参数配置:")
    for joint, alpha in joint_alpha_map.items():
        print(f"  {joint}: {alpha}")
    
    # 初始化平滑器和插值器
    action_smoother = ActionSmoother(
        robot_state_keys,
        window_size=cfg.smoothing_window_size,
        method=cfg.smoothing_method
    )
    action_interpolator = ActionInterpolator(
        robot_state_keys,
        interpolation_steps=cfg.interpolation_steps
    )
    
    print(f"平滑配置:")
    print(f"  方法: {cfg.smoothing_method}")
    print(f"  窗口大小: {cfg.smoothing_window_size}")
    print(f"  插值启用: {cfg.enable_interpolation}")
    print(f"  插值步数: {cfg.interpolation_steps}")
    print(f"  最大角度变化: {cfg.max_delta_pos} rad")

    # --- 频率和延迟统计变量初始化 ---
    last_loop_time = time.time()
    last_action_time = time.time()
    loop_count = 0
    action_count = 0
    print_interval = 1  # 外层循环打印间隔
    action_print_interval = 10  # 内层循环打印间隔（每10个动作打印一次）
    
    # 网络延迟统计
    network_latency_list = []
    max_latency_history = 100  # 保存最近100次延迟记录
    # ------------------------------------

    # 步骤3：运行评估循环
    while True:
        loop_start_time = time.time()
        
        # 获取实时图像
        observation_dict = robot.get_observation()
        obs_time = time.time() - loop_start_time
        
        # 获取动作块（这部分包含网络延迟）
        policy_start_time = time.time()
        action_chunk = policy.get_action(observation_dict, language_instruction)
        policy_time = time.time() - policy_start_time
        
        # 记录网络延迟
        network_latency_list.append(policy_time)
        if len(network_latency_list) > max_latency_history:
            network_latency_list.pop(0)

        # 应用插值（如果启用）
        if cfg.enable_interpolation:
            action_chunk = action_interpolator.interpolate_action_chunk(action_chunk)

        # 执行动作序列
        for i in range(len(action_chunk)):
            action_dict = action_chunk[i]
            
            # 应用平滑算法
            smoothed_action = action_smoother.smooth(action_dict, joint_alpha_map)
            
            # 应用速度限制
            if previous_action is not None:
                for key in smoothed_action:
                    smoothed_action[key] = rad_speed_limit(
                        target_pos=smoothed_action[key],
                        current_pos=previous_action[key],
                        max_delta_pos=cfg.max_delta_pos
                    )
            
            previous_action = smoothed_action.copy()
            
            robot.send_action(smoothed_action)
            time.sleep(cfg.ctrl_period)
            
            # 统计动作执行频率
            action_count += 1
            if action_count % action_print_interval == 0:
                current_time = time.time()
                action_dt = current_time - last_action_time
                action_fps = action_print_interval / action_dt
                print(f"\r[Action] 执行频率: {action_fps:.2f} Hz", end="")
                last_action_time = current_time

        # --- 外层循环频率和网络延迟统计 ---
        loop_count += 1
        if loop_count % print_interval == 0:
            current_time = time.time()
            dt = current_time - last_loop_time
            loop_fps = print_interval / dt
            
            # 计算实际执行的动作数量（考虑插值）
            actual_action_count = len(action_chunk) if cfg.enable_interpolation else cfg.action_horizon
            total_action_fps = (print_interval * actual_action_count) / dt
            
            # 计算动作块执行时间和利用率
            action_execution_time = actual_action_count * cfg.ctrl_period
            idle_time = dt - action_execution_time
            action_chunk_utilization = (action_execution_time / dt) * 100 if dt > 0 else 0
            
            # 计算网络延迟统计
            avg_latency = np.mean(network_latency_list) * 1000  # 毫秒
            min_latency = np.min(network_latency_list) * 1000
            max_latency = np.max(network_latency_list) * 1000
            
            print(f"\n{'='*60}")
            print(f"[Loop {loop_count}] 性能统计 (同步版本)")
            print(f"{'='*60}")
            print(f"周期频率: {loop_fps:.2f} Hz | 实际指令频率: {total_action_fps:.2f} Hz")
            print(f"获取观测耗时: {obs_time*1000:.2f} ms")
            print(f"策略推理耗时: {policy_time*1000:.2f} ms")
            print(f"总循环耗时: {dt*1000:.2f} ms")
            print(f"\n动作块利用统计:")
            print(f"  动作块长度: {cfg.action_horizon} | 插值后: {actual_action_count}")
            print(f"  动作执行时间: {action_execution_time*1000:.2f} ms")
            print(f"  空闲时间: {idle_time*1000:.2f} ms")
            print(f"  动作块利用率: {action_chunk_utilization:.1f}%")
            print(f"\n网络延迟统计 (最近{len(network_latency_list)}次):")
            print(f"  平均延迟: {avg_latency:.2f} ms")
            print(f"  最小延迟: {min_latency:.2f} ms")
            print(f"  最大延迟: {max_latency:.2f} ms")
            print(f"{'='*60}")
            last_loop_time = current_time
        # ------------------------------------

if __name__ == "__main__":
    eval()
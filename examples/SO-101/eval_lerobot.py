"""
================================================================================
GR00T 机器人策略评估脚本
================================================================================

本脚本用于在真实机器人上评估 NVIDIA GR00T 模型的策略推理性能。
支持 SO-100 和 SO-101 系列机械臂，通过远程推理服务器获取动作指令。

主要功能:
---------
1. 机器人控制: 连接并控制 SO-100/SO-101 机械臂
2. 策略推理: 通过网络调用远程 GR00T 模型获取动作块
3. 动作平滑: 提供多种平滑算法(EMA/移动平均/Savitzky-Golay/DCT/卡尔曼滤波)
4. 动作插值: 在动作块内进行插值，使运动更平滑
5. 视频流服务: 提供 Web 界面实时查看摄像头画面
6. 性能监控: 实时统计推理延迟、控制频率等性能指标
7. 动态优化: 根据实时性能自动调整控制参数

架构说明:
---------
- 异步版本(eval_async): 使用观测预取、线程池优化，推荐用于生产环境
- 同步版本(eval_sync): 简单直接的同步调用，用于性能对比测试

使用示例:
---------
python eval_lerobot.py \\
    --robot.type=so101_follower \\
    --robot.port=/dev/ttyACM0 \\
    --robot.id=lil_guy \\
    --robot.cameras="{ wrist: {type: opencv, index_or_path: 9}, front: {type: opencv, index_or_path: 15}}" \\
    --policy_host=10.112.209.136 \\
    --lang_instruction="Grab markers and place into pen holder."

依赖项:
-------
- lerobot: Hugging Face 机器人控制库
- gr00t: NVIDIA GR00T 模型推理服务
- Flask: 视频流 Web 服务器
- scipy: 信号处理(平滑算法)
- numpy: 数值计算

作者: Hanqin Sun
================================================================================
"""

# ==================== 标准库导入 ====================
import asyncio  # 异步编程支持,用于异步评估版本
import logging  # 日志记录,用于记录运行信息和错误
import time  # 时间相关功能,用于性能统计和延时控制
import threading  # 线程支持,用于并发执行和线程锁
from concurrent.futures import ThreadPoolExecutor  # 线程池执行器,用于异步执行阻塞操作
from dataclasses import asdict, dataclass  # 数据类支持,用于配置类定义
from pprint import pformat  # 格式化打印,用于美观地输出配置信息

# ==================== 第三方库导入 ====================
import cv2  # OpenCV,用于图像处理和摄像头操作
import draccus  # 命令行参数解析库,用于配置管理
import matplotlib.pyplot as plt  # Matplotlib,用于图像显示
import numpy as np  # NumPy,用于数值计算和数组操作
from scipy.fftpack import dct, idct  # 离散余弦变换,用于DCT平滑算法
from scipy.signal import savgol_filter  # Savitzky-Golay滤波器,用于信号平滑
from flask import Flask, Response, render_template_string  # Flask,用于视频流Web服务器

# ==================== LeRobot 库导入 ====================
from lerobot.cameras.opencv.configuration_opencv import (  
    OpenCVCameraConfig,  # OpenCV摄像头配置类
)
from lerobot.robots import (  
    Robot,  # 机器人基类
    RobotConfig,  # 机器人配置基类
    koch_follower,  # Koch机器人配置
    make_robot_from_config,  # 从配置创建机器人的工厂函数
    so101_follower  # SO-101机器人配置
)
from lerobot.utils.utils import (
    init_logging,  # 初始化日志系统
    log_say,  # 日志并语音播报
)

# ==================== GR00T 模型服务导入 ====================
from gr00t.eval.service import ExternalRobotInferenceClient  # GR00T外部推理客户端


#################################################################################


class ActionSmoother:
    """
    高级动作平滑器 - 使用多种平滑算法减少机器人动作抖动
    
    该类实现了多种平滑算法，用于处理策略模型输出的动作指令，
    减少因模型输出不稳定或网络延迟导致的机器人动作抖动。
    
    支持的平滑算法:
    ---------------
    - ema: 指数移动平均 (Exponential Moving Average)
           计算公式: smoothed = alpha * new_value + (1 - alpha) * prev_value
           特点: 响应快，适合快速变化的动作
           
    - moving_avg: 简单移动平均 (Simple Moving Average)
                  计算公式: smoothed = mean(history)
                  特点: 平滑效果好，但延迟较大
                  
    - savgol: Savitzky-Golay 滤波器
              基于多项式拟合的平滑方法
              特点: 保持信号形状的同时平滑噪声
              
    - dct: 离散余弦变换平滑
           通过DCT变换去除高频噪声
           特点: 适合处理周期性信号
           
    - kalman: 卡尔曼滤波
              基于状态估计的最优滤波器
              特点: 适合处理带噪声的观测值，可调节过程噪声和测量噪声
    
    - savgol_outlier: Savitzky-Golay滤波 + 离群值剔除
              先使用IQR方法检测并剔除离群值，再应用Savitzky-Golay滤波
              特点: 能有效去除异常突变点，同时保持信号形状，适合处理偶发异常
    
    - one_euro_outlier: One-Euro Filter + 离群值剔除
              自适应低通滤波器，根据信号速度动态调整截止频率
              特点: 在保持平滑的同时减少延迟，结合离群值检测增强鲁棒性
    
    - kalman_predict: 卡尔曼滤波 + 离群值剔除
              基于状态估计的最优滤波器，结合IQR离群值检测
              特点: 能够最优估计真实状态，同时抑制异常值干扰
    
    使用示例:
    ---------
    >>> smoother = ActionSmoother(
    ...     robot_state_keys=['shoulder_pan.pos', 'gripper.pos'],
    ...     window_size=10,
    ...     method='savgol'
    ... )
    >>> smoothed = smoother.smooth(action_dict, joint_alpha_map)
    
    属性:
    -----
    robot_state_keys : list
        机器人状态键名列表，如 ['shoulder_pan.pos', 'gripper.pos']
    window_size : int
        历史数据窗口大小，用于移动平均等算法
    method : str
        平滑算法名称 ('ema', 'moving_avg', 'savgol', 'dct', 'kalman')
    history : dict
        各关节的历史值存储
    """
    
    def __init__(self, robot_state_keys, window_size=5, method='ema', dct_keep_ratio=0.5, savgol_window_length=5, kalman_process_noise=0.01, kalman_measurement_noise=0.1, outlier_threshold=2.0, one_euro_min_cutoff=1.0, one_euro_beta=0.007, one_euro_d_cutoff=1.0):
        """
        初始化动作平滑器
        
        参数:
        -----
        robot_state_keys : list
            机器人状态键名列表，对应各关节名称
            例如: ['shoulder_pan.pos', 'shoulder_lift.pos', 'elbow_flex.pos', 
                   'wrist_flex.pos', 'wrist_roll.pos', 'gripper.pos']
        window_size : int, 默认=5
            历史数据窗口大小，影响平滑程度
            值越大平滑效果越强，但延迟也越大
        method : str, 默认='ema'
            平滑算法选择: 'ema', 'moving_avg', 'savgol', 'dct', 'kalman', 'savgol_outlier', 'one_euro_outlier', 'kalman_predict'
        dct_keep_ratio : float, 默认=0.5
            DCT平滑时保留低频系数的比例 (0.0-1.0)
            值越小越平滑，但可能丢失细节
        savgol_window_length : int, 默认=5
            Savitzky-Golay滤波窗口长度（必须为奇数且>=3）
        kalman_process_noise : float, 默认=0.01
            卡尔曼滤波过程噪声Q，值越大响应越快但噪声更多
        kalman_measurement_noise : float, 默认=0.1
            卡尔曼滤波测量噪声R，值越大平滑效果越强
        outlier_threshold : float, 默认=2.0
            离群值检测阈值（基于IQR方法），值越小剔除越严格
        one_euro_min_cutoff : float, 默认=1.0
            One-Euro Filter最小截止频率（Hz），越小越平滑
        one_euro_beta : float, 默认=0.007
            One-Euro Filter截止频率斜率系数，越大跟踪越快
        one_euro_d_cutoff : float, 默认=1.0
            One-Euro Filter导数截止频率（Hz）
        """
        self.robot_state_keys = robot_state_keys
        self.window_size = window_size
        self.method = method
        self.dct_keep_ratio = dct_keep_ratio
        self.savgol_window_length = savgol_window_length
        self.history = {key: [] for key in robot_state_keys}
        self.current_action = None
        
        self.kalman_process_noise = kalman_process_noise
        self.kalman_measurement_noise = kalman_measurement_noise
        self.kalman_x = {key: None for key in robot_state_keys}
        self.kalman_P = {key: None for key in robot_state_keys}
        
        self.outlier_threshold = outlier_threshold
        
        self.one_euro_min_cutoff = one_euro_min_cutoff
        self.one_euro_beta = one_euro_beta
        self.one_euro_d_cutoff = one_euro_d_cutoff
        self.one_euro_x = {key: None for key in robot_state_keys}
        self.one_euro_dx = {key: None for key in robot_state_keys}
        self.one_euro_t = {key: None for key in robot_state_keys}
        
    def smooth(self, action_dict, joint_alpha_map):
        """
        对动作字典应用平滑算法
        
        参数:
        -----
        action_dict : dict
            原始动作字典，键为关节名，值为目标位置
        joint_alpha_map : dict
            各关节的平滑系数映射（仅用于EMA方法）
            键为关节名，值为alpha系数(0-1)
            alpha越大响应越快，越小越平滑
        
        返回:
        -----
        dict : 平滑后的动作字典
        """
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
                elif self.method == 'kalman':
                    smoothed_action[key] = self._kalman_smooth(
                        action_dict[key], 
                        key
                    )
                elif self.method == 'savgol_outlier':
                    smoothed_action[key] = self._savgol_outlier_smooth(
                        action_dict[key], 
                        key
                    )
                elif self.method == 'one_euro_outlier':
                    smoothed_action[key] = self._one_euro_outlier_smooth(
                        action_dict[key], 
                        key
                    )
                elif self.method == 'kalman_predict':
                    smoothed_action[key] = self._kalman_predict_smooth(
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
        """
        指数移动平均平滑 (Exponential Moving Average)
        
        公式: smoothed = alpha * new_value + (1 - alpha) * prev_value
        
        参数:
        -----
        new_value : float
            新的动作值
        key : str
            关节名称
        alpha : float
            平滑系数 (0-1)，值越大响应越快
        
        返回:
        -----
        float : 平滑后的值
        """
        if len(self.history[key]) == 0:
            smoothed = new_value
        else:
            smoothed = alpha * new_value + (1 - alpha) * self.history[key][-1]
        
        self.history[key].append(smoothed)
        if len(self.history[key]) > self.window_size:
            self.history[key].pop(0)
        
        return smoothed
    
    def _moving_avg_smooth(self, new_value, key):
        """
        简单移动平均平滑 (Simple Moving Average)
        
        计算历史窗口内所有值的平均
        
        参数:
        -----
        new_value : float
            新的动作值
        key : str
            关节名称
        
        返回:
        -----
        float : 平滑后的值（历史均值）
        """
        self.history[key].append(new_value)
        if len(self.history[key]) > self.window_size:
            self.history[key].pop(0)
        
        return np.mean(self.history[key])
    
    def _savgol_smooth(self, new_value, key):
        """
        Savitzky-Golay 滤波器平滑
        
        基于多项式拟合的平滑方法，能保持信号的高频特征
        
        参数:
        -----
        new_value : float
            新的动作值
        key : str
            关节名称
        
        返回:
        -----
        float : 平滑后的值
        """
        self.history[key].append(new_value)
        if len(self.history[key]) > self.window_size:
            self.history[key].pop(0)
        
        if len(self.history[key]) < 3:
            return np.mean(self.history[key])
        
        actual_window = min(len(self.history[key]), self.savgol_window_length)
        if actual_window < 3:
            actual_window = 3
        if actual_window % 2 == 0:
            actual_window -= 1
        
        return savgol_filter(self.history[key], window_length=actual_window, polyorder=2)[-1]
    
    def _dct_smooth(self, new_value, key):
        """
        DCT（离散余弦变换）平滑 - 去除高频噪声
        
        通过离散余弦变换将信号转换到频域，保留低频成分，
        去除高频噪声后逆变换回时域。
        
        参数:
        -----
        new_value : float
            新的动作值
        key : str
            关节名称
        
        返回:
        -----
        float : 平滑后的值
        
        原理:
        -----
        1. 对历史数据进行DCT-II变换
        2. 保留前keep_ratio比例的低频系数
        3. 高频系数置零（去除噪声）
        4. 逆DCT变换得到平滑信号
        """
        self.history[key].append(new_value)
        if len(self.history[key]) > self.window_size:
            self.history[key].pop(0)
        
        if len(self.history[key]) < 3:
            return np.mean(self.history[key])
        
        signal = np.array(self.history[key])
        dct_coeffs = dct(signal, type=2, norm='ortho')
        
        keep_ratio = self.dct_keep_ratio
        keep_count = max(1, int(len(dct_coeffs) * keep_ratio))
        dct_coeffs_filtered = dct_coeffs.copy()
        dct_coeffs_filtered[keep_count:] = 0
        
        smoothed_signal = idct(dct_coeffs_filtered, type=2, norm='ortho')
        
        return smoothed_signal[-1]
    
    def _kalman_smooth(self, new_value, key):
        """
        卡尔曼滤波平滑 - 状态估计滤波器
        
        卡尔曼滤波是一种最优递归滤波器，适合处理带噪声的观测值。
        通过预测-更新循环估计真实状态。
        
        参数:
        -----
        new_value : float
            新的观测值（动作值）
        key : str
            关节名称
        
        返回:
        -----
        float : 状态估计值（平滑后的值）
        
        算法步骤:
        ---------
        1. 预测: x_pred = x_prev, P_pred = P_prev + Q
        2. 计算卡尔曼增益: K = P_pred / (P_pred + R)
        3. 更新: x = x_pred + K * (z - x_pred), P = (1 - K) * P_pred
        
        其中:
        - Q: 过程噪声协方差，越大响应越快
        - R: 测量噪声协方差，越大平滑效果越强
        - K: 卡尔曼增益
        """
        Q = self.kalman_process_noise
        R = self.kalman_measurement_noise
        
        if self.kalman_x[key] is None:
            self.kalman_x[key] = new_value
            self.kalman_P[key] = 1.0
            return new_value
        
        x_pred = self.kalman_x[key]
        P_pred = self.kalman_P[key] + Q
        
        K = P_pred / (P_pred + R)
        
        self.kalman_x[key] = x_pred + K * (new_value - x_pred)
        self.kalman_P[key] = (1 - K) * P_pred
        
        self.history[key].append(self.kalman_x[key])
        if len(self.history[key]) > self.window_size:
            self.history[key].pop(0)
        
        return self.kalman_x[key]
    
    def _savgol_outlier_smooth(self, new_value, key):
        """
        Savitzky-Golay滤波 + 离群值剔除平滑
        
        先使用IQR方法检测并剔除离群值，然后对处理后的数据应用Savitzky-Golay滤波。
        这种组合方法能有效去除异常突变点，同时保持信号的整体形状。
        
        参数:
        -----
        new_value : float
            新的动作值
        key : str
            关节名称
        
        返回:
        -----
        float : 平滑后的值
        
        算法步骤:
        ---------
        1. 将新值添加到历史数据
        2. 使用IQR方法检测离群值
        3. 用中位数或插值替换离群值
        4. 对处理后的数据应用Savitzky-Golay滤波
        """
        self.history[key].append(new_value)
        if len(self.history[key]) > self.window_size:
            self.history[key].pop(0)
        
        if len(self.history[key]) < 3:
            return np.mean(self.history[key])
        
        data = np.array(self.history[key])
        
        Q1 = np.percentile(data, 25)
        Q3 = np.percentile(data, 75)
        IQR = Q3 - Q1
        
        lower_bound = Q1 - self.outlier_threshold * IQR
        upper_bound = Q3 + self.outlier_threshold * IQR
        
        cleaned_data = data.copy()
        outlier_mask = (data < lower_bound) | (data > upper_bound)
        
        if np.any(outlier_mask):
            median_val = np.median(data[~outlier_mask]) if np.any(~outlier_mask) else np.median(data)
            cleaned_data[outlier_mask] = median_val
        
        actual_window = min(len(cleaned_data), self.savgol_window_length)
        if actual_window < 3:
            actual_window = 3
        if actual_window % 2 == 0:
            actual_window -= 1
        
        smoothed = savgol_filter(cleaned_data, window_length=actual_window, polyorder=2)[-1]
        
        return smoothed
    
    def _one_euro_outlier_smooth(self, new_value, key):
        """
        One-Euro Filter + 离群值剔除平滑
        
        One-Euro Filter是一种自适应低通滤波器，能够根据信号速度动态调整截止频率，
        在保持平滑的同时减少延迟。结合IQR离群值检测，先剔除异常值再应用滤波。
        
        参数:
        -----
        new_value : float
            新的动作值
        key : str
            关节名称
        
        返回:
        -----
        float : 平滑后的值
        
        算法步骤:
        ---------
        1. 使用IQR方法检测离群值
        2. 如果是离群值，用历史数据的中位数或One-Euro滤波值替换
        3. 对处理后的值应用One-Euro Filter
        4. 动态调整截止频率：cutoff = min_cutoff + beta * |dx|
        """
        current_time = time.time()
        
        if self.one_euro_x[key] is None:
            self.one_euro_x[key] = new_value
            self.one_euro_dx[key] = 0.0
            self.one_euro_t[key] = current_time
            self.history[key].append(new_value)
            return new_value
        
        dt = current_time - self.one_euro_t[key]
        if dt <= 0:
            dt = 0.001
        
        self.history[key].append(new_value)
        if len(self.history[key]) > self.window_size:
            self.history[key].pop(0)
        
        data = np.array(self.history[key])
        
        Q1 = np.percentile(data, 25)
        Q3 = np.percentile(data, 75)
        IQR = Q3 - Q1
        
        lower_bound = Q1 - self.outlier_threshold * IQR
        upper_bound = Q3 + self.outlier_threshold * IQR
        
        is_outlier = (new_value < lower_bound) | (new_value > upper_bound)
        
        if is_outlier and len(self.history[key]) > 1:
            non_outlier_data = data[~((data < lower_bound) | (data > upper_bound))]
            if len(non_outlier_data) > 0:
                filtered_value = np.median(non_outlier_data)
            else:
                filtered_value = self.one_euro_x[key]
        else:
            filtered_value = new_value
        
        dx = (filtered_value - self.one_euro_x[key]) / dt
        edx = self._one_euro_alpha(dt, self.one_euro_d_cutoff) * dx
        self.one_euro_dx[key] = edx
        
        cutoff = self.one_euro_min_cutoff + self.one_euro_beta * abs(edx)
        alpha = self._one_euro_alpha(dt, cutoff)
        self.one_euro_x[key] = alpha * filtered_value + (1 - alpha) * self.one_euro_x[key]
        self.one_euro_t[key] = current_time
        
        return self.one_euro_x[key]
    
    def _one_euro_alpha(self, dt, cutoff):
        """
        One-Euro Filter的alpha系数计算
        
        参数:
        -----
        dt : float
            时间间隔（秒）
        cutoff : float
            截止频率（Hz）
        
        返回:
        -----
        float : alpha系数
        """
        tau = 1.0 / (2 * np.pi * cutoff)
        return dt / (tau + dt)
    
    def _kalman_predict_smooth(self, new_value, key):
        """
        卡尔曼预测滤波 + 离群值剔除
        
        使用带速度预测的卡尔曼滤波器，结合IQR离群值检测。
        在标准卡尔曼滤波基础上增加了基于速度的预测步骤，
        能够更好地处理快速变化的信号，并抑制异常值。
        
        参数:
        -----
        new_value : float
            新的动作值
        key : str
            关节名称
        
        返回:
        -----
        float : 滤波后的值
        
        算法步骤:
        ---------
        1. 使用IQR方法检测离群值
        2. 如果是离群值，用预测值或历史值替换
        3. 标准卡尔曼滤波预测-更新步骤
        4. 维护状态[x, dx]（位置和速度）
        """
        self.history[key].append(new_value)
        if len(self.history[key]) > self.window_size:
            self.history[key].pop(0)
        
        Q = self.kalman_process_noise
        R = self.kalman_measurement_noise
        
        if self.kalman_x[key] is None:
            self.kalman_x[key] = new_value
            self.kalman_P[key] = 1.0
            return new_value
        
        data = np.array(self.history[key])
        
        Q1 = np.percentile(data, 25)
        Q3 = np.percentile(data, 75)
        IQR = Q3 - Q1
        
        lower_bound = Q1 - self.outlier_threshold * IQR
        upper_bound = Q3 + self.outlier_threshold * IQR
        
        is_outlier = (new_value < lower_bound) | (new_value > upper_bound)
        
        if is_outlier:
            if len(self.history[key]) > 1:
                non_outlier_data = data[~((data < lower_bound) | (data > upper_bound))]
                if len(non_outlier_data) > 0:
                    filtered_value = np.median(non_outlier_data)
                else:
                    filtered_value = self.kalman_x[key]
            else:
                filtered_value = self.kalman_x[key]
        else:
            filtered_value = new_value
        
        x_pred = self.kalman_x[key]
        P_pred = self.kalman_P[key] + Q
        
        K = P_pred / (P_pred + R)
        
        self.kalman_x[key] = x_pred + K * (filtered_value - x_pred)
        self.kalman_P[key] = (1 - K) * P_pred
        
        return self.kalman_x[key]


class ActionInterpolator:
    """
    动作插值器 - 在动作块内进行平滑插值
    
    该类用于在策略模型输出的动作块之间进行线性插值，
    使机器人运动更加平滑连续，避免动作之间的突变。
    
    原理说明:
    ---------
    策略模型通常输出一个动作块（action chunk），包含多个时间步的动作。
    如果直接执行这些动作，可能会在动作之间产生跳变。
    通过插值，在每个动作之间插入中间帧，使运动更平滑。
    
    使用示例:
    ---------
    >>> interpolator = ActionInterpolator(robot_state_keys, interpolation_steps=5)
    >>> interpolated_chunk = interpolator.interpolate_action_chunk(action_chunk)
    """
    
    def __init__(self, robot_state_keys, interpolation_steps=3):
        """
        初始化动作插值器
        
        参数:
        -----
        robot_state_keys : list
            机器人状态键名列表
        interpolation_steps : int, 默认=3
            每两个动作之间的插值步数
            值越大运动越平滑，但执行时间越长
        """
        self.robot_state_keys = robot_state_keys
        self.interpolation_steps = interpolation_steps
        
    def interpolate_action_chunk(self, action_chunk):
        """
        对动作块进行线性插值
        
        在动作块中每两个相邻动作之间插入中间帧，
        使用线性插值计算中间值。
        
        参数:
        -----
        action_chunk : list[dict]
            原始动作块，每个元素是一个动作字典
        
        返回:
        -----
        list[dict] : 插值后的动作块
        
        示例:
        -----
        原始动作块: [A, B, C] (长度3)
        插值步数: 2
        插值后: [A, A->B_1, A->B_2, B, B->C_1, B->C_2, C] (长度7)
        """
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
    """
    观测数据预取器 - 环形缓存区异步预取
    
    该类实现了一个异步观测数据预取机制，使用环形缓存区存储预取的数据。
    通过在后台异步获取观测数据，减少主循环的等待时间，提高整体控制频率。
    
    工作原理:
    ---------
    1. 使用环形缓存区（circular buffer）存储预取的观测数据
    2. 后台线程异步调用 robot.get_observation() 获取数据
    3. 主循环直接从缓存读取，无需等待
    4. 使用线程锁保护机器人串口访问，避免冲突
    
    性能优势:
    ---------
    - 减少主循环阻塞时间
    - 提高控制频率
    - 平滑网络延迟影响
    
    属性:
    -----
    buffer_size : int
        环形缓存区大小
    hit_count : int
        缓存命中次数
    miss_count : int
        缓存未命中次数
    """
    
    def __init__(self, robot, executor, robot_lock, buffer_size=4):
        """
        初始化观测数据预取器
        
        参数:
        -----
        robot : Robot
            机器人实例，用于获取观测数据
        executor : ThreadPoolExecutor
            线程池执行器，用于异步执行
        robot_lock : threading.Lock
            机器人访问锁，保护串口访问
        buffer_size : int, 默认=4
            环形缓存区大小
        """
        self.robot = robot
        self.executor = executor
        self.robot_lock = robot_lock
        self.loop = None
        
        self.buffer_size = buffer_size
        self.buffer = [None] * buffer_size
        self.read_index = 0
        self.write_index = 0
        self.buffer_count = 0
        
        self.prefetch_tasks = []
        self.max_prefetch = 2
        
        self.hit_count = 0
        self.miss_count = 0
        self.prefetch_count = 0
        
    async def start(self):
        """
        启动预取器
        
        初始化事件循环并预填充缓存区
        """
        self.loop = asyncio.get_event_loop()
        await self._prefetch_multiple(self.buffer_size - 1)
        
    async def _prefetch_next(self):
        """
        在后台预取下一个观测数据
        
        检查缓存区状态和活动任务数，决定是否启动新的预取任务
        """
        if self.buffer_count >= self.buffer_size:
            return
            
        active_tasks = [t for t in self.prefetch_tasks if not t.done()]
        if len(active_tasks) >= self.max_prefetch:
            return
            
        task = asyncio.create_task(self._get_observation_async())
        self.prefetch_tasks.append(task)
        self.prefetch_count += 1
        
    async def _prefetch_multiple(self, count):
        """
        预取多个观测数据
        
        参数:
        -----
        count : int
            要预取的数据数量
        """
        for _ in range(count):
            await self._prefetch_next()
            await asyncio.sleep(0.001)
        
    async def _get_observation_async(self):
        """
        异步获取观测数据
        
        使用线程池执行器在后台调用 robot.get_observation()，
        并使用线程锁保护串口访问。
        
        返回:
        -----
        dict : 观测数据字典
        """
        loop = asyncio.get_event_loop()
        
        def get_observation_with_lock():
            with self.robot_lock:
                return self.robot.get_observation()
        
        obs = await loop.run_in_executor(
            self.executor,
            get_observation_with_lock
        )
        
        self.buffer[self.write_index] = obs
        self.write_index = (self.write_index + 1) % self.buffer_size
        self.buffer_count = min(self.buffer_count + 1, self.buffer_size)
        
        return obs
    
    async def get_observation(self):
        """
        获取当前观测数据
        
        从环形缓存区读取数据，如果缓存为空则等待预取完成。
        读取后会触发新的预取任务。
        
        返回:
        -----
        dict : 观测数据字典
        """
        if self.buffer_count == 0:
            self.miss_count += 1
            if self.prefetch_tasks:
                for task in self.prefetch_tasks:
                    if not task.done():
                        await task
                        break
            else:
                await self._get_observation_async()
        else:
            self.hit_count += 1
        
        obs = self.buffer[self.read_index]
        self.read_index = (self.read_index + 1) % self.buffer_size
        self.buffer_count = max(self.buffer_count - 1, 0)
        
        await self._prefetch_next()
        
        return obs
    
    def get_cache_stats(self):
        """
        获取缓存统计信息
        
        返回:
        -----
        dict : 包含命中率、缓存利用率等统计信息
        """
        total = self.hit_count + self.miss_count
        hit_rate = self.hit_count / total if total > 0 else 0
        buffer_utilization = self.buffer_count / self.buffer_size if self.buffer_size > 0 else 0
        return {
            'hit_count': self.hit_count,
            'miss_count': self.miss_count,
            'hit_rate': hit_rate,
            'total': total,
            'buffer_count': self.buffer_count,
            'buffer_utilization': buffer_utilization,
            'prefetch_count': self.prefetch_count
        }
    
    async def close(self):
        """
        关闭预取器
        
        取消所有未完成的预取任务并打印统计信息
        """
        for task in self.prefetch_tasks:
            if not task.done():
                task.cancel()
        
        stats = self.get_cache_stats()
        print(f"\n环形缓存区统计:")
        print(f"  命中次数: {stats['hit_count']}")
        print(f"  未命中次数: {stats['miss_count']}")
        print(f"  命中率: {stats['hit_rate']*100:.2f}%")
        print(f"  缓存区大小: {self.buffer_size}")
        print(f"  当前缓存数: {stats['buffer_count']}")
        print(f"  缓存利用率: {stats['buffer_utilization']*100:.2f}%")
        print(f"  总预取次数: {stats['prefetch_count']}")


class AdaptiveOptimizer:
    """
    动态优化器 - 根据实时性能自动调整参数
    
    该类实现了自适应参数调整机制，根据实时性能指标（如动作块利用率、
    循环频率、推理时间等）自动调整控制参数，以优化机器人控制性能。
    
    调整策略:
    ---------
    1. 动作块利用率过低时，减少插值步数以提高响应速度
    2. 动作块利用率过高时，增加插值步数以提高平滑度
    3. 循环频率过低时，减小控制周期以提高控制频率
    4. 循环频率过高时，增大控制周期以降低CPU负载
    
    属性:
    -----
    enabled : bool
        是否启用动态优化
    interval : int
        调整间隔（每多少次循环调整一次）
    interpolation_steps : int
        当前插值步数
    ctrl_period : float
        当前控制周期
    """
    
    def __init__(self, cfg):
        self.cfg = cfg
        self.enabled = cfg.enable_adaptive
        self.interval = cfg.adaptive_interval
        self.loop_count = 0
        
        # 当前参数
        self.interpolation_steps = cfg.interpolation_steps
        self.ctrl_period = cfg.ctrl_period
        
        # 性能历史
        self.utilization_history = []
        self.loop_fps_history = []
        self.policy_time_history = []
        
        # 调整统计
        self.adjustment_count = 0
        self.interpolation_adjustments = 0
        self.ctrl_period_adjustments = 0
        
    def should_adjust(self):
        """检查是否需要调整参数"""
        if not self.enabled:
            return False
        return self.loop_count % self.interval == 0
    
    def adjust(self, action_chunk_utilization, loop_fps, policy_time):
        """根据性能调整参数"""
        self.loop_count += 1
        
        if not self.should_adjust():
            return False
        
        # 记录历史数据
        self.utilization_history.append(action_chunk_utilization)
        self.loop_fps_history.append(loop_fps)
        self.policy_time_history.append(policy_time)
        
        # 只保留最近50次数据
        if len(self.utilization_history) > 50:
            self.utilization_history.pop(0)
            self.loop_fps_history.pop(0)
            self.policy_time_history.pop(0)
        
        # 计算平均值
        avg_utilization = np.mean(self.utilization_history[-10:])
        avg_loop_fps = np.mean(self.loop_fps_history[-10:])
        avg_policy_time = np.mean(self.policy_time_history[-10:])
        
        # 调整策略
        adjustments = []
        
        # 1. 根据动作块利用率调整插值步数
        if avg_utilization < self.cfg.target_utilization * 0.8:
            # 利用率过低，减少插值
            if self.interpolation_steps > self.cfg.min_interpolation_steps:
                old_value = self.interpolation_steps
                self.interpolation_steps = max(self.interpolation_steps - 1, self.cfg.min_interpolation_steps)
                if old_value != self.interpolation_steps:
                    adjustments.append(f"interpolation_steps: {old_value} → {self.interpolation_steps}")
                    self.interpolation_adjustments += 1
        elif avg_utilization > self.cfg.target_utilization * 1.5:
            # 利用率过高，增加插值
            if self.interpolation_steps < self.cfg.max_interpolation_steps:
                old_value = self.interpolation_steps
                self.interpolation_steps = min(self.interpolation_steps + 1, self.cfg.max_interpolation_steps)
                if old_value != self.interpolation_steps:
                    adjustments.append(f"interpolation_steps: {old_value} → {self.interpolation_steps}")
                    self.interpolation_adjustments += 1
        
        # 2. 根据循环频率调整控制周期
        if avg_loop_fps < 0.5 and self.ctrl_period > self.cfg.min_ctrl_period:
            # 循环频率过低，提高控制频率
            old_value = self.ctrl_period
            self.ctrl_period = max(self.ctrl_period * 0.8, self.cfg.min_ctrl_period)
            if old_value != self.ctrl_period:
                adjustments.append(f"ctrl_period: {old_value*1000:.3f}ms → {self.ctrl_period*1000:.3f}ms")
                self.ctrl_period_adjustments += 1
        elif avg_loop_fps > 1.5 and self.ctrl_period < self.cfg.max_ctrl_period:
            # 循环频率过高，降低控制频率
            old_value = self.ctrl_period
            self.ctrl_period = min(self.ctrl_period * 1.2, self.cfg.max_ctrl_period)
            if old_value != self.ctrl_period:
                adjustments.append(f"ctrl_period: {old_value*1000:.3f}ms → {self.ctrl_period*1000:.3f}ms")
                self.ctrl_period_adjustments += 1
        
        # 记录调整
        if adjustments:
            self.adjustment_count += 1
            print(f"\n{'='*60}")
            print(f"[动态调整 #{self.adjustment_count}] 参数优化")
            print(f"{'='*60}")
            print(f"当前性能: 利用率={avg_utilization:.1f}%, 频率={avg_loop_fps:.2f}Hz, 推理={avg_policy_time:.1f}ms")
            print(f"调整内容:")
            for adj in adjustments:
                print(f"  • {adj}")
            print(f"{'='*60}")
            
            logging.info(f"[动态调整 #{self.adjustment_count}] 参数优化")
            logging.info(f"当前性能: 利用率={avg_utilization:.1f}%, 频率={avg_loop_fps:.2f}Hz, 推理={avg_policy_time:.1f}ms")
            logging.info(f"调整内容:")
            for adj in adjustments:
                logging.info(f"  • {adj}")
            logging.info(f"{'='*60}")
        
        return len(adjustments) > 0
    
    def get_stats(self):
        """获取优化统计"""
        avg_utilization = np.mean(self.utilization_history) if self.utilization_history else 0
        avg_loop_fps = np.mean(self.loop_fps_history) if self.loop_fps_history else 0
        avg_policy_time = np.mean(self.policy_time_history) if self.policy_time_history else 0
        
        return {
            'enabled': self.enabled,
            'loop_count': self.loop_count,
            'adjustment_count': self.adjustment_count,
            'interpolation_adjustments': self.interpolation_adjustments,
            'ctrl_period_adjustments': self.ctrl_period_adjustments,
            'avg_utilization': avg_utilization,
            'avg_loop_fps': avg_loop_fps,
            'avg_policy_time': avg_policy_time,
            'current_interpolation_steps': self.interpolation_steps,
            'current_ctrl_period': self.ctrl_period
        }
    
    def print_stats(self):
        """打印优化统计"""
        stats = self.get_stats()
        print(f"\n动态优化统计:")
        print(f"  状态: {'启用' if stats['enabled'] else '禁用'}")
        print(f"  循环次数: {stats['loop_count']}")
        print(f"  调整次数: {stats['adjustment_count']}")
        print(f"  插值调整: {stats['interpolation_adjustments']}次")
        print(f"  周期调整: {stats['ctrl_period_adjustments']}次")
        print(f"\n当前参数:")
        print(f"  interpolation_steps: {stats['current_interpolation_steps']}")
        print(f"  ctrl_period: {stats['current_ctrl_period']*1000:.3f}ms ({1/stats['current_ctrl_period']:.0f}Hz)")
        print(f"\n平均性能:")
        print(f"  利用率: {stats['avg_utilization']:.1f}%")
        print(f"  频率: {stats['avg_loop_fps']:.2f}Hz")
        print(f"  推理: {stats['avg_policy_time']:.1f}ms")


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
    """GR00T机器人推理客户端 - 连接远程GR00T策略推理服务

    该类封装了与NVIDIA GR00T策略推理服务器的通信,负责:
    1. 准备观测数据(图像和机器人状态)
    2. 调用远程推理服务获取动作块
    3. 解析推理结果并转换为机器人可执行的格式

    支持的机器人:
    -------------
    目前仅支持 SO-100 和 SO-101 系列机械臂
    如需支持其他机器人,需根据 modality.json 修改此代码

    数据格式:
    ---------
    输入:
    - 视频数据: 多个摄像头的图像,格式为 RGB,尺寸 224x224
    - 机器人状态: 6个关节的位置(5个机械臂关节 + 1个夹爪)
    - 语言指令: 自然语言任务描述

    输出:
    - 动作块: 包含多个时间步的动作序列
    - 每个动作包含6个关节的目标位置

    使用示例:
    ---------
    >>> client = Gr00tRobotInferenceClient(
    ...     host="10.112.209.136",
    ...     port=5555,
    ...     camera_keys=['wrist', 'front'],
    ...     robot_state_keys=['shoulder_pan.pos', 'shoulder_lift.pos', ...]
    ... )
    >>> action_chunk, timing = await client.get_action_async(observation_dict, "Grab the pen")
    """
    # 设置默认的 camera_keys 和 robot_state_keys
    def __init__(
        self,
        host="localhost",
        port=5555,
        camera_keys=[],
        robot_state_keys=[],
        show_images=False,
    ):
        """初始化GR00T推理客户端

        参数:
        -----
        host : str, 默认="localhost"
            GR00T推理服务器的主机地址
        port : int, 默认=5555
            GR00T推理服务器的端口号
        camera_keys : list
            摄像头键名列表,如 ['wrist', 'front']
        robot_state_keys : list
            机器人状态键名列表,必须包含6个关节
        show_images : bool, 默认=False
            是否显示观测图像(用于调试)
        """
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
        """异步获取动作块

        使用线程池在后台执行同步推理,避免阻塞事件循环。

        参数:
        -----
        observation_dict : dict
            观测数据字典,包含摄像头图像和机器人状态
        lang : str
            自然语言任务描述

        返回:
        -----
        tuple : (action_chunk, timing_info)
            action_chunk: 动作块列表,每个元素是一个动作字典
            timing_info: 计时信息字典,包含各阶段耗时
        """
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            self.executor,
            self._get_action_sync,
            observation_dict,
            lang
        )

    def _get_action_sync(self, observation_dict, lang: str):
        """同步获取动作块（在线程池中执行）

        这是实际执行推理的函数,包含以下步骤:
        1. 图像预处理: BGR→RGB转换,调整尺寸到224x224
        2. 数据打包: 将观测数据打包成GR00T期望的格式
        3. 网络传输+推理: 发送到服务器并等待结果
        4. 数据解析: 将服务器返回的结果转换为LeRobot格式

        参数:
        -----
        observation_dict : dict
            观测数据字典
        lang : str
            自然语言任务描述

        返回:
        -----
        tuple : (action_chunk, timing_info)
            action_chunk: 动作块列表
            timing_info: 包含各阶段耗时的字典
        """
        timing_info = {}
        
        # 步骤1：图像预处理
        preprocess_start = time.time()
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
        timing_info['preprocess'] = (time.time() - preprocess_start) * 1000  # ms

        # 显示图像
        if self.show_images:
            view_img(obs_dict)

        # 步骤2：数据打包
        pack_start = time.time()
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
        timing_info['pack'] = (time.time() - pack_start) * 1000  # ms

        # 步骤3：网络传输 + 服务器推理
        network_start = time.time()
        action_chunk = self.policy.get_action(obs_dict)
        timing_info['network_inference'] = (time.time() - network_start) * 1000  # ms

        # 步骤4：数据解析
        parse_start = time.time()
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
                
                timing_info['parse'] = (time.time() - parse_start) * 1000  # ms
                timing_info['total'] = timing_info['preprocess'] + timing_info['pack'] + timing_info['network_inference'] + timing_info['parse']
                
                return lerobot_actions, timing_info
            else:
                raise ValueError(f"意外的动作数据格式: {action_data}")
        
        # 原有的字典处理逻辑（向后兼容）
        if isinstance(action_chunk, dict):
            lerobot_actions = []
            action_horizon = action_chunk[f"action.{self.modality_keys[0]}"].shape[0]
            for i in range(action_horizon):
                action_dict = self._convert_to_lerobot_action(action_chunk, i)
                lerobot_actions.append(action_dict)
            timing_info['parse'] = 0
            timing_info['total'] = timing_info['preprocess'] + timing_info['pack'] + timing_info['network_inference']
            return lerobot_actions, timing_info
        
        raise TypeError(f"不支持的 action_chunk 类型: {type(action_chunk)}")

    def get_action(self, observation_dict, lang: str):
        """同步获取动作块（向后兼容）

        这是一个同步包装器,用于向后兼容旧的代码。

        参数:
        -----
        observation_dict : dict
            观测数据字典
        lang : str
            自然语言任务描述

        返回:
        -----
        tuple : (action_chunk, timing_info)
        """
        return self._get_action_sync(observation_dict, lang)

    def _convert_to_lerobot_action(
        self, action_chunk: dict[str, np.array], idx: int
    ) -> dict[str, float]:
        """将动作块转换为LeRobot格式的动作字典

        这是一个魔法函数,将动作块转换为 dict[str, float]
        这是因为动作块是 dict[str, np.array]
        我们想要将其转换为 dict[str, float]
        以便可以发送给机器人

        参数:
        -----
        action_chunk : dict[str, np.array]
            动作块字典,键为模态名,值为动作数组
        idx : int
            要提取的动作索引

        返回:
        -----
        dict[str, float] : LeRobot格式的动作字典
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
    """以黄色文本打印信息到控制台

    使用ANSI转义码将文本设置为黄色,用于在控制台中突出显示重要信息。

    参数:
    -----
    text : str
        要以黄色显示的文本内容

    示例:
    -----
    >>> print_yellow("警告: 这是一个重要信息")
    """
    print("\033[93m {}\033[00m".format(text))


@dataclass
class EvalConfig:
    """评估配置类 - 定义机器人策略评估的所有配置参数

    该数据类包含了评估脚本运行所需的所有配置参数,包括机器人配置、
    策略服务器地址、平滑算法参数、动态优化参数等。

    配置参数说明:
    -------------
    - robot: 机器人配置对象,指定使用的机器人类型和参数
    - policy_host/port: GR00T策略推理服务器的地址和端口
    - action_horizon: 动作块长度,每次推理生成的动作数量
    - lang_instruction: 自然语言指令,描述机器人要执行的任务
    - smoothing_method: 动作平滑算法选择
    - interpolation_steps: 动作插值步数
    - enable_adaptive: 是否启用动态参数优化

    使用示例:
    ---------
    >>> cfg = EvalConfig(
    ...     robot=so101_follower_config,
    ...     policy_host="10.112.209.136",
    ...     lang_instruction="Grab markers and place into pen holder."
    ... )
    """
    robot: RobotConfig  # 要使用的机器人配置
    policy_host: str = "localhost"  # gr00t服务器的主机地址
    policy_port: int = 5555  # gr00t服务器的端口
    action_horizon: int = 4  # 从动作块中执行的动作数量
    lang_instruction: str = "Grab pens and place into pen holder."  # 自然语言任务描述
    play_sounds: bool = False  # 是否播放声音提示
    timeout: int = 60  # 超时时间（秒）
    show_images: bool = False  # 是否显示图像预览
    use_sync: bool = False  # 是否使用同步版本（默认使用异步优化版本）
    enable_video_stream: bool = True  # 是否启用视频流服务
    video_stream_port: int = 5000  # 视频流服务器端口

    ctrl_period: float = 0.003  # 控制周期，单位为秒 0.003s=333Hz
    
    # 平滑算法配置 
    smoothing_method: str = "one_euro_outlier"  # 平滑方法: 'ema', 'moving_avg', 'savgol', 'dct', 'kalman', 'savgol_outlier', 'one_euro_outlier', 'kalman_predict'
    smoothing_window_size: int = 10  # 平滑窗口大小
    savgol_window_length: int = 7  # Savitzky-Golay滤波窗口长度（必须为奇数且>=3）
    enable_interpolation: bool = True  # 是否启用动作块内插值
    interpolation_steps: int = 10 # 每个动作之间的插值步数
    
    # DCT平滑配置
    dct_keep_ratio: float = 0.3  # DCT保留低频系数的比例 (0.1-0.9)，越小越平滑
    
    # 卡尔曼滤波配置
    kalman_process_noise: float = 0.05  # 过程噪声Q，越大响应越快但噪声更多
    kalman_measurement_noise: float = 0.05  # 测量噪声R，越大平滑效果越强
    
    # 离群值剔除配置（用于savgol_outlier、one_euro_outlier和kalman_predict方法）
    outlier_threshold: float = 4  # 离群值检测阈值（基于IQR方法），值越小剔除越严格
    
    # One-Euro Filter配置（用于one_euro_outlier方法）
    one_euro_min_cutoff: float = 0.5  # One-Euro Filter最小截止频率（Hz），越小越平滑 0.5
    one_euro_beta: float = 0.1  # One-Euro Filter截止频率斜率系数，越大跟踪越快0.1
    one_euro_d_cutoff: float = 1.0  # One-Euro Filter导数截止频率（Hz）1.0
    
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
    
    # 动态调整配置
    enable_adaptive: bool = True  # 是否启用动态调整
    adaptive_interval: int = 5  # 动态调整间隔（循环次数）
    min_interpolation_steps: int = 2  # 最小插值步数
    max_interpolation_steps: int = 15  # 最大插值步数
    min_ctrl_period: float = 0.0005  # 最小控制周期（2000Hz）
    max_ctrl_period: float = 0.005  # 最大控制周期（200Hz）
    target_utilization: float = 0.8  # 目标动作块利用率（80%）


def rad_speed_limit(target_pos, current_pos, max_delta_pos=0.5):
    """关节速度限制函数 - 限制关节运动的最大速度

    该函数通过限制单次控制周期内的最大关节角度变化,防止机器人运动过快
    导致的不稳定或危险情况。当目标位置与当前位置的差值超过最大允许值时,
    会按比例缩放所有关节的运动量。

    工作原理:
    ---------
    1. 计算目标位置与当前位置的差值 delta_pos
    2. 计算所有关节中最大的运动幅度 max(|delta_pos|)
    3. 如果最大运动幅度超过 max_delta_pos,计算缩放比例
    4. 按缩放比例调整所有关节的运动量

    参数:
    -----
    target_pos : np.ndarray or list
        目标关节位置数组,形状为 (n_joints,)
    current_pos : np.ndarray or list
        当前关节位置数组,形状为 (n_joints,)
    max_delta_pos : float, 默认=0.5
        单次控制周期内允许的最大关节角度变化（弧度）

    返回:
    -----
    np.ndarray : 限制后的目标位置数组

    示例:
    -----
    >>> current = np.array([0.0, 0.0, 0.0])
    >>> target = np.array([1.0, 0.5, 0.3])
    >>> limited = rad_speed_limit(target, current, max_delta_pos=0.5)
    >>> # limited 将被限制在最大变化0.5弧度以内
    """

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
    """异步评估主函数 - 使用异步优化策略评估机器人性能

    这是评估脚本的异步版本,使用了多种优化技术提高控制频率和性能:
    - 观测数据预取: 使用环形缓存区异步预取观测数据
    - 线程池并发: 使用线程池执行阻塞操作
    - 动作平滑: 多种平滑算法减少动作抖动
    - 动作插值: 在动作块内插值使运动更平滑
    - 动态优化: 根据实时性能自动调整参数
    - 视频流服务: 提供Web界面实时查看摄像头画面

    执行流程:
    ---------
    1. 初始化日志系统和配置
    2. 连接并初始化机器人
    3. 初始化GR00T策略推理客户端
    4. 创建线程锁保护机器人串口访问
    5. 启动视频流服务器(可选)
    6. 初始化观测预取器
    7. 初始化平滑器和插值器
    8. 进入主控制循环:
       - 异步获取观测数据
       - 异步调用策略推理获取动作块
       - 应用插值和平滑
       - 发送动作到机器人
       - 统计性能指标
       - 动态调整参数

    性能统计:
    ---------
    - 循环频率: 外层循环的执行频率
    - 动作频率: 实际发送到机器人的指令频率
    - 网络延迟: 策略推理的延迟
    - 缓存命中率: 预取器的缓存命中率
    - 动作块利用率: 动作块执行时间占总时间的比例

    参数:
    -----
    cfg : EvalConfig
        评估配置对象,包含所有运行参数

    示例:
    -----
    >>> cfg = EvalConfig(
    ...     robot=so101_follower_config,
    ...     policy_host="10.112.209.136",
    ...     lang_instruction="Grab markers and place into pen holder."
    ... )
    >>> await eval_async(cfg)
    """
    # 步骤1：初始化日志系统
    init_logging()
    logging.info(pformat(asdict(cfg)))
    
    # 配置日志文件输出
    log_file = "eval_performance.log"
    file_handler = logging.FileHandler(log_file, mode='w', encoding='utf-8')
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
    logging.getLogger().addHandler(file_handler)
    logging.info(f"日志文件: {log_file}")

    # 步骤2：初始化机器人
    robot = make_robot_from_config(cfg.robot)
    robot.connect()

    # 获取摄像头配置
    camera_keys = list(cfg.robot.cameras.keys())
    print("camera_keys: ", camera_keys)

    log_say("Initializing robot", cfg.play_sounds, blocking=True)

    # 获取语言指令和机器人状态键
    language_instruction = cfg.lang_instruction
    robot_state_keys = list(robot._motors_ft.keys())
    print("robot_state_keys: ", robot_state_keys)

    # 步骤3：初始化策略推理客户端
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

    # 步骤4：创建线程锁保护机器人串口访问
    # 由于机器人串口是共享资源,需要使用锁来避免并发访问冲突
    robot_lock = threading.Lock()
    
    # 步骤5：初始化视频流服务器（如果启用）
    # 提供Web界面实时查看摄像头画面
    video_stream_server = None
    if cfg.enable_video_stream:
        video_stream_server = VideoStreamServer(
            robot=robot,
            robot_lock=robot_lock,
            camera_keys=camera_keys,
            port=cfg.video_stream_port
        )
        video_stream_server.start()
    
    # 步骤6：初始化观测预取器
    # 使用环形缓存区异步预取观测数据,提高控制频率
    executor = ThreadPoolExecutor(max_workers=4)
    obs_prefetcher = ObservationPrefetcher(robot, executor, robot_lock)
    await obs_prefetcher.start()
    
    # 步骤7：初始化平滑器和插值器
    action_smoother = ActionSmoother(
        robot_state_keys,
        window_size=cfg.smoothing_window_size,
        method=cfg.smoothing_method,
        dct_keep_ratio=cfg.dct_keep_ratio,
        savgol_window_length=cfg.savgol_window_length,
        kalman_process_noise=cfg.kalman_process_noise,
        kalman_measurement_noise=cfg.kalman_measurement_noise,
        outlier_threshold=cfg.outlier_threshold,
        one_euro_min_cutoff=cfg.one_euro_min_cutoff,
        one_euro_beta=cfg.one_euro_beta,
        one_euro_d_cutoff=cfg.one_euro_d_cutoff
    )
    action_interpolator = ActionInterpolator(
        robot_state_keys,
        interpolation_steps=cfg.interpolation_steps
    )
    
    # 打印平滑配置信息
    print(f"平滑配置:")
    print(f"  方法: {cfg.smoothing_method}")
    print(f"  窗口大小: {cfg.smoothing_window_size}")
    print(f"  插值启用: {cfg.enable_interpolation}")
    print(f"  插值步数: {cfg.interpolation_steps}")
    print(f"  最大角度变化: {cfg.max_delta_pos} rad")
    if cfg.smoothing_method == 'dct':
        print(f"  DCT保留低频比例: {cfg.dct_keep_ratio}")
    if cfg.smoothing_method == 'kalman':
        print(f"  过程噪声Q: {cfg.kalman_process_noise}")
        print(f"  测量噪声R: {cfg.kalman_measurement_noise}")

    # 初始化前一个动作（用于速度限制）
    previous_action = None
    
    # 创建关节平滑参数映射
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
        dct_keep_ratio=cfg.dct_keep_ratio,
        savgol_window_length=cfg.savgol_window_length,
        kalman_process_noise=cfg.kalman_process_noise,
        kalman_measurement_noise=cfg.kalman_measurement_noise,
        outlier_threshold=cfg.outlier_threshold,
        one_euro_min_cutoff=cfg.one_euro_min_cutoff,
        one_euro_beta=cfg.one_euro_beta,
        one_euro_d_cutoff=cfg.one_euro_d_cutoff
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
    if cfg.smoothing_method == 'kalman':
        print(f"  过程噪声Q: {cfg.kalman_process_noise}")
        print(f"  测量噪声R: {cfg.kalman_measurement_noise}")
    
    # 步骤7：初始化动态优化器
    adaptive_optimizer = AdaptiveOptimizer(cfg)
    if adaptive_optimizer.enabled:
        print(f"\n动态优化配置:")
        print(f"  状态: 启用")
        print(f"  调整间隔: {adaptive_optimizer.interval}次循环")
        print(f"  目标利用率: {cfg.target_utilization*100:.0f}%")
        print(f"  interpolation_steps范围: {cfg.min_interpolation_steps}-{cfg.max_interpolation_steps}")
        print(f"  ctrl_period范围: {cfg.min_ctrl_period*1000:.3f}ms-{cfg.max_ctrl_period*1000:.3f}ms")
    else:
        print(f"\n动态优化: 禁用")

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
    
    # 详细性能统计
    timing_stats = {
        'preprocess': [],
        'pack': [],
        'network_inference': [],
        'parse': [],
        'total': []
    }
    # ------------------------------------

    # 步骤4：运行异步评估循环
    try:
        while True:
            loop_start_time = time.time()
            
            # 直接获取最新观测数据（不使用预取器缓存，避免动作回退）
            obs_start_time = time.time()
            loop = asyncio.get_event_loop()
            
            def get_observation_with_lock():
                with robot_lock:
                    return robot.get_observation()
            
            observation_dict = await loop.run_in_executor(
                executor,
                get_observation_with_lock
            )
            obs_time = time.time() - obs_start_time
            async_obs_time_list.append(obs_time)
            
            # 异步获取动作块（包含详细计时）
            policy_start_time = time.time()
            action_chunk, timing_info = await policy.get_action_async(observation_dict, language_instruction)
            policy_time = time.time() - policy_start_time
            async_policy_time_list.append(policy_time)
            
            # 记录详细性能统计
            for key in timing_stats:
                if key in timing_info:
                    timing_stats[key].append(timing_info[key])
                    if len(timing_stats[key]) > max_latency_history:
                        timing_stats[key].pop(0)
            
            # 记录总网络延迟
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
                
                # 计算详细性能统计
                avg_preprocess = np.mean(timing_stats['preprocess']) if timing_stats['preprocess'] else 0
                avg_pack = np.mean(timing_stats['pack']) if timing_stats['pack'] else 0
                avg_network = np.mean(timing_stats['network_inference']) if timing_stats['network_inference'] else 0
                avg_parse = np.mean(timing_stats['parse']) if timing_stats['parse'] else 0
                avg_total = np.mean(timing_stats['total']) if timing_stats['total'] else 0
                
                print(f"\n{'='*60}")
                print(f"[Loop {loop_count}] 异步性能统计")
                print(f"{'='*60}")
                print(f"周期频率: {loop_fps:.2f} Hz | 实际指令频率: {total_action_fps:.2f} Hz")
                print(f"异步获取观测耗时: {avg_obs_time:.2f} ms (平均)")
                print(f"异步策略推理耗时: {avg_policy_time:.2f} ms (平均)")
                print(f"总循环耗时: {dt*1000:.2f} ms")
                
                logging.info(f"[Loop {loop_count}] 异步性能统计")
                logging.info(f"周期频率: {loop_fps:.2f} Hz | 实际指令频率: {total_action_fps:.2f} Hz")
                logging.info(f"异步获取观测耗时: {avg_obs_time:.2f} ms (平均)")
                logging.info(f"异步策略推理耗时: {avg_policy_time:.2f} ms (平均)")
                logging.info(f"总循环耗时: {dt*1000:.2f} ms")
                
                # 获取缓存统计
                cache_stats = obs_prefetcher.get_cache_stats()
                print(f"\n环形缓存区统计:")
                print(f"  命中次数: {cache_stats['hit_count']}")
                print(f"  未命中次数: {cache_stats['miss_count']}")
                print(f"  命中率: {cache_stats['hit_rate']*100:.2f}%")
                print(f"  缓存区大小: {obs_prefetcher.buffer_size}")
                print(f"  当前缓存数: {cache_stats['buffer_count']}")
                print(f"  缓存利用率: {cache_stats['buffer_utilization']*100:.2f}%")
                print(f"  总预取次数: {cache_stats['prefetch_count']}")
                
                logging.info(f"环形缓存区统计:")
                logging.info(f"  命中次数: {cache_stats['hit_count']}")
                logging.info(f"  未命中次数: {cache_stats['miss_count']}")
                logging.info(f"  命中率: {cache_stats['hit_rate']*100:.2f}%")
                logging.info(f"  缓存区大小: {obs_prefetcher.buffer_size}")
                logging.info(f"  当前缓存数: {cache_stats['buffer_count']}")
                logging.info(f"  缓存利用率: {cache_stats['buffer_utilization']*100:.2f}%")
                logging.info(f"  总预取次数: {cache_stats['prefetch_count']}")
                
                print(f"\n动作块利用统计:")
                print(f"  动作块长度: {cfg.action_horizon} | 插值后: {actual_action_count}")
                print(f"  动作执行时间: {action_execution_time*1000:.2f} ms")
                print(f"  空闲时间: {idle_time*1000:.2f} ms")
                print(f"  动作块利用率: {action_chunk_utilization:.1f}%")
                print(f"\n策略推理详细耗时 (最近{len(timing_stats['total'])}次):")
                print(f"  ├─ 图像预处理: {avg_preprocess:.2f} ms (BGR→RGB, resize)")
                print(f"  ├─ 数据打包: {avg_pack:.2f} ms (格式转换)")
                print(f"  ├─ 网络+推理: {avg_network:.2f} ms (传输+模型推理)")
                print(f"  ├─ 数据解析: {avg_parse:.2f} ms (结果处理)")
                print(f"  └─ 总计: {avg_total:.2f} ms")
                print(f"\n网络延迟统计 (最近{len(network_latency_list)}次):")
                print(f"  平均延迟: {avg_latency:.2f} ms")
                print(f"  最小延迟: {min_latency:.2f} ms")
                print(f"  最大延迟: {max_latency:.2f} ms")
                print(f"{'='*60}")
                
                logging.info(f"动作块利用统计:")
                logging.info(f"  动作块长度: {cfg.action_horizon} | 插值后: {actual_action_count}")
                logging.info(f"  动作执行时间: {action_execution_time*1000:.2f} ms")
                logging.info(f"  空闲时间: {idle_time*1000:.2f} ms")
                logging.info(f"  动作块利用率: {action_chunk_utilization:.1f}%")
                logging.info(f"策略推理详细耗时 (最近{len(timing_stats['total'])}次):")
                logging.info(f"  ├─ 图像预处理: {avg_preprocess:.2f} ms (BGR→RGB, resize)")
                logging.info(f"  ├─ 数据打包: {avg_pack:.2f} ms (格式转换)")
                logging.info(f"  ├─ 网络+推理: {avg_network:.2f} ms (传输+模型推理)")
                logging.info(f"  ├─ 数据解析: {avg_parse:.2f} ms (结果处理)")
                logging.info(f"  └─ 总计: {avg_total:.2f} ms")
                logging.info(f"网络延迟统计 (最近{len(network_latency_list)}次):")
                logging.info(f"  平均延迟: {avg_latency:.2f} ms")
                logging.info(f"  最小延迟: {min_latency:.2f} ms")
                logging.info(f"  最大延迟: {max_latency:.2f} ms")
                logging.info(f"{'='*60}")
                last_loop_time = current_time
            
            # 动态优化：根据性能调整参数
            if adaptive_optimizer.enabled:
                adjusted = adaptive_optimizer.adjust(
                    action_chunk_utilization=action_chunk_utilization,
                    loop_fps=loop_fps,
                    policy_time=policy_time
                )
                
                if adjusted:
                    # 更新插值器配置
                    action_interpolator.interpolation_steps = adaptive_optimizer.interpolation_steps
                    # 更新控制周期
                    cfg.ctrl_period = adaptive_optimizer.ctrl_period
                    # 注意：action_horizon 不需要更新，因为它只在获取动作块时使用
            # ------------------------------------
    finally:
        # 清理资源
        if video_stream_server:
            video_stream_server.stop()
        await obs_prefetcher.close()
        executor.shutdown(wait=True)
        
        # 打印动态优化统计
        if adaptive_optimizer.enabled:
            adaptive_optimizer.print_stats()


@draccus.wrap()
def eval(cfg: EvalConfig):
    """主入口函数 - 根据配置选择异步或同步版本

    这是评估脚本的入口函数,根据配置参数选择运行异步版本还是同步版本。
    默认使用异步版本,可以获得更好的性能。

    参数:
    -----
    cfg : EvalConfig
        评估配置对象,包含所有运行参数

    返回:
    -----
    根据配置返回 eval_async 或 eval_sync 的执行结果

    使用示例:
    ---------
    >>> cfg = EvalConfig(
    ...     robot=so101_follower_config,
    ...     policy_host="10.112.209.136",
    ...     use_sync=False  # 使用异步版本
    ... )
    >>> eval(cfg)
    """
    if cfg.use_sync:
        print("使用同步版本 (use_sync=True)")
        return eval_sync(cfg)
    else:
        print("使用异步优化版本 (use_sync=False)")
        return asyncio.run(eval_async(cfg))


@draccus.wrap()
def eval_sync(cfg: EvalConfig):
    """同步评估主函数 - 使用同步策略评估机器人性能

    这是评估脚本的同步版本,用于性能对比测试。
    与异步版本相比,同步版本没有使用观测预取和线程池优化,
    因此性能较低,但代码更简单,便于理解和调试。

    执行流程:
    ---------
    1. 初始化日志系统和配置
    2. 连接并初始化机器人
    3. 初始化GR00T策略推理客户端
    4. 初始化平滑器和插值器
    5. 进入主控制循环:
       - 同步获取观测数据
       - 同步调用策略推理获取动作块
       - 应用插值和平滑
       - 发送动作到机器人
       - 统计性能指标

    性能特点:
    ---------
    - 控制频率较低,因为每次循环都要等待观测和推理完成
    - 代码结构简单,易于理解和调试
    - 适合用于性能对比和问题排查

    参数:
    -----
    cfg : EvalConfig
        评估配置对象,包含所有运行参数

    示例:
    -----
    >>> cfg = EvalConfig(
    ...     robot=so101_follower_config,
    ...     policy_host="10.112.209.136",
    ...     use_sync=True  # 使用同步版本
    ... )
    >>> eval_sync(cfg)
    """
    init_logging()
    logging.info(pformat(asdict(cfg)))
    
    # 配置日志文件输出
    log_file = "eval_performance_sync.log"
    file_handler = logging.FileHandler(log_file, mode='w', encoding='utf-8')
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
    logging.getLogger().addHandler(file_handler)
    logging.info(f"日志文件: {log_file}")

    # 步骤1：初始化机器人
    robot = make_robot_from_config(cfg.robot)
    robot.connect()

    # 获取摄像头配置
    camera_keys = list(cfg.robot.cameras.keys())
    print("camera_keys: ", camera_keys)

    log_say("Initializing robot", cfg.play_sounds, blocking=True)

    # 获取语言指令和机器人状态键
    language_instruction = cfg.lang_instruction
    robot_state_keys = list(robot._motors_ft.keys())
    print("robot_state_keys: ", robot_state_keys)

    # 步骤2：初始化策略推理客户端
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

    # 初始化前一个动作（用于速度限制）
    previous_action = None
    
    # 创建关节平滑参数映射
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
        dct_keep_ratio=cfg.dct_keep_ratio,
        savgol_window_length=cfg.savgol_window_length,
        kalman_process_noise=cfg.kalman_process_noise,
        kalman_measurement_noise=cfg.kalman_measurement_noise,
        outlier_threshold=cfg.outlier_threshold,
        one_euro_min_cutoff=cfg.one_euro_min_cutoff,
        one_euro_beta=cfg.one_euro_beta,
        one_euro_d_cutoff=cfg.one_euro_d_cutoff
    )
    action_interpolator = ActionInterpolator(
        robot_state_keys,
        interpolation_steps=cfg.interpolation_steps
    )
    
    # 打印平滑配置信息
    print(f"平滑配置:")
    print(f"  方法: {cfg.smoothing_method}")
    print(f"  窗口大小: {cfg.smoothing_window_size}")
    print(f"  插值启用: {cfg.enable_interpolation}")
    print(f"  插值步数: {cfg.interpolation_steps}")
    print(f"  最大角度变化: {cfg.max_delta_pos} rad")

    # --- 频率和延迟统计变量初始化 ---
    last_loop_time = time.time()  # 上一次循环的时间戳
    last_action_time = time.time()  # 上一次动作执行的时间戳
    loop_count = 0  # 循环计数器
    action_count = 0  # 动作计数器
    print_interval = 1  # 外层循环打印间隔
    action_print_interval = 10  # 内层循环打印间隔（每10个动作打印一次）
    
    # 网络延迟统计
    network_latency_list = []
    max_latency_history = 100  # 保存最近100次延迟记录
    # ------------------------------------

    # 步骤3：运行评估主循环
    while True:
        loop_start_time = time.time()
        
        # 获取实时观测数据
        # 同步调用,会阻塞等待观测数据返回
        observation_dict = robot.get_observation()
        obs_time = time.time() - loop_start_time
        
        # 获取动作块（这部分包含网络延迟）
        # 同步调用策略推理,会阻塞等待动作块返回
        policy_start_time = time.time()
        action_chunk, timing_info = policy.get_action(observation_dict, language_instruction)
        policy_time = time.time() - policy_start_time
        
        # 记录网络延迟
        network_latency_list.append(policy_time)
        if len(network_latency_list) > max_latency_history:
            network_latency_list.pop(0)

        # 应用插值（如果启用）
        # 在动作块内进行线性插值,使运动更平滑
        if cfg.enable_interpolation:
            action_chunk = action_interpolator.interpolate_action_chunk(action_chunk)

        # 执行动作序列
        # 遍历动作块中的每个动作,依次发送到机器人
        for i in range(len(action_chunk)):
            action_dict = action_chunk[i]
            
            # 应用平滑算法
            # 使用配置的平滑算法对动作进行平滑处理,减少抖动
            smoothed_action = action_smoother.smooth(action_dict, joint_alpha_map)
            
            # 应用速度限制
            # 限制单次控制周期内的最大关节角度变化,防止运动过快
            if previous_action is not None:
                for key in smoothed_action:
                    smoothed_action[key] = rad_speed_limit(
                        target_pos=smoothed_action[key],
                        current_pos=previous_action[key],
                        max_delta_pos=cfg.max_delta_pos
                    )
            
            # 保存当前动作用于下一次速度限制
            previous_action = smoothed_action.copy()
            
            # 发送动作到机器人
            robot.send_action(smoothed_action)
            
            # 等待控制周期
            # 控制动作执行频率,确保稳定的控制周期
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
            
            # 打印性能统计信息
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
            
            logging.info(f"[Loop {loop_count}] 性能统计 (同步版本)")
            logging.info(f"周期频率: {loop_fps:.2f} Hz | 实际指令频率: {total_action_fps:.2f} Hz")
            logging.info(f"获取观测耗时: {obs_time*1000:.2f} ms")
            logging.info(f"策略推理耗时: {policy_time*1000:.2f} ms")
            logging.info(f"总循环耗时: {dt*1000:.2f} ms")
            logging.info(f"动作块利用统计:")
            logging.info(f"  动作块长度: {cfg.action_horizon} | 插值后: {actual_action_count}")
            logging.info(f"  动作执行时间: {action_execution_time*1000:.2f} ms")
            logging.info(f"  空闲时间: {idle_time*1000:.2f} ms")
            logging.info(f"  动作块利用率: {action_chunk_utilization:.1f}%")
            logging.info(f"网络延迟统计 (最近{len(network_latency_list)}次):")
            logging.info(f"  平均延迟: {avg_latency:.2f} ms")
            logging.info(f"  最小延迟: {min_latency:.2f} ms")
            logging.info(f"  最大延迟: {max_latency:.2f} ms")
            logging.info(f"{'='*60}")
            last_loop_time = current_time
        # ------------------------------------


if __name__ == "__main__":
    """程序入口点

    当直接运行此脚本时,调用 eval() 函数开始评估。
    使用 draccus 库进行命令行参数解析。

    使用示例:
    ---------
    python eval_lerobot.py \\
        --robot.type=so101_follower \\
        --robot.port=/dev/ttyACM0 \\
        --robot.id=lil_guy \\
        --robot.cameras="{ wrist: {type: opencv, index_or_path: 9}, front: {type: opencv, index_or_path: 15}}" \\
        --policy_host=10.112.209.136 \\
        --lang_instruction="Grab markers and place into pen holder."
    """
    eval()
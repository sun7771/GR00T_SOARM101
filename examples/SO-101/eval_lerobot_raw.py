import logging
import time
from dataclasses import asdict, dataclass
from pprint import pformat
import cv2
import draccus
import matplotlib.pyplot as plt
import numpy as np
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

        #获取机械臂动作
    def get_action(self, observation_dict, lang: str):
        # 首先添加图像
        obs_dict = {}
        for key in self.camera_keys:
            img = observation_dict[key]
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
        
        # 调试: 打印 action_chunk 的类型和内容
        print(f"action_chunk type: {type(action_chunk)}")
        print(action_chunk)
        
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



@dataclass
class EvalConfig:
    robot: RobotConfig  # 要使用的机器人
    policy_host: str = "localhost"  # gr00t服务器的主机地址
    policy_port: int = 5555  # gr00t服务器的端口
    # todo：：调整动作块的长度
    action_horizon: int = 8  # 从动作块中执行的动作数量
    lang_instruction: str = "Grab pens and place into pen holder."
    play_sounds: bool = False  # 是否播放声音
    timeout: int = 60  # 超时时间（秒）
    show_images: bool = False  # 是否显示图像

    ctrl_period: float = 0.001  # 控制周期，单位为秒 0.001s=1000Hz
    
    
    shoulder_pan_alpha: float = 0.15    # 肩部转动 - 较大的关节，需要更多平滑
    shoulder_lift_alpha: float = 0.2  # 肩部抬升 - 承重关节，平滑一些
    shoulder_lift_alpha: float = 0.2  # 肩部抬升 - 承重关节，平滑一些
    elbow_flex_alpha: float = 0.15     # 肘部弯曲 - 中等平滑
    wrist_flex_alpha: float = 0.5      # 腕部弯曲 - 精细动作，少一些平滑
    wrist_roll_alpha: float = 0.5     # 腕部旋转 - 快速响应
    gripper_alpha: float = 0.3         # 夹爪 - 需要更多平滑避免抖动


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



@draccus.wrap()
def eval(cfg: EvalConfig):
    init_logging()
    logging.info(pformat(asdict(cfg)))
    
    # 配置日志文件输出
    log_file = "log/eval_performance_raw.log"
    file_handler = logging.FileHandler(log_file, mode='w', encoding='utf-8')
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
    logging.getLogger().addHandler(file_handler)
    logging.info(f"日志文件: {log_file}")

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

        # 执行动作序列
        for i in range(cfg.action_horizon):
            action_dict = action_chunk[i]
            
            if previous_action is not None:
                smoothed_action = {}
                for key in action_dict:
                    if key in joint_alpha_map:
                        alpha = joint_alpha_map[key]
                        smoothed_action[key] = (alpha * action_dict[key] + 
                                              (1 - alpha) * previous_action[key])
                        smoothed_action[key] = rad_speed_limit(
                            target_pos=smoothed_action[key],
                            current_pos=previous_action[key],
                            max_delta_pos=0.5
                        )
                    else:
                        smoothed_action[key] = action_dict[key]
                previous_action = smoothed_action.copy()
            else:
                smoothed_action = action_dict.copy()
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
                total_action_fps = (print_interval * cfg.action_horizon) / dt
                
                # 计算动作块执行时间和利用率
                action_execution_time = cfg.action_horizon * cfg.ctrl_period
                idle_time = dt - action_execution_time
                action_chunk_utilization = (action_execution_time / dt) * 100 if dt > 0 else 0
                
                # 计算网络延迟统计
                avg_latency = np.mean(network_latency_list) * 1000  # 毫秒
                min_latency = np.min(network_latency_list) * 1000
                max_latency = np.max(network_latency_list) * 1000
                
                print(f"\n{'='*60}")
                print(f"[Loop {loop_count}] 性能统计 (Raw版本)")
                print(f"{'='*60}")
                print(f"周期频率: {loop_fps:.2f} Hz | 实际指令频率: {total_action_fps:.2f} Hz")
                print(f"获取观测耗时: {obs_time*1000:.2f} ms")
                print(f"策略推理耗时: {policy_time*1000:.2f} ms")
                print(f"总循环耗时: {dt*1000:.2f} ms")
                print(f"\n动作块利用统计:")
                print(f"  动作块长度: {cfg.action_horizon}")
                print(f"  动作执行时间: {action_execution_time*1000:.2f} ms")
                print(f"  空闲时间: {idle_time*1000:.2f} ms")
                print(f"  动作块利用率: {action_chunk_utilization:.1f}%")
                print(f"\n网络延迟统计 (最近{len(network_latency_list)}次):")
                print(f"  平均延迟: {avg_latency:.2f} ms")
                print(f"  最小延迟: {min_latency:.2f} ms")
                print(f"  最大延迟: {max_latency:.2f} ms")
                print(f"{'='*60}")
                
                logging.info(f"[Loop {loop_count}] 性能统计 (Raw版本)")
                logging.info(f"周期频率: {loop_fps:.2f} Hz | 实际指令频率: {total_action_fps:.2f} Hz")
                logging.info(f"获取观测耗时: {obs_time*1000:.2f} ms")
                logging.info(f"策略推理耗时: {policy_time*1000:.2f} ms")
                logging.info(f"总循环耗时: {dt*1000:.2f} ms")
                logging.info(f"动作块利用统计:")
                logging.info(f"  动作块长度: {cfg.action_horizon}")
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
    eval()
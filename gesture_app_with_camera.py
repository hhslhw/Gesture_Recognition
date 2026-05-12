import tkinter as tk
import customtkinter as ctk
import cv2
import mediapipe as mp
import numpy as np
import torch
import threading
import queue
import time
import os
from PIL import Image, ImageTk
from scipy import interpolate as scipy_interpolate
from st_gcn.net.st_gcn_hand_v6 import Model

# 设置主题
ctk.set_appearance_mode("dark")
ctk.set_default_color_theme("blue")

# ==================== 多流配置 ====================
STREAMS_CONFIG = {
    "joint": {
        "enable": True,
        "model_path": "models/best_new.pth",
        "weight": 1.0
    },
    "bone": {
        "enable": True,
        "model_path": "models/best_bone.pth",
        "weight": 0.90
    },
    "motion": {
        "enable": True,
        "model_path": "models/best_motion.pth",
        "weight": 0.82
    }
}

ACTIONS = ["单手放大", "单手缩小", "双指向上滑动", "双指向下滑动",
           "双指向右滑动", "双指向左滑动", "双指拉近", "双指放大", "拉手靠近"]
NUM_CLASSES = 9
WINDOW_SIZE = 35
CONFIDENCE_THRESHOLD = 0.5
INFERENCE_STRIDE = 5  # 每 N 帧推理一次，降低 CPU/GPU 负载

# ==================== 状态机配置 ====================
# 状态枚举
STATE_IDLE     = 0   # 静止监测
STATE_PREPARE  = 1   # 预备倒计时
STATE_DETECT   = 2   # 正在检测
STATE_COOLDOWN = 3   # 冷却期

MOVEMENT_THRESHOLD = 0.015  # 帧间位移判定阈值（归一化坐标）
PREPARE_TIME       = 0.5    # 预备倒计时时长（秒）
ACTION_COOLDOWN    = 2.0    # 识别后冷却时间（秒）


def interpolate_keypoints_to_window(valid_frames_list, target_frames):
    """
    将有效帧列表线性插值到目标帧数（与训练管线 unified_keypoint_extraction.py 一致）

    Args:
        valid_frames_list: list of (V, C) arrays，仅包含检测成功的帧
        target_frames: 目标帧数（WINDOW_SIZE）

    Returns:
        (target_frames, V, C) float32 array，或 None（若输入为空）
    """
    n = len(valid_frames_list)
    if n == 0:
        return None
    if n == 1:
        return np.repeat(valid_frames_list[0][np.newaxis], target_frames, axis=0).astype(np.float32)

    V, C = valid_frames_list[0].shape
    src_indices = np.arange(n)
    dst_indices = np.linspace(0, n - 1, target_frames)
    result = np.zeros((target_frames, V, C), dtype=np.float32)

    for v in range(V):
        for c in range(C):
            values = np.array([kp[v, c] for kp in valid_frames_list])
            f = scipy_interpolate.interp1d(src_indices, values, kind='linear')
            result[:, v, c] = f(dst_indices)

    return result


def vote_hand_label(hand_labels):
    """
    对手部标签列表进行投票，返回主要手部标签（'Left' 或 'Right'）
    与训练管线 unified_keypoint_extraction.py 的投票机制一致

    Args:
        hand_labels: list of str or None，每帧的手部标签

    Returns:
        'Left' 或 'Right'
    """
    left_count = hand_labels.count('Left')
    right_count = hand_labels.count('Right')
    return 'Left' if left_count > right_count else 'Right'


def load_stream_models(device):
    """加载所有开启的流模型，返回 {stream_name: model} 字典"""
    active_models = {}
    for stream_name, config in STREAMS_CONFIG.items():
        if not config["enable"]:
            continue
        model_path = config["model_path"]
        if not os.path.exists(model_path):
            print(f"⚠️ [{stream_name}] 模型文件不存在: {model_path}，跳过该流")
            continue
        model = Model(
            channel=3, num_class=NUM_CLASSES, window_size=WINDOW_SIZE,
            multiscale=True, use_attention=True, use_data_bn=True
        ).to(device)
        try:
            checkpoint = torch.load(model_path, map_location=device, weights_only=True)
            model.load_state_dict(checkpoint, strict=True)
            model.eval()
            active_models[stream_name] = model
            print(f"✓ [{stream_name}] 流模型加载成功 (权重: {config['weight']})")
        except Exception as e:
            print(f"❌ [{stream_name}] 模型加载失败: {e}")
    return active_models


def infer_multistream(active_models, buffer_joint, device):
    """
    多流 Late Fusion 推理
    
    Args:
        active_models: {stream_name: model} 字典
        buffer_joint: (T, V, C) numpy array，关节流的滑动窗口缓冲区
        device: torch device
    
    Returns:
        (action_str, confidence_float) 或 None（若无模型）
    """
    if not active_models:
        return None

    # buffer_joint shape: (T, V, C) -> transpose to (C, T, V) -> unsqueeze to (1, C, T, V)
    joint_data = buffer_joint.transpose(2, 0, 1)  # (C, T, V)

    # =========================================================================
    # 核心归一化：与 HandFeeder v4 完全一致
    # 用第 0 帧第 0 号关键点（手腕）的坐标作为原点，对整个序列做平移归一化
    # =========================================================================
    origin = joint_data[:, 0, 0]  # (C,) — 第0帧手腕坐标
    joint_data = joint_data - origin[:, np.newaxis, np.newaxis]  # 广播到 (C, T, V)

    joint_tensor = torch.from_numpy(joint_data).float().unsqueeze(0).to(device)  # (1, C, T, V)

    # 骨骼流：当前节点 - 父节点（与 generate_stream_data.py 完全一致）
    # parents[i] 表示第 i 个关键点的父节点索引
    PARENTS = [
        0,           # 0: WRIST（根节点，父节点算作自己，骨骼向量为0）
        0, 1, 2, 3,  # 1-4: 大拇指
        0, 5, 6, 7,  # 5-8: 食指
        0, 9, 10, 11,  # 9-12: 中指
        0, 13, 14, 15, # 13-16: 无名指
        0, 17, 18, 19  # 17-20: 小指
    ]
    bone_data = np.zeros_like(joint_data)
    for v in range(21):
        bone_data[:, :, v] = joint_data[:, :, v] - joint_data[:, :, PARENTS[v]]
    bone_tensor = torch.from_numpy(bone_data).float().unsqueeze(0).to(device)

    # 运动流：帧间差分（motion = frame[t] - frame[t-1]）
    motion_data = np.zeros_like(joint_data)
    motion_data[:, 1:, :] = joint_data[:, 1:, :] - joint_data[:, :-1, :]
    motion_tensor = torch.from_numpy(motion_data).float().unsqueeze(0).to(device)

    stream_tensors = {
        "joint": joint_tensor,
        "bone": bone_tensor,
        "motion": motion_tensor
    }

    fused_scores = None
    with torch.no_grad():
        for stream_name, model in active_models.items():
            if stream_name not in stream_tensors:
                continue
            outputs = model(stream_tensors[stream_name])
            probs = torch.nn.functional.softmax(outputs, dim=1)
            weighted_probs = probs * STREAMS_CONFIG[stream_name]["weight"]
            if fused_scores is None:
                fused_scores = weighted_probs
            else:
                fused_scores += weighted_probs

    if fused_scores is None:
        return None

    # 除以权重总和，将融合分数归一化到 [0, 1] 范围
    total_weight = sum(
        STREAMS_CONFIG[s]["weight"]
        for s in active_models.keys()
        if s in STREAMS_CONFIG
    )
    if total_weight > 0:
        fused_scores = fused_scores / total_weight

    conf, cls = torch.max(fused_scores, 1)
    return ACTIONS[cls.item()], conf.item()


class OfflineProcessorThread(threading.Thread):
    """离线视频处理线程"""
    def __init__(self, path, is_folder, save_data, export_result, callback):
        super().__init__()
        self.path = path
        self.is_folder = is_folder
        self.save_data = save_data
        self.export_result = export_result
        self.callback = callback
        self.running = False
        
        # 初始化MediaPipe
        self.mp_hands = mp.solutions.hands.Hands(
            static_image_mode=False, max_num_hands=1,
            min_detection_confidence=0.5, min_tracking_confidence=0.5
        )
        self.mp_draw = mp.solutions.drawing_utils
        
        # 加载多流模型
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.active_models = load_stream_models(self.device)
        self.model_loaded = len(self.active_models) > 0
    
    def run(self):
        self.running = True
        
        try:
            if self.is_folder:
                # 处理图片文件夹
                self.process_image_folder()
            else:
                # 处理视频文件
                self.process_video_file()
                
        except Exception as e:
            self.callback({"error": str(e)})
            return
    
    def process_video_file(self):
        """处理视频文件"""
        cap = cv2.VideoCapture(self.path)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        buffer = np.zeros((WINDOW_SIZE, 21, 3))
        frame_count = 0
        infer_count = 0  # 用于控制推理步长
        results = []
        keypoint_data = []
        
        while self.running:
            ret, frame = cap.read()
            if not ret:
                break
                
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            hand_results = self.mp_hands.process(rgb_frame)
            
            if hand_results.multi_hand_landmarks:
                # 提取关键点：像素坐标（与训练数据一致）
                h, w = frame.shape[:2]
                norm_landmarks = hand_results.multi_hand_landmarks[0].landmark
                current_kpts = np.array([[lm.x * w, lm.y * h, lm.z] for lm in norm_landmarks])
                
                # 判断左右手并进行镜像反转（以手腕为中心，与训练数据一致）
                if hand_results.multi_handedness:
                    hand_label = hand_results.multi_handedness[0].classification[0].label
                    if hand_label == 'Left':
                        wrist_x = current_kpts[0, 0]
                        current_kpts[:, 0] = 2 * wrist_x - current_kpts[:, 0]
                
                buffer = np.roll(buffer, shift=-1, axis=0)
                buffer[-1] = current_kpts
                frame_count += 1
                infer_count += 1
                
                # 每 INFERENCE_STRIDE 帧推理一次
                if self.model_loaded and frame_count >= WINDOW_SIZE and infer_count >= INFERENCE_STRIDE:
                    infer_count = 0
                    result = infer_multistream(self.active_models, buffer, self.device)
                    if result is not None:
                        action, confidence = result
                        if confidence > CONFIDENCE_THRESHOLD:
                            results.append({"frame": frame_count, "action": action, "confidence": confidence})
                        
                if self.save_data:
                    keypoint_data.append(current_kpts)
            
            # 更新进度
            current_frame = int(cap.get(cv2.CAP_PROP_POS_FRAMES))
            progress = current_frame / total_frames
            self.callback({"progress": progress})
            
        cap.release()
        
        # 保存结果
        if self.save_data and keypoint_data:
            save_path = os.path.join(os.path.dirname(self.path), "keypoint_data.npy")
            np.save(save_path, np.array(keypoint_data))
        
        if self.export_result and results:
            export_path = os.path.join(os.path.dirname(self.path), "detection_results.txt")
            with open(export_path, 'w', encoding='utf-8') as f:
                for result in results:
                    f.write(f"帧 {result['frame']}: {result['action']} ({result['confidence']:.2f})\n")
        
        self.callback({"completed": True, "results": results})
    
    def process_image_folder(self):
        """处理图片文件夹"""
        image_files = [f for f in os.listdir(self.path) if f.endswith(('.jpg', '.jpeg', '.png', '.bmp'))]
        image_files.sort()
        total_images = len(image_files)
        
        buffer = np.zeros((WINDOW_SIZE, 21, 3))
        frame_count = 0
        infer_count = 0
        results = []
        keypoint_data = []
        
        for i, image_file in enumerate(image_files):
            if not self.running:
                break
                
            image_path = os.path.join(self.path, image_file)
            frame = cv2.imread(image_path)
            
            if frame is None:
                continue
                
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            hand_results = self.mp_hands.process(rgb_frame)
            
            if hand_results.multi_hand_landmarks:
                # 提取关键点：像素坐标（与训练数据一致）
                h, w = frame.shape[:2]
                norm_landmarks = hand_results.multi_hand_landmarks[0].landmark
                current_kpts = np.array([[lm.x * w, lm.y * h, lm.z] for lm in norm_landmarks])
                
                # 判断左右手并进行镜像反转（以手腕为中心，与训练数据一致）
                if hand_results.multi_handedness:
                    hand_label = hand_results.multi_handedness[0].classification[0].label
                    if hand_label == 'Left':
                        wrist_x = current_kpts[0, 0]
                        current_kpts[:, 0] = 2 * wrist_x - current_kpts[:, 0]

                buffer = np.roll(buffer, shift=-1, axis=0)
                buffer[-1] = current_kpts
                frame_count += 1
                infer_count += 1
                
                # 每 INFERENCE_STRIDE 帧推理一次
                if self.model_loaded and frame_count >= WINDOW_SIZE and infer_count >= INFERENCE_STRIDE:
                    infer_count = 0
                    result = infer_multistream(self.active_models, buffer, self.device)
                    if result is not None:
                        action, confidence = result
                        if confidence > CONFIDENCE_THRESHOLD:
                            results.append({"frame": frame_count, "action": action, "confidence": confidence})
                        
                if self.save_data:
                    keypoint_data.append(current_kpts)
            
            # 更新进度
            progress = (i + 1) / total_images
            self.callback({"progress": progress})
        
        # 保存结果
        if self.save_data and keypoint_data:
            save_path = os.path.join(self.path, "keypoint_data.npy")
            np.save(save_path, np.array(keypoint_data))
        
        if self.export_result and results:
            export_path = os.path.join(self.path, "detection_results.txt")
            with open(export_path, 'w', encoding='utf-8') as f:
                for result in results:
                    f.write(f"帧 {result['frame']}: {result['action']} ({result['confidence']:.2f})\n")
        
        self.callback({"completed": True, "results": results})
    
    def stop(self):
        self.running = False


class CameraThread(threading.Thread):
    """摄像头线程类（含4状态状态机：静止→预备→检测→冷却）"""
    def __init__(self, frame_queue, result_queue, state_queue):
        super().__init__()
        self.frame_queue = frame_queue
        self.result_queue = result_queue
        self.state_queue = state_queue   # 向 UI 推送状态信息
        self.running = False
        self.cap = None

        # 初始化MediaPipe
        self.mp_hands = mp.solutions.hands.Hands(
            static_image_mode=False, max_num_hands=1,
            min_detection_confidence=0.5, min_tracking_confidence=0.5
        )
        self.mp_draw = mp.solutions.drawing_utils

        # 数据缓冲
        self.buffer = np.zeros((WINDOW_SIZE, 21, 3))
        self.frame_count = 0
        # 原始有效帧列表（用于插值补帧）
        self.raw_frames = []        # list of (21, 3) arrays，仅包含检测成功的帧
        self.raw_hand_labels = []   # 对应每帧的手部标签，用于投票

        # ===== 状态机变量 =====
        self.current_state = STATE_IDLE
        self.state_start_time = time.time()
        self.last_move_time = time.time()
        self.prev_hand_norm = None   # 上一帧归一化坐标，用于运动检测

        # 加载多流模型
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.active_models = load_stream_models(self.device)
        self.model_loaded = len(self.active_models) > 0

    def _push_state(self, state_id, extra=None):
        """向 UI 推送当前状态（非阻塞）"""
        try:
            self.state_queue.put_nowait({"state": state_id, "extra": extra})
        except Exception:
            pass

    def run(self):
        self.running = True
        self.cap = cv2.VideoCapture(0)

        # ===== 性能统计变量 =====
        mp_frame_count = 0
        mp_total_time = 0.0
        infer_latencies = []

        while self.running:
            ret, frame = self.cap.read()
            if not ret:
                break

            frame = cv2.flip(frame, 1)
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            # MediaPipe 检测（计时）
            t_mp_start = time.perf_counter()
            results = self.mp_hands.process(rgb_frame)
            t_mp_end = time.perf_counter()
            mp_total_time += (t_mp_end - t_mp_start)
            mp_frame_count += 1

            if mp_frame_count % 300 == 0 and mp_total_time > 0:
                mp_fps = mp_frame_count / mp_total_time
                print(f"[性能] MediaPipe 平均帧率: {mp_fps:.1f} FPS  (累计 {mp_frame_count} 帧, 总耗时 {mp_total_time:.1f}s)")

            current_time = time.time()
            collect_data = False   # 默认不收集，仅检测状态开启

            if results.multi_hand_landmarks:
                # 绘制手部骨架
                self.mp_draw.draw_landmarks(frame, results.multi_hand_landmarks[0],
                                            mp.solutions.hands.HAND_CONNECTIONS)

                # ---- 提取归一化坐标（仅用于运动检测，不进入模型） ----
                norm_lm = results.multi_hand_landmarks[0].landmark
                curr_hand_norm = np.array([[lm.x, lm.y] for lm in norm_lm])

                is_moving = False
                if self.prev_hand_norm is not None:
                    diff = curr_hand_norm - self.prev_hand_norm
                    max_move = np.max(np.linalg.norm(diff, axis=1))
                    if max_move > MOVEMENT_THRESHOLD:
                        self.last_move_time = current_time
                        is_moving = True
                self.prev_hand_norm = curr_hand_norm

                # ---- 状态机流转 ----
                if self.current_state == STATE_IDLE:
                    # 清空缓冲，等待运动触发
                    self.raw_frames = []
                    self.raw_hand_labels = []
                    self.frame_count = 0
                    self._push_state(STATE_IDLE)
                    if is_moving:
                        self.current_state = STATE_PREPARE
                        self.state_start_time = current_time
                        print("[状态机] IDLE → PREPARE")

                elif self.current_state == STATE_PREPARE:
                    elapsed = current_time - self.state_start_time
                    remaining = PREPARE_TIME - elapsed
                    self._push_state(STATE_PREPARE, round(remaining, 1))
                    if remaining <= 0:
                        self.current_state = STATE_DETECT
                        self.raw_frames = []
                        self.raw_hand_labels = []
                        self.frame_count = 0
                        self.state_start_time = current_time
                        print("[状态机] PREPARE → DETECT")

                elif self.current_state == STATE_DETECT:
                    collect_data = True
                    self._push_state(STATE_DETECT, self.frame_count)

                    if self.model_loaded and self.frame_count >= WINDOW_SIZE:
                        # ---- 补帧：对所有有效帧做线性插值到 WINDOW_SIZE 帧 ----
                        interpolated = interpolate_keypoints_to_window(self.raw_frames, WINDOW_SIZE)
                        if interpolated is not None:
                            # ---- 左右手投票：取前3帧标签投票，统一翻转 ----
                            vote_labels = self.raw_hand_labels[:3]
                            dominant = vote_hand_label(vote_labels)
                            if dominant == 'Left':
                                wrist_x = interpolated[:, 0, 0:1]  # (T, 1)
                                interpolated[:, :, 0] = 2 * wrist_x - interpolated[:, :, 0]

                            t_infer_start = time.perf_counter()
                            result = infer_multistream(self.active_models, interpolated, self.device)
                            t_infer_end = time.perf_counter()
                            latency_ms = (t_infer_end - t_infer_start) * 1000
                            infer_latencies.append(latency_ms)

                            if result is not None:
                                action, confidence = result
                                print(f"[推理] {action} ({confidence:.3f})  延迟: {latency_ms:.1f} ms  有效帧: {len(self.raw_frames)}/{WINDOW_SIZE}")
                                if len(infer_latencies) % 20 == 0:
                                    avg_lat = sum(infer_latencies[-20:]) / 20
                                    print(f"[性能] 近20次推理平均端到端延迟: {avg_lat:.1f} ms")
                                try:
                                    self.result_queue.put_nowait((action, confidence))
                                except Exception:
                                    pass

                        # 无论识别成功与否，强制进入冷却
                        self.current_state = STATE_COOLDOWN
                        self.state_start_time = current_time
                        self.raw_frames = []
                        self.raw_hand_labels = []
                        self.frame_count = 0
                        print("[状态机] DETECT → COOLDOWN")

                elif self.current_state == STATE_COOLDOWN:
                    elapsed = current_time - self.state_start_time
                    remaining = ACTION_COOLDOWN - elapsed
                    self._push_state(STATE_COOLDOWN, round(remaining, 1))
                    if remaining <= 0:
                        self.current_state = STATE_IDLE
                        self.raw_frames = []
                        self.raw_hand_labels = []
                        self.frame_count = 0
                        print("[状态机] COOLDOWN → IDLE")

                # ---- 数据收集（仅检测状态，不做翻转，翻转在推理前统一处理） ----
                if collect_data:
                    h, w = frame.shape[:2]
                    norm_landmarks = results.multi_hand_landmarks[0].landmark
                    current_kpts = np.array([[lm.x * w, lm.y * h, lm.z] for lm in norm_landmarks])

                    # 记录原始关键点（不翻转）和手部标签
                    hand_label = None
                    if results.multi_handedness:
                        hand_label = results.multi_handedness[0].classification[0].label
                    self.raw_frames.append(current_kpts)
                    self.raw_hand_labels.append(hand_label)
                    self.frame_count += 1

            else:
                # 未检测到手：重置运动检测，若在预备/检测阶段则强制回到静止
                self.prev_hand_norm = None
                self._push_state(STATE_IDLE)
                if self.current_state in (STATE_PREPARE, STATE_DETECT):
                    self.current_state = STATE_IDLE
                    self.raw_frames = []
                    self.raw_hand_labels = []
                    self.frame_count = 0
                    print("[状态机] 手丢失 → IDLE")

            # 将帧放入队列
            if not self.frame_queue.full():
                self.frame_queue.put(frame)

            time.sleep(0.03)  # 控制帧率

    def stop(self):
        self.running = False
        if self.cap:
            self.cap.release()


class GestureRecognitionApp(ctk.CTk):
    def __init__(self):
        super().__init__()
        
        # 窗口设置
        self.title("ST-GCN 手势识别系统")
        self.geometry("1200x800")
        self.minsize(1000, 700)
        
        # 创建队列
        self.frame_queue = queue.Queue(maxsize=1)
        self.result_queue = queue.Queue(maxsize=1)
        self.state_queue  = queue.Queue(maxsize=4)   # 状态机状态推送队列
        
        # 摄像头线程
        self.camera_thread = None
        self.camera_running = False
        
        # 主布局
        self.grid_rowconfigure(0, weight=1)
        self.grid_columnconfigure(0, weight=2)
        self.grid_columnconfigure(1, weight=1)
        
        # ===== 左侧：视频显示区 =====
        left_frame = ctk.CTkFrame(self, corner_radius=10)
        left_frame.grid(row=0, column=0, padx=15, pady=15, sticky="nsew")
        left_frame.grid_rowconfigure(1, weight=1)
        left_frame.grid_columnconfigure(0, weight=1)

        # ===== 状态指示器（row=0，固定在视频上方） =====
        state_bar = ctk.CTkFrame(left_frame, corner_radius=8, height=48)
        state_bar.grid(row=0, column=0, padx=10, pady=(10, 0), sticky="ew")
        state_bar.grid_columnconfigure(0, weight=1)
        state_bar.grid_columnconfigure(1, weight=0)
        state_bar.grid_propagate(False)

        # 左侧：状态文字
        self.state_label = ctk.CTkLabel(
            state_bar, text="⏸  等待开启摄像头",
            font=("Arial", 14, "bold"), text_color="#888",
            anchor="w"
        )
        self.state_label.grid(row=0, column=0, padx=12, pady=8, sticky="ew")

        # 右侧：4个状态圆点（静止/预备/检测/冷却）
        dot_frame = ctk.CTkFrame(state_bar, fg_color="transparent")
        dot_frame.grid(row=0, column=1, padx=8, pady=4, sticky="e")

        STATE_LABELS = ["静止", "预备", "检测", "冷却"]
        STATE_COLORS_OFF = ["#333", "#333", "#333", "#333"]
        self._state_dots = []
        self._state_dot_labels = []
        for i, lbl in enumerate(STATE_LABELS):
            col_frame = ctk.CTkFrame(dot_frame, fg_color="transparent")
            col_frame.grid(row=0, column=i, padx=4)
            dot = ctk.CTkLabel(col_frame, text="●", font=("Arial", 18),
                               text_color=STATE_COLORS_OFF[i], width=24)
            dot.grid(row=0, column=0)
            name = ctk.CTkLabel(col_frame, text=lbl, font=("Arial", 9),
                                text_color="#555")
            name.grid(row=1, column=0)
            self._state_dots.append(dot)
            self._state_dot_labels.append(name)

        # 进度条（预备/检测时显示）
        self.state_progress = ctk.CTkProgressBar(left_frame, height=4, corner_radius=2)
        self.state_progress.grid(row=0, column=0, padx=10, pady=(0, 0), sticky="sew")
        self.state_progress.set(0)
        self.state_progress.grid_remove()   # 默认隐藏

        # 视频显示区域
        self.video_label = ctk.CTkLabel(left_frame, text="📷 摄像头画面\n\n点击开启摄像头",
                                      font=("Arial", 16), text_color="#ccc")
        self.video_label.grid(row=1, column=0, padx=10, pady=10, sticky="nsew")
        self.video_label.configure(bg_color="#222", corner_radius=8)
        
        # 控制按钮
        btn_frame = ctk.CTkFrame(left_frame, corner_radius=8)
        btn_frame.grid(row=2, column=0, padx=10, pady=10, sticky="nsew")
        btn_frame.grid_columnconfigure(0, weight=1)
        btn_frame.grid_columnconfigure(1, weight=1)
        
        self.start_btn = ctk.CTkButton(btn_frame, text="▶ 开启摄像头", font=("Arial", 14), 
                                     command=self.start_camera)
        self.start_btn.grid(row=0, column=0, padx=5, pady=10, sticky="ew")
        
        self.stop_btn = ctk.CTkButton(btn_frame, text="⏹ 关闭摄像头", font=("Arial", 14), 
                                    state="disabled", command=self.stop_camera)
        self.stop_btn.grid(row=0, column=1, padx=5, pady=10, sticky="ew")
        
        # 识别结果区域
        result_frame = ctk.CTkFrame(left_frame, corner_radius=10)
        result_frame.grid(row=3, column=0, padx=10, pady=10, sticky="nsew")
        result_frame.grid_rowconfigure(0, weight=1)
        result_frame.grid_columnconfigure(0, weight=1)
        
        result_title = ctk.CTkLabel(result_frame, text="🎯 识别结果", font=("Arial", 16, "bold"), text_color="#00bcd4")
        result_title.grid(row=0, column=0, padx=10, pady=10, sticky="nw")
        
        self.result_label = ctk.CTkLabel(result_frame, text="等待识别...", 
                                      font=("Arial", 24, "bold"), text_color="#4caf50")
        self.result_label.grid(row=1, column=0, padx=20, pady=20, sticky="nsew")
        
        # ===== 右侧：控制面板区 =====
        right_frame = ctk.CTkFrame(self, corner_radius=10)
        right_frame.grid(row=0, column=1, padx=15, pady=15, sticky="nsew")
        right_frame.grid_rowconfigure(0, weight=1)
        right_frame.grid_columnconfigure(0, weight=1)
        
        # 创建标签页
        self.tab_view = ctk.CTkTabview(right_frame, corner_radius=8)
        self.tab_view.grid(row=0, column=0, padx=10, pady=10, sticky="nsew")
        
        # 标签页1：模型管理
        model_tab = self.tab_view.add("🤖 模型管理")
        model_tab.grid_rowconfigure(3, weight=1)
        model_tab.grid_columnconfigure(0, weight=1)
        
        # 扫描 models/ 目录下的 .pth 文件
        def scan_models():
            model_dir = "models"
            if not os.path.exists(model_dir):
                return ["(无模型文件)"]
            files = [f for f in os.listdir(model_dir) if f.endswith('.pth')]
            return files if files else ["(无模型文件)"]
        
        available_models = scan_models()
        
        # 为三个流分别创建配置行
        self.stream_ui = {}  # 存储每个流的 UI 控件引用
        
        stream_defs = [
            ("joint",  "关节流 (Joint)",  "best_new.pth",  1.00),
            ("bone",   "骨骼流 (Bone)",   "best_bone.pth", 0.90),
            ("motion", "运动流 (Motion)", "best_motion.pth", 0.82),
        ]
        
        for row_idx, (stream_name, label_text, default_model, default_weight) in enumerate(stream_defs):
            row_frame = ctk.CTkFrame(model_tab, corner_radius=6)
            row_frame.grid(row=row_idx, column=0, padx=8, pady=4, sticky="ew")
            row_frame.grid_columnconfigure(1, weight=1)
            
            # 启用开关
            enable_var = ctk.BooleanVar(value=STREAMS_CONFIG[stream_name]["enable"])
            chk = ctk.CTkCheckBox(row_frame, text=label_text, variable=enable_var,
                                  font=("Arial", 12, "bold"), width=140)
            chk.grid(row=0, column=0, padx=8, pady=6, sticky="w")
            
            # 模型文件下拉
            # 找到默认选项
            default_val = default_model if default_model in available_models else available_models[0]
            model_var = ctk.StringVar(value=default_val)
            combo = ctk.CTkComboBox(row_frame, values=available_models, variable=model_var,
                                    width=160, font=("Arial", 11))
            combo.grid(row=0, column=1, padx=6, pady=6, sticky="ew")
            
            # 权重输入
            weight_entry = ctk.CTkEntry(row_frame, width=55, font=("Arial", 11),
                                        placeholder_text="权重")
            weight_entry.insert(0, str(default_weight))
            weight_entry.grid(row=0, column=2, padx=6, pady=6)
            
            self.stream_ui[stream_name] = {
                "enable_var": enable_var,
                "model_var": model_var,
                "combo": combo,          # 保存 combo 引用，用于刷新 values
                "weight_entry": weight_entry,
            }
        
        # 刷新模型列表按钮 + 应用配置按钮
        btn_row = ctk.CTkFrame(model_tab, corner_radius=6)
        btn_row.grid(row=3, column=0, padx=8, pady=6, sticky="ew")
        btn_row.grid_columnconfigure(0, weight=1)
        btn_row.grid_columnconfigure(1, weight=1)
        
        refresh_btn = ctk.CTkButton(btn_row, text="🔄 刷新列表", font=("Arial", 12),
                                    command=self._refresh_model_list)
        refresh_btn.grid(row=0, column=0, padx=5, pady=6, sticky="ew")
        
        apply_btn = ctk.CTkButton(btn_row, text="✅ 应用配置", font=("Arial", 12),
                                  fg_color="#2e7d32", hover_color="#1b5e20",
                                  command=self._apply_model_config)
        apply_btn.grid(row=0, column=1, padx=5, pady=6, sticky="ew")
        
        # 状态提示标签
        self.model_status_label = ctk.CTkLabel(model_tab, text="当前使用默认配置",
                                               font=("Arial", 11), text_color="#aaa")
        self.model_status_label.grid(row=4, column=0, padx=8, pady=4, sticky="w")
        
        # 标签页2：识别历史
        history_tab = self.tab_view.add("📋 识别历史")
        history_tab.grid_rowconfigure(0, weight=1)
        history_tab.grid_columnconfigure(0, weight=1)
        
        history_frame = ctk.CTkFrame(history_tab, corner_radius=8)
        history_frame.grid(row=0, column=0, padx=10, pady=10, sticky="nsew")
        
        history_title = ctk.CTkLabel(history_frame, text="最近识别记录", font=("Arial", 14, "bold"), text_color="#00bcd4")
        history_title.grid(row=0, column=0, padx=10, pady=10, sticky="nw")
        
        self.history_labels = []
        for i in range(5):
            item = ctk.CTkLabel(history_frame, text="", font=("Arial", 12))
            item.grid(row=i+1, column=0, padx=10, pady=5, sticky="w")
            self.history_labels.append(item)
        
        # 标签页3：离线检测
        offline_tab = self.tab_view.add("📁 离线检测")
        offline_tab.grid_rowconfigure(0, weight=1)
        offline_tab.grid_columnconfigure(0, weight=1)
        
        video_frame = ctk.CTkFrame(offline_tab, corner_radius=8)
        video_frame.grid(row=0, column=0, padx=10, pady=10, sticky="nsew")
        
        video_title = ctk.CTkLabel(video_frame, text="选择视频文件", font=("Arial", 14, "bold"), text_color="#00bcd4")
        video_title.grid(row=0, column=0, padx=10, pady=10, sticky="nw")
        
        self.video_path_label = ctk.CTkLabel(video_frame, text="未选择视频", font=("Arial", 12))
        self.video_path_label.grid(row=1, column=0, padx=10, pady=5, sticky="w")
        
        btn_frame = ctk.CTkFrame(video_frame)
        btn_frame.grid(row=2, column=0, padx=10, pady=10, sticky="ew")
        btn_frame.grid_columnconfigure(0, weight=1)
        btn_frame.grid_columnconfigure(1, weight=1)
        
        self.select_video_btn = ctk.CTkButton(btn_frame, text="📂 选择视频...", font=("Arial", 14),
                                           command=self.select_video)
        self.select_video_btn.grid(row=0, column=0, padx=5, pady=5, sticky="ew")
        
        self.select_folder_btn = ctk.CTkButton(btn_frame, text="📂 选择图片文件夹...", font=("Arial", 14),
                                            command=self.select_folder)
        self.select_folder_btn.grid(row=0, column=1, padx=5, pady=5, sticky="ew")
        
        options_frame = ctk.CTkFrame(offline_tab, corner_radius=8)
        options_frame.grid(row=1, column=0, padx=10, pady=10, sticky="nsew")
        
        options_title = ctk.CTkLabel(options_frame, text="处理选项", font=("Arial", 14, "bold"), text_color="#00bcd4")
        options_title.grid(row=0, column=0, padx=10, pady=10, sticky="nw")
        
        self.check_save = ctk.CTkCheckBox(options_frame, text="保存关键点数据", font=("Arial", 12))
        self.check_save.grid(row=1, column=0, padx=10, pady=5, sticky="w")
        self.check_save.select()
        
        self.check_export = ctk.CTkCheckBox(options_frame, text="导出识别结果", font=("Arial", 12))
        self.check_export.grid(row=2, column=0, padx=10, pady=5, sticky="w")
        
        self.process_btn = ctk.CTkButton(offline_tab, text="▶️ 开始处理", font=("Arial", 14),
                                      command=self.process_offline)
        self.process_btn.grid(row=2, column=0, padx=10, pady=10, sticky="ew")
        
        # 进度条
        self.progress_bar = ctk.CTkProgressBar(offline_tab)
        self.progress_bar.grid(row=3, column=0, padx=10, pady=10, sticky="ew")
        self.progress_bar.set(0)
        
        # 处理结果显示
        result_frame = ctk.CTkFrame(offline_tab, corner_radius=8)
        result_frame.grid(row=4, column=0, padx=10, pady=10, sticky="nsew")
        
        result_title = ctk.CTkLabel(result_frame, text="处理结果", font=("Arial", 14, "bold"), text_color="#00bcd4")
        result_title.grid(row=0, column=0, padx=10, pady=10, sticky="nw")
        
        self.offline_result_label = ctk.CTkLabel(result_frame, text="等待处理...", font=("Arial", 16))
        self.offline_result_label.grid(row=1, column=0, padx=10, pady=10, sticky="w")
        
        # 标签页4：参数设置
        settings_tab = self.tab_view.add("⚙️ 参数设置")
        settings_tab.grid_rowconfigure(0, weight=1)
        settings_tab.grid_columnconfigure(0, weight=1)
        
        params_frame = ctk.CTkFrame(settings_tab, corner_radius=8)
        params_frame.grid(row=0, column=0, padx=10, pady=10, sticky="nsew")
        
        params_title = ctk.CTkLabel(params_frame, text="识别参数", font=("Arial", 14, "bold"), text_color="#00bcd4")
        params_title.grid(row=0, column=0, padx=10, pady=10, sticky="nw")
        
        # 置信度阈值
        conf_thresh_label = ctk.CTkLabel(params_frame, text="置信度阈值:")
        conf_thresh_label.grid(row=1, column=0, padx=10, pady=5, sticky="w")
        
        self.spin_conf = ctk.CTkEntry(params_frame, width=80)
        self.spin_conf.grid(row=1, column=1, padx=10, pady=5, sticky="w")
        self.spin_conf.insert(0, "85")
        
        # 窗口大小
        window_label = ctk.CTkLabel(params_frame, text="窗口大小:")
        window_label.grid(row=2, column=0, padx=10, pady=5, sticky="w")
        
        self.spin_window = ctk.CTkEntry(params_frame, width=80)
        self.spin_window.grid(row=2, column=1, padx=10, pady=5, sticky="w")
        self.spin_window.insert(0, "35")
        
        # 保存按钮
        self.save_settings_btn = ctk.CTkButton(settings_tab, text="💾 保存设置", font=("Arial", 14),
                                             command=self.save_settings)
        self.save_settings_btn.grid(row=1, column=0, padx=10, pady=10, sticky="ew")
        
        # 更新定时器
        self.timer = self.after(30, self.update_frame)
        
        # 离线处理相关
        self.offline_thread = None
        self.selected_path = None
        self.is_folder = False
    
    def start_camera(self):
        """开启摄像头"""
        if not self.camera_running:
            self.camera_thread = CameraThread(
                self.frame_queue, self.result_queue, self.state_queue
            )
            self.camera_thread.start()
            self.camera_running = True
            self.start_btn.configure(state="disabled")
            self.stop_btn.configure(state="normal")
            self.video_label.configure(text="")
            self._update_state_display(STATE_IDLE)

    def stop_camera(self):
        """关闭摄像头"""
        if self.camera_running:
            self.camera_thread.stop()
            self.camera_thread.join()
            self.camera_running = False
            self.start_btn.configure(state="normal")
            self.stop_btn.configure(state="disabled")
            self.video_label.configure(text="📷 摄像头画面\n\n点击开启摄像头")
            self.result_label.configure(text="等待识别...")
            # 重置状态指示器
            self.state_label.configure(text="⏸  等待开启摄像头", text_color="#888")
            for dot in self._state_dots:
                dot.configure(text_color="#333")
            self.state_progress.grid_remove()

    def update_frame(self):
        """更新视频画面、识别结果和状态指示器"""
        # 更新视频画面
        try:
            if not self.frame_queue.empty():
                frame = self.frame_queue.get()
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                img = Image.fromarray(rgb_frame)
                imgtk = ImageTk.PhotoImage(image=img)
                self.video_label.configure(image=imgtk)
                self.video_label.image = imgtk
        except Exception:
            pass

        # 更新识别结果
        try:
            if not self.result_queue.empty():
                action, confidence = self.result_queue.get()
                if confidence >= CONFIDENCE_THRESHOLD:
                    color = "#4caf50"  # 绿色
                    self.add_to_history(f"{action} ({confidence:.2f})")
                else:
                    color = "#ff9800"  # 橙色（低置信度，仅显示不记录）
                self.result_label.configure(
                    text=f"{action}\n({confidence:.2f})",
                    text_color=color
                )
        except Exception:
            pass

        # 更新状态指示器（只取最新一条，丢弃积压的旧消息）
        try:
            msg = None
            while not self.state_queue.empty():
                msg = self.state_queue.get_nowait()
            if msg is not None:
                self._update_state_display(msg["state"], msg.get("extra"))
        except Exception:
            pass

        # 继续定时更新
        self.timer = self.after(30, self.update_frame)

    def _update_state_display(self, state_id, extra=None):
        """根据状态 ID 更新状态栏 UI"""
        # 各状态的配色和文字
        STATE_CFG = {
            STATE_IDLE:     {"color": "#607d8b", "text": "⏸  静止监测 — 等待手部运动",    "dot": "#607d8b"},
            STATE_PREPARE:  {"color": "#ffc107", "text": "⏳  预备中...",                  "dot": "#ffc107"},
            STATE_DETECT:   {"color": "#4caf50", "text": "🔍  正在检测",                   "dot": "#4caf50"},
            STATE_COOLDOWN: {"color": "#ff7043", "text": "❄️  冷却中...",                  "dot": "#ff7043"},
        }
        cfg = STATE_CFG.get(state_id, STATE_CFG[STATE_IDLE])

        # 更新文字（带倒计时/帧数）
        text = cfg["text"]
        if state_id == STATE_PREPARE and extra is not None:
            text = f"⏳  预备中... {extra}s"
        elif state_id == STATE_COOLDOWN and extra is not None:
            text = f"❄️  冷却中... {extra}s"
        elif state_id == STATE_DETECT and extra is not None:
            text = f"🔍  正在检测  ({extra}/{WINDOW_SIZE} 帧)"
        self.state_label.configure(text=text, text_color=cfg["color"])

        # 更新圆点：当前状态高亮，其余暗色
        DOT_ACTIVE = {
            STATE_IDLE:     "#607d8b",
            STATE_PREPARE:  "#ffc107",
            STATE_DETECT:   "#4caf50",
            STATE_COOLDOWN: "#ff7043",
        }
        for i, dot in enumerate(self._state_dots):
            dot.configure(text_color=DOT_ACTIVE[i] if i == state_id else "#333")

        # 进度条：预备阶段显示倒计时进度，检测阶段显示帧填充进度
        if state_id == STATE_PREPARE and extra is not None:
            self.state_progress.grid()
            progress = 1.0 - (extra / PREPARE_TIME)
            self.state_progress.configure(progress_color="#ffc107")
            self.state_progress.set(max(0.0, min(1.0, progress)))
        elif state_id == STATE_DETECT and extra is not None:
            self.state_progress.grid()
            progress = extra / WINDOW_SIZE
            self.state_progress.configure(progress_color="#4caf50")
            self.state_progress.set(max(0.0, min(1.0, progress)))
        elif state_id == STATE_COOLDOWN and extra is not None:
            self.state_progress.grid()
            progress = 1.0 - (extra / ACTION_COOLDOWN)
            self.state_progress.configure(progress_color="#ff7043")
            self.state_progress.set(max(0.0, min(1.0, progress)))
        else:
            self.state_progress.set(0)
            self.state_progress.grid_remove()
    
    def add_to_history(self, text):
        """添加识别历史记录"""
        # 移除最旧的记录
        for i in range(len(self.history_labels) - 1):
            self.history_labels[i].configure(text=self.history_labels[i+1].cget("text"))
        # 添加新记录
        self.history_labels[-1].configure(text=text)
    
    def select_video(self):
        """选择视频文件"""
        file_path = ctk.filedialog.askopenfilename(
            filetypes=[("视频文件", "*.mp4 *.avi *.mov *.mkv")]
        )
        
        if file_path:
            self.selected_path = file_path
            self.is_folder = False
            self.video_path_label.configure(text=os.path.basename(file_path))
    
    def select_folder(self):
        """选择图片文件夹"""
        folder_path = ctk.filedialog.askdirectory()
        
        if folder_path:
            self.selected_path = folder_path
            self.is_folder = True
            self.video_path_label.configure(text=os.path.basename(folder_path))
    
    def process_offline(self):
        """开始离线处理"""
        if not self.selected_path:
            ctk.CTkMessageBox(message="请先选择视频文件或图片文件夹！", title="警告")
            return
        
        self.process_btn.configure(state="disabled")
        self.progress_bar.set(0)
        self.offline_result_label.configure(text="处理中...")
        
        # 启动处理线程
        self.offline_thread = OfflineProcessorThread(
            path=self.selected_path,
            is_folder=self.is_folder,
            save_data=self.check_save.get(),
            export_result=self.check_export.get(),
            callback=self.offline_callback
        )
        self.offline_thread.start()
    
    def offline_callback(self, data):
        """离线处理回调"""
        if "progress" in data:
            self.progress_bar.set(data["progress"])
        elif "error" in data:
            self.offline_result_label.configure(text=f"处理失败: {data['error']}")
            self.process_btn.configure(state="normal")
        elif "completed" in data:
            results = data.get("results", [])
            if results:
                summary = f"处理完成！\n识别到 {len(results)} 个手势"
                self.offline_result_label.configure(text=summary)
                
                # 添加到历史记录
                for result in results:
                    self.add_to_history(f"{result['action']} ({result['confidence']:.2f})")
            else:
                self.offline_result_label.configure(text="处理完成！未识别到手势")
            self.process_btn.configure(state="normal")
    
    def save_settings(self):
        """保存设置"""
        try:
            conf = int(self.spin_conf.get())
            window = int(self.spin_window.get())
            
            # 更新置信度阈值
            if hasattr(self, 'camera_thread') and self.camera_thread:
                self.camera_thread.CONFIDENCE_THRESHOLD = conf / 100.0
            
            # 更新窗口大小（需要重启才能生效）
            if hasattr(self, 'camera_thread') and self.camera_thread:
                self.camera_thread.WINDOW_SIZE = window
                self.camera_thread.buffer = np.zeros((window, 21, 3))
                self.camera_thread.frame_count = 0
            
            ctk.CTkMessageBox(message="设置已保存！", title="成功")
        except Exception as e:
            ctk.CTkMessageBox(message=f"保存失败: {str(e)}", title="错误")
    def _refresh_model_list(self):
        """刷新 models/ 目录下的模型文件列表"""
        model_dir = "models"
        if not os.path.exists(model_dir):
            files = ["(无模型文件)"]
        else:
            files = [f for f in os.listdir(model_dir) if f.endswith('.pth')]
            if not files:
                files = ["(无模型文件)"]
        
        # 更新所有流的下拉框选项
        for stream_name, ui in self.stream_ui.items():
            current_val = ui["model_var"].get()
            ui["combo"].configure(values=files)
            # 如果当前选中的文件仍然存在，保持选中；否则选第一个
            if current_val not in files:
                ui["model_var"].set(files[0])
        
        self.model_status_label.configure(
            text=f"已刷新，找到 {len(files)} 个模型文件", text_color="#aaa"
        )

    def _apply_model_config(self):
        """读取 UI 配置，更新 STREAMS_CONFIG 并重新加载模型"""
        # 如果摄像头正在运行，先停止
        was_running = self.camera_running
        if was_running:
            self.stop_camera()
        
        # 从 UI 读取配置并更新全局 STREAMS_CONFIG
        errors = []
        for stream_name, ui in self.stream_ui.items():
            enable = ui["enable_var"].get()
            model_file = ui["model_var"].get()
            try:
                weight = float(ui["weight_entry"].get())
            except ValueError:
                errors.append(f"{stream_name} 权重格式错误")
                weight = STREAMS_CONFIG[stream_name]["weight"]
            
            model_path = os.path.join("models", model_file) if model_file != "(无模型文件)" else ""
            STREAMS_CONFIG[stream_name]["enable"] = enable
            STREAMS_CONFIG[stream_name]["model_path"] = model_path
            STREAMS_CONFIG[stream_name]["weight"] = weight
        
        if errors:
            self.model_status_label.configure(
                text="⚠️ " + "；".join(errors), text_color="#ff9800"
            )
            return
        
        # 重新加载模型（在后台线程中执行，避免阻塞 UI）
        self.model_status_label.configure(text="⏳ 正在加载模型...", text_color="#00bcd4")
        self.update()  # 强制刷新 UI
        
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        new_models = load_stream_models(device)
        
        # 将新模型注入到 CameraThread（如果存在）
        if self.camera_thread is not None:
            self.camera_thread.active_models = new_models
            self.camera_thread.model_loaded = len(new_models) > 0
        
        enabled_streams = [k for k, v in STREAMS_CONFIG.items() if v["enable"]]
        loaded_streams = list(new_models.keys())
        
        if not loaded_streams:
            self.model_status_label.configure(
                text="❌ 没有成功加载任何模型", text_color="#f44336"
            )
        else:
            self.model_status_label.configure(
                text=f"✅ 已加载: {', '.join(loaded_streams)}", text_color="#4caf50"
            )
        
        # 如果之前摄像头在运行，重新启动
        if was_running:
            self.start_camera()


if __name__ == "__main__":
    app = GestureRecognitionApp()
    app.mainloop()

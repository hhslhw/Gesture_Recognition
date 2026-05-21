"""
统一的关键点提取脚本
整合视频处理和关键点提取功能，支持补帧和多种姿态估计算法
"""
import cv2
import numpy as np
import os
import random
import torch
from pathlib import Path
from scipy import interpolate

# ==================== 配置区 ====================
CONFIG = {
    # 输入输出路径
    'input_dir': r'E:\work\dataset\jester\classified_jester\train',  # 分类后的视频目录
    'output_base_dir': r'E:\work\gesture_recognition\mediapipe_results_2',  # 输出基础目录
    
    # 动作类别名称（中文）
    'action_classes': [
        "单手放大", "单手缩小", "双指向上滑动", "双指向下滑动",
        "双指向右滑动", "双指向左滑动", "双指拉近", "双指放大", "拉手靠近"
    ],
    
    # 有效帧阈值（少于此值的样本将被丢弃）
    'min_valid_frames': 10,
    
    # 帧数设置
    'target_frames': 35,
    
    # 补帧开关
    'enable_interpolation': True,  # True: 补帧，False: 不补帧
    
    # 算法选择：'mediapipe', 'alphapose', 'mmpose'
    'algorithm': 'mediapipe',
    
    # 每个类别目标样本数
    'target_samples_per_class': 250,  # 2250 / 9 = 250
}

# ==================== 姿态估计算法类 ====================

class PoseEstimator:
    """姿态估计基类"""
    def __init__(self):
        pass
    
    def extract_keypoints(self, frame):
        """从单帧提取关键点"""
        raise NotImplementedError
    
    def get_num_keypoints(self):
        """返回关键点数量"""
        raise NotImplementedError


class MediaPipeEstimator(PoseEstimator):
    """MediaPipe Hands 姿态估计（使用新版 Tasks API）"""
    def __init__(self):
        try:
            import mediapipe as mp
            from mediapipe.tasks import python
            from mediapipe.tasks.python import vision
            
            # 使用新版 Tasks API
            self.mp = mp
            options = vision.HandLandmarkerOptions(
                base_options=python.BaseOptions(
                    model_asset_path='models/hand_landmarker.task'  # 需要下载 .task 模型文件
                ),
                running_mode=vision.RunningMode.IMAGE,  # 图片模式
                num_hands=2,  # 最多检测 2 只手
                min_hand_detection_confidence=0.5,
                min_hand_presence_confidence=0.5,
                min_tracking_confidence=0.5
            )
            
            self.landmarker = vision.HandLandmarker.create_from_options(options)
            print("MediaPipe 模型初始化成功（使用 Tasks API）")
        except ImportError:
            print("警告：MediaPipe 未安装，请先安装：pip install mediapipe")
            self.landmarker = None
        except Exception as e:
            print(f"MediaPipe 初始化失败：{e}")
            import traceback
            traceback.print_exc()
            self.landmarker = None
    
    def extract_keypoints(self, frame):
        """提取手部关键点，返回关键点和左右手信息"""
        if self.landmarker is None:
            return None, None
        
        try:
            # 转换为 RGB 格式
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # 创建 MediaPipe Image 对象
            mp_image = self.mp.Image(
                image_format=self.mp.ImageFormat.SRGB,
                data=frame_rgb
            )
            
            # 执行检测
            detection_result = self.landmarker.detect(mp_image)
            
            # 检查是否有检测结果
            if not detection_result.hand_landmarks:
                return None, None
            
            # 选择置信度最高的手
            if len(detection_result.hand_landmarks) > 1 and detection_result.handedness:
                scores = [h[0].score for h in detection_result.handedness if h]
                best_hand_idx = np.argmax(scores)
            else:
                best_hand_idx = 0
            
            # 提取关键点
            hand_landmarks = detection_result.hand_landmarks[best_hand_idx]
            keypoints = []
            
            # 获取图像尺寸用于坐标转换
            h, w = frame.shape[:2]
            
            for landmark in hand_landmarks:
                # 转换归一化坐标到像素坐标
                x = landmark.x * w
                y = landmark.y * h
                z = landmark.z  # 相对深度值
                keypoints.append([x, y, z])
            
            # 获取左右手信息
            hand_label = None
            if detection_result.handedness and len(detection_result.handedness) > best_hand_idx:
                hand_label = detection_result.handedness[best_hand_idx][0].category_name  # 'Left' 或 'Right'
            
            return np.array(keypoints), hand_label  # (21, 3), 'Left'/'Right'/None
        
        except Exception as e:
            print(f"MediaPipe 推理失败：{e}")
            import traceback
            traceback.print_exc()
        
        return None, None
    
    def get_num_keypoints(self):
        return 21
    
    def close(self):
        """关闭资源"""
        if hasattr(self, 'landmarker') and self.landmarker is not None:
            self.landmarker.close()


class AlphaPoseEstimator(PoseEstimator):
    """AlphaPose 姿态估计（需要安装 AlphaPose）"""
    def __init__(self):
        print("警告：AlphaPose 需要单独安装，当前为占位实现")
        # TODO: 初始化 AlphaPose 模型
    
    def extract_keypoints(self, frame):
        # TODO: 实现 AlphaPose 关键点提取
        raise NotImplementedError("AlphaPose 尚未实现")
    
    def get_num_keypoints(self):
        return 133  # AlphaPose 全身关键点数


class MMPoseEstimator(PoseEstimator):
    """MMPose 手部姿态估计"""
    def __init__(self):
        try:
            from mmpose.apis import MMPoseInferencer
            
            # 使用正确的初始化方式（与 test_mmpose_simple.py 中成功的方法一致）
            # 不传 model 参数，而是通过 pose2d 参数指定模型别名
            self.inferencer = MMPoseInferencer(pose2d='hand')
            print("MMPose 模型初始化成功")
            print(f"使用的模型：hand 2D keypoints")
        except ImportError:
            print("警告：MMPose 未安装，请先安装：pip install mmpose")
            self.inferencer = None
        except Exception as e:
            print(f"MMPose 初始化失败：{e}")
            import traceback
            traceback.print_exc()
            self.inferencer = None
    
    def extract_keypoints(self, frame):
        """提取手部关键点"""
        if self.inferencer is None:
            print("MMPose 推理器未初始化")
            return None
        
        try:
            # 执行推理（不显示结果，只获取关键点）
            # inputs 可以是图像路径、numpy 数组或图像列表
            result_generator = self.inferencer(frame, show=False)
            results = next(result_generator)
            
            # 获取手部关键点
            if 'predictions' in results:
                predictions = results['predictions']
                
                if len(predictions) > 0:
                    pred = predictions[0]
                    
                    if len(pred) > 0:
                        hand_data = pred[0]
                        
                        if 'keypoints' in hand_data:
                            hand_keypoints = hand_data['keypoints']
                            
                            # 如果是 list，转换为 numpy 数组
                            if isinstance(hand_keypoints, list):
                                hand_keypoints = np.array(hand_keypoints)
                            
                            # 如果只有 x,y 坐标，添加 z=0
                            if hand_keypoints.shape[1] == 2:
                                z_coords = np.zeros((hand_keypoints.shape[0], 1))
                                hand_keypoints = np.hstack((hand_keypoints, z_coords))
                            else:
                                # 将 z 坐标置为 0（因为 2D 视频不支持深度估计）
                                hand_keypoints[:, 2] = 0
                            
                            return hand_keypoints  # (21, 3)
        
        except Exception as e:
            print(f"MMPose 推理失败：{e}")
            import traceback
            traceback.print_exc()
        
        return None
    
    def get_num_keypoints(self):
        return 21  # 手部关键点数
    
    def close(self):
        """关闭资源"""
        if hasattr(self, 'inferencer') and self.inferencer is not None:
            # 清理资源
            pass


# ==================== 补帧功能 ====================

def interpolate_keypoints(keypoints_sequence, target_frames):
    """
    对关键点序列进行插值补帧
    
    Args:
        keypoints_sequence: list of (V, C) arrays, 有效帧的关键点
        target_frames: 目标帧数
    
    Returns:
        interpolated: (target_frames, V, C) array
    """
    if len(keypoints_sequence) == 0:
        return None
    
    if len(keypoints_sequence) == 1:
        # 只有 1 帧，复制到所有帧
        return np.repeat(keypoints_sequence[0][np.newaxis, :, :], target_frames, axis=0)
    
    V, C = keypoints_sequence[0].shape
    valid_indices = np.arange(len(keypoints_sequence))
    target_indices = np.linspace(0, len(keypoints_sequence) - 1, target_frames)
    
    interpolated = np.zeros((target_frames, V, C), dtype=np.float32)
    
    for v in range(V):
        for c in range(C):
            values = np.array([kp[v, c] for kp in keypoints_sequence])
            f = interpolate.interp1d(valid_indices, values, kind='linear')
            interpolated[:, v, c] = f(target_indices)
    
    return interpolated


# ==================== 视频处理 ====================

def flip_left_hand_keypoints(keypoints_sequence):
    """
    以手腕为中心，将左手关键点的 X 坐标镜像反转
    
    Args:
        keypoints_sequence: list of (V, C) arrays, 关键点序列
    
    Returns:
        flipped_sequence: list of (V, C) arrays, 反转后的关键点序列
    """
    flipped_sequence = []
    for keypoints in keypoints_sequence:
        # 复制关键点数组
        flipped_kp = keypoints.copy()
        # 手腕是第 0 号关键点
        wrist_x = keypoints[0, 0]
        # 对所有关键点的 X 坐标以手腕为中心镜像
        flipped_kp[:, 0] = 2 * wrist_x - keypoints[:, 0]
        flipped_sequence.append(flipped_kp)
    return flipped_sequence


def process_video_folder(folder_path, estimator, target_frames, enable_interpolation):
    """
    处理单个视频文件夹（包含多帧图像），提取关键点
    
    Args:
        folder_path: 视频文件夹路径
        estimator: 姿态估计器
        target_frames: 目标帧数
        enable_interpolation: 是否启用补帧
    
    Returns:
        keypoints: (C, T, V) array 或 None
        valid_frames: 有效帧数量
        hand_info: 左右手统计信息字典
    """
    try:
        # 获取文件夹中的图像文件
        image_files = [f for f in os.listdir(folder_path) if f.endswith(('.jpg', '.jpeg', '.png', '.bmp'))]
        image_files.sort()
        
        if len(image_files) == 0:
            print(f"警告：文件夹为空 - {folder_path}")
            return None, 0, None
        
        keypoints_sequence = []
        hand_labels = []  # 记录每一帧的左右手信息
        failed_frames = 0
        
        # 第一步：提取所有帧的关键点和左右手信息
        for image_file in image_files:
            image_path = os.path.join(folder_path, image_file)
            
            # 尝试读取图像（处理中文路径）
            try:
                # 使用 PIL 读取图像，支持中文路径
                from PIL import Image
                
                # 使用 PIL 读取图像，支持中文路径
                pil_image = Image.open(image_path)
                frame = np.array(pil_image)
                # 如果是 RGB 格式，转换为 BGR（OpenCV 格式）
                if len(frame.shape) == 3 and frame.shape[2] == 3:
                    frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            except Exception as e:
                # 如果 PIL 失败，尝试 OpenCV
                frame = cv2.imread(image_path)
            
            if frame is None:
                failed_frames += 1
                continue
            
            # 提取关键点和左右手信息
            keypoints, hand_label = estimator.extract_keypoints(frame)
            if keypoints is not None:
                keypoints_sequence.append(keypoints)
                hand_labels.append(hand_label)
        
        valid_frames = len(keypoints_sequence)
        
        # 调试信息
        if valid_frames == 0:
            print(f"警告：文件夹 {folder_path} 无有效关键点")
            print(f"  总帧数：{len(image_files)}, 失败帧数：{failed_frames}")
            return None, 0, None
        
        # 第二步：统计左右手信息
        left_count = hand_labels.count('Left')
        right_count = hand_labels.count('Right')
        unknown_count = hand_labels.count(None)
        
        # 判断主要是哪只手（投票机制）
        dominant_hand = 'Right'  # 默认右手
        if left_count > right_count:
            dominant_hand = 'Left'
        
        hand_info = {
            'dominant_hand': dominant_hand,
            'left_count': left_count,
            'right_count': right_count,
            'unknown_count': unknown_count,
            'total_frames': valid_frames
        }
        
        # 第三步：如果是左手，进行坐标反转
        if dominant_hand == 'Left':
            keypoints_sequence = flip_left_hand_keypoints(keypoints_sequence)
        
        # 第四步：补帧或零填充
        if enable_interpolation:
            keypoints_array = interpolate_keypoints(keypoints_sequence, target_frames)
        else:
            # 零填充
            V, C = keypoints_sequence[0].shape
            keypoints_array = np.zeros((target_frames, V, C), dtype=np.float32)
            for i, kp in enumerate(keypoints_sequence):
                if i < target_frames:
                    keypoints_array[i] = kp
        
        if keypoints_array is None:
            return None, valid_frames, hand_info
        
        # 转换为 (C, T, V) 格式
        keypoints_array = keypoints_array.transpose(2, 0, 1)  # (T, V, C) -> (C, T, V)
        
        return keypoints_array, valid_frames, hand_info
        
    except Exception as e:
        print(f"处理文件夹失败 {folder_path}: {e}")
        import traceback
        traceback.print_exc()
        return None, 0, None


# ==================== 主流程 ====================

def process_dataset(config):
    """处理整个数据集"""
    # 创建输出目录
    output_dir = config['output_base_dir']
    if config['enable_interpolation']:
        output_dir += '_interpolated'
    os.makedirs(output_dir, exist_ok=True)
    
    # 初始化姿态估计器
    algorithm = config['algorithm']
    if algorithm == 'mediapipe':
        estimator = MediaPipeEstimator()
    elif algorithm == 'alphapose':
        estimator = AlphaPoseEstimator()
    elif algorithm == 'mmpose':
        estimator = MMPoseEstimator()
    else:
        raise ValueError(f"未知算法：{algorithm}")
    
    num_keypoints = estimator.get_num_keypoints()
    
    print("=" * 60)
    print("统一关键点提取工具")
    print("=" * 60)
    print(f"算法：{algorithm}")
    print(f"关键点数：{num_keypoints}")
    print(f"目标帧数：{config['target_frames']}")
    print(f"补帧：{'启用' if config['enable_interpolation'] else '禁用'}")
    print(f"有效帧阈值：{config['min_valid_frames']}")
    print(f"每个类别目标样本数：{config['target_samples_per_class']}")
    print(f"输出目录：{output_dir}")
    print("=" * 60)
    
    # 创建样本统计文件
    import csv
    stats_file = os.path.join(output_dir, 'data/dataset_statistics.csv')
    csv_file = open(stats_file, 'w', newline='', encoding='utf-8-sig')
    csv_writer = csv.writer(csv_file)
    csv_writer.writerow(['样本编号', '类别', '样本名称', '输入路径', '主要手部', '左手帧数', '右手帧数', '未知帧数', '总有效帧数', '数据集划分'])
    
    sample_counter = 0  # 全局样本计数器
    
    # 处理每个类别
    for class_idx, action_class in enumerate(config['action_classes']):
        print(f"\n处理类别 {class_idx + 1}: {action_class}...")
        
        # 获取类别文件夹
        class_dir = os.path.join(config['input_dir'], action_class)
        print(f"  类别目录：{class_dir}")
        
        if not os.path.exists(class_dir):
            print(f"  警告：类别目录不存在 - {class_dir}")
            continue
        
        # 获取所有视频文件夹
        try:
            video_folders = [f for f in os.listdir(class_dir) if os.path.isdir(os.path.join(class_dir, f))]
            print(f"  找到视频文件夹数量：{len(video_folders)}")
            
            if len(video_folders) == 0:
                print(f"  警告：类别 {action_class} 无视频文件夹")
                continue
                
        except Exception as e:
            print(f"  读取类别目录失败：{e}")
            continue
        
        # 随机打乱顺序
        random.seed(42)
        random.shuffle(video_folders)
        
        # 提取关键点
        all_keypoints = []
        all_hand_info = []  # 记录每个样本的左右手信息
        all_sample_names = []  # 记录样本名称
        all_sample_paths = []  # 记录样本路径
        total_samples = len(video_folders)
        discarded_samples = 0
        
        # 每个类别目标样本数
        target_samples_per_class = config['target_samples_per_class']
        
        for i, video_folder in enumerate(video_folders):
            # 如果已经收集到足够的样本，停止处理
            if len(all_keypoints) >= target_samples_per_class:
                break
                
            folder_path = os.path.join(class_dir, video_folder)
            
            # 处理视频文件夹
            keypoints, valid_frames, hand_info = process_video_folder(
                folder_path, estimator, config['target_frames'], config['enable_interpolation']
            )
            
            if keypoints is not None:
                if valid_frames >= config['min_valid_frames']:
                    all_keypoints.append(keypoints)
                    all_hand_info.append(hand_info)
                    all_sample_names.append(video_folder)
                    all_sample_paths.append(folder_path)
                else:
                    discarded_samples += 1
            
            if (i + 1) % 50 == 0:
                print(f"  进度：{i + 1}/{total_samples}")
        
        print(f"  总样本数：{total_samples}")
        print(f"  有效样本数：{len(all_keypoints)}")
        print(f"  丢弃样本数：{discarded_samples}")
        
        if len(all_keypoints) == 0:
            print(f"  警告：类别 {action_class} 无有效数据")
            continue
        
        # 确保每个类别有 250 个样本
        if len(all_keypoints) < target_samples_per_class:
            print(f"  警告：类别 {action_class} 有效样本不足，仅收集到 {len(all_keypoints)} 个")
        else:
            all_keypoints = all_keypoints[:target_samples_per_class]
            all_hand_info = all_hand_info[:target_samples_per_class]
            all_sample_names = all_sample_names[:target_samples_per_class]
            all_sample_paths = all_sample_paths[:target_samples_per_class]
            print(f"  已收集 {len(all_keypoints)} 个样本")
        
        # 分割数据集（训练：验证：测试 = 3：1：1）
        total_valid = len(all_keypoints)
        train_size = int(total_valid * 0.6)  # 3/5
        val_size = int(total_valid * 0.2)    # 1/5
        test_size = total_valid - train_size - val_size  # 1/5
        
        train_data = all_keypoints[:train_size]
        val_data = all_keypoints[train_size:train_size + val_size]
        test_data = all_keypoints[train_size + val_size:]
        
        # 保存为 npy
        splits = ['train', 'val', 'test']
        datasets = [train_data, val_data, test_data]
        
        for split_name, dataset in zip(splits, datasets):
            if len(dataset) > 0:
                dataset_array = np.array(dataset, dtype=np.float32)
                output_file = os.path.join(output_dir, f"{class_idx + 1}_{split_name}.npy")
                np.save(output_file, dataset_array)
                print(f"  {split_name}: 保存 {dataset_array.shape} -> {output_file}")
            else:
                print(f"  {split_name}: 无数据")
        
        # 写入统计信息到 CSV
        for idx, (sample_name, sample_path, hand_info) in enumerate(zip(all_sample_names, all_sample_paths, all_hand_info)):
            sample_counter += 1
            
            # 确定数据集划分
            if idx < train_size:
                split = 'train'
            elif idx < train_size + val_size:
                split = 'val'
            else:
                split = 'test'
            
            # 写入 CSV 行
            csv_writer.writerow([
                sample_counter,
                action_class,
                sample_name,
                sample_path,
                hand_info['dominant_hand'] if hand_info else 'Unknown',
                hand_info['left_count'] if hand_info else 0,
                hand_info['right_count'] if hand_info else 0,
                hand_info['unknown_count'] if hand_info else 0,
                hand_info['total_frames'] if hand_info else 0,
                split
            ])
    
    # 关闭 CSV 文件
    csv_file.close()
    print(f"\n样本统计文件已保存：{stats_file}")
    print(f"总样本数：{sample_counter}")
    
    # 清理
    if hasattr(estimator, 'close'):
        estimator.close()
    
    print("\n" + "=" * 60)
    print("处理完成!")
    print("=" * 60)


if __name__ == "__main__":
    process_dataset(CONFIG)

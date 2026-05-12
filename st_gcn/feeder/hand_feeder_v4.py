import os
import numpy as np
import torch
from torch.utils.data import Dataset


class HandFeeder(Dataset):
    """
    Feeder for hand gesture recognition dataset (v4 版)
    Input shape: (N, C, T, V, M=1) 或 (N, C, T, V) → 统一处理为 (N, C, T, V)
    
    新增特性：
        - 支持绝对坐标到相对坐标的归一化 (以第1帧的手腕为原点)
        - 解决不同位置做相同手势导致网络误判的问题（如把“向下滑动”认成“向上滑动”）

    Args:
        data_path: 数据路径 (如 'mediapipe_results' 或 'mmpose_results')
        mode: 'train' / 'val' / 'test'
        window_size: 时间窗口长度
        normalization: 是否进行空间和轨迹归一化（默认设为 True 以大幅提升准确率）
        debug: 是否只加载前10个样本用于快速测试
    """

    def __init__(self,
                 data_path,
                 mode='train',
                 window_size=35,
                 normalization=True,  # 默认开启归一化
                 debug=False):
        self.data_path = data_path
        self.mode = mode
        self.window_size = window_size
        self.normalization = normalization
        self.debug = debug

        self.load_data()

    def load_data(self):
        # 自动扫描所有 .npy 文件
        files = [f for f in os.listdir(self.data_path) if f.endswith('.npy')]

        # 构建样本列表和标签
        self.data = []
        self.labels = []

        for f in files:
            name, ext = os.path.splitext(f)
            label_str, phase = name.split('_')
            
            # 只加载对应阶段的数据
            if phase != self.mode:
                continue

            label = int(label_str) - 1  # 标签从 1~9 转为 0~8
            file_path = os.path.join(self.data_path, f)

            action_data = np.load(file_path)  # shape: (N, C, T, V) 或 (N, C, T, V, M)
            
            # 如果存在多人的 M 维度且 M=1，将其挤压掉
            if len(action_data.shape) == 5:
                action_data = action_data.squeeze(-1) # -> (N, C, T, V)

            if self.debug:
                action_data = action_data[:10]

            self.data.append(action_data)
            self.labels.extend([label] * len(action_data))

        if len(self.data) == 0:
            raise ValueError(f"在 {self.data_path} 目录下找不到 mode 为 {self.mode} 的数据！")

        self.data = np.concatenate(self.data, axis=0)
        self.labels = np.array(self.labels)

        # 输出维度 (N, C, T, V)
        self.N = len(self.labels)
        self.C = self.data.shape[1]  # 通常为3（x, y, z）
        self.T = self.data.shape[2]
        self.V = self.data.shape[3]

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        # 提取当前样本，并进行深度拷贝，防止修改原始数据矩阵
        data_numpy = np.array(self.data[index])  # shape: (C, T, V)
        label = self.labels[index]

        # =========================================================================
        # 核心优化：坐标系归一化 (消除手在画面中绝对位置的影响)
        # =========================================================================
        if self.normalization:
            # 获取第 1 帧 (t=0) 的第 0 号节点 (通常是 WRIST 手腕) 的坐标
            # shape: (C,)
            origin = data_numpy[:, 0, 0]
            
            # 扩展维度为 (C, 1, 1)，利用 numpy 广播机制对整个时空矩阵进行相减
            # 这样处理后，所有动作的起点都在 (0,0,0)，网络可以极其轻易地学习到“相对运动方向”
            data_numpy = data_numpy - origin[:, np.newaxis, np.newaxis]
            
            # (可选进阶) 尺度归一化：可以将坐标除以手部的基准长度（如手腕到中指根部的距离）
            # 避免不同人手掌大小不同带来的误差。目前简单的平移归一化已经足够解决方向混淆问题。

        # 返回形状：(C, T, V)
        return data_numpy, label

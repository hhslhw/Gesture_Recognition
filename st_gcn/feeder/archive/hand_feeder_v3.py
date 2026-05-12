import os
import numpy as np
import torch
from torch.utils.data import Dataset


class HandFeeder(Dataset):
    """
    Feeder for hand gesture recognition dataset (based on MediaPipe keypoints)
    Input shape: (N, C, T, V, M=1) → treated as (N, C, T, V)

    Args:
        data_path: 数据路径
        mode: train/test
        window_size: 时间窗口长度
        normalization: 是否归一化
        debug: 是否只加载前10个样本
    """

    def __init__(self,
                 data_path,
                 mode='train',
                 window_size=35,
                 normalization=False,
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
            if phase != self.mode:
                continue

            label = int(label_str) - 1  # 标签从0开始
            file_path = os.path.join(self.data_path, f)

            action_data = np.load(file_path)  # shape: (N, C, T, V, M=1)

            if self.debug:
                action_data = action_data[:10]

            self.data.append(action_data)
            self.labels.extend([label] * len(action_data))

        self.data = np.concatenate(self.data, axis=0)
        self.labels = np.array(self.labels)

        # 输出维度 (N, C, T, V)
        self.N = len(self.labels)
        self.C = self.data.shape[1]  # 现在C=3（x,y,z）
        self.T = self.data.shape[2]
        self.V = self.data.shape[3]

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        data_numpy = self.data[index]  # shape: (C, T, V)

        label = self.labels[index]

        # 返回形状：(C, T, V)
        return data_numpy, label

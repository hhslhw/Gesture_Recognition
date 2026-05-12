import os
import numpy as np
import torch
from torch.utils.data import Dataset


class HandFeeder(Dataset):
    """
    Feeder for hand gesture recognition dataset (based on MediaPipe keypoints)

    Input shape: (N, C, T, V, M)
        N: sample number
        C: channel (x, y, score)
        T: time frame
        V: joint number per hand (default: 21)
        M: person number (default: 2)
    """

    def __init__(self,
                 data_path,
                 mode='train',
                 window_size=35,
                 num_person=2,
                 normalization=False,
                 debug=False):
        self.data_path = data_path
        self.mode = mode
        self.window_size = window_size
        self.num_person = num_person
        self.normalization = normalization
        self.debug = debug

        self.load_data()

    def load_data(self):
        # 自动扫描所有 .npy 文件
        files = [f for f in os.listdir(self.data_path) if f.endswith('.npy')]
        label_set = set()

        # 构建样本列表和对应标签
        self.data = []
        self.labels = []

        for f in files:
            name, ext = os.path.splitext(f)
            label_str, phase = name.split('_')
            if phase != self.mode:
                continue

            label = int(label_str) - 1  # 转换为从0开始的标签
            file_path = os.path.join(self.data_path, f)

            action_data = np.load(file_path)  # shape: (N, C, T, V, M)
            if self.debug:
                action_data = action_data[:10]

            self.data.append(action_data)
            self.labels.extend([label] * len(action_data))  # 添加多个相同标签

        self.data = np.concatenate(self.data, axis=0)
        self.labels = np.array(self.labels)

        # 归一化处理（可选）
        if self.normalization:
            self.mean = self.data[:, :2].mean()
            self.std = self.data[:, :2].std()
            self.data[:, :2] = (self.data[:, :2] - self.mean) / (self.std + 1e-6)

        # 输出维度 (N, C, T, V, M)
        self.N = len(self.labels)
        self.C = self.data.shape[1]
        self.T = self.data.shape[2]
        self.V = self.data.shape[3]
        self.M = self.data.shape[4]

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        data_numpy = self.data[index]
        label = self.labels[index]

        # 裁剪时间窗口（如果需要）
        if self.window_size > 0 and self.T > self.window_size:
            start = np.random.randint(0, self.T - self.window_size + 1)
            data_numpy = data_numpy[:, start:start + self.window_size, :, :]

        return data_numpy, label

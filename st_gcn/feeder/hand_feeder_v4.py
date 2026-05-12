import os
import numpy as np
import torch
from torch.utils.data import Dataset


class HandFeeder(Dataset):
    """
    手势识别数据集加载器。

    支持以手腕为原点的相对坐标归一化，消除手部绝对位置对识别的干扰。
    数据文件命名格式：{class_id}_{mode}.npy，例如 1_train.npy、2_val.npy。

    Args:
        data_path: npy 数据目录路径
        mode: 数据集划分，'train' / 'val' / 'test'
        window_size: 时间窗口帧数
        normalization: 是否进行坐标归一化（以第0帧手腕为原点做平移）
        debug: 调试模式，每类只加载前10个样本
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
        # 深度拷贝，防止修改原始数据
        data_numpy = np.array(self.data[index])  # (C, T, V)
        label = self.labels[index]

        if self.normalization:
            # 以第0帧手腕关键点（节点0）为原点，对整个时空序列做平移归一化
            # 消除手部绝对位置的影响，使网络专注于学习相对运动方向
            origin = data_numpy[:, 0, 0]  # (C,)
            data_numpy = data_numpy - origin[:, np.newaxis, np.newaxis]

        return data_numpy, label

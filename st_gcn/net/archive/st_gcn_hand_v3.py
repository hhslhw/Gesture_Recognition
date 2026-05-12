import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np
import math

# 假设你已将 net.py 和 unit_gcn.py 放在同目录下
from .net import Unit2D, conv_init, import_class
from .unit_gcn import unit_gcn


# 手部骨架边定义（MediaPipe风格，右手+左手）
def build_hand_graph():
    # MediaPipe 手部 21 关节编号
    hand_joint_names = [
        'WRIST',
        'THUMB_CMC', 'THUMB_MCP', 'THUMB_IP', 'THUMB_TIP',
        'INDEX_FINGER_MCP', 'INDEX_FINGER_PIP', 'INDEX_FINGER_DIP', 'INDEX_FINGER_TIP',
        'MIDDLE_FINGER_MCP', 'MIDDLE_FINGER_PIP', 'MIDDLE_FINGER_DIP', 'MIDDLE_FINGER_TIP',
        'RING_FINGER_MCP', 'RING_FINGER_PIP', 'RING_FINGER_DIP', 'RING_FINGER_TIP',
        'PINKY_MCP', 'PINKY_PIP', 'PINKY_DIP', 'PINKY_TIP'
    ]

    # 右手连接关系（从近端到远端）
    right_hand_edges = [
        (0, 1), (1, 2), (2, 3), (3, 4),  # 大拇指
        (0, 5), (5, 6), (6, 7), (7, 8),  # 食指
        (0, 9), (9, 10), (10, 11), (11, 12),  # 中指
        (0, 13), (13, 14), (14, 15), (15, 16),  # 无名指
        (0, 17), (17, 18), (18, 19), (19, 20)  # 小指
    ]
    # 左手对应右手指序 + 21 偏移
    inward = right_hand_edges
    outward = [(j, i) for (i, j) in inward]
    neighbor = inward + outward
    self_link = [(i, i) for i in range(21)]  # 自环

    A = get_spatial_graph(num_node=21, self_link=self_link, inward=inward, outward=outward)
    return A


# 从 test_samples.py 抽取的图构造函数（简化版）
def edge2mat(link, num_node):
    A = np.zeros((num_node, num_node))
    for i, j in link:
        A[j, i] = 1
    return A

def normalize_digraph(A):
    Dl = np.sum(A, 0)
    num_node = A.shape[0]
    Dn = np.zeros((num_node, num_node))
    for i in range(num_node):
        if Dl[i] > 0:
            Dn[i, i] = Dl[i]**(-1)
    AD = np.dot(A, Dn)
    return AD

def get_spatial_graph(num_node, self_link, inward, outward):
    I = edge2mat(self_link, num_node)
    In = normalize_digraph(edge2mat(inward, num_node))
    Out = normalize_digraph(edge2mat(outward, num_node))
    A = np.stack((I, In, Out))
    return A


# 默认 backbone 配置（可调整）
default_backbone_hand = [
    (64, 64, 1),
    (64, 64, 1),
    (64, 128, 2),
    (128, 128, 1),
    (128, 256, 2),
    (256, 256, 1),
]


class Model(nn.Module):
    """
    Hand-based action recognition using Spatial Temporal Graph Convolutional Networks.

    Input shape: (N, C, T, V=21)
        N: batch size
        C: channel (x, y, score)
        T: time frames
        V: joint number per hand (21)
    """

    def __init__(self,
                 channel=3,  # 默认参数改为3
                 num_class=4,  # 根据你的实际类别数调整
                 window_size=35,
                 num_point=21,
                 use_data_bn=True,
                 backbone_config=None,
                 mask_learning=False,
                 use_local_bn=True,
                 multiscale=False,
                 temporal_kernel_size=9,
                 dropout=0.5):
        super(Model, self).__init__()

        # 输入形状检查
        assert channel == 3, "Expected 3 channels (x, y, z)"
        assert num_point == 21, "This model is designed for 21 hand joints only."

        # 构建手部图结构（21个关节）
        self.A = torch.from_numpy(build_hand_graph()).float()

        self.num_class = num_class
        self.use_data_bn = use_data_bn
        self.multiscale = multiscale

        # BatchNorm 设置
        if self.use_data_bn:
            self.data_bn = nn.BatchNorm1d(channel * num_point)

        kwargs = dict(
            A=self.A,
            mask_learning=mask_learning,
            use_local_bn=use_local_bn,
            dropout=dropout,
            kernel_size=temporal_kernel_size
        )

        if self.multiscale:
            unit = TCN_GCN_unit_multiscale
        else:
            unit = TCN_GCN_unit

        # Backbone 网络配置
        if backbone_config is None:
            backbone_config = default_backbone_hand
        self.backbone = nn.ModuleList([
            unit(in_c, out_c, stride=stride, **kwargs)
            for in_c, out_c, stride in backbone_config
        ])

        backbone_in_c = backbone_config[0][0]
        backbone_out_c = backbone_config[-1][1]

        # Head 层
        self.gcn0 = unit_gcn(
            channel,
            backbone_in_c,
            self.A,
            mask_learning=mask_learning,
            use_local_bn=use_local_bn
        )
        self.tcn0 = Unit2D(backbone_in_c, backbone_in_c, kernel_size=9)

        # Tail 分类层
        self.person_bn = nn.BatchNorm1d(backbone_out_c)
        self.fcn = nn.Conv1d(backbone_out_c, num_class, kernel_size=1)
        conv_init(self.fcn)

    def forward(self, x):
        N, C, T, V = x.size()

        if self.use_data_bn:
            x = x.view(N, C * V, T)
            x = self.data_bn(x)
            x = x.view(N, C, T, V)

        x = self.gcn0(x)
        x = self.tcn0(x)

        for m in self.backbone:
            x = m(x)

        # V pooling - joint average
        x = F.avg_pool2d(x, (1, V))
        x = x.squeeze(-1)  # shape: (B, C, T)

        # T pooling - time average
        T_current = x.shape[2]
        x = F.avg_pool1d(x, T_current)  # shape: (B, C, 1)

        # 分类头
        x = self.fcn(x)  # shape: (B, num_class, 1)
        x = x.squeeze(-1)  # shape: (B, num_class)

        return x


class TCN_GCN_unit(nn.Module):
    def __init__(self,
                 in_channel,
                 out_channel,
                 A,
                 kernel_size=9,
                 stride=1,
                 dropout=0.5,
                 use_local_bn=False,
                 mask_learning=False):
        super(TCN_GCN_unit, self).__init__()
        self.A = A
        self.V = A.size()[-1]
        self.C = in_channel

        self.gcn1 = unit_gcn(
            in_channel,
            out_channel,
            A,
            use_local_bn=use_local_bn,
            mask_learning=mask_learning
        )
        self.tcn1 = Unit2D(
            out_channel,
            out_channel,
            kernel_size=kernel_size,
            dropout=dropout,
            stride=stride
        )
        if (in_channel != out_channel) or (stride != 1):
            self.down1 = Unit2D(in_channel, out_channel, kernel_size=1, stride=stride)
        else:
            self.down1 = None

    def forward(self, x):
        x = self.tcn1(self.gcn1(x)) + (x if self.down1 is None else self.down1(x))
        return x


class TCN_GCN_unit_multiscale(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 A,
                 kernel_size=9,
                 stride=1,
                 **kwargs):
        super(TCN_GCN_unit_multiscale, self).__init__()
        half_out = out_channels // 2
        self.unit_1 = TCN_GCN_unit(in_channels, half_out, A, kernel_size=kernel_size, stride=stride, **kwargs)
        self.unit_2 = TCN_GCN_unit(in_channels, out_channels - half_out, A, kernel_size=kernel_size * 2 - 1, stride=stride, **kwargs)

    def forward(self, x):
        return torch.cat((self.unit_1(x), self.unit_2(x)), dim=1)


# 权重初始化工具函数
def conv_init(module):
    n = module.out_channels
    for k in module.kernel_size:
        n *= k
    module.weight.data.normal_(0, math.sqrt(2. / n))
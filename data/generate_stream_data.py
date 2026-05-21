import os
import numpy as np

def generate_stream_data(source_dir, bone_dir, motion_dir):
    """
    根据原始的关节流 (Joint Stream) 数据，生成骨骼流 (Bone Stream) 和运动流 (Motion Stream) 数据。
    """
    # 如果输出文件夹不存在，则创建
    os.makedirs(bone_dir, exist_ok=True)
    os.makedirs(motion_dir, exist_ok=True)

    # 1. 定义 MediaPipe 21个关键点的父节点连接关系
    # 索引 i 的父节点是 parents[i]
    parents = [
        0,  # 0: WRIST (根节点，父节点算作自己)
        0, 1, 2, 3,  # 1-4: 大拇指 (THUMB)
        0, 5, 6, 7,  # 5-8: 食指 (INDEX_FINGER)
        0, 9, 10, 11,  # 9-12: 中指 (MIDDLE_FINGER)
        0, 13, 14, 15, # 13-16: 无名指 (RING_FINGER)
        0, 17, 18, 19  # 17-20: 小指 (PINKY)
    ]

    # 获取所有的 .npy 文件
    files = [f for f in os.listdir(source_dir) if f.endswith('.npy')]

    if len(files) == 0:
        print(f"❌ 错误: 在 {source_dir} 中找不到 .npy 文件！")
        return

    print("=" * 60)
    print("开始生成双流网络训练数据...")
    print("=" * 60)

    for f in files:
        source_path = os.path.join(source_dir, f)
        
        # 读取原始关节数据, 形状为 (N, C, T, V)
        joint_data = np.load(source_path)
        
        N, C, T, V = joint_data.shape
        assert V == 21, "关键点数量必须是 21！"

        # ========================================================
        # 生成 骨骼流 (Bone Stream) 数据
        # 定义：当前节点坐标 - 父节点坐标 (表征手指弯曲程度和骨骼方向)
        # ========================================================
        bone_data = np.zeros_like(joint_data)
        for v in range(V):
            # 第 v 个节点的骨骼向量 = 第 v 个节点的坐标 - 它父节点的坐标
            bone_data[:, :, :, v] = joint_data[:, :, :, v] - joint_data[:, :, :, parents[v]]
        
        # 根节点(手腕)没有父节点，骨骼向量设为0
        # bone_data[:, :, :, 0] 已经是 0 了，因为 joint_data[0] - joint_data[0] = 0

        bone_out_path = os.path.join(bone_dir, f)
        np.save(bone_out_path, bone_data)

        # ========================================================
        # 生成 运动流 (Motion Stream) 数据
        # 定义：当前帧坐标 - 上一帧坐标 (表征运动速度和绝对方向，极大改善上下滑动混淆)
        # ========================================================
        motion_data = np.zeros_like(joint_data)
        # 从第 1 帧开始，等于当前帧减去前一帧
        motion_data[:, :, 1:, :] = joint_data[:, :, 1:, :] - joint_data[:, :, :-1, :]
        # 第 0 帧的速度默认为 0，保持不变即可

        motion_out_path = os.path.join(motion_dir, f)
        np.save(motion_out_path, motion_data)

        print(f"✓ 已处理: {f} -> 生成了 {N} 个样本的 Bone 和 Motion 特征。")

    print("-" * 60)
    print("🎉 双流数据生成完毕！")
    print(f"骨骼流数据保存在: {bone_dir}")
    print(f"运动流数据保存在: {motion_dir}")
    print("\n【如何训练双流网络？】")
    print("1. 训练骨骼流：把 train.py 里的 args['data_path'] 改为 'mediapipe_bone_results'，运行训练得到 bone_best.pth")
    print("2. 训练运动流：把 train.py 里的 args['data_path'] 改为 'mediapipe_motion_results'，运行训练得到 motion_best.pth")
    print("3. 推理时：将关节流、骨骼流、运动流送入各自的模型，把输出的 softmax 分数 1:1 相加即可！")

if __name__ == "__main__":
    # 配置你的原始数据文件夹
    SOURCE_DIR = "mediapipe_results_2"
    # 输出的骨骼流文件夹
    BONE_DIR = "mediapipe_bone_results_2"
    # 输出的运动流文件夹
    MOTION_DIR = "mediapipe_motion_results_2"

    generate_stream_data(SOURCE_DIR, BONE_DIR, MOTION_DIR)

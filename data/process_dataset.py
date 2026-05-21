import os

# 原始数据路径
DATASET_PATH = "../"  # 当前目录下的四个文件夹
ACTIONS = ["单手放大", "单手缩小", "双指向上滑动", "双指向下滑动","双指向右滑动","双指向左滑动","双指拉近","双指放大","拉手靠近"]  #123456789 动作类别

# 输出路径
PREPROCESSED_PATH = "./preprocessed_dataset_v2"

# 抽样配置
SAMPLES_PER_ACTION = 2500  # 每个动作总样本数 (100训练 + 25测试)
FRAMES_PER_VIDEO = 35  # 目标帧数


def create_directory_structure():
    """创建预处理后的数据目录结构"""
    if not os.path.exists(PREPROCESSED_PATH):
        os.makedirs(PREPROCESSED_PATH)

    for action in ACTIONS:
        action_path = os.path.join(PREPROCESSED_PATH, action)
        if not os.path.exists(action_path):
            os.makedirs(action_path)

        # 创建训练和测试子目录
        for subdir in ['train', 'test']:
            path = os.path.join(action_path, subdir)
            if not os.path.exists(path):
                os.makedirs(path)


import random
import shutil


def select_samples():
    """从每个动作中随机选择样本"""
    for action in ACTIONS:
        print(f"正在处理动作: {action}")
        action_folder = os.path.join(DATASET_PATH, action)
        subfolders = [f for f in os.listdir(action_folder) if os.path.isdir(os.path.join(action_folder, f))]

        # 随机选择样本
        selected_subfolders = random.sample(subfolders, SAMPLES_PER_ACTION)
        train_subfolders = selected_subfolders[:2000]
        test_subfolders = selected_subfolders[2000:]

        # 处理训练样本
        print(f"处理训练样本: {len(train_subfolders)} 个")
        process_and_copy_samples(action, train_subfolders, 'train')

        # 处理测试样本
        print(f"处理测试样本: {len(test_subfolders)} 个")
        process_and_copy_samples(action, test_subfolders, 'test')


def process_and_copy_samples(action, subfolders, subset):
    """处理并复制样本到新目录，调整帧数"""
    for idx, subfolder in enumerate(subfolders):
        src_path = os.path.join(DATASET_PATH, action, subfolder)
        dst_path = os.path.join(PREPROCESSED_PATH, action, subset, subfolder)

        # 创建目标文件夹
        if not os.path.exists(dst_path):
            os.makedirs(dst_path)

        # 获取所有图像文件
        image_files = sorted([f for f in os.listdir(src_path) if f.endswith(('.png', '.jpg', '.jpeg'))])

        # 调整帧数到目标数量
        adjusted_files = adjust_frame_count(image_files, FRAMES_PER_VIDEO)

        # 复制选定的帧
        for new_idx, orig_idx in enumerate(adjusted_files):
            src_file = os.path.join(src_path, image_files[orig_idx])
            dst_file = os.path.join(dst_path, f"{new_idx:03d}{os.path.splitext(image_files[orig_idx])[1]}")
            shutil.copy2(src_file, dst_file)


def adjust_frame_count(image_files, target_frames):
    """调整帧数到目标数量"""
    num_frames = len(image_files)

    if num_frames >= target_frames:
        # 如果帧数足够或过多，均匀采样到目标帧数
        step = num_frames / target_frames
        return [int(i * step) for i in range(target_frames)]
    else:
        # 如果帧数不足，直接返回所有帧（后续在转换为.npy时补齐）
        return list(range(num_frames))
if __name__ == "__main__":
    print("开始数据预处理...")
    create_directory_structure()
    select_samples()
    print("数据预处理完成！")

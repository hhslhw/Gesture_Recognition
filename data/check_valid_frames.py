import numpy as np
import os

def check_valid_frames(npy_file, num_samples=100):
    """
    检查npy文件中前N个样本的有效帧数量
    
    Args:
        npy_file: npy文件路径
        num_samples: 要检查的样本数量
    """
    # 加载数据
    data = np.load(npy_file)
    print(f"文件: {os.path.basename(npy_file)}")
    print(f"数据形状: {data.shape}")
    print(f"数据格式: (样本数, 通道数, 帧数, 关键点数)")
    print("-" * 60)
    
    # 限制样本数量
    actual_samples = min(num_samples, data.shape[0])
    
    # 统计每个样本的有效帧数
    for i in range(actual_samples):
        sample = data[i]  # shape: (C, T, V)
        
        # 检查每一帧是否全为0
        valid_frames = 0
        for t in range(sample.shape[1]):  # 遍历时间维度
            frame = sample[:, t, :]  # shape: (C, V)
            if not np.all(frame == 0):
                valid_frames += 1
        
        print(f"样本 {i+1:3d}: 有效帧数 = {valid_frames:2d} / {sample.shape[1]}")
    
    print("-" * 60)
    print(f"已检查 {actual_samples} 个样本")


if __name__ == "__main__":
    # 指定要检查的文件
    npy_dir = "output_keypoints_v6"
    
    # 列出所有npy文件
    npy_files = [f for f in os.listdir(npy_dir) if f.endswith('.npy')]
    npy_files.sort()
    
    print("可用的npy文件:")
    for idx, f in enumerate(npy_files, 1):
        print(f"{idx}. {f}")
    
    # 选择文件
    choice = input("\n请输入文件编号 (或直接输入文件名): ").strip()
    
    if choice.isdigit() and 1 <= int(choice) <= len(npy_files):
        selected_file = npy_files[int(choice) - 1]
    else:
        selected_file = choice if choice.endswith('.npy') else f"{choice}.npy"
    
    file_path = os.path.join(npy_dir, selected_file)
    
    if os.path.exists(file_path):
        print(f"\n正在检查: {file_path}\n")
        check_valid_frames(file_path, num_samples=100)
    else:
        print(f"错误: 文件不存在 - {file_path}")

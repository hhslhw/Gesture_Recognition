"""
MediaPipe 手部关键点检测简单示例
对单张图片进行推理并保存结果
"""
import cv2
import numpy as np
import matplotlib.pyplot as plt
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

# ==================== 配置 ====================
IMAGE_PATH = '00026.jpg'  # 测试图片路径
OUTPUT_PATH = 'output_keypoints_mediapipe.png'  # 输出图片路径

# ==================== 初始化模型 ====================
print("=" * 60)
print("MediaPipe 手部关键点检测示例")
print("=" * 60)

print("\n1. 初始化 MediaPipe 手部模型...")
try:
    # 配置选项
    options = vision.HandLandmarkerOptions(
        base_options=python.BaseOptions(
            model_asset_path='../models/hand_landmarker.task'  # 需要下载 .task 模型文件
        ),
        running_mode=vision.RunningMode.IMAGE,  # 图片模式（注意参数名是 running_mode）
        num_hands=2,  # 最多检测 2 只手
        min_hand_detection_confidence=0.3,  # 降低检测阈值
        min_hand_presence_confidence=0.3,   # 降低存在阈值
        min_tracking_confidence=0.3         # 降低追踪阈值
    )
    
    # 创建检测器实例
    landmarker = vision.HandLandmarker.create_from_options(options)
    print("✓ MediaPipe 模型初始化成功")
    print(f"  运行模式：IMAGE")
    print(f"  最大检测手数：2")
    print(f"  检测置信度阈值：0.3")
except Exception as e:
    print(f"✗ 模型初始化失败：{e}")
    print("\n请确保已安装 MediaPipe:")
    print("  pip install mediapipe")
    print("\n并下载 hand_landmarker.task 模型文件:")
    print("  https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task")
    exit()

# ==================== 读取图片 ====================
print(f"\n2. 读取图片：{IMAGE_PATH}")
try:
    image = cv2.imread(IMAGE_PATH)
    if image is None:
        print(f"✗ 无法读取图片：{IMAGE_PATH}")
        print("请确保图片存在且路径正确")
        exit()
    print(f"✓ 图片读取成功，尺寸：{image.shape}")
    # 转换为 RGB 格式（MediaPipe 需要 RGB）
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
except Exception as e:
    print(f"✗ 读取图片失败：{e}")
    exit()

# ==================== 执行推理 ====================
print("\n3. 执行手部关键点检测...")
try:
    # 执行检测（需要指定图像格式）
    # 方法 1：使用 mp.Image 对象（推荐）
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=image_rgb)
    detection_result = landmarker.detect(mp_image)
    print("✓ 推理完成")
except Exception as e:
    print(f"✗ 推理失败：{e}")
    import traceback
    traceback.print_exc()
    exit()

# ==================== 处理结果 ====================
print("\n4. 处理检测结果...")

# 打印原始结果信息
print(f"  检测结果类型：{type(detection_result)}")
print(f"  hand_landmarks: {detection_result.hand_landmarks}")
print(f"  handedness: {detection_result.handedness}")

# 检查是否有检测结果
if not detection_result.hand_landmarks:
    print("\n✗ 未检测到手部")
    print("\n可能的原因:")
    print("  1. 图片尺寸太小（当前：100x176）")
    print("  2. 手部在图片中占比太小")
    print("  3. 手部姿势不清晰或光线问题")
    print("  4. 手部不在图片中心区域")
    print("\n建议:")
    print("  - 使用更大尺寸的图片（建议 640x480 以上）")
    print("  - 确保手部占据图片的主要区域")
    print("  - 确保手部姿势清晰，光线充足")
    print("  - 尝试其他包含清晰手部的图片")
    exit()

print(f"✓ 检测到 {len(detection_result.hand_landmarks)} 只手")

# 提取关键点
all_hands_keypoints = []
for i, hand_landmarks in enumerate(detection_result.hand_landmarks):
    print(f"\n  手 {i + 1}:")
    print(f"    关键点数量：{len(hand_landmarks)}")
    
    # 提取 21 个关键点坐标
    keypoints = []
    for landmark in hand_landmarks:
        # MediaPipe 输出的是归一化坐标 [0, 1]
        # 需要转换到像素坐标
        x = landmark.x * image.shape[1]
        y = landmark.y * image.shape[0]
        z = landmark.z  # 相对深度值
        keypoints.append([x, y, z])
    
    keypoints = np.array(keypoints)  # (21, 3)
    all_hands_keypoints.append(keypoints)
    
    print(f"    关键点形状：{keypoints.shape}")
    print(f"    坐标范围:")
    print(f"      X: {keypoints[:, 0].min():.2f} ~ {keypoints[:, 0].max():.2f}")
    print(f"      Y: {keypoints[:, 1].min():.2f} ~ {keypoints[:, 1].max():.2f}")
    print(f"      Z: {keypoints[:, 2].min():.2f} ~ {keypoints[:, 2].max():.2f}")

# 获取第一只手的关键点用于可视化
keypoints = all_hands_keypoints[0]

# 检查左右手信息
if detection_result.handedness:
    for i, handedness in enumerate(detection_result.handedness):
        if handedness:
            hand_label = handedness[0].display_name
            confidence = handedness[0].score
            print(f"\n  手 {i + 1}: {hand_label} (置信度：{confidence:.2f})")

# ==================== 可视化结果 ====================
print(f"\n5. 可视化并保存结果到：{OUTPUT_PATH}")

# 创建画布
fig, ax = plt.subplots(1, 2, figsize=(16, 8))

# 显示原图
ax[0].imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
ax[0].set_title('Original Image', fontsize=14)
ax[0].axis('off')

# 显示带关键点的图像
ax[1].imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
ax[1].set_title('Hand Keypoints Detection (MediaPipe)', fontsize=14)

# 绘制所有检测到的手
for hand_idx, hand_keypoints in enumerate(all_hands_keypoints):
    # 为不同的手选择不同的颜色
    base_color = 'red' if hand_idx == 0 else 'blue'
    
    # 绘制关键点
    for i, (x, y) in enumerate(hand_keypoints[:, :2]):
        # 绘制点
        ax[1].plot(x, y, 'o', color=base_color, markersize=8)
        # 绘制编号
        ax[1].annotate(str(i), (x, y), color='yellow', fontsize=8, 
                      bbox=dict(boxstyle='round,pad=0.3', facecolor='black', alpha=0.7))
    
    # 连接手部骨架（MediaPipe 手部连接顺序）
    connections = [
        (0, 1), (1, 2), (2, 3), (3, 4),  # 拇指
        (0, 5), (5, 6), (6, 7), (7, 8),  # 食指
        (0, 9), (9, 10), (10, 11), (11, 12),  # 中指
        (0, 13), (13, 14), (14, 15), (15, 16),  # 无名指
        (0, 17), (17, 18), (18, 19), (19, 20),  # 小指
        (0, 5), (0, 9), (0, 13), (0, 17)  # 手掌连接
    ]
    
    for i, j in connections:
        ax[1].plot([hand_keypoints[i, 0], hand_keypoints[j, 0]], 
                  [hand_keypoints[i, 1], hand_keypoints[j, 1]], 
                  color=base_color, linewidth=2, alpha=0.7)

ax[1].axis('off')

# 保存图片
plt.tight_layout()
plt.savefig(OUTPUT_PATH, dpi=300, bbox_inches='tight')
print(f"✓ 结果已保存到：{OUTPUT_PATH}")

# ==================== 保存关键点数据 ====================
KEYPOINTS_NPY_PATH = 'keypoints_mediapipe.npy'
# 保存所有手的关键点
np.save(KEYPOINTS_NPY_PATH, np.array(all_hands_keypoints, dtype=object))
print(f"✓ 关键点数据已保存到：{KEYPOINTS_NPY_PATH}")
print(f"  检测到的手数：{len(all_hands_keypoints)}")
print(f"  每只手关键点形状：{keypoints.shape}")
print(f"  数据类型：{keypoints.dtype}")

print("\n" + "=" * 60)
print("处理完成！")
print("=" * 60)
print(f"\n输出文件:")
print(f"  1. 可视化结果：{OUTPUT_PATH}")
print(f"  2. 关键点数据：{KEYPOINTS_NPY_PATH}")
print(f"\n关键点数据说明:")
print(f"  - 形状：(21, 3)")
print(f"  - 每行代表一个关键点")
print(f"  - 列：[x 像素坐标，y 像素坐标，z 相对深度]")
print(f"  - 注意：z 值是相对深度，不是真实深度")

# 清理资源
landmarker.close()

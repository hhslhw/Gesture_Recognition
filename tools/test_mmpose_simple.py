"""
MMPose 手部关键点检测简单示例
对单张图片进行推理并保存结果
"""
import cv2
import numpy as np
import matplotlib.pyplot as plt
from mmpose.apis import MMPoseInferencer

# ==================== 配置 ====================
IMAGE_PATH = 'test_hand.jpg'  # 测试图片路径
OUTPUT_PATH = 'output_keypoints.png'  # 输出图片路径

# ==================== 初始化模型 ====================
print("=" * 60)
print("MMPose 手部关键点检测示例")
print("=" * 60)

print("\n1. 初始化 MMPose 手部模型...")
try:
    # 修改点：不传 model 参数，而是通过 pose2d 参数指定模型别名
    # 'hand' 是 MMPose 内置的手部 2D 模型别名 (基于 RTMPose)
    inferencer = MMPoseInferencer(pose2d='hand') 
    print("✓ 模型初始化成功")
except Exception as e:
    print(f"✗ 模型初始化失败：{e}")
    print("\n建议检查 MMPose 版本或尝试重装:")
    print("  pip install mmpose")
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
except Exception as e:
    print(f"✗ 读取图片失败：{e}")
    exit()

# ==================== 执行推理 ====================
print("\n3. 执行手部关键点检测...")
try:
    # 执行推理
    result_generator = inferencer(image, show=False)
    results = next(result_generator)
    print("✓ 推理完成")
except Exception as e:
    print(f"✗ 推理失败：{e}")
    import traceback
    traceback.print_exc()
    exit()

# ==================== 处理结果 ====================
print("\n4. 处理检测结果...")
if 'predictions' in results:
    predictions = results['predictions']
    
    if len(predictions) > 0:
        pred = predictions[0]
        
        if len(pred) > 0:
            hand_data = pred[0]
            
            if 'keypoints' in hand_data:
                keypoints = hand_data['keypoints']
                
                # 如果是 list，转换为 numpy 数组
                if isinstance(keypoints, list):
                    keypoints = np.array(keypoints)
                
                print(f"✓ 检测到 {len(keypoints)} 个手部关键点")
                print(f"  关键点形状：{keypoints.shape}")
                print(f"  关键点坐标范围:")
                print(f"    X: {keypoints[:, 0].min():.2f} ~ {keypoints[:, 0].max():.2f}")
                print(f"    Y: {keypoints[:, 1].min():.2f} ~ {keypoints[:, 1].max():.2f}")
            else:
                print("✗ 结果中没有 keypoints 字段")
                exit()
        else:
            print("✗ 未检测到手部")
            exit()
    else:
        print("✗ 未检测到手部")
        exit()
else:
    print("✗ 结果格式异常")
    exit()

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
ax[1].set_title('Hand Keypoints Detection', fontsize=14)

# 绘制关键点
for i, (x, y) in enumerate(keypoints[:, :2]):
    # 绘制点
    ax[1].plot(x, y, 'ro', markersize=8)
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
    ax[1].plot([keypoints[i, 0], keypoints[j, 0]], 
              [keypoints[i, 1], keypoints[j, 1]], 
              'b-', linewidth=2, alpha=0.7)

ax[1].axis('off')

# 保存图片
plt.tight_layout()
plt.savefig(OUTPUT_PATH, dpi=300, bbox_inches='tight')
print(f"✓ 结果已保存到：{OUTPUT_PATH}")

# ==================== 保存关键点数据 ====================
KEYPOINTS_NPY_PATH = 'keypoints.npy'
np.save(KEYPOINTS_NPY_PATH, keypoints)
print(f"✓ 关键点数据已保存到：{KEYPOINTS_NPY_PATH}")
print(f"  数据形状：{keypoints.shape}")
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
print(f"  - 列：[x 坐标，y 坐标，z 坐标]")
print(f"  - 注意：z 坐标已置为 0（2D 检测不支持深度）")

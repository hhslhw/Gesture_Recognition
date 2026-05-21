import os
from tqdm import tqdm
import numpy as np
import cv2
import mediapipe as mp

# -----------------------------
# 可配置参数
# -----------------------------
DATASET_PATH = r"E:\work\gesture_recognition\preprocessed_dataset_v3"
ACTIONS = ["1", "2", "3", "4","5", "6", "7", "8","9"]  # 动作类别（可扩展）
OUTPUT_PATH = r"E:\work\gesture_recognition\output_keypoints_v6"
VISUALIZATION_PATH = r"E:\work\gesture_recognition\visualizations_v6"

# 数据维度参数
MAX_TRAIN_SAMPLES = 1000  # 每类最大训练样本数
MAX_TEST_SAMPLES = 200  # 每类最大测试样本数
FRAMES_PER_VIDEO = 35  # 每视频帧数
NUM_KEYPOINTS = 21  # 每手关键点数
COORD_DIM = 3  # 坐标维度 (x, y, z)
MAX_HANDS = 2  # 每帧最大检测手数

# -----------------------------
# 初始化 MediaPipe Hands 模型
# -----------------------------
mp_hands = mp.solutions.hands
# 修改后的Hands初始化代码（5.py第27行）
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=MAX_HANDS,
    model_complexity=1,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5,
)



# -----------------------------
# 提取关键点函数（使用世界坐标）
# -----------------------------
def get_best_hand_keypoints(results):
    """
    获取最佳手的关键点，使用世界坐标(x, y, z)
    返回: (V, C) 数组，其中C=(x,y,z)
    """
    if not results.multi_hand_world_landmarks or not results.multi_handedness:
        return None

    # 选择置信度最高的手
    best_idx = max(
        range(len(results.multi_handedness)),
        key=lambda i: results.multi_handedness[i].classification[0].score
    )

    best_hand = results.multi_hand_world_landmarks[best_idx]

    # 构造关键点数组：[x, y, z]
    keypoints = np.zeros((NUM_KEYPOINTS, COORD_DIM))
    for i, landmark in enumerate(best_hand.landmark):
        keypoints[i, :] = [landmark.x, landmark.y, landmark.z]

    return keypoints


# -----------------------------
# 可视化函数（使用图像坐标）
# -----------------------------
def visualize_and_save_video(folder_path, output_video_path):
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video_writer = None

    image_files = sorted(f for f in os.listdir(folder_path) if f.endswith(('.png', '.jpg', '.jpeg')))

    for image_file in image_files:
        image = cv2.imread(os.path.join(folder_path, image_file))
        results = hands.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

        annotated_image = image.copy()
        if results.multi_hand_landmarks:  # 使用图像坐标进行可视化
            for hand_landmarks in results.multi_hand_landmarks:
                mp.solutions.drawing_utils.draw_landmarks(
                    annotated_image, hand_landmarks, mp_hands.HAND_CONNECTIONS
                )

        if video_writer is None:
            video_writer = cv2.VideoWriter(
                output_video_path, fourcc, 30.0, (image.shape[1], image.shape[0])
            )
        video_writer.write(annotated_image)

    if video_writer:
        video_writer.release()


# -----------------------------
# 数据处理核心函数
# -----------------------------
def process_data(data_folder, max_samples):
    """统一处理训练/测试数据"""
    samples = []
    pbar = tqdm(total=max_samples, desc="样本处理")

    for sample in os.listdir(data_folder):
        if len(samples) >= max_samples or not os.path.isdir(os.path.join(data_folder, sample)):
            continue

        folder_path = os.path.join(data_folder, sample)
        image_files = sorted(f for f in os.listdir(folder_path) if f.endswith(('.png', '.jpg', '.jpeg')))

        valid_frames = 0
        sample_data = np.zeros((COORD_DIM, FRAMES_PER_VIDEO, NUM_KEYPOINTS))

        for frame_idx, image_file in enumerate(image_files[:FRAMES_PER_VIDEO]):
            image = cv2.imread(os.path.join(folder_path, image_file))
            if image is None:
                continue

            results = hands.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
            keypoints = get_best_hand_keypoints(results)

            if keypoints is None:
                continue

            sample_data[:, frame_idx, :] = keypoints.T
            valid_frames += 1

        if valid_frames >= 5:
            samples.append(sample_data)
            pbar.update(1)

    pbar.close()
    return np.array(samples)


# -----------------------------
# 主程序逻辑
# -----------------------------
print("开始关键点检测和可视化...")
# 确保输出目录存在
os.makedirs(OUTPUT_PATH, exist_ok=True)

for action in ACTIONS:
    print(f"\n正在处理动作: {action}")
    action_vis_path = os.path.join(VISUALIZATION_PATH, action)
    os.makedirs(action_vis_path, exist_ok=True)

    # 处理训练数据
    train_folder = os.path.join(DATASET_PATH, action, 'train')
    train_data = process_data(train_folder, MAX_TRAIN_SAMPLES)
    np.save(os.path.join(OUTPUT_PATH, f"{action}_train.npy"), train_data)

    # 保存训练可视化
    vis_train_path = os.path.join(action_vis_path, 'train')
    os.makedirs(vis_train_path, exist_ok=True)
    for sample in os.listdir(train_folder):
        if os.path.isdir(os.path.join(train_folder, sample)):
            visualize_and_save_video(
                os.path.join(train_folder, sample),
                os.path.join(vis_train_path, f"{sample}.mp4")
            )

    # 处理测试数据
    test_folder = os.path.join(DATASET_PATH, action, 'test')
    test_data = process_data(test_folder, MAX_TEST_SAMPLES)
    np.save(os.path.join(OUTPUT_PATH, f"{action}_test.npy"), test_data)

    # 保存测试可视化
    """vis_test_path = os.path.join(action_vis_path, 'test')
    os.makedirs(vis_test_path, exist_ok=True)
    for sample in os.listdir(test_folder):
        if os.path.isdir(os.path.join(test_folder, sample)):
            visualize_and_save_video(
                os.path.join(test_folder, sample),
                os.path.join(vis_test_path, f"{sample}.mp4")
            )"""

print("所有操作完成！")
hands.close()

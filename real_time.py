import cv2
import mediapipe as mp
import numpy as np
import torch
import time
from PIL import Image, ImageDraw, ImageFont
from st_gcn.net.st_gcn_hand_v5 import Model

# ---------------------- 配置区域 ----------------------
ACTIONS = ["单手放大", "单手缩小", "双指向上滑动", "双指向下滑动",
           "双指向右滑动", "双指向左滑动", "双指拉近", "双指放大", "拉手靠近"]
NUM_CLASSES = 9
WINDOW_SIZE = 35
MODEL_PATH = "models/best_model_9.pth"
# 请确保字体路径正确
FONT_PATH = "C:/Windows/Fonts/msyh.ttc"

# 阈值设置
CONFIDENCE_THRESHOLD = 0.85
ACTION_COOLDOWN = 2.0  # 动作识别后（或未识别）的冷却时间
STATIC_THRESHOLD = 1.0  # 判定为静止所需的时间
MOVEMENT_THRESHOLD = 0.015  # 移动判定阈值
PREPARE_TIME = 2.0  # 预备倒计时时长


# ---------------------- UI绘制函数 ----------------------
def draw_ui(img, status_info, history_log, width):
    img_pil = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(img_pil)

    try:
        font_status = ImageFont.truetype(FONT_PATH, 18)
        font_history = ImageFont.truetype(FONT_PATH, 20)
    except:
        font_status = ImageFont.load_default()
        font_history = ImageFont.load_default()

    # 1. 左上角状态
    draw.text((5, 5), status_info["text"], font=font_status, fill=status_info["color"])

    # 2. 右上角历史记录
    start_x = width - 220
    start_y = 5
    draw.text((start_x, start_y), "【识别记录】", font=font_history, fill=(255, 255, 255))

    for i, record in enumerate(reversed(history_log)):
        text = f"{record['action']} {record['conf']}"
        # 如果是 N/A (未识别)，显示灰色，否则显示绿色
        color = (0, 255, 0) if record['conf'] != 'N/A' else (200, 200, 200)
        draw.text((start_x, start_y + 25 + (i * 25)), text, font=font_history, fill=color)

    return cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)


# ---------------------- 模型加载 ----------------------
def load_model():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = Model(
        channel=3, num_class=NUM_CLASSES, window_size=WINDOW_SIZE,
        multiscale=True, use_attention=True, use_data_bn=True
    ).to(device)
    try:
        checkpoint = torch.load(MODEL_PATH, map_location=device)
        model.load_state_dict(checkpoint, strict=True)
    except Exception as e:
        print(f"模型加载失败: {e}")
        exit()
    model.eval()
    return model, device


# ---------------------- 数据缓冲 ----------------------
class DataProcessor:
    def __init__(self, window_size=35):
        self.window_size = window_size
        self.buffer = np.zeros((window_size, 21, 3))
        self.frame_count = 0

    def update_buffer(self, new_frame):
        self.buffer = np.roll(self.buffer, shift=-1, axis=0)
        self.buffer[-1] = new_frame
        self.frame_count += 1

    def clear(self):
        self.buffer = np.zeros((self.window_size, 21, 3))
        self.frame_count = 0

    def get_model_input(self):
        tensor_data = torch.from_numpy(self.buffer.transpose(2, 0, 1))
        return tensor_data.unsqueeze(0).to(torch.float32).contiguous()

    # ---------------------- 主程序 ----------------------


if __name__ == "__main__":
    model, device = load_model()
    processor = DataProcessor(window_size=WINDOW_SIZE)

    mp_hands = mp.solutions.hands.Hands(
        static_image_mode=False, max_num_hands=1,
        min_detection_confidence=0.5, min_tracking_confidence=0.5
    )
    mp_draw = mp.solutions.drawing_utils

    cap = cv2.VideoCapture(0)

    # --- 核心状态控制 ---
    # 状态枚举：0=静止监测, 1=预备倒计时, 2=正在检测, 3=冷却期
    current_state = 0

    state_start_time = time.time()  # 当前状态开始的时间
    last_move_time = time.time()  # 上次手移动的时间（用于静止监测）
    prev_hand_norm = None  # 上一帧坐标

    history_log = []

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret: break

        frame = cv2.flip(frame, 1)
        h, w, _ = frame.shape
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = mp_hands.process(rgb_frame)

        current_time = time.time()
        status_info = {"text": "初始化...", "color": (200, 200, 200)}

        # 默认不收集数据，除非在检测状态
        collect_data = False

        if results.multi_hand_landmarks:
            mp_draw.draw_landmarks(frame, results.multi_hand_landmarks[0], mp.solutions.hands.HAND_CONNECTIONS)

            # --- 1. 基础运动量计算 (只影响静止监测状态) ---
            norm_landmarks = results.multi_hand_landmarks[0].landmark
            curr_hand_norm = np.array([[lm.x, lm.y] for lm in norm_landmarks])

            is_moving = False
            if prev_hand_norm is not None:
                diff = curr_hand_norm - prev_hand_norm
                max_move = np.max(np.linalg.norm(diff, axis=1))
                if max_move > MOVEMENT_THRESHOLD:
                    last_move_time = current_time
                    is_moving = True
            prev_hand_norm = curr_hand_norm

            # --- 2. 状态机流转 ---

            # [状态 0] 静止监测
            if current_state == 0:
                status_info = {"text": "状态: 静止 (等待动作)", "color": (150, 150, 150)}
                processor.clear()

                # 只要检测到移动，立即进入倒计时
                if is_moving:
                    current_state = 1
                    state_start_time = current_time  # 开始倒计时

            # [状态 1] 预备倒计时 (强制 2 秒，不进行识别)
            elif current_state == 1:
                elapsed = current_time - state_start_time
                remaining = PREPARE_TIME - elapsed

                status_info = {"text": f"准备中... {remaining:.1f}s", "color": (255, 255, 0)}  # 黄色

                # 时间到 -> 强制进入检测
                if remaining <= 0:
                    current_state = 2
                    processor.clear()  # 清空缓冲区准备接新数据
                    state_start_time = current_time

            # [状态 2] 正在检测 (收集数据 -> 识别 -> 无论结果如何都进冷却)
            elif current_state == 2:
                status_info = {"text": "正在检测...", "color": (0, 255, 0)}  # 绿色
                collect_data = True  # 开启数据收集标志

                # 如果缓冲区满了，进行推理
                if processor.frame_count >= processor.window_size:
                    # 推理
                    input_tensor = processor.get_model_input().to(device)
                    with torch.no_grad():
                        output = model(input_tensor)
                        probs = torch.nn.functional.softmax(output, dim=1)
                        conf, classes = torch.max(probs, 1)

                    confidence = conf.item()

                    # 结果判定
                    if confidence > CONFIDENCE_THRESHOLD:
                        res_text = ACTIONS[classes.item()]
                        res_conf = f"{confidence:.2f}"
                        # 记录成功
                        history_log.append({'action': res_text, 'conf': res_conf})
                    else:
                        # 记录失败（未识别）
                        history_log.append({'action': "未识别", 'conf': "N/A"})

                    if len(history_log) > 3: history_log.pop(0)

                    # 强制进入冷却
                    current_state = 3
                    state_start_time = current_time
                    processor.clear()

            # [状态 3] 冷却期
            elif current_state == 3:
                elapsed = current_time - state_start_time
                remaining = ACTION_COOLDOWN - elapsed
                status_info = {"text": f"冷却中... {remaining:.1f}s", "color": (255, 165, 0)}  # 橙色

                if remaining <= 0:
                    # 冷却结束，回到静止监测
                    current_state = 0
                    processor.clear()

            # --- 3. 数据收集 (仅在检测状态执行) ---
            if collect_data and results.multi_hand_world_landmarks:
                world_landmarks = results.multi_hand_world_landmarks[0].landmark
                # 提取3D坐标
                current_kpts = np.array([[lm.x, lm.y, lm.z] for lm in world_landmarks])

                # 相对坐标处理 (以手腕为原点)
                #current_kpts = current_kpts - current_kpts[0, :]

                processor.update_buffer(current_kpts)

        else:
            # 未检测到手的情况
            status_info = {"text": "未检测到手", "color": (100, 100, 255)}
            prev_hand_norm = None

            # 策略：如果在 "预备" 或 "检测" 阶段手丢了，强制重置回 "静止监测"
            # 防止程序卡在检测状态等待数据
            if current_state in [1, 2]:
                current_state = 0
                processor.clear()

        # 4. 绘制 UI
        frame = draw_ui(frame, status_info, history_log, w)

        cv2.imshow('ST-GCN Gesture Recognition', frame)
        if cv2.waitKey(1) == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()



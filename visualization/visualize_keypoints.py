"""
从 npy 文件中读取批量关键点数据并可视化
自动遍历所有样本，生成高颜值的手部动态视频
"""
import cv2
import numpy as np
import os

# ==================== 配置 ====================
CONFIG = {
    # 输入 npy 文件路径 (4_test.npy 代表第四类：双指向下滑动)
    'npy_file': '../data/mediapipe_results_2/9_test.npy',
    
    # 输出视频的文件夹
    'output_dir': 'swipe_down_test9',
    
    # 视频帧率
    'fps': 15,
    
    # 画布大小
    'canvas_width': 600,
    'canvas_height': 600,
    
    # 模式选择：True -> 仅在屏幕上连续播放预览; False -> 静默保存所有视频到文件夹
    'live_preview_only': False, 
}

# ==================== 手部关键点连接关系 ====================
HAND_CONNECTIONS = [
    # 拇指
    (0, 1), (1, 2), (2, 3), (3, 4),
    # 食指
    (0, 5), (5, 6), (6, 7), (7, 8),
    # 中指
    (0, 9), (9, 10), (10, 11), (11, 12),
    # 无名指
    (0, 13), (13, 14), (14, 15), (15, 16),
    # 小指
    (0, 17), (17, 18), (18, 19), (19, 20),
    # 手掌连接
    (0, 5), (5, 9), (9, 13), (13, 17),
]

# ==================== 赛博朋克高颜值配色 (BGR格式) ====================
KEYPOINT_COLORS = [
    (180, 105, 255),  # 霓虹粉 - 拇指
    (255, 255, 0),    # 青色 - 食指
    (127, 255, 0),    # 荧光绿 - 中指
    (0, 215, 255),    # 金色 - 无名指
    (0, 165, 255),    # 亮橙色 - 小指
]
PALM_COLOR = (200, 200, 200) # 手掌连线颜色
BG_COLOR = (25, 20, 20)      # 深邃黑灰背景

def normalize_keypoints(keypoints):
    """
    将坐标归一化到 [0, 1] 范围，以适配画布大小
    保持长宽比例，防止手势变形
    """
    if keypoints.shape[-1] >= 2:
        keypoints_2d = keypoints[..., :2]
    else:
        keypoints_2d = keypoints
    
    min_vals = np.min(keypoints_2d, axis=(0, 1), keepdims=True)
    max_vals = np.max(keypoints_2d, axis=(0, 1), keepdims=True)
    
    # 寻找最大跨度以维持真实比例
    scale = np.max(max_vals - min_vals) + 1e-8
    
    # 居中归一化
    normalized = (keypoints_2d - min_vals) / scale
    # 计算偏移量让手掌整体居中
    offset = (1.0 - np.max(normalized, axis=(0,1))) / 2.0
    normalized = normalized + offset
    
    return normalized

def draw_hand_skeleton(frame, keypoints, frame_idx, sample_idx, total_samples):
    """在帧上绘制高颜值手部骨架"""
    h, w = frame.shape[:2]
    
    # 转换到像素坐标，留出边缘 padding
    padding = 100
    pixel_keypoints = keypoints.copy()
    pixel_keypoints[:, 0] = (keypoints[:, 0] * (w - padding * 2)) + padding
    pixel_keypoints[:, 1] = (keypoints[:, 1] * (h - padding * 2)) + padding
    pixel_keypoints = pixel_keypoints.astype(int)
    
    # 1. 绘制带有抗锯齿的连接线 (先画线，后画点)
    for i, j in HAND_CONNECTIONS:
        pt1 = tuple(pixel_keypoints[i])
        pt2 = tuple(pixel_keypoints[j])
        
        # 判断是手指连线还是手掌连线
        is_palm = False
        if (i == 0 and j in [1, 5, 9, 13, 17]) or (i in [5, 9, 13] and j == i+4):
            is_palm = True
            
        color = PALM_COLOR if is_palm else get_color_for_joint(j)
        thickness = 2 if is_palm else 3
        
        cv2.line(frame, pt1, pt2, color, thickness, cv2.LINE_AA)
    
    # 2. 绘制高颜值关键点 (发光效果)
    for idx, (x, y) in enumerate(pixel_keypoints):
        color = (255, 255, 255) if idx == 0 else get_color_for_joint(idx)
        
        # 外发光圈
        cv2.circle(frame, (x, y), 7, color, -1, cv2.LINE_AA)
        # 内置白点
        cv2.circle(frame, (x, y), 3, (255, 255, 255), -1, cv2.LINE_AA)
    
    # 3. 绘制半透明信息背景板
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (w, 60), (0, 0, 0), -1)
    alpha = 0.6
    cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)
    
    # 4. 添加高科技风格文本
    text_info = f"Sample: {sample_idx+1}/{total_samples} | Frame: {frame_idx + 1}/35 | Class: Swipe Down"
    cv2.putText(frame, text_info, (20, 35), cv2.FONT_HERSHEY_DUPLEX, 0.7, (0, 255, 255), 1, cv2.LINE_AA)
    
    return frame

def get_color_for_joint(idx):
    if 1 <= idx <= 4: return KEYPOINT_COLORS[0]
    if 5 <= idx <= 8: return KEYPOINT_COLORS[1]
    if 9 <= idx <= 12: return KEYPOINT_COLORS[2]
    if 13 <= idx <= 16: return KEYPOINT_COLORS[3]
    return KEYPOINT_COLORS[4]

def process_all_samples():
    print("=" * 60)
    print("ST-GCN 高颜值批量关键点可视化工具")
    print("=" * 60)
    
    npy_file = CONFIG['npy_file']
    if not os.path.exists(npy_file):
        print(f"❌ 错误：找不到文件 {npy_file}。请检查路径。")
        return
        
    try:
        data = np.load(npy_file)
        # 将输入统一转为 (N, T, V, C)
        if len(data.shape) == 4 and data.shape[1] in [2, 3]:
            # (N, C, T, V) -> (N, T, V, C)
            data = data.transpose(0, 2, 3, 1)
            
        N, T, V, C = data.shape
        print(f"✓ 成功读取文件：{npy_file}")
        print(f"✓ 共发现 {N} 个测试样本视频 (格式: {T}帧, {V}个关键点)")
    except Exception as e:
        print(f"❌ 读取失败：{e}")
        return

    # 创建输出目录
    if not CONFIG['live_preview_only']:
        os.makedirs(CONFIG['output_dir'], exist_ok=True)
        print(f"✓ 输出目录已准备: {CONFIG['output_dir']}")

    for sample_idx in range(N):
        sample = data[sample_idx]
        normalized_sample = normalize_keypoints(sample)
        
        output_video = os.path.join(CONFIG['output_dir'], f"sample_{sample_idx+1:03d}.mp4")
        
        if not CONFIG['live_preview_only']:
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            video_writer = cv2.VideoWriter(output_video, fourcc, CONFIG['fps'], 
                                           (CONFIG['canvas_width'], CONFIG['canvas_height']))
        
        for frame_idx in range(T):
            # 创建深色画布
            frame = np.zeros((CONFIG['canvas_height'], CONFIG['canvas_width'], 3), dtype=np.uint8)
            frame[:] = BG_COLOR
            
            # 绘制
            annotated_frame = draw_hand_skeleton(frame, normalized_sample[frame_idx], 
                                                 frame_idx, sample_idx, N)
            
            if CONFIG['live_preview_only']:
                cv2.imshow("ST-GCN Skeleton Viewer", annotated_frame)
                # 每帧停留时长，按 ESC 退出
                if cv2.waitKey(int(1000 / CONFIG['fps'])) == 27:
                    cv2.destroyAllWindows()
                    return
            else:
                video_writer.write(annotated_frame)
                
        if not CONFIG['live_preview_only']:
            video_writer.release()
            print(f"  已保存: {output_video}")

    if CONFIG['live_preview_only']:
        cv2.destroyAllWindows()
        
    print("\n" + "=" * 60)
    print("🎉 所有样本处理完成！")
    print("=" * 60)

if __name__ == "__main__":
    process_all_samples()

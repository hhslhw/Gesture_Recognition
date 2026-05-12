import os
import numpy as np
import torch
import time
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix, precision_recall_fscore_support

# 导入 ST-GCN 模型
from st_gcn.net.st_gcn_hand_v6 import Model

# ==================== 多流配置 ====================
# enable: 是否启用该流；weight: Late Fusion 时的分数权重
STREAMS = {
    "joint": {
        "enable": True,
        "model_path": "models/best_new.pth",
        "data_dir": "data/mediapipe_results_2",
        "weight": 0.98
    },
    "bone": {
        "enable": True,
        "model_path": "models/best_bone.pth",
        "data_dir": "data/mediapipe_bone_results_2",
        "weight": 0.90
    },
    "motion": {
        "enable": True,
        "model_path": "models/best_motion.pth",
        "data_dir": "data/mediapipe_motion_results_2",
        "weight": 0.82
    }
}

ACTIONS = [
    "单手放大", "单手缩小", "双指向上滑动", "双指向下滑动",
    "双指向右滑动", "双指向左滑动", "双指拉近", "双指放大", "拉手靠近"
]
NUM_CLASSES = len(ACTIONS)
WINDOW_SIZE = 35

def load_model(model_path, device):
    """辅助函数：加载单个流的模型"""
    model = Model(
        channel=3, 
        num_class=NUM_CLASSES, 
        window_size=WINDOW_SIZE,
        multiscale=True, 
        use_attention=True, 
        use_data_bn=True
    ).to(device)
    
    try:
        checkpoint = torch.load(model_path, map_location=device, weights_only=True)
        model.load_state_dict(checkpoint, strict=True)
        model.eval()
        return model
    except Exception as e:
        print(f"❌ 模型加载失败 ({model_path}): {e}")
        return None

def main():
    print("=" * 60)
    print("ST-GCN 多流(Multi-Stream) 模型测试脚本")
    print("=" * 60)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"▶ 使用计算设备: {device}")

    # 1. 动态加载所有开启的模型
    active_models = {}
    for stream_name, config in STREAMS.items():
        if config["enable"]:
            print(f"▶ 正在初始化 [{stream_name}] 流...")
            
            if not os.path.exists(config["data_dir"]):
                print(f"❌ 错误: 找不到测试集目录 '{config['data_dir']}'")
                return
                
            model = load_model(config["model_path"], device)
            if model is None:
                return
            active_models[stream_name] = model
            print(f"✓ [{stream_name}] 流模型加载成功！(融合权重: {config['weight']})")

    if not active_models:
        print("❌ 错误: 没有任何开启的流！请检查配置。")
        return

    # 统计模型参数量
    total_params = 0
    for name, model in active_models.items():
        params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        total_params += params
        print(f"  - [{name}] 流模型参数量: {params / 1e6:.2f} M")
    
    print(f"  => 多流联合总参数量: {total_params / 1e6:.2f} M")

    # GPU Warmup：确保 GPU 达到稳定热状态，避免首批推理耗时偏高
    print("\n▶ 正在进行 GPU Warmup...")
    use_cuda = device.type == "cuda"
    dummy = torch.randn(50, 3, WINDOW_SIZE, 21).to(device)
    with torch.no_grad():
        for _ in range(10):
            for model in active_models.values():
                _ = model(dummy)
    if use_cuda:
        torch.cuda.synchronize()
    del dummy
    print("✓ Warmup 完成")

    # 初始化统计变量
    total_correct = 0
    total_samples = 0
    all_preds = []
    all_labels = []
    
    # 记录每个类别的耗时
    class_times = []

    # 用于累计纯推理时间（不含数据加载）
    total_inference_time = 0.0

    # 2. 遍历测试集进行多流融合推理
    print("\n▶ 开始进行多流融合测试并收集实验指标...")
    print("-" * 60)
    
    with torch.no_grad():
        for i in range(1, NUM_CLASSES + 1):
            true_label = i - 1
            action_name = ACTIONS[true_label]
            file_name = f"{i}_test.npy"
            
            fused_scores = None
            num_samples = 0
            skip_class = False
            
            # 先完成所有数据加载和预处理（不计入推理耗时）
            stream_tensors = {}
            for stream_name, model in active_models.items():
                config = STREAMS[stream_name]
                file_path = os.path.join(config["data_dir"], file_name)
                
                if not os.path.exists(file_path):
                    print(f"  [类别 {i}] {action_name}: ⚠️ 找不到 {stream_name} 测试文件 '{file_name}'")
                    skip_class = True
                    break
                    
                data = np.load(file_path)
                current_samples = data.shape[0]
                
                if current_samples == 0:
                    skip_class = True
                    break
                
                if num_samples == 0:
                    num_samples = current_samples
                
                # 若存在 M 维度（M=1），挤压掉
                if len(data.shape) == 5:
                    data = data.squeeze(-1)  # (N, C, T, V, 1) -> (N, C, T, V)

                # 坐标归一化：与 HandFeeder 保持一致，以第0帧手腕为原点做平移
                origin = data[:, :, 0, 0]  # (N, C)
                data_normalized = data - origin[:, :, np.newaxis, np.newaxis]
                
                input_tensor = torch.from_numpy(data_normalized).float().to(device)
                stream_tensors[stream_name] = input_tensor

            if skip_class or num_samples == 0:
                continue

            # 仅测量前向传播耗时（不含数据加载）
            if use_cuda:
                torch.cuda.synchronize()
            class_start_time = time.perf_counter()
            
            for stream_name, model in active_models.items():
                input_tensor = stream_tensors[stream_name]
                
                # 前向传播
                outputs = model(input_tensor)
                
                # 多流融合：提取 Softmax 后的概率进行加权相加 (Late Fusion)
                probs = torch.nn.functional.softmax(outputs, dim=1)
                weighted_probs = probs * STREAMS[stream_name]["weight"]
                
                # 累加融合分数
                if fused_scores is None:
                    fused_scores = weighted_probs
                else:
                    fused_scores += weighted_probs

            if use_cuda:
                torch.cuda.synchronize()
            class_end_time = time.perf_counter()
            
            class_time_ms = (class_end_time - class_start_time) / num_samples * 1000 if num_samples > 0 else 0
            class_times.append(class_time_ms)
            total_inference_time += (class_end_time - class_start_time)

            # 取融合分数最大值作为预测类别
            confs, preds = torch.max(fused_scores, 1)
            correct = (preds == true_label).sum().item()

            # 输出预测错误的样本序号
            incorrect_indices = (preds != true_label).nonzero(as_tuple=True)[0].cpu().numpy().tolist()
            if incorrect_indices:
                print(f"  [类别 {i}] {action_name:<10}: ❌ 预测错误的样本序号: {incorrect_indices}")

            # 收集预测和真实标签用于统一计算
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend([true_label] * num_samples)
            
            # 更新整体统计
            total_correct += correct
            total_samples += num_samples
            
            acc = correct / num_samples
            print(f"  [类别 {i}] {action_name:<10}: 样本数 {num_samples:3d} | 正确数 {correct:3d} | 召回率: {acc*100:6.2f}% | 批量单样本耗时: {class_time_ms:.2f} ms")

    # 3. 计算综合指标并导出为 Excel
    print("-" * 60)
    if total_samples > 0:
        overall_acc = total_correct / total_samples * 100
        avg_time = total_inference_time / total_samples * 1000  # ms（纯推理）
        fps = 1000.0 / avg_time if avg_time > 0 else 0
        
        print("▶ 测试完成报告 (综合性能):")
        print(f"  总样本数:   {total_samples}")
        print(f"  总体准确率: {overall_acc:.2f}%")
        print(f"  单样本耗时: {avg_time:.2f} ms")
        print(f"  推理吞吐量: {fps:.2f} FPS")
        print(f"  总参数量:   {total_params / 1e6:.2f} M")

        # 使用 sklearn 计算精确率、召回率、F1
        precision, recall, f1, support = precision_recall_fscore_support(
            all_labels, all_preds, labels=range(NUM_CLASSES), zero_division=0
        )
        
        # 计算逐类准确率 (One-vs-Rest Accuracy): (TP + TN) / 总样本数
        conf_mat = confusion_matrix(all_labels, all_preds, labels=range(NUM_CLASSES))
        class_accuracy = []
        for c in range(NUM_CLASSES):
            tp = conf_mat[c, c]
            fn = conf_mat[c, :].sum() - tp
            fp = conf_mat[:, c].sum() - tp
            tn = conf_mat.sum() - tp - fn - fp
            acc_c = (tp + tn) / (tp + tn + fp + fn) if (tp + tn + fp + fn) > 0 else 0
            class_accuracy.append(acc_c)
        
        # 格式化四位小数
        def fmt4(x): return float(f"{x:.4f}")

        # 构建 DataFrame 用于导出
        df_metrics = pd.DataFrame({
            "动作类别": ACTIONS,
            "准确率 (Accuracy)": [fmt4(a) for a in class_accuracy],
            "精确率 (Precision)": [fmt4(p) for p in precision],
            "召回率 (Recall)": [fmt4(r) for r in recall],
            "F1-Score": [fmt4(f) for f in f1],
            "批量单样本耗时 (ms)": [float(f"{t:.2f}") for t in class_times],
            "样本数 (Support)": support
        })
        
        # 添加总体平均行
        df_metrics.loc["Macro Avg"] = [
            "平均 (Macro)",
            fmt4(np.mean(class_accuracy)),
            fmt4(precision.mean()),
            fmt4(recall.mean()),
            fmt4(f1.mean()),
            float(f"{np.mean(class_times):.2f}"),
            total_samples
        ]

        print("\n▶ 详细分类指标:")
        print(df_metrics.to_string(index=False))

        # 单样本延迟基准测试（batch_size=1，模拟实时推理场景）
        print("\n▶ 正在进行单样本延迟基准测试 (batch_size=1, 100次取平均)...")
        single_dummy = torch.randn(1, 3, WINDOW_SIZE, 21).to(device)

        with torch.no_grad():
            for _ in range(10):
                for model in active_models.values():
                    _ = model(single_dummy)
        if use_cuda:
            torch.cuda.synchronize()
        
        num_bench_runs = 100
        latencies = []
        with torch.no_grad():
            for _ in range(num_bench_runs):
                if use_cuda:
                    torch.cuda.synchronize()
                t0 = time.perf_counter()
                for stream_name, model in active_models.items():
                    _ = model(single_dummy)
                if use_cuda:
                    torch.cuda.synchronize()
                t1 = time.perf_counter()
                latencies.append((t1 - t0) * 1000)  # ms
        
        single_avg_ms = np.mean(latencies)
        single_std_ms = np.std(latencies)
        single_fps = 1000.0 / single_avg_ms if single_avg_ms > 0 else 0
        
        print(f"  单样本推理延迟: {single_avg_ms:.2f} ± {single_std_ms:.2f} ms")
        print(f"  实时推理帧率:   {single_fps:.2f} FPS")
        
        # 混淆矩阵已在前面计算（conf_mat）
        df_conf = pd.DataFrame(conf_mat, index=ACTIONS, columns=ACTIONS)

        # 导出至 Excel
        output_xlsx = "evaluation_report1.xlsx"
        try:
            with pd.ExcelWriter(output_xlsx) as writer:
                df_metrics.to_excel(writer, sheet_name="分类性能指标", index=False)
                df_conf.to_excel(writer, sheet_name="混淆矩阵")
                
                # 记录系统性能至单独的工作表
                df_sys = pd.DataFrame({
                    "指标名称": [
                        "总体准确率", 
                        "总参数量 (M)", 
                        "批量单样本平均耗时 (ms)",
                        "批量推理吞吐量 (FPS)",
                        "单样本推理延迟 (ms, batch=1)",
                        "单样本推理延迟标准差 (ms)",
                        "实时推理帧率 (FPS, batch=1)"
                    ],
                    "数值": [
                        f"{overall_acc:.2f}%", 
                        f"{total_params/1e6:.2f}", 
                        f"{avg_time:.2f}", 
                        f"{fps:.2f}",
                        f"{single_avg_ms:.2f}",
                        f"{single_std_ms:.2f}",
                        f"{single_fps:.2f}"
                    ]
                })
                df_sys.to_excel(writer, sheet_name="系统性能开销", index=False)
            
            print(f"\n✅ 测试报告已导出: {output_xlsx}（分类指标、混淆矩阵、系统性能开销）")
        except Exception as e:
            print(f"⚠️ 导出 Excel 失败 (请确保已安装 pandas 和 openpyxl): {e}")
                
    else:
        print("❌ 未找到任何有效测试样本，无法计算总体准确率。")
    print("=" * 60)

if __name__ == "__main__":
    main()

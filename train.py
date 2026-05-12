import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from torch.optim import Adam
from torch.optim.lr_scheduler import StepLR
import numpy as np
import os
import json
from datetime import datetime
from collections import Counter
from torch.optim.lr_scheduler import CosineAnnealingLR

# 从你的模块导入模型和数据加载器
from st_gcn.net.st_gcn_hand_v6 import Model
from st_gcn.feeder.hand_feeder_v4 import HandFeeder

# 设置参数
args = {
    "data_path": "mediapipe_results_2",
    "num_class": 9,
    "channel": 3,
    "window_size": 35,
    "batch_size": 16,
    "num_epoch": 100,  #  增加训练轮次
    "learning_rate": 5e-4,
    "device": "cuda" if torch.cuda.is_available() else "cpu",
    "model_save_path": "models/best_new.pth",
    "log_dir": "logs",
    "print_freq": 10,
    "use_attention": True  #  全局启用注意力
}

# 创建日志目录
os.makedirs(args["log_dir"], exist_ok=True)

def main():
    print("Loading dataset...")

    # 加载训练集 (v4 版本默认 normalization=True)
    train_dataset = HandFeeder(
        data_path=args["data_path"],
        mode="train",
        window_size=args["window_size"],
        normalization=True,
        debug=args.get("debug", False)
    )

    # 加载验证集（直接使用提取脚本划分好的 val 数据）
    val_dataset = HandFeeder(
        data_path=args["data_path"],
        mode="val",
        window_size=args["window_size"],
        normalization=True,
        debug=args.get("debug", False)
    )

    train_loader = DataLoader(train_dataset, batch_size=args["batch_size"], shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args["batch_size"], shuffle=False)

    # 加载测试集
    test_dataset = HandFeeder(
        data_path=args["data_path"],
        mode="test",
        window_size=args["window_size"],
        normalization=True,
        debug=args.get("debug", False)
    )
    test_loader = DataLoader(test_dataset, batch_size=args["batch_size"], shuffle=False)

    print(f"Train samples: {len(train_dataset)}, Val samples: {len(val_dataset)}, Test samples: {len(test_dataset)}")

    print(" Building model...")
    model = Model(
        channel=args["channel"],
        num_class=args["num_class"],
        window_size=args["window_size"],
        num_point=21,
        use_data_bn=True,
        backbone_config=None,
        mask_learning=False,
        use_local_bn=True,
        multiscale=True,  #  启用多尺度
        use_attention=args["use_attention"],  #  控制注意力开关
        temporal_kernel_size=5,
        dropout=0.5
    ).to(args["device"])

    print(" Model built successfully.")

    criterion = nn.CrossEntropyLoss()
    optimizer = Adam(model.parameters(), lr=args["learning_rate"])
    scheduler = CosineAnnealingLR(optimizer, T_max=args["num_epoch"], eta_min=1e-6)

    best_test_acc = 0.0  # 使用测试集准确率作为保存标准

    # 日志文件
    log_file = os.path.join(args["log_dir"], f"train_log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json")
    log_data = []

    #  早停机制
    patience = 20
    early_stop_counter = 0

    print(" Training started...")
    for epoch in range(args["num_epoch"]):
        model.train()
        total_loss = 0.0
        correct = 0
        total = 0

        for i, (data, label) in enumerate(train_loader):
            data = data.float().to(args["device"])  # shape: (B, C, T, V)
            label = label.long().to(args["device"])  # shape: (B, )

            optimizer.zero_grad()
            output = model(data)  # shape: (B, num_class)
            loss = criterion(output, label)
            loss.backward()
            optimizer.step()

            # 统计准确率
            _, predicted = torch.max(output.data, 1)
            total += label.size(0)
            correct += (predicted == label).sum().item()
            total_loss += loss.item()

            if (i + 1) % args["print_freq"] == 0:
                print(f"Epoch [{epoch + 1}/{args['num_epoch']}], "
                      f"Iter [{i + 1}], Loss: {loss.item():.4f}")

        scheduler.step()
        avg_loss = total_loss / len(train_loader)
        train_acc = 100 * correct / total

        #  验证集评估（仅输出基础准确率）
        val_acc = evaluate(model, val_loader, args["device"], output_report=False)

        #  测试集评估（输出完整报告）
        test_acc = evaluate(model, test_loader, args["device"], output_report=True, epoch=epoch+1)
        print(f" Test Accuracy: {test_acc:.2f}%")

        #  早停机制
        if test_acc > best_test_acc:
            best_test_acc = test_acc
            early_stop_counter = 0
            torch.save(model.state_dict(), args["model_save_path"])
            print(f" Best model saved with test acc: {best_test_acc:.2f}%\n")
        else:
            early_stop_counter += 1
            if early_stop_counter >= patience:
                print(f"Early stopping at epoch {epoch+1}")
                break

        # 记录日志
        log_entry = {
            "epoch": epoch + 1,
            "train_loss": avg_loss,
            "val_acc": val_acc,
            "test_acc": test_acc
        }
        log_data.append(log_entry)

    # 保存日志
    with open(log_file, 'w') as f:
        json.dump(log_data, f, indent=4)
    print(f" Training log saved to {log_file}")


def evaluate(model, data_loader, device, output_report=True, epoch=None):
    """
    评估模型性能
    :param model: 模型
    :param data_loader: 数据加载器
    :param device: 设备
    :param output_report: 是否输出分类报告
    :param epoch: 当前训练轮次（用于文件命名）
    :return: 准确率
    """
    model.eval()
    correct = 0
    total = 0
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for data, label in data_loader:
            data = data.float().to(device)
            label = label.long().to(device)

            output = model(data)
            _, predicted = torch.max(output.data, 1)
            total += label.size(0)
            correct += (predicted == label).sum().item()

            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(label.cpu().numpy())

    acc = 100 * correct / total

    # 可选：输出分类报告
    if output_report:
        try:
            from sklearn.metrics import classification_report, confusion_matrix
            print("\nClassification Report:")
            print(classification_report(all_labels, all_preds, digits=4))
            print("\nConfusion Matrix:")
            print(confusion_matrix(all_labels, all_preds))
        except ImportError:
            print(" sklearn not found, skipping classification report")

    return acc


if __name__ == '__main__':
    main()

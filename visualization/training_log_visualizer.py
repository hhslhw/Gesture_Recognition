import matplotlib.pyplot as plt
import json
from datetime import datetime
import numpy as np


def plot_training_log(log_path="../logs/train_log_20250508_085252.json"):
    """
    从训练日志文件中加载数据并绘制准确率和损失曲线，
    并在准确率曲线中添加验证准确率和测试准确率的最大值水平虚线
    """
    # 加载日志数据
    try:
        with open(log_path, "r") as f:
            log_data = json.load(f)
    except FileNotFoundError:
        print(f"日志文件 {log_path} 未找到，请确认路径是否正确")
        return

    # 提取指标
    epochs = [entry["epoch"] for entry in log_data]
    train_losses = [entry["train_loss"] for entry in log_data]
    train_accs = [entry.get("train_acc", 0) for entry in log_data]
    test_accs = [entry.get("test_acc", 0) for entry in log_data]

    # 计算最大值
    max_train_acc = max(train_accs)
    max_train_epoch = epochs[train_accs.index(max_train_acc)]

    max_test_acc = max(test_accs)
    max_test_epoch = epochs[test_accs.index(max_test_acc)]

    # 计算最小损失
    min_train_loss = min(train_losses)
    min_loss_epoch = epochs[train_losses.index(min_train_loss)]

    # ------------------------ 准确率曲线 ------------------------
    plt.figure(figsize=(12, 6))
    plt.plot(epochs, train_accs, label="Train Accuracy", marker='o')
    plt.plot(epochs, test_accs, label="Test Accuracy", marker='x')

    #  添加最大值水平虚线
    plt.axhline(y=max_train_acc, color='r', linestyle='--',
                label=f'Max Train Acc: {max_train_acc:.2f}% (Epoch {max_train_epoch})')
    plt.axhline(y=max_test_acc, color='b', linestyle='--',
                label=f'Max Test Acc: {max_test_acc:.2f}% (Epoch {max_test_epoch})')

    #  调整纵坐标密度（主刻度每5%一个，次刻度每2.5%一个）
    plt.yticks(np.arange(30, 100, 20))  # 主刻度每5%一个
    ax = plt.gca()
    ax.yaxis.set_minor_locator(plt.MultipleLocator(5))  # 次刻度每2.5%一个
    ax.yaxis.grid(True, which='major', linestyle='-', alpha=0.5)
    ax.yaxis.grid(True, which='minor', linestyle=':', alpha=0.3)

    plt.xlabel("Epoch")
    plt.ylabel("Accuracy (%)")
    plt.title("Train and Test Accuracy Over Epochs")
    plt.legend()
    plt.grid(True)
    plt.savefig("val_test_accuracy_curve_1.png")
    plt.close()

    # ------------------------ 损失曲线 ------------------------
    plt.figure(figsize=(12, 6))
    plt.plot(epochs, train_losses, label="Train Loss", color='orange')

    # ✅ 添加最小值水平虚线
    plt.axhline(y=min_train_loss, color='g', linestyle='--',
                label=f'Min Loss: {min_train_loss:.4f} (Epoch {min_loss_epoch})')

    # ✅ 调整纵坐标密度（主刻度每0.5一个，次刻度每0.25一个）
    plt.yticks(np.arange(0.0, max(train_losses) + 1, 5))  # 主刻度每0.5一个
    ax_loss = plt.gca()
    ax_loss.yaxis.set_minor_locator(plt.MultipleLocator(1))  # 次刻度每0.25一个
    ax_loss.yaxis.grid(True, which='major', linestyle='-', alpha=0.5)
    ax_loss.yaxis.grid(True, which='minor', linestyle=':', alpha=0.3)

    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training Loss Curve")
    plt.legend()
    plt.grid(True)
    plt.savefig("training_loss_curve_1.png")
    plt.close()

    print("准确率曲线和损失曲线已生成！")
    print(f"最大训练准确率: {max_train_acc:.2f}% (Epoch {max_train_epoch})")
    print(f"最大测试准确率: {max_test_acc:.2f}% (Epoch {max_test_epoch})")
    print(f"最小训练损失: {min_train_loss:.4f} (Epoch {min_loss_epoch})")


if __name__ == "__main__":
    plot_training_log()


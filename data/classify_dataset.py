import os
import shutil
import pandas as pd
from pathlib import Path


def organize_dataset(source_dir, labels_path,
                     train_csv, val_csv, test_csv,
                     output_dir="classified_jester"):
    """
    组织Jester数据集到指定结构
    参数：
        source_dir: 原始数据目录（包含数字编号的文件夹）
        labels_path: 类别标签文件路径
        train_csv: 训练集CSV路径
        val_csv: 验证集CSV路径
        test_csv: 测试集CSV路径
        output_dir: 输出目录（默认为organized_jester）
    """
    # 创建输出目录结构
    train_dir = Path(output_dir) / "train"
    val_dir = Path(output_dir) / "validation"
    test_dir = Path(output_dir) / "test"

    # 读取类别标签
    labels = pd.read_csv(labels_path, header=None)[0].tolist()

    # 处理训练集和验证集的函数
    def process_dataset(df, target_dir):
        for _, row in df.iterrows():
            index, label = row[0].split(";")
            src = Path(source_dir) / index
            dst = target_dir / label.strip()
            dst.mkdir(parents=True, exist_ok=True)
            shutil.copytree(src, dst / index)

    # 处理训练集
    train_df = pd.read_csv(train_csv, header=None)
    process_dataset(train_df, train_dir)
    print(f"训练集处理完成，共{len(train_df)}个样本")

    # 处理验证集
    val_df = pd.read_csv(val_csv, header=None)
    process_dataset(val_df, val_dir)
    print(f"验证集处理完成，共{len(val_df)}个样本")

    # 处理测试集（不需要分类）
    test_df = pd.read_csv(test_csv, header=None)
    test_dir.mkdir(parents=True, exist_ok=True)
    for index in test_df[0]:
        src = Path(source_dir) / str(index)
        shutil.copytree(src, test_dir / str(index))
    print(f"测试集处理完成，共{len(test_df)}个样本")


if __name__ == "__main__":
    # 使用示例（请根据实际情况修改路径）
    organize_dataset(
        source_dir="jester",
        labels_path="标注/jester-v1-labels.csv",
        train_csv="标注/jester-v1-train.csv",
        val_csv="标注/jester-v1-validation.csv",
        test_csv="标注/jester-v1-test.csv",
        output_dir="classified_jester"
    )

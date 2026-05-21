"""
将 evaluation_report*.xlsx 中的"混淆矩阵"子表绘制为热力图。

用法:
    python visualization/confusion_matrix_heatmap.py

输出:
    visualization/confusion_matrix_heatmap1.png
    visualization/confusion_matrix_heatmap2.png
    visualization/confusion_matrix_heatmap_combined.png
"""

from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # 使用非交互式后端，避免 Windows 下 GUI 后端崩溃
import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
import openpyxl


# ---------------------------------------------------------------------------
# 中文字体配置（避免中文显示成方块）
# ---------------------------------------------------------------------------
def _setup_chinese_font() -> None:
    """按系统可用情况优先选择中文字体。"""
    candidates = [
        "Microsoft YaHei",
        "SimHei",
        "Source Han Sans SC",
        "Noto Sans CJK SC",
        "PingFang SC",
        "Arial Unicode MS",
    ]
    available = {f.name for f in mpl.font_manager.fontManager.ttflist}
    for name in candidates:
        if name in available:
            mpl.rcParams["font.sans-serif"] = [name]
            break
    mpl.rcParams["axes.unicode_minus"] = False


# ---------------------------------------------------------------------------
# 数据读取
# ---------------------------------------------------------------------------
def load_confusion_matrix(xlsx_path: Path, sheet_index: int = 1):
    """从 xlsx 的指定子表读取混淆矩阵。

    约定子表第一行为列标签（首格留空），第一列为行标签，其余为整数计数。

    返回:
        labels: list[str], 类别名（按行/列同序）
        matrix: np.ndarray of shape (C, C), int
    """
    wb = openpyxl.load_workbook(str(xlsx_path), data_only=True)
    ws = wb.worksheets[sheet_index]

    rows = [[c.value for c in r] for r in ws.iter_rows()]
    if not rows:
        raise ValueError(f"子表为空: {xlsx_path}")

    header = rows[0]
    col_labels = [str(v) for v in header[1:] if v is not None]

    row_labels, data = [], []
    for r in rows[1:]:
        if r[0] is None:
            continue
        row_labels.append(str(r[0]))
        data.append([int(v) if v is not None else 0 for v in r[1 : 1 + len(col_labels)]])

    if row_labels != col_labels:
        raise ValueError(
            f"行/列标签不一致: rows={row_labels}, cols={col_labels} ({xlsx_path})"
        )
    return row_labels, np.asarray(data, dtype=int)


# ---------------------------------------------------------------------------
# 绘图
# ---------------------------------------------------------------------------
def plot_confusion_heatmap(
    matrix: np.ndarray,
    labels: list,
    title: str,
    out_path: Path,
    normalize: bool = False,
    cmap: str = "Blues",
    ax: plt.Axes = None,
) -> plt.Axes:
    """绘制单个混淆矩阵热力图。"""
    cm = matrix.astype(float)
    if normalize:
        row_sum = cm.sum(axis=1, keepdims=True)
        row_sum[row_sum == 0] = 1.0
        cm = cm / row_sum

    standalone = ax is None
    if standalone:
        fig, ax = plt.subplots(figsize=(8.5, 7.0))

    im = ax.imshow(cm, interpolation="nearest", cmap=cmap, vmin=0,
                   vmax=1.0 if normalize else cm.max())

    ax.set_title(title, fontsize=13, pad=12)
    ax.set_xlabel("预测类别 (Predicted)", fontsize=11)
    ax.set_ylabel("真实类别 (True)", fontsize=11)

    n = len(labels)
    ax.set_xticks(np.arange(n))
    ax.set_yticks(np.arange(n))
    ax.set_xticklabels(labels, rotation=40, ha="right", fontsize=10)
    ax.set_yticklabels(labels, fontsize=10)

    # 单元格内标注数值；颜色根据背景深浅自适应
    threshold = (cm.max() + cm.min()) / 2.0 if cm.size else 0.5
    for i in range(n):
        for j in range(n):
            v = cm[i, j]
            if normalize:
                txt = f"{v:.2f}" if v > 0 else "."
            else:
                txt = f"{int(matrix[i, j])}" if matrix[i, j] > 0 else "."
            ax.text(
                j, i, txt,
                ha="center", va="center",
                color="white" if v > threshold else "black",
                fontsize=9,
            )

    # 网格线
    ax.set_xticks(np.arange(n + 1) - 0.5, minor=True)
    ax.set_yticks(np.arange(n + 1) - 0.5, minor=True)
    ax.grid(which="minor", color="lightgray", linestyle="-", linewidth=0.5)
    ax.tick_params(which="minor", bottom=False, left=False)

    if standalone:
        cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        cbar.ax.tick_params(labelsize=9)
        plt.tight_layout()
        out_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(out_path, dpi=200, bbox_inches="tight")
        plt.close(fig)
        print(f"[OK] 已保存: {out_path}")
    else:
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    return ax


def plot_combined(
    items: list,
    out_path: Path,
    normalize: bool = False,
) -> None:
    """把多个混淆矩阵并排画在一张图里，便于对比。"""
    n = len(items)
    fig, axes = plt.subplots(1, n, figsize=(8.5 * n, 7.0))
    if n == 1:
        axes = [axes]
    for ax, (title, labels, mat) in zip(axes, items):
        plot_confusion_heatmap(mat, labels, title, out_path, normalize=normalize, ax=ax)
    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"[OK] 已保存: {out_path}")


# ---------------------------------------------------------------------------
# 主流程
# ---------------------------------------------------------------------------
def main() -> None:
    _setup_chinese_font()

    base = Path(__file__).resolve().parent
    sources = [
        ("评估报告 1 - 混淆矩阵", base / "evaluation_report1.xlsx",
         base / "confusion_matrix_heatmap1.png"),
        ("评估报告 2 - 混淆矩阵", base / "evaluation_report2.xlsx",
         base / "confusion_matrix_heatmap2.png"),
    ]

    loaded = []
    for title, xlsx, out_png in sources:
        if not xlsx.exists():
            print(f"[WARN] 找不到文件，跳过: {xlsx}")
            continue
        labels, matrix = load_confusion_matrix(xlsx)
        # 单图：原始计数
        plot_confusion_heatmap(matrix, labels, title, out_png, normalize=False)
        # 单图：行归一化（召回率视图）
        plot_confusion_heatmap(
            matrix, labels, title + "（按行归一化）",
            out_png.with_name(out_png.stem + "_normalized.png"),
            normalize=True, cmap="Greens",
        )
        loaded.append((title, labels, matrix))

    # 并排对比
    if len(loaded) >= 2:
        plot_combined(loaded, base / "confusion_matrix_heatmap_combined.png", normalize=False)
        plot_combined(
            loaded,
            base / "confusion_matrix_heatmap_combined_normalized.png",
            normalize=True,
        )


if __name__ == "__main__":
    main()

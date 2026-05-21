"""
ST-GCN 精简架构示意图（手绘版，适合论文配图）
==============================================
直接用 matplotlib 按模块分层手绘，避免 torchview 把每个内部算子都展开导致图过长。
按照 default_backbone_hand 的 7 个 block 配置：
    [(64,64,1), (64,64,1), (64,128,2), (128,128,1), (128,128,2), (128,256,1), (256,256,1)]

输出：
    visualization/st_gcn_arch_diagram.png
"""
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
from matplotlib.lines import Line2D

plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False


# ============== 颜色主题 ==============
C_INPUT    = '#90a4ae'
C_BN       = '#b0bec5'
C_STEM     = '#80cbc4'
C_BLOCK_S1 = '#64b5f6'
C_BLOCK_S2 = '#ef9a9a'
C_ATTN     = '#ce93d8'
C_POOL     = '#ffcc80'
C_FC       = '#a5d6a7'
C_OUTPUT   = '#90a4ae'

BACKBONE = [
    (1, 64, 64, 1),
    (2, 64, 64, 1),
    (3, 64, 128, 2),
    (4, 128, 128, 1),
    (5, 128, 128, 2),
    (6, 128, 256, 1),
    (7, 256, 256, 1),
]


def add_box(ax, x, y, w, h, text, color, fontsize=10, fontweight='normal',
            edgecolor='#37474f', text_color='black'):
    box = FancyBboxPatch((x, y), w, h,
                         boxstyle="round,pad=0.02,rounding_size=0.06",
                         linewidth=1.2, facecolor=color,
                         edgecolor=edgecolor, zorder=2)
    ax.add_patch(box)
    ax.text(x + w / 2, y + h / 2, text,
            ha='center', va='center', fontsize=fontsize,
            fontweight=fontweight, color=text_color, zorder=3)


def add_arrow(ax, x1, y1, x2, y2, color='#37474f', lw=1.5):
    arrow = FancyArrowPatch((x1, y1), (x2, y2),
                            arrowstyle='-|>', mutation_scale=14,
                            color=color, lw=lw, zorder=1)
    ax.add_patch(arrow)


def add_tensor_label(ax, x, y, text, fontsize=8, color='#546e7a'):
    ax.text(x, y, text, ha='center', va='center',
            fontsize=fontsize, color=color, style='italic',
            bbox=dict(boxstyle='round,pad=0.2', fc='white',
                      ec='#cfd8dc', lw=0.6), zorder=4)


def draw():
    # ============== 全图坐标系：x=[0,14], y=[0,14] ==============
    fig, ax = plt.subplots(figsize=(13, 9.5))

    # ==========================================================
    # 左半区：主干流程（x = 0.5 ~ 5.5）
    # ==========================================================
    main_x = 1.0
    main_w = 4.0
    main_cx = main_x + main_w / 2

    # ---- 输入 (y=13.0) ----
    add_box(ax, main_x, 13.0, main_w, 0.7,
            '骨架序列输入  (N, 3, 35, 21)\nC=3 (x,y,z)   T=35帧   V=21关节',
            C_INPUT, fontsize=9.5, fontweight='bold')

    # ---- 数据 BN ----
    add_box(ax, main_x, 12.0, main_w, 0.55,
            '数据 BatchNorm   (use_data_bn)',
            C_BN, fontsize=9.5)
    add_arrow(ax, main_cx, 13.0, main_cx, 12.55)

    # ---- Stem ----
    add_box(ax, main_x, 10.85, main_w, 0.85,
            '初始特征头 (Stem)\nGCN_0  →  TCN_0 (k=9)\n3 → 64 通道',
            C_STEM, fontsize=9, fontweight='bold')
    add_arrow(ax, main_cx, 12.0, main_cx, 11.7)
    add_tensor_label(ax, main_cx, 10.62, '(N, 64, 35, 21)')

    # ---- 7 个 backbone blocks ----
    block_h = 0.55
    block_gap = 0.18
    block_top = 9.95
    block_y_list = []
    for idx, (i, c_in, c_out, stride) in enumerate(BACKBONE):
        y = block_top - idx * (block_h + block_gap)
        block_y_list.append(y)
        is_down = stride == 2
        color = C_BLOCK_S2 if is_down else C_BLOCK_S1
        label = f"Block {i}: GCN + TCN(k=5/9)   {c_in}→{c_out}   stride={stride}"
        if is_down:
            label += "  ↓T"
        add_box(ax, main_x, y, main_w, block_h, label, color,
                fontsize=9, fontweight='bold' if is_down else 'normal')
        if idx > 0:
            add_arrow(ax, main_cx, y + block_h + block_gap,
                      main_cx, y + block_h)

    # 顶部 stem -> 第一个 block
    add_arrow(ax, main_cx, 10.85, main_cx, block_y_list[0] + block_h)

    # 7-block 大括号
    last_block_y = block_y_list[-1]
    ax.annotate('', xy=(main_x + main_w + 0.05, last_block_y),
                xytext=(main_x + main_w + 0.05, block_y_list[0] + block_h),
                arrowprops=dict(arrowstyle='-[', lw=1.5,
                                color='#37474f', shrinkA=0, shrinkB=0))
    ax.text(main_x + main_w + 0.40,
            (block_y_list[0] + block_h + last_block_y) / 2,
            '7-block\nbackbone\n(multiscale\n+ residual)',
            fontsize=8.5, ha='left', va='center',
            color='#37474f', fontweight='bold')

    # ---- 注意力 ----
    attn_y = last_block_y - 0.85
    add_box(ax, main_x, attn_y, main_w, 0.6,
            '时空联合注意力\nSpatial Attn  *  Temporal Attn   (gamma-residual)',
            C_ATTN, fontsize=9, fontweight='bold')
    add_arrow(ax, main_cx, last_block_y, main_cx, attn_y + 0.6)
    add_tensor_label(ax, main_cx, last_block_y - 0.26,
                     '(N, 256, ~9, 21)')

    # ---- 全局池化 ----
    pool_y = attn_y - 0.85
    add_box(ax, main_x, pool_y, main_w, 0.55,
            'GAP over V (21 关节)  →  GAP over T',
            C_POOL, fontsize=9)
    add_arrow(ax, main_cx, attn_y, main_cx, pool_y + 0.55)

    # ---- 分类头 ----
    fc_y = pool_y - 0.95
    add_box(ax, main_x, fc_y, main_w, 0.65,
            '分类头 fcn\nConv1×1 + ReLU + Dropout(0.5) + Conv1×1',
            C_FC, fontsize=9)
    add_arrow(ax, main_cx, pool_y, main_cx, fc_y + 0.65)
    add_tensor_label(ax, main_cx, pool_y - 0.18, '(N, 256)')

    # ---- 输出 ----
    out_y = fc_y - 0.85
    add_box(ax, main_x + 0.6, out_y, main_w - 1.2, 0.5,
            '9 类手势 logits   (N, 9)',
            C_OUTPUT, fontsize=10, fontweight='bold')
    add_arrow(ax, main_cx, fc_y, main_cx, out_y + 0.5)

    # ==========================================================
    # 右半区：TCN-GCN block 内部结构（x = 7.5 ~ 12.5）
    # ==========================================================
    inner_x = 7.5
    inner_w = 4.5
    inner_cx = inner_x + inner_w / 2

    # 标题
    ax.text(inner_cx, 13.5, 'TCN-GCN Block 内部结构 (Block 1~7 共享)',
            ha='center', va='center', fontsize=12, fontweight='bold',
            color='#37474f')

    # input feature 标签
    ax.text(inner_cx, 12.85, 'Input Feature',
            ha='center', va='center', fontsize=9.5,
            fontweight='bold', color='#37474f')
    add_arrow(ax, inner_cx, 12.7, inner_cx, 12.4)

    # unit_gcn
    add_box(ax, inner_x + 0.5, 11.8, inner_w - 1, 0.6,
            'unit_gcn  (3 分区图卷积 + BN + ReLU)',
            C_BLOCK_S1, fontsize=9)
    add_arrow(ax, inner_cx, 11.8, inner_cx, 11.4)

    # 分叉点
    fork_y = 11.4
    branch_left_x = inner_x + 1.0
    branch_right_x = inner_x + inner_w - 1.0

    # 分叉横线
    ax.plot([branch_left_x, branch_right_x], [fork_y, fork_y],
            color='#37474f', lw=1.2, zorder=1)
    add_arrow(ax, branch_left_x, fork_y, branch_left_x, 10.85)
    add_arrow(ax, branch_right_x, fork_y, branch_right_x, 10.85)

    # multiscale 两路 TCN
    add_box(ax, branch_left_x - 0.7, 10.25, 1.4, 0.6,
            'TCN  (k=5)\n短时局部',
            C_BLOCK_S2, fontsize=8.5)
    add_box(ax, branch_right_x - 0.7, 10.25, 1.4, 0.6,
            'TCN  (k=9)\n长时全局',
            C_BLOCK_S2, fontsize=8.5)

    # 汇合
    merge_y = 9.85
    ax.plot([branch_left_x, branch_right_x], [merge_y, merge_y],
            color='#37474f', lw=1.2, zorder=1)
    add_arrow(ax, branch_left_x, 10.25, branch_left_x, merge_y)
    add_arrow(ax, branch_right_x, 10.25, branch_right_x, merge_y)
    add_arrow(ax, inner_cx, merge_y, inner_cx, 9.5)

    # Concat
    add_box(ax, inner_x + 1.2, 8.95, inner_w - 2.4, 0.5,
            'Concat 通道维',
            '#fff59d', fontsize=9)
    add_arrow(ax, inner_cx, 8.95, inner_cx, 8.5)

    # Residual Add
    add_box(ax, inner_x + 1.2, 7.95, inner_w - 2.4, 0.5,
            'Residual  Add  ( + )',
            '#fff59d', fontsize=10, fontweight='bold')

    # Output
    add_arrow(ax, inner_cx, 7.95, inner_cx, 7.6)
    ax.text(inner_cx, 7.4, 'Output Feature',
            ha='center', va='center', fontsize=9.5,
            fontweight='bold', color='#37474f')

    # 残差捷径（右弧线，从 input 直连到 add）
    shortcut = FancyArrowPatch(
        (inner_x + inner_w - 0.05, 12.6),
        (inner_x + inner_w - 0.45, 8.20),
        connectionstyle="arc3,rad=-0.45",
        arrowstyle='-|>', mutation_scale=14,
        color='#e57373', lw=1.6, linestyle='--', zorder=1)
    ax.add_patch(shortcut)
    ax.text(inner_x + inner_w + 0.0, 10.4,
            'shortcut\n(1×1 调通道/步幅)',
            fontsize=8.5, color='#e57373', ha='left', va='center',
            style='italic', fontweight='bold')

    # ==========================================================
    # 图例（右下角空白处）
    # ==========================================================
    legend = [
        Line2D([0], [0], marker='s', color='w', markerfacecolor=C_STEM,
               markersize=11, label='Stem 初始特征头'),
        Line2D([0], [0], marker='s', color='w', markerfacecolor=C_BLOCK_S1,
               markersize=11, label='TCN-GCN Block (s=1)'),
        Line2D([0], [0], marker='s', color='w', markerfacecolor=C_BLOCK_S2,
               markersize=11, label='TCN-GCN Block (s=2 ↓T)'),
        Line2D([0], [0], marker='s', color='w', markerfacecolor=C_ATTN,
               markersize=11, label='ST-Attention'),
        Line2D([0], [0], marker='s', color='w', markerfacecolor=C_POOL,
               markersize=11, label='Global Avg Pool'),
        Line2D([0], [0], marker='s', color='w', markerfacecolor=C_FC,
               markersize=11, label='分类头 fcn'),
    ]
    leg = ax.legend(handles=legend, loc='lower right',
                    bbox_to_anchor=(0.98, 0.02),
                    fontsize=9, framealpha=0.95, ncol=2,
                    title='图例', title_fontsize=10)
    leg.get_frame().set_edgecolor('#cfd8dc')

    # ==========================================================
    # 总标题
    # ==========================================================
    fig.suptitle("ST-GCN 手势识别网络精简架构",
                 fontsize=15, fontweight='bold', y=0.97)

    # 坐标范围
    ax.set_xlim(0, 14)
    ax.set_ylim(out_y - 0.3, 14.0)
    ax.set_aspect('auto')
    ax.axis('off')

    plt.tight_layout()
    out_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                            "st_gcn_arch_diagram.png")
    plt.savefig(out_path, dpi=220, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print(f"[OK] ST-GCN 精简架构图已保存: {out_path}")


if __name__ == "__main__":
    draw()
    print("Done.")

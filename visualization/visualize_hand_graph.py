"""
手部 21 关节图结构可视化（三分区分别绘制）
==========================================
将 ST-GCN 的三种空间分区（Self-link / Inward / Outward）拆分为三个子图，
让分区差异一目了然。

输出：
    visualization/hand_graph_partitions.png   —— 三联图（推荐放论文）
    visualization/hand_graph_topology.png     —— 单图整体拓扑（保留）
    visualization/hand_graph_adjacency.png    —— 三个邻接矩阵 heatmap
"""
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import networkx as nx

plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False


# ============== build_hand_graph 的纯 numpy 实现 ==============
def _edge2mat(link, num_node):
    A = np.zeros((num_node, num_node))
    for i, j in link:
        A[j, i] = 1
    return A


def _normalize_digraph(A):
    Dl = np.sum(A, 0)
    n = A.shape[0]
    Dn = np.zeros((n, n))
    for i in range(n):
        if Dl[i] > 0:
            Dn[i, i] = Dl[i] ** -1
    return np.dot(A, Dn)


def build_hand_graph_np():
    inward = [
        (0, 1), (1, 2), (2, 3), (3, 4),
        (0, 5), (5, 6), (6, 7), (7, 8),
        (0, 9), (9, 10), (10, 11), (11, 12),
        (0, 13), (13, 14), (14, 15), (15, 16),
        (0, 17), (17, 18), (18, 19), (19, 20),
    ]
    outward = [(j, i) for (i, j) in inward]
    self_link = [(i, i) for i in range(21)]
    I = _edge2mat(self_link, 21)
    In = _normalize_digraph(_edge2mat(inward, 21))
    Out = _normalize_digraph(_edge2mat(outward, 21))
    return np.stack((I, In, Out))


# ============== 21 个关节的标准布局 ==============
POS = {
    0:  (0.00, 0.00),
    1:  (-1.10, 0.55), 2: (-1.55, 1.30), 3: (-1.85, 2.00), 4: (-2.10, 2.65),
    5:  (-0.75, 2.10), 6: (-0.85, 3.30), 7: (-0.90, 4.00), 8: (-0.95, 4.65),
    9:  (-0.05, 2.30), 10: (-0.05, 3.65), 11: (-0.05, 4.45), 12: (-0.05, 5.15),
    13: (0.65, 2.20), 14: (0.80, 3.45), 15: (0.90, 4.20), 16: (0.95, 4.85),
    17: (1.30, 1.95), 18: (1.65, 3.00), 19: (1.85, 3.70), 20: (2.00, 4.30),
}

INWARD = [
    (0, 1), (1, 2), (2, 3), (3, 4),
    (0, 5), (5, 6), (6, 7), (7, 8),
    (0, 9), (9, 10), (10, 11), (11, 12),
    (0, 13), (13, 14), (14, 15), (15, 16),
    (0, 17), (17, 18), (18, 19), (19, 20),
]
OUTWARD = [(j, i) for (i, j) in INWARD]


def _draw_skeleton_base(ax):
    """画灰色骨架作为底图（不带方向）"""
    G = nx.Graph()
    G.add_nodes_from(range(21))
    G.add_edges_from(INWARD)
    nx.draw_networkx_edges(G, POS, edge_color='#d0d0d0', width=1.2, ax=ax)


def _draw_nodes(ax, highlight_color, edgecolor='#212121'):
    nodes_x = [POS[i][0] for i in range(21)]
    nodes_y = [POS[i][1] for i in range(21)]
    ax.scatter(nodes_x, nodes_y, s=320, c=highlight_color,
               edgecolors=edgecolor, linewidths=1.4, zorder=3)
    for i, (x, y) in POS.items():
        ax.text(x, y, str(i), ha='center', va='center',
                fontsize=7.5, color='white', fontweight='bold', zorder=4)


def _common_axis(ax, title, subtitle=None):
    """主标题 + 副标题使用不同字体大小，用 ax.text 控制位置"""
    ax.set_xlim(-2.7, 2.6)
    ax.set_ylim(-0.8, 5.8)
    ax.set_aspect('equal')
    ax.axis('off')
    
    # 手动放置标题避免被紧凑布局压缩
    ax.text(0.5, 1.05, title, transform=ax.transAxes,
            ha='center', va='bottom', fontsize=15, fontweight='bold')
    if subtitle:
        ax.text(0.5, 0.98, subtitle, transform=ax.transAxes,
                ha='center', va='top', fontsize=11, color='#555')


def plot_partitions(save_path):
    """三联图：分别展示 Self-link / Inward / Outward 三个分区"""
    fig, axes = plt.subplots(1, 3, figsize=(15.5, 6.2),
                             gridspec_kw={'wspace': 0.05})

    # -------- (a) Self-link --------
    ax = axes[0]
    _draw_skeleton_base(ax)
    _draw_nodes(ax, '#37474f')
    # 绕每个节点画自环
    for i, (x, y) in POS.items():
        from matplotlib.patches import Arc
        loop = Arc((x, y + 0.30), 0.45, 0.45, angle=0,
                   theta1=0, theta2=300,
                   color='#37474f', linewidth=2.0, zorder=2)
        ax.add_patch(loop)
        # 加箭头尾巴 (修改箭头方向，使其看起来连回自身)
        ax.annotate('', xy=(x + 0.18, y + 0.11), xytext=(x + 0.22, y + 0.20),
                    arrowprops=dict(arrowstyle='-|>', color='#37474f', lw=2.0),
                    zorder=2)
    _common_axis(ax, "(a) Self-link 分区",
                 "保留节点自身特征 (A[0])")

    # -------- (b) Inward --------
    ax = axes[1]
    _draw_skeleton_base(ax)
    # 向心边：指尖→根
    for (src, tgt) in OUTWARD:  # outward 的方向是 j→i, 即指尖→根
        x1, y1 = POS[src]
        x2, y2 = POS[tgt]
        ax.annotate('', xy=(x2, y2), xytext=(x1, y1),
                    arrowprops=dict(arrowstyle='-|>', color='#1565c0',
                                    lw=2.0, shrinkA=10, shrinkB=10),
                    zorder=2)
    _draw_nodes(ax, '#1565c0')
    _common_axis(ax, "(b) Inward 分区",
                 "向心方向: 指尖 → 手腕 (A[1])")

    # -------- (c) Outward --------
    ax = axes[2]
    _draw_skeleton_base(ax)
    # 离心边：根→指尖
    for (src, tgt) in INWARD:
        x1, y1 = POS[src]
        x2, y2 = POS[tgt]
        ax.annotate('', xy=(x2, y2), xytext=(x1, y1),
                    arrowprops=dict(arrowstyle='-|>', color='#ef6c00',
                                    lw=2.0, shrinkA=10, shrinkB=10),
                    zorder=2)
    _draw_nodes(ax, '#ef6c00')
    _common_axis(ax, "(c) Outward 分区",
                 "离心方向: 手腕 → 指尖 (A[2])")

    fig.suptitle("ST-GCN 手部图结构空间划分策略 (Spatial Partition)",
                 fontsize=15, y=0.99, fontweight='bold')

    plt.tight_layout()
    plt.savefig(save_path, dpi=220, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print(f"[OK] 三分区拆分图已保存: {save_path}")


def plot_topology_combined(save_path):
    """整体拓扑图（仅画无向骨架，按手指分色）"""
    finger_colors = {
        'wrist':  '#37474f',
        'thumb':  '#e57373',
        'index':  '#64b5f6',
        'middle': '#81c784',
        'ring':   '#ffb74d',
        'pinky':  '#ba68c8',
    }
    node_colors = (
        [finger_colors['wrist']]
        + [finger_colors['thumb']] * 4
        + [finger_colors['index']] * 4
        + [finger_colors['middle']] * 4
        + [finger_colors['ring']] * 4
        + [finger_colors['pinky']] * 4
    )

    fig, ax = plt.subplots(figsize=(6.5, 7.5))
    G = nx.Graph()
    G.add_nodes_from(range(21))
    G.add_edges_from(INWARD)
    nx.draw_networkx_edges(G, POS, edge_color='#9e9e9e', width=2.0, ax=ax)
    nx.draw_networkx_nodes(G, POS, node_size=520, node_color=node_colors,
                           edgecolors='#212121', linewidths=1.4, ax=ax)
    nx.draw_networkx_labels(G, POS,
                            labels={i: str(i) for i in range(21)},
                            font_size=8, font_color='white',
                            font_weight='bold', ax=ax)

    from matplotlib.lines import Line2D
    legend = [
        Line2D([0], [0], marker='o', color='w', markerfacecolor=finger_colors['wrist'],
               markersize=10, label='手腕 (Wrist)'),
        Line2D([0], [0], marker='o', color='w', markerfacecolor=finger_colors['thumb'],
               markersize=10, label='大拇指 (Thumb)'),
        Line2D([0], [0], marker='o', color='w', markerfacecolor=finger_colors['index'],
               markersize=10, label='食指 (Index)'),
        Line2D([0], [0], marker='o', color='w', markerfacecolor=finger_colors['middle'],
               markersize=10, label='中指 (Middle)'),
        Line2D([0], [0], marker='o', color='w', markerfacecolor=finger_colors['ring'],
               markersize=10, label='无名指 (Ring)'),
        Line2D([0], [0], marker='o', color='w', markerfacecolor=finger_colors['pinky'],
               markersize=10, label='小指 (Pinky)'),
    ]
    ax.legend(handles=legend, loc='lower right', fontsize=9, framealpha=0.95)
    ax.set_title("MediaPipe 手部 21 关节拓扑结构", fontsize=13, pad=10)
    ax.set_xlim(-2.7, 2.6)
    ax.set_ylim(-0.8, 5.8)
    ax.set_aspect('equal')
    ax.axis('off')
    plt.tight_layout()
    plt.savefig(save_path, dpi=220, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print(f"[OK] 整体拓扑图已保存: {save_path}")


def plot_adjacency_heatmaps(save_path):
    A = build_hand_graph_np()
    titles = ['Self-link (自环)', 'Inward (向心)', 'Outward (离心)']
    cmaps = ['Greys', 'Blues', 'Oranges']
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    for k in range(3):
        ax = axes[k]
        im = ax.imshow(A[k], cmap=cmaps[k], aspect='equal',
                       vmin=0, vmax=max(0.3, A[k].max()))
        ax.set_title(f"A[{k}] : {titles[k]}", fontsize=12)
        ax.set_xlabel("Source Joint")
        ax.set_ylabel("Target Joint" if k == 0 else "")
        ax.set_xticks(range(0, 21, 4))
        ax.set_yticks(range(0, 21, 4))
        plt.colorbar(im, ax=ax, fraction=0.045, pad=0.04)
    fig.suptitle("ST-GCN 三分区邻接矩阵（Inward/Outward 已做入度归一化）",
                 fontsize=13, y=1.02)
    plt.tight_layout()
    plt.savefig(save_path, dpi=200, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print(f"[OK] 邻接矩阵 heatmap 已保存: {save_path}")


if __name__ == "__main__":
    out_dir = os.path.dirname(os.path.abspath(__file__))
    plot_partitions(os.path.join(out_dir, "hand_graph_partitions.png"))
    plot_topology_combined(os.path.join(out_dir, "hand_graph_topology.png"))
    plot_adjacency_heatmaps(os.path.join(out_dir, "hand_graph_adjacency.png"))
    print("Done.")

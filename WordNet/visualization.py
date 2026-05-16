# ---------------------------------------------
# 可视化邻接矩阵（使用 matplotlib）
# 输入：本地 adj_matrix_similarity.pt 和 adj_matrix_structural.pt
# 输出：两个热力图，展示 30x30 的连接结构
# ---------------------------------------------

import torch
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# 设置中文字体和绘图风格
plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['axes.unicode_minus'] = False
sns.set_style("whitegrid")

# ----------------------------
# 加载两个邻接矩阵
# ----------------------------

try:
    A_sim = torch.load('./adj_matrix_similarity.pt')      # (30, 30) 加权
    A_str = torch.load('./adj_matrix_structural.pt')      # (30, 30) 二值
except FileNotFoundError as e:
    raise FileNotFoundError(f"❌ 文件未找到：{e}\n请确保已生成两个邻接矩阵文件")

assert A_sim.shape == (30, 30) and A_str.shape == (30, 30), "❌ 矩阵形状应为 (30, 30)"

# 转为 numpy
A_sim_np = A_sim.numpy()
A_str_np = A_str.numpy()

# 标签名称（每组6个）
original_labels = ['Usability', 'Support', 'Reliability', 'Performance', 'Miscellaneous']
words_per_label = 6
n_labels = len(original_labels)

# 生成 xticklabels（可选：只标每组第一个词的位置）
tick_labels = [f"{lbl}\n({i*6})" for i, lbl in enumerate(original_labels)]
tick_pos = [i * words_per_label for i in range(n_labels)]  # [0, 6, 12, 18, 24]

# ----------------------------
# 绘图：子图 1 - 结构邻接矩阵
# ----------------------------

plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
sns.heatmap(A_str_np, cmap='Blues', cbar=True, square=True, xticklabels=False, yticklabels=False)
# 手动添加组分隔线
for pos in tick_pos[1:]:  # 在 6,12,18,24 处画线
    plt.axvline(x=pos, color='red', linewidth=2)
    plt.axhline(y=pos, color='red', linewidth=2)

plt.title("Structural Adjacency Matrix\n(Block Diagonal: Within-group Full Connection)", fontsize=10)
plt.xlabel("Node ID")
plt.ylabel("Node ID")

# ----------------------------
# 绘图：子图 2 - 语义相似度邻接矩阵
# ----------------------------

plt.subplot(1, 2, 2)
cmap = sns.diverging_palette(240, 10, as_cmap=True)  # 蓝-白-红
sns.heatmap(A_sim_np, cmap='viridis', cbar=True, square=True, xticklabels=False, yticklabels=False)

# 同样画出组边界
for pos in tick_pos[1:]:
    plt.axvline(x=pos, color='red', linewidth=2)
    plt.axhline(y=pos, color='red', linewidth=2)

plt.title("Semantic Adjacency Matrix\n(Cosine Similarity > 0.5)", fontsize=10)
plt.xlabel("Node ID")
plt.ylabel("Node ID")

# ----------------------------
# 调整布局并显示
# ----------------------------

plt.tight_layout()
plt.suptitle("Comparison of Two Adjacency Matrices (30 Nodes)", 
             fontsize=14, y=1.02, weight='bold')
plt.show()
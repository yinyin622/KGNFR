# ---------------------------------------------
# 构建邻接矩阵（完全独立运行）
# 输入：本地 node_features.pt (30, 768)
# 输出：
#   - adj_matrix_similarity.pt：基于余弦相似度
#   - adj_matrix_structural.pt：基于预定义分组结构
# ---------------------------------------------

import torch
import torch.nn.functional as F
import numpy as np

# ----------------------------
# Step 1: 加载节点特征
# ----------------------------

FEATURE_PATH = './WordNet/node_features.pt'

try:
    X = torch.load(FEATURE_PATH)  # shape: (30, 768)
except FileNotFoundError:
    raise FileNotFoundError(f"特征文件未找到：{FEATURE_PATH}\n请确保已生成 node_features.pt")

assert X.dim() == 2 and X.size(0) == 24 and X.size(1) == 768, \
    f"特征张量形状错误，期望 (30, 768)，得到 {X.shape}"

print(f"成功加载节点特征：{X.shape}")

# ----------------------------
# 方法 1：基于语义相似度的邻接矩阵（加权）
# ----------------------------

# 计算归一化余弦相似度矩阵
sim_matrix = F.cosine_similarity(
    X.unsqueeze(1),  # (30, 1, 768)
    X.unsqueeze(0),  # (1, 30, 768)
    dim=2
)  # -> (30, 30)

# 可选：阈值过滤（只保留相似度 > 0.5 的边）
threshold = 0.5
adj_sim = torch.where(sim_matrix > threshold, sim_matrix, torch.tensor(0.0))

# 确保对称（无向图）
adj_sim = (adj_sim + adj_sim.T) / 2

# 移除自环（后续 GCN 可显式添加）
adj_sim_no_self = adj_sim.clone()
adj_sim_no_self.fill_diagonal_(0.0)

# 保存
SIM_PATH = './WordNet/adj_matrix_similarity.pt'
torch.save(adj_sim_no_self, SIM_PATH)
print(f"语义邻接矩阵（加权）已保存：{SIM_PATH}")
print(f"范围: [{adj_sim_no_self.min():.3f}, {adj_sim_no_self.max():.3f}]")
print(f"非零元素: {((adj_sim_no_self > 0)).sum().item()}")

# ----------------------------
# 方法 2：基于结构分组的邻接矩阵（二值，块对角）
# ----------------------------

total_nodes = 24
n_labels = 4
words_per_label = 6

assert total_nodes == n_labels * words_per_label, "❌ 节点总数应为 4 × 6 = 30"

# 初始化结构邻接矩阵
adj_struct = torch.zeros(total_nodes, total_nodes, dtype=torch.float32)

# 规则：每个 label 对应的 6 个节点内部全连接
for i in range(n_labels):
    start_idx = i * words_per_label
    end_idx = start_idx + words_per_label
    adj_struct[start_idx:end_idx, start_idx:end_idx] = 1.0

# 移除自环
adj_struct.fill_diagonal_(0.0)

# 确保对称（虽然是对称的）
adj_struct = (adj_struct + adj_struct.T) / 2

# 保存
STRUCT_PATH = './WordNet/adj_matrix_structural.pt'
torch.save(adj_struct, STRUCT_PATH)
print(f"结构邻接矩阵（二值块对角）已保存：{STRUCT_PATH}")
print(f"每组 {words_per_label} 个节点内部全连接")
print(f"共 {n_labels} 个独立组，无跨组连接")
print(f"非零元素: {adj_struct.sum().item()}")  # 应为 5 * 6*5 = 150

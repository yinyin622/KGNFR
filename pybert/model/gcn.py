# Kipf 提出的 GCN 实现
# 层数可通过 num_layers=2 控制

import torch
import torch.nn as nn
import torch.nn.functional as F

class GCNLayer(nn.Module):
    def __init__(self, in_features, out_features):
        super().__init__()
        self.linear = nn.Linear(in_features, out_features)

    def forward(self, x, adj):
        # adj: (N, N) 对称归一化邻接矩阵（已加自环）
        # x: (N, in_features)
        support = self.linear(x)  # (N, out_features)
        output = torch.matmul(adj, support)  # (N, out_features)
        return F.relu(output)

class GCN(nn.Module):
    def __init__(self, in_features, hidden_features, out_features, num_layers=2):
        super().__init__()
        layers = []
        self.num_layers = num_layers 
        dims = [in_features] + [hidden_features] * (num_layers - 1) + [out_features]
        for i in range(num_layers):
            layers.append(GCNLayer(dims[i], dims[i+1]))
        self.layers = nn.ModuleList(layers)

    def forward(self, x, adj):
        for layer in self.layers:
            x = layer(x, adj)
        return x
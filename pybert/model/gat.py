# gat.py
# 图注意力
import torch
import torch.nn as nn
import torch.nn.functional as F

class BatchGAT(nn.Module):
    """
    支持 (B, N, D) 输入的 GAT 层（无预定义邻接矩阵，全连接）
    """
    def __init__(self, in_features, out_features, num_heads=4, dropout=0.1):
        super().__init__()
        self.num_heads = num_heads
        self.out_features = out_features
        self.dropout = dropout

        # 线性变换（所有头共享）
        self.W = nn.Linear(in_features, out_features * num_heads, bias=False)
        self.a = nn.Parameter(torch.empty(2 * out_features, num_heads, dtype=torch.float32))
        # self.a = nn.Parameter(torch.Tensor(2 * out_features, num_heads))
        self.leakyrelu = nn.LeakyReLU(0.2)
        self.reset_parameters()


    def reset_parameters(self):
        # 所有初始化都在 CPU 上进行（安全）
        nn.init.xavier_uniform_(self.W.weight)
        nn.init.xavier_uniform_(self.a)  # 使用 .data 更安全
        print(self.a.device)

    def forward(self, x):
        """
        x: (B, N, D_in)
        returns: (B, N, D_out)
        """
        B, N, D_in = x.shape
        D_out = self.out_features

        # 线性变换
        h = self.W(x).view(B, N, self.num_heads, D_out)  # (B, N, H, D_out)
        h = h.transpose(1, 2)  # (B, H, N, D_out)

        # 计算注意力系数 e_ij
        # a^T [h_i || h_j] -> (B, H, N, N)
        h_i = h.unsqueeze(3).expand(-1, -1, -1, N, -1)  # (B, H, N, N, D_out)
        h_j = h.unsqueeze(2).expand(-1, -1, N, -1, -1)  # (B, H, N, N, D_out)
        concat = torch.cat([h_i, h_j], dim=-1)  # (B, H, N, N, 2*D_out)

        assert not torch.isnan(self.a).any()

        e = torch.einsum('bhijd,dh->bhij', concat, self.a)  # (B, H, N, N)

        # 关键改进：添加缩放因子（类似 Transformer）
        e = e / (D_out ** 0.5)

        # 可选：保留或移除 LeakyReLU（建议先移除）
        # e = self.leakyrelu(e)

        # 数值稳定
        e = e - e.max(dim=-1, keepdim=True)[0]

        assert not torch.isnan(e).any()

        attention = F.softmax(e, dim=-1)

        assert not torch.isnan(attention).any()

        attention = F.dropout(attention, p=self.dropout, training=self.training)

        # 加权聚合
        h_prime = torch.matmul(attention, h)  # (B, H, N, D_out)
        h_prime = h_prime.transpose(1, 2).contiguous()  # (B, N, H, D_out)
        h_prime = h_prime.view(B, N, -1)  # (B, N, H*D_out)

        return h_prime
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import BertPreTrainedModel, BertModel

# ========== 保留你原来的 GCNLayer 和 GCN ==========
class GCNLayer(nn.Module):
    def __init__(self, in_features, out_features):
        super().__init__()
        self.linear = nn.Linear(in_features, out_features)

    def forward(self, x, adj):
        # x: (batch, N, in_features)
        # adj: (batch, N, N) 或 (N, N) —— 我们支持 batch-wise
        support = self.linear(x)  # (batch, N, out_features)
        if adj.dim() == 2:
            adj = adj.unsqueeze(0)  # (1, N, N) -> broadcast
        output = torch.bmm(adj, support)  # (batch, N, out_features)
        return F.relu(output)


class GCN(nn.Module):
    def __init__(self, in_features, hidden_features, out_features, num_layers=2):
        super().__init__()
        layers = []
        dims = [in_features] + [hidden_features] * (num_layers - 1) + [out_features]
        for i in range(num_layers):
            layers.append(GCNLayer(dims[i], dims[i+1]))
        self.layers = nn.ModuleList(layers)

    def forward(self, x, adj):
        for layer in self.layers:
            x = layer(x, adj)
        return x


# ========== 新模型：BERT + GCN for Multi-Label Classification ==========
class BertForMultiLable(BertPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.bert = BertModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

        # GCN 参数
        self.gcn_hidden = 256
        self.gcn_out = 256
        self.num_gcn_layers = 2

        self.gcn = GCN(
            in_features=config.hidden_size,
            hidden_features=self.gcn_hidden,
            out_features=self.gcn_out,
            num_layers=self.num_gcn_layers
        )

        # 分类头：从 GCN 输出聚合后映射到标签
        self.classifier = nn.Linear(self.gcn_out, config.num_labels)

        # 可选：是否使用可学习的邻接矩阵（这里我们先用固定全连接）
        self.use_learnable_adj = False  # 设为 True 可启用可学习邻接
        if self.use_learnable_adj:
            # 初始化一个可学习的邻接参数（对所有样本共享）
            max_seq_len = config.max_position_embeddings  # 通常 512
            self.adj_param = nn.Parameter(torch.randn(max_seq_len, max_seq_len))
            # 或者 per-batch 动态计算（见下文）

        self.init_weights()

    def build_adjacency_matrix(self, seq_len, device, attention_mask=None):
        """
        构建邻接矩阵（带自环的全连接图）
        可扩展为：基于 attention、kNN、依存句法等
        """
        # 方法1：固定全连接 + 自环（最简单）
        adj = torch.ones(seq_len, seq_len, device=device)
        
        # 可选：归一化（对称归一化 D^{-0.5} A D^{-0.5}）
        degree = torch.sum(adj, dim=1)
        degree_inv_sqrt = torch.pow(degree, -0.5)
        degree_inv_sqrt[torch.isinf(degree_inv_sqrt)] = 0.
        D_inv_sqrt = torch.diag(degree_inv_sqrt)
        adj_norm = D_inv_sqrt @ adj @ D_inv_sqrt

        return adj_norm

    def forward(self, input_ids, token_type_ids=None, attention_mask=None, labels=None):
        # Step 1: BERT 编码
        outputs = self.bert(
            input_ids,
            token_type_ids=token_type_ids,
            attention_mask=attention_mask
        )
        sequence_output = outputs[0]  # (batch_size, seq_len, hidden_size)
        sequence_output = self.dropout(sequence_output)

        batch_size, seq_len, hidden_size = sequence_output.shape
        device = sequence_output.device

        # Step 2: 构建邻接矩阵（每个样本独立 or 共享？）
        # 简单起见：每个样本用相同结构（全连接），但 mask 掉 padding
        adj = self.build_adjacency_matrix(seq_len, device)  # (seq_len, seq_len)

        # 如果有 attention_mask，可以 mask 掉 padding 节点的影响（可选）
        if attention_mask is not None:
            # 扩展 mask 到邻接矩阵
            mask = attention_mask.unsqueeze(-1).float()  # (batch, seq_len, 1)
            mask_matrix = mask @ mask.transpose(-1, -2)  # (batch, seq_len, seq_len)
            # 将 adj 广播并与 mask 相乘
            adj = adj.unsqueeze(0) * mask_matrix  # (batch, seq_len, seq_len)

            # 重新归一化（考虑 mask 后的度）
            degree = torch.sum(adj, dim=-1)  # (batch, seq_len)
            degree_inv_sqrt = torch.pow(degree, -0.5)
            degree_inv_sqrt[torch.isinf(degree_inv_sqrt)] = 0.
            D_inv_sqrt = torch.diag_embed(degree_inv_sqrt)  # (batch, seq_len, seq_len)
            adj = D_inv_sqrt @ adj @ D_inv_sqrt

        # Step 3: GCN 传播
        gcn_output = self.gcn(sequence_output, adj)  # (batch, seq_len, gcn_out)

        # Step 4: 聚合节点表示 → 句子表示
        # 方式1: 取 [CLS] token（index=0）
        sentence_repr = gcn_output[:, 0, :]  # (batch, gcn_out)

        # 方式2（可选）: mean pooling over valid tokens
        # if attention_mask is not None:
        #     masked_output = gcn_output * attention_mask.unsqueeze(-1)
        #     sentence_repr = masked_output.sum(dim=1) / attention_mask.sum(dim=1, keepdim=True).clamp(min=1)
        # else:
        #     sentence_repr = gcn_output.mean(dim=1)

        # Step 5: 分类
        logits = self.classifier(sentence_repr)  # (batch, num_labels)

        return logits
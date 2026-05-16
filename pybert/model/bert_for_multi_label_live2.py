# bert_for_multi_label_pair.py
import torch
import torch.nn as nn
from transformers import BertPreTrainedModel, BertModel, AutoTokenizer
from pathlib import Path
import json


class SimpleGCNLayer(nn.Module):
    def __init__(self, in_features, out_features):
        super().__init__()
        self.linear = nn.Linear(in_features, out_features)
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.linear.weight)
        if self.linear.bias is not None:
            nn.init.zeros_(self.linear.bias)

    def forward(self, x, adj_norm):
        # x: (B, N, F)
        # adj_norm: (N, N) —— 共享邻接矩阵（固定结构）
        x = torch.matmul(adj_norm, x)  # (B, N, F)
        x = self.linear(x)
        return x


class BertForMultiLable(BertPreTrainedModel):

    ALPHA = 0.2
    GCN_LAYERS = 1
    TEMPERATURE = 0.5  # 用于相似度差异化处理（0.1~5.0，越小差异变得越大）

    def __init__(self, config, alpha=ALPHA, gcn_layers=GCN_LAYERS):
        super().__init__(config)
        self.bert = BertModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier = nn.Linear(config.hidden_size, config.num_labels)
        self.alpha = alpha
        self.temperature = self.TEMPERATURE
        self.bert_model_dir = './pybert/pretrain/bert/base-uncased'

        # GCN layers
        self.gcn_layers = gcn_layers
        gcn_modules = []
        for _ in range(gcn_layers):
            gcn_modules.append(SimpleGCNLayer(config.hidden_size, config.hidden_size))
        self.gcn = nn.ModuleList(gcn_modules)

        # === 加载知识节点（必须与训练时一致）===
        # 扩展 17 节点图
        knowledge_file = Path("./WordNet/knowledge_nodes.txt")

        # 原始 4 节点图
        # knowledge_file = Path("./WordNet/error.txt")

        if knowledge_file.exists():
            with open(knowledge_file, "r", encoding="utf-8") as f:
                self.knowledge_nodes = [line.strip() for line in f if line.strip()]
            print(f"[INFO] Loaded {len(self.knowledge_nodes)} knowledge nodes from {knowledge_file}")
        else:
            self.knowledge_nodes = ['Usability', 'Support', 'Reliability', 'Performance']
            print("[WARNING] knowledge_nodes.txt not found, using default labels.")

        # 验证 num_labels 与 knowledge_nodes 中原始标签数量一致
        original_labels = ['Usability', 'Support', 'Reliability', 'Performance']
        assert config.num_labels == len(original_labels), "config.num_labels must be 4"
        self.original_labels = original_labels

        # Tokenizer 用于构造 pair 输入
        self.tokenizer = AutoTokenizer.from_pretrained(self.bert_model_dir)
        print(f"[INFO] Tokenizer loaded from: {self.bert_model_dir}")

        self.init_weights()

    def forward(self, input_ids, token_type_ids=None, attention_mask=None):
        batch_size = input_ids.size(0)
        device = input_ids.device
        K = len(self.knowledge_nodes)

        # Step 1: 原始句子特征表示（用于残差计算）
        outputs_orig = self.bert(input_ids, token_type_ids=token_type_ids, attention_mask=attention_mask)
        sent_repr = self.dropout(outputs_orig[1])  # (B, 768)

        # Step 2: 构造 (sentence, knowledge_node) pairs
        label_token_ids = []
        for label in self.knowledge_nodes:
            ids = self.tokenizer.encode(label, add_special_tokens=False)
            label_token_ids.append(ids)

        all_input_ids = []
        all_attention_masks = []
        all_token_type_ids = []

        for b in range(batch_size):
            sent_ids = input_ids[b]
            sent_len = attention_mask[b].sum().item()
            sent_part = sent_ids[:sent_len].tolist()

            for label_ids in label_token_ids:
                input_pair = [self.tokenizer.cls_token_id] + sent_part[1:] + [self.tokenizer.sep_token_id] + label_ids + [self.tokenizer.sep_token_id]
                token_type = [0] * (len(sent_part) + 1) + [1] * (len(label_ids) + 1)

                if len(input_pair) > 128:
                    input_pair = input_pair[:128]
                    token_type = token_type[:128]
                else:
                    pad_len = 128 - len(input_pair)
                    input_pair += [self.tokenizer.pad_token_id] * pad_len
                    token_type += [0] * pad_len

                all_input_ids.append(input_pair)
                all_attention_masks.append([1 if x != self.tokenizer.pad_token_id else 0 for x in input_pair])
                all_token_type_ids.append(token_type)

        pair_input_ids = torch.tensor(all_input_ids, device=device)
        pair_attention_mask = torch.tensor(all_attention_masks, device=device)
        pair_token_type_ids = torch.tensor(all_token_type_ids, device=device)

        # Step 3: 编码所有 pairs
        pair_outputs = self.bert(
            input_ids=pair_input_ids,
            attention_mask=pair_attention_mask,
            token_type_ids=pair_token_type_ids
        )
        label_reprs_flat = pair_outputs[1]  # (B * K, 768)
        label_reprs = label_reprs_flat.view(batch_size, K, -1)  # (B, K, 768)

        # Step 4: 构建图节点：[sent, node1, ..., nodeK]
        x = torch.cat([sent_repr.unsqueeze(1), label_reprs], dim=1)  # (B, 1+K, 768)

        # Step 5: ✅ 动态构建邻接矩阵 —— 基于相似度 + 温度缩放
        N = 1 + K
        # 计算 sent_repr 与每个 label_repr 的余弦相似度
        sent_norm = torch.nn.functional.normalize(sent_repr, dim=-1)       # (B, 768)
        label_norm = torch.nn.functional.normalize(label_reprs, dim=-1)   # (B, K, 768)

        # (B, 1, 768) @ (B, 768, K) -> (B, 1, K) -> (B, K)
        sim = torch.bmm(sent_norm.unsqueeze(1), label_norm.transpose(1, 2)).squeeze(1)  # (B, K)

        # ✅ 应用温度缩放：放大差异
        scaled_sim = sim / self.temperature  # (B, K)

        # 可选：是否 softmax？—— 如果希望边权重和为1，可用；否则直接用 scaled_sim
        # 这里我们采用 **非归一化的加权邻接**，但确保对称
        weights = torch.softmax(scaled_sim, dim=-1)  # 或者直接用 scaled_sim.clamp(min=0)
        # 也可以用：weights = torch.relu(scaled_sim)  # 保留正相关

        # 构建邻接矩阵 (B, N, N)
        adj = torch.zeros(batch_size, N, N, device=device)
        adj[:, 0, 0] = 1.0  # 自环

        # 填充 sent ↔ knowledge 节点的边（对称）
        for i in range(K):
            adj[:, 0, 1 + i] = weights[:, i]
            adj[:, 1 + i, 0] = weights[:, i]

        # Step 6: 对每个样本的邻接矩阵做对称归一化
        # D^{-1/2} A D^{-1/2}
        D = adj.sum(dim=-1, keepdim=True)  # (B, N, 1)
        D_inv_sqrt = D.clamp(min=1e-8).pow(-0.5)  # (B, N, 1)
        adj_norm = D_inv_sqrt * adj * D_inv_sqrt.transpose(-2, -1)  # (B, N, N)

        # Step 7: GCN propagation —— 支持 batch-wise adj
        h = x
        for i, gcn_layer in enumerate(self.gcn):
            # 手动实现 batch matmul: (B, N, F) = (B, N, N) @ (B, N, F)
            h = torch.bmm(adj_norm, h)  # (B, N, F)
            h = gcn_layer.linear(h)
            if i < len(self.gcn) - 1:
                h = torch.relu(h)

        # Step 8: 取出增强后的句子表示
        enhanced_sent = h[:, 0, :]  # (B, 768)

        # Step 9: 残差融合
        final_repr = (1 - self.alpha) * sent_repr + self.alpha * enhanced_sent

        logits = self.classifier(final_repr)  # (B, 4)

        return logits

    def save_pretrained(self, save_directory, **kwargs):
        super().save_pretrained(save_directory, **kwargs)
        config_path = Path(save_directory) / "gcn_config.json"
        with open(config_path, 'w') as f:
            json.dump({
                "alpha": self.alpha,
                "gcn_layers": self.gcn_layers,
                "temperature": self.temperature,
            }, f)

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, *model_args, **kwargs):
        kwargs.pop('alpha', None)
        kwargs.pop('gcn_layers', None)
        model = super().from_pretrained(pretrained_model_name_or_path, *model_args, **kwargs)
        return model
# bert_for_multi_label_dynamic.py
import torch
import torch.nn as nn
from transformers import BertPreTrainedModel, BertModel, AutoTokenizer
from pathlib import Path
import json
import os

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
        # adj_norm: (N, N) —— 共享邻接矩阵
        x = torch.matmul(adj_norm, x)  # (B, N, F)
        x = self.linear(x)
        return x


class BertForMultiLable(BertPreTrainedModel):

    # 20251207：改这里！
    ALPHA = 0.35
    GCN_LAYERS = 2

    def __init__(self, config, alpha=ALPHA, gcn_layers=GCN_LAYERS):
        super().__init__(config)
        self.bert = BertModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier = nn.Linear(config.hidden_size, config.num_labels)
        self.alpha = alpha

        # 手写 GCN layers
        self.gcn_layers = gcn_layers
        gcn_modules = []
        for _ in range(gcn_layers):
            gcn_modules.append(SimpleGCNLayer(config.hidden_size, config.hidden_size))
        self.gcn = nn.ModuleList(gcn_modules)

        # 固定标签顺序（必须与数据集一致！）
        self.label_texts = ['Usability', 'Support', 'Reliability', 'Performance']
        assert len(self.label_texts) == config.num_labels, "类别数量错误！"

        # 关键修复：从 config._name_or_path 加载 tokenizer
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(config._name_or_path)
            print(f"[INFO] Tokenizer loaded from: {config._name_or_path}")
        except Exception as e:
            print(f"[WARNING] Failed to load tokenizer: {e}")
            self.tokenizer = None

        self.init_weights()

    def forward(self, input_ids, token_type_ids=None, attention_mask=None):
        batch_size = input_ids.size(0)
        device = input_ids.device

        # Step 1: 原始句子表示
        outputs_orig = self.bert(input_ids, token_type_ids=token_type_ids, attention_mask=attention_mask)
        sent_repr = self.dropout(outputs_orig[1])  # (B, 768)

        # Step 2: 动态构造 [SENT] [SEP] label 批量输入
        if self.tokenizer is not None:
            sentences = [
                self.tokenizer.decode(ids, skip_special_tokens=True)
                for ids in input_ids
            ]
        else:
            # fallback：用空字符串避免崩溃（但会影响 prompt 质量）
            sentences = [""] * batch_size

        all_sentences = []
        all_labels = []
        for sent in sentences:
            for label in self.label_texts:
                all_sentences.append(sent)
                all_labels.append(label)

        # 一次性编码所有 (sentence, label) pairs
        if self.tokenizer is not None:
            pair_inputs = self.tokenizer(
                all_sentences,
                text_pair=all_labels,
                return_tensors='pt',
                padding=True,
                truncation=True,
                max_length=128,
                add_special_tokens=True
            ).to(device)
        else:
            # 最后兜底：生成全零输入（不理想，但保证不 crash）
            L = 128
            num_pairs = len(all_sentences)
            pair_inputs = {
                'input_ids': torch.zeros(num_pairs, L, dtype=torch.long, device=device),
                'attention_mask': torch.ones(num_pairs, L, dtype=torch.long, device=device),
                'token_type_ids': torch.zeros(num_pairs, L, dtype=torch.long, device=device)
            }

        # with torch.no_grad():
        pair_outputs = self.bert(**pair_inputs)
        label_reprs_flat = pair_outputs[1]  # (B * num_labels, 768)
        label_reprs = label_reprs_flat.view(batch_size, len(self.label_texts), -1)  # (B, 5, 768)

        # Step 3: 构建图节点：[sent, label1, ..., label5] → (B, 6, 768)
        x = torch.cat([sent_repr.unsqueeze(1), label_reprs], dim=1)  # (B, 6, 768)

        # Step 4: 构建全连接邻接矩阵（含自环）并归一化
        N = x.size(1)  # 6
        adj = torch.ones(N, N, device=device)
        D = adj.sum(dim=-1, keepdim=True)  # (6, 1)
        D_inv_sqrt = D.pow(-0.5)
        adj_norm = D_inv_sqrt * adj * D_inv_sqrt.t()  # (6, 6)

        # Step 5: GCN propagation
        h = x
        for i, gcn_layer in enumerate(self.gcn):
            h = gcn_layer(h, adj_norm)
            if i < len(self.gcn) - 1:
                h = torch.relu(h)

        # Step 6: 取出增强后的句子表示（第0个节点）
        enhanced_sent = h[:, 0, :]  # (B, 768)

        # Step 7: 残差融合
        final_repr = (1 - self.alpha) * sent_repr + self.alpha * enhanced_sent

        # Step 8: 分类
        logits = self.classifier(final_repr)
        return logits


    # 模型保存
    def save_pretrained(self, save_directory, **kwargs):
        super().save_pretrained(save_directory, **kwargs)
        # 保存超参数，以便检验
        config_path = Path(save_directory) / "gcn_config.json"
        with open(config_path, 'w') as f:
            json.dump({
                "alpha": self.alpha,
                "gcn_layers": self.gcn_layers,
            }, f)


    # 模型加载
    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, *model_args, **kwargs):
        # 移除可能传入的自定义参数，避免干扰父类
        kwargs.pop('alpha', None)
        kwargs.pop('gcn_layers', None)

        # 提前校验（关键！）
        config_path = Path(pretrained_model_name_or_path) / "gcn_config.json"
        if config_path.exists():
            with open(config_path, 'r') as f:
                saved_cfg = json.load(f)
            if saved_cfg.get("alpha") != cls.ALPHA or saved_cfg.get("gcn_layers") != cls.GCN_LAYERS:
                raise ValueError("配置不匹配！")

        # 使用当前代码的硬编码默认值构建模型
        model = super().from_pretrained(pretrained_model_name_or_path, 
        *model_args, **kwargs)

        # 此时 node_features, edge_weights, gcn 参数都已从 pytorch_model.bin 正确加载！

        return model


    # @classmethod
    # def from_pretrained(cls, pretrained_model_name_or_path, *model_args, **kwargs):
    #     # 移除自定义参数，避免传给父类
    #     kwargs.pop('alpha', None)
    #     kwargs.pop('gcn_layers', None)
    #     model = super().from_pretrained(pretrained_model_name_or_path, *model_args, **kwargs)

    #     # 加载额外配置
    #     config_path = Path(pretrained_model_name_or_path) / "model_config.json"
    #     if config_path.exists():
    #         with open(config_path, 'r') as f:
    #             cfg = json.load(f)
    #         model.alpha = cfg.get("alpha", 0.5)
    #         model.gcn_layers = cfg.get("gcn_layers", 1)
    #     return model
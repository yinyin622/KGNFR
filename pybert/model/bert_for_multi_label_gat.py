# bert_for_multi_label_gat.py
# 使用 GAT 模型
import torch
import torch.nn as nn
from transformers import BertPreTrainedModel, BertModel
from .gat import BatchGAT  # ← 替换为 GAT
from pathlib import Path
import json

# ----------------------------
# 知识图谱路径
# ----------------------------
# 使用：节点原型知识图谱
NODE_FEATURES_PATH = './WordNet/node_meta_features.pt'

# 注意：这里不再需要 ADJ_MATRIX_PATH！

# ----------------------------
# 主模型
# ----------------------------

class BertForMultiLable(BertPreTrainedModel):
    
    ALPHA = 0.5
    GAT_LAYERS = 1  # 请尽量使用 1 层, GAT一层应付它完全够用

    def __init__(self, config, alpha=ALPHA, gat_layers=GAT_LAYERS):
        super().__init__(config)
        self.bert = BertModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier = nn.Linear(config.hidden_size, config.num_labels)

        self.alpha = alpha

        num_heads = 4
        
        # 初始化 GAT（替代 GCN）
        self.gat_layers = nn.ModuleList([
            BatchGAT(
                in_features=768,
                out_features=768 // 4,  # 4 heads → 768
                num_heads=num_heads,
                dropout=config.hidden_dropout_prob
            ) for _ in range(gat_layers)
        ])
        self.gat_norms = nn.ModuleList([nn.LayerNorm(768) for _ in range(gat_layers)])

        # 加载知识图谱节点特征（4 个类别原型）
        node_features = torch.load(NODE_FEATURES_PATH, map_location='cpu')
        # assert node_features.shape == (4, 768), f"Expected (4, 768), got {node_features.shape}"
        self.register_buffer('node_features', node_features)  # (4, 768)
        self.num_kg_nodes = node_features.size(0)  # = 4

        # self.init_weights()

    # def init_weights(self):
    #     super().init_weights()  # 初始化 BERT 部分
    #     # 显式初始化 GAT 参数
    #     for gat_layer in self.gat_layers:
    #         gat_layer.reset_parameters()

    def forward(self, input_ids, token_type_ids=None, attention_mask=None):
        batch_size = input_ids.size(0)
        device = input_ids.device

        # Step 1: BERT 编码
        outputs = self.bert(input_ids, token_type_ids=token_type_ids, attention_mask=attention_mask)
        anchor_features = self.dropout(outputs[1])  # (B, 768)

        # Step 2: 构造 batch 特征矩阵 X ∈ (B, N+1, 768)
        kg_nodes = self.node_features.unsqueeze(0).expand(batch_size, -1, -1)  # (B, 4, 768)
        batch_x = torch.cat([kg_nodes, anchor_features.unsqueeze(1)], dim=1)   # (B, 5, 768)

        # Step 3: 多层 GAT（无需邻接矩阵！）
        x = batch_x
        for gat, norm in zip(self.gat_layers, self.gat_norms):
            x_res = x
            x = gat(x)  # (B, 5, 768)
            x = norm(x + x_res)  # 残差 + LayerNorm


        # Step 4: 取出增强后的意图锚点
        enhanced_anchor = x[:, -1, :]  # (B, 768)

        # Step 5: 残差融合
        final_feature = (1 - self.alpha) * anchor_features + self.alpha * enhanced_anchor

        # Step 6: 分类
        logits = self.classifier(final_feature)
        return logits

    def save_pretrained(self, save_directory, **kwargs):
        super().save_pretrained(save_directory, **kwargs)
        config_path = Path(save_directory) / "gat_config.json"
        with open(config_path, 'w') as f:
            json.dump({
                "alpha": self.alpha,
                "gat_layers": len(self.gat_layers),
            }, f)

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, *model_args, **kwargs):
        kwargs.pop('alpha', None)
        kwargs.pop('gat_layers', None)
        
        config_path = Path(pretrained_model_name_or_path) / "gat_config.json"
        if config_path.exists():
            with open(config_path, 'r') as f:
                saved_cfg = json.load(f)
            if saved_cfg.get("alpha") != cls.ALPHA or saved_cfg.get("gat_layers") != cls.GAT_LAYERS:
                raise ValueError("配置不匹配！")

        model = super().from_pretrained(pretrained_model_name_or_path, *model_args, **kwargs)
        return model
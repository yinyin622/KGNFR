
import torch
import torch.nn as nn
from transformers import BertPreTrainedModel, BertModel
from .gcn import GCN
from pathlib import Path
import json

# NODE_FEATURES_PATH = './WordNet/node_features.pt'
# ADJ_MATRIX_PATH = './WordNet/adj_matrix_similarity.pt'
#NODE_FEATURES_PATH = './WordNet/node_features_prompt.pt'
#ADJ_MATRIX_PATH = './WordNet/adj_matrix_prompt_similarity.pt'
# NODE_FEATURES_PATH = './WordNet/node_meta_features.pt'
# ADJ_MATRIX_PATH = './WordNet/adj_matrix_meta.pt'

class BertForMultiLable(BertPreTrainedModel):
    
    ALPHA = 0.9
    GCN_LAYERS = 1
    TEMPERATURE = 1  

    def __init__(self, config, alpha=ALPHA, gcn_layers=GCN_LAYERS):
        super().__init__(config)
        self.bert = BertModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier = nn.Linear(config.hidden_size, config.num_labels)

   
        self.alpha = alpha
        self.temperature = self.TEMPERATURE
        

        self.gcn = GCN(
            in_features=768,
            hidden_features=768,
            out_features=768,
            num_layers=gcn_layers
        )

      
        node_features = torch.load(NODE_FEATURES_PATH, map_location='cpu')
        edge_weights = torch.load(ADJ_MATRIX_PATH, map_location='cpu')

       
        self.register_buffer('node_features', node_features)           # (N, 768)
        self.register_buffer('edge_weights', edge_weights)         # (N, N)


        self.num_kg_nodes = self.node_features.size(0)

        self.init_weights()



    def enhance_anchor_kg_edges(self, batch_adj, temperature=0.5):
        """
        对 batch_adj 中 "意图锚点 ↔ 知识节点" 的边应用温度缩放 + softmax，
        以增大高相关性边的权重，抑制低相关性边。
        
        Args:
            batch_adj: (B, N+1, N+1) 邻接矩阵，其中最后一个是 anchor 节点
            temperature: 温度参数，越小越锐化
        
        Returns:
            enhanced_adj: (B, N+1, N+1) 锐化后的邻接矩阵
        """
        B, total_nodes, _ = batch_adj.shape
        N = total_nodes - 1  # 知识节点数


        enhanced_adj = batch_adj.clone()

        # 提取 anchor → KG 的边 (B, N)
        anchor_to_kg = batch_adj[:, -1, :N]  # (B, N)

       
        # 注意：这里假设原始值 ∈ [0, 1]（来自 sigmoid），但 softmax 不依赖此假设
        scaled = anchor_to_kg / temperature
        sharpened_weights = torch.softmax(scaled, dim=-1)  # (B, N)，和为1

        # 写回双向边
        enhanced_adj[:, -1, :N] = sharpened_weights
        enhanced_adj[:, :N, -1] = sharpened_weights  # 保持对称

        # 可选：保留 anchor 自环为 1
        enhanced_adj[:, -1, -1] = 1.0

        return enhanced_adj


    def forward(self, input_ids, token_type_ids=None, attention_mask=None):


        batch_size = input_ids.size(0)
        device = input_ids.device  # 确保所有张量在相同设备

        # Step 1: BERT 编码 → “意图锚点” 特征
        outputs = self.bert(input_ids, token_type_ids=token_type_ids, attention_mask=attention_mask)
        anchor_features = outputs[1]  # (B, 768)
        anchor_features = self.dropout(anchor_features)  # (B, 768)

        # Step 2: 计算相似度
        sim_matrix = torch.mm(anchor_features, self.node_features.t())  # (B, N)

        # Step 3: 构造 batch 邻接矩阵 (B, N+1, N+1)
        total_nodes = self.num_kg_nodes + 1
        batch_adj = torch.zeros(batch_size, total_nodes, total_nodes, device=device)

        # Block 1: 知识图谱内部连接（+ 自环）
        A_kg_clean = self.edge_weights.fill_diagonal_(0).clone()  # 移除可能的自环
        A_kg = A_kg_clean + torch.eye(self.num_kg_nodes, device=device)  # (N, N)
        batch_adj[:, :self.num_kg_nodes, :self.num_kg_nodes] = A_kg.unsqueeze(0)  # (B, N, N)

        # Block 2: 意图锚点 ↔ 知识节点（双向）— 使用原始相似度（或可选 relu）
        # 注意：sim_matrix 可能含负值，我们只保留正相关（可选）
        edge_weights = torch.relu(sim_matrix)  # (B, N)，将负相似度置0；也可直接用 sim_matrix
        batch_adj[:, -1, :self.num_kg_nodes] = edge_weights
        batch_adj[:, :self.num_kg_nodes, -1] = edge_weights

        # Block 3: 意图锚点自环
        batch_adj = self.enhance_anchor_kg_edges(batch_adj, self.temperature)

        # Step 4: 对称归一化（有防除零措施）
        D = batch_adj.sum(dim=-1)  # (B, N+1)
        eps = 1e-12
        D_inv_sqrt = (D + eps).pow(-0.5).unsqueeze(-1)  # (B, N+1, 1)
        batch_adj_norm = D_inv_sqrt * batch_adj * D_inv_sqrt.transpose(1, 2)  # (B, N+1, N+1)

        # Step 5: 构造特征矩阵
        batch_x = torch.cat([
            self.node_features.unsqueeze(0).expand(batch_size, -1, -1),  # (B, N, 768)
            anchor_features.unsqueeze(1)                                 # (B, 1, 768)
        ], dim=1)  # (B, N+1, 768)

        # Step 6: 批量运行 GCN
        enhanced_x = self.gcn(batch_x, batch_adj_norm)  # (B, N+1, 768)

        # Step 7: 取出意图锚点（最后一个节点）
        enhanced_anchor = enhanced_x[:, -1, :]  # (B, 768)

        # Step 8: 残差融合
        final_feature = (1 - self.alpha) * anchor_features + self.alpha * enhanced_anchor

        # Step 9: 分类
        logits = self.classifier(final_feature)

        return logits
    



    def save_pretrained(self, save_directory, **kwargs):
        super().save_pretrained(save_directory, **kwargs)
 
        config_path = Path(save_directory) / "gcn_config.json"
        with open(config_path, 'w') as f:
            json.dump({
                "alpha": self.alpha,
                "gcn_layers": self.gcn.num_layers,
                "temperature": self.temperature,
            }, f)

   
    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, *model_args, **kwargs):
        # 移除可能传入的自定义参数，避免干扰父类
        kwargs.pop('alpha', None)
        kwargs.pop('gcn_layers', None)
        kwargs.pop('temperature', None)
        
        # 提前校验（关键！）
        config_path = Path(pretrained_model_name_or_path) / "gcn_config.json"
        if config_path.exists():
            with open(config_path, 'r') as f:
                saved_cfg = json.load(f)
            if saved_cfg.get("alpha") != cls.ALPHA or saved_cfg.get("gcn_layers") != cls.GCN_LAYERS:
                raise ValueError("配置不匹配！")

        model = super().from_pretrained(pretrained_model_name_or_path, *model_args, **kwargs)
        
        # 此时 node_features, edge_weights, gcn 参数都已从 pytorch_model.bin 正确加载！

        return model


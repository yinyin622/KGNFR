# bert_for_multi_label_kg.py
import torch
import torch.nn as nn
from transformers import BertPreTrainedModel, BertModel
from .gcn import GCN
from pathlib import Path
import json

# ----------------------------
# 知识图谱路径
# ----------------------------
# 节点特征路径
NODE_FEATURES_PATH = './WordNet/node_features_prompt.pt'
# 邻接矩阵路径
ADJ_MATRIX_PATH = './WordNet/adj_matrix_prompt_similarity.pt'

# ----------------------------
# 主模型
# ----------------------------

class BertForMultiLable(BertPreTrainedModel):
    
    # 20251207: 更安全的加载方式
    # 在这里定义 
    ALPHA = 0.25
    GCN_LAYERS = 2

    # 修改为常量
    def __init__(self, config, alpha=ALPHA, gcn_layers=GCN_LAYERS):
        # 在这里修改此处的 alpha 与 gcn_layers 的值
        # 测试模型时，alpha 与 gcn_layers 的值要跟 outputs/.../gcn_config.json 一样
        # 不然会报错
        super().__init__(config)
        self.bert = BertModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier = nn.Linear(config.hidden_size, config.num_labels)

        # 特征融合的超参数
        self.alpha = alpha
        
        # 初始化 GCN
        self.gcn = GCN(
            in_features=768,
            hidden_features=768,
            out_features=768,
            num_layers=gcn_layers
        )

        # --- 关键修改：将全局变量转为 buffer ---
        # 加载知识图谱数据（只在初始化时加载一次）
        node_features = torch.load(NODE_FEATURES_PATH, map_location='cpu')
        edge_weights = torch.load(ADJ_MATRIX_PATH, map_location='cpu')

        # 注册为 buffer，自动随模型迁移设备
        self.register_buffer('node_features', node_features)           # (N, 768)
        self.register_buffer('edge_weights', edge_weights)         # (N, N)

        # 动态获取节点数量
        self.num_kg_nodes = self.node_features.size(0)

        self.init_weights()

    def forward(self, input_ids, token_type_ids=None, attention_mask=None):

        # 获取批次大小（Batch_size，简称 B）
        batch_size = input_ids.size(0)
        device = input_ids.device  # 确保所有张量在相同设备

        # Step 1: BERT 编码 → “意图锚点” 特征
        outputs = self.bert(input_ids, token_type_ids=token_type_ids, attention_mask=attention_mask)
        anchor_features = outputs[1]  # (B, 768)
        anchor_features = self.dropout(anchor_features)  # (B, 768)

        # Step 2: 计算相似度
        sim_matrix = torch.mm(anchor_features, self.node_features.t())  # (B, N)
        edge_weights = torch.sigmoid(sim_matrix)  # (B, N)

        # Step 3: 构造 batch 邻接矩阵 (B, N+1, N+1)
        total_nodes = self.num_kg_nodes + 1
        batch_adj = torch.zeros(batch_size, total_nodes, total_nodes, device=device)

        # Block 1: 知识图谱内部连接（+ 自环）
        A_kg_clean = self.edge_weights.fill_diagonal_(0).clone()  # 移除可能的自环
        A_kg = A_kg_clean + torch.eye(self.num_kg_nodes, device=device)  # (N, N)
        batch_adj[:, :self.num_kg_nodes, :self.num_kg_nodes] = A_kg.unsqueeze(0)  # (B, N, N)

        # Block 2: 意图锚点 ↔ 知识节点（双向）
        batch_adj[:, -1, :self.num_kg_nodes] = edge_weights  # anchor → knowledge
        batch_adj[:, :self.num_kg_nodes, -1] = edge_weights  # knowledge → anchor

        # Block 3: 意图锚点自环
        batch_adj[:, -1, -1] = 1.0

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

    # 重写模型保存
    # def save_pretrained(self, save_directory, **kwargs):
    #     """
    #     保存整个模型：BERT + GCN + classifier + buffers
    #     """
    #     super().save_pretrained(save_directory, **kwargs)

    #     # 1. 保存 GCN 权重
    #     gcn_path = Path(save_directory) / "gcn.bin"
    #     torch.save(self.gcn.state_dict(), gcn_path)

    #     # 2. 保存节点特征和边：额外 buffer（node_features, edge_weights）
    #     buffer_path = Path(save_directory) / "kg_buffers.bin"
    #     torch.save({
    #         'node_features': self.node_features.data,
    #         'edge_weights': self.edge_weights.data,
    #     }, buffer_path)

    #     # 3. 保存超参数（alpha, gcn_layers）
    #     config_path = Path(save_directory) / "model_config.json"
    #     with open(config_path, 'w') as f:
    #         json.dump({
    #             "alpha": self.alpha,
    #             "gcn_layers": self.gcn.num_layers,
    #         }, f)


    # # 模型保存
    # def save_pretrained(self, save_directory, **kwargs):
    #     super().save_pretrained(save_directory, **kwargs)

    #     # 保存 GCN 和 buffers
    #     torch.save(self.gcn.state_dict(), Path(save_directory) / "gcn.bin")
    #     torch.save({
    #         'node_features': self.node_features.data,
    #         'edge_weights': self.edge_weights.data,
    #     }, Path(save_directory) / "kg_buffers.bin")

    #     # 保存超参数（alpha, gcn_layers）
    #     config_path = Path(save_directory) / "model_config.json"
    #     with open(config_path, 'w') as f:
    #         json.dump({
    #             "alpha": self.alpha,
    #             "gcn_layers": self.gcn.num_layers,
    #         }, f)

    # 删除所有 gcn.bin / kg_buffers.bin 相关代码！

    # 20251207
    # 简化模型保存
    def save_pretrained(self, save_directory, **kwargs):
        super().save_pretrained(save_directory, **kwargs)
        # 只需保存超参数（用于校验）
        config_path = Path(save_directory) / "gcn_config.json"
        with open(config_path, 'w') as f:
            json.dump({
                "alpha": self.alpha,
                "gcn_layers": self.gcn.num_layers,
            }, f)

    # 20251207
    # 完善模型checkpoint加载
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

        model = super().from_pretrained(pretrained_model_name_or_path, *model_args, **kwargs)
        
        # 此时 node_features, edge_weights, gcn 参数都已从 pytorch_model.bin 正确加载！

        return model


    # 模型加载
    # @classmethod
    # def from_pretrained(cls, pretrained_model_name_or_path, *model_args, **kwargs):
    #     # 移除可能传入的自定义参数，避免干扰父类
    #     kwargs.pop('alpha', None)
    #     kwargs.pop('gcn_layers', None)

    #     # 使用当前代码的硬编码值构造模型（通过 __init__ 默认参数）
    #     model = super().from_pretrained(pretrained_model_name_or_path, *model_args, **kwargs)

    #     # 步骤1：校验当前设置 是否与 model_config.json 是否一致
    #     config_path = Path(pretrained_model_name_or_path) / "model_config.json"
    #     if config_path.exists():
    #         with open(config_path, 'r') as f:
    #             saved_cfg = json.load(f)
    #         saved_alpha = saved_cfg.get("alpha", cls.ALPHA)
    #         saved_gcn_layers = saved_cfg.get("gcn_layers", cls.GCN_LAYERS)

    #         if saved_alpha != cls.ALPHA or saved_gcn_layers != cls.GCN_LAYERS:
    #             raise ValueError(
    #                 f"错误，你现在加载的模型是:\n"
    #                 f"  alpha={saved_alpha}, gcn_layers={saved_gcn_layers}\n"
    #                 f"但是你在上面设置的是:\n"
    #                 f"  alpha={cls.ALPHA}, gcn_layers={cls.GCN_LAYERS}\n"
    #                 f"你要修改上面的设置，要与现在加载的模型一致，不然跑不了！"
    #             )
    #     else:
    #         print(f"[WARNING] 没有在 {pretrained_model_name_or_path} 下找到 model_config.json ！")

    #     # 步骤2：加载 GCN 权重（结构已由硬编码保证一致）
    #     gcn_path = Path(pretrained_model_name_or_path) / "gcn.bin"
    #     if gcn_path.exists():
    #         state_dict = torch.load(gcn_path, map_location=model.device)
    #         model.gcn.load_state_dict(state_dict)
    #     else:
    #         print(f"[WARNING] 找不到 gcn.bin，GCN 模型参数未正确加载！")

    #     # 步骤3：加载 KG buffers（node_features & edge_weights）
    #     buffer_path = Path(pretrained_model_name_or_path) / "kg_buffers.bin"
    #     if buffer_path.exists():
    #         buffers = torch.load(buffer_path, map_location=model.device)
    #         # 使用 .data.copy_() 安全覆盖 buffer 内容
    #         model.node_features.data.copy_(buffers['node_features'])
    #         model.edge_weights.data.copy_(buffers['edge_weights'])
    #     else:
    #         print(f"[INFO] 找不到 kg_buffers.bin，知识图谱未正确加载！")

    #     return model


    # 重新模型重新加载
    # @classmethod
    # def from_pretrained(cls, pretrained_model_name_or_path, *model_args, **kwargs):
    #     """
    #     加载完整模型：包括 BERT、GCN、buffers 和超参数
    #     """
    #     # 确保 kwargs 中没有 alpha/gcn_layers（避免意外传入）
    #     kwargs.pop('alpha', None)
    #     kwargs.pop('gcn_layers', None)

    #     # 1. 先加载超参数
    #     config_path = Path(pretrained_model_name_or_path) / "model_config.json"
    #     if config_path.exists():
    #         import json
    #         with open(config_path, 'r') as f:
    #             model_config = json.load(f)
    #         alpha = model_config.get("alpha", 0.5)
    #         gcn_layers = model_config.get("gcn_layers", 2)
    #     else:
    #         alpha = kwargs.pop('alpha', 0.5)
    #         gcn_layers = kwargs.pop('gcn_layers', 2)

    #     # 2. 初始化模型（会调用 __init__）
    #     model = super().from_pretrained(pretrained_model_name_or_path, *model_args, **kwargs)

    #     # 3. 手动设置 alpha（__init__ 里用了 alpha）
    #     model.alpha = alpha

    #     # 4. 加载 GCN 权重
    #     gcn_path = Path(pretrained_model_name_or_path) / "gcn.bin"
    #     if gcn_path.exists():
    #         model.gcn.load_state_dict(torch.load(gcn_path, map_location=model.device))

    #     # 5. 加载 buffers
    #     buffer_path = Path(pretrained_model_name_or_path) / "kg_buffers.bin"
    #     if buffer_path.exists():
    #         buffers = torch.load(buffer_path, map_location=model.device)
    #         model.node_features.data.copy_(buffers['node_features'])
    #         model.edge_weights.data.copy_(buffers['edge_weights'])

    #     return model

    # @classmethod
    # def from_pretrained(cls, pretrained_model_name_or_path, *model_args, **kwargs):
    #     # 关键：不要试图读取旧的 alpha/gcn_layers
    #     # 直接让 __init__ 使用当前代码的默认值！

    #     # 确保 kwargs 中没有 alpha/gcn_layers（避免意外传入）
    #     kwargs.pop('alpha', None)
    #     kwargs.pop('gcn_layers', None)

    #     # 调用父类，它会用 config 调用 cls(config)，即你的 __init__(config)
    #     model = super().from_pretrained(pretrained_model_name_or_path, *model_args, **kwargs)

    #     # 加载 GCN 权重（必须与当前 gcn_layers 一致！）
    #     gcn_path = Path(pretrained_model_name_or_path) / "gcn.bin"
    #     if gcn_path.exists():
    #         state_dict = torch.load(gcn_path, map_location=model.device)
    #         # 可选：增加 strict=False 防止轻微 mismatch（不推荐用于层数变化）
    #         model.gcn.load_state_dict(state_dict)

    #     # 加载 KG buffers
    #     buffer_path = Path(pretrained_model_name_or_path) / "kg_buffers.bin"
    #     if buffer_path.exists():
    #         buffers = torch.load(buffer_path, map_location=model.device)
    #         model.node_features.data.copy_(buffers['node_features'])
    #         model.edge_weights.data.copy_(buffers['edge_weights'])
 
        # return model

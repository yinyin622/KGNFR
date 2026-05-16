import pandas as pd
import torch
import numpy as np
from transformers import BertTokenizer, BertModel
from tqdm import tqdm
import os

# ----------------------------
# Step 1: 配置与加载本地 BERT
# ----------------------------

# 请确保路径正确
bert_model_dir = 'D:/#第一个喵喵/NFRKG_4class/pybert/pretrain/bert/base-uncased'

print("Loading local BERT model and tokenizer...")
tokenizer = BertTokenizer.from_pretrained(bert_model_dir)
model = BertModel.from_pretrained(bert_model_dir)
model.eval()

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)

# ----------------------------
# Step 2: 数据加载与预处理
# ----------------------------

DATA_PATH = "pybert/dataset/review_origin_train.csv"
OUTPUT_PATH = "node_meta_features.pt"
CATEGORIES = ['Usa', 'Sup', 'Rel', 'Per']
MAX_LENGTH = 128
BATCH_SIZE = 32

print("Loading dataset...")
df = pd.read_csv(DATA_PATH)

# 确保必要列存在
for col in CATEGORIES:
    assert col in df.columns, f"Missing column: {col}"

# 去除空 review
df = df.dropna(subset=['review']).reset_index(drop=True)
reviews = df['review'].tolist()

# 收集每个类别的正样本索引
category_indices = {cat: [] for cat in CATEGORIES}
for idx, row in df.iterrows():
    for cat in CATEGORIES:
        if row[cat] == 1:
            category_indices[cat].append(idx)

# ----------------------------
# Step 3: 批量提取 BERT [CLS] 嵌入（无 prompt，直接 encode review）
# ----------------------------

print("Tokenizing all reviews...")
encodings = tokenizer(
    reviews,
    truncation=True,
    padding=True,
    max_length=MAX_LENGTH,
    return_tensors='pt'
)

print("Extracting BERT [CLS] embeddings...")
all_cls_embeddings = []
total = len(reviews)
with torch.no_grad():
    for i in tqdm(range(0, total, BATCH_SIZE)):
        batch = {k: v[i:i+BATCH_SIZE].to(device) for k, v in encodings.items()}
        outputs = model(**batch)
        cls_emb = outputs.last_hidden_state[:, 0, :]  # [batch_size, 768]
        all_cls_embeddings.append(cls_emb.cpu())

all_cls_embeddings = torch.cat(all_cls_embeddings, dim=0)  # [N, 768]

# ----------------------------
# Step 4: 构建类别原型（策略 A）
# ----------------------------

prototype_list = []
for cat in CATEGORIES:
    indices = category_indices[cat]
    if len(indices) == 0:
        print(f"Warning: No positive samples for category '{cat}'. Using zero vector.")
        proto = torch.zeros(768)
    else:
        selected = all_cls_embeddings[indices]  # [M, 768]
        proto = torch.mean(selected, dim=0)     # [768]
    prototype_list.append(proto)

node_features = torch.stack(prototype_list, dim=0)  # [4, 768]

# ----------------------------
# Step 5: 保存结果
# ----------------------------

torch.save(node_features, OUTPUT_PATH)
print(f"\n Successfully saved node meta features to: {os.path.abspath(OUTPUT_PATH)}")
print(f"Shape: {node_features.shape}")
print(f"Category order: {CATEGORIES}")
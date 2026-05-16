import nltk
import torch
import numpy as np
from transformers import BertTokenizer, BertModel
from nltk.corpus import wordnet as wn
import random

# ----------------------------
# Step 1: 加载本地 BERT 模型和分词器
# ----------------------------

bert_model_dir = 'D:/#第一个喵喵/NFRKG_20251104/NFRKG/pybert/pretrain/bert/base-uncased'

tokenizer = BertTokenizer.from_pretrained(bert_model_dir)
model = BertModel.from_pretrained(bert_model_dir)
model.eval()  # 推理模式

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)

# ----------------------------
# Step 2: 增强版 WordNet 扩展函数（返回词 + 类型）
# ----------------------------

def get_related_words_with_type(word, depth=2, total_related=5):
    """
    返回格式: List[Tuple[word, type]]，type ∈ {'hyper', 'hypo', 'synonym'}
    最后添加 (word, 'self') 作为第六项
    """
    candidates = []  # 改用 list 保持顺序，存 (word, type)

    all_synsets = wn.synsets(word)
    if not all_synsets:
        return [(word, 'self')] * 6  # 极端情况

    primary_synset = all_synsets[0]

    # 1. 上位词（更抽象） 苹果 <-- 水果
    for hyper in primary_synset.closure(lambda s: s.hypernyms(), depth):
        lemma = hyper.lemmas()[0].name().replace('_', ' ')
        if lemma.lower() != word.lower():
            candidates.append((lemma, 'hyper'))

    # 2. 下位词（更具体） 水果 --> 苹果
    for hypo in primary_synset.closure(lambda s: s.hyponyms(), depth):
        lemma = hypo.lemmas()[0].name().replace('_', ' ')
        if lemma.lower() != word.lower():
            candidates.append((lemma, 'hypo'))

    # 3. 同义词 番茄 <-> 西红柿
    for lemma in primary_synset.lemmas():
        name = lemma.name().replace('_', ' ')
        if name.lower() != word.lower():
            candidates.append((name, 'synonym'))

    # 4. 补充：其他 synsets 的词
    if len(candidates) < total_related:
        for synset in all_synsets[1:]:
            for lemma in synset.lemmas():
                name = lemma.name().replace('_', ' ')
                if name.lower() != word.lower():
                    candidates.append((name, 'synonym'))
            for hyper in synset.closure(lambda s: s.hypernyms(), 1):
                lemma = hyper.lemmas()[0].name().replace('_', ' ')
                if lemma.lower() != word.lower():
                    candidates.append((lemma, 'hyper'))
            for hypo in synset.closure(lambda s: s.hyponyms(), 1):
                lemma = hypo.lemmas()[0].name().replace('_', ' ')
                if lemma.lower() != word.lower():
                    candidates.append((lemma, 'hypo'))

    # 去重：按小写去重，保留第一个出现的
    seen = set()
    unique_candidates = []
    for w, t in candidates:
        key = (w.lower(), t)
        if key not in seen:
            seen.add(key)
            unique_candidates.append((w, t))

    # 随机打乱并取前5个
    random.shuffle(unique_candidates)
    selected = unique_candidates[:total_related]

    # 如果不足5个，用已有词循环补（避免拼接）
    pool = unique_candidates if unique_candidates else [(f"{word}_related", 'synonym')]
    while len(selected) < total_related:
        for item in pool:
            if len(selected) >= total_related:
                break
            if (item[0].lower(), item[1]) not in seen:
                seen.add((item[0].lower(), item[1]))
                selected.append(item)

    # 添加原词作为第六个，类型为 'self'
    final_list = selected + [(word, 'self')]
    assert len(final_list) == 6, f"Expected 6 items, got {len(final_list)}"
    return final_list

# ----------------------------
# Step 3: 编码文本为向量
# ----------------------------

def encode_word(word):
    inputs = tokenizer(word, return_tensors='pt', padding=True, truncation=True, max_length=16)
    inputs = {k: v.to(device) for k, v in inputs.items()}
    with torch.no_grad():
        outputs = model(**inputs)
    embedding = outputs.last_hidden_state[:, 0, :]  # (1, 768)
    return embedding.cpu()

# ----------------------------
# Step 4: 主流程
# ----------------------------

original_labels = ['Usability', 'Support', 'Reliability', 'Performance']

all_embeddings = []
expanded_words = []  # 存储 "label ~arrow~ word" 字符串

# 定义不同类型对应的箭头符号
ARROW_MAP = {
    'hyper':   '<--',   # 上位词：抽象化
    'hypo':    '-->',   # 下位词：具体化
    'synonym': '<->',   # 同义词
    'self':    '==='   # 自身节点
}

for label in original_labels:
    related_items = get_related_words_with_type(label)
    print(f"Label '{label}' -> Expanded: {[f'{w}({t})' for w,t in related_items]}")
    for word, word_type in related_items:
        vec = encode_word(word)
        all_embeddings.append(vec)
        arrow = ARROW_MAP.get(word_type, '->')
        expanded_words.append(f"{label} {arrow} {word}")

# 拼接成 (30, 768)
embedding_tensor = torch.cat(all_embeddings, dim=0)
print(f"✅ Final embeddings shape: {embedding_tensor.shape}")  # torch.Size([30, 768])

# ----------------------------
# Step 5: 保存结果
# ----------------------------

output_path = './WordNet/node_features.pt'
torch.save(embedding_tensor, output_path)

words_info_path = './WordNet/word_mapping.txt'
with open(words_info_path, 'w', encoding='utf-8') as f:
    for line in expanded_words:
        f.write(line + '\n')

print(f"✅ Embeddings saved to: {output_path}")
print(f"✅ Word mapping saved to: {words_info_path}")
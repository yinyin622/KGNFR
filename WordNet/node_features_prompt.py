import nltk
import torch
import random
from transformers import BertTokenizer, BertModel
from nltk.corpus import wordnet as wn

# ----------------------------
# Step 1: 加载本地 BERT 模型和分词器
# ----------------------------

bert_model_dir = './pybert/pretrain/bert/base-uncased'
# bert_model_dir = 'D:/#第一个喵喵/NFRKG_20251104/NFRKG/pybert/pretrain/bert/base-uncased'

tokenizer = BertTokenizer.from_pretrained(bert_model_dir)
model = BertModel.from_pretrained(bert_model_dir)
model.eval()  # 推理模式

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)

# ----------------------------
# Step 2: 增强版 WordNet 扩展函数（保持不变）
# ----------------------------

def get_related_words_with_type(word, depth=2, total_related=5):
    candidates = []
    all_synsets = wn.synsets(word)
    if not all_synsets:
        return [(word, 'self')] * 6

    primary_synset = all_synsets[0]

    for hyper in primary_synset.closure(lambda s: s.hypernyms(), depth):
        lemma = hyper.lemmas()[0].name().replace('_', ' ')
        if lemma.lower() != word.lower():
            candidates.append((lemma, 'hyper'))

    for hypo in primary_synset.closure(lambda s: s.hyponyms(), depth):
        lemma = hypo.lemmas()[0].name().replace('_', ' ')
        if lemma.lower() != word.lower():
            candidates.append((lemma, 'hypo'))

    for lemma in primary_synset.lemmas():
        name = lemma.name().replace('_', ' ')
        if name.lower() != word.lower():
            candidates.append((name, 'synonym'))

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

    seen = set()
    unique_candidates = []
    for w, t in candidates:
        key = (w.lower(), t)
        if key not in seen:
            seen.add(key)
            unique_candidates.append((w, t))

    random.shuffle(unique_candidates)
    selected = unique_candidates[:total_related]

    pool = unique_candidates if unique_candidates else [(f"{word}_related", 'synonym')]
    while len(selected) < total_related:
        for item in pool:
            if len(selected) >= total_related:
                break
            if (item[0].lower(), item[1]) not in seen:
                seen.add((item[0].lower(), item[1]))
                selected.append(item)

    final_list = selected + [(word, 'self')]
    assert len(final_list) == 6, f"Expected 6 items, got {len(final_list)}"
    return final_list

# ----------------------------
# Step 3: 使用 Prompt 编码单词（关键修改！）
# ----------------------------

# 定义任务导向的 prompt 模板（可调整）
PROMPTS = [
    "The software should be {}.",
    "This requirement specifies that the system must be {}.",
    "It is a {} requirement for the system.",
    "The system needs to have high {}.",
    "This is a non-functional requirement about {}.",
    "Users expect the system to be {}."
]

def encode_word_with_prompt(word, use_average=True):
    """
    对给定单词使用多个 prompt 进行 BERT 编码。
    :param word: 输入词（如 'Usability'）
    :param use_average: 是否对多个 prompt 取平均；若 False，则返回多个向量（用于扩展节点）
    :return: (1, 768) 或 (num_prompts, 768)
    """
    embeddings = []
    for prompt in PROMPTS:
        text = prompt.format(word)
        inputs = tokenizer(
            text,
            return_tensors='pt',
            padding=True,
            truncation=True,
            max_length=32  # prompt 较短，32 足够
        )
        inputs = {k: v.to(device) for k, v in inputs.items()}
        with torch.no_grad():
            outputs = model(**inputs)
        cls_emb = outputs.last_hidden_state[:, 0, :]  # (1, 768)
        embeddings.append(cls_emb.cpu())
    
    if use_average:
        # 对多个 prompt 取平均 → 更鲁棒的“任务语境”表示
        avg_emb = torch.mean(torch.stack(embeddings, dim=0), dim=0)  # (1, 768)
        return avg_emb
    else:
        # 返回多个向量（可用于每个 prompt 作为一个独立节点）
        return torch.cat(embeddings, dim=0)  # (num_prompts, 768)

# ----------------------------
# Step 4: 主流程（适配新编码方式）
# ----------------------------

original_labels = ['Usability', 'Support', 'Reliability', 'Performance']

all_embeddings = []
expanded_words = []

ARROW_MAP = {
    'hyper':   '<--',
    'hypo':    '-->',
    'synonym': '<->',
    'self':    '==='
}

for label in original_labels:
    related_items = get_related_words_with_type(label)
    print(f"Label '{label}' -> Expanded: {[f'{w}({t})' for w,t in related_items]}")
    for word, word_type in related_items:
        # 使用 prompt 编码（默认取平均）
        vec = encode_word_with_prompt(word, use_average=True)  # (1, 768)
        all_embeddings.append(vec)
        arrow = ARROW_MAP.get(word_type, '->')
        expanded_words.append(f"{label} {arrow} {word}")

# 拼接成 (30, 768)
embedding_tensor = torch.cat(all_embeddings, dim=0)
print(f"✅ Final embeddings shape: {embedding_tensor.shape}")  # 应仍为 [30, 768]

# ----------------------------
# Step 5: 保存结果
# ----------------------------

output_path = './WordNet/node_features_prompt.pt'
torch.save(embedding_tensor, output_path)

words_info_path = './WordNet/word_mapping_prompt.txt'
with open(words_info_path, 'w', encoding='utf-8') as f:
    for line in expanded_words:
        f.write(line + '\n')

print(f"✅ Prompt-based embeddings saved to: {output_path}")
print(f"✅ Word mapping saved to: {words_info_path}")
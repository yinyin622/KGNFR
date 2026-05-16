import nltk

# 下载资源（首次运行需要）
nltk.download('wordnet')
nltk.download('omw-1.4')

from nltk.corpus import wordnet as wn

def get_related_words(word, depth=2):
    synsets = wn.synsets(word)
    related = set()
    for syn in synsets:
        # 添加上位词（更抽象的概念）
        for hyper in syn.closure(lambda s: s.hypernyms(), depth):
            related.add(hyper.name().split('.')[0])  # 取词干
        # 添加下位词（更具体的例子）
        for hypo in syn.closure(lambda s: s.hyponyms(), depth):
            related.add(hypo.name().split('.')[0])
        # 添加同义词
        for lemma in syn.lemmas():
            related.add(lemma.name().replace('_', ' '))
    return list(related)

# 示例：扩展 "security"
security_concepts = get_related_words("security")
print(security_concepts[:10])
# 输出可能包含: ['safety', 'protection', 'defense', 'shield', 'guard', ...]
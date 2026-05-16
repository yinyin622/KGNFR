import nltk

# 下载知识图谱资源（只有首次运行需要）
# nltk.download('wordnet')
# nltk.download('omw-1.4')

from nltk.corpus import wordnet as wn

# 词语数量
num_words = 5

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
print(security_concepts[:num_words])
# 输出可能包含: ['safety', 'protection', 'defense', 'shield', 'guard', ...]

# 上述用到的GCN过程中，邻接矩阵都是运行时运算的，既然我的特征向量都存在本地了，干脆写个代码，把邻接矩阵这个torch张量存在本地好了，等到要用的时候再加载。我已经把图的节点特征向量（torch shape = (36, 768)，即36个节点的特征向量，之前漏掉）存在了本地的pt文件，现在我有两种构建这个邻接矩阵的思路：1、预定义（二值化），
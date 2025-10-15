# 第 1 章 : 文本表示与词向量

自然语言处理（NLP）的第一步是把文本转换为计算机可以理解的形式，也就是数字表示。第 1 章会带你从最简单的文本编码开始，一步步理解为什么要用词向量，以及如何训练和使用它们。

---

## 1.1 学习目标

完成本章后，你将能：

1. 理解文本数字化的概念
2. 掌握 One-Hot、TF-IDF、Word2Vec 等文本表示方法
3. 理解 Word2Vec 的 CBOW 和 Skip-gram 原理
4. 能用 Python 实现文本表示和词向量训练
5. 可视化词向量，理解语义关系

---

## 1.2 文本数字化的概念

### 为什么要数字化

计算机只能理解数字，而文本是字符组成的自然语言。为了让模型处理文本，需要把词或句子表示为数值。

文本表示的质量直接影响 NLP 模型的性能：

* **简单表示**（One-Hot）易实现，但无法捕捉语义
* **稠密表示**（Word2Vec / GloVe）可以表达词之间的相似性

---

### 文本表示方式概览

| 方法               | 核心思想       | 优点     | 缺点          |
| ---------------- | ---------- | ------ | ----------- |
| One-Hot          | 每个词一个唯一向量  | 简单，易实现 | 高维稀疏，无法表达语义 |
| TF-IDF           | 基于词频和逆文档频率 | 强调区分词  | 无上下文信息      |
| Word2Vec / GloVe | 将词嵌入低维向量空间 | 捕获语义关系 | 需要训练语料      |

---

## 1.3 文本预处理

在文本表示之前，需要做一些清洗和处理：

1. **分词**：把句子拆成词
2. **小写化**：英文统一小写
3. **去标点和特殊符号**
4. **去停用词**：删除高频但无意义词（如 "the", "is"）
5. **词形还原或词干化**（可选）

**示例（英文文本）**：

```python
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords

nltk.download('punkt')
nltk.download('stopwords')

text = "Natural Language Processing (NLP) is fascinating and fun!"
tokens = word_tokenize(text.lower())  # 分词并小写化
filtered = [w for w in tokens if w.isalpha() and w not in stopwords.words('english')]
print(filtered)
```

输出：

```
['natural', 'language', 'processing', 'nlp', 'fascinating', 'fun']
```

---

## 1.4 One-Hot 表示

### 概念

* 为每个词分配唯一向量
* 向量长度 = 词表大小
* 该词位置为 1，其余为 0

### 数学表示

$$
\text{one_hot}(w_i) = [0,0,...,1,...,0]
$$

### 示例

词表：`["I", "love", "NLP"]`

| 词      | One-Hot |
| ------ | ------- |
| "I"    | [1,0,0] |
| "love" | [0,1,0] |
| "NLP"  | [0,0,1] |

### Python 实现

```python
import numpy as np

vocab = ["I", "love", "NLP"]
word_to_idx = {word:i for i, word in enumerate(vocab)}

def one_hot(word):
    vec = np.zeros(len(vocab))
    vec[word_to_idx[word]] = 1
    return vec

print(one_hot("love"))  # 输出: [0. 1. 0.]
```

> 缺点：高维稀疏，无法表示词语之间的语义相似性。

---

## 1.5 TF-IDF（Term Frequency - Inverse Document Frequency）

### 概念

衡量词在文档中的重要性：

* **TF（词频）**：词在文档中出现频率
* **IDF（逆文档频率）**：词在语料中出现越频繁，权重越低

### 数学公式

$$
\text{tf}(t,d) = \frac{\text{count}(t,d)}{\text{total words in } d}
$$

$$
\text{idf}(t) = \log \frac{N}{1 + df_t}
$$

$$
\text{tfidf}(t,d) = \text{tf}(t,d) \times \text{idf}(t)
$$

* $N$：语料库中文档总数
* $df_t$：包含词 $t$ 的文档数

### Python 示例

```python
from sklearn.feature_extraction.text import TfidfVectorizer

docs = ["I love NLP", "NLP is fascinating", "I enjoy learning NLP"]
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(docs)
print(vectorizer.get_feature_names_out())
print(X.toarray())
```

> 优点：弱化高频词影响
> 缺点：无法捕捉上下文语义

---

## 1.6 词向量（Word Embedding）

### 1.6.1 概念

* 将词映射到低维向量空间
* 稠密向量表示，能捕获语义相似性
* 示例：

$$
\text{vec}("king") - \text{vec}("man") + \text{vec}("woman") \approx \text{vec}("queen")
$$

### 1.6.2 Word2Vec 原理

Word2Vec 是一种经典词嵌入方法，有两种训练策略：

1. **CBOW（Continuous Bag-of-Words）**

   * 根据上下文预测中心词
   * 举例：句子 “I love NLP”

     * 上下文 = ["I", "NLP"]
     * 中心词 = "love"

   **公式**：

   $$
   \max \sum_t \log P(w_t | \text{context}(w_t))
   $$

2. **Skip-gram**

   * 根据中心词预测上下文
   * 举例：中心词 = "love"，上下文 = ["I", "NLP"]

   **公式**：

   $$
   \max \sum_{t} \sum_{-c \le j \le c, j\ne 0} \log P(w_{t+j}|w_t)
   $$

> CBOW 对小语料表现更稳定，Skip-gram 对低频词更敏感。

---

### 1.6.3 Word2Vec Python 实现

```python
from gensim.models import Word2Vec

sentences = [["I","love","nlp"], ["nlp","is","fun"], ["I","enjoy","learning","nlp"]]

# Skip-gram 模型训练
model = Word2Vec(sentences, vector_size=50, window=2, min_count=1, sg=1, epochs=100)

# 查看词向量
print("词向量 nlp:\n", model.wv['nlp'])

# 找相似词
print("与 nlp 相似的词:", model.wv.most_similar('nlp'))
```

---

### 1.6.4 词向量可视化

```python
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

words = ['I','love','nlp','fun','learning']
vecs = [model.wv[w] for w in words]

pca = PCA(n_components=2)
vecs_2d = pca.fit_transform(vecs)

plt.figure(figsize=(6,6))
for i, word in enumerate(words):
    plt.scatter(vecs_2d[i,0], vecs_2d[i,1])
    plt.text(vecs_2d[i,0]+0.01, vecs_2d[i,1]+0.01, word)
plt.show()
```

> 你会看到语义相关的词更接近，比如 “nlp” 和 “learning”。

---

## 1.7 本章小结

1. 文本表示是 NLP 的基础，影响模型效果
2. One-Hot 和 TF-IDF 简单，但无法捕捉语义
3. Word2Vec（CBOW / Skip-gram）通过上下文学习词向量
4. 词向量能在低维空间表示语义关系
5. 可视化词向量帮助理解语义距离

---

如果你希望，我可以接着帮你生成 **第 1 章完整 Jupyter Notebook 版本**，包含：

* 文本预处理
* One-Hot、TF-IDF、Word2Vec 实战
* 词向量可视化

这样你可以直接运行、修改、实验，学习体验会更直观。

你希望我生成这个 notebook 吗？

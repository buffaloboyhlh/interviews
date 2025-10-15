# 🧩 第 1 章：认识 NLTK

### 什么是 NLTK？

NLTK（Natural Language Toolkit）是 Python 最经典的 NLP 教学与实验库。它提供了：

* 上百个语料库（corpus）和词典资源（如 WordNet）；
* 各类文本处理工具（分词、标注、命名实体识别等）；
* 统计模型和分类算法（朴素贝叶斯、决策树等）；
* 强大的可视化功能（句法树、频率分布等）。

简而言之，NLTK 就像是 NLP 的“积木盒”——你可以用它拼出各种自然语言处理系统。

### 安装与配置

```bash
pip install nltk
```

第一次使用：

```python
import nltk
nltk.download('all')  # 或按需下载部分资源
```

---

# 🧩 第 2 章：文本预处理（Text Preprocessing）

语言模型的输入必须是干净、结构化的文本。
而原始文本通常充满噪音：标点符号、HTML 标签、大小写不统一……
这一步的目标是把自然语言转化为“计算机能理解的词序列”。

### 2.1 分词（Tokenization）

```python
from nltk.tokenize import word_tokenize, sent_tokenize

text = "Natural Language Processing lets computers understand human language."
print(sent_tokenize(text))
print(word_tokenize(text))
```

输出：

```
['Natural Language Processing lets computers understand human language.']
['Natural', 'Language', 'Processing', 'lets', 'computers', 'understand', 'human', 'language', '.']
```

### 2.2 停用词（Stopwords）

停用词指如 “the”、“is”、“and” 这类语义贡献很小的常用词。

```python
from nltk.corpus import stopwords

stop_words = set(stopwords.words('english'))
tokens = [w for w in word_tokenize(text.lower()) if w.isalpha() and w not in stop_words]
print(tokens)
```

### 2.3 词干提取（Stemming）与词形还原（Lemmatization）

两种词形标准化方式：

```python
from nltk.stem import PorterStemmer, WordNetLemmatizer

stemmer = PorterStemmer()
lemmatizer = WordNetLemmatizer()

print(stemmer.stem("studies"))     # stem: study
print(lemmatizer.lemmatize("studies", pos='v'))  # lemma: study
```

---

# 🧩 第 3 章：词性标注（POS Tagging）

词性标注让计算机知道每个词在句子中的语法功能。

```python
import nltk
sentence = "The quick brown fox jumps over the lazy dog."
tokens = nltk.word_tokenize(sentence)
tagged = nltk.pos_tag(tokens)
print(tagged)
```

输出：

```
[('The', 'DT'), ('quick', 'JJ'), ('brown', 'JJ'), ('fox', 'NN'), ...]
```

常见标签：

* NN：名词
* JJ：形容词
* VB：动词
* RB：副词

NLTK 的词性标注器底层是一个 **隐马尔可夫模型（HMM）**。

---

# 🧩 第 4 章：命名实体识别（NER）

NER = Named Entity Recognition，用来识别文本中的实体，如人名、地点、组织等。

```python
from nltk import ne_chunk

sentence = "Elon Musk founded SpaceX in California."
tokens = nltk.word_tokenize(sentence)
tags = nltk.pos_tag(tokens)
tree = ne_chunk(tags)
print(tree)
```

输出树中会标注实体类别，如：

```
(ORGANIZATION SpaceX)
(GPE California)
(PERSON Elon Musk)
```

---

# 🧩 第 5 章：句法分析（Parsing）

句法分析是语言理解的“结构”部分。

### 例：使用上下文无关文法（CFG）

```python
from nltk import CFG
from nltk.parse import ChartParser

grammar = CFG.fromstring("""
S -> NP VP
NP -> DT NN
VP -> VB NP
DT -> 'the'
NN -> 'cat' | 'dog'
VB -> 'chased' | 'saw'
""")

parser = ChartParser(grammar)
sentence = ['the', 'dog', 'chased', 'the', 'cat']

for tree in parser.parse(sentence):
    print(tree)
    tree.draw()
```

这会绘制出一棵句法树，让你看到“主语-谓语-宾语”的结构。

---

# 🧩 第 6 章：语义分析（WordNet）

NLTK 内置 WordNet —— 一个庞大的英语语义网。

```python
from nltk.corpus import wordnet as wn

word = 'car'
synsets = wn.synsets(word)
print(synsets[0].definition())       # 定义
print(synsets[0].lemmas())           # 同义词
print(synsets[0].hypernyms())        # 上位词
print(synsets[0].hyponyms())         # 下位词
```

你可以利用 WordNet 实现：

* **同义词扩展**（synonym expansion）
* **语义相似度计算**
* **概念层级分析**

---

# 🧩 第 7 章：文本特征提取与分类

### 7.1 词频统计

```python
from nltk import FreqDist

text = "Data science is the study of data using statistics and machine learning."
tokens = word_tokenize(text.lower())
fdist = FreqDist(tokens)
print(fdist.most_common(5))
fdist.plot(20)
```

### 7.2 简单文本分类（朴素贝叶斯）

使用 NLTK 自带的电影评论语料：

```python
from nltk.corpus import movie_reviews
from nltk.classify import NaiveBayesClassifier

def extract_features(words):
    return {word: True for word in words}

data = [(extract_features(movie_reviews.words(fileid)), category)
        for category in movie_reviews.categories()
        for fileid in movie_reviews.fileids(category)]

train, test = data[:1900], data[1900:]
classifier = NaiveBayesClassifier.train(train)
print("Accuracy:", nltk.classify.accuracy(classifier, test))
classifier.show_most_informative_features(10)
```

---

# 🧩 第 8 章：进阶与综合应用

### 8.1 搭配分析（Collocations）

找出词语搭配频繁的组合：

```python
from nltk.collocations import BigramCollocationFinder
from nltk.metrics import BigramAssocMeasures

tokens = word_tokenize("She sells sea shells by the sea shore.")
finder = BigramCollocationFinder.from_words(tokens)
print(finder.nbest(BigramAssocMeasures.likelihood_ratio, 5))
```

### 8.2 关键词提取与共现网络

你可以基于词频、TF-IDF、互信息（PMI）构建词图，探索语义结构。

### 8.3 主题建模（LDA）

虽然 NLTK 不直接实现 LDA，但你可以用它预处理文本，然后交给 gensim 训练主题模型。


# 🧩 第 9 章：情感分析（Sentiment Analysis）

情感分析是 NLP 的经典任务之一。目标是判断文本的情绪极性（正面、负面、中性）。

NLTK 提供了两条路径：

1. 用自带语料 + 传统分类器（如朴素贝叶斯）
2. 用内置的 VADER 模型（专为社交媒体短文本优化）

---

## 9.1 用电影评论语料训练朴素贝叶斯情感分类器

```python
import nltk
from nltk.corpus import movie_reviews
from nltk.classify import NaiveBayesClassifier
from nltk.classify.util import accuracy

def extract_features(words):
    return {word: True for word in words}

# 加载语料库
documents = [(extract_features(movie_reviews.words(fileid)), category)
             for category in movie_reviews.categories()
             for fileid in movie_reviews.fileids(category)]

train_set, test_set = documents[:1900], documents[1900:]
classifier = NaiveBayesClassifier.train(train_set)

print("Accuracy:", accuracy(classifier, test_set))
classifier.show_most_informative_features(10)
```

输出示例：

```
Accuracy: 0.82
Most Informative Features
   outstanding = True           pos : neg = 9.0 : 1.0
   awful = True                 neg : pos = 8.0 : 1.0
   ...
```

**原理简述**：
朴素贝叶斯分类器基于词语的条件概率：
[
P(\text{label}|\text{words}) \propto P(\text{label}) \times \prod_i P(w_i|\text{label})
]
假设词语独立（这是“朴素”的地方），在大语料上效果仍然惊人地稳健。

---

## 9.2 使用 VADER 做社交媒体情感分析

VADER（Valence Aware Dictionary for sEntiment Reasoning）是一个**基于词典 + 规则**的情感分析器，尤其擅长处理带表情符号、缩写、感叹号的短文本（如推特、评论）。

```python
from nltk.sentiment import SentimentIntensityAnalyzer

sia = SentimentIntensityAnalyzer()

sentences = [
    "I love this movie! It's amazing 😊",
    "The plot was terrible and boring...",
    "Not bad, but not great either."
]

for s in sentences:
    print(s, "→", sia.polarity_scores(s))
```

输出：

```
I love this movie! ... → {'neg': 0.0, 'neu': 0.3, 'pos': 0.7, 'compound': 0.85}
The plot was terrible ... → {'neg': 0.6, 'neu': 0.4, 'pos': 0.0, 'compound': -0.78}
```

`compound` 分数在 [-1, 1] 间，表示整体情绪强度。
VADER 不需要训练，适合快速分析推特、评论、弹幕等文本。

---

## 9.3 用 NLTK 构建情感词典

如果想自定义情感分析规则，你可以从 WordNet 构建自己的**情绪词典**。

```python
from nltk.corpus import wordnet as wn

positive = ['good', 'happy', 'excellent', 'fortunate', 'correct', 'superior']
negative = ['bad', 'sad', 'terrible', 'poor', 'wrong', 'inferior']

def get_synonyms(word):
    syns = set()
    for s in wn.synsets(word):
        for lemma in s.lemmas():
            syns.add(lemma.name())
    return syns

pos_lexicon = set()
neg_lexicon = set()
for w in positive: pos_lexicon.update(get_synonyms(w))
for w in negative: neg_lexicon.update(get_synonyms(w))

print("Positive words:", list(pos_lexicon)[:10])
print("Negative words:", list(neg_lexicon)[:10])
```

你可以基于这些词集合做词频统计或 TF-IDF 计算，从而手工构造“情感得分”。

---

# 🧩 第 10 章：NLTK 与现代 NLP（BERT / GPT 的衔接）

NLTK 是传统 NLP 的基石，
而现代 NLP 模型（如 BERT、GPT、T5）代表的是**深度语义理解的新时代**。
这一章讲如何把 NLTK 的数据管线与深度模型结合。

---

## 10.1 使用 NLTK 做前处理，送入 Transformer 模型

```python
from nltk.tokenize import word_tokenize
from transformers import pipeline

# 1. 用 NLTK 做分词、清洗
text = "Natural language processing is fascinating but complex!"
tokens = [w.lower() for w in word_tokenize(text) if w.isalpha()]
print("Cleaned tokens:", tokens)

# 2. 用现代模型分析
analyzer = pipeline("sentiment-analysis")
print(analyzer(text))
```

输出：

```
Cleaned tokens: ['natural', 'language', 'processing', 'is', 'fascinating', 'but', 'complex']
[{'label': 'POSITIVE', 'score': 0.9998}]
```

NLTK 负责**预处理与特征化**；Transformers 负责**语义建模与推理**。
这正是现代 NLP 的最佳实践。

---

## 10.2 从词袋到词向量的演进

NLTK 主要处理**符号级文本**：单词、标点、句法树。
但在深度学习中，我们需要**数值化表示**（word embeddings）。

词袋模型（Bag-of-Words, BoW） → Word2Vec → GloVe → Transformer Embeddings
这是一条“从统计到语义”的演化路径。

可以用 NLTK 构造 BoW 特征：

```python
from nltk import FreqDist
from sklearn.feature_extraction.text import CountVectorizer

corpus = ["I love NLP", "NLP loves deep learning"]
vectorizer = CountVectorizer()
X = vectorizer.fit_transform(corpus)
print(vectorizer.get_feature_names_out())
print(X.toarray())
```

输出：

```
['deep' 'learning' 'love' 'loves' 'nlp']
[[0 0 1 0 1]
 [1 1 0 1 1]]
```

---

## 10.3 语义相似度：用 WordNet + 预训练模型结合

```python
from nltk.corpus import wordnet as wn

dog = wn.synset('dog.n.01')
cat = wn.synset('cat.n.01')
car = wn.synset('car.n.01')

print("dog-cat:", dog.wup_similarity(cat))
print("dog-car:", dog.wup_similarity(car))
```

输出：

```
dog-cat: 0.857
dog-car: 0.6
```

WordNet 的相似度基于语义层级距离。
若配合 BERT 词嵌入（transformers），你可以构建混合模型，实现：

* 概念层级的推理（WordNet）
* 语境化语义距离（BERT Embedding）

---

## 10.4 现代 NLP 与 NLTK 的哲学差别

| 层面   | NLTK              | Transformers   |
| ---- | ----------------- | -------------- |
| 核心思想 | 语言规则 + 统计         | 语义表征 + 神经网络    |
| 数据形式 | 符号（tokens, trees） | 向量（embeddings） |
| 可解释性 | 强                 | 弱              |
| 精度   | 中                 | 极高             |
| 适用场景 | 教学、可视化、基础NLP      | 商业应用、大模型微调     |

NLTK 教你理解语言的结构；
Transformer 让机器“感受”语言的语义。
二者结合，才是真正的“精通 NLP”。

---

## 🎓 完整学习建议路线

1. **掌握语言学基础：**
   分词 → 词性标注 → 句法树
2. **理解统计 NLP：**
   朴素贝叶斯、TF-IDF、搭配分析
3. **学习语义网络：**
   WordNet、相似度计算、同义词扩展
4. **过渡到深度模型：**
   了解 word2vec、Transformer、BERT
5. **综合实践项目：**
   情感分析、关键词提取、主题建模

---

NLTK 就像自然语言世界的显微镜——
透过它，你能看到语言的结构、规律与奇妙的混沌。
而当你把它与现代神经网络结合，你就能在计算机中“重建意义”。

---

是否希望我在此基础上，继续扩展出一份
**「NLTK 实战项目集」**（例如：文本分类、舆情分析、关键词提取、知识图谱构建等实战教程）？


# NLP (自然语言处理)

## 一、基础教程

### 🌍 一、NLP 是什么？

**自然语言处理（Natural Language Processing, NLP）**
是一门让计算机能够**理解、生成和处理人类语言**的科学。

它融合了：

* 语言学（语法、语义、上下文）
* 计算机科学（算法、工程实现）
* 机器学习 / 深度学习（模型训练）

---

### 🧩 二、NLP 的核心任务

| 类别   | 任务               | 示例                      |
| ---- | ---------------- | ----------------------- |
| 文本分类 | 情感分析、垃圾邮件识别      | “这家餐厅真棒！” → 积极          |
| 序列标注 | 命名实体识别（NER）、词性标注 | “张三 在 北京” → 人名 / 地名     |
| 文本匹配 | 相似度计算、语义检索       | “天气预报”和“今日气象”相似         |
| 文本生成 | 翻译、摘要、对话         | “Translate: 你好 → Hello” |
| 问答系统 | 抽取式 / 生成式 QA     | “中国首都是哪？”→ 北京           |
| 语言建模 | 下一词预测            | “我今天很____” → 高兴         |

---

### 🧠 三、文本预处理基础

1. **分词（Tokenization）**

   * 英文：空格、标点分割
   * 中文：jieba、pkuseg、HanLP
   * 子词（Subword）：BPE、WordPiece、Unigram

2. **停用词（Stopwords）**：去掉高频无意义词（如“的”“了”“and”）

3. **词形还原（Lemmatization）**

4. **向量化（Vectorization）**：将文本转为数字表示

   * One-hot encoding
   * Bag-of-Words (BoW)
   * TF-IDF
   * Word Embedding (Word2Vec, GloVe)

---

### 🔣 四、词向量（Word Embedding）

**目标**：将每个词映射为稠密向量，使相似词语向量接近。
$$
\text{Embedding}: \text{Word} \to \mathbb{R}^d
$$

##### 1️⃣ Word2Vec

* 模型结构：Skip-gram / CBOW
* Skip-gram 目标函数：
  $$
  \max_\theta \sum_{t=1}^T \sum_{-c \le j \le c, j\neq 0} \log P(w_{t+j} \mid w_t)
  $$

其中：
$$
P(w_O \mid w_I) = \frac{\exp(v_{w_O}^\top v_{w_I})}{\sum_{w}\exp(v_w^\top v_{w_I})}
$$

##### 2️⃣ GloVe

* 基于词共现矩阵 $X_{ij}$ 的对数回归：
  $$
  J = \sum_{i,j=1}^{V} f(X_{ij})(w_i^T \tilde{w}_j + b_i + \tilde{b}*j - \log X*{ij})^2
  $$

---

### 🔁 五、语言模型（Language Model）

预测一个句子中词语的概率分布：

$$
P(w_1, w_2, ..., w_T) = \prod_{t=1}^{T} P(w_t \mid w_1, ..., w_{t-1})
$$

##### 1️⃣ N-gram 模型

近似假设：
$$
P(w_t \mid w_1,...,w_{t-1}) \approx P(w_t \mid w_{t-n+1},...,w_{t-1})
$$

##### 2️⃣ 神经语言模型

用神经网络（RNN / Transformer）代替概率表：
$$
h_t = f(W_{xh}x_t + W_{hh}h_{t-1})
$$
$$
P(w_t \mid h_t) = \text{softmax}(Wh_t)
$$

---

### 🔂 六、序列模型（RNN / LSTM / GRU）

#### 1️⃣ RNN

递归结构：
$$
h_t = \tanh(W_{xh}x_t + W_{hh}h_{t-1})
$$
但存在梯度消失/爆炸问题。

#### 2️⃣ LSTM（长短期记忆网络）

通过“门控机制”保留长期依赖：
$$
f_t = \sigma(W_f[x_t,h_{t-1}]+b_f)
$$
$$
i_t = \sigma(W_i[x_t,h_{t-1}]+b_i)
$$
$$
o_t = \sigma(W_o[x_t,h_{t-1}]+b_o)
$$
$$
c_t = f_t*c_{t-1}+i_t*\tanh(W_c[x_t,h_{t-1}]+b_c)
$$
$$
h_t = o_t*\tanh(c_t)
$$

---

### ⚡ 七、Transformer 架构

Transformer 取代 RNN，成为 NLP 的核心模型。

#### Self-Attention 公式：

$$
\text{Attention}(Q,K,V)=\text{softmax}\left(\frac{QK^\top}{\sqrt{d_k}}\right)V
$$

每个词向量与序列中其他词交互，捕获全局依赖。

**优势：**

* 并行计算
* 长程依赖捕捉能力强
* 更易扩展（预训练大模型）

---

### 🧬 八、预训练语言模型（PLM）

| 模型            | 架构              | 目标任务         | 应用        |
| ------------- | --------------- | ------------ | --------- |
| **BERT**      | Encoder         | 掩码语言模型 (MLM) | 分类、NER、QA |
| **GPT**       | Decoder         | 自回归生成        | 对话、写作     |
| **T5 / BART** | Encoder–Decoder | 文本到文本        | 翻译、摘要、问答  |

---

### 🧰 九、NLP 任务与模型对应

| 任务    | 输入输出      | 模型            | 损失函数                |
| ----- | --------- | ------------- | ------------------- |
| 文本分类  | 文本 → 类别   | BERT + Linear | CrossEntropy        |
| 序列标注  | 文本 → 标签序列 | BERT + CRF    | NLL                 |
| 机器翻译  | 句子 → 句子   | Transformer   | CrossEntropy        |
| 摘要生成  | 文本 → 摘要   | BART / T5     | NLL                 |
| QA 问答 | 文本 → 答案   | BERT / T5     | Span / CrossEntropy |

---

### 🧪 十、实战示例：中文情感分类（BERT 微调）

```python
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments
from datasets import load_dataset

model_name = "bert-base-chinese"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2)

# 加载数据
dataset = load_dataset("csv", data_files={"train": "train.csv", "test": "test.csv"})
dataset = dataset.map(lambda e: tokenizer(e["text"], padding="max_length", truncation=True), batched=True)

args = TrainingArguments(
    output_dir="./output",
    per_device_train_batch_size=16,
    num_train_epochs=3,
    evaluation_strategy="epoch",
    learning_rate=2e-5
)

trainer = Trainer(model=model, args=args,
                  train_dataset=dataset["train"],
                  eval_dataset=dataset["test"])

trainer.train()
```

---

### 📊 十一、评估指标

| 任务类型 | 指标                              |
| ---- | ------------------------------- |
| 分类   | Accuracy, Precision, Recall, F1 |
| 序列标注 | Token-level / Entity-level F1   |
| 生成   | BLEU, ROUGE                     |
| 语言模型 | Perplexity                      |
| 检索   | MRR, Recall@k, NDCG@k           |

---

### ⚙️ 十二、优化与微调技巧

* 学习率：$1e^{-5} \sim 5e^{-5}$
* warmup + linear decay 调度
* Dropout、权重衰减防止过拟合
* 冻结底层，先训练上层
* 小样本：使用 **LoRA / PEFT / Prompt-tuning**

---

### 🧠 十三、工程与部署

* 推理优化：FP16 / 量化 / ONNX / TensorRT
* 服务化：FastAPI + TorchServe
* 检索增强（RAG）：LLM + 向量数据库（FAISS / Chroma）
* 模型监控：延迟、准确率漂移、日志分析

---

### 🧭 十四、学习路径建议

| 阶段              | 学习目标                 | 推荐实践           |
| --------------- | -------------------- | -------------- |
| 1️⃣ NLP 基础      | 理解词袋、TF-IDF、Word2Vec | 做文本分类          |
| 2️⃣ 序列模型        | 掌握 RNN / LSTM        | 做情感分析、NER      |
| 3️⃣ Transformer | 理解 Self-Attention    | 实现机器翻译         |
| 4️⃣ 预训练模型       | 使用 BERT / GPT        | 做 QA、摘要        |
| 5️⃣ 工程实践        | 模型优化与部署              | FastAPI + ONNX |

---

### 📚 十五、推荐学习资源

* 书籍：

  * 《Speech and Language Processing》—— Jurafsky & Martin
  * 《Deep Learning for NLP》—— Yoav Goldberg
  * 《自然语言处理综论》—— 周明

* 开源课程：

  * CS224N（Stanford NLP）
  * Hugging Face Transformers 官方教程

---

### ✅ 十六、总结

| 模块    | 关键词                     |
| ----- | ----------------------- |
| 文本表示  | Tokenization, Embedding |
| 序列建模  | RNN, LSTM, Attention    |
| 现代架构  | Transformer, BERT, GPT  |
| 任务映射  | 分类 / 生成 / QA            |
| 优化与部署 | 微调、蒸馏、ONNX              |

---

如果你想进一步学习，我可以为你生成一个分阶段系统课程：

> 📘「从零开始学 NLP」系列
> 第 1 章：文本表示与词向量
> 第 2 章：RNN/LSTM 序列模型
> 第 3 章：Transformer 与 Self-Attention
> 第 4 章：BERT / GPT 原理与微调
> 第 5 章：NLP 实战与部署

是否希望我帮你制定这个 **系统化 NLP 学习路线表（附练习任务和代码）**？

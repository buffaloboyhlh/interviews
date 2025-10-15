
# 第 3 章 — Transformer 与 Self-Attention

Transformer 是现代 NLP 的核心架构，广泛应用于 BERT、GPT 等模型。它的最大特点是**彻底摆脱了 RNN/LSTM 的顺序计算**，通过 **Self-Attention** 同时处理整个序列，实现高效并行和全局上下文建模。

---

## 3.1 学习目标

完成本章后，你将能够：

1. 理解 Transformer 的整体架构（Encoder 与 Decoder）
2. 掌握 Self-Attention 的原理、公式与直观理解
3. 理解 Multi-Head Attention 和位置编码
4. 用 PyTorch 实现基本的 Self-Attention 和 Transformer
5. 理解 Transformer 相较于 RNN 的优势

---

## 3.2 Transformer 概览

### 3.2.1 架构组成

Transformer 由 **Encoder** 和 **Decoder** 两部分组成：

* **Encoder**：处理输入序列，生成上下文表示
* **Decoder**：接收 Encoder 输出，生成目标序列

**Encoder 核心模块**：

1. Multi-Head Self-Attention
2. 前馈全连接网络（Feed-Forward Network）
3. 残差连接 + 层归一化

**Decoder 核心模块**：

1. Masked Multi-Head Self-Attention（避免看到未来信息）
2. Encoder-Decoder Attention（将输入序列信息与当前生成序列对齐）
3. 前馈全连接网络 + 残差连接

> Encoder-Decoder 架构常用于机器翻译，单独 Encoder（如 BERT）用于理解任务，单独 Decoder（如 GPT）用于生成任务。

---

## 3.3 Self-Attention 原理

Self-Attention 是 Transformer 的核心，它让每个词可以**关注序列中所有其他词**，从而捕获全局上下文。

### 3.3.1 输入与输出

* 输入序列：$X = [x_1, x_2, ..., x_n]$，每个 $x_i$ 是词向量
* 输出序列：$Z = [z_1, z_2, ..., z_n]$，每个 $z_i$ 是上下文向量

> 直观理解：Self-Attention 就像每个词都在问“在理解我自己的意义时，其他词的重要性是多少”，然后根据权重整合信息。

---

### 3.3.2 Self-Attention 计算公式

1. **生成 Query、Key、Value 向量**：

$$
Q = X W^Q, \quad K = X W^K, \quad V = X W^V
$$

* $W^Q, W^K, W^V$：可学习的权重矩阵
* $Q$：Query（提问）
* $K$：Key（回答的关键）
* $V$：Value（实际信息）

2. **计算注意力分数**：

$$
\text{Attention}(Q,K,V) = \text{softmax}\Big(\frac{Q K^\top}{\sqrt{d_k}}\Big) V
$$

* $d_k$：Key 向量维度，用 $\sqrt{d_k}$ 缩放避免分数过大
* $Q K^\top$：衡量 Query 与每个 Key 的相似度
* softmax：将相似度转换为权重

> 直观理解：每个词对序列中所有词的“关注程度”被量化，得到加权信息。

---

### 3.3.3 Self-Attention 举例

句子：**"The cat sat on the mat"**

* Query: "cat"
* Key/Value: 所有词
* Attention 权重可能显示：

  * "sat": 0.4
  * "mat": 0.3
  * "The": 0.05

> 说明“cat”会更关注与其语义相关的词，“sat”和“mat”的权重较高。

---

## 3.4 Multi-Head Attention

单个注意力头可能捕捉的信息有限，**Multi-Head Attention** 用多个注意力头捕捉不同语义关系：

$$
\text{MultiHead}(Q,K,V) = \text{Concat}(\text{head}_1, ..., \text{head}_h) W^O
$$

* 每个 head 有独立的 $W^Q, W^K, W^V$
* 可以在不同子空间关注不同信息
* 最后通过 $W^O$ 整合多头信息

> 类比：一群专家分别关注序列的不同角度，然后汇总意见。

---

## 3.5 位置编码（Positional Encoding）

Transformer 不像 RNN 那样自然感知顺序，因此需要显式位置编码：

$$
PE_{(pos,2i)} = \sin\Big(\frac{pos}{10000^{2i/d_\text{model}}}\Big), \quad
PE_{(pos,2i+1)} = \cos\Big(\frac{pos}{10000^{2i/d_\text{model}}}\Big)
$$

* $pos$：词在序列的位置
* $i$：向量维度索引

> 直观理解：正弦/余弦波不同频率编码位置，使模型区分顺序，同时允许插值预测。

---

## 3.6 前馈全连接网络（Feed-Forward）

每个 Encoder/Decoder 层还包含一个前馈网络：

$$
\text{FFN}(x) = \text{ReLU}(x W_1 + b_1) W_2 + b_2
$$

* 独立处理每个位置
* 增加非线性表达能力
* 配合残差连接和 LayerNorm

---

## 3.7 残差连接与层归一化

每一层使用残差连接和 Layer Normalization：

$$
\text{Output} = \text{LayerNorm}(x + \text{Sublayer}(x))
$$

* 防止梯度消失
* 加快训练收敛
* 保持信息流通顺畅

---

## 3.8 Transformer Python 示例（PyTorch）

```python
import torch
import torch.nn as nn

# 输入：batch_size=2, seq_len=5, embedding_dim=512
x = torch.randn(2,5,512)

# Multi-Head Attention
mha = nn.MultiheadAttention(embed_dim=512, num_heads=8, batch_first=True)
out, attn_weights = mha(x, x, x)

print("输出形状:", out.shape)        # (2,5,512)
print("注意力权重形状:", attn_weights.shape)  # (2,8,5,5)
```

> 注意力权重可以可视化，观察每个词关注序列中哪些词。

---

## 3.9 Transformer 优势

| 特性    | RNN      | Transformer   |
| ----- | -------- | ------------- |
| 并行计算  | 否，必须顺序处理 | 是，全序列并行       |
| 长距离依赖 | 难捕捉      | 易捕捉，全局注意力     |
| 训练速度  | 慢        | 快             |
| 表达能力  | 有限       | 强，多头注意力捕捉复杂语义 |
| 适用任务  | 小规模序列    | 大规模预训练 & 生成   |

---

## 3.10 Transformer 直观理解

* Self-Attention：每个词“看”整个序列，找出相关信息
* Multi-Head Attention：多个“专家”，捕捉不同语义
* 前馈网络 + 残差连接：处理复杂非线性关系，同时保证信息流
* 位置编码：告诉模型词的顺序

> Transformer 的并行处理和全局注意力让模型能够快速理解长文本语义。

---

## 3.11 本章小结

1. Transformer 彻底改变了 NLP 序列处理方式
2. Self-Attention 是核心思想，实现全局上下文建模
3. Multi-Head Attention 提升了语义捕捉能力
4. 位置编码解决序列顺序问题
5. PyTorch 提供高效实现，可直接实验和可视化

---

我可以帮你生成 **第 3 章完整 Jupyter Notebook**，包含：

* Self-Attention 可视化
* Multi-Head Attention 示例
* 位置编码可视化
* 简单 Transformer 前向传播实验

这样你可以直接运行实验，更直观理解 Transformer 工作原理。

你希望我生成这个 notebook 吗？

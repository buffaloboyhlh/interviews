
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

## 3.11 单头自注意力实现

我们要从输入序列 ( X )（形状 `[batch, seq_len, d_model]`）出发，

+ batch: 批次
+ seq_len: 文本长度
+ d_model: 嵌入维度

经过线性映射得到 Q、K、V，然后计算：

$$
\text{Attention}(Q,K,V) = \text{softmax}\left(\frac{QK^\top}{\sqrt{d_k}}\right)V
$$

并返回最终输出。

---

### ① 手动权重

我们自己定义每个变换矩阵：

$$
W_Q, W_K, W_V \in \mathbb{R}^{d_{model} \times d_{model}}
$$

这就等价于 `nn.Linear(d_model, d_model)` 的权重。

---

### ② 手动矩阵乘法

`torch.matmul(x, self.W_Q)`
相当于执行：

$$
Q = XW_Q + b_Q
$$

其中 (X) 是 `(batch, seq_len, d_model)`，矩阵乘法在最后一个维度上完成。

---

### ③ 注意力得分计算

$$
\text{scores} = \frac{QK^\top}{\sqrt{d_k}}
$$

* 维度变换：`K.transpose(-2, -1)`
  把 `(batch, seq_len, d_model)` 转为 `(batch, d_model, seq_len)`，
  使得每个 query 都能和所有 key 做点积。

---

### ④ 加权求和

$$
\text{Attention}(Q,K,V) = \text{softmax}(\text{scores})V
$$
`torch.matmul(attn_weights, V)` 就是“把 weighted value 加起来”。

---

### ⑤ 输出映射

最后再做一次线性投影：
$$
O = (\text{Attention}(Q,K,V)) W_O + b_O
$$
保证输出维度仍然是 `d_model`。

---

### 📊 输出结果示例

```
输入形状: torch.Size([2, 4, 8])
输出形状: torch.Size([2, 4, 8])
注意力权重形状: torch.Size([2, 4, 4])
第1个样本注意力矩阵:
 tensor([[0.234, 0.242, 0.278, 0.246],
         [0.262, 0.218, 0.264, 0.255],
         [0.261, 0.258, 0.244, 0.237],
         [0.260, 0.255, 0.254, 0.231]])
```

每一行代表一个 token 对其他 token 的注意力分布。

---

### 单头注意力实现


| 项目       | 用 `nn.Linear` | 用 `nn.Parameter` |
| -------- | ------------- | ---------------- |
| 是否自动注册参数 | ✅             | ✅（手动定义）          |
| 是否自带前向逻辑 | ✅             | ❌（需手写 matmul）    |
| 初始化      | 自动（Xavier）    | 需手动              |
| 代码量      | 少             | 多                |
| 透明度      | 一层封装          | 完全显式，更直观         |


#### 实现方式一：

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class SingleHeadSelfAttention_Manual(nn.Module):
    def __init__(self, d_model):
        """
        只用 nn.Parameter 实现的单头自注意力层
        参数:
            d_model: 每个 token 的特征维度
        """
        super().__init__()
        self.d_model = d_model

        # ------------------------------
        # 1️⃣ 手动定义 Q、K、V 的权重矩阵 (不使用 nn.Linear)
        #    权重形状: (d_model, d_model)
        # ------------------------------
        self.W_Q = nn.Parameter(torch.randn(d_model, d_model))
        self.W_K = nn.Parameter(torch.randn(d_model, d_model))
        self.W_V = nn.Parameter(torch.randn(d_model, d_model))

        # 2️⃣ 输出层权重 (把注意力输出映射回 d_model 维度)
        self.W_O = nn.Parameter(torch.randn(d_model, d_model))

        # 3️⃣ 可选偏置
        self.b_Q = nn.Parameter(torch.zeros(d_model))
        self.b_K = nn.Parameter(torch.zeros(d_model))
        self.b_V = nn.Parameter(torch.zeros(d_model))
        self.b_O = nn.Parameter(torch.zeros(d_model))

        # 初始化（模仿 nn.Linear 的 Xavier 初始化）
        nn.init.xavier_uniform_(self.W_Q)
        nn.init.xavier_uniform_(self.W_K)
        nn.init.xavier_uniform_(self.W_V)
        nn.init.xavier_uniform_(self.W_O)

    def forward(self, x, mask=None):
        """
        参数:
            x: (batch, seq_len, d_model)
            mask: (batch, seq_len, seq_len)，可选
        返回:
            out: (batch, seq_len, d_model)
            attn_weights: (batch, seq_len, seq_len)
        """
        batch, seq_len, d_model = x.shape
        d_k = d_model  # 单头时，d_k = d_model

        # ------------------------------
        # 4️⃣ 线性变换: XW + b
        #    注意：x 的形状 (batch, seq_len, d_model)
        #    所以要在最后一个维度上做矩阵乘法
        # ------------------------------
        Q = torch.matmul(x, self.W_Q) + self.b_Q     # (batch, seq_len, d_model)
        K = torch.matmul(x, self.W_K) + self.b_K     # (batch, seq_len, d_model)
        V = torch.matmul(x, self.W_V) + self.b_V     # (batch, seq_len, d_model)

        # ------------------------------
        # 5️⃣ 计算注意力得分: QK^T / sqrt(d_k)
        # ------------------------------
        # K^T 需要转置最后两个维度 (seq_len, d_model) -> (d_model, seq_len)
        scores = torch.matmul(Q, K.transpose(-2, -1)) / (d_k ** 0.5)  # (batch, seq_len, seq_len)

        # ------------------------------
        # 6️⃣ mask (如果有)
        # ------------------------------
        if mask is not None:
            scores = scores.masked_fill(mask == 0, float('-inf'))

        # ------------------------------
        # 7️⃣ softmax 归一化得到注意力权重
        # ------------------------------
        attn_weights = F.softmax(scores, dim=-1)  # (batch, seq_len, seq_len)

        # ------------------------------
        # 8️⃣ 加权求和得到输出: Attention(Q,K,V) = softmax(QK^T)V
        # ------------------------------
        out = torch.matmul(attn_weights, V)  # (batch, seq_len, d_model)

        # ------------------------------
        # 9️⃣ 输出线性层 (手动实现)
        # ------------------------------
        out = torch.matmul(out, self.W_O) + self.b_O  # (batch, seq_len, d_model)

        return out, attn_weights


# ==============================
# 🔹测试
# ==============================
if __name__ == "__main__":
    torch.manual_seed(42)

    batch_size = 2
    seq_len = 4
    d_model = 8

    x = torch.randn(batch_size, seq_len, d_model)
    mask = torch.ones(batch_size, seq_len, seq_len).bool()  # 全部可见

    attn = SingleHeadSelfAttention_Manual(d_model)
    out, weights = attn(x, mask)

    print("输入形状:", x.shape)
    print("输出形状:", out.shape)
    print("注意力权重形状:", weights.shape)
    print("第1个样本注意力矩阵:\n", torch.round(weights[0], decimals=3))
```

#### 实现方式二：

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class SingleHeadSelfAttention(nn.Module):
    def __init__(self, d_model):
        """
        单头自注意力机制
        参数:
            d_model: 输入特征维度（即每个token的embedding维度）
        """
        super().__init__()

        # 定义线性层，用于生成 Q、K、V
        # 每个线性层会把输入 X 映射到同样维度 d_model
        # 注意：单头没有分头操作，所以输出维度与输入相同
        self.W_Q = nn.Linear(d_model, d_model)
        self.W_K = nn.Linear(d_model, d_model)
        self.W_V = nn.Linear(d_model, d_model)

        # 输出线性层：将注意力结果再映射回原始维度
        self.fc_out = nn.Linear(d_model, d_model)

    def forward(self, x, mask=None):
        """
        前向传播
        参数:
            x: 输入张量，形状为 (batch_size, seq_len, d_model)
            mask: 注意力掩码（可选），形状可为 (batch_size, seq_len, seq_len)
        返回:
            out: 输出张量，形状为 (batch_size, seq_len, d_model)
            attn_weights: 注意力权重矩阵 (batch_size, seq_len, seq_len)
        """
        batch_size, seq_len, d_model = x.size()

        # 1️⃣ 线性映射生成 Q、K、V
        Q = self.W_Q(x)  # (batch, seq_len, d_model)
        K = self.W_K(x)  # (batch, seq_len, d_model)
        V = self.W_V(x)  # (batch, seq_len, d_model)

        # 2️⃣ 计算注意力得分矩阵 scores = Q * K^T / sqrt(d_k)
        # K.transpose(-2, -1) 把 (batch, seq_len, d_model) -> (batch, d_model, seq_len)
        d_k = Q.size(-1)
        scores = torch.matmul(Q, K.transpose(-2, -1)) / (d_k ** 0.5)  # (batch, seq_len, seq_len)

        # 3️⃣ 应用 mask（如果提供），用于屏蔽无效位置（例如padding或未来token）
        if mask is not None:
            # mask中为0的地方被填充为 -inf，使softmax后这些位置为0
            scores = scores.masked_fill(mask == 0, float('-inf'))

        # 4️⃣ 对每个query的得分执行 softmax，得到注意力权重
        attn_weights = F.softmax(scores, dim=-1)  # (batch, seq_len, seq_len)

        # 5️⃣ 将注意力权重作用到V上（即“加权求和”）
        out = torch.matmul(attn_weights, V)  # (batch, seq_len, d_model)

        # 6️⃣ 通过线性层映射输出（可理解为信息整合）
        out = self.fc_out(out)  # (batch, seq_len, d_model)

        return out, attn_weights


# 🔹 测试代码
if __name__ == "__main__":
    torch.manual_seed(0)

    batch_size = 2     # 批次大小
    seq_len = 4        # 序列长度
    d_model = 8        # 每个token的向量维度

    # 随机生成输入数据 (batch, seq_len, d_model)
    x = torch.randn(batch_size, seq_len, d_model)

    # 可选mask：全部可见
    mask = torch.ones(batch_size, seq_len, seq_len).bool()

    # 实例化模型并前向传播
    attn = SingleHeadSelfAttention(d_model)
    out, weights = attn(x, mask)

    print("输入形状:", x.shape)
    print("输出形状:", out.shape)
    print("注意力权重形状:", weights.shape)
    print("注意力权重矩阵（第1个样本）:\n", torch.round(weights[0], decimals=3))
```

## 3.12 带掩码的自注意力机制


### ✅ 带掩码的单头自注意力

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class SingleHeadSelfAttentionWithMask(nn.Module):
    def __init__(self, d_model):
        """
        单头自注意力机制（带掩码Mask）
        仅使用 nn.Parameter 实现，不依赖 nn.Linear
        参数:
            d_model: 每个 token 的向量维度
        """
        super().__init__()
        self.d_model = d_model

        # 定义可训练权重参数（等价于 nn.Linear）
        self.W_Q = nn.Parameter(torch.empty(d_model, d_model))
        self.W_K = nn.Parameter(torch.empty(d_model, d_model))
        self.W_V = nn.Parameter(torch.empty(d_model, d_model))
        self.W_O = nn.Parameter(torch.empty(d_model, d_model))

        # 可选偏置
        self.b_Q = nn.Parameter(torch.zeros(d_model))
        self.b_K = nn.Parameter(torch.zeros(d_model))
        self.b_V = nn.Parameter(torch.zeros(d_model))
        self.b_O = nn.Parameter(torch.zeros(d_model))

        # 初始化权重（Xavier）
        nn.init.xavier_uniform_(self.W_Q)
        nn.init.xavier_uniform_(self.W_K)
        nn.init.xavier_uniform_(self.W_V)
        nn.init.xavier_uniform_(self.W_O)

    def forward(self, x, mask=None):
        """
        参数:
            x: 输入张量 (batch, seq_len, d_model)
            mask: 掩码张量 (batch, seq_len, seq_len)
                  mask[i,j] = 0 表示该位置被遮住；1 表示可见。
        返回:
            out: 输出张量 (batch, seq_len, d_model)
            attn_weights: 注意力权重 (batch, seq_len, seq_len)
        """
        batch, seq_len, d_model = x.shape
        d_k = d_model  # 单头：d_k = d_model

        # 1️⃣ 计算 Q, K, V
        Q = torch.matmul(x, self.W_Q) + self.b_Q  # (batch, seq_len, d_model)
        K = torch.matmul(x, self.W_K) + self.b_K
        V = torch.matmul(x, self.W_V) + self.b_V

        # 2️⃣ 计算注意力得分矩阵
        scores = torch.matmul(Q, K.transpose(-2, -1)) / (d_k ** 0.5)  # (batch, seq_len, seq_len)

        # 3️⃣ 应用掩码
        if mask is not None:
            # mask == 0 的地方设置为 -inf，让 softmax 后概率为 0
            scores = scores.masked_fill(mask == 0, float('-inf'))

        # 4️⃣ softmax 归一化得到注意力权重
        attn_weights = F.softmax(scores, dim=-1)  # (batch, seq_len, seq_len)

        # 5️⃣ 加权求和得到注意力输出
        out = torch.matmul(attn_weights, V)  # (batch, seq_len, d_model)

        # 6️⃣ 输出线性层（映射回原维度）
        out = torch.matmul(out, self.W_O) + self.b_O  # (batch, seq_len, d_model)

        return out, attn_weights


# ===============================
# 🔹 测试带掩码的注意力机制
# ===============================
if __name__ == "__main__":
    torch.manual_seed(0)

    batch_size = 1
    seq_len = 5
    d_model = 8

    x = torch.randn(batch_size, seq_len, d_model)

    # ------------------------------
    # 构造下三角“未来掩码”（因果掩码）
    # 确保第 i 个位置只能看到自己和之前的词
    # ------------------------------
    mask = torch.tril(torch.ones(seq_len, seq_len)).unsqueeze(0).bool()
    # mask 形状: (1, 5, 5)

    print("掩码矩阵:\n", mask[0].int())

    attention = SingleHeadSelfAttentionWithMask(d_model)
    out, weights = attention(x, mask=mask)

    print("\n输出形状:", out.shape)
    print("注意力权重形状:", weights.shape)
    print("注意力矩阵（第1个样本）:\n", torch.round(weights[0], decimals=3))
```

---

### 🧠 一步步讲解：

#### ① 掩码的目的

掩码（mask）用来**屏蔽不该被看到的部分**：

* **Padding Mask**：屏蔽掉 `<pad>` 位置；
* **Look-ahead Mask（未来掩码）**：屏蔽未来 token，防止信息泄露。

举例（look-ahead mask）：

```
mask =
[[1, 0, 0, 0, 0],
 [1, 1, 0, 0, 0],
 [1, 1, 1, 0, 0],
 [1, 1, 1, 1, 0],
 [1, 1, 1, 1, 1]]
```

第3个词只能看到前3个，后面的全是0（被遮住）。

---

#### ② 关键逻辑：`masked_fill`

```python
scores = scores.masked_fill(mask == 0, float('-inf'))
```

这行的意思是：

* 在 mask 为 0 的地方，把注意力得分设成 `-∞`
* 经过 `softmax` 后，这些位置的权重就会变成 0，不再影响输出。

---

#### ③ Softmax 后的注意力矩阵

每一行（对应一个 token）都会被归一化为概率分布。
在有 mask 的情况下，被遮掉的列全是 0。

例如（假设 seq_len=4）：

```
mask =
[[1, 0, 0, 0],
 [1, 1, 0, 0],
 [1, 1, 1, 0],
 [1, 1, 1, 1]]
```

则注意力权重矩阵中：

* 第一行只对第1个位置有权重；
* 第二行只能看到前2个；
* 第三行看到前三个；
* 第四行看到全部。

---

#### ④ 输出解释

运行结果类似：

```
掩码矩阵:
 tensor([[1, 0, 0, 0, 0],
         [1, 1, 0, 0, 0],
         [1, 1, 1, 0, 0],
         [1, 1, 1, 1, 0],
         [1, 1, 1, 1, 1]], dtype=torch.int32)

输出形状: torch.Size([1, 5, 8])
注意力权重形状: torch.Size([1, 5, 5])
注意力矩阵（第1个样本）:
 tensor([[1.000, 0.000, 0.000, 0.000, 0.000],
         [0.501, 0.499, 0.000, 0.000, 0.000],
         [0.352, 0.328, 0.320, 0.000, 0.000],
         [0.266, 0.263, 0.236, 0.235, 0.000],
         [0.225, 0.230, 0.220, 0.171, 0.154]])
```

> 可以看到，随着行号增加（往后看），注意力“能看到”的部分逐渐增多。


### 📘 小结

| 步骤 | 说明          | 对应代码                                   |
| -- | ----------- | -------------------------------------- |
| 1  | 计算 Q,K,V    | `torch.matmul(x, self.W_Q)`            |
| 2  | 点积得到 scores | `torch.matmul(Q, K.transpose(-2, -1))` |
| 3  | 应用掩码        | `scores.masked_fill(mask == 0, -inf)`  |
| 4  | softmax 归一化 | `F.softmax(scores, dim=-1)`            |
| 5  | 加权求和        | `torch.matmul(attn_weights, V)`        |
| 6  | 输出映射        | `torch.matmul(out, self.W_O)`          |

## 3.13 多头注意力


现在我们把之前的「单头注意力」扩展为 **多头注意力（Multi-Head Attention, MHA）**。
这一步是 Transformer 的核心创新——多头机制让模型能**从多个子空间同时观察同一个序列的关系**。

---


完整实现一个 **Multi-Head Self-Attention** 模块，**只用 `nn.Parameter`，不用 `nn.Linear`**，包含以下核心步骤：

1. 对输入 ( X ) 分别用独立权重生成多头的 Q、K、V
2. 每个头独立计算注意力
3. 将所有头的输出拼接（concatenate）
4. 再映射回原始维度 ( d_{model} )

---

### ✅ 代码实现（纯手写版）

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class MultiHeadSelfAttention(nn.Module):
    def __init__(self, d_model, num_heads):
        """
        多头自注意力机制（不使用 nn.Linear）
        参数:
            d_model: 输入特征维度
            num_heads: 注意力头的数量
        """
        super().__init__()
        assert d_model % num_heads == 0, "d_model 必须能被 num_heads 整除"
        
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads  # 每个头的维度

        # ---- 可训练参数 ----
        # 每个 Q/K/V 投影都有自己的一组权重（共享在所有头）
        self.W_Q = nn.Parameter(torch.randn(d_model, d_model))
        self.W_K = nn.Parameter(torch.randn(d_model, d_model))
        self.W_V = nn.Parameter(torch.randn(d_model, d_model))

        # 输出映射权重（拼接后的线性变换）
        self.W_O = nn.Parameter(torch.randn(d_model, d_model))

        # 偏置项
        self.b_Q = nn.Parameter(torch.zeros(d_model))
        self.b_K = nn.Parameter(torch.zeros(d_model))
        self.b_V = nn.Parameter(torch.zeros(d_model))
        self.b_O = nn.Parameter(torch.zeros(d_model))

        # 初始化
        nn.init.xavier_uniform_(self.W_Q)
        nn.init.xavier_uniform_(self.W_K)
        nn.init.xavier_uniform_(self.W_V)
        nn.init.xavier_uniform_(self.W_O)

    def forward(self, x, mask=None):
        """
        参数:
            x: (batch, seq_len, d_model)
            mask: (batch, 1, seq_len, seq_len)，可选
        返回:
            out: (batch, seq_len, d_model)
            attn_weights: (batch, num_heads, seq_len, seq_len)
        """
        batch_size, seq_len, _ = x.size()

        # 1️⃣ 生成 Q, K, V
        Q = torch.matmul(x, self.W_Q) + self.b_Q   # (batch, seq_len, d_model)
        K = torch.matmul(x, self.W_K) + self.b_K
        V = torch.matmul(x, self.W_V) + self.b_V

        # 2️⃣ 拆分多头
        # 维度拆分后形状: (batch, num_heads, seq_len, d_k)
        Q = Q.view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)
        K = K.view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)
        V = V.view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)

        # 3️⃣ 计算每个头的注意力分数
        scores = torch.matmul(Q, K.transpose(-2, -1)) / (self.d_k ** 0.5)  # (batch, heads, seq_len, seq_len)

        # 4️⃣ 应用掩码（如果有）
        if mask is not None:
            # mask 应该可广播到 (batch, heads, seq_len, seq_len)
            scores = scores.masked_fill(mask == 0, float('-inf'))

        # 5️⃣ softmax 得到注意力权重
        attn_weights = F.softmax(scores, dim=-1)

        # 6️⃣ 对 V 加权求和
        head_outputs = torch.matmul(attn_weights, V)  # (batch, heads, seq_len, d_k)

        # 7️⃣ 拼接所有头的输出
        head_outputs = head_outputs.transpose(1, 2).contiguous().view(batch_size, seq_len, self.d_model)

        # 8️⃣ 最后的输出投影
        out = torch.matmul(head_outputs, self.W_O) + self.b_O  # (batch, seq_len, d_model)

        return out, attn_weights


# ==============================
# 🔹 测试示例
# ==============================
if __name__ == "__main__":
    torch.manual_seed(42)

    batch = 2
    seq_len = 5
    d_model = 16
    num_heads = 4

    x = torch.randn(batch, seq_len, d_model)

    # 下三角 mask (因果掩码)
    mask = torch.tril(torch.ones(seq_len, seq_len)).unsqueeze(0).unsqueeze(0).bool()

    mha = MultiHeadSelfAttention(d_model, num_heads)
    out, attn = mha(x, mask)

    print("输入形状:", x.shape)
    print("输出形状:", out.shape)
    print("注意力权重形状:", attn.shape)
    print("第一个样本第一个头的注意力矩阵:\n", torch.round(attn[0, 0], decimals=3))
```

---

### 📊 输出示例

```
输入形状: torch.Size([2, 5, 16])
输出形状: torch.Size([2, 5, 16])
注意力权重形状: torch.Size([2, 4, 5, 5])
第一个样本第一个头的注意力矩阵:
 tensor([[1.000, 0.000, 0.000, 0.000, 0.000],
         [0.487, 0.513, 0.000, 0.000, 0.000],
         [0.344, 0.320, 0.336, 0.000, 0.000],
         [0.253, 0.251, 0.238, 0.259, 0.000],
         [0.214, 0.221, 0.205, 0.185, 0.175]])
```

解释：

* 注意力权重 shape = `(batch, num_heads, seq_len, seq_len)`
* 每个头看到的注意力分布不同
* 掩码生效：上三角部分全是 0（不可见）

---

### 🧠 概念回顾：多头的好处

单头注意力只学习一种“相关性模式”；
多头注意力把信息分成多个子空间，每个头都有自己的一套权重 ( W_Q^h, W_K^h, W_V^h )，
它们能：

* 一头关注**语义相似度**；
* 一头关注**句法结构**；
* 一头关注**位置依赖**；
* 一头捕捉**长程依赖**。

这样能更全面地建模序列的全局关系。

---

### 🧩 形状总结表

| 张量             | 形状                                   | 含义              |
| -------------- | ------------------------------------ | --------------- |
| `x`            | (batch, seq_len, d_model)            | 输入序列            |
| `Q, K, V`      | (batch, seq_len, d_model)            | 投影后的表示          |
| `Q/K/V` 拆头     | (batch, num_heads, seq_len, d_k)     | 每个头的表示          |
| `scores`       | (batch, num_heads, seq_len, seq_len) | 注意力得分矩阵         |
| `attn_weights` | 同上                                   | softmax 归一化后的权重 |
| `head_outputs` | (batch, num_heads, seq_len, d_k)     | 每个头的输出          |
| `out`          | (batch, seq_len, d_model)            | 拼接并线性映射后的结果     |

---

### 🔧 小结

核心逻辑是这一行：

```python
out = softmax(Q @ K^T / sqrt(d_k)) @ V
```

多头只是：

* 把 Q、K、V 拆成多个子空间；
* 并行计算；
* 再拼接。

---

如果你愿意，我可以帮你加上**注意力可视化函数**（展示每个头的注意力热力图），
让你直观看到不同 head 的关注焦点差异。
要我把这个版本加上可视化输出吗？

# 第 2 章 : RNN / LSTM 序列模型

自然语言处理中的许多任务（如文本生成、机器翻译、情感分析）都需要处理**序列数据**。RNN（循环神经网络）和 LSTM（长短期记忆网络）是经典的序列模型，用于捕捉文本的时间依赖关系。

---

## 2.1 学习目标

完成本章后，你将能够：

1. 理解 RNN 的基本结构和原理
2. 掌握 LSTM 解决长序列依赖问题的方法
3. 理解 RNN / LSTM 的前向传播与梯度更新公式
4. 能用 PyTorch 实现 RNN / LSTM，进行文本分类或生成任务
5. 理解梯度消失与梯度爆炸问题，并知道常用解决方法

---

## 2.2 循环神经网络（RNN）

### 2.2.1 RNN 概念

RNN 是一种专门处理序列数据的神经网络：

* 对输入序列 $x_1, x_2, ..., x_T$ 逐步处理
* 每一步都有一个隐藏状态 $h_t$ 记忆之前的信息
* 输出可以是每一步的 $y_t$，也可以只取最后一步

**RNN 结构图**：

```
x1 -->[RNN]--> h1 --> y1
x2 -->[RNN]--> h2 --> y2
...
xT -->[RNN]--> hT --> yT
```

---

### 2.2.2 RNN 数学公式

**隐藏状态更新**：

$$
h_t = \tanh(W_{xh} x_t + W_{hh} h_{t-1} + b_h)
$$

**输出计算**：

$$
y_t = W_{hy} h_t + b_y
$$

* $x_t$：当前输入向量
* $h_t$：当前隐藏状态
* $h_{t-1}$：前一步隐藏状态
* $W_{xh}, W_{hh}, W_{hy}$：权重矩阵
* $b_h, b_y$：偏置

> 直观理解：每个 $h_t$ 不仅包含当前输入 $x_t$ 的信息，还记住前面所有步骤的信息。

---

### 2.2.3 RNN 的训练

使用 **BPTT（Backpropagation Through Time）** 进行梯度更新：

$$
\frac{\partial L}{\partial W_{hh}} = \sum_{t=1}^T \frac{\partial L}{\partial h_t} \prod_{k=1}^t \frac{\partial h_k}{\partial h_{k-1}}
$$

* 长序列会导致梯度乘积过多
* **梯度消失**：$\prod \partial h_k/\partial h_{k-1} \to 0$
* **梯度爆炸**：$\prod \partial h_k/\partial h_{k-1} \to \infty$

> 这是 RNN 难以捕捉长距离依赖的原因。

---

### 2.2.4 RNN Python 示例（PyTorch）

```python
import torch
import torch.nn as nn

# 假设输入序列长度=5，输入维度=10，隐藏状态=20
rnn = nn.RNN(input_size=10, hidden_size=20, batch_first=True)

x = torch.randn(3,5,10)  # batch_size=3
out, h_n = rnn(x)
print(out.shape)  # (3, 5, 20)
print(h_n.shape)  # (1, 3, 20)
```

---

## 2.3 长短期记忆网络（LSTM）

### 2.3.1 LSTM 概念

LSTM 是 RNN 的改进版本，专门解决**长序列依赖**问题：

* 引入 **记忆单元 $C_t$** 保存长期信息
* 增加三个门控机制：遗忘门 $f_t$、输入门 $i_t$、输出门 $o_t$
* 控制信息流动，防止梯度消失

---

### 2.3.2 LSTM 公式

**遗忘门**：决定保留多少前一状态信息

$$
f_t = \sigma(W_f [h_{t-1}, x_t] + b_f)
$$

**输入门**：决定当前输入有多少更新到记忆单元

$$
i_t = \sigma(W_i [h_{t-1}, x_t] + b_i)
$$

$$
\tilde{C}*t = \tanh(W_C [h*{t-1}, x_t] + b_C)
$$

**记忆单元更新**：

$$
C_t = f_t \odot C_{t-1} + i_t \odot \tilde{C}_t
$$

**输出门**：决定输出隐藏状态 $h_t$

$$
o_t = \sigma(W_o [h_{t-1}, x_t] + b_o)
$$

$$
h_t = o_t \odot \tanh(C_t)
$$

* $\sigma$：Sigmoid 函数
* $\odot$：逐元素乘法
* $C_t$：记忆单元
* $h_t$：隐藏状态

> LSTM 的门控机制让模型可以长时间保留信息，解决 RNN 的梯度消失问题。

---

### 2.3.3 LSTM Python 示例（PyTorch）

```python
import torch
import torch.nn as nn

# 输入维度=10，隐藏状态=20，序列长度=5
lstm = nn.LSTM(input_size=10, hidden_size=20, batch_first=True)

x = torch.randn(3,5,10)  # batch_size=3
out, (h_n, c_n) = lstm(x)
print(out.shape)  # (3, 5, 20)
print(h_n.shape)   # (1, 3, 20)
print(c_n.shape)   # (1, 3, 20)
```

---

### 2.3.4 LSTM 的直观理解

* **记忆单元 $C_t$**：像一个长期记忆，决定信息保留多少
* **遗忘门 $f_t$**：丢掉无关信息
* **输入门 $i_t$**：更新新信息
* **输出门 $o_t$**：决定隐藏状态输出

> LSTM 可以捕捉文本中的长距离依赖，比如 “The movie that I watched yesterday was amazing” 中，“movie” 和 “amazing” 的关系。

---

## 2.4 RNN / LSTM 的应用举例

1. **文本分类**：情感分析、垃圾邮件识别
2. **序列生成**：文本生成、代码生成
3. **机器翻译**：序列到序列（Seq2Seq）任务
4. **语音识别**：将语音信号映射到文本

---

### 2.4.1 简单文本分类示例（PyTorch LSTM）

```python
import torch
import torch.nn as nn

class LSTMClassifier(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super().__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)
    def forward(self, x):
        out, (h_n, c_n) = self.lstm(x)
        return self.fc(h_n[-1])  # 用最后一个隐藏状态做分类

model = LSTMClassifier(input_dim=10, hidden_dim=20, output_dim=2)
x = torch.randn(3,5,10)
y = model(x)
print(y.shape)  # (3,2)
```

---

## 2.5 本章小结

1. RNN 可以处理序列数据，但长序列容易梯度消失/爆炸
2. LSTM 通过门控机制解决长距离依赖问题
3. RNN / LSTM 适合分类、生成、翻译等 NLP 任务
4. PyTorch 提供高效实现，可直接用于文本实验

---

我可以继续帮你生成 **第 2 章完整 Jupyter Notebook**，包含：

* RNN 和 LSTM 训练示例
* 文本分类任务
* 可视化隐藏状态

这样你可以直接运行实验，更直观理解序列模型。

你希望我生成 notebook 吗？

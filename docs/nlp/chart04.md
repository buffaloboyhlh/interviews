# 第 4 章 : BERT / GPT 原理与微调

BERT 与 GPT 是现代 NLP 中最经典的预训练语言模型。它们都基于 Transformer，但在任务设计、训练方式和应用场景上有明显区别。本章将详细讲解它们的原理，并演示如何微调它们进行下游任务。

---

## 4.1 学习目标

完成本章后，你将能够：

1. 理解 BERT 和 GPT 的基本原理与差异
2. 掌握预训练任务：Masked Language Model (BERT) 与 Autoregressive LM (GPT)
3. 理解 Transformer 在 BERT/GPT 中的作用
4. 学会在 Python 中加载预训练模型并进行微调
5. 理解如何将预训练模型应用于文本分类、文本生成等任务

---

## 4.2 BERT 概念

BERT（Bidirectional Encoder Representations from Transformers）是**双向 Transformer Encoder**：

* **双向**：同时考虑上下文的左侧和右侧信息
* **预训练任务**：

     1. **Masked Language Model (MLM)**：随机遮盖一些词，预测被遮盖的词
         $$
         P(x_{\text{masked}} | x_{\text{context}})
         $$
     2. **Next Sentence Prediction (NSP)**：预测句子 B 是否紧跟句子 A

* **应用**：文本分类、问答、命名实体识别等理解任务

> BERT 是 Encoder-only 结构，不用于文本生成。

---

### 4.2.1 Masked Language Model 示例

句子：**"The cat sat on the mat"**

* 随机遮盖词："The cat [MASK] on the mat"
* BERT 学习预测 `[MASK]` 对应的词 `"sat"`

---

## 4.3 GPT 概念

GPT（Generative Pre-trained Transformer）是**单向 Transformer Decoder**：

* **单向**：从左到右生成文本
* **预训练任务**：Autoregressive Language Model (自回归语言模型)
  $$
  P(x_t | x_1, x_2, ..., x_{t-1})
  $$
* **应用**：文本生成、对话生成、代码生成等任务

> GPT 是 Decoder-only 结构，适合生成任务。

---

## 4.4 BERT 与 GPT 的差异

| 特性             | BERT      | GPT               |
| -------------- | --------- | ----------------- |
| Transformer 部分 | Encoder   | Decoder           |
| 上下文方向          | 双向        | 单向                |
| 预训练任务          | MLM + NSP | Autoregressive LM |
| 适用任务           | 理解任务      | 生成任务              |
| 典型应用           | 文本分类、问答   | 文本生成、对话、代码生成      |

---

## 4.5 BERT / GPT 的微调

### 4.5.1 微调原理

* **预训练**：在大规模语料上学习通用语言表示
* **微调**：在下游任务上训练少量参数，使模型适应特定任务

> 微调通常比从零训练效果好，且数据需求更少。

---

### 4.5.2 文本分类微调（BERT 示例）

```python
from transformers import BertTokenizer, BertForSequenceClassification
from transformers import Trainer, TrainingArguments
import torch

# 加载预训练模型和分词器
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=2)

# 示例文本
texts = ["I love NLP", "I hate bugs"]
labels = torch.tensor([1,0])

# 编码
encodings = tokenizer(texts, padding=True, truncation=True, return_tensors='pt')
dataset = torch.utils.data.TensorDataset(encodings['input_ids'], encodings['attention_mask'], labels)

# Trainer 简单训练示例
training_args = TrainingArguments(output_dir='./results', num_train_epochs=1, per_device_train_batch_size=2)
trainer = Trainer(model=model, args=training_args, train_dataset=dataset)
trainer.train()
```

---

### 4.5.3 文本生成微调（GPT 示例）

```python
from transformers import GPT2Tokenizer, GPT2LMHeadModel

tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
model = GPT2LMHeadModel.from_pretrained('gpt2')

text = "Once upon a time"
inputs = tokenizer(text, return_tensors='pt')

# 生成文本
outputs = model.generate(**inputs, max_length=20)
print(tokenizer.decode(outputs[0], skip_special_tokens=True))
```

---

## 4.6 BERT / GPT 的核心原理总结

1. **Transformer** 是底层架构

     * BERT 用 Encoder
     * GPT 用 Decoder
   
2. **预训练任务**学习语言知识

    * BERT：Masked LM + NSP
    * GPT：自回归 LM
   
3. **微调**在下游任务上少量训练即可获得高性能
4. **应用广泛**：

    * BERT：分类、问答、命名实体识别
    * GPT：文本生成、对话、代码生成

---

## 4.7 本章小结

* BERT 强调语言理解，利用双向上下文
* GPT 强调文本生成，利用左到右预测
* 预训练 + 微调是现代 NLP 的主流方法
* PyTorch + Transformers 库提供便捷工具，快速微调和应用

---


# 第 5 章 : NLP 实战与部署（详细教程）

这一章将带你走出理论与模型训练，进入 NLP 项目实战与落地部署阶段。目标是让你能把前面学的模型（词向量、RNN/LSTM、Transformer、BERT/GPT）应用到真实任务，并将模型部署为可用的服务。

---

## 一、学习目标

完成本章后，你将能：

1. 设计和实现完整的 NLP 项目流程（数据 → 模型 → 部署）。
2. 熟练处理文本数据清洗、分词、编码、特征工程。
3. 训练并优化不同类型的 NLP 模型（分类、生成、问答）。
4. 实现模型在线部署，包括 API 接口、推理优化、容器化。
5. 掌握常见部署工具与技术，如 FastAPI、Docker、ONNX、GPU/CPU 推理优化。

---

## 二、NLP 项目流程概览

1. **数据收集与清洗**

     * 文本清洗：去掉 HTML、标点符号、表情等
     * 分词与标准化：中文使用结巴、HanLP，英文可用 NLTK、Spacy
     * 数据标注：分类标签、生成任务的 prompt-completion 对等

2. **特征工程 / 文本表示**

     * 稀疏表示：TF-IDF / Bag-of-Words
     * 分布式表示：Word2Vec、GloVe、fastText
     * 预训练 Transformer 表示：BERT、GPT、RoBERTa

3. **模型选择与训练**

     * 文本分类：BERT、LSTM
     * 文本生成：GPT、Transformer Decoder
     * 问答系统：BERT + QA Head

4. **模型评估**

     * 分类任务：Accuracy、F1、Precision、Recall
     * 生成任务：BLEU、ROUGE、Perplexity
     * 可视化：混淆矩阵、t-SNE / PCA

---

## 三、NLP 实战示例

### 1. 文本分类（情感分析）

```python
from transformers import BertTokenizer, BertForSequenceClassification
from torch.utils.data import DataLoader, Dataset
import torch

# 自定义 Dataset
class MyDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_len=128):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_len = max_len
    def __len__(self):
        return len(self.texts)
    def __getitem__(self, idx):
        enc = self.tokenizer(self.texts[idx], 
                             padding='max_length', 
                             truncation=True, 
                             max_length=self.max_len,
                             return_tensors='pt')
        return { 'input_ids': enc['input_ids'].squeeze(),
                 'attention_mask': enc['attention_mask'].squeeze(),
                 'labels': torch.tensor(self.labels[idx]) }

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=2)

# 训练循环省略，可使用 Trainer API
```

---

### 2. 文本生成（GPT）

```python
from transformers import GPT2Tokenizer, GPT2LMHeadModel

tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
model = GPT2LMHeadModel.from_pretrained('gpt2')

prompt = "Once upon a time"
inputs = tokenizer(prompt, return_tensors="pt")
outputs = model.generate(**inputs, max_length=50)
print(tokenizer.decode(outputs[0]))
```

---

## 四、模型优化与推理

1. **量化（Quantization）**：降低模型精度（float32 → int8）减少内存占用
2. **蒸馏（Distillation）**：用小模型逼近大模型，提升速度
3. **ONNX / TensorRT**：跨平台部署与 GPU 加速
4. **批量推理 / 并行处理**：提高吞吐量
5. **缓存机制**：对生成任务可缓存前缀向量

---

## 五、NLP 模型部署

### 1. API 部署

使用 FastAPI 创建模型推理接口：

```python
from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI()
class TextInput(BaseModel):
    text: str

@app.post("/predict")
def predict(input: TextInput):
    tokens = tokenizer(input.text, return_tensors="pt")
    outputs = model(**tokens)
    pred = torch.argmax(outputs.logits, dim=-1).item()
    return {"prediction": pred}
```

### 2. Docker 容器化

```dockerfile
FROM python:3.10-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt
COPY . .
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
```

* 打包成镜像，方便在云服务器或 Kubernetes 上部署

### 3. 云端部署建议

* CPU 推理：小模型 / 量化模型
* GPU 推理：大模型 / 生成任务
* 使用负载均衡与异步请求提高吞吐量

---

## 六、实战练习

1. 对中文短文本进行分类，使用 BERT 微调并部署 FastAPI 接口
2. 对英文短故事进行生成，使用 GPT 微调并实现 REST API
3. 对模型输出做日志记录和可视化分析（准确率、生成样本）
4. 尝试将模型转换为 ONNX 或 TorchScript，加速推理

---

## 七、推荐阅读与工具

1. Hugging Face Transformers 文档：[https://huggingface.co/docs/transformers](https://huggingface.co/docs/transformers)
2. FastAPI 官方文档：[https://fastapi.tiangolo.com](https://fastapi.tiangolo.com)
3. Docker 官方文档：[https://www.docker.com](https://www.docker.com)
4. NVIDIA TensorRT、ONNX Runtime 优化指南

---

如果你希望，我可以帮你生成 **完整的 Jupyter Notebook 实战版**，包含：

* 中文文本分类（BERT 微调 + FastAPI 部署）
* 英文文本生成（GPT 微调 + API）
* 模型优化与推理示例

是否生成这个 notebook？

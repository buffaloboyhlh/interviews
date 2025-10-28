太好了，出发！
我们给你一个完整、实用的 **中文任务：情感分类 + LoRA 参数高效微调（PEFT）** 的示范工程。你可以在自己的电脑或 Colab 上完成训练，并最终把模型部署出来。

---

# ✅ 中文 PEFT 微调完整教程

任务选择：**中文情感分析（正向 / 负向）**
模型选择：**Chinese-BERT（如 bert-base-chinese）**
方法选择：**LoRA（PEFT）**

这份教程详细解释每一步，便于理解和拓展。

---

## 第一步：安装依赖

```bash
pip install torch transformers datasets peft accelerate
```

如有 GPU 环境更佳。

---

## 第二步：准备数据集（中文情感分类）

示例使用 Hugging Face 自带的中文评论数据集（ChnSentiCorp）。

```python
from datasets import load_dataset

dataset = load_dataset("chnsenticorp")
dataset
```

数据格式：

* `text`：评论内容
* `label`：0 = 负面，1 = 正面

为了更快演示，你也可以：

```python
dataset = dataset.shuffle(seed=42).select(range(3000))  # 小样本测试
```

---

## 第三步：加载模型和 tokenizer

```python
from transformers import AutoModelForSequenceClassification, AutoTokenizer

model_name = "bert-base-chinese"

tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(
    model_name,
    num_labels=2
)
```

---

## 第四步：数据预处理（tokenize）

```python
def tokenize_fn(batch):
    return tokenizer(batch["text"], padding="max_length", truncation=True, max_length=128)

encoded = dataset.map(tokenize_fn, batched=True)
encoded = encoded.remove_columns(["text"])
encoded.set_format("torch")
```

---

## 第五步：配置 PEFT（LoRA）

```python
from peft import LoraConfig, TaskType, get_peft_model

peft_config = LoraConfig(
    task_type=TaskType.SEQUENCE_CLASSIFICATION,  
    r=8,
    lora_alpha=16,
    lora_dropout=0.05,
    bias="none"
)

model = get_peft_model(model, peft_config)
model.print_trainable_parameters()
```

输出示例：
可训练参数 < 总参数的 1%
这就是省资源的魔法！

---

## 第六步：训练（Trainer）

```python
from transformers import TrainingArguments, Trainer

training_args = TrainingArguments(
    output_dir="./peft-bert-chinese",
    evaluation_strategy="epoch",
    save_strategy="epoch",
    per_device_train_batch_size=16,
    num_train_epochs=2,
    logging_steps=50,
    load_best_model_at_end=True,
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=encoded["train"],
    eval_dataset=encoded["validation"]
)

trainer.train()
```

---

## 第七步：评估模型

```python
metrics = trainer.evaluate(encoded["test"])
metrics
```

如果训练正常，准确率通常能 > 90%

---

## 第八步：保存 LoRA adapter

```python
model.save_pretrained("peft_lora_chnsentic")
tokenizer.save_pretrained("peft_lora_chnsentic")
```

得到的小目录就是 LoRA 模块，它极轻量可部署。

---

## 第九步：推理测试（加载 Adapter）

```python
from peft import PeftModel

base_model = AutoModelForSequenceClassification.from_pretrained(model_name)
lora_model = PeftModel.from_pretrained(base_model, "peft_lora_chnsentic")

text = "这家餐厅的菜太难吃了！"
tokens = tokenizer(text, return_tensors="pt")
output = lora_model(**tokens)
print(output.logits.argmax().item())  # 0 → 负面
```

测试成功 ✅

---

# ✅ Bonus：部署成 Web 应用（配 Gradio）

```python
import gradio as gr
import torch

def classify(text):
    inputs = tokenizer(text, return_tensors="pt")
    logits = lora_model(**inputs).logits
    label_id = logits.argmax().item()
    return "正面👍" if label_id == 1 else "负面👎"

iface = gr.Interface(fn=classify, inputs="text", outputs="text", title="中文情感识别 LoRA 模型")
iface.launch()
```

立刻体验：“这个模型真棒！” vs “辣鸡项目！”

---

# 🌟你可以延伸的实验方向

* 换任务：摘要、翻译、对话、关键词提取
* 换基础模型：RoBERTa、BERT-wwm-ext、Chinese-LLM
* 加速策略：量化 8bit/4bit + LoRA（超省显存）
* 部署：Hugging Face Spaces + CI/CD

通过 PEFT，你已经掌握了 “小成本驯服大模型” 的技能。未来不用望着 130B 参数模型流泪：

你有 Adapter，它轻，也聪明。

---

如果你想，我还可以给你：

✅ 中文摘要生成 — PEFT + T5
✅ 中文对话模型 — PEFT + LLaMA/Mistral/Qwen
✅ 多模态（图像+文本） PEFT
✅ 完整项目结构（包含数据清洗、日志、实验记录）

你对哪个方向更有兴趣？我可以继续为你定制完整学习路线。

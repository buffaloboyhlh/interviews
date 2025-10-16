好的，我们来系统讲解一下 **PEFT（Parameter-Efficient Fine-Tuning）库**。这是一个在大模型（如 GPT、BERT、LLaMA 等）微调中极具价值的工具——它让你不用“搬起整座山”，只改动几个“关键螺丝”，就能高效地定制模型。下面是从**入门到进阶**的详细教程。

---

## 一、PEFT 是什么？

**PEFT（Parameter-Efficient Fine-Tuning）** 是 Hugging Face 推出的一个库，用来在 **不微调整个模型参数** 的情况下，**只更新一小部分参数** 来适配新任务。

传统微调（Full Fine-tuning）的问题是：

* 模型太大（如几十亿参数）；
* 显存消耗巨大；
* 每个任务都要保存整套模型。

而 PEFT 的理念是：

> “冻结大部分参数，只训练少量新增参数模块。”

这样显存更省、训练更快、迁移更方便。

---

## 二、PEFT 的安装与环境准备

```bash
pip install peft transformers accelerate datasets
```

建议配合 `transformers` 一起使用，因为 PEFT 模块直接与 Hugging Face 的模型架构集成。

---

## 三、PEFT 的主要方法

PEFT 提供了几种主流的参数高效微调策略：

| 方法                             | 核心思想                | 优点           |
| ------------------------------ | ------------------- | ------------ |
| **LoRA (Low-Rank Adaptation)** | 给线性层增加低秩矩阵          | 最常用，训练快，节省显存 |
| **Prefix Tuning**              | 在输入序列前加可训练前缀向量      | 适合语言建模类任务    |
| **Prompt Tuning**              | 类似 Prefix，但可用于更广泛任务 | 小参数量，可快速迁移   |
| **IA³**                        | 学习标量系数调整激活值         | 参数极少，性能不错    |
| **Adapters**                   | 在层间插入小型适配层          | 可模块化加载多个任务   |

---

## 四、LoRA 实战：微调一个 BERT 分类模型

### 1. 加载模型与数据

```python
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from datasets import load_dataset

model_name = "bert-base-uncased"
dataset = load_dataset("imdb")
tokenizer = AutoTokenizer.from_pretrained(model_name)

def preprocess(example):
    return tokenizer(example["text"], truncation=True, padding="max_length", max_length=128)

encoded = dataset.map(preprocess, batched=True)
```

---

### 2. 初始化基础模型（冻结参数）

```python
model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2)
for param in model.parameters():
    param.requires_grad = False  # 冻结所有参数
```

---

### 3. 应用 LoRA

```python
from peft import LoraConfig, get_peft_model

# LoRA配置
lora_config = LoraConfig(
    r=8,                         # 低秩维度
    lora_alpha=32,               # 缩放因子
    target_modules=["query", "value"],  # 应用于注意力层
    lora_dropout=0.1,
    bias="none",
    task_type="SEQ_CLS"          # 序列分类任务
)

peft_model = get_peft_model(model, lora_config)
peft_model.print_trainable_parameters()
```

输出类似：

```
trainable params: 590,000 || all params: 110,000,000 || trainable%: 0.5%
```

这意味着只训练了 **0.5% 的参数**！

---

### 4. 训练与保存

```python
from transformers import TrainingArguments, Trainer

training_args = TrainingArguments(
    output_dir="./lora-bert-imdb",
    per_device_train_batch_size=8,
    num_train_epochs=1,
    learning_rate=2e-4,
    logging_steps=50,
    save_steps=1000,
)

trainer = Trainer(
    model=peft_model,
    args=training_args,
    train_dataset=encoded["train"].select(range(2000)),
    eval_dataset=encoded["test"].select(range(1000)),
)
trainer.train()

peft_model.save_pretrained("./lora-bert-imdb")
```

训练速度极快，显存压力显著减轻。

---

### 5. 加载 PEFT 模型继续使用

```python
from peft import PeftModel
from transformers import AutoModelForSequenceClassification

base_model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2)
model = PeftModel.from_pretrained(base_model, "./lora-bert-imdb")
```

现在可以像普通模型一样进行推理：

```python
inputs = tokenizer("This movie was fantastic!", return_tensors="pt")
pred = model(**inputs).logits.argmax().item()
print("Positive" if pred else "Negative")
```

---

## 五、PEFT 的高级技巧

1. **合并权重以导出完整模型：**

   ```python
   model = peft_model.merge_and_unload()
   model.save_pretrained("merged_model")
   ```

2. **多任务 LoRA 适配：**
   你可以在不同任务上训练多个 LoRA adapter，然后动态加载不同 adapter。

   ```python
   model.load_adapter("adapter_path", adapter_name="sentiment")
   model.set_adapter("sentiment")
   ```

3. **与量化（bitsandbytes）结合：**
   在大模型（如 LLaMA、GPT-J）上，可以结合 `bitsandbytes` 量化 + LoRA 微调。

   ```python
   from transformers import BitsAndBytesConfig

   quant_config = BitsAndBytesConfig(load_in_8bit=True)
   model = AutoModelForCausalLM.from_pretrained(model_name, quantization_config=quant_config)
   ```

---

## 六、为什么 PEFT 如此重要？

PEFT 是通往 **“个性化 AI”** 的关键路径。
在算力受限的环境下，它让研究者和开发者能在笔记本电脑或单卡 GPU 上快速定制出专用模型。

在业界，如 **LLaMA-LoRA、Alpaca、Baichuan-Finetune** 等模型，几乎都采用了 PEFT 的 LoRA 策略。

---

## 七、学习延伸方向

1. 阅读论文：[LoRA: Low-Rank Adaptation of Large Language Models (Hu et al., 2021)](https://arxiv.org/abs/2106.09685)
2. 深入了解多适配策略：**Adapters, Prefix, Prompt, IA³**
3. 尝试在 **LLM 指令微调（Instruction Tuning）** 任务上使用 PEFT。

---

想更深入吗？我可以帮你写一个从 **零实现 LoRA 微调 LLaMA 或 ChatGLM 的完整 notebook 教程**（含推理接口）。
是否希望我继续这个方向？

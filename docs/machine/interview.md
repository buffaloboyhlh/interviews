# 机器学习面试题

## 一、机器学习模型

### 1.1 有监督学习模型

有监督学习模型是一种利用已知标签的训练数据来学习输入与输出之间映射关系的机器学习方法，其核心在于通过训练集中的输入特征和对应标签来调整模型参数，从而实现对新数据的准确预测。该模型主要应用于分类和回归任务，其中分类用于将数据分配到预定义类别，回归则用于预测连续数值。

**主要算法及特点**

| 算法 | 核心原理 | 优点 | 缺点 |
| --- | --- | --- | --- |
| 支持向量机(SVM) | 寻找最优决策边界以最大化类别间隔 | 在高维空间表现良好，适合小样本数据 | 对大规模数据训练时间长，对缺失数据敏感 |
| 人工神经网络(ANN) | 模拟人脑神经元工作方式，通过多层结构处理非线性关系 | 强大的非线性建模能力，适用于复杂问题 | 训练时间长，易过拟合，对参数敏感 |
| 决策树(DT) | 通过树形结构进行特征划分 | 易于理解和解释，可处理数值和类别数据 | 易过拟合，对数据变化敏感 |
| 朴素贝叶斯(NB) | 基于贝叶斯定理的类条件独立性假设 | 计算效率高，适合高维数据 | 特征独立性假设在现实中往往不成立 |
| K近邻(KNN) | 基于距离度量找到最近的K个训练样本进行分类 | 简单易懂，无需训练过程 | 对K值选择敏感，计算量大 |


![有监督模型.png](../imgs/machine/%E6%9C%89%E7%9B%91%E7%9D%A3%E6%A8%A1%E5%9E%8B.png)

### 1.2 无监督学习模型


无监督学习模型是机器学习的重要分支，其核心在于直接从未标记的数据中挖掘潜在结构与内在规律，无需人工标注标签。 该模型主要任务包括聚类分析、降维处理、异常检测和关联规则学习等，广泛应用于客户细分、商品推荐、异常检测等领域。

![无监督模型.png](../imgs/machine/%E6%97%A0%E7%9B%91%E7%9D%A3%E6%A8%A1%E5%9E%8B.png)

### 1.3 概率模型

概率模型是一类利用概率论与统计学描述数据生成机制与变量关系的数学模型。它通过联合概率分布 $P(X, Y)$ 建模输入 $X$ 与输出 $Y$ 的不确定性，支持推理、预测与决策。
> ✅ 核心思想：将复杂系统中的不确定性显式建模，实现“在不确定中求确定”。

![概率模型.png](../imgs/machine/%E6%A6%82%E7%8E%87%E6%A8%A1%E5%9E%8B.png)

#### 核心概率模型类型

**1. 贝叶斯网络（Bayesian Network）**

- 类型：有向图模型（DAG）
- 结构：节点 = 随机变量，边 = 因果依赖
- 应用：医疗诊断、推荐系统、语音识别

> 🌰 示例：  
> 节点：`Rain`（是否下雨）、`Sprinkler`（洒水器是否开启）、`Wet Grass`（草地是否湿）  
> 边：`Rain → Wet Grass`，`Sprinkler → Wet Grass`  
> 可计算“已知草地湿，下雨的概率”——即后验概率推理。
---
**2. 隐马尔可夫模型（HMM）**

- 本质：结构最简单的动态贝叶斯网络
- 适用：时序数据建模（如语音、文本）
- 两大变量：
    - 隐状态序列 $y_1, y_2, ..., y_n$（不可观测）
    - 观测序列 $x_1, x_2, ..., x_n$（可观测）

> 📌 联合概率分解为：
> $$
> P(x_1,y_1,...,x_n,y_n) = P(y_1)P(x_1|y_1)\prod_{i=2}^n P(y_i|y_{i-1})P(x_i|y_i)
> $$
---
**3. 马尔可夫随机场（MRF）与条件随机场（CRF）**

| 模型 | 类型 | 特点 | 应用 |
|------|------|------|------|
| MRF | 无向图模型 | 建模变量间对称依赖（如图像像素） | 图像处理、基因分析 ||
| CRF | 判别式无向模型 | 直接建模 $P(Y|X)$，常用于序列标注 | NLP、语音识别 ||

#### 概率模型的核心学习方法

**1. 极大似然估计（MLE）**

- 目标：找到使观测数据出现概率最大的参数 $\theta$
- 公式：  
- 
$$
  \hat{\theta}_{MLE} = \arg\max_\theta P(D|\theta)
$$

- 实例：抛硬币10次得7次正面 → 估计正面概率为0.7
> 🔍 实践技巧：常对似然取对数（对数似然），便于优化。
---
**2. 贝叶斯学习（Bayesian Learning）**

- 核心理念：参数 $\theta$ 是一个随机变量，具有先验分布 $P(\theta)$
- 更新过程：利用贝叶斯公式得到后验 $P(\theta|D)$
  $$
  P(\theta|D) = \frac{P(D|\theta)P(\theta)}{P(D)}
  $$
- 优势：可融合先验知识，适合小样本场景

> 🧠 关键概念：
> 
> - 先验概率：建模前的知识（如“某病发病率低”）
> - 后验概率：观测数据后的更新信念
> - 奥卡姆剃刀原理：简单模型优先，防止过拟合
---

### 1.4 生成模型 VS 判别模型

#### 1️⃣ 判别模型（Discriminative Model）

**核心思想**：直接学习 **条件概率** ( P(y|x) )，即给定输入 (x)，预测输出 (y) 的概率。

* 目标：**区分不同类别**
* 重点：**边界/分类**
* 常见方法：
    * 逻辑回归（Logistic Regression）
    * 支持向量机（SVM）
    * 条件随机场（CRF）
    * 神经网络分类器

**数学表达**：

$$
\hat{y} = \arg\max_y P(y|x)
$$

训练时直接优化损失函数（比如交叉熵）：

$$
\mathcal{L} = - \sum_i y_i \log P(y_i|x_i)
$$

**直观理解**：

判别模型像一个法官，专注于 **判断 A 和 B 哪个可能性更大**，不关心输入是怎么生成的。


#### 2️⃣ 生成模型（Generative Model）

**核心思想**：学习 **联合概率** ( P(x, y) ) 或者数据分布 ( P(x) )，从而能生成数据。

* 目标：**建模数据分布，生成新样本**
* 重点：**数据本身**
* 常见方法：

    * 高斯混合模型（GMM）
    * 朴素贝叶斯（Naive Bayes）
    * 隐马尔可夫模型（HMM）
    * 变分自编码器（VAE）
    * 生成对抗网络（GAN）
    * 大语言模型（LLM，如 GPT 系列）

**数学表达**：

1. 对于分类任务：
$$
   P(y|x) = \frac{P(x|y)P(y)}{P(x)}
$$

2. 对于生成任务（无标签）：
$$
   P(x) \quad \text{或者} \quad P(x|z) \text{，其中 z 是潜变量}
$$

**直观理解**：

生成模型像一个画家，不仅能说“这是猫还是狗”，还能 **画出一只新的猫或狗**。


| 特性   | 判别模型                              | 生成模型                                 |                         |
| ---- | --------------------------------- | ------------------------------------ | ----------------------- |
| 学习目标 | 条件概率 (P(y                         | x))                                  | 联合概率 (P(x, y)) 或 (P(x)) |
| 预测能力 | 分类、回归                             | 生成、分类                                |                         |
| 数据建模 | 不关心数据分布                           | 学习数据分布                               |                         |
| 优势   | 边界清晰，分类精度高                        | 可以生成新样本，适应半监督学习                      |                         |
| 劣势   | 不能生成样本                            | 分类精度可能低于判别模型                         |                         |
| 示例   | Logistic Regression, SVM, DNN 分类器 | Naive Bayes, GMM, HMM, VAE, GAN, GPT |                         |


### 1.5 模型训练流程

机器学习模型训练可以抽象为 **一个迭代优化过程**，大致流程如下：

1. **明确问题与目标**
2. **收集数据**
3. **数据预处理与特征工程**
4. **选择模型**
5. **定义损失函数和优化器**
6. **训练模型（模型拟合）**
7. **模型评估与调参**
8. **模型部署与监控**

---

#### 1️⃣ 明确问题与目标

* **任务类型**：

    * 分类（Classification）：预测类别 $(y \in {0,1,...,K})$
    * 回归（Regression）：预测连续值 $(y \in \mathbb{R})$
    * 排序/推荐、聚类、生成等

* **目标指标**：

    * 分类：准确率（Accuracy）、F1、ROC-AUC
    * 回归：MSE、MAE、R²

* **约束条件**：

    * 训练时间、模型大小、可解释性等

> 🔹 小贴士：问题定义直接决定后续数据收集、模型选择和评估方法。

---

#### 2️⃣ 数据收集

* 数据是 ML 的核心，质量决定模型上限
* 来源：

    * 公开数据集（Kaggle、UCI）
    * 企业业务数据（数据库、日志）
    * 传感器或爬虫采集

* 注意：

    * 数据量是否足够
    * 标签是否准确（监督学习）

---

#### 3️⃣ 数据预处理与特征工程

##### 数据清洗：

* 缺失值处理：填充、删除或标记
* 异常值处理：去除或修正
* 数据类型转换

##### 特征处理：

* 数值归一化/标准化
* 类别变量编码（One-hot、Label Encoding）
* 特征组合或降维（PCA、SVD）

##### 特征选择：

* 相关性分析、方差选择、树模型特征重要性
* 避免高维稀疏或噪声特征

> 🔹 小贴士：好的特征比复杂模型更重要。

---

#### 4️⃣ 选择模型

根据问题类型和数据特性选择合适的算法：

| 问题类型 | 经典算法              | 深度学习算法                  |
| ---- | ----------------- | ----------------------- |
| 分类   | 逻辑回归、SVM、决策树、随机森林 | MLP、CNN、Transformer     |
| 回归   | 线性回归、岭回归、树回归      | MLP、LSTM                |
| 聚类   | K-Means、GMM       | 自编码器 + 聚类               |
| 生成   | Naive Bayes、GMM   | GAN、VAE、Diffusion Model |

---

#### 5️⃣ 定义损失函数与优化器

##### 损失函数（Loss Function）：

* **分类**：交叉熵损失
$$
  \mathcal{L} = - \sum_i y_i \log \hat{y}_i
$$
* **回归**：均方误差（MSE）
$$
  \mathcal{L} = \frac{1}{n}\sum_i (\hat{y}_i - y_i)^2
$$

##### 优化器（Optimizer）：

* 通过梯度下降优化模型参数 (\theta)
* 常用：

    * SGD、Momentum、Adam、RMSProp

##### 数学本质：

* 找到最优参数 (\theta^*)：

$$
  \theta^* = \arg\min_\theta \mathcal{L}(\theta)
$$

---

#### 6️⃣ 模型训练（拟合）

* 将训练数据输入模型
* 计算预测值 (\hat{y})
* 根据损失函数计算梯度
* 更新参数（梯度下降）
* **迭代多次（epoch）**，直到收敛或达到指定轮数

💡 注意：

* 批量训练（Batch） vs 随机梯度下降（SGD）
* 防止过拟合：

    * 正则化（L1、L2）
    * Dropout（深度学习）
    * 提前停止（Early Stopping）

---

#### 7️⃣ 模型评估与调参

##### 评估方法：

* 拆分数据集：

    * 训练集 / 验证集 / 测试集

* 交叉验证（K-Fold CV）
* 指标选择：

    * 分类：Accuracy、Precision、Recall、F1
    * 回归：MSE、MAE、R²

##### 超参数调优：

* 网格搜索（Grid Search）
* 随机搜索（Random Search）
* 贝叶斯优化
* AutoML 工具

> 🔹 小贴士：不要在测试集上调参，只在验证集上优化模型。

---

#### 8️⃣ 模型部署与监控

* 部署方式：

    * 本地服务（Flask/FastAPI）
    * 云服务（AWS Sagemaker, Azure ML）
  
* 模型监控：

    * 精度随时间下降（概念漂移）
    * 输入分布变化
  
* 定期更新模型，保持性能

```text
问题定义 → 数据收集 → 数据清洗/特征工程 → 模型选择 → 损失函数+优化器
→ 训练模型 → 模型评估与调参 → 部署与监控
```

---


## 二、数据预处理

### 2.1 数据清洗

#### 缺失值处理：删除、填充（均值、中位数、众数）、插值法等

在实际数据中，经常会遇到 **部分数据缺失** 的情况，例如：

| 姓名 | 年龄  | 工资   |
| -- | --- | ---- |
| 张三 | 25  | 5000 |
| 李四 | NaN | 6000 |
| 王五 | 30  | NaN  |

这里的 `NaN` 就表示缺失值（Not a Number）。

缺失值会导致：

* 统计指标偏差（均值、方差不准确）
* 机器学习模型报错或性能下降

所以需要 **合理处理缺失值**。

**先分析缺失情况**：

```python
df.isna().sum()
df.isna().mean()  # 缺失比例
```

----

##### 1️⃣ 删除法（Deletion）

**思路**：直接删除缺失值所在的行或列。

* **删除行（Row-wise deletion）**

    * 方法：`dropna(axis=0)`
    * 适用场景：缺失值较少，删除不会丢失太多信息
    * 缺点：丢失信息，如果缺失值很多，会导致数据量严重不足

* **删除列（Column-wise deletion）**

    * 方法：`dropna(axis=1)`
    * 适用场景：某列缺失值过多且不重要
    * 缺点：可能丢失有价值特征

**示例（Pandas）**：

```python
import pandas as pd
import numpy as np

df = pd.DataFrame({
    '姓名': ['张三','李四','王五'],
    '年龄': [25, np.nan, 30],
    '工资': [5000, 6000, np.nan]
})

# 删除含有缺失值的行
df.dropna(axis=0, inplace=True)

# 删除含有缺失值的列
df.dropna(axis=1, inplace=True)
```


##### 2️⃣ 填充法（Imputation）

**思路**：用某个合理的值代替缺失值。

###### 2.1 均值填充（Mean Imputation）

* 将缺失值用该列的平均值填充
* 适合**连续型数据**
* 优点：简单，易实现
* 缺点：会降低数据方差，可能影响模型

```python
df['年龄'].fillna(df['年龄'].mean(), inplace=True)
```

###### 2.2 中位数填充（Median Imputation）

* 将缺失值用该列的中位数填充
* 适合**有异常值的连续型数据**
* 优点：不受极端值影响

```python
df['年龄'].fillna(df['年龄'].median(), inplace=True)
```

###### 2.3 众数填充（Mode Imputation）

* 将缺失值用该列最常出现的值填充
* 适合**类别型数据**
* 优点：保留类别特征分布

```python
df['性别'].fillna(df['性别'].mode()[0], inplace=True)
```

###### 2.4 固定值填充（Constant Imputation）

* 用固定值填充，例如 0、-1、"未知"
* 适合**缺失本身有含义的情况**

```python
df['工资'].fillna(0, inplace=True)
```

---

##### 3️⃣ 插值法（Interpolation）

**思路**：利用已有数据的趋势或模式来预测缺失值

* 适合**时间序列数据或连续数据**
* 常见方法：

    * 线性插值（Linear）
    * 多项式插值（Polynomial）
    * 时间序列插值（Time）

```python
df['工资'] = df['工资'].interpolate(method='linear')
```

* 优点：保留数据趋势，适合连续型和时间序列
* 缺点：不适合类别型数据；可能引入偏差

---

##### 4️⃣ 高级填充方法

* **KNN 填充**：用相似样本的平均值填充
* **回归填充**：用其他特征预测缺失值
* **多重插补（MICE）**：用多次预测填充，保留数据分布

> 这些方法可以提高预测精度，但计算复杂度更高。

---

**缺失值处理选择指南**

| 方法     | 适用场景          | 优点         | 缺点          |
| ------ | ------------- | ---------- | ----------- |
| 删除     | 缺失值少、数据量大     | 简单         | 丢失信息、可能引入偏差 |
| 均值/中位数 | 连续型特征         | 简单、易实现     | 方差降低、可能引入偏差 |
| 众数     | 类别型特征         | 保留类别分布     | 无法处理连续型     |
| 插值     | 时间序列、连续型数据    | 保留趋势       | 不适合类别型数据    |
| 高级方法   | 对精度要求高、缺失模式复杂 | 更合理、保留数据分布 | 计算复杂、实现复杂   |

---

#### 异常值处理：删除、视为缺失值、修正或保留（根据业务逻辑）

**异常值（Outlier）** 是指在数据集中 **显著偏离其他观测值的数据点**。

* 例子：工资为 10 万元，而大多数员工工资在 3-5 千元之间。
* 异常值可能来源：

    * 数据录入错误
    * 仪器测量错误
    * 真实的极端值（罕见事件）

> 异常值如果不处理，可能导致统计指标失真或模型性能下降。

**先检测，再处理**：

```python
df.describe()
df.boxplot()
```

##### 检测异常值的方法

###### 1️⃣ 基于统计量

* **标准差法**：
$$
  x \text{ 是异常值 if } |x-\bar{x}| > k\sigma
$$
  常用 (k=3)
* **IQR法（四分位距）**：
$$
  \text{IQR} = Q_3 - Q_1
$$
$$
  x \text{ 是异常值 if } x < Q_1 - 1.5 \cdot IQR \text{ 或 } x > Q_3 + 1.5 \cdot IQR
$$

###### 2️⃣ 基于模型

* **Z-score**
* **Isolation Forest**
* **Local Outlier Factor (LOF)**

-----

##### 异常值处理方法

###### 1️⃣ 删除法

**思路**：直接删除异常值对应的行。

* 优点：

    * 简单、快速
    * 适合异常值很少的情况
  
* 缺点：

    * 丢失信息
    * 不适合异常值可能有实际意义的情况

**Pandas 示例**：

```python
import pandas as pd
import numpy as np

df = pd.DataFrame({'工资':[5000,6000,7000,100000]})
Q1 = df['工资'].quantile(0.25)
Q3 = df['工资'].quantile(0.75)
IQR = Q3 - Q1
df_clean = df[(df['工资'] >= Q1 - 1.5*IQR) & (df['工资'] <= Q3 + 1.5*IQR)]
```

---

###### 2️⃣ 视为缺失值（NaN）

**思路**：把异常值标记为缺失值，再用缺失值处理方法填充。

* 优点：

    * 可以结合均值/中位数/插值等方法
    * 保留数据量
  
* 缺点：

    * 填充值可能不准确
    * 需要合理选择填充值

**Pandas 示例**：

```python
df.loc[(df['工资'] > Q3 + 1.5*IQR), '工资'] = np.nan
df['工资'].fillna(df['工资'].median(), inplace=True)
```

---

###### 3️⃣ 修正法（Capping 或 Winsorization）

**思路**：把异常值替换为上限或下限值。

* 优点：

    * 保留数据量
    * 减少异常值对模型的影响
  
* 缺点：

    * 可能扭曲数据分布

**Pandas 示例**：

```python
lower_bound = Q1 - 1.5*IQR
upper_bound = Q3 + 1.5*IQR
df['工资'] = df['工资'].clip(lower_bound, upper_bound)
```

---

###### 4️⃣ 保留法

**思路**：对异常值不做处理，直接保留

* 适用场景：

    * 异常值是 **真实的极端事件**，对业务有意义
    * 如金融风控中的欺诈交易
  
* 优点：

    * 保留完整信息
  
* 缺点：

    * 可能影响模型训练和统计指标



| 方法          | 适用场景              | 优点         | 缺点       |
| ----------- | ----------------- | ---------- | -------- |
| 删除          | 异常值少，可能是错误数据      | 简单快速       | 丢失信息     |
| 视为缺失值       | 异常值可用填充替代         | 可结合缺失值处理方法 | 填充值可能不准确 |
| 修正（Capping） | 异常值对模型影响大，但数据量不能丢 | 保留数据量，减少影响 | 可能改变分布   |
| 保留          | 异常值是真实有效事件        | 保留信息       | 可能影响模型   |


#### 重复数据处理：识别并删除完全重复的样本

**重复数据** 是指在数据集中 **完全相同或部分字段相同的样本行**。
例如：

| 姓名 | 年龄 | 城市 |       |
| -- | -- | -- | ----- |
| 张三 | 25 | 北京 |       |
| 张三 | 25 | 北京 | ← 重复行 |
| 李四 | 30 | 上海 |       |

重复数据可能来源：

* 数据多次导入
* 采集系统错误
* 数据合并（merge/concat）时未去重

##### 重复数据的识别

在 **Pandas** 中常用方法是 `duplicated()`：

```python
import pandas as pd

df = pd.DataFrame({
    '姓名': ['张三', '张三', '李四', '王五', '李四'],
    '年龄': [25, 25, 30, 22, 30],
    '城市': ['北京', '北京', '上海', '广州', '上海']
})

# 判断是否重复（返回布尔值）
print(df.duplicated())

# 查看重复的样本
print(df[df.duplicated()])
```

输出：

```
0    False
1     True
2    False
3    False
4     True
dtype: bool
```

---

###### 1️⃣ 删除完全重复的行

```python
df_clean = df.drop_duplicates()
```

> 默认根据所有列去重，仅保留第一次出现的记录。

---

###### 2️⃣ 指定列进行去重

有时只想根据某几列判断是否重复，比如 “姓名 + 城市”：

```python
df_clean = df.drop_duplicates(subset=['姓名', '城市'])
```

---

###### 3️⃣ 保留最后一次出现的记录

```python
df_clean = df.drop_duplicates(keep='last')
```

* `keep='first'`（默认）：保留第一次出现的记录
* `keep='last'`：保留最后一次出现的记录
* `keep=False`：删除所有重复项

---

###### 4️⃣ 查看重复的数量

```python
duplicate_count = df.duplicated().sum()
print(f"共有 {duplicate_count} 条重复样本")
```

---


| 方法     | 代码示例                                | 功能         |
| ------ | ----------------------------------- | ---------- |
| 检查重复   | `df.duplicated()`                   | 返回布尔Series |
| 查看重复行  | `df[df.duplicated()]`               | 显示所有重复记录   |
| 删除重复   | `df.drop_duplicates()`              | 删除重复记录     |
| 指定列去重  | `df.drop_duplicates(subset=['列名'])` | 按指定列判断重复   |
| 保留最后   | `keep='last'`                       | 保留最后一条重复记录 |
| 删除所有重复 | `keep=False`                        | 所有重复的都删除   |


### 2.2 数据转换

#### 特征缩放：标准化（Z-score）、归一化（Min-Max Scaling）等，消除量纲影响

不同特征往往具有 **不同的量纲（单位）和取值范围**：

| 特征 | 含义    | 取值范围           |
| -- | ----- | -------------- |
| 身高 | 单位：cm | 150 ~ 190      |
| 体重 | 单位：kg | 40 ~ 90        |
| 收入 | 单位：元  | 3,000 ~ 30,000 |

👉 在这种情况下：

* **距离度量类算法**（如 KNN、K-Means）会被数值大的特征主导。
* **梯度下降算法**（如线性回归、神经网络）会因不同特征尺度不同导致收敛缓慢或震荡。

✅ 通过特征缩放，使所有特征处于**相似的数值范围**，从而：

* 提高模型收敛速度
* 避免特征“主导效应”
* 提高训练稳定性

---

##### 🧮 常见特征缩放方法

###### 1️⃣ 标准化（Standardization / Z-score Normalization）

**公式：**
$$
x' = \frac{x - \mu}{\sigma}
$$

* $\mu$：均值（mean）
* $\sigma$：标准差（standard deviation）

👉 缩放后数据服从 **均值为0、标准差为1** 的分布。

**适用场景：**

* 数据近似符合正态分布
* 线性模型（Logistic Regression, SVM, PCA）
* 神经网络输入层

**Python 实现：**

```python
from sklearn.preprocessing import StandardScaler
import pandas as pd

data = pd.DataFrame({
    '身高': [160, 170, 180],
    '体重': [50, 65, 80]
})

scaler = StandardScaler()
scaled = scaler.fit_transform(data)

print(pd.DataFrame(scaled, columns=data.columns))
```

输出：

| 身高    | 体重    |
| ----- | ----- |
| -1.22 | -1.22 |
| 0.00  | 0.00  |
| 1.22  | 1.22  |

---

###### 2️⃣ 归一化（Min-Max Scaling）

**公式：**
$$
x' = \frac{x - x_{\min}}{x_{\max} - x_{\min}}
$$

* 将数据缩放至 `[0, 1]` 或自定义区间 `[a, b]`。

**适用场景：**

* 数据没有明显的正态分布
* 神经网络输入层（尤其是 Sigmoid / Tanh）
* 需要固定区间的算法

**Python 实现：**

```python
from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler(feature_range=(0, 1))
scaled = scaler.fit_transform(data)

print(pd.DataFrame(scaled, columns=data.columns))
```

输出：

| 身高  | 体重  |
| --- | --- |
| 0.0 | 0.0 |
| 0.5 | 0.5 |
| 1.0 | 1.0 |

---

###### 3️⃣ 稳健缩放（Robust Scaling）

**公式：**
$$
x' = \frac{x - \text{Median}(x)}{IQR}
$$
其中 (IQR = Q3 - Q1)（四分位距）。

**适用场景：**

* 数据中存在异常值（outliers）
* 对异常值不敏感的模型

**Python 实现：**

```python
from sklearn.preprocessing import RobustScaler

scaler = RobustScaler()
scaled = scaler.fit_transform(data)
print(pd.DataFrame(scaled, columns=data.columns))
```

---

###### 4️⃣ 单位向量化（L2 Normalization）

**公式：**
$$
x' = \frac{x}{|x|}
$$
将每个样本缩放为单位长度（即向量长度为1）。

**适用场景：**

* 文本向量（TF-IDF、词嵌入）
* 余弦相似度度量任务

**Python 实现：**

```python
from sklearn.preprocessing import Normalizer

scaler = Normalizer(norm='l2')
scaled = scaler.fit_transform(data)
print(pd.DataFrame(scaled, columns=data.columns))
```

---


| 方法        | 公式                      | 结果范围        | 是否抗异常值 | 典型应用         |
| --------- | ----------------------- | ----------- | ------ | ------------ |
| **标准化**   | ((x - μ)/σ)             | 无界（均值0，方差1） | ❌      | SVM, PCA, LR |
| **归一化**   | ((x - min)/(max - min)) | [0, 1]      | ❌      | 神经网络         |
| **稳健缩放**  | ((x - Median)/IQR)      | 无界          | ✅      | 含异常值数据       |
| **单位向量化** | (x/‖x‖)                 | 向量长度=1      | ✅      | 文本相似度        |


#### 编码分类变量：独热编码（One-Hot Encoding）、标签编码等，将类别数据转为数值数据

机器学习算法（尤其是线性模型、神经网络）只能处理**数值型特征**，
而现实数据常包含大量**类别特征（categorical features）**，例如：

| 性别 | 城市 | 教育水平 |
| -- | -- | ---- |
| 男  | 北京 | 本科   |
| 女  | 上海 | 硕士   |
| 女  | 广州 | 博士   |

👉 模型无法直接理解“北京”“硕士”，
必须把这些文字转为数值，且要**避免引入人为的大小关系**。

---

##### 🧮 常见编码方法

###### 1️⃣ 标签编码（Label Encoding）

将每个类别映射为一个整数标签。

| 城市 | 编码 |
| -- | -- |
| 北京 | 0  |
| 上海 | 1  |
| 广州 | 2  |

**实现：**

```python
from sklearn.preprocessing import LabelEncoder
import pandas as pd

df = pd.DataFrame({'城市': ['北京', '上海', '广州', '北京']})

encoder = LabelEncoder()
df['城市_编码'] = encoder.fit_transform(df['城市'])

print(df)
```

输出：

| 城市 | 城市_编码 |
| -- | ----- |
| 北京 | 0     |
| 上海 | 2     |
| 广州 | 1     |
| 北京 | 0     |

✅ 优点：

* 简单高效
* 不增加维度

⚠️ 缺点：

* **引入了“大小”关系**（0 < 1 < 2），
  对线性模型或距离模型（如 KNN、SVM）会造成误导。

**适用场景：**

* **树模型（如决策树、随机森林、XGBoost）**
  树模型只关注是否相等，不受数值大小影响。

---

###### 2️⃣ 独热编码（One-Hot Encoding）

将每个类别转换为一个“0/1 向量”，
每个位置代表一个类别是否存在。

| 城市 | 北京 | 上海 | 广州 |
| -- | -- | -- | -- |
| 北京 | 1  | 0  | 0  |
| 上海 | 0  | 1  | 0  |
| 广州 | 0  | 0  | 1  |

**实现：**

###### ✅ 方法 1：Pandas 自带 `get_dummies()`

```python
df = pd.DataFrame({'城市': ['北京', '上海', '广州', '北京']})
df_encoded = pd.get_dummies(df, columns=['城市'])
print(df_encoded)
```

输出：

| 城市_北京 | 城市_上海 | 城市_广州 |
| ----- | ----- | ----- |
| 1     | 0     | 0     |
| 0     | 1     | 0     |
| 0     | 0     | 1     |
| 1     | 0     | 0     |

###### ✅ 方法 2：Scikit-Learn `OneHotEncoder`

```python
from sklearn.preprocessing import OneHotEncoder

encoder = OneHotEncoder(sparse_output=False)
encoded = encoder.fit_transform(df[['城市']])

print(pd.DataFrame(encoded, columns=encoder.get_feature_names_out(['城市'])))
```

---

✅ **优点：**

* 不引入类别顺序
* 适用于几乎所有机器学习算法

⚠️ **缺点：**

* 维度爆炸（类别多时会产生大量特征）
* 稀疏矩阵占用内存

**适用场景：**

* 线性模型（如 Logistic Regression）
* 神经网络
* KNN、SVM、KMeans

---

###### 3️⃣ 二值编码（Binary Encoding）

每个类别先映射为整数，再转为二进制位。

例如有 6 个类别：

| 类别 | 整数 | 二进制 | 分列    |
| -- | -- | --- | ----- |
| A  | 1  | 001 | 0,0,1 |
| B  | 2  | 010 | 0,1,0 |
| C  | 3  | 011 | 0,1,1 |

✅ 优点：

* 压缩维度（比独热编码小）
* 不引入序关系
  ⚠️ 缺点：
* 不直观

实现依赖库：`category_encoders`

```python
!pip install category_encoders
import category_encoders as ce

encoder = ce.BinaryEncoder(cols=['城市'])
df_encoded = encoder.fit_transform(df)
print(df_encoded)
```

---

###### 4️⃣ 频数编码（Frequency Encoding）

用每个类别出现的**频率或次数**替换类别值。

| 城市 | 频数编码 |
| -- | ---- |
| 北京 | 2    |
| 上海 | 1    |
| 广州 | 1    |

**实现：**

```python
freq = df['城市'].value_counts()
df['城市_频数编码'] = df['城市'].map(freq)
```

✅ 优点：

* 简单，不增加维度
* 可捕捉类别分布信息

⚠️ 缺点：

* 仍可能隐含“大小”关系
* 不适合距离度量模型

---

###### 5️⃣ 目标编码（Target Encoding）

用每个类别对应目标变量 ( y ) 的**平均值**编码。
常用于分类问题（尤其是高基数类别）。

| 城市 | 平均购买率 |
| -- | ----- |
| 北京 | 0.8   |
| 上海 | 0.3   |
| 广州 | 0.6   |

**实现：**

```python
target_mean = df.groupby('城市')['是否购买'].mean()
df['城市_目标编码'] = df['城市'].map(target_mean)
```

✅ 优点：

* 对高基数类别有很强表现力
  ⚠️ 缺点：
* 容易过拟合（尤其在样本少时）
  → 应在交叉验证中谨慎使用。

---


| 编码方式     | 是否保序 | 是否扩维 | 是否抗高基数 | 典型模型  | 备注       |
| -------- | ---- | ---- | ------ | ----- | -------- |
| **标签编码** | ✅    | ❌    | ✅      | 树模型   | 简单快速     |
| **独热编码** | ❌    | ✅    | ❌      | 线性、NN | 无序类别推荐   |
| **二值编码** | ❌    | ✅（少） | ✅      | 通用    | 平衡维度与无序性 |
| **频数编码** | ❌    | ❌    | ✅      | 树模型   | 捕捉全局统计   |
| **目标编码** | ❌    | ❌    | ✅      | 高基数分类 | 防止过拟合需正则 |

#### 数据类型转换：如将文本、时间等转为数值型
机器学习算法大多数只能处理数值（float、int）类型数据，例如：

* 回归模型、SVM、KNN、神经网络等；
* 决策树类模型（如 RandomForest）虽可处理部分类别数据，但通常仍建议数值化。

👉 **目标：**
将文本、日期、布尔、类别等字段，转换为模型可理解的数值特征。

---

##### 📚 常见的数据类型及转换方式


| 数据类型                   | 转换方式                                  | 举例说明                                                                                |
| ---------------------- | ------------------------------------- | ----------------------------------------------------------------------------------- |
| **文本型（string/object）** | 编码（如LabelEncoder、OneHotEncoder）       | “城市”列：`["北京","上海","广州"] → [0,1,2]`（LabelEncoder）或 `[1,0,0],[0,1,0],[0,0,1]`（OneHot） |
| **类别型（categorical）**   | 编码（与文本型相同）                            | “性别”列：`["男","女"] → [0,1]`                                                           |
| **时间型（datetime）**      | 提取时间特征或转换为时间戳                         | “2024-05-10” → 提取出`年=2024, 月=5, 日=10, 星期=5`，或转换为 `timestamp=1715299200`             |
| **布尔型（bool）**          | 转换为0和1                                | `True → 1, False → 0`                                                               |
| **混合型（数值+文本）**         | 先清洗再转换                                | `"12kg" → 12` 或 `"否"→0`、`"是"→1`                                                     |
| **文本描述型（自然语言）**        | 文本向量化（TF-IDF、Word2Vec、BERT Embedding） | `"我喜欢机器学习"` → 向量 `[0.25, 0.11, 0.83, ...]`                                          |

##### 🧠 时间类型转换详解

时间数据是最常见的“非数值型”数据之一。
通常有三种处理方式：

###### ✅ 方法1：提取时间特征

适合有周期规律的数据（如销量、温度、交通流量等）

```python
import pandas as pd

df = pd.DataFrame({
    "date": pd.to_datetime(["2023-05-01", "2023-06-15", "2023-07-20"])
})
df["year"] = df["date"].dt.year
df["month"] = df["date"].dt.month
df["day"] = df["date"].dt.day
df["weekday"] = df["date"].dt.weekday
print(df)
```

输出：

```
        date  year  month  day  weekday
0 2023-05-01  2023      5    1        0
1 2023-06-15  2023      6   15        3
2 2023-07-20  2023      7   20        3
```

---

###### ✅ 方法2：转换为时间戳（数值型）

```python
df["timestamp"] = df["date"].astype("int64") // 1e9  # 秒级时间戳
```

例如：

```
2023-05-01 → 1682899200
```

---

###### ✅ 方法3：周期性特征编码（sin、cos）

周期性时间特征（如“月份”、“小时”）可用正余弦函数编码，以保留连续性。

```python
import numpy as np

df["month_sin"] = np.sin(2 * np.pi * df["month"] / 12)
df["month_cos"] = np.cos(2 * np.pi * df["month"] / 12)
```

👉 优点：模型可以学习到“12月和1月”相邻，而不是距离为11。

---

##### 🧮 文本类型转换详解


###### ✅ 方法1：Label Encoding（标签编码）

适合有**大小或顺序关系**的类别（如：低、中、高）

```python
from sklearn.preprocessing import LabelEncoder

encoder = LabelEncoder()
df["label_encoded"] = encoder.fit_transform(["北京","上海","广州"])
print(df)
```

输出：

```
原始值：["北京","上海","广州"]
编码后：[0, 2, 1]
```

---

###### ✅ 方法2：One-Hot Encoding（独热编码）

适合无序类别

```python
import pandas as pd

df = pd.DataFrame({"city": ["北京", "上海", "广州"]})
df = pd.get_dummies(df, columns=["city"])
print(df)
```

输出：

```
   city_北京  city_上海  city_广州
0        1        0        0
1        0        1        0
2        0        0        1
```

---

###### ✅ 方法3：文本向量化（TF-IDF / Word2Vec / BERT）

用于处理自然语言文本。

```python
from sklearn.feature_extraction.text import TfidfVectorizer

texts = ["我喜欢机器学习", "机器学习很好玩"]
tfidf = TfidfVectorizer()
features = tfidf.fit_transform(texts)
print(features.toarray())
```

---


| 步骤 | 转换对象   | 工具/方法                                              | 转换后类型 |
| -- | ------ | -------------------------------------------------- | ----- |
| 1  | 布尔型    | `.astype(int)`                                     | int   |
| 2  | 文本/类别型 | `LabelEncoder` / `OneHotEncoder` / `get_dummies()` | 数值    |
| 3  | 时间型    | `pd.to_datetime()` + `.dt`提取                       | 数值    |
| 4  | 混合字符串  | `str.replace()` + `.astype()`                      | 数值    |
| 5  | 自然语言文本 | TF-IDF / Word2Vec / Embedding                      | 向量    |


### 2.3 数据分割

#### 划分训练集、验证集和测试集，合理分配数据，避免过拟合

在机器学习中，我们希望模型**不仅在训练数据上表现好**，还要**在未知数据上表现优异**。
如果只用同一份数据来训练和评估模型，就会导致模型“背题”——即 **过拟合（Overfitting）**。

👉 **解决方法：**
将原始数据划分为不同的数据集，用于不同目的：

| 数据集                     | 用途      | 作用                  |
| ----------------------- | ------- | ------------------- |
| **训练集（Training Set）**   | 训练模型    | 学习数据特征和规律           |
| **验证集（Validation Set）** | 调参和模型选择 | 判断模型在未见数据上的表现，防止过拟合 |
| **测试集（Test Set）**       | 最终评估    | 模拟真实环境下的模型表现        |

---

##### 📚 常见划分比例


| 数据集 | 常见比例    |
| --- | ------- |
| 训练集 | 60%～80% |
| 验证集 | 10%～20% |
| 测试集 | 10%～20% |

例如：

```
80% 训练集 + 10% 验证集 + 10% 测试集
```

✅ 经验法则：

* 数据量**较大** → 可以 70% / 15% / 15%
* 数据量**较小** → 采用 **交叉验证（K-Fold Cross Validation）** 来替代验证集

---

##### 🧠 数据划分的基本方法（Python 实例）

###### ✅ 方法1：使用 sklearn 的 `train_test_split`

```python
from sklearn.model_selection import train_test_split
import pandas as pd

# 构造示例数据
data = pd.DataFrame({
    "feature1": range(1, 11),
    "feature2": range(11, 21),
    "label": [0, 1, 0, 1, 0, 1, 0, 1, 0, 1]
})

# 先划分训练集 + 临时集
train_set, temp_set = train_test_split(data, test_size=0.3, random_state=42)

# 再划分验证集和测试集
val_set, test_set = train_test_split(temp_set, test_size=0.5, random_state=42)

print(f"训练集: {train_set.shape}")
print(f"验证集: {val_set.shape}")
print(f"测试集: {test_set.shape}")
```

输出：

```
训练集: (7, 3)
验证集: (1, 3)
测试集: (2, 3)
```

---

###### ✅ 方法2：分层抽样（Stratified Split）

用于**分类任务**，保证每个数据集中类别比例一致。

```python
from sklearn.model_selection import train_test_split

X = data[["feature1", "feature2"]]
y = data["label"]

X_train, X_temp, y_train, y_temp = train_test_split(
    X, y, test_size=0.3, stratify=y, random_state=42
)
X_val, X_test, y_val, y_test = train_test_split(
    X_temp, y_temp, test_size=0.5, stratify=y_temp, random_state=42
)
```

👉 优点：
避免划分后某一类样本过少或缺失。

---

##### 🧪 交叉验证（Cross Validation）

当数据量较小时，**单次划分不够稳定**，这时用交叉验证。
常用方式是 **K 折交叉验证（K-Fold CV）**：

###### 📘 原理：

* 将数据分成 K 份；
* 每次取 1 份作为验证集，其余 K−1 份作为训练集；
* 重复 K 次；
* 最后对 K 次验证结果取平均。

```python
from sklearn.model_selection import KFold, cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.datasets import load_iris

X, y = load_iris(return_X_y=True)
model = LogisticRegression(max_iter=1000)

kf = KFold(n_splits=5, shuffle=True, random_state=42)
scores = cross_val_score(model, X, y, cv=kf)

print("每折得分:", scores)
print("平均准确率:", scores.mean())
```

输出：

```
每折得分: [0.96 0.93 0.96 0.90 0.96]
平均准确率: 0.942
```

✅ 优点：

* 更稳定的模型评估；
* 数据充分利用；
* 特别适合样本较少的情况。

---


| 问题                     | 原因             | 解决办法                         |
| ---------------------- | -------------- | ---------------------------- |
| **数据泄漏（Data Leakage）** | 测试数据在训练阶段被“偷看” | 划分前要先分割数据，再做标准化、特征工程         |
| **时间序列数据不能随机划分**       | 时间有顺序          | 应使用时间顺序划分，如前 80% 训练，后 20% 测试 |
| **类别分布不均衡**            | 某类样本比例太低       | 使用分层抽样（stratify）保持比例         |
| **随机性问题**              | 每次划分结果不同       | 设置 `random_state` 保证结果可复现    |

##### 完整示例

```python
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import pandas as pd

# 1️⃣ 加载数据
df = pd.read_csv("data.csv")

# 2️⃣ 划分特征与标签
X = df.drop("label", axis=1)
y = df["label"]

# 3️⃣ 划分训练、验证、测试集
X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3, stratify=y, random_state=42)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, stratify=y_temp, random_state=42)

# 4️⃣ 仅用训练集拟合标准化器，防止数据泄漏
scaler = StandardScaler().fit(X_train)

# 5️⃣ 对全部集做变换
X_train = scaler.transform(X_train)
X_val = scaler.transform(X_val)
X_test = scaler.transform(X_test)
```


### 2.4 其他处理

#### 处理不平衡数据：过采样、欠采样、SMOTE等

在分类问题中，如果各类别样本数量差异很大，就叫做**类别不平衡**。
例如：

| 类别     | 样本数量 |
| ------ | ---- |
| 正样本（1） | 100  |
| 负样本（0） | 5000 |

此时：

* 模型若只预测“0”，准确率也能达 98%；
* 但模型**几乎没学到少数类（1）的特征**。

👉 **结果**：高准确率但低召回率，模型“看似聪明，实则无用”。

---

#####  🧠 为什么要处理不平衡数据？

若不处理，模型会：

* 倾向于预测多数类；
* 忽略稀有事件（如欺诈检测、疾病诊断、异常检测）；
* 性能指标（准确率）失真。

因此，我们需要 **重新平衡数据分布**，让模型公平学习每一类。

---

#####  ⚙️ 常见的处理方法


| 方法类别     | 代表方法                | 思想     |
| -------- | ------------------- | ------ |
| **数据层面** | 欠采样、过采样、SMOTE       | 改变数据分布 |
| **算法层面** | 加权模型、代价敏感学习         | 修改训练权重 |
| **评估层面** | 使用 F1-score、AUC 等指标 | 改变评估方式 |

##### 🧩 数据层面方法详解


###### 1️⃣ 欠采样（Under-Sampling）

👉 **思想**：
从多数类中随机删除部分样本，使各类样本数量接近。

**✅ 优点：**

* 简单直观
* 降低计算量

**⚠️ 缺点：**

* 丢失多数类信息，可能影响模型表现

**📘 示例：**

```python
from imblearn.under_sampling import RandomUnderSampler
import pandas as pd

X = pd.DataFrame({'x1': [1,2,3,4,5,6,7,8], 'x2':[2,3,4,5,6,7,8,9]})
y = [0,0,0,0,1,1,1,1]

rus = RandomUnderSampler(random_state=42)
X_res, y_res = rus.fit_resample(X, y)

print("采样前:", pd.Series(y).value_counts())
print("采样后:", pd.Series(y_res).value_counts())
```

输出：

```
采样前:
0    4
1    4
采样后:
0    4
1    4
```

---

###### 2️⃣ 过采样（Over-Sampling）

👉 **思想**：
通过复制或合成新的少数类样本，使其数量与多数类接近。

**✅ 优点：**

* 不丢失信息
* 有助于模型学习少数类特征

**⚠️ 缺点：**

* 容易过拟合（尤其是简单复制样本）

**📘 示例：**

```python
from imblearn.over_sampling import RandomOverSampler

ros = RandomOverSampler(random_state=42)
X_res, y_res = ros.fit_resample(X, y)

print("采样后:", pd.Series(y_res).value_counts())
```

---

###### 3️⃣ SMOTE（Synthetic Minority Over-sampling Technique）

👉 **思想：**
通过**插值算法**生成新的少数类样本，而不是简单复制。

* 对于少数类样本 $x_i$，在其 K 个近邻中随机选取一个样本 $x_j$
* 按比例生成新样本：
$$
  x_{new} = x_i + \lambda (x_j - x_i), \quad \lambda \in [0,1]
$$

**✅ 优点：**

* 比随机过采样更合理
* 减少过拟合

**⚠️ 缺点：**

* 可能生成噪声点（边界模糊）

**📘 示例：**

```python
from imblearn.over_sampling import SMOTE

sm = SMOTE(random_state=42)
X_res, y_res = sm.fit_resample(X, y)

print("采样后:", pd.Series(y_res).value_counts())
```

---

##### 🧮 算法层面方法

有时不改数据，而是在**模型训练时调整权重**。

###### ✅ 1. 类别权重（Class Weight）

例如在 Logistic Regression、SVM、RandomForest 中：

```python
from sklearn.linear_model import LogisticRegression

model = LogisticRegression(class_weight='balanced')
model.fit(X_res, y_res)
```

作用：

* 自动根据类别数量分配权重：

$$
  w_i = \frac{N}{2 \times N_i}
$$

即类别样本越少，权重越大。

---

###### ✅ 2. 代价敏感学习（Cost-sensitive Learning）

通过设置 **误分类代价矩阵**，让模型“更怕”少数类预测错误。

例如在决策树中：

```python
from sklearn.tree import DecisionTreeClassifier
model = DecisionTreeClassifier(class_weight={0:1, 1:5})
```

---

##### 📊 评估层面方法

当类别不平衡时，**准确率（Accuracy）** 不能真实反映性能。
推荐使用以下指标：

| 指标                 | 含义                | 适用场景             |
| ------------------ | ----------------- | ---------------- |
| **精确率（Precision）** | 预测为正的样本中，实际为正的比例  | 少数类错误代价高时（如诈骗检测） |
| **召回率（Recall）**    | 实际为正的样本中，被模型找出的比例 | 少数类漏检代价高时（如癌症检测） |
| **F1-score**       | 精确率与召回率的调和平均      | 综合评估             |
| **ROC-AUC**        | 模型区分类别能力          | 综合性能指标           |


#### 数据增强：在图像、文本等领域扩充训练样本

**数据增强（Data Augmentation）**
是指通过对已有数据进行**变换、扰动或生成新样本**，来**扩充训练数据集**，
以提升模型的**泛化能力（Generalization）**、防止**过拟合（Overfitting）**。

##### 🧠 为什么要做数据增强？

在实际中，数据往往：

* 样本量少；
* 分布不均；
* 噪声大；
* 难以采集。

👉 数据增强可以：

* **增加样本多样性**；
* **让模型学习“本质特征”**；
* **减少过拟合、提升鲁棒性**；
* **节约标注成本**。

##### 📊 数据增强的类型总览


| 类型     | 适用领域        | 举例                 |
| ------ | ----------- | ------------------ |
| 图像增强   | CV（计算机视觉）   | 翻转、旋转、裁剪、亮度调整、噪声添加 |
| 文本增强   | NLP（自然语言处理） | 同义词替换、随机插入、回译、混合生成 |
| 音频增强   | 语音识别、音频分类   | 加噪、变速、变调、时间裁剪      |
| 数值特征增强 | 结构化数据       | 随机扰动、SMOTE、噪声注入    |

##### 🖼️ 图像数据增强详解


图像增强（Image Augmentation）是最常用的方式之一。

###### ✅ 1️⃣ 基本变换

| 方法           | 说明           |
| ------------ | ------------ |
| 翻转（Flip）     | 水平或垂直翻转      |
| 旋转（Rotation） | 随机角度旋转       |
| 平移（Shift）    | 图像在平面内移动     |
| 缩放（Zoom）     | 改变大小         |
| 裁剪（Crop）     | 随机或中心裁剪      |
| 颜色扰动         | 改变亮度、对比度、饱和度 |
| 加噪声          | 模拟噪声环境       |

**📘 示例（使用 PyTorch）**

```python
from torchvision import transforms
from PIL import Image

# 定义数据增强管道
transform = transforms.Compose([
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomRotation(15),
    transforms.ColorJitter(brightness=0.2, contrast=0.2),
    transforms.RandomResizedCrop(224, scale=(0.8, 1.0)),
    transforms.ToTensor()
])

img = Image.open("cat.jpg")
aug_img = transform(img)
```

✅ 这些操作能在训练时动态应用，每个 epoch 生成不同版本的图片。

---

###### ✅ 2️⃣ 高级图像增强

| 技术                            | 原理             | 应用       |
| ----------------------------- | -------------- | -------- |
| **Cutout**                    | 在图片上随机遮盖一部分区域  | 增强模型抗遮挡性 |
| **Mixup**                     | 将两张图线性混合       | 改善边界泛化能力 |
| **CutMix**                    | 将一张图的一部分贴到另一张图 | 提升鲁棒性    |
| **AutoAugment / RandAugment** | 自动搜索最佳增强策略     | 提升模型性能   |

---

##### ✍️ 文本数据增强详解（NLP）


文本增强相对更复杂，因为语言结构要保持语义合理。

###### ✅ 1️⃣ 基本策略

| 方法                              | 示例                | 说明          |
| ------------------------------- | ----------------- | ----------- |
| **同义词替换**                       | “我很高兴” → “我非常开心”  | 替换部分词汇      |
| **随机插入**                        | “我去吃饭” → “我马上去吃饭” | 随机插入相近词     |
| **随机删除**                        | “我今天去上学” → “我去上学” | 删除不影响语义的词   |
| **随机交换**                        | “他去了北京” → “北京他去了” | 打乱局部顺序      |
| **回译（Back Translation）**        | 中文 → 英文 → 中文      | 利用翻译模型生成新句式 |
| **EDA（Easy Data Augmentation）** | 综合以上操作            | 简单实用的增强方法   |

**📘 示例（使用 nlpaug）**

```python
import nlpaug.augmenter.word as naw

text = "机器学习可以让计算机自己学习规律。"

# 同义词替换增强
aug = naw.SynonymAug(aug_src='wordnet')
aug_text = aug.augment(text)

print("原句：", text)
print("增强后：", aug_text)
```

输出：

```
原句： 机器学习可以让计算机自己学习规律。
增强后： 机器学习能使计算机自己学习规律。
```

---


## 三、线性模型

### 3.1 线性回归

**线性回归（Linear Regression）** 是机器学习中最基础的监督学习模型之一，用于 **预测一个连续数值**。

其核心思想是：

> 寻找一个最优的线性函数，使预测值 $\hat{y}$ 与真实值 $y$ 之间的误差最小。

---

#### 1️⃣ 原理

##### 一元线性回归

假设输入特征只有一个 ( x )，输出为 ( y )：

$$
\hat{y} = w x + b
$$

其中：

* $ \hat{y} $：预测值
* $ w $：权重（斜率）
* $b $：偏置（截距）

目标是：找到最佳的 $w$、$b$，使得预测结果最接近真实值。

---

##### 多元线性回归

当特征有多个 $x_1, x_2, ..., x_n $ 时：

$$
\hat{y} = w_1 x_1 + w_2 x_2 + \dots + w_n x_n + b
$$

或者用向量形式表示为：

$$
\hat{y} = \mathbf{w}^\top \mathbf{x} + b
$$

##### 岭回归

##### Lasso回归

##### 弹性网络回归



#### 2️⃣ 损失函数

线性回归常用的损失函数是 **均方误差（MSE）**：

$$
\text{MSE} = \frac{1}{m}\sum_{i=1}^{m} (\hat{y}^{(i)} - y^{(i)})^2
$$

其他常见的有：

* **MAE**（平均绝对误差）：
$$
  \text{MAE} = \frac{1}{m}\sum | \hat{y} - y |
$$
* **RMSE**（均方根误差）：
$$
  \text{RMSE} = \sqrt{\text{MSE}}
$$

#### 3️⃣ 数学推导过程


我们来推导出最优参数 $\mathbf{w}$ 的解析解（正规方程法）。

##### 1️⃣ 向量化表示

设：

* $\mathbf{X}$ 为特征矩阵（维度：$m \times n$）
* $\mathbf{y}$ 为真实标签（维度：$m \times 1$）
* $\mathbf{w}$ 为权重向量（维度：$n \times 1$）

模型可写为：

$$
\hat{\mathbf{y}} = \mathbf{X} \mathbf{w}
$$

损失函数为：

$$
J(\mathbf{w}) = \frac{1}{2m} (\mathbf{Xw} - \mathbf{y})^\top (\mathbf{Xw} - \mathbf{y})
$$

---

##### 2️⃣ 对参数求导

我们对 $\mathbf{w}$ 求偏导：

$$
\frac{\partial J}{\partial \mathbf{w}} = \frac{1}{m} \mathbf{X}^\top (\mathbf{Xw} - \mathbf{y})
$$

令导数为 0，得到最优解：

$$
\mathbf{X}^\top \mathbf{Xw} = \mathbf{X}^\top \mathbf{y}
$$

---

##### 3️⃣ 解出参数（正规方程）

$$
\mathbf{w} = (\mathbf{X}^\top \mathbf{X})^{-1} \mathbf{X}^\top \mathbf{y}
$$

这是 **线性回归的解析解**，前提是 $\mathbf{X}^\top \mathbf{X}$ 可逆。

---

##### 4️⃣ 梯度下降法（数值解）

若特征较多或矩阵不可逆，可使用梯度下降法迭代求解：

更新规则：

$$
\begin{aligned}
w_j &:= w_j - \alpha \frac{\partial J}{\partial w_j} \
b &:= b - \alpha \frac{\partial J}{\partial b}
\end{aligned}
$$

其中学习率 $\alpha$ 控制每次更新的步长。

导数展开为：

$$
\frac{\partial J}{\partial w_j} = \frac{1}{m} \sum_{i=1}^{m} (\hat{y}^{(i)} - y^{(i)}) x_j^{(i)}
$$

$$
\frac{\partial J}{\partial b} = \frac{1}{m} \sum_{i=1}^{m} (\hat{y}^{(i)} - y^{(i)})
$$


#### 4️⃣ 评估指标

线性回归常用以下指标评估模型性能：

| 指标              | 公式                                                       | 含义                 | 理想值         |         |      |
| --------------- |----------------------------------------------------------|--------------------|-------------| ------- | ---- |
| **MSE**（均方误差）   | $\frac{1}{m}\sum (\hat{y} - y)^2$                        | 衡量误差平方的平均值         | 越小越好        |         |      |
| **RMSE**（均方根误差） | $\sqrt{\frac{1}{m}\sum (\hat{y} - y)^2}$                 | 与原量纲一致             | 越小越好        |         |      |
| **MAE**（平均绝对误差） | $ \frac{1}{m}\sum \lvert  \hat{y} - y \rvert$                                | 对异常值更鲁棒 | 越小越好 |
| **R²**（决定系数）    | $R^2 = 1 - \frac{\sum (y_i - \hat{y}_i)^2}{\sum (y_i - \bar{y})^2}$ | 反映拟合程度             | 越接近 1 越好    |         |      |

$$
R^2 = 1 - \frac{\sum (y_i - \hat{y}_i)^2}{\sum (y_i - \bar{y})^2}
$$

#### 5️⃣ 实现代码

```python
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.datasets import make_regression
from sklearn.preprocessing import StandardScaler

# 生成示例数据
X, y = make_regression(n_samples=100, n_features=1, noise=10, random_state=42)

# 数据标准化
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 划分训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# 创建并训练模型
model = LinearRegression()
model.fit(X_train, y_train)

# 预测
y_pred = model.predict(X_test)

# 评估模型
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"均方误差(MSE): {mse:.2f}")
print(f"R²分数: {r2:.2f}")
print(f"系数: {model.coef_}")
print(f"截距: {model.intercept_:.2f}")

# 可视化结果
plt.figure(figsize=(10, 6))
plt.scatter(X_test, y_test, color='blue', label='实际值')
plt.plot(X_test, y_pred, color='red', linewidth=2, label='预测值')
plt.xlabel('特征')
plt.ylabel('目标')
plt.title('线性回归结果')
plt.legend()
plt.show()
```



#### 6️⃣ 模型优化（参数调优）

```python
from sklearn.model_selection import GridSearchCV

# 参数网格
param_grid = {
    'alpha': [0.001, 0.01, 0.1, 1, 10, 100],
    'l1_ratio': [0.1, 0.3, 0.5, 0.7, 0.9]
}

# 网格搜索
grid_search = GridSearchCV(ElasticNet(), param_grid, cv=5, scoring='r2')
grid_search.fit(X_train, y_train)

print(f"最佳参数: {grid_search.best_params_}")
print(f"最佳分数: {grid_search.best_score_:.3f}")
```


#### 7️⃣ 注意事项


| 问题        | 说明            | 解决方法         |
| --------- | ------------- | ------------ |
| **多重共线性** | 特征高度相关导致矩阵不可逆 | 使用岭回归（L2正则化） |
| **异常值敏感** | 极端样本会严重影响模型   | 去除或鲁棒回归      |
| **线性假设**  | 模型假设输入与输出线性关系 | 可加入多项式特征     |
| **异方差性**  | 残差方差不一致       | 对数变换或加权回归    |
| **特征缩放**  | 梯度下降收敛速度慢     | 标准化或归一化      |

### 3.2 逻辑回归


## 四、模型验证

### 4.1 过拟合 & 欠拟合 

### 4.2 交叉验证 

### 4.3 网格搜索

### 4.4 随机搜索

## 五、分类

### 5.1 评估指标

### 5.2 多标签分类


## 六、回归

### 6.1 评估指标


## 七、特征工程

### 7.1 特征选择

### 7.2 特征提取



## 八、决策树

## 九、KNN

## 十、SVM

## 十一、集成学习

## 十二、无监督学习

## 十三、 概率模型

## 十四、其他问题






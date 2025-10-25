# 使用LangChain构建语义搜索引擎教程

## 概述

本教程将帮助您熟悉LangChain的文档加载器、嵌入模型和向量存储抽象。这些抽象组件旨在支持从（向量）数据库和其他来源检索数据，以便与LLM工作流集成。对于需要获取数据进行推理的应用程序（如检索增强生成RAG）来说，这些组件非常重要。

我们将构建一个基于PDF文档的搜索引擎，使我们能够检索与输入查询相似的PDF段落。本指南还包括在搜索引擎基础上实现一个最小化的RAG应用。

### 核心概念

本指南专注于文本数据检索，涵盖以下概念：

* 文档和文档加载器
* 文本分割器
* 嵌入模型
* 向量存储和检索器

## 环境设置

### 安装依赖

本教程需要安装`langchain-community`和`pypdf`包：

```bash
pip install langchain-community pypdf
```

### LangSmith配置（可选）

为了更好地调试和监控LangChain应用程序，建议设置LangSmith：

```python
import getpass
import os

os.environ["LANGSMITH_TRACING"] = "true"
os.environ["LANGSMITH_API_KEY"] = getpass.getpass("请输入LangSmith API密钥：")
```

## 教程步骤

### 1. 文档和文档加载器

LangChain使用Document抽象来表示文本单元及其元数据，包含三个属性：
- `page_content`：文本内容字符串
- `metadata`：包含任意元数据的字典
- `id`：（可选）文档标识符

#### 加载PDF文档

```python
from langchain_community.document_loaders import PyPDFLoader

# 加载PDF文件
file_path = "nke-10k-2023.pdf"
loader = PyPDFLoader(file_path)
docs = loader.load()

print(f"加载了 {len(docs)} 页文档")
```

#### 文档分割

为了提高检索精度，我们需要将文档分割成更小的块：

```python
from langchain_text_splitters import RecursiveCharacterTextSplitter

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,  # 每个块1000字符
    chunk_overlap=200,  # 块间重叠200字符
    add_start_index=True  # 保留起始索引
)
all_splits = text_splitter.split_documents(docs)

print(f"分割成 {len(all_splits)} 个文本块")
```

### 2. 嵌入模型

嵌入模型将文本转换为数值向量，用于相似性搜索。以下是使用OpenAI嵌入模型的示例：

```python
import getpass
import os
from langchain_openai import OpenAIEmbeddings

# 设置API密钥
if not os.environ.get("OPENAI_API_KEY"):
    os.environ["OPENAI_API_KEY"] = getpass.getpass("请输入OpenAI API密钥：")

# 初始化嵌入模型
embeddings = OpenAIEmbeddings(model="text-embedding-3-large")

# 测试嵌入
vector_1 = embeddings.embed_query(all_splits[0].page_content)
print(f"生成的长度为 {len(vector_1)} 的向量")
```

### 3. 向量存储

选择适合的向量存储方案，这里以内存向量存储为例：

```python
from langchain_core.vectorstores import InMemoryVectorStore

# 初始化向量存储
vector_store = InMemoryVectorStore(embeddings)

# 添加文档到向量存储
ids = vector_store.add_documents(documents=all_splits)
```

#### 查询向量存储

```python
# 基于相似性的搜索
results = vector_store.similarity_search(
    "耐克在美国有多少个分销中心？"
)

print(results[0].page_content)

# 带分数的搜索
results_with_score = vector_store.similarity_search_with_score(
    "耐克2023年的收入是多少？"
)
doc, score = results_with_score[0]
print(f"相似度分数：{score}")
```

### 4. 检索器

检索器是Runnable对象，可以轻松集成到更复杂的应用中：

```python
# 从向量存储创建检索器
retriever = vector_store.as_retriever(
    search_type="similarity",
    search_kwargs={"k": 1}  # 返回最相似的1个文档
)

# 批量查询
results = retriever.batch([
    "耐克在美国有多少个分销中心？",
    "耐克是什么时候成立的？"
])
```

## 构建完整的语义搜索引擎

以下是完整的代码示例：

```python
import os
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_core.vectorstores import InMemoryVectorStore

def build_semantic_search_engine(pdf_path, api_key):
    """构建语义搜索引擎"""
    
    # 1. 设置API密钥
    os.environ["OPENAI_API_KEY"] = api_key
    
    # 2. 加载文档
    print("正在加载PDF文档...")
    loader = PyPDFLoader(pdf_path)
    docs = loader.load()
    
    # 3. 分割文档
    print("正在分割文档...")
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        add_start_index=True
    )
    all_splits = text_splitter.split_documents(docs)
    
    # 4. 初始化嵌入模型
    print("正在初始化嵌入模型...")
    embeddings = OpenAIEmbeddings(model="text-embedding-3-large")
    
    # 5. 创建向量存储
    print("正在构建向量索引...")
    vector_store = InMemoryVectorStore(embeddings)
    vector_store.add_documents(documents=all_splits)
    
    # 6. 创建检索器
    retriever = vector_store.as_retriever(
        search_type="similarity",
        search_kwargs={"k": 3}  # 返回最相似的3个文档
    )
    
    print("语义搜索引擎构建完成！")
    return retriever

# 使用示例
if __name__ == "__main__":
    # 构建搜索引擎
    search_engine = build_semantic_search_engine(
        pdf_path="nke-10k-2023.pdf",
        api_key="your-openai-api-key"
    )
    
    # 进行查询
    query = "耐克2023年的财务表现如何？"
    results = search_engine.invoke(query)
    
    print(f"查询：{query}")
    print("检索结果：")
    for i, doc in enumerate(results, 1):
        print(f"\n--- 结果 {i} ---")
        print(doc.page_content[:500] + "...")  # 显示前500个字符
        print(f"来源：{doc.metadata.get('source', '未知')}")
```

## 进阶功能

### 最大边际相关性搜索

为了避免返回过于相似的结果，可以使用MMR搜索：

```python
retriever = vector_store.as_retriever(
    search_type="mmr",
    search_kwargs={"k": 3, "fetch_k": 10}  # 从10个候选中选择3个最不同的
)
```

### 相似度阈值过滤

```python
retriever = vector_store.as_retriever(
    search_type="similarity_score_threshold",
    search_kwargs={"k": 3, "score_threshold": 0.7}  # 只返回相似度大于0.7的文档
)
```

## 故障排除

1. **内存不足**：对于大型文档，考虑使用持久化向量存储（如Chroma、FAISS）
2. **API限制**：注意嵌入模型的API调用限制和成本
3. **分割策略**：根据文档类型调整chunk_size和chunk_overlap参数

## 下一步

- 探索更多[文档加载器集成](/oss/python/integrations/document_loaders/)
- 了解不同的[嵌入模型选项](/oss/python/integrations/text_embedding/)
- 尝试各种[向量存储解决方案](/oss/python/integrations/vectorstores/)
- 学习如何构建完整的[RAG应用程序](/oss/python/langchain/rag/)

这个语义搜索引擎可以作为构建更复杂AI应用的基础，如智能问答系统、文档分析工具等。
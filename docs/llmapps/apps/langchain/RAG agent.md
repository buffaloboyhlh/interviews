# 基于LangChain创建RAG agent

## 项目概述

基于这篇博客 https://lilianweng.github.io/posts/2023-06-23-agent/ 作为数据源，建立RAG agent。

## 项目实现

### 1️⃣ 搭建基础环境

```python
import os

os.environ['LANGSMITH_TRACING'] ='true'
os.environ['LANGSMITH_API_KEY'] = '***************************************'
```

### 2️⃣ 加载文档

```python
import bs4
from langchain_community.document_loaders import WebBaseLoader

# 只保留 文章的名称、标题和内容
bs4_strainer = bs4.SoupStrainer(class_=("post-title", "post-header", "post-content"))
loader = WebBaseLoader(
    web_path=("https://lilianweng.github.io/posts/2023-06-23-agent/",),
    bs_kwargs={"parse_only": bs4_strainer},
)

docs = loader.load() # 加载文档

assert len(docs) == 1
print(f"总字数：{len(docs[0].page_content)}")
print(f"前500个字的内容：{docs[0].page_content[:500]}")
```

### 3️⃣ 文档分割

```python
from langchain_text_splitters import RecursiveCharacterTextSplitter

text_spliter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200,
    add_start_index=True
)

all_splits = text_spliter.split_documents(docs)
print(f"把文档拆分成{len(all_splits)}个子文档")
```

### 4️⃣ 存储文档

```python
from langchain_milvus import Milvus
from langchain_ollama import OllamaEmbeddings

# 嵌入
embeddings = OllamaEmbeddings(model="llama3.2:latest") # 嵌入

# 向量数据库
uri = 'http://localhost:19530'
vector_store = Milvus(
    embedding_function=embeddings,
    connection_args={'uri':uri}
)

# 存入到向量库中
document_ids = vector_store.add_documents(documents=all_splits)
print(f"前三个document_ids:{document_ids[:3]}")
```

### 5️⃣ 创建Agent

#### 5.1 定义工具

```python
from langchain.tools import tool

@tool(response_format="content_and_artifact")
def retrieve_context(query:str):
    '''检索上下文帮助问题回答'''
    retrieved_docs = vector_store.similarity_search(query=query,k=2) # 检索文章，并返回2个文档
    serialized = "\n\n".join([
        (f"Source:{doc.metadata}\nContent:{doc.page_content}")
        for doc in retrieved_docs
    ])
    return serialized,retrieved_docs
```

#### 5.2 创建智能体

```python
from langchain.agents import create_agent
from langchain_ollama import ChatOllama
from langchain.messages import HumanMessage
# 工具集
tools = [retrieve_context]
# 系统提示词
system_prompt = (
    "You have access to a tool that retrieves context from a blog post. "
    "Use the tool to help answer user queries."
)

# 选择模型
model = ChatOllama(
    model="qwen3:1.7b",
    temperature=0.8
)

# 创建agent
agent = create_agent(model=model, system_prompt=system_prompt,tools=tools)

query = (
    "What is the standard method for Task Decomposition?\n\n"
    "Once you get the answer, look up common extensions of that method."
)

for event in agent.stream(
    {"messages": [{"role": "user", "content": query}]},
    stream_mode="values",
):
    event["messages"][-1].pretty_print()
```

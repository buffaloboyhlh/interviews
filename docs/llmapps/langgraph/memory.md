# LangGraph 内存管理完整教程

## 概述

在AI应用中，内存（Memory）是实现多轮对话和上下文共享的关键组件。LangGraph提供了两种类型的内存管理：

- **短期内存**：用于跟踪多轮对话的线程级持久化
- **长期内存**：用于跨会话存储用户特定或应用级别的数据

## 短期内存配置

### 基础配置

```python
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import StateGraph

# 创建内存检查点保存器
checkpointer = InMemorySaver()

# 构建图并配置检查点
builder = StateGraph(...)
graph = builder.compile(checkpointer=checkpointer)

# 使用线程ID调用图
graph.invoke(
    {"messages": [{"role": "user", "content": "hi! i am Bob"}]},
    {"configurable": {"thread_id": "1"}},
)
```

### 生产环境配置

在生产环境中，建议使用数据库支持的检查点：

```python
from langgraph.checkpoint.postgres import PostgresSaver

DB_URI = "postgresql://postgres:postgres@localhost:5442/postgres?sslmode=disable"
with PostgresSaver.from_conn_string(DB_URI) as checkpointer:
    builder = StateGraph(...)
    graph = builder.compile(checkpointer=checkpointer)
```

#### PostgreSQL 示例

```bash
pip install -U "psycopg[binary,pool]" langgraph langgraph-checkpoint-postgres
```

**同步版本：**
```python
from langchain.chat_models import init_chat_model
from langgraph.graph import StateGraph, MessagesState, START
from langgraph.checkpoint.postgres import PostgresSaver

model = init_chat_model(model="anthropic:claude-3-5-haiku-latest")

DB_URI = "postgresql://postgres:postgres@localhost:5442/postgres?sslmode=disable"
with PostgresSaver.from_conn_string(DB_URI) as checkpointer:
    # 首次使用时需要设置：checkpointer.setup()
    
    def call_model(state: MessagesState):
        response = model.invoke(state["messages"])
        return {"messages": response}
    
    builder = StateGraph(MessagesState)
    builder.add_node(call_model)
    builder.add_edge(START, "call_model")
    
    graph = builder.compile(checkpointer=checkpointer)
    
    config = {"configurable": {"thread_id": "1"}}
    
    # 多轮对话示例
    for chunk in graph.stream(
        {"messages": [{"role": "user", "content": "hi! I'm bob"}]},
        config,
        stream_mode="values"
    ):
        chunk["messages"][-1].pretty_print()
    
    for chunk in graph.stream(
        {"messages": [{"role": "user", "content": "what's my name?"}]},
        config,
        stream_mode="values"
    ):
        chunk["messages"][-1].pretty_print()
```

#### MongoDB 示例

```bash
pip install -U pymongo langgraph langgraph-checkpoint-mongodb
```

```python
from langgraph.checkpoint.mongodb import MongoDBSaver

DB_URI = "localhost:27017"
with MongoDBSaver.from_conn_string(DB_URI) as checkpointer:
    # 配置图...
```

#### Redis 示例

```bash
pip install -U langgraph langgraph-checkpoint-redis
```

```python
from langgraph.checkpoint.redis import RedisSaver

DB_URI = "redis://localhost:6379"
with RedisSaver.from_conn_string(DB_URI) as checkpointer:
    # 首次使用时需要设置：checkpointer.setup()
    # 配置图...
```

### 在子图中使用内存

如果图中包含子图，只需在父图编译时提供检查点，LangGraph会自动传播到子图：

```python
from langgraph.graph import START, StateGraph
from langgraph.checkpoint.memory import InMemorySaver
from typing import TypedDict

class State(TypedDict):
    foo: str

# 子图配置
def subgraph_node_1(state: State):
    return {"foo": state["foo"] + "bar"}

subgraph_builder = StateGraph(State)
subgraph_builder.add_node(subgraph_node_1)
subgraph_builder.add_edge(START, "subgraph_node_1")
subgraph = subgraph_builder.compile()

# 父图配置
builder = StateGraph(State)
builder.add_node("node_1", subgraph)
builder.add_edge(START, "node_1")

checkpointer = InMemorySaver()
graph = builder.compile(checkpointer=checkpointer)
```

如果希望子图拥有独立内存：

```python
subgraph_builder = StateGraph(...)
subgraph = subgraph_builder.compile(checkpointer=True)
```

## 长期内存配置

### 基础配置

```python
from langgraph.store.memory import InMemoryStore
from langgraph.graph import StateGraph

store = InMemoryStore()

builder = StateGraph(...)
graph = builder.compile(store=store)
```

### 生产环境配置

```python
from langgraph.store.postgres import PostgresStore

DB_URI = "postgresql://postgres:postgres@localhost:5442/postgres?sslmode=disable"
with PostgresStore.from_conn_string(DB_URI) as store:
    builder = StateGraph(...)
    graph = builder.compile(store=store)
```

#### PostgreSQL 存储示例

```python
from langchain_core.runnables import RunnableConfig
from langchain.chat_models import init_chat_model
from langgraph.graph import StateGraph, MessagesState, START
from langgraph.checkpoint.postgres import PostgresSaver
from langgraph.store.postgres import PostgresStore
from langgraph.store.base import BaseStore
import uuid

model = init_chat_model(model="anthropic:claude-3-5-haiku-latest")

DB_URI = "postgresql://postgres:postgres@localhost:5442/postgres?sslmode=disable"

with (
    PostgresStore.from_conn_string(DB_URI) as store,
    PostgresSaver.from_conn_string(DB_URI) as checkpointer,
):
    # store.setup()  # 首次使用时需要设置
    # checkpointer.setup()
    
    def call_model(
        state: MessagesState,
        config: RunnableConfig,
        *,
        store: BaseStore,
    ):
        user_id = config["configurable"]["user_id"]
        namespace = ("memories", user_id)
        
        # 搜索相关记忆
        memories = store.search(namespace, query=str(state["messages"][-1].content))
        info = "\n".join([d.value["data"] for d in memories])
        system_msg = f"You are a helpful assistant talking to the user. User info: {info}"
        
        # 如果用户要求记住，存储新记忆
        last_message = state["messages"][-1]
        if "remember" in last_message.content.lower():
            memory = "User name is Bob"
            store.put(namespace, str(uuid.uuid4()), {"data": memory})
        
        response = model.invoke(
            [{"role": "system", "content": system_msg}] + state["messages"]
        )
        return {"messages": response}
    
    builder = StateGraph(MessagesState)
    builder.add_node(call_model)
    builder.add_edge(START, "call_model")
    
    graph = builder.compile(
        checkpointer=checkpointer,
        store=store,
    )
    
    # 跨线程共享用户记忆
    config1 = {"configurable": {"thread_id": "1", "user_id": "1"}}
    config2 = {"configurable": {"thread_id": "2", "user_id": "1"}}
```

### 语义搜索

启用语义搜索功能：

```python
from langchain.embeddings import init_embeddings
from langgraph.store.memory import InMemoryStore

# 创建支持语义搜索的存储
embeddings = init_embeddings("openai:text-embedding-3-small")
store = InMemoryStore(
    index={
        "embed": embeddings,
        "dims": 1536,
    }
)

# 存储数据
store.put(("user_123", "memories"), "1", {"text": "I love pizza"})
store.put(("user_123", "memories"), "2", {"text": "I am a plumber"})

# 语义搜索
items = store.search(
    ("user_123", "memories"), query="I'm hungry", limit=1
)
```

## 短期内存管理策略

### 消息修剪

当对话历史超过LLM上下文窗口时，可以修剪消息：

```python
from langchain_core.messages.utils import trim_messages, count_tokens_approximately

def call_model(state: MessagesState):
    messages = trim_messages(
        state["messages"],
        strategy="last",  # 保留最后的消息
        token_counter=count_tokens_approximately,
        max_tokens=128,   # 最大token数
        start_on="human",
        end_on=("human", "tool"),
    )
    response = model.invoke(messages)
    return {"messages": [response]}
```

### 消息删除

删除特定消息：

```python
from langchain.messages import RemoveMessage

def delete_messages(state):
    messages = state["messages"]
    if len(messages) > 2:
        # 删除最早的两条消息
        return {"messages": [RemoveMessage(id=m.id) for m in messages[:2]]}
```

删除所有消息：

```python
from langgraph.graph.message import REMOVE_ALL_MESSAGES

def delete_all_messages(state):
    return {"messages": [RemoveMessage(id=REMOVE_ALL_MESSAGES)]}
```

### 消息摘要

使用摘要来压缩对话历史：

```python
from typing import Any, TypedDict
from langchain.messages import AnyMessage
from langgraph.graph import StateGraph, START, MessagesState
from langmem.short_term import SummarizationNode, RunningSummary

class State(MessagesState):
    context: dict[str, RunningSummary]

summarization_node = SummarizationNode(
    token_counter=count_tokens_approximately,
    model=summarization_model,
    max_tokens=256,
    max_tokens_before_summary=256,
    max_summary_tokens=128,
)

# 在图中使用摘要节点
builder = StateGraph(State)
builder.add_node(call_model)
builder.add_node("summarize", summarization_node)
builder.add_edge(START, "summarize")
builder.add_edge("summarize", "call_model")
```

## 检查点管理

### 查看线程状态

```python
config = {"configurable": {"thread_id": "1"}}

# 使用图API
state_snapshot = graph.get_state(config)

# 使用检查点API
checkpoint_tuple = checkpointer.get_tuple(config)
```

### 查看线程历史

```python
config = {"configurable": {"thread_id": "1"}}

# 获取状态历史
history = list(graph.get_state_history(config))

# 获取检查点历史
checkpoints = list(checkpointer.list(config))
```

### 删除线程检查点

```python
thread_id = "1"
checkpointer.delete_thread(thread_id)
```

## 最佳实践

1. **生产环境**：始终使用数据库支持的检查点和存储
2. **内存管理**：根据对话长度选择合适的策略（修剪、删除或摘要）
3. **错误处理**：确保消息删除后的历史仍然有效
4. **性能优化**：使用语义搜索提高长期内存的检索效率
5. **多租户**：使用不同的命名空间隔离不同用户的数据

## 预构建内存工具

LangMem是LangChain维护的库，提供了管理长期记忆的工具。参考[LangMem文档](https://langchain-ai.github.io/langmem/)获取更多使用示例。

通过合理配置短期和长期内存，可以构建出能够进行复杂多轮对话并保持上下文连贯性的AI应用。
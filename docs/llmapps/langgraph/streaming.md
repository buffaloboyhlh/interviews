# LangGraph 流式传输完整教程

## 概述

LangGraph 实现了流式传输系统，能够实时展示更新。流式传输对于提升基于 LLM 的应用程序的响应能力至关重要。通过逐步显示输出（即使在完整响应准备好之前），流式传输显著改善了用户体验，特别是在处理 LLM 延迟时。

## 支持的流模式

| 模式 | 描述 |
|------|------|
| `values` | 在图的每个步骤后流式传输状态的完整值 |
| `updates` | 在图的每个步骤后流式传输状态的更新 |
| `custom` | 从图节点内部流式传输自定义数据 |
| `messages` | 从任何调用 LLM 的图节点流式传输 2 元组 (LLM token, 元数据) |
| `debug` | 在图执行过程中流式传输尽可能多的信息 |

## 基础用法

### 基本流式传输示例

```python
# 同步流式传输
for chunk in graph.stream(inputs, stream_mode="updates"):
    print(chunk)

# 异步流式传输
async for chunk in graph.astream(inputs, stream_mode="updates"):
    print(chunk)
```

### 完整示例

```python
from typing import TypedDict
from langgraph.graph import StateGraph, START, END

class State(TypedDict):
    topic: str
    joke: str

def refine_topic(state: State):
    return {"topic": state["topic"] + " and cats"}

def generate_joke(state: State):
    return {"joke": f"This is a joke about {state['topic']}"}

# 构建图
graph = (
    StateGraph(State)
    .add_node(refine_topic)
    .add_node(generate_joke)
    .add_edge(START, "refine_topic")
    .add_edge("refine_topic", "generate_joke")
    .add_edge("generate_joke", END)
    .compile()
)

# 流式传输更新
for chunk in graph.stream(
    {"topic": "ice cream"},
    stream_mode="updates",
):
    print(chunk)
```

输出：
```
{'refine_topic': {'topic': 'ice cream and cats'}}
{'generate_joke': {'joke': 'This is a joke about ice cream and cats'}}
```

## 多模式流式传输

可以同时流式传输多种模式：

```python
for mode, chunk in graph.stream(
    inputs, 
    stream_mode=["updates", "custom"]
):
    print(f"{mode}: {chunk}")
```

## 流式传输图状态

### 使用 updates 模式

流式传输状态更新（仅显示变化的部分）：

```python
for chunk in graph.stream(
    {"topic": "ice cream"},
    stream_mode="updates",
):
    print(chunk)
```

### 使用 values 模式

流式传输完整状态值：

```python
for chunk in graph.stream(
    {"topic": "ice cream"},
    stream_mode="values",
):
    print(chunk)
```

## 流式传输子图输出

包含子图输出的流式传输：

```python
for chunk in graph.stream(
    {"foo": "foo"},
    subgraphs=True,  # 启用子图流式传输
    stream_mode="updates",
):
    print(chunk)
```

## 流式传输 LLM Token

### 基础 LLM Token 流式传输

```python
from langchain.chat_models import init_chat_model
from langgraph.graph import StateGraph, START

model = init_chat_model(model="openai:gpt-4o-mini")

def call_model(state):
    """调用 LLM 生成关于主题的笑话"""
    model_response = model.invoke([
        {"role": "user", "content": f"Generate a joke about {state['topic']}"}
    ])
    return {"joke": model_response.content}

graph = (
    StateGraph(State)
    .add_node(call_model)
    .add_edge(START, "call_model")
    .compile()
)

# 流式传输 LLM token
for message_chunk, metadata in graph.stream(
    {"topic": "ice cream"},
    stream_mode="messages",
):
    if message_chunk.content:
        print(message_chunk.content, end="|", flush=True)
```

### 按标签过滤 LLM 调用

```python
# 为不同模型设置标签
joke_model = init_chat_model(model="openai:gpt-4o-mini", tags=['joke'])
poem_model = init_chat_model(model="openai:gpt-4o-mini", tags=['poem'])

# 流式传输时按标签过滤
async for msg, metadata in graph.astream(
    {"topic": "cats"},
    stream_mode="messages",
):
    if metadata["tags"] == ["joke"]:
        print(msg.content, end="|", flush=True)
```

### 按节点过滤

```python
for msg, metadata in graph.stream(
    inputs,
    stream_mode="messages",
):
    # 只流式传输特定节点的 token
    if msg.content and metadata["langgraph_node"] == "write_poem":
        print(msg.content, end="|", flush=True)
```

## 流式传输自定义数据

### 从节点流式传输自定义数据

```python
from typing import TypedDict
from langgraph.config import get_stream_writer
from langgraph.graph import StateGraph, START

class State(TypedDict):
    query: str
    answer: str

def node(state: State):
    # 获取流写入器发送自定义数据
    writer = get_stream_writer()
    # 发送自定义数据
    writer({"custom_key": "Generating custom data inside node"})
    return {"answer": "some data"}

graph = (
    StateGraph(State)
    .add_node(node)
    .add_edge(START, "node")
    .compile()
)

# 接收自定义数据
for chunk in graph.stream(inputs, stream_mode="custom"):
    print(chunk)
```

### 从工具流式传输自定义数据

```python
from langchain.tools import tool
from langgraph.config import get_stream_writer

@tool
def query_database(query: str) -> str:
    """查询数据库"""
    writer = get_stream_writer()
    # 发送进度更新
    writer({"data": "Retrieved 0/100 records", "type": "progress"})
    # 执行查询
    writer({"data": "Retrieved 100/100 records", "type": "progress"})
    return "some-answer"
```

## 与任意 LLM 一起使用

即使 LLM API 没有实现 LangChain 聊天模型接口，也可以使用自定义模式进行流式传输：

```python
from langgraph.config import get_stream_writer

def call_arbitrary_model(state):
    """调用任意模型并流式传输输出"""
    writer = get_stream_writer()
    
    # 使用自定义流式客户端
    for chunk in your_custom_streaming_client(state["topic"]):
        # 将自定义数据发送到流
        writer({"custom_llm_chunk": chunk})
    
    return {"result": "completed"}

# 接收自定义数据
for chunk in graph.stream(
    {"topic": "cats"},
    stream_mode="custom",
):
    print(chunk)
```

## 调试模式

使用调试模式获取详细信息：

```python
for chunk in graph.stream(
    {"topic": "ice cream"},
    stream_mode="debug",
):
    print(chunk)
```

## 禁用特定聊天模型的流式传输

对于不支持流式传输的模型：

```python
from langchain.chat_models import init_chat_model

model = init_chat_model(
    "anthropic:claude-sonnet-4-5",
    disable_streaming=True  # 禁用流式传输
)
```

## Python < 3.11 的异步处理

### 手动传递配置

```python
async def call_model(state, config):
    topic = state["topic"]
    # 必须显式传递 config
    joke_response = await model.ainvoke(
        [{"role": "user", "content": f"Write a joke about {topic}"}],
        config,  # 显式传递配置
    )
    return {"joke": joke_response.content}
```

### 异步自定义流式传输

```python
from langgraph.types import StreamWriter

async def generate_joke(state: State, writer: StreamWriter):
    # 在异步函数中直接使用 writer 参数
    writer({"custom_key": "Streaming custom data"})
    return {"joke": f"This is a joke about {state['topic']}"}
```

## 最佳实践

1. **选择合适的流模式**：根据需求选择 `updates`、`values`、`messages` 或 `custom` 模式
2. **合理使用过滤**：使用标签或节点名称过滤流式输出
3. **处理异步场景**：在 Python < 3.11 中注意手动传递配置
4. **错误处理**：在流式传输过程中添加适当的错误处理机制
5. **性能考虑**：避免在流式传输中执行阻塞操作

通过本教程，您可以充分利用 LangGraph 的流式传输功能，构建响应迅速、用户体验良好的 LLM 应用程序。
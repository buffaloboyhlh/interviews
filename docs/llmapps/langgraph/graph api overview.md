# LangGraph 入门教程：构建智能代理工作流

## 概述

LangGraph 是一个用于构建智能代理工作流的框架，它将工作流程建模为**图**。通过定义状态、节点和边，你可以创建复杂、可循环的工作流。

## 核心概念

### 1. 图的基本组成

LangGraph 工作流由三个核心组件构成：

- **State（状态）**：共享数据结构，表示应用的当前快照
- **Nodes（节点）**：执行具体逻辑的函数，接收状态并返回更新
- **Edges（边）**：决定下一个执行节点的路由逻辑

> **核心原则**：*节点执行工作，边决定下一步做什么*

## 快速开始

### 步骤1：定义状态

首先定义图的状态结构，通常使用 `TypedDict`：

```python
from typing_extensions import TypedDict

class State(TypedDict):
    user_input: str
    processed_result: str
    conversation_history: list
```

### 步骤2：创建图构建器

```python
from langgraph.graph import StateGraph

builder = StateGraph(State)
```

### 步骤3：添加节点

节点是执行具体工作的函数：

```python
def process_input(state: State):
    # 处理用户输入
    processed = state["user_input"].upper()
    return {"processed_result": processed}

def generate_response(state: State):
    # 生成响应
    response = f"Processed: {state['processed_result']}"
    return {"conversation_history": [response]}

# 添加节点到图中
builder.add_node("process_input", process_input)
builder.add_node("generate_response", generate_response)
```

### 步骤4：定义边

连接节点，定义执行流程：

```python
from langgraph.graph import START, END

# 设置入口点
builder.add_edge(START, "process_input")
# 连接处理节点到响应节点
builder.add_edge("process_input", "generate_response")
# 设置结束点
builder.add_edge("generate_response", END)
```

### 步骤5：编译图

**必须编译后才能使用**：

```python
graph = builder.compile()
```

### 步骤6：执行图

```python
# 输入初始状态
result = graph.invoke({"user_input": "hello world"})
print(result)
# 输出: {'user_input': 'hello world', 'processed_result': 'HELLO WORLD', 'conversation_history': ['Processed: HELLO WORLD']}
```

## 深入理解状态管理

### 状态归约器

归约器定义如何更新状态：

```python
from typing import Annotated
from operator import add

class State(TypedDict):
    # 默认归约器：覆盖更新
    current_value: int
    # 使用add归约器：追加更新
    history: Annotated[list, add]
```

### 消息处理

处理对话消息的常用模式：

```python
from langchain.messages import AnyMessage
from langgraph.graph.message import add_messages

class GraphState(TypedDict):
    messages: Annotated[list[AnyMessage], add_messages]
```

或者使用预定义的 `MessagesState`：

```python
from langgraph.graph import MessagesState

class State(MessagesState):
    additional_data: str
```

## 高级节点功能

### 节点参数

节点可以接收多种参数：

```python
def advanced_node(
    state: State, 
    config: RunnableConfig, 
    runtime: Runtime
):
    print(f"Thread ID: {config['configurable']['thread_id']}")
    # 执行节点逻辑
    return {"result": "success"}
```

### 节点缓存

对计算密集型节点启用缓存：

```python
from langgraph.types import CachePolicy

# 设置缓存策略（TTL=5秒）
cache_policy = CachePolicy(ttl=5)
builder.add_node("expensive_node", expensive_function, cache_policy=cache_policy)
```

## 复杂路由控制

### 条件边

根据条件决定下一个节点：

```python
def routing_function(state: State):
    if len(state["user_input"]) > 10:
        return "long_input_node"
    else:
        return "short_input_node"

builder.add_conditional_edges("process_input", routing_function)
```

### 使用映射表

```python
route_map = {
    "long": "long_input_node",
    "short": "short_input_node"
}
builder.add_conditional_edges("process_input", routing_function, route_map)
```

## 高级特性

### Send API

用于动态生成边（如map-reduce模式）：

```python
from langgraph.types import Send

def split_and_process(state: State):
    words = state["user_input"].split()
    return [Send("process_word", {"word": word}) for word in words]
```

### Command API

结合状态更新和路由控制：

```python
from langgraph.types import Command
from typing import Literal

def smart_node(state: State) -> Command[Literal["next_node"]]:
    if state["value"] > 100:
        return Command(
            update={"status": "high"},
            goto="handle_high_value"
        )
    else:
        return Command(
            update={"status": "low"}, 
            goto="handle_low_value"
        )
```

## 运行时配置

### 添加上下文

```python
from dataclasses import dataclass

@dataclass
class ContextSchema:
    user_id: str
    api_key: str

graph = StateGraph(State, context_schema=ContextSchema)

# 使用上下文
def context_aware_node(state: State, runtime: Runtime[ContextSchema]):
    user_id = runtime.context.user_id
    # 使用上下文信息
```

### 递归限制

防止无限循环：

```python
# 限制最大执行步数
result = graph.invoke(
    inputs, 
    config={"recursion_limit": 50}
)
```

## 完整示例

下面是一个完整的对话代理示例：

```python
from typing_extensions import TypedDict
from typing import Annotated
from langgraph.graph import StateGraph, MessagesState, add_messages
from langchain.messages import HumanMessage, AIMessage

class State(MessagesState):
    needs_clarification: bool = False

def process_input(state: State):
    last_message = state["messages"][-1]
    
    if "?" in last_message.content:
        return {"needs_clarification": True}
    return {"needs_clarification": False}

def generate_clarification(state: State):
    return {"messages": [AIMessage(content="Could you please clarify your question?")]}

def generate_response(state: State):
    last_message = state["messages"][-1]
    response = f"I understand you said: {last_message.content}"
    return {"messages": [AIMessage(content=response)]}

def route_conversation(state: State):
    if state["needs_clarification"]:
        return "generate_clarification"
    else:
        return "generate_response"

# 构建图
builder = StateGraph(State)
builder.add_node("process_input", process_input)
builder.add_node("generate_clarification", generate_clarification)
builder.add_node("generate_response", generate_response)

builder.add_edge(START, "process_input")
builder.add_conditional_edges("process_input", route_conversation)
builder.add_edge("generate_clarification", END)
builder.add_edge("generate_response", END)

graph = builder.compile()

# 使用图
result = graph.invoke({
    "messages": [HumanMessage(content="Hello, how are you?")]
})
```

## 最佳实践

1. **状态设计**：保持状态简洁，只包含必要的数据
2. **节点职责**：每个节点应该只负责一个明确的职责
3. **错误处理**：在节点中添加适当的错误处理逻辑
4. **测试**：对每个节点进行单元测试，对整个工作流进行集成测试
5. **监控**：利用LangGraph的追踪功能监控工作流执行

## 总结

LangGraph 提供了一个强大的框架来构建复杂的代理工作流。通过理解状态管理、节点执行和路由控制，你可以创建高效、可维护的智能应用。记住关键的工作流程：**定义状态 → 添加节点 → 连接边 → 编译执行**。

开始构建你的第一个LangGraph工作流吧！
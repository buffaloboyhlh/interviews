# LangGraph 子图使用教程

## 概述

本教程将详细介绍如何在 LangGraph 中使用子图（Subgraphs）。子图是指在一个图中作为节点使用的另一个图，这种设计模式在构建复杂系统时非常有用。

## 子图的优势

- **构建多智能体系统**：每个智能体可以作为独立的子图
- **节点复用**：在多个图中重复使用同一组节点
- **分布式开发**：不同团队可以独立开发不同的子图部分，只要保持接口规范即可

## 环境设置

首先安装必要的依赖：

```bash
# 使用 pip
pip install -U langgraph

# 使用 uv
uv add langgraph
```

**提示**：建议设置 [LangSmith](https://smith.langchain.com) 来监控和调试 LangGraph 应用。

## 两种子图实现方式

### 1. 从节点调用图

当子图和父图有完全不同的状态模式时，可以使用这种方式。

```python
from typing_extensions import TypedDict
from langgraph.graph.state import StateGraph, START

# 定义子图状态
class SubgraphState(TypedDict):
    bar: str

# 子图实现
def subgraph_node_1(state: SubgraphState):
    return {"bar": "hi! " + state["bar"]}

subgraph_builder = StateGraph(SubgraphState)
subgraph_builder.add_node(subgraph_node_1)
subgraph_builder.add_edge(START, "subgraph_node_1")
subgraph = subgraph_builder.compile()

# 父图实现
class State(TypedDict):
    foo: str

def call_subgraph(state: State):
    # 将父图状态转换为子图状态
    subgraph_output = subgraph.invoke({"bar": state["foo"]})
    # 将子图响应转换回父图状态
    return {"foo": subgraph_output["bar"]}

builder = StateGraph(State)
builder.add_node("node_1", call_subgraph)
builder.add_edge(START, "node_1")
graph = builder.compile()
```

### 2. 将图作为节点添加

当子图和父图共享状态键时，可以直接将子图作为节点添加到父图中。

```python
from typing_extensions import TypedDict
from langgraph.graph.state import StateGraph, START

class State(TypedDict):
    foo: str  # 共享的状态键

# 子图实现
def subgraph_node_1(state: State):
    return {"foo": "hi! " + state["foo"]}

subgraph_builder = StateGraph(State)
subgraph_builder.add_node(subgraph_node_1)
subgraph_builder.add_edge(START, "subgraph_node_1")
subgraph = subgraph_builder.compile()

# 父图实现 - 直接添加子图作为节点
builder = StateGraph(State)
builder.add_node("node_1", subgraph)  # 直接传递编译后的子图
builder.add_edge(START, "node_1")
graph = builder.compile()
```

## 添加持久化

### 为父图和子图共享存储

```python
from langgraph.graph import START, StateGraph
from langgraph.checkpoint.memory import MemorySaver

class State(TypedDict):
    foo: str

# 子图实现
def subgraph_node_1(state: State):
    return {"foo": state["foo"] + "bar"}

subgraph_builder = StateGraph(State)
subgraph_builder.add_node(subgraph_node_1)
subgraph_builder.add_edge(START, "subgraph_node_1")
subgraph = subgraph_builder.compile()

# 父图实现
builder = StateGraph(State)
builder.add_node("node_1", subgraph)
builder.add_edge(START, "node_1")

# 只为父图提供检查点，子图会自动继承
checkpointer = MemorySaver()
graph = builder.compile(checkpointer=checkpointer)
```

### 为子图设置独立存储

在多智能体系统中，可能需要为每个子图设置独立的存储：

```python
subgraph_builder = StateGraph(State)
subgraph = subgraph_builder.compile(checkpointer=True)  # 子图有自己的存储
```

## 查看子图状态

当启用持久化时，可以查看子图的状态。**注意：只能在子图被中断时查看其状态**。

```python
# 获取父图状态
parent_state = graph.get_state(config)

# 获取子图状态（仅在子图中断时可用）
subgraph_state = graph.get_state(config, subgraphs=True).tasks[0].state
```

## 流式输出子图结果

要在流式输出中包含子图的输出，可以在流式方法中设置 `subgraphs=True`：

```python
for chunk in graph.stream(
    {"foo": "foo"},
    subgraphs=True,  # 包含子图输出
    stream_mode="updates",
):
    print(chunk)
```

## 完整示例

### 共享状态模式的完整示例

```python
from typing_extensions import TypedDict
from langgraph.graph.state import StateGraph, START

# 定义子图状态
class SubgraphState(TypedDict):
    foo: str  # 与父图共享
    bar: str  # 子图私有

def subgraph_node_1(state: SubgraphState):
    return {"bar": "bar"}

def subgraph_node_2(state: SubgraphState):
    return {"foo": state["foo"] + state["bar"]}

subgraph_builder = StateGraph(SubgraphState)
subgraph_builder.add_node(subgraph_node_1)
subgraph_builder.add_node(subgraph_node_2)
subgraph_builder.add_edge(START, "subgraph_node_1")
subgraph_builder.add_edge("subgraph_node_1", "subgraph_node_2")
subgraph = subgraph_builder.compile()

# 父图实现
class ParentState(TypedDict):
    foo: str

def node_1(state: ParentState):
    return {"foo": "hi! " + state["foo"]}

builder = StateGraph(ParentState)
builder.add_node("node_1", node_1)
builder.add_node("node_2", subgraph)
builder.add_edge(START, "node_1")
builder.add_edge("node_1", "node_2")
graph = builder.compile()

# 执行并查看结果
for chunk in graph.stream({"foo": "foo"}):
    print(chunk)
```

## 使用场景建议

1. **选择从节点调用图**：当子图和父图状态模式完全不同时
2. **选择将图作为节点添加**：当子图和父图共享部分状态键时
3. **多级子图**：可以构建父图->子图->孙图的多级结构
4. **独立存储**：在多智能体系统中，为每个智能体设置独立的消息历史存储

通过合理使用子图，可以构建出结构清晰、易于维护的复杂图工作流。
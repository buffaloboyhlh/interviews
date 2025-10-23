# LangGraph Graph API 教程

本教程将演示 LangGraph Graph API 的基础知识，包括状态管理、常见图结构（序列、分支、循环）的构建，以及 LangGraph 的控制特性。

## 安装与设置

首先安装 `langgraph`：

```bash
pip install -U langgraph
```

或使用 uv：

```bash
uv add langgraph
```

**提示**：设置 LangSmith 以获得更好的调试体验，可以快速发现并改进 LangGraph 项目的性能问题。

## 定义和更新状态

### 定义状态

在 LangGraph 中，状态可以是 `TypedDict`、`Pydantic` 模型或数据类。以下使用 `TypedDict`：

```python
from langchain.messages import AnyMessage
from typing_extensions import TypedDict

class State(TypedDict):
    messages: list[AnyMessage]
    extra_field: int
```

### 更新状态

节点是读取和更新状态的 Python 函数：

```python
from langchain.messages import AIMessage

def node(state: State):
    messages = state["messages"]
    new_message = AIMessage("Hello!")
    return {"messages": messages + [new_message], "extra_field": 10}
```

**警告**：节点应直接返回状态更新，而不是改变状态。

构建图：

```python
from langgraph.graph import StateGraph

builder = StateGraph(State)
builder.add_node(node)
builder.set_entry_point("node")
graph = builder.compile()
```

可视化图：

```python
from IPython.display import Image, display

display(Image(graph.get_graph().draw_mermaid_png()))
```

调用图：

```python
from langchain.messages import HumanMessage

result = graph.invoke({"messages": [HumanMessage("Hi")]})
```

### 使用 Reducers 处理状态更新

每个状态键可以有独立的 reducer 函数来控制更新处理方式：

```python
from typing_extensions import Annotated

def add(left, right):
    return left + right

class State(TypedDict):
    messages: Annotated[list[AnyMessage], add]
    extra_field: int
```

现在节点可以简化：

```python
def node(state: State):
    new_message = AIMessage("Hello!")
    return {"messages": [new_message], "extra_field": 10}
```

### MessagesState

LangGraph 包含内置的 `add_messages` reducer：

```python
from langgraph.graph.message import add_messages

class State(TypedDict):
    messages: Annotated[list[AnyMessage], add_messages]
    extra_field: int
```

也可以使用预构建的 `MessagesState`：

```python
from langgraph.graph import MessagesState

class State(MessagesState):
    extra_field: int
```

## 运行时配置

添加运行时配置：

1. 指定配置模式
2. 在节点或条件边函数签名中添加配置
3. 将配置传递到图中

```python
from langgraph.graph import END, StateGraph, START
from langgraph.runtime import Runtime
from typing_extensions import TypedDict

# 1. 指定配置模式
class ContextSchema(TypedDict):
    my_runtime_value: str

# 2. 定义访问配置的图
class State(TypedDict):
    my_state_value: str

def node(state: State, runtime: Runtime[ContextSchema]):
    if runtime.context["my_runtime_value"] == "a":
        return {"my_state_value": 1}
    elif runtime.context["my_runtime_value"] == "b":
        return {"my_state_value": 2}
    else:
        raise ValueError("Unknown values.")

builder = StateGraph(State, context_schema=ContextSchema)
builder.add_node(node)
builder.add_edge(START, "node")
builder.add_edge("node", END)

graph = builder.compile()

# 3. 在运行时传入配置
print(graph.invoke({}, context={"my_runtime_value": "a"}))
```

## 重试策略

为节点添加重试策略：

```python
from langgraph.types import RetryPolicy

builder.add_node(
    "node_name",
    node_function,
    retry_policy=RetryPolicy(),
)
```

## 节点缓存

配置节点缓存策略：

```python
from langgraph.types import CachePolicy
from langgraph.cache.memory import InMemoryCache

builder.add_node(
    "node_name",
    node_function,
    cache_policy=CachePolicy(ttl=120),
)

graph = builder.compile(cache=InMemoryCache())
```

## 创建步骤序列

### 基本序列

```python
from langgraph.graph import START, StateGraph

builder = StateGraph(State)

# 添加节点
builder.add_node(step_1)
builder.add_node(step_2)
builder.add_node(step_3)

# 添加边
builder.add_edge(START, "step_1")
builder.add_edge("step_1", "step_2")
builder.add_edge("step_2", "step_3")
```

### 使用内置简写

```python
builder = StateGraph(State).add_sequence([step_1, step_2, step_3])
builder.add_edge(START, "step_1")
```

## 创建分支

### 并行执行节点

```python
import operator
from typing import Annotated
from typing_extensions import TypedDict
from langgraph.graph import StateGraph, START, END

class State(TypedDict):
    aggregate: Annotated[list, operator.add]

def a(state: State):
    print(f'Adding "A" to {state["aggregate"]}')
    return {"aggregate": ["A"]}

def b(state: State):
    print(f'Adding "B" to {state["aggregate"]}')
    return {"aggregate": ["B"]}

def c(state: State):
    print(f'Adding "C" to {state["aggregate"]}')
    return {"aggregate": ["C"]}

def d(state: State):
    print(f'Adding "D" to {state["aggregate"]}')
    return {"aggregate": ["D"]}

builder = StateGraph(State)
builder.add_node(a)
builder.add_node(b)
builder.add_node(c)
builder.add_node(d)
builder.add_edge(START, "a")
builder.add_edge("a", "b")
builder.add_edge("a", "c")
builder.add_edge("b", "d")
builder.add_edge("c", "d")
builder.add_edge("d", END)
graph = builder.compile()
```

### 条件分支

```python
from typing import Literal

def conditional_edge(state: State) -> Literal["b", "c"]:
    return state["which"]

builder.add_conditional_edges("a", conditional_edge)
```

## Map-Reduce 和 Send API

使用 Send API 实现 map-reduce 模式：

```python
from langgraph.graph import StateGraph, START, END
from langgraph.types import Send
from typing_extensions import TypedDict, Annotated
import operator

class OverallState(TypedDict):
    topic: str
    subjects: list[str]
    jokes: Annotated[list[str], operator.add]
    best_selected_joke: str

def generate_topics(state: OverallState):
    return {"subjects": ["lions", "elephants", "penguins"]}

def generate_joke(state: OverallState):
    joke_map = {
        "lions": "Why don't lions like fast food? Because they can't catch it!",
        "elephants": "Why don't elephants use computers? They're afraid of the mouse!",
        "penguins": "Why don't penguins like talking to strangers at parties? Because they find it hard to break the ice."
    }
    return {"jokes": [joke_map[state["subject"]]]}

def continue_to_jokes(state: OverallState):
    return [Send("generate_joke", {"subject": s}) for s in state["subjects"]]

builder = StateGraph(OverallState)
builder.add_node("generate_topics", generate_topics)
builder.add_node("generate_joke", generate_joke)
builder.add_edge(START, "generate_topics")
builder.add_conditional_edges("generate_topics", continue_to_jokes, ["generate_joke"])
```

## 创建和控制循环

### 基本循环

```python
import operator
from typing import Annotated, Literal
from typing_extensions import TypedDict
from langgraph.graph import StateGraph, START, END

class State(TypedDict):
    aggregate: Annotated[list, operator.add]

def a(state: State):
    print(f'Node A sees {state["aggregate"]}')
    return {"aggregate": ["A"]}

def b(state: State):
    print(f'Node B sees {state["aggregate"]}')
    return {"aggregate": ["B"]}

# 定义节点
builder = StateGraph(State)
builder.add_node(a)
builder.add_node(b)

# 定义边
def route(state: State) -> Literal["b", END]:
    if len(state["aggregate"]) < 7:
        return "b"
    else:
        return END

builder.add_edge(START, "a")
builder.add_conditional_edges("a", route)
builder.add_edge("b", "a")
graph = builder.compile()
```

### 设置递归限制

```python
from langgraph.errors import GraphRecursionError

try:
    graph.invoke({"aggregate": []}, {"recursion_limit": 4})
except GraphRecursionError:
    print("Recursion Error")
```

## 异步支持

将同步实现转换为异步实现：

```python
from langchain.chat_models import init_chat_model
from langgraph.graph import MessagesState, StateGraph

async def node(state: MessagesState):
    new_message = await llm.ainvoke(state["messages"])
    return {"messages": [new_message]}

builder = StateGraph(MessagesState).add_node(node).set_entry_point("node")
graph = builder.compile()

input_message = {"role": "user", "content": "Hello"}
result = await graph.ainvoke({"messages": [input_message]})
```

## 使用 Command 结合控制流和状态更新

使用 `Command` 对象在同一节点中执行状态更新和决定下一个节点：

```python
import random
from typing_extensions import TypedDict, Literal
from langgraph.graph import StateGraph, START
from langgraph.types import Command

class State(TypedDict):
    foo: str

def node_a(state: State) -> Command[Literal["node_b", "node_c"]]:
    print("Called A")
    value = random.choice(["b", "c"])
    if value == "b":
        goto = "node_b"
    else:
        goto = "node_c"

    return Command(
        update={"foo": value},
        goto=goto,
    )

def node_b(state: State):
    print("Called B")
    return {"foo": state["foo"] + "b"}

def node_c(state: State):
    print("Called C")
    return {"foo": state["foo"] + "c"}

builder = StateGraph(State)
builder.add_edge(START, "node_a")
builder.add_node(node_a)
builder.add_node(node_b)
builder.add_node(node_c)

graph = builder.compile()
```

## 可视化图

### Mermaid 语法

```python
print(app.get_graph().draw_mermaid())
```

### PNG 图像

```python
from IPython.display import Image, display

display(Image(app.get_graph().draw_mermaid_png()))
```

本教程涵盖了 LangGraph Graph API 的核心概念，包括状态管理、序列、分支、循环、异步支持和可视化。这些基础将帮助您构建复杂的 AI 应用程序工作流。
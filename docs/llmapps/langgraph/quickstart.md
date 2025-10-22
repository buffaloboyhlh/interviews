# LangGraph 快速入门教程：构建计算器智能体

本教程将教你使用 LangGraph 的两种不同 API 来构建一个计算器智能体：

- **Graph API**：通过定义节点和边来构建智能体图
- **Functional API**：通过单个函数定义智能体逻辑

> **前置要求**：你需要设置 [Claude (Anthropic)](https://www.anthropic.com/) 账户并获取 API 密钥，然后在终端中设置 `ANTHROPIC_API_KEY` 环境变量。

## 方法一：使用 Graph API

Graph API 适合喜欢通过可视化节点和边来构建智能体的开发者。

### 步骤 1：定义工具和模型

首先，我们定义 Claude Sonnet 4.5 模型和数学计算工具：

```python
from langchain.tools import tool
from langchain.chat_models import init_chat_model

# 初始化模型
model = init_chat_model(
    "anthropic:claude-sonnet-4-5",
    temperature=0
)

# 定义数学工具
@tool
def multiply(a: int, b: int) -> int:
    """Multiply `a` and `b`."""
    return a * b

@tool
def add(a: int, b: int) -> int:
    """Adds `a` and `b`."""
    return a + b

@tool
def divide(a: int, b: int) -> float:
    """Divide `a` and `b`."""
    return a / b

# 将工具绑定到模型
tools = [add, multiply, divide]
tools_by_name = {tool.name: tool for tool in tools}
model_with_tools = model.bind_tools(tools)
```

### 步骤 2：定义状态

状态用于存储消息和追踪 LLM 调用次数：

```python
from langchain.messages import AnyMessage
from typing_extensions import TypedDict, Annotated
import operator

class MessagesState(TypedDict):
    messages: Annotated[list[AnyMessage], operator.add]  # 自动追加新消息
    llm_calls: int
```

> **提示**：`Annotated` 类型配合 `operator.add` 确保新消息会追加到现有列表中，而不是替换它。

### 步骤 3：定义模型节点

这个节点负责调用 LLM 并决定是否调用工具：

```python
from langchain.messages import SystemMessage

def llm_call(state: dict):
    """LLM 决定是否调用工具"""
    return {
        "messages": [
            model_with_tools.invoke(
                [
                    SystemMessage(
                        content="You are a helpful assistant tasked with performing arithmetic on a set of inputs."
                    )
                ] + state["messages"]
            )
        ],
        "llm_calls": state.get('llm_calls', 0) + 1  # 追踪调用次数
    }
```

### 步骤 4：定义工具节点

这个节点负责执行工具调用并返回结果：

```python
from langchain.messages import ToolMessage

def tool_node(state: dict):
    """执行工具调用"""
    result = []
    for tool_call in state["messages"][-1].tool_calls:
        tool = tools_by_name[tool_call["name"]]
        observation = tool.invoke(tool_call["args"])
        result.append(ToolMessage(content=observation, tool_call_id=tool_call["id"]))
    return {"messages": result}
```

### 步骤 5：定义结束逻辑

条件边函数决定是继续调用工具还是结束对话：

```python
from typing import Literal
from langgraph.graph import StateGraph, START, END

def should_continue(state: MessagesState) -> Literal["tool_node", END]:
    """根据 LLM 是否调用工具来决定继续还是停止"""
    messages = state["messages"]
    last_message = messages[-1]

    # 如果 LLM 调用了工具，则执行工具节点
    if last_message.tool_calls:
        return "tool_node"
    
    # 否则结束对话（回复用户）
    return END
```

### 步骤 6：构建和编译智能体

使用 `StateGraph` 类构建工作流并编译：

```python
# 构建工作流
agent_builder = StateGraph(MessagesState)

# 添加节点
agent_builder.add_node("llm_call", llm_call)
agent_builder.add_node("tool_node", tool_node)

# 添加边连接节点
agent_builder.add_edge(START, "llm_call")
agent_builder.add_conditional_edges(
    "llm_call",
    should_continue,
    ["tool_node", END]
)
agent_builder.add_edge("tool_node", "llm_call")

# 编译智能体
agent = agent_builder.compile()

# 可视化智能体图
from IPython.display import Image, display
display(Image(agent.get_graph(xray=True).draw_mermaid_png()))

# 测试智能体
from langchain.messages import HumanMessage
messages = [HumanMessage(content="Add 3 and 4.")]
result = agent.invoke({"messages": messages})
for m in result["messages"]:
    m.pretty_print()
```

**恭喜！** 你已经使用 Graph API 成功构建了第一个智能体！

## 方法二：使用 Functional API

Functional API 适合喜欢在单个函数中定义逻辑的开发者。

### 步骤 1：定义工具和模型

工具和模型定义与 Graph API 相同：

```python
from langchain.tools import tool
from langchain.chat_models import init_chat_model

model = init_chat_model(
    "anthropic:claude-sonnet-4-5",
    temperature=0
)

# 定义相同的数学工具
@tool
def multiply(a: int, b: int) -> int:
    """Multiply `a` and `b`."""
    return a * b

@tool
def add(a: int, b: int) -> int:
    """Adds `a` and `b`."""
    return a + b

@tool
def divide(a: int, b: int) -> float:
    """Divide `a` and `b`."""
    return a / b

# 绑定工具到模型
tools = [add, multiply, divide]
tools_by_name = {tool.name: tool for tool in tools}
model_with_tools = model.bind_tools(tools)

from langgraph.graph import add_messages
from langchain.messages import SystemMessage, HumanMessage, ToolCall
from langchain_core.messages import BaseMessage
from langgraph.func import entrypoint, task
```

### 步骤 2：定义模型节点

使用 `@task` 装饰器标记可执行任务：

```python
@task
def call_llm(messages: list[BaseMessage]):
    """LLM 决定是否调用工具"""
    return model_with_tools.invoke(
        [
            SystemMessage(
                content="You are a helpful assistant tasked with performing arithmetic on a set of inputs."
            )
        ] + messages
    )
```

> **提示**：`@task` 装饰器标记的函数可以作为智能体的一部分执行，可以同步或异步调用。

### 步骤 3：定义工具节点

```python
@task
def call_tool(tool_call: ToolCall):
    """执行工具调用"""
    tool = tools_by_name[tool_call["name"]]
    return tool.invoke(tool_call)
```

### 步骤 4：定义智能体

使用 `@entrypoint` 函数构建智能体：

```python
@entrypoint()
def agent(messages: list[BaseMessage]):
    model_response = call_llm(messages).result()

    while True:
        if not model_response.tool_calls:
            break

        # 执行工具
        tool_result_futures = [
            call_tool(tool_call) for tool_call in model_response.tool_calls
        ]
        tool_results = [fut.result() for fut in tool_result_futures]
        messages = add_messages(messages, [model_response, *tool_results])
        model_response = call_llm(messages).result()

    messages = add_messages(messages, model_response)
    return messages

# 测试智能体
messages = [HumanMessage(content="Add 3 and 4.")]
for chunk in agent.stream(messages, stream_mode="updates"):
    print(chunk)
    print("\n")
```

> **注意**：在 Functional API 中，你不需要显式定义节点和边，而是在单个函数中使用标准的控制流逻辑（循环、条件语句）。

**恭喜！** 你已经使用 Functional API 成功构建了第一个智能体！

## 两种方法的比较

| 特性 | Graph API | Functional API |
|------|-----------|----------------|
| **可视化** | ✅ 支持图可视化 | ❌ 无可视化 |
| **控制精度** | ✅ 细粒度控制 | ⚠️ 中等控制 |
| **学习曲线** | 较陡峭 | 较平缓 |
| **代码复杂度** | 较高 | 较低 |
| **适用场景** | 复杂工作流 | 简单到中等复杂度 |

## 下一步学习建议

- 深入了解 [Graph API 概述](/oss/python/langgraph/graph-api)
- 学习 [Functional API 概述](/oss/python/langgraph/functional-api)
- 探索更复杂的智能体模式和应用场景

两种 API 都能构建强大的智能体，选择哪种取决于你的具体需求和个人偏好。Graph API 提供更多控制和可视化能力，而 Functional API 则更简洁易用。
# LangGraph 完全入门教程

## 什么是 LangGraph？

LangGraph 是一个低级别的编排框架和运行时，专门用于构建、管理和部署长时间运行的有状态智能体。它被 Klarna、Replit、Elastic 等领先公司广泛使用，专注于智能体的**编排**核心能力。

> **重要通知**: LangGraph v1.0 已正式发布！如需查看完整变更列表和升级指南，请参阅[发布说明](/oss/python/releases/langgraph-v1)和[迁移指南](/oss/python/migrate/langgraph-v1)。

## 核心优势

### 🛡️ 持久执行 (Durable Execution)
构建能够从故障中恢复并长期运行的智能体，支持从中断处继续执行。

### 👥 人工干预 (Human-in-the-loop)
在任何时间点检查和修改智能体状态，实现人工监督。

### 🧠 全面记忆系统 (Comprehensive Memory)
创建具有短期工作记忆和长期会话记忆的有状态智能体。

### 🔍 LangSmith 调试
通过可视化工具深度洞察复杂智能体行为，追踪执行路径和状态转换。

### 🚀 生产就绪部署
为有状态、长时间运行的工作流提供可扩展的基础设施。

## 安装指南

### 使用 pip 安装
```bash
pip install -U langgraph
```

### 使用 uv 安装
```bash
uv add langgraph
```

## 快速开始：Hello World 示例

让我们创建一个简单的 LangGraph 应用来理解基本概念：

```python
from langgraph.graph import StateGraph, MessagesState, START, END

# 定义模拟的 LLM 节点
def mock_llm(state: MessagesState):
    return {"messages": [{"role": "ai", "content": "hello world"}]}

# 创建状态图
graph = StateGraph(MessagesState)

# 添加节点
graph.add_node("mock_llm", mock_llm)

# 建立连接关系
graph.add_edge(START, "mock_llm")  # 从开始到 LLM 节点
graph.add_edge("mock_llm", END)   # 从 LLM 节点到结束

# 编译图
graph = graph.compile()

# 执行图
result = graph.invoke({"messages": [{"role": "user", "content": "hi!"}]})
print(result)
```

## 核心概念详解

### 状态图 (StateGraph)
LangGraph 的核心是状态图，它定义了智能体的执行流程：

```python
# 创建状态图，指定状态类型
graph = StateGraph(MessagesState)
```

### 节点 (Nodes)
节点是图的基本构建块，每个节点执行特定的任务：

```python
def my_node(state: MessagesState):
    # 处理状态并返回更新
    new_message = {"role": "ai", "content": "处理完成"}
    return {"messages": state["messages"] + [new_message]}
```

### 边 (Edges)
边定义了节点之间的执行路径：

```python
graph.add_edge(START, "first_node")      # 从开始到第一个节点
graph.add_edge("first_node", "second_node")  # 节点之间的连接
graph.add_edge("second_node", END)       # 从节点到结束
```

## 进阶示例：条件工作流

创建更复杂的有条件执行的工作流：

```python
from langgraph.graph import StateGraph, MessagesState, START, END
from typing import Literal

def router(state: MessagesState) -> Literal["end", "continue"]:
    last_message = state["messages"][-1]["content"]
    if "结束" in last_message or "stop" in last_message.lower():
        return "end"
    else:
        return "continue"

def process_message(state: MessagesState):
    last_message = state["messages"][-1]["content"]
    response = f"已处理您的消息: {last_message}"
    return {"messages": [{"role": "ai", "content": response}]}

def final_response(state: MessagesState):
    return {"messages": [{"role": "ai", "content": "对话结束，感谢使用！"}]}

# 构建图
graph = StateGraph(MessagesState)
graph.add_node("router", router)
graph.add_node("process", process_message)
graph.add_node("final", final_response)

# 设置条件边
graph.add_conditional_edges(
    "router",
    router,
    {
        "continue": "process",
        "end": "final"
    }
)

graph.add_edge("process", "router")  # 循环回到路由节点
graph.add_edge("final", END)

graph = graph.compile()
```

## LangGraph 生态系统集成

### 与 LangSmith 集成
获得完整的可观测性：

```python
# 设置环境变量
import os
os.environ["LANGSMITH_API_KEY"] = "your-api-key"
os.environ["LANGSMITH_PROJECT"] = "your-project-name"

# 现在所有的调用都会被追踪
result = graph.invoke({"messages": [{"role": "user", "content": "hi"}]})
```

### 与 LangChain 组件集成
虽然 LangGraph 可以独立使用，但与 LangChain 集成可以提供更丰富的功能：

```python
from langchain_community.chat_models import ChatOpenAI
from langchain_core.messages import HumanMessage

def llm_node(state: MessagesState):
    llm = ChatOpenAI(model="gpt-4")
    response = llm.invoke(state["messages"])
    return {"messages": [response]}
```

## 生产环境最佳实践

### 1. 错误处理
```python
def robust_node(state: MessagesState):
    try:
        # 你的业务逻辑
        return {"messages": [{"role": "ai", "content": "成功"}]}
    except Exception as e:
        return {"messages": [{"role": "ai", "content": f"处理出错: {str(e)}"}]}
```

### 2. 状态持久化
```python
# 保存检查点
checkpoint = graph.get_state()
# 恢复执行
graph.invoke({"messages": [...]}, config={"configurable": {"thread_id": "123"}})
```

### 3. 流式输出
```python
for chunk in graph.stream({"messages": [...]}):
    print("收到更新:", chunk)
```

## 故障排除

### 常见问题

1. **状态类型不匹配**
   - 确保所有节点返回的状态结构与图定义的类型一致

2. **循环依赖**
   - 使用条件边避免无限循环

3. **内存管理**
   - 对于长时间运行的工作流，定期清理不需要的状态

### 获取帮助

- 遇到问题？[提交 issue](https://github.com/langchain-ai/docs/issues/new?template=02-langgraph.yml&labels=langgraph,python)
- 查看 [v0.x 文档](https://langchain-ai.github.io/langgraph/)（归档版本）
- 通过 [MCP](/use-these-docs) 连接这些文档到 Claude、VSCode 等工具获取实时答案

## 下一步

- 深入学习[持久执行](/oss/python/langgraph/durable-execution)
- 了解[人工干预](/oss/python/langgraph/interrupts)功能
- 探索[内存管理](/oss/python/concepts/memory)概念
- 查看[生产部署](/langsmith/deployments)指南

LangGraph 为你提供了构建复杂、有状态智能体应用所需的所有底层能力，让你能够专注于业务逻辑而不是基础设施。
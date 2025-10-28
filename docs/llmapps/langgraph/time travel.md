# LangGraph 时间旅行使用教程

## 什么是时间旅行？

在基于模型决策的非确定性系统（如LLM驱动的智能体）中，时间旅行功能让您能够详细检查决策过程：

- 🔍 **理解推理逻辑**：分析导致成功结果的各个步骤
- 🐛 **调试错误**：识别错误发生的位置和原因  
- 🔄 **探索替代方案**：测试不同路径以发现更好的解决方案

## 时间旅行核心概念

LangGraph的时间旅行功能允许您从之前的检查点恢复执行——可以重放相同状态，也可以修改状态来探索替代方案。无论哪种情况，恢复过去的执行都会在历史中创建一个新的分支。

## 使用步骤

### 1. 运行图

首先使用初始输入运行图：

```python
config = {
    "configurable": {
        "thread_id": uuid.uuid4(),
    }
}
state = graph.invoke({}, config)
```

### 2. 识别检查点

获取执行历史并定位所需的检查点：

```python
# 状态按时间倒序返回
states = list(graph.get_state_history(config))

for state in states:
    print(f"下一步节点: {state.next}")
    print(f"检查点ID: {state.config['configurable']['checkpoint_id']}")
    print()
```

或者，在目标节点前设置[中断]，然后在中断处找到最近的检查点。

### 3. 更新状态（可选）

在检查点修改图状态：

```python
new_config = graph.update_state(
    selected_state.config, 
    values={"topic": "新的主题"}
)
```

`update_state`会创建一个与同一线程关联但具有新检查点ID的新检查点。

### 4. 从检查点恢复执行

使用适当的`thread_id`和`checkpoint_id`恢复执行：

```python
graph.invoke(None, new_config)
```

## 完整工作流程示例

### 环境设置

```python
%%capture --no-stderr
pip install --quiet -U langgraph langchain_anthropic

import getpass
import os

def _set_env(var: str):
    if not os.environ.get(var):
        os.environ[var] = getpass.getpass(f"{var}: ")

_set_env("ANTHROPIC_API_KEY")
```

### 构建工作流

```python
import uuid
from typing_extensions import TypedDict, NotRequired
from langgraph.graph import StateGraph, START, END
from langchain.chat_models import init_chat_model
from langgraph.checkpoint.memory import InMemorySaver

# 定义状态结构
class State(TypedDict):
    topic: NotRequired[str]
    joke: NotRequired[str]

# 初始化模型
model = init_chat_model("anthropic:claude-sonnet-4-5", temperature=0)

def generate_topic(state: State):
    """生成笑话主题的LLM调用"""
    msg = model.invoke("给我一个有趣的笑话主题")
    return {"topic": msg.content}

def write_joke(state: State):
    """基于主题写笑话的LLM调用"""
    msg = model.invoke(f"写一个关于{state['topic']}的短笑话")
    return {"joke": msg.content}

# 构建工作流
workflow = StateGraph(State)
workflow.add_node("generate_topic", generate_topic)
workflow.add_node("write_joke", write_joke)
workflow.add_edge(START, "generate_topic")
workflow.add_edge("generate_topic", "write_joke")
workflow.add_edge("write_joke", END)

# 编译图
checkpointer = InMemorySaver()
graph = workflow.compile(checkpointer=checkpointer)
```

### 执行时间旅行

```python
# 1. 首次运行
config = {"configurable": {"thread_id": uuid.uuid4()}}
state = graph.invoke({}, config)
print("原始结果:", state["joke"])

# 2. 识别检查点
states = list(graph.get_state_history(config))
selected_state = states[1]  # 选择write_joke之前的检查点

# 3. 修改状态
new_config = graph.update_state(
    selected_state.config, 
    values={"topic": "程序员的生活"}
)

# 4. 恢复执行
new_state = graph.invoke(None, new_config)
print("修改后的结果:", new_state["joke"])
```

## 使用场景

### 调试分析
当智能体产生意外结果时，使用时间旅行回溯到关键决策点，分析推理过程。

### 方案对比
从同一检查点出发，尝试不同的状态修改，比较多种解决方案的效果。

### 性能优化
识别执行瓶颈，通过修改状态测试更高效的执行路径。

## 注意事项

- 时间旅行会创建新的执行分支，不影响原始执行历史
- 确保检查点ID正确，避免从错误的状态恢复
- 状态修改应符合图的预期输入格式
- 内存检查点适用于开发环境，生产环境建议使用持久化存储

通过时间旅行功能，您可以更深入地理解和优化基于LLM的智能体行为，提高系统的可靠性和性能。
# LangGraph 工作流与智能体教程

本教程将介绍如何使用 LangGraph 构建工作流和智能代理系统。

## 概述

### 工作流 vs 智能体

- **工作流**：具有预定义的代码路径，按特定顺序执行
- **智能体**：动态的，能够定义自己的流程和工具使用方式

LangGraph 在构建智能体和工作流时提供了多种优势，包括持久化、流式处理和调试支持。

![agent_workflow.avif](../../imgs/llm/agent_workflow.avif)

## 环境设置

### 1. 安装依赖

```bash
pip install langchain_core langchain-anthropic langgraph
```

### 2. 初始化 LLM

```python
import os
import getpass
from langchain_anthropic import ChatAnthropic

def _set_env(var: str):
    if not os.environ.get(var):
        os.environ[var] = getpass.getpass(f"{var}: ")

_set_env("ANTHROPIC_API_KEY")

llm = ChatAnthropic(model="claude-sonnet-4-5")
```

## LLM 增强功能

工作流和智能体系统基于 LLM 及其各种增强功能：

- **工具调用**：让 LLM 能够使用外部工具
- **结构化输出**：确保输出符合预定义格式
- **短期记忆**：为 LLM 提供上下文记忆

![augmented_llm.avif](../../imgs/llm/augmented_llm.avif)

### 结构化输出示例

```python
from pydantic import BaseModel, Field

class SearchQuery(BaseModel):
    search_query: str = Field(None, description="优化的网络搜索查询")
    justification: str = Field(None, description="为什么此查询与用户请求相关")

# 增强 LLM 以支持结构化输出
structured_llm = llm.with_structured_output(SearchQuery)

# 调用增强后的 LLM
output = structured_llm.invoke("钙 CT 评分与高胆固醇有什么关系？")
```

### 工具调用示例

```python
# 定义工具
def multiply(a: int, b: int) -> int:
    return a * b

# 为 LLM 绑定工具
llm_with_tools = llm.bind_tools([multiply])

# 调用 LLM
msg = llm_with_tools.invoke("2 乘以 3 等于多少？")

# 获取工具调用
print(msg.tool_calls)
```

## 工作流模式

### 1. 提示链 (Prompt Chaining)

提示链是指每个 LLM 调用处理前一个调用的输出，适用于可以分解为较小、可验证步骤的任务。

**应用场景**：
- 将文档翻译成不同语言
- 验证生成内容的一致性

![prompt_chain.avif](../../imgs/llm/prompt_chain.avif)

#### Graph API 实现

```python
from typing_extensions import TypedDict
from langgraph.graph import StateGraph, START, END

# 图状态定义
class State(TypedDict):
    topic: str
    joke: str
    improved_joke: str
    final_joke: str

# 节点函数
def generate_joke(state: State):
    """生成初始笑话"""
    msg = llm.invoke(f"写一个关于 {state['topic']} 的短笑话")
    return {"joke": msg.content}

def check_punchline(state: State):
    """检查笑话是否有笑点"""
    if "?" in state["joke"] or "!" in state["joke"]:
        return "Pass"
    return "Fail"

def improve_joke(state: State):
    """改进笑话"""
    msg = llm.invoke(f"通过添加文字游戏让这个笑话更有趣：{state['joke']}")
    return {"improved_joke": msg.content}

def polish_joke(state: State):
    """最终润色"""
    msg = llm.invoke(f"为这个笑话添加一个意想不到的转折：{state['improved_joke']}")
    return {"final_joke": msg.content}

# 构建工作流
workflow = StateGraph(State)
workflow.add_node("generate_joke", generate_joke)
workflow.add_node("improve_joke", improve_joke)
workflow.add_node("polish_joke", polish_joke)

# 添加边连接节点
workflow.add_edge(START, "generate_joke")
workflow.add_conditional_edges(
    "generate_joke", check_punchline, {"Fail": "improve_joke", "Pass": END}
)
workflow.add_edge("improve_joke", "polish_joke")
workflow.add_edge("polish_joke", END)

# 编译并执行
chain = workflow.compile()
state = chain.invoke({"topic": "猫"})
```

### 2. 并行化 (Parallelization)

并行化让 LLM 同时处理任务，可以提高速度或增加输出置信度。

**应用场景**：
- 同时处理多个独立子任务
- 多次运行同一任务以检查不同输出

![parallelization.avif](../../imgs/llm/parallelization.avif)

#### Graph API 实现

```python
class State(TypedDict):
    topic: str
    joke: str
    story: str
    poem: str
    combined_output: str

def call_llm_1(state: State):
    msg = llm.invoke(f"写一个关于 {state['topic']} 的笑话")
    return {"joke": msg.content}

def call_llm_2(state: State):
    msg = llm.invoke(f"写一个关于 {state['topic']} 的故事")
    return {"story": msg.content}

def call_llm_3(state: State):
    msg = llm.invoke(f"写一首关于 {state['topic']} 的诗")
    return {"poem": msg.content}

def aggregator(state: State):
    combined = f"这是关于 {state['topic']} 的故事、笑话和诗歌！\n\n"
    combined += f"故事：\n{state['story']}\n\n"
    combined += f"笑话：\n{state['joke']}\n\n"
    combined += f"诗歌：\n{state['poem']}"
    return {"combined_output": combined}

# 构建并行工作流
parallel_builder = StateGraph(State)
parallel_builder.add_node("call_llm_1", call_llm_1)
parallel_builder.add_node("call_llm_2", call_llm_2)
parallel_builder.add_node("call_llm_3", call_llm_3)
parallel_builder.add_node("aggregator", aggregator)

# 设置并行执行路径
parallel_builder.add_edge(START, "call_llm_1")
parallel_builder.add_edge(START, "call_llm_2")
parallel_builder.add_edge(START, "call_llm_3")
parallel_builder.add_edge("call_llm_1", "aggregator")
parallel_builder.add_edge("call_llm_2", "aggregator")
parallel_builder.add_edge("call_llm_3", "aggregator")
parallel_builder.add_edge("aggregator", END)

parallel_workflow = parallel_builder.compile()
```

### 3. 路由 (Routing)

路由工作流处理输入并将其定向到特定任务，适用于复杂任务的专门流程。

![routing.avif](../../imgs/llm/routing.avif)

#### Graph API 实现

```python
from typing_extensions import Literal
from pydantic import BaseModel, Field
from langchain.messages import HumanMessage, SystemMessage

class Route(BaseModel):
    step: Literal["poem", "story", "joke"] = Field(
        None, description="路由过程中的下一步"
    )

# 增强 LLM 以支持路由决策
router = llm.with_structured_output(Route)

class State(TypedDict):
    input: str
    decision: str
    output: str

def llm_call_router(state: State):
    """路由输入到适当的节点"""
    decision = router.invoke([
        SystemMessage(content="根据用户请求将输入路由到故事、笑话或诗歌。"),
        HumanMessage(content=state["input"])
    ])
    return {"decision": decision.step}

def route_decision(state: State):
    if state["decision"] == "story":
        return "llm_call_1"
    elif state["decision"] == "joke":
        return "llm_call_2"
    elif state["decision"] == "poem":
        return "llm_call_3"
```

### 4. 编排器-工作者 (Orchestrator-Worker)

在这种配置中，编排器分解任务、委派子任务给工作者，并将工作者输出合成为最终结果。

![worker.avif](../../imgs/llm/worker.avif)

#### 使用 Send API 实现

```python
from langgraph.types import Send
from typing import Annotated, List
import operator

class Section(BaseModel):
    name: str = Field(description="报告部分的名称")
    description: str = Field(description="本节要涵盖的主要主题和概念的简要概述")

class Sections(BaseModel):
    sections: List[Section] = Field(description="报告的各部分")

planner = llm.with_structured_output(Sections)

class State(TypedDict):
    topic: str
    sections: list[Section]
    completed_sections: Annotated[list, operator.add]
    final_report: str

class WorkerState(TypedDict):
    section: Section
    completed_sections: Annotated[list, operator.add]

def orchestrator(state: State):
    """编排器生成报告计划"""
    report_sections = planner.invoke([
        SystemMessage(content="生成报告计划。"),
        HumanMessage(content=f"这是报告主题：{state['topic']}")
    ])
    return {"sections": report_sections.sections}

def llm_call(state: WorkerState):
    """工作者编写报告部分"""
    section = llm.invoke([
        SystemMessage(content="按照提供的名称和描述编写报告部分。"),
        HumanMessage(content=f"这是部分名称：{state['section'].name} 和描述：{state['section'].description}")
    ])
    return {"completed_sections": [section.content]}

def assign_workers(state: State):
    """为计划中的每个部分分配工作者"""
    return [Send("llm_call", {"section": s}) for s in state["sections"]]
```

### 5. 评估器-优化器 (Evaluator-Optimizer)

在这种工作流中，一个 LLM 调用创建响应，另一个评估该响应。如果需要改进，提供反馈并重新创建响应。

![evaluator_optimizer.avif](../../imgs/llm/evaluator_optimizer.avif)


#### Graph API 实现

```python
class Feedback(BaseModel):
    grade: Literal["funny", "not funny"] = Field(description="决定笑话是否有趣")
    feedback: str = Field(description="如果笑话不好笑，提供改进反馈")

evaluator = llm.with_structured_output(Feedback)

class State(TypedDict):
    joke: str
    topic: str
    feedback: str
    funny_or_not: str

def llm_call_generator(state: State):
    """LLM 生成笑话"""
    if state.get("feedback"):
        msg = llm.invoke(f"写一个关于 {state['topic']} 的笑话，但要考虑反馈：{state['feedback']}")
    else:
        msg = llm.invoke(f"写一个关于 {state['topic']} 的笑话")
    return {"joke": msg.content}

def llm_call_evaluator(state: State):
    """LLM 评估笑话"""
    grade = evaluator.invoke(f"评价这个笑话：{state['joke']}")
    return {"funny_or_not": grade.grade, "feedback": grade.feedback}

def route_joke(state: State):
    """根据评估器反馈路由回笑话生成器或结束"""
    if state["funny_or_not"] == "funny":
        return "Accepted"
    elif state["funny_or_not"] == "not funny":
        return "Rejected + Feedback"
```

## 智能体 (Agents)

智能体通常是使用工具执行操作的 LLM，在连续反馈循环中运行，用于问题和解决方案不可预测的情况。

![agent.avif](../../imgs/llm/agent.avif)

### 工具定义

```python
from langchain.tools import tool

@tool
def multiply(a: int, b: int) -> int:
    """将 `a` 和 `b` 相乘"""
    return a * b

@tool
def add(a: int, b: int) -> int:
    """将 `a` 和 `b` 相加"""
    return a + b

@tool
def divide(a: int, b: int) -> float:
    """将 `a` 除以 `b`"""
    return a / b

# 为 LLM 绑定工具
tools = [add, multiply, divide]
tools_by_name = {tool.name: tool for tool in tools}
llm_with_tools = llm.bind_tools(tools)
```

### Graph API 实现智能体

```python
from langgraph.graph import MessagesState
from langchain.messages import SystemMessage, HumanMessage, ToolMessage

def llm_call(state: MessagesState):
    """LLM 决定是否调用工具"""
    return {
        "messages": [
            llm_with_tools.invoke([
                SystemMessage(content="你是一个有帮助的助手，负责对一组输入执行算术运算。")
            ] + state["messages"])
        ]
    }

def tool_node(state: dict):
    """执行工具调用"""
    result = []
    for tool_call in state["messages"][-1].tool_calls:
        tool = tools_by_name[tool_call["name"]]
        observation = tool.invoke(tool_call["args"])
        result.append(ToolMessage(content=observation, tool_call_id=tool_call["id"]))
    return {"messages": result}

def should_continue(state: MessagesState) -> Literal["tool_node", END]:
    """根据 LLM 是否进行工具调用来决定是否继续循环"""
    messages = state["messages"]
    last_message = messages[-1]
    
    if last_message.tool_calls:
        return "tool_node"
    return END

# 构建智能体
agent_builder = StateGraph(MessagesState)
agent_builder.add_node("llm_call", llm_call)
agent_builder.add_node("tool_node", tool_node)

agent_builder.add_edge(START, "llm_call")
agent_builder.add_conditional_edges("llm_call", should_continue, ["tool_node", END])
agent_builder.add_edge("tool_node", "llm_call")

agent = agent_builder.compile()
```

## 总结

本教程介绍了 LangGraph 中的主要工作流和智能体模式：

1. **提示链**：顺序执行，适用于可分解任务
2. **并行化**：同时执行，提高效率
3. **路由**：根据输入定向到特定任务
4. **编排器-工作者**：动态任务分解和分配
5. **评估器-优化器**：迭代改进输出质量
6. **智能体**：自主决策和工具使用

每种模式都有其适用场景，选择合适的模式取决于具体任务的需求和复杂性。LangGraph 提供了灵活的 API 来构建这些模式，无论是使用 Graph API 还是 Functional API。
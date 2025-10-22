# LangGraph 工作流与智能体完全指南

本教程将深入探讨 LangGraph 中的工作流和智能体模式，帮助你理解如何构建复杂的 AI 应用系统。

## 工作流 vs 智能体：核心区别

**工作流**具有预定义的代码路径，按特定顺序执行
**智能体**是动态的，定义自己的流程和工具使用

![工作流与智能体对比](https://mintcdn.com/langchain-5e9cc07a/-_xGPoyjhyiDWTPJ/oss/images/agent_workflow.png)

LangGraph 为构建工作流和智能体提供了多项优势，包括[持久性](/oss/python/langgraph/persistence)、[流式传输](/oss/python/langgraph/streaming)、调试支持以及[部署](/oss/python/langgraph/deploy)能力。

## 环境设置

### 安装依赖
```bash
pip install langchain_core langchain-anthropic langgraph
```

### 初始化 LLM
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

工作流和智能体系统基于 LLM 及其各种增强功能构建。[工具调用](/oss/python/langchain/tools)、[结构化输出](/oss/python/langchain/structured-output)和[短期记忆](/oss/python/langchain/short-term-memory)是定制 LLM 以满足需求的几种选择。

![LLM 增强功能](https://mintcdn.com/langchain-5e9cc07a/-_xGPoyjhyiDWTPJ/oss/images/augmented_llm.png)

### 结构化输出示例
```python
from pydantic import BaseModel, Field

class SearchQuery(BaseModel):
    search_query: str = Field(None, description="优化网络搜索的查询")
    justification: str = Field(
        None, description="此查询与用户请求相关的原因"
    )

# 使用结构化输出增强 LLM
structured_llm = llm.with_structured_output(SearchQuery)

# 调用增强的 LLM
output = structured_llm.invoke("钙 CT 评分与高胆固醇有什么关系？")
```

### 工具调用示例
```python
# 定义工具
def multiply(a: int, b: int) -> int:
    return a * b

# 使用工具增强 LLM
llm_with_tools = llm.bind_tools([multiply])

# 调用触发工具调用的 LLM
msg = llm_with_tools.invoke("2 乘以 3 等于多少？")

# 获取工具调用
msg.tool_calls
```

## 1. 提示链 (Prompt Chaining)

提示链是指每个 LLM 调用处理前一个调用的输出。通常用于执行可以分解为较小、可验证步骤的明确定义任务。

**适用场景：**
- 将文档翻译成不同语言
- 验证生成内容的一致性

![提示链](https://mintcdn.com/langchain-5e9cc07a/dL5Sn6Cmy9pwtY0V/oss/images/prompt_chain.png)

### Graph API 实现
```python
from typing_extensions import TypedDict
from langgraph.graph import StateGraph, START, END
from IPython.display import Image, display

# 图状态
class State(TypedDict):
    topic: str
    joke: str
    improved_joke: str
    final_joke: str

# 节点函数
def generate_joke(state: State):
    """第一个 LLM 调用生成初始笑话"""
    msg = llm.invoke(f"写一个关于 {state['topic']} 的短笑话")
    return {"joke": msg.content}

def check_punchline(state: State):
    """检查笑话是否有笑点"""
    if "?" in state["joke"] or "!" in state["joke"]:
        return "Pass"
    return "Fail"

def improve_joke(state: State):
    """第二个 LLM 调用改进笑话"""
    msg = llm.invoke(f"通过添加文字游戏使这个笑话更有趣: {state['joke']}")
    return {"improved_joke": msg.content}

def polish_joke(state: State):
    """第三个 LLM 调用进行最终润色"""
    msg = llm.invoke(f"为这个笑话添加一个意想不到的转折: {state['improved_joke']}")
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

# 编译
chain = workflow.compile()

# 显示工作流
display(Image(chain.get_graph().draw_mermaid_png()))

# 调用
state = chain.invoke({"topic": "猫"})
```

### Functional API 实现
```python
from langgraph.func import entrypoint, task

@task
def generate_joke(topic: str):
    """第一个 LLM 调用生成初始笑话"""
    msg = llm.invoke(f"写一个关于 {topic} 的短笑话")
    return msg.content

def check_punchline(joke: str):
    """检查笑话是否有笑点"""
    if "?" in joke or "!" in joke:
        return "Fail"
    return "Pass"

@task
def improve_joke(joke: str):
    """第二个 LLM 调用改进笑话"""
    msg = llm.invoke(f"通过添加文字游戏使这个笑话更有趣: {joke}")
    return msg.content

@task
def polish_joke(joke: str):
    """第三个 LLM 调用进行最终润色"""
    msg = llm.invoke(f"为这个笑话添加一个意想不到的转折: {joke}")
    return msg.content

@entrypoint()
def prompt_chaining_workflow(topic: str):
    original_joke = generate_joke(topic).result()
    if check_punchline(original_joke) == "Pass":
        return original_joke
    
    improved_joke = improve_joke(original_joke).result()
    return polish_joke(improved_joke).result()

# 调用
for step in prompt_chaining_workflow.stream("猫", stream_mode="updates"):
    print(step)
```

## 2. 并行化 (Parallelization)

通过并行化，LLM 同时处理任务。这可以通过同时运行多个独立子任务，或多次运行同一任务以检查不同输出来实现。

**适用场景：**
- 拆分子任务并行运行，提高速度
- 多次运行任务检查不同输出，增加置信度

![并行化](https://mintcdn.com/langchain-5e9cc07a/dL5Sn6Cmy9pwtY0V/oss/images/parallelization.png)

### Graph API 实现
```python
# 图状态
class State(TypedDict):
    topic: str
    joke: str
    story: str
    poem: str
    combined_output: str

# 节点函数
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
    combined = f"这是关于 {state['topic']} 的故事、笑话和诗！\n\n"
    combined += f"故事:\n{state['story']}\n\n"
    combined += f"笑话:\n{state['joke']}\n\n"
    combined += f"诗:\n{state['poem']}"
    return {"combined_output": combined}

# 构建工作流
parallel_builder = StateGraph(State)
parallel_builder.add_node("call_llm_1", call_llm_1)
parallel_builder.add_node("call_llm_2", call_llm_2)
parallel_builder.add_node("call_llm_3", call_llm_3)
parallel_builder.add_node("aggregator", aggregator)

# 添加边连接节点
parallel_builder.add_edge(START, "call_llm_1")
parallel_builder.add_edge(START, "call_llm_2")
parallel_builder.add_edge(START, "call_llm_3")
parallel_builder.add_edge("call_llm_1", "aggregator")
parallel_builder.add_edge("call_llm_2", "aggregator")
parallel_builder.add_edge("call_llm_3", "aggregator")
parallel_builder.add_edge("aggregator", END)

parallel_workflow = parallel_builder.compile()
```

## 3. 路由 (Routing)

路由工作流处理输入，然后将它们定向到特定于上下文的任务。这允许你为复杂任务定义专门的流程。

![路由](https://mintcdn.com/langchain-5e9cc07a/dL5Sn6Cmy9pwtY0V/oss/images/routing.png)

### Graph API 实现
```python
from typing_extensions import Literal
from pydantic import BaseModel, Field
from langchain.messages import HumanMessage, SystemMessage

# 用于路由逻辑的结构化输出模式
class Route(BaseModel):
    step: Literal["poem", "story", "joke"] = Field(
        None, description="路由过程中的下一步"
    )

# 使用结构化输出增强 LLM
router = llm.with_structured_output(Route)

# 状态
class State(TypedDict):
    input: str
    decision: str
    output: str

# 节点函数
def llm_call_1(state: State):
    result = llm.invoke(state["input"])
    return {"output": result.content}

def llm_call_2(state: State):
    result = llm.invoke(state["input"])
    return {"output": result.content}

def llm_call_3(state: State):
    result = llm.invoke(state["input"])
    return {"output": result.content}

def llm_call_router(state: State):
    decision = router.invoke([
        SystemMessage(content="根据用户请求将输入路由到故事、笑话或诗歌。"),
        HumanMessage(content=state["input"]),
    ])
    return {"decision": decision.step}

# 条件边函数路由到适当的节点
def route_decision(state: State):
    if state["decision"] == "story":
        return "llm_call_1"
    elif state["decision"] == "joke":
        return "llm_call_2"
    elif state["decision"] == "poem":
        return "llm_call_3"

# 构建工作流
router_builder = StateGraph(State)
router_builder.add_node("llm_call_1", llm_call_1)
router_builder.add_node("llm_call_2", llm_call_2)
router_builder.add_node("llm_call_3", llm_call_3)
router_builder.add_node("llm_call_router", llm_call_router)

router_builder.add_edge(START, "llm_call_router")
router_builder.add_conditional_edges(
    "llm_call_router",
    route_decision,
    {
        "llm_call_1": "llm_call_1",
        "llm_call_2": "llm_call_2",
        "llm_call_3": "llm_call_3",
    },
)
router_builder.add_edge("llm_call_1", END)
router_builder.add_edge("llm_call_2", END)
router_builder.add_edge("llm_call_3", END)

router_workflow = router_builder.compile()
```

## 4. 编排器-工作器 (Orchestrator-Worker)

在编排器-工作器配置中，编排器：
- 将任务分解为子任务
- 将子任务委托给工作器
- 将工作器输出合成为最终结果

![编排器-工作器](https://mintcdn.com/langchain-5e9cc07a/ybiAaBfoBvFquMDz/oss/images/worker.png)

### 使用 Send API 的高级实现
```python
from langgraph.types import Send
from typing import Annotated, List
import operator

# 图状态
class State(TypedDict):
    topic: str  # 报告主题
    sections: list  # 报告部分列表
    completed_sections: Annotated[list, operator.add]  # 所有工作器并行写入此键
    final_report: str  # 最终报告

# 工作器状态
class WorkerState(TypedDict):
    section: dict
    completed_sections: Annotated[list, operator.add]

# 节点函数
def orchestrator(state: State):
    report_sections = planner.invoke([
        SystemMessage(content="生成报告计划。"),
        HumanMessage(content=f"这是报告主题: {state['topic']}"),
    ])
    return {"sections": report_sections.sections}

def llm_call(state: WorkerState):
    section = llm.invoke([
        SystemMessage(content="按照提供的名称和描述编写报告部分。"),
        HumanMessage(content=f"这是部分名称: {state['section'].name} 和描述: {state['section'].description}")
    ])
    return {"completed_sections": [section.content]}

def synthesizer(state: State):
    completed_sections = state["completed_sections"]
    completed_report_sections = "\n\n---\n\n".join(completed_sections)
    return {"final_report": completed_report_sections}

# 条件边函数为计划中的每个部分创建工作器
def assign_workers(state: State):
    return [Send("llm_call", {"section": s}) for s in state["sections"]]

# 构建工作流
orchestrator_worker_builder = StateGraph(State)
orchestrator_worker_builder.add_node("orchestrator", orchestrator)
orchestrator_worker_builder.add_node("llm_call", llm_call)
orchestrator_worker_builder.add_node("synthesizer", synthesizer)

orchestrator_worker_builder.add_edge(START, "orchestrator")
orchestrator_worker_builder.add_conditional_edges("orchestrator", assign_workers, ["llm_call"])
orchestrator_worker_builder.add_edge("llm_call", "synthesizer")
orchestrator_worker_builder.add_edge("synthesizer", END)

orchestrator_worker = orchestrator_worker_builder.compile()
```

## 5. 评估器-优化器 (Evaluator-Optimizer)

在评估器-优化器工作流中，一个 LLM 调用创建响应，另一个评估该响应。如果评估器或[人工干预](/oss/python/langgraph/interrupts)确定响应需要改进，则提供反馈并重新创建响应。此循环持续进行，直到生成可接受的响应。

![评估器-优化器](https://mintcdn.com/langchain-5e9cc07a/-_xGPoyjhyiDWTPJ/oss/images/evaluator_optimizer.png)

### Graph API 实现
```python
# 图状态
class State(TypedDict):
    joke: str
    topic: str
    feedback: str
    funny_or_not: str

# 用于评估的结构化输出模式
class Feedback(BaseModel):
    grade: Literal["funny", "not funny"] = Field(
        description="决定笑话是否有趣。",
    )
    feedback: str = Field(
        description="如果笑话不有趣，提供如何改进的反馈。",
    )

# 使用结构化输出增强 LLM
evaluator = llm.with_structured_output(Feedback)

# 节点函数
def llm_call_generator(state: State):
    if state.get("feedback"):
        msg = llm.invoke(
            f"写一个关于 {state['topic']} 的笑话，但要考虑反馈: {state['feedback']}"
        )
    else:
        msg = llm.invoke(f"写一个关于 {state['topic']} 的笑话")
    return {"joke": msg.content}

def llm_call_evaluator(state: State):
    grade = evaluator.invoke(f"评价笑话 {state['joke']}")
    return {"funny_or_not": grade.grade, "feedback": grade.feedback}

# 条件边函数根据评估器的反馈路由回笑话生成器或结束
def route_joke(state: State):
    if state["funny_or_not"] == "funny":
        return "Accepted"
    elif state["funny_or_not"] == "not funny":
        return "Rejected + Feedback"

# 构建工作流
optimizer_builder = StateGraph(State)
optimizer_builder.add_node("llm_call_generator", llm_call_generator)
optimizer_builder.add_node("llm_call_evaluator", llm_call_evaluator)

optimizer_builder.add_edge(START, "llm_call_generator")
optimizer_builder.add_edge("llm_call_generator", "llm_call_evaluator")
optimizer_builder.add_conditional_edges(
    "llm_call_evaluator",
    route_joke,
    {
        "Accepted": END,
        "Rejected + Feedback": "llm_call_generator",
    },
)

optimizer_workflow = optimizer_builder.compile()
```

## 6. 智能体 (Agents)

智能体通常实现为使用[工具](/oss/python/langchain/tools)执行操作的 LLM。它们在连续反馈循环中操作，用于问题和解决方案不可预测的情况。

![智能体](https://mintcdn.com/langchain-5e9cc07a/-_xGPoyjhyiDWTPJ/oss/images/agent.png)

### 工具定义
```python
from langchain.tools import tool

# 定义工具
@tool
def multiply(a: int, b: int) -> int:
    """将 `a` 和 `b` 相乘。"""
    return a * b

@tool
def add(a: int, b: int) -> int:
    """将 `a` 和 `b` 相加。"""
    return a + b

@tool
def divide(a: int, b: int) -> float:
    """将 `a` 除以 `b`。"""
    return a / b

# 使用工具增强 LLM
tools = [add, multiply, divide]
tools_by_name = {tool.name: tool for tool in tools}
llm_with_tools = llm.bind_tools(tools)
```

### Graph API 实现
```python
from langgraph.graph import MessagesState
from langchain.messages import SystemMessage, HumanMessage, ToolMessage

# 节点函数
def llm_call(state: MessagesState):
    return {
        "messages": [
            llm_with_tools.invoke(
                [
                    SystemMessage(
                        content="你是一个有帮助的助手，负责对一组输入执行算术运算。"
                    )
                ] + state["messages"]
            )
        ]
    }

def tool_node(state: dict):
    result = []
    for tool_call in state["messages"][-1].tool_calls:
        tool = tools_by_name[tool_call["name"]]
        observation = tool.invoke(tool_call["args"])
        result.append(ToolMessage(content=observation, tool_call_id=tool_call["id"]))
    return {"messages": result}

# 条件边函数根据 LLM 是否进行工具调用路由到工具节点或结束
def should_continue(state: MessagesState) -> Literal["tool_node", END]:
    messages = state["messages"]
    last_message = messages[-1]

    if last_message.tool_calls:
        return "tool_node"
    return END

# 构建工作流
agent_builder = StateGraph(MessagesState)
agent_builder.add_node("llm_call", llm_call)
agent_builder.add_node("tool_node", tool_node)

agent_builder.add_edge(START, "llm_call")
agent_builder.add_conditional_edges(
    "llm_call",
    should_continue,
    ["tool_node", END]
)
agent_builder.add_edge("tool_node", "llm_call")

agent = agent_builder.compile()
```

## 模式选择指南

| 模式 | 适用场景 | 复杂度 | 控制级别 |
|------|----------|--------|----------|
| **提示链** | 顺序处理、内容生成 | 低 | 高 |
| **并行化** | 独立任务、性能优化 | 中 | 中 |
| **路由** | 分类处理、多路径决策 | 中 | 高 |
| **编排器-工作器** | 复杂任务分解、动态子任务 | 高 | 高 |
| **评估器-优化器** | 质量保证、迭代改进 | 高 | 中 |
| **智能体** | 动态决策、工具使用 | 高 | 低 |

## 最佳实践

1. **状态设计**：保持状态简洁，只存储必要数据
2. **错误处理**：为不同节点类型实施适当的错误处理策略
3. **可观察性**：使用 LangSmith 进行监控和调试
4. **测试**：为每个节点和工作流创建单元测试
5. **文档**：为复杂工作流提供清晰的文档和可视化

## 下一步

- 查看[快速入门](/oss/python/langchain/quickstart)开始使用智能体
- 了解[智能体工作原理](/oss/python/langchain/agents)
- 探索[高级模式](/oss/python/langgraph/advanced-patterns)
- 学习[部署最佳实践](/oss/python/langgraph/deploy)

通过掌握这些模式，你可以构建从简单工作流到复杂自主智能体的各种 AI 应用系统。
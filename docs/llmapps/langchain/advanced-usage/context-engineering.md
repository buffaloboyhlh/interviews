# 智能体上下文工程完整教程

## 概述：为什么智能体会失败？

构建智能体（或任何LLM应用）的难点在于使它们足够可靠。虽然原型可能运行良好，但在实际使用中常常失败。

### 智能体失败的原因

当智能体失败时，通常是因为内部的LLM调用采取了错误的行动/没有达到我们的预期。LLM失败有两个主要原因：

1. 底层LLM能力不足
2. 没有将"正确的"上下文传递给LLM

**更常见的是第二个原因导致智能体不可靠。**

**上下文工程**是指以正确的格式提供正确的信息和工具，使LLM能够完成任务。这是AI工程师的首要工作。缺乏"正确的"上下文是构建更可靠智能体的首要障碍，而LangChain的智能体抽象设计独特，旨在促进上下文工程。

## 智能体循环

典型的智能体循环包含两个主要步骤：

1. **模型调用** - 使用提示和可用工具调用LLM，返回响应或执行工具的请求
2. **工具执行** - 执行LLM请求的工具，返回工具结果

这个循环持续进行，直到LLM决定结束。

## 可控因素

要构建可靠的智能体，你需要控制智能体循环每一步发生的事情，以及步骤之间发生的事情。

| 上下文类型 | 控制内容 | 瞬态或持久化 |
|----------|---------|-------------|
| **模型上下文** | 模型调用的内容（指令、消息历史、工具、响应格式） | 瞬态 |
| **工具上下文** | 工具可以访问和产生的内容（对状态、存储、运行时上下文的读写） | 持久化 |
| **生命周期上下文** | 模型和工具调用之间发生的事情（摘要、护栏、日志记录等） | 持久化 |

### 数据源

在整个过程中，你的智能体访问（读/写）不同的数据源：

| 数据源 | 又名 | 范围 | 示例 |
|--------|------|------|------|
| **运行时上下文** | 静态配置 | 会话范围 | 用户ID、API密钥、数据库连接、权限、环境设置 |
| **状态** | 短期记忆 | 会话范围 | 当前消息、上传的文件、认证状态、工具结果 |
| **存储** | 长期记忆 | 跨会话 | 用户偏好、提取的见解、记忆、历史数据 |

## 模型上下文

控制每次模型调用的内容 - 指令、可用工具、使用哪个模型以及输出格式。这些决策直接影响可靠性和成本。

### 系统提示

系统提示设置LLM的行为和能力。不同的用户、上下文或对话阶段需要不同的指令。

**基于状态的动态提示**
```python
from langchain.agents import create_agent
from langchain.agents.middleware import dynamic_prompt, ModelRequest

@dynamic_prompt
def state_aware_prompt(request: ModelRequest) -> str:
    message_count = len(request.messages)

    base = "你是一个有用的助手。"

    if message_count > 10:
        base += "\n这是一个长对话 - 请特别简洁。"

    return base

agent = create_agent(
    model="openai:gpt-4o",
    tools=[...],
    middleware=[state_aware_prompt]
)
```

**基于存储的用户偏好提示**
```python
from dataclasses import dataclass

@dataclass
class Context:
    user_id: str

@dynamic_prompt
def store_aware_prompt(request: ModelRequest) -> str:
    user_id = request.runtime.context.user_id
    store = request.runtime.store
    user_prefs = store.get(("preferences",), user_id)

    base = "你是一个有用的助手。"

    if user_prefs:
        style = user_prefs.value.get("communication_style", "balanced")
        base += f"\n用户偏好{style}风格的回复。"

    return base
```

### 消息管理

消息构成发送给LLM的提示。管理消息内容对于确保LLM有正确的信息来良好响应至关重要。

**注入文件上下文**
```python
from langchain.agents.middleware import wrap_model_call, ModelRequest, ModelResponse

@wrap_model_call
def inject_file_context(request: ModelRequest, handler) -> ModelResponse:
    uploaded_files = request.state.get("uploaded_files", [])

    if uploaded_files:
        file_descriptions = []
        for file in uploaded_files:
            file_descriptions.append(f"- {file['name']} ({file['type']}): {file['summary']}")

        file_context = f"""本次对话中你可以访问的文件：
{chr(10).join(file_descriptions)}

回答问题时请参考这些文件。"""

        messages = [
            *request.messages,
            {"role": "user", "content": file_context},
        ]
        request = request.override(messages=messages)

    return handler(request)
```

### 工具管理

工具让模型能够与数据库、API和外部系统交互。你定义和选择工具的方式直接影响模型能否有效完成任务。

**工具定义最佳实践**
```python
from langchain.tools import tool

@tool(parse_docstring=True)
def search_orders(user_id: str, status: str, limit: int = 10) -> str:
    """按状态搜索用户订单。

    当用户询问订单历史或想要检查订单状态时使用此工具。
    始终按提供的状态进行过滤。

    参数：
        user_id: 用户的唯一标识符
        status: 订单状态：'pending'、'shipped' 或 'delivered'
        limit: 返回的最大结果数
    """
    # 实现代码
    pass
```

**基于状态动态选择工具**
```python
@wrap_model_call
def state_based_tools(request: ModelRequest, handler) -> ModelResponse:
    state = request.state
    is_authenticated = state.get("authenticated", False)
    message_count = len(state["messages"])

    if not is_authenticated:
        tools = [t for t in request.tools if t.name.startswith("public_")]
        request = request.override(tools=tools)
    elif message_count < 5:
        tools = [t for t in request.tools if t.name != "advanced_search"]
        request = request.override(tools=tools)

    return handler(request)
```

### 模型选择

不同的模型有不同的优势、成本和上下文窗口。根据当前任务选择合适的模型。

**基于对话长度选择模型**
```python
from langchain.chat_models import init_chat_model

large_model = init_chat_model("anthropic:claude-sonnet-4-5")
standard_model = init_chat_model("openai:gpt-4o")
efficient_model = init_chat_model("openai:gpt-4o-mini")

@wrap_model_call
def state_based_model(request: ModelRequest, handler) -> ModelResponse:
    message_count = len(request.messages)

    if message_count > 20:
        model = large_model
    elif message_count > 10:
        model = standard_model
    else:
        model = efficient_model

    request = request.override(model=model)
    return handler(request)
```

### 响应格式

结构化输出将非结构化文本转换为经过验证的结构化数据。

**定义响应格式**
```python
from pydantic import BaseModel, Field

class CustomerSupportTicket(BaseModel):
    """从客户消息中提取的结构化票据信息。"""

    category: str = Field(description="问题类别：'billing'、'technical'、'account' 或 'product'")
    priority: str = Field(description="紧急程度：'low'、'medium'、'high' 或 'critical'")
    summary: str = Field(description="客户问题的一句话摘要")
    customer_sentiment: str = Field(description="客户情绪：'frustrated'、'neutral' 或 'satisfied'")
```

**基于状态动态选择响应格式**
```python
class SimpleResponse(BaseModel):
    answer: str = Field(description="简短回答")

class DetailedResponse(BaseModel):
    answer: str = Field(description="详细回答")
    reasoning: str = Field(description="推理过程说明")
    confidence: float = Field(description="置信度分数0-1")

@wrap_model_call
def state_based_output(request: ModelRequest, handler) -> ModelResponse:
    message_count = len(request.messages)

    if message_count < 3:
        request = request.override(response_format=SimpleResponse)
    else:
        request = request.override(response_format=DetailedResponse)

    return handler(request)
```

## 工具上下文

工具的特殊之处在于它们既读取又写入上下文。

### 读取上下文

大多数真实世界的工具需要的不仅仅是LLM的参数。它们需要用户ID进行数据库查询、API密钥访问外部服务，或当前会话状态来做决策。

**从运行时上下文读取配置**
```python
from dataclasses import dataclass

@dataclass
class Context:
    user_id: str
    api_key: str
    db_connection: str

@tool
def fetch_user_data(query: str, runtime: ToolRuntime[Context]) -> str:
    """使用运行时上下文配置获取数据。"""
    user_id = runtime.context.user_id
    api_key = runtime.context.api_key
    db_connection = runtime.context.db_connection

    results = perform_database_query(db_connection, query, api_key)
    return f"为用户 {user_id} 找到 {len(results)} 条结果"
```

### 写入上下文

工具结果可用于帮助智能体完成给定任务。工具既可以直接向模型返回结果，也可以更新智能体的记忆，使重要的上下文在后续步骤中可用。

**写入状态跟踪会话信息**
```python
from langgraph.types import Command

@tool
def authenticate_user(password: str, runtime: ToolRuntime) -> Command:
    """验证用户并更新状态。"""
    if password == "correct":
        return Command(update={"authenticated": True})
    else:
        return Command(update={"authenticated": False})
```

**写入存储持久化数据**
```python
@tool
def save_preference(preference_key: str, preference_value: str, runtime: ToolRuntime[Context]) -> str:
    """将用户偏好保存到存储。"""
    user_id = runtime.context.user_id
    store = runtime.store
    
    existing_prefs = store.get(("preferences",), user_id)
    prefs = existing_prefs.value if existing_prefs else {}
    prefs[preference_key] = preference_value
    
    store.put(("preferences",), user_id, prefs)
    return f"保存偏好：{preference_key} = {preference_value}"
```

## 生命周期上下文

控制**核心智能体步骤之间**发生的事情 - 拦截数据流以实现横切关注点，如摘要、护栏和日志记录。

### 自动摘要

最常见的生命周期模式之一是在对话历史过长时自动压缩。与模型上下文中显示的瞬态消息修剪不同，摘要**持久更新状态** - 永久用摘要替换旧消息，该摘要将保存供所有未来轮次使用。

**使用内置摘要中间件**
```python
from langchain.agents.middleware import SummarizationMiddleware

agent = create_agent(
    model="openai:gpt-4o",
    tools=[...],
    middleware=[
        SummarizationMiddleware(
            model="openai:gpt-4o-mini",
            max_tokens_before_summary=4000,  # 在4000个token时触发摘要
            messages_to_keep=20,  # 摘要后保留最后20条消息
        ),
    ],
)
```

### 自定义生命周期钩子

**Before Model钩子**
```python
from langchain.agents.middleware import before_model

@before_model
def log_before_model(state: AgentState, runtime: Runtime[Context]) -> dict | None:
    user_name = runtime.context.user_name
    session_id = runtime.context.session_id
    print(f"为用户 {user_name} 处理请求 (会话: {session_id})")
    return None
```

**After Model钩子**
```python
from langchain.agents.middleware import after_model

@after_model
def log_after_model(state: AgentState, runtime: Runtime[Context]) -> dict | None:
    user_name = runtime.context.user_name
    session_id = runtime.context.session_id
    
    if runtime.store and state["messages"]:
        last_message = state["messages"][-1]
        runtime.store.set(
            ("conversations",), 
            session_id, 
            {
                "user": user_name,
                "last_interaction": str(last_message.content)[:100],
                "timestamp": "2024-01-01"
            }
        )
    
    return None
```

## 完整示例：构建上下文感知的客户服务智能体

```python
from dataclasses import dataclass
from langchain.agents import create_agent
from langchain.agents.middleware import (
    dynamic_prompt, wrap_model_call, before_model, after_model,
    SummarizationMiddleware
)
from langchain.tools import tool, ToolRuntime
from langgraph.types import Command
from pydantic import BaseModel, Field

# 上下文定义
@dataclass
class CustomerContext:
    user_id: str
    user_tier: str  # basic, premium, vip
    language: str = "zh-CN"
    permissions: list[str] = None

    def __post_init__(self):
        if self.permissions is None:
            self.permissions = ["basic_support"]

# 响应格式
class SupportResponse(BaseModel):
    answer: str = Field(description="对客户问题的回答")
    next_steps: list[str] = Field(description="建议的后续步骤")
    confidence: float = Field(description="回答的置信度")

# 动态系统提示
@dynamic_prompt
def customer_aware_prompt(request: ModelRequest) -> str:
    context = request.runtime.context
    base = f"你是客户服务助手，正在为{context.user_tier}级别用户服务。"
    
    if context.user_tier == "vip":
        base += " 这是VIP用户，请提供优先和个性化服务。"
    elif context.user_tier == "premium":
        base += " 这是高级用户，请提供优质服务。"
    
    if "zh-CN" in context.language:
        base += " 请使用中文回复。"
    
    return base

# 工具：获取用户历史
@tool
def get_user_ticket_history(runtime: ToolRuntime[CustomerContext]) -> str:
    """获取用户的工单历史。"""
    user_id = runtime.context.user_id
    store = runtime.store
    
    history = store.get(("ticket_history",), user_id)
    if history:
        return f"用户最近的工单：{history.value}"
    return "用户没有历史工单记录"

# 工具：创建新工单
@tool
def create_support_ticket(issue: str, runtime: ToolRuntime[CustomerContext]) -> Command:
    """为用户创建支持工单。"""
    user_id = runtime.context.user_id
    store = runtime.store
    
    # 获取现有历史
    history = store.get(("ticket_history",), user_id)
    tickets = history.value if history else []
    
    # 添加新工单
    new_ticket = {
        "issue": issue,
        "timestamp": "2024-01-01",
        "status": "open"
    }
    tickets.append(new_ticket)
    
    # 更新存储
    store.put(("ticket_history",), user_id, tickets)
    
    return Command(
        update={"last_ticket_created": new_ticket},
        message=f"已创建工单：{issue}"
    )

# 创建完整的智能体
agent = create_agent(
    model="openai:gpt-4o",
    tools=[get_user_ticket_history, create_support_ticket],
    middleware=[
        customer_aware_prompt,
        SummarizationMiddleware(
            model="openai:gpt-4o-mini",
            max_tokens_before_summary=3000,
        ),
    ],
    context_schema=CustomerContext,
    response_format=SupportResponse
)

# 使用示例
result = agent.invoke(
    {"messages": [{"role": "user", "content": "我的订单有问题"}]},
    context=CustomerContext(
        user_id="user_123",
        user_tier="premium",
        language="zh-CN"
    )
)
```

## 最佳实践

1. **从简单开始** - 从静态提示和工具开始，仅在需要时添加动态功能
2. **增量测试** - 一次添加一个上下文工程功能
3. **监控性能** - 跟踪模型调用、token使用和延迟
4. **使用内置中间件** - 利用`SummarizationMiddleware`等现有组件
5. **记录上下文策略** - 明确说明正在传递什么上下文以及原因
6. **理解瞬态与持久化**：模型上下文更改是瞬态的（每次调用），而生命周期上下文更改会持久化到状态

通过有效实施上下文工程，你可以显著提高智能体的可靠性和实用性，使其能够处理复杂的现实世界任务。
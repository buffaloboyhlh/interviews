# LangChain Runtime 使用教程

## 概述

LangChain 的 [`create_agent`](https://reference.langchain.com/python/langchain/agents/#langchain.agents.create_agent) 在底层运行在 LangGraph 的 Runtime 之上。

LangGraph 暴露了一个 [Runtime](https://reference.langchain.com/python/langgraph/runtime/#langgraph.runtime.Runtime) 对象，包含以下信息：

1. **Context**：静态信息，如用户 ID、数据库连接或其他代理调用依赖项
2. **Store**：用于[长期记忆](/oss/python/langchain/long-term-memory)的 [BaseStore](https://reference.langchain.com/python/langgraph/store/#langgraph.store.base.BaseStore) 实例
3. **Stream writer**：通过 `"custom"` 流模式流式传输信息的对象

你可以在[工具内部](#在工具中访问)和[中间件内部](#在中间件中访问)访问 Runtime 信息。

## 基本用法

### 定义 Context Schema

使用 [`create_agent`](https://reference.langchain.com/python/langchain/agents/#langchain.agents.create_agent) 创建代理时，可以指定 `context_schema` 来定义存储在代理 [Runtime](https://reference.langchain.com/python/langgraph/runtime/#langgraph.runtime.Runtime) 中的 `context` 结构。

调用代理时，传递带有相关配置的 `context` 参数：

```python
from dataclasses import dataclass
from langchain.agents import create_agent

@dataclass
class Context:
    user_name: str

agent = create_agent(
    model="openai:gpt-5-nano",
    tools=[...],
    context_schema=Context  # 定义 context 结构
)

# 调用时传递 context
agent.invoke(
    {"messages": [{"role": "user", "content": "What's my name?"}]},
    context=Context(user_name="John Smith")  # 设置具体 context 值
)
```

## 在工具中访问 Runtime

你可以在工具内部访问 Runtime 信息来：

* 访问 context
* 读取或写入长期记忆
* 写入[自定义流](/oss/python/langchain/streaming#custom-updates)（例如工具进度/更新）

使用 `ToolRuntime` 参数在工具内部访问 [Runtime](https://reference.langchain.com/python/langgraph/runtime/#langgraph.runtime.Runtime) 对象。

### 示例：访问用户信息和长期记忆

```python
from dataclasses import dataclass
from langchain.tools import tool, ToolRuntime

@dataclass
class Context:
    user_id: str

@tool
def fetch_user_email_preferences(runtime: ToolRuntime[Context]) -> str:
    """从存储中获取用户的电子邮件偏好设置。"""
    # 访问 context 中的用户 ID
    user_id = runtime.context.user_id

    # 默认偏好设置
    preferences: str = "用户偏好简洁礼貌的电子邮件。"

    # 如果存在存储，尝试从长期记忆中获取用户偏好
    if runtime.store:
        # 从存储中获取用户记忆
        memory = runtime.store.get(("users",), user_id)
        if memory:
            preferences = memory.value["preferences"]

    return preferences
```

### 示例：写入自定义流和存储

```python
from langchain.tools import tool, ToolRuntime

@tool
def process_user_data(runtime: ToolRuntime[Context]) -> str:
    """处理用户数据并更新进度。"""
    user_id = runtime.context.user_id
    
    # 向自定义流写入进度更新
    if runtime.stream_writer:
        runtime.stream_writer.send({
            "type": "progress",
            "message": "开始处理用户数据...",
            "user_id": user_id
        })
    
    # 模拟数据处理
    processed_data = f"已处理用户 {user_id} 的数据"
    
    # 将结果保存到长期记忆
    if runtime.store:
        runtime.store.set(
            ("processed_data",), 
            user_id, 
            {"data": processed_data, "timestamp": "2024-01-01"}
        )
    
    # 发送完成通知
    if runtime.stream_writer:
        runtime.stream_writer.send({
            "type": "progress", 
            "message": "用户数据处理完成",
            "user_id": user_id
        })
    
    return processed_data
```

## 在中间件中访问 Runtime

你可以在中间件中访问 Runtime 信息来创建动态提示、修改消息或基于用户上下文控制代理行为。

使用 `request.runtime` 在中间件装饰器内部访问 [Runtime](https://reference.langchain.com/python/langgraph/runtime/#langgraph.runtime.Runtime) 对象。Runtime 对象在传递给中间件函数的 [`ModelRequest`](https://reference.langchain.com/python/langchain/middleware/#langchain.agents.middleware.ModelRequest) 参数中可用。

### 动态提示示例

```python
from dataclasses import dataclass
from langchain.agents import create_agent
from langchain.agents.middleware import dynamic_prompt, ModelRequest

@dataclass
class Context:
    user_name: str
    user_role: str

# 动态系统提示
@dynamic_prompt
def dynamic_system_prompt(request: ModelRequest) -> str:
    user_name = request.runtime.context.user_name
    user_role = request.runtime.context.user_role
    
    # 基于用户角色定制系统提示
    if user_role == "admin":
        system_prompt = f"你是管理员 {user_name} 的助手。你可以访问所有系统功能。"
    elif user_role == "customer":
        system_prompt = f"你是客户 {user_name} 的助手。请提供友好的客户服务。"
    else:
        system_prompt = f"你是 {user_name} 的助手。请提供有帮助的回答。"
    
    return system_prompt
```

### Before/After Model 钩子示例

```python
from dataclasses import dataclass
from langchain.agents import create_agent, AgentState
from langchain.agents.middleware import before_model, after_model
from langgraph.runtime import Runtime

@dataclass
class Context:
    user_name: str
    session_id: str

# Before model 钩子 - 模型调用前执行
@before_model
def log_before_model(state: AgentState, runtime: Runtime[Context]) -> dict | None:
    user_name = runtime.context.user_name
    session_id = runtime.context.session_id
    
    print(f"为用户 {user_name} 处理请求 (会话: {session_id})")
    
    # 可以在此修改状态或添加验证
    return None

# After model 钩子 - 模型调用后执行
@after_model
def log_after_model(state: AgentState, runtime: Runtime[Context]) -> dict | None:
    user_name = runtime.context.user_name
    session_id = runtime.context.session_id
    
    print(f"完成用户 {user_name} 的请求 (会话: {session_id})")
    
    # 记录到长期记忆
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

### 完整的代理配置示例

```python
from dataclasses import dataclass
from langchain.agents import create_agent, AgentState
from langchain.agents.middleware import dynamic_prompt, before_model, after_model, ModelRequest
from langgraph.runtime import Runtime

@dataclass
class Context:
    user_name: str
    user_role: str
    session_id: str
    api_key: str  # 可以包含敏感信息，如 API 密钥

# 动态提示
@dynamic_prompt
def role_based_prompt(request: ModelRequest) -> str:
    context = request.runtime.context
    base_prompt = f"你正在与 {context.user_name} ({context.user_role}) 对话。"
    
    if context.user_role == "premium":
        base_prompt += " 这是尊享用户，请提供优先服务。"
    elif context.user_role == "trial":
        base_prompt += " 这是试用用户，请友好地介绍我们的服务。"
    
    return base_prompt

# 请求验证中间件
@before_model
def validate_request(state: AgentState, runtime: Runtime[Context]) -> dict | None:
    # 检查必要的上下文信息
    if not runtime.context.api_key:
        return {
            "messages": [{
                "role": "assistant", 
                "content": "认证信息缺失，请联系管理员。"
            }]
        }
    
    # 可以添加其他验证逻辑
    return None

# 创建配置完整的代理
agent = create_agent(
    model="openai:gpt-5-nano",
    tools=[fetch_user_email_preferences, process_user_data],
    middleware=[role_based_prompt, validate_request, log_before_model, log_after_model],
    context_schema=Context
)

# 使用完整上下文调用代理
result = agent.invoke(
    {"messages": [{"role": "user", "content": "帮我查看我的账户信息"}]},
    context=Context(
        user_name="张三",
        user_role="premium", 
        session_id="session_123",
        api_key="sk-xxx"
    )
)
```

## 最佳实践

### 1. 合理设计 Context Schema

```python
from dataclasses import dataclass
from typing import Optional

@dataclass
class AppContext:
    user_id: str
    user_tier: str = "basic"  # basic, premium, admin
    language: str = "zh-CN"
    timezone: str = "Asia/Shanghai"
    permissions: list[str] = None
    
    def __post_init__(self):
        if self.permissions is None:
            self.permissions = []
```

### 2. 错误处理

```python
from langchain.tools import tool, ToolRuntime

@tool
def safe_tool_operation(runtime: ToolRuntime[Context]) -> str:
    """带有错误处理的工具示例。"""
    try:
        # 检查必要的上下文
        if not hasattr(runtime.context, 'user_id'):
            return "错误：缺少用户ID"
        
        # 业务逻辑
        user_id = runtime.context.user_id
        return f"处理用户 {user_id} 的请求"
        
    except Exception as e:
        # 记录错误到流
        if runtime.stream_writer:
            runtime.stream_writer.send({
                "type": "error",
                "message": f"工具执行失败: {str(e)}"
            })
        return f"操作失败: {str(e)}"
```

### 3. 性能优化

```python
from langchain.agents.middleware import before_model

@before_model
def cache_check(state: AgentState, runtime: Runtime[Context]) -> dict | None:
    """检查缓存以避免重复处理。"""
    if runtime.store:
        # 基于消息内容生成缓存键
        cache_key = hash(str(state["messages"]))
        cached_result = runtime.store.get(("cache",), str(cache_key))
        
        if cached_result:
            # 返回缓存结果
            return {
                "messages": [{
                    "role": "assistant",
                    "content": cached_result.value["response"]
                }]
            }
    
    return None
```

通过合理使用 Runtime 功能，你可以创建更加智能、个性化和高效的 AI 代理应用。
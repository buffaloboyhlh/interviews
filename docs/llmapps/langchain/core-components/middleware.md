# LangChain Middleware 

## 概述

Middleware（中间件）在 LangChain 中提供了对 Agent 执行过程的精细控制能力。它允许你在 Agent 执行的各个关键节点插入自定义逻辑，实现监控、修改、控制和强制执行等功能。

![](https://mintcdn.com/langchain-5e9cc07a/RAP6mjwE5G00xYsA/oss/images/middleware_final.png?fit=max&auto=format&n=RAP6mjwE5G00xYsA&q=85&s=eb4404b137edec6f6f0c8ccb8323eaf1)


## 核心概念

### Agent 执行流程

标准的 Agent 执行循环包含三个主要步骤：

1. 调用模型
2. 选择要执行的工具
3. 当不再调用工具时结束

```python
# 基础 Agent 创建
from langchain.agents import create_agent

agent = create_agent(
    model="openai:gpt-4o",
    tools=[...]
)
```

### Middleware 执行钩子

Middleware 在以下关键节点暴露钩子：

- **before_model**: 在模型调用之前
- **after_model**: 在模型响应之后
- **wrap_model_call**: 围绕模型调用（完全控制）
- **wrap_tool_call**: 围绕工具调用

## 内置 Middleware 使用指南

### 1. SummarizationMiddleware - 对话总结

自动总结对话历史以避免超出 token 限制。

```python
from langchain.agents import create_agent
from langchain.agents.middleware import SummarizationMiddleware

agent = create_agent(
    model="openai:gpt-4o",
    tools=[weather_tool, calculator_tool],
    middleware=[
        SummarizationMiddleware(
            model="openai:gpt-4o-mini",
            max_tokens_before_summary=4000,  # 在 4000 tokens 时触发总结
            messages_to_keep=20,  # 总结后保留最近 20 条消息
        ),
    ],
)
```

**适用场景**：

- 长时间运行的对话
- 多轮对话且有大量历史记录
- 需要保留完整对话上下文的应用

### 2. HumanInTheLoopMiddleware - 人工审核

在执行工具调用前暂停，等待人工批准、编辑或拒绝。

```python
from langchain.agents import create_agent
from langchain.agents.middleware import HumanInTheLoopMiddleware
from langgraph.checkpoint.memory import InMemorySaver

agent = create_agent(
    model="openai:gpt-4o",
    tools=[read_email_tool, send_email_tool],
    checkpointer=InMemorySaver(),  # 必须使用 checkpointer
    middleware=[
        HumanInTheLoopMiddleware(
            interrupt_on={
                "send_email_tool": {  # 发送邮件需要人工审核
                    "allowed_decisions": ["approve", "edit", "reject"],
                },
                "read_email_tool": False,  # 自动批准读取邮件
            }
        ),
    ],
)
```

**重要提示**：必须配置 checkpointer 来维持中断间的状态。

### 3. ModelCallLimitMiddleware - 模型调用限制

限制模型调用次数，防止无限循环或过高成本。

```python
from langchain.agents import create_agent
from langchain.agents.middleware import ModelCallLimitMiddleware

agent = create_agent(
    model="openai:gpt-4o",
    tools=[...],
    middleware=[
        ModelCallLimitMiddleware(
            thread_limit=10,  # 每个线程最多 10 次调用
            run_limit=5,     # 每次运行最多 5 次调用
            exit_behavior="end",  # 达到限制时优雅结束
        ),
    ],
)
```

### 4. ToolCallLimitMiddleware - 工具调用限制

限制特定工具或所有工具的调用次数。

```python
from langchain.agents import create_agent
from langchain.agents.middleware import ToolCallLimitMiddleware

# 限制所有工具调用
global_limiter = ToolCallLimitMiddleware(thread_limit=20, run_limit=10)

# 限制特定工具
search_limiter = ToolCallLimitMiddleware(
    tool_name="search",
    thread_limit=5,
    run_limit=3,
)

agent = create_agent(
    model="openai:gpt-4o",
    tools=[...],
    middleware=[global_limiter, search_limiter],
)
```

### 5. ModelFallbackMiddleware - 模型故障转移

当主模型失败时自动切换到备用模型。

```python
from langchain.agents import create_agent
from langchain.agents.middleware import ModelFallbackMiddleware

agent = create_agent(
    model="openai:gpt-4o",  # 主模型
    tools=[...],
    middleware=[
        ModelFallbackMiddleware(
            "openai:gpt-4o-mini",  # 第一备用模型
            "anthropic:claude-3-5-sonnet-20241022",  # 第二备用模型
        ),
    ],
)
```

### 6. PIIMiddleware - 个人信息检测

检测和处理对话中的个人身份信息。

```python
from langchain.agents import create_agent
from langchain.agents.middleware import PIIMiddleware

agent = create_agent(
    model="openai:gpt-4o",
    tools=[...],
    middleware=[
        # 在用户输入中屏蔽邮箱
        PIIMiddleware("email", strategy="redact", apply_to_input=True),
        # 掩码信用卡号（显示最后4位）
        PIIMiddleware("credit_card", strategy="mask", apply_to_input=True),
        # 自定义 API 密钥检测
        PIIMiddleware(
            "api_key",
            detector=r"sk-[a-zA-Z0-9]{32}",
            strategy="block",  # 检测到时报错
        ),
    ],
)
```

**处理策略**：

- `block`: 检测到时抛出异常
- `redact`: 替换为 `[REDACTED_TYPE]`
- `mask`: 部分掩码
- `hash`: 替换为确定性哈希值

### 7. ToolRetryMiddleware - 工具重试

对失败的工具调用进行指数退避重试。

```python
from langchain.agents import create_agent
from langchain.agents.middleware import ToolRetryMiddleware

agent = create_agent(
    model="openai:gpt-4o",
    tools=[search_tool, database_tool],
    middleware=[
        ToolRetryMiddleware(
            max_retries=3,      # 最多重试 3 次
            backoff_factor=2.0, # 指数退避乘数
            initial_delay=1.0,  # 初始延迟 1 秒
            max_delay=60.0,     # 最大延迟 60 秒
            jitter=True,        # 添加随机抖动
        ),
    ],
)
```

## 自定义 Middleware 开发

### 装饰器方式（简单场景）

适用于只需要单个钩子的简单中间件。

```python
from langchain.agents.middleware import before_model, after_model, wrap_model_call
from langchain.agents.middleware import AgentState
from langchain.messages import AIMessage

# 模型调用前日志记录
@before_model
def log_before_model(state: AgentState, runtime) -> dict | None:
    print(f"准备调用模型，消息数量: {len(state['messages'])}")
    return None

# 模型调用后验证
@after_model
def validate_output(state: AgentState, runtime) -> dict | None:
    last_message = state["messages"][-1]
    if "BLOCKED" in last_message.content:
        return {
            "messages": [AIMessage("我无法响应这个请求。")],
            "jump_to": "end"
        }
    return None

# 模型调用重试包装
@wrap_model_call
def retry_model(request, handler):
    for attempt in range(3):
        try:
            return handler(request)
        except Exception as e:
            if attempt == 2:
                raise
            print(f"重试 {attempt + 1}/3，错误: {e}")
```

### 类方式（复杂场景）

适用于需要多个钩子或复杂配置的中间件。

```python
from langchain.agents.middleware import AgentMiddleware, AgentState
from langchain.agents.middleware import ModelRequest, ModelResponse
from typing import Callable, Any

class ComprehensiveLoggingMiddleware(AgentMiddleware):
    def __init__(self, log_level: str = "INFO"):
        super().__init__()
        self.log_level = log_level
    
    def before_model(self, state: AgentState, runtime) -> dict[str, Any] | None:
        print(f"[{self.log_level}] 模型调用前 - 消息数: {len(state['messages'])}")
        return None
    
    def after_model(self, state: AgentState, runtime) -> dict[str, Any] | None:
        last_msg = state["messages"][-1]
        print(f"[{self.log_level}] 模型响应: {last_msg.content[:100]}...")
        return None
    
    def wrap_model_call(
        self,
        request: ModelRequest,
        handler: Callable[[ModelRequest], ModelResponse],
    ) -> ModelResponse:
        print(f"[{self.log_level}] 开始模型调用")
        start_time = time.time()
        
        try:
            response = handler(request)
            duration = time.time() - start_time
            print(f"[{self.log_level}] 模型调用成功，耗时: {duration:.2f}s")
            return response
        except Exception as e:
            duration = time.time() - start_time
            print(f"[{self.log_level}] 模型调用失败，耗时: {duration:.2f}s，错误: {e}")
            raise
```

### 自定义状态管理

Middleware 可以扩展 Agent 的状态结构：

```python
from typing_extensions import NotRequired
from typing import Any

class CustomState(AgentState):
    model_call_count: NotRequired[int]
    user_preferences: NotRequired[dict]

class StateAwareMiddleware(AgentMiddleware[CustomState]):
    state_schema = CustomState

    def before_model(self, state: CustomState, runtime) -> dict[str, Any] | None:
        # 访问自定义状态
        call_count = state.get("model_call_count", 0)
        preferences = state.get("user_preferences", {})
        
        if call_count > 50:
            return {"jump_to": "end"}
            
        return None

    def after_model(self, state: CustomState, runtime) -> dict[str, Any] | None:
        # 更新自定义状态
        return {
            "model_call_count": state.get("model_call_count", 0) + 1
        }
```

## 高级用例

### 动态工具选择

根据上下文智能选择相关工具：

```python
class SmartToolSelectorMiddleware(AgentMiddleware):
    def wrap_model_call(
        self,
        request: ModelRequest,
        handler: Callable[[ModelRequest], ModelResponse],
    ) -> ModelResponse:
        # 基于对话内容选择相关工具
        user_message = request.state["messages"][-1].content.lower()
        
        if "weather" in user_message:
            relevant_tools = [t for t in request.tools if "weather" in t.name]
        elif "calculate" in user_message:
            relevant_tools = [t for t in request.tools if "calc" in t.name or "math" in t.name]
        else:
            relevant_tools = request.tools[:5]  # 限制工具数量
        
        request.tools = relevant_tools
        return handler(request)
```

### 权限控制中间件

```python
class PermissionMiddleware(AgentMiddleware):
    def __init__(self, user_roles: dict):
        super().__init__()
        self.user_roles = user_roles
    
    def before_model(self, state: AgentState, runtime) -> dict[str, Any] | None:
        user_id = runtime.context.get("user_id")
        user_role = self.user_roles.get(user_id, "guest")
        
        # 基于角色限制工具访问
        if user_role == "guest":
            restricted_tools = ["delete", "admin", "config"]
            for tool in state.get("tools", []):
                if any(restricted in tool.name for restricted in restricted_tools):
                    return {
                        "messages": [AIMessage("权限不足，无法使用该功能")],
                        "jump_to": "end"
                    }
        return None
```

## 最佳实践

### 1. 执行顺序管理

```python
# Middleware 执行顺序很重要
agent = create_agent(
    model="openai:gpt-4o",
    middleware=[
        LoggingMiddleware(),      # 最先执行 - 基础日志
        ValidationMiddleware(),   # 其次 - 输入验证
        SecurityMiddleware(),     # 安全检查
        BusinessLogicMiddleware(), # 业务逻辑
        FallbackMiddleware()      # 最后 - 故障处理
    ],
    tools=[...]
)
```

### 2. 错误处理

```python
class RobustMiddleware(AgentMiddleware):
    def wrap_model_call(self, request, handler):
        try:
            return handler(request)
        except Exception as e:
            # 记录错误但不中断执行
            logger.error(f"Middleware error: {e}")
            # 返回降级响应
            return ModelResponse(
                messages=[AIMessage("系统暂时不可用，请稍后重试。")]
            )
```

### 3. 性能优化

```python
class CachingMiddleware(AgentMiddleware):
    def __init__(self):
        super().__init__()
        self.cache = {}
    
    def wrap_model_call(self, request, handler):
        cache_key = self._generate_cache_key(request)
        
        if cache_key in self.cache:
            return self.cache[cache_key]
        
        response = handler(request)
        self.cache[cache_key] = response
        return response
    
    def _generate_cache_key(self, request):
        # 基于请求内容生成缓存键
        return hash(str(request.messages))
```

## 完整示例

下面是一个综合使用多个 Middleware 的生产级示例：

```python
from langchain.agents import create_agent
from langchain.agents.middleware import (
    SummarizationMiddleware,
    ModelCallLimitMiddleware,
    ToolRetryMiddleware,
    PIIMiddleware
)

# 创建具备完整中间件栈的 Agent
production_agent = create_agent(
    model="openai:gpt-4o",
    tools=[search_tool, calculator_tool, database_tool, email_tool],
    middleware=[
        # 安全和合规
        PIIMiddleware("email", strategy="redact", apply_to_input=True),
        PIIMiddleware("credit_card", strategy="mask", apply_to_input=True),
        
        # 性能和稳定性
        SummarizationMiddleware(
            model="openai:gpt-4o-mini",
            max_tokens_before_summary=3000,
        ),
        ToolRetryMiddleware(
            max_retries=2,
            backoff_factor=1.5,
        ),
        
        # 资源控制
        ModelCallLimitMiddleware(thread_limit=20, run_limit=10),
        
        # 自定义业务逻辑
        CustomLoggingMiddleware(),
        PermissionMiddleware(user_roles=USER_ROLES),
    ],
    checkpointer=InMemorySaver(),  # 用于维持状态
)
```

## 总结

LangChain Middleware 提供了强大的扩展能力，让你能够：

- **监控**：跟踪 Agent 行为，记录日志和分析
- **修改**：转换提示、工具选择和输出格式
- **控制**：添加重试、故障转移和提前终止逻辑
- **强制执行**：应用速率限制、防护栏和 PII 检测

通过合理使用内置 Middleware 和开发自定义 Middleware，你可以构建出更加健壮、安全和高效的 AI Agent 应用。
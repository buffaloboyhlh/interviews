# 使用 Guardrails 构建安全合规的 AI 应用教程

## 什么是 Guardrails？

Guardrails（安全护栏）是在 AI 代理执行的关键节点进行内容验证和过滤的安全检查机制。它们帮助开发者构建安全、合规的 AI 应用，能够在问题发生前检测敏感信息、执行内容策略、验证输出并防止不安全行为。

### 主要应用场景

- **防止 PII 泄露** - 保护个人身份信息
- **检测和阻止提示注入攻击** - 防范恶意输入
- **阻止不当或有害内容** - 过滤违规内容
- **执行业务规则和合规要求** - 满足行业规范
- **验证输出质量和准确性** - 确保响应可靠性

## Guardrails 的两种实现方式

### 1. 确定性 Guardrails
使用基于规则的逻辑，如正则表达式、关键词匹配或显式检查。速度快、可预测且成本低，但可能遗漏复杂违规情况。

### 2. 基于模型的 Guardrails
使用 LLM 或分类器进行语义理解评估。能捕获规则可能遗漏的细微问题，但速度较慢且成本更高。

## 使用内置 Guardrails

### PII 检测中间件

LangChain 提供了内置的 PII 检测中间件，可识别电子邮件、信用卡、IP 地址等常见敏感信息。

```python
from langchain.agents import create_agent
from langchain.agents.middleware import PIIMiddleware

agent = create_agent(
    model="openai:gpt-4o",
    tools=[customer_service_tool, email_tool],
    middleware=[
        # 在发送给模型前对用户输入中的电子邮件进行脱敏
        PIIMiddleware(
            "email",
            strategy="redact",
            apply_to_input=True,
        ),
        # 对用户输入中的信用卡进行掩码处理
        PIIMiddleware(
            "credit_card",
            strategy="mask",
            apply_to_input=True,
        ),
        # 检测到 API 密钥时抛出错误
        PIIMiddleware(
            "api_key",
            detector=r"sk-[a-zA-Z0-9]{32}",
            strategy="block",
            apply_to_input=True,
        ),
    ],
)

# 当用户提供 PII 时，将根据策略进行处理
result = agent.invoke({
    "messages": [{"role": "user", "content": "My email is john.doe@example.com and card is 4532-1234-5678-9010"}]
})
```

**PII 处理策略：**

- `redact` - 替换为 `[REDACTED_TYPE]`
- `mask` - 部分遮蔽（如显示最后4位）
- `hash` - 替换为确定性哈希值
- `block` - 检测到时抛出异常

### 人工介入中间件

对于高风险操作，可以要求人工审批后再执行。

```python
from langchain.agents import create_agent
from langchain.agents.middleware import HumanInTheLoopMiddleware
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.types import Command

agent = create_agent(
    model="openai:gpt-4o",
    tools=[search_tool, send_email_tool, delete_database_tool],
    middleware=[
        HumanInTheLoopMiddleware(
            interrupt_on={
                # 敏感操作需要审批
                "send_email": True,
                "delete_database": True,
                # 安全操作自动批准
                "search": False,
            }
        ),
    ],
    checkpointer=InMemorySaver(),  # 持久化状态
)

config = {"configurable": {"thread_id": "some_id"}}

# 代理将在执行敏感工具前暂停并等待批准
result = agent.invoke(
    {"messages": [{"role": "user", "content": "Send an email to the team"}]},
    config=config
)

# 人工批准后继续执行
result = agent.invoke(
    Command(resume={"decisions": [{"type": "approve"}]}),
    config=config
)
```

## 创建自定义 Guardrails

### Before Agent Guardrails

在代理开始执行前验证请求，适用于会话级检查，如身份验证、速率限制或阻止不当请求。

**类语法：**
```python
from typing import Any
from langchain.agents.middleware import AgentMiddleware, AgentState, hook_config
from langgraph.runtime import Runtime

class ContentFilterMiddleware(AgentMiddleware):
    """确定性护栏：阻止包含禁用关键词的请求"""
    
    def __init__(self, banned_keywords: list[str]):
        super().__init__()
        self.banned_keywords = [kw.lower() for kw in banned_keywords]
    
    @hook_config(can_jump_to=["end"])
    def before_agent(self, state: AgentState, runtime: Runtime) -> dict[str, Any] | None:
        if not state["messages"]:
            return None
            
        first_message = state["messages"][0]
        if first_message.type != "human":
            return None
            
        content = first_message.content.lower()
        
        for keyword in self.banned_keywords:
            if keyword in content:
                return {
                    "messages": [{
                        "role": "assistant",
                        "content": "I cannot process requests containing inappropriate content."
                    }],
                    "jump_to": "end"
                }
        return None

# 使用自定义护栏
agent = create_agent(
    model="openai:gpt-4o",
    tools=[search_tool, calculator_tool],
    middleware=[ContentFilterMiddleware(["hack", "exploit", "malware"])],
)
```

**装饰器语法：**
```python
from typing import Any
from langchain.agents.middleware import before_agent, AgentState, hook_config
from langgraph.runtime import Runtime

banned_keywords = ["hack", "exploit", "malware"]

@before_agent(can_jump_to=["end"])
def content_filter(state: AgentState, runtime: Runtime) -> dict[str, Any] | None:
    """确定性护栏：阻止包含禁用关键词的请求"""
    if not state["messages"]:
        return None
        
    first_message = state["messages"][0]
    if first_message.type != "human":
        return None
        
    content = first_message.content.lower()
    
    for keyword in banned_keywords:
        if keyword in content:
            return {
                "messages": [{
                    "role": "assistant",
                    "content": "I cannot process requests containing inappropriate content."
                }],
                "jump_to": "end"
            }
    return None
```

### After Agent Guardrails

在返回给用户前验证最终输出，适用于基于模型的安全检查、质量验证或最终合规扫描。

**类语法：**
```python
from langchain.agents.middleware import AgentMiddleware, AgentState, hook_config
from langgraph.runtime import Runtime
from langchain_core.messages import AIMessage
from langchain.chat_models import init_chat_model
from typing import Any

class SafetyGuardrailMiddleware(AgentMiddleware):
    """基于模型的护栏：使用 LLM 评估响应安全性"""
    
    def __init__(self):
        super().__init__()
        self.safety_model = init_chat_model("openai:gpt-4o-mini")
    
    @hook_config(can_jump_to=["end"])
    def after_agent(self, state: AgentState, runtime: Runtime) -> dict[str, Any] | None:
        if not state["messages"]:
            return None
            
        last_message = state["messages"][-1]
        if not isinstance(last_message, AIMessage):
            return None
            
        # 使用模型评估安全性
        safety_prompt = f"""评估此响应是否安全适当。
        仅回复 'SAFE' 或 'UNSAFE'。
        
        响应：{last_message.content}"""
        
        result = self.safety_model.invoke([{"role": "user", "content": safety_prompt}])
        
        if "UNSAFE" in result.content:
            return {
                "messages": [{
                    "role": "assistant", 
                    "content": "I cannot provide that response."
                }],
                "jump_to": "end"
            }
        return None
```

## 组合多个 Guardrails

你可以通过将多个 Guardrails 添加到中间件数组中来堆叠它们，构建分层保护：

```python
agent = create_agent(
    model="openai:gpt-4o",
    tools=[search_tool, send_email_tool],
    middleware=[
        # 第1层：确定性输入过滤器（代理前）
        ContentFilterMiddleware(banned_keywords=["hack", "exploit"]),
        
        # 第2层：PII 保护（模型前后）
        PIIMiddleware("email", strategy="redact", apply_to_input=True),
        PIIMiddleware("email", strategy="redact", apply_to_output=True),
        
        # 第3层：敏感工具的人工审批
        HumanInTheLoopMiddleware(interrupt_on={"send_email": True}),
        
        # 第4层：基于模型的安全检查（代理后）
        SafetyGuardrailMiddleware(),
    ],
)
```

## 最佳实践

1. **分层防御**：结合确定性和基于模型的 Guardrails
2. **早期拦截**：在代理执行前尽可能早地拦截问题
3. **成本平衡**：对高频率操作使用确定性检查，对关键决策使用模型检查
4. **持续测试**：定期测试安全机制的有效性
5. **人工监督**：对高风险操作保留人工审批环节

通过合理配置 Guardrails，你可以显著提升 AI 应用的安全性和合规性，同时保持用户体验的流畅性。
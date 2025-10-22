# LangChain 短期记忆

## 概述

短期记忆系统让 AI Agent 能够记住单次对话或线程中的先前交互信息。这对于构建能够理解上下文、学习用户偏好并保持连贯对话的智能应用至关重要。

### 核心概念

- **线程（Thread）**：组织多次交互的会话，类似电子邮件对话
- **检查点（Checkpointer）**：负责状态的持久化存储
- **状态管理**：Agent 通过状态来维护对话历史和自定义信息

## 基础设置

### 1. 启用短期记忆

```python
from langchain.agents import create_agent
from langgraph.checkpoint.memory import InMemorySaver

# 创建带有短期记忆的 Agent
agent = create_agent(
    model="openai:gpt-4o",
    tools=[get_user_info],
    checkpointer=InMemorySaver(),  # 启用内存检查点
)

# 使用线程ID来区分不同对话
result = agent.invoke(
    {"messages": [{"role": "user", "content": "Hi! My name is Bob."}]},
    {"configurable": {"thread_id": "1"}},  # 指定线程ID
)
```

### 2. 生产环境配置

```python
from langchain.agents import create_agent
from langgraph.checkpoint.postgres import PostgresSaver

# 安装依赖：pip install langgraph-checkpoint-postgres
DB_URI = "postgresql://postgres:postgres@localhost:5442/postgres?sslmode=disable"

with PostgresSaver.from_conn_string(DB_URI) as checkpointer:
    checkpointer.setup()  # 自动创建数据库表
    
    agent = create_agent(
        model="openai:gpt-4o",
        tools=[get_user_info],
        checkpointer=checkpointer,  # 使用 PostgreSQL 检查点
    )
```

## 自定义 Agent 状态

### 扩展默认状态

```python
from langchain.agents import create_agent, AgentState
from langgraph.checkpoint.memory import InMemorySaver
from typing import Optional, Dict, List

class CustomAgentState(AgentState):
    """自定义 Agent 状态"""
    user_id: str
    preferences: Dict[str, str]
    conversation_topics: List[str]
    last_active: Optional[str] = None

# 创建使用自定义状态的 Agent
agent = create_agent(
    model="openai:gpt-4o",
    tools=[get_user_info],
    state_schema=CustomAgentState,  # 使用自定义状态模式
    checkpointer=InMemorySaver(),
)

# 调用时传入自定义状态
result = agent.invoke(
    {
        "messages": [{"role": "user", "content": "Hello"}],
        "user_id": "user_123",
        "preferences": {"theme": "dark", "language": "zh-CN"},
        "conversation_topics": ["technology", "programming"]
    },
    {"configurable": {"thread_id": "1"}}
)
```

## 内存管理策略

### 1. 消息修剪（Trim Messages）

当对话历史过长时，修剪消息以适配上下文窗口。

```python
from langchain.messages import RemoveMessage
from langgraph.graph.message import REMOVE_ALL_MESSAGES
from langchain.agents.middleware import before_model
from langgraph.runtime import Runtime
from typing import Any

@before_model
def trim_messages(state: AgentState, runtime: Runtime) -> dict[str, Any] | None:
    """保留最近几条消息以适配上下文窗口"""
    messages = state["messages"]
    
    # 如果消息数量不多，不需要修剪
    if len(messages) <= 4:
        return None
    
    # 保留系统消息和最近的3条消息
    system_messages = [msg for msg in messages if msg.type == "system"]
    recent_messages = messages[-3:]
    
    new_messages = system_messages + recent_messages
    
    return {
        "messages": [
            RemoveMessage(id=REMOVE_ALL_MESSAGES),  # 移除所有现有消息
            *new_messages  # 添加修剪后的消息
        ]
    }

# 使用修剪中间件的 Agent
agent = create_agent(
    model="openai:gpt-4o",
    tools=[],
    middleware=[trim_messages],
    checkpointer=InMemorySaver(),
)
```

### 2. 消息删除（Delete Messages）

永久删除特定消息以管理对话历史。

```python
from langchain.agents.middleware import after_model
from langchain.messages import RemoveMessage

@after_model
def delete_old_messages(state: AgentState, runtime: Runtime) -> dict | None:
    """删除旧消息以保持对话可管理"""
    messages = state["messages"]
    
    # 如果消息超过5条，删除最早的两条
    if len(messages) > 5:
        messages_to_remove = messages[:2]
        return {
            "messages": [RemoveMessage(id=msg.id) for msg in messages_to_remove]
        }
    
    return None

# 使用删除中间件的 Agent
agent = create_agent(
    model="openai:gpt-4o",
    tools=[],
    middleware=[delete_old_messages],
    checkpointer=InMemorySaver(),
)
```

### 3. 消息总结（Summarize Messages）

使用总结中间件自动总结长对话历史。

```python
from langchain.agents import create_agent
from langchain.agents.middleware import SummarizationMiddleware
from langgraph.checkpoint.memory import InMemorySaver

checkpointer = InMemorySaver()

agent = create_agent(
    model="openai:gpt-4o",
    tools=[],
    middleware=[
        SummarizationMiddleware(
            model="openai:gpt-4o-mini",  # 使用更便宜的模型进行总结
            max_tokens_before_summary=2000,  # 在2000个token时触发总结
            messages_to_keep=10,  # 总结后保留最近10条消息
            summary_prompt="请总结之前的对话，保留关键信息：",
        )
    ],
    checkpointer=checkpointer,
)

# 测试长对话
config = {"configurable": {"thread_id": "1"}}

agent.invoke({"messages": "Hi, my name is Bob"}, config)
agent.invoke({"messages": "I'm a software engineer from Beijing"}, config)
agent.invoke({"messages": "I enjoy hiking and reading books"}, config)
agent.invoke({"messages": "My favorite programming language is Python"}, config)

# 即使经过多次对话，Agent 仍然记得用户信息
final_response = agent.invoke({"messages": "Can you remind me what I told you about myself?"}, config)
print(final_response["messages"][-1].content)
```

## 访问和操作内存

### 1. 在工具中访问内存

```python
from langchain.tools import tool, ToolRuntime
from langchain.agents import create_agent, AgentState

class UserState(AgentState):
    user_profile: dict
    conversation_count: int

@tool
def get_user_profile(runtime: ToolRuntime) -> str:
    """获取用户档案信息"""
    state = runtime.state
    user_profile = state.get("user_profile", {})
    conversation_count = state.get("conversation_count", 0)
    
    return f"用户档案: {user_profile}, 对话次数: {conversation_count}"

@tool  
def update_user_preference(runtime: ToolRuntime, preference: str, value: str) -> str:
    """更新用户偏好"""
    from langgraph.types import Command
    
    # 更新状态
    return Command(update={
        "user_profile": {
            **runtime.state.get("user_profile", {}),
            preference: value
        },
        "conversation_count": runtime.state.get("conversation_count", 0) + 1
    })

agent = create_agent(
    model="openai:gpt-4o",
    tools=[get_user_profile, update_user_preference],
    state_schema=UserState,
    checkpointer=InMemorySaver(),
)
```

### 2. 使用动态提示

```python
from langchain.agents.middleware import dynamic_prompt, ModelRequest
from typing import TypedDict

class ConversationContext(TypedDict):
    user_name: str
    user_role: str

@dynamic_prompt
def personalized_system_prompt(request: ModelRequest) -> str:
    """基于用户上下文的动态系统提示"""
    context = request.runtime.context
    user_name = context.get("user_name", "用户")
    user_role = context.get("user_role", "访客")
    
    return f"""
    你是一个有帮助的助手，正在与{user_name}对话。
    {user_name}的身份是：{user_role}
    
    请根据对话历史提供个性化的回应。
    保持友好和专业的态度。
    """

def get_weather(city: str) -> str:
    """获取天气信息"""
    return f"{city}的天气是晴朗的，25°C"

agent = create_agent(
    model="openai:gpt-4o",
    tools=[get_weather],
    middleware=[personalized_system_prompt],
    context_schema=ConversationContext,
)

# 使用上下文调用
result = agent.invoke(
    {"messages": [{"role": "user", "content": "今天天气怎么样？"}]},
    context=ConversationContext(user_name="张三", user_role="软件工程师")
)
```

### 3. Before Model 中间件

在模型调用前访问和修改状态。

```python
from langchain.agents.middleware import before_model
from langchain.messages import SystemMessage

@before_model
def enhance_with_context(state: AgentState, runtime: Runtime) -> dict[str, Any] | None:
    """在模型调用前增强上下文"""
    messages = state["messages"]
    
    # 获取用户信息
    user_id = state.get("user_id", "unknown")
    preferences = state.get("preferences", {})
    
    # 添加系统消息提供上下文
    context_message = SystemMessage(content=f"""
    当前用户ID: {user_id}
    用户偏好: {preferences}
    请根据以上信息提供个性化服务。
    """)
    
    # 将上下文消息添加到对话开始
    enhanced_messages = [context_message] + messages
    
    return {"messages": enhanced_messages}

agent = create_agent(
    model="openai:gpt-4o",
    tools=[],
    middleware=[enhance_with_context],
    state_schema=CustomAgentState,
    checkpointer=InMemorySaver(),
)
```

### 4. After Model 中间件

在模型调用后处理响应和状态。

```python
from langchain.agents.middleware import after_model
from langchain.messages import RemoveMessage

@after_model
def track_conversation_metrics(state: AgentState, runtime: Runtime) -> dict | None:
    """跟踪对话指标并清理敏感信息"""
    messages = state["messages"]
    
    # 更新对话统计
    conversation_count = state.get("conversation_count", 0) + 1
    last_active = datetime.now().isoformat()
    
    # 检查并移除包含敏感信息的消息
    sensitive_keywords = ["密码", "secret", "password", "token"]
    messages_to_remove = []
    
    for msg in messages:
        if any(keyword in msg.content.lower() for keyword in sensitive_keywords):
            messages_to_remove.append(msg)
    
    updates = {
        "conversation_count": conversation_count,
        "last_active": last_active
    }
    
    if messages_to_remove:
        updates["messages"] = [RemoveMessage(id=msg.id) for msg in messages_to_remove]
    
    return updates

agent = create_agent(
    model="openai:gpt-4o",
    tools=[],
    middleware=[track_conversation_metrics],
    state_schema=CustomAgentState,
    checkpointer=InMemorySaver(),
)
```

## 实际应用场景

### 场景1：个性化客户服务

```python
from datetime import datetime
from typing import Dict, List, Optional

class CustomerServiceState(AgentState):
    customer_id: str
    ticket_history: List[Dict]
    customer_tier: str  # "standard", "premium", "vip"
    last_issue: Optional[str] = None
    satisfaction_score: Optional[int] = None

def create_customer_service_agent():
    """创建客户服务 Agent"""
    
    @tool
    def create_support_ticket(runtime: ToolRuntime, issue: str, priority: str) -> str:
        """创建支持工单"""
        from langgraph.types import Command
        
        ticket = {
            "id": f"TICKET_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            "issue": issue,
            "priority": priority,
            "created_at": datetime.now().isoformat(),
            "status": "open"
        }
        
        return Command(update={
            "ticket_history": runtime.state.get("ticket_history", []) + [ticket],
            "last_issue": issue
        })
    
    @tool
    def get_customer_history(runtime: ToolRuntime) -> str:
        """获取客户历史记录"""
        state = runtime.state
        ticket_history = state.get("ticket_history", [])
        customer_tier = state.get("customer_tier", "standard")
        
        if not ticket_history:
            return "这是该客户的第一次联系"
        
        last_ticket = ticket_history[-1]
        return f"""
        客户等级: {customer_tier}
        总工单数: {len(ticket_history)}
        最近问题: {last_ticket['issue']}
        最近工单状态: {last_ticket['status']}
        """
    
    @before_model
    def add_customer_context(state: CustomerServiceState, runtime: Runtime) -> dict | None:
        """添加客户上下文"""
        customer_tier = state.get("customer_tier", "standard")
        ticket_count = len(state.get("ticket_history", []))
        
        tier_benefits = {
            "standard": "标准支持（24小时内响应）",
            "premium": "优先支持（4小时内响应）", 
            "vip": "专属支持（1小时内响应）"
        }
        
        context_msg = f"""
        当前客户等级: {customer_tier}
        支持级别: {tier_benefits.get(customer_tier, '标准支持')}
        历史工单数量: {ticket_count}
        """
        
        if state.get("last_issue"):
            context_msg += f"\n最近报告的问题: {state['last_issue']}"
        
        return {
            "messages": [SystemMessage(content=context_msg)] + state["messages"]
        }
    
    return create_agent(
        model="openai:gpt-4o",
        tools=[create_support_ticket, get_customer_history],
        middleware=[add_customer_context],
        state_schema=CustomerServiceState,
        checkpointer=InMemorySaver(),
    )

# 使用示例
service_agent = create_customer_service_agent()

# 第一次交互
result1 = service_agent.invoke(
    {
        "messages": [{"role": "user", "content": "我的账户无法登录"}],
        "customer_id": "cust_123",
        "customer_tier": "premium",
        "ticket_history": []
    },
    {"configurable": {"thread_id": "cust_123"}}
)

# 后续交互 - Agent 会记住客户历史
result2 = service_agent.invoke(
    {
        "messages": [{"role": "user", "content": "查看我的支持历史"}]
    },
    {"configurable": {"thread_id": "cust_123"}}
)
```

### 场景2：智能学习助手

```python
class LearningAssistantState(AgentState):
    student_level: str  # "beginner", "intermediate", "advanced"
    learning_topics: List[str]
    completed_lessons: List[Dict]
    weak_areas: List[str]
    learning_style: str  # "visual", "auditory", "kinesthetic"

def create_learning_assistant():
    """创建学习助手 Agent"""
    
    @tool
    def track_progress(runtime: ToolRuntime, topic: str, score: int) -> str:
        """跟踪学习进度"""
        from langgraph.types import Command
        
        lesson = {
            "topic": topic,
            "score": score,
            "completed_at": datetime.now().isoformat()
        }
        
        completed_lessons = runtime.state.get("completed_lessons", []) + [lesson]
        
        # 自动识别薄弱领域
        weak_areas = []
        if score < 70:
            weak_areas = list(set(runtime.state.get("weak_areas", []) + [topic]))
        
        return Command(update={
            "completed_lessons": completed_lessons,
            "weak_areas": weak_areas
        })
    
    @tool
    def get_study_recommendations(runtime: ToolRuntime) -> str:
        """获取学习建议"""
        state = runtime.state
        weak_areas = state.get("weak_areas", [])
        learning_style = state.get("learning_style", "visual")
        student_level = state.get("student_level", "beginner")
        
        recommendations = []
        
        if weak_areas:
            recommendations.append(f"需要加强的领域: {', '.join(weak_areas)}")
        
        style_suggestions = {
            "visual": "建议使用图表和视频学习",
            "auditory": "建议收听讲解和参与讨论", 
            "kinesthetic": "建议通过实践练习学习"
        }
        
        recommendations.append(style_suggestions.get(learning_style, "多种方式结合学习"))
        recommendations.append(f"适合{student_level}水平的学习材料")
        
        return "\n".join(recommendations)
    
    @dynamic_prompt
    def personalized_learning_prompt(request: ModelRequest) -> str:
        """个性化学习提示"""
        state = request.state
        context = request.runtime.context
        
        student_name = context.get("student_name", "同学")
        learning_style = state.get("learning_style", "visual")
        student_level = state.get("student_level", "beginner")
        
        return f"""
        你是一个耐心的学习助手，正在帮助{student_name}学习。
        
        学生信息：
        - 学习风格: {learning_style}
        - 当前水平: {student_level}
        - 已完成课程: {len(state.get('completed_lessons', []))}个
        
        请根据学生的学习风格和水平提供个性化的指导。
        对于{learning_style}型学习者，使用适合的教学方法。
        """
    
    return create_agent(
        model="openai:gpt-4o",
        tools=[track_progress, get_study_recommendations],
        middleware=[personalized_learning_prompt],
        state_schema=LearningAssistantState,
        checkpointer=InMemorySaver(),
    )

# 使用示例
learning_agent = create_learning_assistant()

# 初始化学习状态
learning_agent.invoke(
    {
        "messages": [{"role": "user", "content": "我想学习Python编程"}],
        "student_level": "beginner",
        "learning_style": "visual",
        "learning_topics": ["Python", "编程基础"],
        "completed_lessons": [],
        "weak_areas": []
    },
    {"configurable": {"thread_id": "student_123"}},
    context={"student_name": "小明"}
)
```

### 场景3：电商推荐系统

```python
class EcommerceState(AgentState):
    user_id: str
    browse_history: List[Dict]
    purchase_history: List[Dict]
    interests: List[str]
    budget_range: str
    preferred_categories: List[str]

def create_ecommerce_agent():
    """创建电商推荐 Agent"""
    
    @tool
    def track_browse_behavior(runtime: ToolRuntime, product: str, category: str) -> str:
        """跟踪浏览行为"""
        from langgraph.types import Command
        
        browse_record = {
            "product": product,
            "category": category,
            "timestamp": datetime.now().isoformat()
        }
        
        # 更新浏览历史和兴趣
        browse_history = runtime.state.get("browse_history", []) + [browse_record]
        interests = list(set(runtime.state.get("interests", []) + [category]))
        
        return Command(update={
            "browse_history": browse_history,
            "interests": interests
        })
    
    @tool
    def get_personalized_recommendations(runtime: ToolRuntime) -> str:
        """获取个性化推荐"""
        state = runtime.state
        interests = state.get("interests", [])
        budget_range = state.get("budget_range", "medium")
        preferred_categories = state.get("preferred_categories", [])
        
        # 基于用户行为生成推荐逻辑
        recommendations = []
        
        if interests:
            recommendations.append(f"基于您的兴趣推荐: {', '.join(interests[:3])} 相关商品")
        
        budget_map = {
            "low": "经济实惠型",
            "medium": "性价比型", 
            "high": "高端品质型"
        }
        
        recommendations.append(f"符合您{budget_map.get(budget_range, '中等')}预算的商品")
        
        return "\n".join(recommendations)
    
    @before_model
    def enhance_with_shopping_context(state: EcommerceState, runtime: Runtime) -> dict | None:
        """增强购物上下文"""
        interests = state.get("interests", [])
        purchase_count = len(state.get("purchase_history", []))
        browse_count = len(state.get("browse_history", []))
        
        context_msg = f"""
        购物助手上下文：
        - 用户兴趣: {', '.join(interests) if interests else '尚未确定'}
        - 浏览历史: {browse_count} 次
        - 购买记录: {purchase_count} 次
        - 预算范围: {state.get('budget_range', '未设置')}
        """
        
        return {
            "messages": [SystemMessage(content=context_msg)] + state["messages"]
        }
    
    return create_agent(
        model="openai:gpt-4o",
        tools=[track_browse_behavior, get_personalized_recommendations],
        middleware=[enhance_with_shopping_context],
        state_schema=EcommerceState,
        checkpointer=InMemorySaver(),
    )
```

## 最佳实践

### 1. 状态设计原则

```python
class WellDesignedState(AgentState):
    """良好设计的状态类示例"""
    
    # 必需的核心字段
    user_id: str
    
    # 会话相关字段
    session_start: str
    interaction_count: int = 0
    
    # 业务相关字段
    user_preferences: Dict[str, Any] = {}
    recent_actions: List[str] = []
    
    # 性能优化字段
    last_summary: Optional[str] = None
    tokens_used: int = 0
    
    def should_summarize(self) -> bool:
        """判断是否需要总结"""
        return self.interaction_count > 10 or self.tokens_used > 3000
```

### 2. 内存管理策略

```python
def create_memory_optimized_agent():
    """创建内存优化的 Agent"""
    
    @before_model
    def smart_memory_management(state: AgentState, runtime: Runtime) -> dict | None:
        """智能内存管理"""
        messages = state["messages"]
        
        # 基于不同条件采取不同策略
        if len(messages) > 20:
            # 消息过多时进行总结
            return {"messages": messages[-10:]}  # 保留最近10条
        
        elif state.get("tokens_used", 0) > 4000:
            # Token 使用过多时修剪
            return {"messages": messages[-8:]}
        
        return None
    
    @after_model
    def update_usage_metrics(state: AgentState, runtime: Runtime) -> dict | None:
        """更新使用指标"""
        # 估算 token 使用量（简化版）
        message_content = " ".join([msg.content for msg in state["messages"]])
        estimated_tokens = len(message_content) // 4
        
        return {
            "interaction_count": state.get("interaction_count", 0) + 1,
            "tokens_used": state.get("tokens_used", 0) + estimated_tokens
        }
    
    return create_agent(
        model="openai:gpt-4o",
        tools=[],
        middleware=[smart_memory_management, update_usage_metrics],
        state_schema=WellDesignedState,
        checkpointer=InMemorySaver(),
    )
```

### 3. 错误处理和恢复

```python
def create_robust_agent():
    """创建健壮的 Agent"""
    
    @after_model
    def handle_memory_errors(state: AgentState, runtime: Runtime) -> dict | None:
        """处理内存相关错误"""
        try:
            # 检查状态健康度
            messages = state["messages"]
            
            if len(messages) > 100:
                # 消息过多，自动清理
                return {"messages": messages[-20:]}
            
            return None
            
        except Exception as e:
            # 发生错误时恢复到最后已知良好状态
            print(f"内存处理错误: {e}")
            return None  # 保持当前状态
    
    return create_agent(
        model="openai:gpt-4o",
        tools=[],
        middleware=[handle_memory_errors],
        checkpointer=InMemorySaver(),
    )
```

## 总结

LangChain 的短期记忆系统提供了强大的对话状态管理能力：

- **灵活的状态设计**：支持自定义状态字段
- **多种存储后端**：内存、PostgreSQL 等
- **智能内存管理**：修剪、删除、总结等策略
- **全方位访问**：通过工具、中间件等访问和修改状态
- **生产级可靠性**：错误处理和性能优化

通过合理使用短期记忆，可以构建出真正理解用户上下文、提供个性化体验的智能应用。
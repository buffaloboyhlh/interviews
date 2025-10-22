# LangChain 长期记忆使用教程

## 概述

LangChain 智能体使用 [LangGraph 持久化](/oss/python/langgraph/persistence#memory-store) 来实现长期记忆。这是一个更高级的主题，需要了解 LangGraph 才能使用。

## 内存存储

LangGraph 将长期记忆作为 JSON 文档存储在 [store](/oss/python/langgraph/persistence#memory-store) 中。

每个记忆都在自定义的 `namespace`（类似于文件夹）和不同的 `key`（如文件名）下组织。命名空间通常包括用户或组织 ID 或其他标签，以便更容易地组织信息。

这种结构支持记忆的分层组织。然后通过内容过滤器支持跨命名空间的搜索。

### 基础存储操作

```python
from langgraph.store.memory import InMemoryStore

def embed(texts: list[str]) -> list[list[float]]:
    # 替换为实际的嵌入函数或 LangChain 嵌入对象
    return [[1.0, 2.0] * len(texts)]

# InMemoryStore 将数据保存到内存字典中。在生产环境中使用数据库支持的存储。
store = InMemoryStore(index={"embed": embed, "dims": 2})
user_id = "my-user"
application_context = "chitchat"
namespace = (user_id, application_context)

# 存储记忆
store.put(
    namespace,
    "a-memory",
    {
        "rules": [
            "用户喜欢简短、直接的语言",
            "用户只说英语和Python",
        ],
        "my-key": "my-value",
    },
)

# 通过 ID 获取"记忆"
item = store.get(namespace, "a-memory")

# 在此命名空间内搜索"记忆"，基于内容等价性过滤，按向量相似度排序
items = store.search(
    namespace, filter={"my-key": "my-value"}, query="语言偏好"
)
```

有关内存存储的更多信息，请参阅 [持久化](/oss/python/langgraph/persistence#memory-store) 指南。

## 在工具中读取长期记忆

```python
from dataclasses import dataclass
from langchain.agents import create_agent
from langchain.tools import tool, ToolRuntime
from langgraph.store.memory import InMemoryStore

@dataclass
class Context:
    user_id: str

# InMemoryStore 将数据保存到内存字典中。在生产环境中使用数据库支持的存储。
store = InMemoryStore()

# 使用 put 方法将示例数据写入存储
store.put(
    ("users",),  # 用于将相关数据分组在一起的命名空间（用户数据的用户命名空间）
    "user_123",  # 命名空间内的键（用户 ID 作为键）
    {
        "name": "张三",
        "language": "中文",
        "preferences": {
            "theme": "dark",
            "notifications": True
        }
    }  # 为给定用户存储的数据
)

@tool
def get_user_info(runtime: ToolRuntime[Context]) -> str:
    """查找用户信息。"""
    # 访问存储 - 与提供给 `create_agent` 的存储相同
    store = runtime.store
    user_id = runtime.context.user_id
    # 从存储中检索数据 - 返回带有值和元数据的 StoreValue 对象
    user_info = store.get(("users",), user_id)
    return str(user_info.value) if user_info else "未知用户"

agent = create_agent(
    model="anthropic:claude-sonnet-4-5",
    tools=[get_user_info],
    # 将存储传递给智能体 - 使智能体在运行工具时能够访问存储
    store=store,
    context_schema=Context
)

# 运行智能体
agent.invoke(
    {"messages": [{"role": "user", "content": "查找用户信息"}]},
    context=Context(user_id="user_123")
)
```

### 高级记忆读取示例

```python
from dataclasses import dataclass
from typing import List, Dict, Any
from langchain.agents import create_agent
from langchain.tools import tool, ToolRuntime
from langgraph.store.memory import InMemoryStore

@dataclass
class Context:
    user_id: str
    session_id: str

store = InMemoryStore()

# 初始化一些示例记忆数据
def initialize_sample_data():
    # 用户偏好
    store.put(("users", "preferences"), "user_123", {
        "language": "zh-CN",
        "timezone": "Asia/Shanghai",
        "communication_style": "concise",
        "topics_of_interest": ["AI", "编程", "科技"]
    })
    
    # 对话历史摘要
    store.put(("users", "conversations"), "user_123", {
        "recent_topics": ["Python编程", "机器学习", "LangChain使用"],
        "preferred_detail_level": "detailed",
        "conversation_count": 15
    })
    
    # 用户技能和知识
    store.put(("users", "skills"), "user_123", {
        "programming_languages": ["Python", "JavaScript"],
        "frameworks": ["Django", "React"],
        "experience_level": "intermediate"
    })

@tool
def get_user_preferences(runtime: ToolRuntime[Context]) -> str:
    """获取用户偏好设置。"""
    store = runtime.store
    user_id = runtime.context.user_id
    
    preferences = store.get(("users", "preferences"), user_id)
    if not preferences:
        return "未找到用户偏好设置"
    
    pref_data = preferences.value
    return f"""
用户偏好:
- 语言: {pref_data.get('language', '未设置')}
- 时区: {pref_data.get('timezone', '未设置')}
- 沟通风格: {pref_data.get('communication_style', '未设置')}
- 感兴趣的话题: {', '.join(pref_data.get('topics_of_interest', []))}
"""

@tool
def get_conversation_context(runtime: ToolRuntime[Context]) -> str:
    """获取对话上下文和历史。"""
    store = runtime.store
    user_id = runtime.context.user_id
    
    conv_data = store.get(("users", "conversations"), user_id)
    if not conv_data:
        return "未找到对话历史"
    
    data = conv_data.value
    return f"""
对话上下文:
- 最近话题: {', '.join(data.get('recent_topics', []))}
- 偏好详细程度: {data.get('preferred_detail_level', '未设置')}
- 对话次数: {data.get('conversation_count', 0)}
"""

@tool
def search_user_memories(runtime: ToolRuntime[Context], query: str) -> str:
    """基于查询搜索用户记忆。"""
    store = runtime.store
    user_id = runtime.context.user_id
    
    # 在所有用户相关的命名空间中搜索
    namespaces = [("users", "preferences"), ("users", "conversations"), ("users", "skills")]
    results = []
    
    for namespace in namespaces:
        items = store.search(namespace, query=query)
        for item in items:
            results.append(f"命名空间 {namespace}: {item.value}")
    
    if not results:
        return f"未找到与 '{query}' 相关的记忆"
    
    return "搜索结果:\n" + "\n".join(results)

# 初始化示例数据
initialize_sample_data()

# 创建智能体
agent = create_agent(
    model="anthropic:claude-sonnet-4-5",
    tools=[get_user_preferences, get_conversation_context, search_user_memories],
    store=store,
    context_schema=Context
)

# 使用示例
result = agent.invoke(
    {"messages": [{"role": "user", "content": "告诉我关于我的偏好和对话历史"}]},
    context=Context(user_id="user_123", session_id="session_456")
)
```

## 在工具中写入长期记忆

```python
from dataclasses import dataclass
from typing_extensions import TypedDict
from langchain.agents import create_agent
from langchain.tools import tool, ToolRuntime
from langgraph.store.memory import InMemoryStore

# InMemoryStore 将数据保存到内存字典中。在生产环境中使用数据库支持的存储。
store = InMemoryStore()

@dataclass
class Context:
    user_id: str

# TypedDict 为 LLM 定义用户信息的结构
class UserInfo(TypedDict):
    name: str

# 允许智能体更新用户信息的工具（对聊天应用程序有用）
@tool
def save_user_info(user_info: UserInfo, runtime: ToolRuntime[Context]) -> str:
    """保存用户信息。"""
    # 访问存储 - 与提供给 `create_agent` 的存储相同
    store = runtime.store
    user_id = runtime.context.user_id
    # 将数据存储在存储中（命名空间，键，数据）
    store.put(("users",), user_id, user_info)
    return "成功保存用户信息。"

agent = create_agent(
    model="anthropic:claude-sonnet-4-5",
    tools=[save_user_info],
    store=store,
    context_schema=Context
)

# 运行智能体
agent.invoke(
    {"messages": [{"role": "user", "content": "我的名字是张三"}]},
    # user_id 在上下文中传递以标识正在更新谁的信息
    context=Context(user_id="user_123")
)

# 你可以直接访问存储来获取值
print(store.get(("users",), "user_123").value)
```

### 高级记忆写入示例

```python
from dataclasses import dataclass
from typing import Dict, Any, List
from datetime import datetime
import json
from langchain.agents import create_agent
from langchain.tools import tool, ToolRuntime
from langgraph.store.memory import InMemoryStore

@dataclass
class Context:
    user_id: str
    session_id: str

store = InMemoryStore()

@tool
def save_user_preference(runtime: ToolRuntime[Context], preference_type: str, value: str) -> str:
    """保存用户偏好设置。"""
    store = runtime.store
    user_id = runtime.context.user_id
    
    # 获取现有的偏好设置
    existing_prefs = store.get(("users", "preferences"), user_id)
    prefs_data = existing_prefs.value if existing_prefs else {}
    
    # 更新偏好设置
    prefs_data[preference_type] = value
    prefs_data["last_updated"] = datetime.now().isoformat()
    
    # 保存回存储
    store.put(("users", "preferences"), user_id, prefs_data)
    return f"成功保存偏好设置: {preference_type} = {value}"

@tool
def record_conversation_topic(runtime: ToolRuntime[Context], topic: str) -> str:
    """记录对话话题以供将来参考。"""
    store = runtime.store
    user_id = runtime.context.user_id
    
    # 获取现有的对话历史
    existing_conv = store.get(("users", "conversations"), user_id)
    conv_data = existing_conv.value if existing_conv else {
        "recent_topics": [],
        "conversation_count": 0,
        "first_interaction": datetime.now().isoformat()
    }
    
    # 更新话题列表（保持最近10个话题）
    recent_topics = conv_data.get("recent_topics", [])
    if topic not in recent_topics:
        recent_topics.insert(0, topic)
        recent_topics = recent_topics[:10]  # 只保留最近10个话题
    
    # 更新对话计数
    conv_data["recent_topics"] = recent_topics
    conv_data["conversation_count"] = conv_data.get("conversation_count", 0) + 1
    conv_data["last_interaction"] = datetime.now().isoformat()
    
    # 保存回存储
    store.put(("users", "conversations"), user_id, conv_data)
    return f"已记录话题: {topic}"

@tool
def save_learning_progress(runtime: ToolRuntime[Context], topic: str, progress: str, notes: str = "") -> str:
    """保存用户的学习进度和笔记。"""
    store = runtime.store
    user_id = runtime.context.user_id
    
    # 获取现有的学习记录
    existing_learning = store.get(("users", "learning"), user_id)
    learning_data = existing_learning.value if existing_learning else {"topics": {}}
    
    # 更新特定话题的进度
    learning_data["topics"][topic] = {
        "progress": progress,
        "notes": notes,
        "last_updated": datetime.now().isoformat()
    }
    learning_data["overall_last_updated"] = datetime.now().isoformat()
    
    # 保存回存储
    store.put(("users", "learning"), user_id, learning_data)
    return f"已保存 {topic} 的学习进度: {progress}"

@tool
def save_feedback(runtime: ToolRuntime[Context], feedback: str, rating: int) -> str:
    """保存用户反馈和评分。"""
    store = runtime.store
    user_id = runtime.context.user_id
    session_id = runtime.context.session_id
    
    # 获取现有的反馈
    existing_feedback = store.get(("users", "feedback"), user_id)
    feedback_data = existing_feedback.value if existing_feedback else {"feedbacks": []}
    
    # 添加新反馈
    new_feedback = {
        "session_id": session_id,
        "feedback": feedback,
        "rating": rating,
        "timestamp": datetime.now().isoformat()
    }
    feedback_data["feedbacks"].append(new_feedback)
    
    # 保存回存储
    store.put(("users", "feedback"), user_id, feedback_data)
    return f"感谢您的反馈！评分: {rating}/5"

# 创建智能体
agent = create_agent(
    model="anthropic:claude-sonnet-4-5",
    tools=[save_user_preference, record_conversation_topic, save_learning_progress, save_feedback],
    store=store,
    context_schema=Context
)

# 使用示例
result = agent.invoke(
    {"messages": [{"role": "user", "content": "我喜欢详细的技术解释，请记录下来"}]},
    context=Context(user_id="user_123", session_id="session_456")
)
```

## 完整的长期记忆系统

```python
from dataclasses import dataclass
from typing import Dict, Any, List, Optional
from datetime import datetime
import json
from langchain.agents import create_agent
from langchain.tools import tool, ToolRuntime
from langchain.agents.middleware import dynamic_prompt
from langgraph.store.memory import InMemoryStore

@dataclass
class UserContext:
    user_id: str
    session_id: str
    user_name: Optional[str] = None

class LongTermMemorySystem:
    def __init__(self):
        self.store = InMemoryStore()
    
    def initialize_user(self, user_id: str, initial_data: Dict[str, Any] = None):
        """初始化新用户的记忆存储"""
        if initial_data is None:
            initial_data = {}
        
        base_data = {
            "created_at": datetime.now().isoformat(),
            "last_active": datetime.now().isoformat(),
            "conversation_count": 0,
            **initial_data
        }
        
        self.store.put(("users", "profile"), user_id, base_data)
        self.store.put(("users", "preferences"), user_id, {})
        self.store.put(("users", "conversations"), user_id, {"recent_topics": []})
        self.store.put(("users", "learning"), user_id, {"topics": {}})
    
    def get_user_summary(self, user_id: str) -> str:
        """获取用户记忆摘要"""
        profile = self.store.get(("users", "profile"), user_id)
        preferences = self.store.get(("users", "preferences"), user_id)
        conversations = self.store.get(("users", "conversations"), user_id)
        
        if not profile:
            return f"用户 {user_id} 未初始化"
        
        profile_data = profile.value
        prefs_data = preferences.value if preferences else {}
        conv_data = conversations.value if conversations else {}
        
        return f"""
用户记忆摘要:
- 用户ID: {user_id}
- 创建时间: {profile_data.get('created_at', '未知')}
- 最后活跃: {profile_data.get('last_active', '未知')}
- 对话次数: {profile_data.get('conversation_count', 0)}
- 偏好设置: {len(prefs_data)} 项
- 最近话题: {len(conv_data.get('recent_topics', []))} 个
"""

# 创建记忆系统实例
memory_system = LongTermMemorySystem()

# 动态提示，基于用户记忆个性化
@dynamic_prompt
def personalized_prompt(request) -> str:
    """基于用户记忆的个性化提示"""
    user_id = request.runtime.context.user_id
    
    # 获取用户偏好
    preferences = request.runtime.store.get(("users", "preferences"), user_id)
    prefs_data = preferences.value if preferences else {}
    
    # 构建个性化提示
    base_prompt = "你是一个有用的AI助手。"
    
    # 添加个性化元素
    if prefs_data.get("communication_style"):
        style = prefs_data["communication_style"]
        if style == "concise":
            base_prompt += " 请提供简洁、直接的回答。"
        elif style == "detailed":
            base_prompt += " 请提供详细、全面的解释。"
        elif style == "friendly":
            base_prompt += " 请使用友好、热情的语气。"
    
    if prefs_data.get("language") == "zh-CN":
        base_prompt += " 请使用中文进行交流。"
    
    return base_prompt

@tool
def update_user_profile(runtime: ToolRuntime[UserContext], updates: Dict[str, Any]) -> str:
    """更新用户个人资料信息。"""
    store = runtime.store
    user_id = runtime.context.user_id
    
    # 获取现有个人资料
    existing_profile = store.get(("users", "profile"), user_id)
    profile_data = existing_profile.value if existing_profile else {}
    
    # 应用更新
    profile_data.update(updates)
    profile_data["last_updated"] = datetime.now().isoformat()
    
    # 保存回存储
    store.put(("users", "profile"), user_id, profile_data)
    
    # 如果提供了用户名，更新上下文
    if "name" in updates and runtime.context.user_name is None:
        runtime.context.user_name = updates["name"]
    
    return "个人资料已更新"

@tool
def get_memory_summary(runtime: ToolRuntime[UserContext]) -> str:
    """获取用户记忆系统的完整摘要。"""
    return memory_system.get_user_summary(runtime.context.user_id)

@tool
def search_memories(runtime: ToolRuntime[UserContext], query: str, namespace: str = "all") -> str:
    """在用户记忆中搜索特定信息。"""
    store = runtime.store
    user_id = runtime.context.user_id
    
    namespaces_to_search = []
    if namespace == "all":
        namespaces_to_search = [("users", "profile"), ("users", "preferences"), 
                               ("users", "conversations"), ("users", "learning")]
    else:
        namespaces_to_search = [("users", namespace)]
    
    results = []
    for ns in namespaces_to_search:
        items = store.search(ns, query=query)
        for item in items:
            # 格式化结果
            if isinstance(item.value, dict):
                formatted_value = json.dumps(item.value, ensure_ascii=False, indent=2)
            else:
                formatted_value = str(item.value)
            
            results.append(f"=== {ns} ===\n{formatted_value}")
    
    if not results:
        return f"在记忆中未找到与 '{query}' 相关的内容"
    
    return "\n\n".join(results)

# 初始化示例用户
memory_system.initialize_user("user_123", {"name": "张三", "initial_setup": True})

# 创建完整的记忆增强智能体
agent = create_agent(
    model="anthropic:claude-sonnet-4-5",
    tools=[update_user_profile, get_memory_summary, search_memories],
    middleware=[personalized_prompt],
    store=memory_system.store,
    context_schema=UserContext
)

# 使用示例
def demonstrate_memory_system():
    print("=== 长期记忆系统演示 ===")
    
    # 第一次交互
    result1 = agent.invoke(
        {"messages": [{"role": "user", "content": "请更新我的个人资料，设置我的沟通风格为详细"}]},
        context=UserContext(user_id="user_123", session_id="session_1")
    )
    print("第一次交互结果:", result1['messages'][-1].content)
    
    # 第二次交互 - 系统会记住偏好
    result2 = agent.invoke(
        {"messages": [{"role": "user", "content": "告诉我关于我的记忆系统的信息"}]},
        context=UserContext(user_id="user_123", session_id="session_2")
    )
    print("第二次交互结果:", result2['messages'][-1].content)
    
    # 搜索记忆
    result3 = agent.invoke(
        {"messages": [{"role": "user", "content": "搜索我的个人资料信息"}]},
        context=UserContext(user_id="user_123", session_id="session_3")
    )
    print("搜索结果:", result3['messages'][-1].content)

# 运行演示
demonstrate_memory_system()
```

## 生产环境建议

### 1. 使用持久化存储

```python
# 在生产环境中，使用数据库支持的存储而不是 InMemoryStore
from langgraph.store.postgres import PostgresStore
import os

# 配置 PostgreSQL 存储
def create_production_store():
    return PostgresStore(
        connection_string=os.getenv("DATABASE_URL"),
        # 其他配置选项...
    )
```

### 2. 记忆清理策略

```python
from datetime import datetime, timedelta

class MemoryManager:
    def __init__(self, store):
        self.store = store
    
    def cleanup_old_memories(self, user_id: str, days_old: int = 30):
        """清理旧的用户记忆"""
        cutoff_date = datetime.now() - timedelta(days=days_old)
        
        # 这里可以实现具体的清理逻辑
        # 例如删除过时的对话记录、学习进度等
        pass
    
    def compress_conversation_history(self, user_id: str):
        """压缩对话历史以减少存储空间"""
        conversations = self.store.get(("users", "conversations"), user_id)
        if conversations:
            conv_data = conversations.value
            # 保留最近的话题，归档旧话题
            if "recent_topics" in conv_data:
                conv_data["recent_topics"] = conv_data["recent_topics"][:20]  # 只保留最近20个
                self.store.put(("users", "conversations"), user_id, conv_data)
```

### 3. 记忆备份和恢复

```python
import json

class MemoryBackup:
    def __init__(self, store):
        self.store = store
    
    def export_user_memories(self, user_id: str) -> Dict[str, Any]:
        """导出用户的所有记忆"""
        namespaces = ["profile", "preferences", "conversations", "learning", "feedback"]
        memories = {}
        
        for namespace in namespaces:
            memory = self.store.get(("users", namespace), user_id)
            if memory:
                memories[namespace] = memory.value
        
        return memories
    
    def import_user_memories(self, user_id: str, memories: Dict[str, Any]):
        """导入用户记忆"""
        for namespace, data in memories.items():
            self.store.put(("users", namespace), user_id, data)
```

## 总结

长期记忆系统为 LangChain 智能体提供了持久化记忆能力，使它们能够：

1. **记住用户偏好** - 沟通风格、语言偏好、兴趣话题等
2. **维护对话上下文** - 跟踪对话历史、话题趋势
3. **记录学习进度** - 保存用户的学习轨迹和成就
4. **个性化交互** - 基于历史交互提供更相关的响应

**关键最佳实践**：
- 在生产环境中使用数据库支持的存储
- 实现记忆清理和压缩策略
- 设计合理的命名空间结构
- 提供记忆备份和恢复机制
- 尊重用户隐私，提供记忆管理选项

通过合理实现长期记忆系统，你可以创建真正个性化、上下文感知的 AI 应用，为用户提供持续且一致的体验。
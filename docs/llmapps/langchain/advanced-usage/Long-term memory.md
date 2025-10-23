# 🧠 LangChain 长期记忆（Long-term Memory）教程

## 一、概念简介

在 LangChain 框架中，**长期记忆（Long-term Memory）** 通过 [LangGraph 持久化机制（Persistence）](https://python.langchain.com/oss/langgraph/persistence#memory-store) 实现。
它允许智能体（Agent）在多个会话之间保存和检索上下文信息，而不仅仅依赖于短期对话上下文。

换句话说，这让你的 Agent 拥有“记忆力”：
它能**记住用户的偏好、历史操作或资料**，并在未来的对话中引用这些内容。

---

## 二、记忆的存储结构

LangGraph 使用类似文件系统的结构来组织记忆数据。
每一条记忆由两个关键部分组成：

* **namespace（命名空间）**：相当于文件夹，用于分组。例如可以用用户ID、应用场景区分。
* **key（键）**：类似文件名，用于唯一标识某条具体记忆。

每条记忆的内容以 JSON 文档形式保存。
示例结构如下：

```python
from langgraph.store.memory import InMemoryStore

def embed(texts: list[str]) -> list[list[float]]:
    # 实际使用时应替换为真实嵌入函数
    return [[1.0, 2.0] * len(texts)]

# 创建一个内存型存储（开发阶段使用，生产应换成数据库）
store = InMemoryStore(index={"embed": embed, "dims": 2})

user_id = "my-user"
app_context = "chitchat"
namespace = (user_id, app_context)

# 写入一条记忆
store.put(
    namespace,
    "a-memory",
    {
        "rules": [
            "User likes short, direct language",
            "User only speaks English & Python",
        ],
        "my-key": "my-value",
    },
)

# 根据 key 获取记忆
item = store.get(namespace, "a-memory")

# 在命名空间内搜索记忆（根据内容过滤并按向量相似度排序）
items = store.search(
    namespace, filter={"my-key": "my-value"}, query="language preferences"
)
```

在生产环境中，你可以将 `InMemoryStore` 替换为数据库后端，以持久化存储记忆。

---

## 三、在工具中读取长期记忆

在 LangChain 中，Agent 的工具（Tool）可以直接访问长期记忆，用于在执行任务时查找用户信息。
下面的例子展示了一个可以读取用户资料的工具：

```python
from dataclasses import dataclass
from langchain.agents import create_agent
from langchain.tools import tool, ToolRuntime
from langgraph.store.memory import InMemoryStore

@dataclass
class Context:
    user_id: str

store = InMemoryStore()

# 预先写入一条示例数据
store.put(
    ("users",), 
    "user_123", 
    {"name": "John Smith", "language": "English"}
)

@tool
def get_user_info(runtime: ToolRuntime[Context]) -> str:
    """查找用户信息"""
    store = runtime.store
    user_id = runtime.context.user_id
    user_info = store.get(("users",), user_id)
    return str(user_info.value) if user_info else "Unknown user"

agent = create_agent(
    model="anthropic:claude-sonnet-4-5",
    tools=[get_user_info],
    store=store,
    context_schema=Context
)

# 运行 Agent
agent.invoke(
    {"messages": [{"role": "user", "content": "look up user information"}]},
    context=Context(user_id="user_123")
)
```

运行后，Agent 会通过工具从记忆存储中读取用户资料。

---

## 四、在工具中写入长期记忆

除了读取外，Agent 也可以通过工具**动态写入记忆**，这让它能在对话过程中更新用户信息。

下面是一个保存用户信息的示例：

```python
from dataclasses import dataclass
from typing_extensions import TypedDict
from langchain.agents import create_agent
from langchain.tools import tool, ToolRuntime
from langgraph.store.memory import InMemoryStore

store = InMemoryStore()

@dataclass
class Context:
    user_id: str

class UserInfo(TypedDict):
    name: str

@tool
def save_user_info(user_info: UserInfo, runtime: ToolRuntime[Context]) -> str:
    """保存用户信息"""
    store = runtime.store
    user_id = runtime.context.user_id
    store.put(("users",), user_id, user_info)
    return "Successfully saved user info."

agent = create_agent(
    model="anthropic:claude-sonnet-4-5",
    tools=[save_user_info],
    store=store,
    context_schema=Context
)

# 运行 Agent，动态更新记忆
agent.invoke(
    {"messages": [{"role": "user", "content": "My name is John Smith"}]},
    context=Context(user_id="user_123")
)

# 验证是否写入成功
store.get(("users",), "user_123").value
```

结果中可以看到，用户数据被成功存储到 `store` 中。

---

## 五、进阶：使用数据库或云端存储

`InMemoryStore` 仅适合开发测试阶段。
在生产环境中，你应使用持久化后端，如：

* SQLite、PostgreSQL、MongoDB
* 云服务（如 AWS DynamoDB、Redis、FireStore 等）

LangGraph 的持久化接口是统一的，这意味着你可以替换底层存储，而不影响上层逻辑。

---

## 六、总结

长期记忆让 LangChain Agent 拥有更“人性化”的上下文意识。
通过它，Agent 能够：

* 记住用户的历史与偏好
* 跨会话检索并使用先前信息
* 动态更新知识、个性与行为模式

在构建**个性化助理、对话机器人或持续学习型智能体**时，长期记忆是不可或缺的能力。

---

## 延伸阅读

* [LangGraph Persistence 文档](https://python.langchain.com/oss/langgraph/persistence#memory-store)
* [LangChain Agents 教程](https://python.langchain.com/docs/modules/agents/)
* [MCP 协议集成方式](/use-these-docs)

---

如果你希望我帮你把这份教程改写成 **教学视频脚本** 或 **企业级部署版本（使用 PostgreSQL 作为后端）**，我可以继续扩展。你想我往哪个方向延伸？

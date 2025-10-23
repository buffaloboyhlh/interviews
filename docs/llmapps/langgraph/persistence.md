# 🧠 LangGraph 持久化机制教程（Persistence Tutorial）

LangGraph 是 LangChain 团队推出的一个用于可视化与可编程化语言模型工作流的框架。
它的一大核心能力——**持久化（Persistence）**，通过“**检查点（Checkpoint）**”与“**线程（Thread）**”系统，实现了状态的保存、恢复、分支、回放与共享。

这使得你能像玩时间机器一样在对话与任务流中穿梭：保存过去、编辑现在、分叉未来。

---

## 一、持久化的核心概念

LangGraph 内置一个“检查点系统（Checkpointer）”，每当图（Graph）运行一个“超级步骤（super-step）”，它就自动保存当前状态（State）的快照，称为 **Checkpoint**。
这些检查点属于某个“线程（Thread）”，每个线程就像一次独立的执行会话或对话历史。

得益于这种机制，LangGraph 能够实现：

* **人类介入（Human-in-the-loop）**：随时查看与修改状态；
* **记忆（Memory）**：保存长期上下文；
* **时间旅行（Time Travel）**：回放任意历史状态；
* **容错（Fault-tolerance）**：任务中断可恢复。

> ✅ 提示：使用 LangGraph API 时，这一切都自动完成，无需手动管理。

---

## 二、线程（Threads）

线程是持久化状态的载体。
每当你执行一个带有 checkpointer 的图时，必须指定一个唯一的 `thread_id`：

```python
config = {"configurable": {"thread_id": "1"}}
```

这个线程会保存整个执行过程中产生的所有检查点（checkpoints）。
稍后我们可以通过 `thread_id` 来访问：

* 最新状态；
* 历史状态；
* 任意时间点的快照；
* 从任意检查点继续执行。

---

## 三、检查点（Checkpoints）

每个 **Checkpoint** 就是一张状态快照，包含以下核心信息：

* `values`: 当前图中各通道（channel）的状态值；
* `config`: 运行时配置；
* `metadata`: 元数据（如执行节点、错误、步骤编号等）；
* `next`: 下一个待执行节点；
* `tasks`: 当前任务信息（可含错误、暂停、或中断数据）。

这些快照被连续保存，就构成了完整的线程执行历史。

---

### 示例：创建并运行一个简单图

```python
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import InMemorySaver
from typing import Annotated
from typing_extensions import TypedDict
from operator import add

class State(TypedDict):
    foo: str
    bar: Annotated[list[str], add]

def node_a(state: State):
    return {"foo": "a", "bar": ["a"]}

def node_b(state: State):
    return {"foo": "b", "bar": ["b"]}

# 定义图结构
workflow = StateGraph(State)
workflow.add_node(node_a)
workflow.add_node(node_b)
workflow.add_edge(START, "node_a")
workflow.add_edge("node_a", "node_b")
workflow.add_edge("node_b", END)

# 添加内存型 checkpointer
checkpointer = InMemorySaver()
graph = workflow.compile(checkpointer=checkpointer)

# 指定线程并执行
config = {"configurable": {"thread_id": "1"}}
graph.invoke({"foo": ""}, config)
```

执行后，LangGraph 自动生成 4 个检查点，分别对应：

1. 初始状态（待执行节点：START）
2. 输入加载（待执行节点：node_a）
3. node_a 执行后（待执行节点：node_b）
4. node_b 执行后（执行结束）

---

## 四、读取状态

### 1. 获取最新状态

```python
graph.get_state({"configurable": {"thread_id": "1"}})
```

返回值是一个 `StateSnapshot` 对象。

### 2. 获取特定检查点的状态

```python
config = {"configurable": {
    "thread_id": "1",
    "checkpoint_id": "1ef663ba-28fe-6528-8002-5a559208592c"
}}
graph.get_state(config)
```

### 3. 获取整个状态历史

```python
config = {"configurable": {"thread_id": "1"}}
history = list(graph.get_state_history(config))
```

返回一个时间倒序排列的 `StateSnapshot` 列表。

---

## 五、重放（Replay）

“重放”允许你从任意历史状态重新运行图。
这相当于“时间旅行”到一个旧状态，然后从那里创建新的分支。

```python
config = {
    "configurable": {
        "thread_id": "1",
        "checkpoint_id": "0c62ca34-ac19-445d-bbb0-5b4984975b2a"
    }
}
graph.invoke(None, config=config)
```

LangGraph 会自动识别哪些步骤已执行过，只“重放”这些步骤，而不是重新计算。之后的步骤则会被真正执行（相当于时间线分叉）。

---

## 六、编辑状态（Update State）

`update_state()` 方法允许你直接修改状态，甚至“伪造”节点输出。

```python
graph.update_state(config, {"foo": 2, "bar": ["b"]})
```

如果 `bar` 有 reducer（如 `add`），则会合并：

```
原状态: {"foo": 1, "bar": ["a"]}
更新后: {"foo": 2, "bar": ["a", "b"]}
```

如果要模拟节点执行，可使用 `as_node` 参数：

```python
graph.update_state(config, {"foo": 3}, as_node="node_b")
```

---

## 七、跨线程共享记忆（Memory Store）

Checkpointer 保存状态在**单个线程内**，而有时我们希望在**不同线程之间共享记忆**（如同一个用户的多轮对话）。
这就需要 **Store（存储）** 接口。

LangGraph 提供 `InMemoryStore`，用于跨线程存储与检索信息。

```python
from langgraph.store.memory import InMemoryStore
store = InMemoryStore()
```

### 存储用户记忆

```python
import uuid
user_id = "1"
namespace = (user_id, "memories")
memory_id = str(uuid.uuid4())
memory = {"food_preference": "I like pizza"}

store.put(namespace, memory_id, memory)
```

### 检索记忆

```python
memories = store.search(namespace)
print(memories[-1].dict())
```

返回的对象包含：

* `value`: 实际内容；
* `namespace`: 命名空间；
* `created_at` / `updated_at`: 时间戳。

---

## 八、语义检索（Semantic Search）

Store 不仅能做关键词检索，还能进行**语义匹配**。

启用方法：

```python
from langchain.embeddings import init_embeddings

store = InMemoryStore(
    index={
        "embed": init_embeddings("openai:text-embedding-3-small"),
        "dims": 1536,
        "fields": ["food_preference", "$"]
    }
)
```

查询：

```python
store.search(namespace, query="用户喜欢吃什么？", limit=3)
```

---

## 九、在 LangGraph 中集成 Store

你可以同时在编译时传入 checkpointer 与 store：

```python
from langgraph.checkpoint.memory import InMemorySaver

checkpointer = InMemorySaver()
graph = graph.compile(checkpointer=checkpointer, store=store)
```

在节点中即可使用 Store：

```python
def update_memory(state, config, *, store):
    user_id = config["configurable"]["user_id"]
    namespace = (user_id, "memories")
    store.put(namespace, str(uuid.uuid4()), {"memory": "User likes pizza"})
```

多个线程（thread_id 不同）共享同一 user_id 即可访问同一记忆。

---

## 十、Checkpointer 实现库

LangGraph 的持久化底层由多个可选的 Checkpointer 库实现：

| 库                               | 存储类型       | 适用场景      |
| ------------------------------- | ---------- | --------- |
| `langgraph-checkpoint`          | 内存         | 默认内置，快速实验 |
| `langgraph-checkpoint-sqlite`   | SQLite     | 本地持久化     |
| `langgraph-checkpoint-postgres` | PostgreSQL | 生产环境级持久化  |

所有实现都符合 `BaseCheckpointSaver` 接口。

---

## 🎯 总结

LangGraph 的持久化系统让工作流不再是“一次性”的。
通过 **Checkpointer + Thread + Store** 的三层架构，它让你的 AI 系统拥有：

* 可回溯的状态；
* 可持久化的上下文；
* 可共享的长期记忆；
* 可重放与分叉的执行路径。

换句话说，LangGraph 让“AI 程序的时间”成为一种可编程资源。

---

如果你希望我继续补充下一篇教程，比如
👉「LangGraph 时间旅行 (Time Travel) 实战」
或
👉「LangGraph 中的 Store 与语义检索实战指南」
我可以直接在此基础上续写。
你想我往哪个方向展开？

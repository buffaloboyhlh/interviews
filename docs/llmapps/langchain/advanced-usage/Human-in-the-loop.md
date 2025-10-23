# 🧭 Human-in-the-Loop（人类在环）教程

## 一、概念简介

在自动化智能系统中，我们希望 AI 具备强大的自主决策能力，但在一些关键操作上（例如删除数据库记录、写入文件、发送邮件），**完全交由模型自动执行是不安全的**。
这时就需要 **Human-in-the-Loop（HITL）** 机制。

**HITL 中间件** 允许你在 Agent 执行工具调用（Tool Call）之前，插入一个人工审查环节。
当模型计划执行高风险或敏感操作时，它会**暂停执行**，等待人工批准、修改或拒绝。

换句话说，HITL 是 AI 系统的**“安全阀”** —— 它让你决定何时让人类介入、如何恢复执行。

---

## 二、HITL 的工作原理

1. **模型生成计划**：Agent 输出一个工具调用请求（例如执行 SQL）。
2. **中间件检查**：HITL 中间件根据预先设定的策略检查这个调用。
3. **触发中断**：若该操作需要人工干预，中间件会发出一个 *interrupt（中断）* 信号，暂停执行。
4. **保存状态**：当前对话状态通过 LangGraph 的持久化层（Persistence Layer）保存。
5. **人工审查**：人类可以选择：

   * ✅ `approve`（批准执行）
   * ✏️ `edit`（修改执行参数）
   * ❌ `reject`（拒绝并反馈原因）
6. **恢复执行**：系统根据人工决策继续运行或放弃该操作。

---

## 三、三种人工决策类型

| 决策类型          | 说明       | 示例          |
| ------------- | -------- | ----------- |
| ✅ **approve** | 批准操作并执行  | 发送邮件草稿原样发送  |
| ✏️ **edit**   | 修改参数后再执行 | 改动邮件收件人后再发送 |
| ❌ **reject**  | 拒绝执行并反馈  | 拒绝草稿并说明改进建议 |

> 提示：
> 修改工具参数（`edit`）时建议小幅调整。过大改动可能导致模型重新规划，引发多次执行或不可预期行为。

---

## 四、HITL 配置示例

在创建 Agent 时，将 `HumanInTheLoopMiddleware` 添加到 `middleware` 列表中即可启用 HITL。
通过 `interrupt_on` 字典来定义哪些工具需要人工审批，以及允许的决策类型。

```python
from langchain.agents import create_agent
from langchain.agents.middleware import HumanInTheLoopMiddleware
from langgraph.checkpoint.memory import InMemorySaver

agent = create_agent(
    model="openai:gpt-4o",
    tools=[write_file_tool, execute_sql_tool, read_data_tool],
    middleware=[
        HumanInTheLoopMiddleware(
            interrupt_on={
                "write_file": True,  # 允许approve/edit/reject三种操作
                "execute_sql": {"allowed_decisions": ["approve", "reject"]},  # 禁止edit修改
                "read_data": False,  # 安全操作，不需要人工审批
            },
            description_prefix="工具调用等待人工审批",
        ),
    ],
    # HITL 依赖检查点机制来在中断后恢复状态
    checkpointer=InMemorySaver(),  # 测试用内存保存器
)
```

> ⚙️ 在生产环境中，请使用持久化的保存器（如 `AsyncPostgresSaver`）以保证在服务重启后仍能恢复执行状态。
> 详见 LangGraph [Checkpoint 文档](https://reference.langchain.com/python/langgraph/checkpoints/#langgraph.checkpoint.postgres.aio.AsyncPostgresSaver)。

---

## 五、触发中断与人工审查流程

当你运行 Agent 时，它会一直执行，直到遇到需要人工干预的工具调用。
此时，返回结果会包含一个 `__interrupt__` 字段，其中列出所有待审查的操作。

### 示例

```python
from langgraph.types import Command

config = {"configurable": {"thread_id": "user_session_001"}}

# 调用触发潜在危险操作
result = agent.invoke(
    {
        "messages": [
            {"role": "user", "content": "Delete old records from the database"}
        ]
    },
    config=config
)

print(result['__interrupt__'])
```

输出中断请求，内容类似：

```python
[
  Interrupt(
    value={
      "action_requests": [
        {
          "name": "execute_sql",
          "arguments": {"query": "DELETE FROM records WHERE created_at < NOW() - INTERVAL '30 days';"},
          "description": "工具调用等待人工审批\n\nTool: execute_sql\nArgs: {...}"
        }
      ],
      "review_configs": [
        {
          "action_name": "execute_sql",
          "allowed_decisions": ["approve", "reject"]
        }
      ]
    }
  )
]
```

---

## 六、提供人工决策（恢复执行）

执行被中断后，你可以使用 `Command(resume=...)` 提供人工决策并恢复流程。

### ✅ 批准执行

```python
agent.invoke(
    Command(
        resume={
            "decisions": [{"type": "approve"}]
        }
    ),
    config=config
)
```

---

### ✏️ 修改参数后执行

```python
agent.invoke(
    Command(
        resume={
            "decisions": [
                {
                    "type": "edit",
                    "edited_action": {
                        "name": "execute_sql",
                        "args": {"query": "DELETE FROM records WHERE created_at < NOW() - INTERVAL '60 days';"}
                    }
                }
            ]
        }
    ),
    config=config
)
```

---

### ❌ 拒绝执行并提供反馈

```python
agent.invoke(
    Command(
        resume={
            "decisions": [
                {
                    "type": "reject",
                    "message": "不允许删除数据，请仅归档旧记录。"
                }
            ]
        }
    ),
    config=config
)
```

---

### 多个操作同时中断时

必须按中断请求中操作出现的顺序依次提供决策：

```python
{
  "decisions": [
    {"type": "approve"},
    {
      "type": "edit",
      "edited_action": {"name": "tool_name", "args": {"param": "new_value"}}
    },
    {
      "type": "reject",
      "message": "该操作未获批准"
    }
  ]
}
```

---

## 七、HITL 执行生命周期详解

1. Agent 向模型请求响应。
2. 模型返回包含工具调用的响应。
3. HITL 中间件在 `after_model` 阶段检查这些调用。
4. 若发现需要审查的操作，构建 `HITLRequest` 并触发 `interrupt`。
5. Agent 暂停，等待人工审查输入。
6. 根据人工决策：

   * 执行批准或修改后的工具调用；
   * 对拒绝的调用生成反馈消息；
   * 恢复继续执行模型推理。

---

## 八、自定义 HITL 逻辑

如果你希望实现更复杂的人机交互流程（例如多级审批或条件触发），可以直接使用：

* `interrupt` 原语（LangGraph 提供）
* 自定义中间件（继承 `BaseMiddleware`）

可参考 [LangChain Middleware 文档](https://reference.langchain.com/python/langchain/middleware) 了解扩展方式。

---

## 九、总结

**Human-in-the-Loop（HITL）** 是让 AI 具备“可控自治”的关键机制。
它带来的好处包括：

* ✅ **安全性**：避免模型自动执行危险操作
* 🧠 **可解释性**：人工审查过程提供决策依据
* 🔄 **可恢复性**：通过 checkpoint 能安全暂停与继续执行

未来，HITL 将成为企业级智能代理系统的标准组成部分，使 AI 既能自主执行，又不失人类掌控。

---

是否希望我继续写一篇《实战篇：用 HITL 审查 SQL + 文件写入操作》的进阶教程？我可以展示完整代码与中断处理流程。

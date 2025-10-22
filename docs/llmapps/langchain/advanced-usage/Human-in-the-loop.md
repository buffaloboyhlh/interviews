# Human-in-the-Loop (HITL) 中间件使用教程

## 概述

Human-in-the-Loop (HITL) 中间件允许你在智能体工具调用中添加人工监督。当模型提出可能需要审查的操作时（例如写入文件或执行 SQL），中间件可以暂停执行并等待决策。

## 工作原理

HITL 中间件通过检查每个工具调用是否符合可配置的策略来工作。如果需要干预，中间件会发出[中断](https://reference.langchain.com/python/langgraph/types/#langgraph.types.interrupt)来停止执行。图状态使用 LangGraph 的[持久层](/oss/python/langgraph/persistence)保存，因此执行可以安全暂停并在稍后恢复。

人工决策然后决定接下来会发生什么：操作可以按原样批准（`approve`）、在运行前修改（`edit`）或拒绝并提供反馈（`reject`）。

## 中断决策类型

中间件定义了三种内置的人工响应中断方式：

| 决策类型 | 描述 | 示例用例 |
|---------|------|----------|
| ✅ `approve` | 操作按原样批准并无更改执行 | 按原样发送电子邮件草稿 |
| ✏️ `edit` | 工具调用在修改后执行 | 在发送电子邮件前更改收件人 |
| ❌ `reject` | 工具调用被拒绝，解释信息添加到对话中 | 拒绝电子邮件草稿并解释如何重写 |

每个工具可用的决策类型取决于你在 `interrupt_on` 中配置的策略。当多个工具调用同时暂停时，每个操作都需要单独的决策。决策必须按照中断请求中操作出现的相同顺序提供。

**提示**：当**编辑**工具参数时，请保守地进行更改。对原始参数的显著修改可能导致模型重新评估其方法，并可能多次执行工具或采取意外操作。

## 配置中断

要使用 HITL，请在创建智能体时将中间件添加到智能体的 `middleware` 列表中。

你配置一个工具操作到允许的决策类型的映射。当工具调用匹配映射中的操作时，中间件将中断执行。

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
                "write_file": True,  # 允许所有决策（approve、edit、reject）
                "execute_sql": {"allowed_decisions": ["approve", "reject"]},  # 不允许编辑
                # 安全操作，不需要批准
                "read_data": False,
            },
            # 中断消息的前缀 - 与工具名称和参数组合形成完整消息
            # 例如："Tool execution pending approval: execute_sql with query='DELETE FROM...'"
            # 单个工具可以通过在其中断配置中指定 "description" 来覆盖此设置
            description_prefix="工具执行待批准",
        ),
    ],
    # Human-in-the-loop 需要检查点处理来处理中断。
    # 在生产环境中，使用持久性检查点如 AsyncPostgresSaver。
    checkpointer=InMemorySaver(),
)
```

**重要信息**：你必须配置一个检查点以在中断之间保持图状态。在生产环境中，使用持久性检查点如 [`AsyncPostgresSaver`](https://reference.langchain.com/python/langgraph/checkpoints/#langgraph.checkpoint.postgres.aio.AsyncPostgresSaver)。对于测试或原型设计，使用 [`InMemorySaver`](https://reference.langchain.com/python/langgraph/checkpoints/#langgraph.checkpoint.memory.InMemorySaver)。

在调用智能体时，传递包含**线程 ID** 的 `config` 以将执行与会话线程关联。有关详细信息，请参阅 [LangGraph 中断文档](/oss/python/langgraph/interrupts)。

## 响应中断

当你调用智能体时，它会运行直到完成或引发中断。当工具调用匹配你在 `interrupt_on` 中配置的策略时，会触发中断。在这种情况下，调用结果将包括一个 `__interrupt__` 字段，其中包含需要审查的操作。然后你可以将这些操作呈现给审查者，并在提供决策后恢复执行。

```python
from langgraph.types import Command

# Human-in-the-loop 利用 LangGraph 的持久层。
# 你必须提供线程 ID 以将执行与会话线程关联，
# 这样对话可以暂停和恢复（这是人工审查所需的）。
config = {"configurable": {"thread_id": "some_id"}}

# 运行图直到遇到中断。
result = agent.invoke(
    {
        "messages": [
            {
                "role": "user",
                "content": "从数据库中删除旧记录",
            }
        ]
    },
    config=config
)

# 中断包含完整的 HITL 请求，包括 action_requests 和 review_configs
print(result['__interrupt__'])
# 输出示例：
# [
#    Interrupt(
#       value={
#          'action_requests': [
#             {
#                'name': 'execute_sql',
#                'arguments': {'query': 'DELETE FROM records WHERE created_at < NOW() - INTERVAL \'30 days\';'},
#                'description': '工具执行待批准\n\n工具: execute_sql\n参数: {...}'
#             }
#          ],
#          'review_configs': [
#             {
#                'action_name': 'execute_sql',
#                'allowed_decisions': ['approve', 'reject']
#             }
#          ]
#       }
#    )
# ]

# 使用批准决策恢复
agent.invoke(
    Command(
        resume={"decisions": [{"type": "approve"}]}  # 或 "edit", "reject"
    ),
    config=config  # 相同的线程 ID 以恢复暂停的对话
)
```

### 决策类型

#### ✅ 批准 (approve)

使用 `approve` 按原样批准工具调用并无更改执行。

```python
agent.invoke(
    Command(
        # 决策以列表形式提供，每个待审查操作一个。
        # 决策的顺序必须与 `__interrupt__` 请求中列出的操作顺序匹配。
        resume={
            "decisions": [
                {
                    "type": "approve",
                }
            ]
        }
    ),
    config=config  # 相同的线程 ID 以恢复暂停的对话
)
```

#### ✏️ 编辑 (edit)

使用 `edit` 在执行前修改工具调用。提供编辑后的操作，包括新的工具名称和参数。

```python
agent.invoke(
    Command(
        # 决策以列表形式提供，每个待审查操作一个。
        # 决策的顺序必须与 `__interrupt__` 请求中列出的操作顺序匹配。
        resume={
            "decisions": [
                {
                    "type": "edit",
                    # 编辑后的操作，包含工具名称和参数
                    "edited_action": {
                        # 要调用的工具名称。
                        # 通常与原始操作相同。
                        "name": "execute_sql",
                        # 传递给工具的参数。
                        "args": {"query": "SELECT * FROM records LIMIT 10"},
                    }
                }
            ]
        }
    ),
    config=config  # 相同的线程 ID 以恢复暂停的对话
)
```

**提示**：当**编辑**工具参数时，请保守地进行更改。对原始参数的显著修改可能导致模型重新评估其方法，并可能多次执行工具或采取意外操作。

#### ❌ 拒绝 (reject)

使用 `reject` 拒绝工具调用并提供反馈而不是执行。

```python
agent.invoke(
    Command(
        # 决策以列表形式提供，每个待审查操作一个。
        # 决策的顺序必须与 `__interrupt__` 请求中列出的操作顺序匹配。
        resume={
            "decisions": [
                {
                    "type": "reject",
                    # 关于为什么拒绝操作的说明
                    "message": "不，这是错误的，因为...，应该这样做...",
                }
            ]
        }
    ),
    config=config  # 相同的线程 ID 以恢复暂停的对话
)
```

`message` 作为反馈添加到对话中，以帮助智能体理解为什么操作被拒绝以及它应该做什么。

### 多个决策

当多个操作待审查时，为每个操作提供一个决策，顺序与中断中出现的顺序相同：

```python
{
    "decisions": [
        {"type": "approve"},
        {
            "type": "edit",
            "edited_action": {
                "name": "tool_name",
                "args": {"param": "new_value"}
            }
        },
        {
            "type": "reject",
            "message": "此操作不允许"
        }
    ]
}
```

## 完整示例

### 示例 1：文件操作和数据库操作

```python
from langchain.agents import create_agent
from langchain.agents.middleware import HumanInTheLoopMiddleware
from langgraph.checkpoint.memory import InMemorySaver
from langchain.tools import tool
from langgraph.types import Command

# 定义工具
@tool
def write_file(filepath: str, content: str) -> str:
    """写入文件"""
    try:
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(content)
        return f"成功写入文件: {filepath}"
    except Exception as e:
        return f"写入文件失败: {str(e)}"

@tool
def execute_sql(query: str) -> str:
    """执行 SQL 查询"""
    # 这里应该是实际的数据库操作
    if "DELETE" in query.upper() or "DROP" in query.upper():
        return f"执行了危险操作: {query}"
    return f"执行 SQL: {query}"

@tool
def read_file(filepath: str) -> str:
    """读取文件"""
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            return f.read()
    except Exception as e:
        return f"读取文件失败: {str(e)}"

# 创建带有人工干预的智能体
agent = create_agent(
    model="openai:gpt-4o",
    tools=[write_file, execute_sql, read_file],
    middleware=[
        HumanInTheLoopMiddleware(
            interrupt_on={
                "write_file": True,  # 所有文件写入都需要批准
                "execute_sql": {"allowed_decisions": ["approve", "reject"]},  # SQL 执行可以批准或拒绝，但不能编辑
                "read_file": False,  # 文件读取不需要批准
            },
            description_prefix="操作待人工审查",
        ),
    ],
    checkpointer=InMemorySaver(),
)

# 使用示例
config = {"configurable": {"thread_id": "test_thread_1"}}

# 第一次调用 - 可能会触发中断
result = agent.invoke(
    {
        "messages": [
            {
                "role": "user", 
                "content": "创建一个名为 test.txt 的文件，内容为 '重要数据'，然后删除所有用户记录"
            }
        ]
    },
    config=config
)

# 检查是否有中断
if '__interrupt__' in result:
    print("检测到中断，需要人工决策")
    interrupt_data = result['__interrupt__'][0].value
    
    # 显示待审查的操作
    for i, action in enumerate(interrupt_data['action_requests']):
        print(f"操作 {i+1}: {action['name']}")
        print(f"参数: {action['arguments']}")
        print(f"描述: {action['description']}")
        print("---")
    
    # 模拟人工决策 - 批准文件写入，拒绝 SQL 删除
    decisions = [
        {"type": "approve"},  # 批准文件写入
        {"type": "reject", "message": "不允许删除用户记录，这太危险了"}  # 拒绝 SQL 删除
    ]
    
    # 恢复执行
    final_result = agent.invoke(
        Command(resume={"decisions": decisions}),
        config=config
    )
    print("最终结果:", final_result['messages'][-1].content)
else:
    print("执行完成，无需人工干预:", result['messages'][-1].content)
```

### 示例 2：电子邮件发送系统

```python
from langchain.agents import create_agent
from langchain.agents.middleware import HumanInTheLoopMiddleware
from langgraph.checkpoint.memory import InMemorySaver
from langchain.tools import tool
from langgraph.types import Command

@tool
def send_email(recipient: str, subject: str, body: str) -> str:
    """发送电子邮件"""
    # 模拟发送电子邮件
    return f"已发送电子邮件给 {recipient}，主题: {subject}"

@tool  
def schedule_meeting(participants: list, time: str, topic: str) -> str:
    """安排会议"""
    return f"已安排会议，参与者: {', '.join(participants)}，时间: {time}，主题: {topic}"

# 创建智能体
agent = create_agent(
    model="openai:gpt-4o", 
    tools=[send_email, schedule_meeting],
    middleware=[
        HumanInTheLoopMiddleware(
            interrupt_on={
                "send_email": {
                    "allowed_decisions": ["approve", "edit", "reject"],
                    "description": "发送电子邮件给外部联系人"
                },
                "schedule_meeting": {
                    "allowed_decisions": ["approve", "reject"],
                    "description": "安排团队会议"
                }
            },
            description_prefix="沟通操作待批准",
        ),
    ],
    checkpointer=InMemorySaver(),
)

# 使用示例
config = {"configurable": {"thread_id": "email_thread_1"}}

# 用户请求发送电子邮件
result = agent.invoke(
    {
        "messages": [
            {
                "role": "user",
                "content": "给客户 John 发送一封电子邮件，主题是项目更新，内容是项目进展顺利"
            }
        ]
    },
    config=config
)

if '__interrupt__' in result:
    print("需要审查电子邮件发送")
    
    # 人工决定编辑电子邮件
    decisions = [{
        "type": "edit",
        "edited_action": {
            "name": "send_email",
            "args": {
                "recipient": "john@client.com",
                "subject": "项目进度更新 - 需要您的反馈",
                "body": "亲爱的 John，项目目前进展顺利。我们已经完成了第一阶段的主要功能。请查看附件中的详细报告并提供您的反馈。"
            }
        }
    }]
    
    final_result = agent.invoke(
        Command(resume={"decisions": decisions}),
        config=config
    )
    print("电子邮件已发送:", final_result['messages'][-1].content)
```

## 执行生命周期

中间件定义了一个 `after_model` 钩子，该钩子在模型生成响应后但在任何工具调用执行之前运行：

1. 智能体调用模型生成响应
2. 中间件检查响应中的工具调用
3. 如果任何调用需要人工输入，中间件构建一个包含 `action_requests` 和 `review_configs` 的 `HITLRequest` 并调用 [interrupt](https://reference.langchain.com/python/langgraph/types/#langgraph.types.interrupt)
4. 智能体等待人工决策
5. 基于 `HITLResponse` 决策，中间件执行批准或编辑的调用，为拒绝的调用合成 [ToolMessage](https://reference.langchain.com/python/langchain/messages/#langchain.messages.ToolMessage)，并恢复执行

## 自定义 HITL 逻辑

对于更专业的工作流程，你可以直接使用 [interrupt](https://reference.langchain.com/python/langgraph/types/#langgraph.types.interrupt) 原语和 [middleware](/oss/python/langchain/middleware) 抽象构建自定义 HITL 逻辑。

查看上面的[执行生命周期](#执行生命周期)以了解如何将中断集成到智能体的操作中。

## 最佳实践

1. **合理配置中断策略**：只为真正需要人工监督的操作启用中断
2. **提供清晰的描述**：使用 `description` 字段为审查者提供足够的上下文
3. **考虑用户体验**：在编辑操作时保持参数的一致性，避免导致智能体困惑
4. **使用持久性检查点**：在生产环境中使用数据库支持的检查点
5. **记录决策**：考虑记录人工决策以供审计和培训目的

通过合理使用 HITL 中间件，你可以在保持自动化效率的同时，为关键操作添加必要的人工监督，确保系统的安全性和可靠性。
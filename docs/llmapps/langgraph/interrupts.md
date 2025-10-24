# LangGraph 中断功能完整教程

## 什么是中断？

中断（Interrupts）允许你在图执行的特定点暂停，并在继续之前等待外部输入。这实现了"人在回路"（human-in-the-loop）模式，让你能够在需要外部输入时暂停执行。

### 核心特性

- **动态中断**：可以在代码的任何位置放置中断，并基于应用逻辑条件触发
- **状态持久化**：触发中断时，LangGraph 使用持久化层保存图状态
- **无限等待**：中断后会一直等待，直到你明确恢复执行

## 基础使用

### 设置中断

要使用中断功能，你需要三个关键组件：

```python
from langgraph.types import interrupt

def approval_node(state: State):
    # 暂停执行并请求批准
    approved = interrupt("你是否批准此操作？")
    
    # 恢复时，Command(resume=...)的值会在这里返回
    return {"approved": approved}
```

### 配置要求

1. **检查点器（Checkpointer）**：用于持久化图状态（生产环境使用持久化检查点器）
2. **线程ID（thread_id）**：在配置中指定，用于标识要恢复的状态
3. **JSON可序列化**：传递给 `interrupt()` 的值必须是 JSON 可序列化的

## 中断工作流程

### 1. 触发中断

当调用 `interrupt()` 时：

1. **图执行暂停**：在调用 `interrupt` 的精确点暂停
2. **状态保存**：使用检查点器保存当前状态
3. **返回值**：中断值通过 `__interrupt__` 字段返回给调用者
4. **无限等待**：图会一直等待，直到你恢复执行
5. **值传递**：恢复时的响应值会成为 `interrupt()` 调用的返回值

### 2. 恢复中断

```python
from langgraph.types import Command

# 初始运行 - 遇到中断并暂停
config = {"configurable": {"thread_id": "thread-1"}}
result = graph.invoke({"input": "data"}, config=config)

# 检查中断内容
print(result["__interrupt__"])
# 输出: [Interrupt(value='你是否批准此操作？')]

# 使用人类响应恢复执行
# resume 的值会成为节点内 interrupt() 的返回值
graph.invoke(Command(resume=True), config=config)
```

**恢复要点**：

- 必须使用**相同的线程ID**
- `Command(resume=...)` 的值成为 `interrupt()` 的返回值
- 节点会从头开始重新执行

## 常见使用模式

### 1. 审批工作流

在关键操作前暂停并请求批准：

```python
from typing import Literal
from langgraph.types import interrupt, Command

def approval_node(state: State) -> Command[Literal["proceed", "cancel"]]:
    # 暂停执行，payload 会出现在 result["__interrupt__"] 中
    is_approved = interrupt({
        "question": "是否要继续执行此操作？",
        "details": state["action_details"]
    })

    # 基于响应路由
    if is_approved:
        return Command(goto="proceed")
    else:
        return Command(goto="cancel")
```

恢复方式：
```python
# 批准
graph.invoke(Command(resume=True), config=config)

# 拒绝
graph.invoke(Command(resume=False), config=config)
```

### 2. 审查和编辑状态

让人类在继续之前审查和编辑图状态：

```python
def review_node(state: State):
    # 暂停并显示当前内容供审查
    edited_content = interrupt({
        "instruction": "审查并编辑此内容",
        "content": state["generated_text"]
    })

    # 使用编辑后的版本更新状态
    return {"generated_text": edited_content}
```

恢复时提供编辑内容：
```python
graph.invoke(
    Command(resume="编辑和改进后的文本"),
    config=config
)
```

### 3. 在工具中使用中断

在工具函数内部直接放置中断：

```python
from langchain.tools import tool
from langgraph.types import interrupt

@tool
def send_email(to: str, subject: str, body: str):
    """发送邮件给收件人"""

    # 在发送前暂停
    response = interrupt({
        "action": "send_email",
        "to": to,
        "subject": subject,
        "body": body,
        "message": "是否批准发送此邮件？"
    })

    if response.get("action") == "approve":
        # 恢复值可以在执行前覆盖输入
        final_to = response.get("to", to)
        final_subject = response.get("subject", subject)
        final_body = response.get("body", body)
        return f"邮件已发送至 {final_to}"
    return "用户取消了邮件"
```

### 4. 验证人类输入

使用多个 `interrupt` 调用来验证输入：

```python
def get_age_node(state: State):
    prompt = "你的年龄是多少？"

    while True:
        answer = interrupt(prompt)

        # 验证输入
        if isinstance(answer, int) and answer > 0:
            # 有效输入 - 继续
            break
        else:
            # 无效输入 - 用更明确的提示重新询问
            prompt = f"'{answer}' 不是有效的年龄。请输入正数。"

    return {"age": answer}
```

## 中断的重要规则

### 1. 不要将 `interrupt` 调用包裹在 try/except 中

✅ **正确做法**：
```python
def node_a(state: State):
    # 先处理中断，再单独处理错误条件
    interrupt("你的名字是什么？")
    try:
        fetch_data()  # 这里可能失败
    except Exception as e:
        print(e)
    return state
```

❌ **错误做法**：
```python
def node_a(state: State):
    try:
        interrupt("你的名字是什么？")
    except Exception as e:
        print(e)  # 这会捕获中断异常！
    return state
```

### 2. 不要在节点内重新排序 `interrupt` 调用

✅ **正确做法** - 保持一致的调用顺序：
```python
def node_a(state: State):
    name = interrupt("你的名字？")
    age = interrupt("你的年龄？")
    city = interrupt("你的城市？")
    return {"name": name, "age": age, "city": city}
```

❌ **错误做法** - 条件性跳过中断：
```python
def node_a(state: State):
    name = interrupt("你的名字？")
    
    # 第一次运行可能跳过此中断
    # 恢复时可能不会跳过 - 导致索引不匹配
    if state.get("needs_age"):
        age = interrupt("你的年龄？")
    
    city = interrupt("你的城市？")
    return {"name": name, "city": city}
```

### 3. 不要在 `interrupt` 调用中传递复杂值

✅ **正确做法** - 使用简单可序列化类型：
```python
def node_a(state: State):
    # 传递简单类型
    name = interrupt("你的名字？")
    
    # 传递包含简单值的字典
    response = interrupt({
        "question": "输入用户详情",
        "fields": ["name", "email", "age"]
    })
    return {"user": response}
```

❌ **错误做法** - 传递复杂对象：
```python
def node_a(state: State):
    class DataProcessor:
        def __init__(self, config):
            self.config = config
    
    processor = DataProcessor({"mode": "strict"})
    
    # 这会失败，因为实例无法序列化
    response = interrupt({
        "question": "输入要处理的数据",
        "processor": processor  # 无法序列化！
    })
    return {"result": response}
```

### 4. `interrupt` 调用前的副作用必须是幂等的

✅ **正确做法** - 使用幂等操作或在中断后执行副作用：
```python
def node_a(state: State):
    # ✅ 使用幂等的 upsert 操作
    db.upsert_user(user_id=state["user_id"], status="pending_approval")
    
    approved = interrupt("是否批准此更改？")
    
    # ✅ 副作用放在中断之后
    if approved:
        db.create_audit_log(user_id=state["user_id"], action="approved")
    
    return {"approved": approved}
```

❌ **错误做法** - 在中断前执行非幂等操作：
```python
def node_a(state: State):
    # ❌ 在中断前创建新记录
    # 每次恢复都会创建重复记录
    audit_id = db.create_audit_log({
        "user_id": state["user_id"],
        "action": "pending_approval"
    })
    
    approved = interrupt("是否批准此更改？")
    return {"approved": approved, "audit_id": audit_id}
```

## 调试技巧

### 使用静态中断作为断点

在编译时设置断点：
```python
graph = builder.compile(
    interrupt_before=["node_a"],      # 在节点执行前暂停
    interrupt_after=["node_b", "node_c"],  # 在节点执行后暂停
    checkpointer=checkpointer,
)

# 运行到第一个断点
graph.invoke(inputs, config=config)

# 继续执行到下一个断点
graph.invoke(None, config=config)
```

在运行时设置断点：
```python
graph.invoke(
    inputs,
    interrupt_before=["node_a"],
    interrupt_after=["node_b", "node_c"],
    config=config,
)
```

### 使用 LangGraph Studio

你可以使用 LangGraph Studio 在 UI 中设置静态中断，并在执行过程中随时检查图状态。

## 总结

中断功能为 LangGraph 提供了强大的人机协作能力。关键要点：

1. **动态控制**：可以在代码的任何位置条件性地触发中断
2. **状态持久化**：中断状态会被安全保存，支持长时间暂停
3. **灵活恢复**：恢复时可以传递任意 JSON 可序列化的值
4. **模式丰富**：支持审批、审查、验证等多种人机交互模式
5. **遵循规则**：注意中断的使用规则，避免常见陷阱

通过合理使用中断，你可以构建出更加灵活、安全、可控的 AI 应用系统。
# LangGraph 中断机制完整教程

## 什么是中断？

中断（Interrupts）允许你在特定点暂停图执行，等待外部输入后再继续。这实现了"人在回路"（human-in-the-loop）模式，让你能够在需要外部输入时暂停执行。

当触发中断时，LangGraph 会使用其[持久化层](/oss/python/langgraph/persistence)保存图状态，并无限期等待直到你恢复执行。

## 核心概念

### 中断的工作原理

中断通过在图的节点中调用 `interrupt()` 函数来工作：

- 中断是**动态的** - 可以在代码的任何位置放置，并基于应用逻辑条件触发
- 检查点保持你的位置 - 检查点器保存确切的图状态，即使处于错误状态也能稍后恢复
- `thread_id` 是指针 - 设置 `config={"configurable": {"thread_id": ...}}` 来告诉检查点器加载哪个状态
- 中断负载显示为 `__interrupt__` - 传递给 `interrupt()` 的值会在 `__interrupt__` 字段中返回给调用者

## 基础用法

### 1. 使用 `interrupt` 暂停执行

要使用中断，你需要：

1. **检查点器** - 持久化图状态（生产环境使用持久化检查点器）
2. **线程 ID** - 在配置中指定，以便运行时知道从哪个状态恢复
3. **调用 `interrupt()`** - 在需要暂停的地方调用

```python
from langgraph.types import interrupt

def approval_node(state: State):
    # 暂停并请求批准
    approved = interrupt("你是否批准此操作？")
    
    # 当你恢复时，Command(resume=...) 的值会在这里返回
    return {"approved": approved}
```

当调用 `interrupt` 时会发生：

1. **图执行被挂起**在调用 `interrupt` 的确切位置
2. **状态被保存**以便稍后恢复执行
3. **值返回给调用者**在 `__interrupt__` 字段下
4. **图无限期等待**直到你恢复执行
5. **响应传回节点**当你恢复时，成为 `interrupt()` 调用的返回值

### 2. 恢复中断

中断暂停执行后，通过使用包含恢复值的 `Command` 再次调用图来恢复：

```python
from langgraph.types import Command

# 初始运行 - 遇到中断并暂停
config = {"configurable": {"thread_id": "thread-1"}}
result = graph.invoke({"input": "data"}, config=config)

# 检查中断内容
print(result["__interrupt__"])
# > [Interrupt(value='你是否批准此操作？')]

# 用人类的响应恢复
# resume 负载成为节点内 interrupt() 的返回值
graph.invoke(Command(resume=True), config=config)
```

**恢复的关键点：**

- 必须使用与中断发生时**相同的线程 ID**
- 传递给 `Command(resume=...)` 的值成为 `interrupt` 调用的返回值
- 节点从调用 `interrupt` 的节点开头重新开始
- 可以传递任何 JSON 可序列化的值作为恢复值

## 常用模式

### 1. 批准或拒绝工作流

最常见的用途是在关键操作前暂停并请求批准：

```python
from typing import Literal
from langgraph.types import interrupt, Command

def approval_node(state: State) -> Command[Literal["proceed", "cancel"]]:
    # 暂停执行；负载显示在 result["__interrupt__"] 中
    is_approved = interrupt({
        "question": "你是否要继续执行此操作？",
        "details": state["action_details"]
    })

    # 基于响应路由
    if is_approved:
        return Command(goto="proceed")  # 在提供恢复负载后运行
    else:
        return Command(goto="cancel")
```

恢复图时，传递 `true` 批准或 `false` 拒绝：

```python
# 批准
graph.invoke(Command(resume=True), config=config)

# 拒绝
graph.invoke(Command(resume=False), config=config)
```

### 2. 审查和编辑状态

让人类在继续前审查和编辑图状态：

```python
from langgraph.types import interrupt

def review_node(state: State):
    # 暂停并显示当前内容供审查
    edited_content = interrupt({
        "instruction": "审查并编辑此内容",
        "content": state["generated_text"]
    })

    # 用编辑后的版本更新状态
    return {"generated_text": edited_content}
```

恢复时提供编辑后的内容：

```python
graph.invoke(
    Command(resume="编辑和改进后的文本"),  # 值成为 interrupt() 的返回值
    config=config
)
```

### 3. 工具中的中断

在工具函数内部直接放置中断：

```python
from langchain.tools import tool
from langgraph.types import interrupt

@tool
def send_email(to: str, subject: str, body: str):
    """发送邮件给收件人"""

    # 在发送前暂停；负载显示在 result["__interrupt__"] 中
    response = interrupt({
        "action": "send_email",
        "to": to,
        "subject": subject,
        "body": body,
        "message": "批准发送此邮件吗？"
    })

    if response.get("action") == "approve":
        # 恢复值可以在执行前覆盖输入
        final_to = response.get("to", to)
        final_subject = response.get("subject", subject)
        final_body = response.get("body", body)
        return f"邮件已发送给 {final_to}，主题 '{final_subject}'"
    return "用户取消了邮件"
```

### 4. 验证人类输入

使用多个 `interrupt` 调用来验证输入并在无效时重新询问：

```python
from langgraph.types import interrupt

def get_age_node(state: State):
    prompt = "你的年龄是多少？"

    while True:
        answer = interrupt(prompt)  # 负载显示在 result["__interrupt__"] 中

        # 验证输入
        if isinstance(answer, int) and answer > 0:
            # 有效输入 - 继续
            break
        else:
            # 无效输入 - 用更具体的提示再次询问
            prompt = f"'{answer}' 不是有效的年龄。请输入正数。"

    return {"age": answer}
```

## 中断规则

### 1. 不要用 try/except 包装 `interrupt` 调用

中断通过抛出特殊异常来工作，如果包装在 try/except 中会捕获这个异常：

```python
# ✅ 正确：将中断与错误处理代码分开
def node_a(state: State):
    interrupt("你叫什么名字？")
    try:
        fetch_data()  # 这可能会失败
    except Exception as e:
        print(e)
    return state

# ❌ 错误：用 try/except 包装 interrupt
def node_a(state: State):
    try:
        interrupt("你叫什么名字？")
    except Exception as e:
        print(e)
    return state
```

### 2. 不要在节点内重新排序 `interrupt` 调用

当节点包含多个中断调用时，匹配是**严格基于索引的**：

```python
# ✅ 正确：中断调用每次以相同顺序发生
def node_a(state: State):
    name = interrupt("你叫什么名字？")
    age = interrupt("你的年龄是多少？")
    city = interrupt("你在哪个城市？")
    return {"name": name, "age": age, "city": city}

# ❌ 错误：有条件地跳过中断会改变顺序
def node_a(state: State):
    name = interrupt("你叫什么名字？")
    if state.get("needs_age"):  # 条件可能在不同运行间变化
        age = interrupt("你的年龄是多少？")
    city = interrupt("你在哪个城市？")
    return {"name": name, "city": city}
```

### 3. 不要在 `interrupt` 调用中返回复杂值

只使用可合理序列化的值：

```python
# ✅ 正确：传递简单的可序列化类型
def node_a(state: State):
    name = interrupt("你叫什么名字？")
    count = interrupt(42)
    approved = interrupt(True)
    return {"name": name, "count": count, "approved": approved}

# ❌ 错误：传递函数、类实例或其他复杂对象
def node_a(state: State):
    response = interrupt({
        "question": "你叫什么名字？",
        "validator": validate_input  # 这会失败
    })
    return {"name": response}
```

### 4. `interrupt` 之前调用的副作用必须是幂等的

因为中断通过重新运行它们被调用的节点来工作：

```python
# ✅ 正确：在中断后放置副作用
def node_a(state: State):
    approved = interrupt("批准此更改吗？")
    if approved:
        db.create_audit_log(user_id=state["user_id"], action="approved")
    return {"approved": approved}

# ❌ 错误：在中断前执行非幂等操作
def node_a(state: State):
    audit_id = db.create_audit_log({  # 每次恢复都会创建新记录
        "user_id": state["user_id"],
        "action": "pending_approval"
    })
    approved = interrupt("批准此更改吗？")
    return {"approved": approved, "audit_id": audit_id}
```

## 调试与中断

### 静态中断（断点）

使用静态中断作为断点来逐步执行图：

```python
# 编译时设置
graph = builder.compile(
    interrupt_before=["node_a"],  # 在节点执行前暂停
    interrupt_after=["node_b", "node_c"],  # 在节点执行后暂停
    checkpointer=checkpointer,
)

# 运行时设置
graph.invoke(
    inputs,
    interrupt_before=["node_a"],
    interrupt_after=["node_b", "node_c"],
    config=config,
)

# 恢复执行
graph.invoke(None, config=config)
```

### 使用 LangGraph Studio

可以使用 [LangGraph Studio](/langsmith/studio) 在 UI 中设置静态中断，并在执行过程中的任何点检查图状态。

## 总结

中断是 LangGraph 中实现人在回路模式的核心功能。通过合理使用中断，你可以：

- 在关键操作前请求人工批准
- 让人类审查和编辑 LLM 输出
- 验证人类输入并在无效时重新提示
- 在工具执行前进行人工审查

记住遵循中断的最佳实践，特别是关于副作用幂等性和中断调用顺序的规则，以确保可靠的行为。

在生产环境中，确保使用持久的检查点器（如数据库支持的检查点器）来可靠地保存和恢复图状态。
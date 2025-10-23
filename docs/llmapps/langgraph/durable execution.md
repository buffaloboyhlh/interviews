# LangGraph 持久化执行教程

## 什么是持久化执行？

持久化执行是一种技术，允许流程或工作流在关键点保存进度，从而能够暂停并在之后从离开的地方准确恢复。这在以下场景中特别有用：

- **人工介入场景**：用户可以在继续之前检查、验证或修改流程
- **长时间运行任务**：可能遇到中断或错误的任务（如 LLM 调用超时）

通过保存已完成的工作，持久化执行确保流程可以在不重新处理先前步骤的情况下恢复——即使在长时间延迟后（如一周后）。

## 启用持久化执行

### 1. 配置持久化存储

要启用持久化执行，需要在编译图时指定一个检查点存储器：

```python
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import StateGraph

# 创建状态图
builder = StateGraph(State)
builder.add_node("process_data", process_data)
builder.add_edge(START, "process_data")

# 指定检查点存储器
checkpointer = InMemorySaver()

# 编译图并启用持久化
graph = builder.compile(checkpointer=checkpointer)
```

### 2. 指定线程标识符

执行工作流时需要提供线程ID来跟踪特定实例的执行历史：

```python
import uuid

# 生成唯一的线程ID
thread_id = uuid.uuid4()
config = {"configurable": {"thread_id": thread_id}}

# 执行工作流
graph.invoke({"input": "data"}, config)
```

## 确保确定性和一致性重放

当工作流恢复时，代码**不会**从停止的同一行代码恢复，而是从适当的起点重新开始。因此，必须遵循以下准则：

### 使用任务包装非确定性操作

将任何非确定性操作（如随机数生成）或具有副作用的操作（如文件写入、API调用）包装在 `@task` 装饰器中：

```python
from langgraph.func import task
import requests

@task
def make_api_call(url: str):
    """包装API调用"""
    return requests.get(url).text[:100]

def process_data(state: State):
    """处理数据的节点"""
    # 使用任务来确保确定性
    api_results = [make_api_call(url) for url in state['urls']]
    results = [result.result() for result in api_results]
    return {"results": results}
```

### 最佳实践

- **避免重复工作**：将多个副作用操作包装在单独的任务中
- **封装非确定性操作**：确保工作流遵循精确记录的执行序列
- **使用幂等操作**：确保副作用操作可以安全重试

## 持久化模式

LangGraph 提供三种持久化模式，平衡性能和数据一致性：

### 1. `"exit"` 模式
仅在图执行完成时持久化更改。性能最佳，但不保存中间状态。

```python
graph.stream(
    {"input": "test"},
    durability="exit",
    config=config
)
```

### 2. `"async"` 模式
在下一步执行时异步持久化更改。提供良好的性能和持久性平衡。

```python
graph.stream(
    {"input": "test"},
    durability="async", 
    config=config
)
```

### 3. `"sync"` 模式
在下一步开始前同步持久化更改。提供最高的持久性保证。

```python
graph.stream(
    {"input": "test"},
    durability="sync",
    config=config
)
```

## 在节点中使用任务

如果节点包含多个操作，可以将每个操作转换为任务：

### 原始实现（有问题）

```python
def call_api(state: State):
    """有问题的实现 - 直接进行API调用"""
    result = requests.get(state['url']).text[:100]  # 副作用操作
    return {"result": result}
```

### 使用任务的改进实现

```python
@task
def _make_request(url: str):
    """将API调用包装为任务"""
    return requests.get(url).text[:100]

def call_api(state: State):
    """改进的实现 - 使用任务"""
    requests = [_make_request(url) for url in state['urls']]
    results = [request.result() for request in requests]
    return {"results": results}
```

## 恢复工作流

### 1. 暂停和恢复工作流

使用中断机制在特定点暂停工作流：

```python
from langgraph.types import interrupt, Command

# 在节点中设置中断点
def review_step(state: State):
    # ... 处理逻辑
    interrupt()  # 暂停执行等待人工审核
    return state

# 恢复工作流
command = Command(resume=True, update={"approved": True})
graph.stream(None, config=config, command=command)
```

### 2. 从错误中恢复

工作流可以从最后一个成功的检查点自动恢复：

```python
try:
    # 首次执行
    graph.invoke({"input": "data"}, config)
except Exception:
    # 从错误中恢复 - 输入为 None
    graph.invoke(None, config)
```

## 恢复起点

工作流恢复时的起点取决于使用的API：

- **StateGraph**：从执行停止的**节点**开始
- **子图调用**：从调用被停止子图的**父节点**开始
- **Functional API**：从执行停止的**入口点**开始

## 完整示例

```python
from typing import List, Optional
from typing_extensions import TypedDict
import uuid
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.func import task
from langgraph.graph import StateGraph, START, END
import requests

# 定义状态
class ProcessingState(TypedDict):
    urls: List[str]
    results: Optional[List[str]]
    processed: bool

# 包装API调用为任务
@task
def fetch_url_content(url: str):
    """获取URL内容"""
    return requests.get(url).text[:500]

def process_urls(state: ProcessingState):
    """处理URL的节点"""
    # 使用任务并行处理URL
    fetch_tasks = [fetch_url_content(url) for url in state['urls']]
    results = [task.result() for task in fetch_tasks]
    
    return {
        "results": results,
        "processed": True
    }

# 构建图
builder = StateGraph(ProcessingState)
builder.add_node("process_urls", process_urls)
builder.add_edge(START, "process_urls")
builder.add_edge("process_urls", END)

# 启用持久化
checkpointer = InMemorySaver()
graph = builder.compile(checkpointer=checkpointer)

# 执行工作流
thread_id = uuid.uuid4()
config = {"configurable": {"thread_id": thread_id}}

try:
    # 首次执行
    result = graph.invoke({
        "urls": ["https://example.com", "https://example.org"]
    }, config)
    print("处理完成:", result)
    
except Exception as e:
    print(f"执行出错: {e}")
    # 从错误中恢复
    recovery_result = graph.invoke(None, config)
    print("恢复后的结果:", recovery_result)
```

通过遵循本教程中的模式，您可以构建健壮的、支持持久化执行的 LangGraph 工作流，确保在中断或错误后能够可靠恢复。
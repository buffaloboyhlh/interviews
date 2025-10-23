# LangGraph 思维模式完全指南

> 通过将客户支持邮件智能体分解为离散步骤，学习如何用 LangGraph 构建智能体

LangGraph 可以改变你构建智能体的思维方式。使用 LangGraph 构建智能体时，你首先需要将其分解为称为**节点**的离散步骤。然后，描述每个节点的不同决策和转换。最后，通过共享的**状态**将节点连接起来，每个节点都可以读取和写入这个状态。本教程将指导你完成构建客户支持邮件智能体的思维过程。

## 从要自动化的流程开始

假设你需要构建一个处理客户支持邮件的 AI 智能体。产品团队给了你以下要求：

智能体应该能够：

* 读取传入的客户邮件
* 按紧急程度和主题进行分类
* 搜索相关文档来回答问题
* 起草适当的回复
* 将复杂问题转交给人工代理
* 需要时安排跟进

需要处理的示例场景：

1. 简单产品问题："如何重置密码？"
2. 错误报告："选择 PDF 格式时导出功能崩溃"
3. 紧急账单问题："我的订阅被重复收费了！"
4. 功能请求："能否为移动应用添加深色模式？"
5. 复杂技术问题："我们的 API 集成间歇性失败，出现 504 错误"

要在 LangGraph 中实现智能体，你通常需要遵循相同的五个步骤。

## 步骤 1：将工作流程映射为离散步骤

首先识别流程中的不同步骤。每个步骤将成为一个**节点**（执行特定操作的函数）。然后勾画这些步骤如何相互连接。

![langgraph的工作流程.png](../../imgs/llm/langgraph%E7%9A%84%E5%B7%A5%E4%BD%9C%E6%B5%81%E7%A8%8B.png)

箭头显示可能的路径，但实际选择哪条路径的决策发生在每个节点内部。

现在你已经识别了工作流程中的组件，让我们理解每个节点需要做什么：

* **读取邮件**：提取和解析邮件内容
* **分类意图**：使用 LLM 分类紧急程度和主题，然后路由到适当的操作
* **文档搜索**：查询知识库获取相关信息
* **错误跟踪**：在跟踪系统中创建或更新问题
* **起草回复**：生成适当的响应
* **人工审核**：转交给人工代理进行批准或处理
* **发送回复**：发送邮件回复

> **提示**：注意有些节点决定下一步去哪里（分类意图、起草回复、人工审核），而其他节点总是进入相同的下一步（读取邮件总是到分类意图，文档搜索总是到起草回复）。

## 步骤 2：识别每个步骤需要做什么

对于图中的每个节点，确定它代表什么类型的操作以及需要什么上下文才能正常工作。

### LLM 步骤

当步骤需要理解、分析、生成文本或做出推理决策时使用：

**分类意图节点**

* 静态上下文（提示）：分类类别、紧急程度定义、响应格式
* 动态上下文（来自状态）：邮件内容、发件人信息
* 期望结果：确定路由的结构化分类

**起草回复节点**

* 静态上下文（提示）：语气指南、公司政策、响应模板
* 动态上下文（来自状态）：分类结果、搜索结果、客户历史
* 期望结果：准备审核的专业邮件回复

### 数据步骤

当步骤需要从外部源检索信息时使用：

**文档搜索节点**

* 参数：根据意图和主题构建的查询
* 重试策略：是，对暂时性故障使用指数退避
* 缓存：可以缓存常见查询以减少 API 调用

**客户历史查询**

* 参数：来自状态的客户邮件或 ID
* 重试策略：是，但如果不可用则回退到基本信息
* 缓存：是，使用生存时间来平衡新鲜度和性能

### 操作步骤

当步骤需要执行外部操作时使用：

**发送回复节点**

* 执行时机：批准后（人工或自动）
* 重试策略：是，对网络问题使用指数退避
* 不应缓存：每次发送都是唯一操作

**错误跟踪节点**

* 执行时机：意图为"错误"时总是执行
* 重试策略：是，关键是不能丢失错误报告
* 返回：要在响应中包含的工单 ID

### 用户输入步骤

当步骤需要人工干预时使用：

**人工审核节点**

* 决策上下文：原始邮件、草稿回复、紧急程度、分类
* 期望输入格式：批准布尔值加上可选的编辑响应
* 触发时机：高紧急程度、复杂问题或质量担忧

## 步骤 3：设计你的状态

状态是你的智能体中所有节点都可以访问的共享[内存](/oss/python/concepts/memory)。可以把它看作你的智能体在处理过程中用来跟踪一切学习和决策的笔记本。

### 什么应该放在状态中？

对每个数据片段问自己这些问题：

**包含在状态中**

* 是否需要跨步骤持久化？如果是，就放在状态中

**不要存储**

* 能否从其他数据推导出来？如果是，在需要时计算而不是存储在状态中

对于我们的邮件智能体，我们需要跟踪：

* 原始邮件和发件人信息（无法重建这些）
* 分类结果（多个下游节点需要）
* 搜索结果和客户数据（重新获取成本高）
* 草稿回复（需要通过审核持久化）
* 执行元数据（用于调试和恢复）

### 保持状态原始，按需格式化提示

> **关键原则**：你的状态应该存储原始数据，而不是格式化文本。在需要时在节点内部格式化提示。

这种分离意味着：

* 不同节点可以为了各自的需要以不同方式格式化相同的数据
* 你可以更改提示模板而不修改状态模式
* 调试更清晰 - 你可以看到每个节点接收到的确切数据
* 你的智能体可以演进而不破坏现有状态

让我们定义我们的状态：

```python
from typing import TypedDict, Literal

# 定义邮件分类结构
class EmailClassification(TypedDict):
    intent: Literal["question", "bug", "billing", "feature", "complex"]
    urgency: Literal["low", "medium", "high", "critical"]
    topic: str
    summary: str

class EmailAgentState(TypedDict):
    # 原始邮件数据
    email_content: str
    sender_email: str
    email_id: str

    # 分类结果
    classification: EmailClassification | None

    # 原始搜索/API 结果
    search_results: list[str] | None  # 原始文档块列表
    customer_history: dict | None  # 来自 CRM 的原始客户数据

    # 生成的内容
    draft_response: str | None
```

注意状态只包含原始数据 - 没有提示模板，没有格式化字符串，没有指令。分类输出作为单个字典存储，直接来自 LLM。

## 步骤 4：构建你的节点

现在我们实现每个步骤作为一个函数。LangGraph 中的节点只是一个 Python 函数，它接受当前状态并返回对其的更新。

### 适当处理错误

不同类型的错误需要不同的处理策略：

| 错误类型 | 谁修复 | 策略 | 使用时机 |
|---------|--------|------|----------|
| 暂时性错误（网络问题、速率限制） | 系统（自动） | 重试策略 | 通常重试能解决的临时故障 |
| LLM 可恢复错误（工具故障、解析问题） | LLM | 将错误存储在状态中并循环回去 | LLM 可以看到错误并调整方法 |
| 用户可修复错误（缺少信息、指令不清） | 人工 | 使用 `interrupt()` 暂停 | 需要用户输入才能继续 |
| 意外错误 | 开发者 | 让它们冒泡 | 需要调试的未知问题 |

**暂时性错误处理**
```python
from langgraph.types import RetryPolicy

workflow.add_node(
    "search_documentation",
    search_documentation,
    retry_policy=RetryPolicy(max_attempts=3, initial_interval=1.0)
)
```

**LLM 可恢复错误处理**
```python
from langgraph.types import Command

def execute_tool(state: State) -> Command[Literal["agent", "execute_tool"]]:
    try:
        result = run_tool(state['tool_call'])
        return Command(update={"tool_result": result}, goto="agent")
    except ToolError as e:
        # 让 LLM 看到问题并重试
        return Command(
            update={"tool_result": f"工具错误: {str(e)}"},
            goto="agent"
        )
```

**用户可修复错误处理**
```python
from langgraph.types import Command

def lookup_customer_history(state: State) -> Command[Literal["draft_response"]]:
    if not state.get('customer_id'):
        user_input = interrupt({
            "message": "需要客户 ID",
            "request": "请提供客户账户 ID 以查询订阅历史"
        })
        return Command(
            update={"customer_id": user_input['customer_id']},
            goto="lookup_customer_history"
        )
    # 现在继续查询
    customer_data = fetch_customer_history(state['customer_id'])
    return Command(update={"customer_history": customer_data}, goto="draft_response")
```

**意外错误处理**
```python
def send_reply(state: EmailAgentState):
    try:
        email_service.send(state["draft_response"])
    except Exception:
        raise  # 抛出意外错误
```

### 实现我们的邮件智能体节点

我们将每个节点实现为一个简单的函数。记住：节点接受状态，执行工作，并返回更新。

**读取和分类节点**
```python
from typing import Literal
from langgraph.graph import StateGraph, START, END
from langgraph.types import interrupt, Command, RetryPolicy
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage

llm = ChatOpenAI(model="gpt-4")

def read_email(state: EmailAgentState) -> dict:
    """提取和解析邮件内容"""
    # 在生产环境中，这将连接到你的邮件服务
    return {
        "messages": [HumanMessage(content=f"处理邮件: {state['email_content']}")]
    }

def classify_intent(state: EmailAgentState) -> Command[Literal["search_documentation", "human_review", "draft_response", "bug_tracking"]]:
    """使用 LLM 分类邮件意图和紧急程度，然后相应路由"""

    # 创建返回 EmailClassification 字典的结构化 LLM
    structured_llm = llm.with_structured_output(EmailClassification)

    # 按需格式化提示，不存储在状态中
    classification_prompt = f"""
    分析此客户邮件并分类：

    邮件: {state['email_content']}
    发件人: {state['sender_email']}

    提供分类，包括意图、紧急程度、主题和摘要。
    """

    # 直接获取结构化响应作为字典
    classification = structured_llm.invoke(classification_prompt)

    # 根据分类确定下一个节点
    if classification['intent'] == 'billing' or classification['urgency'] == 'critical':
        goto = "human_review"
    elif classification['intent'] in ['question', 'feature']:
        goto = "search_documentation"
    elif classification['intent'] == 'bug':
        goto = "bug_tracking"
    else:
        goto = "draft_response"

    # 将分类作为单个字典存储在状态中
    return Command(
        update={"classification": classification},
        goto=goto
    )
```

**搜索和跟踪节点**
```python
def search_documentation(state: EmailAgentState) -> Command[Literal["draft_response"]]:
    """搜索知识库获取相关信息"""

    # 从分类构建搜索查询
    classification = state.get('classification', {})
    query = f"{classification.get('intent', '')} {classification.get('topic', '')}"

    try:
        # 在此实现你的搜索逻辑
        # 存储原始搜索结果，不是格式化文本
        search_results = [
            "通过设置 > 安全 > 更改密码重置密码",
            "密码必须至少12个字符",
            "包含大写字母、小写字母、数字和符号"
        ]
    except SearchAPIError as e:
        # 对于可恢复的搜索错误，存储错误并继续
        search_results = [f"搜索暂时不可用: {str(e)}"]

    return Command(
        update={"search_results": search_results},  # 存储原始结果或错误
        goto="draft_response"
    )

def bug_tracking(state: EmailAgentState) -> Command[Literal["draft_response"]]:
    """在错误跟踪系统中创建或更新工单"""

    # 通过 API 在你的错误跟踪系统中创建工单
    ticket_id = "BUG-12345"  # 将通过 API 创建

    return Command(
        update={
            "search_results": [f"错误工单 {ticket_id} 已创建"],
            "current_step": "bug_tracked"
        },
        goto="draft_response"
    )
```

**响应节点**
```python
def draft_response(state: EmailAgentState) -> Command[Literal["human_review", "send_reply"]]:
    """使用上下文生成响应并根据质量路由"""

    classification = state.get('classification', {})

    # 按需从原始状态数据格式化上下文
    context_sections = []

    if state.get('search_results'):
        # 为提示格式化搜索结果
        formatted_docs = "\n".join([f"- {doc}" for doc in state['search_results']])
        context_sections.append(f"相关文档:\n{formatted_docs}")

    if state.get('customer_history'):
        # 为提示格式化客户数据
        context_sections.append(f"客户层级: {state['customer_history'].get('tier', 'standard')}")

    # 使用格式化上下文构建提示
    draft_prompt = f"""
    起草对此客户邮件的回复：
    {state['email_content']}

    邮件意图: {classification.get('intent', 'unknown')}
    紧急程度: {classification.get('urgency', 'medium')}

    {chr(10).join(context_sections)}

    指南：
    - 专业且乐于助人
    - 解决他们的具体问题
    - 相关时使用提供的文档
    """

    response = llm.invoke(draft_prompt)

    # 根据紧急程度和意图确定是否需要人工审核
    needs_review = (
        classification.get('urgency') in ['high', 'critical'] or
        classification.get('intent') == 'complex'
    )

    # 路由到适当的下一节点
    goto = "human_review" if needs_review else "send_reply"

    return Command(
        update={"draft_response": response.content},  # 只存储原始响应
        goto=goto
    )

def human_review(state: EmailAgentState) -> Command[Literal["send_reply", END]]:
    """使用 interrupt 暂停人工审核并根据决策路由"""

    classification = state.get('classification', {})

    # interrupt() 必须首先出现 - 它之前的任何代码都将在恢复时重新运行
    human_decision = interrupt({
        "email_id": state['email_id'],
        "original_email": state['email_content'],
        "draft_response": state['draft_response'],
        "urgency": classification.get('urgency'),
        "intent": classification.get('intent'),
        "action": "请审核并批准/编辑此响应"
    })

    # 现在处理人工决策
    if human_decision.get("approved"):
        return Command(
            update={"draft_response": human_decision.get("edited_response", state['draft_response'])},
            goto="send_reply"
        )
    else:
        # 拒绝意味着人工将直接处理
        return Command(update={}, goto=END)

def send_reply(state: EmailAgentState) -> dict:
    """发送邮件回复"""
    # 与邮件服务集成
    print(f"发送回复: {state['draft_response'][:100]}...")
    return {}
```

## 步骤 5：将它们连接起来

现在我们将节点连接成一个工作图。由于我们的节点处理自己的路由决策，我们只需要一些基本的边。

要使用 `interrupt()` 启用[人工干预](/oss/python/langgraph/interrupts)，我们需要使用[检查点](/oss/python/langgraph/persistence)进行编译以在运行之间保存状态：

```python
from langgraph.checkpoint.memory import MemorySaver
from langgraph.types import RetryPolicy

# 创建图
workflow = StateGraph(EmailAgentState)

# 添加带有适当错误处理的节点
workflow.add_node("read_email", read_email)
workflow.add_node("classify_intent", classify_intent)

# 为可能有暂时性故障的节点添加重试策略
workflow.add_node(
    "search_documentation",
    search_documentation,
    retry_policy=RetryPolicy(max_attempts=3)
)
workflow.add_node("bug_tracking", bug_tracking)
workflow.add_node("draft_response", draft_response)
workflow.add_node("human_review", human_review)
workflow.add_node("send_reply", send_reply)

# 只添加基本边
workflow.add_edge(START, "read_email")
workflow.add_edge("read_email", "classify_intent")
workflow.add_edge("send_reply", END)

# 使用检查点进行持久化编译
memory = MemorySaver()
app = workflow.compile(checkpointer=memory)
```

图结构是最小的，因为路由通过 [`Command`](https://reference.langchain.com/python/langgraph/types/#langgraph.types.Command) 对象在节点内部发生。每个节点使用类型提示如 `Command[Literal["node1", "node2"]]` 声明它可以去哪里，使流程明确且可追踪。

### 测试你的智能体

让我们用一个需要人工审核的紧急账单问题来运行我们的智能体：

```python
# 测试紧急账单问题
initial_state = {
    "email_content": "我的订阅被重复收费了！这很紧急！",
    "sender_email": "customer@example.com",
    "email_id": "email_123",
    "messages": []
}

# 使用 thread_id 进行持久化运行
config = {"configurable": {"thread_id": "customer_123"}}
result = app.invoke(initial_state, config)
# 图将在 human_review 处暂停
print(f"准备审核的草稿: {result['draft_response'][:100]}...")

# 准备就绪时，提供人工输入以恢复
from langgraph.types import Command

human_response = Command(
    resume={
        "approved": True,
        "edited_response": "我们为重复收费诚挚道歉。我已立即启动退款..."
    }
)

# 恢复执行
final_result = app.invoke(human_response, config)
print(f"邮件发送成功！")
```

当图遇到 `interrupt()` 时暂停，将所有内容保存到检查点，并等待。它可以在几天后恢复，从停止的地方准确继续。`thread_id` 确保此对话的所有状态都一起保存。

## 总结和下一步

### 关键见解

构建这个邮件智能体向我们展示了 LangGraph 的思维方式：

**分解为离散步骤**
* 每个节点做好一件事。这种分解支持流式进度更新、可以暂停和恢复的持久执行，以及清晰的调试，因为你可以在步骤之间检查状态。

**状态是共享内存**
* 存储原始数据，而不是格式化文本。这让不同节点以不同方式使用相同的信息。

**节点是函数**
* 它们接受状态，执行工作，并返回更新。当需要做出路由决策时，它们指定状态更新和下一个目的地。

**错误是流程的一部分**
* 暂时性故障重试，LLM 可恢复错误循环回去并提供上下文，用户可修复问题暂停等待输入，意外错误冒泡用于调试。

**人工输入是一等公民**
* `interrupt()` 函数无限期暂停执行，保存所有状态，并在你提供输入时从停止的地方准确恢复。当与节点中的其他操作结合时，它必须首先出现。

**图结构自然出现**
* 你定义基本连接，你的节点处理自己的路由逻辑。这使控制流明确且可追踪 - 你总是可以通过查看当前节点来理解你的智能体下一步将做什么。

### 高级考虑

**节点粒度权衡**
你可能会想：为什么不将"读取邮件"和"分类意图"合并为一个节点？或者为什么将文档搜索与起草回复分开？

答案涉及弹性与可观察性之间的权衡。

**弹性考虑：** LangGraph 的[持久执行](/oss/python/langgraph/durable-execution)在节点边界创建检查点。当工作流程在中断或故障后恢复时，它从执行停止的节点开始。较小的节点意味着更频繁的检查点，这意味着如果出现问题，重复的工作更少。如果你将多个操作合并到一个大节点中，接近末尾的失败意味着从该节点开始重新执行一切。

我们为邮件智能体选择这种分解的原因：

* **外部服务隔离：** 文档搜索和错误跟踪是单独的节点，因为它们调用外部 API。如果搜索服务缓慢或失败，我们希望将其与 LLM 调用隔离。我们可以为这些特定节点添加重试策略而不影响其他节点。

* **中间可见性：** 将"分类意图"作为自己的节点让我们可以在采取行动之前检查 LLM 的决定。这对于调试和监控很有价值 - 你可以准确看到智能体何时以及为何路由到人工审核。

* **不同的故障模式：** LLM 调用、数据库查询和邮件发送有不同的重试策略。单独的节点让你可以独立配置这些。

* **可重用性和测试：** 较小的节点更容易单独测试并在其他工作流程中重用。

不同的有效方法：你可以将"读取邮件"和"分类意图"合并为单个节点。你将失去在分类之前检查原始邮件的能力，并且在该节点出现任何故障时会重复这两个操作。对于大多数应用，单独节点的可观察性和调试好处是值得的权衡。

**控制检查点行为：** 你可以使用[持久性模式](/oss/python/langgraph/durable-execution#durability-modes)调整检查点的写入时间。默认的 `"async"` 模式在后台写入检查点以获得良好的性能，同时保持持久性。使用 `"exit"` 模式仅在完成时检查点（对于不需要中间执行恢复的长时间运行图更快），或使用 `"sync"` 模式保证在继续下一步之前写入检查点（当你需要确保状态在继续执行之前持久化时很有用）。

### 从这里去哪里

这是关于使用 LangGraph 构建智能体的思维方式的介绍。你可以用以下内容扩展这个基础：

**人工干预模式**
* 学习如何在执行前添加工具批准、批量批准和其他模式

**子图**
* 为复杂的多步骤操作创建子图

**流式传输**
* 添加流式传输以向用户显示实时进度

**可观察性**
* 使用 LangSmith 添加可观察性以进行调试和监控

**工具集成**
* 集成更多工具以进行网络搜索、数据库查询和 API 调用

**重试逻辑**
* 为失败的操作实现指数退避的重试逻辑

通过掌握这些核心概念，你现在可以构建复杂、有状态、生产就绪的智能体，能够处理现实世界的业务流程，同时保持对执行流程的完全控制和可见性。
# LangChain Agents（智能体）

## 概述

Agents（代理）将语言模型与工具相结合，创建能够推理任务、决定使用哪些工具并迭代工作以找到解决方案的系统。

### 核心概念

- **推理引擎**：语言模型负责思考和决策
- **工具调用**：代理可以调用外部工具执行操作
- **迭代过程**：代理在循环中工作直到达到停止条件
- **状态管理**：代理维护对话历史和自定义状态

## 基础 Agent 创建

### 1. 简单 Agent

```python
from langchain.agents import create_agent
from langchain.tools import tool

@tool
def search_web(query: str) -> str:
    """在网络上搜索信息。"""
    return f"关于 '{query}' 的搜索结果：相关文章、新闻和信息"

@tool
def get_weather(location: str) -> str:
    """获取指定位置的天气信息。"""
    return f"{location}的天气：晴朗，25°C"

# 创建基础 Agent
agent = create_agent(
    model="openai:gpt-4o",
    tools=[search_web, get_weather]
)
```

### 2. 调用 Agent

```python
# 基础调用
result = agent.invoke(
    {"messages": [{"role": "user", "content": "北京今天天气怎么样？"}]}
)

print(result["messages"][-1].content)
```

## 核心组件详解

### 1. 模型配置

#### 静态模型配置

```python
from langchain.agents import create_agent
from langchain_openai import ChatOpenAI

# 方法1：使用模型标识符字符串
agent1 = create_agent(
    "openai:gpt-4o",  # 自动推断为 OpenAI GPT-4o
    tools=[search_web, get_weather]
)

# 方法2：使用模型实例（推荐用于生产环境）
model = ChatOpenAI(
    model="gpt-4o",
    temperature=0.1,      # 控制创造性
    max_tokens=1000,      # 最大输出长度
    timeout=30,           # 超时设置
    # 其他参数...
)

agent2 = create_agent(
    model=model,
    tools=[search_web, get_weather]
)
```

#### 动态模型选择

```python
from langchain.agents.middleware import wrap_model_call, ModelRequest
from langchain_openai import ChatOpenAI

# 定义不同模型
basic_model = ChatOpenAI(model="gpt-4o-mini")  # 经济型模型
advanced_model = ChatOpenAI(model="gpt-4o")    # 高级模型

@wrap_model_call
def dynamic_model_selection(request: ModelRequest, handler):
    """基于对话复杂度选择模型"""
    messages = request.state["messages"]
    message_count = len(messages)
    
    # 复杂对话使用高级模型
    if message_count > 5 or any("复杂" in msg.content for msg in messages if hasattr(msg, 'content')):
        request.model = advanced_model
    else:
        request.model = basic_model
    
    return handler(request)

# 创建支持动态模型选择的 Agent
agent = create_agent(
    model=basic_model,  # 默认模型
    tools=[search_web, get_weather],
    middleware=[dynamic_model_selection]
)
```

### 2. 工具系统

#### 基础工具定义

```python
from langchain.tools import tool
from datetime import datetime

@tool
def calculator(expression: str) -> str:
    """计算数学表达式。"""
    try:
        result = eval(expression)
        return f"{expression} = {result}"
    except Exception as e:
        return f"计算错误: {str(e)}"

@tool
def get_current_time(timezone: str = "UTC") -> str:
    """获取指定时区的当前时间。"""
    now = datetime.now()
    return f"{timezone}时区当前时间: {now.strftime('%Y-%m-%d %H:%M:%S')}"

@tool
def search_products(query: str, category: str = "all") -> str:
    """搜索产品信息。"""
    return f"在 '{category}' 类别中找到关于 '{query}' 的产品"

# 创建包含多个工具的 Agent
agent = create_agent(
    model="openai:gpt-4o",
    tools=[calculator, get_current_time, search_products]
)
```

#### 工具错误处理

```python
from langchain.agents.middleware import wrap_tool_call
from langchain_core.messages import ToolMessage

@wrap_tool_call
def handle_tool_errors(request, handler):
    """自定义工具错误处理"""
    try:
        return handler(request)
    except Exception as e:
        # 返回友好的错误信息
        error_message = f"工具执行失败：{str(e)}。请检查输入参数并重试。"
        
        return ToolMessage(
            content=error_message,
            tool_call_id=request.tool_call["id"]
        )

agent = create_agent(
    model="openai:gpt-4o",
    tools=[calculator, get_current_time],
    middleware=[handle_tool_errors]
)
```

### 3. 系统提示词

#### 静态系统提示词

```python
agent = create_agent(
    model="openai:gpt-4o",
    tools=[search_web, get_weather],
    system_prompt="""你是一个专业、友好的助手。请遵循以下指导原则：
    1. 回答要准确、简洁
    2. 使用工具获取最新信息
    3. 如果用户问题涉及专业领域，请说明信息来源
    4. 对不确定的信息要明确说明
    """
)
```

#### 动态系统提示词

```python
from langchain.agents.middleware import dynamic_prompt, ModelRequest
from typing import TypedDict

class UserContext(TypedDict):
    user_level: str  # "beginner", "intermediate", "expert"
    language: str

@dynamic_prompt
def adaptive_system_prompt(request: ModelRequest) -> str:
    """基于用户水平和语言生成动态系统提示词"""
    context = request.runtime.context
    user_level = context.get("user_level", "beginner")
    language = context.get("language", "zh-CN")
    
    base_prompt = "你是一个有帮助的AI助手。"
    
    # 根据用户水平调整提示词
    level_prompts = {
        "beginner": "请用简单易懂的语言解释概念，避免专业术语。",
        "intermediate": "提供平衡的解答，包含基本概念和一些进阶信息。",
        "expert": "提供详细的技术分析，可以使用专业术语。"
    }
    
    level_prompt = level_prompts.get(user_level, level_prompts["beginner"])
    
    # 语言特定提示
    if language == "zh-CN":
        language_prompt = "请使用中文回复，保持语言的自然和流畅。"
    else:
        language_prompt = "Please respond in natural and fluent language."
    
    return f"{base_prompt} {level_prompt} {language_prompt}"

# 创建支持动态提示词的 Agent
agent = create_agent(
    model="openai:gpt-4o",
    tools=[search_web, get_weather],
    middleware=[adaptive_system_prompt],
    context_schema=UserContext
)

# 使用上下文调用
result = agent.invoke(
    {"messages": [{"role": "user", "content": "解释机器学习"}]},
    context={"user_level": "beginner", "language": "zh-CN"}
)
```

## 高级功能

### 1. 结构化输出

```python
from pydantic import BaseModel, Field
from langchain.agents import create_agent
from langchain.agents.structured_output import ToolStrategy

class ProductReview(BaseModel):
    """产品评价分析"""
    product_name: str = Field(description="产品名称")
    rating: int = Field(description="评分(1-5)", ge=1, le=5)
    positive_points: list[str] = Field(description="优点列表")
    negative_points: list[str] = Field(description="缺点列表")
    summary: str = Field(description="总体评价总结")

class CustomerInfo(BaseModel):
    """客户信息提取"""
    name: str = Field(description="客户姓名")
    email: str = Field(description="邮箱地址")
    phone: str = Field(description="电话号码")
    interests: list[str] = Field(description="兴趣列表")

# 使用 ToolStrategy 实现结构化输出
agent = create_agent(
    model="openai:gpt-4o",
    tools=[search_web],
    response_format=ToolStrategy(ProductReview)  # 或 CustomerInfo
)

# 调用并获取结构化输出
result = agent.invoke({
    "messages": [{
        "role": "user", 
        "content": "分析这个产品评价：'iPhone 15 Pro 太棒了！相机质量优秀，电池续航也很好，就是价格有点贵。评分5/5'"
    }]
})

structured_data = result["structured_response"]
print(f"产品: {structured_data.product_name}")
print(f"评分: {structured_data.rating}")
print(f"优点: {', '.join(structured_data.positive_points)}")
```

### 2. 内存管理

#### 自定义状态管理

```python
from typing import TypedDict, List, Optional
from langchain.agents import AgentState, create_agent

class CustomAgentState(AgentState):
    """自定义 Agent 状态"""
    user_preferences: dict
    conversation_topics: List[str]
    interaction_count: int = 0
    last_active: Optional[str] = None

# 通过 state_schema 定义状态
agent = create_agent(
    model="openai:gpt-4o",
    tools=[search_web, get_weather],
    state_schema=CustomAgentState
)

# 使用自定义状态调用
result = agent.invoke({
    "messages": [{"role": "user", "content": "我喜欢技术类内容"}],
    "user_preferences": {"category": "technology", "detail_level": "high"},
    "conversation_topics": ["AI", "编程"],
    "interaction_count": 1
})
```

#### 通过中间件管理状态

```python
from langchain.agents.middleware import AgentMiddleware
from typing import Any

class UserPreferencesMiddleware(AgentMiddleware):
    """用户偏好管理中间件"""
    state_schema = CustomAgentState
    
    def before_model(self, state: CustomAgentState, runtime) -> dict[str, Any] | None:
        """在模型调用前处理用户偏好"""
        preferences = state.get("user_preferences", {})
        
        # 基于用户偏好调整行为
        if preferences.get("detail_level") == "high":
            # 可以在这里修改消息或添加上下文
            pass
            
        return None
    
    def after_model(self, state: CustomAgentState, runtime) -> dict[str, Any] | None:
        """在模型调用后更新交互统计"""
        return {
            "interaction_count": state.get("interaction_count", 0) + 1,
            "last_active": "2024-01-01T10:00:00"  # 实际使用中应为当前时间
        }

agent = create_agent(
    model="openai:gpt-4o",
    tools=[search_web, get_weather],
    middleware=[UserPreferencesMiddleware()]
)
```

### 3. 流式传输

```python
def stream_agent_progress():
    """流式传输 Agent 执行进度"""
    print("开始流式传输 Agent 执行...")
    
    for chunk in agent.stream(
        {"messages": [{"role": "user", "content": "搜索AI最新发展并总结要点"}]},
        stream_mode="values"  # 也可以使用 "updates" 或 "messages"
    ):
        # 获取最新消息
        latest_message = chunk["messages"][-1]
        
        # 处理AI回复
        if hasattr(latest_message, 'content') and latest_message.content:
            print(f"🤖 AI: {latest_message.content}")
        
        # 处理工具调用
        elif hasattr(latest_message, 'tool_calls') and latest_message.tool_calls:
            for tool_call in latest_message.tool_calls:
                print(f"🛠️  调用工具: {tool_call['name']}")
                print(f"   参数: {tool_call['args']}")

# 调用流式传输
stream_agent_progress()
```

### 4. 中间件系统

```python
from langchain.agents.middleware import before_model, after_model, wrap_tool_call
from langchain.messages import RemoveMessage
from langgraph.graph.message import REMOVE_ALL_MESSAGES

@before_model
def trim_long_conversations(state, runtime):
    """修剪过长的对话历史"""
    messages = state["messages"]
    
    if len(messages) > 10:
        # 保留系统消息和最近5条消息
        system_messages = [msg for msg in messages if msg.type == "system"]
        recent_messages = messages[-5:]
        
        return {
            "messages": [
                RemoveMessage(id=REMOVE_ALL_MESSAGES),
                *system_messages,
                *recent_messages
            ]
        }
    
    return None

@after_model
def validate_response_content(state, runtime):
    """验证模型响应内容"""
    last_message = state["messages"][-1]
    
    # 检查是否包含不当内容
    inappropriate_keywords = ["暴力", "仇恨", "歧视"]
    if any(keyword in last_message.content for keyword in inappropriate_keywords):
        return {
            "messages": [
                RemoveMessage(id=last_message.id),
                *state["messages"][:-1]
            ]
        }
    
    return None

@wrap_tool_call
def log_tool_execution(request, handler):
    """记录工具执行日志"""
    tool_name = request.tool_call["name"]
    tool_args = request.tool_call["args"]
    
    print(f"📝 开始执行工具: {tool_name}")
    print(f"   参数: {tool_args}")
    
    start_time = time.time()
    result = handler(request)
    execution_time = time.time() - start_time
    
    print(f"✅ 工具执行完成: {tool_name} (耗时: {execution_time:.2f}s)")
    
    return result

# 创建包含多个中间件的 Agent
agent = create_agent(
    model="openai:gpt-4o",
    tools=[search_web, get_weather, calculator],
    middleware=[
        trim_long_conversations,
        validate_response_content,
        log_tool_execution
    ]
)
```

## 实际应用场景

### 场景1：客户服务 Agent

```python
from langchain.tools import tool
from datetime import datetime

class CustomerServiceAgent:
    """客户服务 Agent"""
    
    def __init__(self):
        self.agent = self._create_agent()
    
    @tool
    def check_order_status(self, order_id: str) -> str:
        """检查订单状态。"""
        # 模拟订单数据库
        orders = {
            "ORD001": {"status": "已发货", "tracking": "SF123456789"},
            "ORD002": {"status": "处理中", "tracking": None},
            "ORD003": {"status": "已送达", "tracking": "SF987654321"}
        }
        
        if order_id in orders:
            order = orders[order_id]
            result = f"订单 {order_id} 状态: {order['status']}"
            if order['tracking']:
                result += f"\n物流单号: {order['tracking']}"
            return result
        return f"未找到订单 {order_id}"
    
    @tool
    def process_refund(self, order_id: str, reason: str) -> str:
        """处理退款申请。"""
        return f"订单 {order_id} 的退款申请已提交。原因: {reason}\n预计3-5个工作日处理完成。"
    
    @tool
    def get_faq(self, category: str) -> str:
        """获取常见问题解答。"""
        faqs = {
            "shipping": "配送时间：普通快递3-5天，加急快递1-2天",
            "returns": "退换货政策：7天无理由退货，30天质量问题的换货",
            "payment": "支付方式：支持支付宝、微信支付、银行卡"
        }
        return faqs.get(category, "暂无该类别常见问题")
    
    def _create_agent(self):
        """创建客户服务 Agent"""
        return create_agent(
            model="openai:gpt-4o",
            tools=[self.check_order_status, self.process_refund, self.get_faq],
            system_prompt="""你是一个专业的客户服务代表。请遵循以下原则：
            1. 始终保持友好和专业的态度
            2. 准确回答客户问题
            3. 使用工具获取最新信息
            4. 对于复杂问题，提供清晰的后续步骤
            5. 如果无法解决问题，建议联系人工客服
            """,
            state_schema=CustomAgentState
        )
    
    def handle_customer_query(self, query: str, user_id: str):
        """处理客户查询"""
        return self.agent.invoke({
            "messages": [{"role": "user", "content": query}],
            "user_preferences": {"user_id": user_id}
        })

# 使用示例
service_agent = CustomerServiceAgent()
result = service_agent.handle_customer_query("我的订单ORD001状态如何？", "user123")
```

### 场景2：数据分析 Agent

```python
import pandas as pd
import numpy as np
from io import StringIO

class DataAnalysisAgent:
    """数据分析 Agent"""
    
    def __init__(self):
        self.agent = self._create_agent()
        self.current_dataset = None
    
    @tool
    def load_csv_data(self, csv_content: str) -> str:
        """加载CSV数据。"""
        try:
            self.current_dataset = pd.read_csv(StringIO(csv_content))
            stats = {
                "行数": len(self.current_dataset),
                "列数": len(self.current_dataset.columns),
                "列名": list(self.current_dataset.columns)
            }
            return f"数据加载成功！统计信息: {stats}"
        except Exception as e:
            return f"数据加载失败: {str(e)}"
    
    @tool
    def describe_dataset(self) -> str:
        """描述数据集基本信息。"""
        if self.current_dataset is None:
            return "请先加载数据"
        
        description = self.current_dataset.describe()
        return f"数据集描述:\n{description}"
    
    @tool
    def calculate_correlation(self, column1: str, column2: str) -> str:
        """计算两列之间的相关性。"""
        if self.current_dataset is None:
            return "请先加载数据"
        
        if column1 not in self.current_dataset.columns or column2 not in self.current_dataset.columns:
            return "指定的列不存在"
        
        correlation = self.current_dataset[column1].corr(self.current_dataset[column2])
        return f"{column1} 和 {column2} 的相关性: {correlation:.3f}"
    
    @tool
    def filter_data(self, condition: str) -> str:
        """根据条件过滤数据。"""
        if self.current_dataset is None:
            return "请先加载数据"
        
        try:
            filtered_data = self.current_dataset.query(condition)
            return f"过滤后数据: {len(filtered_data)} 行"
        except Exception as e:
            return f"过滤条件错误: {str(e)}"
    
    def _create_agent(self):
        """创建数据分析 Agent"""
        return create_agent(
            model="openai:gpt-4o",
            tools=[
                self.load_csv_data, 
                self.describe_dataset, 
                self.calculate_correlation,
                self.filter_data
            ],
            system_prompt="""你是一个数据分析专家。请帮助用户：
            1. 加载和分析数据
            2. 提供数据统计信息
            3. 计算指标和相关性
            4. 解释分析结果的含义
            5. 用通俗易懂的语言解释技术概念
            """
        )

# 使用示例
analysis_agent = DataAnalysisAgent()

# 模拟CSV数据
sample_data = """name,age,salary,department
张三,25,50000,技术部
李四,30,60000,销售部
王五,35,70000,技术部
赵六,28,55000,市场部"""

result = analysis_agent.agent.invoke({
    "messages": [{"role": "user", "content": f"请分析以下数据:\n{sample_data}"}]
})
```

### 场景3：研究助手 Agent

```python
class ResearchAssistantAgent:
    """研究助手 Agent"""
    
    def __init__(self):
        self.agent = self._create_agent()
        self.research_topics = []
    
    @tool
    def search_academic_papers(self, topic: str, max_results: int = 5) -> str:
        """搜索学术论文。"""
        # 模拟学术搜索
        papers = [
            f"《{topic}的最新研究进展》- 作者A et al.",
            f"《{topic}在实践中的应用》- 作者B et al.", 
            f"《{topic}的未来发展趋势》- 作者C et al."
        ]
        return f"找到 {len(papers)} 篇相关论文:\n" + "\n".join(papers[:max_results])
    
    @tool
    def summarize_research_topic(self, topic: str) -> str:
        """总结研究主题。"""
        self.research_topics.append(topic)
        return f"""
        {topic} 研究总结：
        1. 核心概念：{topic}涉及多个交叉学科领域
        2. 当前热点：AI驱动的{topic}研究正在兴起
        3. 主要挑战：数据质量和算法可解释性
        4. 未来方向：自动化、智能化{topic}解决方案
        """
    
    @tool
    def compare_topics(self, topic1: str, topic2: str) -> str:
        """比较两个研究主题。"""
        return f"""
        {topic1} vs {topic2} 比较：
        
        相似点：
        - 都是前沿技术领域
        - 都需要跨学科知识
        - 都有广泛的应用场景
        
        不同点：
        - {topic1}更注重理论发展
        - {topic2}更注重实践应用
        - 技术栈和研究方法有所不同
        """
    
    @tool
    def generate_research_questions(self, topic: str) -> str:
        """生成研究问题。"""
        questions = [
            f"{topic}如何影响相关行业？",
            f"{topic}面临的主要技术挑战是什么？",
            f"{topic}的未来发展方向有哪些？",
            f"如何评估{topic}的实际效果？"
        ]
        return "潜在研究问题:\n" + "\n".join([f"{i+1}. {q}" for i, q in enumerate(questions)])
    
    def _create_agent(self):
        """创建研究助手 Agent"""
        return create_agent(
            model="openai:gpt-4o",
            tools=[
                self.search_academic_papers,
                self.summarize_research_topic, 
                self.compare_topics,
                self.generate_research_questions
            ],
            system_prompt="""你是一个专业的研究助手。请帮助用户：
            1. 搜索相关学术文献
            2. 总结研究主题和趋势
            3. 比较不同研究方向
            4. 生成有价值的研究问题
            5. 提供研究方法和建议
            
            请保持专业性和准确性，引用可靠的来源。
            """
        )

# 使用示例
research_agent = ResearchAssistantAgent()
result = research_agent.agent.invoke({
    "messages": [{"role": "user", "content": "帮我研究人工智能在医疗领域的应用"}]
})
```

## 最佳实践

### 1. Agent 设计原则

```python
def create_well_designed_agent():
    """创建良好设计的 Agent"""
    
    # 1. 明确的工具定义
    @tool
    def specific_tool(param1: str, param2: int = 10) -> str:
        """执行特定任务的工具。
        
        Args:
            param1: 主要参数描述
            param2: 可选参数，默认值10
        """
        return f"处理结果: {param1} * {param2}"
    
    # 2. 清晰的系统提示词
    system_prompt = """
    你是一个专业的助手。请遵循：
    - 准确回答，不编造信息
    - 使用工具获取真实数据
    - 对不确定的内容要说明
    - 保持友好和专业
    """
    
    # 3. 适当的中间件
    @before_model
    def ensure_proper_context(state, runtime):
        """确保适当的上下文"""
        messages = state["messages"]
        if len(messages) > 0:
            last_message = messages[-1]
            # 可以在这里添加上下文验证逻辑
            pass
        return None
    
    # 创建 Agent
    return create_agent(
        model="openai:gpt-4o",
        tools=[specific_tool],
        system_prompt=system_prompt,
        middleware=[ensure_proper_context]
    )
```

### 2. 错误处理策略

```python
class RobustAgent:
    """健壮的 Agent 实现"""
    
    def __init__(self):
        self.agent = self._create_robust_agent()
    
    def _create_robust_agent(self):
        """创建健壮的 Agent"""
        
        @wrap_tool_call
        def comprehensive_error_handling(request, handler):
            """全面的错误处理"""
            try:
                # 参数验证
                tool_call = request.tool_call
                if not self._validate_tool_inputs(tool_call):
                    return ToolMessage(
                        content="参数验证失败，请检查输入格式",
                        tool_call_id=tool_call["id"]
                    )
                
                return handler(request)
                
            except Exception as e:
                # 记录错误
                print(f"工具执行错误: {e}")
                
                # 返回用户友好的错误信息
                return ToolMessage(
                    content="服务暂时不可用，请稍后重试",
                    tool_call_id=request.tool_call["id"]
                )
        
        @wrap_model_call  
        def model_fallback(request, handler):
            """模型调用降级策略"""
            try:
                return handler(request)
            except Exception as e:
                # 如果主要模型失败，可以在这里切换到备用模型
                print(f"模型调用失败: {e}")
                raise  # 或实现降级逻辑
        
        return create_agent(
            model="openai:gpt-4o",
            tools=[search_web, get_weather],
            middleware=[comprehensive_error_handling, model_fallback]
        )
    
    def _validate_tool_inputs(self, tool_call):
        """验证工具输入参数"""
        # 实现参数验证逻辑
        return True
```

### 3. 性能优化

```python
class OptimizedAgent:
    """性能优化的 Agent"""
    
    def __init__(self):
        self.agent = self._create_optimized_agent()
        self.response_cache = {}  # 简单缓存
    
    def _create_optimized_agent(self):
        """创建性能优化的 Agent"""
        
        @before_model
        def check_cache(state, runtime):
            """检查缓存以避免重复处理"""
            user_message = state["messages"][-1].content
            cache_key = hash(user_message)
            
            if cache_key in self.response_cache:
                # 返回缓存响应
                return self.response_cache[cache_key]
            
            return None
        
        @after_model
        def update_cache(state, runtime):
            """更新响应缓存"""
            if len(state["messages"]) > 0:
                last_message = state["messages"][-1]
                user_message = state["messages"][-2].content  # 假设上一条是用户消息
                cache_key = hash(user_message)
                self.response_cache[cache_key] = {"messages": [last_message]}
            
            return None
        
        @wrap_tool_call
        def timeout_protection(request, handler):
            """工具调用超时保护"""
            import signal
            import time
            
            def timeout_handler(signum, frame):
                raise TimeoutError("工具执行超时")
            
            # 设置超时（5秒）
            signal.signal(signal.SIGALRM, timeout_handler)
            signal.alarm(5)
            
            try:
                result = handler(request)
                signal.alarm(0)  # 取消超时
                return result
            except TimeoutError:
                return ToolMessage(
                    content="工具执行超时，请简化请求或稍后重试",
                    tool_call_id=request.tool_call["id"]
                )
        
        return create_agent(
            model="openai:gpt-4o",
            tools=[search_web, get_weather],
            middleware=[check_cache, update_cache, timeout_protection]
        )
```

## 总结

LangChain Agents 提供了强大的AI应用构建能力：

- **灵活配置**：支持多种模型、工具和提示词配置
- **强大扩展**：通过中间件系统实现高度定制化
- **状态管理**：内置对话状态和自定义状态管理
- **生产就绪**：包含错误处理、性能优化等生产级特性
- **实时交互**：支持流式传输和进度监控

通过合理设计和使用 Agents，可以构建出能够处理复杂任务、与外部系统交互并提供智能服务的AI应用系统。
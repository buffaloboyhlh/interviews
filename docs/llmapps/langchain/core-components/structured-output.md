# LangChain 结构化输出

## 概述

结构化输出允许 Agent 以特定、可预测的格式返回数据。与解析自然语言响应不同，你可以获得 JSON 对象、Pydantic 模型或数据类形式的结构化数据，这些数据可以直接在你的应用程序中使用。

### 核心优势

- **可预测性**：数据格式固定，便于后续处理
- **类型安全**：自动验证和类型检查
- **直接集成**：无需手动解析，可直接在代码中使用
- **错误处理**：内置验证和重试机制

## 基础用法

LangChain 的 `create_agent` 自动处理结构化输出。用户设置所需的结构化输出模式，当模型生成结构化数据时，它会被捕获、验证并返回到 Agent 状态的 `'structured_response'` 键中。

```python
from langchain.agents import create_agent

# 基本语法
agent = create_agent(
    model="openai:gpt-4o",
    tools=[...],
    response_format=YourSchema  # 结构化输出模式
)
```

## 响应格式策略

### 1. ProviderStrategy（提供者策略）

当模型提供商原生支持结构化输出时使用（目前支持 OpenAI 和 Grok），这是最可靠的方法。

```python
# LangChain 会自动选择 ProviderStrategy
agent = create_agent(
    model="openai:gpt-4o",
    tools=tools,
    response_format=ContactInfo  # 自动选择最佳策略
)
```

### 2. ToolStrategy（工具调用策略）

对于不支持原生结构化输出的模型，使用工具调用来实现相同效果。

```python
from langchain.agents.structured_output import ToolStrategy

agent = create_agent(
    model="anthropic:claude-3-5-sonnet",
    tools=tools,
    response_format=ToolStrategy(ContactInfo)
)
```

### 3. 自动选择策略

直接传递模式类型，LangChain 会自动选择最佳策略：

```python
# LangChain 根据模型能力自动选择
agent = create_agent(
    model="openai:gpt-4o",  # 支持原生结构化输出 → ProviderStrategy
    tools=tools,
    response_format=ContactInfo  # 自动选择
)
```

## 模式定义方式

### 1. Pydantic 模型（推荐）

```python
from pydantic import BaseModel, Field
from typing import Optional, List
from langchain.agents import create_agent

class ContactInfo(BaseModel):
    """联系人信息"""
    name: str = Field(description="姓名")
    email: str = Field(description="邮箱地址")
    phone: Optional[str] = Field(description="电话号码")
    tags: List[str] = Field(description="标签列表")

class ProductReview(BaseModel):
    """产品评价分析"""
    rating: int = Field(description="评分(1-5)", ge=1, le=5)
    sentiment: str = Field(description="情感倾向")
    key_points: List[str] = Field(description="关键点")
    summary: str = Field(description="总结")

# 使用示例
agent = create_agent(
    model="openai:gpt-4o",
    tools=[],
    response_format=ContactInfo
)

result = agent.invoke({
    "messages": [{
        "role": "user", 
        "content": "提取联系人：张三，邮箱zhangsan@example.com，电话13800138000，标签：VIP客户、技术部"
    }]
})

print(result["structured_response"])
# ContactInfo(name='张三', email='zhangsan@example.com', phone='13800138000', tags=['VIP客户', '技术部'])
```

### 2. Dataclass

```python
from dataclasses import dataclass
from typing import List, Optional

@dataclass
class MeetingNotes:
    """会议记录"""
    topic: str                    # 会议主题
    participants: List[str]       # 参会人员
    decisions: List[str]          # 决策事项
    action_items: List[str]       # 行动项
    next_meeting: Optional[str]   # 下次会议时间

agent = create_agent(
    model="openai:gpt-4o",
    tools=[],
    response_format=MeetingNotes
)
```

### 3. TypedDict

```python
from typing_extensions import TypedDict, List, Optional

class CustomerOrder(TypedDict):
    """客户订单"""
    order_id: str
    customer_name: str
    items: List[str]
    total_amount: float
    status: str

agent = create_agent(
    model="openai:gpt-4o",
    tools=[],
    response_format=CustomerOrder
)
```

### 4. JSON Schema

```python
order_schema = {
    "type": "object",
    "description": "客户订单信息",
    "properties": {
        "order_id": {"type": "string", "description": "订单ID"},
        "customer_name": {"type": "string", "description": "客户姓名"},
        "items": {
            "type": "array",
            "items": {"type": "string"},
            "description": "商品列表"
        },
        "total_amount": {"type": "number", "description": "总金额"},
        "status": {"type": "string", "description": "订单状态"}
    },
    "required": ["order_id", "customer_name", "items", "total_amount"]
}

agent = create_agent(
    model="openai:gpt-4o",
    tools=[],
    response_format=order_schema
)
```

## 高级功能

### 1. 联合类型（多模式选择）

```python
from typing import Union
from pydantic import BaseModel, Field

class ProductQuery(BaseModel):
    """产品查询"""
    product_name: str = Field(description="产品名称")
    features: List[str] = Field(description="产品特性")

class TechnicalSupport(BaseModel):
    """技术支持请求"""
    issue_type: str = Field(description="问题类型")
    severity: str = Field(description="严重程度")
    description: str = Field(description="问题描述")

class SalesInquiry(BaseModel):
    """销售咨询"""
    interest_level: str = Field(description="兴趣等级")
    budget_range: str = Field(description="预算范围")
    timeline: str = Field(description="时间线")

# 模型根据上下文选择最合适的模式
agent = create_agent(
    model="openai:gpt-4o",
    tools=[],
    response_format=ToolStrategy(
        Union[ProductQuery, TechnicalSupport, SalesInquiry]
    )
)

# 模型会自动选择 TechnicalSupport
result = agent.invoke({
    "messages": [{
        "role": "user", 
        "content": "我的应用程序无法启动，显示错误代码500，需要紧急帮助"
    }]
})
```

### 2. 自定义工具消息内容

```python
from langchain.agents.structured_output import ToolStrategy

class BugReport(BaseModel):
    """Bug报告"""
    title: str = Field(description="问题标题")
    severity: str = Field(description="严重程度")
    steps_to_reproduce: List[str] = Field(description="重现步骤")
    expected_behavior: str = Field(description="预期行为")
    actual_behavior: str = Field(description="实际行为")

agent = create_agent(
    model="openai:gpt-4o",
    tools=[],
    response_format=ToolStrategy(
        schema=BugReport,
        tool_message_content="✅ Bug报告已成功记录到系统！"
    )
)
```

### 3. 复杂嵌套结构

```python
from typing import List, Optional
from pydantic import BaseModel, Field

class Address(BaseModel):
    """地址信息"""
    street: str = Field(description="街道")
    city: str = Field(description="城市")
    country: str = Field(description="国家")
    postal_code: str = Field(description="邮编")

class OrderItem(BaseModel):
    """订单项"""
    product_name: str = Field(description="商品名称")
    quantity: int = Field(description="数量")
    price: float = Field(description="单价")

class CustomerOrder(BaseModel):
    """完整客户订单"""
    order_id: str = Field(description="订单ID")
    customer_name: str = Field(description="客户姓名")
    shipping_address: Address = Field(description="配送地址")
    items: List[OrderItem] = Field(description="订单项列表")
    total_amount: float = Field(description="总金额")

agent = create_agent(
    model="openai:gpt-4o",
    tools=[],
    response_format=CustomerOrder
)
```

## 错误处理

### 1. 基本错误处理

```python
from langchain.agents.structured_output import ToolStrategy

class ProductRating(BaseModel):
    rating: int = Field(description="评分(1-5)", ge=1, le=5)
    comment: str = Field(description="评价内容")

# 默认错误处理（自动重试）
agent = create_agent(
    model="openai:gpt-4o",
    tools=[],
    response_format=ToolStrategy(ProductRating)  # handle_errors=True
)
```

### 2. 自定义错误消息

```python
agent = create_agent(
    model="openai:gpt-4o",
    tools=[],
    response_format=ToolStrategy(
        schema=ProductRating,
        handle_errors="请提供1-5分的评分和有效的评价内容。"
    )
)
```

### 3. 特定异常处理

```python
# 只处理特定类型的异常
agent = create_agent(
    model="openai:gpt-4o",
    tools=[],
    response_format=ToolStrategy(
        schema=ProductRating,
        handle_errors=ValueError  # 只对 ValueError 重试
    )
)

# 处理多种异常类型
agent = create_agent(
    model="openai:gpt-4o",
    tools=[],
    response_format=ToolStrategy(
        schema=ProductRating,
        handle_errors=(ValueError, TypeError)  # 对两种异常重试
    )
)
```

### 4. 自定义错误处理函数

```python
def custom_error_handler(error: Exception) -> str:
    if "rating" in str(error):
        return "评分必须在1-5之间，请修正。"
    elif "comment" in str(error):
        return "评价内容不能为空，请补充。"
    else:
        return f"格式错误：{str(error)}"

agent = create_agent(
    model="openai:gpt-4o",
    tools=[],
    response_format=ToolStrategy(
        schema=ProductRating,
        handle_errors=custom_error_handler
    )
)
```

### 5. 禁用错误处理

```python
# 所有错误都会直接抛出
agent = create_agent(
    model="openai:gpt-4o",
    tools=[],
    response_format=ToolStrategy(
        schema=ProductRating,
        handle_errors=False  # 不进行错误处理
    )
)
```

## 实际应用场景

### 场景1：客户服务自动化

```python
from typing import Literal
from pydantic import BaseModel, Field

class CustomerServiceTicket(BaseModel):
    """客户服务工单"""
    ticket_type: Literal["technical", "billing", "general", "complaint"] = Field(description="工单类型")
    priority: Literal["low", "medium", "high", "urgent"] = Field(description优先级")
    customer_issue: str = Field(description="客户问题描述")
    suggested_solution: str = Field(description="建议解决方案")
    follow_up_required: bool = Field(description="是否需要跟进")

class CustomerServiceAgent:
    def __init__(self):
        self.agent = create_agent(
            model="openai:gpt-4o",
            tools=[],  # 可以集成知识库搜索等工具
            response_format=CustomerServiceTicket
        )
    
    def process_customer_message(self, message: str):
        result = self.agent.invoke({
            "messages": [{"role": "user", "content": message}]
        })
        
        ticket = result["structured_response"]
        self._route_ticket(ticket)
        return ticket
    
    def _route_ticket(self, ticket: CustomerServiceTicket):
        # 根据工单类型和优先级路由到不同团队
        if ticket.ticket_type == "technical" and ticket.priority in ["high", "urgent"]:
            print("🚨 紧急技术问题 - 路由到技术团队")
        elif ticket.ticket_type == "billing":
            print("💰 账单问题 - 路由到财务团队")
        # ... 其他路由逻辑
```

### 场景2：内容分析和提取

```python
from datetime import datetime
from typing import List, Optional

class NewsArticle(BaseModel):
    """新闻文章分析"""
    headline: str = Field(description="标题")
    summary: str = Field(description="摘要")
    key_entities: List[str] = Field(description="关键实体")
    sentiment: str = Field(description="情感倾向")
    categories: List[str] = Field(description="分类")
    publish_date: Optional[str] = Field(description="发布日期")

class ContentAnalyzer:
    def __init__(self):
        self.agent = create_agent(
            model="openai:gpt-4o",
            tools=[],
            response_format=NewsArticle
        )
    
    def analyze_article(self, content: str):
        result = self.agent.invoke({
            "messages": [{
                "role": "user", 
                "content": f"分析以下新闻内容：\n\n{content}"
            }]
        })
        return result["structured_response"]

# 使用示例
analyzer = ContentAnalyzer()
article_content = """
今日，某科技公司发布了新一代AI芯片，性能提升200%。
该芯片采用5nm工艺，功耗降低30%。CEO张三表示，
这将推动人工智能应用的快速发展。
"""

analysis = analyzer.analyze_article(article_content)
print(f"标题: {analysis.headline}")
print(f"情感: {analysis.sentiment}")
print(f"关键实体: {analysis.key_entities}")
```

### 场景3：电子商务产品信息提取

```python
from typing import List, Optional
from decimal import Decimal

class ProductInfo(BaseModel):
    """产品信息提取"""
    name: str = Field(description="产品名称")
    brand: Optional[str] = Field(description="品牌")
    price: Optional[Decimal] = Field(description="价格")
    features: List[str] = Field(description="产品特性")
    specifications: dict = Field(description="规格参数")
    availability: bool = Field(description="是否有货")

class EcommerceParser:
    def __init__(self):
        self.agent = create_agent(
            model="openai:gpt-4o",
            tools=[],
            response_format=ProductInfo
        )
    
    def parse_product_description(self, description: str):
        result = self.agent.invoke({
            "messages": [{
                "role": "user",
                "content": f"从以下描述中提取产品信息：\n\n{description}"
            }]
        })
        return result["structured_response"]

# 使用示例
parser = EcommerceParser()
product_desc = """
苹果 iPhone 15 Pro Max，256GB，钛金属材质
价格：¥9,999
特性：A17 Pro芯片、4800万像素主摄、5倍光学变焦
规格：重量221g，6.7英寸超视网膜XDR显示屏
库存充足，次日达
"""

product_info = parser.parse_product_description(product_desc)
```

## 最佳实践

### 1. 设计有效的模式

```python
# ✅ 好的模式设计
class EffectiveSchema(BaseModel):
    # 明确的字段描述
    name: str = Field(description="用户姓名")
    # 适当的约束
    age: int = Field(description="年龄", ge=0, le=150)
    # 合理的可选字段
    email: Optional[str] = Field(description="邮箱地址")
    # 清晰的枚举值
    status: Literal["active", "inactive", "pending"] = Field(description="状态")

# ❌ 避免的模式设计
class PoorSchema(BaseModel):
    # 描述不清晰
    field1: str
    # 约束不明确
    field2: int
    # 过于复杂的嵌套
    data: Dict[str, Any]
```

### 2. 错误处理策略

```python
# 根据使用场景选择合适的错误处理
def create_robust_agent(schema):
    return create_agent(
        model="openai:gpt-4o",
        tools=[],
        response_format=ToolStrategy(
            schema=schema,
            handle_errors=lambda e: f"格式错误，请重新输入：{str(e)}"
        ),
        system_prompt="请严格按照要求的格式输出数据。"
    )
```

### 3. 性能优化

```python
# 重用 Agent 实例
class StructuredOutputService:
    def __init__(self):
        self._agents = {}
    
    def get_agent(self, schema):
        schema_key = str(schema)
        if schema_key not in self._agents:
            self._agents[schema_key] = create_agent(
                model="openai:gpt-4o",
                tools=[],
                response_format=schema
            )
        return self._agents[schema_key]
```

## 故障排除

### 常见问题及解决方案

1. **模型不返回结构化数据**
     - 检查模型是否支持工具调用
     - 验证模式定义是否清晰
     - 添加更详细的字段描述

2. **验证错误频繁**
     - 简化模式结构
     - 放宽字段约束
     - 提供更明确的系统提示

3. **性能问题**
     - 重用 Agent 实例
     - 使用更简单的模式
     - 考虑使用 ProviderStrategy（如果可用）

## 总结

LangChain 的结构化输出功能为构建可靠的 AI 应用提供了强大基础：

- **灵活的模式定义**：支持多种模式类型
- **智能的策略选择**：自动选择最佳实现方式
- **强大的错误处理**：内置验证和重试机制
- **生产级可靠性**：适合企业级应用

通过合理使用结构化输出，你可以构建出更加稳定、可维护的 AI 应用系统。
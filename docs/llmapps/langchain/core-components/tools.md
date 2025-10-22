# LangChain Tools 

## 概述

Tools（工具）是 AI Agent 调用以执行操作的组件。它们通过定义良好的输入和输出来扩展模型能力，使模型能够与外部系统（如 API、数据库、文件系统）进行交互。

### 核心概念

- **结构化交互**：Tools 提供模型与外部系统的结构化接口
- **封装性**：封装可调用函数及其输入模式
- **智能调用**：模型决定是否调用工具以及使用什么参数

## 创建工具

### 1. 基础工具定义

使用 `@tool` 装饰器创建工具，函数文档字符串会自动成为工具描述：

```python
from langchain.tools import tool

@tool
def search_database(query: str, limit: int = 10) -> str:
    """在客户数据库中搜索匹配查询的记录。

    Args:
        query: 要查找的搜索词
        limit: 返回的最大结果数
    """
    # 模拟数据库搜索
    return f"找到 {limit} 条关于 '{query}' 的结果"

# 使用工具
result = search_database.invoke({"query": "客户投诉", "limit": 5})
print(result)
```

### 2. 自定义工具属性

#### 自定义工具名称

```python
@tool("web_search")  # 自定义名称
def search_web(query: str) -> str:
    """在网络上搜索信息。"""
    return f"搜索 '{query}' 的结果"

print(search_web.name)  # 输出: web_search
```

#### 自定义工具描述

```python
@tool("calculator", description="执行算术计算。用于任何数学问题。")
def calculate(expression: str) -> str:
    """评估数学表达式。"""
    return str(eval(expression))
```

### 3. 高级模式定义

#### 使用 Pydantic 模型定义复杂输入

```python
from pydantic import BaseModel, Field
from typing import Literal, List

class WeatherInput(BaseModel):
    """天气查询的输入参数。"""
    location: str = Field(description="城市名称或坐标")
    units: Literal["celsius", "fahrenheit"] = Field(
        default="celsius",
        description="温度单位偏好"
    )
    include_forecast: bool = Field(
        default=False,
        description="包含5天预报"
    )
    forecast_days: int = Field(
        default=5,
        ge=1,
        le=10,
        description="预报天数（1-10）"
    )

@tool(args_schema=WeatherInput)
def get_weather(
    location: str, 
    units: str = "celsius", 
    include_forecast: bool = False,
    forecast_days: int = 5
) -> str:
    """获取当前天气和可选预报。"""
    temp = 22 if units == "celsius" else 72
    result = f"{location}当前天气: {temp}度 {units}"
    
    if include_forecast:
        result += f"\n未来{forecast_days}天预报: 晴朗"
    
    return result
```

#### 使用 JSON Schema 定义

```python
weather_schema = {
    "type": "object",
    "properties": {
        "location": {"type": "string"},
        "units": {"type": "string", "enum": ["celsius", "fahrenheit"]},
        "include_forecast": {"type": "boolean"},
        "forecast_days": {"type": "integer", "minimum": 1, "maximum": 10}
    },
    "required": ["location"]
}

@tool(args_schema=weather_schema)
def get_weather_json(
    location: str, 
    units: str = "celsius", 
    include_forecast: bool = False,
    forecast_days: int = 5
) -> str:
    """使用 JSON Schema 定义获取天气信息。"""
    temp = 22 if units == "celsius" else 72
    result = f"{location}当前天气: {temp}度 {units}"
    
    if include_forecast:
        result += f"\n未来{forecast_days}天预报: 晴朗"
    
    return result
```

## 访问上下文

Tools 最强大的功能是能够访问 Agent 状态、运行时上下文和长期记忆，从而实现上下文感知决策和个性化响应。

### ToolRuntime 概述

`ToolRuntime` 是一个统一的参数，提供工具访问以下信息的能力：

- **State**：执行过程中的可变数据（消息、计数器、自定义字段）
- **Context**：不可变配置（用户 ID、会话详情、应用特定配置）
- **Store**：跨对话的持久长期记忆
- **Stream Writer**：工具执行时流式传输自定义更新
- **Config**：执行的 RunnableConfig
- **Tool Call ID**：当前工具调用的 ID

### 访问状态（State）

```python
from langchain.tools import tool, ToolRuntime

@tool
def analyze_conversation(runtime: ToolRuntime) -> str:
    """分析当前对话状态。"""
    messages = runtime.state["messages"]
    
    # 统计不同类型的消息
    human_count = sum(1 for m in messages if m.type == "human")
    ai_count = sum(1 for m in messages if m.type == "ai")
    tool_count = sum(1 for m in messages if m.type == "tool")
    
    return f"对话统计: {human_count}条用户消息, {ai_count}条AI回复, {tool_count}条工具结果"

@tool
def get_user_preference(pref_name: str, runtime: ToolRuntime) -> str:
    """获取用户偏好设置。"""
    preferences = runtime.state.get("user_preferences", {})
    return preferences.get(pref_name, "未设置")
```

**重要提示**：`runtime` 参数对模型不可见，模型只能看到其他参数。

### 更新状态（使用 Command）

```python
from langgraph.types import Command
from langchain.messages import RemoveMessage
from langgraph.graph.message import REMOVE_ALL_MESSAGES

@tool
def clear_conversation(runtime: ToolRuntime) -> Command:
    """清除对话历史。"""
    return Command(
        update={
            "messages": [RemoveMessage(id=REMOVE_ALL_MESSAGES)],
        }
    )

@tool
def update_user_profile(name: str, age: int, runtime: ToolRuntime) -> Command:
    """更新用户档案。"""
    return Command(
        update={
            "user_profile": {
                "name": name,
                "age": age,
                "updated_at": "2024-01-01"
            }
        }
    )
```

### 访问上下文（Context）

```python
from dataclasses import dataclass
from langchain.agents import create_agent
from langchain.tools import tool, ToolRuntime

# 模拟用户数据库
USER_DATABASE = {
    "user123": {
        "name": "张三",
        "account_type": "高级会员",
        "balance": 5000,
        "email": "zhangsan@example.com"
    },
    "user456": {
        "name": "李四",
        "account_type": "标准会员",
        "balance": 1200,
        "email": "lisi@example.com"
    }
}

@dataclass
class UserContext:
    user_id: str

@tool
def get_account_info(runtime: ToolRuntime[UserContext]) -> str:
    """获取当前用户的账户信息。"""
    user_id = runtime.context.user_id
    
    if user_id in USER_DATABASE:
        user = USER_DATABASE[user_id]
        return f"""
        账户信息:
        - 姓名: {user['name']}
        - 账户类型: {user['account_type']}
        - 余额: ¥{user['balance']}
        - 邮箱: {user['email']}
        """
    return "用户未找到"

@tool
def transfer_funds(amount: float, to_user: str, runtime: ToolRuntime[UserContext]) -> str:
    """转账到其他用户。"""
    from_user_id = runtime.context.user_id
    
    if from_user_id not in USER_DATABASE or to_user not in USER_DATABASE:
        return "用户不存在"
    
    from_user = USER_DATABASE[from_user_id]
    to_user_info = USER_DATABASE[to_user]
    
    if from_user["balance"] < amount:
        return "余额不足"
    
    # 模拟转账操作
    from_user["balance"] -= amount
    to_user_info["balance"] += amount
    
    return f"成功转账 ¥{amount} 给 {to_user_info['name']}"

# 创建使用上下文的 Agent
agent = create_agent(
    model="openai:gpt-4o",
    tools=[get_account_info, transfer_funds],
    context_schema=UserContext,
    system_prompt="你是一个金融助手。"
)

# 使用上下文调用
result = agent.invoke(
    {"messages": [{"role": "user", "content": "查看我的账户余额"}]},
    context=UserContext(user_id="user123")
)
```

### 访问存储（Store）- 长期记忆

```python
from typing import Any
from langgraph.store.memory import InMemoryStore
from langchain.agents import create_agent
from langchain.tools import tool, ToolRuntime

@tool
def save_user_preferences(user_id: str, preferences: dict, runtime: ToolRuntime) -> str:
    """保存用户偏好设置到长期存储。"""
    store = runtime.store
    store.put(("user_preferences",), user_id, preferences)
    return "用户偏好设置已保存"

@tool
def get_user_preferences(user_id: str, runtime: ToolRuntime) -> str:
    """从长期存储获取用户偏好设置。"""
    store = runtime.store
    preferences = store.get(("user_preferences",), user_id)
    
    if preferences and preferences.value:
        prefs = preferences.value
        return f"用户 {user_id} 的偏好设置: {prefs}"
    else:
        return f"未找到用户 {user_id} 的偏好设置"

@tool
def save_conversation_summary(conversation_id: str, summary: str, runtime: ToolRuntime) -> str:
    """保存对话总结到长期存储。"""
    store = runtime.store
    store.put(("conversations",), conversation_id, {
        "summary": summary,
        "timestamp": "2024-01-01T10:00:00"
    })
    return "对话总结已保存"

# 创建使用存储的 Agent
store = InMemoryStore()
agent = create_agent(
    model="openai:gpt-4o",
    tools=[save_user_preferences, get_user_preferences, save_conversation_summary],
    store=store
)

# 第一次会话：保存用户偏好
agent.invoke({
    "messages": [{
        "role": "user", 
        "content": "保存用户123的偏好：语言=中文，主题=深色，通知=开启"
    }]
})

# 后续会话：获取用户偏好
agent.invoke({
    "messages": [{
        "role": "user", 
        "content": "获取用户123的偏好设置"
    }]
})
```

### 使用流写入器（Stream Writer）

```python
from langchain.tools import tool, ToolRuntime
import time

@tool
def process_large_data(data_source: str, runtime: ToolRuntime) -> str:
    """处理大型数据集的工具，带进度反馈。"""
    writer = runtime.stream_writer
    
    writer(f"🔄 开始处理数据源: {data_source}")
    writer("📊 连接数据源...")
    time.sleep(0.5)
    
    writer("🔍 读取数据...")
    time.sleep(1)
    
    # 模拟处理步骤
    steps = ["数据清洗", "特征提取", "模型训练", "结果分析"]
    for i, step in enumerate(steps, 1):
        writer(f"⏳ 步骤 {i}/{len(steps)}: {step}")
        time.sleep(0.8)
    
    writer("✅ 数据处理完成")
    return f"成功处理 {data_source}，生成分析报告"

@tool
def search_with_progress(query: str, runtime: ToolRuntime) -> str:
    """带进度反馈的搜索工具。"""
    writer = runtime.stream_writer
    
    writer(f"🔍 开始搜索: {query}")
    writer("🌐 连接搜索引擎...")
    time.sleep(0.3)
    
    writer("📡 发送搜索请求...")
    time.sleep(0.5)
    
    writer("📄 解析搜索结果...")
    time.sleep(0.7)
    
    writer("✅ 搜索完成")
    return f"找到关于 '{query}' 的 15 个相关结果"
```

## 实际应用场景

### 场景1：电商客服系统

```python
from datetime import datetime
from typing import Dict, List
from langchain.tools import tool, ToolRuntime

class EcommerceTools:
    """电商客服工具集"""
    
    @staticmethod
    @tool
    def check_order_status(order_id: str, runtime: ToolRuntime) -> str:
        """检查订单状态。"""
        # 模拟订单数据库
        orders = {
            "ORD001": {"status": "已发货", "tracking": "SF123456789", "items": ["商品A", "商品B"]},
            "ORD002": {"status": "处理中", "tracking": None, "items": ["商品C"]},
            "ORD003": {"status": "已送达", "tracking": "SF987654321", "items": ["商品D"]}
        }
        
        if order_id in orders:
            order = orders[order_id]
            result = f"订单 {order_id} 状态: {order['status']}"
            if order['tracking']:
                result += f"\n物流单号: {order['tracking']}"
            result += f"\n商品: {', '.join(order['items'])}"
            return result
        else:
            return f"未找到订单 {order_id}"
    
    @staticmethod
    @tool
    def get_product_info(product_id: str, runtime: ToolRuntime) -> str:
        """获取商品信息。"""
        products = {
            "P001": {"name": "智能手机", "price": 2999, "stock": 50, "description": "最新款智能手机"},
            "P002": {"name": "笔记本电脑", "price": 5999, "stock": 25, "description": "高性能笔记本电脑"},
            "P003": {"name": "无线耳机", "price": 399, "stock": 100, "description": "降噪无线耳机"}
        }
        
        if product_id in products:
            product = products[product_id]
            return f"""
            {product['name']}
            - 价格: ¥{product['price']}
            - 库存: {product['stock']}件
            - 描述: {product['description']}
            """
        else:
            return f"未找到商品 {product_id}"
    
    @staticmethod
    @tool
    def process_return(request_id: str, reason: str, runtime: ToolRuntime) -> Command:
        """处理退货申请。"""
        from langgraph.types import Command
        
        # 模拟处理退货
        return_info = {
            "request_id": request_id,
            "reason": reason,
            "status": "处理中",
            "processed_at": datetime.now().isoformat()
        }
        
        return Command(
            update={
                "return_requests": runtime.state.get("return_requests", []) + [return_info]
            }
        )

# 创建电商客服 Agent
ecommerce_agent = create_agent(
    model="openai:gpt-4o",
    tools=[
        EcommerceTools.check_order_status,
        EcommerceTools.get_product_info,
        EcommerceTools.process_return
    ],
    system_prompt="你是一个专业的电商客服助手。"
)
```

### 场景2：智能数据分析工具

```python
import pandas as pd
import numpy as np
from io import StringIO
from langchain.tools import tool, ToolRuntime

class DataAnalysisTools:
    """数据分析工具集"""
    
    @staticmethod
    @tool
    def load_csv_data(csv_content: str, runtime: ToolRuntime) -> Command:
        """加载 CSV 数据到分析环境。"""
        from langgraph.types import Command
        
        try:
            # 从 CSV 字符串创建 DataFrame
            df = pd.read_csv(StringIO(csv_content))
            
            # 返回数据统计信息
            stats = {
                "rows": len(df),
                "columns": len(df.columns),
                "columns_list": list(df.columns),
                "memory_usage": df.memory_usage(deep=True).sum()
            }
            
            return Command(
                update={
                    "current_dataset": df.to_dict(),
                    "dataset_stats": stats
                }
            )
        except Exception as e:
            return f"加载数据失败: {str(e)}"
    
    @staticmethod
    @tool
    def describe_dataset(runtime: ToolRuntime) -> str:
        """描述当前数据集的基本信息。"""
        stats = runtime.state.get("dataset_stats", {})
        
        if not stats:
            return "没有加载的数据集"
        
        return f"""
        数据集信息:
        - 行数: {stats['rows']}
        - 列数: {stats['columns']}
        - 列名: {', '.join(stats['columns_list'])}
        - 内存使用: {stats['memory_usage']} 字节
        """
    
    @staticmethod
    @tool
    def calculate_statistics(column: str, runtime: ToolRuntime) -> str:
        """计算指定列的统计信息。"""
        dataset = runtime.state.get("current_dataset", {})
        
        if not dataset:
            return "没有加载的数据集"
        
        try:
            df = pd.DataFrame(dataset)
            
            if column not in df.columns:
                return f"列 '{column}' 不存在"
            
            series = df[column]
            stats = {
                "count": len(series),
                "mean": series.mean(),
                "std": series.std(),
                "min": series.min(),
                "max": series.max(),
                "null_count": series.isnull().sum()
            }
            
            return f"""
            {column} 列统计信息:
            - 数量: {stats['count']}
            - 平均值: {stats['mean']:.2f}
            - 标准差: {stats['std']:.2f}
            - 最小值: {stats['min']}
            - 最大值: {stats['max']}
            - 空值数量: {stats['null_count']}
            """
        except Exception as e:
            return f"计算统计信息失败: {str(e)}"

# 创建数据分析 Agent
data_analysis_agent = create_agent(
    model="openai:gpt-4o",
    tools=[
        DataAnalysisTools.load_csv_data,
        DataAnalysisTools.describe_dataset,
        DataAnalysisTools.calculate_statistics
    ],
    system_prompt="你是一个数据分析助手，帮助用户分析和理解数据。"
)
```

### 场景3：项目管理工具

```python
from typing import List, Dict
from datetime import datetime, timedelta
from langchain.tools import tool, ToolRuntime

class ProjectManagementTools:
    """项目管理工具集"""
    
    @staticmethod
    @tool
    def create_task(title: str, description: str, assignee: str, due_date: str, runtime: ToolRuntime) -> Command:
        """创建新任务。"""
        from langgraph.types import Command
        
        task = {
            "id": f"TASK_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            "title": title,
            "description": description,
            "assignee": assignee,
            "due_date": due_date,
            "status": "待开始",
            "created_at": datetime.now().isoformat()
        }
        
        return Command(
            update={
                "project_tasks": runtime.state.get("project_tasks", []) + [task]
            }
        )
    
    @staticmethod
    @tool
    def update_task_status(task_id: str, new_status: str, runtime: ToolRuntime) -> Command:
        """更新任务状态。"""
        from langgraph.types import Command
        
        tasks = runtime.state.get("project_tasks", [])
        updated_tasks = []
        task_found = False
        
        for task in tasks:
            if task["id"] == task_id:
                task["status"] = new_status
                task["updated_at"] = datetime.now().isoformat()
                task_found = True
            updated_tasks.append(task)
        
        if task_found:
            return Command(update={"project_tasks": updated_tasks})
        else:
            return f"未找到任务 {task_id}"
    
    @staticmethod
    @tool
    def get_project_progress(runtime: ToolRuntime) -> str:
        """获取项目进度概览。"""
        tasks = runtime.state.get("project_tasks", [])
        
        if not tasks:
            return "项目中没有任务"
        
        status_count = {}
        for task in tasks:
            status = task["status"]
            status_count[status] = status_count.get(status, 0) + 1
        
        total_tasks = len(tasks)
        completed_tasks = status_count.get("已完成", 0)
        progress_percentage = (completed_tasks / total_tasks) * 100
        
        return f"""
        项目进度概览:
        - 总任务数: {total_tasks}
        - 已完成: {completed_tasks}
        - 进行中: {status_count.get('进行中', 0)}
        - 待开始: {status_count.get('待开始', 0)}
        - 总体进度: {progress_percentage:.1f}%
        """
    
    @staticmethod
    @tool
    def assign_task(task_id: str, new_assignee: str, runtime: ToolRuntime) -> Command:
        """重新分配任务。"""
        from langgraph.types import Command
        
        tasks = runtime.state.get("project_tasks", [])
        updated_tasks = []
        task_found = False
        
        for task in tasks:
            if task["id"] == task_id:
                old_assignee = task["assignee"]
                task["assignee"] = new_assignee
                task["updated_at"] = datetime.now().isoformat()
                task_found = True
            updated_tasks.append(task)
        
        if task_found:
            return Command(
                update={
                    "project_tasks": updated_tasks
                }
            )
        else:
            return f"未找到任务 {task_id}"

# 创建项目管理 Agent
project_agent = create_agent(
    model="openai:gpt-4o",
    tools=[
        ProjectManagementTools.create_task,
        ProjectManagementTools.update_task_status,
        ProjectManagementTools.get_project_progress,
        ProjectManagementTools.assign_task
    ],
    system_prompt="你是一个项目管理助手，帮助团队管理任务和跟踪进度。"
)
```

## 最佳实践

### 1. 工具设计原则

```python
from pydantic import BaseModel, Field
from typing import Optional

class WellDesignedTool:
    """良好设计的工具示例"""
    
    @staticmethod
    @tool
    def search_products(
        query: str,
        category: Optional[str] = None,
        price_range: Optional[str] = None,
        sort_by: str = "relevance",
        runtime: ToolRuntime
    ) -> str:
        """搜索产品信息。
        
        Args:
            query: 搜索关键词（必需）
            category: 产品类别筛选（可选）
            price_range: 价格范围筛选，如 "100-500"（可选）
            sort_by: 排序方式：relevance（相关度）、price_asc（价格升序）、price_desc（价格降序）
        """
        # 清晰的参数说明
        # 合理的默认值
        # 完整的错误处理
        
        writer = runtime.stream_writer
        writer(f"🔍 搜索产品: {query}")
        
        if category:
            writer(f"📁 筛选类别: {category}")
        if price_range:
            writer(f"💰 价格范围: {price_range}")
        
        # 模拟搜索逻辑
        writer("📊 获取搜索结果...")
        
        return f"找到 15 个匹配 '{query}' 的产品"
```

### 2. 错误处理

```python
from langchain.tools import tool, ToolRuntime

class RobustTools:
    """健壮的工具设计"""
    
    @staticmethod
    @tool
    def safe_api_call(api_endpoint: str, params: dict, runtime: ToolRuntime) -> str:
        """安全的 API 调用工具。"""
        import requests
        import time
        
        writer = runtime.stream_writer
        writer(f"🌐 调用 API: {api_endpoint}")
        
        max_retries = 3
        for attempt in range(max_retries):
            try:
                response = requests.get(api_endpoint, params=params, timeout=10)
                response.raise_for_status()
                return f"API 调用成功: {response.json()}"
                
            except requests.exceptions.Timeout:
                writer(f"⏰ 请求超时 (尝试 {attempt + 1}/{max_retries})")
                if attempt == max_retries - 1:
                    return "错误: API 请求超时"
                time.sleep(1)
                
            except requests.exceptions.RequestException as e:
                return f"API 调用错误: {str(e)}"
    
    @staticmethod
    @tool
    def validate_and_process_data(data: str, runtime: ToolRuntime) -> str:
        """验证和处理数据。"""
        writer = runtime.stream_writer
        
        # 数据验证
        if not data or not data.strip():
            return "错误: 数据不能为空"
        
        writer("✅ 数据验证通过")
        writer("🔄 处理数据...")
        
        try:
            # 模拟数据处理
            processed = data.upper()
            return f"处理后的数据: {processed}"
        except Exception as e:
            return f"数据处理错误: {str(e)}"
```

### 3. 性能优化

```python
from langchain.tools import tool, ToolRuntime
import functools

class OptimizedTools:
    """性能优化的工具"""
    
    # 使用缓存避免重复计算
    @functools.lru_cache(maxsize=100)
    def _expensive_calculation(self, input_data: str) -> str:
        """模拟昂贵的计算。"""
        # 模拟复杂计算
        return f"计算结果: {input_data.upper()}"
    
    @tool
    def cached_calculation(self, input_data: str, runtime: ToolRuntime) -> str:
        """使用缓存的昂贵计算。"""
        writer = runtime.stream_writer
        writer("⚡ 使用缓存计算...")
        
        return self._expensive_calculation(input_data)
    
    @tool
    def batch_processing(self, items: list, runtime: ToolRuntime) -> str:
        """批量处理工具。"""
        writer = runtime.stream_writer
        
        writer(f"📦 开始批量处理 {len(items)} 个项目")
        
        results = []
        batch_size = 5
        
        for i in range(0, len(items), batch_size):
            batch = items[i:i + batch_size]
            writer(f"处理批次 {i//batch_size + 1}/{(len(items)-1)//batch_size + 1}")
            
            # 模拟批量处理
            batch_results = [f"处理: {item}" for item in batch]
            results.extend(batch_results)
        
        writer("✅ 批量处理完成")
        return f"成功处理 {len(results)} 个项目"
```

## 总结

LangChain Tools 提供了强大的能力来扩展 AI Agent 的功能：

- **简单创建**：使用 `@tool` 装饰器快速定义工具
- **灵活定制**：支持自定义名称、描述和复杂输入模式
- **上下文感知**：通过 `ToolRuntime` 访问状态、上下文、存储等
- **实时反馈**：使用流写入器提供执行进度
- **生产就绪**：包含错误处理、性能优化等最佳实践

通过合理设计和使用 Tools，可以构建出能够与各种外部系统交互的智能 Agent，实现真正的自动化工作流程。
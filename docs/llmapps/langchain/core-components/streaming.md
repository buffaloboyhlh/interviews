# LangChain 流式传输

## 概述

LangChain 的流式传输系统可以实时展示更新，这对于构建响应迅速的 LLM 应用至关重要。通过逐步显示输出（即使在完整响应准备好之前），流式传输显著改善了用户体验，特别是在处理 LLM 的延迟时。

### 流式传输的优势

- **实时反馈**：用户可以看到处理进度
- **降低感知延迟**：即使总时间相同，用户体验更好
- **调试友好**：可以观察每个步骤的执行情况
- **灵活控制**：支持多种流式传输模式

## 基础设置

### 创建基础 Agent

```python
from langchain.agents import create_agent

# 创建一个简单的工具函数
def get_weather(city: str) -> str:
    """获取指定城市的天气"""
    return f"{city}的天气是晴朗的，25°C"

# 创建 Agent
agent = create_agent(
    model="openai:gpt-4o",
    tools=[get_weather],
)
```

## 流式传输模式

### 1. 代理进度流 (Agent Progress)

使用 `stream_mode="updates"` 来流式传输代理的每个步骤进度。

```python
def stream_agent_progress():
    """流式传输代理执行进度"""
    print("=== 代理进度流式传输 ===")
    
    for chunk in agent.stream(
        {"messages": [{"role": "user", "content": "北京和上海的天气怎么样？"}]},
        stream_mode="updates",  # 关键参数
    ):
        for step, data in chunk.items():
            print(f"步骤: {step}")
            if 'messages' in data and data['messages']:
                last_message = data['messages'][-1]
                print(f"内容: {last_message.content}")
                if hasattr(last_message, 'tool_calls') and last_message.tool_calls:
                    print(f"工具调用: {last_message.tool_calls}")
            print("-" * 50)

# 调用示例
stream_agent_progress()
```

**输出示例：**
```
步骤: model
内容: 
工具调用: [{'name': 'get_weather', 'args': {'city': '北京'}, 'id': 'call_123'}]
--------------------------------------------------
步骤: tools
内容: 北京的天气是晴朗的，25°C
--------------------------------------------------
步骤: model
内容: 北京天气晴朗，25°C。接下来查询上海天气...
工具调用: [{'name': 'get_weather', 'args': {'city': '上海'}, 'id': 'call_456'}]
--------------------------------------------------
步骤: tools
内容: 上海的天气是晴朗的，25°C
--------------------------------------------------
步骤: model
内容: 北京和上海都是晴朗天气，25°C。
--------------------------------------------------
```

### 2. LLM Token 流 (LLM Tokens)

使用 `stream_mode="messages"` 来流式传输 LLM 生成的每个 token。

```python
def stream_llm_tokens():
    """流式传输 LLM 生成的 tokens"""
    print("=== LLM Token 流式传输 ===")
    
    for token, metadata in agent.stream(
        {"messages": [{"role": "user", "content": "上海的天气如何？"}]},
        stream_mode="messages",  # 关键参数
    ):
        node_name = metadata.get('langgraph_node', 'unknown')
        
        if hasattr(token, 'content_blocks') and token.content_blocks:
            for block in token.content_blocks:
                if block.get('type') == 'text' and block.get('text'):
                    print(f"[{node_name}] {block['text']}", end='', flush=True)
                elif block.get('type') == 'tool_call_chunk':
                    print(f"\n[工具调用] {block.get('name', '')} {block.get('args', '')}")
        
    print()  # 最终换行

# 调用示例
stream_llm_tokens()
```

**输出示例：**
```
[model] 让我
[model] 来查询
[model] 一下
[model] 上海
[model] 的天气
[model] ...
[工具调用] get_weather {"city":"上海"}
[tools] 上海的天气是晴朗的，25°C
[model] 上海
[model] 的天气
[model] 是晴朗的
[model] ，25°C
[model] 。
```

### 3. 自定义更新流 (Custom Updates)

在工具中使用 `get_stream_writer()` 来发送自定义的流式更新。

```python
from langgraph.config import get_stream_writer

def create_custom_streaming_tool():
    """创建支持自定义流式传输的工具"""
    
    def search_products(query: str, max_results: int = 5) -> str:
        """搜索产品信息"""
        writer = get_stream_writer()
        
        # 发送自定义进度更新
        writer(f"🔍 开始搜索: {query}")
        writer(f"📊 最大结果数: {max_results}")
        
        # 模拟搜索过程
        writer("⏳ 连接数据库...")
        # 模拟数据库查询
        writer("✅ 数据库连接成功")
        
        writer("🔎 执行搜索查询...")
        # 模拟搜索逻辑
        import time
        time.sleep(0.5)
        
        writer(f"📦 找到 3 个相关产品")
        
        # 返回最终结果
        return f"搜索 '{query}' 找到 3 个产品: 产品A, 产品B, 产品C"
    
    return search_products

def stream_custom_updates():
    """流式传输自定义更新"""
    print("=== 自定义更新流式传输 ===")
    
    search_tool = create_custom_streaming_tool()
    custom_agent = create_agent(
        model="openai:gpt-4o",
        tools=[search_tool],
    )
    
    for chunk in custom_agent.stream(
        {"messages": [{"role": "user", "content": "搜索笔记本电脑"}]},
        stream_mode="custom"  # 关键参数
    ):
        print(f"自定义更新: {chunk}")

# 调用示例
stream_custom_updates()
```

**输出示例：**
```
自定义更新: 🔍 开始搜索: 笔记本电脑
自定义更新: 📊 最大结果数: 5
自定义更新: ⏳ 连接数据库...
自定义更新: ✅ 数据库连接成功
自定义更新: 🔎 执行搜索查询...
自定义更新: 📦 找到 3 个相关产品
```

### 4. 多模式流式传输

可以同时使用多种流式传输模式。

```python
def stream_multiple_modes():
    """同时使用多种流式传输模式"""
    print("=== 多模式流式传输 ===")
    
    # 创建支持自定义流的工具
    def advanced_weather_tool(city: str) -> str:
        """高级天气查询工具"""
        writer = get_stream_writer()
        writer(f"🌤️  开始查询 {city} 的天气")
        writer("📡 连接气象API...")
        writer("🔍 获取实时数据...")
        return f"{city}的天气：晴朗，25°C，湿度60%"
    
    multi_agent = create_agent(
        model="openai:gpt-4o",
        tools=[advanced_weather_tool],
    )
    
    for stream_mode, chunk in multi_agent.stream(
        {"messages": [{"role": "user", "content": "查询杭州的天气"}]},
        stream_mode=["updates", "custom", "messages"]  # 多种模式
    ):
        print(f"模式: {stream_mode}")
        print(f"内容: {chunk}")
        print("-" * 30)

# 调用示例
stream_multiple_modes()
```

## 实际应用场景

### 场景1：实时聊天应用

```python
import asyncio
from langchain.agents import create_agent

class StreamingChatApp:
    """支持流式传输的聊天应用"""
    
    def __init__(self):
        self.agent = create_agent(
            model="openai:gpt-4o",
            tools=[self.get_weather, self.search_web],
        )
    
    def get_weather(self, city: str) -> str:
        """获取天气信息"""
        writer = get_stream_writer()
        writer(f"查询{city}的天气...")
        # 模拟API调用
        return f"{city}: 25°C, 晴朗"
    
    def search_web(self, query: str) -> str:
        """网页搜索"""
        writer = get_stream_writer()
        writer(f"搜索: {query}")
        writer("正在获取最新信息...")
        return f"关于'{query}'的搜索结果..."
    
    async def chat_stream(self, message: str):
        """流式聊天"""
        print(f"用户: {message}")
        print("助手: ", end="", flush=True)
        
        full_response = ""
        for token, metadata in self.agent.stream(
            {"messages": [{"role": "user", "content": message}]},
            stream_mode="messages",
        ):
            if hasattr(token, 'content_blocks'):
                for block in token.content_blocks:
                    if block.get('type') == 'text' and block.get('text'):
                        text = block['text']
                        print(text, end='', flush=True)
                        full_response += text
        
        print()  # 换行
        return full_response

# 使用示例
async def demo_chat():
    app = StreamingChatApp()
    await app.chat_stream("今天杭州天气怎么样？然后搜索AI最新发展")
```

### 场景2：进度监控仪表板

```python
from typing import Dict, Any
import json

class ProgressMonitor:
    """进度监控器"""
    
    def __init__(self):
        self.progress_data = {
            'total_steps': 0,
            'completed_steps': 0,
            'current_step': '',
            'details': []
        }
    
    def update_progress(self, step: str, details: str = ""):
        """更新进度"""
        self.progress_data['current_step'] = step
        self.progress_data['details'].append({
            'step': step,
            'details': details,
            'timestamp': str(datetime.now())
        })
        self.progress_data['completed_steps'] += 1
        
        # 发送到前端（模拟）
        print(f"进度更新: {json.dumps(self.progress_data, ensure_ascii=False)}")

def create_monitored_tools(monitor: ProgressMonitor):
    """创建被监控的工具"""
    
    def research_topic(topic: str) -> str:
        """研究主题"""
        writer = get_stream_writer()
        
        monitor.update_progress('research', f"开始研究: {topic}")
        writer(f"🔬 研究主题: {topic}")
        
        # 模拟研究步骤
        steps = [
            "收集相关资料",
            "分析关键信息", 
            "整理研究结果",
            "生成总结报告"
        ]
        
        for step in steps:
            monitor.update_progress('research', step)
            writer(f"✅ {step}")
            import time
            time.sleep(0.3)
        
        return f"关于{topic}的研究完成"
    
    return research_topic

def monitored_agent_demo():
    """被监控的Agent演示"""
    monitor = ProgressMonitor()
    research_tool = create_monitored_tools(monitor)
    
    agent = create_agent(
        model="openai:gpt-4o",
        tools=[research_tool],
    )
    
    print("开始监控Agent执行...")
    for stream_mode, chunk in agent.stream(
        {"messages": [{"role": "user", "content": "研究人工智能在医疗领域的应用"}]},
        stream_mode=["updates", "custom"]
    ):
        if stream_mode == "custom":
            print(f"自定义事件: {chunk}")

# 调用示例
monitored_agent_demo()
```

### 场景3：实时数据流处理

```python
import time
from datetime import datetime

class RealTimeDataProcessor:
    """实时数据处理器"""
    
    def __init__(self):
        self.agent = create_agent(
            model="openai:gpt-4o",
            tools=[self.process_data_stream],
        )
        self.data_buffer = []
    
    def process_data_stream(self, data_type: str, count: int = 10) -> str:
        """处理数据流"""
        writer = get_stream_writer()
        
        writer(f"开始处理 {data_type} 数据流...")
        writer(f"预计处理 {count} 条数据")
        
        # 模拟数据流处理
        for i in range(count):
            # 模拟数据处理
            processed_item = f"{data_type}_item_{i+1}"
            self.data_buffer.append(processed_item)
            
            # 发送进度更新
            progress = (i + 1) / count * 100
            writer(f"📊 进度: {progress:.1f}% - 已处理: {processed_item}")
            
            # 模拟处理时间
            time.sleep(0.1)
        
        writer("✅ 数据流处理完成")
        return f"成功处理 {count} 条{data_type}数据"
    
    def start_processing(self, data_type: str):
        """开始处理"""
        print(f"开始实时处理 {data_type} 数据...")
        
        for stream_mode, chunk in self.agent.stream(
            {"messages": [{"role": "user", "content": f"处理{data_type}数据流"}]},
            stream_mode=["custom", "updates"]
        ):
            if stream_mode == "custom":
                print(f"{datetime.now().strftime('%H:%M:%S')} - {chunk}")

# 使用示例
processor = RealTimeDataProcessor()
processor.start_processing("传感器")
```

## 高级功能

### 1. 错误处理和重试

```python
def create_robust_streaming_tool():
    """创建健壮的流式传输工具"""
    
    def robust_operation(operation: str) -> str:
        """健壮的操作"""
        writer = get_stream_writer()
        
        try:
            writer(f"🟡 开始执行: {operation}")
            
            # 模拟可能失败的操作
            if "fail" in operation:
                raise Exception("模拟操作失败")
            
            writer("🟢 操作执行中...")
            time.sleep(1)
            writer("✅ 操作完成")
            
            return f"操作 '{operation}' 成功完成"
            
        except Exception as e:
            writer(f"🔴 操作失败: {str(e)}")
            writer("🔄 尝试重试...")
            # 这里可以添加重试逻辑
            return f"操作 '{operation}' 失败: {str(e)}"
    
    return robust_operation

def error_handling_demo():
    """错误处理演示"""
    robust_tool = create_robust_streaming_tool()
    agent = create_agent(
        model="openai:gpt-4o",
        tools=[robust_tool],
    )
    
    print("测试正常操作:")
    for chunk in agent.stream(
        {"messages": [{"role": "user", "content": "执行正常操作"}]},
        stream_mode="custom"
    ):
        print(chunk)
    
    print("\n测试失败操作:")
    for chunk in agent.stream(
        {"messages": [{"role": "user", "content": "执行失败操作"}]},
        stream_mode="custom"
    ):
        print(chunk)
```

### 2. 性能优化

```python
class OptimizedStreaming:
    """优化流式传输性能"""
    
    def __init__(self):
        self.batch_size = 5
        self.message_buffer = []
    
    def batch_process_tool(self, items: list) -> str:
        """批量处理工具"""
        writer = get_stream_writer()
        
        writer(f"🔄 开始批量处理 {len(items)} 个项目")
        
        for i, item in enumerate(items):
            # 处理每个项目
            writer(f"处理项目 {i+1}/{len(items)}: {item}")
            
            # 模拟处理
            time.sleep(0.1)
            
            # 每处理完一批发送更新
            if (i + 1) % self.batch_size == 0:
                writer(f"📦 已完成 {i+1} 个项目")
        
        writer("✅ 批量处理完成")
        return f"成功处理 {len(items)} 个项目"
    
    def optimized_stream_demo(self):
        """优化流式传输演示"""
        agent = create_agent(
            model="openai:gpt-4o",
            tools=[self.batch_process_tool],
        )
        
        items = [f"item_{i}" for i in range(1, 16)]
        
        for chunk in agent.stream(
            {"messages": [{"role": "user", "content": f"批量处理这些项目: {items}"}]},
            stream_mode="custom"
        ):
            print(chunk)
```

## 最佳实践

### 1. 选择合适的流式传输模式

```python
def choose_stream_mode(use_case: str):
    """根据使用场景选择合适的流式传输模式"""
    mode_recommendations = {
        "chat_application": "messages",  # 聊天应用：需要实时显示文字
        "progress_tracking": ["updates", "custom"],  # 进度跟踪：需要步骤和自定义更新
        "debugging": "updates",  # 调试：需要看到每个步骤
        "data_processing": ["custom", "messages"],  # 数据处理：需要进度和结果
        "real_time_monitoring": ["updates", "custom", "messages"]  # 实时监控：全部信息
    }
    
    return mode_recommendations.get(use_case, "updates")

# 使用示例
chat_mode = choose_stream_mode("chat_application")
debug_mode = choose_stream_mode("debugging")
```

### 2. 处理流式传输错误

```python
def safe_stream_invoke(agent, input_data, stream_mode="updates", max_retries=3):
    """安全的流式调用"""
    for attempt in range(max_retries):
        try:
            for chunk in agent.stream(input_data, stream_mode=stream_mode):
                yield chunk
            break  # 成功完成，退出重试循环
        except Exception as e:
            print(f"流式传输错误 (尝试 {attempt + 1}/{max_retries}): {e}")
            if attempt == max_retries - 1:
                raise  # 最后一次尝试仍然失败，抛出异常
            time.sleep(1)  # 等待后重试

# 使用示例
for chunk in safe_stream_invoke(
    agent,
    {"messages": [{"role": "user", "content": "查询天气"}]},
    stream_mode="messages"
):
    print(chunk)
```

## 故障排除

### 常见问题及解决方案

1. **流式传输不工作**
   - 检查模型是否支持流式传输
   - 确认 `stream_mode` 参数设置正确
   - 验证网络连接

2. **自定义更新不显示**
   - 确保在工具中正确使用 `get_stream_writer()`
   - 检查 `stream_mode` 包含 "custom"
   - 确认在 LangGraph 执行上下文中调用

3. **性能问题**
   - 减少不必要的流式更新
   - 使用合适的批处理大小
   - 考虑禁用某些流式模式

## 总结

LangChain 的流式传输系统提供了强大的实时更新能力：

- **多种模式**：代理进度、LLM tokens、自定义更新
- **灵活组合**：可以同时使用多种流式模式
- **实际应用**：适用于聊天、监控、数据处理等场景
- **健壮性**：包含错误处理和性能优化

通过合理使用流式传输，可以显著提升应用的响应性和用户体验。
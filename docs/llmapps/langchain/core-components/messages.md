# LangChain Messages

## 概述

Messages（消息）是 LangChain 中模型上下文的基本单位。它们表示模型的输入和输出，携带与 LLM 交互时表示对话状态所需的内容和元数据。

消息对象包含：

- **角色**：标识消息类型（如 `system`、`user`）
- **内容**：表示消息的实际内容（如文本、图像、音频等）
- **元数据**：可选字段，如响应信息、消息 ID 和令牌使用情况

## 基础用法

### 1. 创建消息对象

```python
from langchain.messages import HumanMessage, AIMessage, SystemMessage

# 系统消息 - 设置模型行为
system_msg = SystemMessage("你是一个有帮助的助手。")

# 用户消息 - 代表用户输入
human_msg = HumanMessage("你好，请介绍一下人工智能。")

# AI 消息 - 模型生成的响应
ai_msg = AIMessage("人工智能是...")

print(f"系统消息: {system_msg.content}")
print(f"用户消息: {human_msg.content}")
print(f"AI 消息: {ai_msg.content}")
```

### 2. 与模型一起使用

```python
from langchain.chat_models import init_chat_model

# 初始化模型
model = init_chat_model("openai:gpt-4o")

# 使用消息列表调用模型
messages = [
    SystemMessage("你是一个专业的AI助手。"),
    HumanMessage("请解释机器学习的基本概念。")
]

response = model.invoke(messages)
print(f"模型回复: {response.content}")
```

## 消息类型详解

### 1. 系统消息 (SystemMessage)

系统消息用于设置模型的初始指令，定义模型的行为方式。

```python
from langchain.messages import SystemMessage

# 基础指令
system_basic = SystemMessage("你是一个有帮助的助手。")

# 详细角色设定
system_detailed = SystemMessage("""
你是一位资深软件工程师，具有10年Python开发经验。
请遵循以下原则：
1. 提供详细的代码示例
2. 解释技术概念时要清晰易懂
3. 对于复杂问题，分步骤解答
4. 保持专业和友好的态度
""")

# 特定领域专家
system_expert = SystemMessage("""
你是一位金融分析师，专注于股票市场分析。
请：
- 使用专业的金融术语
- 提供数据支持的分析
- 给出风险提示
- 保持客观中立
""")
```

### 2. 用户消息 (HumanMessage)

用户消息代表用户的输入，可以包含文本、图像、音频等多种内容。

```python
from langchain.messages import HumanMessage

# 基础文本消息
human_basic = HumanMessage("什么是Python的装饰器？")

# 带元数据的消息
human_with_metadata = HumanMessage(
    content="你好，我需要帮助！",
    name="张三",  # 可选：标识不同用户
    id="msg_001"  # 可选：唯一标识符
)

# 多模态消息（后续详细讲解）
human_multimodal = HumanMessage(
    content=[
        {"type": "text", "text": "描述这张图片的内容"},
        {"type": "image", "url": "https://example.com/image.jpg"}
    ]
)

print(f"消息内容: {human_basic.content}")
print(f"用户名称: {human_with_metadata.name}")
print(f"消息ID: {human_with_metadata.id}")
```

### 3. AI 消息 (AIMessage)

AI 消息表示模型的输出响应，包含内容、工具调用和元数据。

```python
from langchain.messages import AIMessage

# 调用模型获取AI消息
response = model.invoke("解释神经网络的工作原理")
print(f"响应类型: {type(response)}")  # <class 'langchain_core.messages.AIMessage'>

# 手动创建AI消息（用于对话历史）
manual_ai_msg = AIMessage(
    content="神经网络是受人脑启发的一种机器学习模型...",
    id="ai_msg_001"
)

# 访问AI消息的属性
print(f"文本内容: {response.content}")
print(f"消息ID: {response.id}")
print(f"使用元数据: {response.usage_metadata}")
print(f"响应元数据: {response.response_metadata}")
```

#### AI 消息的重要属性

```python
# 获取完整的响应信息
response = model.invoke("请详细说明深度学习")

print("=== AI消息属性 ===")
print(f"文本内容: {response.content}")
print(f"内容块: {response.content_blocks}")
print(f"工具调用: {response.tool_calls}")
print(f"消息ID: {response.id}")
print(f"令牌使用: {response.usage_metadata}")
print(f"响应元数据: {response.response_metadata}")

# 令牌使用统计示例
if response.usage_metadata:
    print(f"输入令牌: {response.usage_metadata.get('input_tokens', 'N/A')}")
    print(f"输出令牌: {response.usage_metadata.get('output_tokens', 'N/A')}")
    print(f"总令牌: {response.usage_metadata.get('total_tokens', 'N/A')}")
```

### 4. 工具消息 (ToolMessage)

工具消息用于将工具执行结果传递回模型。

```python
from langchain.messages import ToolMessage

# 模拟工具调用后的结果消息
tool_message = ToolMessage(
    content="北京天气：晴朗，25°C，湿度60%",  # 工具执行结果
    tool_call_id="call_123",  # 必须与工具调用ID匹配
    name="get_weather"  # 工具名称
)

print(f"工具结果: {tool_message.content}")
print(f"关联的工具调用ID: {tool_message.tool_call_id}")
print(f"工具名称: {tool_message.name}")
```

#### 使用 artifact 存储额外数据

```python
# 使用 artifact 存储不发送给模型的额外数据
tool_message_with_artifact = ToolMessage(
    content="检索到相关文档内容...",
    tool_call_id="call_456",
    name="search_documents",
    artifact={
        "document_ids": ["doc_123", "doc_456"],
        "source_urls": ["https://example.com/doc1", "https://example.com/doc2"],
        "confidence_scores": [0.95, 0.87]
    }
)

print(f"发送给模型的内容: {tool_message_with_artifact.content}")
print(f"额外数据: {tool_message_with_artifact.artifact}")
```

## 消息内容格式

### 1. 文本提示（简单格式）

```python
# 直接使用字符串（自动转换为HumanMessage）
response = model.invoke("写一首关于春天的诗")

# 等同于
response = model.invoke(HumanMessage("写一首关于春天的诗"))
```

**适用场景：**
- 单一独立请求
- 不需要对话历史
- 代码简洁性要求高

### 2. 消息对象列表

```python
from langchain.messages import SystemMessage, HumanMessage, AIMessage

messages = [
    SystemMessage("你是一位诗人，擅长写中文古诗。"),
    HumanMessage("写一首关于春天的七言绝句。"),
    AIMessage("春风吹绿柳丝长，花开满园蝶舞忙。"),
    HumanMessage("再写一首关于秋天的。")
]

response = model.invoke(messages)
print(response.content)
```

**适用场景：**
- 管理多轮对话
- 处理多模态内容
- 包含系统指令

### 3. 字典格式（OpenAI 兼容）

```python
# 使用OpenAI聊天完成格式
messages = [
    {"role": "system", "content": "你是一位专业翻译。"},
    {"role": "user", "content": "Translate: Hello, how are you?"},
    {"role": "assistant", "content": "你好，你好吗？"},
    {"role": "user", "content": "Translate: I love programming."}
]

response = model.invoke(messages)
print(response.content)  # 输出：我喜欢编程。
```

## 标准内容块

LangChain 提供了跨提供商的标准消息内容表示。

### 1. 内容块类型

```python
from langchain.messages import AIMessage

# 创建包含标准内容块的消息
message = AIMessage(
    content_blocks=[
        {
            "type": "text",
            "text": "这是主要的文本回复。",
            "annotations": [
                {
                    "type": "citation",
                    "start_index": 0,
                    "end_index": 5,
                    "text": "参考来源1"
                }
            ]
        },
        {
            "type": "reasoning",
            "reasoning": "首先，用户询问的是...然后我考虑了..."
        }
    ]
)

# 访问标准化的内容块
for block in message.content_blocks:
    print(f"块类型: {block['type']}")
    if block['type'] == 'text':
        print(f"文本内容: {block['text']}")
    elif block['type'] == 'reasoning':
        print(f"推理过程: {block['reasoning']}")
```

### 2. 多模态内容

#### 图像内容

```python
from langchain.messages import HumanMessage
import base64

# 从URL使用图像
message_with_image_url = HumanMessage(
    content_blocks=[
        {
            "type": "text", 
            "text": "描述这张图片中的内容"
        },
        {
            "type": "image",
            "url": "https://example.com/image.jpg"
        }
    ]
)

# 从base64数据使用图像
def encode_image_to_base64(image_path):
    """将图像编码为base64"""
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')

# 假设有本地图像文件
# image_base64 = encode_image_to_base64("path/to/image.jpg")

message_with_image_base64 = HumanMessage(
    content_blocks=[
        {
            "type": "text",
            "text": "分析这张医学影像"
        },
        {
            "type": "image",
            "base64": "base64_encoded_string_here",  # 替换为实际的base64字符串
            "mime_type": "image/jpeg"
        }
    ]
)
```

#### 文档内容

```python
# PDF文档
message_with_pdf = HumanMessage(
    content_blocks=[
        {
            "type": "text",
            "text": "总结这份文档的主要内容"
        },
        {
            "type": "file",
            "url": "https://example.com/document.pdf",
            "mime_type": "application/pdf"
        }
    ]
)

# 文本文件
message_with_text_file = HumanMessage(
    content_blocks=[
        {
            "type": "text-plain",
            "text": "文件内容...",
            "mime_type": "text/plain"
        }
    ]
)
```

#### 音频和视频内容

```python
# 音频内容
message_with_audio = HumanMessage(
    content_blocks=[
        {
            "type": "text",
            "text": "转写这段音频内容"
        },
        {
            "type": "audio",
            "base64": "base64_encoded_audio_here",
            "mime_type": "audio/wav"
        }
    ]
)

# 视频内容
message_with_video = HumanMessage(
    content_blocks=[
        {
            "type": "text",
            "text": "描述视频中的场景"
        },
        {
            "type": "video",
            "url": "https://example.com/video.mp4",
            "mime_type": "video/mp4"
        }
    ]
)
```

## 高级功能

### 1. 工具调用集成

```python
from langchain.tools import tool
from langchain.messages import AIMessage, ToolMessage

@tool
def get_weather(location: str) -> str:
    """获取指定位置的天气信息。"""
    # 模拟天气数据
    weather_data = {
        "北京": "晴朗，25°C",
        "上海": "多云，23°C", 
        "广州": "阵雨，28°C"
    }
    return weather_data.get(location, "未知地点")

# 绑定工具到模型
model_with_tools = model.bind_tools([get_weather])

# 模拟工具调用流程
def simulate_tool_call():
    # 用户询问天气
    user_message = HumanMessage("北京和上海的天气怎么样？")
    
    # 模型生成工具调用
    ai_response = model_with_tools.invoke([user_message])
    
    print("=== 模型工具调用 ===")
    for tool_call in ai_response.tool_calls:
        print(f"工具: {tool_call['name']}")
        print(f"参数: {tool_call['args']}")
        print(f"调用ID: {tool_call['id']}")
        
        # 执行工具
        if tool_call['name'] == 'get_weather':
            location = tool_call['args']['location']
            result = get_weather.invoke({"location": location})
            
            # 创建工具消息
            tool_msg = ToolMessage(
                content=result,
                tool_call_id=tool_call['id'],
                name="get_weather"
            )
            
            print(f"工具结果: {tool_msg.content}")
    
    return ai_response

# 运行示例
response = simulate_tool_call()
```

### 2. 流式消息处理

```python
def stream_message_example():
    """流式消息处理示例"""
    print("开始流式处理...")
    
    chunks = []
    full_message = None
    
    for chunk in model.stream("解释人工智能的发展历史"):
        chunks.append(chunk)
        
        # 实时输出文本内容
        if hasattr(chunk, 'content') and chunk.content:
            print(chunk.content, end="", flush=True)
        
        # 累积完整消息
        full_message = chunk if full_message is None else full_message + chunk
    
    print("\n=== 流式处理完成 ===")
    print(f"收到块数: {len(chunks)}")
    print(f"完整消息类型: {type(full_message)}")
    print(f"完整内容: {full_message.content}")
    
    return full_message

# 运行流式示例
final_message = stream_message_example()
```

### 3. 消息转换和序列化

```python
def message_conversion_examples():
    """消息转换示例"""
    
    # 创建消息
    human_msg = HumanMessage("你好，世界！")
    ai_msg = AIMessage("你好！我是AI助手。")
    
    print("=== 原始消息 ===")
    print(f"HumanMessage: {human_msg}")
    print(f"AIMessage: {ai_msg}")
    
    # 转换为字典
    human_dict = human_msg.dict()
    ai_dict = ai_msg.dict()
    
    print("\n=== 字典表示 ===")
    print(f"HumanMessage字典: {human_dict}")
    print(f"AIMessage字典: {ai_dict}")
    
    # 访问内容块
    print("\n=== 内容块 ===")
    print(f"HumanMessage内容块: {human_msg.content_blocks}")
    print(f"AIMessage内容块: {ai_msg.content_blocks}")
    
    #  pretty print
    print("\n=== 美观打印 ===")
    human_msg.pretty_print()

# 运行转换示例
message_conversion_examples()
```

## 实际应用场景

### 场景1：对话管理系统

```python
class ConversationManager:
    """对话管理器"""
    
    def __init__(self, model):
        self.model = model
        self.conversation_history = []
    
    def add_system_message(self, content: str):
        """添加系统消息"""
        system_msg = SystemMessage(content)
        self.conversation_history.append(system_msg)
    
    def add_user_message(self, content: str, user_name: str = None):
        """添加用户消息"""
        human_msg = HumanMessage(content, name=user_name)
        self.conversation_history.append(human_msg)
    
    def add_ai_message(self, content: str):
        """添加AI消息"""
        ai_msg = AIMessage(content)
        self.conversation_history.append(ai_msg)
    
    def get_ai_response(self, user_input: str, user_name: str = None) -> str:
        """获取AI响应"""
        # 添加用户消息
        self.add_user_message(user_input, user_name)
        
        # 调用模型（只使用最近10条消息以避免上下文过长）
        recent_messages = self.conversation_history[-10:] if len(self.conversation_history) > 10 else self.conversation_history
        
        # 获取AI响应
        response = self.model.invoke(recent_messages)
        
        # 添加AI消息到历史
        self.add_ai_message(response.content)
        
        return response.content
    
    def get_conversation_summary(self) -> dict:
        """获取对话摘要"""
        user_msgs = [msg for msg in self.conversation_history if isinstance(msg, HumanMessage)]
        ai_msgs = [msg for msg in self.conversation_history if isinstance(msg, AIMessage)]
        
        return {
            "total_messages": len(self.conversation_history),
            "user_messages": len(user_msgs),
            "ai_messages": len(ai_msgs),
            "last_user_message": user_msgs[-1].content if user_msgs else None,
            "last_ai_message": ai_msgs[-1].content if ai_msgs else None
        }

# 使用示例
manager = ConversationManager(model)
manager.add_system_message("你是一个友好的客服助手。")

# 进行对话
response1 = manager.get_ai_response("你好，我想查询订单状态")
print(f"AI回复1: {response1}")

response2 = manager.get_ai_response("我的订单号是12345")
print(f"AI回复2: {response2}")

# 获取对话统计
summary = manager.get_conversation_summary()
print(f"对话统计: {summary}")
```

### 场景2：多模态内容处理

```python
class MultimodalProcessor:
    """多模态内容处理器"""
    
    def __init__(self, model):
        self.model = model
    
    def analyze_image_with_text(self, image_url: str, question: str) -> str:
        """分析图像并回答问题"""
        message = HumanMessage(
            content_blocks=[
                {
                    "type": "text",
                    "text": question
                },
                {
                    "type": "image",
                    "url": image_url
                }
            ]
        )
        
        response = self.model.invoke([message])
        return response.content
    
    def process_document_qa(self, document_url: str, questions: list) -> dict:
        """处理文档问答"""
        results = {}
        
        for question in questions:
            message = HumanMessage(
                content_blocks=[
                    {
                        "type": "text",
                        "text": question
                    },
                    {
                        "type": "file",
                        "url": document_url,
                        "mime_type": "application/pdf"
                    }
                ]
            )
            
            response = self.model.invoke([message])
            results[question] = response.content
        
        return results
    
    def create_content_with_citations(self, topic: str, sources: list) -> AIMessage:
        """创建带引用的内容"""
        # 模拟带引用的响应
        citations = [{"type": "citation", "text": source, "url": f"https://example.com/{i}"} 
                    for i, source in enumerate(sources)]
        
        response = AIMessage(
            content_blocks=[
                {
                    "type": "text",
                    "text": f"关于{topic}的详细说明...",
                    "annotations": citations
                }
            ]
        )
        
        return response

# 使用示例
processor = MultimodalProcessor(model)

# 图像分析（假设有图像URL）
# image_analysis = processor.analyze_image_with_text(
#     "https://example.com/medical-image.jpg",
#     "描述这张医学影像中的异常情况"
# )

# 文档问答（假设有文档URL）
# doc_qa = processor.process_document_qa(
#     "https://example.com/research-paper.pdf",
#     ["研究的主要发现是什么？", "研究方法是什么？"]
# )

# 创建带引用的内容
cited_content = processor.create_content_with_citations(
    "人工智能伦理",
    ["AI伦理指南2023", "机器学习道德标准", "AI治理白皮书"]
)

print("带引用的内容:")
cited_content.pretty_print()
```

### 场景3：工具调用工作流

```python
class ToolWorkflowManager:
    """工具调用工作流管理器"""
    
    def __init__(self, model):
        self.model = model
        self.available_tools = {}
    
    def register_tool(self, tool):
        """注册工具"""
        self.available_tools[tool.name] = tool
    
    def execute_workflow(self, user_query: str) -> str:
        """执行工具调用工作流"""
        # 绑定所有可用工具
        tool_list = list(self.available_tools.values())
        model_with_tools = self.model.bind_tools(tool_list)
        
        messages = [HumanMessage(user_query)]
        max_iterations = 5  # 防止无限循环
        
        for iteration in range(max_iterations):
            # 获取模型响应
            ai_response = model_with_tools.invoke(messages)
            messages.append(ai_response)
            
            # 如果没有工具调用，返回最终响应
            if not ai_response.tool_calls:
                return ai_response.content
            
            # 执行所有工具调用
            for tool_call in ai_response.tool_calls:
                tool_name = tool_call["name"]
                if tool_name in self.available_tools:
                    # 执行工具
                    result = self.available_tools[tool_name].invoke(tool_call["args"])
                    
                    # 创建工具消息
                    tool_msg = ToolMessage(
                        content=str(result),
                        tool_call_id=tool_call["id"],
                        name=tool_name
                    )
                    messages.append(tool_msg)
                else:
                    # 工具未找到
                    error_msg = ToolMessage(
                        content=f"错误：工具 '{tool_name}' 未找到",
                        tool_call_id=tool_call["id"],
                        name=tool_name
                    )
                    messages.append(error_msg)
        
        return "达到最大迭代次数，工作流终止"

# 定义一些工具
@tool
def calculate_bmi(weight: float, height: float) -> str:
    """计算BMI指数"""
    bmi = weight / (height ** 2)
    category = "偏瘦" if bmi < 18.5 else "正常" if bmi < 24 else "超重" if bmi < 28 else "肥胖"
    return f"BMI: {bmi:.1f} ({category})"

@tool
def get_time(timezone: str = "UTC") -> str:
    """获取指定时区的当前时间"""
    from datetime import datetime
    now = datetime.now()
    return f"{timezone}时间: {now.strftime('%Y-%m-%d %H:%M:%S')}"

# 使用示例
workflow_manager = ToolWorkflowManager(model)
workflow_manager.register_tool(calculate_bmi)
workflow_manager.register_tool(get_time)

# 执行复杂查询
result = workflow_manager.execute_workflow(
    "请计算我的BMI，我体重70公斤，身高1.75米，然后告诉我现在北京时间"
)

print("工作流结果:", result)
```

## 最佳实践

### 1. 消息管理

```python
class MessageBestPractices:
    """消息管理最佳实践"""
    
    @staticmethod
    def create_effective_system_prompt(role: str, guidelines: list) -> SystemMessage:
        """创建有效的系统提示"""
        guidelines_text = "\n".join([f"{i+1}. {guideline}" for i, guideline in enumerate(guidelines)])
        
        prompt = f"""
        角色: {role}
        
        指导原则:
        {guidelines_text}
        
        请始终遵循以上原则进行回复。
        """
        
        return SystemMessage(prompt.strip())
    
    @staticmethod
    def manage_conversation_length(messages: list, max_messages: int = 20) -> list:
        """管理对话长度"""
        if len(messages) <= max_messages:
            return messages
        
        # 保留系统消息和最近的消息
        system_messages = [msg for msg in messages if isinstance(msg, SystemMessage)]
        recent_messages = messages[-(max_messages - len(system_messages)):]
        
        return system_messages + recent_messages
    
    @staticmethod
    def extract_usage_statistics(messages: list) -> dict:
        """提取使用统计"""
        ai_messages = [msg for msg in messages if isinstance(msg, AIMessage)]
        
        total_input_tokens = 0
        total_output_tokens = 0
        
        for msg in ai_messages:
            if hasattr(msg, 'usage_metadata') and msg.usage_metadata:
                total_input_tokens += msg.usage_metadata.get('input_tokens', 0)
                total_output_tokens += msg.usage_metadata.get('output_tokens', 0)
        
        return {
            "total_ai_messages": len(ai_messages),
            "total_input_tokens": total_input_tokens,
            "total_output_tokens": total_output_tokens,
            "total_tokens": total_input_tokens + total_output_tokens
        }

# 使用最佳实践
system_prompt = MessageBestPractices.create_effective_system_prompt(
    "专业技术顾问",
    [
        "提供准确的技术信息",
        "使用代码示例说明概念", 
        "解释复杂概念时要清晰",
        "保持专业和友好的态度"
    ]
)

print("系统提示:", system_prompt.content)
```

### 2. 错误处理

```python
class MessageErrorHandler:
    """消息错误处理"""
    
    @staticmethod
    def safe_message_creation(content, message_type="human", **kwargs):
        """安全创建消息"""
        try:
            if message_type == "human":
                return HumanMessage(content, **kwargs)
            elif message_type == "ai":
                return AIMessage(content, **kwargs)
            elif message_type == "system":
                return SystemMessage(content, **kwargs)
            elif message_type == "tool":
                return ToolMessage(content, **kwargs)
            else:
                raise ValueError(f"未知的消息类型: {message_type}")
        except Exception as e:
            print(f"创建消息时出错: {e}")
            return None
    
    @staticmethod
    def validate_tool_message(tool_call_id: str, content: str) -> bool:
        """验证工具消息"""
        if not tool_call_id:
            print("错误: 工具消息缺少 tool_call_id")
            return False
        
        if not content:
            print("警告: 工具消息内容为空")
        
        return True
    
    @staticmethod
    def handle_streaming_errors(stream_generator):
        """处理流式错误"""
        try:
            for chunk in stream_generator:
                yield chunk
        except Exception as e:
            print(f"流式处理错误: {e}")
            # 返回错误消息
            yield AIMessage(f"抱歉，处理过程中出现错误: {str(e)}")

# 使用错误处理
safe_msg = MessageErrorHandler.safe_message_creation(
    "正常内容",
    "human",
    name="test_user"
)

print("安全创建的消息:", safe_msg.content if safe_msg else "创建失败")
```

## 总结

LangChain Messages 提供了强大而灵活的方式来管理AI对话：

- **多种消息类型**：SystemMessage、HumanMessage、AIMessage、ToolMessage
- **丰富的内容支持**：文本、图像、音频、视频、文档等多模态内容
- **标准化内容块**：跨提供商的一致内容表示
- **工具调用集成**：完整的工具调用和工作流支持
- **生产级特性**：错误处理、性能优化、使用统计

通过合理使用Messages，可以构建出能够处理复杂对话、多模态内容和工具调用的智能AI应用系统。
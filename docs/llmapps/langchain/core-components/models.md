# LangChain Models 

## 概述

大语言模型（LLMs）是能够像人类一样解释和生成文本的强大AI工具。它们足够通用，可以编写内容、翻译语言、总结和回答问题，而无需为每个任务进行专门训练。

除了文本生成，许多模型还支持：

- **工具调用** - 调用外部工具并在响应中使用结果
- **结构化输出** - 模型的响应被约束为遵循定义的格式
- **多模态** - 处理和返回文本以外的数据，如图像、音频和视频
- **推理** - 模型执行多步推理以得出结论

模型是Agent的推理引擎，驱动Agent的决策过程。您选择的模型的质量和能力直接影响Agent的可靠性和性能。

## 基础用法

### 1. 初始化模型

#### 使用 `init_chat_model`（推荐）

```python
import os
from langchain.chat_models import init_chat_model

# 设置API密钥
os.environ["OPENAI_API_KEY"] = "sk-..."

# 初始化模型
model = init_chat_model("openai:gpt-4o")

# 基本调用
response = model.invoke("为什么鹦鹉会说话？")
print(response.content)
```

#### 使用模型类

```python
from langchain_openai import ChatOpenAI

# 直接使用模型类
model = ChatOpenAI(
    model="gpt-4o",
    temperature=0.7,
    max_tokens=1000,
    timeout=30
)

response = model.invoke("解释量子计算")
print(response.content)
```

### 2. 支持的提供商

```python
# Anthropic
from langchain_anthropic import ChatAnthropic
model = ChatAnthropic(model="claude-3-5-sonnet-20241022")

# Google Gemini
from langchain_google_genai import ChatGoogleGenerativeAI
model = ChatGoogleGenerativeAI(model="gemini-2.0-flash-exp")

# Azure OpenAI
from langchain_openai import AzureChatOpenAI
model = AzureChatOpenAI(
    azure_deployment="your-deployment-name",
    openai_api_version="2023-05-15"
)
```

## 参数配置

### 常用参数

```python
model = init_chat_model(
    "openai:gpt-4o",
    # 核心参数
    temperature=0.7,      # 控制随机性 (0-1)
    max_tokens=1000,      # 最大输出长度
    timeout=30,           # 超时时间（秒）
    max_retries=3,        # 最大重试次数
    
    # 高级参数
    top_p=0.9,           # 核采样参数
    frequency_penalty=0.1, # 频率惩罚
    presence_penalty=0.1,  # 存在惩罚
)

response = model.invoke("写一个关于AI的短故事")
```

## 调用方式

### 1. 单次调用（Invoke）

```python
# 单条消息
response = model.invoke("Python的主要特点是什么？")
print(response.content)

# 对话历史
messages = [
    {"role": "system", "content": "你是一个有帮助的助手。"},
    {"role": "user", "content": "教我Python"},
    {"role": "assistant", "content": "Python是一种高级编程语言..."},
    {"role": "user", "content": "它的主要应用领域是什么？"}
]

response = model.invoke(messages)
print(response.content)
```

### 2. 流式调用（Stream）

```python
print("AI回复: ", end="", flush=True)

for chunk in model.stream("解释机器学习的基本概念"):
    if hasattr(chunk, 'content'):
        print(chunk.content, end="", flush=True)
print()  # 换行

# 或者累积完整的消息
full_response = None
for chunk in model.stream("天气如何影响心情？"):
    full_response = chunk if full_response is None else full_response + chunk

print(f"\n完整回复: {full_response.content}")
```

### 3. 批量调用（Batch）

```python
# 基本批量处理
questions = [
    "什么是人工智能？",
    "解释深度学习",
    "机器学习的应用场景",
    "神经网络如何工作"
]

responses = model.batch(questions)
for i, response in enumerate(responses):
    print(f"问题 {i+1}: {response.content[:100]}...")

# 异步完成批量处理
print("按完成顺序输出:")
for response in model.batch_as_completed(questions):
    print(f"收到回复: {response.content[:50]}...")

# 控制并发数
responses = model.batch(
    questions,
    config={'max_concurrency': 2}  # 限制同时2个请求
)
```

## 工具调用

### 1. 绑定工具

```python
from langchain.tools import tool

@tool
def get_weather(location: str) -> str:
    """获取指定位置的天气信息。"""
    return f"{location}的天气：晴朗，25°C"

@tool
def calculator(expression: str) -> str:
    """计算数学表达式。"""
    try:
        result = eval(expression)
        return f"{expression} = {result}"
    except:
        return "计算错误"

# 绑定工具到模型
model_with_tools = model.bind_tools([get_weather, calculator])

# 调用带工具的模型
response = model_with_tools.invoke("北京今天天气怎么样？然后计算 25 * 4")
print("工具调用:", response.tool_calls)
```

### 2. 工具执行循环

```python
def execute_tool_calls(model, messages, tools):
    """执行工具调用循环"""
    # 模型生成工具调用
    ai_msg = model.invoke(messages)
    messages.append(ai_msg)
    
    # 执行所有工具调用
    for tool_call in ai_msg.tool_calls:
        tool_name = tool_call["name"]
        tool_args = tool_call["args"]
        
        # 找到对应的工具并执行
        for tool in tools:
            if tool.name == tool_name:
                result = tool.invoke(tool_args)
                messages.append({
                    "role": "tool",
                    "content": result,
                    "tool_call_id": tool_call["id"]
                })
                break
    
    # 获取最终回复
    final_response = model.invoke(messages)
    return final_response

# 使用示例
tools = [get_weather, calculator]
messages = [{"role": "user", "content": "北京天气如何？然后计算 15 + 27"}]
result = execute_tool_calls(model_with_tools, messages, tools)
print("最终回复:", result.content)
```

### 3. 高级工具功能

```python
# 强制使用特定工具
forced_model = model.bind_tools(
    [get_weather], 
    tool_choice="get_weather"  # 强制使用天气工具
)

# 禁用并行工具调用
sequential_model = model.bind_tools(
    [get_weather, calculator],
    parallel_tool_calls=False  # 顺序执行工具
)

# 流式工具调用
print("流式工具调用:")
for chunk in model_with_tools.stream("查询北京和上海的天气"):
    if hasattr(chunk, 'tool_call_chunks') and chunk.tool_call_chunks:
        for tool_chunk in chunk.tool_call_chunks:
            if tool_chunk.get('name'):
                print(f"工具: {tool_chunk['name']}")
            if tool_chunk.get('args'):
                print(f"参数: {tool_chunk['args']}")
```

## 结构化输出

### 1. Pydantic 模型

```python
from pydantic import BaseModel, Field
from typing import List

class Movie(BaseModel):
    """电影信息"""
    title: str = Field(description="电影标题")
    year: int = Field(description="上映年份")
    director: str = Field(description="导演")
    rating: float = Field(description="评分(0-10)")
    genres: List[str] = Field(description="类型列表")

class ProductReview(BaseModel):
    """产品评价"""
    product_name: str = Field(description="产品名称")
    rating: int = Field(description="评分(1-5)")
    pros: List[str] = Field(description="优点")
    cons: List[str] = Field(description="缺点")
    summary: str = Field(description="总结")

# 使用结构化输出
structured_model = model.with_structured_output(Movie)
response = structured_model.invoke("提供电影《盗梦空间》的详细信息")

print(f"标题: {response.title}")
print(f"年份: {response.year}")
print(f"导演: {response.director}")
print(f"评分: {response.rating}")
```

### 2. 包含原始响应

```python
# 同时获取解析结果和原始消息
structured_model_with_raw = model.with_structured_output(
    Movie, 
    include_raw=True
)

result = structured_model_with_raw.invoke("描述电影《阿凡达》")

print("解析结果:", result.parsed)
print("原始消息:", result.raw)
print("解析错误:", result.parsing_error)
```

### 3. 复杂嵌套结构

```python
from typing import Optional

class Actor(BaseModel):
    """演员信息"""
    name: str = Field(description="演员姓名")
    character: str = Field(description="扮演角色")

class MovieDetails(BaseModel):
    """详细电影信息"""
    title: str = Field(description="电影标题")
    year: int = Field(description="上映年份")
    director: str = Field(description="导演")
    cast: List[Actor] = Field(description="演员表")
    budget: Optional[float] = Field(description="预算（百万美元）")
    box_office: Optional[float] = Field(description="票房（百万美元）")

# 使用嵌套结构
detailed_model = model.with_structured_output(MovieDetails)
response = detailed_model.invoke("提供《泰坦尼克号》的完整信息")

print(f"电影: {response.title} ({response.year})")
print(f"导演: {response.director}")
print("主演:")
for actor in response.cast:
    print(f"  - {actor.name} 饰演 {actor.character}")
```

## 高级功能

### 1. 多模态处理

```python
from langchain_core.messages import HumanMessage
import base64

# 处理图像（模拟）
def encode_image(image_path):
    """编码图像为base64"""
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')

# 创建多模态消息
multimodal_message = [
    {
        "type": "text",
        "text": "描述这张图片中的内容"
    },
    {
        "type": "image_url",
        "image_url": {
            "url": "data:image/jpeg;base64,..."  # 实际使用中替换为真实base64数据
        }
    }
]

# 支持多模态的模型调用
response = model.invoke(multimodal_message)
print("图像描述:", response.content)
```

### 2. 推理过程

```python
# 流式推理过程
print("推理过程:")
for chunk in model.stream("为什么天空是蓝色的？"):
    # 检查推理块
    if hasattr(chunk, 'content_blocks'):
        for block in chunk.content_blocks:
            if block.get("type") == "reasoning" and block.get("reasoning"):
                print(f"推理: {block['reasoning']}")
            elif block.get("type") == "text" and block.get("text"):
                print(f"回答: {block['text']}")

# 获取完整推理
response = model.invoke("解释全球变暖的原因", reasoning_effort="high")
reasoning_blocks = [b for b in response.content_blocks if b.get("type") == "reasoning"]
if reasoning_blocks:
    print("完整推理过程:")
    for block in reasoning_blocks:
        print(block.get("reasoning", ""))
```

### 3. 本地模型

```python
# 使用 Ollama 运行本地模型
from langchain_community.chat_models import ChatOllama

local_model = ChatOllama(
    model="llama2",  # 或其它本地模型
    temperature=0.7,
    num_predict=1000
)

response = local_model.invoke("用中文解释机器学习")
print("本地模型回复:", response.content)
```

### 4. 速率限制

```python
from langchain_core.rate_limiters import InMemoryRateLimiter

# 创建速率限制器
rate_limiter = InMemoryRateLimiter(
    requests_per_second=1,      # 每秒1个请求
    check_every_n_seconds=0.1,  # 每100ms检查一次
    max_bucket_size=5          # 最大突发请求数
)

model_with_limiter = init_chat_model(
    "openai:gpt-4o",
    rate_limiter=rate_limiter
)

# 受速率限制的调用
for i in range(3):
    response = model_with_limiter.invoke(f"问题 {i+1}: 什么是AI？")
    print(f"回复 {i+1}: {response.content[:50]}...")
```

### 5. 令牌使用统计

```python
from langchain_core.callbacks import get_usage_metadata_callback

# 使用上下文管理器跟踪令牌使用
with get_usage_metadata_callback() as callback:
    response1 = model.invoke("解释神经网络")
    response2 = model.invoke("什么是深度学习")
    
    print("令牌使用统计:")
    print(callback.usage_metadata)

# 直接从响应获取令牌信息
response = model.invoke("写一个Python函数计算斐波那契数列")
if hasattr(response, 'response_metadata'):
    usage = response.response_metadata.get('token_usage', {})
    print(f"输入令牌: {usage.get('prompt_tokens', 'N/A')}")
    print(f"输出令牌: {usage.get('completion_tokens', 'N/A')}")
    print(f"总令牌: {usage.get('total_tokens', 'N/A')}")
```

### 6. 可配置模型

```python
# 创建运行时可配置的模型
configurable_model = init_chat_model(
    temperature=0,
    configurable_fields=("model", "temperature", "max_tokens")
)

# 使用不同配置调用
response1 = configurable_model.invoke(
    "解释机器学习",
    config={"configurable": {"model": "gpt-4o", "temperature": 0.7}}
)

response2 = configurable_model.invoke(
    "写一首诗", 
    config={"configurable": {"model": "gpt-4o", "temperature": 0.9}}
)

print("技术解释:", response1.content[:100])
print("诗歌创作:", response2.content[:100])
```

## 实际应用场景

### 场景1：内容生成

```python
class ContentGenerator:
    """内容生成器"""
    
    def __init__(self, model):
        self.model = model
    
    def generate_blog_post(self, topic: str, style: str = "informative") -> str:
        """生成博客文章"""
        prompt = f"""
        以{style}的风格写一篇关于{topic}的博客文章。
        要求：
        1. 标题吸引人
        2. 结构清晰（引言、正文、结论）
        3. 包含具体例子
        4. 字数800-1000字
        """
        
        response = self.model.invoke(prompt)
        return response.content
    
    def generate_social_media_post(self, topic: str, platform: str) -> str:
        """生成社交媒体帖子"""
        platform_formats = {
            "twitter": "280字符以内，使用话题标签",
            "linkedin": "专业风格，聚焦行业见解",
            "instagram": "轻松有趣，使用表情符号"
        }
        
        format_guide = platform_formats.get(platform, "简洁有力")
        prompt = f"为{platform}创建关于{topic}的帖子。要求：{format_guide}"
        
        response = self.model.invoke(prompt)
        return response.content

# 使用示例
generator = ContentGenerator(model)
blog_post = generator.generate_blog_post("人工智能的未来", "专业")
twitter_post = generator.generate_social_media_post("机器学习", "twitter")

print("博客文章:", blog_post[:200])
print("推特帖子:", twitter_post)
```

### 场景2：数据分析助手

```python
import json
from typing import Dict, Any

class DataAnalysisAssistant:
    """数据分析助手"""
    
    def __init__(self, model):
        self.model = model
        # 配置结构化输出用于数据分析
        self.analysis_model = model.with_structured_output(DataAnalysisResult)
    
    def analyze_dataset(self, data_description: str, questions: List[str]) -> Dict[str, Any]:
        """分析数据集"""
        prompt = f"""
        数据集描述: {data_description}
        
        请分析这个数据集并回答以下问题:
        {chr(10).join(f'{i+1}. {q}' for i, q in enumerate(questions))}
        
        提供:
        - 关键洞察
        - 潜在模式
        - 建议的进一步分析
        """
        
        response = self.analysis_model.invoke(prompt)
        return response.dict()
    
    def generate_sql_query(self, requirement: str, schema: str) -> str:
        """生成SQL查询"""
        prompt = f"""
        数据库模式: {schema}
        
        需求: {requirement}
        
        请生成一个优化的SQL查询来满足这个需求。
        同时解释查询的逻辑。
        """
        
        response = self.model.invoke(prompt)
        return response.content

# 数据结构定义
class DataAnalysisResult(BaseModel):
    key_insights: List[str]
    patterns: List[str]
    recommendations: List[str]
    summary: str

# 使用示例
assistant = DataAnalysisAssistant(model)

schema = """
用户表(users): id, name, age, city, signup_date
订单表(orders): id, user_id, amount, order_date, status
"""

analysis = assistant.analyze_dataset(
    "电商平台的用户和订单数据",
    ["用户年龄分布如何？", "哪个城市的用户最活跃？", "订单趋势如何？"]
)

sql_query = assistant.generate_sql_query(
    "查询最近30天每个城市的订单总量",
    schema
)

print("分析结果:", analysis)
print("SQL查询:", sql_query)
```

### 场景3：代码助手

```python
class CodeAssistant:
    """代码助手"""
    
    def __init__(self, model):
        self.model = model
    
    def explain_code(self, code: str, language: str = "python") -> str:
        """解释代码"""
        prompt = f"""
        请解释以下{language}代码:
        
        ```{language}
        {code}
        ```
        
        解释应该包括:
        1. 代码的功能
        2. 关键逻辑步骤
        3. 可能的改进建议
        """
        
        response = self.model.invoke(prompt)
        return response.content
    
    def debug_code(self, code: str, error: str, language: str = "python") -> str:
        """调试代码"""
        prompt = f"""
        请帮助调试以下{language}代码:
        
        ```{language}
        {code}
        ```
        
        错误信息: {error}
        
        请提供:
        1. 错误原因分析
        2. 修复建议
        3. 修复后的代码
        """
        
        response = self.model.invoke(prompt)
        return response.content
    
    def generate_test_cases(self, code: str, language: str = "python") -> str:
        """生成测试用例"""
        prompt = f"""
        为以下{language}代码生成测试用例:
        
        ```{language}
        {code}
        ```
        
        包括:
        1. 正常情况测试
        2. 边界情况测试  
        3. 错误情况测试
        """
        
        response = self.model.invoke(prompt)
        return response.content

# 使用示例
code_assistant = CodeAssistant(model)

sample_code = """
def fibonacci(n):
    if n <= 1:
        return n
    else:
        return fibonacci(n-1) + fibonacci(n-2)
"""

explanation = code_assistant.explain_code(sample_code)
test_cases = code_assistant.generate_test_cases(sample_code)

print("代码解释:", explanation)
print("测试用例:", test_cases)
```

## 最佳实践

### 1. 错误处理

```python
from tenacity import retry, stop_after_attempt, wait_exponential
import asyncio

class RobustModelClient:
    """健壮的模型客户端"""
    
    def __init__(self, model):
        self.model = model
    
    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=4, max=10)
    )
    def invoke_with_retry(self, prompt: str, **kwargs):
        """带重试的调用"""
        try:
            return self.model.invoke(prompt, **kwargs)
        except Exception as e:
            print(f"调用失败: {e}, 进行重试...")
            raise
    
    def safe_batch_process(self, prompts: List[str], batch_size: int = 5):
        """安全的批量处理"""
        results = []
        
        for i in range(0, len(prompts), batch_size):
            batch = prompts[i:i + batch_size]
            try:
                batch_results = self.model.batch(
                    batch, 
                    config={'max_concurrency': 2}
                )
                results.extend(batch_results)
                print(f"完成批次 {i//batch_size + 1}")
            except Exception as e:
                print(f"批次 {i//batch_size + 1} 失败: {e}")
                # 可以在这里添加重试逻辑
        
        return results

# 使用示例
robust_client = RobustModelClient(model)

try:
    response = robust_client.invoke_with_retry(
        "解释量子力学", 
        temperature=0.7
    )
    print("成功获取响应")
except Exception as e:
    print(f"所有重试都失败了: {e}")
```

### 2. 性能优化

```python
import time
from functools import lru_cache

class OptimizedModelHandler:
    """优化的模型处理器"""
    
    def __init__(self, model):
        self.model = model
        self.response_cache = {}
    
    @lru_cache(maxsize=100)
    def cached_invoke(self, prompt: str, temperature: float = 0.7) -> str:
        """带缓存的调用"""
        cache_key = hash(prompt + str(temperature))
        
        if cache_key in self.response_cache:
            return self.response_cache[cache_key]
        
        start_time = time.time()
        response = self.model.invoke(prompt, temperature=temperature)
        execution_time = time.time() - start_time
        
        self.response_cache[cache_key] = response.content
        print(f"新请求 - 耗时: {execution_time:.2f}s")
        
        return response.content
    
    def batch_optimized(self, prompts: List[str], **kwargs):
        """优化的批量处理"""
        # 去重
        unique_prompts = list(set(prompts))
        
        # 批量处理唯一提示
        unique_responses = self.model.batch(unique_prompts, **kwargs)
        
        # 构建响应映射
        response_map = {prompt: resp.content for prompt, resp in zip(unique_prompts, unique_responses)}
        
        # 按原始顺序返回
        return [response_map[prompt] for prompt in prompts]

# 使用示例
optimized_handler = OptimizedModelHandler(model)

# 重复请求会使用缓存
for i in range(3):
    result = optimized_handler.cached_invoke("什么是Python？")
    print(f"请求 {i+1}: {result[:50]}...")
```

### 3. 成本控制

```python
class CostAwareModelClient:
    """成本感知的模型客户端"""
    
    def __init__(self, model, budget_limit: int = 1000000):  # 100万token限制
        self.model = model
        self.budget_limit = budget_limit
        self.tokens_used = 0
        self.requests_count = 0
    
    def track_usage(self, response):
        """跟踪令牌使用"""
        if hasattr(response, 'response_metadata'):
            usage = response.response_metadata.get('token_usage', {})
            tokens = usage.get('total_tokens', 0)
            self.tokens_used += tokens
            self.requests_count += 1
            
            print(f"本次使用: {tokens} tokens")
            print(f"累计使用: {self.tokens_used}/{self.budget_limit} tokens")
            
            if self.tokens_used >= self.budget_limit:
                print("警告: 接近预算限制！")
    
    def invoke_with_budget(self, prompt: str, **kwargs):
        """带预算控制的调用"""
        if self.tokens_used >= self.budget_limit:
            raise Exception("已超过预算限制")
        
        response = self.model.invoke(prompt, **kwargs)
        self.track_usage(response)
        return response
    
    def get_usage_stats(self):
        """获取使用统计"""
        return {
            'tokens_used': self.tokens_used,
            'requests_count': self.requests_count,
            'budget_remaining': self.budget_limit - self.tokens_used,
            'utilization_percentage': (self.tokens_used / self.budget_limit) * 100
        }

# 使用示例
cost_aware_client = CostAwareModelClient(model, budget_limit=5000)  # 5000token测试限制

try:
    response1 = cost_aware_client.invoke_with_budget("解释机器学习")
    response2 = cost_aware_client.invoke_with_budget("什么是深度学习")
    
    stats = cost_aware_client.get_usage_stats()
    print("使用统计:", stats)
    
except Exception as e:
    print(f"调用失败: {e}")
```

## 总结

LangChain Models 提供了强大而灵活的方式来使用各种大语言模型：

- **多提供商支持**：OpenAI、Anthropic、Google、Azure等
- **多种调用方式**：单次调用、流式调用、批量调用
- **工具集成**：绑定和执行外部工具
- **结构化输出**：确保响应格式符合预期
- **高级功能**：多模态、推理、本地部署等
- **生产就绪**：错误处理、性能优化、成本控制

通过合理使用这些功能，您可以构建出强大、可靠且成本效益高的AI应用系统。
# Model Context Protocol (MCP) 使用教程

## 概述

Model Context Protocol (MCP) 是一个开放协议，标准化了应用程序如何向 LLM 提供工具和上下文。LangChain 智能体可以使用 [`langchain-mcp-adapters`](https://github.com/langchain-ai/langchain-mcp-adapters) 库来使用 MCP 服务器上定义的工具。

## 安装

安装 `langchain-mcp-adapters` 库以在 LangGraph 中使用 MCP 工具：


```bash  
  pip install langchain-mcp-adapters
```

## 传输类型

MCP 支持不同的客户端-服务器通信传输机制：

* **stdio**：客户端将服务器作为子进程启动，通过标准输入/输出进行通信。适用于本地工具和简单设置。
* **Streamable HTTP**：服务器作为独立进程运行，处理 HTTP 请求。支持远程连接和多个客户端。
* **Server-Sent Events (SSE)**：Streamable HTTP 的变体，针对实时流通信进行了优化。

## 使用 MCP 工具

`langchain-mcp-adapters` 使智能体能够使用一个或多个 MCP 服务器上定义的工具。

### 访问多个 MCP 服务器

```python
from langchain_mcp_adapters.client import MultiServerMCPClient
from langchain.agents import create_agent

# 配置多个 MCP 服务器
client = MultiServerMCPClient(
    {
        "math": {
            "transport": "stdio",  # 本地子进程通信
            "command": "python",
            # 指向你的 math_server.py 文件的绝对路径
            "args": ["/path/to/math_server.py"],
        },
        "weather": {
            "transport": "streamable_http",  # 基于 HTTP 的远程服务器
            # 确保你的天气服务器在端口 8000 上启动
            "url": "http://localhost:8000/mcp",
        }
    }
)

# 获取所有工具
tools = await client.get_tools()
agent = create_agent(
    "anthropic:claude-sonnet-4-5",
    tools  # 使用 MCP 工具
)

# 使用数学工具
math_response = await agent.ainvoke(
    {"messages": [{"role": "user", "content": "what's (3 + 5) x 12?"}]}
)

# 使用天气工具
weather_response = await agent.ainvoke(
    {"messages": [{"role": "user", "content": "what is the weather in nyc?"}]}
)
```

**注意**：`MultiServerMCPClient` **默认是无状态的**。每个工具调用都会创建一个新的 MCP `ClientSession`，执行工具，然后进行清理。

## 创建自定义 MCP 服务器

要创建自己的 MCP 服务器，可以使用 `mcp` 库。该库提供了一种简单的方法来定义[工具](https://modelcontextprotocol.io/docs/learn/server-concepts#tools-ai-actions)并将其作为服务器运行。

<CodeGroup>
  ```bash pip
  pip install mcp
  ```

  ```bash uv
  uv add mcp
  ```
</CodeGroup>

### 数学服务器示例（stdio 传输）

```python
from mcp.server.fastmcp import FastMCP

# 创建 MCP 服务器实例
mcp = FastMCP("Math")

@mcp.tool()
def add(a: int, b: int) -> int:
    """Add two numbers"""
    return a + b

@mcp.tool()
def multiply(a: int, b: int) -> int:
    """Multiply two numbers"""
    return a * b

@mcp.tool()
def divide(a: float, b: float) -> float:
    """Divide two numbers"""
    if b == 0:
        return "Error: Division by zero"
    return a / b

if __name__ == "__main__":
    # 使用 stdio 传输运行服务器
    mcp.run(transport="stdio")
```

### 天气服务器示例（streamable HTTP 传输）

```python
from mcp.server.fastmcp import FastMCP
import asyncio

mcp = FastMCP("Weather")

@mcp.tool()
async def get_weather(location: str) -> str:
    """Get weather for location."""
    # 模拟天气数据获取
    await asyncio.sleep(0.1)  # 模拟网络延迟
    weather_data = {
        "New York": "Sunny, 25°C",
        "London": "Cloudy, 15°C", 
        "Tokyo": "Rainy, 20°C",
        "Beijing": "Clear, 22°C"
    }
    return weather_data.get(location, f"Weather data not available for {location}")

@mcp.tool()
async def get_forecast(location: str, days: int = 3) -> str:
    """Get weather forecast for location."""
    forecasts = {
        1: "Sunny tomorrow",
        3: "Sunny, then cloudy, then rainy",
        5: "Mixed conditions throughout the week"
    }
    return forecasts.get(days, f"Forecast for {days} days not available")

if __name__ == "__main__":
    # 使用 streamable HTTP 传输运行服务器
    mcp.run(transport="streamable-http", host="localhost", port=8000)
```

### 文件操作服务器示例

```python
from mcp.server.fastmcp import FastMCP
import os
import json

mcp = FastMCP("FileOperations")

@mcp.tool()
def read_file(filepath: str) -> str:
    """Read content from a file."""
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            return f.read()
    except FileNotFoundError:
        return f"Error: File {filepath} not found"
    except Exception as e:
        return f"Error reading file: {str(e)}"

@mcp.tool()
def write_file(filepath: str, content: str) -> str:
    """Write content to a file."""
    try:
        # 确保目录存在
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(content)
        return f"Successfully wrote to {filepath}"
    except Exception as e:
        return f"Error writing file: {str(e)}"

@mcp.tool()
def list_files(directory: str = ".") -> str:
    """List files in a directory."""
    try:
        files = os.listdir(directory)
        return json.dumps(files, indent=2)
    except Exception as e:
        return f"Error listing directory: {str(e)}"

if __name__ == "__main__":
    mcp.run(transport="stdio")
```

## 状态化工具使用

对于需要在工具调用之间维护上下文的状态化服务器，使用 `client.session()` 创建持久的 `ClientSession`。

```python
from langchain_mcp_adapters.tools import load_mcp_tools
from langchain_mcp_adapters.client import MultiServerMCPClient

# 配置 MCP 客户端
client = MultiServerMCPClient({
    "database": {
        "transport": "stdio",
        "command": "python",
        "args": ["/path/to/database_server.py"],
    }
})

# 使用会话进行状态化操作
async with client.session("database") as session:
    tools = await load_mcp_tools(session)
    
    # 创建使用这些工具的智能体
    agent = create_agent(
        "openai:gpt-4o",
        tools
    )
    
    # 在会话中执行多个相关操作
    result1 = await agent.ainvoke({
        "messages": [{"role": "user", "content": "Connect to the database"}]
    })
    
    result2 = await agent.ainvoke({
        "messages": [{"role": "user", "content": "Query user data"}]
    })
```

### 状态化数据库服务器示例

```python
from mcp.server.fastmcp import FastMCP
from typing import Dict, Any

mcp = FastMCP("Database")

class DatabaseSession:
    def __init__(self):
        self.connection = None
        self.data = {}  # 模拟数据库数据
    
    def connect(self, database_name: str) -> str:
        """连接到数据库"""
        self.connection = database_name
        self.data = {
            "users": [
                {"id": 1, "name": "Alice", "email": "alice@example.com"},
                {"id": 2, "name": "Bob", "email": "bob@example.com"}
            ],
            "orders": [
                {"id": 1, "user_id": 1, "product": "Laptop", "amount": 1200},
                {"id": 2, "user_id": 2, "product": "Mouse", "amount": 50}
            ]
        }
        return f"Connected to database: {database_name}"
    
    def query(self, table: str, conditions: Dict[str, Any] = None) -> str:
        """查询数据库表"""
        if not self.connection:
            return "Error: Not connected to any database"
        
        if table not in self.data:
            return f"Error: Table {table} not found"
        
        results = self.data[table]
        if conditions:
            # 简单的条件过滤
            filtered_results = []
            for item in results:
                match = True
                for key, value in conditions.items():
                    if item.get(key) != value:
                        match = False
                        break
                if match:
                    filtered_results.append(item)
            results = filtered_results
        
        return json.dumps(results, indent=2)
    
    def insert(self, table: str, data: Dict[str, Any]) -> str:
        """向表中插入数据"""
        if not self.connection:
            return "Error: Not connected to any database"
        
        if table not in self.data:
            self.data[table] = []
        
        # 生成新ID
        new_id = max([item.get('id', 0) for item in self.data[table]], default=0) + 1
        data['id'] = new_id
        self.data[table].append(data)
        
        return f"Inserted record with ID {new_id} into {table}"

# 创建会话管理器
session_manager = DatabaseSession()

@mcp.tool()
def connect_to_database(database_name: str) -> str:
    """Connect to a database"""
    return session_manager.connect(database_name)

@mcp.tool()
def query_table(table: str, user_id: int = None) -> str:
    """Query data from a table"""
    conditions = {}
    if user_id is not None:
        conditions['user_id'] = user_id
    return session_manager.query(table, conditions)

@mcp.tool()
def insert_record(table: str, data: Dict[str, Any]) -> str:
    """Insert a new record into a table"""
    return session_manager.insert(table, data)

if __name__ == "__main__":
    mcp.run(transport="stdio")
```

## 完整示例：集成多个 MCP 服务器的智能体

```python
import asyncio
from langchain_mcp_adapters.client import MultiServerMCPClient
from langchain.agents import create_agent
from langchain.agents.middleware import dynamic_prompt

@dynamic_prompt
def mcp_tools_prompt(request) -> str:
    """动态提示，说明可用的 MCP 工具"""
    tool_names = [tool.name for tool in request.tools]
    return f"""
你是一个智能助手，可以访问以下工具：
{', '.join(tool_names)}

请根据用户请求选择合适的工具来完成任务。
"""

async def main():
    # 配置多个 MCP 服务器
    client = MultiServerMCPClient(
        {
            "math": {
                "transport": "stdio",
                "command": "python", 
                "args": ["math_server.py"],
            },
            "weather": {
                "transport": "streamable_http",
                "url": "http://localhost:8000/mcp",
            },
            "files": {
                "transport": "stdio",
                "command": "python",
                "args": ["file_server.py"],
            }
        }
    )
    
    # 获取所有工具
    tools = await client.get_tools()
    
    # 创建智能体
    agent = create_agent(
        "anthropic:claude-sonnet-4-5",
        tools,
        middleware=[mcp_tools_prompt]
    )
    
    # 测试不同的工具
    test_cases = [
        "计算 15 乘以 8 加上 23 等于多少？",
        "纽约的天气怎么样？",
        "列出当前目录的文件",
        "创建一个名为 test.txt 的文件，内容为 'Hello MCP!'"
    ]
    
    for question in test_cases:
        print(f"\n问题: {question}")
        response = await agent.ainvoke({
            "messages": [{"role": "user", "content": question}]
        })
        print(f"回答: {response['messages'][-1].content}")

if __name__ == "__main__":
    asyncio.run(main())
```

## 最佳实践

### 1. 错误处理

```python
from langchain_mcp_adapters.client import MultiServerMCPClient
import asyncio

async def safe_mcp_operation():
    try:
        client = MultiServerMCPClient({
            "math": {
                "transport": "stdio",
                "command": "python",
                "args": ["math_server.py"],
            }
        })
        
        # 检查服务器是否可用
        async with client.session("math") as session:
            tools = await session.list_tools()
            if not tools:
                print("警告：数学服务器没有提供任何工具")
                return
            
            # 正常使用工具...
            
    except Exception as e:
        print(f"MCP 操作失败: {e}")
        # 回退到不使用 MCP 工具的模式
```

### 2. 性能优化

```python
import asyncio
from langchain_mcp_adapters.client import MultiServerMCPClient

class MCPManager:
    def __init__(self):
        self.client = None
        self.tools_cache = None
    
    async def initialize(self):
        """延迟初始化 MCP 客户端"""
        if self.client is None:
            self.client = MultiServerMCPClient({
                "math": {"transport": "stdio", "command": "python", "args": ["math_server.py"]},
                "weather": {"transport": "streamable_http", "url": "http://localhost:8000/mcp"},
            })
            self.tools_cache = await self.client.get_tools()
    
    async def get_tools(self):
        """获取缓存的工具"""
        if self.tools_cache is None:
            await self.initialize()
        return self.tools_cache

# 使用管理器
mcp_manager = MCPManager()
```

### 3. 配置管理

```python
import yaml
from langchain_mcp_adapters.client import MultiServerMCPClient

def load_mcp_config(config_path: str):
    """从 YAML 文件加载 MCP 配置"""
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    
    return MultiServerMCPClient(config)

# config.yaml 示例
"""
math:
  transport: stdio
  command: python
  args: ["/apps/math_server.py"]
  
weather:
  transport: streamable_http  
  url: "http://weather-service:8000/mcp"
  
files:
  transport: stdio
  command: python
  args: ["/apps/file_server.py"]
"""
```

## 故障排除

### 常见问题

1. **服务器连接失败**
     - 检查服务器进程是否正在运行
     - 验证传输类型和连接参数
     - 检查防火墙和网络设置

2. **工具不可用**
     - 确保服务器正确注册了工具
     - 检查工具名称和参数定义
     - 验证 MCP 协议版本兼容性

3. **性能问题**
     - 对于频繁使用的工具，考虑使用会话模式
     - 优化服务器实现，减少启动时间
     - 使用 HTTP 传输时考虑连接池

## 附加资源

* [MCP 官方文档](https://modelcontextprotocol.io/introduction)
* [MCP 传输文档](https://modelcontextprotocol.io/docs/concepts/transports)
* [`langchain-mcp-adapters` GitHub 仓库](https://github.com/langchain-ai/langchain-mcp-adapters)

通过本教程，你应该能够成功地在 LangChain 智能体中集成和使用 MCP 工具，创建更加强大和可扩展的 AI 应用。
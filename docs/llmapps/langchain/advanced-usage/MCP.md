# 🧭 Model Context Protocol (MCP) 教程

## 一、什么是 MCP？

**Model Context Protocol（MCP）** 是一种开放协议，用于标准化 **应用程序与大语言模型（LLM）之间** 的上下文与工具交互方式。
简单来说，它定义了一种“语言模型访问外部世界的统一方式”。

传统上，LLM 只能处理纯文本输入，但通过 MCP，你可以让模型访问：

* 本地或远程工具（Tool）
* 外部 API 或服务（例如天气、计算、数据库）
* 统一上下文管理（Context Management）

MCP 的目标是让不同平台的工具、插件、上下文模块之间实现互操作，就像 HTTP 让不同网站之间可以交互一样。

---

## 二、MCP 与 LangChain 的结合

LangChain 提供了 [`langchain-mcp-adapters`](https://github.com/langchain-ai/langchain-mcp-adapters) 库，使 LangChain Agent 可以直接访问 MCP 定义的工具。
这意味着你可以像调用普通函数一样，让 Agent 使用外部 MCP Server 提供的功能。

---

## 三、安装

### 使用 `pip`

```bash
pip install langchain-mcp-adapters
```

### 使用 `uv`

```bash
uv add langchain-mcp-adapters
```

安装完成后，你就可以在 LangGraph 或 LangChain 中使用 MCP 工具了。

---

## 四、MCP 的通信机制（Transport Types）

MCP 支持多种通信机制（传输层协议）用于客户端与服务器之间的数据交互：

| 类型                           | 说明                               | 适用场景        |
| ---------------------------- | -------------------------------- | ----------- |
| **stdio**                    | 使用标准输入/输出进行通信，客户端启动服务器进程并通过管道通信。 | 本地运行、简单调试   |
| **streamable-http**          | 使用 HTTP 通信，支持多客户端并发访问。           | 远程部署、Web 服务 |
| **SSE (Server-Sent Events)** | HTTP 的实时流式通信变体。                  | 实时数据更新，如聊天流 |

例如：

* `stdio` 更适合运行在你本机的轻量工具。
* `streamable_http` 则适合部署在云端的多用户环境。

---

## 五、在 LangChain 中使用 MCP 工具

`MultiServerMCPClient` 允许连接多个 MCP Server，让一个 Agent 同时使用多个外部工具。

### 示例代码

```python
from langchain_mcp_adapters.client import MultiServerMCPClient
from langchain.agents import create_agent

# 定义两个 MCP Server：一个数学计算，一个天气服务
client = MultiServerMCPClient({
    "math": {
        "transport": "stdio",
        "command": "python",
        "args": ["/path/to/math_server.py"],  # 本地路径
    },
    "weather": {
        "transport": "streamable_http",
        "url": "http://localhost:8000/mcp",  # 远程服务
    }
})

tools = await client.get_tools()

agent = create_agent("anthropic:claude-sonnet-4-5", tools)

# 让模型调用 math 工具
math_response = await agent.ainvoke({
    "messages": [{"role": "user", "content": "what's (3 + 5) x 12?"}]
})

# 调用 weather 工具
weather_response = await agent.ainvoke({
    "messages": [{"role": "user", "content": "what is the weather in nyc?"}]
})
```

> ⚙️ 注意：
> `MultiServerMCPClient` 默认是**无状态的（stateless）**，每次调用都会重新创建一个会话并在执行后关闭。

---

## 六、自定义 MCP Server

要创建自己的 MCP Server，可以使用 [`mcp`](https://pypi.org/project/mcp/) 库。
这个库让你很容易定义工具（Tool）并将其暴露为 MCP 服务。

### 安装

```bash
pip install mcp
```

---

### 示例 1：数学服务器（本地 stdio 模式）

```python
from mcp.server.fastmcp import FastMCP

mcp = FastMCP("Math")

@mcp.tool()
def add(a: int, b: int) -> int:
    """加法"""
    return a + b

@mcp.tool()
def multiply(a: int, b: int) -> int:
    """乘法"""
    return a * b

if __name__ == "__main__":
    mcp.run(transport="stdio")
```

运行这个脚本后，它会作为一个本地 MCP Server 运行，客户端可以通过 `stdio` 调用。

---

### 示例 2：天气服务器（HTTP 模式）

```python
from mcp.server.fastmcp import FastMCP

mcp = FastMCP("Weather")

@mcp.tool()
async def get_weather(location: str) -> str:
    """获取指定城市的天气"""
    return "It's always sunny in New York"

if __name__ == "__main__":
    mcp.run(transport="streamable-http")
```

启动后，该服务会监听一个 HTTP 端口（默认 8000），支持远程访问。

---

## 七、保持有状态的会话（Stateful Session）

有时你需要让 MCP Server 记住上一次的状态，比如上下文、缓存或用户数据。
这种情况下，可以使用 `client.session()` 创建持久会话。

### 示例

```python
from langchain_mcp_adapters.tools import load_mcp_tools
from langchain_mcp_adapters.client import MultiServerMCPClient

client = MultiServerMCPClient({...})

# 创建持久化 session
async with client.session("math") as session:
    tools = await load_mcp_tools(session)
    # 在同一个会话中多次调用工具
```

---

## 八、延伸阅读

* [官方 MCP 文档](https://modelcontextprotocol.io/introduction)
* [MCP Transport 机制说明](https://modelcontextprotocol.io/docs/concepts/transports)
* [langchain-mcp-adapters 源码](https://github.com/langchain-ai/langchain-mcp-adapters)

---

## 九、总结

MCP 为 LLM 工具调用建立了一个**开放、通用的协议层**。
它带来的关键优势包括：

1. **统一标准**：不同工具和服务之间实现一致的接口。
2. **多语言支持**：任何实现 MCP 协议的语言都能交互。
3. **LangChain 无缝集成**：可直接扩展 Agent 的外部能力。

在未来，MCP 很可能成为 AI Agent 世界中的“API 协议层”，让模型能够像浏览器访问网页那样访问工具与上下文。

---

如果你想，我可以接着写一篇 **“手动从零实现一个最小 MCP 工具链”** 教程，展示从自定义 Server 到 LangChain Agent 全流程调试。是否要继续？

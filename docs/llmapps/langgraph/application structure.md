# LangGraph 应用结构完整教程

## 概述

本教程将详细介绍如何构建一个完整的 LangGraph 应用程序。LangGraph 应用由一个或多个图、配置文件、依赖文件以及可选的环境变量文件组成。

## 核心概念

要成功部署 LangGraph 应用，需要提供以下关键组件：

### 1. 配置文件 (`langgraph.json`)
这是应用的核心配置文件，指定了依赖项、图和环境变量。

### 2. 图 (Graphs)
实现应用逻辑的流程图或状态机。

### 3. 依赖管理
应用运行所需的包依赖。

### 4. 环境变量
应用运行所需的环境配置。

## 项目结构详解

### Python 项目结构（使用 requirements.txt）

```plaintext
my-app/
├── my_agent/                    # 项目主代码目录
│   ├── utils/                   # 工具和工具函数目录
│   │   ├── __init__.py
│   │   ├── tools.py            # 图中使用的工具函数
│   │   ├── nodes.py            # 图节点函数
│   │   └── state.py            # 状态定义
│   ├── __init__.py
│   └── agent.py                # 图构建代码
├── .env                        # 环境变量文件
├── requirements.txt            # Python 包依赖
└── langgraph.json             # LangGraph 配置文件
```

### Python 项目结构（使用 pyproject.toml）

```plaintext
my-app/
├── my_agent/
│   ├── utils/
│   │   ├── __init__.py
│   │   ├── tools.py
│   │   ├── nodes.py
│   │   └── state.py
│   ├── __init__.py
│   └── agent.py
├── .env
├── langgraph.json
└── pyproject.toml             # 项目依赖配置
```

## 配置文件详解

### langgraph.json 结构

```json
{
  "dependencies": ["langchain_openai", "./your_package"],
  "graphs": {
    "my_agent": "./your_package/your_file.py:agent"
  },
  "env": "./.env"
}
```

### 配置参数说明

- **dependencies**: 依赖包列表，支持本地包和 PyPI 包
- **graphs**: 图定义，格式为 `"图名": "文件路径:变量名"`
- **env**: 环境变量文件路径

## 依赖管理

### 方法一：requirements.txt

```txt
langchain_openai>=0.1.0
langgraph>=0.1.0
requests>=2.31.0
```

### 方法二：pyproject.toml

```toml
[build-system]
requires = ["setuptools", "wheel"]

[project]
name = "my-agent"
dependencies = [
    "langchain_openai>=0.1.0",
    "langgraph>=0.1.0",
]

[tool.setuptools.packages.find]
where = ["."]
```

## 图定义示例

### agent.py - 图构建代码

```python
from langgraph.graph import Graph
from .utils.nodes import process_input, generate_response
from .utils.state import AgentState

def create_agent_graph():
    """创建代理图"""
    graph = Graph()
    
    # 添加节点
    graph.add_node("process", process_input)
    graph.add_node("generate", generate_response)
    
    # 设置边
    graph.set_entry_point("process")
    graph.add_edge("process", "generate")
    graph.set_finish_point("generate")
    
    return graph.compile()

# 导出的图实例
agent = create_agent_graph()
```

### state.py - 状态定义

```python
from typing import TypedDict, List, Annotated
from langgraph.graph import add_messages

class AgentState(TypedDict):
    messages: Annotated[List, add_messages]
    input_data: str
    processed_data: dict
```

### nodes.py - 节点函数

```python
from .state import AgentState

def process_input(state: AgentState) -> AgentState:
    """处理输入数据的节点"""
    # 处理逻辑
    state["processed_data"] = {"content": state["input_data"]}
    return state

def generate_response(state: AgentState) -> AgentState:
    """生成响应的节点"""
    # 响应生成逻辑
    state["messages"].append({"role": "assistant", "content": "Response"})
    return state
```

### tools.py - 工具函数

```python
from langchain.tools import tool

@tool
def search_tool(query: str) -> str:
    """搜索工具"""
    # 实现搜索逻辑
    return f"Search results for: {query}"
```

## 环境变量配置

### .env 文件示例

```env
OPENAI_API_KEY=your_openai_api_key_here
LANGCHAIN_API_KEY=your_langchain_api_key
DATABASE_URL=your_database_connection_string
```

## 部署配置最佳实践

### 1. 完整的 langgraph.json 配置

```json
{
  "dependencies": [
    "langchain_openai",
    "langgraph",
    "pydantic>=2.0.0",
    "./my_agent"
  ],
  "graphs": {
    "chat_agent": "./my_agent/agent.py:agent",
    "analysis_agent": "./my_agent/analysis.py:analyzer"
  },
  "env": "./.env",
  "dockerfile_lines": [
    "RUN apt-get update && apt-get install -y curl",
    "RUN pip install --upgrade pip"
  ]
}
```

### 2. 多图应用结构

对于包含多个图的应用：

```plaintext
multi-agent-app/
├── agents/
│   ├── chat_agent.py
│   ├── analysis_agent.py
│   └── utils/
├── shared/
│   ├── state.py
│   └── tools.py
├── .env
├── requirements.txt
└── langgraph.json
```

## 开发工作流程

### 步骤 1：初始化项目结构
```bash
mkdir my-langgraph-app
cd my-langgraph-app
mkdir -p my_agent/utils
touch my_agent/__init__.py my_agent/utils/__init__.py
```

### 步骤 2：创建核心文件
```bash
touch my_agent/agent.py my_agent/utils/state.py
touch my_agent/utils/nodes.py my_agent/utils/tools.py
touch .env requirements.txt langgraph.json
```

### 步骤 3：配置依赖和环境
```bash
# requirements.txt
echo "langchain_openai" >> requirements.txt
echo "langgraph" >> requirements.txt

# .env
echo "OPENAI_API_KEY=your_key_here" >> .env
```

### 步骤 4：测试和部署
```bash
# 本地测试
python -c "from my_agent.agent import agent; print('Graph compiled successfully')"

# 使用 LangGraph CLI 部署
langgraph deploy
```

## 故障排除

### 常见问题

1. **依赖解析失败**
   - 检查依赖包名称和版本
   - 验证本地包路径是否正确

2. **图加载错误**
   - 确认文件路径和变量名正确
   - 检查图编译是否有语法错误

3. **环境变量缺失**
   - 验证 `.env` 文件存在且格式正确
   - 检查生产环境变量配置

### 调试技巧

```python
# 在 agent.py 中添加调试信息
def create_agent_graph():
    print("开始构建图...")
    graph = Graph()
    # ... 构建逻辑
    compiled_graph = graph.compile()
    print("图构建完成")
    return compiled_graph
```

## 总结

通过本教程，您应该能够：

- ✅ 理解 LangGraph 应用的核心组件
- ✅ 创建标准的项目结构
- ✅ 配置正确的依赖管理
- ✅ 定义和构建功能图
- ✅ 设置环境变量和配置文件
- ✅ 准备应用进行部署

遵循这些最佳实践将确保您的 LangGraph 应用结构清晰、易于维护，并能够顺利部署到生产环境。
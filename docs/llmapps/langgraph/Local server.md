# LangGraph 本地服务器运行完全指南

本教程将指导你如何在本地运行 LangGraph 应用程序，包括安装、配置、启动和测试完整流程。

## 前置要求

开始之前，请确保你已准备好：

* **Python 3.11 或更高版本**
* [LangSmith](https://smith.langchain.com/settings) API 密钥（免费注册）

## 步骤 1：安装 LangGraph CLI

首先安装 LangGraph 命令行工具：

### 使用 pip 安装
```bash
# 需要 Python >= 3.11
pip install -U "langgraph-cli[inmem]"
```

### 使用 uv 安装
```bash
# 需要 Python >= 3.11
uv add langgraph-cli[inmem]
```

## 步骤 2：创建 LangGraph 应用 🌱

从模板创建一个新的 LangGraph 应用：

```shell
langgraph new path/to/your/app --template new-langgraph-project-python
```

这个模板展示了一个单节点应用，你可以基于此扩展自己的逻辑。

> **提示：更多模板选择**
> 
> 如果使用 `langgraph new` 时不指定模板，会出现交互式菜单让你从可用模板列表中选择。

## 步骤 3：安装依赖

进入新创建的 LangGraph 应用根目录，以 `edit` 模式安装依赖，这样服务器的更改会立即生效：

### 使用 pip
```bash
cd path/to/your/app
pip install -e .
```

### 使用 uv
```bash
cd path/to/your/app
uv add .
```

## 步骤 4：配置环境变量

在你的新 LangGraph 应用根目录中，你会找到 `.env.example` 文件。创建一个 `.env` 文件并复制内容，填入必要的 API 密钥：

```bash
LANGSMITH_API_KEY=lsv2_你的实际API密钥
```

## 步骤 5：启动 LangGraph 服务器 🚀

在本地启动 LangGraph API 服务器：

```shell
langgraph dev
```

成功启动后，你会看到类似以下输出：

```
>    Ready!
>
>    - API: [http://localhost:2024](http://localhost:2024/)
>
>    - Docs: http://localhost:2024/docs
>
>    - LangGraph Studio Web UI: https://smith.langchain.com/studio/?baseUrl=http://127.0.0.1:2024
```

> **重要说明**：`langgraph dev` 命令以内存模式启动 LangGraph 服务器，适用于开发和测试。生产环境请使用持久化存储后端部署。详见[托管概述](/langsmith/hosting)。

## 步骤 6：在 Studio 中测试应用

[Studio](/langsmith/studio) 是一个专门的 UI 界面，可以连接到 LangGraph API 服务器，用于可视化、交互和调试你的应用。

访问 `langgraph dev` 命令输出中提供的 URL 来在 Studio 中测试你的图：

```
>    - LangGraph Studio Web UI: https://smith.langchain.com/studio/?baseUrl=http://127.0.0.1:2024
```

如果 LangGraph 服务器运行在自定义主机/端口上，请更新 baseURL 参数。

<details>
<summary>Safari 浏览器兼容性说明</summary>

由于 Safari 在连接 localhost 服务器时有限制，使用 `--tunnel` 标志创建安全隧道：

```shell
langgraph dev --tunnel
```
</details>

## 步骤 7：测试 API

### 方法一：Python SDK（异步）

1. 安装 LangGraph Python SDK：

```shell
pip install langgraph-sdk
```

2. 发送消息到助手（无线程运行）：

```python
from langgraph_sdk import get_client
import asyncio

# 连接到本地服务器
client = get_client(url="http://localhost:2024")

async def main():
    async for chunk in client.runs.stream(
        None,  # 无线程运行
        "agent",  # 助手名称，在 langgraph.json 中定义
        input={
            "messages": [{
                "role": "human",
                "content": "What is LangGraph?",
            }],
        },
    ):
        print(f"收到新事件类型: {chunk.event}...")
        print(chunk.data)
        print("\n\n")

asyncio.run(main())
```

### 方法二：Python SDK（同步）

1. 安装 LangGraph Python SDK：

```shell
pip install langgraph-sdk
```

2. 发送消息到助手：

```python
from langgraph_sdk import get_sync_client

# 连接到本地服务器
client = get_sync_client(url="http://localhost:2024")

for chunk in client.runs.stream(
    None,  # 无线程运行
    "agent",  # 助手名称
    input={
        "messages": [{
            "role": "human",
            "content": "What is LangGraph?",
        }],
    },
    stream_mode="messages-tuple",
):
    print(f"收到新事件类型: {chunk.event}...")
    print(chunk.data)
    print("\n\n")
```

### 方法三：REST API

使用 curl 命令直接测试 API：

```bash
curl -s --request POST \
    --url "http://localhost:2024/runs/stream" \
    --header 'Content-Type: application/json' \
    --data "{
        \"assistant_id\": \"agent\",
        \"input\": {
            \"messages\": [
                {
                    \"role\": \"human\",
                    \"content\": \"What is LangGraph?\"
                }
            ]
        },
        \"stream_mode\": \"messages-tuple\"
    }"
```

## 故障排除

### 常见问题

1. **端口冲突**
       - 如果 2024 端口被占用，使用 `--port` 参数指定其他端口：

```shell
         langgraph dev --port 3030
```

2. **API 密钥错误**

      - 确保 `.env` 文件中的 `LANGSMITH_API_KEY` 设置正确

3. **依赖安装失败**
     - 确保 Python 版本 >= 3.11
     - 尝试使用虚拟环境

4. **模板创建失败**
     - 检查网络连接
     - 尝试使用不同的模板名称

### 调试技巧

1. **查看详细日志**
```shell
   langgraph dev --verbose
```

2. **检查应用配置**
     - 确认 `langgraph.json` 文件配置正确
     - 验证助手名称与代码中使用的名称一致

3. **测试连接**

```bash
   curl http://localhost:2024/health
```

## 项目结构说明

成功创建项目后，你会看到以下典型结构：

```
your-app/
├── langgraph.json          # 应用配置文件
├── pyproject.toml          # 项目依赖配置
├── .env.example            # 环境变量示例
├── src/
│   └── your_app/
│       ├── __init__.py
│       └── graph.py        # 主要的图定义
└── README.md
```

## 下一步

现在你已经在本地成功运行了 LangGraph 应用，接下来可以：

* [部署快速入门](/langsmith/deployment-quickstart)：使用 LangSmith 部署你的 LangGraph 应用
* [LangSmith 基础](/langsmith/home)：学习 LangSmith 核心概念
* [Python SDK 参考](https://reference.langchain.com/python/platform/python_sdk/)：探索 Python SDK API 参考文档

## 总结

通过本教程，你已经学会了：
- 安装和配置 LangGraph CLI
- 从模板创建新应用
- 启动本地开发服务器
- 使用 Studio 界面测试应用
- 通过多种方式调用 API

现在你可以开始构建和测试自己的 LangGraph 智能体应用了！
```python
import streamlit as st
from langchain_ollama import ChatOllama
from langchain.agents import create_agent
from langchain.tools import tool
from langchain.messages import HumanMessage
from langchain.agents.middleware import HumanInTheLoopMiddleware
from langgraph.types import Command
from langgraph.checkpoint.memory import InMemorySaver

@tool
def add_num(a: int, b: int) -> int:
    '''加法'''
    return a + b

@tool
def minus_num(a: int, b: int) -> int:
    '''减法'''
    return a - b

model = ChatOllama(
    model='qwen3:4b'
)

system_prompt = '''
        你是一个数学计算助手。你可以帮助用户进行加法和减法运算。
        当用户请求计算时，选择合适的工具并执行计算。
        所有计算操作都需要人工审核确认。
'''

agent = create_agent(
    model=model,
    tools=[add_num, minus_num],
    system_prompt=system_prompt,
    middleware=[
        HumanInTheLoopMiddleware(
            interrupt_on={'add_num':True, 'minus_num':True},
            description_prefix="计算待审核",
        ),
    ],
    checkpointer=InMemorySaver(),
)

## 使用streamlit布局页面

st.set_page_config(page_title="SQL Agent Demo", page_icon="🧠")
st.title('LangChain人工审核示例')

if "thread_id" not in st.session_state:
    st.session_state.thread_id = "thread-1"

config = {"configurable": {"thread_id": st.session_state.thread_id}}

question = st.text_input('请输入要计算的数学公式：')

if st.button("开始计算"):
    st.session_state.result = []
    st.session_state.pending = None

    for step in agent.stream({'messages':[HumanMessage(question)]},config=config,stream_mode='values'):

        # 中断等待审核
        if '__interrupt__' in step:
            interrupt = step['__interrupt__'][0]
            st.session_state.pending = interrupt
            st.error('计算已暂停，等待人工审核')
            break

        if 'messages' in step:
            st.session_state.result.append(step['messages'][-1].content)

# ✅ 人工批准继续执行
if "pending" in st.session_state and st.session_state.pending:
    if st.button("✅ 批准执行计算"):
        for step in agent.stream(
            Command(resume={"decisions": [{"type": "approve"}]}),
            config=config,
            stream_mode="values"
        ):
            if "messages" in step:
                st.session_state.result.append(step["messages"][-1].content)

        st.session_state.pending = None
        st.success("✅ 审核完成，执行成功")

# 输出结果
if "result" in st.session_state:
    for r in st.session_state.result:
        st.write(f'运算结果：{r}')
```

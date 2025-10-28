# Streamlit 完整教程：从基础到高级

## 概念介绍

### 什么是 Streamlit？
Streamlit 是一个开源的 Python 框架，专门用于快速构建和分享数据科学和机器学习 Web 应用。它让数据科学家和工程师能够用简单的 Python 脚本创建交互式、美观的 Web 应用，而无需前端开发经验。

### 核心特点
- **简单易用**：只需几行 Python 代码
- **实时更新**：代码保存后应用立即更新
- **丰富组件**：内置多种交互式组件
- **无需前端**：纯 Python，无需 HTML/CSS/JavaScript
- **数据集成**：完美支持 Pandas、Matplotlib、Plotly 等

## 基础代码示例

### 1. 安装和基础应用

```python
# 安装 streamlit
# pip install streamlit

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# 设置页面配置（必须在最前面）
st.set_page_config(
    page_title="我的 Streamlit 应用",
    page_icon="🚀",
    layout="wide",  # "wide" 或 "centered"
    initial_sidebar_state="expanded",  # "auto", "expanded", "collapsed"
    menu_items={
        'Get Help': 'https://docs.streamlit.io/',
        'Report a bug': "https://github.com/streamlit/streamlit/issues",
        'About': "# 这是我的第一个 Streamlit 应用!"
    }
)

# 标题和文本
st.title("🎯 我的 Streamlit 教程应用")
st.header("这是主标题")
st.subheader("这是副标题")

# Markdown 支持
st.markdown("""
这是 **粗体** 和 *斜体* 文本
- 列表项 1
- 列表项 2
- 列表项 3

[这是一个链接](https://streamlit.io)
""")

# 代码块
st.code("""
import streamlit as st
st.write('Hello World!')
""", language='python')
```

## 页面布局详解

### 2. 侧边栏布局

```python
# 侧边栏 - 所有以 st.sidebar 开头的组件都会显示在侧边栏
st.sidebar.title("🎛️ 控制面板")
st.sidebar.markdown("这里是应用的配置选项")

# 侧边栏组件
sidebar_option = st.sidebar.radio(
    "选择功能",
    ["数据查看", "数据分析", "数据可视化"]
)

# 侧边栏文件上传
uploaded_file = st.sidebar.file_uploader(
    "上传数据文件",
    type=['csv', 'xlsx', 'txt']
)

# 侧边栏下载按钮
if st.sidebar.button("下载示例数据"):
    # 创建示例数据供下载
    sample_data = pd.DataFrame({
        'x': range(100),
        'y': np.random.randn(100)
    })
    csv = sample_data.to_csv(index=False)
    st.sidebar.download_button(
        label="下载 CSV",
        data=csv,
        file_name="sample_data.csv",
        mime="text/csv"
    )
```

### 3. 列布局

```python
# 列布局示例
st.header("📐 列布局示例")

# 创建等宽列
col1, col2, col3 = st.columns(3)

with col1:
    st.subheader("第一列")
    st.metric("温度", "25°C", "1.2°C")
    st.button("列1按钮", key="btn1")

with col2:
    st.subheader("第二列")
    st.metric("湿度", "60%", "-5%")
    st.button("列2按钮", key="btn2")

with col3:
    st.subheader("第三列")
    st.metric("压力", "1013 hPa", "2 hPa")
    st.button("列3按钮", key="btn3")

# 创建不等宽列
st.header("不等宽列布局")
wide_col, narrow_col = st.columns([3, 1])

with wide_col:
    st.subheader("主要内容区")
    # 生成示例数据
    data = pd.DataFrame({
        '姓名': ['张三', '李四', '王五', '赵六'],
        '年龄': [25, 30, 35, 28],
        '城市': ['北京', '上海', '广州', '深圳'],
        '分数': [85, 92, 78, 96]
    })
    st.dataframe(data, use_container_width=True)

with narrow_col:
    st.subheader("控制区")
    show_age = st.checkbox("显示年龄")
    show_city = st.checkbox("显示城市")
    theme = st.selectbox("主题", ["浅色", "深色"])
```

### 4. 容器和扩展器

```python
# 容器示例
st.header("📦 容器和扩展器")

# 使用容器组织相关内容
with st.container():
    st.subheader("相关功能组")
    col1, col2 = st.columns(2)
    
    with col1:
        name = st.text_input("姓名")
        email = st.text_input("邮箱")
    
    with col2:
        phone = st.text_input("电话")
        department = st.selectbox("部门", ["技术", "市场", "销售", "人事"])
    
    st.info("这是一个信息容器，相关的内容可以放在一起")

# 扩展器（可折叠内容）
with st.expander("📊 点击查看详细统计信息", expanded=False):
    st.write("### 数据统计")
    if 'data' in locals():
        st.write(f"数据行数: {len(data)}")
        st.write(f"数据列数: {len(data.columns)}")
        st.write("数据类型:")
        st.write(data.dtypes)
    
    # 在扩展器内部还可以有图表
    fig, ax = plt.subplots()
    if 'data' in locals() and '年龄' in data.columns:
        ax.hist(data['年龄'], bins=10, alpha=0.7, color='skyblue')
        ax.set_title('年龄分布')
        st.pyplot(fig)

# 多个扩展器
expander1 = st.expander("📝 使用说明")
with expander1:
    st.write("""
    这是一个 Streamlit 应用的使用说明：
    1. 首先在侧边栏上传数据
    2. 然后选择分析功能
    3. 查看结果和可视化
    """)

expander2 = st.expander("⚙️ 高级设置")
with expander2:
    precision = st.slider("计算精度", 1, 10, 2)
    auto_refresh = st.checkbox("自动刷新")
    refresh_interval = st.number_input("刷新间隔(秒)", 1, 60, 5)
```

### 5. 标签页布局

```python
# 标签页布局 (Streamlit 1.23.0+)
st.header("📑 标签页布局")

tab1, tab2, tab3, tab4 = st.tabs(["📊 数据", "📈 图表", "🔧 设置", "ℹ️ 关于"])

with tab1:
    st.subheader("数据管理")
    if 'data' in locals():
        st.dataframe(data, use_container_width=True)
        
        # 数据编辑功能
        st.subheader("数据编辑")
        edited_data = st.data_editor(data, num_rows="dynamic")
        if st.button("保存更改"):
            st.success("数据已更新！")

with tab2:
    st.subheader("数据可视化")
    
    if 'data' in locals():
        chart_type = st.selectbox("选择图表类型", ["折线图", "柱状图", "散点图"])
        
        if chart_type == "折线图":
            st.line_chart(data.set_index('姓名')['分数'])
        elif chart_type == "柱状图":
            st.bar_chart(data.set_index('姓名')['分数'])
        elif chart_type == "散点图":
            fig = px.scatter(data, x='年龄', y='分数', text='姓名')
            st.plotly_chart(fig, use_container_width=True)

with tab3:
    st.subheader("应用设置")
    
    # 主题设置
    theme = st.selectbox("选择主题", ["浅色", "深色", "系统默认"])
    
    # 数据设置
    st.subheader("数据设置")
    decimal_places = st.slider("小数位数", 0, 6, 2)
    date_format = st.selectbox("日期格式", ["YYYY-MM-DD", "DD/MM/YYYY", "MM/DD/YYYY"])
    
    # 保存设置
    if st.button("应用设置"):
        st.success("设置已保存！")

with tab4:
    st.subheader("关于应用")
    st.write("""
    ## Streamlit 数据仪表板
    
    **版本**: 1.0.0
    **作者**: Your Name
    **描述**: 这是一个用于数据分析和可视化的 Streamlit 应用
    
    ### 功能特性
    - 数据上传和查看
    - 交互式数据分析
    - 多种可视化图表
    - 可自定义的设置
    """)
```

## Session State 详解

### 6. Session State 基础

```python
# Session State 管理应用状态
st.header("💾 Session State 管理")

# 初始化 session state
if 'counter' not in st.session_state:
    st.session_state.counter = 0
if 'user_data' not in st.session_state:
    st.session_state.user_data = {}
if 'app_state' not in st.session_state:
    st.session_state.app_state = {
        'theme': 'light',
        'language': 'zh',
        'data_loaded': False
    }

# 计数器示例
col1, col2, col3 = st.columns(3)

with col1:
    if st.button("增加计数"):
        st.session_state.counter += 1

with col2:
    if st.button("减少计数"):
        st.session_state.counter -= 1

with col3:
    if st.button("重置计数"):
        st.session_state.counter = 0

st.metric("当前计数", st.session_state.counter)

# 显示所有 session state
with st.expander("查看所有 Session State"):
    st.write("计数器:", st.session_state.counter)
    st.write("用户数据:", st.session_state.user_data)
    st.write("应用状态:", st.session_state.app_state)
```

### 7. Session State 高级用法

```python
# 复杂状态管理
st.header("🔄 复杂状态管理")

# 用户会话管理
if 'user_session' not in st.session_state:
    st.session_state.user_session = {
        'logged_in': False,
        'username': '',
        'preferences': {},
        'history': []
    }

# 登录系统
st.subheader("用户会话管理")

if not st.session_state.user_session['logged_in']:
    with st.form("login_form"):
        username = st.text_input("用户名")
        password = st.text_input("密码", type="password")
        submit = st.form_submit_button("登录")
        
        if submit:
            if username and password:  # 简单验证
                st.session_state.user_session.update({
                    'logged_in': True,
                    'username': username,
                    'login_time': pd.Timestamp.now()
                })
                st.success(f"欢迎 {username}!")
                st.rerun()
else:
    st.success(f"已登录为: {st.session_state.user_session['username']}")
    
    # 用户偏好设置
    with st.form("preferences_form"):
        st.subheader("用户偏好设置")
        theme = st.selectbox("主题", ["浅色", "深色", "自动"])
        language = st.selectbox("语言", ["中文", "英文"])
        notifications = st.checkbox("启用通知")
        
        if st.form_submit_button("保存偏好"):
            st.session_state.user_session['preferences'] = {
                'theme': theme,
                'language': language,
                'notifications': notifications
            }
            st.success("偏好设置已保存!")
    
    # 退出登录
    if st.button("退出登录"):
        # 保存历史记录
        if 'history' not in st.session_state.user_session:
            st.session_state.user_session['history'] = []
        
        st.session_state.user_session['history'].append({
            'action': 'logout',
            'time': pd.Timestamp.now()
        })
        
        st.session_state.user_session['logged_in'] = False
        st.info("已退出登录")
        st.rerun()

# 购物车示例
st.subheader("购物车示例")

if 'shopping_cart' not in st.session_state:
    st.session_state.shopping_cart = []

products = [
    {"id": 1, "name": "笔记本电脑", "price": 5999},
    {"id": 2, "name": "无线鼠标", "price": 199},
    {"id": 3, "name": "机械键盘", "price": 599},
    {"id": 4, "name": "显示器", "price": 1299},
]

# 显示商品列表
st.write("### 商品列表")
for product in products:
    col1, col2, col3 = st.columns([3, 1, 1])
    with col1:
        st.write(f"**{product['name']}** - ¥{product['price']}")
    with col2:
        if st.button(f"加入购物车", key=f"add_{product['id']}"):
            st.session_state.shopping_cart.append(product)
            st.success(f"已添加 {product['name']} 到购物车")
    with col3:
        if st.button(f"移除", key=f"remove_{product['id']}"):
            # 移除最后一个匹配的商品
            for i in range(len(st.session_state.shopping_cart)-1, -1, -1):
                if st.session_state.shopping_cart[i]['id'] == product['id']:
                    st.session_state.shopping_cart.pop(i)
                    st.info(f"已移除 {product['name']}")
                    break

# 显示购物车
st.write("### 购物车")
if st.session_state.shopping_cart:
    cart_df = pd.DataFrame(st.session_state.shopping_cart)
    st.dataframe(cart_df)
    
    total_price = sum(item['price'] for item in st.session_state.shopping_cart)
    st.metric("总价", f"¥{total_price}")
    
    if st.button("清空购物车"):
        st.session_state.shopping_cart = []
        st.rerun()
else:
    st.info("购物车为空")
```

### 8. 表单和状态结合

```python
# 表单状态管理
st.header("📝 表单状态管理")

# 多步骤表单
if 'form_step' not in st.session_state:
    st.session_state.form_step = 1
if 'form_data' not in st.session_state:
    st.session_state.form_data = {}

# 步骤指示器
steps = ["基本信息", "详细资料", "确认信息"]
current_step = st.session_state.form_step

# 显示进度
progress = current_step / len(steps)
st.progress(progress)
st.write(f"步骤 {current_step}/{len(steps)}: {steps[current_step-1]}")

# 多步骤表单内容
if current_step == 1:
    with st.form("step1_form"):
        st.subheader("基本信息")
        name = st.text_input("姓名", value=st.session_state.form_data.get('name', ''))
        email = st.text_input("邮箱", value=st.session_state.form_data.get('email', ''))
        
        if st.form_submit_button("下一步"):
            st.session_state.form_data.update({
                'name': name,
                'email': email
            })
            st.session_state.form_step = 2
            st.rerun()

elif current_step == 2:
    with st.form("step2_form"):
        st.subheader("详细资料")
        age = st.number_input("年龄", min_value=0, max_value=150, 
                             value=st.session_state.form_data.get('age', 25))
        city = st.selectbox("城市", ["北京", "上海", "广州", "深圳", "其他"],
                           index=st.session_state.form_data.get('city_index', 0))
        
        col1, col2 = st.columns(2)
        with col1:
            if st.form_submit_button("上一步"):
                st.session_state.form_step = 1
                st.rerun()
        with col2:
            if st.form_submit_button("下一步"):
                st.session_state.form_data.update({
                    'age': age,
                    'city': city
                })
                st.session_state.form_step = 3
                st.rerun()

elif current_step == 3:
    st.subheader("确认信息")
    st.write("请确认您输入的信息:")
    st.json(st.session_state.form_data)
    
    col1, col2 = st.columns(2)
    with col1:
        if st.button("返回修改"):
            st.session_state.form_step = 2
            st.rerun()
    with col2:
        if st.button("提交"):
            st.success("表单提交成功!")
            # 重置表单
            st.session_state.form_step = 1
            st.session_state.form_data = {}
            st.rerun()
```

### 9. 缓存和性能优化

```python
# 缓存机制
st.header("⚡ 缓存和性能优化")

# 数据缓存
@st.cache_data(ttl=3600)  # 缓存1小时
def load_large_dataset(file_path):
    """模拟加载大型数据集"""
    st.info("正在加载数据...")
    time.sleep(2)  # 模拟耗时操作
    return pd.DataFrame({
        'id': range(1000),
        'value': np.random.randn(1000),
        'category': np.random.choice(['A', 'B', 'C'], 1000)
    })

# 资源缓存
@st.cache_resource
def get_expensive_model():
    """模拟加载昂贵模型"""
    st.info("正在加载模型...")
    time.sleep(3)
    return "模拟的机器学习模型"

# 缓存使用示例
if st.button("加载数据(使用缓存)"):
    data = load_large_dataset("dummy_path.csv")
    st.write(f"数据加载完成，共 {len(data)} 行")
    
if st.button("加载模型(使用缓存)"):
    model = get_expensive_model()
    st.success(f"模型加载完成: {model}")

# 清空缓存
if st.button("清空所有缓存"):
    st.cache_data.clear()
    st.cache_resource.clear()
    st.success("缓存已清空!")
```

## 完整应用示例

### 10. 综合应用：数据分析仪表板

```python
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta

# 应用配置
st.set_page_config(
    page_title="高级数据分析仪表板",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# 初始化 session state
if 'dashboard' not in st.session_state:
    st.session_state.dashboard = {
        'data_loaded': False,
        'current_view': 'overview',
        'filters': {},
        'charts_config': {}
    }

def main():
    # 标题和描述
    st.title("📊 高级数据分析仪表板")
    st.markdown("---")
    
    # 侧边栏
    with st.sidebar:
        st.header("控制面板")
        
        # 数据上传
        uploaded_file = st.file_uploader(
            "上传数据文件 (CSV)",
            type=['csv'],
            key="file_uploader"
        )
        
        if uploaded_file is not None:
            try:
                data = pd.read_csv(uploaded_file)
                st.session_state.dashboard['data'] = data
                st.session_state.dashboard['data_loaded'] = True
                st.success(f"数据加载成功! 共 {len(data)} 行, {len(data.columns)} 列")
            except Exception as e:
                st.error(f"数据加载失败: {e}")
        
        # 视图选择
        if st.session_state.dashboard['data_loaded']:
            view_options = {
                "overview": "数据概览",
                "explore": "数据探索", 
                "visualize": "数据可视化",
                "analyze": "高级分析"
            }
            
            selected_view = st.radio(
                "选择视图",
                options=list(view_options.keys()),
                format_func=lambda x: view_options[x]
            )
            st.session_state.dashboard['current_view'] = selected_view
    
    # 主内容区
    if st.session_state.dashboard['data_loaded']:
        data = st.session_state.dashboard['data']
        current_view = st.session_state.dashboard['current_view']
        
        if current_view == 'overview':
            show_data_overview(data)
        elif current_view == 'explore':
            show_data_exploration(data)
        elif current_view == 'visualize':
            show_data_visualization(data)
        elif current_view == 'analyze':
            show_advanced_analysis(data)
    else:
        show_welcome_screen()

def show_welcome_screen():
    """欢迎界面"""
    st.header("欢迎使用数据分析仪表板")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.info("""
        ### 🚀 功能特性
        - 数据上传和预览
        - 交互式数据探索
        - 多种可视化图表
        - 高级统计分析
        - 实时数据过滤
        """)
    
    with col2:
        st.warning("""
        ### 📝 使用说明
        1. 在左侧边栏上传 CSV 文件
        2. 选择不同的视图模式
        3. 使用交互式控件探索数据
        4. 保存和导出分析结果
        """)
    
    # 示例数据
    if st.button("加载示例数据"):
        # 生成示例数据
        np.random.seed(42)
        sample_data = pd.DataFrame({
            '日期': pd.date_range('2023-01-01', periods=100),
            '销售额': np.random.normal(1000, 200, 100).cumsum(),
            '客户数': np.random.poisson(50, 100),
            '产品类别': np.random.choice(['电子产品', '服装', '食品', '家居'], 100),
            '地区': np.random.choice(['华北', '华东', '华南', '西部'], 100)
        })
        st.session_state.dashboard['data'] = sample_data
        st.session_state.dashboard['data_loaded'] = True
        st.rerun()

def show_data_overview(data):
    """数据概览视图"""
    st.header("📈 数据概览")
    
    # 关键指标
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("总行数", len(data))
    with col2:
        st.metric("总列数", len(data.columns))
    with col3:
        st.metric("缺失值", data.isnull().sum().sum())
    with col4:
        st.metric("内存使用", f"{data.memory_usage(deep=True).sum() / 1024**2:.2f} MB")
    
    # 数据预览
    st.subheader("数据预览")
    tab1, tab2, tab3 = st.tabs(["前几行", "后几行", "随机样本"])
    
    with tab1:
        st.dataframe(data.head(10), use_container_width=True)
    with tab2:
        st.dataframe(data.tail(10), use_container_width=True)
    with tab3:
        st.dataframe(data.sample(10), use_container_width=True)
    
    # 数据类型信息
    st.subheader("数据类型信息")
    dtype_info = pd.DataFrame({
        '列名': data.columns,
        '数据类型': data.dtypes,
        '非空值数量': data.count(),
        '空值数量': data.isnull().sum()
    })
    st.dataframe(dtype_info, use_container_width=True)

def show_data_exploration(data):
    """数据探索视图"""
    st.header("🔍 数据探索")
    
    # 列选择器
    col1, col2 = st.columns(2)
    
    with col1:
        selected_columns = st.multiselect(
            "选择要分析的列",
            options=data.columns.tolist(),
            default=data.columns.tolist()[:3] if len(data.columns) >= 3 else data.columns.tolist()
        )
    
    with col2:
        # 数据过滤
        st.subheader("数据过滤")
        filter_col = st.selectbox("过滤列", [None] + data.select_dtypes(include=[np.number]).columns.tolist())
        if filter_col:
            min_val, max_val = float(data[filter_col].min()), float(data[filter_col].max())
            filter_range = st.slider(
                f"选择 {filter_col} 范围",
                min_val, max_val, (min_val, max_val)
            )
            filtered_data = data[(data[filter_col] >= filter_range[0]) & (data[filter_col] <= filter_range[1])]
        else:
            filtered_data = data
    
    if selected_columns:
        # 显示选中的列数据
        st.subheader("选中的数据")
        st.dataframe(filtered_data[selected_columns], use_container_width=True)
        
        # 描述性统计
        st.subheader("描述性统计")
        st.dataframe(filtered_data[selected_columns].describe(), use_container_width=True)
        
        # 相关性分析（如果有多列数值数据）
        numeric_cols = filtered_data[selected_columns].select_dtypes(include=[np.number]).columns
        if len(numeric_cols) > 1:
            st.subheader("相关性矩阵")
            corr_matrix = filtered_data[numeric_cols].corr()
            fig = px.imshow(corr_matrix, text_auto=True, aspect="auto")
            st.plotly_chart(fig, use_container_width=True)

def show_data_visualization(data):
    """数据可视化视图"""
    st.header("📊 数据可视化")
    
    # 图表类型选择
    chart_type = st.selectbox(
        "选择图表类型",
        ["散点图", "折线图", "柱状图", "直方图", "箱线图", "热力图"]
    )
    
    col1, col2 = st.columns(2)
    
    with col1:
        # X 轴选择
        x_axis = st.selectbox("X 轴", data.columns.tolist())
    
    with col2:
        # Y 轴选择（如果是数值型图表）
        if chart_type in ["散点图", "折线图", "柱状图"]:
            y_axis = st.selectbox("Y 轴", data.select_dtypes(include=[np.number]).columns.tolist())
        else:
            y_axis = None
    
    # 颜色分组
    color_by = st.selectbox("按颜色分组", [None] + data.select_dtypes(include=['object']).columns.tolist())
    
    # 生成图表
    if chart_type == "散点图" and y_axis:
        fig = px.scatter(data, x=x_axis, y=y_axis, color=color_by, hover_data=data.columns)
        st.plotly_chart(fig, use_container_width=True)
    
    elif chart_type == "折线图" and y_axis:
        # 如果 x 轴是日期，自动排序
        if pd.api.types.is_datetime64_any_dtype(data[x_axis]):
            sorted_data = data.sort_values(x_axis)
        else:
            sorted_data = data
        fig = px.line(sorted_data, x=x_axis, y=y_axis, color=color_by)
        st.plotly_chart(fig, use_container_width=True)
    
    elif chart_type == "柱状图" and y_axis:
        fig = px.bar(data, x=x_axis, y=y_axis, color=color_by)
        st.plotly_chart(fig, use_container_width=True)
    
    elif chart_type == "直方图":
        fig = px.histogram(data, x=x_axis, color=color_by, nbins=20)
        st.plotly_chart(fig, use_container_width=True)
    
    elif chart_type == "箱线图" and y_axis:
        fig = px.box(data, x=x_axis, y=y_axis, color=color_by)
        st.plotly_chart(fig, use_container_width=True)
    
    elif chart_type == "热力图":
        numeric_data = data.select_dtypes(include=[np.number])
        if len(numeric_data.columns) > 1:
            fig = px.imshow(numeric_data.corr(), text_auto=True, aspect="auto")
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.warning("需要至少两个数值列来生成热力图")

def show_advanced_analysis(data):
    """高级分析视图"""
    st.header("🧠 高级分析")
    
    # 时间序列分析（如果数据包含日期列）
    date_columns = data.select_dtypes(include=['datetime64']).columns
    if len(date_columns) > 0:
        st.subheader("时间序列分析")
        
        date_col = st.selectbox("选择日期列", date_columns)
        value_col = st.selectbox("选择数值列", data.select_dtypes(include=[np.number]).columns)
        
        if date_col and value_col:
            # 确保数据按日期排序
            time_series_data = data.sort_values(date_col).set_index(date_col)
            
            # 移动平均
            window = st.slider("移动平均窗口", 1, 30, 7)
            time_series_data[f'{value_col}_MA'] = time_series_data[value_col].rolling(window=window).mean()
            
            # 绘制时间序列和移动平均
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=time_series_data.index,
                y=time_series_data[value_col],
                name='原始数据',
                line=dict(color='blue')
            ))
            fig.add_trace(go.Scatter(
                x=time_series_data.index,
                y=time_series_data[f'{value_col}_MA'],
                name=f'{window}期移动平均',
                line=dict(color='red', dash='dash')
            ))
            fig.update_layout(title=f"{value_col} 时间序列分析")
            st.plotly_chart(fig, use_container_width=True)
    
    # 聚类分析
    st.subheader("聚类分析")
    if st.checkbox("启用聚类分析"):
        numeric_cols = data.select_dtypes(include=[np.number]).columns.tolist()
        if len(numeric_cols) >= 2:
            from sklearn.cluster import KMeans
            from sklearn.preprocessing import StandardScaler
            
            selected_cluster_cols = st.multiselect(
                "选择聚类特征",
                numeric_cols,
                default=numeric_cols[:2]
            )
            
            n_clusters = st.slider("聚类数量", 2, 10, 3)
            
            if len(selected_cluster_cols) >= 2:
                # 准备数据
                cluster_data = data[selected_cluster_cols].dropna()
                scaler = StandardScaler()
                scaled_data = scaler.fit_transform(cluster_data)
                
                # 执行聚类
                kmeans = KMeans(n_clusters=n_clusters, random_state=42)
                clusters = kmeans.fit_predict(scaled_data)
                
                # 可视化结果
                cluster_data['Cluster'] = clusters
                fig = px.scatter(
                    cluster_data, 
                    x=selected_cluster_cols[0], 
                    y=selected_cluster_cols[1],
                    color='Cluster',
                    title="K-means 聚类结果"
                )
                st.plotly_chart(fig, use_container_width=True)
        else:
            st.warning("需要至少两个数值列进行聚类分析")

if __name__ == "__main__":
    main()
```

## 运行应用

要运行 Streamlit 应用，在终端中执行：

```bash
streamlit run your_app.py
```

## 部署应用

### 本地部署
```bash
streamlit run app.py
```

### 部署到 Streamlit Cloud
1. 将代码推送到 GitHub
2. 访问 [share.streamlit.io](https://share.streamlit.io)
3. 连接 GitHub 仓库
4. 选择分支和文件路径

### 部署到其他平台
- **Heroku**: 使用 Procfile 和 requirements.txt
- **AWS/Azure**: 使用 Docker 容器
- **Hugging Face**: 支持 Streamlit 空间

## 总结

这个完整教程涵盖了 Streamlit 的所有重要概念：

### 核心概念掌握：
1. **页面布局**：侧边栏、列、容器、标签页、扩展器
2. **Session State**：状态管理、用户会话、表单状态、购物车模式
3. **数据展示**：表格、图表、指标、交互组件
4. **性能优化**：缓存机制、懒加载
5. **部署发布**：多平台部署选项

### 最佳实践：
- 合理使用 Session State 管理应用状态
- 使用缓存优化性能
- 采用模块化设计组织代码
- 提供清晰的用户引导和反馈
- 考虑响应式布局设计

这个教程提供了从基础到高级的完整学习路径，你可以基于这些模式构建复杂的数据应用！
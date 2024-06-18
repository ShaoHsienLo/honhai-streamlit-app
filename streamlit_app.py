import pandas as pd
import numpy as np
from scipy.spatial import distance
import plotly.graph_objects as go
import streamlit as st
import os


def find_closest(values_list, input_values):
    distances = []
    
    for idx, pair in enumerate(values_list):
        dist = distance.euclidean(pair, input_values)
        distances.append((dist, idx))
    
    # 按距离排序
    distances.sort(key=lambda x: x[0])
    
    # 获取前两个最小距离的索引
    closest_indices = [distances[0][1], distances[1][1]]
    closest_pairs = [values_list[closest_indices[0]], values_list[closest_indices[1]]]
    
    return closest_pairs, closest_indices


st.set_page_config(layout="wide")
st.title("電池添加劑充放電趨勢分析工具")

current_dir = os.path.dirname(os.path.abspath(__file__))
excel_file_path = os.path.join(current_dir, 'cycle test_T8(20240527提供).xlsx')
data = pd.read_excel(excel_file_path, sheet_name='LT data', usecols='A:P')
# data = pd.read_excel('cycle test_T8(20240527提供).xlsx', sheet_name='LT data', usecols='A:P')

y1 = data.iloc[:, 3].dropna().reset_index(drop=True)
y2 = data.iloc[:, 6].dropna().reset_index(drop=True)
y3 = data.iloc[:, 9].dropna().reset_index(drop=True)
y4 = data.iloc[:, 12].dropna().reset_index(drop=True)
ys = [y1, y2, y3, y4]
names = ['I', 'II', 'III', 'IV']

slopes = []
intercepts = []
for y in ys:
    x = np.arange(len(y)) + 1
    slope, intercept = np.polyfit(x, y, 1)
    slopes.append(slope)
    intercepts.append(intercept)

values_list = [[2, 1.5], [2.5, 1.5], [2, 1], [2.5, 1]]
# 使用 Streamlit 滑动条调整 input_values
col1, col2, col3, col4 = st.columns(4)
with col1:
    input_value1 = st.slider('添加劑 E', 2.0, 2.5, 2.0)
with col2:
    input_value3 = st.slider('添加劑 F', 0.0, 1.0, 0.5)
with col3:
    input_value2 = st.slider('添加劑 G', 1.0, 1.5, 1.0)
with col4:
    input_value4 = st.slider('添加劑 H', 0.0, 1.5, 1.0)
input_values = [input_value1, input_value2]

start_button = st.button("開始分析")

if start_button:
    st.write("")
    closest_pairs, closest_indices = find_closest(values_list, input_values)
    # st.write(f"The two closest pairs to **{input_value1}, {input_value1}** are: **{closest_pairs[0]}, {closest_pairs[1]}**")
    # st.markdown(f"""The two closest pairs to <span style="color:red; font-weight:bold">{input_value1}, {input_value2}</span> are: 
    # <span style="color:blue; font-weight:bold">{closest_pairs[0]}, {closest_pairs[1]}</span>""", unsafe_allow_html=True)
    # st.write(f"Their indices in the list are: **{closest_indices}**")

    # 颜色列表
    colors = ['blue', 'red', 'green', 'purple']

    # 创建图形对象
    fig = go.Figure()

    # 添加每组数据
    for i in range(len(ys)):
        # 添加數組點
        fig.add_trace(go.Scatter(x=list(range(len(ys[i]))), y=ys[i], mode='lines', name=names[i], line=dict(color=colors[i])))

        # 計算斜率和截距
        # x = np.arange(len(ys[i])) + 1
        # slope, intercept = np.polyfit(x, ys[i], 1)
        
        # 添加斜率和截距的直線
        # trend_line_y = slope * x + intercept
        # fig.add_trace(go.Scatter(x=x, y=trend_line_y, mode='lines', name=f'{names[i]} trend', line=dict(color='black')))

    # 提取指定索引的斜率和截距
    index1 = closest_indices[0]  # 第一个索引
    index2 = closest_indices[1]  # 第二个索引

    slope1, intercept1 = slopes[index1], intercepts[index1]
    slope2, intercept2 = slopes[index2], intercepts[index2]

    # 计算平均斜率和截距
    avg_slope = (slope1 + slope2) / 2
    avg_intercept = (intercept1 + intercept2) / 2

    # 绘制介于两条直线中间的新直线
    x = np.arange(399, len(ys[0])) + 1
    y = avg_slope * x + avg_intercept
    fig.add_trace(go.Scatter(x=x, y=y, mode='lines', name='預測趨勢線', line=dict(color='black', dash='dash')))

    # 更新布局
    fig.update_layout(title='Four Sets of Floating Point Numbers with Trends',
                    xaxis_title='Cycle No',
                    yaxis_title='Capacity ratio')

    # 在 Streamlit 中显示图形
    st.plotly_chart(fig, use_container_width=True)

    # 显示图形
    # fig.show()
    # fig.write_html('capacity.html')

    # 计算指定循环次数的Capacity ratio
    idx = [400, 600, 800, 1000]
    capacity_ratios = [avg_slope * i + avg_intercept for i in idx]
    capacity_ratios_percent = [ratio * 100 for ratio in capacity_ratios]

    # 创建DataFrame
    df_results = pd.DataFrame({
        'Cycle No.': idx,
        'Capacity ratio (%)': capacity_ratios_percent
    })
    df_results.index = df_results.index + 1

    # 在Streamlit中显示DataFrame
    st.write("循環性能Capacity Ratio落點數值預測")
    st.dataframe(df_results)

    st.write("分析結束！")
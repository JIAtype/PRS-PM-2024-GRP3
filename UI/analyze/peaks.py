import streamlit as st
import os
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import matplotlib.pyplot as plt
from scipy.signal import find_peaks as findPeaks
from sklearn.preprocessing import MinMaxScaler

st.title("Dynamic Time Series Analysis: ")
st.header("📈 Peak Detection")
st.markdown("""
    This interactive tool allows you to upload your dataset and perform an analysis on two key time series: Amount Paid and Increased Number of Member.
""")

UPLOAD_FOLDER = "UI/data"
image_path = os.path.join("UI/images", "A.png")
# File upload section
if os.path.isdir(UPLOAD_FOLDER) and os.listdir(UPLOAD_FOLDER):
    uploaded_files = [f for f in os.listdir(UPLOAD_FOLDER) if f != ".DS_Store"]
    if uploaded_files:
        uploaded_files = ["Select File"] + uploaded_files
        selected_file = st.selectbox("Please select your file to start the analysis:", uploaded_files)
        df = None
        if selected_file != "Select File":
            file_path = os.path.join(UPLOAD_FOLDER, selected_file)
            if selected_file.endswith(".csv"):
                df = pd.read_csv(file_path)
            elif selected_file.endswith(".xlsx"):
                df = pd.read_excel(file_path)

            # Required columns for analysis
            required_columns = ["month", "AmtPaid", "IncreasedNM"]

            if df is not None:
                st.write(f"Preview 3 rows of data from **{selected_file}**:")
                st.dataframe(df.head(3).style.set_table_attributes('style="width: 100%; border-collapse: collapse;"'))
                if all(col in df.columns for col in required_columns) and len(df.columns) == len(required_columns):
                    st.header("🌟 View Analysis Results")
                    figures = []
                    scaler = MinMaxScaler()
                    df[['AmtPaid', 'IncreasedNM']] = scaler.fit_transform(df[['AmtPaid', 'IncreasedNM']])
                    Y = df['AmtPaid'].values
                    Z = df['IncreasedNM'].values
                    month_labels = [f'{(i % 12) + 1}/{2015 + (i // 12)}' for i in range(len(Y))]

                    #第一个
                    allPks, _ = findPeaks(Y)
                    fig = go.Figure()
                    fig.add_trace(go.Scatter(
                        x=np.arange(len(Y)),  
                        y=Y,
                        mode='lines',
                        name='Total Revenue',
                        line=dict(color='blue')
                    ))
                    fig.add_trace(go.Scatter(
                        x=allPks,
                        y=Y[allPks],
                        mode='markers+text',
                        name='Peaks',
                        marker=dict(color='red', size=8),
                        text=[month_labels[i] for i in allPks],  # 显示峰值索引
                        textposition='top center'
                    ))
                    fig.update_layout(
                        title='Total Revenue with Peaks',
                        xaxis_title='Month',
                        yaxis_title='Normalized Total Revenue'
                    )
                    tick_indices = np.arange(0, len(month_labels), 3)  # 每3个月取一个索引
                    fig.update_xaxes(tickvals=tick_indices, ticktext=[month_labels[i] for i in tick_indices])
                    figures.append(fig)

                    #第二个
                    allPks, _ = findPeaks(Z)
                    fig = go.Figure()
                    fig.add_trace(go.Scatter(
                        x=np.arange(len(Z)),  
                        y=Z,
                        mode='lines',
                        name='Increased New Membership',
                        line=dict(color='blue')
                    ))
                    fig.add_trace(go.Scatter(
                        x=allPks,
                        y=Z[allPks],
                        mode='markers+text',
                        name='Peaks',
                        marker=dict(color='red', size=8),
                        text=[month_labels[i] for i in allPks],  # 显示峰值索引
                        textposition='top center'
                    ))
                    fig.update_layout(
                        title='Increased New Membership with Peaks',
                        xaxis_title='Month',
                        yaxis_title='Normalized Increased New Membership'
                    )
                    tick_indices = np.arange(0, len(month_labels), 3)  # 每3个月取一个索引
                    fig.update_xaxes(tickvals=tick_indices, ticktext=[month_labels[i] for i in tick_indices])
                    figures.append(fig)

                    for i in range(0, len(figures), 2):  # 每三张图生成一行
                        cols = st.columns(2)  # 创建三列
                        for j in range(2):
                            index = i + j  # 计算当前图表的索引
                            if index < len(figures):  # 如果图表存在
                                with cols[j]:  # 在第 j 列中
                                    fig = figures[index]
                                    # fig.update_layout(template='plotly_white')
                                    st.plotly_chart(fig, use_container_width=True)
                                    # st.pyplot(fig, use_container_width=True)
                                    # st.plotly_chart(fig, use_container_width=True)  # 使用 Streamlit 的方法渲染图表
                            else:  # 如果没有图表，显示图片
                                with cols[j]:
                                    st.image(image_path, use_column_width=True)  # 显示备用图片
                    st.header(f"📑 View Raw Data From ***{selected_file}***:")
                    st.dataframe(df.style.set_table_attributes('style="width: 100%; border-collapse: collapse;"'))  # 显示原始数据
                else:
                    st.warning(f"The selected file should only have the following columns: {', '.join(required_columns)}. Please upload a file with the correct format.", icon="⚠️")
            else:
                st.warning("Please select a valid file before Analyze.", icon="⚠️")
    else:
        st.warning("Please upload one or more files in the upload files section to get started!", icon="⚠️")
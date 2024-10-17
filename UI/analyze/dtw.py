import streamlit as st
import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import plotly.graph_objects as go
from sklearn.preprocessing import MinMaxScaler

st.title("⏳ Dynamic Time Warping (DTW) Analysis: ")
st.header("Similarity Between Amount Paid and Increased Number of Member")
st.markdown("""
    This page demonstrates the use of Dynamic Time Warping (DTW) to analyze the similarity between two time series: Amount Paid and Increased Number. By aligning the sequences, DTW reveals how they are related over time, even when they are not perfectly synchronized.
""")

UPLOAD_FOLDER = "UI/data"
image_path = os.path.join("UI/images", "A.png")
# File upload section
if os.path.isdir(UPLOAD_FOLDER) and os.listdir(UPLOAD_FOLDER):
    uploaded_files = [f for f in os.listdir(UPLOAD_FOLDER) if f != ".DS_Store"]
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
                df['month'] = pd.to_datetime(df['month'])
                # 数据归一化
                scaler = MinMaxScaler()
                df[['AmtPaid', 'IncreasedNM']] = scaler.fit_transform(df[['AmtPaid', 'IncreasedNM']])
                # 创建 Plotly 图表
                fig = go.Figure()
                fig.add_trace(go.Scatter(x=df['month'], y=df['AmtPaid'], 
                mode='lines+markers', name='Amount Paid (Normalized)',
                line=dict(color='#1f77b4', width=2), marker=dict(color='#5fad56', size=6)))
                fig.add_trace(go.Scatter(x=df['month'], y=df['IncreasedNM'], 
                mode='lines+markers', name='Increased Number of Member (Normalized)',
                line=dict(color='#ff7f0e', width=2), marker=dict(color='#1446a0', size=6)))
                fig.update_layout(
                    title='Normalized Values Over Time',
                    xaxis_title='Month',
                    yaxis_title='Normalized Values',
                    legend=dict(font=dict(size=12)),
                    template='plotly_white',  # 设置白色背景
                    xaxis=dict(showgrid=True, gridcolor='LightGray'),
                    yaxis=dict(showgrid=True, gridcolor='LightGray')
                )
                figures.append(fig)

                #DTW的图
                # Define the plotting functions
                def pltDistances(dists, xlab="X", ylab="Y", clrmap="Viridis"):
                    fig = go.Figure(data=go.Heatmap(z=dists, colorscale=clrmap))
                    fig.update_layout(title='Distance Matrix', xaxis_title=xlab, yaxis_title=ylab, 
                                    xaxis=dict(showgrid=True, gridcolor='LightGray'), 
                                    yaxis=dict(showgrid=True, gridcolor='LightGray'))
                    return fig

                def pltCostAndPath(acuCost, path, xlab="X", ylab="Y", clrmap="Viridis"):
                    fig = pltDistances(acuCost, xlab=xlab, ylab=ylab, clrmap=clrmap)
                    px = [pt[0] for pt in path]
                    py = [pt[1] for pt in path]
                    fig.add_trace(go.Scatter(x=px, y=py, mode='lines', line=dict(color='red', dash='dot'), name='Path'))
                    fig.update_layout(title='Accumulated Cost')
                    return fig

                def normalize(sequence):
                    return (sequence - np.min(sequence)) / (np.max(sequence) - np.min(sequence))

                def pltWarp(s1, s2, path, df, xlab="Month", ylab="Value"):
                    s1_normalized = normalize(s1)
                    s2_normalized = normalize(s2)
                    month_data = df['month']
                    fig = go.Figure()
                    for [idx1, idx2] in path:
                        fig.add_trace(go.Scatter(x=[month_data[idx1], month_data[idx2]], y=[s1_normalized[idx1], s2_normalized[idx2]], 
                        mode='lines', line=dict(color='red', dash='dot'), name='Warping Line',
                        showlegend=(idx1 == path[0][0] and idx2 == path[0][1])))
                    fig.add_trace(go.Scatter(x=month_data, y=s1_normalized,
                    mode='lines+markers', name="Amount Paid (Normalized)", marker=dict(color='#5fad56', size=6)))
                    fig.add_trace(go.Scatter(x=month_data, y=s2_normalized,
                    mode='lines+markers', name="Increased Number of Member (Normalized)", marker=dict(color='#1446a0', size=6)))
                    fig.update_layout(title='Warping Plot', xaxis_title=xlab, yaxis_title=ylab, xaxis=dict(tickformat="%Y"))
                    return fig

                # Define the DTW computation functions
                def computeDists(x, y):
                    dists = np.zeros((len(y), len(x)))
                    for i in range(len(y)):
                        for j in range(len(x)):
                            dists[i, j] = (y[i] - x[j])**2
                    return dists

                def computeAcuCost(dists):
                    acuCost = np.zeros(dists.shape)
                    acuCost[0, 0] = dists[0, 0]
                    
                    # Accumulated costs along the first row
                    for j in range(1, dists.shape[1]):
                        acuCost[0, j] = dists[0, j] + acuCost[0, j-1]
                        
                    # Accumulated costs along the first column
                    for i in range(1, dists.shape[0]):
                        acuCost[i, 0] = dists[i, 0] + acuCost[i-1, 0]    
                    
                    # Accumulated costs for the rest of the matrix
                    for i in range(1, dists.shape[0]):
                        for j in range(1, dists.shape[1]):
                            acuCost[i, j] = min(acuCost[i-1, j-1], acuCost[i-1, j], acuCost[i, j-1]) + dists[i, j]
                    
                    return acuCost

                # Find the optimal warping path
                def findWarpPath(acuCost):
                    path = []
                    i, j = acuCost.shape[0] - 1, acuCost.shape[1] - 1
                    path.append((i, j))
                    
                    while i > 0 or j > 0:
                        if i == 0:
                            j -= 1
                        elif j == 0:
                            i -= 1
                        else:
                            min_val = min(acuCost[i-1, j-1], acuCost[i-1, j], acuCost[i, j-1])
                            if min_val == acuCost[i-1, j-1]:
                                i -= 1
                                j -= 1
                            elif min_val == acuCost[i-1, j]:
                                i -= 1
                            else:
                                j -= 1
                        path.append((i, j))
                    
                    path.reverse()
                    return path

                x = df['AmtPaid'].values
                y = df['IncreasedNM'].values
                dists = computeDists(x, y)
                acuCost = computeAcuCost(dists)
                path = findWarpPath(acuCost)
                plt3 = pltWarp(x, y, path, df, xlab="Month", ylab="Value")
                figures.append(plt3)
                plt1 = pltDistances(dists, xlab="Amount Paid", ylab="Increased Number of Member")
                figures.append(plt1)
                plt2 = pltCostAndPath(acuCost, path, xlab="Amount Paid", ylab="Increased Number of Member")
                figures.append(plt2)
                

                for i in range(0, len(figures), 2):  # 每三张图生成一行
                    cols = st.columns(2)  # 创建三列
                    for j in range(2):
                        index = i + j  # 计算当前图表的索引
                        if index < len(figures):  # 如果图表存在
                            with cols[j]:  # 在第 j 列中
                                fig = figures[index]
                                fig.update_layout(template='plotly_white')
                                st.plotly_chart(fig, use_container_width=True)  # 使用 Streamlit 的方法渲染图表
                        else:  # 如果没有图表，显示图片
                            with cols[j]:
                                st.image(image_path, use_column_width=True)  # 显示备用图片
                st.header(f"📑 View Raw Data From ***{selected_file}***:")
                st.dataframe(df.style.set_table_attributes('style="width: 100%; border-collapse: collapse;"'))  # 显示原始数据
            else:
                st.warning(f"The selected file should only have the following columns: {', '.join(required_columns)}. Please upload a file with the correct format.", icon="⚠️")
        else:
            st.warning("Please select a valid file before clicking 'Analyze'.", icon="⚠️")
else:
    st.warning("Please upload one or more files in the upload files section to get started!", icon="⚠️")
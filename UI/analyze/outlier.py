import streamlit as st
import os
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
import matplotlib.pyplot as plt
from scipy.signal import find_peaks as findPeaks
from sklearn.preprocessing import MinMaxScaler

st.title("📊 Member Data Analysis: ")
st.header(" Visual Insights on Demographics and Spending")
st.markdown("""
    This page you can easily view and analyze key characteristics across different member data.
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
            required_columns = ["MemID", "MemGen_x", "MemAge_x", "MemDuration_M_x", "ASPT_x", "MaxSPT_x", "MinSPT_x", 
                                "ANT_x", "APDR_x", "APinFavShop_x", "ATRinFavShop_x", "NGinFavShop_x", 
                                "NFavinFavShop_x", "MemGen_y", "MemAge_y", "MemDuration_M_y", "ASPT_y", 
                                "MaxSPT_y", "MinSPT_y", "ANT_y", "APDR_y", "APinFavShop_y", "ATRinFavShop_y", 
                                "NGinFavShop_y", "NFavinFavShop_y"]

            if df is not None:
                st.write(f"Preview 3 rows of data from **{selected_file}**:")
                st.dataframe(df.head(3).style.set_table_attributes('style="width: 100%; border-collapse: collapse;"'))
                if all(col in df.columns for col in required_columns) and len(df.columns) == len(required_columns):
                    st.header("🌟 View Analysis Results")
                    figures = []
                    #第一个图
                    # Scatter plot for males (MemGen_x = 1) and females (MemGen_x = 0)
                    df['MemGen_x'] = df['MemGen_x'].astype('category')
                    df['MemGen_x_display'] = df['MemGen_x'].map({0: 'Female', 1: 'Male'})
                    color_sequence = ['#ec407a', '#29b6f6']
                    fig = px.scatter(df, x='MemAge_x', y='MemGen_x', 
                    color='MemGen_x', labels={'MemAge_x': 'Member Age', 'MemGen_x': 'Member Gender', 'MemGen_x_display': 'Member Gender'},
                    title="Scatter Plot of Member Age vs Gender", color_discrete_sequence=color_sequence,
                    hover_data={'MemGen_x': False, 'MemGen_x_display': True})
                    fig.for_each_trace(lambda t: t.update(name='Female' if t.name == '0' else 'Male'))
                    fig.update_traces(marker=dict(size=12), selector=dict(mode='markers'))
                    figures.append(fig)

                    #第二个图
                    age = df['MemAge_x']
                    gender = df['MemGen_x']
                    df['Gender'] = df['MemGen_x'].map({0: 'Female', 1: 'Male'})
                    fig = px.histogram(df, 
                                    x='MemAge_x', 
                                    color='Gender', 
                                    nbins=50,  # Set the number of bins
                                    color_discrete_map={'Female': '#ec407a', 'Male': '#29b6f6'},  # Set the colors for females and males
                                    labels={'MemAge_x': 'Age', 'Gender': 'Gender'},
                                    title='Age Distribution for Males and Females')
                    fig.update_layout(
                        barmode='overlay',  # Overlay the histograms
                        xaxis_title='Member Age',
                        yaxis_title='Frequency',
                        legend_title='Gender'
                    )
                    figures.append(fig)

                    #第三个图
                    df['MemGen_x'] = df['MemGen_x'].astype('category')
                    df['MemGen_x_display'] = df['MemGen_x'].map({0: 'Female', 1: 'Male'})
                    color_sequence = ['#ec407a', '#29b6f6']
                    fig = px.scatter(df, x='MemDuration_M_x', y='MemGen_x', 
                    color='MemGen_x', labels={'MemDuration_M_x': 'Membership Duration (Months)', 'MemGen_x': 'Member Gender', 'MemGen_x_display': 'Member Gender'},
                    title="Scatter Plot of Member Duration vs Gender", color_discrete_sequence=color_sequence,
                    hover_data={'MemGen_x': False, 'MemGen_x_display': True})
                    fig.for_each_trace(lambda t: t.update(name='Female' if t.name == '0' else 'Male'))
                    fig.update_traces(marker=dict(size=12), selector=dict(mode='markers'))
                    figures.append(fig)

                    #第四个图
                    duration = df['MemDuration_M_x']
                    gender = df['MemGen_x']
                    df['Gender'] = df['MemGen_x'].map({0: 'Female', 1: 'Male'})
                    fig = px.histogram(df, 
                                    x='MemDuration_M_x', 
                                    color='Gender', 
                                    nbins=50,  # Set the number of bins
                                    color_discrete_map={'Female': '#ec407a', 'Male': '#29b6f6'},  # Set the colors for females and males
                                    labels={'MemDuration_M_x': 'Membership Duration (Months)', 'Gender': 'Gender'},
                                    title='Duration Distribution for Males and Females')
                    fig.update_layout(
                        barmode='overlay',  # Overlay the histograms
                        xaxis_title='Membership Duration (Months)',
                        yaxis_title='Frequency',
                        legend_title='Gender'
                    )
                    figures.append(fig)

                    #第五个图
                    df['MemGen_x'] = df['MemGen_x'].astype('category')
                    df['MemGen_x_display'] = df['MemGen_x'].map({0: 'Female', 1: 'Male'})
                    color_sequence = ['#ec407a', '#29b6f6']
                    fig = px.scatter(df, x='ASPT_x', y='MemGen_x', 
                    color='MemGen_x', labels={'ASPT_x': 'Averaged Spending Per Transaction', 'MemGen_x': 'Member Gender', 'MemGen_x_display': 'Member Gender'},
                    title="Scatter Plot of Member Averaged Spending vs Gender", color_discrete_sequence=color_sequence,
                    hover_data={'MemGen_x': False, 'MemGen_x_display': True})
                    fig.for_each_trace(lambda t: t.update(name='Female' if t.name == '0' else 'Male'))
                    fig.update_traces(marker=dict(size=12), selector=dict(mode='markers'))
                    figures.append(fig)

                    #第六个图
                    duration = df['ASPT_x']
                    gender = df['MemGen_x']
                    df['Gender'] = df['MemGen_x'].map({0: 'Female', 1: 'Male'})
                    fig = px.histogram(df, 
                                    x='ASPT_x', 
                                    color='Gender', 
                                    nbins=50,  # Set the number of bins
                                    color_discrete_map={'Female': '#ec407a', 'Male': '#29b6f6'},  # Set the colors for females and males
                                    labels={'ASPT_x': 'Averaged Spending Per Transaction', 'Gender': 'Gender'},
                                    title='ASPT Distribution for Males and Females')
                    fig.update_layout(
                        barmode='overlay',  # Overlay the histograms
                        xaxis_title='Averaged Spending Per Transaction',
                        yaxis_title='Frequency',
                        legend_title='Gender'
                    )
                    figures.append(fig)

                    #排列图表
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
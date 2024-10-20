import os
import pandas as pd
import plotly.graph_objects as go
import joblib
from sklearn.preprocessing import StandardScaler
from math import pi
import streamlit as st

# 加载模型
Ka = joblib.load('model/kmeans_A6.pkl')
Kf = joblib.load('model/kmeans_F8.pkl')
Km = joblib.load('model/kmeans_M6.pkl')

def generate_cluster_radar_chart(df, model, fig_name):
    """
    生成雷达图展示聚类结果。
    """
    numerical_features = df.select_dtypes(include='number').columns
    scaler = StandardScaler()
    df_scaled = scaler.fit_transform(df[numerical_features])
    cluster_labels = model.predict(df_scaled)

    data = pd.DataFrame(df_scaled, columns=numerical_features)
    data['Cluster'] = cluster_labels
    mean_values = data.groupby('Cluster').mean().reset_index()
    feature_names_mapping = {
        'MemGen_x': 'Gender',
        'ASPT_x': 'Average Spending Per Transaction',
        'MaxSPT_x': 'Maximum Spending',
        'MinSPT_x': 'Minimum Spending',
        'ANT_x': 'Average Number of Transactions',
        'APDR_x': 'Average Purchase Frequency',
        'APinFavShop_x': 'Average Spending in Favorite Shop',
        'ATRinFavShop_x': 'Average Transactions in Favorite Shop',
        'NGinFavShop_x': 'Number of Visits to Favorite Shop',
        'NFavinFavShop_x': 'Number of Favorites in Favorite Shop',
        'MemAge_x': 'Age',
        'MemDuration_M_x': 'Duration (Months)',
    }
    features = mean_values.columns[1:]
    new_feature_names = [feature_names_mapping[feat] for feat in features]
    num_vars = len(features)
    angles = [n / float(num_vars) * 2 * pi for n in range(num_vars)]
    angles += angles[:1]

    fig = go.Figure()
    for i in range(len(mean_values)):
        values = mean_values.loc[i].drop('Cluster').values.tolist()
        values += values[:1]
        fig.add_trace(go.Scatterpolar(
            r=values,
            theta=new_feature_names,
            fill='toself',
            name=f'Cluster {i}'
        ))
    fig.update_layout(
        title= fig_name ,
        polar=dict(
            radialaxis=dict(visible=True),
            angularaxis=dict(tickvals=list(range(len(features))), ticktext=new_feature_names,)
        )
    )
    return fig, cluster_labels

def plot_cluster_pie(cluster, pie_name):
    # 计算每个聚类的计数
    cluster_counts = pd.Series(cluster).value_counts().reset_index()
    cluster_counts.columns = ['Cluster', 'Count']
    cluster_counts['Renamed'] = ['Cluster ' + str(i) for i in cluster_counts['Cluster']]
    
    # 创建饼图
    pie_fig = go.Figure(data=[go.Pie(labels=cluster_counts['Renamed'], values=cluster_counts['Count'], hole=0.3)])
    pie_fig.update_layout(title_text=pie_name)
    
    # 显示图表
    return pie_fig

# 主界面部分
st.title("🎨 Consumer Profiling (Clustering)​")
st.header("K-Means")
st.markdown("""
    This interactive tool allows you to upload your dataset and perform clustering analysis on consumers.
    We use K-Means clustering to profile consumers based on key features.
""")

UPLOAD_FOLDER = "UI/data"
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
                                "NFavinFavShop_x"]

            if df is not None:
                st.write(f"Preview 3 rows of data from **{selected_file}**:")
                st.dataframe(df.head(3).style.set_table_attributes('style="width: 100%; border-collapse: collapse;"'))
                if all(col in df.columns for col in required_columns) and len(df.columns) == len(required_columns):
                    st.header("🌟 View Analysis Results")
                    analysis_mode = st.radio(
                        "Select Analysis Mode:",
                        ["All Consumers", "Females", "Males", "View All Above"]
                    )
                    fig_all, cluster_all = generate_cluster_radar_chart(df, Ka, "Clustering for All Consumers")
                    pie_all = plot_cluster_pie(cluster_all, "Proportion of All Consumer Clusters")
                    df_female = df[df['MemGen_x'] == 0]
                    fig_female, cluster_female = generate_cluster_radar_chart(df_female, Kf, "Clustering for Female Consumers")
                    pie_female = plot_cluster_pie(cluster_female, "Proportion of Female Consumer Clusters")
                    df_male = df[df['MemGen_x'] == 1]
                    fig_male, cluster_male = generate_cluster_radar_chart(df_male, Km, "Clustering for Male Consumers")
                    pie_male = plot_cluster_pie(cluster_male, "Proportion of Male Consumer Clusters")

                    if analysis_mode == "All Consumers":
                        st.plotly_chart(fig_all)
                        st.plotly_chart(pie_all)

                    if analysis_mode == "Females":
                        st.plotly_chart(fig_female)
                        st.plotly_chart(pie_female)

                    if analysis_mode == "Males":
                        st.plotly_chart(fig_male)
                        st.plotly_chart(pie_male)

                    # 如果选择“查看所有三种聚类图”，将三个图表一起显示
                    if analysis_mode == "View All Above":

                        col1, col2, col3 = st.columns(3)
                        with col1:
                            st.plotly_chart(fig_all, use_container_width=True)
                            st.plotly_chart(pie_all, use_container_width=True)
                        with col2:
                            st.plotly_chart(fig_female, use_container_width=True)
                            st.plotly_chart(pie_female, use_container_width=True)
                        with col3:
                            st.plotly_chart(fig_male, use_container_width=True)
                            st.plotly_chart(pie_male, use_container_width=True)

                    st.header(f"📑 View Raw Data From ***{selected_file}***:")
                    st.dataframe(df.style.set_table_attributes('style="width: 100%; border-collapse: collapse;"'))  # 显示原始数据
                else:
                        st.warning(f"The selected file should only have the following columns: {', '.join(required_columns)}. Please upload a file with the correct format.", icon="⚠️")
            else:
                st.warning("Please select a valid file before Analyze.", icon="⚠️")
    else:
        st.warning("Please upload one or more files in the upload files section to get started!", icon="⚠️")
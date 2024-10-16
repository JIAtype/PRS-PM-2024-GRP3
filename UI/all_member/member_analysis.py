import streamlit as st
import pandas as pd
import plotly.express as px
import os
import numpy as np
import plotly.graph_objects as go

# 设置标题
st.title("💡 Current Member Consumer Analysis:")

UPLOAD_FOLDER = "data"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
current_member_file = "current_member.xlsx"
file_path = os.path.join(UPLOAD_FOLDER, current_member_file)
image_path = os.path.join("images", "A.png")

missing_data = []

if os.path.exists(file_path):
    df = pd.read_excel(file_path)
    required_columns = ["Member ID", "Member Gender", "Member Age", "Member Duration(Month)"]
    
    if all(col in df.columns for col in required_columns):
        figures = []

        # 显示男性与女性会员的比例。
        if "Member Gender" in df.columns:
            pie_data1 = df["Member Gender"].value_counts().reset_index()
            pie_data1.columns = ['Gender', 'Count']
            # 定义颜色
            color_sequence = [' #ec407a ', '#29b6f6 ']#先粉色后蓝色
            fig1 = px.pie(
                pie_data1,
                values='Count',
                names='Gender',
                title='Gender Ratio Pie Chart',
                hole=0.2,
                color_discrete_sequence=color_sequence  # 使用指定的颜色
            )
            fig1.update_traces(
                textinfo='percent+label',  # 显示百分比和标签
                # textfont_size=14,           # 文本大小
                marker=dict(line=dict(color='#FFFFFF', width=2))  # 添加白色边框
            )
            fig1.update_layout(
                # title_font_size=20,          # 标题字体大小
                legend_title_text='Gender',     # 图例标题
                legend=dict(orientation="h"),  # 图例横向排列
                xaxis=dict(showgrid=True),
                yaxis=dict(showgrid=True)
            )
            figures.append(fig1)
        else:
            missing_data.append("Member Gender")
        # 会员年龄分布
        if "Member Age" in df.columns:
            # 将年龄分组
            age_bins = [0, 18, 30, 40, 50, 60, 70, 80, 90, 100]  # 设定年龄区间
            age_labels = ['0-17', '18-29', '30-39', '40-49', '50-59', '60-69', '70-79', '80-89', '90+']
            # 使用 pd.cut() 来分组并计算每个年龄段的会员数量
            df['Age'] = pd.cut(df['Member Age'], bins=age_bins, labels=age_labels, right=False)
            age_distribution = df['Age'].value_counts().reset_index()
            age_distribution.columns = ['Age', 'Count']
            # 排序年龄分组
            age_distribution = age_distribution.sort_values('Age')
            # 绘制折线图
            fig2 = px.line(
                age_distribution,
                x='Age',
                y='Count',
                title='Age Distribution Line Graph',
                markers=True,
                line_shape='linear'
            )
            fig2.update_traces(line=dict(color='#9ccc65 '))
            fig2.update_traces(marker=dict(color='#66bb6a')) 
            # 更新图表布局
            fig2.update_layout(
                xaxis_title='Age',
                yaxis_title='Count',
                legend_title_text='Age',
                xaxis=dict(showgrid=True),
                yaxis=dict(showgrid=True)
            )
            figures.append(fig2)
        else:
            missing_data.append("Member Age")

        # 会员持续时间分布
        if "Member Duration(Month)" in df.columns:
            # 将持续时间分组
            duration_bins = [0, 12, 24, 36, 48, 60, 72, 84, 96, 120]  # 持续时间区间（单位：个月）
            duration_labels = ['0-12 months', '13-24 months', '25-36 months', '37-48 months', '49-60 months', '61-72 months', '73-84 months', '85-96 months', '97-120 months']
            # 使用 pd.cut() 来分组并计算每个持续时间段的会员数量
            df['Duration'] = pd.cut(df['Member Duration(Month)'], bins=duration_bins, labels=duration_labels, right=False)
            duration_distribution = df['Duration'].value_counts().reset_index()
            duration_distribution.columns = ['Duration', 'Count']
            # 排序持续时间分组
            duration_distribution = duration_distribution.sort_values('Duration')
            # 绘制柱状图
            fig3 = px.bar(
                duration_distribution,
                x='Duration',
                y='Count',
                title='Duration Distribution Histogram',
                color='Count',
                color_continuous_scale=[(0, '#ab47bc'), (1, '#5c6bc0')]
            )
            # 更新图表布局
            fig3.update_layout(
                xaxis_title='Duration',
                yaxis_title='Count',
                legend_title_text='Distribution of Duration',
                xaxis=dict(showgrid=True),
                yaxis=dict(showgrid=True)
            )
            figures.append(fig3)
        else:
            missing_data.append("Member Duration(Month)")
        
        #显示平均每月消费与交易次数之间的关系。
        if all(col in df.columns for col in ["Member Duration(Month)", "Averaged Spending Per Transaction", "Averaged Number of Transaction"]):
            df['Monthly Spending'] = df['Averaged Spending Per Transaction'] * df['Averaged Number of Transaction']
            monthly_data = df.groupby('Member ID').agg({
                'Monthly Spending': 'mean',  # 计算每月平均消费
                'Averaged Number of Transaction': 'mean'  # 计算每月交易次数
            }).reset_index()
            # 绘制散点图
            fig7 = px.scatter(
                monthly_data,
                x='Monthly Spending',
                y='Averaged Number of Transaction',
                title='Monthly Spending and Frequency',
                labels={'Monthly Spending': 'Average Monthly Spending', 'Averaged Number of Transaction': 'Average Number of Transactions'},
                color='Averaged Number of Transaction',  # 根据交易数量变化颜色
                color_continuous_scale=['#ff7043', '#ef5350']  # 浅色渐变
            )
            # 更新图表布局
            fig7.update_layout(
                xaxis_title='Spending',
                yaxis_title='Transactions',
                xaxis=dict(showgrid=True),
                yaxis=dict(showgrid=True)
            )
            figures.append(fig7)
        else:
            missing_data.append("Averaged Spending Per Transaction")
            missing_data.append("Averaged Number of Transaction")

        #热力图，将折扣率和平均消费划分成网格，使用颜色深浅来表示每个区域的消费者数量，从而更直观地看到折扣率与消费之间的聚集趋势。
        if all(col in df.columns for col in ["Averaged Purchasing Discount Rate", "Averaged Spending Per Transaction"]):
            # 计算折扣率与平均消费的聚合数据
            heatmap_data = df.groupby(['Averaged Purchasing Discount Rate', 'Averaged Spending Per Transaction']).size().reset_index(name='Consumer Count')
            # 绘制热力图
            fig8 = px.density_heatmap(
                heatmap_data,
                x='Averaged Purchasing Discount Rate',
                y='Averaged Spending Per Transaction',
                z='Consumer Count',
                color_continuous_scale=['#ffee58','#ef5350'],  # 颜色渐变，选择适合的颜色
                title='Density by Discount Rate and Spending'
            )
            # 更新图表布局
            fig8.update_layout(
                xaxis_title='Averaged Purchasing Discount Rate',
                yaxis_title='Averaged Spending',
                xaxis=dict(showgrid=True),
                yaxis=dict(showgrid=True)
            )
            figures.append(fig8)
        else:
            missing_data.append("Averaged Purchasing Discount Rate")

        # 根据年龄段（如青少年、青年、中年、老年）分组，使用箱线图展示每个年龄段的平均消费、最大消费和最小消费。
        # 这可以直观比较不同年龄组的消费特征。
        if all(col in df.columns for col in ["Member Age", "Averaged Spending Per Transaction", "Maximun Spending Per Transaction", "Minimun Spending Per Transaction"]):
            bins = [0, 18, 30, 45, 60, 75, 100]  # 定义年龄段
            labels = ['0-18', '19-30', '31-45', '46-60', '61-75', '76-100']  # 年龄段标签
            df['Age'] = pd.cut(df['Member Age'], bins=bins, labels=labels, right=False)
            # 计算每个年龄段的每月消费均值
            spending_summary = df.groupby('Age').agg({
                'Averaged Spending Per Transaction': 'mean',
                'Maximun Spending Per Transaction': 'mean',
                'Minimun Spending Per Transaction': 'mean'
            }).reset_index()
            # 将数据转换为长格式
            long_df = pd.melt(spending_summary, id_vars='Age', 
                            value_vars=['Averaged Spending Per Transaction', 
                                        'Maximun Spending Per Transaction', 
                                        'Minimun Spending Per Transaction'], 
                            var_name='Spending Type', value_name='Amount')
            long_df['Spending Type'] = long_df['Spending Type'].replace({
                'Averaged Spending Per Transaction': 'Avg',
                'Maximun Spending Per Transaction': 'Max',
                'Minimun Spending Per Transaction': 'Min'
            })
            # 创建箱线图
            fig9 = px.box(long_df, x='Age', y='Amount', color='Spending Type', 
                        title='Average Max Min Spending Per Transaction by Age',
                        category_orders={'Age': labels}, color_discrete_sequence=['#9ccc65', '#ef5350', '#42a5f5'])
            # 更新布局
            fig9.update_layout(
                yaxis_title='Spending Per Transaction',
                xaxis_title='Age',
                xaxis=dict(showgrid=True),
                yaxis=dict(showgrid=True)
            )
            figures.append(fig9)
        else:
            missing_data.append("Maximun Spending Per Transaction")
            missing_data.append("Minimun Spending Per Transaction")

        # 比较男性和女性在每笔交易中的平均消费。
        if all(col in df.columns for col in ["Member Age", "Averaged Spending Per Transaction"]):
            bins = [0, 18, 30, 45, 60, 75, 100]  # 定义年龄段
            labels = ['0-18', '19-30', '31-45', '46-60', '61-75', '76-100']  # 年龄段标签
            df['Age'] = pd.cut(df['Member Age'], bins=bins, labels=labels, right=False)
            # 计算每个年龄段的每月消费总和
            monthly_spending = df.groupby('Age').agg({'Averaged Spending Per Transaction': 'mean'}).reset_index()
            # 绘制折线图
            fig4 = px.line(
                monthly_spending,
                x='Age',
                y='Averaged Spending Per Transaction',
                title='Monthly Spending Trend by Age',
                markers=True
            )
            fig4.update_traces(line=dict(color='#b39ddb')) 
            fig4.update_traces(marker=dict(color='#7e57c2')) 
            # 更新图表布局
            fig4.update_layout(
                xaxis_title='Age',
                yaxis_title='Average Spending',
                xaxis=dict(showgrid=True),
                yaxis=dict(showgrid=True)
            )
            # 将图表添加到 figures 列表中
            figures.append(fig4)

        # 展示不同年龄段的会员在消费上的趋势，每月消费变化。
        if all(col in df.columns for col in ["Member Gender", "Averaged Spending Per Transaction"]):
            avg_spending = df.groupby("Member Gender")["Averaged Spending Per Transaction"].mean().reset_index()
            avg_spending.columns = ['Gender', 'Averaged Spending Per Transaction'] 
            # 绘制柱状图
            fig5 = px.bar(
                avg_spending,
                x='Gender',
                y='Averaged Spending Per Transaction',
                title='Monthly Spending Trend by Gender',
                color='Averaged Spending Per Transaction',
                color_continuous_scale=['#ec407a', '#29b6f6']  #粉色 蓝色
            )
            # 更新图表布局
            fig5.update_layout(
                xaxis_title='Gender',
                yaxis_title='Average Spending',
                legend_title_text='Averaged Spending',
                xaxis=dict(showgrid=True),
                yaxis=dict(showgrid=True)
            )
            figures.append(fig5)

        #分析会员在系统中持续时间与其消费的关系
        if all(col in df.columns for col in ["Member Duration(Month)", "Averaged Spending Per Transaction"]):
            bins = [0, 12, 24, 36, 48, 60, 72, 84, 96, 120]  # 定义持续时间段（以月份为单位）
            labels = ['0-12 months', '13-24 months', '25-36 months', '37-48 months', '49-60 months', '61-72 months', '73-84 months', '85-96 months', '97-120 months']  # 持续时间段标签
            df['Duration'] = pd.cut(df['Member Duration(Month)'], bins=bins, labels=labels, right=False)
            avg_spending_duration = df.groupby('Duration').agg({'Averaged Spending Per Transaction': 'mean'}).reset_index()
            # 绘制折线图
            fig6 = px.line(
                avg_spending_duration,
                x='Duration',
                y='Averaged Spending Per Transaction',
                title='Monthly Spending Trend by Duration',
                markers=True
            )
            fig6.update_traces(line=dict(color='#ffa726'))
            fig6.update_traces(marker=dict(color='#ff7043'))
            # 更新图表布局
            fig6.update_layout(
                xaxis_title='Member Duration (Months)',
                yaxis_title='Average Spending',
                xaxis=dict(showgrid=True),
                yaxis=dict(showgrid=True)
            )
            figures.append(fig6)

        for i in range(0, len(figures), 3):  # 每三张图生成一行
            cols = st.columns(3)  # 创建三列
            for j in range(3):
                index = i + j  # 计算当前图表的索引
                if index < len(figures):  # 如果图表存在
                    with cols[j]:  # 在第 j 列中
                        fig = figures[index]
                        fig.update_layout(
                            width=300,
                            height=300,
                            margin=dict(l=20, r=20, t=50, b=20)
                        )
                        st.plotly_chart(fig, use_container_width=True)  # 渲染图表
                else:  # 如果没有图表，显示图片
                    with cols[j]:
                        st.image(image_path, use_column_width=True)  # 显示备用图片

        if missing_data:
            st.warning(f"For the chart(s) that failed to generate, the following essential data is missing: {', '.join(missing_data)}. Please re-upload the file.", icon="⚠️")

    else:
        st.warning(f"The uploaded file does not contain the required basic information columns: {', '.join(required_columns)}. Please upload a file with correct format.", icon="⚠️")
else:
    st.warning("No file named 'current_member.xlsx' found in the folder for uploading files. Please upload the file to get started!", icon="⚠️")

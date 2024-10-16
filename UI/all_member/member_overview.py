import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import plotly.express as px
import os

# 设置标题
st.title("📑 Current Member Consumer Data Preview:")

UPLOAD_FOLDER = "data"
# 确保文件夹存在
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
# 指定要读取的文件名
current_member_file = "current_member.xlsx"
file_path = os.path.join(UPLOAD_FOLDER, current_member_file)

# 检查文件是否存在
if os.path.exists(file_path):
    # 读取文件
    df = pd.read_excel(file_path)
    st.dataframe(df)

    # 检查列名
    required_columns = ["Member ID", "Member Gender", "Member Age", "Member Duration(Month)"]
    if all(col in df.columns for col in required_columns):
        st.success("✅ File contains the required columns")
    else:
        st.warning(f"The uploaded file does not contain the required basic information columns: {', '.join(required_columns)}. Please upload a file with correct format.", icon="⚠️")
else:
    st.warning("No file named 'current_member.xlsx' found in the folder for uploading files. Please upload the file to get started!", icon="⚠️")

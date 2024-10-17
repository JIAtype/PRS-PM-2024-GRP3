import streamlit as st
import pandas as pd
import os

st.title("📑 Show Data")
st.markdown("This page allows you to view data files in the upload folder. Please select a file below.")

UPLOAD_FOLDER = "UI/data"

# 检查上传文件夹是否为空
if os.listdir(UPLOAD_FOLDER):
    uploaded_files = [f for f in os.listdir(UPLOAD_FOLDER) if f != ".DS_Store"]
    uploaded_files = ["Select File"] + uploaded_files
    selected_file = st.selectbox("Please select a file to visualize:", uploaded_files)

    if selected_file != "Select File":  # 仅当选择有效文件时读取数据
        file_path = os.path.join(UPLOAD_FOLDER, selected_file)

        # 检测文件类型并读取数据
        if selected_file.endswith(".csv"):
            df = pd.read_csv(file_path)
        elif selected_file.endswith(".xlsx"):
            df = pd.read_excel(file_path)
        else:
            df = pd.DataFrame()
            st.error("Please select a valid file to delete.")

        # 显示数据
        st.write(f"###Raw Data from **{selected_file}**:")
        st.dataframe(df.style.set_table_attributes('style="width: 100%; border-collapse: collapse;"'))

else:
    st.warning("Please upload one or more files in the upload section.", icon="⚠️")


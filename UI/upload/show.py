import streamlit as st
import pandas as pd
import os

st.title("📑 Show Data")
st.markdown("This page allows you to view data files in the upload folder. Please select a file below.")

UPLOAD_FOLDER = "UI/data"

# 检查上传文件夹是否存在
if not os.path.exists(UPLOAD_FOLDER):
    st.error("Upload folder does not exist. Please check the path.")
else:
    # 检查上传文件夹是否为空
    uploaded_files = [f for f in os.listdir(UPLOAD_FOLDER) if f != ".DS_Store"]
    
    if uploaded_files:
        uploaded_files = ["Select File"] + uploaded_files
        selected_file = st.selectbox("Please select a file to visualize:", uploaded_files)

        if selected_file != "Select File":  # 仅当选择有效文件时读取数据
            file_path = os.path.join(UPLOAD_FOLDER, selected_file)

            # 检测文件类型并读取数据
            try:
                if selected_file.endswith(".csv"):
                    df = pd.read_csv(file_path)
                elif selected_file.endswith(".xlsx"):
                    df = pd.read_excel(file_path)
                else:
                    st.error("Please select a valid file to visualize.")
                    df = pd.DataFrame()

                # 显示数据
                st.write(f"### Raw Data from **{selected_file}**:")
                st.dataframe(df.style.set_table_attributes('style="width: 100%; border-collapse: collapse;"'))
            except Exception as e:
                st.error(f"Error reading file: {e}")
    else:
        st.warning("Please upload one or more files in the upload section.", icon="⚠️")
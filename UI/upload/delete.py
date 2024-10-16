import streamlit as st
import pandas as pd
import os

st.title("🗑️ Delete Data")
st.markdown("This page allows you to delete data files in the upload folder. Please select a file below.")

UPLOAD_FOLDER = "UI/data"

# 检查上传文件夹是否为空
if os.listdir(UPLOAD_FOLDER):
    uploaded_files = [f for f in os.listdir(UPLOAD_FOLDER) if f != ".DS_Store"]
    uploaded_files = ["Select File"] + uploaded_files
    selected_file = st.selectbox("Please select a file to delete:", uploaded_files)

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
        st.write(f"### Data from ***{selected_file}***:")
        st.dataframe(df.head(6).style.set_table_attributes('style="width: 100%; border-collapse: collapse;"'))
        
        st.write(f"Confirm **deletion** of ***{selected_file}***? ")
        if st.button("Confirm Delete ❗"):
            os.remove(file_path)
            st.success("The file has been deleted!")
    # else:
    #     st.error("Please select a valid file to delete.")
else:
    st.warning("Please upload one or more files in the upload section.", icon="⚠️")


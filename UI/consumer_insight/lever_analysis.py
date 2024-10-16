import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import os

dt = joblib.load('model/decision_tree_model.pkl')

st.title("Consumer Lever Insight")

if st.session_state.role == "Admin":
    UPLOAD_FOLDER = "a_data"
if st.session_state.role == "Clerk":
    UPLOAD_FOLDER = "c_data"

if os.path.isdir(UPLOAD_FOLDER) and os.listdir(UPLOAD_FOLDER):
    uploaded_files = os.listdir(UPLOAD_FOLDER)
    # 在下拉框中添加一个默认选项
    uploaded_files = ["Select File"] + uploaded_files
    # st.markdown("Please select the file to visualize")
    selected_file = st.selectbox("Please select the file to visualize here:", uploaded_files)
    df = None
    if selected_file != "Select File":  # 仅当选择有效文件时读取数据
        # 获取文件的完整路径
        file_path = os.path.join(UPLOAD_FOLDER, selected_file)
        # 检测文件类型并读取数据
        if selected_file.endswith(".csv"):
            df = pd.read_csv(file_path)
        elif selected_file.endswith(".xlsx"):
            df = pd.read_excel(file_path)
else:
    st.warning("Please upload one or more files in the upload files section to get started!", icon="⚠️")

required_columns = ["MemID", "MemGen_x", "MemAge_x", "MemDuration_M_x","ASPT_x", "MaxSPT_x", "MinSPT_x", "ANT_x", "APDR_x", "APinFavShop_x", "ATRinFavShop_x", "NGinFavShop_x", "NFavinFavShop_x", "MemGen_y", "MemAge_y", "MemDuration_M_y", "ASPT_y", "MaxSPT_y", "MinSPT_y", "ANT_y", "APDR_y", "APinFavShop_y", "ATRinFavShop_y", "NGinFavShop_y", "NFavinFavShop_y", "ProdName"]

if st.button("analysis"):
    if df is not None:
        if all(col in df.columns for col in required_columns):
            # Data preprocessing for MemAge_y
            bin_edges = [15, 20, 30, 40, 50, 60, 70, 85]
            bin_labels = ['15-20', '21-30', '31-40', '41-50', '51-60', '61-70', '71-85']
            df['MemAge_y_binned'] = pd.cut(df['MemAge_y'], bins=bin_edges, labels=bin_labels, include_lowest=True)
            df['MemAge_y_binned_code'] = df['MemAge_y_binned'].cat.codes

            # Binning for ASPT_x and ANT_x using thresholds
            top_30_asptx_threshold = 2113.916
            top_30_antx_threshold = 0.363636364

            # Create binned columns for ASPT_x and ANT_x as numeric codes (0 or 1)
            df['ASPT_x_binned_code'] = df['ASPT_x'].apply(lambda x: 1 if x >= top_30_asptx_threshold else 0)
            df['ANT_x_binned_code'] = df['ANT_x'].apply(lambda x: 1 if x >= top_30_antx_threshold else 0)

            features_to_extract = ['MemAge_y_binned_code', 'ASPT_x_binned_code', 'ANT_x_binned_code']

            if all(feature in df.columns for feature in features_to_extract):
                data_points = df[features_to_extract].values  # Get all rows
                predictions = dt.predict(data_points)
                df['Predicted_Class'] = predictions

                def highlight_rows(row):
                    return ['background-color: yellow' if row['Predicted_Class'] == 1 else '' for _ in row]

                results = df[['MemID', 'Predicted_Class']]
                st.subheader("分析后的结果：")
                styled_results = results.style.apply(highlight_rows, axis=1)
                st.dataframe(styled_results)
                # st.write(results)
            else:
                st.warning("Data is missing one or more of the features needed for analysis.", icon="⚠️")
        else:
            st.warning(f"The selected file does not contain the required columns: {', '.join(required_columns)}. Please upload a file with the correct format.", icon="⚠️")
    else:
        st.warning("Please select a valid file before clicking 'Analyze'.", icon="⚠️")
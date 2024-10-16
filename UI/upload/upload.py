import streamlit as st
import os

# Set the upload directory
UPLOAD_FOLDER = "UI/data"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

st.title("📁 File Upload")
st.write("Welcome to the File Upload System! Please upload your CSV or XLSX files.")

uploaded_files = st.file_uploader("Choose files to upload", 
                                    type=["csv", "xlsx"], 
                                    accept_multiple_files=True, 
                                    label_visibility="collapsed")

if uploaded_files:
    for uploaded_file in uploaded_files:
        file_path = os.path.join(UPLOAD_FOLDER, uploaded_file.name)
        with open(file_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        st.success(f"✅ File '{uploaded_file.name}' has been successfully uploaded to `{file_path}`")
# else:
#     st.warning('Note: No files are being uploaded right now.', icon="⚠️")
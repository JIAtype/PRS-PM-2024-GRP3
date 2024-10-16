import streamlit as st
import time

# 使用 st.set_page_config() 来设置页面的宽度和其他布局选项。
st.set_page_config(layout="wide")

upload = st.Page(
    "UI/upload/upload.py",
    title="Upload Files",
    icon=":material/upload_file:",
)
show = st.Page(
    "UI/upload/show.py",
    title="View Files",
    icon=":material/task:",
)
delete = st.Page(
    "UI/upload/delete.py",
    title="Delete Files",
    icon=":material/scan_delete:",
)
dtw = st.Page(
    "UI/analyze/dtw.py",
    title="Relationship",
    icon=":material/analytics:", 
)
peaks = st.Page(
    "UI/analyze/peaks.py",
    title="Peaks",
    icon=":material/timeline:", 
)
outlier = st.Page(
    "UI/analyze/outlier.py",
    title="Outlier",
    icon=":material/area_chart:", 
)
kmeans = st.Page(
    "UI/analyze/kmeans.py",
    title="Category",
    icon=":material/donut_small:", 
)
svip = st.Page(
    "UI/svip.py",
    title="SVIP",
    icon=":material/stars:", 
)

upload_pages = [upload, show, delete]
analyze = [dtw, peaks, outlier, kmeans]
prediction = [svip]

st.logo("UI/images/horizontal_blue.png")

# 按照希望的顺序排列页面组
page_dict = {}
page_dict["File Operations"] = upload_pages
page_dict["Analyze"] = analyze
page_dict["Prediction"] = prediction

pg = st.navigation(page_dict)

# 启动应用
pg.run()
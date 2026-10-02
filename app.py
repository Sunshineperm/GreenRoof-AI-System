import streamlit as st
from PIL import Image
from modules.statistics import coverage_rate
from modules.heat_analysis import pearson_analysis

st.set_page_config(page_title="GreenRoof AI", layout="wide")
st.title("GreenRoof AI")
st.subheader("绿色屋顶遥感智能识别与城市热环境评估平台")

img = st.file_uploader("上传遥感影像", type=["png","jpg","jpeg","tif","tiff"])
if img:
    st.image(Image.open(img), caption="输入遥感影像")
    st.success("影像已加载")

st.info("GR-Net权重: models_checkpoint_epoch50.pth")
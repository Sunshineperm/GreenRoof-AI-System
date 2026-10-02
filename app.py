"""Run the complete original RoofScope working surface inside Streamlit.

The local frontend includes its own trained GR-Net ONNX model and all assets.
It does not embed the original hosted site or require that site to stay online.
"""
from pathlib import Path
import streamlit as st
import streamlit.components.v1 as components

st.set_page_config(page_title="绿顶智析 · RoofScope", page_icon="🌿", layout="wide", initial_sidebar_state="collapsed")
FRONTEND = Path(__file__).resolve().parent / "frontend"
if not (FRONTEND / "index.html").is_file():
    st.error("缺少 frontend 文件夹。请把完整更新包解压到 app.py 所在目录，然后重新部署。")
    st.stop()
if not (FRONTEND / "model" / "manifest.json").is_file():
    st.error("缺少内置训练模型，请同时上传 frontend/model 文件夹。")
    st.stop()

st.markdown("""<style>
[data-testid="stHeader"] { background: transparent; height: 0; }
[data-testid="stMainBlockContainer"] { max-width: 100%; padding: 0; }
[data-testid="stVerticalBlock"] { gap: 0; }
[data-testid="stElementContainer"]:has(iframe) { width: 100%; }
iframe[title*="roofscope_workbench"] { width: 100%; border: 0; }
footer { display: none; }
</style>""", unsafe_allow_html=True)
workbench = components.declare_component("roofscope_workbench", path=str(FRONTEND))
workbench(key="roofscope-full-workbench", default=None)

import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

# -----------------------------
# Page Configuration
# -----------------------------
st.set_page_config(
    page_title="Gender Classification AI",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# -----------------------------
# Custom CSS (Apple Inspired)
# -----------------------------
st.markdown("""
<style>

html, body, [class*="css"]{
    background:#0d1117;
    color:white;
    font-family: -apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,sans-serif;
}

.block-container{
    padding-top:2rem;
    padding-bottom:2rem;
}

.main-title{
    text-align:center;
    font-size:60px;
    font-weight:700;
    margin-bottom:10px;
}

.subtitle{
    text-align:center;
    color:#9CA3AF;
    font-size:22px;
    margin-bottom:50px;
}

.card{
    background:#161b22;
    border-radius:25px;
    padding:30px;
    box-shadow:0px 8px 30px rgba(0,0,0,.35);
}

.predictButton>button{
    width:100%;
    border-radius:15px;
    height:60px;
    font-size:20px;
    font-weight:bold;
}

.result{
    font-size:40px;
    text-align:center;
    font-weight:bold;
}

footer{
    visibility:hidden;
}

</style>
""", unsafe_allow_html=True)

# -----------------------------
# Header
# -----------------------------
st.markdown(
    '<div class="main-title">Gender Classification AI</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="subtitle">Minimal • Fast • Deep Learning Powered</div>',
    unsafe_allow_html=True
)
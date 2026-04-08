import streamlit as st
import pandas as pd
import numpy as np
import os

# 设置页面标题和图标
st.set_page_config(
    page_title="Wildfire Response Dashboard",
    page_icon="🔥",
    layout="wide"
)

st.title("🔥 Wildfire Detection & Integrated Response System")
st.markdown("""
*This dashboard integrates Risk Prediction (Obj 1), Spread Simulation (Obj 2), and Disaster Reporting (Obj 3).*
""")

# --- Sidebar: Integration Status ---
st.sidebar.header("System Integration Status")
st.sidebar.success("Frontend: Operational")
st.sidebar.info("Model Backend: Pending Schema Sync")

# --- Objective 1: Risk Prediction (Your Core Task) ---
st.header("1. Fire Risk Monitoring (Objective 1)")
col1, col2 = st.columns([2, 1])

with col1:
    st.subheader("Regional Risk Map")
    # 模拟地图数据，等 Li-Hsun 的 GeoJSON 出来后直接替换
    map_data = pd.DataFrame(
        np.random.randn(10, 2) / [50, 50] + [37.77, -122.41],
        columns=['lat', 'lon']
    )
    st.map(map_data)
    st.caption("Live visualization of XGBoost risk scores across 1km² grids.")

with col2:
    st.subheader("Key Risk Indicators")
    # 这里体现你对 TDD 12个特征的理解
    st.metric(label="Average Temperature", value="32°C", delta="1.2°C")
    st.metric(label="Soil Moisture (SMAP)", value="12%", delta="-2%", delta_color="inverse")
    st.progress(85, text="High Risk Level Detected")

# --- Objective 2: Spread Simulation ---
st.divider()
st.header("2. Fire Spread Simulation (Objective 2)")
st.write("Dynamic simulation of fire perimeter using **Cell2Fire**.")

# 预留给 Mohammed 的接口
if st.button("Run Latest Simulation"):
    st.warning("Awaiting GeoJSON sequence from Model-Pipeline/outputs/simulation/")
else:
    st.info("Select a historical fire event or active hotspot to simulate spread.")

# --- Objective 3: Disaster Reporting ---
st.divider()
st.header("3. Automated Disaster Report (Objective 3)")
st.subheader("Gemini 3.1 Flash-Lite Generation")

# 模拟报告内容
report_placeholder = """
**Situation Report: Southern California Sector**
- **Incident Level**: Grade 3 (Critical)
- **Primary Cause**: Low humidity + High wind gust.
- **Recommended Action**: Deploy assets to Sector B; notify local FEMA NRI communities.
"""
st.text_area("Latest AI-Generated Report", value=report_placeholder, height=150)
st.button("Export to PDF")

# --- Footer ---
st.markdown("---")
st.caption("Developed by Bohan (ChrisZhao5) | Northeastern University ML Project 2026")
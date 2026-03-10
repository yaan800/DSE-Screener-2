"""
DSE Technical Screener - Minimal Working Version
"""

import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime
import plotly.graph_objects as go
from plotly.subplots import make_subplots

st.set_page_config(page_title="DSE Technical Screener", page_icon="📈", layout="wide")

st.title("📈 DSE Technical Screener")
st.markdown("Professional stock analysis with technical indicators")

# Sidebar for file upload
with st.sidebar:
    st.header("📊 Upload Data")
    price_file = st.file_uploader("Upload Price Data (CSV)", type=['csv'])
    volume_file = st.file_uploader("Upload Volume Data (CSV)", type=['csv'])

# Main content
if price_file is None or volume_file is None:
    st.warning("⚠️ Please upload both Price and Volume CSV files")
    st.stop()

try:
    # Load price data
    price_df = pd.read_csv(price_file)
    st.success(f"✓ Price file loaded: {price_df.shape[0]} rows, {price_df.shape[1]} columns")
    
    # Load volume data
    volume_df = pd.read_csv(volume_file)
    st.success(f"✓ Volume file loaded: {volume_df.shape[0]} rows, {volume_df.shape[1]} columns")
    
    # Display first few rows
    st.subheader("Price Data Preview")
    st.dataframe(price_df.head())
    
    st.subheader("Volume Data Preview")
    st.dataframe(volume_df.head())
    
except Exception as e:
    st.error(f"Error loading files: {str(e)}")
    st.stop()

st.success("✓ App is working! Files loaded successfully.")

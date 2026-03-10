"""
DSE Technical Screener - Fixed Version
Proper date parsing for DD-Mon format (09-Mar, 08-Mar, etc.)
"""

import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import warnings
warnings.filterwarnings('ignore')

st.set_page_config(page_title="DSE Technical Screener", page_icon="📈", layout="wide")

st.title("📈 DSE Technical Screener")
st.markdown("Professional stock analysis with technical indicators")

# Sidebar for file upload
with st.sidebar:
    st.header("📊 Upload Data")
    price_file = st.file_uploader("Upload Price Data (CSV)", type=['csv'])
    volume_file = st.file_uploader("Upload Volume Data (CSV)", type=['csv'])
    
    st.subheader("Select Timeframe")
    timeframe = st.radio("Timeframe", ['Daily', 'Weekly', 'Monthly'])
    
    st.subheader("Select Indicators")
    show_sma = st.checkbox("SMA (20, 50)", value=True)
    show_bb = st.checkbox("Bollinger Bands", value=True)
    show_rsi = st.checkbox("RSI", value=True)
    show_macd = st.checkbox("MACD", value=True)
    show_mfi = st.checkbox("MFI", value=True)

if price_file is None or volume_file is None:
    st.warning("⚠️ Please upload both Price and Volume CSV files")
    st.stop()

def load_data_from_file(df, file_type='price'):
    """Load and parse data with proper date handling"""
    try:
        # Get ticker column (first column)
        ticker_col = df.columns[0]
        
        # Melt the dataframe from wide to long format
        melted = df.melt(id_vars=[ticker_col], var_name='Date', value_name=file_type.capitalize())
        melted.columns = ['Ticker', 'Date', file_type.capitalize()]
        
        # Convert date - handle DD-Mon format (e.g., 09-Mar, 08-Mar)
        # Try multiple formats
        try:
            # First try: DD-Mon format with current year
            melted['Date'] = pd.to_datetime(melted['Date'], format='%d-%b', errors='coerce')
            current_year = pd.Timestamp.now().year
            melted['Date'] = melted['Date'].apply(lambda x: x.replace(year=current_year) if pd.notna(x) else x)
        except:
            pass
        
        # If still have NaT values, try other formats
        if melted['Date'].isna().any():
            try:
                melted.loc[melted['Date'].isna(), 'Date'] = pd.to_datetime(
                    melted.loc[melted['Date'].isna(), 'Date'], 
                    format='%d-%b-%y', 
                    errors='coerce'
                )
            except:
                pass
        
        # Remove rows with invalid dates
        melted = melted.dropna(subset=['Date'])
        
        # Convert price/volume to numeric - handle commas in numbers
        melted[file_type.capitalize()] = melted[file_type.capitalize()].astype(str).str.replace(',', '')
        melted[file_type.capitalize()] = pd.to_numeric(melted[file_type.capitalize()], errors='coerce')
        melted = melted.dropna(subset=[file_type.capitalize()])
        
        # Sort by date (oldest first)
        melted = melted.sort_values('Date').reset_index(drop=True)
        
        return melted
    except Exception as e:
        st.error(f"Error loading file: {str(e)}")
        return None

# Load and process data
with st.spinner("Loading data..."):
    price_df = load_data_from_file(pd.read_csv(price_file), 'price')
    volume_df = load_data_from_file(pd.read_csv(volume_file), 'volume')
    
    if price_df is None or volume_df is None:
        st.error("Failed to load data files")
        st.stop()
    
    # Merge data
    merged = price_df.merge(volume_df, on=['Ticker', 'Date'], how='inner')
    
    if len(merged) == 0:
        st.error("No matching data between price and volume files")
        st.stop()
    
    # Add OHLC columns
    merged['Open'] = merged['Price']
    merged['High'] = merged['Price']
    merged['Low'] = merged['Price']
    merged['Close'] = merged['Price']
    
    stocks = merged['Ticker'].unique()
    stock_counts = merged.groupby('Ticker').size()
    valid_stocks = stock_counts[stock_counts >= 10].index.tolist()
    
    if len(valid_stocks) == 0:
        st.error("❌ No valid data to analyze - not enough data points per stock")
        st.stop()

st.success(f"✓ Loaded {len(valid_stocks)} stocks with sufficient data")

# Calculate indicators
def calculate_rsi(prices, period=14):
    delta = prices.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / (loss + 1e-10)
    return 100 - (100 / (1 + rs))

def calculate_macd(prices, fast=12, slow=26, signal=9):
    ema_fast = prices.ewm(span=fast, adjust=False).mean()
    ema_slow = prices.ewm(span=slow, adjust=False).mean()
    macd = ema_fast - ema_slow
    macd_signal = macd.ewm(span=signal, adjust=False).mean()
    return macd, macd_signal

def calculate_bollinger_bands(prices, period=20, std_dev=2.0):
    sma = prices.rolling(window=period).mean()
    std = prices.rolling(window=period).std()
    upper = sma + (std * std_dev)
    lower = sma - (std * std_dev)
    return upper, sma, lower

def calculate_mfi(high, low, close, volume, period=14):
    typical_price = (high + low + close) / 3
    money_flow = typical_price * volume
    positive_flow = money_flow.where(typical_price > typical_price.shift(), 0)
    negative_flow = money_flow.where(typical_price < typical_price.shift(), 0)
    positive_mf = positive_flow.rolling(window=period).sum()
    negative_mf = negative_flow.rolling(window=period).sum()
    mfi_ratio = positive_mf / (negative_mf + 1e-10)
    return 100 - (100 / (1 + mfi_ratio))

# Main dashboard
tab1, tab2 = st.tabs(["📊 Dashboard", "🔍 Stock Explorer"])

with tab1:
    st.header("Market Overview")
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Total Stocks", len(valid_stocks))
    with col2:
        st.metric("Data Points", len(merged))
    with col3:
        avg_points = len(merged) / len(valid_stocks)
        st.metric("Avg Points/Stock", f"{avg_points:.0f}")
    
    st.subheader("Stocks with Data")
    st.write(stock_counts[stock_counts >= 10].sort_values(ascending=False))

with tab2:
    st.header("Stock Analysis")
    
    selected_stock = st.selectbox("Select Stock", sorted(valid_stocks))
    
    if selected_stock:
        stock_data = merged[merged['Ticker'] == selected_stock].sort_values('Date').copy()
        
        if len(stock_data) >= 10:
            # Calculate indicators
            stock_data['RSI'] = calculate_rsi(stock_data['Close'])
            stock_data['MACD'], stock_data['MACD_Signal'] = calculate_macd(stock_data['Close'])
            stock_data['BB_Upper'], stock_data['BB_Middle'], stock_data['BB_Lower'] = calculate_bollinger_bands(stock_data['Close'])
            stock_data['SMA_20'] = stock_data['Close'].rolling(window=20).mean()
            stock_data['SMA_50'] = stock_data['Close'].rolling(window=50).mean()
            stock_data['MFI'] = calculate_mfi(stock_data['High'], stock_data['Low'], stock_data['Close'], stock_data['Volume'])
            
            # Display metrics
            latest = stock_data.iloc[-1]
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Price", f"{latest['Close']:.2f}")
            with col2:
                st.metric("RSI", f"{latest['RSI']:.1f}")
            with col3:
                st.metric("MACD", f"{latest['MACD']:.4f}")
            with col4:
                st.metric("MFI", f"{latest['MFI']:.1f}")
            
            # Create candlestick chart
            st.subheader("Price Chart")
            
            fig = go.Figure()
            
            # Candlestick
            fig.add_trace(go.Candlestick(
                x=stock_data['Date'],
                open=stock_data['Open'],
                high=stock_data['High'],
                low=stock_data['Low'],
                close=stock_data['Close'],
                name='OHLC'
            ))
            
            # Moving averages
            if show_sma:
                fig.add_trace(go.Scatter(
                    x=stock_data['Date'], y=stock_data['SMA_20'],
                    name='SMA 20', line=dict(color='orange', width=1)
                ))
                fig.add_trace(go.Scatter(
                    x=stock_data['Date'], y=stock_data['SMA_50'],
                    name='SMA 50', line=dict(color='blue', width=1)
                ))
            
            fig.update_layout(
                title=f"{selected_stock} - Price Chart",
                height=600,
                hovermode='x unified',
                template='plotly_dark',
                xaxis_rangeslider_visible=False
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Indicators table
            st.subheader("Technical Indicators")
            display_cols = ['Date', 'Close', 'RSI', 'MACD', 'MFI', 'SMA_20', 'SMA_50']
            display_df = stock_data[display_cols].tail(20).copy()
            display_df['Date'] = display_df['Date'].dt.strftime('%Y-%m-%d')
            st.dataframe(display_df, use_container_width=True)
        else:
            st.warning(f"Not enough data for {selected_stock}")

st.success("✓ App loaded successfully!")

"""
DSE Technical Screener - Redesigned Version
Professional layout with stock table, TradingView-like charts, and indicator screener
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

if price_file is None or volume_file is None:
    st.warning("⚠️ Please upload both Price and Volume CSV files")
    st.stop()

def load_data_from_file(df, file_type='price'):
    """Load and parse data with proper date handling"""
    try:
        ticker_col = df.columns[0]
        melted = df.melt(id_vars=[ticker_col], var_name='Date', value_name=file_type.capitalize())
        melted.columns = ['Ticker', 'Date', file_type.capitalize()]
        
        try:
            melted['Date'] = pd.to_datetime(melted['Date'], format='%d-%b', errors='coerce')
            current_year = pd.Timestamp.now().year
            melted['Date'] = melted['Date'].apply(lambda x: x.replace(year=current_year) if pd.notna(x) else x)
        except:
            pass
        
        if melted['Date'].isna().any():
            try:
                melted.loc[melted['Date'].isna(), 'Date'] = pd.to_datetime(
                    melted.loc[melted['Date'].isna(), 'Date'], 
                    format='%d-%b-%y', 
                    errors='coerce'
                )
            except:
                pass
        
        melted = melted.dropna(subset=['Date'])
        melted[file_type.capitalize()] = melted[file_type.capitalize()].astype(str).str.replace(',', '')
        melted[file_type.capitalize()] = pd.to_numeric(melted[file_type.capitalize()], errors='coerce')
        melted = melted.dropna(subset=[file_type.capitalize()])
        melted = melted.sort_values('Date').reset_index(drop=True)
        
        return melted
    except Exception as e:
        st.error(f"Error loading file: {str(e)}")
        return None

# Load data
with st.spinner("Loading data..."):
    price_df = load_data_from_file(pd.read_csv(price_file), 'price')
    volume_df = load_data_from_file(pd.read_csv(volume_file), 'volume')
    
    if price_df is None or volume_df is None:
        st.error("Failed to load data files")
        st.stop()
    
    merged = price_df.merge(volume_df, on=['Ticker', 'Date'], how='inner')
    
    if len(merged) == 0:
        st.error("No matching data between price and volume files")
        st.stop()
    
    merged['Open'] = merged['Price']
    merged['High'] = merged['Price']
    merged['Low'] = merged['Price']
    merged['Close'] = merged['Price']
    
    stocks = merged['Ticker'].unique()
    stock_counts = merged.groupby('Ticker').size()
    valid_stocks = stock_counts[stock_counts >= 10].index.tolist()
    
    if len(valid_stocks) == 0:
        st.error("❌ No valid data to analyze")
        st.stop()

# Calculate indicators
def calculate_rsi(prices, period=14):
    delta = prices.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / (loss + 1e-10)
    return 100 - (100 / (1 + rs))

def calculate_stoch_rsi(prices, period=14, k=3, d=3):
    rsi = calculate_rsi(prices, period)
    lowest_rsi = rsi.rolling(window=period).min()
    highest_rsi = rsi.rolling(window=period).max()
    stoch_rsi = 100 * (rsi - lowest_rsi) / (highest_rsi - lowest_rsi + 1e-10)
    k_line = stoch_rsi.rolling(window=k).mean()
    d_line = k_line.rolling(window=d).mean()
    return stoch_rsi, k_line, d_line

def calculate_macd(prices, fast=12, slow=26, signal=9):
    ema_fast = prices.ewm(span=fast, adjust=False).mean()
    ema_slow = prices.ewm(span=slow, adjust=False).mean()
    macd = ema_fast - ema_slow
    macd_signal = macd.ewm(span=signal, adjust=False).mean()
    histogram = macd - macd_signal
    return macd, macd_signal, histogram

def calculate_bollinger_bands(prices, period=20, std_dev=2.0):
    sma = prices.rolling(window=period).mean()
    std = prices.rolling(window=period).std()
    upper = sma + (std * std_dev)
    lower = sma - (std * std_dev)
    return upper, sma, lower

def calculate_ema(prices, period):
    return prices.ewm(span=period, adjust=False).mean()

def calculate_mfi(high, low, close, volume, period=14):
    typical_price = (high + low + close) / 3
    money_flow = typical_price * volume
    positive_flow = money_flow.where(typical_price > typical_price.shift(), 0)
    negative_flow = money_flow.where(typical_price < typical_price.shift(), 0)
    positive_mf = positive_flow.rolling(window=period).sum()
    negative_mf = negative_flow.rolling(window=period).sum()
    mfi_ratio = positive_mf / (negative_mf + 1e-10)
    return 100 - (100 / (1 + mfi_ratio))

def calculate_atr(high, low, close, period=14):
    high_low = high - low
    high_close = abs(high - close.shift())
    low_close = abs(low - close.shift())
    tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    atr = tr.rolling(window=period).mean()
    return atr

def calculate_adx(high, low, close, period=14):
    plus_dm = high.diff()
    minus_dm = -low.diff()
    plus_dm[plus_dm < 0] = 0
    minus_dm[minus_dm < 0] = 0
    
    tr = pd.concat([
        high - low,
        abs(high - close.shift()),
        abs(low - close.shift())
    ], axis=1).max(axis=1)
    
    atr = tr.rolling(window=period).mean()
    plus_di = 100 * (plus_dm.rolling(window=period).mean() / (atr + 1e-10))
    minus_di = 100 * (minus_dm.rolling(window=period).mean() / (atr + 1e-10))
    
    di_diff = abs(plus_di - minus_di)
    di_sum = plus_di + minus_di
    dx = 100 * (di_diff / (di_sum + 1e-10))
    adx = dx.rolling(window=period).mean()
    
    return adx

def create_tradingview_chart(stock_data, selected_stock, show_bb=True, show_ema=True):
    """Create TradingView-like candlestick chart"""
    fig = go.Figure()
    
    # Candlestick
    fig.add_trace(go.Candlestick(
        x=stock_data['Date'],
        open=stock_data['Open'],
        high=stock_data['High'],
        low=stock_data['Low'],
        close=stock_data['Close'],
        name='OHLC',
        increasing_line_color='#26a69a',
        decreasing_line_color='#ef5350'
    ))
    
    # Bollinger Bands
    if show_bb and 'BB_Upper' in stock_data.columns:
        fig.add_trace(go.Scatter(
            x=stock_data['Date'], y=stock_data['BB_Upper'],
            name='BB Upper', line=dict(color='rgba(100,100,255,0.3)', width=1)
        ))
        fig.add_trace(go.Scatter(
            x=stock_data['Date'], y=stock_data['BB_Lower'],
            name='BB Lower', line=dict(color='rgba(100,100,255,0.3)', width=1),
            fill='tonexty', fillcolor='rgba(100,100,255,0.1)'
        ))
    
    # EMAs
    if show_ema:
        if 'EMA_12' in stock_data.columns:
            fig.add_trace(go.Scatter(
                x=stock_data['Date'], y=stock_data['EMA_12'],
                name='EMA 12', line=dict(color='#ff9800', width=1)
            ))
        if 'EMA_26' in stock_data.columns:
            fig.add_trace(go.Scatter(
                x=stock_data['Date'], y=stock_data['EMA_26'],
                name='EMA 26', line=dict(color='#2196f3', width=1)
            ))
    
    fig.update_layout(
        title=f"{selected_stock} - Price Chart",
        height=600,
        hovermode='x unified',
        template='plotly_dark',
        xaxis_rangeslider_visible=False,
        plot_bgcolor='#1a1a1a',
        paper_bgcolor='#0f0f0f',
        font=dict(color='#ffffff', size=11),
        margin=dict(l=50, r=50, t=80, b=50)
    )
    
    return fig

# Initialize session state
if 'selected_stock_dashboard' not in st.session_state:
    st.session_state.selected_stock_dashboard = None
if 'selected_indicator' not in st.session_state:
    st.session_state.selected_indicator = 'RSI'
if 'selected_stock_screener' not in st.session_state:
    st.session_state.selected_stock_screener = None
if 'screener_filter' not in st.session_state:
    st.session_state.screener_filter = 'BB_Lower'

# Tabs
tab1, tab2 = st.tabs(["📊 Dashboard", "🔍 Indicator Screener"])

with tab1:
    st.header("Stock List with Technical Indicators")
    
    # Calculate all indicators for all stocks
    all_stock_data = {}
    for stock in valid_stocks:
        stock_data = merged[merged['Ticker'] == stock].sort_values('Date').copy()
        stock_data['RSI'] = calculate_rsi(stock_data['Close'])
        stock_data['MACD'], stock_data['MACD_Signal'], stock_data['MACD_Histogram'] = calculate_macd(stock_data['Close'])
        stock_data['BB_Upper'], stock_data['BB_Middle'], stock_data['BB_Lower'] = calculate_bollinger_bands(stock_data['Close'])
        stock_data['EMA_12'] = calculate_ema(stock_data['Close'], 12)
        stock_data['EMA_26'] = calculate_ema(stock_data['Close'], 26)
        stock_data['MFI'] = calculate_mfi(stock_data['High'], stock_data['Low'], stock_data['Close'], stock_data['Volume'])
        stock_data['ATR'] = calculate_atr(stock_data['High'], stock_data['Low'], stock_data['Close'])
        stock_data['ADX'] = calculate_adx(stock_data['High'], stock_data['Low'], stock_data['Close'])
        stock_data['Stoch_RSI'], stock_data['Stoch_K'], stock_data['Stoch_D'] = calculate_stoch_rsi(stock_data['Close'])
        all_stock_data[stock] = stock_data
    
    # Create summary table
    summary_data = []
    for stock in valid_stocks:
        latest = all_stock_data[stock].iloc[-1]
        summary_data.append({
            'Stock': stock,
            'Price': f"{latest['Close']:.2f}",
            'RSI': f"{latest['RSI']:.1f}",
            'Stoch RSI': f"{latest['Stoch_RSI']:.1f}",
            'MACD': f"{latest['MACD']:.4f}",
            'MFI': f"{latest['MFI']:.1f}",
            'ATR': f"{latest['ATR']:.2f}",
            'ADX': f"{latest['ADX']:.1f}",
            'BB_Upper': f"{latest['BB_Upper']:.2f}",
            'BB_Lower': f"{latest['BB_Lower']:.2f}",
            'EMA_12': f"{latest['EMA_12']:.2f}",
            'EMA_26': f"{latest['EMA_26']:.2f}"
        })
    
    summary_df = pd.DataFrame(summary_data)
    
    # Display clickable table
    st.subheader("Click on a stock to view detailed chart")
    
    # Create columns for stock buttons
    cols = st.columns(5)
    for idx, stock in enumerate(sorted(valid_stocks)):
        col_idx = idx % 5
        with cols[col_idx]:
            if st.button(stock, key=f"stock_btn_{stock}", use_container_width=True):
                st.session_state.selected_stock_dashboard = stock
    
    # Display table
    st.dataframe(summary_df, use_container_width=True, hide_index=True)
    
    # Show selected stock chart
    if st.session_state.selected_stock_dashboard:
        st.divider()
        st.subheader(f"Chart: {st.session_state.selected_stock_dashboard}")
        
        stock_data = all_stock_data[st.session_state.selected_stock_dashboard]
        
        # Candlestick chart with BB and EMA
        fig = create_tradingview_chart(stock_data, st.session_state.selected_stock_dashboard, show_bb=True, show_ema=True)
        st.plotly_chart(fig, use_container_width=True)
        
        # Indicator selector
        st.subheader("Technical Indicators")
        selected_indicator = st.selectbox(
            "Select Indicator",
            ['RSI', 'Stochastic RSI', 'MACD', 'MFI', 'ATR', 'ADX'],
            key='indicator_selector'
        )
        
        # Create indicator chart
        fig_indicator = make_subplots(specs=[[{"secondary_y": False}]])
        
        if selected_indicator == 'RSI':
            fig_indicator.add_trace(go.Scatter(
                x=stock_data['Date'], y=stock_data['RSI'],
                name='RSI', line=dict(color='#ff9800', width=2)
            ))
            fig_indicator.add_hline(y=70, line_dash="dash", line_color="red", annotation_text="Overbought")
            fig_indicator.add_hline(y=30, line_dash="dash", line_color="green", annotation_text="Oversold")
            fig_indicator.update_yaxes(range=[0, 100])
        
        elif selected_indicator == 'Stochastic RSI':
            fig_indicator.add_trace(go.Scatter(
                x=stock_data['Date'], y=stock_data['Stoch_K'],
                name='%K', line=dict(color='#2196f3', width=2)
            ))
            fig_indicator.add_trace(go.Scatter(
                x=stock_data['Date'], y=stock_data['Stoch_D'],
                name='%D', line=dict(color='#ff9800', width=2)
            ))
            fig_indicator.add_hline(y=80, line_dash="dash", line_color="red")
            fig_indicator.add_hline(y=20, line_dash="dash", line_color="green")
            fig_indicator.update_yaxes(range=[0, 100])
        
        elif selected_indicator == 'MACD':
            fig_indicator.add_trace(go.Scatter(
                x=stock_data['Date'], y=stock_data['MACD'],
                name='MACD', line=dict(color='#2196f3', width=2)
            ))
            fig_indicator.add_trace(go.Scatter(
                x=stock_data['Date'], y=stock_data['MACD_Signal'],
                name='Signal', line=dict(color='#ff9800', width=2)
            ))
            colors = ['#26a69a' if h > 0 else '#ef5350' for h in stock_data['MACD_Histogram']]
            fig_indicator.add_trace(go.Bar(
                x=stock_data['Date'], y=stock_data['MACD_Histogram'],
                name='Histogram', marker=dict(color=colors)
            ))
            fig_indicator.add_hline(y=0, line_dash="dash", line_color="gray")
        
        elif selected_indicator == 'MFI':
            fig_indicator.add_trace(go.Scatter(
                x=stock_data['Date'], y=stock_data['MFI'],
                name='MFI', line=dict(color='#4caf50', width=2)
            ))
            fig_indicator.add_hline(y=80, line_dash="dash", line_color="red")
            fig_indicator.add_hline(y=20, line_dash="dash", line_color="green")
            fig_indicator.update_yaxes(range=[0, 100])
        
        elif selected_indicator == 'ATR':
            fig_indicator.add_trace(go.Scatter(
                x=stock_data['Date'], y=stock_data['ATR'],
                name='ATR', line=dict(color='#9c27b0', width=2)
            ))
        
        elif selected_indicator == 'ADX':
            fig_indicator.add_trace(go.Scatter(
                x=stock_data['Date'], y=stock_data['ADX'],
                name='ADX', line=dict(color='#f44336', width=2)
            ))
            fig_indicator.add_hline(y=25, line_dash="dash", line_color="orange", annotation_text="Trend Strength")
        
        fig_indicator.update_layout(
            title=f"{selected_indicator} - {st.session_state.selected_stock_dashboard}",
            height=400,
            hovermode='x unified',
            template='plotly_dark',
            plot_bgcolor='#1a1a1a',
            paper_bgcolor='#0f0f0f',
            font=dict(color='#ffffff'),
            margin=dict(l=50, r=50, t=80, b=50)
        )
        
        st.plotly_chart(fig_indicator, use_container_width=True)

with tab2:
    st.header("Indicator-Based Screener")
    
    # Screener filters
    col1, col2, col3 = st.columns(3)
    
    with col1:
        screener_type = st.selectbox(
            "Select Screener",
            [
                'Bollinger Bands - Below Lower Band',
                'RSI - Oversold (< 30)',
                'RSI - Overbought (> 70)',
                'Stochastic RSI - Oversold',
                'Stochastic RSI - Overbought',
                'MACD - Bullish Crossover',
                'MFI - Oversold (< 20)',
                'MFI - Overbought (> 80)',
                'ADX - Strong Trend (> 25)'
            ]
        )
    
    # Calculate screener results
    screener_results = []
    
    for stock in valid_stocks:
        stock_data = all_stock_data[stock]
        latest = stock_data.iloc[-1]
        
        match = False
        reason = ""
        
        if screener_type == 'Bollinger Bands - Below Lower Band':
            if latest['Close'] < latest['BB_Lower']:
                match = True
                reason = f"Price: {latest['Close']:.2f} < BB Lower: {latest['BB_Lower']:.2f}"
        
        elif screener_type == 'RSI - Oversold (< 30)':
            if latest['RSI'] < 30:
                match = True
                reason = f"RSI: {latest['RSI']:.1f}"
        
        elif screener_type == 'RSI - Overbought (> 70)':
            if latest['RSI'] > 70:
                match = True
                reason = f"RSI: {latest['RSI']:.1f}"
        
        elif screener_type == 'Stochastic RSI - Oversold':
            if latest['Stoch_RSI'] < 20:
                match = True
                reason = f"Stoch RSI: {latest['Stoch_RSI']:.1f}"
        
        elif screener_type == 'Stochastic RSI - Overbought':
            if latest['Stoch_RSI'] > 80:
                match = True
                reason = f"Stoch RSI: {latest['Stoch_RSI']:.1f}"
        
        elif screener_type == 'MACD - Bullish Crossover':
            if latest['MACD'] > latest['MACD_Signal']:
                match = True
                reason = f"MACD: {latest['MACD']:.4f} > Signal: {latest['MACD_Signal']:.4f}"
        
        elif screener_type == 'MFI - Oversold (< 20)':
            if latest['MFI'] < 20:
                match = True
                reason = f"MFI: {latest['MFI']:.1f}"
        
        elif screener_type == 'MFI - Overbought (> 80)':
            if latest['MFI'] > 80:
                match = True
                reason = f"MFI: {latest['MFI']:.1f}"
        
        elif screener_type == 'ADX - Strong Trend (> 25)':
            if latest['ADX'] > 25:
                match = True
                reason = f"ADX: {latest['ADX']:.1f}"
        
        if match:
            screener_results.append({
                'Stock': stock,
                'Price': f"{latest['Close']:.2f}",
                'Reason': reason
            })
    
    # Display screener results
    if screener_results:
        results_df = pd.DataFrame(screener_results)
        st.subheader(f"Stocks Matching: {screener_type}")
        st.write(f"Found {len(screener_results)} stocks")
        
        # Display as clickable buttons
        cols = st.columns(5)
        for idx, row in enumerate(results_df.iterrows()):
            col_idx = idx % 5
            stock = row[1]['Stock']
            with cols[col_idx]:
                if st.button(stock, key=f"screener_stock_{stock}", use_container_width=True):
                    st.session_state.selected_stock_screener = stock
        
        # Display table
        st.dataframe(results_df, use_container_width=True, hide_index=True)
        
        # Show selected stock from screener
        if st.session_state.selected_stock_screener:
            st.divider()
            st.subheader(f"Chart: {st.session_state.selected_stock_screener}")
            
            stock_data = all_stock_data[st.session_state.selected_stock_screener]
            fig = create_tradingview_chart(stock_data, st.session_state.selected_stock_screener, show_bb=True, show_ema=True)
            st.plotly_chart(fig, use_container_width=True)
    else:
        st.info(f"No stocks found matching: {screener_type}")

st.success("✓ App loaded successfully!")

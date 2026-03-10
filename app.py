"""
DSE Technical Screener - Streamlit Application

Professional interactive stock screener with technical indicators,
anti-manipulation detection, and multi-timeframe analysis.
"""

import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import io
from indicators import TechnicalIndicators
from charting import CandlestickChart, create_comparison_chart


# Page configuration
st.set_page_config(
    page_title="DSE Technical Screener",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .metric-card {
        background-color: #1f1f1f;
        padding: 20px;
        border-radius: 10px;
        border-left: 4px solid #00d4ff;
        margin: 10px 0;
    }
    .buy-signal {
        color: #00ff00;
        font-weight: bold;
    }
    .sell-signal {
        color: #ff0000;
        font-weight: bold;
    }
    .hold-signal {
        color: #ffaa00;
        font-weight: bold;
    }
    .warning-box {
        background-color: #332200;
        padding: 15px;
        border-radius: 5px;
        border-left: 4px solid #ff6600;
        margin: 10px 0;
    }
</style>
""", unsafe_allow_html=True)


def load_data_from_file(uploaded_file, file_type='price'):
    """Load and parse TSV/CSV file with flexible date handling"""
    try:
        # Read file
        df = pd.read_csv(uploaded_file, sep='\t')
        
        # Check if first row contains dates (header row)
        first_col = df.columns[0]
        first_row_values = df.iloc[0].values
        
        # Try to detect if first row is a header with dates
        is_header_row = False
        if len(first_row_values) > 0:
            try:
                # Try parsing first non-ticker value as date
                pd.to_datetime(str(first_row_values[1]))
                is_header_row = True
            except:
                is_header_row = False
        
        # If first row is dates, use it as column names
        if is_header_row and len(df) > 1:
            # Reconstruct: first row becomes column names
            new_cols = [df.iloc[0, 0]] + list(df.iloc[0, 1:])
            df.columns = new_cols
            df = df.iloc[1:].reset_index(drop=True)
        
        # Reshape from wide format (stocks as columns) to long format
        ticker_col = df.columns[0]
        date_cols = df.columns[1:]
        
        # Melt the dataframe
        melted = df.melt(id_vars=[ticker_col], var_name='Date', value_name=file_type.capitalize())
        melted.columns = ['Ticker', 'Date', file_type.capitalize()]
        
        # Convert date - handle various formats
        try:
            melted['Date'] = pd.to_datetime(melted['Date'])
        except:
            st.warning(f"Could not parse some dates. Attempting alternative format...")
            melted['Date'] = pd.to_datetime(melted['Date'], errors='coerce')
        
        # Remove rows with invalid dates
        melted = melted.dropna(subset=['Date'])
        
        # Convert price/volume to numeric
        melted[file_type.capitalize()] = pd.to_numeric(melted[file_type.capitalize()], errors='coerce')
        melted = melted.dropna(subset=[file_type.capitalize()])
        
        # IMPORTANT: Sort by date to ensure chronological order (oldest first)
        # This handles both forward and reverse chronological data
        melted = melted.sort_values('Date').reset_index(drop=True)
        
        return melted
    except Exception as e:
        st.error(f"Error loading file: {str(e)}")
        return None


def process_stock_data(price_df, volume_df):
    """Combine price and volume data, calculate indicators"""
    try:
        # Merge price and volume
        merged = price_df.merge(volume_df, on=['Ticker', 'Date'], how='inner')
        merged = merged.sort_values(['Ticker', 'Date'])
        
        # For each stock, we need OHLC data
        # If only Close price is available, use it for all OHLC
        if 'Price' in merged.columns:
            merged['Open'] = merged['Price']
            merged['High'] = merged['Price']
            merged['Low'] = merged['Price']
            merged['Close'] = merged['Price']
        
        return merged
    except Exception as e:
        st.error(f"Error processing data: {str(e)}")
        return None


def get_timeframe_data(df, timeframe='D'):
    """Resample data to different timeframes"""
    if timeframe == 'D':
        return df  # Daily
    elif timeframe == 'W':
        # Weekly
        return df.set_index('Date').resample('W').agg({
            'Open': 'first',
            'High': 'max',
            'Low': 'min',
            'Close': 'last',
            'Volume': 'sum'
        }).reset_index()
    elif timeframe == 'M':
        # Monthly
        return df.set_index('Date').resample('M').agg({
            'Open': 'first',
            'High': 'max',
            'Low': 'min',
            'Close': 'last',
            'Volume': 'sum'
        }).reset_index()
    return df


def main():
    st.title("📈 DSE Technical Screener")
    st.markdown("Professional stock analysis with anti-manipulation detection")
    
    # Sidebar
    with st.sidebar:
        st.header("📊 Configuration")
        
        # File upload
        st.subheader("1. Upload Data Files")
        price_file = st.file_uploader("Upload Price Data (TSV/CSV)", type=['tsv', 'csv'], key='price')
        volume_file = st.file_uploader("Upload Volume Data (TSV/CSV)", type=['tsv', 'csv'], key='volume')
        
        if price_file and volume_file:
            st.success("✓ Files uploaded")
        else:
            st.warning("⚠️ Please upload both files")
            return
        
        # Timeframe selection
        st.subheader("2. Select Timeframe")
        timeframe = st.radio("Timeframe", ['Daily', 'Weekly', 'Monthly'], 
                            format_func=lambda x: {'Daily': 'Daily (D)', 'Weekly': 'Weekly (W)', 'Monthly': 'Monthly (M)'}[x])
        timeframe_map = {'Daily': 'D', 'Weekly': 'W', 'Monthly': 'M'}
        
        # Indicators selection
        st.subheader("3. Select Indicators")
        show_sma = st.checkbox("SMA (20, 50, 200)", value=True)
        show_bb = st.checkbox("Bollinger Bands", value=True)
        show_rsi = st.checkbox("RSI", value=True)
        show_stoch = st.checkbox("Stochastic RSI", value=True)
        show_macd = st.checkbox("MACD", value=True)
        show_mfi = st.checkbox("MFI", value=True)
        
        # Filtering options
        st.subheader("4. Filters")
        signal_filter = st.selectbox("Filter by Signal", ["All", "BUY", "SELL", "HOLD"])
        min_confidence = st.slider("Minimum Confidence", -100, 100, 0)
        min_liquidity = st.slider("Minimum Liquidity Score", 0, 100, 30)
    
    # Load and process data
    price_df = load_data_from_file(price_file, 'price')
    volume_df = load_data_from_file(volume_file, 'volume')
    
    if price_df is None or volume_df is None:
        return
    
    # Process data
    with st.spinner("Processing data..."):
        merged_data = process_stock_data(price_df, volume_df)
        
        if merged_data is None:
            return
        
        # Calculate indicators for all stocks
        all_results = []
        stocks = merged_data['Ticker'].unique()
        
        progress_bar = st.progress(0)
        for idx, stock in enumerate(stocks):
            stock_data = merged_data[merged_data['Ticker'] == stock].copy()
            
            # Resample to timeframe
            stock_data = get_timeframe_data(stock_data, timeframe_map[timeframe])
            
            if len(stock_data) < 20:
                continue
            
            # Calculate indicators
            indicators = TechnicalIndicators(stock_data)
            result_df = indicators.calculate_all()
            result_df['Ticker'] = stock
            all_results.append(result_df)
            
            progress_bar.progress((idx + 1) / len(stocks))
        
        if not all_results:
            st.error("No valid data to analyze")
            return
        
        full_results = pd.concat(all_results, ignore_index=True)
    
    # Create tabs
    tab1, tab2, tab3 = st.tabs(["📊 Dashboard", "🔍 Stock Explorer", "📈 Detailed Analysis"])
    
    # TAB 1: DASHBOARD
    with tab1:
        st.header("Screener Dashboard")
        
        # Get latest data for each stock
        latest_data = full_results.sort_values('Date').groupby('Ticker').tail(1)
        
        # Apply filters
        filtered_data = latest_data.copy()
        
        if signal_filter != "All":
            filtered_data = filtered_data[filtered_data['Signal'] == signal_filter]
        
        filtered_data = filtered_data[filtered_data['Confidence'] >= min_confidence]
        filtered_data = filtered_data[filtered_data['Liquidity_Score'] >= min_liquidity]
        
        # Summary metrics
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Total Stocks Analyzed", len(latest_data))
        
        with col2:
            buy_count = len(latest_data[latest_data['Signal'] == 'BUY'])
            st.metric("BUY Signals", buy_count, delta=f"{(buy_count/len(latest_data)*100):.1f}%")
        
        with col3:
            sell_count = len(latest_data[latest_data['Signal'] == 'SELL'])
            st.metric("SELL Signals", sell_count, delta=f"{(sell_count/len(latest_data)*100):.1f}%")
        
        with col4:
            hold_count = len(latest_data[latest_data['Signal'] == 'HOLD'])
            st.metric("HOLD Signals", hold_count, delta=f"{(hold_count/len(latest_data)*100):.1f}%")
        
        st.divider()
        
        # Top recommendations
        st.subheader("🎯 Top Recommendations")
        
        top_buy = latest_data[latest_data['Signal'] == 'BUY'].nlargest(10, 'Confidence')
        
        if len(top_buy) > 0:
            display_cols = ['Ticker', 'Close', 'RSI', 'MACD', 'MFI', 'Signal', 'Confidence', 'Liquidity_Score']
            available_cols = [col for col in display_cols if col in top_buy.columns]
            
            st.dataframe(
                top_buy[available_cols].style.format({
                    'Close': '{:.2f}',
                    'RSI': '{:.1f}',
                    'MACD': '{:.4f}',
                    'MFI': '{:.1f}',
                    'Confidence': '{:.0f}',
                    'Liquidity_Score': '{:.1f}'
                }),
                use_container_width=True
            )
        else:
            st.info("No BUY signals found with current filters")
        
        st.divider()
        
        # Signal distribution
        st.subheader("📊 Signal Distribution")
        
        col1, col2 = st.columns(2)
        
        with col1:
            signal_counts = latest_data['Signal'].value_counts()
            st.bar_chart(signal_counts)
        
        with col2:
            # Confidence distribution
            confidence_data = latest_data[latest_data['Signal'] == 'BUY']['Confidence']
            if len(confidence_data) > 0:
                st.histogram(confidence_data, bins=20, title="BUY Signal Confidence Distribution")
    
    # TAB 2: STOCK EXPLORER
    with tab2:
        st.header("Stock Explorer")
        
        # Get latest data
        latest_data = full_results.sort_values('Date').groupby('Ticker').tail(1)
        
        # Stock selector
        selected_stock = st.selectbox("Select Stock", sorted(latest_data['Ticker'].unique()))
        
        if selected_stock:
            stock_data = full_results[full_results['Ticker'] == selected_stock].sort_values('Date')
            latest = stock_data.iloc[-1]
            
            # Display stock info
            col1, col2, col3, col4, col5 = st.columns(5)
            
            with col1:
                st.metric("Current Price", f"{latest['Close']:.2f}")
            
            with col2:
                signal_color = "🟢" if latest['Signal'] == 'BUY' else "🔴" if latest['Signal'] == 'SELL' else "🟡"
                st.metric("Signal", f"{signal_color} {latest['Signal']}")
            
            with col3:
                st.metric("Confidence", f"{latest['Confidence']:.0f}")
            
            with col4:
                st.metric("RSI", f"{latest['RSI']:.1f}")
            
            with col5:
                st.metric("Liquidity", f"{latest['Liquidity_Score']:.1f}")
            
            st.divider()
            
            # Candlestick chart
            st.subheader("Price Chart")
            
            indicators_to_show = []
            if show_sma:
                indicators_to_show.extend(['SMA_20', 'SMA_50', 'SMA_200'])
            if show_bb:
                indicators_to_show.append('BB')
            if show_rsi:
                indicators_to_show.append('RSI')
            if show_stoch:
                indicators_to_show.append('Stoch_RSI')
            if show_macd:
                indicators_to_show.append('MACD')
            if show_mfi:
                indicators_to_show.append('MFI')
            
            chart = CandlestickChart(stock_data, f"{selected_stock} - {timeframe}")
            fig = chart.create_chart(indicators=indicators_to_show, height=700)
            st.plotly_chart(fig, use_container_width=True)
            
            st.divider()
            
            # Detailed indicators table
            st.subheader("Technical Indicators")
            
            indicator_cols = ['Date', 'Close', 'RSI', 'Stoch_RSI', 'MACD', 'MFI', 'ATR', 'ADX', 'Liquidity_Score', 'Signal', 'Confidence']
            available_indicator_cols = [col for col in indicator_cols if col in stock_data.columns]
            
            display_df = stock_data[available_indicator_cols].tail(20).copy()
            display_df['Date'] = display_df['Date'].dt.strftime('%Y-%m-%d')
            
            st.dataframe(
                display_df.style.format({
                    'Close': '{:.2f}',
                    'RSI': '{:.1f}',
                    'Stoch_RSI': '{:.1f}',
                    'MACD': '{:.4f}',
                    'MFI': '{:.1f}',
                    'ATR': '{:.2f}',
                    'ADX': '{:.1f}',
                    'Liquidity_Score': '{:.1f}',
                    'Confidence': '{:.0f}'
                }),
                use_container_width=True
            )
            
            st.divider()
            
            # Signal reasons
            st.subheader("📋 Signal Analysis")
            
            latest_reason = latest['Signal_Reason']
            if latest_reason:
                st.info(latest_reason)
            
            # Anti-manipulation warnings
            st.subheader("⚠️ Risk Indicators")
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                if latest.get('Volume_Price_Divergence', False):
                    st.warning("🔴 Volume anomaly detected")
                else:
                    st.success("✓ Normal volume")
            
            with col2:
                if latest.get('Low_Liquidity_Trap', False):
                    st.warning("🔴 Low liquidity trap risk")
                else:
                    st.success("✓ Good liquidity")
            
            with col3:
                if latest.get('Fake_Breakout', False):
                    st.warning("🔴 Potential fake breakout")
                else:
                    st.success("✓ No fake breakout detected")
    
    # TAB 3: DETAILED ANALYSIS
    with tab3:
        st.header("Detailed Analysis")
        
        # Get latest data
        latest_data = full_results.sort_values('Date').groupby('Ticker').tail(1)
        
        # Full data table
        st.subheader("All Stocks - Latest Data")
        
        display_cols = ['Ticker', 'Close', 'RSI', 'MACD', 'MFI', 'Signal', 'Confidence', 'Liquidity_Score', 'Volume_Ratio']
        available_cols = [col for col in display_cols if col in latest_data.columns]
        
        display_df = latest_data[available_cols].sort_values('Confidence', ascending=False)
        
        st.dataframe(
            display_df.style.format({
                'Close': '{:.2f}',
                'RSI': '{:.1f}',
                'MACD': '{:.4f}',
                'MFI': '{:.1f}',
                'Confidence': '{:.0f}',
                'Liquidity_Score': '{:.1f}',
                'Volume_Ratio': '{:.2f}'
            }),
            use_container_width=True
        )
        
        # Export data
        st.divider()
        st.subheader("📥 Export Data")
        
        csv = display_df.to_csv(index=False)
        st.download_button(
            label="Download CSV",
            data=csv,
            file_name=f"dse_screener_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
            mime="text/csv"
        )


if __name__ == "__main__":
    main()

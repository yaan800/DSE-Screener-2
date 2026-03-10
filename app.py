"""
DSE Technical Screener - Complete Single File Application
Professional interactive stock screener with technical indicators and anti-manipulation detection
"""

import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime
import plotly.graph_objects as go
from plotly.subplots import make_subplots


# ============================================================================
# TECHNICAL INDICATORS MODULE
# ============================================================================

class TechnicalIndicators:
    """Calculate technical indicators and generate trading signals"""
    
    def __init__(self, df):
        self.df = df.copy().sort_values('Date').reset_index(drop=True)
        
    def calculate_all(self):
        """Calculate all indicators and return enhanced dataframe"""
        self._calculate_rsi()
        self._calculate_stochastic_rsi()
        self._calculate_macd()
        self._calculate_bollinger_bands()
        self._calculate_moving_averages()
        self._calculate_atr()
        self._calculate_mfi()
        self._calculate_obv()
        self._calculate_adx()
        self._detect_volume_anomalies()
        self._detect_price_manipulation()
        self._detect_divergences()
        self._calculate_liquidity_score()
        self._generate_trading_signals()
        return self.df
    
    def _calculate_rsi(self, period=14):
        delta = self.df['Close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / (loss + 1e-10)
        self.df['RSI'] = 100 - (100 / (1 + rs))
        self.df['RSI'] = self.df['RSI'].fillna(50)
    
    def _calculate_stochastic_rsi(self, period=14, smooth_k=3, smooth_d=3):
        rsi = self.df['RSI']
        lowest_rsi = rsi.rolling(window=period).min()
        highest_rsi = rsi.rolling(window=period).max()
        stoch_rsi = 100 * (rsi - lowest_rsi) / (highest_rsi - lowest_rsi + 1e-10)
        self.df['Stoch_RSI'] = stoch_rsi.fillna(50)
        self.df['Stoch_K'] = stoch_rsi.rolling(window=smooth_k).mean().fillna(50)
        self.df['Stoch_D'] = self.df['Stoch_K'].rolling(window=smooth_d).mean().fillna(50)
    
    def _calculate_macd(self, fast=12, slow=26, signal=9):
        ema_fast = self.df['Close'].ewm(span=fast, adjust=False).mean()
        ema_slow = self.df['Close'].ewm(span=slow, adjust=False).mean()
        self.df['MACD'] = ema_fast - ema_slow
        self.df['MACD_Signal'] = self.df['MACD'].ewm(span=signal, adjust=False).mean()
        self.df['MACD_Histogram'] = self.df['MACD'] - self.df['MACD_Signal']
    
    def _calculate_bollinger_bands(self, period=20, std_dev=2.0):
        sma = self.df['Close'].rolling(window=period).mean()
        std = self.df['Close'].rolling(window=period).std()
        self.df['BB_Upper'] = sma + (std * std_dev)
        self.df['BB_Middle'] = sma
        self.df['BB_Lower'] = sma - (std * std_dev)
        self.df['BB_Width'] = self.df['BB_Upper'] - self.df['BB_Lower']
        self.df['BB_Position'] = (self.df['Close'] - self.df['BB_Lower']) / (self.df['BB_Width'] + 1e-10)
    
    def _calculate_moving_averages(self):
        self.df['SMA_20'] = self.df['Close'].rolling(window=20).mean()
        self.df['SMA_50'] = self.df['Close'].rolling(window=50).mean()
        self.df['SMA_200'] = self.df['Close'].rolling(window=200).mean()
        self.df['EMA_12'] = self.df['Close'].ewm(span=12, adjust=False).mean()
        self.df['EMA_26'] = self.df['Close'].ewm(span=26, adjust=False).mean()
    
    def _calculate_atr(self, period=14):
        high_low = self.df['High'] - self.df['Low']
        high_close = abs(self.df['High'] - self.df['Close'].shift())
        low_close = abs(self.df['Low'] - self.df['Close'].shift())
        tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
        self.df['ATR'] = tr.rolling(window=period).mean()
        self.df['ATR_Percent'] = (self.df['ATR'] / self.df['Close']) * 100
    
    def _calculate_mfi(self, period=14):
        typical_price = (self.df['High'] + self.df['Low'] + self.df['Close']) / 3
        money_flow = typical_price * self.df['Volume']
        positive_flow = money_flow.where(typical_price > typical_price.shift(), 0)
        negative_flow = money_flow.where(typical_price < typical_price.shift(), 0)
        positive_mf = positive_flow.rolling(window=period).sum()
        negative_mf = negative_flow.rolling(window=period).sum()
        mfi_ratio = positive_mf / (negative_mf + 1e-10)
        self.df['MFI'] = 100 - (100 / (1 + mfi_ratio))
        self.df['MFI'] = self.df['MFI'].fillna(50)
    
    def _calculate_obv(self):
        obv = (np.sign(self.df['Close'].diff()) * self.df['Volume']).fillna(0).cumsum()
        self.df['OBV'] = obv
        self.df['OBV_EMA'] = obv.ewm(span=20, adjust=False).mean()
    
    def _calculate_adx(self, period=14):
        high_diff = self.df['High'].diff()
        low_diff = -self.df['Low'].diff()
        plus_dm = high_diff.where((high_diff > low_diff) & (high_diff > 0), 0)
        minus_dm = low_diff.where((low_diff > high_diff) & (low_diff > 0), 0)
        tr = self._calculate_true_range()
        plus_di = 100 * (plus_dm.rolling(window=period).mean() / (tr.rolling(window=period).mean() + 1e-10))
        minus_di = 100 * (minus_dm.rolling(window=period).mean() / (tr.rolling(window=period).mean() + 1e-10))
        di_diff = abs(plus_di - minus_di)
        di_sum = plus_di + minus_di
        dx = 100 * (di_diff / (di_sum + 1e-10))
        self.df['ADX'] = dx.rolling(window=period).mean()
        self.df['Plus_DI'] = plus_di
        self.df['Minus_DI'] = minus_di
    
    def _calculate_true_range(self):
        high_low = self.df['High'] - self.df['Low']
        high_close = abs(self.df['High'] - self.df['Close'].shift())
        low_close = abs(self.df['Low'] - self.df['Close'].shift())
        return pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    
    def _detect_volume_anomalies(self):
        avg_volume = self.df['Volume'].rolling(window=20).mean()
        volume_ratio = self.df['Volume'] / (avg_volume + 1e-10)
        self.df['Volume_Anomaly'] = volume_ratio > 2.5
        self.df['Volume_Ratio'] = volume_ratio
        price_change_pct = abs(self.df['Close'].pct_change()) * 100
        self.df['Volume_Price_Divergence'] = (self.df['Volume_Anomaly']) & (price_change_pct < 2)
    
    def _detect_price_manipulation(self):
        self.df['Price_Spike'] = abs(self.df['Close'].pct_change()) > 5
        self.df['Reversal_Next'] = self.df['Price_Spike'].shift(-1) & \
                                   (self.df['Close'].pct_change().shift(-1) * self.df['Close'].pct_change() < 0)
        avg_vol = self.df['Volume'].rolling(window=20).mean()
        price_move = abs(self.df['Close'].pct_change()) * 100
        self.df['Low_Liquidity_Trap'] = (price_move > 3) & (self.df['Volume'] < avg_vol * 0.5)
        self.df['Above_BB_Upper'] = self.df['Close'] > self.df['BB_Upper']
        self.df['Fake_Breakout'] = self.df['Above_BB_Upper'] & \
                                   (self.df['Close'].shift(-1) < self.df['BB_Middle'])
    
    def _detect_divergences(self):
        self.df['RSI_Divergence'] = False
        if len(self.df) > 20:
            price_higher = self.df['Close'] > self.df['Close'].shift(20)
            rsi_lower = self.df['RSI'] < self.df['RSI'].shift(20)
            self.df['RSI_Divergence'] = price_higher & rsi_lower
        
        self.df['MACD_Divergence'] = False
        if len(self.df) > 20:
            price_higher = self.df['Close'] > self.df['Close'].shift(20)
            macd_lower = self.df['MACD'] < self.df['MACD'].shift(20)
            self.df['MACD_Divergence'] = price_higher & macd_lower
    
    def _calculate_liquidity_score(self):
        avg_vol = self.df['Volume'].rolling(window=20).mean()
        vol_score = (self.df['Volume'] / (avg_vol + 1e-10)).clip(0, 5) * 20
        spread = ((self.df['High'] - self.df['Low']) / self.df['Close']) * 100
        spread_score = (1 - (spread / spread.max()).clip(0, 1)) * 30
        volatility = self.df['Close'].pct_change().rolling(window=20).std() * 100
        vol_stability_score = (1 - (volatility / volatility.max()).clip(0, 1)) * 50
        self.df['Liquidity_Score'] = vol_score + spread_score + vol_stability_score
        self.df['Liquidity_Score'] = self.df['Liquidity_Score'].clip(0, 100)
    
    def _generate_trading_signals(self):
        signals = []
        confidence = []
        reasons = []
        
        for idx in range(len(self.df)):
            row = self.df.iloc[idx]
            signal = "HOLD"
            conf = 0
            reason = []
            
            if row.get('Volume_Price_Divergence', False):
                signal = "HOLD"
                conf = -50
                reason.append("⚠️ Volume anomaly")
            elif row.get('Low_Liquidity_Trap', False):
                signal = "HOLD"
                conf = -40
                reason.append("⚠️ Low liquidity trap")
            elif row.get('Fake_Breakout', False):
                signal = "HOLD"
                conf = -30
                reason.append("⚠️ Fake breakout")
            else:
                buy_score = 0
                
                if row['RSI'] < 30:
                    buy_score += 20
                    reason.append("✓ RSI oversold")
                elif row['RSI'] < 40:
                    buy_score += 10
                
                if row['Stoch_RSI'] < 20 and row['Stoch_K'] < row['Stoch_D']:
                    buy_score += 15
                    reason.append("✓ Stoch oversold")
                
                if row['MACD'] > row['MACD_Signal'] and row['MACD_Histogram'] > 0:
                    buy_score += 20
                    reason.append("✓ MACD bullish")
                elif row['MACD'] > row['MACD_Signal']:
                    buy_score += 10
                
                if row['MFI'] < 30:
                    buy_score += 15
                    reason.append("✓ MFI oversold")
                
                if row['Close'] < row['BB_Lower']:
                    buy_score += 15
                    reason.append("✓ Below BB Lower")
                elif row['Close'] < row['BB_Middle']:
                    buy_score += 5
                
                if row['Close'] > row['SMA_20'] > row['SMA_50']:
                    buy_score += 10
                    reason.append("✓ Bullish MA")
                
                if row['OBV'] > row['OBV_EMA']:
                    buy_score += 5
                
                if row['ADX'] > 25:
                    buy_score += 5
                    reason.append("✓ Strong trend")
                
                if row.get('RSI_Divergence', False):
                    buy_score -= 10
                
                if buy_score >= 60:
                    signal = "BUY"
                    conf = min(buy_score, 100)
                elif buy_score >= 40:
                    signal = "BUY"
                    conf = buy_score
                elif buy_score >= 20:
                    signal = "HOLD"
                    conf = buy_score
                else:
                    sell_score = 0
                    reason = []
                    
                    if row['RSI'] > 70:
                        sell_score += 20
                        reason.append("✓ RSI overbought")
                    elif row['RSI'] > 60:
                        sell_score += 10
                    
                    if row['Stoch_RSI'] > 80 and row['Stoch_K'] > row['Stoch_D']:
                        sell_score += 15
                        reason.append("✓ Stoch overbought")
                    
                    if row['MACD'] < row['MACD_Signal'] and row['MACD_Histogram'] < 0:
                        sell_score += 20
                        reason.append("✓ MACD bearish")
                    
                    if row['MFI'] > 70:
                        sell_score += 15
                        reason.append("✓ MFI overbought")
                    
                    if row['Close'] > row['BB_Upper']:
                        sell_score += 15
                        reason.append("✓ Above BB Upper")
                    
                    if sell_score >= 40:
                        signal = "SELL"
                        conf = -min(sell_score, 100)
                    else:
                        signal = "HOLD"
                        conf = 0
            
            signals.append(signal)
            confidence.append(conf)
            reasons.append(" | ".join(reason) if reason else "No clear signal")
        
        self.df['Signal'] = signals
        self.df['Confidence'] = confidence
        self.df['Signal_Reason'] = reasons


# ============================================================================
# CHARTING MODULE
# ============================================================================

def create_candlestick_chart(df, title="Stock Price", indicators=None, height=600):
    """Create candlestick chart with indicators"""
    if indicators is None:
        indicators = []
    
    num_subplots = 1
    if 'RSI' in indicators or 'Stoch_RSI' in indicators:
        num_subplots += 1
    if 'MACD' in indicators:
        num_subplots += 1
    if 'MFI' in indicators:
        num_subplots += 1
    
    row_heights = [0.5] + [0.15] * (num_subplots - 1)
    
    fig = make_subplots(
        rows=num_subplots,
        cols=1,
        shared_xaxes=True,
        row_heights=row_heights,
        vertical_spacing=0.08
    )
    
    # Candlestick
    fig.add_trace(
        go.Candlestick(
            x=df['Date'],
            open=df['Open'],
            high=df['High'],
            low=df['Low'],
            close=df['Close'],
            name='OHLC'
        ),
        row=1, col=1
    )
    
    # Moving averages
    if 'SMA_20' in indicators and 'SMA_20' in df.columns:
        fig.add_trace(
            go.Scatter(x=df['Date'], y=df['SMA_20'], name='SMA 20', 
                      line=dict(color='orange', width=1)),
            row=1, col=1
        )
    
    if 'SMA_50' in indicators and 'SMA_50' in df.columns:
        fig.add_trace(
            go.Scatter(x=df['Date'], y=df['SMA_50'], name='SMA 50', 
                      line=dict(color='blue', width=1)),
            row=1, col=1
        )
    
    if 'SMA_200' in indicators and 'SMA_200' in df.columns:
        fig.add_trace(
            go.Scatter(x=df['Date'], y=df['SMA_200'], name='SMA 200', 
                      line=dict(color='red', width=1)),
            row=1, col=1
        )
    
    # Bollinger Bands
    if 'BB' in indicators and 'BB_Upper' in df.columns:
        fig.add_trace(
            go.Scatter(x=df['Date'], y=df['BB_Upper'], name='BB Upper',
                      line=dict(color='rgba(0,0,0,0)')),
            row=1, col=1
        )
        fig.add_trace(
            go.Scatter(x=df['Date'], y=df['BB_Lower'], name='BB Lower',
                      line=dict(color='rgba(0,0,0,0)'),
                      fill='tonexty', fillcolor='rgba(0,100,200,0.1)'),
            row=1, col=1
        )
    
    current_row = 2
    
    # RSI
    if 'RSI' in indicators and 'RSI' in df.columns:
        fig.add_trace(
            go.Scatter(x=df['Date'], y=df['RSI'], name='RSI',
                      line=dict(color='purple', width=2)),
            row=current_row, col=1
        )
        fig.add_hline(y=70, line_dash="dash", line_color="red", row=current_row, col=1)
        fig.add_hline(y=30, line_dash="dash", line_color="green", row=current_row, col=1)
        current_row += 1
    
    # Stochastic RSI
    if 'Stoch_RSI' in indicators and 'Stoch_RSI' in df.columns:
        fig.add_trace(
            go.Scatter(x=df['Date'], y=df['Stoch_RSI'], name='Stoch RSI',
                      line=dict(color='darkviolet', width=2)),
            row=current_row, col=1
        )
        fig.add_hline(y=80, line_dash="dash", line_color="red", row=current_row, col=1)
        fig.add_hline(y=20, line_dash="dash", line_color="green", row=current_row, col=1)
        current_row += 1
    
    # MACD
    if 'MACD' in indicators and 'MACD' in df.columns:
        fig.add_trace(
            go.Scatter(x=df['Date'], y=df['MACD'], name='MACD',
                      line=dict(color='blue', width=2)),
            row=current_row, col=1
        )
        fig.add_trace(
            go.Scatter(x=df['Date'], y=df['MACD_Signal'], name='Signal',
                      line=dict(color='red', width=2)),
            row=current_row, col=1
        )
        colors = ['green' if h > 0 else 'red' for h in df['MACD_Histogram']]
        fig.add_trace(
            go.Bar(x=df['Date'], y=df['MACD_Histogram'], name='Histogram',
                  marker=dict(color=colors), showlegend=False),
            row=current_row, col=1
        )
        fig.add_hline(y=0, line_dash="dash", line_color="gray", row=current_row, col=1)
        current_row += 1
    
    # MFI
    if 'MFI' in indicators and 'MFI' in df.columns:
        fig.add_trace(
            go.Scatter(x=df['Date'], y=df['MFI'], name='MFI',
                      line=dict(color='green', width=2)),
            row=current_row, col=1
        )
        fig.add_hline(y=80, line_dash="dash", line_color="red", row=current_row, col=1)
        fig.add_hline(y=20, line_dash="dash", line_color="green", row=current_row, col=1)
    
    fig.update_layout(
        title=title,
        height=height,
        hovermode='x unified',
        template='plotly_dark',
        xaxis_rangeslider_visible=False,
        margin=dict(l=50, r=50, t=80, b=50)
    )
    
    return fig


# ============================================================================
# STREAMLIT APP
# ============================================================================

st.set_page_config(
    page_title="DSE Technical Screener",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
    .metric-card {
        background-color: #1f1f1f;
        padding: 20px;
        border-radius: 10px;
        border-left: 4px solid #00d4ff;
        margin: 10px 0;
    }
</style>
""", unsafe_allow_html=True)


def load_data_from_file(uploaded_file, file_type='price'):
    """Load and parse TSV/CSV file"""
    try:
        df = pd.read_csv(uploaded_file, sep='\t')
        
        first_col = df.columns[0]
        first_row_values = df.iloc[0].values
        
        is_header_row = False
        if len(first_row_values) > 0:
            try:
                pd.to_datetime(str(first_row_values[1]))
                is_header_row = True
            except:
                is_header_row = False
        
        if is_header_row and len(df) > 1:
            new_cols = [df.iloc[0, 0]] + list(df.iloc[0, 1:])
            df.columns = new_cols
            df = df.iloc[1:].reset_index(drop=True)
        
        ticker_col = df.columns[0]
        date_cols = df.columns[1:]
        
        melted = df.melt(id_vars=[ticker_col], var_name='Date', value_name=file_type.capitalize())
        melted.columns = ['Ticker', 'Date', file_type.capitalize()]
        
        try:
            melted['Date'] = pd.to_datetime(melted['Date'])
        except:
            melted['Date'] = pd.to_datetime(melted['Date'], errors='coerce')
        
        melted = melted.dropna(subset=['Date'])
        melted[file_type.capitalize()] = pd.to_numeric(melted[file_type.capitalize()], errors='coerce')
        melted = melted.dropna(subset=[file_type.capitalize()])
        melted = melted.sort_values('Date').reset_index(drop=True)
        
        return melted
    except Exception as e:
        st.error(f"Error loading file: {str(e)}")
        return None


def process_stock_data(price_df, volume_df):
    """Combine price and volume data"""
    try:
        merged = price_df.merge(volume_df, on=['Ticker', 'Date'], how='inner')
        merged = merged.sort_values(['Ticker', 'Date'])
        
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
        return df
    elif timeframe == 'W':
        return df.set_index('Date').resample('W').agg({
            'Open': 'first',
            'High': 'max',
            'Low': 'min',
            'Close': 'last',
            'Volume': 'sum'
        }).reset_index()
    elif timeframe == 'M':
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
    
    with st.sidebar:
        st.header("📊 Configuration")
        
        st.subheader("1. Upload Data Files")
        price_file = st.file_uploader("Upload Price Data (TSV/CSV)", type=['tsv', 'csv'], key='price')
        volume_file = st.file_uploader("Upload Volume Data (TSV/CSV)", type=['tsv', 'csv'], key='volume')
        
        if not (price_file and volume_file):
            st.warning("⚠️ Please upload both files")
            return
        
        st.success("✓ Files uploaded")
        
        st.subheader("2. Select Timeframe")
        timeframe = st.radio("Timeframe", ['Daily', 'Weekly', 'Monthly'])
        timeframe_map = {'Daily': 'D', 'Weekly': 'W', 'Monthly': 'M'}
        
        st.subheader("3. Select Indicators")
        show_sma = st.checkbox("SMA (20, 50, 200)", value=True)
        show_bb = st.checkbox("Bollinger Bands", value=True)
        show_rsi = st.checkbox("RSI", value=True)
        show_stoch = st.checkbox("Stochastic RSI", value=True)
        show_macd = st.checkbox("MACD", value=True)
        show_mfi = st.checkbox("MFI", value=True)
        
        st.subheader("4. Filters")
        signal_filter = st.selectbox("Filter by Signal", ["All", "BUY", "SELL", "HOLD"])
        min_confidence = st.slider("Minimum Confidence", -100, 100, 0)
        min_liquidity = st.slider("Minimum Liquidity Score", 0, 100, 30)
    
    price_df = load_data_from_file(price_file, 'price')
    volume_df = load_data_from_file(volume_file, 'volume')
    
    if price_df is None or volume_df is None:
        return
    
    with st.spinner("Processing data..."):
        merged_data = process_stock_data(price_df, volume_df)
        
        if merged_data is None:
            return
        
        all_results = []
        stocks = merged_data['Ticker'].unique()
        
        progress_bar = st.progress(0)
        for idx, stock in enumerate(stocks):
            stock_data = merged_data[merged_data['Ticker'] == stock].copy()
            stock_data = get_timeframe_data(stock_data, timeframe_map[timeframe])
            
            if len(stock_data) < 20:
                continue
            
            indicators = TechnicalIndicators(stock_data)
            result_df = indicators.calculate_all()
            result_df['Ticker'] = stock
            all_results.append(result_df)
            
            progress_bar.progress((idx + 1) / len(stocks))
        
        if not all_results:
            st.error("No valid data to analyze")
            return
        
        full_results = pd.concat(all_results, ignore_index=True)
    
    tab1, tab2, tab3 = st.tabs(["📊 Dashboard", "🔍 Stock Explorer", "📈 Detailed Analysis"])
    
    # TAB 1: DASHBOARD
    with tab1:
        st.header("Screener Dashboard")
        
        latest_data = full_results.sort_values('Date').groupby('Ticker').tail(1)
        
        filtered_data = latest_data.copy()
        if signal_filter != "All":
            filtered_data = filtered_data[filtered_data['Signal'] == signal_filter]
        filtered_data = filtered_data[filtered_data['Confidence'] >= min_confidence]
        filtered_data = filtered_data[filtered_data['Liquidity_Score'] >= min_liquidity]
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Total Stocks", len(latest_data))
        with col2:
            buy_count = len(latest_data[latest_data['Signal'] == 'BUY'])
            st.metric("BUY Signals", buy_count)
        with col3:
            sell_count = len(latest_data[latest_data['Signal'] == 'SELL'])
            st.metric("SELL Signals", sell_count)
        with col4:
            hold_count = len(latest_data[latest_data['Signal'] == 'HOLD'])
            st.metric("HOLD Signals", hold_count)
        
        st.divider()
        st.subheader("🎯 Top Recommendations")
        
        top_buy = latest_data[latest_data['Signal'] == 'BUY'].nlargest(10, 'Confidence')
        
        if len(top_buy) > 0:
            display_cols = ['Ticker', 'Close', 'RSI', 'MACD', 'MFI', 'Signal', 'Confidence', 'Liquidity_Score']
            available_cols = [col for col in display_cols if col in top_buy.columns]
            st.dataframe(top_buy[available_cols], use_container_width=True)
        else:
            st.info("No BUY signals found")
        
        st.divider()
        st.subheader("📊 Signal Distribution")
        signal_counts = latest_data['Signal'].value_counts()
        st.bar_chart(signal_counts)
    
    # TAB 2: STOCK EXPLORER
    with tab2:
        st.header("Stock Explorer")
        
        latest_data = full_results.sort_values('Date').groupby('Ticker').tail(1)
        selected_stock = st.selectbox("Select Stock", sorted(latest_data['Ticker'].unique()))
        
        if selected_stock:
            stock_data = full_results[full_results['Ticker'] == selected_stock].sort_values('Date')
            latest = stock_data.iloc[-1]
            
            col1, col2, col3, col4, col5 = st.columns(5)
            
            with col1:
                st.metric("Price", f"{latest['Close']:.2f}")
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
            
            fig = create_candlestick_chart(stock_data, f"{selected_stock} - {timeframe}", 
                                         indicators=indicators_to_show, height=700)
            st.plotly_chart(fig, use_container_width=True)
            
            st.divider()
            st.subheader("Technical Indicators")
            
            indicator_cols = ['Date', 'Close', 'RSI', 'Stoch_RSI', 'MACD', 'MFI', 'ATR', 'ADX', 'Liquidity_Score', 'Signal', 'Confidence']
            available_cols = [col for col in indicator_cols if col in stock_data.columns]
            
            display_df = stock_data[available_cols].tail(20).copy()
            display_df['Date'] = display_df['Date'].dt.strftime('%Y-%m-%d')
            
            st.dataframe(display_df, use_container_width=True)
    
    # TAB 3: DETAILED ANALYSIS
    with tab3:
        st.header("Detailed Analysis")
        
        latest_data = full_results.sort_values('Date').groupby('Ticker').tail(1)
        
        st.subheader("All Stocks - Latest Data")
        
        display_cols = ['Ticker', 'Close', 'RSI', 'MACD', 'MFI', 'Signal', 'Confidence', 'Liquidity_Score', 'Volume_Ratio']
        available_cols = [col for col in display_cols if col in latest_data.columns]
        
        display_df = latest_data[available_cols].sort_values('Confidence', ascending=False)
        
        st.dataframe(display_df, use_container_width=True)
        
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

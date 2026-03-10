"""
DSE Technical Screener - Indicators Module

Calculates technical indicators with anti-manipulation detection strategies
used by professional market operators globally.
"""

import pandas as pd
import numpy as np
from typing import Dict, Tuple, List


class TechnicalIndicators:
    """Calculate technical indicators and generate trading signals"""
    
    def __init__(self, df: pd.DataFrame):
        """
        Initialize with price data
        
        Args:
            df: DataFrame with columns ['Date', 'Open', 'High', 'Low', 'Close', 'Volume']
        """
        self.df = df.copy().sort_values('Date').reset_index(drop=True)
        self.signals = {}
        
    def calculate_all(self) -> pd.DataFrame:
        """Calculate all indicators and return enhanced dataframe"""
        
        # Basic indicators
        self._calculate_rsi()
        self._calculate_stochastic_rsi()
        self._calculate_macd()
        self._calculate_bollinger_bands()
        self._calculate_moving_averages()
        self._calculate_atr()
        self._calculate_mfi()
        self._calculate_obv()
        self._calculate_adx()
        
        # Anti-manipulation strategies
        self._detect_volume_anomalies()
        self._detect_price_manipulation()
        self._detect_divergences()
        self._calculate_liquidity_score()
        
        # Generate final signals
        self._generate_trading_signals()
        
        return self.df
    
    def _calculate_rsi(self, period: int = 14):
        """Relative Strength Index"""
        delta = self.df['Close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / (loss + 1e-10)
        self.df['RSI'] = 100 - (100 / (1 + rs))
        self.df['RSI'] = self.df['RSI'].fillna(50)
    
    def _calculate_stochastic_rsi(self, period: int = 14, smooth_k: int = 3, smooth_d: int = 3):
        """Stochastic RSI - More sensitive than RSI"""
        rsi = self.df['RSI']
        lowest_rsi = rsi.rolling(window=period).min()
        highest_rsi = rsi.rolling(window=period).max()
        
        stoch_rsi = 100 * (rsi - lowest_rsi) / (highest_rsi - lowest_rsi + 1e-10)
        self.df['Stoch_RSI'] = stoch_rsi.fillna(50)
        
        # K and D lines
        self.df['Stoch_K'] = stoch_rsi.rolling(window=smooth_k).mean()
        self.df['Stoch_D'] = self.df['Stoch_K'].rolling(window=smooth_d).mean()
        
        self.df['Stoch_K'] = self.df['Stoch_K'].fillna(50)
        self.df['Stoch_D'] = self.df['Stoch_D'].fillna(50)
    
    def _calculate_macd(self, fast: int = 12, slow: int = 26, signal: int = 9):
        """MACD - Momentum indicator"""
        ema_fast = self.df['Close'].ewm(span=fast, adjust=False).mean()
        ema_slow = self.df['Close'].ewm(span=slow, adjust=False).mean()
        
        self.df['MACD'] = ema_fast - ema_slow
        self.df['MACD_Signal'] = self.df['MACD'].ewm(span=signal, adjust=False).mean()
        self.df['MACD_Histogram'] = self.df['MACD'] - self.df['MACD_Signal']
    
    def _calculate_bollinger_bands(self, period: int = 20, std_dev: float = 2.0):
        """Bollinger Bands - Volatility indicator"""
        sma = self.df['Close'].rolling(window=period).mean()
        std = self.df['Close'].rolling(window=period).std()
        
        self.df['BB_Upper'] = sma + (std * std_dev)
        self.df['BB_Middle'] = sma
        self.df['BB_Lower'] = sma - (std * std_dev)
        self.df['BB_Width'] = self.df['BB_Upper'] - self.df['BB_Lower']
        self.df['BB_Position'] = (self.df['Close'] - self.df['BB_Lower']) / (self.df['BB_Width'] + 1e-10)
    
    def _calculate_moving_averages(self):
        """SMA and EMA"""
        self.df['SMA_20'] = self.df['Close'].rolling(window=20).mean()
        self.df['SMA_50'] = self.df['Close'].rolling(window=50).mean()
        self.df['SMA_200'] = self.df['Close'].rolling(window=200).mean()
        
        self.df['EMA_12'] = self.df['Close'].ewm(span=12, adjust=False).mean()
        self.df['EMA_26'] = self.df['Close'].ewm(span=26, adjust=False).mean()
    
    def _calculate_atr(self, period: int = 14):
        """Average True Range - Volatility measure"""
        high_low = self.df['High'] - self.df['Low']
        high_close = abs(self.df['High'] - self.df['Close'].shift())
        low_close = abs(self.df['Low'] - self.df['Close'].shift())
        
        tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
        self.df['ATR'] = tr.rolling(window=period).mean()
        self.df['ATR_Percent'] = (self.df['ATR'] / self.df['Close']) * 100
    
    def _calculate_mfi(self, period: int = 14):
        """Money Flow Index - Volume-weighted RSI"""
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
        """On-Balance Volume - Cumulative volume indicator"""
        obv = (np.sign(self.df['Close'].diff()) * self.df['Volume']).fillna(0).cumsum()
        self.df['OBV'] = obv
        self.df['OBV_EMA'] = obv.ewm(span=20, adjust=False).mean()
    
    def _calculate_adx(self, period: int = 14):
        """Average Directional Index - Trend strength"""
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
    
    def _calculate_true_range(self) -> pd.Series:
        """Helper: Calculate True Range"""
        high_low = self.df['High'] - self.df['Low']
        high_close = abs(self.df['High'] - self.df['Close'].shift())
        low_close = abs(self.df['Low'] - self.df['Close'].shift())
        return pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    
    def _detect_volume_anomalies(self):
        """Detect abnormal volume patterns - common manipulation sign"""
        avg_volume = self.df['Volume'].rolling(window=20).mean()
        volume_ratio = self.df['Volume'] / (avg_volume + 1e-10)
        
        self.df['Volume_Anomaly'] = volume_ratio > 2.5  # 2.5x normal volume
        self.df['Volume_Ratio'] = volume_ratio
        
        # Spike without price movement = potential manipulation
        price_change_pct = abs(self.df['Close'].pct_change()) * 100
        self.df['Volume_Price_Divergence'] = (self.df['Volume_Anomaly']) & (price_change_pct < 2)
    
    def _detect_price_manipulation(self):
        """Detect common price manipulation patterns"""
        # Pattern 1: Sudden spike followed by reversal (trap)
        self.df['Price_Spike'] = abs(self.df['Close'].pct_change()) > 5
        self.df['Reversal_Next'] = self.df['Price_Spike'].shift(-1) & \
                                   (self.df['Close'].pct_change().shift(-1) * self.df['Close'].pct_change() < 0)
        
        # Pattern 2: Low liquidity trap (high price move on low volume)
        avg_vol = self.df['Volume'].rolling(window=20).mean()
        price_move = abs(self.df['Close'].pct_change()) * 100
        self.df['Low_Liquidity_Trap'] = (price_move > 3) & (self.df['Volume'] < avg_vol * 0.5)
        
        # Pattern 3: Fake breakout (break above resistance then reversal)
        self.df['Above_BB_Upper'] = self.df['Close'] > self.df['BB_Upper']
        self.df['Fake_Breakout'] = self.df['Above_BB_Upper'] & \
                                   (self.df['Close'].shift(-1) < self.df['BB_Middle'])
    
    def _detect_divergences(self):
        """Detect bullish/bearish divergences - weak signals"""
        # RSI divergence
        self.df['RSI_Divergence'] = False
        if len(self.df) > 20:
            # Bearish: Price higher but RSI lower
            price_higher = self.df['Close'] > self.df['Close'].shift(20)
            rsi_lower = self.df['RSI'] < self.df['RSI'].shift(20)
            self.df['RSI_Divergence'] = price_higher & rsi_lower
        
        # MACD divergence
        self.df['MACD_Divergence'] = False
        if len(self.df) > 20:
            price_higher = self.df['Close'] > self.df['Close'].shift(20)
            macd_lower = self.df['MACD'] < self.df['MACD'].shift(20)
            self.df['MACD_Divergence'] = price_higher & macd_lower
    
    def _calculate_liquidity_score(self):
        """Calculate liquidity score (0-100) - higher is better"""
        # Factors: volume, spread, volatility
        avg_vol = self.df['Volume'].rolling(window=20).mean()
        vol_score = (self.df['Volume'] / (avg_vol + 1e-10)).clip(0, 5) * 20
        
        # Bid-ask spread proxy (using high-low)
        spread = ((self.df['High'] - self.df['Low']) / self.df['Close']) * 100
        spread_score = (1 - (spread / spread.max()).clip(0, 1)) * 30
        
        # Volatility score (lower volatility = better liquidity)
        volatility = self.df['Close'].pct_change().rolling(window=20).std() * 100
        vol_stability_score = (1 - (volatility / volatility.max()).clip(0, 1)) * 50
        
        self.df['Liquidity_Score'] = vol_score + spread_score + vol_stability_score
        self.df['Liquidity_Score'] = self.df['Liquidity_Score'].clip(0, 100)
    
    def _generate_trading_signals(self):
        """Generate BUY/HOLD/SELL signals with manipulation detection"""
        signals = []
        confidence = []
        reasons = []
        
        for idx in range(len(self.df)):
            row = self.df.iloc[idx]
            signal = "HOLD"
            conf = 0
            reason = []
            
            # Check for manipulation red flags
            if row.get('Volume_Price_Divergence', False):
                signal = "HOLD"
                conf = -50
                reason.append("⚠️ Volume anomaly without price movement")
            elif row.get('Low_Liquidity_Trap', False):
                signal = "HOLD"
                conf = -40
                reason.append("⚠️ Low liquidity trap detected")
            elif row.get('Fake_Breakout', False):
                signal = "HOLD"
                conf = -30
                reason.append("⚠️ Potential fake breakout")
            else:
                # BUY signals
                buy_score = 0
                
                # RSI oversold
                if row['RSI'] < 30:
                    buy_score += 20
                    reason.append("✓ RSI oversold")
                elif row['RSI'] < 40:
                    buy_score += 10
                    reason.append("✓ RSI low")
                
                # Stochastic RSI
                if row['Stoch_RSI'] < 20 and row['Stoch_K'] < row['Stoch_D']:
                    buy_score += 15
                    reason.append("✓ Stoch RSI oversold")
                
                # MACD bullish
                if row['MACD'] > row['MACD_Signal'] and row['MACD_Histogram'] > 0:
                    buy_score += 20
                    reason.append("✓ MACD bullish crossover")
                elif row['MACD'] > row['MACD_Signal']:
                    buy_score += 10
                    reason.append("✓ MACD above signal")
                
                # MFI oversold
                if row['MFI'] < 30:
                    buy_score += 15
                    reason.append("✓ MFI oversold")
                
                # Bollinger Bands
                if row['Close'] < row['BB_Lower']:
                    buy_score += 15
                    reason.append("✓ Price below lower BB")
                elif row['Close'] < row['BB_Middle']:
                    buy_score += 5
                    reason.append("✓ Price below middle BB")
                
                # Moving average alignment
                if row['Close'] > row['SMA_20'] > row['SMA_50']:
                    buy_score += 10
                    reason.append("✓ Bullish MA alignment")
                
                # OBV confirmation
                if row['OBV'] > row['OBV_EMA']:
                    buy_score += 5
                    reason.append("✓ OBV positive")
                
                # ADX trend strength
                if row['ADX'] > 25:
                    buy_score += 5
                    reason.append("✓ Strong trend")
                
                # Check for divergences (weak signals)
                if row.get('RSI_Divergence', False):
                    buy_score -= 10
                    reason.append("⚠️ RSI divergence")
                
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
                    # SELL signals
                    sell_score = 0
                    reason = []
                    
                    if row['RSI'] > 70:
                        sell_score += 20
                        reason.append("✓ RSI overbought")
                    elif row['RSI'] > 60:
                        sell_score += 10
                        reason.append("✓ RSI high")
                    
                    if row['Stoch_RSI'] > 80 and row['Stoch_K'] > row['Stoch_D']:
                        sell_score += 15
                        reason.append("✓ Stoch RSI overbought")
                    
                    if row['MACD'] < row['MACD_Signal'] and row['MACD_Histogram'] < 0:
                        sell_score += 20
                        reason.append("✓ MACD bearish crossover")
                    
                    if row['MFI'] > 70:
                        sell_score += 15
                        reason.append("✓ MFI overbought")
                    
                    if row['Close'] > row['BB_Upper']:
                        sell_score += 15
                        reason.append("✓ Price above upper BB")
                    
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
    
    def get_summary(self) -> Dict:
        """Get summary of latest indicators"""
        latest = self.df.iloc[-1]
        
        return {
            'Price': latest['Close'],
            'RSI': latest['RSI'],
            'Stoch_RSI': latest['Stoch_RSI'],
            'MACD': latest['MACD'],
            'MACD_Signal': latest['MACD_Signal'],
            'MFI': latest['MFI'],
            'ATR': latest['ATR'],
            'ATR_Percent': latest['ATR_Percent'],
            'BB_Upper': latest['BB_Upper'],
            'BB_Lower': latest['BB_Lower'],
            'SMA_20': latest['SMA_20'],
            'SMA_50': latest['SMA_50'],
            'EMA_12': latest['EMA_12'],
            'EMA_26': latest['EMA_26'],
            'ADX': latest['ADX'],
            'Liquidity_Score': latest['Liquidity_Score'],
            'Volume_Ratio': latest['Volume_Ratio'],
            'Signal': latest['Signal'],
            'Confidence': latest['Confidence'],
            'Signal_Reason': latest['Signal_Reason'],
        }

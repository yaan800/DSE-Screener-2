"""
DSE Technical Screener - Charting Module

Interactive candlestick charts with overlayable technical indicators
"""

import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from typing import List, Optional


class CandlestickChart:
    """Create interactive candlestick charts with technical indicators"""
    
    def __init__(self, df: pd.DataFrame, title: str = "Stock Price"):
        """
        Initialize chart
        
        Args:
            df: DataFrame with OHLCV data and indicators
            title: Chart title
        """
        self.df = df.copy()
        self.title = title
    
    def create_chart(self, 
                     indicators: List[str] = None,
                     height: int = 600,
                     show_volume: bool = True) -> go.Figure:
        """
        Create candlestick chart with selected indicators
        
        Args:
            indicators: List of indicators to display (e.g., ['RSI', 'MACD', 'BB'])
            height: Chart height
            show_volume: Show volume subplot
            
        Returns:
            Plotly figure
        """
        if indicators is None:
            indicators = []
        
        # Determine number of subplots
        num_subplots = 1  # Candlestick
        if show_volume:
            num_subplots += 1
        
        # Count indicator subplots
        indicator_subplots = 0
        if 'RSI' in indicators or 'Stoch_RSI' in indicators:
            indicator_subplots += 1
        if 'MACD' in indicators:
            indicator_subplots += 1
        if 'MFI' in indicators:
            indicator_subplots += 1
        
        num_subplots += indicator_subplots
        
        # Create subplots
        row_heights = [0.4]  # Candlestick
        if show_volume:
            row_heights.append(0.15)  # Volume
        row_heights.extend([0.15] * indicator_subplots)  # Indicators
        
        fig = make_subplots(
            rows=num_subplots,
            cols=1,
            shared_xaxes=True,
            row_heights=row_heights,
            vertical_spacing=0.08,
            subplot_titles=self._get_subplot_titles(indicators, show_volume)
        )
        
        # Add candlestick
        fig.add_trace(
            go.Candlestick(
                x=self.df['Date'],
                open=self.df['Open'],
                high=self.df['High'],
                low=self.df['Low'],
                close=self.df['Close'],
                name='OHLC',
                hovertemplate='<b>%{x|%Y-%m-%d}</b>  
' +
                            'Open: %{open:.2f}  
' +
                            'High: %{high:.2f}  
' +
                            'Low: %{low:.2f}  
' +
                            'Close: %{close:.2f}<extra></extra>'
            ),
            row=1, col=1
        )
        
        # Add moving averages to candlestick
        if 'SMA_20' in indicators and 'SMA_20' in self.df.columns:
            fig.add_trace(
                go.Scatter(
                    x=self.df['Date'],
                    y=self.df['SMA_20'],
                    name='SMA 20',
                    line=dict(color='orange', width=1),
                    hovertemplate='SMA 20: %{y:.2f}<extra></extra>'
                ),
                row=1, col=1
            )
        
        if 'SMA_50' in indicators and 'SMA_50' in self.df.columns:
            fig.add_trace(
                go.Scatter(
                    x=self.df['Date'],
                    y=self.df['SMA_50'],
                    name='SMA 50',
                    line=dict(color='blue', width=1),
                    hovertemplate='SMA 50: %{y:.2f}<extra></extra>'
                ),
                row=1, col=1
            )
        
        if 'SMA_200' in indicators and 'SMA_200' in self.df.columns:
            fig.add_trace(
                go.Scatter(
                    x=self.df['Date'],
                    y=self.df['SMA_200'],
                    name='SMA 200',
                    line=dict(color='red', width=1),
                    hovertemplate='SMA 200: %{y:.2f}<extra></extra>'
                ),
                row=1, col=1
            )
        
        # Add Bollinger Bands
        if 'BB' in indicators and 'BB_Upper' in self.df.columns:
            fig.add_trace(
                go.Scatter(
                    x=self.df['Date'],
                    y=self.df['BB_Upper'],
                    name='BB Upper',
                    line=dict(color='rgba(0,0,0,0)'),
                    hovertemplate='BB Upper: %{y:.2f}<extra></extra>'
                ),
                row=1, col=1
            )
            
            fig.add_trace(
                go.Scatter(
                    x=self.df['Date'],
                    y=self.df['BB_Lower'],
                    name='BB Lower',
                    line=dict(color='rgba(0,0,0,0)'),
                    fill='tonexty',
                    fillcolor='rgba(0,100,200,0.1)',
                    hovertemplate='BB Lower: %{y:.2f}<extra></extra>'
                ),
                row=1, col=1
            )
        
        current_row = 2
        
        # Add volume
        if show_volume:
            colors = ['red' if close < open_ else 'green' 
                     for close, open_ in zip(self.df['Close'], self.df['Open'])]
            
            fig.add_trace(
                go.Bar(
                    x=self.df['Date'],
                    y=self.df['Volume'],
                    name='Volume',
                    marker=dict(color=colors),
                    hovertemplate='Volume: %{y:,.0f}<extra></extra>',
                    showlegend=False
                ),
                row=current_row, col=1
            )
            current_row += 1
        
        # Add RSI
        if 'RSI' in indicators and 'RSI' in self.df.columns:
            fig.add_trace(
                go.Scatter(
                    x=self.df['Date'],
                    y=self.df['RSI'],
                    name='RSI',
                    line=dict(color='purple', width=2),
                    hovertemplate='RSI: %{y:.2f}<extra></extra>'
                ),
                row=current_row, col=1
            )
            
            # Add overbought/oversold lines
            fig.add_hline(y=70, line_dash="dash", line_color="red", 
                         row=current_row, col=1, annotation_text="Overbought")
            fig.add_hline(y=30, line_dash="dash", line_color="green", 
                         row=current_row, col=1, annotation_text="Oversold")
            current_row += 1
        
        # Add Stochastic RSI
        if 'Stoch_RSI' in indicators and 'Stoch_RSI' in self.df.columns:
            fig.add_trace(
                go.Scatter(
                    x=self.df['Date'],
                    y=self.df['Stoch_RSI'],
                    name='Stoch RSI',
                    line=dict(color='darkviolet', width=2),
                    hovertemplate='Stoch RSI: %{y:.2f}<extra></extra>'
                ),
                row=current_row, col=1
            )
            
            fig.add_trace(
                go.Scatter(
                    x=self.df['Date'],
                    y=self.df['Stoch_K'],
                    name='Stoch K',
                    line=dict(color='blue', width=1, dash='dot'),
                    hovertemplate='Stoch K: %{y:.2f}<extra></extra>'
                ),
                row=current_row, col=1
            )
            
            fig.add_trace(
                go.Scatter(
                    x=self.df['Date'],
                    y=self.df['Stoch_D'],
                    name='Stoch D',
                    line=dict(color='red', width=1, dash='dot'),
                    hovertemplate='Stoch D: %{y:.2f}<extra></extra>'
                ),
                row=current_row, col=1
            )
            
            fig.add_hline(y=80, line_dash="dash", line_color="red", 
                         row=current_row, col=1)
            fig.add_hline(y=20, line_dash="dash", line_color="green", 
                         row=current_row, col=1)
            current_row += 1
        
        # Add MACD
        if 'MACD' in indicators and 'MACD' in self.df.columns:
            fig.add_trace(
                go.Scatter(
                    x=self.df['Date'],
                    y=self.df['MACD'],
                    name='MACD',
                    line=dict(color='blue', width=2),
                    hovertemplate='MACD: %{y:.4f}<extra></extra>'
                ),
                row=current_row, col=1
            )
            
            fig.add_trace(
                go.Scatter(
                    x=self.df['Date'],
                    y=self.df['MACD_Signal'],
                    name='Signal',
                    line=dict(color='red', width=2),
                    hovertemplate='Signal: %{y:.4f}<extra></extra>'
                ),
                row=current_row, col=1
            )
            
            colors = ['green' if h > 0 else 'red' for h in self.df['MACD_Histogram']]
            fig.add_trace(
                go.Bar(
                    x=self.df['Date'],
                    y=self.df['MACD_Histogram'],
                    name='Histogram',
                    marker=dict(color=colors),
                    hovertemplate='Histogram: %{y:.4f}<extra></extra>',
                    showlegend=False
                ),
                row=current_row, col=1
            )
            
            fig.add_hline(y=0, line_dash="dash", line_color="gray", 
                         row=current_row, col=1)
            current_row += 1
        
        # Add MFI
        if 'MFI' in indicators and 'MFI' in self.df.columns:
            fig.add_trace(
                go.Scatter(
                    x=self.df['Date'],
                    y=self.df['MFI'],
                    name='MFI',
                    line=dict(color='green', width=2),
                    hovertemplate='MFI: %{y:.2f}<extra></extra>'
                ),
                row=current_row, col=1
            )
            
            fig.add_hline(y=80, line_dash="dash", line_color="red", 
                         row=current_row, col=1, annotation_text="Overbought")
            fig.add_hline(y=20, line_dash="dash", line_color="green", 
                         row=current_row, col=1, annotation_text="Oversold")
            current_row += 1
        
        # Update layout
        fig.update_layout(
            title=self.title,
            height=height,
            hovermode='x unified',
            template='plotly_dark',
            font=dict(size=10),
            margin=dict(l=50, r=50, t=80, b=50),
        )
        
        # Update x-axis
        fig.update_xaxes(rangeslider_visible=False, row=1, col=1)
        
        return fig
    
    def _get_subplot_titles(self, indicators: List[str], show_volume: bool) -> tuple:
        """Generate subplot titles"""
        titles = ['Price']
        
        if show_volume:
            titles.append('Volume')
        
        if 'RSI' in indicators or 'Stoch_RSI' in indicators:
            titles.append('Momentum (RSI/Stoch RSI)')
        
        if 'MACD' in indicators:
            titles.append('MACD')
        
        if 'MFI' in indicators:
            titles.append('MFI')
        
        return tuple(titles)
    
    def create_simple_chart(self, height: int = 400) -> go.Figure:
        """Create simple candlestick chart without indicators"""
        fig = go.Figure(data=[go.Candlestick(
            x=self.df['Date'],
            open=self.df['Open'],
            high=self.df['High'],
            low=self.df['Low'],
            close=self.df['Close'],
            name='OHLC'
        )])
        
        fig.update_layout(
            title=self.title,
            height=height,
            hovermode='x unified',
            template='plotly_dark',
            xaxis_rangeslider_visible=False,
            margin=dict(l=50, r=50, t=80, b=50),
        )
        
        return fig


def create_comparison_chart(dataframes: dict, height: int = 500) -> go.Figure:
    """
    Create comparison chart for multiple stocks
    
    Args:
        dataframes: Dict of {stock_name: DataFrame}
        height: Chart height
    """
    fig = go.Figure()
    
    for stock_name, df in dataframes.items():
        fig.add_trace(go.Scatter(
            x=df['Date'],
            y=df['Close'],
            name=stock_name,
            mode='lines',
            hovertemplate=f'{stock_name}: %{{y:.2f}}<extra></extra>'
        ))
    
    fig.update_layout(
        title='Stock Price Comparison',
        xaxis_title='Date',
        yaxis_title='Price',
        height=height,
        hovermode='x unified',
        template='plotly_dark',
        margin=dict(l=50, r=50, t=80, b=50),
    )
    
    return fig

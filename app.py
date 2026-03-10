"""
DSE Technical Screener - Debug Version
Shows detailed information about data loading and parsing
"""

import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime

st.set_page_config(page_title="DSE Technical Screener - Debug", page_icon="📈", layout="wide")

st.title("📈 DSE Technical Screener - Debug Mode")
st.markdown("Debugging data loading and parsing issues")

# Sidebar for file upload
with st.sidebar:
    st.header("📊 Upload Data")
    price_file = st.file_uploader("Upload Price Data (CSV)", type=['csv'])
    volume_file = st.file_uploader("Upload Volume Data (CSV)", type=['csv'])

if price_file is None or volume_file is None:
    st.warning("⚠️ Please upload both Price and Volume CSV files")
    st.stop()

st.write("---")
st.header("Step 1: Load Raw Files")

try:
    # Load price data
    price_df_raw = pd.read_csv(price_file)
    st.success(f"✓ Price file loaded: {price_df_raw.shape[0]} rows, {price_df_raw.shape[1]} columns")
    st.write("**Price file first row:**")
    st.write(price_df_raw.iloc[0])
    
    # Load volume data
    volume_df_raw = pd.read_csv(volume_file)
    st.success(f"✓ Volume file loaded: {volume_df_raw.shape[0]} rows, {volume_df_raw.shape[1]} columns")
    st.write("**Volume file first row:**")
    st.write(volume_df_raw.iloc[0])
    
except Exception as e:
    st.error(f"❌ Error loading files: {str(e)}")
    st.stop()

st.write("---")
st.header("Step 2: Parse and Transform Data")

def load_data_from_file(df, file_type='price'):
    """Load and parse data with detailed debugging"""
    try:
        # Get ticker column (first column)
        ticker_col = df.columns[0]
        date_cols = df.columns[1:]
        
        st.write(f"**Ticker column:** {ticker_col}")
        st.write(f"**Number of date columns:** {len(date_cols)}")
        st.write(f"**First 5 date columns:** {list(date_cols[:5])}")
        
        # Melt the dataframe from wide to long format
        melted = df.melt(id_vars=[ticker_col], var_name='Date', value_name=file_type.capitalize())
        melted.columns = ['Ticker', 'Date', file_type.capitalize()]
        
        st.write(f"**After melting:** {melted.shape[0]} rows")
        st.write("**First 5 rows after melting:**")
        st.write(melted.head())
        
        # Convert date - handle DD-Mon format (e.g., 09-Mar, 08-Mar)
        st.write(f"**Parsing dates (format: DD-Mon)...**")
        try:
            # Try DD-Mon format first
            melted['Date'] = pd.to_datetime(melted['Date'], format='%d-%b')
            # Add year (assume current year)
            current_year = pd.Timestamp.now().year
            melted['Date'] = melted['Date'].apply(lambda x: x.replace(year=current_year))
            st.success(f"✓ Dates parsed successfully (year: {current_year})")
        except Exception as e:
            st.warning(f"⚠️ DD-Mon format failed: {str(e)}")
            try:
                # Try other common formats
                melted['Date'] = pd.to_datetime(melted['Date'])
                st.success("✓ Dates parsed with default format")
            except Exception as e2:
                st.error(f"❌ Date parsing failed: {str(e2)}")
                melted['Date'] = pd.to_datetime(melted['Date'], errors='coerce')
        
        st.write("**First 5 dates after parsing:**")
        st.write(melted['Date'].head())
        
        # Remove rows with invalid dates
        invalid_dates = melted['Date'].isna().sum()
        if invalid_dates > 0:
            st.warning(f"⚠️ Found {invalid_dates} invalid dates - removing them")
        melted = melted.dropna(subset=['Date'])
        st.write(f"**After removing invalid dates:** {melted.shape[0]} rows")
        
        # Convert price/volume to numeric - handle commas in numbers (e.g., 201,932)
        st.write(f"**Converting {file_type} to numeric...**")
        st.write(f"**Sample values before conversion:** {melted[file_type.capitalize()].head().tolist()}")
        
        melted[file_type.capitalize()] = melted[file_type.capitalize()].astype(str).str.replace(',', '')
        melted[file_type.capitalize()] = pd.to_numeric(melted[file_type.capitalize()], errors='coerce')
        
        st.write(f"**Sample values after conversion:** {melted[file_type.capitalize()].head().tolist()}")
        
        invalid_values = melted[file_type.capitalize()].isna().sum()
        if invalid_values > 0:
            st.warning(f"⚠️ Found {invalid_values} invalid {file_type} values - removing them")
        melted = melted.dropna(subset=[file_type.capitalize()])
        st.write(f"**After removing invalid values:** {melted.shape[0]} rows")
        
        # Sort by date (oldest first)
        melted = melted.sort_values('Date').reset_index(drop=True)
        st.success(f"✓ Data sorted by date (oldest first)")
        
        st.write("**Final data sample:**")
        st.write(melted.head(10))
        
        return melted
    except Exception as e:
        st.error(f"❌ Error in load_data_from_file: {str(e)}")
        import traceback
        st.error(traceback.format_exc())
        return None

st.subheader("Parsing Price Data:")
price_df = load_data_from_file(price_df_raw, 'price')

if price_df is None:
    st.error("❌ Failed to parse price data")
    st.stop()

st.write("---")
st.subheader("Parsing Volume Data:")
volume_df = load_data_from_file(volume_df_raw, 'volume')

if volume_df is None:
    st.error("❌ Failed to parse volume data")
    st.stop()

st.write("---")
st.header("Step 3: Merge Data")

try:
    merged = price_df.merge(volume_df, on=['Ticker', 'Date'], how='inner')
    st.success(f"✓ Data merged: {merged.shape[0]} rows")
    st.write(f"**Columns:** {list(merged.columns)}")
    st.write("**First 10 rows:**")
    st.write(merged.head(10))
except Exception as e:
    st.error(f"❌ Error merging data: {str(e)}")
    st.stop()

st.write("---")
st.header("Step 4: Check Data Per Stock")

stocks = merged['Ticker'].unique()
st.write(f"**Total unique stocks:** {len(stocks)}")

stock_counts = merged.groupby('Ticker').size()
st.write(f"**Data points per stock (first 20):**")
st.write(stock_counts.head(20))

stocks_with_enough_data = stock_counts[stock_counts >= 10]
st.write(f"**Stocks with 10+ data points:** {len(stocks_with_enough_data)}")

if len(stocks_with_enough_data) == 0:
    st.error("❌ No stocks have enough data points (need 10+)")
else:
    st.success(f"✓ {len(stocks_with_enough_data)} stocks ready for analysis")
    st.write("**Stocks with enough data:**")
    st.write(stocks_with_enough_data.sort_values(ascending=False))

st.write("---")
st.header("✓ Debug Complete!")
st.success("If you see this message, data loading is working correctly!")

import streamlit as st
import pandas as pd
import requests
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime, timedelta

# Page config
st.set_page_config(page_title="Forex Dashboard with Technical Indicators", layout="wide")

# Title
st.title("🪙 Forex Trading Dashboard with Technical Indicators")

# Sidebar for user inputs
st.sidebar.header("Select Currency Pair")
base_currencies = ["USD", "EUR", "GBP", "JPY", "AUD", "CAD"]
quote_currencies = ["USD", "EUR", "GBP", "JPY", "AUD", "CAD"]
base = st.sidebar.selectbox("Base Currency", base_currencies)
quote = st.sidebar.selectbox("Quote Currency", quote_currencies, index=1 if base == "USD" else 0)

# Ensure base != quote
if base == quote:
    st.sidebar.warning("Please select different currencies.")
    st.stop()

pair = f"{base}{quote}=X"

days = st.sidebar.slider("Historical Days", 30, 365, 30)  # Min 30 for all indicators

# Technical Indicators Selection
st.sidebar.subheader("Technical Indicators")
show_sma = st.sidebar.checkbox("Simple Moving Average (SMA 20)", True)
show_ema = st.sidebar.checkbox("Exponential Moving Average (EMA 20)", True)
show_bb = st.sidebar.checkbox("Bollinger Bands (20, 2)", True)  # Bollinger Bands
show_rsi = st.sidebar.checkbox("RSI (14)", True)
show_macd = st.sidebar.checkbox("MACD", True)

# Free API for forex data
@st.cache_data(ttl=300)  # Cache for 5 minutes
def fetch_forex_data(pair, days):
    end_date = datetime.now().strftime("%Y-%m-%d")
    start_date = (datetime.now() - timedelta(days=days)).strftime("%Y-%m-%d")
    
    url = f"https://api.exchangerate.host/timeseries?start_date={start_date}&end_date={end_date}&base={base}&symbols={quote}"
    
    try:
        response = requests.get(url)
        data = response.json()
        if "rates" in data:
            df = pd.DataFrame(data["rates"]).T
            df.index = pd.to_datetime(df.index)
            df = df.reset_index().rename(columns={"index": "Date", quote: "Rate"})
            df["Date"] = pd.to_datetime(df["Date"])
            df = df.sort_values("Date").reset_index(drop=True)  # Ensure sorted
            return df
        else:
            st.error("API error: " + data.get("error", "Unknown"))
            return pd.DataFrame()
    except Exception as e:
        st.error(f"Error fetching data: {e}")
        return pd.DataFrame()

# Fetch data
df = fetch_forex_data(pair, days)

# Compute Technical Indicators (native pandas implementation)
@st.cache_data
def compute_indicators(df):
    if df.empty or len(df) < 30:  # Need min data for indicators
        return df
    
    df = df.copy()
    
    # SMA and EMA
    if show_sma:
        df['SMA_20'] = df['Rate'].rolling(window=20).mean()
    if show_ema:
        df['EMA_20'] = df['Rate'].ewm(span=20).mean()
    
    # Bollinger Bands
    if show_bb:
        df['BBM_20_2.0'] = df['Rate'].rolling(window=20).mean()
        bb_std = df['Rate'].rolling(window=20).std()
        df['BBU_20_2.0'] = df['BBM_20_2.0'] + (bb_std * 2)
        df['BBL_20_2.0'] = df['BBM_20_2.0'] - (bb_std * 2)
    
    # RSI
    if show_rsi:
        def calculate_rsi(prices, window=14):
            delta = prices.diff()
            gain = delta.where(delta > 0, 0)
            loss = -delta.where(delta < 0, 0)
            avg_gain = gain.rolling(window=window, min_periods=1).mean()
            avg_loss = loss.rolling(window=window, min_periods=1).mean()
            rs = avg_gain / avg_loss
            rsi = 100 - (100 / (1 + rs))
            return rsi
        df['RSI'] = calculate_rsi(df['Rate'])
    
    # MACD
    if show_macd:
        ema12 = df['Rate'].ewm(span=12).mean()
        ema26 = df['Rate'].ewm(span=26).mean()
        df['MACD'] = ema12 - ema26
        df['MACD_Signal'] = df['MACD'].ewm(span=9).mean()
        df['MACD_Hist'] = df['MACD'] - df['MACD_Signal']
    
    return df

if not df.empty:
    df = compute_indicators(df)

# Main content
col1, col2 = st.columns(2)

with col1:
    current_rate = df["Rate"].iloc[-1] if not df.empty else "N/A"
    delta = df["Rate"].iloc[-1] - df["Rate"].iloc[0] if len(df) > 1 else 0
    st.metric("Current Rate", current_rate, delta)

with col2:
    if not df.empty and len(df) > 1:
        change_pct = ((df["Rate"].iloc[-1] - df["Rate"].iloc[0]) / df["Rate"].iloc[0]) * 100
        st.metric("Change (Period)", f"{change_pct:.2f}%", change_pct)

# Chart with Indicators
if not df.empty and len(df) >= 30:
    # Determine number of subplots
    num_rows = 1
    if show_rsi: num_rows += 1
    if show_macd: num_rows += 1
    
    fig = make_subplots(
        rows=num_rows, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.05,
        subplot_titles=(f"{base}/{quote} Price", "RSI" if show_rsi else "", "MACD" if show_macd else ""),
        row_heights=[0.6] + [0.2] * (num_rows - 1)
    )
    
    # Price and MAs/Bands (row 1)
    fig.add_trace(go.Scatter(x=df['Date'], y=df['Rate'], name='Rate', line=dict(color='blue')), row=1, col=1)
    if show_sma and 'SMA_20' in df.columns:
        fig.add_trace(go.Scatter(x=df['Date'], y=df['SMA_20'], name='SMA 20', line=dict(color='orange')), row=1, col=1)
    if show_ema and 'EMA_20' in df.columns:
        fig.add_trace(go.Scatter(x=df['Date'], y=df['EMA_20'], name='EMA 20', line=dict(color='green')), row=1, col=1)
    
    # Bollinger Bands
    if show_bb and all(col in df.columns for col in ['BBU_20_2.0', 'BBL_20_2.0']):
        fig.add_trace(go.Scatter(x=df['Date'], y=df['BBU_20_2.0'], name='BB Upper', line=dict(color='red', dash='dash')), row=1, col=1)
        fig.add_trace(go.Scatter(x=df['Date'], y=df['BBL_20_2.0'], name='BB Lower', line=dict(color='green', dash='dash')), row=1, col=1)
        # Fill between bands
        fig.add_trace(go.Scatter(
            x=pd.concat([df['Date'], df['Date'][::-1]]), 
            y=pd.concat([df['BBU_20_2.0'], df['BBL_20_2.0'][::-1]]), 
            fill='toself', fillcolor='rgba(128,128,128,0.2)', 
            line=dict(color='rgba(255,255,255,0)'), name='BB Band', showlegend=False
        ), row=1, col=1)
        if show_sma:  # BB middle is SMA, avoid duplicate if SMA shown
            pass
        else:
            fig.add_trace(go.Scatter(x=df['Date'], y=df['BBM_20_2.0'], name='BB Middle', line=dict(color='orange')), row=1, col=1)
    
    # RSI (row 2 if present)
    rsi_row = 2 if show_rsi else None
    if show_rsi and rsi_row and 'RSI' in df.columns:
        fig.add_trace(go.Scatter(x=df['Date'], y=df['RSI'], name='RSI', line=dict(color='purple')), row=rsi_row, col=1)
        fig.add_hline(y=70, line_dash="dash", line_color="red", row=rsi_row, col=1)
        fig.add_hline(y=30, line_dash="dash", line_color="green", row=rsi_row, col=1)
    
    # MACD (row 3 if present)
    macd_row = 3 if show_macd else None
    if show_macd and macd_row and all(col in df.columns for col in ['MACD', 'MACD_Signal', 'MACD_Hist']):
        fig.add_trace(go.Scatter(x=df['Date'], y=df['MACD'], name='MACD', line=dict(color='blue')), row=macd_row, col=1)
        fig.add_trace(go.Scatter(x=df['Date'], y=df['MACD_Signal'], name='Signal', line=dict(color='red')), row=macd_row, col=1)
        fig.add_trace(go.Bar(x=df['Date'], y=df['MACD_Hist'], name='Histogram', marker_color='gray'), row=macd_row, col=1)
    
    fig.update_layout(height=600 + 200 * (num_rows - 1), title=f"{base}/{quote} Exchange Rate & Indicators (Last {days} Days)", xaxis_rangeslider_visible=False)
    fig.update_xaxes(title_text="Date", row=num_rows, col=1)
    fig.update_yaxes(title_text="Rate", row=1, col=1)
    if show_rsi and rsi_row:
        fig.update_yaxes(title_text="RSI", row=rsi_row, col=1, range=[0, 100])
    if show_macd and macd_row:
        fig.update_yaxes(title_text="MACD", row=macd_row, col=1)
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Data table with indicators
    st.subheader("Historical Data with Indicators")
    display_cols = ['Date', 'Rate']
    if show_sma and 'SMA_20' in df.columns: display_cols.append('SMA_20')
    if show_ema and 'EMA_20' in df.columns: display_cols.append('EMA_20')
    if show_bb and all(col in df.columns for col in ['BBU_20_2.0', 'BBM_20_2.0', 'BBL_20_2.0']):
        display_cols.extend(['BBU_20_2.0', 'BBM_20_2.0', 'BBL_20_2.0'])
    if show_rsi and 'RSI' in df.columns: display_cols.append('RSI')
    if show_macd and all(col in df.columns for col in ['MACD', 'MACD_Signal', 'MACD_Hist']):
        display_cols.extend(['MACD', 'MACD_Signal', 'MACD_Hist'])
    st.dataframe(df[display_cols], use_container_width=True)
else:
    st.info("No data available or insufficient data for indicators (need at least 30 days). Check API or increase days.")

# Footer
st.sidebar.markdown("---")
st.sidebar.info("Built with Streamlit & native Pandas. No external TA library needed. For real trading, use a licensed broker.")

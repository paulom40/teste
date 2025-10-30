import streamlit as st
import pandas as pd
import yfinance as yf
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime, timedelta
import numpy as np

# Set page config for professional look
st.set_page_config(
    page_title="Professional Forex Dashboard",
    page_icon="💱",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for professional styling
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #1f77b4;
    }
</style>
""", unsafe_allow_html=True)

# Header
st.markdown('<h1 class="main-header">💱 Professional Forex Trading Dashboard</h1>', unsafe_allow_html=True)
st.markdown("---")

# Sidebar: Inputs
st.sidebar.header("📊 Dashboard Controls")
base = st.sidebar.selectbox("Base Currency", ["USD", "EUR", "GBP", "JPY", "AUD", "CAD"], index=0)
quote = st.sidebar.selectbox("Quote Currency", ["EUR", "GBP", "JPY", "AUD", "CAD", "USD"], index=0)
days = st.sidebar.slider("Historical Days", min_value=5, max_value=365, value=30, step=5)

if base == quote:
    st.sidebar.error("Select different currencies.")
    st.stop()

ticker = f"{quote}{base}=X"

# Fetch data
@st.cache_data(ttl=300)  # Cache for 5 minutes
def fetch_data(ticker, days):
    try:
        data = yf.download(ticker, period=f"{days}d", interval="1d", progress=False)
        if data.empty:
            return pd.DataFrame()
        df = data.reset_index()
        df['Rate'] = 1 / df['Close']  # Quote per base
        df = df[['Date', 'Rate']].copy()
        df['Date'] = pd.to_datetime(df['Date'])
        return df.sort_values('Date').reset_index(drop=True)
    except Exception as e:
        st.error(f"Data fetch error: {e}")
        return pd.DataFrame()

df = fetch_data(ticker, days)

# Indicators
@st.cache_data
def compute_indicators(df):
    if len(df) < 20:
        return df
    df = df.copy()
    # SMA 20
    df['SMA_20'] = df['Rate'].rolling(window=20).mean()
    # RSI 14
    delta = df['Rate'].diff()
    gain = delta.where(delta > 0, 0).rolling(14).mean()
    loss = -delta.where(delta < 0, 0).rolling(14).mean()
    rs = gain / loss
    df['RSI'] = 100 - (100 / (1 + rs))
    return df

if not df.empty:
    df = compute_indicators(df)

# Signal Logic
signal = "HOLD"
color = "blue"
if not df.empty:
    rsi_val = df['RSI'].iloc[-1]
    sma_val = df['SMA_20'].iloc[-1]
    rate_val = df['Rate'].iloc[-1]
    if rsi_val < 30 and rate_val > sma_val:
        signal = "BUY"
        color = "green"
    elif rsi_val > 70 and rate_val < sma_val:
        signal = "SELL"
        color = "red"

# Main Layout: Tabs for Professional Sections
tab1, tab2, tab3 = st.tabs(["📈 Overview", "📊 Analysis", "💼 Trade Simulator"])

with tab1:
    # Metrics Row
    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        st.metric("Current Rate", f"{df['Rate'].iloc[-1]:.5f}" if not df.empty else "N/A")
        st.markdown('</div>', unsafe_allow_html=True)
    with col2:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        change_pct = ((df['Rate'].iloc[-1] - df['Rate'].iloc[0]) / df['Rate'].iloc[0] * 100) if not df.empty and len(df) > 1 else 0
        st.metric("Period Change", f"{change_pct:.2f}%", delta_color="normal")
        st.markdown('</div>', unsafe_allow_html=True)
    with col3:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        st.metric("Signal", signal, delta_color=color)
        st.markdown('</div>', unsafe_allow_html=True)
    
    # Quick Chart
    if not df.empty:
        fig = make_subplots(specs=[[{"secondary_y": True}]])
        fig.add_trace(go.Scatter(x=df['Date'], y=df['Rate'], name='Rate', line=dict(color='#1f77b4')), secondary_y=False)
        fig.add_trace(go.Scatter(x=df['Date'], y=df['SMA_20'], name='SMA 20', line=dict(color='#ff7f0e')), secondary_y=False)
        fig.add_trace(go.Scatter(x=df['Date'], y=df['RSI'], name='RSI', line=dict(color='#9467bd')), secondary_y=True)
        fig.update_xaxes(title_text="Date")
        fig.update_yaxes(title_text="Rate", secondary_y=False)
        fig.update_yaxes(title_text="RSI", secondary_y=True, range=[0, 100])
        fig.update_layout(title=f"{base}/{quote} Overview - Last {days} Days", height=500)
        st.plotly_chart(fig, use_container_width=True)

with tab2:
    st.header("Technical Analysis")
    if not df.empty:
        col1, col2 = st.columns(2)
        with col1:
            st.subheader("Rate & SMA")
            fig1 = go.Figure()
            fig1.add_trace(go.Scatter(x=df['Date'], y=df['Rate'], name='Rate', line=dict(color='blue')))
            fig1.add_trace(go.Scatter(x=df['Date'], y=df['SMA_20'], name='SMA 20', line=dict(color='orange')))
            fig1.update_layout(title="Price Trend", xaxis_title="Date", yaxis_title="Rate")
            st.plotly_chart(fig1, use_container_width=True)
        
        with col2:
            st.subheader("RSI Momentum")
            fig2 = go.Figure()
            fig2.add_trace(go.Scatter(x=df['Date'], y=df['RSI'], name='RSI', line=dict(color='purple')))
            fig2.add_hline(y=70, line_dash="dash", line_color="red", annotation_text="Overbought")
            fig2.add_hline(y=30, line_dash="dash", line_color="green", annotation_text="Oversold")
            fig2.update_layout(title="RSI Indicator", xaxis_title="Date", yaxis_title="RSI", yaxis_range=[0, 100])
            st.plotly_chart(fig2, use_container_width=True)
        
        # Data Table
        st.subheader("Historical Data")
        display_cols = ['Date', 'Rate', 'SMA_20', 'RSI']
        st.dataframe(df[display_cols].round(4), use_container_width=True)

with tab3:
    st.header("Trade Simulator")
    stake = st.number_input("Investment Stake ($)", min_value=100.0, max_value=100000.0, value=1000.0, step=100.0)
    risk_pct = st.slider("Risk per Trade (%)", 1.0, 5.0, 2.0)
    
    if st.button("Simulate Trade", type="primary") and signal != "HOLD":
        entry_price = df['Rate'].iloc[-1] if not df.empty else 1.0
        position_size = (stake * (risk_pct / 100)) / entry_price
        
        # Simulate outcome based on signal
        if signal == "BUY":
            pnl = position_size * entry_price * 0.03  # 3% assumed gain
            outcome = "Profitable"
        else:
            pnl = -position_size * entry_price * 0.02  # 2% assumed loss
            outcome = "Loss"
        
        st.success(f"**Simulation Result**: {signal} {base}/{quote} | Entry: {entry_price:.5f} | P&L: ${pnl:.2f} | Outcome: {outcome}")
        
        # Simple backtest summary
        if len(df) > 1:
            total_change = (df['Rate'].iloc[-1] - df['Rate'].iloc[0]) / df['Rate'].iloc[0] * 100
            st.metric("Hypothetical Return (Period)", f"{total_change:.2f}%")

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #666;'>
    <p>Professional Forex Dashboard | Built with Streamlit & yFinance | For Educational Use Only | Trading Involves Risk</p>
</div>
""", unsafe_allow_html=True)

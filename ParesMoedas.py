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

# Custom CSS for professional styling (Enhanced with more vibrant gradients)
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
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    .signal-container {
        text-align: center;
        padding: 1rem;
        border-radius: 0.75rem;
        font-size: 1.8rem;
        font-weight: bold;
        margin: 0.5rem 0;
        box-shadow: 0 4px 8px rgba(0,0,0,0.15);
        transition: transform 0.2s ease-in-out;
    }
    .signal-container:hover {
        transform: scale(1.05);
    }
    .signal-buy {
        background: linear-gradient(135deg, #28a745, #20c997);
        color: white;
        border: 2px solid #28a745;
    }
    .signal-sell {
        background: linear-gradient(135deg, #dc3545, #fd7e14);
        color: white;
        border: 2px solid #dc3545;
    }
    .signal-hold {
        background: linear-gradient(135deg, #17a2b8, #6f42c1);
        color: white;
        border: 2px solid #17a2b8;
    }
    .live-indicator {
        position: fixed;
        top: 10px;
        right: 10px;
        background: #28a745;
        color: white;
        padding: 0.5rem 1rem;
        border-radius: 20px;
        font-weight: bold;
        z-index: 1000;
        animation: pulse 2s infinite;
    }
    @keyframes pulse {
        0% { opacity: 1; }
        50% { opacity: 0.7; }
        100% { opacity: 1; }
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
interval = st.sidebar.selectbox("Intraday Interval", ["1m", "5m", "15m", "1h"], index=2)  # New: Interval selection
period = st.sidebar.selectbox("Period", ["1d", "5d", "1mo"], index=1)  # New: Period for intraday

# Live Mode Toggle
live_mode = st.sidebar.checkbox("🚨 Enable Live Mode (Auto-Refresh & Alerts every 60s)")

if base == quote:
    st.sidebar.error("Select different currencies.")
    st.stop()

ticker = f"{quote}{base}=X"

# Fetch data (Updated for intraday)
@st.cache_data(ttl=60 if live_mode else 300)  # Shorter cache in live mode
def fetch_data(ticker, period, interval):
    try:
        data = yf.download(ticker, period=period, interval=interval, progress=False)
        if data.empty:
            return pd.DataFrame()
        df = data.reset_index()
        df['Rate'] = 1 / df['Close']  # Quote per base
        df = df[['Datetime', 'Rate', 'Open', 'High', 'Low', 'Volume']].copy()  # Include OHLCV
        df['Datetime'] = pd.to_datetime(df['Datetime'])
        return df.sort_values('Datetime').reset_index(drop=True)
    except Exception as e:
        st.error(f"Data fetch error: {e}")
        return pd.DataFrame()

df = fetch_data(ticker, period, interval)

# Indicators (Added MACD)
@st.cache_data
def compute_indicators(df):
    if len(df) < 26:  # Min for MACD
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
    # MACD
    ema12 = df['Rate'].ewm(span=12).mean()
    ema26 = df['Rate'].ewm(span=26).mean()
    df['MACD'] = ema12 - ema26
    df['MACD_Signal'] = df['MACD'].ewm(span=9).mean()
    df['MACD_Hist'] = df['MACD'] - df['MACD_Signal']
    return df

if not df.empty:
    df = compute_indicators(df)

# Signal Logic
signal = "HOLD"
signal_class = "signal-hold"
signal_icon = "⏸️"
if not df.empty:
    rsi_val = df['RSI'].iloc[-1]
    sma_val = df['SMA_20'].iloc[-1]
    rate_val = df['Rate'].iloc[-1]
    macd_val = df['MACD'].iloc[-1]
    macd_signal = df['MACD_Signal'].iloc[-1]
    if rsi_val < 30 and rate_val > sma_val and macd_val > macd_signal:
        signal = "BUY"
        signal_class = "signal-buy"
        signal_icon = "📈"
    elif rsi_val > 70 and rate_val < sma_val and macd_val < macd_signal:
        signal = "SELL"
        signal_class = "signal-sell"
        signal_icon = "📉"

# Real-Time Alerts Logic
if 'last_signal' not in st.session_state:
    st.session_state.last_signal = signal

if signal != st.session_state.last_signal:
    if signal == "BUY":
        st.toast("🚨 ALERT: BUY Signal Triggered! RSI low + Rate > SMA + MACD Bullish.", icon="📈")
    elif signal == "SELL":
        st.toast("🚨 ALERT: SELL Signal Triggered! RSI high + Rate < SMA + MACD Bearish.", icon="📉")
    else:
        st.toast("📊 Signal Updated: HOLD", icon="⏸️")
    st.session_state.last_signal = signal

# Live Mode Polling (Non-blocking with JS)
if live_mode:
    st.markdown('<div class="live-indicator">🔴 LIVE MODE ACTIVE</div>', unsafe_allow_html=True)
    # Non-blocking JS timer for auto-rerun every 60s
    st.components.v1.html(
        f"""
        <script>
            setTimeout(() => {{
                window.parent.document.querySelector('.stAppViewContainer').dispatchEvent(new Event('rerun'));
            }}, 60000);
        </script>
        """,
        height=0
    )

# Manual Refresh
if st.sidebar.button("🔄 Refresh Now"):
    st.rerun()

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
        st.markdown(f'<div class="{signal_class} signal-container">{signal_icon} {signal}</div>', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)
    
    # Quick Chart (Updated with MACD subplot)
    if not df.empty:
        fig = make_subplots(
            rows=2, cols=1,
            shared_xaxes=True,
            vertical_spacing=0.1,
            subplot_titles=(f"{base}/{quote} Price & Indicators ({interval} interval)", "MACD"),
            row_heights=[0.7, 0.3]
        )
        fig.add_trace(go.Scatter(x=df['Datetime'], y=df['Rate'], name='Rate', line=dict(color='#1f77b4')), row=1, col=1)
        fig.add_trace(go.Scatter(x=df['Datetime'], y=df['SMA_20'], name='SMA 20', line=dict(color='#ff7f0e')), row=1, col=1)
        fig.add_trace(go.Scatter(x=df['Datetime'], y=df['RSI'], name='RSI', line=dict(color='#9467bd'), yaxis="y2"), row=1, col=1)
        # MACD
        fig.add_trace(go.Scatter(x=df['Datetime'], y=df['MACD'], name='MACD', line=dict(color='blue')), row=2, col=1)
        fig.add_trace(go.Scatter(x=df['Datetime'], y=df['MACD_Signal'], name='Signal', line=dict(color='red')), row=2, col=1)
        fig.add_trace(go.Bar(x=df['Datetime'], y=df['MACD_Hist'], name='Histogram', marker_color='gray'), row=2, col=1)
        fig.update_xaxes(title_text="Date", row=2, col=1)
        fig.update_yaxes(title_text="Rate", row=1, col=1, secondary_y=False)
        fig.update_yaxes(title_text="RSI", row=1, col=1, secondary_y=True, range=[0, 100])
        fig.update_yaxes(title_text="MACD", row=2, col=1)
        fig.update_layout(title=f"{base}/{quote} Overview - {period} ({interval})", height=600)
        st.plotly_chart(fig, use_container_width=True)

with tab2:
    st.header("Technical Analysis")
    if not df.empty:
        col1, col2 = st.columns(2)
        with col1:
            st.subheader("Rate & SMA")
            fig1 = go.Figure()
            fig1.add_trace(go.Scatter(x=df['Datetime'], y=df['Rate'], name='Rate', line=dict(color='blue')))
            fig1.add_trace(go.Scatter(x=df['Datetime'], y=df['SMA_20'], name='SMA 20', line=dict(color='orange')))
            fig1.update_layout(title="Price Trend", xaxis_title="Datetime", yaxis_title="Rate")
            st.plotly_chart(fig1, use_container_width=True)
        
        with col2:
            st.subheader("RSI Momentum")
            fig2 = go.Figure()
            fig2.add_trace(go.Scatter(x=df['Datetime'], y=df['RSI'], name='RSI', line=dict(color='purple')))
            fig2.add_hline(y=70, line_dash="dash", line_color="red", annotation_text="Overbought")
            fig2.add_hline(y=30, line_dash="dash", line_color="green", annotation_text="Oversold")
            fig2.update_layout(title="RSI Indicator", xaxis_title="Datetime", yaxis_title="RSI", yaxis_range=[0, 100])
            st.plotly_chart(fig2, use_container_width=True)
        
        # New: MACD Chart
        st.subheader("MACD Momentum")
        fig3 = make_subplots(specs=[[{"secondary_y": False}]])
        fig3.add_trace(go.Scatter(x=df['Datetime'], y=df['MACD'], name='MACD', line=dict(color='blue')), secondary_y=False)
        fig3.add_trace(go.Scatter(x=df['Datetime'], y=df['MACD_Signal'], name='Signal', line=dict(color='red')), secondary_y=False)
        fig3.add_trace(go.Bar(x=df['Datetime'], y=df['MACD_Hist'], name='Histogram', marker_color='gray'), secondary_y=False)
        fig3.update_layout(title="MACD Indicator (12,26,9)", xaxis_title="Datetime", yaxis_title="MACD")
        st.plotly_chart(fig3, use_container_width=True)
        
        # Data Table (Updated with OHLCV)
        st.subheader("Intraday Data")
        display_cols = ['Datetime', 'Rate', 'Open', 'High', 'Low', 'Volume', 'SMA_20', 'RSI', 'MACD', 'MACD_Signal', 'MACD_Hist']
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

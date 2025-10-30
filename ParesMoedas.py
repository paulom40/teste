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
    page_icon="Forex Trading",
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
    .trade-execution {
        background: #fff3cd;
        border: 1px solid #ffeaa7;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

# Header
st.markdown('<h1 class="main-header">Forex Trading Professional Forex Trading Dashboard</h1>', unsafe_allow_html=True)
st.markdown("---")

# Sidebar: Inputs
st.sidebar.header("Dashboard Controls")
base = st.sidebar.selectbox("Base Currency", ["USD", "EUR", "GBP", "JPY", "AUD", "CAD"], index=0)
quote = st.sidebar.selectbox("Quote Currency", ["EUR", "GBP", "JPY", "AUD", "CAD", "USD"], index=0)
interval = st.sidebar.selectbox("Intraday Interval", ["1m", "5m", "15m", "1h"], index=2)
period = st.sidebar.selectbox("Period", ["1d", "5d", "1mo"], index=1)

# Live Mode & Auto-Trade Toggle
live_mode = st.sidebar.checkbox("Enable Live Mode (Auto-Refresh every 60s)")
auto_trade = st.sidebar.checkbox("Enable Auto-Trade (Simulated Execution on Signals)")

if base == quote:
    st.sidebar.error("Select different currencies.")
    st.stop()

ticker = f"{quote}{base}=X"

# Predefined top 10 pairs for scanning
TOP_PAIRS = [
    ("EUR", "USD"), ("GBP", "USD"), ("USD", "JPY"), ("AUD", "USD"),
    ("USD", "CAD"), ("USD", "CHF"), ("NZD", "USD"), ("EUR", "GBP"),
    ("EUR", "JPY"), ("GBP", "JPY")
]

# Fetch data
@st.cache_data(ttl=60 if live_mode else 300)
def fetch_data(ticker, period, interval):
    try:
        data = yf.download(ticker, period=period, interval=interval, progress=False)
        if data.empty:
            return pd.DataFrame()
        df = data.reset_index()
        df['Rate'] = 1 / df['Close']
        df = df[['Datetime', 'Rate']].copy()
        df['Datetime'] = pd.to_datetime(df['Datetime'])
        return df.sort_values('Datetime').reset_index(drop=True)
    except Exception as e:
        st.error(f"Data fetch error: {e}")
        return pd.DataFrame()

# Compute indicators
@st.cache_data
def compute_indicators(df):
    if len(df) < 26:
        return df
    df = df.copy()
    df['SMA_20'] = df['Rate'].rolling(20).mean()
    delta = df['Rate'].diff()
    gain = delta.where(delta > 0, 0).rolling(14).mean()
    loss = -delta.where(delta < 0, 0).rolling(14).mean()
    rs = gain / loss
    df['RSI'] = 100 - (100 / (1 + rs))
    ema12 = df['Rate'].ewm(span=12).mean()
    ema26 = df['Rate'].ewm(span=26).mean()
    df['MACD'] = ema12 - ema26
    df['MACD_Signal'] = df['MACD'].ewm(span=9).mean()
    return df

# Analyze single pair
def analyze_pair(base_cur, quote_cur):
    ticker = f"{quote_cur}{base_cur}=X"
    df = fetch_data(ticker, period, interval)
    if df.empty or len(df) < 26:
        return None
    df = compute_indicators(df)
    rate = df['Rate'].iloc[-1]
    rsi = df['RSI'].iloc[-1]
    sma = df['SMA_20'].iloc[-1]
    macd = df['MACD'].iloc[-1]
    macd_sig = df['MACD_Signal'].iloc[-1]
    
    signal = "HOLD"
    strength = 0.0
    if rsi < 30 and rate > sma and macd > macd_sig:
        signal = "BUY"
        strength = 0.8
    elif rsi > 70 and rate < sma and macd < macd_sig:
        signal = "SELL"
        strength = 0.7
    
    return {
        "pair": f"{base_cur}/{quote_cur}",
        "rate": rate,
        "rsi": rsi,
        "signal": signal,
        "strength": strength
    }

# Scan all pairs
st.subheader("Real-Time Signal Scanner: Top 10 Pairs")
scan_results = []
for b, q in TOP_PAIRS:
    result = analyze_pair(b, q)
    if result:
        scan_results.append(result)

if scan_results:
    scan_df = pd.DataFrame(scan_results)
    scan_df = scan_df.sort_values("strength", ascending=False)
    
    # Display scanner table
    st.dataframe(
        scan_df[['pair', 'rate', 'rsi', 'signal', 'strength']].round(4),
        use_container_width=True
    )
    
    # Auto-Trade Execution
    if auto_trade and live_mode:
        strong_signals = scan_df[scan_df['strength'] > 0.6]
        if not strong_signals.empty:
            st.markdown("### Auto-Trade Execution (Simulated)")
            executed = []
            for _, row in strong_signals.iterrows():
                entry = row['rate']
                sl = entry * (0.98 if row['signal'] == "BUY" else 1.02)
                tp = entry * (1.04 if row['signal'] == "BUY" else 0.96)
                size = 0.5  # mini lot
                pnl = size * 100000 * (0.02 if row['signal'] == "BUY" else -0.02)
                executed.append({
                    "Pair": row['pair'],
                    "Signal": row['signal'],
                    "Entry": f"{entry:.5f}",
                    "SL": f"{sl:.5f}",
                    "TP": f"{tp:.5f}",
                    "P&L": f"${pnl:.0f}",
                    "Status": "Executed"
                })
                st.toast(f"EXECUTED {row['signal']} {row['pair']} @ {entry:.5f}")
            
            exec_df = pd.DataFrame(executed)
            st.dataframe(exec_df, use_container_width=True)
            total_pnl = sum(float(p[1:]) for p in exec_df['P&L'])
            st.success(f"Total Simulated P&L: ${total_pnl:.0f}")

# Live Mode Polling
if live_mode:
    st.markdown('<div class="live-indicator">LIVE MODE ACTIVE</div>', unsafe_allow_html=True)
    st.components.v1.html(
        """
        <script>
            setTimeout(() => {
                window.parent.document.querySelector('.stAppViewContainer').dispatchEvent(new Event('rerun'));
            }, 60000);
        </script>
        """,
        height=0
    )

# Manual Refresh
if st.sidebar.button("Refresh Now"):
    st.rerun()

# Main Tabs
tab1, tab2, tab3 = st.tabs(["Overview", "Analysis", "Trade Simulator"])

with tab1:
    df = fetch_data(ticker, period, interval)
    if not df.empty:
        df = compute_indicators(df)
        col1, col2, col3 = st.columns(3)
        with col1:
            st.markdown('<div class="metric-card">', unsafe_allow_html=True)
            st.metric("Current Rate", f"{df['Rate'].iloc[-1]:.5f}")
            st.markdown('</div>', unsafe_allow_html=True)
        with col2:
            st.markdown('<div class="metric-card">', unsafe_allow_html=True)
            change = (df['Rate'].iloc[-1] - df['Rate'].iloc[0]) / df['Rate'].iloc[0] * 100
            st.metric("Period Change", f"{change:.2f}%")
            st.markdown('</div>', unsafe_allow_html=True)
        with col3:
            signal = "BUY" if df['RSI'].iloc[-1] < 30 and df['Rate'].iloc[-1] > df['SMA_20'].iloc[-1] and df['MACD'].iloc[-1] > df['MACD_Signal'].iloc[-1] else "SELL" if df['RSI'].iloc[-1] > 70 and df['Rate'].iloc[-1] < df['SMA_20'].iloc[-1] and df['MACD'].iloc[-1] < df['MACD_Signal'].iloc[-1] else "HOLD"
            cls = "signal-buy" if signal == "BUY" else "signal-sell" if signal == "SELL" else "signal-hold"
            icon = "Up" if signal == "BUY" else "Down" if signal == "SELL" else "Pause"
            st.markdown(f'<div class="metric-card"><div class="{cls} signal-container">{icon} {signal}</div></div>', unsafe_allow_html=True)
        
        fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.1,
                           subplot_titles=(f"{base}/{quote} Price & Indicators", "MACD"), row_heights=[0.7, 0.3])
        fig.add_trace(go.Scatter(x=df['Datetime'], y=df['Rate'], name='Rate', line=dict(color='#1f77b4')), row=1, col=1)
        fig.add_trace(go.Scatter(x=df['Datetime'], y=df['SMA_20'], name='SMA 20', line=dict(color='#ff7f0e')), row=1, col=1)
        fig.add_trace(go.Scatter(x=df['Datetime'], y=df['RSI'], name='RSI', line=dict(color='#9467bd'), yaxis="y2"), row=1, col=1)
        fig.add_trace(go.Scatter(x=df['Datetime'], y=df['MACD'], name='MACD', line=dict(color='blue')), row=2, col=1)
        fig.add_trace(go.Scatter(x=df['Datetime'], y=df['MACD_Signal'], name='Signal', line=dict(color='red')), row=2, col=1)
        fig.update_layout(height=600)
        st.plotly_chart(fig, use_container_width=True)

with tab2:
    st.header("Technical Analysis")
    if not df.empty:
        col1, col2 = st.columns(2)
        with col1:
            st.subheader("Price & SMA")
            fig1 = go.Figure()
            fig1.add_trace(go.Scatter(x=df['Datetime'], y=df['Rate'], name='Rate'))
            fig1.add_trace(go.Scatter(x=df['Datetime'], y=df['SMA_20'], name='SMA 20'))
            st.plotly_chart(fig1, use_container_width=True)
        with col2:
            st.subheader("RSI")
            fig2 = go.Figure()
            fig2.add_trace(go.Scatter(x=df['Datetime'], y=df['RSI'], name='RSI'))
            fig2.add_hline(y=70, line_dash="dash", line_color="red")
            fig2.add_hline(y=30, line_dash="dash", line_color="green")
            st.plotly_chart(fig2, use_container_width=True)

with tab3:
    st.header("Manual Trade Simulator")
    stake = st.number_input("Stake ($)", 100.0, 10000.0, 1000.0)
    if st.button("Simulate BUY"):
        st.success(f"Simulated BUY @ {df['Rate'].iloc[-1]:.5f} | P&L: +${stake*0.03:.0f}")
    if st.button("Simulate SELL"):
        st.success(f"Simulated SELL @ {df['Rate'].iloc[-1]:.5f} | P&L: -${stake*0.02:.0f}")

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #666;'>
    <p>Professional Forex Dashboard | Built with Streamlit & yFinance | For Educational Use Only | Trading Involves Risk</p>
</div>
""", unsafe_allow_html=True)

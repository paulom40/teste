import streamlit as st
import pandas as pd
import yfinance as yf
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np

# ------------------------------------------------------------------ #
# Page config & minified CSS
# ------------------------------------------------------------------ #
st.set_page_config(page_title="Forex Dashboard", page_icon="Forex", layout="wide", initial_sidebar_state="expanded")
st.markdown("""
<style>
    .main-header{font-size:3rem;color:#1f77b4;text-align:center;margin-bottom:2rem}
    .metric-card{background:#f0f2f6;padding:1rem;border-radius:.5rem;border-left:4px solid #1f77b4;box-shadow:0 2px 4px rgba(0,0,0,.1)}
    .signal-container{text-align:center;padding:1rem;border-radius:.75rem;font-size:1.8rem;font-weight:bold;margin:.5rem 0;box-shadow:0 4px 8px rgba(0,0,0,.15);transition:transform .2s}
    .signal-container:hover{transform:scale(1.05)}
    .signal-buy{background:linear-gradient(135deg,#28a745,#20c997);color:#fff;border:2px solid #28a745}
    .signal-sell{background:linear-gradient(135deg,#dc3545,#fd7e14);color:#fff;border:2px solid #dc3545}
    .signal-hold{background:linear-gradient(135deg,#17a2b8,#6f42c1);color:#fff;border:2px solid #17a2b8}
    .live-indicator{position:fixed;top:10px;right:10px;background:#28a745;color:#fff;padding:.5rem 1rem;border-radius:20px;font-weight:bold;z-index:1000;animation:pulse 2s infinite}
    @keyframes pulse{0%,100%{opacity:1}50%{opacity:.7}}
</style>
""", unsafe_allow_html=True)

st.markdown('<h1 class="main-header">Professional Forex Dashboard</h1>', unsafe_allow_html=True)
st.markdown("---")

# ------------------------------------------------------------------ #
# Sidebar
# ------------------------------------------------------------------ #
st.sidebar.header("Controls")
base   = st.sidebar.selectbox("Base",  ["USD","EUR","GBP","JPY","AUD","CAD"], index=0)
quote  = st.sidebar.selectbox("Quote", ["EUR","GBP","JPY","AUD","CAD","USD"], index=0)
interval = st.sidebar.selectbox("Interval", ["1m","5m","15m","1h"], index=2)
period   = st.sidebar.selectbox("Period",   ["1d","5d","1mo"], index=1)
live_mode = st.sidebar.checkbox("Live Mode (60 s)")
auto_trade = st.sidebar.checkbox("Auto-Trade (simulated)")

if base == quote:
    st.sidebar.error("Pick different currencies.")
    st.stop()

ticker = f"{quote}{base}=X"

TOP_PAIRS = [
    ("EUR","USD"),("GBP","USD"),("USD","JPY"),("AUD","USD"),
    ("USD","CAD"),("USD","CHF"),("NZD","USD"),("EUR","GBP"),
    ("EUR","JPY"),("GBP","JPY")
]

# ------------------------------------------------------------------ #
# Data fetch (keep Close!)
# ------------------------------------------------------------------ #
@st.cache_data(ttl=60 if live_mode else 300, show_spinner=False)
def fetch_data(ticker, period, interval):
    data = yf.download(ticker, period=period, interval=interval,
                       progress=False, auto_adjust=True)
    if data.empty:
        return pd.DataFrame()
    df = data[['Open','High','Low','Close']].copy()
    df['Rate'] = 1 / df['Close']
    df = df.reset_index()[['Datetime','Rate','Open','High','Low','Close']]
    df['Datetime'] = pd.to_datetime(df['Datetime'])
    return df

# ------------------------------------------------------------------ #
# Price-action on the **last 3 candles only**
# ------------------------------------------------------------------ #
def detect_price_action(last3: pd.DataFrame) -> str:
    if len(last3) < 3:
        return "No Pattern"
    c, p1 = last3.iloc[-1], last3.iloc[-2]

    # Bullish Engulfing
    if (p1['Close'] < p1['Open'] and
        c['Open']   < p1['Close'] and
        c['Close']  > p1['Open']  and
        c['Close']  > c['Open']):
        return "Bullish Engulfing"

    # Bearish Engulfing
    if (p1['Close'] > p1['Open'] and
        c['Open']   > p1['Close'] and
        c['Close']  < p1['Open']  and
        c['Close']  < c['Open']):
        return "Bearish Engulfing"

    body   = abs(c['Close'] - c['Open'])
    lower  = min(c['Open'], c['Close']) - c['Low']
    upper  = c['High'] - max(c['Open'], c['Close'])

    if lower > 2*body and upper < body and c['Close'] > c['Open']:
        return "Hammer"
    if upper > 2*body and lower < body and c['Close'] < c['Open']:
        return "Shooting Star"

    return "No Pattern"

# ------------------------------------------------------------------ #
# Indicators (vectorized, single pass)
# ------------------------------------------------------------------ #
@st.cache_data(show_spinner=False)
def compute_indicators(df: pd.DataFrame) -> pd.DataFrame:
    if len(df) < 26:
        return df

    r = df['Rate']
    df['SMA_20'] = r.rolling(20).mean()

    delta = r.diff()
    gain  = delta.clip(lower=0).rolling(14).mean()
    loss  = -delta.clip(upper=0).rolling(14).mean()
    rs    = gain / loss.replace(0, np.nan)
    df['RSI'] = 100 - (100 / (1 + rs))

    ema12 = r.ewm(span=12, adjust=False).mean()
    ema26 = r.ewm(span=26, adjust=False).mean()
    macd  = ema12 - ema26
    df['MACD'] = macd
    df['MACD_Signal'] = macd.ewm(span=9, adjust=False).mean()

    # Price action only on the newest candle
    df['Price_Action'] = "No Pattern"
    if len(df) >= 3:
        df.iloc[-1, df.columns.get_loc('Price_Action')] = detect_price_action(df.tail(3))

    return df[['Datetime','Rate','SMA_20','RSI','MACD','MACD_Signal','Price_Action']]

# ------------------------------------------------------------------ #
# Pair analysis (fast)
# ------------------------------------------------------------------ #
def analyze_pair(b, q):
    df = fetch_data(f"{q}{b}=X", period, interval)
    if df.empty or len(df) < 26:
        return None
    df = compute_indicators(df)
    r  = df.iloc[-1]

    signal = "HOLD"
    strength = 0.0
    pattern = r['Price_Action']

    if (r['RSI'] < 30 and r['Rate'] > r['SMA_20'] and r['MACD'] > r['MACD_Signal'] and
        pattern in ("Bullish Engulfing","Hammer")):
        signal, strength = "BUY", 0.8 + (0.2 if pattern != "No Pattern" else 0)
    elif (r['RSI'] > 70 and r['Rate'] < r['SMA_20'] and r['MACD'] < r['MACD_Signal'] and
          pattern in ("Bearish Engulfing","Shooting Star")):
        signal, strength = "SELL", 0.7 + (0.3 if pattern != "No Pattern" else 0)

    return {
        "pair": f"{b}/{q}",
        "rate": r['Rate'],
        "rsi":  r['RSI'],
        "pattern": pattern,
        "signal": signal,
        "strength": min(strength, 1.0)
    }

# ------------------------------------------------------------------ #
# Scanner
# ------------------------------------------------------------------ #
st.subheader("Real-Time Scanner – Top 10 Pairs")
with st.spinner("Scanning…"):
    results = [analyze_pair(b, q) for b, q in TOP_PAIRS]
    results = [r for r in results if r]

if results:
    scan = pd.DataFrame(results).sort_values("strength", ascending=False)
    st.dataframe(scan.round(4), use_container_width=True)

    # ---------- Auto-trade (simulated) ----------
    if auto_trade and live_mode and scan['strength'].max() > 0.7:
        strong = scan[scan['strength'] > 0.7]
        st.markdown("### Auto-Trade (simulated)")
        execs = []
        for _, row in strong.iterrows():
            e = row['rate']
            sl = e * (0.98 if row['signal'] == "BUY" else 1.02)
            tp = e * (1.04 if row['signal'] == "BUY" else 0.96)
            pnl = 50 if row['signal'] == "BUY" else -40
            execs.append({
                "Pair": row['pair'],
                "Signal": row['signal'],
                "Entry": f"{e:.5f}",
                "SL": f"{sl:.5f}",
                "TP": f"{tp:.5f}",
                "P&L": f"${pnl}"
            })
            st.toast(f"EXECUTED {row['signal']} {row['pair']} @ {e:.5f}")
        st.dataframe(pd.DataFrame(execs), use_container_width=True)
        total = sum(int(p['P&L'][1:]) * (1 if 'BUY' in p['Signal'] else -1) for p in execs)
        st.success(f"Total Simulated P&L: ${total}")

# ------------------------------------------------------------------ #
# Live refresh (non-blocking)
# ------------------------------------------------------------------ #
if live_mode:
    st.markdown('<div class="live-indicator">LIVE</div>', unsafe_allow_html=True)
    st.components.v1.html(
        "<script>setTimeout(() => window.parent.document.querySelector('.stAppViewContainer').dispatchEvent(new Event('rerun')), 60000);</script>",
        height=0)

if st.sidebar.button("Refresh Now"):
    st.rerun()

# ------------------------------------------------------------------ #
# Tabs – Overview / Analysis / Simulator
# ------------------------------------------------------------------ #
tab1, tab2, tab3 = st.tabs(["Overview", "Analysis", "Simulator"])

with tab1:
    df = fetch_data(ticker, period, interval)
    if not df.empty:
        df = compute_indicators(df)
        r  = df.iloc[-1]

        c1, c2, c3 = st.columns(3)
        with c1:
            st.markdown(f'<div class="metric-card"><b>Rate:</b> {r["Rate"]:.5f}</div>', unsafe_allow_html=True)
        with c2:
            chg = (r["Rate"] - df["Rate"].iloc[0]) / df["Rate"].iloc[0] * 100
            st.markdown(f'<div class="metric-card"><b>Change:</b> {chg:+.2f}%</div>', unsafe_allow_html=True)
        with c3:
            sig = ("BUY" if r["RSI"] < 30 and r["Rate"] > r["SMA_20"] and r["MACD"] > r["MACD_Signal"] and
                   r["Price_Action"] in ("Bullish Engulfing","Hammer")
                   else "SELL" if r["RSI"] > 70 and r["Rate"] < r["SMA_20"] and r["MACD"] < r["MACD_Signal"] and
                   r["Price_Action"] in ("Bearish Engulfing","Shooting Star")
                   else "HOLD")
            cls = "signal-buy" if sig == "BUY" else "signal-sell" if sig == "SELL" else "signal-hold"
            ico = "Up" if sig == "BUY" else "Down" if sig == "SELL" else "Pause"
            st.markdown(f'<div class="metric-card"><div class="{cls} signal-container">{ico} {sig}</div></div>', unsafe_allow_html=True)

        fig = make_subplots(rows=2, cols=1, shared_xaxes=True, row_heights=[0.7,0.3])
        fig.add_trace(go.Scatter(x=df['Datetime'], y=df['Rate'], name='Rate'), row=1, col=1)
        fig.add_trace(go.Scatter(x=df['Datetime'], y=df['SMA_20'], name='SMA'), row=1, col=1)
        fig.add_trace(go.Scatter(x=df['Datetime'], y=df['RSI'], name='RSI'), row=1, col=1)
        fig.add_trace(go.Scatter(x=df['Datetime'], y=df['MACD'], name='MACD'), row=2, col=1)
        fig.add_trace(go.Scatter(x=df['Datetime'], y=df['MACD_Signal'], name='Signal'), row=2, col=1)
        if r['Price_Action'] != "No Pattern":
            fig.add_annotation(x=df['Datetime'].iloc[-1], y=r['Rate'], text=r['Price_Action'], row=1, col=1)
        st.plotly_chart(fig, use_container_width=True)

with tab2:
    if not df.empty:
        c1, c2 = st.columns(2)
        with c1:
            st.subheader("Price & SMA")
            f = go.Figure()
            f.add_trace(go.Scatter(x=df['Datetime'], y=df['Rate'], name='Rate'))
            f.add_trace(go.Scatter(x=df['Datetime'], y=df['SMA_20'], name='SMA'))
            st.plotly_chart(f, use_container_width=True)
        with c2:
            st.subheader("RSI")
            f = go.Figure()
            f.add_trace(go.Scatter(x=df['Datetime'], y=df['RSI'], name='RSI'))
            f.add_hline(y=70, line_dash="dash", line_color="red")
            f.add_hline(y=30, line_dash="dash", line_color="green")
            st.plotly_chart(f, use_container_width=True)

with tab3:
    stake = st.number_input("Stake ($)", 100.0, 10000.0, 1000.0)
    if st.button("Simulate BUY"):
        st.success(f"Simulated BUY @ {df['Rate'].iloc[-1]:.5f} | P&L +${stake*0.03:.0f}")
    if st.button("Simulate SELL"):
        st.success(f"Simulated SELL @ {df['Rate'].iloc[-1]:.5f} | P&L -${stake*0.02:.0f}")

# ------------------------------------------------------------------ #
# Footer
# ------------------------------------------------------------------ #
st.markdown("---")
st.markdown("<p style='text-align:center;color:#666'>Professional Forex Dashboard | Streamlit + yFinance | Educational Use Only</p>", unsafe_allow_html=True)

import streamlit as st
import pandas as pd
import yfinance as yf
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np

# ------------------------------------------------------------------ #
# Page & CSS
# ------------------------------------------------------------------ #
st.set_page_config(page_title="Forex Scanner", page_icon="Chart", layout="wide", initial_sidebar_state="expanded")
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

st.markdown('<h1 class="main-header">Real-Time Forex Scanner</h1>', unsafe_allow_html=True)
st.markdown("---")

# ------------------------------------------------------------------ #
# Sidebar
# ------------------------------------------------------------------ #
st.sidebar.header("Controls")
base   = st.sidebar.selectbox("Base",  ["USD","EUR","GBP","JPY","AUD","CAD"], index=0)
quote  = st.sidebar.selectbox("Quote", ["EUR","GBP","JPY","AUD","CAD","USD"], index=0)
interval = st.sidebar.selectbox("Interval", ["1m","5m","15m","1h"], index=2)
period   = st.sidebar.selectbox("Period",   ["1d","5d","1mo"], index=1)
live_mode = st.sidebar.checkbox("Live Mode (60s)")
auto_trade = st.sidebar.checkbox("Auto-Trade (Simulated)")

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
# Fetch Data
# ------------------------------------------------------------------ #
@st.cache_data(ttl=60 if live_mode else 300, show_spinner=False)
def fetch_data(ticker, period, interval):
    try:
        data = yf.download(ticker, period=period, interval=interval,
                           progress=False, auto_adjust=True)
        if data.empty:
            return pd.DataFrame()
        df = data[['Open','High','Low','Close']].copy()
        df['Rate'] = 1.0 / df['Close']
        df = df.reset_index()
        df['Datetime'] = pd.to_datetime(df['Datetime'])
        return df[['Datetime','Rate','Open','High','Low','Close']]
    except Exception:
        return pd.DataFrame()

# ------------------------------------------------------------------ #
# Price Action – **scalars only**
# ------------------------------------------------------------------ #
def detect_price_action(last3: pd.DataFrame) -> str:
    if len(last3) < 3:
        return "No Pattern"

    c  = last3.iloc[-1]   # current
    p1 = last3.iloc[-2]   # previous

    # ----> scalar extraction + NaN guard <----
    try:
        o, h, l, cl = float(c['Open']), float(c['High']), float(c['Low']), float(c['Close'])
        po, ph, pl, pcl = float(p1['Open']), float(p1['High']), float(p1['Low']), float(p1['Close'])
    except (ValueError, TypeError):
        return "No Pattern"

    # Bullish Engulfing
    if (pcl < po and o < pcl and cl > po and cl > o):
        return "Bullish Engulfing"

    # Bearish Engulfing
    if (pcl > po and o > pcl and cl < po and cl < o):
        return "Bearish Engulfing"

    body  = abs(cl - o)
    lower = min(o, cl) - l
    upper = h - max(o, cl)

    if lower > 2 * body and upper < body and cl > o:
        return "Hammer"
    if upper > 2 * body and lower < body and cl < o:
        return "Shooting Star"

    return "No Pattern"

# ------------------------------------------------------------------ #
# Indicators (vectorized)
# ------------------------------------------------------------------ #
@st.cache_data(show_spinner=False)
def compute_indicators(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty or len(df) < 26:
        df['Price_Action'] = "No Pattern"
        return df

    r = df['Rate']
    df['SMA_20'] = r.rolling(20).mean()
    delta = r.diff()
    gain = delta.clip(lower=0).rolling(14).mean()
    loss = -delta.clip(upper=0).rolling(14).mean()
    rs = gain / loss.replace(0, np.nan)
    df['RSI'] = 100 - (100 / (1 + rs))

    ema12 = r.ewm(span=12, adjust=False).mean()
    ema26 = r.ewm(span=26, adjust=False).mean()
    macd = ema12 - ema26
    df['MACD'] = macd
    df['MACD_Signal'] = macd.ewm(span=9, adjust=False).mean()

    # ----> price action on the newest candle only <----
    df['Price_Action'] = "No Pattern"
    if len(df) >= 3:
        pattern = detect_price_action(df.tail(3))
        df.loc[df.index[-1], 'Price_Action'] = pattern

    return df[['Datetime','Rate','SMA_20','RSI','MACD','MACD_Signal','Price_Action']]

# ------------------------------------------------------------------ #
# Analyze Pair
# ------------------------------------------------------------------ #
def analyze_pair(b, q):
    try:
        df = fetch_data(f"{q}{b}=X", period, interval)
        if df.empty or len(df) < 26:
            return None
        df = compute_indicators(df)
        r = df.iloc[-1]

        signal = "HOLD"
        strength = 0.0
        pattern = r['Price_Action']

        if (r['RSI'] < 30 and r['Rate'] > r['SMA_20'] and r['MACD'] > r['MACD_Signal'] and
            pattern in ("Bullish Engulfing", "Hammer")):
            signal = "BUY"
            strength = 0.8 + (0.2 if pattern != "No Pattern" else 0)
        elif (r['RSI'] > 70 and r['Rate'] < r['SMA_20'] and r['MACD'] < r['MACD_Signal'] and
              pattern in ("Bearish Engulfing", "Shooting Star")):
            signal = "SELL"
            strength = 0.7 + (0.3 if pattern != "No Pattern" else 0)

        return {
            "pair": f"{b}/{q}",
            "rate": r['Rate'],
            "rsi": r['RSI'],
            "pattern": pattern,
            "signal": signal,
            "strength": min(strength, 1.0)
        }
    except Exception:
        return None

# ------------------------------------------------------------------ #
# Scanner
# ------------------------------------------------------------------ #
st.subheader("Real-Time Scanner – Top 10 Pairs")
with st.spinner("Scanning 10 pairs..."):
    results = [analyze_pair(b, q) for b, q in TOP_PAIRS]
    results = [r for r in results if r]

if results:
    scan = pd.DataFrame(results).sort_values("strength", ascending=False)
    st.dataframe(scan.round(4), use_container_width=True)

    # ---------- Auto-Trade (simulated) ----------
    if auto_trade and live_mode and scan['strength'].max() > 0.7:
        strong = scan[scan['strength'] > 0.7]
        st.markdown("### Auto-Trade (Simulated)")
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
        st.success(f"Total P&L: ${total}")
else:
    st.info("No data for selected pairs.")

# ------------------------------------------------------------------ #
# Live Mode
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
        r = df.iloc[-1]
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
    rate = df['Rate'].iloc[-1] if not df.empty else 1.0
    if st.button("Simulate BUY"):
        st.success(f"Simulated BUY @ {rate:.5f} | P&L +${stake*0.03:.0f}")
    if st.button("Simulate SELL"):
        st.success(f"Simulated SELL @ {rate:.5f} | P&L -${stake*0.02:.0f}")

# ------------------------------------------------------------------ #
# Footer
# ------------------------------------------------------------------ #
st.markdown("---")
st.markdown("<p style='text-align:center;color:#666'>Streamlit + yFinance | Educational Use Only</p>", unsafe_allow_html=True)

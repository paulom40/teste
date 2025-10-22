import streamlit as st
import pandas as pd
import numpy as np
import requests
import time
from datetime import datetime, timedelta
import ta  # Technical Analysis library; install with pip install ta

# Page configuration
st.set_page_config(page_title="Real-Time Trading Demo with Indicators", layout="wide")

# Fixed parameters
initial_bank = 1000
stake = 10
profit_target = 10
stop_loss = 20
pip_value = stake / 10  # € per pip (1€ per pip for simplicity)

# Indicator parameters
ma_period = 20  # Simple Moving Average period
rsi_period = 14  # RSI period
rsi_overbought = 70
rsi_oversold = 30

# Pip sizes for pairs
pip_sizes = {
    "EUR/USD": 0.0001,
    "GBP/USD": 0.0001,
    "USD/JPY": 0.01,
    "AUD/USD": 0.0001,
    "USD/CAD": 0.0001,
    "NZD/USD": 0.0001
}

# Trading pairs
trading_pairs = [
    "EUR/USD", "GBP/USD", "USD/JPY", "AUD/USD", "USD/CAD", "NZD/USD"
]

# Function to fetch historical prices from Exchange API (daily for last 30 days)
@st.cache_data(ttl=3600)  # Cache for 1 hour
def get_historical_prices(pair, days=30):
    base_currency = "usd"  # API base
    if "/" in pair:
        from_curr, to_curr = pair.split("/")
        # For pairs like EUR/USD, fetch USD/EUR and invert
        target_curr = from_curr.lower() if to_curr == "USD" else to_curr.lower()
        url = f"https://api.exchangerate.host/timeseries?start_date={ (datetime.now() - timedelta(days=days)).strftime('%Y-%m-%d') }&end_date={datetime.now().strftime('%Y-%m-%d')}&base={base_currency}&symbols={target_curr}"
    else:
        return None
    
    try:
        resp = requests.get(url, timeout=10)
        if resp.status_code == 200:
            data = resp.json()
            if "rates" in data:
                dates = sorted(data["rates"].keys())
                prices = []
                for date in dates:
                    rate = data["rates"][date][target_curr]
                    if to_curr == "USD":
                        price = 1 / rate if rate > 0 else 0
                    else:
                        price = rate
                    prices.append({"date": date, "close": price})
                df = pd.DataFrame(prices)
                df["close"] = pd.to_numeric(df["close"])
                return df
    except Exception:
        pass
    return None

# Function to compute indicators and signal
def compute_indicators_and_signal(historical_df, current_price):
    if historical_df is None or len(historical_df) < max(ma_period, rsi_period):
        return None, None, "Insufficient Data"
    
    # Append current price if not in historical
    latest_date = historical_df["date"].max()
    if pd.to_datetime(historical_df["date"].iloc[-1]) < datetime.now().date():
        historical_df = pd.concat([historical_df, pd.DataFrame({"date": [datetime.now().strftime("%Y-%m-%d")], "close": [current_price]})], ignore_index=True)
    
    # Compute SMA
    historical_df["sma"] = historical_df["close"].rolling(window=ma_period).mean()
    
    # Compute RSI using ta library
    historical_df["rsi"] = ta.momentum.RSIIndicator(historical_df["close"], window=rsi_period).rsi()
    
    current_sma = historical_df["sma"].iloc[-1]
    current_rsi = historical_df["rsi"].iloc[-1]
    
    # Simple signal logic
    if current_price > current_sma and current_rsi < rsi_overbought:
        signal = "Buy (Long)"
    elif current_price < current_sma and current_rsi > rsi_oversold:
        signal = "Sell (Short)"
    else:
        signal = "Hold"
    
    return current_sma, current_rsi, signal

# Session state initialization
if "bankroll" not in st.session_state:
    st.session_state.bankroll = initial_bank
if "active_trades" not in st.session_state:
    st.session_state.active_trades = []
if "simulation_running" not in st.session_state:
    st.session_state.simulation_running = False
if "prices" not in st.session_state:
    # Initial prices (will be updated with live)
    st.session_state.prices = {}
if "historical_data" not in st.session_state:
    st.session_state.historical_data = {}
if "indicators" not in st.session_state:
    st.session_state.indicators = {}
if "start_time" not in st.session_state:
    st.session_state.start_time = None
if "last_update" not in st.session_state:
    st.session_state.last_update = None
if "api_last_fetched" not in st.session_state:
    st.session_state.api_last_fetched = None

# Title
st.title("Real-Time Trading Demo with Indicators")

# Sidebar for parameters and controls
st.sidebar.header("Trading Parameters")
st.sidebar.metric("Initial Bankroll", f"{initial_bank}€")
st.sidebar.metric("Stake per Trade", f"{stake}€")
st.sidebar.metric("Profit Target", f"{profit_target}€")
st.sidebar.metric("Stop Loss", f"{stop_loss}€")
st.sidebar.metric("Pip Value", f"{pip_value}€ per pip")

st.sidebar.header("Indicator Parameters")
st.sidebar.slider("MA Period", 5, 50, ma_period, key="ma_period")
st.sidebar.slider("RSI Period", 5, 30, rsi_period, key="rsi_period")
st.sidebar.slider("RSI Overbought", 50, 90, rsi_overbought, key="rsi_ob")
st.sidebar.slider("RSI Oversold", 10, 50, rsi_oversold, key="rsi_os")

st.sidebar.header("Simulation Controls")
if st.sidebar.button("Start Simulation"):
    st.session_state.simulation_running = True
    st.session_state.start_time = time.time()
    if st.session_state.last_update is None:
        st.session_state.last_update = time.time()
if st.sidebar.button("Stop Simulation"):
    st.session_state.simulation_running = False

if st.sidebar.button("Refresh Live Prices & Indicators"):
    # Fetch live and historical
    live_prices = get_live_prices()
    if live_prices:
        for pair in trading_pairs:
            if pair in live_prices:
                st.session_state.prices[pair] = {
                    "price": live_prices[pair],
                    "pip_size": pip_sizes[pair]
                }
                # Fetch historical for indicators
                hist_df = get_historical_prices(pair)
                if hist_df is not None:
                    st.session_state.historical_data[pair] = hist_df
                    sma, rsi, signal = compute_indicators_and_signal(hist_df, live_prices[pair])
                    st.session_state.indicators[pair] = {"sma": sma, "rsi": rsi, "signal": signal}
    st.session_state.api_last_fetched = datetime.now().strftime("%H:%M:%S")
    st.rerun()

# Current bankroll display
col1, col2 = st.columns([3, 1])
with col1:
    st.metric("Current Bankroll", f"{st.session_state.bankroll:.2f}€")
with col2:
    if st.button("Reset Bankroll"):
        st.session_state.bankroll = initial_bank
        st.session_state.active_trades = []
        st.session_state.prices = {}
        st.session_state.historical_data = {}
        st.session_state.indicators = {}
        st.session_state.simulation_running = False
        st.session_state.last_update = None
        st.session_state.api_last_fetched = None
        st.rerun()

# Function to fetch live prices (from previous)
@st.cache_data(ttl=300)
def get_live_prices():
    url = "https://api.exchangerate.host/latest?base=usd"  # Updated to use exchangerate.host for consistency with historical
    try:
        resp = requests.get(url, timeout=10)
        if resp.status_code == 200:
            data = resp.json()
            if "rates" in data:
                rates = data["rates"]
                prices = {}
                # EUR/USD: 1 / usd_to_eur
                usd_to_eur = rates.get("EUR", 0)
                if usd_to_eur > 0:
                    prices["EUR/USD"] = round(1 / usd_to_eur, 5)
                
                usd_to_gbp = rates.get("GBP", 0)
                if usd_to_gbp > 0:
                    prices["GBP/USD"] = round(1 / usd_to_gbp, 5)
                
                prices["USD/JPY"] = round(rates.get("JPY", 0), 2)
                
                usd_to_aud = rates.get("AUD", 0)
                if usd_to_aud > 0:
                    prices["AUD/USD"] = round(1 / usd_to_aud, 5)
                
                prices["USD/CAD"] = round(rates.get("CAD", 0), 5)
                
                usd_to_nzd = rates.get("NZD", 0)
                if usd_to_nzd > 0:
                    prices["NZD/USD"] = round(1 / usd_to_nzd, 5)
                
                return prices
    except Exception:
        pass
    return None

# Initialize prices and indicators if not set
if not st.session_state.prices:
    live_prices = get_live_prices()
    if live_prices:
        for pair in trading_pairs:
            if pair in live_prices:
                st.session_state.prices[pair] = {
                    "price": live_prices[pair],
                    "pip_size": pip_sizes[pair]
                }
                hist_df = get_historical_prices(pair)
                if hist_df is not None:
                    st.session_state.historical_data[pair] = hist_df
                    sma, rsi, signal = compute_indicators_and_signal(hist_df, live_prices[pair])
                    st.session_state.indicators[pair] = {"sma": sma, "rsi": rsi, "signal": signal}

# Update global params from sidebar
ma_period = st.session_state.ma_period
rsi_period = st.session_state.rsi_period
rsi_overbought = st.session_state.rsi_ob
rsi_oversold = st.session_state.rsi_os

# Recompute indicators if params changed
if st.session_state.indicators:
    for pair in trading_pairs:
        if pair in st.session_state.prices and pair in st.session_state.historical_data:
            hist_df = st.session_state.historical_data[pair].copy()
            hist_df["sma"] = hist_df["close"].rolling(window=ma_period).mean()
            hist_df["rsi"] = ta.momentum.RSIIndicator(hist_df["close"], window=rsi_period).rsi()
            current_price = st.session_state.prices[pair]["price"]
            current_sma = hist_df["sma"].iloc[-1]
            current_rsi = hist_df["rsi"].iloc[-1]
            if current_price > current_sma and current_rsi < rsi_overbought:
                signal = "Buy (Long)"
            elif current_price < current_sma and current_rsi > rsi_oversold:
                signal = "Sell (Short)"
            else:
                signal = "Hold"
            st.session_state.indicators[pair] = {"sma": current_sma, "rsi": current_rsi, "signal": signal}
    st.rerun()

# Function to format price
def format_price(price, pip_size):
    if pip_size == 0.01:
        return f"{price:.2f}"
    else:
        return f"{price:.4f}"

# Active trades display
if st.session_state.active_trades:
    st.subheader("Active Trades")
    active_trades_display = []
    for trade in st.session_state.active_trades:
        t = trade.copy()
        current_price = st.session_state.prices[trade["Pair"]]["price"]
        pip_size = trade["pip_size"]
        is_long = trade["Direction"] == "Long"
        entry = trade["Entry Price"]
        
        if is_long:
            delta = current_price - entry
        else:
            delta = entry - current_price
        
        pips = delta / pip_size
        t["Current Price"] = format_price(current_price, pip_size)
        t["Entry Price"] = format_price(entry, pip_size)
        t["TP Price"] = format_price(trade["TP_price"], pip_size)
        t["SL Price"] = format_price(trade["SL_price"], pip_size)
        t["Pips"] = round(pips, 1)
        t["Current P&L (€)"] = round(pips * pip_value, 2)
        
        active_trades_display.append(t)
    
    trades_df = pd.DataFrame(active_trades_display)
    st.dataframe(trades_df, use_container_width=True)

# Available pairs for trading with indicators
st.subheader("Trading Pairs - Signals from Indicators")
pairs_data = []
for pair in trading_pairs:
    if pair in st.session_state.prices and pair in st.session_state.indicators:
        data = st.session_state.prices[pair]
        ind = st.session_state.indicators[pair]
        pairs_data.append({
            "Pair": pair,
            "Current Price": format_price(data["price"], data["pip_size"]),
            "MA (" + str(ma_period) + ")": f"{ind['sma']:.4f}" if ind['sma'] else "N/A",
            "RSI (" + str(rsi_period) + ")": f"{ind['rsi']:.1f}" if ind['rsi'] is not np.nan else "N/A",
            "Signal": ind['signal'],
            "Last Updated": st.session_state.api_last_fetched or "Initial"
        })

pairs_df = pd.DataFrame(pairs_data)
st.dataframe(pairs_df, use_container_width=True)

# Trade entry (now with suggested direction from signal)
selected_pair = st.selectbox("Select Pair to Trade", list(st.session_state.prices.keys()))
current_signal = st.session_state.indicators.get(selected_pair, {}).get("signal", "Hold")
suggested_dir = "Long" if "Buy" in current_signal else "Short" if "Sell" in current_signal else "None"
st.info(f"Suggested Direction from Indicators: **{current_signal}** ({suggested_dir})")

direction = st.selectbox("Trade Direction", ["Long", "Short"], index=0 if suggested_dir == "Long" else 1 if suggested_dir == "Short" else 0)

if st.button("Open Trade"):
    if st.session_state.bankroll >= stake:
        entry = st.session_state.prices[selected_pair]["price"]
        pip_size = st.session_state.prices[selected_pair]["pip_size"]
        tp_pips = profit_target / pip_value
        sl_pips = stop_loss / pip_value
        
        trade = {
            "Pair": selected_pair,
            "Direction": direction,
            "Entry Price": entry,
            "Stake": stake,
            "TP": profit_target,
            "SL": stop_loss,
            "pip_size": pip_size,
            "Open Time": datetime.now().strftime("%H:%M:%S"),
            "Status": "Open"
        }
        
        is_long = direction == "Long"
        if is_long:
            trade["TP_price"] = entry + tp_pips * pip_size
            trade["SL_price"] = entry - sl_pips * pip_size
        else:
            trade["TP_price"] = entry - tp_pips * pip_size
            trade["SL_price"] = entry + sl_pips * pip_size
        
        st.session_state.active_trades.append(trade)
        st.session_state.bankroll -= stake
        st.rerun()
    else:
        st.error("Insufficient bankroll!")

# Simulation logic (fetches live every 30s, but indicators update on refresh)
if st.session_state.simulation_running:
    current_time = time.time()
    if st.session_state.last_update is None:
        st.session_state.last_update = current_time
    
    elapsed = current_time - st.session_state.last_update
    if elapsed >= 30:
        live_prices = get_live_prices()
        if live_prices:
            updated = False
            for pair in trading_pairs:
                if pair in live_prices:
                    old_price = st.session_state.prices[pair]["price"]
                    st.session_state.prices[pair]["price"] = live_prices[pair]
                    if abs(live_prices[pair] - old_price) > 0.0001:  # If price changed
                        hist_df = st.session_state.historical_data.get(pair)
                        if hist_df is not None:
                            sma, rsi, signal = compute_indicators_and_signal(hist_df, live_prices[pair])
                            st.session_state.indicators[pair] = {"sma": sma, "rsi": rsi, "signal": signal}
                            updated = True
                    st.session_state.api_last_fetched = datetime.now().strftime("%H:%M:%S")
            
            # Check active trades
            for trade in st.session_state.active_trades[:]:
                current_price = st.session_state.prices[trade["Pair"]]["price"]
                is_long = trade["Direction"] == "Long"
                
                if is_long:
                    tp_hit = current_price >= trade["TP_price"]
                    sl_hit = current_price <= trade["SL_price"]
                else:
                    tp_hit = current_price <= trade["TP_price"]
                    sl_hit = current_price >= trade["SL_price"]
                
                if tp_hit:
                    trade["Status"] = "Closed (TP Hit)"
                    st.session_state.bankroll += stake + profit_target
                    st.session_state.active_trades.remove(trade)
                elif sl_hit:
                    trade["Status"] = "Closed (SL Hit)"
                    st.session_state.bankroll += stake - stop_loss
                    st.session_state.active_trades.remove(trade)
        
        st.session_state.last_update = current_time
        if updated:
            st.rerun()
    
    next_update = 30 - (current_time - st.session_state.last_update)
    st.info(f"🔄 Live data feed active... Next price check in {next_update:.0f} seconds. Indicators update on refresh or significant price change.")

# Notes
st.subheader("Demo Notes")
st.info("""
- **Indicators**: Uses SMA (trend) and RSI (momentum) for signals. Buy if price > SMA & RSI < 70; Sell if price < SMA & RSI > 30; else Hold. Adjustable in sidebar.
- **Data**: Live from exchangerate.host (daily rates). Historical (30 days) fetched for indicators. Requires `pip install ta` for RSI calc.
- **Signals**: Guide trade direction—override manually if needed. Auto-trading not implemented (demo focuses on signals).
- **P&L**: Pip-based. TP/SL on price hits.
- **Risk**: Educational only—indicators aren't foolproof; backtest in real scenarios.
- **Enhancements**: Add more indicators (e.g., MACD via ta), charts with plotly, or auto-trade on signals.
- Run with: `streamlit run app.py` (requires `pip install streamlit pandas numpy requests ta`)
""")

# Footer
st.markdown("---")
st.caption("Built with Streamlit. Data from exchangerate.host – for educational purposes only.")

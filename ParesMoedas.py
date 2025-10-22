import streamlit as st
import pandas as pd
import numpy as np
import requests
import time
from datetime import datetime

# Page configuration
st.set_page_config(page_title="Real-Time Trading Demo", layout="wide")

# Fixed parameters
initial_bank = 1000
stake = 10
profit_target = 10
stop_loss = 20
pip_value = stake / 10  # € per pip (1€ per pip for simplicity)

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

# Function to fetch live prices from Exchange API
@st.cache_data(ttl=300)  # Cache for 5 minutes to respect any implicit limits
def get_live_prices():
    url = "https://cdn.jsdelivr.net/npm/@fawazahmed0/currency-api@latest/v1/currencies/usd.json"
    fallback_url = "https://latest.currency-api.pages.dev/v1/currencies/usd.json"
    
    try:
        resp = requests.get(url, timeout=10)
        if resp.status_code == 200:
            data = resp.json()
        else:
            # Fallback
            resp = requests.get(fallback_url, timeout=10)
            if resp.status_code == 200:
                data = resp.json()
            else:
                return None
    except Exception:
        return None
    
    prices = {}
    # EUR/USD: 1 / eur_per_usd
    eur_per_usd = data.get("eur", 0)
    if eur_per_usd > 0:
        prices["EUR/USD"] = round(1 / eur_per_usd, 5)
    
    # GBP/USD: 1 / gbp_per_usd
    gbp_per_usd = data.get("gbp", 0)
    if gbp_per_usd > 0:
        prices["GBP/USD"] = round(1 / gbp_per_usd, 5)
    
    # USD/JPY: direct
    jpy_per_usd = data.get("jpy", 0)
    prices["USD/JPY"] = round(jpy_per_usd, 2)
    
    # AUD/USD: 1 / aud_per_usd
    aud_per_usd = data.get("aud", 0)
    if aud_per_usd > 0:
        prices["AUD/USD"] = round(1 / aud_per_usd, 5)
    
    # USD/CAD: direct
    cad_per_usd = data.get("cad", 0)
    prices["USD/CAD"] = round(cad_per_usd, 5)
    
    # NZD/USD: 1 / nzd_per_usd
    nzd_per_usd = data.get("nzd", 0)
    if nzd_per_usd > 0:
        prices["NZD/USD"] = round(1 / nzd_per_usd, 5)
    
    return prices

# Session state initialization
if "bankroll" not in st.session_state:
    st.session_state.bankroll = initial_bank
if "active_trades" not in st.session_state:
    st.session_state.active_trades = []
if "simulation_running" not in st.session_state:
    st.session_state.simulation_running = False
if "prices" not in st.session_state:
    live_prices = get_live_prices()
    if live_prices:
        st.session_state.prices = {}
        for pair in trading_pairs:
            if pair in live_prices:
                st.session_state.prices[pair] = {
                    "price": live_prices[pair],
                    "pip_size": pip_sizes[pair]
                }
    else:
        # Fallback to initial simulated prices if API fails
        initial_prices = {
            "EUR/USD": 1.0850, "GBP/USD": 1.2950, "USD/JPY": 150.20,
            "AUD/USD": 0.6750, "USD/CAD": 1.3850, "NZD/USD": 0.6150
        }
        st.session_state.prices = {}
        for pair in trading_pairs:
            st.session_state.prices[pair] = {
                "price": initial_prices[pair],
                "pip_size": pip_sizes[pair]
            }
if "start_time" not in st.session_state:
    st.session_state.start_time = None
if "last_update" not in st.session_state:
    st.session_state.last_update = None
if "api_last_fetched" not in st.session_state:
    st.session_state.api_last_fetched = None

# Title
st.title("Real-Time Trading Demo")

# Sidebar for parameters and controls
st.sidebar.header("Trading Parameters")
st.sidebar.metric("Initial Bankroll", f"{initial_bank}€")
st.sidebar.metric("Stake per Trade", f"{stake}€")
st.sidebar.metric("Profit Target", f"{profit_target}€")
st.sidebar.metric("Stop Loss", f"{stop_loss}€")
st.sidebar.metric("Pip Value", f"{pip_value}€ per pip")

st.sidebar.header("Simulation Controls")
if st.sidebar.button("Start Simulation"):
    st.session_state.simulation_running = True
    st.session_state.start_time = time.time()
    if st.session_state.last_update is None:
        st.session_state.last_update = time.time()
if st.sidebar.button("Stop Simulation"):
    st.session_state.simulation_running = False

if st.sidebar.button("Refresh Live Prices"):
    live_prices = get_live_prices()
    if live_prices:
        for pair in trading_pairs:
            if pair in live_prices:
                st.session_state.prices[pair]["price"] = live_prices[pair]
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
        live_prices = get_live_prices()
        if live_prices:
            st.session_state.prices = {}
            for pair in trading_pairs:
                if pair in live_prices:
                    st.session_state.prices[pair] = {
                        "price": live_prices[pair],
                        "pip_size": pip_sizes[pair]
                    }
        st.session_state.simulation_running = False
        st.session_state.last_update = None
        st.session_state.api_last_fetched = None
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

# Available pairs for trading
st.subheader("Trading Pairs - Enter a Trade")
pairs_data = [
    {
        "Pair": pair,
        "Current Price": format_price(data["price"], data["pip_size"]),
        "Last Updated": st.session_state.api_last_fetched or "Initial"
    }
    for pair, data in st.session_state.prices.items()
]
pairs_df = pd.DataFrame(pairs_data)
st.dataframe(pairs_df, use_container_width=True)

# Trade entry
selected_pair = st.selectbox("Select Pair to Trade", list(st.session_state.prices.keys()))
direction = st.selectbox("Trade Direction", ["Long", "Short"])

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

# Simulation logic (now fetches live prices periodically)
if st.session_state.simulation_running:
    current_time = time.time()
    if st.session_state.last_update is None:
        st.session_state.last_update = current_time
    
    elapsed = current_time - st.session_state.last_update
    if elapsed >= 30:  # Fetch every 30 seconds
        # Fetch live prices
        live_prices = get_live_prices()
        if live_prices:
            for pair in trading_pairs:
                if pair in live_prices:
                    st.session_state.prices[pair]["price"] = live_prices[pair]
            st.session_state.api_last_fetched = datetime.now().strftime("%H:%M:%S")
        
        # Check active trades for TP/SL hits (using updated prices)
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
        st.rerun()
    
    # Placeholder for real-time update indicator
    next_update = 30 - (current_time - st.session_state.last_update)
    st.info(f"🔄 Live data feed active... Next price check in {next_update:.0f} seconds. (API updates daily)")

# Notes
st.subheader("Demo Notes")
st.info("""
- **Live Data Source**: Powered by [Exchange API](https://github.com/fawazahmed0/exchange-api) – free, no key, daily updated rates for 200+ currencies. Fetches USD-based rates and derives pairs (e.g., EUR/USD = 1 / EUR per USD).
- **P&L Calculation**: Pip-based with realistic forex sizing. TP/SL checked on price updates.
- **Real-Time**: API provides latest daily rates (refreshed periodically in demo). For intra-day ticks, consider premium APIs like Finnhub.
- **Trade Mechanics**: Stake as margin proxy. Unrealized P&L updates live.
- **Risk**: Educational demo—real trading risks capital.
- **Enhancements**: Add charts, alerts, or error handling for API downtime.
- Run with: `streamlit run app.py` (requires `pip install streamlit pandas numpy requests`)
""")

# Footer
st.markdown("---")
st.caption("Built with Streamlit. Data from Exchange API – for educational purposes only.")

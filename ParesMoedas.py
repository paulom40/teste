import streamlit as st
import pandas as pd
import numpy as np
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

# Trading pairs with initial prices (simulated; in production, fetch from API)
trading_pairs = [
    "EUR/USD", "GBP/USD", "USD/JPY", "AUD/USD", "USD/CAD", "NZD/USD"
]

initial_prices = {
    "EUR/USD": 1.0850,
    "GBP/USD": 1.2950,
    "USD/JPY": 150.20,
    "AUD/USD": 0.6750,
    "USD/CAD": 1.3850,
    "NZD/USD": 0.6150
}

initial_directions = {
    "EUR/USD": 1,
    "GBP/USD": -1,
    "USD/JPY": 1,
    "AUD/USD": 1,
    "USD/CAD": -1,
    "NZD/USD": 1
}

# Session state initialization
if "bankroll" not in st.session_state:
    st.session_state.bankroll = initial_bank
if "active_trades" not in st.session_state:
    st.session_state.active_trades = []
if "simulation_running" not in st.session_state:
    st.session_state.simulation_running = False
if "prices" not in st.session_state:
    st.session_state.prices = {}
    for pair in trading_pairs:
        st.session_state.prices[pair] = {
            "price": initial_prices[pair],
            "direction": initial_directions[pair],
            "pip_size": pip_sizes[pair]
        }
if "start_time" not in st.session_state:
    st.session_state.start_time = None
if "last_update" not in st.session_state:
    st.session_state.last_update = None

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

# Current bankroll display
col1, col2 = st.columns([3, 1])
with col1:
    st.metric("Current Bankroll", f"{st.session_state.bankroll:.2f}€")
with col2:
    if st.button("Reset Bankroll"):
        st.session_state.bankroll = initial_bank
        st.session_state.active_trades = []
        st.session_state.prices = {}
        for pair in trading_pairs:
            st.session_state.prices[pair] = {
                "price": initial_prices[pair],
                "direction": initial_directions[pair],
                "pip_size": pip_sizes[pair]
            }
        st.session_state.simulation_running = False
        st.session_state.last_update = None
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
        "Bias": "Up" if data["direction"] > 0 else "Down"
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

# Simulation logic
if st.session_state.simulation_running:
    current_time = time.time()
    if st.session_state.last_update is None:
        st.session_state.last_update = current_time
    
    elapsed = current_time - st.session_state.last_update
    if elapsed >= 2:
        # Update prices
        for pair, data in st.session_state.prices.items():
            pip_vol = np.random.normal(0, 0.5)  # Random walk volatility in pips
            bias_pip = data["direction"] * np.random.uniform(0, 0.2)  # Biased step in pips
            total_pip_change = bias_pip + pip_vol
            data["price"] += total_pip_change * data["pip_size"]
            data["price"] = max(0.0001, data["price"])  # Prevent negative/zero
        
        # Check active trades for TP/SL hits
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
    next_update = 2 - (current_time - st.session_state.last_update)
    st.info(f"🔄 Simulation running... Next price update in {next_update:.1f} seconds.")

# Notes
st.subheader("Demo Notes")
st.info("""
- **P&L Calculation**: Realistic forex pip-based P&L with 1€ per pip (adjusted for pair pip sizes, e.g., 0.0001 for EUR/USD, 0.01 for USD/JPY). TP/SL enforced via exact price levels.
- **Simulation**: Biased random walk (0.5 pip std dev + bias) every 2 seconds. In reality, integrate with APIs like Polygon.io for live forex data.
- **Trade Mechanics**: Stake deducted on entry (as margin proxy). Closes add back stake + P&L. Current unrealized P&L shown live.
- **Risk**: Demo only—real trading risks capital. Assumes consistent pip value across pairs for simplicity.
- **Enhancements**: Add charts (`st.line_chart`), alerts, or WebSockets for true real-time.
- Run with: `streamlit run app.py`
""")

# Footer
st.markdown("---")
st.caption("Built with Streamlit. Simulation for educational purposes only.")

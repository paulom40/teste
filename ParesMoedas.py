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

# Initial prices (fallback)
initial_prices = {
    "EUR/USD": 1.0850,
    "GBP/USD": 1.2950,
    "USD/JPY": 150.20,
    "AUD/USD": 0.6750,
    "USD/CAD": 1.3850,
    "NZD/USD": 0.6150
}

# Function to generate simulated historical prices (fallback)
def generate_simulated_historical(pair, days=30):
    np.random.seed(42)  # For reproducibility
    base_price = initial_prices[pair]
    prices = []
    dates = pd.date_range(end=datetime.now(), periods=days).strftime("%Y-%m-%d")
    for date in dates:
        # Simulate random walk with small volatility
        change = np.random.normal(0, 0.001)  # Adjust volatility per pair
        base_price += change * base_price
        prices.append({"date": date, "close": base_price})
    df = pd.DataFrame(prices)
    df["close"] = pd.to_numeric(df["close"])
    return df

# Function to fetch historical prices from Exchange API (daily for last 30 days)
@st.cache_data(ttl=3600)  # Cache for 1 hour
def get_historical_prices(pair, days=30):
    base_currency = "usd"  # API base
    if "/" in pair:
        from_curr, to_curr = pair.split("/")
        # For pairs like EUR/USD, fetch USD/EUR and invert
        target_curr = from_curr.lower() if to_curr == "USD" else to_curr.lower()
        start_date = (datetime.now() - timedelta(days=days)).strftime('%Y-%m-%d')
        end_date = datetime.now().strftime('%Y-%m-%d')
        url = f"https://api.exchangerate.host/timeseries?start_date={start_date}&end_date={end_date}&base={base_currency}&symbols={target_curr}"
    else:
        return None
    
    try:
        resp = requests.get(url, timeout=10)
        if resp.status_code == 200:
            data = resp.json()
            if "rates" in data and data["success"]:
                dates = sorted(data["rates"].keys())
                prices = []
                for date in dates:
                    rate = data["rates"][date][target_curr]
                    if to_curr == "USD" and rate > 0:
                        price = 1 / rate
                    else:
                        price = rate
                    prices.append({"date": date, "close": price})
                df = pd.DataFrame(prices)
                df["close"] = pd.to_numeric(df["close"])
                return df
    except Exception as e:
        st.warning(f"API fetch failed for {pair}: {e}. Using simulated data.")
    # Fallback to simulated
    return generate_simulated_historical(pair, days)

# Function to fetch live prices (from previous)
@st.cache_data(ttl=300)
def get_live_prices():
    url = "https://api.exchangerate.host/latest?base=usd"
    try:
        resp = requests.get(url, timeout=10)
        if resp.status_code == 200:
            data = resp.json()
            if "rates" in data and data["success"]:
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
    except Exception as e:
        st.warning(f"Live API fetch failed: {e}. Using initial prices.")
    return None

# Function to compute indicators and signal
def compute_indicators_and_signal(historical_df, current_price):
    if historical_df is None or len(historical_df) < max(ma_period, rsi_period):
        return None, None, "Insufficient Data"
    
    # Append current price if not in historical
    latest_date = pd.to_datetime(historical_df["date"].iloc[-1])
    now_date = datetime.now().date()
    if latest_date.date() < now_date:
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
if "closed_trades" not in st.session_state:
    st.session_state.closed_trades = []
if "simulation_running" not in st.session_state:
    st.session_state.simulation_running = False
if "prices" not in st.session_state:
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
ma_period = st.sidebar.slider("MA Period", 5, 50, 20)
rsi_period = st.sidebar.slider("RSI Period", 5, 30, 14)
rsi_overbought = st.sidebar.slider("RSI Overbought", 50, 90, 70)
rsi_oversold = st.sidebar.slider("RSI Oversold", 10, 50, 30)

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
    else:
        # Use initial if API fails
        for pair in trading_pairs:
            st.session_state.prices[pair] = {
                "price": initial_prices[pair],
                "pip_size": pip_sizes[pair]
            }
            hist_df = get_historical_prices(pair)  # Will fallback to simulated
            st.session_state.historical_data[pair] = hist_df
            sma, rsi, signal = compute_indicators_and_signal(hist_df, initial_prices[pair])
            st.session_state.indicators[pair] = {"sma": sma, "rsi": rsi, "signal": signal}
        st.session_state.api_last_fetched = "Fallback (Simulated)"
    st.rerun()

# Current bankroll display
col1, col2 = st.columns([3, 1])
with col1:
    st.metric("Current Bankroll", f"{st.session_state.bankroll:.2f}€")
with col2:
    if st.button("Reset Bankroll"):
        st.session_state.bankroll = initial_bank
        st.session_state.active_trades = []
        st.session_state.closed_trades = []
        st.session_state.prices = {}
        st.session_state.historical_data = {}
        st.session_state.indicators = {}
        st.session_state.simulation_running = False
        st.session_state.last_update = None
        st.session_state.api_last_fetched = None
        st.rerun()

# Initialize prices and indicators if not set
if not st.session_state.prices:
    # Use initial prices and simulated historical
    for pair in trading_pairs:
        st.session_state.prices[pair] = {
            "price": initial_prices[pair],
            "pip_size": pip_sizes[pair]
        }
        hist_df = get_historical_prices(pair)  # Falls back to simulated if API fails
        st.session_state.historical_data[pair] = hist_df
        sma, rsi, signal = compute_indicators_and_signal(hist_df, initial_prices[pair])
        st.session_state.indicators[pair] = {"sma": sma, "rsi": rsi, "signal": signal}
    st.session_state.api_last_fetched = "Initial (Simulated)"

# Recompute indicators if params changed (store in session for slider changes)
if "ma_period" not in st.session_state:
    st.session_state.ma_period = ma_period
    st.session_state.rsi_period = rsi_period
    st.session_state.rsi_overbought = rsi_overbought
    st.session_state.rsi_oversold = rsi_oversold

if st.session_state.ma_period != ma_period or st.session_state.rsi_period != rsi_period or \
   st.session_state.rsi_overbought != rsi_overbought or st.session_state.rsi_oversold != rsi_oversold:
    st.session_state.ma_period = ma_period
    st.session_state.rsi_period = rsi_period
    st.session_state.rsi_overbought = rsi_overbought
    st.session_state.rsi_oversold = rsi_oversold
    for pair in trading_pairs:
        if pair in st.session_state.prices and pair in st.session_state.historical_data:
            hist_df = st.session_state.historical_data[pair].copy()
            sma, rsi, signal = compute_indicators_and_signal(hist_df, st.session_state.prices[pair]["price"])
            st.session_state.indicators[pair] = {"sma": sma, "rsi": rsi, "signal": signal}
    st.rerun()

# Use session state params
ma_period = st.session_state.ma_period
rsi_period = st.session_state.rsi_period
rsi_overbought = st.session_state.rsi_overbought
rsi_oversold = st.session_state.rsi_oversold

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
        if trade["Pair"] in st.session_state.prices:
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
        else:
            t["Current Price"] = "N/A"
            t["Pips"] = "N/A"
            t["Current P&L (€)"] = "N/A"
        
        active_trades_display.append(t)
    
    trades_df = pd.DataFrame(active_trades_display)
    st.dataframe(trades_df, use_container_width=True)

# Trading Results table
st.subheader("Trading Results")
if st.session_state.closed_trades:
    results_display = []
    for trade in st.session_state.closed_trades:
        t = trade.copy()
        pip_size = trade["pip_size"]
        t["Entry Price"] = format_price(trade["Entry Price"], pip_size)
        t["Exit Price"] = format_price(trade["Exit Price"], pip_size)
        # Calculate duration roughly (in seconds for simplicity, assuming same day)
        open_time = datetime.strptime(trade["Open Time"], "%H:%M:%S")
        close_time = datetime.strptime(trade["Close Time"], "%H:%M:%S")
        duration = (close_time - open_time).total_seconds()
        if duration < 0:  # If cross day, approximate
            duration += 86400  # Add one day in seconds
        t["Duration (s)"] = round(duration, 1)
        results_display.append(t)
    
    results_df = pd.DataFrame(results_display)[["Pair", "Direction", "Entry Price", "Exit Price", "P&L (€)", "Duration (s)", "Status"]]
    st.dataframe(results_df, use_container_width=True)
    
    # Summary metrics
    total_trades = len(st.session_state.closed_trades)
    total_pnl = sum(trade["P&L"] for trade in st.session_state.closed_trades)
    wins = sum(1 for trade in st.session_state.closed_trades if trade["P&L"] > 0)
    win_rate = (wins / total_trades * 100) if total_trades > 0 else 0
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total Trades", total_trades)
    col2.metric("Total P&L (€)", f"{total_pnl:.2f}")
    col3.metric("Wins", wins)
    col4.metric("Win Rate (%)", f"{win_rate:.1f}")
else:
    st.info("No closed trades yet. Open some trades and run the simulation to see results.")

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
            "MA (" + str(ma_period) + ")": f"{ind['sma']:.4f}" if ind['sma'] is not None else "N/A",
            "RSI (" + str(rsi_period) + ")": f"{ind['rsi']:.1f}" if ind['rsi'] is not None and not np.isnan(ind['rsi']) else "N/A",
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
    if selected_pair in st.session_state.prices and st.session_state.bankroll >= stake:
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
        st.error("Insufficient bankroll or invalid pair!")

# Simulation logic (fetches live every 30s, but indicators update on refresh)
if st.session_state.simulation_running:
    current_time = time.time()
    if st.session_state.last_update is None:
        st.session_state.last_update = current_time
    
    elapsed = current_time - st.session_state.last_update
    if elapsed >= 30:
        live_prices = get_live_prices()
        updated = False
        if live_prices:
            for pair in trading_pairs:
                if pair in live_prices and pair in pip_sizes:
                    old_price = st.session_state.prices.get(pair, {}).get("price", 0)
                    st.session_state.prices[pair] = {
                        "price": live_prices[pair],
                        "pip_size": pip_sizes[pair]
                    }
                    if abs(live_prices[pair] - old_price) > 0.0001:  # If price changed
                        hist_df = st.session_state.historical_data.get(pair)
                        if hist_df is not None:
                            sma, rsi, signal = compute_indicators_and_signal(hist_df, live_prices[pair])
                            st.session_state.indicators[pair] = {"sma": sma, "rsi": rsi, "signal": signal}
                            updated = True
                    st.session_state.api_last_fetched = datetime.now().strftime("%H:%M:%S")
        else:
            # Simulate small changes for demo
            for pair in trading_pairs:
                if pair in st.session_state.prices and "pip_size" in st.session_state.prices[pair]:
                    pip_size = st.session_state.prices[pair]["pip_size"]
                    change = np.random.normal(0, 0.0005 / pip_size) * pip_size  # Small pip change
                    old_price = st.session_state.prices[pair]["price"]
                    st.session_state.prices[pair]["price"] += change
                    if abs(change) > 0:
                        hist_df = st.session_state.historical_data.get(pair)
                        if hist_df is not None:
                            sma, rsi, signal = compute_indicators_and_signal(hist_df, st.session_state.prices[pair]["price"])
                            st.session_state.indicators[pair] = {"sma": sma, "rsi": rsi, "signal": signal}
                            updated = True
        
        # Check active trades
        for trade in st.session_state.active_trades[:]:
            if trade["Pair"] in st.session_state.prices:
                current_price = st.session_state.prices[trade["Pair"]]["price"]
                is_long = trade["Direction"] == "Long"
                
                tp_hit = False
                sl_hit = False
                if is_long:
                    tp_hit = current_price >= trade["TP_price"]
                    sl_hit = current_price <= trade["SL_price"]
                else:
                    tp_hit = current_price <= trade["TP_price"]
                    sl_hit = current_price >= trade["SL_price"]
                
                if tp_hit or sl_hit:
                    # Close the trade
                    closed_trade = trade.copy()
                    closed_trade["Exit Price"] = current_price
                    closed_trade["Close Time"] = datetime.now().strftime("%H:%M:%S")
                    closed_trade["Status"] = "Closed (TP Hit)" if tp_hit else "Closed (SL Hit)"
                    closed_trade["P&L"] = profit_target if tp_hit else -stop_loss
                    st.session_state.closed_trades.append(closed_trade)
                    
                    # Update bankroll
                    pnl = closed_trade["P&L"]
                    st.session_state.bankroll += stake + pnl
                    
                    # Remove from active
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
- **Data**: Attempts live from exchangerate.host (daily rates). Falls back to simulated data if API unavailable. Historical (30 days) used for indicators. Requires `pip install ta` for RSI calc.
- **Signals**: Guide trade direction—override manually if needed. Auto-trading not implemented (demo focuses on signals).
- **P&L**: Pip-based. TP/SL on price hits.
- **Risk**: Educational only—indicators aren't foolproof; backtest in real scenarios.
- **Enhancements**: Add more indicators (e.g., MACD via ta), charts with plotly, or auto-trade on signals.
- Run with: `streamlit run app.py` (requires `pip install streamlit pandas numpy requests ta`)
""")

# Footer
st.markdown("---")
st.caption("Built with Streamlit. Data from exchangerate.host or simulated – for educational purposes only.")

import streamlit as st
import pandas as pd
import yfinance as yf
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime, timedelta
import numpy as np
from oandapyV20 import API  # New: OANDA API wrapper
from oandapyV20.exceptions import V20Error
from oandapyV20.endpoints import orders as orders_endpoint
from oandapyV20.endpoints import accounts as accounts_endpoint

# Page config
st.set_page_config(page_title="Auto-Trading Forex System", layout="wide")

# Title
st.title("🤖 Auto-Trading Forex System with Broker API Integration")

# Sidebar for user inputs
st.sidebar.header("Select Currency Pair")
base_currencies = ["USD", "EUR", "GBP", "JPY", "AUD", "CAD"]
quote_currencies = ["USD", "EUR", "GBP", "JPY", "AUD", "CAD"]
base = st.sidebar.selectbox("Base Currency", base_currencies)
quote = st.sidebar.selectbox("Quote Currency", quote_currencies, index=1 if base == "USD" else 0)

if base == quote:
    st.sidebar.warning("Please select different currencies.")
    st.stop()

ticker = f"{quote}{base}=X"

days = st.sidebar.slider("Historical Days", 5, 365, 30)

# Manual Stake Input
st.sidebar.subheader("Trading Parameters")
stake = st.sidebar.number_input("Manual Stake Amount ($)", min_value=1.0, value=1000.0, step=100.0)

# Auto-Trading Settings
st.sidebar.subheader("Auto-Trading Strategy")
strategy = st.sidebar.selectbox("Strategy", ["RSI Oversold/Overbought", "MA Crossover", "Combined"])
rsi_threshold_low = st.sidebar.slider("RSI Buy Threshold", 20, 40, 30)
rsi_threshold_high = st.sidebar.slider("RSI Sell Threshold", 60, 80, 70)
use_stop_loss = st.sidebar.checkbox("Use Stop Loss (2%)", True)
use_take_profit = st.sidebar.checkbox("Use Take Profit (4%)", True)

# OANDA Broker API Configuration (New)
st.sidebar.subheader("Broker API (OANDA v20)")
api_token = st.sidebar.text_input("API Token", type="password", help="Get from https://www.oanda.com/account/settings/api")
account_id = st.sidebar.text_input("Account ID", help="Your OANDA account ID (demo/live)")
environment = st.sidebar.selectbox("Environment", ["practice", "live"], help="Use 'practice' for testing")
if st.sidebar.button("Test API Connection"):
    if api_token and account_id:
        try:
            api = API(access_token=api_token, environment=environment)
            r = accounts_endpoint.AccountDetails(accountID=account_id)
            api.request(r)
            st.sidebar.success("✅ API Connected Successfully!")
        except V20Error as e:
            st.sidebar.error(f"❌ API Error: {e}")
    else:
        st.sidebar.warning("Enter API Token and Account ID to test.")

# Technical Indicators Selection (for visualization only; all computed for strategies)
st.sidebar.subheader("Display Indicators")
show_sma = st.sidebar.checkbox("Simple Moving Average (SMA 20)", True)
show_ema = st.sidebar.checkbox("Exponential Moving Average (EMA 20)", True)
show_bb = st.sidebar.checkbox("Bollinger Bands (20, 2)", True)
show_rsi = st.sidebar.checkbox("RSI (14)", True)
show_macd = st.sidebar.checkbox("MACD", True)

# Real-Time Refresh Button
if st.sidebar.button("🔄 Refresh Data & Check Alerts"):
    st.rerun()

# Fetch forex data using yfinance (for charts; OANDA for orders)
@st.cache_data(ttl=60)  # Cache for 1 minute for more "real-time" feel
def fetch_forex_data(ticker, days):
    try:
        data = yf.download(ticker, period=f"{days}d", interval="1d", progress=False)
        if not data.empty:
            df = data.reset_index()
            df['Rate'] = 1 / df['Close']  # Adjust to quote per base
            df = df[['Date', 'Rate']]
            df['Date'] = pd.to_datetime(df['Date'])
            df = df.sort_values("Date").reset_index(drop=True)
            return df
        else:
            st.warning("No data returned from yfinance.")
            return pd.DataFrame()
    except Exception as e:
        st.error(f"Error fetching data: {e}")
        return pd.DataFrame()

df = fetch_forex_data(ticker, days)

# Compute Technical Indicators (always compute all for strategies)
@st.cache_data
def compute_indicators(df):
    if df.empty or len(df) < 26:  # Min for MACD
        return df
    
    df = df.copy()
    
    # Always compute
    df['SMA_20'] = df['Rate'].rolling(window=20).mean()
    df['EMA_20'] = df['Rate'].ewm(span=20).mean()
    
    df['BBM_20_2.0'] = df['Rate'].rolling(window=20).mean()
    bb_std = df['Rate'].rolling(window=20).std()
    df['BBU_20_2.0'] = df['BBM_20_2.0'] + (bb_std * 2)
    df['BBL_20_2.0'] = df['BBM_20_2.0'] - (bb_std * 2)
    
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
    
    ema12 = df['Rate'].ewm(span=12).mean()
    ema26 = df['Rate'].ewm(span=26).mean()
    df['MACD'] = ema12 - ema26
    df['MACD_Signal'] = df['MACD'].ewm(span=9).mean()
    df['MACD_Hist'] = df['MACD'] - df['MACD_Signal']
    
    return df

if not df.empty:
    df = compute_indicators(df)

# Auto-Trading Logic (Simulation)
@st.cache_data
def generate_signals(df, strategy, rsi_low, rsi_high):
    if df.empty:
        return df
    df = df.copy()
    signals = []
    positions = []
    current_position = None
    
    for i in range(len(df)):
        signal = 'HOLD'
        
        if strategy == "RSI Oversold/Overbought":
            rsi_val = df['RSI'].iloc[i]
            if pd.notna(rsi_val):
                if rsi_val < rsi_low and current_position != 'LONG':
                    signal = 'BUY'
                    current_position = 'LONG'
                elif rsi_val > rsi_high and current_position != 'SHORT':
                    signal = 'SELL'
                    current_position = 'SHORT'
        
        elif strategy == "MA Crossover":
            if i > 0 and pd.notna(df['SMA_20'].iloc[i]) and pd.notna(df['EMA_20'].iloc[i]):
                prev_sma = df['SMA_20'].iloc[i-1]
                prev_ema = df['EMA_20'].iloc[i-1]
                if df['EMA_20'].iloc[i] > df['SMA_20'].iloc[i] and prev_ema <= prev_sma:
                    signal = 'BUY'
                    current_position = 'LONG'
                elif df['EMA_20'].iloc[i] < df['SMA_20'].iloc[i] and prev_ema >= prev_sma:
                    signal = 'SELL'
                    current_position = 'SHORT'
        
        elif strategy == "Combined":
            rsi_val = df['RSI'].iloc[i] if pd.notna(df['RSI'].iloc[i]) else 50
            ema_val = df['EMA_20'].iloc[i]
            sma_val = df['SMA_20'].iloc[i]
            if pd.notna(ema_val) and pd.notna(sma_val):
                ma_buy = ema_val > sma_val
                ma_sell = ema_val < sma_val
            else:
                ma_buy = ma_sell = False
            if i > 0:
                prev_ema = df['EMA_20'].iloc[i-1]
                prev_sma = df['SMA_20'].iloc[i-1]
                if pd.notna(prev_ema) and pd.notna(prev_sma):
                    ma_buy = ma_buy and prev_ema <= prev_sma
                    ma_sell = ma_sell and prev_ema >= prev_sma
            rsi_buy = rsi_val < rsi_low
            rsi_sell = rsi_val > rsi_high
            if rsi_buy and ma_buy and current_position != 'LONG':
                signal = 'BUY'
                current_position = 'LONG'
            elif (rsi_sell or ma_sell) and current_position != 'SHORT':
                signal = 'SELL'
                current_position = 'SHORT'
        
        signals.append(signal)
        positions.append(current_position)
    
    df['Signal'] = signals
    df['Position'] = positions
    return df

if not df.empty:
    df = generate_signals(df, strategy, rsi_threshold_low, rsi_threshold_high)

# Current Signal & Real-Time Alert
if not df.empty:
    current_signal = df['Signal'].iloc[-1]
    last_signal = st.session_state.get('last_signal', 'HOLD')
    
    if current_signal != last_signal:
        if current_signal == 'BUY':
            st.toast("🚨 ALERT: BUY Signal Detected! Consider entering a long position.", icon="📈")
        elif current_signal == 'SELL':
            st.toast("🚨 ALERT: SELL Signal Detected! Consider entering a short position.", icon="📉")
        else:
            st.toast("📊 Signal Updated: HOLD", icon="⏸️")
        st.session_state.last_signal = current_signal
    
    st.sidebar.metric("Current Signal", current_signal)
    if current_signal == 'BUY':
        st.sidebar.success("Recommendation: BUY")
    elif current_signal == 'SELL':
        st.sidebar.error("Recommendation: SELL")
    else:
        st.sidebar.info("Recommendation: HOLD")

# Place Real Order via OANDA API (New: Replaces Simulate)
if st.sidebar.button("🚀 Place Real Order (Demo/Live)"):
    if not df.empty and current_signal != 'HOLD' and api_token and account_id:
        entry_price = df['Rate'].iloc[-1]
        # OANDA instrument format: e.g., EUR_USD
        instrument = f"{quote}_{base}"
        units = int(stake / entry_price * 10000)  # Approximate units (pip value adjustment; customize for pair)
        if current_signal == 'SELL':
            units = -units  # Negative for sell
        
        try:
            api = API(access_token=api_token, environment=environment)
            # Market order
            order_data = {
                "order": {
                    "instrument": instrument,
                    "units": str(units),
                    "type": "MARKET",
                    "positionFill": "DEFAULT"
                }
            }
            r = orders_endpoint.OrderCreate(accountID=account_id, data=order_data)
            api.request(r)
            st.success(f"✅ Order Placed: {current_signal} {abs(units)} units of {instrument} at ~{entry_price:.5f}")
            
            # Optional: Add SL/TP (requires more config; simplified here)
            if use_stop_loss or use_take_profit:
                st.info("SL/TP not auto-added; manage manually in OANDA dashboard.")
            
            # Log to history
            pnl = 0  # Real P&L tracked by broker
            trade_type = "Long" if units > 0 else "Short"
            st.session_state.trade_history = st.session_state.get('trade_history', []) + [{
                'Date': datetime.now().strftime("%Y-%m-%d %H:%M"),
                'Pair': f"{base}/{quote}",
                'Type': trade_type,
                'Units': units,
                'Entry': entry_price,
                'P&L': pnl,  # Update later from API
                'Status': 'Live'
            }]
        except V20Error as e:
            st.error(f"❌ Order Error: {e}")
        except Exception as e:
            st.error(f"Unexpected error: {e}")
    else:
        st.warning("Enter valid signal, API details, or use 'Simulate' for demo.")

# Simulate Trade (Fallback)
if st.sidebar.button("Simulate Trade (No API)"):
    if not df.empty and current_signal != 'HOLD':
        entry_price = df['Rate'].iloc[-1]
        position_size = stake / entry_price
        
        if use_take_profit:
            tp_mult = 1.04 if current_signal == 'BUY' else 0.96
            exit_price = entry_price * tp_mult
        else:
            tp_mult = 1.01 if current_signal == 'BUY' else 0.99
            exit_price = entry_price * tp_mult
        
        if current_signal == 'BUY':
            pnl = (exit_price - entry_price) * position_size
            trade_type = "Long"
        else:
            pnl = (entry_price - exit_price) * position_size
            trade_type = "Short"
        
        st.session_state.trade_history = st.session_state.get('trade_history', []) + [{
            'Date': datetime.now().strftime("%Y-%m-%d %H:%M"),
            'Pair': f"{base}/{quote}",
            'Type': trade_type,
            'Entry': entry_price,
            'Stake': stake,
            'P&L': pnl,
            'Exit': exit_price,
            'Status': 'Simulated'
        }]
        st.success(f"Simulated {trade_type} Trade: Entry {entry_price:.5f}, Exit {exit_price:.5f}, P&L ${pnl:.2f}")

# Trade History
if 'trade_history' in st.session_state and st.session_state.trade_history:
    st.subheader("Trade History")
    history_df = pd.DataFrame(st.session_state.trade_history)
    st.dataframe(history_df)
    total_pnl = history_df[history_df['Status'] == 'Simulated']['P&L'].sum() if 'P&L' in history_df.columns else 0
    st.metric("Simulated Total P&L", f"${total_pnl:.2f}")

# Main content: Metrics
col1, col2 = st.columns(2)
with col1:
    current_rate = df["Rate"].iloc[-1] if not df.empty else "N/A"
    delta = df["Rate"].iloc[-1] - df["Rate"].iloc[0] if len(df) > 1 else 0
    st.metric("Current Rate", f"{current_rate:.5f}" if isinstance(current_rate, (int, float)) else current_rate, delta)

with col2:
    if not df.empty and len(df) > 1:
        change_pct = ((df["Rate"].iloc[-1] - df["Rate"].iloc[0]) / df["Rate"].iloc[0]) * 100
        st.metric("Change (Period)", f"{change_pct:.2f}%", change_pct)

# Chart with Indicators and Signals
if not df.empty and len(df) >= 26:
    num_rows = 1
    if show_rsi: num_rows += 1
    if show_macd: num_rows += 1
    
    fig = make_subplots(
        rows=num_rows, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.05,
        subplot_titles=(f"{base}/{quote} Rate & Signals", "RSI" if show_rsi else "", "MACD" if show_macd else ""),
        row_heights=[0.6] + [0.2] * (num_rows - 1)
    )
    
    # Price and Indicators
    fig.add_trace(go.Scatter(x=df['Date'], y=df['Rate'], name='Rate', line=dict(color='blue')), row=1, col=1)
    if show_sma:
        fig.add_trace(go.Scatter(x=df['Date'], y=df['SMA_20'], name='SMA 20', line=dict(color='orange')), row=1, col=1)
    if show_ema:
        fig.add_trace(go.Scatter(x=df['Date'], y=df['EMA_20'], name='EMA 20', line=dict(color='green')), row=1, col=1)
    
    if show_bb:
        fig.add_trace(go.Scatter(x=df['Date'], y=df['BBU_20_2.0'], name='BB Upper', line=dict(color='red', dash='dash')), row=1, col=1)
        fig.add_trace(go.Scatter(x=df['Date'], y=df['BBL_20_2.0'], name='BB Lower', line=dict(color='green', dash='dash')), row=1, col=1)
        fig.add_trace(go.Scatter(
            x=pd.concat([df['Date'], df['Date'][::-1]]), 
            y=pd.concat([df['BBU_20_2.0'], df['BBL_20_2.0'][::-1]]), 
            fill='toself', fillcolor='rgba(128,128,128,0.2)', 
            line=dict(color='rgba(255,255,255,0)'), name='BB Band', showlegend=False
        ), row=1, col=1)
        if not show_sma:
            fig.add_trace(go.Scatter(x=df['Date'], y=df['BBM_20_2.0'], name='BB Middle', line=dict(color='orange')), row=1, col=1)
    
    # Signals
    buy_signals = df[df['Signal'] == 'BUY']
    sell_signals = df[df['Signal'] == 'SELL']
    if not buy_signals.empty:
        fig.add_trace(go.Scatter(x=buy_signals['Date'], y=buy_signals['Rate'], mode='markers', marker=dict(color='green', size=10, symbol='triangle-up'), name='Buy Signal'), row=1, col=1)
    if not sell_signals.empty:
        fig.add_trace(go.Scatter(x=sell_signals['Date'], y=sell_signals['Rate'], mode='markers', marker=dict(color='red', size=10, symbol='triangle-down'), name='Sell Signal'), row=1, col=1)
    
    # RSI subplot
    if show_rsi:
        rsi_row = 2
        fig.add_trace(go.Scatter(x=df['Date'], y=df['RSI'], name='RSI', line=dict(color='purple')), row=rsi_row, col=1)
        fig.add_hline(y=rsi_threshold_high, line_dash="dash", line_color="red", row=rsi_row, col=1)
        fig.add_hline(y=rsi_threshold_low, line_dash="dash", line_color="green", row=rsi_row, col=1)
        num_rows = max(num_rows, 2)
    
    # MACD subplot
    if show_macd:
        macd_row = 3 if show_rsi else 2
        fig.add_trace(go.Scatter(x=df['Date'], y=df['MACD'], name='MACD', line=dict(color='blue')), row=macd_row, col=1)
        fig.add_trace(go.Scatter(x=df['Date'], y=df['MACD_Signal'], name='Signal', line=dict(color='red')), row=macd_row, col=1)
        fig.add_trace(go.Bar(x=df['Date'], y=df['MACD_Hist'], name='Histogram', marker_color='gray'), row=macd_row, col=1)
        num_rows = max(num_rows, macd_row)
    
    fig.update_layout(height=600 + 200 * (num_rows - 1), title=f"{base}/{quote} Chart with Auto-Trading Signals (Last {days} Days)", xaxis_rangeslider_visible=False)
    fig.update_xaxes(title_text="Date", row=num_rows, col=1)
    fig.update_yaxes(title_text="Rate", row=1, col=1)
    if show_rsi:
        fig.update_yaxes(title_text="RSI", row=2, col=1, range=[0, 100])
    if show_macd:
        fig.update_yaxes(title_text="MACD", row=macd_row, col=1)
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Data table
    st.subheader("Historical Data with Signals")
    display_cols = ['Date', 'Rate', 'Signal', 'Position']
    if show_sma: display_cols.append('SMA_20')
    if show_ema: display_cols.append('EMA_20')
    if show_bb: display_cols.extend(['BBU_20_2.0', 'BBM_20_2.0', 'BBL_20_2.0'])
    if show_rsi: display_cols.append('RSI')
    if show_macd: display_cols.extend(['MACD', 'MACD_Signal', 'MACD_Hist'])
    st.dataframe(df[display_cols].round(5), use_container_width=True)
else:
    st.info("No data available. Try adjusting parameters.")

# Footer
st.sidebar.markdown("---")
st.sidebar.info("⚠️ REAL TRADING RISK: Use 'practice' environment for testing. This integrates OANDA v20 API for market orders. Customize units/SL/TP for production. Built with Streamlit & yfinance/oandapyV20.")

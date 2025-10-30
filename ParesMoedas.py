import streamlit as st
import pandas as pd
import yfinance as yf
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime, timedelta
import numpy as np

# Page config
st.set_page_config(page_title="Auto-Trading Forex System", layout="wide")

# Title
st.title("🤖 Auto-Trading Forex System with Manual Stake")

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

# Technical Indicators Selection (for visualization)
st.sidebar.subheader("Display Indicators")
show_sma = st.sidebar.checkbox("Simple Moving Average (SMA 20)", True)
show_ema = st.sidebar.checkbox("Exponential Moving Average (EMA 20)", True)
show_bb = st.sidebar.checkbox("Bollinger Bands (20, 2)", True)
show_rsi = st.sidebar.checkbox("RSI (14)", True)
show_macd = st.sidebar.checkbox("MACD", True)

# Fetch forex data using yfinance
@st.cache_data(ttl=300)
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

# Compute Technical Indicators
@st.cache_data
def compute_indicators(df, show_sma, show_ema, show_bb, show_rsi, show_macd):
    if df.empty or len(df) < 14:
        return df
    
    df = df.copy()
    
    if show_sma:
        df['SMA_20'] = df['Rate'].rolling(window=20).mean()
    if show_ema:
        df['EMA_20'] = df['Rate'].ewm(span=20).mean()
    
    if show_bb:
        df['BBM_20_2.0'] = df['Rate'].rolling(window=20).mean()
        bb_std = df['Rate'].rolling(window=20).std()
        df['BBU_20_2.0'] = df['BBM_20_2.0'] + (bb_std * 2)
        df['BBL_20_2.0'] = df['BBM_20_2.0'] - (bb_std * 2)
    
    if show_rsi:
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
    
    if show_macd:
        ema12 = df['Rate'].ewm(span=12).mean()
        ema26 = df['Rate'].ewm(span=26).mean()
        df['MACD'] = ema12 - ema26
        df['MACD_Signal'] = df['MACD'].ewm(span=9).mean()
        df['MACD_Hist'] = df['MACD'] - df['MACD_Signal']
    
    return df

if not df.empty:
    df = compute_indicators(df, show_sma, show_ema, show_bb, show_rsi, show_macd)

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
        row = df.iloc[i]
        signal = 'HOLD'
        
        if strategy == "RSI Oversold/Overbought" and 'RSI' in df.columns:
            if row['RSI'] < rsi_low and current_position != 'LONG':
                signal = 'BUY'
                current_position = 'LONG'
            elif row['RSI'] > rsi_high and current_position != 'SHORT':
                signal = 'SELL'
                current_position = 'SHORT'
        
        elif strategy == "MA Crossover" and 'SMA_20' in df.columns and 'EMA_20' in df.columns:
            if i > 0:
                prev_sma = df.iloc[i-1]['SMA_20']
                prev_ema = df.iloc[i-1]['EMA_20']
                if row['EMA_20'] > row['SMA_20'] and prev_ema <= prev_sma:
                    signal = 'BUY'
                    current_position = 'LONG'
                elif row['EMA_20'] < row['SMA_20'] and prev_ema >= prev_sma:
                    signal = 'SELL'
                    current_position = 'SHORT'
        
        elif strategy == "Combined":
            rsi_buy = row['RSI'] < rsi_low if 'RSI' in df.columns else False
            rsi_sell = row['RSI'] > rsi_high if 'RSI' in df.columns else False
            ma_buy = (row['EMA_20'] > row['SMA_20']) if all(col in df.columns for col in ['EMA_20', 'SMA_20']) else False
            ma_sell = (row['EMA_20'] < row['SMA_20']) if all(col in df.columns for col in ['EMA_20', 'SMA_20']) else False
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

# Current Signal
if not df.empty:
    current_signal = df['Signal'].iloc[-1]
    st.sidebar.metric("Current Signal", current_signal)
    if current_signal == 'BUY':
        st.sidebar.success("Recommendation: BUY")
    elif current_signal == 'SELL':
        st.sidebar.error("Recommendation: SELL")
    else:
        st.sidebar.info("Recommendation: HOLD")

# Simulate Trade Button
if st.sidebar.button("Simulate Trade with Stake"):
    if not df.empty and current_signal != 'HOLD':
        entry_price = df['Rate'].iloc[-1]
        position_size = stake / entry_price  # Units of quote currency
        
        # Simulate P&L based on next day's price (or random for demo; in real, use future data)
        # For demo, assume 1% move in signal direction
        if current_signal == 'BUY':
            exit_price = entry_price * 1.01  # Simulated profit
            pnl = (exit_price - entry_price) * position_size
            trade_type = "Long"
        else:
            exit_price = entry_price * 0.99  # Simulated profit for short
            pnl = (entry_price - exit_price) * position_size
            trade_type = "Short"
        
        # Apply SL/TP if enabled
        sl_price = entry_price * 0.98 if use_stop_loss and trade_type == "Long" else entry_price * 1.02
        tp_price = entry_price * 1.04 if use_take_profit and trade_type == "Long" else entry_price * 0.96
        # Simplified: assume hits TP for demo
        
        st.session_state.trade_history = st.session_state.get('trade_history', []) + [{
            'Date': datetime.now().strftime("%Y-%m-%d %H:%M"),
            'Pair': f"{base}/{quote}",
            'Type': trade_type,
            'Entry': entry_price,
            'Stake': stake,
            'P&L': pnl,
            'Exit': exit_price
        }]
        st.success(f"Simulated {trade_type} Trade: Entry {entry_price:.5f}, Exit {exit_price:.5f}, P&L ${pnl:.2f}")

# Trade History
if 'trade_history' in st.session_state and st.session_state.trade_history:
    st.subheader("Trade History")
    history_df = pd.DataFrame(st.session_state.trade_history)
    st.dataframe(history_df)
    total_pnl = history_df['P&L'].sum()
    st.metric("Total P&L", f"${total_pnl:.2f}")

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
min_days_for_chart = 20 if any([show_sma, show_ema, show_bb]) else 14 if show_rsi else 26 if show_macd else 1
if not df.empty and len(df) >= min_days_for_chart:
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
    if show_sma and 'SMA_20' in df.columns:
        fig.add_trace(go.Scatter(x=df['Date'], y=df['SMA_20'], name='SMA 20', line=dict(color='orange')), row=1, col=1)
    if show_ema and 'EMA_20' in df.columns:
        fig.add_trace(go.Scatter(x=df['Date'], y=df['EMA_20'], name='EMA 20', line=dict(color='green')), row=1, col=1)
    
    if show_bb and all(col in df.columns for col in ['BBU_20_2.0', 'BBL_20_2.0']):
        fig.add_trace(go.Scatter(x=df['Date'], y=df['BBU_20_2.0'], name='BB Upper', line=dict(color='red', dash='dash')), row=1, col=1)
        fig.add_trace(go.Scatter(x=df['Date'], y=df['BBL_20_2.0'], name='BB Lower', line=dict(color='green', dash='dash')), row=1, col=1)
        fig.add_trace(go.Scatter(
            x=pd.concat([df['Date'], df['Date'][::-1]]), 
            y=pd.concat([df['BBU_20_2.0'], df['BBL_20_2.0'][::-1]]), 
            fill='toself', fillcolor='rgba(128,128,128,0.2)', 
            line=dict(color='rgba(255,255,255,0)'), name='BB Band', showlegend=False
        ), row=1, col=1)
        if not show_sma and 'BBM_20_2.0' in df.columns:
            fig.add_trace(go.Scatter(x=df['Date'], y=df['BBM_20_2.0'], name='BB Middle', line=dict(color='orange')), row=1, col=1)
    
    # Signals
    buy_signals = df[df['Signal'] == 'BUY']
    sell_signals = df[df['Signal'] == 'SELL']
    if not buy_signals.empty:
        fig.add_trace(go.Scatter(x=buy_signals['Date'], y=buy_signals['Rate'], mode='markers', marker=dict(color='green', size=10, symbol='triangle-up'), name='Buy Signal'), row=1, col=1)
    if not sell_signals.empty:
        fig.add_trace(go.Scatter(x=sell_signals['Date'], y=sell_signals['Rate'], mode='markers', marker=dict(color='red', size=10, symbol='triangle-down'), name='Sell Signal'), row=1, col=1)
    
    # RSI and MACD subplots
    if show_rsi and 'RSI' in df.columns:
        rsi_row = 2
        fig.add_trace(go.Scatter(x=df['Date'], y=df['RSI'], name='RSI', line=dict(color='purple')), row=rsi_row, col=1)
        fig.add_hline(y=rsi_threshold_high, line_dash="dash", line_color="red", row=rsi_row, col=1)
        fig.add_hline(y=rsi_threshold_low, line_dash="dash", line_color="green", row=rsi_row, col=1)
    
    if show_macd and all(col in df.columns for col in ['MACD', 'MACD_Signal', 'MACD_Hist']):
        macd_row = 3 if show_rsi else 2
        fig.add_trace(go.Scatter(x=df['Date'], y=df['MACD'], name='MACD', line=dict(color='blue')), row=macd_row, col=1)
        fig.add_trace(go.Scatter(x=df['Date'], y=df['MACD_Signal'], name='Signal', line=dict(color='red')), row=macd_row, col=1)
        fig.add_trace(go.Bar(x=df['Date'], y=df['MACD_Hist'], name='Histogram', marker_color='gray'), row=macd_row, col=1)
    
    num_rows_final = num_rows if not show_macd or not show_rsi else num_rows
    fig.update_layout(height=600 + 200 * (num_rows_final - 1), title=f"{base}/{quote} Chart with Auto-Trading Signals (Last {days} Days)", xaxis_rangeslider_visible=False)
    fig.update_xaxes(title_text="Date", row=num_rows_final, col=1)
    fig.update_yaxes(title_text="Rate", row=1, col=1)
    if show_rsi:
        fig.update_yaxes(title_text="RSI", row=2, col=1, range=[0, 100])
    if show_macd:
        fig.update_yaxes(title_text="MACD", row=num_rows_final, col=1)
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Data table
    st.subheader("Historical Data with Signals")
    display_cols = ['Date', 'Rate', 'Signal', 'Position']
    if show_sma and 'SMA_20' in df.columns: display_cols.append('SMA_20')
    if show_ema and 'EMA_20' in df.columns: display_cols.append('EMA_20')
    if show_bb and all(col in df.columns for col in ['BBU_20_2.0', 'BBM_20_2.0', 'BBL_20_2.0']):
        display_cols.extend(['BBU_20_2.0', 'BBM_20_2.0', 'BBL_20_2.0'])
    if show_rsi and 'RSI' in df.columns: display_cols.append('RSI')
    if show_macd and all(col in df.columns for col in ['MACD', 'MACD_Signal', 'MACD_Hist']):
        display_cols.extend(['MACD', 'MACD_Signal', 'MACD_Hist'])
    st.dataframe(df[display_cols].round(5), use_container_width=True)
else:
    st.info("No data available. Try adjusting parameters.")

# Footer
st.sidebar.markdown("---")
st.sidebar.info("⚠️ SIMULATION ONLY: This is for educational purposes. Do not use for real trading without proper risk management and a licensed broker. Built with Streamlit & yfinance.")

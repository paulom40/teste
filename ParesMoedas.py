import streamlit as st
import pandas as pd
import numpy as np
import requests
import time
from datetime import datetime, timedelta
import ta  # Technical Analysis library
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Page configuration with modern theme
st.set_page_config(
    page_title="Real-Time Trading Demo with Indicators",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for modern styling
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-align: center;
        margin-bottom: 2rem;
        font-weight: bold;
    }
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1rem;
        border-radius: 10px;
        color: white;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }
    .profit-positive {
        color: #00ff88;
        font-weight: bold;
    }
    .profit-negative {
        color: #ff4444;
        font-weight: bold;
    }
    .sidebar .sidebar-content {
        background: linear-gradient(180deg, #2c3e50 0%, #3498db 100%);
    }
    .trade-buy {
        background-color: rgba(0, 255, 136, 0.1);
        border-left: 4px solid #00ff88;
        padding: 0.5rem;
        margin: 0.2rem 0;
        border-radius: 4px;
    }
    .trade-sell {
        background-color: rgba(255, 68, 68, 0.1);
        border-left: 4px solid #ff4444;
        padding: 0.5rem;
        margin: 0.2rem 0;
        border-radius: 4px;
    }
</style>
""", unsafe_allow_html=True)

# Fixed parameters
initial_bank = 1000
stake = 15
profit_target = 10
stop_loss = 30
pip_value = stake / 10

# Indicator parameters
ma_period = 20
rsi_period = 14
rsi_overbought = 70
rsi_oversold = 30
macd_fast = 12
macd_slow = 26
macd_signal = 9

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

# Initialize session state
if 'bank_balance' not in st.session_state:
    st.session_state.bank_balance = initial_bank
if 'open_trades' not in st.session_state:
    st.session_state.open_trades = []
if 'trade_history' not in st.session_state:
    st.session_state.trade_history = []
if 'current_prices' not in st.session_state:
    st.session_state.current_prices = initial_prices.copy()
if 'price_history' not in st.session_state:
    st.session_state.price_history = {}

# Function to format price
def format_price(price, pip_size):
    if pip_size == 0.01:
        return f"{price:.2f}"
    else:
        return f"{price:.4f}"

# Function to generate simulated historical prices
def generate_simulated_historical(pair, days=30):
    np.random.seed(42)
    base_price = initial_prices[pair]
    prices = []
    current_time = datetime.now()
    for i in range(days * 24):  # Hourly data for 30 days
        date = current_time - timedelta(hours=days * 24 - i)
        change = np.random.normal(0, 0.001)
        base_price += change * base_price
        prices.append({
            "date": date,
            "open": base_price * (1 - abs(change)),
            "high": base_price * (1 + abs(change)),
            "low": base_price * (1 - abs(change * 2)),
            "close": base_price
        })
    df = pd.DataFrame(prices)
    return df

# Function to calculate technical indicators
def calculate_indicators(df):
    df['MA'] = ta.trend.sma_indicator(df['close'], window=ma_period)
    df['RSI'] = ta.momentum.rsi(df['close'], window=rsi_period)
    
    macd = ta.trend.MACD(df['close'], window_fast=macd_fast, window_slow=macd_slow, window_sign=macd_signal)
    df['MACD'] = macd.macd()
    df['MACD_Signal'] = macd.macd_signal()
    df['MACD_Histogram'] = macd.macd_diff()
    
    return df

# Function to generate trading signals
def generate_signals(df):
    latest = df.iloc[-1]
    signals = []
    
    # RSI signals
    if latest['RSI'] < rsi_oversold:
        signals.append("RSI Oversold - BUY")
    elif latest['RSI'] > rsi_overbought:
        signals.append("RSI Overbought - SELL")
    
    # MACD signals
    if latest['MACD'] > latest['MACD_Signal'] and df.iloc[-2]['MACD'] <= df.iloc[-2]['MACD_Signal']:
        signals.append("MACD Bullish Crossover - BUY")
    elif latest['MACD'] < latest['MACD_Signal'] and df.iloc[-2]['MACD'] >= df.iloc[-2]['MACD_Signal']:
        signals.append("MACD Bearish Crossover - SELL")
    
    # Moving Average signals
    if latest['close'] > latest['MA'] and df.iloc[-2]['close'] <= df.iloc[-2]['MA']:
        signals.append("Price above MA - BUY")
    elif latest['close'] < latest['MA'] and df.iloc[-2]['close'] >= df.iloc[-2]['MA']:
        signals.append("Price below MA - SELL")
    
    return signals

# Function to simulate price movement
def simulate_price_movement(pair):
    current_price = st.session_state.current_prices[pair]
    volatility = 0.0005  # Increased volatility for more movement
    
    # Add some trend based on recent signals
    trend_bias = 0
    if st.session_state.price_history.get(pair) is not None:
        df = st.session_state.price_history[pair]
        if len(df) > 10:
            signals = generate_signals(df)
            for signal in signals:
                if "BUY" in signal:
                    trend_bias += 0.0002
                elif "SELL" in signal:
                    trend_bias -= 0.0002
    
    change = np.random.normal(trend_bias, volatility)
    new_price = current_price + change * current_price
    st.session_state.current_prices[pair] = new_price
    
    # Update price history
    if pair not in st.session_state.price_history:
        st.session_state.price_history[pair] = generate_simulated_historical(pair, 7)
    
    new_row = pd.DataFrame([{
        'date': datetime.now(),
        'open': current_price,
        'high': max(current_price, new_price),
        'low': min(current_price, new_price),
        'close': new_price
    }])
    
    st.session_state.price_history[pair] = pd.concat([
        st.session_state.price_history[pair], new_row
    ]).tail(200)  # Keep last 200 periods

# Function to execute trade
def execute_trade(pair, direction, entry_price):
    if st.session_state.bank_balance >= stake:
        trade = {
            'id': len(st.session_state.trade_history) + 1,
            'pair': pair,
            'direction': direction,
            'entry_price': entry_price,
            'stake': stake,
            'time': datetime.now(),
            'status': 'open',
            'profit_loss': 0
        }
        st.session_state.open_trades.append(trade)
        st.session_state.bank_balance -= stake
        return True
    return False

# Function to update open trades
def update_trades():
    for trade in st.session_state.open_trades[:]:
        if trade['status'] == 'open':
            current_price = st.session_state.current_prices[trade['pair']]
            pip_size = pip_sizes[trade['pair']]
            
            if trade['direction'] == 'BUY':
                pips = (current_price - trade['entry_price']) / pip_size
            else:  # SELL
                pips = (trade['entry_price'] - current_price) / pip_size
            
            profit_loss = pips * pip_value
            trade['profit_loss'] = profit_loss
            trade['current_price'] = current_price
            
            # Check for profit target or stop loss
            if profit_loss >= profit_target:
                trade['status'] = 'closed'
                trade['close_time'] = datetime.now()
                trade['close_price'] = current_price
                st.session_state.bank_balance += stake + profit_loss
                st.session_state.trade_history.append(trade.copy())
                st.session_state.open_trades.remove(trade)
            elif profit_loss <= -stop_loss:
                trade['status'] = 'closed'
                trade['close_time'] = datetime.now()
                trade['close_price'] = current_price
                st.session_state.bank_balance += stake + profit_loss
                st.session_state.trade_history.append(trade.copy())
                st.session_state.open_trades.remove(trade)

# Main application layout
st.markdown('<h1 class="main-header">📈 Real-Time Trading Dashboard</h1>', unsafe_allow_html=True)

# Sidebar
with st.sidebar:
    st.markdown("## 🎯 Trading Controls")
    
    selected_pair = st.selectbox("Select Trading Pair", trading_pairs)
    
    col1, col2 = st.columns(2)
    with col1:
        if st.button("🎯 BUY", use_container_width=True):
            execute_trade(selected_pair, 'BUY', st.session_state.current_prices[selected_pair])
    with col2:
        if st.button("📉 SELL", use_container_width=True):
            execute_trade(selected_pair, 'SELL', st.session_state.current_prices[selected_pair])
    
    st.markdown("---")
    st.markdown("## ⚙️ Trading Parameters")
    st.write(f"**Stake:** €{stake}")
    st.write(f"**Profit Target:** +{profit_target} pips")
    st.write(f"**Stop Loss:** -{stop_loss} pips")
    st.write(f"**Risk/Reward:** 1:{profit_target/stop_loss:.2f}")
    
    st.markdown("---")
    st.markdown("## 📊 Technical Indicators")
    st.write(f"**MA Period:** {ma_period}")
    st.write(f"**RSI Period:** {rsi_period}")
    st.write(f"**RSI Overbought:** {rsi_overbought}")
    st.write(f"**RSI Oversold:** {rsi_oversold}")

# Main content
col1, col2, col3 = st.columns(3)

with col1:
    st.markdown(f"""
    <div class="metric-card">
        <h3>💰 Bank Balance</h3>
        <h2>€{st.session_state.bank_balance:.2f}</h2>
    </div>
    """, unsafe_allow_html=True)

with col2:
    total_profit = sum(trade['profit_loss'] for trade in st.session_state.trade_history)
    profit_class = "profit-positive" if total_profit >= 0 else "profit-negative"
    st.markdown(f"""
    <div class="metric-card">
        <h3>📊 Total P&L</h3>
        <h2 class="{profit_class}">€{total_profit:.2f}</h2>
    </div>
    """, unsafe_allow_html=True)

with col3:
    st.markdown(f"""
    <div class="metric-card">
        <h3>🔓 Open Trades</h3>
        <h2>{len(st.session_state.open_trades)}</h2>
    </div>
    """, unsafe_allow_html=True)

# Price and chart section
st.markdown("---")
col1, col2 = st.columns([2, 1])

with col1:
    st.markdown(f"## 📈 {selected_pair} - Live Chart")
    
    # Initialize price history if not exists
    if selected_pair not in st.session_state.price_history:
        st.session_state.price_history[selected_pair] = generate_simulated_historical(selected_pair, 7)
    
    # Simulate price movement
    simulate_price_movement(selected_pair)
    
    # Calculate indicators
    df = calculate_indicators(st.session_state.price_history[selected_pair])
    
    # Create chart
    fig = make_subplots(rows=3, cols=1, 
                       shared_xaxes=True,
                       vertical_spacing=0.05,
                       subplot_titles=('Price with Moving Average', 'RSI', 'MACD'),
                       row_heights=[0.5, 0.25, 0.25])
    
    # Price and MA
    fig.add_trace(go.Scatter(x=df['date'], y=df['close'], name='Price', line=dict(color='#00ff88')), row=1, col=1)
    fig.add_trace(go.Scatter(x=df['date'], y=df['MA'], name=f'MA{ma_period}', line=dict(color='#ff4444', dash='dash')), row=1, col=1)
    
    # RSI
    fig.add_trace(go.Scatter(x=df['date'], y=df['RSI'], name='RSI', line=dict(color='#ffaa00')), row=2, col=1)
    fig.add_hline(y=rsi_overbought, line_dash="dash", line_color="red", row=2, col=1)
    fig.add_hline(y=rsi_oversold, line_dash="dash", line_color="green", row=2, col=1)
    
    # MACD
    fig.add_trace(go.Scatter(x=df['date'], y=df['MACD'], name='MACD', line=dict(color='#00ff88')), row=3, col=1)
    fig.add_trace(go.Scatter(x=df['date'], y=df['MACD_Signal'], name='Signal', line=dict(color='#ff4444')), row=3, col=1)
    fig.add_trace(go.Bar(x=df['date'], y=df['MACD_Histogram'], name='Histogram', marker_color='#777777'), row=3, col=1)
    
    fig.update_layout(height=600, showlegend=True, template="plotly_dark")
    st.plotly_chart(fig, use_container_width=True)

with col2:
    st.markdown("## 🎯 Trading Signals")
    
    # Generate signals
    signals = generate_signals(df)
    if signals:
        for signal in signals:
            if "BUY" in signal:
                st.success(f"✅ {signal}")
            else:
                st.error(f"❌ {signal}")
    else:
        st.info("🔍 No strong signals detected")
    
    st.markdown("---")
    st.markdown("## 💰 Current Prices")
    for pair in trading_pairs:
        pip_size = pip_sizes[pair]
        current_price = st.session_state.current_prices[pair]
        price_change = ""
        
        if len(st.session_state.price_history.get(pair, pd.DataFrame())) > 1:
            prev_price = st.session_state.price_history[pair].iloc[-2]['close']
            change_pct = ((current_price - prev_price) / prev_price) * 100
            price_change = f" ({change_pct:+.2f}%)"
        
        st.write(f"**{pair}:** {format_price(current_price, pip_size)}{price_change}")

# Trades section
st.markdown("---")
col1, col2 = st.columns(2)

with col1:
    st.markdown("## 🔓 Open Trades")
    if st.session_state.open_trades:
        for trade in st.session_state.open_trades:
            trade_class = "trade-buy" if trade['direction'] == 'BUY' else "trade-sell"
            st.markdown(f"""
            <div class="{trade_class}">
                <strong>{trade['pair']} {trade['direction']}</strong><br>
                Entry: {format_price(trade['entry_price'], pip_sizes[trade['pair']])} | 
                P&L: <span class="{'profit-positive' if trade['profit_loss'] >= 0 else 'profit-negative'}">€{trade['profit_loss']:.2f}</span>
            </div>
            """, unsafe_allow_html=True)
    else:
        st.info("No open trades")

with col2:
    st.markdown("## 📋 Trade History")
    if st.session_state.trade_history[-5:]:  # Show last 5 trades
        for trade in st.session_state.trade_history[-5:][::-1]:
            trade_class = "trade-buy" if trade['direction'] == 'BUY' else "trade-sell"
            result_class = "profit-positive" if trade['profit_loss'] >= 0 else "profit-negative"
            st.markdown(f"""
            <div class="{trade_class}">
                <strong>{trade['pair']} {trade['direction']}</strong><br>
                Result: <span class="{result_class}">€{trade['profit_loss']:.2f}</span> | 
                {trade['time'].strftime('%H:%M:%S')}
            </div>
            """, unsafe_allow_html=True)
    else:
        st.info("No trade history")

# Auto-refresh
if st.button("🔄 Refresh Data"):
    st.rerun()

# Auto-refresh every 5 seconds
time.sleep(5)
st.rerun()

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
if 'last_refresh' not in st.session_state:
    st.session_state.last_refresh = datetime.now()

# Function to format price
def format_price(price, pip_size):
    if pip_size == 0.01:
        return f"{price:.2f}"
    else:
        return f"{price:.4f}"

# Function to generate simulated historical prices with proper structure
def generate_simulated_historical(pair, periods=100):
    np.random.seed(42)
    base_price = initial_prices[pair]
    prices = []
    current_time = datetime.now()
    
    for i in range(periods):
        date = current_time - timedelta(hours=periods - i - 1)
        
        # Generate OHLC data with some randomness
        open_price = base_price
        change = np.random.normal(0, 0.001)
        close_price = base_price * (1 + change)
        high_price = max(open_price, close_price) * (1 + abs(np.random.normal(0, 0.0005)))
        low_price = min(open_price, close_price) * (1 - abs(np.random.normal(0, 0.0005)))
        
        prices.append({
            "date": date,
            "open": open_price,
            "high": high_price,
            "low": low_price,
            "close": close_price
        })
        
        base_price = close_price
    
    df = pd.DataFrame(prices)
    return df

# Function to calculate technical indicators with error handling
def calculate_indicators(df):
    try:
        # Create a copy to avoid modifying original
        df_indicators = df.copy()
        
        # Ensure we have enough data for calculations
        min_data_required = max(ma_period, rsi_period, macd_slow)
        
        if len(df_indicators) < min_data_required:
            # Initialize columns with NaN if not enough data
            df_indicators['MA'] = np.nan
            df_indicators['RSI'] = np.nan
            df_indicators['MACD'] = np.nan
            df_indicators['MACD_Signal'] = np.nan
            df_indicators['MACD_Histogram'] = np.nan
            return df_indicators
        
        # Calculate Moving Average
        df_indicators['MA'] = ta.trend.sma_indicator(df_indicators['close'], window=ma_period)
        
        # Calculate RSI
        df_indicators['RSI'] = ta.momentum.rsi(df_indicators['close'], window=rsi_period)
        
        # Calculate MACD
        macd_indicator = ta.trend.MACD(
            df_indicators['close'], 
            window_fast=macd_fast, 
            window_slow=macd_slow, 
            window_sign=macd_signal
        )
        df_indicators['MACD'] = macd_indicator.macd()
        df_indicators['MACD_Signal'] = macd_indicator.macd_signal()
        df_indicators['MACD_Histogram'] = macd_indicator.macd_diff()
        
        return df_indicators
        
    except Exception as e:
        st.error(f"Error calculating indicators: {e}")
        # Return original dataframe if calculation fails
        df['MA'] = np.nan
        df['RSI'] = np.nan
        df['MACD'] = np.nan
        df['MACD_Signal'] = np.nan
        df['MACD_Histogram'] = np.nan
        return df

# Function to generate trading signals with safe data access
def generate_signals(df):
    signals = []
    
    try:
        if len(df) < 2:
            return signals
            
        latest = df.iloc[-1]
        previous = df.iloc[-2]
        
        # Check if we have valid indicator values (not NaN)
        has_valid_rsi = 'RSI' in df.columns and pd.notna(latest['RSI']) and pd.notna(previous['RSI'])
        has_valid_macd = ('MACD' in df.columns and 'MACD_Signal' in df.columns and 
                         pd.notna(latest['MACD']) and pd.notna(latest['MACD_Signal']) and 
                         pd.notna(previous['MACD']) and pd.notna(previous['MACD_Signal']))
        has_valid_ma = 'MA' in df.columns and pd.notna(latest['MA']) and pd.notna(previous['MA'])
        
        # RSI signals
        if has_valid_rsi:
            if latest['RSI'] < rsi_oversold:
                signals.append("RSI Oversold - BUY")
            elif latest['RSI'] > rsi_overbought:
                signals.append("RSI Overbought - SELL")
        
        # MACD signals
        if has_valid_macd:
            if latest['MACD'] > latest['MACD_Signal'] and previous['MACD'] <= previous['MACD_Signal']:
                signals.append("MACD Bullish Crossover - BUY")
            elif latest['MACD'] < latest['MACD_Signal'] and previous['MACD'] >= previous['MACD_Signal']:
                signals.append("MACD Bearish Crossover - SELL")
        
        # Moving Average signals
        if has_valid_ma:
            if latest['close'] > latest['MA'] and previous['close'] <= previous['MA']:
                signals.append("Price above MA - BUY")
            elif latest['close'] < latest['MA'] and previous['close'] >= previous['MA']:
                signals.append("Price below MA - SELL")
                
    except Exception as e:
        # Don't show error for missing data, just return no signals
        if "RSI" not in str(e):
            st.error(f"Error generating signals: {e}")
    
    return signals

# Function to simulate price movement
def simulate_price_movement(pair):
    try:
        current_price = st.session_state.current_prices[pair]
        volatility = 0.0005
        
        # Initialize price history if not exists
        if pair not in st.session_state.price_history:
            st.session_state.price_history[pair] = generate_simulated_historical(pair, 100)
        
        # Add some trend based on recent signals if available
        trend_bias = 0
        df = st.session_state.price_history[pair]
        if len(df) > 10:
            df_with_indicators = calculate_indicators(df)
            signals = generate_signals(df_with_indicators)
            for signal in signals:
                if "BUY" in signal:
                    trend_bias += 0.0002
                elif "SELL" in signal:
                    trend_bias -= 0.0002
        
        change = np.random.normal(trend_bias, volatility)
        new_price = current_price * (1 + change)
        st.session_state.current_prices[pair] = new_price
        
        # Add new price data
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
        
    except Exception as e:
        st.error(f"Error simulating price movement for {pair}: {e}")

# Function to execute trade
def execute_trade(pair, direction, entry_price):
    try:
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
    except Exception as e:
        st.error(f"Error executing trade: {e}")
        return False

# Function to update open trades
def update_trades():
    try:
        trades_to_remove = []
        for i, trade in enumerate(st.session_state.open_trades):
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
                    trades_to_remove.append(i)
                elif profit_loss <= -stop_loss:
                    trade['status'] = 'closed'
                    trade['close_time'] = datetime.now()
                    trade['close_price'] = current_price
                    st.session_state.bank_balance += stake + profit_loss
                    st.session_state.trade_history.append(trade.copy())
                    trades_to_remove.append(i)
        
        # Remove closed trades in reverse order
        for i in sorted(trades_to_remove, reverse=True):
            if i < len(st.session_state.open_trades):
                st.session_state.open_trades.pop(i)
                
    except Exception as e:
        st.error(f"Error updating trades: {e}")

# Initialize price history for all pairs
for pair in trading_pairs:
    if pair not in st.session_state.price_history:
        st.session_state.price_history[pair] = generate_simulated_historical(pair, 100)

# Main application layout
st.markdown('<h1 class="main-header">📈 Real-Time Trading Dashboard</h1>', unsafe_allow_html=True)

# Sidebar
with st.sidebar:
    st.markdown("## 🎯 Trading Controls")
    
    selected_pair = st.selectbox("Select Trading Pair", trading_pairs)
    
    col1, col2 = st.columns(2)
    with col1:
        if st.button("🎯 BUY", use_container_width=True, type="primary"):
            if execute_trade(selected_pair, 'BUY', st.session_state.current_prices[selected_pair]):
                st.success("Buy order executed!")
            else:
                st.error("Insufficient funds!")
    with col2:
        if st.button("📉 SELL", use_container_width=True, type="secondary"):
            if execute_trade(selected_pair, 'SELL', st.session_state.current_prices[selected_pair]):
                st.success("Sell order executed!")
            else:
                st.error("Insufficient funds!")
    
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
    
    # Manual refresh control
    if st.button("🔄 Force Refresh"):
        st.session_state.last_refresh = datetime.now()
        st.rerun()

# Update prices and trades
for pair in trading_pairs:
    simulate_price_movement(pair)
update_trades()

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
    
    if selected_pair in st.session_state.price_history:
        df = st.session_state.price_history[selected_pair].copy()
        
        # Calculate indicators
        df_with_indicators = calculate_indicators(df)
        
        # Create chart only if we have enough data
        if len(df_with_indicators) > ma_period and pd.notna(df_with_indicators['MA'].iloc[-1]):
            fig = make_subplots(rows=3, cols=1, 
                               shared_xaxes=True,
                               vertical_spacing=0.05,
                               subplot_titles=('Price with Moving Average', 'RSI', 'MACD'),
                               row_heights=[0.5, 0.25, 0.25])
            
            # Price and MA
            fig.add_trace(go.Scatter(x=df_with_indicators['date'], 
                                   y=df_with_indicators['close'], 
                                   name='Price', 
                                   line=dict(color='#00ff88')), row=1, col=1)
            fig.add_trace(go.Scatter(x=df_with_indicators['date'], 
                                   y=df_with_indicators['MA'], 
                                   name=f'MA{ma_period}', 
                                   line=dict(color='#ff4444', dash='dash')), row=1, col=1)
            
            # RSI - only if we have valid data
            valid_rsi_data = df_with_indicators['RSI'].dropna()
            if len(valid_rsi_data) > 0:
                fig.add_trace(go.Scatter(x=df_with_indicators['date'], 
                                       y=df_with_indicators['RSI'], 
                                       name='RSI', 
                                       line=dict(color='#ffaa00')), row=2, col=1)
                fig.add_hline(y=rsi_overbought, line_dash="dash", line_color="red", row=2, col=1)
                fig.add_hline(y=rsi_oversold, line_dash="dash", line_color="green", row=2, col=1)
            else:
                fig.add_annotation(x=0.5, y=0.5, xref="paper", yref="paper",
                                 text="RSI data not available yet",
                                 showarrow=False, row=2, col=1)
            
            # MACD - only if we have valid data
            if 'MACD' in df_with_indicators.columns:
                valid_macd_data = df_with_indicators['MACD'].dropna()
                if len(valid_macd_data) > 0:
                    fig.add_trace(go.Scatter(x=df_with_indicators['date'], 
                                           y=df_with_indicators['MACD'], 
                                           name='MACD', 
                                           line=dict(color='#00ff88')), row=3, col=1)
                    fig.add_trace(go.Scatter(x=df_with_indicators['date'], 
                                           y=df_with_indicators['MACD_Signal'], 
                                           name='Signal', 
                                           line=dict(color='#ff4444')), row=3, col=1)
                    
                    if 'MACD_Histogram' in df_with_indicators.columns:
                        fig.add_trace(go.Bar(x=df_with_indicators['date'], 
                                           y=df_with_indicators['MACD_Histogram'], 
                                           name='Histogram', 
                                           marker_color='#777777'), row=3, col=1)
                else:
                    fig.add_annotation(x=0.5, y=0.5, xref="paper", yref="paper",
                                     text="MACD data not available yet",
                                     showarrow=False, row=3, col=1)
            
            fig.update_layout(height=600, showlegend=True, template="plotly_dark")
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("📊 Collecting market data... Chart will appear soon.")
    else:
        st.info("🔄 Initializing price data...")

with col2:
    st.markdown("## 🎯 Trading Signals")
    
    if selected_pair in st.session_state.price_history:
        df = st.session_state.price_history[selected_pair].copy()
        df_with_indicators = calculate_indicators(df)
        signals = generate_signals(df_with_indicators)
        
        if signals:
            for signal in signals:
                if "BUY" in signal:
                    st.success(f"✅ {signal}")
                else:
                    st.error(f"❌ {signal}")
        else:
            st.info("🔍 No strong signals detected")
    else:
        st.info("🔄 Waiting for data...")
    
    st.markdown("---")
    st.markdown("## 💰 Current Prices")
    for pair in trading_pairs:
        pip_size = pip_sizes[pair]
        current_price = st.session_state.current_prices[pair]
        
        # Calculate price change
        price_change = ""
        if pair in st.session_state.price_history and len(st.session_state.price_history[pair]) > 1:
            df_pair = st.session_state.price_history[pair]
            if len(df_pair) >= 2:
                prev_price = df_pair.iloc[-2]['close']
                change_pct = ((current_price - prev_price) / prev_price) * 100
                change_color = "profit-positive" if change_pct >= 0 else "profit-negative"
                price_change = f" <span class='{change_color}'>({change_pct:+.2f}%)</span>"
        
        st.markdown(f"**{pair}:** {format_price(current_price, pip_size)}{price_change}", unsafe_allow_html=True)

# Trades section
st.markdown("---")
col1, col2 = st.columns(2)

with col1:
    st.markdown("## 🔓 Open Trades")
    if st.session_state.open_trades:
        for trade in st.session_state.open_trades:
            trade_class = "trade-buy" if trade['direction'] == 'BUY' else "trade-sell"
            current_pl = trade['profit_loss']
            pl_class = "profit-positive" if current_pl >= 0 else "profit-negative"
            
            st.markdown(f"""
            <div class="{trade_class}">
                <strong>{trade['pair']} {trade['direction']}</strong><br>
                Entry: {format_price(trade['entry_price'], pip_sizes[trade['pair']])}<br>
                Current: {format_price(trade.get('current_price', trade['entry_price']), pip_sizes[trade['pair']])}<br>
                P&L: <span class="{pl_class}">€{current_pl:.2f}</span>
            </div>
            """, unsafe_allow_html=True)
    else:
        st.info("No open trades")

with col2:
    st.markdown("## 📋 Trade History")
    recent_trades = st.session_state.trade_history[-10:]  # Show last 10 trades
    if recent_trades:
        for trade in reversed(recent_trades):
            trade_class = "trade-buy" if trade['direction'] == 'BUY' else "trade-sell"
            result_class = "profit-positive" if trade['profit_loss'] >= 0 else "profit-negative"
            
            st.markdown(f"""
            <div class="{trade_class}">
                <strong>{trade['pair']} {trade['direction']}</strong><br>
                Result: <span class="{result_class}">€{trade['profit_loss']:.2f}</span><br>
                {trade.get('close_time', trade['time']).strftime('%H:%M:%S')}
            </div>
            """, unsafe_allow_html=True)
    else:
        st.info("No trade history yet")

# Auto-refresh info
st.markdown("---")
refresh_time = st.session_state.last_refresh.strftime("%H:%M:%S")
st.markdown(f"**Last update:** {refresh_time} | *Auto-refreshing every 5 seconds*")

# Auto-refresh
time.sleep(5)
st.rerun()

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
    page_title="Auto Trading Bot with Signal Detection",
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
    .signal-strong-buy {
        background: linear-gradient(135deg, #00ff88 0%, #00cc66 100%);
        color: white;
        padding: 0.5rem;
        border-radius: 5px;
        margin: 0.2rem 0;
        text-align: center;
        font-weight: bold;
    }
    .signal-strong-sell {
        background: linear-gradient(135deg, #ff4444 0%, #cc0000 100%);
        color: white;
        padding: 0.5rem;
        border-radius: 5px;
        margin: 0.2rem 0;
        text-align: center;
        font-weight: bold;
    }
    .signal-weak {
        background: linear-gradient(135deg, #ffaa00 0%, #ff8800 100%);
        color: white;
        padding: 0.5rem;
        border-radius: 5px;
        margin: 0.2rem 0;
        text-align: center;
        font-weight: bold;
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
    .auto-trade-active {
        background: linear-gradient(135deg, #00ff88 0%, #00cc66 100%);
        color: white;
        padding: 0.5rem;
        border-radius: 10px;
        text-align: center;
        font-weight: bold;
        animation: pulse 2s infinite;
    }
    @keyframes pulse {
        0% { opacity: 1; }
        50% { opacity: 0.7; }
        100% { opacity: 1; }
    }
</style>
""", unsafe_allow_html=True)

# Trading parameters
initial_bank = 5000
max_trades_per_pair = 2
stake = 100
profit_target = 15
stop_loss = 10
pip_value = stake / 10

# Indicator parameters
ma_fast = 10
ma_slow = 20
rsi_period = 14
rsi_overbought = 70
rsi_oversold = 30
macd_fast = 12
macd_slow = 26
macd_signal = 9
stoch_k = 14
stoch_d = 3

# Signal strength thresholds
STRONG_SIGNAL_THRESHOLD = 3  # Number of confirming indicators
WEAK_SIGNAL_THRESHOLD = 2    # Minimum indicators for weak signal

# Pip sizes for pairs
pip_sizes = {
    "EUR/USD": 0.0001,
    "GBP/USD": 0.0001,
    "USD/JPY": 0.01,
    "AUD/USD": 0.0001,
    "USD/CAD": 0.0001,
    "NZD/USD": 0.0001
}

# Trading pairs with different volatilities
trading_pairs = [
    "EUR/USD", "GBP/USD", "USD/JPY", "AUD/USD", "USD/CAD", "NZD/USD"
]

# Initial prices
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
if 'auto_trading' not in st.session_state:
    st.session_state.auto_trading = False
if 'signal_history' not in st.session_state:
    st.session_state.signal_history = {}
if 'last_auto_trade' not in st.session_state:
    st.session_state.last_auto_trade = {}

# Function to generate simulated historical prices
def generate_simulated_historical(pair, periods=200):
    np.random.seed(42)
    base_price = initial_prices[pair]
    prices = []
    current_time = datetime.now()
    
    for i in range(periods):
        date = current_time - timedelta(hours=periods - i - 1)
        
        # Generate OHLC data with realistic volatility
        open_price = base_price
        change = np.random.normal(0, 0.002)  # Increased volatility for better signals
        close_price = base_price * (1 + change)
        high_price = max(open_price, close_price) * (1 + abs(np.random.normal(0, 0.001)))
        low_price = min(open_price, close_price) * (1 - abs(np.random.normal(0, 0.001)))
        
        prices.append({
            "date": date,
            "open": open_price,
            "high": high_price,
            "low": low_price,
            "close": close_price
        })
        
        base_price = close_price
    
    return pd.DataFrame(prices)

# Function to calculate advanced technical indicators
def calculate_advanced_indicators(df):
    try:
        df_indicators = df.copy()
        
        # Moving Averages
        df_indicators['MA_Fast'] = ta.trend.sma_indicator(df_indicators['close'], window=ma_fast)
        df_indicators['MA_Slow'] = ta.trend.sma_indicator(df_indicators['close'], window=ma_slow)
        
        # RSI
        df_indicators['RSI'] = ta.momentum.rsi(df_indicators['close'], window=rsi_period)
        
        # MACD
        macd_indicator = ta.trend.MACD(df_indicators['close'], window_fast=macd_fast, 
                                     window_slow=macd_slow, window_sign=macd_signal)
        df_indicators['MACD'] = macd_indicator.macd()
        df_indicators['MACD_Signal'] = macd_indicator.macd_signal()
        df_indicators['MACD_Histogram'] = macd_indicator.macd_diff()
        
        # Stochastic
        stoch_indicator = ta.momentum.StochasticOscillator(df_indicators['high'], df_indicators['low'], 
                                                         df_indicators['close'], window=stoch_k, 
                                                         smooth_window=stoch_d)
        df_indicators['Stoch_K'] = stoch_indicator.stoch()
        df_indicators['Stoch_D'] = stoch_indicator.stoch_signal()
        
        # Bollinger Bands
        bollinger = ta.volatility.BollingerBands(df_indicators['close'], window=20, window_dev=2)
        df_indicators['BB_Upper'] = bollinger.bollinger_hband()
        df_indicators['BB_Lower'] = bollinger.bollinger_lband()
        df_indicators['BB_Middle'] = bollinger.bollinger_mavg()
        
        return df_indicators
        
    except Exception as e:
        return df

# Function to detect strong trading signals with scoring system
def detect_strong_signals(df):
    signals = []
    buy_score = 0
    sell_score = 0
    signal_details = []
    
    try:
        if len(df) < 30:
            return signals, buy_score, sell_score, signal_details
            
        latest = df.iloc[-1]
        previous = df.iloc[-2]
        
        # Check if we have valid indicator values
        has_valid_indicators = all(pd.notna(latest.get(col, np.nan)) for col in 
                                 ['MA_Fast', 'MA_Slow', 'RSI', 'MACD', 'MACD_Signal', 'Stoch_K', 'Stoch_D'])
        
        if not has_valid_indicators:
            return signals, buy_score, sell_score, signal_details
        
        # 1. Moving Average Crossover (Strong signal)
        if latest['MA_Fast'] > latest['MA_Slow'] and previous['MA_Fast'] <= previous['MA_Slow']:
            buy_score += 2
            signal_details.append("MA Crossover Bullish")
        elif latest['MA_Fast'] < latest['MA_Slow'] and previous['MA_Fast'] >= previous['MA_Slow']:
            sell_score += 2
            signal_details.append("MA Crossover Bearish")
        
        # 2. RSI Signals
        if latest['RSI'] < 30:  # Strong oversold
            buy_score += 2
            signal_details.append("RSI Strong Oversold")
        elif latest['RSI'] < rsi_oversold:  # Regular oversold
            buy_score += 1
            signal_details.append("RSI Oversold")
        elif latest['RSI'] > 80:  # Strong overbought
            sell_score += 2
            signal_details.append("RSI Strong Overbought")
        elif latest['RSI'] > rsi_overbought:  # Regular overbought
            sell_score += 1
            signal_details.append("RSI Overbought")
        
        # 3. MACD Signals
        if latest['MACD'] > latest['MACD_Signal'] and previous['MACD'] <= previous['MACD_Signal']:
            buy_score += 2
            signal_details.append("MACD Bullish Crossover")
        elif latest['MACD'] < latest['MACD_Signal'] and previous['MACD'] >= previous['MACD_Signal']:
            sell_score += 2
            signal_details.append("MACD Bearish Crossover")
        
        # 4. Stochastic Signals
        if latest['Stoch_K'] < 20 and latest['Stoch_D'] < 20:
            buy_score += 1
            signal_details.append("Stochastic Oversold")
        elif latest['Stoch_K'] > 80 and latest['Stoch_D'] > 80:
            sell_score += 1
            signal_details.append("Stochastic Overbought")
        
        # 5. Bollinger Bands
        if latest['close'] < latest['BB_Lower']:
            buy_score += 1
            signal_details.append("Below Lower Bollinger Band")
        elif latest['close'] > latest['BB_Upper']:
            sell_score += 1
            signal_details.append("Above Upper Bollinger Band")
        
        # Determine final signals based on scores
        if buy_score >= STRONG_SIGNAL_THRESHOLD:
            signals.append(("STRONG BUY", buy_score, signal_details))
        elif buy_score >= WEAK_SIGNAL_THRESHOLD:
            signals.append(("WEAK BUY", buy_score, signal_details))
        
        if sell_score >= STRONG_SIGNAL_THRESHOLD:
            signals.append(("STRONG SELL", sell_score, signal_details))
        elif sell_score >= WEAK_SIGNAL_THRESHOLD:
            signals.append(("WEAK SELL", sell_score, signal_details))
            
    except Exception as e:
        pass
    
    return signals, buy_score, sell_score, signal_details

# Function to check if we can open a new trade for a pair
def can_open_trade(pair, direction):
    # Count current open trades for this pair
    current_trades = [t for t in st.session_state.open_trades if t['pair'] == pair]
    
    if len(current_trades) >= max_trades_per_pair:
        return False
    
    # Don't open same direction trade if we already have one
    same_direction_trades = [t for t in current_trades if t['direction'] == direction]
    if same_direction_trades:
        return False
    
    return True

# Function to execute auto trade based on signals
def execute_auto_trades():
    if not st.session_state.auto_trading:
        return
    
    auto_trades_executed = []
    
    for pair in trading_pairs:
        if pair in st.session_state.price_history:
            df = st.session_state.price_history[pair].copy()
            df_with_indicators = calculate_advanced_indicators(df)
            
            signals, buy_score, sell_score, details = detect_strong_signals(df_with_indicators)
            
            # Store signal history for display
            st.session_state.signal_history[pair] = {
                'signals': signals,
                'buy_score': buy_score,
                'sell_score': sell_score,
                'time': datetime.now(),
                'details': details
            }
            
            # Execute trades based on strong signals
            for signal_type, score, signal_details in signals:
                if "STRONG BUY" in signal_type and can_open_trade(pair, 'BUY'):
                    if execute_trade(pair, 'BUY', st.session_state.current_prices[pair]):
                        auto_trades_executed.append(f"AUTO BUY {pair} (Score: {score})")
                        st.session_state.last_auto_trade[pair] = datetime.now()
                
                elif "STRONG SELL" in signal_type and can_open_trade(pair, 'SELL'):
                    if execute_trade(pair, 'SELL', st.session_state.current_prices[pair]):
                        auto_trades_executed.append(f"AUTO SELL {pair} (Score: {score})")
                        st.session_state.last_auto_trade[pair] = datetime.now()
    
    return auto_trades_executed

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
                'profit_loss': 0,
                'type': 'AUTO' if st.session_state.auto_trading else 'MANUAL'
            }
            st.session_state.open_trades.append(trade)
            st.session_state.bank_balance -= stake
            return True
        return False
    except Exception as e:
        return False

# Function to simulate price movement with trends
def simulate_price_movement(pair):
    try:
        current_price = st.session_state.current_prices[pair]
        
        # Add some market trends and patterns
        volatility = 0.001
        
        # Check for recent strong signals to influence price direction
        trend_bias = 0
        if pair in st.session_state.signal_history:
            signal_info = st.session_state.signal_history[pair]
            if signal_info['buy_score'] > signal_info['sell_score']:
                trend_bias += 0.0003
            elif signal_info['sell_score'] > signal_info['buy_score']:
                trend_bias -= 0.0003
        
        change = np.random.normal(trend_bias, volatility)
        new_price = current_price * (1 + change)
        st.session_state.current_prices[pair] = new_price
        
        # Initialize price history if not exists
        if pair not in st.session_state.price_history:
            st.session_state.price_history[pair] = generate_simulated_historical(pair, 200)
        
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
        ]).tail(300)
        
    except Exception as e:
        pass

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
        
        # Remove closed trades
        for i in sorted(trades_to_remove, reverse=True):
            if i < len(st.session_state.open_trades):
                st.session_state.open_trades.pop(i)
                
    except Exception as e:
        pass

# Initialize price history
for pair in trading_pairs:
    if pair not in st.session_state.price_history:
        st.session_state.price_history[pair] = generate_simulated_historical(pair, 200)

# Main application layout
st.markdown('<h1 class="main-header">🤖 Auto Trading Bot</h1>', unsafe_allow_html=True)

# Sidebar
with st.sidebar:
    st.markdown("## 🎯 Trading Controls")
    
    # Auto Trading Toggle
    col1, col2 = st.columns(2)
    with col1:
        if st.button("🚀 Start Auto Trading", use_container_width=True, type="primary"):
            st.session_state.auto_trading = True
            st.success("Auto Trading Started!")
    with col2:
        if st.button("🛑 Stop Auto Trading", use_container_width=True, type="secondary"):
            st.session_state.auto_trading = False
            st.warning("Auto Trading Stopped!")
    
    st.markdown("---")
    st.markdown("## 📊 Trading Parameters")
    st.write(f"**Bank:** €{st.session_state.bank_balance:.2f}")
    st.write(f"**Stake per trade:** €{stake}")
    st.write(f"**Profit Target:** +{profit_target} pips")
    st.write(f"**Stop Loss:** -{stop_loss} pips")
    st.write(f"**Max trades per pair:** {max_trades_per_pair}")
    
    st.markdown("---")
    st.markdown("## ⚙️ Signal Settings")
    st.write(f"**Strong Signal Threshold:** {STRONG_SIGNAL_THRESHOLD}+ indicators")
    st.write(f"**Weak Signal Threshold:** {WEAK_SIGNAL_THRESHOLD}+ indicators")
    
    st.markdown("---")
    st.markdown("## 📈 Indicators Used")
    st.write("• Moving Average Crossover")
    st.write("• RSI (Oversold/Overbought)")
    st.write("• MACD Crossover")
    st.write("• Stochastic Oscillator")
    st.write("• Bollinger Bands")

# Update prices and execute auto trades
for pair in trading_pairs:
    simulate_price_movement(pair)

auto_trades_executed = execute_auto_trades()
update_trades()

# Main dashboard
col1, col2, col3, col4 = st.columns(4)

with col1:
    status_color = "auto-trade-active" if st.session_state.auto_trading else "metric-card"
    status_text = "ACTIVE" if st.session_state.auto_trading else "INACTIVE"
    st.markdown(f"""
    <div class="{status_color}">
        <h3>🤖 Auto Trading</h3>
        <h2>{status_text}</h2>
    </div>
    """, unsafe_allow_html=True)

with col2:
    st.markdown(f"""
    <div class="metric-card">
        <h3>💰 Bank Balance</h3>
        <h2>€{st.session_state.bank_balance:.2f}</h2>
    </div>
    """, unsafe_allow_html=True)

with col3:
    total_profit = sum(trade['profit_loss'] for trade in st.session_state.trade_history)
    profit_class = "profit-positive" if total_profit >= 0 else "profit-negative"
    st.markdown(f"""
    <div class="metric-card">
        <h3>📊 Total P&L</h3>
        <h2 class="{profit_class}">€{total_profit:.2f}</h2>
    </div>
    """, unsafe_allow_html=True)

with col4:
    st.markdown(f"""
    <div class="metric-card">
        <h3>🔓 Open Trades</h3>
        <h2>{len(st.session_state.open_trades)}</h2>
    </div>
    """, unsafe_allow_html=True)

# Show auto trade executions
if auto_trades_executed:
    st.markdown("---")
    st.markdown("## 🤖 Auto Trade Executions")
    for trade in auto_trades_executed:
        if "BUY" in trade:
            st.success(f"✅ {trade}")
        else:
            st.error(f"❌ {trade}")

# Signal Strength Dashboard
st.markdown("---")
st.markdown("## 🎯 Live Signal Detection")

signal_cols = st.columns(len(trading_pairs))

for idx, pair in enumerate(trading_pairs):
    with signal_cols[idx]:
        st.markdown(f"**{pair}**")
        
        if pair in st.session_state.signal_history:
            signal_info = st.session_state.signal_history[pair]
            buy_score = signal_info['buy_score']
            sell_score = signal_info['sell_score']
            
            # Display signal strength
            if buy_score >= STRONG_SIGNAL_THRESHOLD:
                st.markdown(f'<div class="signal-strong-buy">STRONG BUY<br>Score: {buy_score}</div>', unsafe_allow_html=True)
            elif sell_score >= STRONG_SIGNAL_THRESHOLD:
                st.markdown(f'<div class="signal-strong-sell">STRONG SELL<br>Score: {sell_score}</div>', unsafe_allow_html=True)
            elif buy_score >= WEAK_SIGNAL_THRESHOLD:
                st.markdown(f'<div class="signal-weak">WEAK BUY<br>Score: {buy_score}</div>', unsafe_allow_html=True)
            elif sell_score >= WEAK_SIGNAL_THRESHOLD:
                st.markdown(f'<div class="signal-weak">WEAK SELL<br>Score: {sell_score}</div>', unsafe_allow_html=True)
            else:
                st.info("No Signal")
            
            # Show last update
            last_update = signal_info['time'].strftime("%H:%M:%S")
            st.caption(f"Last: {last_update}")
        else:
            st.info("Analyzing...")

# Detailed charts for selected pair
st.markdown("---")
selected_pair = st.selectbox("Select Pair for Detailed Analysis", trading_pairs)

col1, col2 = st.columns([2, 1])

with col1:
    st.markdown(f"## 📈 {selected_pair} - Advanced Analysis")
    
    if selected_pair in st.session_state.price_history:
        df = st.session_state.price_history[selected_pair].copy()
        df_with_indicators = calculate_advanced_indicators(df)
        
        # Create advanced chart
        fig = make_subplots(rows=4, cols=1, 
                           shared_xaxes=True,
                           vertical_spacing=0.03,
                           subplot_titles=('Price with MA & Bollinger Bands', 'RSI', 'MACD', 'Stochastic'),
                           row_heights=[0.4, 0.2, 0.2, 0.2])
        
        # Price with MA and Bollinger Bands
        fig.add_trace(go.Scatter(x=df_with_indicators['date'], y=df_with_indicators['close'], 
                               name='Price', line=dict(color='#00ff88')), row=1, col=1)
        fig.add_trace(go.Scatter(x=df_with_indicators['date'], y=df_with_indicators['MA_Fast'], 
                               name=f'MA{ma_fast}', line=dict(color='#ff4444')), row=1, col=1)
        fig.add_trace(go.Scatter(x=df_with_indicators['date'], y=df_with_indicators['MA_Slow'], 
                               name=f'MA{ma_slow}', line=dict(color='#4444ff')), row=1, col=1)
        
        if 'BB_Upper' in df_with_indicators.columns:
            fig.add_trace(go.Scatter(x=df_with_indicators['date'], y=df_with_indicators['BB_Upper'], 
                                   name='BB Upper', line=dict(color='#888888', dash='dash')), row=1, col=1)
            fig.add_trace(go.Scatter(x=df_with_indicators['date'], y=df_with_indicators['BB_Lower'], 
                                   name='BB Lower', line=dict(color='#888888', dash='dash')), row=1, col=1)
        
        # RSI
        fig.add_trace(go.Scatter(x=df_with_indicators['date'], y=df_with_indicators['RSI'], 
                               name='RSI', line=dict(color='#ffaa00')), row=2, col=1)
        fig.add_hline(y=rsi_overbought, line_dash="dash", line_color="red", row=2, col=1)
        fig.add_hline(y=rsi_oversold, line_dash="dash", line_color="green", row=2, col=1)
        
        # MACD
        fig.add_trace(go.Scatter(x=df_with_indicators['date'], y=df_with_indicators['MACD'], 
                               name='MACD', line=dict(color='#00ff88')), row=3, col=1)
        fig.add_trace(go.Scatter(x=df_with_indicators['date'], y=df_with_indicators['MACD_Signal'], 
                               name='Signal', line=dict(color='#ff4444')), row=3, col=1)
        fig.add_trace(go.Bar(x=df_with_indicators['date'], y=df_with_indicators['MACD_Histogram'], 
                           name='Histogram', marker_color='#777777'), row=3, col=1)
        
        # Stochastic
        fig.add_trace(go.Scatter(x=df_with_indicators['date'], y=df_with_indicators['Stoch_K'], 
                               name='Stoch %K', line=dict(color='#00ff88')), row=4, col=1)
        fig.add_trace(go.Scatter(x=df_with_indicators['date'], y=df_with_indicators['Stoch_D'], 
                               name='Stoch %D', line=dict(color='#ff4444')), row=4, col=1)
        fig.add_hline(y=80, line_dash="dash", line_color="red", row=4, col=1)
        fig.add_hline(y=20, line_dash="dash", line_color="green", row=4, col=1)
        
        fig.update_layout(height=800, showlegend=True, template="plotly_dark")
        st.plotly_chart(fig, use_container_width=True)

with col2:
    st.markdown("## 📊 Signal Details")
    
    if selected_pair in st.session_state.signal_history:
        signal_info = st.session_state.signal_history[selected_pair]
        
        st.metric("Buy Signal Score", signal_info['buy_score'])
        st.metric("Sell Signal Score", signal_info['sell_score'])
        
        st.markdown("### Signal Components:")
        for detail in signal_info.get('details', []):
            st.write(f"• {detail}")
        
        st.markdown("### Trading Rules:")
        st.write(f"• Strong Signal: ≥{STRONG_SIGNAL_THRESHOLD} confirming indicators")
        st.write(f"• Auto-executes STRONG signals only")
        st.write(f"• Max {max_trades_per_pair} trades per pair")
        st.write(f"• No same-direction duplicates")

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
            trade_type = trade.get('type', 'MANUAL')
            
            st.markdown(f"""
            <div class="{trade_class}">
                <strong>{trade['pair']} {trade['direction']} ({trade_type})</strong><br>
                Entry: {trade['entry_price']:.4f}<br>
                P&L: <span class="{pl_class}">€{current_pl:.2f}</span><br>
                <small>{trade['time'].strftime('%H:%M:%S')}</small>
            </div>
            """, unsafe_allow_html=True)
    else:
        st.info("No open trades")

with col2:
    st.markdown("## 📋 Recent Auto Trades")
    auto_trades = [t for t in st.session_state.trade_history if t.get('type') == 'AUTO']
    recent_trades = auto_trades[-8:] if auto_trades else []
    
    if recent_trades:
        for trade in reversed(recent_trades):
            trade_class = "trade-buy" if trade['direction'] == 'BUY' else "trade-sell"
            result_class = "profit-positive" if trade['profit_loss'] >= 0 else "profit-negative"
            
            st.markdown(f"""
            <div class="{trade_class}">
                <strong>{trade['pair']} {trade['direction']}</strong><br>
                Result: <span class="{result_class}">€{trade['profit_loss']:.2f}</span><br>
                <small>{trade.get('close_time', trade['time']).strftime('%H:%M:%S')}</small>
            </div>
            """, unsafe_allow_html=True)
    else:
        st.info("No auto trade history yet")

# Performance metrics
st.markdown("---")
st.markdown("## 📈 Performance Metrics")

if st.session_state.trade_history:
    total_trades = len(st.session_state.trade_history)
    winning_trades = len([t for t in st.session_state.trade_history if t['profit_loss'] > 0])
    losing_trades = len([t for t in st.session_state.trade_history if t['profit_loss'] < 0])
    win_rate = (winning_trades / total_trades) * 100 if total_trades > 0 else 0
    
    auto_trades = [t for t in st.session_state.trade_history if t.get('type') == 'AUTO']
    manual_trades = [t for t in st.session_state.trade_history if t.get('type') == 'MANUAL']
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Trades", total_trades)
    with col2:
        st.metric("Win Rate", f"{win_rate:.1f}%")
    with col3:
        st.metric("Auto Trades", len(auto_trades))
    with col4:
        st.metric("Manual Trades", len(manual_trades))

# Auto-refresh
st.markdown("---")
st.markdown("🔄 Auto-refreshing every 3 seconds...")

time.sleep(3)
st.rerun()

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
    page_title="Auto Trading Bot - 2 Indicator Agreement",
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
    .signal-no-trade {
        background: linear-gradient(135deg, #666666 0%, #999999 100%);
        color: white;
        padding: 0.5rem;
        border-radius: 5px;
        margin: 0.2rem 0;
        text-align: center;
        font-weight: bold;
    }
    .signal-mixed {
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
    .indicator-active {
        color: #00ff88;
        font-weight: bold;
    }
    .indicator-conflict {
        color: #ffaa00;
        font-weight: bold;
    }
    .pair-card {
        background: linear-gradient(135deg, #2c3e50 0%, #3498db 100%);
        padding: 1rem;
        border-radius: 10px;
        color: white;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }
    .agreement-buy {
        border: 3px solid #00ff88;
    }
    .agreement-sell {
        border: 3px solid #ff4444;
    }
    .no-agreement {
        border: 3px solid #ffaa00;
    }
</style>
""", unsafe_allow_html=True)

# Trading parameters
initial_bank = 1000
stake = 10  # €10 per trade
profit_target = 10  # +10 pips
stop_loss = 10     # -10 pips
pip_value = stake / 10  # €1 per pip

# Indicator parameters
ma_fast = 10
ma_slow = 20
rsi_period = 14
rsi_overbought = 70
rsi_oversold = 30
macd_fast = 12
macd_slow = 26
macd_signal = 9

# Signal threshold - Require BOTH indicators to agree
REQUIRED_INDICATORS = 2

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

# Initial prices
initial_prices = {
    "EUR/USD": 1.0850,
    "GBP/USD": 1.2950,
    "USD/JPY": 150.20,
    "AUD/USD": 0.6750,
    "USD/CAD": 1.3850,
    "NZD/USD": 0.6150
}

# Initialize session state with proper structure
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
if 'all_signals' not in st.session_state:
    st.session_state.all_signals = {}

# Initialize signal history for all pairs
for pair in trading_pairs:
    if pair not in st.session_state.signal_history:
        st.session_state.signal_history[pair] = {
            'signals': [],
            'buy_indicators': [],
            'sell_indicators': [],
            'time': datetime.now(),
            'buy_count': 0,
            'sell_count': 0,
            'agreement': 'NONE'  # NONE, BUY, SELL, MIXED
        }

# Function to generate simulated historical prices
def generate_simulated_historical(pair, periods=200):
    np.random.seed(hash(pair) % 10000)  # Different seed for each pair
    base_price = initial_prices[pair]
    prices = []
    current_time = datetime.now()
    
    for i in range(periods):
        date = current_time - timedelta(hours=periods - i - 1)
        
        # Generate OHLC data with realistic volatility
        open_price = base_price
        change = np.random.normal(0, 0.0015)
        close_price = base_price * (1 + change)
        high_price = max(open_price, close_price) * (1 + abs(np.random.normal(0, 0.0008)))
        low_price = min(open_price, close_price) * (1 - abs(np.random.normal(0, 0.0008)))
        
        prices.append({
            "date": date,
            "open": open_price,
            "high": high_price,
            "low": low_price,
            "close": close_price
        })
        
        base_price = close_price
    
    return pd.DataFrame(prices)

# Function to calculate technical indicators
def calculate_indicators(df):
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
        
        return df_indicators
        
    except Exception as e:
        return df

# Function to detect trading signals for ALL pairs - UPDATED FOR AGREEMENT
def scan_all_pairs_signals():
    all_signals = {}
    
    for pair in trading_pairs:
        if pair in st.session_state.price_history:
            df = st.session_state.price_history[pair].copy()
            df_with_indicators = calculate_indicators(df)
            
            signals, buy_indicators, sell_indicators, agreement = detect_trading_signals(df_with_indicators)
            
            # Store signals for this pair
            all_signals[pair] = {
                'signals': signals,
                'buy_indicators': buy_indicators,
                'sell_indicators': sell_indicators,
                'time': datetime.now(),
                'buy_count': len(buy_indicators),
                'sell_count': len(sell_indicators),
                'agreement': agreement,
                'current_price': st.session_state.current_prices[pair],
                'price_change': calculate_price_change(pair)
            }
            
            # Also update signal history
            st.session_state.signal_history[pair] = all_signals[pair]
    
    return all_signals

# Function to calculate price change
def calculate_price_change(pair):
    if pair in st.session_state.price_history and len(st.session_state.price_history[pair]) > 1:
        df = st.session_state.price_history[pair]
        current_price = df.iloc[-1]['close']
        previous_price = df.iloc[-2]['close']
        change = ((current_price - previous_price) / previous_price) * 100
        return change
    return 0

# Function to detect trading signals with BOTH INDICATORS AGREEMENT
def detect_trading_signals(df):
    buy_indicators = []
    sell_indicators = []
    
    try:
        if len(df) < max(ma_slow, macd_slow, rsi_period) + 5:
            return [], [], [], 'NONE'
            
        latest = df.iloc[-1]
        previous = df.iloc[-2]
        
        # Check if we have valid indicator values
        has_valid_data = all(pd.notna(latest.get(col, np.nan)) for col in 
                           ['MA_Fast', 'MA_Slow', 'RSI', 'MACD', 'MACD_Signal'])
        
        if not has_valid_data:
            return [], [], [], 'NONE'
        
        # 1. Moving Average Crossover Signal
        if pd.notna(latest['MA_Fast']) and pd.notna(latest['MA_Slow']):
            if latest['MA_Fast'] > latest['MA_Slow'] and previous['MA_Fast'] <= previous['MA_Slow']:
                buy_indicators.append("MA Crossover")
            elif latest['MA_Fast'] < latest['MA_Slow'] and previous['MA_Fast'] >= previous['MA_Slow']:
                sell_indicators.append("MA Crossover")
        
        # 2. RSI Signals
        if pd.notna(latest['RSI']):
            if latest['RSI'] < rsi_oversold:
                buy_indicators.append("RSI Oversold")
            elif latest['RSI'] > rsi_overbought:
                sell_indicators.append("RSI Overbought")
        
        # 3. MACD Signals
        if pd.notna(latest['MACD']) and pd.notna(latest['MACD_Signal']):
            if latest['MACD'] > latest['MACD_Signal'] and previous['MACD'] <= previous['MACD_Signal']:
                buy_indicators.append("MACD Bullish")
            elif latest['MACD'] < latest['MACD_Signal'] and previous['MACD'] >= previous['MACD_Signal']:
                sell_indicators.append("MACD Bearish")
        
        # Determine agreement type
        total_buy = len(buy_indicators)
        total_sell = len(sell_indicators)
        
        # Check for perfect agreement (both indicators say the same thing)
        if total_buy == REQUIRED_INDICATORS and total_sell == 0:
            agreement = 'BUY'
            signals = [("BUY", total_buy, buy_indicators)]
        elif total_sell == REQUIRED_INDICATORS and total_buy == 0:
            agreement = 'SELL'
            signals = [("SELL", total_sell, sell_indicators)]
        elif total_buy > 0 and total_sell > 0:
            agreement = 'MIXED'
            signals = []
        else:
            agreement = 'NONE'
            signals = []
            
        return signals, buy_indicators, sell_indicators, agreement
        
    except Exception as e:
        return [], [], [], 'NONE'

# Function to check if we can open a new trade for a pair
def can_open_trade(pair, direction):
    # Count current open trades for this pair in the same direction
    same_direction_trades = [t for t in st.session_state.open_trades 
                           if t['pair'] == pair and t['direction'] == direction]
    
    # No same-direction duplicates allowed
    if same_direction_trades:
        return False
    
    # Check bank balance
    if st.session_state.bank_balance < stake:
        return False
    
    return True

# Function to execute auto trade based on signals for ALL pairs - UPDATED FOR AGREEMENT
def execute_auto_trades():
    if not st.session_state.auto_trading:
        return []
    
    auto_trades_executed = []
    
    # Scan all pairs for signals
    all_signals = scan_all_pairs_signals()
    
    for pair, signal_info in all_signals.items():
        signals = signal_info.get('signals', [])
        agreement = signal_info.get('agreement', 'NONE')
        
        # Execute trades only when BOTH indicators agree on the same direction
        for signal_type, count, indicators in signals:
            if agreement == 'BUY' and signal_type == "BUY" and can_open_trade(pair, 'BUY'):
                if execute_trade(pair, 'BUY', st.session_state.current_prices[pair]):
                    auto_trades_executed.append(f"AUTO BUY {pair} (BOTH indicators agree: {', '.join(indicators)})")
                    st.session_state.last_auto_trade[pair] = datetime.now()
            
            elif agreement == 'SELL' and signal_type == "SELL" and can_open_trade(pair, 'SELL'):
                if execute_trade(pair, 'SELL', st.session_state.current_prices[pair]):
                    auto_trades_executed.append(f"AUTO SELL {pair} (BOTH indicators agree: {', '.join(indicators)})")
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

# Function to simulate price movement for ALL pairs
def simulate_all_prices_movement():
    for pair in trading_pairs:
        try:
            current_price = st.session_state.current_prices[pair]
            
            # Moderate volatility for better signal generation
            volatility = 0.0008
            
            # Small trend bias based on recent signals
            trend_bias = 0
            if pair in st.session_state.signal_history:
                signal_info = st.session_state.signal_history[pair]
                agreement = signal_info.get('agreement', 'NONE')
                if agreement == 'BUY':
                    trend_bias += 0.0003
                elif agreement == 'SELL':
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
            ]).tail(250)
            
        except Exception as e:
            continue

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

# Initialize price history for all pairs
for pair in trading_pairs:
    if pair not in st.session_state.price_history:
        st.session_state.price_history[pair] = generate_simulated_historical(pair, 200)

# Main application layout
st.markdown('<h1 class="main-header">🤖 2 Indicator Agreement Strategy</h1>', unsafe_allow_html=True)

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
    st.markdown("## ⚙️ Trading Parameters")
    st.write(f"**Stake per trade:** €{stake}")
    st.write(f"**Profit Target:** +{profit_target} pips")
    st.write(f"**Stop Loss:** -{stop_loss} pips")
    st.write(f"**Required Agreement:** {REQUIRED_INDICATORS} indicators")
    st.write(f"**Bank Balance:** €{st.session_state.bank_balance:.2f}")
    
    st.markdown("---")
    st.markdown("## 📊 Monitoring Pairs")
    for pair in trading_pairs:
        st.write(f"• {pair}")
    
    st.markdown("---")
    st.markdown("## 🎯 Trading Rules")
    st.write(f"• Enter only when **BOTH indicators agree**")
    st.write("• **No same-direction duplicates** per pair")
    st.write("• **No maximum trades** per pair")
    st.write("• 1:1 Risk/Reward ratio")
    st.write("• **BUY:** Both indicators say BUY")
    st.write("• **SELL:** Both indicators say SELL")
    st.write("• **NO TRADE:** Mixed signals")

# Update prices and execute auto trades
simulate_all_prices_movement()
st.session_state.all_signals = scan_all_pairs_signals()
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

# ALL PAIRS SIGNAL DASHBOARD - UPDATED FOR AGREEMENT
st.markdown("---")
st.markdown("## 📊 All Pairs Agreement Dashboard")

# Create a grid layout for all pairs
cols = st.columns(3)

for idx, pair in enumerate(trading_pairs):
    with cols[idx % 3]:
        signal_info = st.session_state.all_signals.get(pair, {})
        buy_count = signal_info.get('buy_count', 0)
        sell_count = signal_info.get('sell_count', 0)
        agreement = signal_info.get('agreement', 'NONE')
        current_price = signal_info.get('current_price', st.session_state.current_prices.get(pair, 0))
        price_change = signal_info.get('price_change', 0)
        
        # Count current trades for this pair
        current_trades = [t for t in st.session_state.open_trades if t['pair'] == pair]
        buy_trades = len([t for t in current_trades if t['direction'] == 'BUY'])
        sell_trades = len([t for t in current_trades if t['direction'] == 'SELL'])
        
        # Price display
        pip_size = pip_sizes[pair]
        if pip_size == 0.01:
            price_display = f"{current_price:.2f}"
        else:
            price_display = f"{current_price:.4f}"
        
        # Determine signal type and styling based on AGREEMENT
        if agreement == 'BUY':
            signal_class = "signal-strong-buy"
            signal_text = "BOTH SAY BUY"
            signal_emoji = "🟢"
            border_class = "agreement-buy"
        elif agreement == 'SELL':
            signal_class = "signal-strong-sell"
            signal_text = "BOTH SAY SELL"
            signal_emoji = "🔴"
            border_class = "agreement-sell"
        elif agreement == 'MIXED':
            signal_class = "signal-mixed"
            signal_text = "MIXED SIGNALS"
            signal_emoji = "🟡"
            border_class = "no-agreement"
        else:
            signal_class = "signal-no-trade"
            signal_text = "NO AGREEMENT"
            signal_emoji = "⚪"
            border_class = "no-agreement"
        
        # Price change styling
        change_color = "#00ff88" if price_change >= 0 else "#ff4444"
        change_emoji = "📈" if price_change >= 0 else "📉"
        
        st.markdown(f"""
        <div class="pair-card {border_class}">
            <h3>{pair} {signal_emoji}</h3>
            <div class="{signal_class}">
                {signal_text}<br>
                Buy: {buy_count}/2 • Sell: {sell_count}/2
            </div>
            <div style="margin-top: 0.5rem;">
                <strong>Price: {price_display}</strong><br>
                <span style="color: {change_color};">
                    {change_emoji} {price_change:+.2f}%
                </span><br>
                <small>Open Trades: {buy_trades} BUY, {sell_trades} SELL</small>
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        # Show active indicators
        with st.expander(f"View {pair} Details"):
            buy_indicators = signal_info.get('buy_indicators', [])
            sell_indicators = signal_info.get('sell_indicators', [])
            
            st.write("**Buy Indicators:**")
            for indicator in buy_indicators:
                st.markdown(f"<span class='indicator-active'>✅ {indicator}</span>", unsafe_allow_html=True)
            if not buy_indicators:
                st.write("None")
            
            st.write("**Sell Indicators:**")
            for indicator in sell_indicators:
                st.markdown(f"<span class='indicator-active'>❌ {indicator}</span>", unsafe_allow_html=True)
            if not sell_indicators:
                st.write("None")
            
            # Agreement status
            if agreement == 'BUY':
                if buy_trades == 0:
                    st.success(f"🎯 **PERFECT BUY AGREEMENT** - Both indicators say BUY!")
                else:
                    st.info(f"📊 Already have BUY position - Both indicators still agree on BUY")
            elif agreement == 'SELL':
                if sell_trades == 0:
                    st.error(f"🎯 **PERFECT SELL AGREEMENT** - Both indicators say SELL!")
                else:
                    st.info(f"📊 Already have SELL position - Both indicators still agree on SELL")
            elif agreement == 'MIXED':
                st.warning(f"⚠️ **MIXED SIGNALS** - Indicators disagree (BUY: {buy_count}, SELL: {sell_count})")
            else:
                st.info("⏸️ **NO AGREEMENT** - Waiting for both indicators to agree")

# Agreement Summary
st.markdown("---")
st.markdown("## 📈 Agreement Summary")

perfect_buy_pairs = [p for p in trading_pairs 
                    if st.session_state.all_signals.get(p, {}).get('agreement') == 'BUY']
perfect_sell_pairs = [p for p in trading_pairs 
                     if st.session_state.all_signals.get(p, {}).get('agreement') == 'SELL']
mixed_pairs = [p for p in trading_pairs 
              if st.session_state.all_signals.get(p, {}).get('agreement') == 'MIXED']

# Filter pairs that can actually be traded (no same-direction duplicates)
tradable_buy_pairs = [p for p in perfect_buy_pairs 
                     if len([t for t in st.session_state.open_trades if t['pair'] == p and t['direction'] == 'BUY']) == 0]
tradable_sell_pairs = [p for p in perfect_sell_pairs 
                      if len([t for t in st.session_state.open_trades if t['pair'] == p and t['direction'] == 'SELL']) == 0]

col1, col2, col3 = st.columns(3)

with col1:
    st.metric("Perfect BUY Agreement", f"{len(tradable_buy_pairs)}/{len(perfect_buy_pairs)} tradable")
    if tradable_buy_pairs:
        st.write("**Ready to BUY:**")
        for pair in tradable_buy_pairs:
            st.success(f"✅ {pair} - BOTH indicators say BUY")
    elif perfect_buy_pairs:
        st.write("**Already trading BUY:**")
        for pair in perfect_buy_pairs:
            st.info(f"📊 {pair} - Already have BUY position")

with col2:
    st.metric("Perfect SELL Agreement", f"{len(tradable_sell_pairs)}/{len(perfect_sell_pairs)} tradable")
    if tradable_sell_pairs:
        st.write("**Ready to SELL:**")
        for pair in tradable_sell_pairs:
            st.error(f"❌ {pair} - BOTH indicators say SELL")
    elif perfect_sell_pairs:
        st.write("**Already trading SELL:**")
        for pair in perfect_sell_pairs:
            st.info(f"📊 {pair} - Already have SELL position")

with col3:
    st.metric("Mixed Signals", len(mixed_pairs))
    if mixed_pairs:
        st.write("**Indicators disagree:**")
        for pair in mixed_pairs:
            signal_info = st.session_state.all_signals.get(pair, {})
            st.warning(f"⚠️ {pair} - BUY: {signal_info.get('buy_count', 0)}, SELL: {signal_info.get('sell_count', 0)}")

# Detailed Analysis for Selected Pair
st.markdown("---")
st.markdown("## 🔍 Detailed Pair Analysis")

selected_pair = st.selectbox("Select Pair for Detailed Chart Analysis", trading_pairs)

col1, col2 = st.columns([2, 1])

with col1:
    st.markdown(f"## 📈 {selected_pair} - Technical Analysis")
    
    if selected_pair in st.session_state.price_history:
        df = st.session_state.price_history[selected_pair].copy()
        df_with_indicators = calculate_indicators(df)
        
        # Create chart
        fig = make_subplots(rows=3, cols=1, 
                           shared_xaxes=True,
                           vertical_spacing=0.05,
                           subplot_titles=('Price with Moving Averages', 'RSI', 'MACD'),
                           row_heights=[0.5, 0.25, 0.25])
        
        # Price with MAs
        fig.add_trace(go.Scatter(x=df_with_indicators['date'], y=df_with_indicators['close'], 
                               name='Price', line=dict(color='#00ff88')), row=1, col=1)
        fig.add_trace(go.Scatter(x=df_with_indicators['date'], y=df_with_indicators['MA_Fast'], 
                               name=f'MA{ma_fast}', line=dict(color='#ff4444')), row=1, col=1)
        fig.add_trace(go.Scatter(x=df_with_indicators['date'], y=df_with_indicators['MA_Slow'], 
                               name=f'MA{ma_slow}', line=dict(color='#4444ff')), row=1, col=1)
        
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
        
        fig.update_layout(height=600, showlegend=True, template="plotly_dark")
        st.plotly_chart(fig, use_container_width=True)

with col2:
    st.markdown("## 📊 Current Agreement")
    
    # Safe access to signal info
    signal_info = st.session_state.all_signals.get(selected_pair, {})
    buy_count = signal_info.get('buy_count', 0)
    sell_count = signal_info.get('sell_count', 0)
    agreement = signal_info.get('agreement', 'NONE')
    buy_indicators = signal_info.get('buy_indicators', [])
    sell_indicators = signal_info.get('sell_indicators', [])
    
    # Count current trades
    current_trades = [t for t in st.session_state.open_trades if t['pair'] == selected_pair]
    buy_trades = len([t for t in current_trades if t['direction'] == 'BUY'])
    sell_trades = len([t for t in current_trades if t['direction'] == 'SELL'])
    
    st.metric("Agreement Status", agreement)
    st.metric("Buy Indicators", f"{buy_count}/{REQUIRED_INDICATORS}")
    st.metric("Sell Indicators", f"{sell_count}/{REQUIRED_INDICATORS}")
    st.metric("Open BUY Trades", buy_trades)
    st.metric("Open SELL Trades", sell_trades)
    
    st.markdown("### Buy Signals:")
    for indicator in buy_indicators:
        st.markdown(f"<span class='indicator-active'>✅ {indicator}</span>", unsafe_allow_html=True)
    if not buy_indicators:
        st.write("None")
    
    st.markdown("### Sell Signals:")
    for indicator in sell_indicators:
        st.markdown(f"<span class='indicator-active'>❌ {indicator}</span>", unsafe_allow_html=True)
    if not sell_indicators:
        st.write("None")
    
    # Trading status based on AGREEMENT
    if agreement == 'BUY':
        if buy_trades == 0:
            st.success(f"🎯 **PERFECT BUY AGREEMENT**\n\nBoth indicators confirm BUY signal!")
        else:
            st.info(f"📊 **Already Trading BUY**\n\nBoth indicators still agree on BUY")
    elif agreement == 'SELL':
        if sell_trades == 0:
            st.error(f"🎯 **PERFECT SELL AGREEMENT**\n\nBoth indicators confirm SELL signal!")
        else:
            st.info(f"📊 **Already Trading SELL**\n\nBoth indicators still agree on SELL")
    elif agreement == 'MIXED':
        st.warning(f"⚠️ **MIXED SIGNALS**\n\nIndicators disagree: {buy_count} BUY vs {sell_count} SELL")
    else:
        st.info("⏸️ **NO AGREEMENT**\n\nWaiting for both indicators to agree")

# Trades section
st.markdown("---")
col1, col2 = st.columns(2)

with col1:
    st.markdown("## 🔓 Open Trades")
    if st.session_state.open_trades:
        # Group trades by pair
        trades_by_pair = {}
        for trade in st.session_state.open_trades:
            if trade['pair'] not in trades_by_pair:
                trades_by_pair[trade['pair']] = []
            trades_by_pair[trade['pair']].append(trade)
        
        for pair, trades in trades_by_pair.items():
            st.markdown(f"**{pair}**")
            for trade in trades:
                trade_class = "trade-buy" if trade['direction'] == 'BUY' else "trade-sell"
                current_pl = trade['profit_loss']
                pl_class = "profit-positive" if current_pl >= 0 else "profit-negative"
                trade_type = trade.get('type', 'MANUAL')
                
                # Calculate pips
                pip_size = pip_sizes[trade['pair']]
                current_price = trade.get('current_price', trade['entry_price'])
                if trade['direction'] == 'BUY':
                    pips = (current_price - trade['entry_price']) / pip_size
                else:
                    pips = (trade['entry_price'] - current_price) / pip_size
                
                st.markdown(f"""
                <div class="{trade_class}">
                    <strong>{trade['direction']} ({trade_type})</strong><br>
                    Entry: {trade['entry_price']:.4f}<br>
                    P&L: <span class="{pl_class}">€{current_pl:.2f}</span><br>
                    Pips: <span class="{pl_class}">{pips:+.1f}</span><br>
                    <small>{trade['time'].strftime('%H:%M:%S')}</small>
                </div>
                """, unsafe_allow_html=True)
    else:
        st.info("No open trades")

with col2:
    st.markdown("## 📋 Trade History")
    recent_trades = st.session_state.trade_history[-10:] if st.session_state.trade_history else []
    
    if recent_trades:
        for trade in reversed(recent_trades):
            trade_class = "trade-buy" if trade['direction'] == 'BUY' else "trade-sell"
            result_class = "profit-positive" if trade['profit_loss'] >= 0 else "profit-negative"
            trade_type = trade.get('type', 'MANUAL')
            
            st.markdown(f"""
            <div class="{trade_class}">
                <strong>{trade['pair']} {trade['direction']} ({trade_type})</strong><br>
                Result: <span class="{result_class}">€{trade['profit_loss']:.2f}</span><br>
                <small>{trade.get('close_time', trade['time']).strftime('%H:%M:%S')}</small>
            </div>
            """, unsafe_allow_html=True)
    else:
        st.info("No trade history yet")

# Auto-refresh
st.markdown("---")
st.markdown("🔄 Auto-refreshing every 3 seconds...")

time.sleep(3)
st.rerun()

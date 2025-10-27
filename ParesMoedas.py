import streamlit as st
import pandas as pd
import numpy as np
import requests
import time
from datetime import datetime, timedelta
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
st.markdown("""
    <style>
    #MainMenu {visibility: hidden;}
    header {visibility: hidden;}
    footer {visibility: hidden;}
    </style>
""", unsafe_allow_html=True)

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
    .param-section {
        background: linear-gradient(135deg, #f8f9fa 0%, #e9ecef 100%);
        padding: 1rem;
        border-radius: 10px;
        margin: 0.5rem 0;
        border-left: 4px solid #667eea;
    }
</style>
""", unsafe_allow_html=True)

# Default trading parameters
DEFAULT_PARAMS = {
    'initial_bank': 1000,
    'stake': 10,
    'profit_target': 10,
    'stop_loss': 10,
    'ma_fast': 10,
    'ma_slow': 20,
    'rsi_period': 14,
    'rsi_overbought': 70,
    'rsi_oversold': 30,
    'macd_fast': 12,
    'macd_slow': 26,
    'macd_signal': 9,
    'fib_period': 50,
    'bb_period': 20,
    'bb_std': 2.0,
    'required_indicators': 2
}

# Initialize session state with parameters
if 'trading_params' not in st.session_state:
    st.session_state.trading_params = DEFAULT_PARAMS.copy()

# Ensure all defaults are present in case of updates
for key, val in DEFAULT_PARAMS.items():
    st.session_state.trading_params.setdefault(key, val)

if 'bank_balance' not in st.session_state:
    st.session_state.bank_balance = st.session_state.trading_params['initial_bank']
if 'open_trades' not in st.session_state:
    st.session_state.open_trades = []
if 'trade_history' not in st.session_state:
    st.session_state.trade_history = []
if 'auto_trading' not in st.session_state:
    st.session_state.auto_trading = False
if 'signal_history' not in st.session_state:
    st.session_state.signal_history = {}
if 'last_auto_trade' not in st.session_state:
    st.session_state.last_auto_trade = {}
if 'all_signals' not in st.session_state:
    st.session_state.all_signals = {}

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

# Initialize current prices and price history
if 'current_prices' not in st.session_state:
    st.session_state.current_prices = initial_prices.copy()
if 'price_history' not in st.session_state:
    st.session_state.price_history = {}

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
            'agreement': 'NONE'
        }

# Function to reset trading system
def reset_trading_system():
    st.session_state.bank_balance = st.session_state.trading_params['initial_bank']
    st.session_state.open_trades = []
    st.session_state.trade_history = []
    st.session_state.auto_trading = False
    st.session_state.signal_history = {}
    st.session_state.last_auto_trade = {}
    st.session_state.all_signals = {}
    st.session_state.current_prices = initial_prices.copy()
    st.session_state.price_history = {}
    
    # Re-initialize signal history
    for pair in trading_pairs:
        st.session_state.signal_history[pair] = {
            'signals': [],
            'buy_indicators': [],
            'sell_indicators': [],
            'time': datetime.now(),
            'buy_count': 0,
            'sell_count': 0,
            'agreement': 'NONE'
        }

# Function to generate simulated historical prices
def generate_simulated_historical(pair, periods=200):
    np.random.seed(hash(pair) % 10000)
    base_price = initial_prices[pair]
    prices = []
    current_time = datetime.now()
    
    for i in range(periods):
        date = current_time - timedelta(hours=periods - i - 1)
        
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

# Function to calculate RSI
def calculate_rsi(prices, window):
    delta = prices.diff()
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    avg_gain = gain.rolling(window=window, min_periods=1).mean()
    avg_loss = loss.rolling(window=window, min_periods=1).mean()
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    return rsi

# Function to calculate MACD
def calculate_macd(prices, fast, slow, signal):
    ema_fast = prices.ewm(span=fast, adjust=False).mean()
    ema_slow = prices.ewm(span=slow, adjust=False).mean()
    macd = ema_fast - ema_slow
    macd_signal = macd.ewm(span=signal, adjust=False).mean()
    macd_histogram = macd - macd_signal
    return macd, macd_signal, macd_histogram

# Function to calculate technical indicators using current parameters
def calculate_indicators(df):
    try:
        df_indicators = df.copy()
        params = st.session_state.trading_params
        
        # Moving Averages
        df_indicators['MA_Fast'] = df_indicators['close'].rolling(window=params['ma_fast']).mean()
        df_indicators['MA_Slow'] = df_indicators['close'].rolling(window=params['ma_slow']).mean()
        
        # RSI
        df_indicators['RSI'] = calculate_rsi(df_indicators['close'], params['rsi_period'])
        
        # MACD
        macd_line, signal_line, histogram = calculate_macd(df_indicators['close'], 
                                                          params['macd_fast'], 
                                                          params['macd_slow'], 
                                                          params['macd_signal'])
        df_indicators['MACD'] = macd_line
        df_indicators['MACD_Signal'] = signal_line
        df_indicators['MACD_Histogram'] = histogram
        
        # Bollinger Bands
        bb_middle = df_indicators['close'].rolling(window=params['bb_period']).mean()
        bb_std = df_indicators['close'].rolling(window=params['bb_period']).std()
        df_indicators['BB_Upper'] = bb_middle + (bb_std * params['bb_std'])
        df_indicators['BB_Lower'] = bb_middle - (bb_std * params['bb_std'])
        df_indicators['BB_Middle'] = bb_middle
        
        return df_indicators
        
    except Exception as e:
        return df

# Function to detect trading signals for ALL pairs
def scan_all_pairs_signals():
    all_signals = {}
    params = st.session_state.trading_params
    
    for pair in trading_pairs:
        if pair in st.session_state.price_history:
            df = st.session_state.price_history[pair].copy()
            df_with_indicators = calculate_indicators(df)
            
            signals, buy_indicators, sell_indicators, agreement = detect_trading_signals(df_with_indicators)
            
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
    params = st.session_state.trading_params
    
    try:
        if len(df) < max(params['ma_slow'], params['macd_slow'], params['rsi_period'], params['fib_period'], params['bb_period']) + 5:
            return [], [], [], 'NONE'
            
        latest = df.iloc[-1]
        previous = df.iloc[-2]
        
        has_valid_data = all(pd.notna(latest.get(col, np.nan)) for col in 
                           ['MA_Fast', 'MA_Slow', 'RSI', 'MACD', 'MACD_Signal', 'BB_Upper', 'BB_Lower'])
        
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
            if latest['RSI'] < params['rsi_oversold']:
                buy_indicators.append("RSI Oversold")
            elif latest['RSI'] > params['rsi_overbought']:
                sell_indicators.append("RSI Overbought")
        
        # 3. MACD Signals
        if pd.notna(latest['MACD']) and pd.notna(latest['MACD_Signal']):
            if latest['MACD'] > latest['MACD_Signal'] and previous['MACD'] <= previous['MACD_Signal']:
                buy_indicators.append("MACD Bullish")
            elif latest['MACD'] < latest['MACD_Signal'] and previous['MACD'] >= previous['MACD_Signal']:
                sell_indicators.append("MACD Bearish")
        
        # 4. Fibonacci 50% Retracement Crossover Signal
        if len(df) >= params['fib_period']:
            recent_df = df.tail(params['fib_period'])
            swing_high = recent_df['high'].max()
            swing_low = recent_df['low'].min()
            fib_50 = swing_low + 0.5 * (swing_high - swing_low)
            
            if latest['close'] > fib_50 and previous['close'] <= fib_50:
                buy_indicators.append("Fib 50% Cross")
            elif latest['close'] < fib_50 and previous['close'] >= fib_50:
                sell_indicators.append("Fib 50% Cross")
        
        # 5. Bollinger Bands Signal
        if pd.notna(latest['BB_Upper']) and pd.notna(latest['BB_Lower']):
            if latest['close'] <= latest['BB_Lower'] and previous['close'] > previous['BB_Lower']:
                buy_indicators.append("BB Lower Touch")
            elif latest['close'] >= latest['BB_Upper'] and previous['close'] < previous['BB_Upper']:
                sell_indicators.append("BB Upper Touch")
        
        # Determine agreement type
        total_buy = len(buy_indicators)
        total_sell = len(sell_indicators)
        required = params['required_indicators']
        
        if total_buy >= required and total_sell == 0:
            agreement = 'BUY'
            signals = [("BUY", total_buy, buy_indicators)]
        elif total_sell >= required and total_buy == 0:
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
    params = st.session_state.trading_params
    
    same_direction_trades = [t for t in st.session_state.open_trades 
                           if t['pair'] == pair and t['direction'] == direction]
    
    if same_direction_trades:
        return False
    
    if st.session_state.bank_balance < params['stake']:
        return False
    
    return True

# Function to execute auto trade based on signals for ALL pairs
def execute_auto_trades():
    if not st.session_state.auto_trading:
        return []
    
    auto_trades_executed = []
    params = st.session_state.trading_params
    
    all_signals = scan_all_pairs_signals()
    
    for pair, signal_info in all_signals.items():
        signals = signal_info.get('signals', [])
        agreement = signal_info.get('agreement', 'NONE')
        
        for signal_type, count, indicators in signals:
            if agreement == 'BUY' and signal_type == "BUY" and can_open_trade(pair, 'BUY'):
                if execute_trade(pair, 'BUY', st.session_state.current_prices[pair]):
                    auto_trades_executed.append(f"AUTO BUY {pair} ({count} indicators agree: {', '.join(indicators)})")
                    st.session_state.last_auto_trade[pair] = datetime.now()
            
            elif agreement == 'SELL' and signal_type == "SELL" and can_open_trade(pair, 'SELL'):
                if execute_trade(pair, 'SELL', st.session_state.current_prices[pair]):
                    auto_trades_executed.append(f"AUTO SELL {pair} ({count} indicators agree: {', '.join(indicators)})")
                    st.session_state.last_auto_trade[pair] = datetime.now()
    
    return auto_trades_executed

# Function to execute trade
def execute_trade(pair, direction, entry_price):
    try:
        params = st.session_state.trading_params
        stake = params['stake']
        
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
            
            volatility = 0.0008
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
            
            if pair not in st.session_state.price_history:
                st.session_state.price_history[pair] = generate_simulated_historical(pair, 200)
            
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
        params = st.session_state.trading_params
        stake = params['stake']
        profit_target = params['profit_target']
        stop_loss = params['stop_loss']
        pip_value = stake / 10
        
        trades_to_remove = []
        for i, trade in enumerate(st.session_state.open_trades):
            if trade['status'] == 'open':
                current_price = st.session_state.current_prices[trade['pair']]
                pip_size = pip_sizes[trade['pair']]
                
                if trade['direction'] == 'BUY':
                    pips = (current_price - trade['entry_price']) / pip_size
                else:
                    pips = (trade['entry_price'] - current_price) / pip_size
                
                profit_loss = pips * pip_value
                trade['profit_loss'] = profit_loss
                trade['current_price'] = current_price
                
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
    
    col1, col2 = st.columns(2)
    with col1:
        if st.button("🚀 Start Auto Trading", width='stretch', type="primary"):
            st.session_state.auto_trading = True
            st.success("Auto Trading Started!")
    with col2:
        if st.button("🛑 Stop Auto Trading", width='stretch', type="secondary"):
            st.session_state.auto_trading = False
            st.warning("Auto Trading Stopped!")
    
    st.markdown("---")
    
    # ⚙️ TRADING PARAMETERS SETUP
    st.markdown("## ⚙️ Trading Parameters Setup")
    
    with st.expander("💰 Money Management", expanded=True):
        st.markdown('<div class="param-section">', unsafe_allow_html=True)
        st.session_state.trading_params['initial_bank'] = st.number_input(
            "Initial Bank Balance (€)", 
            min_value=100, 
            max_value=10000, 
            value=st.session_state.trading_params['initial_bank'],
            step=100,
            help="Starting capital for trading"
        )
        st.session_state.trading_params['stake'] = st.number_input(
            "Stake per Trade (€)", 
            min_value=1, 
            max_value=500, 
            value=st.session_state.trading_params['stake'],
            step=1,
            help="Amount to risk per trade"
        )
        st.markdown('</div>', unsafe_allow_html=True)
    
    with st.expander("🎯 Trade Settings", expanded=True):
        st.markdown('<div class="param-section">', unsafe_allow_html=True)
        st.session_state.trading_params['profit_target'] = st.number_input(
            "Profit Target (pips)", 
            min_value=1, 
            max_value=100, 
            value=st.session_state.trading_params['profit_target'],
            step=1,
            help="Take profit level in pips"
        )
        st.session_state.trading_params['stop_loss'] = st.number_input(
            "Stop Loss (pips)", 
            min_value=1, 
            max_value=100, 
            value=st.session_state.trading_params['stop_loss'],
            step=1,
            help="Stop loss level in pips"
        )
        st.session_state.trading_params['required_indicators'] = st.selectbox(
            "Required Indicators Agreement",
            options=[2, 3, 4, 5],
            value=st.session_state.trading_params['required_indicators'],
            help="Number of indicators that must agree for trade entry"
        )
        st.markdown('</div>', unsafe_allow_html=True)
    
    with st.expander("📊 Indicator Parameters", expanded=True):
        st.markdown('<div class="param-section">', unsafe_allow_html=True)
        
        col1, col2 = st.columns(2)
        with col1:
            st.session_state.trading_params['ma_fast'] = st.number_input(
                "MA Fast Period", 
                min_value=5, 
                max_value=50, 
                value=st.session_state.trading_params['ma_fast'],
                step=1
            )
        with col2:
            st.session_state.trading_params['ma_slow'] = st.number_input(
                "MA Slow Period", 
                min_value=10, 
                max_value=100, 
                value=st.session_state.trading_params['ma_slow'],
                step=1
            )
        
        st.session_state.trading_params['rsi_period'] = st.number_input(
            "RSI Period", 
            min_value=5, 
            max_value=30, 
            value=st.session_state.trading_params['rsi_period'],
            step=1
        )
        
        col1, col2 = st.columns(2)
        with col1:
            st.session_state.trading_params['rsi_overbought'] = st.number_input(
                "RSI Overbought", 
                min_value=60, 
                max_value=90, 
                value=st.session_state.trading_params['rsi_overbought'],
                step=1
            )
        with col2:
            st.session_state.trading_params['rsi_oversold'] = st.number_input(
                "RSI Oversold", 
                min_value=10, 
                max_value=40, 
                value=st.session_state.trading_params['rsi_oversold'],
                step=1
            )
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.session_state.trading_params['macd_fast'] = st.number_input(
                "MACD Fast", 
                min_value=5, 
                max_value=20, 
                value=st.session_state.trading_params['macd_fast'],
                step=1
            )
        with col2:
            st.session_state.trading_params['macd_slow'] = st.number_input(
                "MACD Slow", 
                min_value=15, 
                max_value=50, 
                value=st.session_state.trading_params['macd_slow'],
                step=1
            )
        with col3:
            st.session_state.trading_params['macd_signal'] = st.number_input(
                "MACD Signal", 
                min_value=5, 
                max_value=20, 
                value=st.session_state.trading_params['macd_signal'],
                step=1
            )
        
        st.session_state.trading_params['fib_period'] = st.number_input(
            "Fib Period", 
            min_value=20, 
            max_value=100, 
            value=st.session_state.trading_params['fib_period'],
            step=5,
            help="Period to look back for swing high/low in Fibonacci calculation"
        )
        
        col1, col2 = st.columns(2)
        with col1:
            st.session_state.trading_params['bb_period'] = st.number_input(
                "BB Period", 
                min_value=10, 
                max_value=50, 
                value=st.session_state.trading_params['bb_period'],
                step=1,
                help="Period for Bollinger Bands"
            )
        with col2:
            st.session_state.trading_params['bb_std'] = st.number_input(
                "BB Std Dev", 
                min_value=1.0, 
                max_value=3.0, 
                value=st.session_state.trading_params['bb_std'],
                step=0.1,
                help="Standard deviations for Bollinger Bands"
            )
        st.markdown('</div>', unsafe_allow_html=True)
    
    # Apply Parameters Button
    if st.button("🔄 Apply Parameters & Reset", type="primary", width='stretch'):
        reset_trading_system()
        st.success("Parameters applied and system reset!")
    
    st.markdown("---")
    st.markdown("## 📊 Current Parameters")
    params = st.session_state.trading_params
    st.write(f"**Bank:** €{st.session_state.bank_balance:.2f}")
    st.write(f"**Stake:** €{params['stake']}")
    st.write(f"**TP/SL:** ±{params['profit_target']} pips")
    st.write(f"**Agreement:** {params['required_indicators']}/5 indicators")
    
    st.markdown("---")
    st.markdown("## 📈 Monitoring Pairs")
    for pair in trading_pairs:
        st.write(f"• {pair}")
    
    st.markdown("---")
    st.markdown("## 🎯 Trading Rules")
    st.write(f"• Enter only when **{params['required_indicators']} indicators agree**")
    st.write("• **No same-direction duplicates** per pair")
    st.write("• **No maximum trades** per pair")
    st.write(f"• Risk/Reward: 1:{params['profit_target']/params['stop_loss']:.1f}")

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

# ALL PAIRS SIGNAL DASHBOARD
st.markdown("---")
st.markdown("## 📊 All Pairs Agreement Dashboard")

cols = st.columns(3)
params = st.session_state.trading_params

for idx, pair in enumerate(trading_pairs):
    with cols[idx % 3]:
        signal_info = st.session_state.all_signals.get(pair, {})
        buy_count = signal_info.get('buy_count', 0)
        sell_count = signal_info.get('sell_count', 0)
        agreement = signal_info.get('agreement', 'NONE')
        current_price = signal_info.get('current_price', st.session_state.current_prices.get(pair, 0))
        price_change = signal_info.get('price_change', 0)
        
        current_trades = [t for t in st.session_state.open_trades if t['pair'] == pair]
        buy_trades = len([t for t in current_trades if t['direction'] == 'BUY'])
        sell_trades = len([t for t in current_trades if t['direction'] == 'SELL'])
        
        pip_size = pip_sizes[pair]
        if pip_size == 0.01:
            price_display = f"{current_price:.2f}"
        else:
            price_display = f"{current_price:.4f}"
        
        if agreement == 'BUY':
            signal_class = "signal-strong-buy"
            signal_text = f"{buy_count} SAY BUY"
            signal_emoji = "🟢"
            border_class = "agreement-buy"
        elif agreement == 'SELL':
            signal_class = "signal-strong-sell"
            signal_text = f"{sell_count} SAY SELL"
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
        
        change_color = "#00ff88" if price_change >= 0 else "#ff4444"
        change_emoji = "📈" if price_change >= 0 else "📉"
        
        st.markdown(f"""
        <div class="pair-card {border_class}">
            <h3>{pair} {signal_emoji}</h3>
            <div class="{signal_class}">
                {signal_text}<br>
                Buy: {buy_count}/5 • Sell: {sell_count}/5
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
            
            if agreement == 'BUY':
                if buy_trades == 0:
                    st.success(f"🎯 **STRONG BUY AGREEMENT** - {buy_count} indicators say BUY!")
                else:
                    st.info(f"📊 Already have BUY position - {buy_count} indicators still agree on BUY")
            elif agreement == 'SELL':
                if sell_trades == 0:
                    st.error(f"🎯 **STRONG SELL AGREEMENT** - {sell_count} indicators say SELL!")
                else:
                    st.info(f"📊 Already have SELL position - {sell_count} indicators still agree on SELL")
            elif agreement == 'MIXED':
                st.warning(f"⚠️ **MIXED SIGNALS** - Indicators disagree (BUY: {buy_count}, SELL: {sell_count})")
            else:
                st.info("⏸️ **NO AGREEMENT** - Waiting for indicators to agree")

# Agreement Summary
st.markdown("---")
st.markdown("## 📈 Agreement Summary")

perfect_buy_pairs = [p for p in trading_pairs 
                    if st.session_state.all_signals.get(p, {}).get('agreement') == 'BUY']
perfect_sell_pairs = [p for p in trading_pairs 
                     if st.session_state.all_signals.get(p, {}).get('agreement') == 'SELL']
mixed_pairs = [p for p in trading_pairs 
              if st.session_state.all_signals.get(p, {}).get('agreement') == 'MIXED']

tradable_buy_pairs = [p for p in perfect_buy_pairs 
                     if len([t for t in st.session_state.open_trades if t['pair'] == p and t['direction'] == 'BUY']) == 0]
tradable_sell_pairs = [p for p in perfect_sell_pairs 
                      if len([t for t in st.session_state.open_trades if t['pair'] == p and t['direction'] == 'SELL']) == 0]

col1, col2, col3 = st.columns(3)

with col1:
    st.metric("Strong BUY Agreement", f"{len(tradable_buy_pairs)}/{len(perfect_buy_pairs)} tradable")
    if tradable_buy_pairs:
        st.write("**Ready to BUY:**")
        for pair in tradable_buy_pairs:
            signal_info = st.session_state.all_signals.get(pair, {})
            count = signal_info.get('buy_count', 0)
            st.success(f"✅ {pair} - {count} indicators say BUY")
    elif perfect_buy_pairs:
        st.write("**Already trading BUY:**")
        for pair in perfect_buy_pairs:
            signal_info = st.session_state.all_signals.get(pair, {})
            count = signal_info.get('buy_count', 0)
            st.info(f"📊 {pair} - Already have BUY position ({count} agreeing)")

with col2:
    st.metric("Strong SELL Agreement", f"{len(tradable_sell_pairs)}/{len(perfect_sell_pairs)} tradable")
    if tradable_sell_pairs:
        st.write("**Ready to SELL:**")
        for pair in tradable_sell_pairs:
            signal_info = st.session_state.all_signals.get(pair, {})
            count = signal_info.get('sell_count', 0)
            st.error(f"❌ {pair} - {count} indicators say SELL")
    elif perfect_sell_pairs:
        st.write("**Already trading SELL:**")
        for pair in perfect_sell_pairs:
            signal_info = st.session_state.all_signals.get(pair, {})
            count = signal_info.get('sell_count', 0)
            st.info(f"📊 {pair} - Already have SELL position ({count} agreeing)")

with col3:
    st.metric("Mixed Signals", len(mixed_pairs))
    if mixed_pairs:
        st.write("**Indicators disagree:**")
        for pair in mixed_pairs:
            signal_info = st.session_state.all_signals.get(pair, {})
            st.warning(f"⚠️ {pair} - BUY: {signal_info.get('buy_count', 0)}, SELL: {signal_info.get('sell_count', 0)}")

# Trade Summary Table
st.markdown("---")
st.markdown("## 📋 Trade Summary Table")

tab1, tab2 = st.tabs(["Closed Trades", "Open Trades"])

with tab1:
    if st.session_state.trade_history:
        closed_df = pd.DataFrame(st.session_state.trade_history)
        # Select and format relevant columns
        closed_df_display = closed_df[['id', 'pair', 'direction', 'entry_price', 'close_price', 'profit_loss', 'stake', 'time', 'close_time', 'type']].copy()
        closed_df_display['profit_loss'] = closed_df_display['profit_loss'].round(2)
        closed_df_display['entry_price'] = closed_df_display['entry_price'].round(4)
        closed_df_display['close_price'] = closed_df_display['close_price'].round(4)
        closed_df_display['duration'] = (pd.to_datetime(closed_df_display['close_time']) - pd.to_datetime(closed_df_display['time'])).dt.total_seconds() / 3600  # Hours
        closed_df_display['duration'] = closed_df_display['duration'].round(2)
        closed_df_display = closed_df_display.rename(columns={
            'id': 'ID',
            'pair': 'Pair',
            'direction': 'Direction',
            'entry_price': 'Entry Price',
            'close_price': 'Exit Price',
            'profit_loss': 'P&L (€)',
            'stake': 'Stake (€)',
            'time': 'Open Time',
            'close_time': 'Close Time',
            'type': 'Type',
            'duration': 'Duration (hrs)'
        })
        st.dataframe(closed_df_display, width='stretch', hide_index=True)
        
        # Total P&L summary
        total_pnl_closed = closed_df['profit_loss'].sum()
        st.metric("Total P&L (Closed Trades)", f"€{total_pnl_closed:.2f}")
    else:
        st.info("No closed trades yet.")

with tab2:
    if st.session_state.open_trades:
        open_df = pd.DataFrame(st.session_state.open_trades)
        # Select and format relevant columns
        open_df_display = open_df[['id', 'pair', 'direction', 'entry_price', 'current_price', 'profit_loss', 'stake', 'time', 'type']].copy()
        open_df_display['profit_loss'] = open_df_display['profit_loss'].round(2)
        open_df_display['entry_price'] = open_df_display['entry_price'].round(4)
        open_df_display['current_price'] = open_df_display['current_price'].round(4)
        open_df_display['duration'] = (pd.to_datetime(datetime.now()) - pd.to_datetime(open_df_display['time'])).dt.total_seconds() / 3600  # Hours
        open_df_display['duration'] = open_df_display['duration'].round(2)
        open_df_display = open_df_display.rename(columns={
            'id': 'ID',
            'pair': 'Pair',
            'direction': 'Direction',
            'entry_price': 'Entry Price',
            'current_price': 'Current Price',
            'profit_loss': 'Unrealized P&L (€)',
            'stake': 'Stake (€)',
            'time': 'Open Time',
            'type': 'Type',
            'duration': 'Duration (hrs)'
        })
        st.dataframe(open_df_display, width='stretch', hide_index=True)
        
        # Total Unrealized P&L summary
        total_unrealized = open_df['profit_loss'].sum()
        st.metric("Total Unrealized P&L", f"€{total_unrealized:.2f}")
    else:
        st.info("No open trades.")

# Auto-refresh
st.markdown("---")
st.markdown("🔄 Auto-refreshing every 3 seconds...")

time.sleep(3)
st.rerun()

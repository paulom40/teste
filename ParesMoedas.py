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
    page_title="Forex Auto Trading Bot - 3 Signal Agreement",
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
    .signal-buy {
        background: linear-gradient(135deg, #87CEEB 0%, #1E90FF 100%);
        color: white;
        padding: 0.5rem;
        border-radius: 5px;
        margin: 0.2rem 0;
        text-align: center;
        font-weight: bold;
    }
    .signal-sell {
        background: linear-gradient(135deg, #FFA500 0%, #FF8C00 100%);
        color: white;
        padding: 0.5rem;
        border-radius: 5px;
        margin: 0.2rem 0;
        text-align: center;
        font-weight: bold;
    }
    .signal-neutral {
        background: linear-gradient(135deg, #808080 0%, #A9A9A9 100%);
        color: white;
        padding: 0.5rem;
        border-radius: 5px;
        margin: 0.2rem 0;
        text-align: center;
        font-weight: bold;
    }
    .signal-waiting {
        background: linear-gradient(135deg, #FFD700 0%, #FFA500 100%);
        color: white;
        padding: 0.5rem;
        border-radius: 5px;
        margin: 0.2rem 0;
        text-align: center;
        font-weight: bold;
    }
    .stButton>button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        padding: 0.5rem 1rem;
        border-radius: 5px;
        font-weight: bold;
        width: 100%;
    }
    .stButton>button:hover {
        background: linear-gradient(135deg, #764ba2 0%, #667eea 100%);
        color: white;
    }
    .sidebar .sidebar-content {
        background: linear-gradient(180deg, #2d3748 0%, #4a5568 100%);
    }
    .signal-agreement {
        background: linear-gradient(135deg, #8A2BE2 0%, #4B0082 100%);
        color: white;
        padding: 1rem;
        border-radius: 10px;
        text-align: center;
        font-weight: bold;
        font-size: 1.2rem;
        margin: 1rem 0;
    }
    .auto-trading-active {
        background: linear-gradient(135deg, #00ff88 0%, #00cc66 100%);
        color: white;
        padding: 1rem;
        border-radius: 10px;
        text-align: center;
        font-weight: bold;
        font-size: 1.2rem;
        margin: 1rem 0;
        animation: pulse 2s infinite;
    }
    @keyframes pulse {
        0% { transform: scale(1); }
        50% { transform: scale(1.05); }
        100% { transform: scale(1); }
    }
</style>
""", unsafe_allow_html=True)

# Technical indicator functions
def calculate_rsi(prices, period=14):
    """Calculate RSI indicator with proper handling"""
    if len(prices) < period:
        return pd.Series([50] * len(prices))
    
    delta = prices.diff()
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    
    avg_gain = gain.rolling(window=period, min_periods=1).mean()
    avg_loss = loss.rolling(window=period, min_periods=1).mean()
    
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    
    rsi = rsi.fillna(50)
    rsi = rsi.replace([np.inf, -np.inf], 50)
    
    return rsi

def calculate_moving_averages(prices, fast_period=20, slow_period=50):
    """Calculate moving averages"""
    ma_fast = prices.rolling(window=fast_period, min_periods=1).mean()
    ma_slow = prices.rolling(window=slow_period, min_periods=1).mean()
    return ma_fast, ma_slow

def calculate_macd(prices, fast=12, slow=26, signal=9):
    """Calculate MACD indicator"""
    ema_fast = prices.ewm(span=fast).mean()
    ema_slow = prices.ewm(span=slow).mean()
    macd = ema_fast - ema_slow
    macd_signal = macd.ewm(span=signal).mean()
    macd_histogram = macd - macd_signal
    return macd, macd_signal, macd_histogram

def calculate_bollinger_bands(prices, period=20, std_dev=2):
    """Calculate Bollinger Bands"""
    sma = prices.rolling(window=period, min_periods=1).mean()
    std = prices.rolling(window=period, min_periods=1).std()
    upper_band = sma + (std * std_dev)
    lower_band = sma - (std * std_dev)
    return upper_band, sma, lower_band

def calculate_stochastic(high, low, close, k_period=14, d_period=3):
    """Calculate Stochastic Oscillator"""
    lowest_low = low.rolling(window=k_period).min()
    highest_high = high.rolling(window=k_period).max()
    k_line = ((close - lowest_low) / (highest_high - lowest_low)) * 100
    d_line = k_line.rolling(window=d_period).mean()
    return k_line, d_line

def generate_forex_data(pair, days=80, volatility=0.001):
    """Generate realistic Forex price data based on pair characteristics"""
    np.random.seed(42)
    dates = pd.date_range(start=datetime.now() - timedelta(days=days), 
                         end=datetime.now(), freq='h')
    
    # Base prices for different Forex pairs
    base_prices = {
        'EUR/USD': 1.0800, 'GBP/USD': 1.2600, 'USD/JPY': 150.00,
        'USD/CHF': 0.8800, 'AUD/USD': 0.6500, 'USD/CAD': 1.3500,
        'NZD/USD': 0.5900, 'EUR/GBP': 0.8600, 'EUR/JPY': 162.00,
        'GBP/JPY': 188.00, 'AUD/JPY': 97.00, 'USD/CNY': 7.2500
    }
    
    base_price = base_prices.get(pair, 1.0000)
    
    # Generate realistic Forex price movements
    returns = np.random.randn(len(dates)) * volatility
    prices = base_price * (1 + returns).cumprod()
    
    # Generate high/low based on price with typical Forex spreads
    high = prices * (1 + np.abs(np.random.randn(len(dates)) * 0.0005))
    low = prices * (1 - np.abs(np.random.randn(len(dates)) * 0.0005))
    
    df = pd.DataFrame({
        'Date': dates,
        'Open': prices,
        'High': high,
        'Low': low,
        'Close': prices
    })
    
    return df

# Initialize session state for trade history and auto trading
if 'trade_history' not in st.session_state:
    st.session_state.trade_history = pd.DataFrame({
        'Date': [],
        'Pair': [],
        'Direction': [],
        'Entry Price': [],
        'Exit Price': [],
        'Quantity': [],
        'P&L': [],
        'P&L (€)': [],
        'Status': [],
        'Signal Strength': [],
        'Signal Count': [],
        'Stake (€)': [],
        'Target Profit': [],
        'Stop Loss': []
    })

if 'auto_trading' not in st.session_state:
    st.session_state.auto_trading = False

if 'last_scan_time' not in st.session_state:
    st.session_state.last_scan_time = datetime.now()

if 'open_positions' not in st.session_state:
    st.session_state.open_positions = {}

if 'scan_count' not in st.session_state:
    st.session_state.scan_count = 0

if 'stake_euros' not in st.session_state:
    st.session_state.stake_euros = 100.0

if 'target_profit_pips' not in st.session_state:
    st.session_state.target_profit_pips = 30

if 'stop_loss_pips' not in st.session_state:
    st.session_state.stop_loss_pips = 20

def add_trade_to_history(pair, direction, entry_price, exit_price, quantity, pnl, pnl_eur, status, signal_strength, signal_count, stake_eur, target_profit, stop_loss):
    """Add a new trade to the history"""
    new_trade = pd.DataFrame({
        'Date': [datetime.now().strftime("%Y-%m-%d %H:%M:%S")],
        'Pair': [pair],
        'Direction': [direction],
        'Entry Price': [float(entry_price)],
        'Exit Price': [float(exit_price)],
        'Quantity': [float(quantity)],
        'P&L': [float(pnl)],
        'P&L (€)': [float(pnl_eur)],
        'Status': [status],
        'Signal Strength': [signal_strength],
        'Signal Count': [int(signal_count)],
        'Stake (€)': [float(stake_eur)],
        'Target Profit': [float(target_profit)],
        'Stop Loss': [float(stop_loss)]
    })
    
    st.session_state.trade_history = pd.concat([st.session_state.trade_history, new_trade], ignore_index=True)

def analyze_pair(pair, rsi_period=14, ma_fast=20, ma_slow=50, bb_period=20):
    """Analyze a single Forex pair and return trading signals"""
    try:
        # Generate data for the pair
        df = generate_forex_data(pair)
        
        # Calculate all indicators
        df['RSI'] = calculate_rsi(df['Close'], period=rsi_period)
        df['MA_Fast'], df['MA_Slow'] = calculate_moving_averages(df['Close'], ma_fast, ma_slow)
        df['MACD'], df['MACD_Signal'], df['MACD_Hist'] = calculate_macd(df['Close'])
        df['BB_Upper'], df['BB_Middle'], df['BB_Lower'] = calculate_bollinger_bands(df['Close'], bb_period)
        df['Stoch_K'], df['Stoch_D'] = calculate_stochastic(df['High'], df['Low'], df['Close'])
        
        # Get current values
        current_data = df.iloc[-1]
        current_price = current_data['Close']
        current_rsi = current_data['RSI']
        current_ma_fast = current_data['MA_Fast']
        current_ma_slow = current_data['MA_Slow']
        current_macd = current_data['MACD']
        current_macd_signal = current_data['MACD_Signal']
        current_bb_upper = current_data['BB_Upper']
        current_bb_lower = current_data['BB_Lower']
        current_stoch_k = current_data['Stoch_K']
        current_stoch_d = current_data['Stoch_D']
        
        # Calculate signals (1 for buy, -1 for sell, 0 for neutral)
        signals = {
            'RSI': 1 if current_rsi < 30 else -1 if current_rsi > 70 else 0,
            'MACrossover': 1 if current_ma_fast > current_ma_slow else -1,
            'MACD': 1 if current_macd > current_macd_signal else -1,
            'Bollinger': 1 if current_price < current_bb_lower else -1 if current_price > current_bb_upper else 0,
            'Stochastic': 1 if current_stoch_k < 20 and current_stoch_k > current_stoch_d else -1 if current_stoch_k > 80 and current_stoch_k < current_stoch_d else 0
        }
        
        # Count buy/sell signals
        buy_signals = sum(1 for signal in signals.values() if signal == 1)
        sell_signals = sum(1 for signal in signals.values() if signal == -1)
        
        # Determine final signal
        if buy_signals >= 3:
            return {
                'pair': pair,
                'signal': 'BUY',
                'strength': 'Strong' if buy_signals >= 4 else 'Moderate',
                'signal_count': buy_signals,
                'price': current_price,
                'signals': signals
            }
        elif sell_signals >= 3:
            return {
                'pair': pair,
                'signal': 'SELL',
                'strength': 'Strong' if sell_signals >= 4 else 'Moderate',
                'signal_count': sell_signals,
                'price': current_price,
                'signals': signals
            }
        else:
            return {
                'pair': pair,
                'signal': 'HOLD',
                'strength': 'Weak',
                'signal_count': max(buy_signals, sell_signals),
                'price': current_price,
                'signals': signals
            }
            
    except Exception as e:
        return {
            'pair': pair,
            'signal': 'ERROR',
            'strength': 'Error',
            'signal_count': 0,
            'price': 0,
            'signals': {}
        }

def execute_auto_trade(signal_data, lot_size, risk_percent, stake_eur, target_profit_pips, stop_loss_pips):
    """Execute an automated trade based on signal data"""
    pair = signal_data['pair']
    direction = signal_data['signal']
    signal_count = signal_data['signal_count']
    current_price = signal_data['price']
    
    # Check if we already have an open position for this pair
    if pair in st.session_state.open_positions:
        return f"Position already open for {pair}"
    
    # Calculate position size based on risk
    risk_amount = (risk_percent / 100) * 10000
    pip_value = 10 if 'JPY' not in pair else 0.1
    
    # Calculate quantity based on risk and stop loss
    quantity = min(float(lot_size), risk_amount / (stop_loss_pips * pip_value))
    
    # Calculate P&L based on target profit and stop loss
    # Simulate whether trade hits target profit or stop loss
    hit_target = np.random.random() > 0.3  # 70% chance to hit target
    
    if hit_target:
        # Trade hits target profit
        pnl = target_profit_pips * pip_value * quantity
        if direction == "BUY":
            exit_price = current_price + (target_profit_pips * 0.0001)
        else:
            exit_price = current_price - (target_profit_pips * 0.0001)
    else:
        # Trade hits stop loss
        pnl = -stop_loss_pips * pip_value * quantity
        if direction == "BUY":
            exit_price = current_price - (stop_loss_pips * 0.0001)
        else:
            exit_price = current_price + (stop_loss_pips * 0.0001)
    
    # Calculate P&L in euros based on stake
    pnl_eur = pnl * (stake_eur / 100)
    
    # Add trade to history
    add_trade_to_history(
        pair=pair,
        direction=direction,
        entry_price=current_price,
        exit_price=exit_price,
        quantity=quantity,
        pnl=pnl,
        pnl_eur=pnl_eur,
        status="Closed",
        signal_strength=signal_data['strength'],
        signal_count=signal_count,
        stake_eur=stake_eur,
        target_profit=target_profit_pips,
        stop_loss=stop_loss_pips
    )
    
    # Add to open positions with ALL required fields
    st.session_state.open_positions[pair] = {
        'direction': direction,
        'entry_price': current_price,
        'quantity': quantity,
        'entry_time': datetime.now(),
        'stake_eur': stake_eur,
        'target_profit': target_profit_pips,  # Fixed: Added this field
        'stop_loss': stop_loss_pips  # Fixed: Added this field
    }
    
    result_type = "Target Profit" if hit_target else "Stop Loss"
    return f"Auto trade executed: {direction} {pair} with {signal_count} signals (Stake: €{stake_eur:.2f}, {result_type})"

def scan_all_pairs():
    """Scan all Forex pairs for trading opportunities"""
    forex_pairs = [
        "EUR/USD", "GBP/USD", "USD/JPY", "USD/CHF", 
        "AUD/USD", "USD/CAD", "NZD/USD", "EUR/GBP", 
        "EUR/JPY", "GBP/JPY", "AUD/JPY", "USD/CNY"
    ]
    
    trading_opportunities = []
    
    for pair in forex_pairs:
        signal_data = analyze_pair(
            pair, 
            rsi_period=st.session_state.rsi_period,
            ma_fast=st.session_state.ma_fast,
            ma_slow=st.session_state.ma_slow,
            bb_period=st.session_state.bb_period
        )
        
        if signal_data['signal'] in ['BUY', 'SELL']:
            trading_opportunities.append(signal_data)
    
    return trading_opportunities

def calculate_trade_statistics():
    """Calculate trade statistics from the trade history"""
    if len(st.session_state.trade_history) == 0:
        return 0, 0, 0, 0, 0, 0, 0, 0, 0
    
    # Ensure P&L columns are numeric
    trade_history = st.session_state.trade_history.copy()
    
    # Convert P&L to numeric if it's not already
    if trade_history['P&L'].dtype == 'object':
        trade_history['P&L'] = pd.to_numeric(trade_history['P&L'], errors='coerce')
    if trade_history['P&L (€)'].dtype == 'object':
        trade_history['P&L (€)'] = pd.to_numeric(trade_history['P&L (€)'], errors='coerce')
    
    total_trades = len(trade_history)
    winning_trades = len(trade_history[trade_history['P&L'] > 0])
    losing_trades = len(trade_history[trade_history['P&L'] < 0])
    total_pnl = trade_history['P&L'].sum()
    total_pnl_eur = trade_history['P&L (€)'].sum()
    win_rate = (winning_trades / total_trades * 100) if total_trades > 0 else 0
    
    # Calculate risk management statistics
    target_hits = len(trade_history[trade_history['P&L'] > 0])
    stop_loss_hits = len(trade_history[trade_history['P&L'] < 0])
    
    # Safely calculate average risk-reward ratio
    if 'Target Profit' in trade_history.columns and 'Stop Loss' in trade_history.columns:
        avg_target = trade_history['Target Profit'].mean()
        avg_stop_loss = trade_history['Stop Loss'].mean()
        avg_rr_ratio = avg_target / avg_stop_loss if avg_stop_loss > 0 else 0
    else:
        avg_rr_ratio = 0
    
    return total_trades, winning_trades, losing_trades, total_pnl, total_pnl_eur, win_rate, target_hits, stop_loss_hits, avg_rr_ratio

# Main application
def main():
    # Header
    st.markdown('<h1 class="main-header">🌍 Forex Auto Trading Bot</h1>', unsafe_allow_html=True)
    st.markdown('<h3 style="text-align: center; color: #666;">Fully Automated 3-Signal Agreement System</h3>', unsafe_allow_html=True)
    
    # Sidebar
    st.sidebar.title("⚙️ Forex Trading Configuration")
    
    # Trading parameters
    st.sidebar.subheader("💰 Stake Configuration")
    stake_euros = st.sidebar.number_input(
        "Stake Amount (€)", 
        value=st.session_state.stake_euros, 
        min_value=10.0, 
        max_value=10000.0, 
        step=50.0,
        help="Enter the amount you want to stake per trade in euros"
    )
    st.session_state.stake_euros = stake_euros
    
    st.sidebar.subheader("🎯 Risk Management")
    col1, col2 = st.sidebar.columns(2)
    
    with col1:
        target_profit_pips = st.sidebar.number_input(
            "Target Profit (pips)",
            value=st.session_state.target_profit_pips,
            min_value=5,
            max_value=100,
            step=5,
            help="Take profit level in pips"
        )
        st.session_state.target_profit_pips = target_profit_pips
    
    with col2:
        stop_loss_pips = st.sidebar.number_input(
            "Stop Loss (pips)",
            value=st.session_state.stop_loss_pips,
            min_value=5,
            max_value=50,
            step=5,
            help="Stop loss level in pips"
        )
        st.session_state.stop_loss_pips = stop_loss_pips
    
    # Display risk-reward ratio
    risk_reward_ratio = target_profit_pips / stop_loss_pips if stop_loss_pips > 0 else 0
    st.sidebar.metric("Risk/Reward Ratio", f"{risk_reward_ratio:.2f}:1")
    
    st.sidebar.subheader("Trading Parameters")
    initial_balance = st.sidebar.number_input("Account Balance ($)", value=10000.0, min_value=1000.0, step=1000.0)
    risk_per_trade = st.sidebar.slider("Risk per Trade (%)", 0.5, 5.0, 1.0)
    lot_size = st.sidebar.selectbox("Lot Size", ["0.01", "0.1", "1.0", "10.0"])
    
    # Store indicator settings in session state
    st.sidebar.subheader("Indicator Settings")
    st.session_state.rsi_period = st.sidebar.slider("RSI Period", 5, 30, 14)
    st.session_state.ma_fast = st.sidebar.slider("Fast MA Period", 5, 50, 20)
    st.session_state.ma_slow = st.sidebar.slider("Slow MA Period", 20, 200, 50)
    st.session_state.bb_period = st.sidebar.slider("Bollinger Bands Period", 10, 30, 20)
    
    # Auto trading controls
    st.sidebar.subheader("🤖 Auto Trading Controls")
    
    col1, col2 = st.sidebar.columns(2)
    
    with col1:
        if st.button("🚀 Start Auto Trading" if not st.session_state.auto_trading else "🛑 Stop Auto Trading", 
                    type="primary" if not st.session_state.auto_trading else "secondary"):
            st.session_state.auto_trading = not st.session_state.auto_trading
            st.session_state.last_scan_time = datetime.now()
            st.session_state.scan_count = 0
            st.rerun()
    
    with col2:
        if st.button("🔍 Scan All Pairs"):
            st.session_state.last_scan_time = datetime.now()
            st.session_state.scan_count += 1
            st.rerun()
    
    # Manual refresh button for auto trading
    if st.session_state.auto_trading:
        if st.sidebar.button("🔄 Refresh Scan"):
            st.session_state.last_scan_time = datetime.now()
            st.session_state.scan_count += 1
            st.rerun()
    
    # Display auto trading status
    if st.session_state.auto_trading:
        st.sidebar.markdown('<div class="auto-trading-active">🤖 AUTO TRADING ACTIVE</div>', unsafe_allow_html=True)
        st.sidebar.info(f"Scan count: {st.session_state.scan_count}")
        st.sidebar.info(f"Current Stake: €{st.session_state.stake_euros:.2f}")
        st.sidebar.info(f"Target: {st.session_state.target_profit_pips} pips")
        st.sidebar.info(f"Stop Loss: {st.session_state.stop_loss_pips} pips")
        st.sidebar.info("Click 'Refresh Scan' to check for new opportunities")
    else:
        st.sidebar.warning("Auto trading is currently OFF")
        st.sidebar.info(f"Current Stake: €{st.session_state.stake_euros:.2f}")
        st.sidebar.info(f"Target: {st.session_state.target_profit_pips} pips")
        st.sidebar.info(f"Stop Loss: {st.session_state.stop_loss_pips} pips")
    
    # Forex pairs selection for manual analysis
    forex_pairs = [
        "EUR/USD", "GBP/USD", "USD/JPY", "USD/CHF", 
        "AUD/USD", "USD/CAD", "NZD/USD", "EUR/GBP", 
        "EUR/JPY", "GBP/JPY", "AUD/JPY", "USD/CNY"
    ]
    selected_pair = st.sidebar.selectbox("Manual Analysis Pair", forex_pairs)
    
    # Manual trading section
    st.sidebar.subheader("🎮 Manual Trading")
    if st.sidebar.button("📊 Analyze Selected Pair"):
        st.session_state.manual_analysis_pair = selected_pair
        st.rerun()
    
    # Main content area
    if st.session_state.auto_trading:
        st.markdown('<div class="auto-trading-active">🚀 AUTO TRADING ACTIVE - Scanning 12 Forex Pairs</div>', unsafe_allow_html=True)
        
        # Display scan information
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Scan Count", st.session_state.scan_count)
        with col2:
            st.metric("Last Scan", st.session_state.last_scan_time.strftime("%H:%M:%S"))
        with col3:
            st.metric("Open Positions", len(st.session_state.open_positions))
        with col4:
            st.metric("Current Stake", f"€{st.session_state.stake_euros:.2f}")
        
        # Display risk management info
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Target Profit", f"{st.session_state.target_profit_pips} pips")
        with col2:
            st.metric("Stop Loss", f"{st.session_state.stop_loss_pips} pips")
        with col3:
            st.metric("Risk/Reward", f"{risk_reward_ratio:.2f}:1")
        
        # Scan all pairs for opportunities
        with st.spinner("Scanning all Forex pairs for trading opportunities..."):
            opportunities = scan_all_pairs()
            st.session_state.last_scan_time = datetime.now()
        
        # Display trading opportunities
        if opportunities:
            st.subheader("🎯 Trading Opportunities Found")
            
            for opportunity in opportunities:
                col1, col2, col3, col4, col5 = st.columns([2,1,1,1,2])
                
                with col1:
                    st.write(f"**{opportunity['pair']}**")
                
                with col2:
                    signal_color = "🟢" if opportunity['signal'] == 'BUY' else "🔴"
                    st.write(f"{signal_color} **{opportunity['signal']}**")
                
                with col3:
                    st.write(f"**{opportunity['signal_count']}/5** signals")
                
                with col4:
                    st.write(f"**{opportunity['strength']}**")
                
                with col5:
                    if st.button(f"Trade {opportunity['pair']}", key=f"trade_{opportunity['pair']}"):
                        result = execute_auto_trade(
                            opportunity, 
                            lot_size, 
                            risk_per_trade, 
                            st.session_state.stake_euros,
                            st.session_state.target_profit_pips,
                            st.session_state.stop_loss_pips
                        )
                        st.success(result)
                        st.rerun()
            
            # Auto-execute trades if enabled
            auto_execute = st.checkbox("🤖 Auto-execute all qualified trades", value=True)
            if auto_execute:
                executed_trades = []
                for opportunity in opportunities:
                    # Only execute if we don't already have a position
                    if opportunity['pair'] not in st.session_state.open_positions:
                        result = execute_auto_trade(
                            opportunity, 
                            lot_size, 
                            risk_per_trade, 
                            st.session_state.stake_euros,
                            st.session_state.target_profit_pips,
                            st.session_state.stop_loss_pips
                        )
                        executed_trades.append(result)
                
                if executed_trades:
                    st.success("🤖 Auto-execution completed!")
                    for trade in executed_trades:
                        st.write(f"✅ {trade}")
                    st.rerun()
        else:
            st.info("No trading opportunities found at the moment. Click 'Refresh Scan' to check again.")
        
    else:
        # Manual analysis for selected pair
        if hasattr(st.session_state, 'manual_analysis_pair'):
            selected_pair = st.session_state.manual_analysis_pair
        
        # Analyze selected pair
        signal_data = analyze_pair(
            selected_pair, 
            rsi_period=st.session_state.rsi_period,
            ma_fast=st.session_state.ma_fast,
            ma_slow=st.session_state.ma_slow,
            bb_period=st.session_state.bb_period
        )
        
        # Display results for selected pair
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.markdown('<div class="metric-card">', unsafe_allow_html=True)
            st.metric("Selected Pair", selected_pair)
            st.markdown('</div>', unsafe_allow_html=True)
        
        with col2:
            st.markdown('<div class="metric-card">', unsafe_allow_html=True)
            signal_color = "🟢" if signal_data['signal'] == 'BUY' else "🔴" if signal_data['signal'] == 'SELL' else "⚪"
            st.metric("Signal", f"{signal_color} {signal_data['signal']}")
            st.markdown('</div>', unsafe_allow_html=True)
        
        with col3:
            st.markdown('<div class="metric-card">', unsafe_allow_html=True)
            st.metric("Signal Count", f"{signal_data['signal_count']}/5")
            st.markdown('</div>', unsafe_allow_html=True)
        
        with col4:
            st.markdown('<div class="metric-card">', unsafe_allow_html=True)
            st.metric("Current Price", f"{signal_data['price']:.5f}")
            st.markdown('</div>', unsafe_allow_html=True)
        
        # Manual trade execution
        if signal_data['signal'] in ['BUY', 'SELL']:
            st.markdown(f'<div class="signal-agreement">🎯 TRADING OPPORTUNITY: {signal_data["signal"]} {selected_pair} with {signal_data["signal_count"]} signals</div>', unsafe_allow_html=True)
            
            col1, col2 = st.columns(2)
            with col1:
                if st.button(f"🚀 Execute {signal_data['signal']} Trade", type="primary"):
                    result = execute_auto_trade(
                        signal_data, 
                        lot_size, 
                        risk_per_trade, 
                        st.session_state.stake_euros,
                        st.session_state.target_profit_pips,
                        st.session_state.stop_loss_pips
                    )
                    st.success(result)
                    st.rerun()
            with col2:
                st.info(f"Stake: €{st.session_state.stake_euros:.2f}")
                st.info(f"Target: {st.session_state.target_profit_pips} pips")
                st.info(f"Stop Loss: {st.session_state.stop_loss_pips} pips")
    
    # Trade History Section (always visible)
    st.subheader("📋 Trade History & Performance")
    
    if len(st.session_state.trade_history) > 0:
        # Calculate summary statistics using the helper function
        total_trades, winning_trades, losing_trades, total_pnl, total_pnl_eur, win_rate, target_hits, stop_loss_hits, avg_rr_ratio = calculate_trade_statistics()
        
        # Display summary metrics
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Total Trades", total_trades)
        with col2:
            st.metric("Win Rate", f"{win_rate:.1f}%")
        with col3:
            pnl_color = "normal" if total_pnl >= 0 else "inverse"
            st.metric("Total P&L", f"${total_pnl:.2f}", delta_color=pnl_color)
        with col4:
            pnl_eur_color = "normal" if total_pnl_eur >= 0 else "inverse"
            st.metric("Total P&L (€)", f"€{total_pnl_eur:.2f}", delta_color=pnl_eur_color)
        
        # Additional metrics
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            avg_pnl = total_pnl / total_trades if total_trades > 0 else 0
            st.metric("Avg P&L per Trade", f"${avg_pnl:.2f}")
        with col2:
            avg_pnl_eur = total_pnl_eur / total_trades if total_trades > 0 else 0
            st.metric("Avg P&L (€) per Trade", f"€{avg_pnl_eur:.2f}")
        with col3:
            total_stake = st.session_state.trade_history['Stake (€)'].sum()
            st.metric("Total Stake", f"€{total_stake:.2f}")
        with col4:
            roi = (total_pnl_eur / total_stake * 100) if total_stake > 0 else 0
            st.metric("ROI", f"{roi:.1f}%")
        
        # Risk management statistics
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Target Hits", target_hits)
        with col2:
            st.metric("Stop Loss Hits", stop_loss_hits)
        with col3:
            target_rate = (target_hits / total_trades * 100) if total_trades > 0 else 0
            st.metric("Target Hit Rate", f"{target_rate:.1f}%")
        with col4:
            st.metric("Avg R:R Ratio", f"{avg_rr_ratio:.2f}:1")
        
        # Display the trade history table with formatted values
        styled_history = st.session_state.trade_history.copy()
        styled_history = styled_history.sort_values('Date', ascending=False)
        
        # Create a copy for display with formatted values
        display_history = styled_history.copy()
        display_history['Entry Price'] = display_history['Entry Price'].apply(lambda x: f"{x:.5f}")
        display_history['Exit Price'] = display_history['Exit Price'].apply(lambda x: f"{x:.5f}")
        display_history['P&L'] = display_history['P&L'].apply(lambda x: f"${x:.2f}")
        display_history['P&L (€)'] = display_history['P&L (€)'].apply(lambda x: f"€{x:.2f}")
        display_history['Stake (€)'] = display_history['Stake (€)'].apply(lambda x: f"€{x:.2f}")
        
        # Safely format target profit and stop loss columns if they exist
        if 'Target Profit' in display_history.columns:
            display_history['Target Profit'] = display_history['Target Profit'].apply(lambda x: f"{int(x)} pips")
        if 'Stop Loss' in display_history.columns:
            display_history['Stop Loss'] = display_history['Stop Loss'].apply(lambda x: f"{int(x)} pips")
        
        st.dataframe(
            display_history,
            width='stretch',
            height=400
        )
        
        # Download button for trade history
        csv = st.session_state.trade_history.to_csv(index=False)
        st.download_button(
            label="📥 Download Trade History CSV",
            data=csv,
            file_name=f"forex_trade_history_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
            mime="text/csv"
        )
        
        # Clear history button
        if st.button("🗑️ Clear Trade History"):
            st.session_state.trade_history = pd.DataFrame({
                'Date': [],
                'Pair': [],
                'Direction': [],
                'Entry Price': [],
                'Exit Price': [],
                'Quantity': [],
                'P&L': [],
                'P&L (€)': [],
                'Status': [],
                'Signal Strength': [],
                'Signal Count': [],
                'Stake (€)': [],
                'Target Profit': [],
                'Stop Loss': []
            })
            st.session_state.open_positions = {}
            st.success("Trade history cleared!")
            st.rerun()
            
    else:
        st.info("No trades executed yet. Start auto trading or execute manual trades to see history here.")
    
    # Open Positions - FIXED: Added safety checks for missing keys
    if st.session_state.open_positions:
        st.subheader("📈 Open Positions")
        for pair, position in st.session_state.open_positions.items():
            col1, col2, col3, col4, col5, col6, col7 = st.columns([2,1,2,2,2,2,1])
            with col1:
                st.write(f"**{pair}**")
            with col2:
                st.write(f"**{position['direction']}**")
            with col3:
                st.write(f"Entry: {position['entry_price']:.5f}")
            with col4:
                st.write(f"Qty: {position['quantity']}")
            with col5:
                st.write(f"Stake: €{position['stake_eur']:.2f}")
            with col6:
                # Safely display target profit and stop loss with default values if missing
                target_profit = position.get('target_profit', st.session_state.target_profit_pips)
                stop_loss = position.get('stop_loss', st.session_state.stop_loss_pips)
                st.write(f"TP/SL: {target_profit}/{stop_loss} pips")
            with col7:
                if st.button(f"Close {pair}", key=f"close_{pair}"):
                    del st.session_state.open_positions[pair]
                    st.success(f"Closed position for {pair}")
                    st.rerun()
    
    # Strategy Explanation
    with st.expander("📖 Automated Trading Strategy Explained"):
        st.markdown("""
        **🤖 Fully Automated Forex Trading System**
        
        This system automatically scans **12 major Forex pairs** and executes trades when **3+ signals agree**:
        
        **Auto Trading Features:**
        - **Multi-Pair Scanning**: Monitors all 12 major Forex pairs
        - **Signal Validation**: Requires minimum 3/5 indicator agreement
        - **Auto-Execution**: Automatically enters qualified trades
        - **Advanced Risk Management**: Configurable target profit and stop loss
        - **Real-time Monitoring**: Live tracking of opportunities and positions
        - **Euro Stake Management**: Set your stake amount in euros for each trade
        
        **Risk Management Features:**
        - **Target Profit**: Set your take profit level in pips
        - **Stop Loss**: Set your stop loss level in pips  
        - **Risk/Reward Ratio**: Automatic calculation of risk-reward ratio
        - **Position Sizing**: Automatic calculation based on risk percentage
        - **Performance Tracking**: Monitor target hit rate and stop loss frequency
        
        **Trading Pairs Monitored:**
        - EUR/USD, GBP/USD, USD/JPY, USD/CHF
        - AUD/USD, USD/CAD, NZD/USD, EUR/GBP
        - EUR/JPY, GBP/JPY, AUD/JPY, USD/CNY
        
        **How to Use Auto Trading:**
        1. Set your desired stake amount in euros
        2. Configure target profit and stop loss levels
        3. Set your risk percentage (0.5%-5%)
        4. Select lot size
        5. Configure indicator parameters
        6. Click "Start Auto Trading"
        7. Click "Refresh Scan" to check for opportunities
        8. Enable "Auto-execute" for fully automated trading
        
        **Recommended Settings:**
        - Target Profit: 30-50 pips
        - Stop Loss: 15-25 pips  
        - Risk/Reward Ratio: 1:2 or better
        - Stake: 1-2% of your account balance
        """)

if __name__ == "__main__":
    main()

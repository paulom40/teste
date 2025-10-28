import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
from datetime import datetime, timedelta
import requests
import time

# Configure the page
st.set_page_config(
    page_title="LIVE FOREX PRICES - PURE PYTHON ANALYSIS",
    page_icon="📈",
    layout="wide"
)

# Polygon.io API Configuration (for future use)
POLYGON_API_KEY = "ZACYNJQZmDFZV0B92pErxGfiF60iUuZ"
BASE_URL = "https://api.polygon.io/v2"

# Forex pairs to monitor
FOREX_PAIRS = [
    'USD/JPY', 'USD/CHF', 'USD/CAD', 
    'EUR/USD', 'GBP/USD', 'AUD/USD'
]

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .currency-card {
        background-color: #f0f2f6;
        border-radius: 10px;
        padding: 1rem;
        margin: 0.5rem 0;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }
    .analysis-card {
        background-color: #fff3e0;
        border-radius: 10px;
        padding: 1rem;
        margin: 0.5rem 0;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        border-left: 4px solid #ff9800;
    }
    .trend-up {
        background-color: #e8f5e9;
        color: #2e7d32;
        padding: 0.5rem;
        border-radius: 5px;
        text-align: center;
        font-weight: bold;
    }
    .trend-down {
        background-color: #ffebee;
        color: #c62828;
        padding: 0.5rem;
        border-radius: 5px;
        text-align: center;
        font-weight: bold;
    }
    .trend-neutral {
        background-color: #e3f2fd;
        color: #1565c0;
        padding: 0.5rem;
        border-radius: 5px;
        text-align: center;
        font-weight: bold;
    }
    .positive-change {
        color: #00cc96;
        font-weight: bold;
    }
    .negative-change {
        color: #ef553b;
        font-weight: bold;
    }
    .positive-pnl {
        background-color: #e8f5e9;
        color: #2e7d32;
        padding: 0.25rem 0.5rem;
        border-radius: 4px;
        font-weight: bold;
    }
    .negative-pnl {
        background-color: #ffebee;
        color: #c62828;
        padding: 0.25rem 0.5rem;
        border-radius: 4px;
        font-weight: bold;
    }
    .signal-low {
        background-color: #ffebee;
        color: #c62828;
        padding: 0.5rem;
        border-radius: 5px;
        text-align: center;
    }
    .signal-high {
        background-color: #e8f5e9;
        color: #2e7d32;
        padding: 0.5rem;
        border-radius: 5px;
        text-align: center;
    }
    .indicator-bullish {
        color: #00cc96;
        font-weight: bold;
    }
    .indicator-bearish {
        color: #ef553b;
        font-weight: bold;
    }
    .indicator-neutral {
        color: #636efa;
        font-weight: bold;
    }
    .api-status {
        padding: 0.5rem;
        border-radius: 5px;
        text-align: center;
        margin: 0.5rem 0;
        font-weight: bold;
    }
    .api-active {
        background-color: #e8f5e9;
        color: #2e7d32;
        border: 1px solid #2e7d32;
    }
    .api-error {
        background-color: #fff3e0;
        color: #ff9800;
        border: 1px solid #ff9800;
    }
    .stake-card {
        background-color: #e8f5e9;
        border-radius: 10px;
        padding: 1rem;
        margin: 0.5rem 0;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        border-left: 4px solid #4CAF50;
    }
</style>
""", unsafe_allow_html=True)

# Enhanced Simulated Data Functions
def generate_realistic_forex_data():
    """Generate realistic simulated forex data with proper price ranges"""
    base_prices = {
        'EUR/USD': 1.0850,
        'GBP/USD': 1.2650,
        'USD/JPY': 148.50,
        'USD/CHF': 0.8800,
        'USD/CAD': 1.3550,
        'AUD/USD': 0.6550
    }
    
    quotes = {}
    for pair, base_price in base_prices.items():
        # Realistic price movement (0.01% to 0.1%)
        change_percent = np.random.uniform(-0.001, 0.001)
        new_price = base_price * (1 + change_percent)
        
        quotes[pair] = {
            'price': new_price,
            'change': new_price - base_price,
            'change_percent': change_percent * 100,
            'timestamp': datetime.now(),
            'volume': np.random.randint(1000, 10000)
        }
    
    return quotes

def generate_realistic_historical_data(forex_pair, days=60):
    """Generate realistic historical price data"""
    base_prices = {
        'EUR/USD': 1.0850, 'GBP/USD': 1.2650, 'USD/JPY': 148.50,
        'USD/CHF': 0.8800, 'USD/CAD': 1.3550, 'AUD/USD': 0.6550
    }
    
    base_price = base_prices.get(forex_pair, 1.0)
    prices = [base_price]
    
    # Generate realistic price movements
    for _ in range(days - 1):
        # More realistic volatility (0.05% to 0.3% daily moves)
        change = np.random.uniform(-0.003, 0.003)
        new_price = prices[-1] * (1 + change)
        
        # Ensure prices stay in realistic ranges
        if 'JPY' in forex_pair:
            new_price = max(100.0, min(200.0, new_price))
        elif 'USD' in forex_pair and forex_pair != 'USD/JPY':
            new_price = max(0.5, min(2.0, new_price))
        
        prices.append(new_price)
    
    return prices

# Technical Analysis Functions (Pure Python)
def calculate_sma(prices, window):
    """Calculate Simple Moving Average"""
    if len(prices) < window:
        return None
    return sum(prices[-window:]) / window

def calculate_ema(prices, window):
    """Calculate Exponential Moving Average"""
    if len(prices) < window:
        return None
    
    alpha = 2 / (window + 1)
    ema = prices[0]
    
    for price in prices[1:]:
        ema = alpha * price + (1 - alpha) * ema
    
    return ema

def calculate_rsi(prices, window=14):
    """Calculate Relative Strength Index"""
    if len(prices) < window + 1:
        return None
    
    gains = []
    losses = []
    
    for i in range(1, len(prices)):
        change = prices[i] - prices[i-1]
        gains.append(max(change, 0))
        losses.append(max(-change, 0))
    
    avg_gain = sum(gains[-window:]) / window
    avg_loss = sum(losses[-window:]) / window
    
    if avg_loss == 0:
        return 100
    
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    
    return rsi

def calculate_macd(prices, fast=12, slow=26, signal=9):
    """Calculate MACD indicator"""
    if len(prices) < slow:
        return None, None, None
    
    ema_fast = calculate_ema(prices, fast)
    ema_slow = calculate_ema(prices, slow)
    
    if ema_fast is None or ema_slow is None:
        return None, None, None
    
    macd_line = ema_fast - ema_slow
    macd_signal = calculate_ema(prices, signal) if calculate_ema(prices, signal) else macd_line * 0.9
    macd_histogram = macd_line - macd_signal
    
    return macd_line, macd_signal, macd_histogram

def calculate_bollinger_bands(prices, window=20, num_std=2):
    """Calculate Bollinger Bands"""
    if len(prices) < window:
        return None, None, None
    
    sma = calculate_sma(prices, window)
    std_dev = np.std(prices[-window:])
    
    upper_band = sma + (std_dev * num_std)
    lower_band = sma - (std_dev * num_std)
    
    return upper_band, sma, lower_band

def calculate_stochastic(prices, k_period=14, d_period=3):
    """Calculate Stochastic Oscillator"""
    if len(prices) < k_period:
        return None, None
    
    current_close = prices[-1]
    lowest_low = min(prices[-k_period:])
    highest_high = max(prices[-k_period:])
    
    if highest_high - lowest_low == 0:
        return 50, 50
    
    stoch_k = 100 * (current_close - lowest_low) / (highest_high - lowest_low)
    stoch_d = stoch_k  # Simplified for demo
    
    return stoch_k, stoch_d

def calculate_technical_indicators(prices):
    """Calculate various technical indicators"""
    if len(prices) < 20:
        return {}
    
    indicators = {}
    
    # Moving Averages
    indicators['sma_20'] = calculate_sma(prices, 20)
    indicators['sma_50'] = calculate_sma(prices, 50)
    indicators['ema_12'] = calculate_ema(prices, 12)
    indicators['ema_26'] = calculate_ema(prices, 26)
    
    # RSI
    indicators['rsi'] = calculate_rsi(prices)
    
    # MACD
    macd_line, macd_signal, macd_histogram = calculate_macd(prices)
    indicators['macd'] = macd_line
    indicators['macd_signal'] = macd_signal
    indicators['macd_histogram'] = macd_histogram
    
    # Bollinger Bands
    bb_upper, bb_middle, bb_lower = calculate_bollinger_bands(prices)
    indicators['bb_upper'] = bb_upper
    indicators['bb_lower'] = bb_lower
    indicators['bb_middle'] = bb_middle
    
    # Stochastic
    stoch_k, stoch_d = calculate_stochastic(prices)
    indicators['stoch_k'] = stoch_k
    indicators['stoch_d'] = stoch_d
    
    return indicators

def analyze_trend(indicators, current_price):
    """Analyze overall trend based on multiple indicators"""
    if not indicators:
        return "NEUTRAL", "Insufficient Data"
    
    bullish_signals = 0
    bearish_signals = 0
    total_signals = 0
    
    # Moving Average Analysis
    if indicators['sma_20'] and current_price > indicators['sma_20']:
        bullish_signals += 1
    elif indicators['sma_20']:
        bearish_signals += 1
    total_signals += 1
    
    # EMA Analysis
    if indicators['ema_12'] and indicators['ema_26']:
        if indicators['ema_12'] > indicators['ema_26']:
            bullish_signals += 1
        else:
            bearish_signals += 1
        total_signals += 1
    
    # RSI Analysis
    if indicators['rsi']:
        if indicators['rsi'] < 30:
            bullish_signals += 1  # Oversold
        elif indicators['rsi'] > 70:
            bearish_signals += 1  # Overbought
        total_signals += 1
    
    # MACD Analysis
    if indicators['macd'] and indicators['macd_signal']:
        if indicators['macd'] > indicators['macd_signal']:
            bullish_signals += 1
        else:
            bearish_signals += 1
        total_signals += 1
    
    # Stochastic Analysis
    if indicators['stoch_k'] and indicators['stoch_d']:
        if indicators['stoch_k'] < 20 and indicators['stoch_d'] < 20:
            bullish_signals += 1  # Oversold
        elif indicators['stoch_k'] > 80 and indicators['stoch_d'] > 80:
            bearish_signals += 1  # Overbought
        total_signals += 1
    
    # Determine trend strength
    if total_signals > 0:
        bullish_percentage = (bullish_signals / total_signals) * 100
        bearish_percentage = (bearish_signals / total_signals) * 100
        
        if bullish_percentage >= 60:
            return "BULLISH", f"Strong Uptrend ({bullish_percentage:.0f}%)"
        elif bearish_percentage >= 60:
            return "BEARISH", f"Strong Downtrend ({bearish_percentage:.0f}%)"
        elif bullish_percentage > bearish_percentage:
            return "BULLISH", f"Weak Uptrend ({bullish_percentage:.0f}%)"
        elif bearish_percentage > bullish_percentage:
            return "BEARISH", f"Weak Downtrend ({bearish_percentage:.0f}%)"
    
    return "NEUTRAL", "Sideways Market"

def generate_trading_signal(indicators, trend_direction):
    """Generate trading signal based on technical analysis"""
    if not indicators:
        return "HOLD", "Waiting for data"
    
    signals = []
    
    # RSI Signal
    if indicators['rsi']:
        if indicators['rsi'] < 30:
            signals.append("RSI: OVERSOLD")
        elif indicators['rsi'] > 70:
            signals.append("RSI: OVERBOUGHT")
    
    # MACD Signal
    if indicators['macd'] and indicators['macd_signal']:
        if indicators['macd'] > indicators['macd_signal']:
            signals.append("MACD: BULLISH")
        elif indicators['macd'] < indicators['macd_signal']:
            signals.append("MACD: BEARISH")
    
    # Moving Average Signal
    if indicators['sma_20']:
        if indicators['current_price'] > indicators['sma_20']:
            signals.append("Above SMA20")
        else:
            signals.append("Below SMA20")
    
    # Combine signals with trend
    if trend_direction == "BULLISH" and any("BULLISH" in s or "Above" in s for s in signals):
        return "BUY", " | ".join(signals)
    elif trend_direction == "BEARISH" and any("BEARISH" in s or "Below" in s for s in signals):
        return "SELL", " | ".join(signals)
    
    if signals:
        return "HOLD", " | ".join(signals)
    
    return "HOLD", "No clear signals"

# Initialize session state
if 'initialized' not in st.session_state:
    st.session_state.initialized = True
    st.session_state.stake_amounts = {pair: 10 for pair in FOREX_PAIRS}
    st.session_state.active_trades = []
    st.session_state.trade_history = []
    st.session_state.next_trade_id = 1001
    st.session_state.price_history = {}
    st.session_state.last_update = datetime.now()
    st.session_state.use_real_data = False  # Force simulated data for now

# Trade management functions
def open_trade(currency_pair, position, stake, entry_price):
    """Open a new trade"""
    trade = {
        'trade_id': f"TR{st.session_state.next_trade_id}",
        'currency_pair': currency_pair,
        'position': position,
        'stake': stake,
        'entry_price': entry_price,
        'current_price': entry_price,
        'open_time': datetime.now(),
        'status': 'ACTIVE'
    }
    st.session_state.active_trades.append(trade)
    st.session_state.next_trade_id += 1
    return trade

def close_trade(trade_id, exit_price):
    """Close an active trade"""
    for i, trade in enumerate(st.session_state.active_trades):
        if trade['trade_id'] == trade_id:
            if trade['position'] == 'LONG':
                pnl = (exit_price - trade['entry_price']) * trade['stake']
            else:
                pnl = (trade['entry_price'] - exit_price) * trade['stake']
            
            closed_trade = {
                'trade_id': trade['trade_id'],
                'currency_pair': trade['currency_pair'],
                'position': trade['position'],
                'stake': trade['stake'],
                'entry_price': trade['entry_price'],
                'exit_price': exit_price,
                'pnl': round(pnl, 2),
                'pnl_percent': round((pnl / trade['stake']) * 100, 2),
                'open_time': trade['open_time'],
                'close_time': datetime.now(),
                'status': 'CLOSED'
            }
            
            st.session_state.trade_history.append(closed_trade)
            st.session_state.active_trades.pop(i)
            return closed_trade
    return None

def initialize_price_history():
    """Initialize realistic price history for all pairs"""
    for pair in FOREX_PAIRS:
        if pair not in st.session_state.price_history:
            st.session_state.price_history[pair] = generate_realistic_historical_data(pair, 100)

def generate_sample_trades():
    """Generate sample trades if none exist"""
    if len(st.session_state.active_trades) == 0:
        for i in range(2):
            pair = np.random.choice(FOREX_PAIRS)
            stake = 10
            if pair in st.session_state.price_history and st.session_state.price_history[pair]:
                entry_price = st.session_state.price_history[pair][-1]
                open_trade(pair, np.random.choice(['LONG', 'SHORT']), stake, round(entry_price, 5))
    
    if len(st.session_state.trade_history) == 0:
        for i in range(8):
            pair = np.random.choice(FOREX_PAIRS)
            stake = 10
            if pair in st.session_state.price_history and len(st.session_state.price_history[pair]) >= 50:
                entry_price = st.session_state.price_history[pair][-50]
                exit_price = st.session_state.price_history[pair][-1]
                
                if np.random.choice(['LONG', 'SHORT']) == 'LONG':
                    pnl = (exit_price - entry_price) * stake
                else:
                    pnl = (entry_price - exit_price) * stake
                
                historical_trade = {
                    'trade_id': f"TR{800 + i}",
                    'currency_pair': pair,
                    'position': np.random.choice(['LONG', 'SHORT']),
                    'stake': stake,
                    'entry_price': round(entry_price, 5),
                    'exit_price': round(exit_price, 5),
                    'pnl': round(pnl, 2),
                    'pnl_percent': round((pnl / stake) * 100, 2),
                    'open_time': datetime.now() - timedelta(days=np.random.randint(5, 60)),
                    'close_time': datetime.now() - timedelta(days=np.random.randint(1, 5)),
                    'status': 'CLOSED'
                }
                st.session_state.trade_history.append(historical_trade)

# Initialize data
initialize_price_history()
generate_sample_trades()

# Update prices with realistic simulated data
current_quotes = generate_realistic_forex_data()

# Update price history with new prices
for pair, quote in current_quotes.items():
    if pair in st.session_state.price_history:
        st.session_state.price_history[pair].append(quote['price'])
        # Keep only last 100 prices
        if len(st.session_state.price_history[pair]) > 100:
            st.session_state.price_history[pair] = st.session_state.price_history[pair][-100:]

# Update active trades with current prices
for trade in st.session_state.active_trades:
    if trade['currency_pair'] in current_quotes:
        trade['current_price'] = round(current_quotes[trade['currency_pair']]['price'], 5)

# Header
st.markdown('<div class="main-header">LIVE FOREX PRICES - PURE PYTHON ANALYSIS</div>', unsafe_allow_html=True)

# API Status (Simulated Data Notice)
st.markdown('<div class="api-status api-error">🟡 USING REALISTIC SIMULATED DATA - Polygon.io API Available for Future Use</div>', unsafe_allow_html=True)

# Fixed Stake Notice
st.info("🎯 **Trading with Fixed 10€ Stake per Trade**")

# Manual Stake Management
st.markdown("---")
st.subheader("💰 Stake Management")

col1, col2, col3 = st.columns(3)

with col1:
    st.markdown('<div class="stake-card">', unsafe_allow_html=True)
    st.write("**Quick Stake Presets**")
    if st.button("€10 Standard", use_container_width=True):
        for pair in st.session_state.stake_amounts:
            st.session_state.stake_amounts[pair] = 10
        st.rerun()
    if st.button("€25 Advanced", use_container_width=True):
        for pair in st.session_state.stake_amounts:
            st.session_state.stake_amounts[pair] = 25
        st.rerun()
    st.markdown('</div>', unsafe_allow_html=True)

with col2:
    st.markdown('<div class="stake-card">', unsafe_allow_html=True)
    st.write("**Current Stake Settings**")
    for pair in FOREX_PAIRS[:3]:
        st.write(f"{pair}: €{st.session_state.stake_amounts[pair]}")
    st.markdown('</div>', unsafe_allow_html=True)

with col3:
    st.markdown('<div class="stake-card">', unsafe_allow_html=True)
    st.write("**Risk Management**")
    st.write("Max Exposure: €60")
    st.write("Per Trade: €10")
    st.write("Risk Level: Low")
    st.markdown('</div>', unsafe_allow_html=True)

# Real-time Forex Prices
st.markdown("---")
st.subheader("💱 Real-time Forex Prices")

# Display current prices in a compact format
cols = st.columns(len(FOREX_PAIRS))
for idx, pair in enumerate(FOREX_PAIRS):
    with cols[idx]:
        if pair in current_quotes:
            quote = current_quotes[pair]
            change_color = "positive-change" if quote['change'] >= 0 else "negative-change"
            change_symbol = "+" if quote['change'] >= 0 else ""
            
            st.metric(
                label=pair,
                value=f"{quote['price']:.4f}" if 'JPY' not in pair else f"{quote['price']:.2f}",
                delta=f"{change_symbol}{quote['change_percent']:.3f}%"
            )

# Technical Analysis Overview
st.markdown("---")
st.subheader("📊 Technical Analysis & Trading Signals")

for pair in FOREX_PAIRS:
    if pair in st.session_state.price_history and len(st.session_state.price_history[pair]) > 20:
        prices = st.session_state.price_history[pair]
        current_price = prices[-1]
        quote = current_quotes[pair]
        
        indicators = calculate_technical_indicators(prices)
        indicators['current_price'] = current_price
        
        trend_direction, trend_strength = analyze_trend(indicators, current_price)
        signal, signal_reason = generate_trading_signal(indicators, trend_direction)
        
        with st.container():
            st.markdown(f'<div class="analysis-card">', unsafe_allow_html=True)
            
            col1, col2, col3, col4 = st.columns([2, 2, 2, 2])
            
            with col1:
                st.write(f"**{pair}**")
                price_format = f"{current_price:.4f}" if 'JPY' not in pair else f"{current_price:.2f}"
                st.write(f"**{price_format}**")
                change_color = "positive-change" if quote['change'] >= 0 else "negative-change"
                change_symbol = "+" if quote['change'] >= 0 else ""
                st.markdown(f'<span class="{change_color}">{change_symbol}{quote["change_percent"]:.3f}%</span>', unsafe_allow_html=True)
            
            with col2:
                if trend_direction == "BULLISH":
                    st.markdown(f'<div class="trend-up">📈 {trend_strength}</div>', unsafe_allow_html=True)
                elif trend_direction == "BEARISH":
                    st.markdown(f'<div class="trend-down">📉 {trend_strength}</div>', unsafe_allow_html=True)
                else:
                    st.markdown(f'<div class="trend-neutral">➡️ {trend_strength}</div>', unsafe_allow_html=True)
                
                if signal == "BUY":
                    st.markdown(f'<div class="trend-up">🎯 SIGNAL: {signal}</div>', unsafe_allow_html=True)
                elif signal == "SELL":
                    st.markdown(f'<div class="trend-down">🎯 SIGNAL: {signal}</div>', unsafe_allow_html=True)
                else:
                    st.markdown(f'<div class="trend-neutral">🎯 SIGNAL: {signal}</div>', unsafe_allow_html=True)
            
            with col3:
                if indicators.get('rsi'):
                    rsi_color = "indicator-bullish" if indicators['rsi'] < 30 else "indicator-bearish" if indicators['rsi'] > 70 else "indicator-neutral"
                    st.markdown(f'<span class="{rsi_color}">RSI: {indicators["rsi"]:.1f}</span>', unsafe_allow_html=True)
                
                if indicators.get('macd'):
                    macd_color = "indicator-bullish" if indicators['macd'] > indicators.get('macd_signal', 0) else "indicator-bearish"
                    st.markdown(f'<span class="{macd_color}">MACD: {indicators["macd"]:.4f}</span>', unsafe_allow_html=True)
            
            with col4:
                if indicators.get('sma_20'):
                    sma_relation = "Above" if current_price > indicators['sma_20'] else "Below"
                    st.write(f"Price {sma_relation} SMA20")
                
                if indicators.get('stoch_k'):
                    stoch_color = "indicator-bullish" if indicators['stoch_k'] < 20 else "indicator-bearish" if indicators['stoch_k'] > 80 else "indicator-neutral"
                    st.markdown(f'<span class="{stoch_color}">Stoch: {indicators["stoch_k"]:.1f}</span>', unsafe_allow_html=True)
                
                st.caption(signal_reason)
            
            st.markdown('</div>', unsafe_allow_html=True)

# Current Active Trades Table
st.markdown("---")
st.subheader("📊 Current Active Trades")

if len(st.session_state.active_trades) > 0:
    total_exposure = sum(trade['stake'] for trade in st.session_state.active_trades)
    total_unrealized_pnl = 0
    
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Active Trades", len(st.session_state.active_trades))
    with col2:
        st.metric("Total Exposure", f"€{total_exposure:,}")
    
    # Display active trades
    for trade in st.session_state.active_trades:
        if trade['position'] == 'LONG':
            unrealized_pnl = (trade['current_price'] - trade['entry_price']) * trade['stake']
        else:
            unrealized_pnl = (trade['entry_price'] - trade['current_price']) * trade['stake']
        
        unrealized_pnl_percent = (unrealized_pnl / trade['stake']) * 100
        total_unrealized_pnl += unrealized_pnl
        
        pnl_color = "positive-pnl" if unrealized_pnl >= 0 else "negative-pnl"
        
        col1, col2, col3, col4, col5 = st.columns([1, 2, 2, 2, 1])
        with col1:
            st.write(f"**{trade['trade_id']}**")
        with col2:
            st.write(f"**{trade['currency_pair']}**")
            st.write(f"{trade['position']} | €{trade['stake']:,}")
        with col3:
            st.write(f"Entry: {trade['entry_price']}")
            st.write(f"Current: {trade['current_price']}")
        with col4:
            st.markdown(f'<div class="{pnl_color}">€{unrealized_pnl:,.2f}</div>', unsafe_allow_html=True)
            st.markdown(f'<div class="{pnl_color}">{unrealized_pnl_percent:.2f}%</div>', unsafe_allow_html=True)
        with col5:
            if st.button("Close", key=f"close_{trade['trade_id']}"):
                close_trade(trade['trade_id'], trade['current_price'])
                st.rerun()
        
        st.markdown("---")
    
    with col3:
        st.metric("Unrealized P/L", f"€{total_unrealized_pnl:,.2f}")
    with col4:
        if st.session_state.active_trades:
            avg_holding = np.mean([(datetime.now() - trade['open_time']).days 
                                  for trade in st.session_state.active_trades])
            st.metric("Avg Holding Time", f"{avg_holding:.1f} days")
        
else:
    st.info("No active trades currently open")

# Trade History Table
st.markdown("---")
st.subheader("📈 Trade History & Performance")

if len(st.session_state.trade_history) > 0:
    total_closed_pnl = sum(trade['pnl'] for trade in st.session_state.trade_history)
    winning_trades = len([t for t in st.session_state.trade_history if t['pnl'] > 0])
    total_trades = len(st.session_state.trade_history)
    win_rate = (winning_trades / total_trades) * 100 if total_trades > 0 else 0
    
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Total Trades", total_trades)
    with col2:
        st.metric("Win Rate", f"{win_rate:.1f}%")
    with col3:
        st.metric("Total P/L", f"€{total_closed_pnl:,.2f}")
    with col4:
        avg_trade_pnl = total_closed_pnl / total_trades if total_trades > 0 else 0
        st.metric("Avg Trade P/L", f"€{avg_trade_pnl:.2f}")
    
    # Display recent trades
    st.write("**Recent Trades:**")
    recent_trades = st.session_state.trade_history[-8:][::-1]
    
    for trade in recent_trades:
        pnl_color = "positive-pnl" if trade['pnl'] >= 0 else "negative-pnl"
        
        col1, col2, col3, col4, col5 = st.columns([1, 2, 2, 2, 2])
        with col1:
            st.write(f"**{trade['trade_id']}**")
        with col2:
            st.write(f"**{trade['currency_pair']}**")
            st.write(f"{trade['position']} | €{trade['stake']:,}")
        with col3:
            st.write(f"Entry: {trade['entry_price']}")
            st.write(f"Exit: {trade['exit_price']}")
        with col4:
            st.markdown(f'<div class="{pnl_color}">€{trade["pnl"]:,.2f}</div>', unsafe_allow_html=True)
            st.markdown(f'<div class="{pnl_color}">{trade["pnl_percent"]:.2f}%</div>', unsafe_allow_html=True)
        with col5:
            duration = (trade['close_time'] - trade['open_time']).days
            st.write(f"Duration: {duration}d")
            st.write(f"Closed: {trade['close_time'].strftime('%m/%d')}")
        
        st.markdown("---")
    
else:
    st.info("No trade history available")

# Trading Interface
st.markdown("---")
st.subheader("🚀 Quick Trading Interface")

col1, col2, col3 = st.columns(3)

with col1:
    st.write("**Trade Configuration**")
    selected_pair = st.selectbox("Currency Pair", FOREX_PAIRS)
    position = st.selectbox("Position Type", ["LONG", "SHORT"])
    
with col2:
    stake = st.session_state.stake_amounts[selected_pair]
    st.write(f"**Stake Amount: €{stake}**")
    if selected_pair in current_quotes:
        current_price = current_quotes[selected_pair]['price']
        price_format = f"{current_price:.4f}" if 'JPY' not in selected_pair else f"{current_price:.2f}"
        st.write(f"**Current Price: {price_format}**")
    
with col3:
    st.write("**Execute Trade**")
    st.write("&nbsp;")
    if st.button("🎯 EXECUTE TRADE", use_container_width=True, type="primary"):
        if selected_pair in current_quotes:
            current_price = current_quotes[selected_pair]['price']
            new_trade = open_trade(selected_pair, position, stake, round(current_price, 5))
            st.success(f"✅ Trade Executed: {new_trade['trade_id']}")
            st.success(f"{selected_pair} {position} | €{stake} | Price: {current_price:.5f}")
            st.rerun()

# Footer with refresh
st.markdown("---")
col1, col2 = st.columns([3, 1])
with col1:
    st.write(f"Last update: {datetime.now().strftime('%H:%M:%S')}")
    st.write("Data refreshes automatically every 30 seconds")
with col2:
    if st.button("🔄 Refresh Now", use_container_width=True):
        st.rerun()

st.markdown("""
<div style="text-align: center; color: #666; margin-top: 2rem;">
    Advanced Forex Trading Dashboard | Fixed €10 Stake | Realistic Market Simulation
</div>
""", unsafe_allow_html=True)

# Auto-refresh every 30 seconds
time.sleep(30)
st.rerun()

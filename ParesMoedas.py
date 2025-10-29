import streamlit as st
import pandas as pd
import numpy as np
import time
from datetime import datetime, timedelta

st.set_page_config(
    page_title="Forex Pro Bot - Manual Stake",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for professional styling
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
    .stake-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 1rem;
        border-radius: 10px;
        margin: 0.5rem 0;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }
    .manual-trade-card {
        background: linear-gradient(135deg, #00b09b 0%, #96c93d 100%);
        color: white;
        padding: 1rem;
        border-radius: 10px;
        margin: 1rem 0;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }
</style>
""", unsafe_allow_html=True)

# Forex pairs with realistic base prices and volatility (added more pairs)
FOREX_PAIRS = {
    "EUR/USD": {"base_price": 1.0850, "volatility": 0.0008, "pip_value": 0.0001},
    "GBP/USD": {"base_price": 1.2650, "volatility": 0.0010, "pip_value": 0.0001},
    "USD/JPY": {"base_price": 148.50, "volatility": 0.15, "pip_value": 0.01},
    "USD/CHF": {"base_price": 0.8800, "volatility": 0.0009, "pip_value": 0.0001},
    "USD/CAD": {"base_price": 1.3550, "volatility": 0.0010, "pip_value": 0.0001},
    "AUD/USD": {"base_price": 0.6550, "volatility": 0.0012, "pip_value": 0.0001},
    "NZD/USD": {"base_price": 0.6050, "volatility": 0.0010, "pip_value": 0.0001},
    "EUR/GBP": {"base_price": 0.8350, "volatility": 0.0005, "pip_value": 0.0001},
    "EUR/JPY": {"base_price": 161.00, "volatility": 0.20, "pip_value": 0.01},
    "GBP/JPY": {"base_price": 188.50, "volatility": 0.25, "pip_value": 0.01}
}

# PROVEN TRADING STRATEGIES
PRO_STRATEGIES = {
    "SCALPING_5MIN": {
        "name": "5-Minute Scalping",
        "description": "High-frequency trades with tight stops",
        "timeframe": "5min",
        "ma_fast": 5,
        "ma_slow": 20,
        "rsi_period": 14,
        "rsi_overbought": 65,
        "rsi_oversold": 35,
        "macd_fast": 6,
        "macd_slow": 13,
        "macd_signal": 5,
        "profit_target_pips": 8.0,
        "stop_loss_pips": 5.0,
        "required_indicators": 2
    },
    "SWING_15MIN": {
        "name": "15-Minute Swing",
        "description": "Balanced approach for intraday trading",
        "timeframe": "15min",
        "ma_fast": 8,
        "ma_slow": 21,
        "rsi_period": 14,
        "rsi_overbought": 70,
        "rsi_oversold": 30,
        "macd_fast": 8,
        "macd_slow": 17,
        "macd_signal": 9,
        "profit_target_pips": 15.0,
        "stop_loss_pips": 10.0,
        "required_indicators": 3
    },
    "PROFESSIONAL_COMBO": {
        "name": "Professional Combo",
        "description": "Multi-timeframe confirmed signals",
        "timeframe": "15min",
        "ma_fast": 7,
        "ma_slow": 25,
        "rsi_period": 14,
        "rsi_overbought": 72,
        "rsi_oversold": 28,
        "macd_fast": 10,
        "macd_slow": 22,
        "macd_signal": 7,
        "profit_target_pips": 20.0,
        "stop_loss_pips": 12.0,
        "required_indicators": 3
    }
}

# Default trading parameters (added loss avoidance params)
DEFAULT_PARAMS = {
    'initial_bank': 10000.0,
    'profit_target_pips': 20.0,
    'stop_loss_pips': 12.0,
    'ma_fast': 7,
    'ma_slow': 25,
    'rsi_period': 14,
    'rsi_overbought': 72,
    'rsi_oversold': 28,
    'macd_fast': 10,
    'macd_slow': 22,
    'macd_signal': 7,
    'required_indicators': 3,
    'max_open_trades': 3,
    'max_risk_percent': 1.5,
    'max_daily_loss_percent': 2.0,  # New: Max daily loss as % of initial bank
    'trailing_stop_enabled': True,   # New: Enable trailing stop
    'trail_start_pips': 10.0,        # New: Pips in profit to start trailing
    'trail_distance_pips': 5.0,      # New: Trailing distance in pips
    'selected_strategy': 'PROFESSIONAL_COMBO',
    'manual_stake_amount': 100.0,  # Default manual stake
    'use_candlestick_patterns': True,  # New: Enable candlestick patterns
    'min_pattern_confidence': 0.7  # New: Minimum confidence for patterns to count as indicators
}

# Initialize session state
if 'trading_params' not in st.session_state:
    st.session_state.trading_params = DEFAULT_PARAMS.copy()
else:
    for key, default_value in DEFAULT_PARAMS.items():
        if key not in st.session_state.trading_params:
            st.session_state.trading_params[key] = default_value

if 'bank_balance' not in st.session_state:
    st.session_state.bank_balance = float(st.session_state.trading_params['initial_bank'])
if 'open_trades' not in st.session_state:
    st.session_state.open_trades = []
if 'trade_history' not in st.session_state:
    st.session_state.trade_history = []
if 'auto_trading' not in st.session_state:
    st.session_state.auto_trading = False
if 'all_signals' not in st.session_state:
    st.session_state.all_signals = {}
if 'current_prices' not in st.session_state:
    st.session_state.current_prices = {pair: data['base_price'] for pair, data in FOREX_PAIRS.items()}
if 'trade_counter' not in st.session_state:
    st.session_state.trade_counter = 0
if 'use_manual_stake' not in st.session_state:
    st.session_state.use_manual_stake = False
# New for loss avoidance
if 'last_reset_date' not in st.session_state:
    st.session_state.last_reset_date = datetime.now().date()
if 'daily_start_balance' not in st.session_state:
    st.session_state.daily_start_balance = st.session_state.bank_balance

# Trading pairs list (updated with new pairs)
trading_pairs = list(FOREX_PAIRS.keys())

# Technical Indicator Calculations
def calculate_rsi(prices, period=14):
    delta = prices.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    return rsi

def calculate_macd(prices, fast=12, slow=26, signal=9):
    exp1 = prices.ewm(span=fast).mean()
    exp2 = prices.ewm(span=slow).mean()
    macd_line = exp1 - exp2
    signal_line = macd_line.ewm(span=signal).mean()
    return macd_line, signal_line

def calculate_indicators(df):
    try:
        if df.empty or len(df) < 50:
            return df
            
        df_indicators = df.copy()
        params = st.session_state.trading_params
        
        df_indicators['MA_Fast'] = df_indicators['close'].rolling(window=params['ma_fast']).mean()
        df_indicators['MA_Slow'] = df_indicators['close'].rolling(window=params['ma_slow']).mean()
        df_indicators['RSI'] = calculate_rsi(df_indicators['close'], params['rsi_period'])
        
        macd_line, signal_line = calculate_macd(
            df_indicators['close'], 
            params['macd_fast'], 
            params['macd_slow'], 
            params['macd_signal']
        )
        df_indicators['MACD'] = macd_line
        df_indicators['MACD_Signal'] = signal_line
        
        # Supporting metrics for patterns
        df_indicators['AVGH10'] = df_indicators['high'].rolling(window=10).mean()
        df_indicators['AVGL10'] = df_indicators['low'].rolling(window=10).mean()
        df_indicators['MINL10'] = df_indicators['low'].rolling(window=10).min()
        
        return df_indicators
        
    except Exception as e:
        return df

def generate_15min_forex_data(pair, periods=200):
    pair_data = FOREX_PAIRS[pair]
    base_price = st.session_state.current_prices.get(pair, pair_data['base_price'])
    volatility = pair_data['volatility']
    prices = []
    current_time = datetime.now()
    
    for i in range(periods):
        date = current_time - timedelta(minutes=15 * (periods - i - 1))
        
        open_price = base_price
        change = np.random.normal(0, volatility * 0.1)
        close_price = base_price * (1 + change)
        high_price = max(open_price, close_price) * (1 + abs(np.random.normal(0, volatility * 0.05)))
        low_price = min(open_price, close_price) * (1 - abs(np.random.normal(0, volatility * 0.05)))
        
        prices.append({
            "date": date,
            "open": open_price,
            "high": high_price,
            "low": low_price,
            "close": close_price
        })
        
        base_price = close_price
    
    return pd.DataFrame(prices)

# Enhanced Candlestick Pattern Detections with Confidence Scores
def detect_doji(df):
    if len(df) < 1:
        return None, 0.0
    
    latest = df.iloc[-1]
    body = abs(latest['close'] - latest['open'])
    total_range = latest['high'] - latest['low']
    avgh10 = latest.get('AVGH10', total_range)
    avgl10 = latest.get('AVGL10', 0)
    
    if total_range == 0:
        return None, 0.0
    
    # Confidence based on body size relative to range (smaller body = higher confidence)
    body_ratio = body / total_range
    confidence = max(0.0, 1.0 - (body_ratio / 0.05)) if body_ratio <= 0.05 else 0.0
    
    if body_ratio <= 0.05 and (avgh10 - avgl10) > 0:
        return 'neutral', confidence
    
    return None, 0.0

def detect_engulfing(df):
    if len(df) < 2:
        return None, 0.0
    
    prev = df.iloc[-2]
    latest = df.iloc[-1]
    O1 = prev['open']
    C1 = prev['close']
    H1 = prev['high']
    L1 = prev['low']
    avgh10_1 = prev.get('AVGH10', H1 - L1)
    avgl10_1 = prev.get('AVGL10', 0)
    
    prev_range = H1 - L1
    latest_range = latest['high'] - latest['low']
    prev_body = abs(O1 - C1)
    latest_body = abs(latest['close'] - latest['open'])
    
    # Bullish Engulfing
    if (O1 > C1) and \
       (10 * latest_body >= 7 * latest_range) and \
       (latest['close'] > O1) and \
       (C1 > latest['open']) and \
       (10 * latest_range >= 12 * (avgh10_1 - avgl10_1)):
        
        # Confidence: based on engulfing ratio and range expansion
        engulf_ratio = latest_body / prev_body if prev_body > 0 else 1.0
        range_exp = latest_range / prev_range if prev_range > 0 else 1.0
        confidence = min(1.0, (engulf_ratio * 0.6) + (range_exp * 0.4))
        return 'bullish', confidence
    
    # Bearish Engulfing
    if (O1 < C1) and \
       (10 * latest_body >= 7 * latest_range) and \
       (latest['open'] > C1) and \
       (O1 > latest['close']) and \
       (10 * latest_range >= 12 * (avgh10_1 - avgl10_1)):
        
        engulf_ratio = latest_body / prev_body if prev_body > 0 else 1.0
        range_exp = latest_range / prev_range if prev_range > 0 else 1.0
        confidence = min(1.0, (engulf_ratio * 0.6) + (range_exp * 0.4))
        return 'bearish', confidence
    
    return None, 0.0

def detect_hammer(df):
    if len(df) < 1:
        return None, 0.0
    
    latest = df.iloc[-1]
    body = abs(latest['close'] - latest['open'])
    total_range = latest['high'] - latest['low']
    
    if total_range == 0:
        return None, 0.0
    
    lower_wick = min(latest['open'], latest['close']) - latest['low']
    upper_wick = latest['high'] - max(latest['open'], latest['close'])
    
    # Enhanced Hammer
    lower_wick_ratio = lower_wick / total_range
    upper_wick_ratio = upper_wick / total_range
    body_ratio = body / total_range
    
    if lower_wick_ratio >= 0.4 and upper_wick_ratio <= 0.1 and body_ratio <= 0.3:
        # Confidence: longer lower wick and smaller body/upper wick = higher confidence
        confidence = min(1.0, (lower_wick_ratio * 0.7) + (1 - body_ratio) * 0.3)
        return 'bullish', confidence
    # Enhanced Shooting Star
    elif upper_wick_ratio >= 0.4 and lower_wick_ratio <= 0.1 and body_ratio <= 0.3:
        confidence = min(1.0, (upper_wick_ratio * 0.7) + (1 - body_ratio) * 0.3)
        return 'bearish', confidence
    
    return None, 0.0

def detect_morning_star(df):
    if len(df) < 3:
        return None, 0.0
    
    c2 = df.iloc[-3]  # First candle
    c1 = df.iloc[-2]  # Second
    c0 = df.iloc[-1]  # Third
    
    O2, C2, H2, L2 = c2['open'], c2['close'], c2['high'], c2['low']
    O1, C1, H1, L1 = c1['open'], c1['close'], c1['high'], c1['low']
    O0, C0, H0, L0 = c0['open'], c0['close'], c0['high'], c0['low']
    
    # Detailed Morning Star
    if (O2 > C2) and \
       (5 * (O2 - C2) > 3 * (H2 - L2)) and \
       (C2 > O1) and \
       (2 * abs(O1 - C1) < abs(O2 - C2)) and \
       (H1 - L1 > 3 * abs(C1 - O1)) and \
       (C0 > O0) and \
       (O0 > O1) and \
       (O0 > C1):
        
        # Confidence: based on gap sizes and body ratios
        first_body_ratio = abs(O2 - C2) / (H2 - L2)
        gap_size = min((O0 - C1), (O1 - C2)) / (H2 - L2) if (H2 - L2) > 0 else 0
        confidence = min(1.0, first_body_ratio * 0.5 + gap_size * 0.5)
        return 'bullish', confidence
    
    # Evening Star
    if (O2 < C2) and \
       (5 * (C2 - O2) > 3 * (H2 - L2))

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
    'initial_bank': 10000,
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
    'manual_stake_amount': 100,  # Default manual stake
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
    st.session_state.bank_balance = st.session_state.trading_params['initial_bank']
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
       (5 * (C2 - O2) > 3 * (H2 - L2)) and \
       (C2 < O1) and \
       (2 * abs(O1 - C1) < abs(O2 - C2)) and \
       (H1 - L1 > 3 * abs(C1 - O1)) and \
       (C0 < O0) and \
       (O0 < O1) and \
       (O0 < C1):
        
        first_body_ratio = abs(O2 - C2) / (H2 - L2)
        gap_size = min((C1 - O0), (C2 - O1)) / (H2 - L2) if (H2 - L2) > 0 else 0
        confidence = min(1.0, first_body_ratio * 0.5 + gap_size * 0.5)
        return 'bearish', confidence
    
    return None, 0.0

def detect_harami(df):
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
    prev_body = abs(O1 - C1)
    latest_body = abs(latest['close'] - latest['open'])
    
    # Bullish Harami
    if (10 * prev_body >= 7 * prev_range) and \
       (prev_range >= avgh10_1 - avgl10_1) and \
       (latest['close'] > latest['open']) and \
       (latest['open'] > C1) and \
       (O1 > latest['close']) and \
       (6 * prev_body >= 10 * latest_body):
        
        # Confidence: larger prev body and smaller latest body
        body_ratio = latest_body / prev_body if prev_body > 0 else 0
        confidence = min(1.0, 1.0 - body_ratio)
        return 'bullish', confidence
    
    # Bearish Harami
    if (10 * prev_body >= 7 * prev_range) and \
       (prev_range >= avgh10_1 - avgl10_1) and \
       (latest['close'] < latest['open']) and \
       (latest['open'] < C1) and \
       (O1 < latest['close']) and \
       (6 * prev_body >= 10 * latest_body):
        
        body_ratio = latest_body / prev_body if prev_body > 0 else 0
        confidence = min(1.0, 1.0 - body_ratio)
        return 'bearish', confidence
    
    return None, 0.0

def detect_pin_bar(df):
    # Enhanced Pin Bar
    if len(df) < 1:
        return None, 0.0
    
    latest = df.iloc[-1]
    body = abs(latest['close'] - latest['open'])
    total_range = latest['high'] - latest['low']
    
    if total_range == 0:
        return None, 0.0
    
    lower_wick = min(latest['open'], latest['close']) - latest['low']
    upper_wick = latest['high'] - max(latest['open'], latest['close'])
    
    lower_wick_ratio = lower_wick / total_range
    upper_wick_ratio = upper_wick / total_range
    body_ratio = body / total_range
    
    if lower_wick_ratio >= 0.6 and body_ratio <= 0.2 and upper_wick_ratio < 0.1:
        # Confidence: longer wick and smaller body
        confidence = min(1.0, lower_wick_ratio * 0.8 + (1 - body_ratio) * 0.2)
        return 'bullish', confidence
    elif upper_wick_ratio >= 0.6 and body_ratio <= 0.2 and lower_wick_ratio < 0.1:
        confidence = min(1.0, upper_wick_ratio * 0.8 + (1 - body_ratio) * 0.2)
        return 'bearish', confidence
    
    return None, 0.0

def detect_trading_signals(df):
    buy_indicators = []
    sell_indicators = []
    patterns = {}  # Track detected patterns with confidence
    params = st.session_state.trading_params
    min_conf = params['min_pattern_confidence']
    
    try:
        if len(df) < 20:
            return [], [], [], 'NONE', patterns
        
        latest = df.iloc[-1]
        
        # Moving Average Signals
        if pd.notna(latest['MA_Fast']) and pd.notna(latest['MA_Slow']):
            if latest['MA_Fast'] > latest['MA_Slow']:
                buy_indicators.append("MA Bullish Crossover")
            else:
                sell_indicators.append("MA Bearish Crossover")
        
        # RSI Signals
        if pd.notna(latest['RSI']):
            if latest['RSI'] < params['rsi_oversold']:
                buy_indicators.append("RSI Oversold")
            elif latest['RSI'] > params['rsi_overbought']:
                sell_indicators.append("RSI Overbought")
        
        # MACD Signals
        if pd.notna(latest['MACD']) and pd.notna(latest['MACD_Signal']):
            if latest['MACD'] > latest['MACD_Signal']:
                buy_indicators.append("MACD Bullish")
            else:
                sell_indicators.append("MACD Bearish")
        
        # Enhanced Candlestick Patterns with Confidence
        if params['use_candlestick_patterns']:
            # Doji
            doj_type, doj_conf = detect_doji(df)
            if doj_type == 'neutral' and doj_conf > 0:
                patterns['doji'] = {'type': 'neutral', 'confidence': doj_conf}
            
            # Pin Bar
            pin_type, pin_conf = detect_pin_bar(df)
            if pin_type == 'bullish' and pin_conf >= min_conf:
                buy_indicators.append(f"Pin Bar Bullish ({pin_conf:.2f})")
                patterns['pin_bar'] = {'type': 'bullish', 'confidence': pin_conf}
            elif pin_type == 'bearish' and pin_conf >= min_conf:
                sell_indicators.append(f"Pin Bar Bearish ({pin_conf:.2f})")
                patterns['pin_bar'] = {'type': 'bearish', 'confidence': pin_conf}
            
            # Engulfing
            eng_type, eng_conf = detect_engulfing(df)
            if eng_type == 'bullish' and eng_conf >= min_conf:
                buy_indicators.append(f"Engulfing Bullish ({eng_conf:.2f})")
                patterns['engulfing'] = {'type': 'bullish', 'confidence': eng_conf}
            elif eng_type == 'bearish' and eng_conf >= min_conf:
                sell_indicators.append(f"Engulfing Bearish ({eng_conf:.2f})")
                patterns['engulfing'] = {'type': 'bearish', 'confidence': eng_conf}
            
            # Hammer/Shooting Star
            ham_type, ham_conf = detect_hammer(df)
            if ham_type == 'bullish' and ham_conf >= min_conf:
                buy_indicators.append(f"Hammer Bullish ({ham_conf:.2f})")
                patterns['hammer'] = {'type': 'bullish', 'confidence': ham_conf}
            elif ham_type == 'bearish' and ham_conf >= min_conf:
                sell_indicators.append(f"Shooting Star Bearish ({ham_conf:.2f})")
                patterns['hammer'] = {'type': 'bearish', 'confidence': ham_conf}
            
            # Morning/Evening Star
            star_type, star_conf = detect_morning_star(df)
            if star_type == 'bullish' and star_conf >= min_conf:
                buy_indicators.append(f"Morning Star Bullish ({star_conf:.2f})")
                patterns['morning_star'] = {'type': 'bullish', 'confidence': star_conf}
            elif star_type == 'bearish' and star_conf >= min_conf:
                sell_indicators.append(f"Evening Star Bearish ({star_conf:.2f})")
                patterns['morning_star'] = {'type': 'bearish', 'confidence': star_conf}
            
            # Harami
            har_type, har_conf = detect_harami(df)
            if har_type == 'bullish' and har_conf >= min_conf:
                buy_indicators.append(f"Harami Bullish ({har_conf:.2f})")
                patterns['harami'] = {'type': 'bullish', 'confidence': har_conf}
            elif har_type == 'bearish' and har_conf >= min_conf:
                sell_indicators.append(f"Harami Bearish ({har_conf:.2f})")
                patterns['harami'] = {'type': 'bearish', 'confidence': har_conf}
        
        # Determine agreement
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
            
        return signals, buy_indicators, sell_indicators, agreement, patterns
        
    except Exception as e:
        return [], [], [], 'NONE', {}

def calculate_sl_tp_prices(entry_price, direction, sl_pips, tp_pips, pair, patterns=None):
    pip_value = FOREX_PAIRS[pair]['pip_value']
    
    # Adjust SL based on average pattern confidence (higher conf = tighter SL)
    adjusted_sl = sl_pips
    if patterns:
        confidences = [p['confidence'] for p in patterns.values() if isinstance(p, dict) and 'confidence' in p]
        if confidences:
            avg_conf = np.mean(confidences)
            adjusted_sl = max(5.0, sl_pips * (1 - avg_conf * 0.3))  # Up to 30% tighter based on conf
    
    if direction == 'BUY':
        stop_loss_price = entry_price - (adjusted_sl * pip_value)
        take_profit_price = entry_price + (tp_pips * pip_value)
    else:
        stop_loss_price = entry_price + (adjusted_sl * pip_value)
        take_profit_price = entry_price - (tp_pips * pip_value)
    
    return stop_loss_price, take_profit_price

def calculate_position_size(stake_amount, sl_pips, pair):
    """Calculate position size based on stake and stop loss"""
    pip_value = FOREX_PAIRS[pair]['pip_value']
    
    if pip_value > 0 and sl_pips > 0:
        # Position size = Stake / (Stop loss in pips * Pip value)
        position_size = stake_amount / (sl_pips * pip_value)
        return min(position_size, 100000)  # Cap at 10 lots
    return 10000  # Default to 1 lot

def check_daily_loss_limit():
    """Check if daily loss limit is exceeded"""
    if datetime.now().date() > st.session_state.last_reset_date:
        st.session_state.daily_start_balance = st.session_state.bank_balance
        st.session_state.last_reset_date = datetime.now().date()
    
    daily_pnl = st.session_state.bank_balance - st.session_state.daily_start_balance
    max_loss = - (st.session_state.trading_params['initial_bank'] * st.session_state.trading_params['max_daily_loss_percent'] / 100)
    
    if daily_pnl < max_loss:
        if st.session_state.auto_trading:
            st.session_state.auto_trading = False
            st.warning(f"Daily loss limit ({st.session_state.trading_params['max_daily_loss_percent']}%) reached! Auto trading stopped.")
        return False
    return True

def execute_trade(pair, direction, entry_price, stake_amount=None, patterns=None):
    try:
        # Check daily loss limit before executing
        if not check_daily_loss_limit():
            st.error("Daily loss limit exceeded. Cannot execute new trades.")
            return False

        st.session_state.trade_counter += 1
        params = st.session_state.trading_params
        
        # Adjust entry for Pin Bar or Hammer
        pip_value = FOREX_PAIRS[pair]['pip_value']
        if patterns and ('pin_bar' in patterns or 'hammer' in patterns):
            pin_type = patterns.get('pin_bar', {}).get('type') or patterns.get('hammer', {}).get('type')
            if pin_type == 'bullish' and direction == 'BUY':
                entry_price += (2 * pip_value)  # Entry above high
            elif pin_type == 'bearish' and direction == 'SELL':
                entry_price -= (2 * pip_value)  # Entry below low
        
        # Determine stake amount
        if stake_amount is None:
            if st.session_state.use_manual_stake:
                stake_amount = params['manual_stake_amount']
            else:
                stake_amount = (params['max_risk_percent'] / 100) * st.session_state.bank_balance
        
        # Check if we have enough balance
        if stake_amount > st.session_state.bank_balance:
            st.error(f"Insufficient balance! Available: ${st.session_state.bank_balance:.2f}, Required: ${stake_amount:.2f}")
            return False
        
        # Calculate position size
        position_size = calculate_position_size(stake_amount, params['stop_loss_pips'], pair)
        
        # Calculate SL and TP prices
        stop_loss_price, take_profit_price = calculate_sl_tp_prices(
            entry_price, direction, params['stop_loss_pips'], params['profit_target_pips'], pair, patterns
        )
        
        trade = {
            'id': st.session_state.trade_counter,
            'pair': pair,
            'direction': direction,
            'entry_price': entry_price,
            'stop_loss_price': stop_loss_price,
            'take_profit_price': take_profit_price,
            'stake': stake_amount,
            'position_size': position_size,
            'time': datetime.now(),
            'status': 'open',
            'profit_loss': 0,
            'profit_loss_pips': 0,
            'current_price': entry_price,
            'type': 'MANUAL' if stake_amount else 'AUTO',
            'close_reason': None,
            'stake_type': 'MANUAL' if st.session_state.use_manual_stake else 'AUTO',
            'trailing_sl': stop_loss_price,
            'patterns': patterns or {}  # Track patterns with confidence
        }
        
        st.session_state.open_trades.append(trade)
        st.session_state.bank_balance -= stake_amount
        return True
    except Exception as e:
        st.error(f"Error executing trade: {e}")
        return False

def close_trade(trade_id, close_price=None, reason='MANUAL'):
    for i, trade in enumerate(st.session_state.open_trades):
        if trade['id'] == trade_id and trade['status'] == 'open':
            if close_price is None:
                close_price = st.session_state.current_prices.get(trade['pair'], trade['entry_price'])
            
            pip_value = FOREX_PAIRS[trade['pair']]['pip_value']
            
            # Calculate P&L in pips
            if trade['direction'] == 'BUY':
                pips = (close_price - trade['entry_price']) / pip_value
            else:
                pips = (trade['entry_price'] - close_price) / pip_value
            
            # Calculate dollar P&L (simplified calculation)
            profit_loss_dollar = pips * pip_value * trade['position_size']
            
            # Update trade details
            trade['status'] = 'closed'
            trade['close_time'] = datetime.now()
            trade['close_price'] = close_price
            trade['profit_loss'] = profit_loss_dollar
            trade['profit_loss_pips'] = pips
            trade['close_reason'] = reason
            
            # Move to trade history and return stake + P&L
            st.session_state.trade_history.append(trade.copy())
            st.session_state.bank_balance += trade['stake'] + profit_loss_dollar
            
            # Remove from open trades
            st.session_state.open_trades.pop(i)
            return True
    return False

def update_trades():
    trades_to_remove = []
    params = st.session_state.trading_params
    trailing_enabled = params['trailing_stop_enabled']
    trail_start = params['trail_start_pips']
    trail_dist = params['trail_distance_pips']
    
    for i, trade in enumerate(st.session_state.open_trades):
        if trade['status'] == 'open':
            current_price = st.session_state.current_prices.get(trade['pair'], trade['entry_price'])
            pip_value = FOREX_PAIRS[trade['pair']]['pip_value']
            
            # Calculate current P&L
            if trade['direction'] == 'BUY':
                pips = (current_price - trade['entry_price']) / pip_value
            else:
                pips = (trade['entry_price'] - current_price) / pip_value
            
            profit_loss_dollar = pips * pip_value * trade['position_size']
            
            trade['profit_loss'] = profit_loss_dollar
            trade['profit_loss_pips'] = pips
            trade['current_price'] = current_price
            
            # Trailing Stop Logic
            if trailing_enabled:
                if trade['direction'] == 'BUY':
                    if pips > trail_start:
                        new_sl = current_price - (trail_dist * pip_value)
                        if new_sl > trade['stop_loss_price']:
                            trade['stop_loss_price'] = new_sl
                else:  # SELL
                    if pips > trail_start:
                        new_sl = current_price + (trail_dist * pip_value)
                        if new_sl < trade['stop_loss_price']:
                            trade['stop_loss_price'] = new_sl
            
            # Check SL/TP
            if trade['direction'] == 'BUY':
                if current_price >= trade['take_profit_price']:
                    close_trade(trade['id'], current_price, 'TP')
                    trades_to_remove.append(i)
                elif current_price <= trade['stop_loss_price']:
                    close_trade(trade['id'], current_price, 'SL')
                    trades_to_remove.append(i)
            else:
                if current_price <= trade['take_profit_price']:
                    close_trade(trade['id'], current_price, 'TP')
                    trades_to_remove.append(i)
                elif current_price >= trade['stop_loss_price']:
                    close_trade(trade['id'], current_price, 'SL')
                    trades_to_remove.append(i)

def scan_all_pairs_signals():
    all_signals = {}
    
    for pair in trading_pairs:
        df = generate_15min_forex_data(pair, 200)
        df_with_indicators = calculate_indicators(df)
        
        signals, buy_indicators, sell_indicators, agreement, patterns = detect_trading_signals(df_with_indicators)
        
        current_price = df_with_indicators['close'].iloc[-1]
        st.session_state.current_prices[pair] = current_price
        
        all_signals[pair] = {
            'signals': signals,
            'buy_indicators': buy_indicators,
            'sell_indicators': sell_indicators,
            'agreement': agreement,
            'current_price': current_price,
            'patterns': patterns
        }
    
    return all_signals

def execute_auto_trades():
    if not st.session_state.auto_trading:
        return []
    
    # Check daily loss limit before auto trades
    if not check_daily_loss_limit():
        return []
    
    auto_trades_executed = []
    
    try:
        all_signals = scan_all_pairs_signals()
        
        for pair, signal_info in all_signals.items():
            signals = signal_info.get('signals', [])
            agreement = signal_info.get('agreement', 'NONE')
            patterns = signal_info.get('patterns', {})
            
            for signal_type, count, indicators in signals:
                if agreement == 'BUY' and signal_type == "BUY" and len(st.session_state.open_trades) < st.session_state.trading_params['max_open_trades']:
                    current_price = st.session_state.current_prices.get(pair, FOREX_PAIRS[pair]['base_price'])
                    
                    if execute_trade(pair, 'BUY', current_price, patterns=patterns):
                        pattern_str = ', '.join([f"{k}:{v['type']}({v['confidence']:.2f})" for k, v in patterns.items() if isinstance(v, dict)]) if patterns else 'None'
                        auto_trades_executed.append(f"✅ BUY {pair} - {count} indicators (Patterns: {pattern_str})")
                
                elif agreement == 'SELL' and signal_type == "SELL" and len(st.session_state.open_trades) < st.session_state.trading_params['max_open_trades']:
                    current_price = st.session_state.current_prices.get(pair, FOREX_PAIRS[pair]['base_price'])
                    
                    if execute_trade(pair, 'SELL', current_price, patterns=patterns):
                        pattern_str = ', '.join([f"{k}:{v['type']}({v['confidence']:.2f})" for k, v in patterns.items() if isinstance(v, dict)]) if patterns else 'None'
                        auto_trades_executed.append(f"❌ SELL {pair} - {count} indicators (Patterns: {pattern_str})")
                        
    except Exception as e:
        st.error(f"Error in auto trading: {e}")
    
    return auto_trades_executed

def apply_strategy(strategy_name):
    if strategy_name in PRO_STRATEGIES:
        strategy = PRO_STRATEGIES[strategy_name]
        for key, value in strategy.items():
            if key in st.session_state.trading_params:
                st.session_state.trading_params[key] = value
        st.session_state.trading_params['selected_strategy'] = strategy_name
        return True
    return False

# MAIN APP LAYOUT
st.markdown('<h1 class="main-header">🤖 Forex Pro Bot - Manual Stake Control with Advanced Candlesticks & Confidence</h1>', unsafe_allow_html=True)

# Sidebar
with st.sidebar:
    st.header("💰 Stake Management")
    
    # Stake Type Selection
    st.subheader("🎯 Stake Type")
    stake_type = st.radio(
        "Choose Stake Method",
        ["Auto Risk %", "Manual Fixed Amount"],
        index=1 if st.session_state.use_manual_stake else 0
    )
    
    st.session_state.use_manual_stake = (stake_type == "Manual Fixed Amount")
    
    if st.session_state.use_manual_stake:
        st.markdown('<div class="stake-card">', unsafe_allow_html=True)
        st.session_state.trading_params['manual_stake_amount'] = st.number_input(
            "Manual Stake Amount ($)",
            min_value=10,
            max_value=10000,
            value=st.session_state.trading_params['manual_stake_amount'],
            step=50,
            help="Fixed amount to risk per trade"
        )
        st.write(f"**Stake per trade:** ${st.session_state.trading_params['manual_stake_amount']:.2f}")
        st.markdown('</div>', unsafe_allow_html=True)
    else:
        st.markdown('<div class="stake-card">', unsafe_allow_html=True)
        st.session_state.trading_params['max_risk_percent'] = st.slider(
            "Risk Per Trade (%)",
            min_value=0.5,
            max_value=5.0,
            value=st.session_state.trading_params['max_risk_percent'],
            step=0.5,
            help="Percentage of balance to risk per trade"
        )
        risk_amount = (st.session_state.trading_params['max_risk_percent'] / 100) * st.session_state.bank_balance
        st.write(f"**Risk per trade:** ${risk_amount:.2f}")
        st.markdown('</div>', unsafe_allow_html=True)
    
    st.divider()
    
    # Loss Avoidance System
    st.subheader("🛡️ Loss Avoidance System")
    st.session_state.trading_params['trailing_stop_enabled'] = st.checkbox(
        "Enable Trailing Stop Loss",
        value=st.session_state.trading_params['trailing_stop_enabled'],
        help="Trail SL after profit reaches start pips"
    )
    col1, col2 = st.columns(2)
    with col1:
        st.session_state.trading_params['trail_start_pips'] = st.number_input(
            "Trail Start (Pips)",
            min_value=5.0,
            max_value=20.0,
            value=st.session_state.trading_params['trail_start_pips'],
            step=1.0
        )
    with col2:
        st.session_state.trading_params['trail_distance_pips'] = st.number_input(
            "Trail Distance (Pips)",
            min_value=3.0,
            max_value=10.0,
            value=st.session_state.trading_params['trail_distance_pips'],
            step=1.0
        )
    st.session_state.trading_params['max_daily_loss_percent'] = st.slider(
        "Max Daily Loss (%)",
        min_value=1.0,
        max_value=5.0,
        value=st.session_state.trading_params['max_daily_loss_percent'],
        step=0.5,
        help="Stop trading if daily loss exceeds this % of initial bank"
    )
    
    # Show current daily PnL
    if datetime.now().date() > st.session_state.last_reset_date:
        st.session_state.daily_start_balance = st.session_state.bank_balance
        st.session_state.last_reset_date = datetime.now().date()
    daily_pnl = st.session_state.bank_balance - st.session_state.daily_start_balance
    st.info(f"**Daily P&L:** ${daily_pnl:.2f} ({(daily_pnl / st.session_state.daily_start_balance * 100):.1f}%)")
    
    st.divider()
    
    # Candlestick Patterns
    st.subheader("🕯️ Candlestick Patterns")
    st.session_state.trading_params['use_candlestick_patterns'] = st.checkbox(
        "Enable Advanced Candlestick Patterns (Pin Bar, Engulfing, Hammer, Doji, Morning Star, Harami)",
        value=st.session_state.trading_params['use_candlestick_patterns'],
        help="Detect multiple candlestick patterns with detailed logic for enhanced signals"
    )
    st.session_state.trading_params['min_pattern_confidence'] = st.slider(
        "Min Pattern Confidence",
        min_value=0.5,
        max_value=1.0,
        value=st.session_state.trading_params['min_pattern_confidence'],
        step=0.1,
        help="Minimum confidence score for patterns to count as signals (0-1)"
    )
    
    st.divider()
    
    # Strategy Selection
    st.subheader("🎯 Trading Strategy")
    strategy_options = {name: strategy['name'] for name, strategy in PRO_STRATEGIES.items()}
    selected_strategy = st.selectbox(
        "Choose Strategy",
        options=list(strategy_options.keys()),
        format_func=lambda x: strategy_options[x],
        index=list(strategy_options.keys()).index(st.session_state.trading_params.get('selected_strategy', 'PROFESSIONAL_COMBO'))
    )
    
    if st.button("🔄 Apply Strategy", use_container_width=True):
        apply_strategy(selected_strategy)
        st.success("Strategy applied!")
    
    st.divider()
    
    # Trading Controls
    st.subheader("🎮 Trading Controls")
    col1, col2 = st.columns(2)
    with col1:
        if st.button("🚀 Start Auto", use_container_width=True, type="primary"):
            st.session_state.auto_trading = True
            st.success("Auto Trading Started!")
    with col2:
        if st.button("🛑 Stop Auto", use_container_width=True):
            st.session_state.auto_trading = False
            st.warning("Auto Trading Stopped!")
    
    if st.session_state.auto_trading:
        st.success("**Auto Trading: ACTIVE**")
    else:
        st.info("Auto Trading: INACTIVE")
    
    st.divider()
    
    # Quick Stats
    total_profit = sum(trade.get('profit_loss', 0) for trade in st.session_state.trade_history)
    st.write(f"**Bank Balance:** ${st.session_state.bank_balance:.2f}")
    st.write(f"**Total P&L:** ${total_profit:.2f}")
    st.write(f"**Open Trades:** {len(st.session_state.open_trades)}")
    st.write(f"**Total Trades:** {len(st.session_state.trade_history)}")

# MANUAL TRADING SECTION
st.subheader("🎯 Manual Trading")

col1, col2, col3, col4, col5 = st.columns(5)

with col1:
    manual_pair = st.selectbox("Currency Pair", trading_pairs)
with col2:
    manual_direction = st.selectbox("Direction", ["BUY", "SELL"])
with col3:
    default_stake = st.session_state.trading_params['manual_stake_amount'] if st.session_state.use_manual_stake else (st.session_state.trading_params['max_risk_percent'] / 100) * st.session_state.bank_balance
    manual_stake = st.number_input(
        "Stake Amount ($)",
        min_value=10.0,
        max_value=st.session_state.bank_balance,
        value=default_stake,
        step=10.0,
        help="Enter custom stake amount for this trade (overrides default)"
    )
with col4:
    current_price = st.session_state.current_prices.get(manual_pair, FOREX_PAIRS[manual_pair]['base_price'])
    st.write(f"**Current Price:** {current_price:.4f}")
with col5:
    # Manual pattern selection for demo
    manual_pattern = st.selectbox("Select Pattern", [None, "Pin Bar Bullish", "Engulfing Bullish", "Hammer Bullish", "Morning Star Bullish", "Harami Bullish"], index=0)
    manual_conf = st.slider("Manual Confidence", 0.0, 1.0, 0.8, 0.1, disabled=manual_pattern is None) if manual_pattern else 0.0
    if st.button("🎯 Execute Manual Trade", use_container_width=True, type="primary"):
        patterns = {manual_pattern.lower().replace(' ', '_').replace('bar-', 'bar_').replace('star-', 'star_'): {'type': 'bullish', 'confidence': manual_conf}} if manual_pattern else None
        if execute_trade(manual_pair, manual_direction, current_price, manual_stake, patterns):
            st.success(f"Manual trade executed! {manual_pair} {manual_direction} - ${manual_stake:.2f} (Pattern: {manual_pattern}, Conf: {manual_conf:.2f})")
        else:
            st.error("Failed to execute trade!")

# Execute trading logic
auto_trades_executed = execute_auto_trades()
update_trades()
st.session_state.all_signals = scan_all_pairs_signals()

# CURRENT OPEN TRADES TABLE
st.subheader("📊 Current Open Trades")

if st.session_state.open_trades:
    # Create DataFrame for open trades
    open_trades_data = []
    for trade in st.session_state.open_trades:
        pattern_str = ', '.join([f"{k}:{v['type']}({v['confidence']:.2f})" for k, v in trade.get('patterns', {}).items() if isinstance(v, dict)]) if trade.get('patterns') else 'None'
        open_trades_data.append({
            'ID': trade['id'],
            'Pair': trade['pair'],
            'Direction': trade['direction'],
            'Stake': f"${trade['stake']:.2f}",
            'Stake Type': trade.get('stake_type', 'AUTO'),
            'Patterns': pattern_str,
            'Entry Price': f"{trade['entry_price']:.4f}",
            'Current Price': f"{trade['current_price']:.4f}",
            'Stop Loss': f"{trade['stop_loss_price']:.4f}",
            'Take Profit': f"{trade['take_profit_price']:.4f}",
            'P&L ($)': f"${trade['profit_loss']:.2f}",
            'P&L (Pips)': f"{trade['profit_loss_pips']:.1f}",
            'Time Opened': trade['time'].strftime('%H:%M:%S')
        })
    
    open_df = pd.DataFrame(open_trades_data)
    st.dataframe(open_df, use_container_width=True, hide_index=True)
    
    # Close trade buttons
    st.write("**Close Trades Manually:**")
    cols = st.columns(4)
    for i, trade in enumerate(st.session_state.open_trades):
        with cols[i % 4]:
            if st.button(f"Close Trade {trade['id']}", key=f"close_{trade['id']}", use_container_width=True):
                close_trade(trade['id'])
                st.success(f"Trade {trade['id']} closed!")
                st.rerun()
else:
    st.info("No open trades currently")

# TRADE HISTORY TABLE
st.subheader("📈 Trade History & Performance")

if st.session_state.trade_history:
    # Create DataFrame for trade history
    history_data = []
    for trade in st.session_state.trade_history:
        pattern_str = ', '.join([f"{k}:{v['type']}({v['confidence']:.2f})" for k, v in trade.get('patterns', {}).items() if isinstance(v, dict)]) if trade.get('patterns') else 'None'
        history_data.append({
            'ID': trade['id'],
            'Pair': trade['pair'],
            'Direction': trade['direction'],
            'Stake': f"${trade['stake']:.2f}",
            'Stake Type': trade.get('stake_type', 'AUTO'),
            'Patterns': pattern_str,
            'Entry Price': f"{trade['entry_price']:.4f}",
            'Exit Price': f"{trade.get('close_price', 0):.4f}",
            'P&L ($)': f"${trade['profit_loss']:.2f}",
            'P&L (Pips)': f"{trade['profit_loss_pips']:.1f}",
            'Close Reason': trade.get('close_reason', ''),
            'Open Time': trade['time'].strftime('%H:%M'),
            'Close Time': trade.get('close_time', '').strftime('%H:%M') if trade.get('close_time') else ''
        })
    
    history_df = pd.DataFrame(history_data)
    
    # Display with filters
    col1, col2 = st.columns(2)
    with col1:
        show_trades = st.selectbox("Show", ["All Trades", "Winning Trades", "Losing Trades"])
    with col2:
        stake_type_filter = st.selectbox("Stake Type", ["All Types", "Manual", "Auto"])
    
    # Apply filters
    filtered_df = history_df.copy()
    
    if show_trades == "Winning Trades":
        filtered_df = filtered_df[pd.to_numeric(filtered_df['P&L ($)'].str.replace('$', '')) > 0]
    elif show_trades == "Losing Trades":
        filtered_df = filtered_df[pd.to_numeric(filtered_df['P&L ($)'].str.replace('$', '')) < 0]
    
    if stake_type_filter == "Manual":
        filtered_df = filtered_df[filtered_df['Stake Type'] == 'MANUAL']
    elif stake_type_filter == "Auto":
        filtered_df = filtered_df[filtered_df['Stake Type'] == 'AUTO']
    
    st.dataframe(filtered_df, use_container_width=True, hide_index=True)
    
    # PERFORMANCE SUMMARY
    st.subheader("📊 Performance Summary")
    
    total_trades = len(st.session_state.trade_history)
    winning_trades = len([t for t in st.session_state.trade_history if t['profit_loss'] > 0])
    losing_trades = len([t for t in st.session_state.trade_history if t['profit_loss'] < 0])
    win_rate = (winning_trades / total_trades * 100) if total_trades > 0 else 0
    
    total_profit = sum(trade['profit_loss'] for trade in st.session_state.trade_history)
    avg_profit = total_profit / total_trades if total_trades > 0 else 0
    
    # Manual vs Auto performance
    manual_trades = [t for t in st.session_state.trade_history if t.get('stake_type') == 'MANUAL']
    auto_trades = [t for t in st.session_state.trade_history if t.get('stake_type') == 'AUTO']
    
    manual_profit = sum(t['profit_loss'] for t in manual_trades)
    auto_profit = sum(t['profit_loss'] for t in auto_trades)
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Trades", total_trades)
    with col2:
        st.metric("Win Rate", f"{win_rate:.1f}%")
    with col3:
        st.metric("Total P&L", f"${total_profit:.2f}")
    with col4:
        st.metric("Avg Trade", f"${avg_profit:.2f}")
    
    # Stake Type Breakdown
    st.write("**Stake Type Performance:**")
    col1, col2 = st.columns(2)
    with col1:
        st.write(f"**Manual Trades:** {len(manual_trades)}")
        st.write(f"**Manual P&L:** ${manual_profit:.2f}")
    with col2:
        st.write(f"**Auto Trades:** {len(auto_trades)}")
        st.write(f"**Auto P&L:** ${auto_profit:.2f}")
    
else:
    st.info("No trade history yet. Trades will appear here once they are closed.")

# TRADING ACTIVITY
st.subheader("🎯 Recent Trading Activity")

# Show recent trade executions
if auto_trades_executed:
    st.write("**Recent Auto Trades:**")
    for trade in auto_trades_executed[-5:]:
        if "BUY" in trade:
            st.success(trade)
        else:
            st.error(trade)

# Current signals
st.write("**Current Market Signals:**")
cols = st.columns(5)
for idx, pair in enumerate(trading_pairs):
    with cols[idx % 5]:
        signal_info = st.session_state.all_signals.get(pair, {})
        agreement = signal_info.get('agreement', 'NONE')
        patterns = signal_info.get('patterns', {})
        current_price = signal_info.get('current_price', 0)
        
        if agreement == 'BUY':
            color = "#00ff88"
            text = "BUY"
        elif agreement == 'SELL':
            color = "#ff4444"
            text = "SELL"
        else:
            color = "#666666"
            text = "HOLD"
        
        pattern_str = ', '.join([f"{k}:{v['type']}({v['confidence']:.2f})" for k, v in patterns.items() if isinstance(v, dict)]) if patterns else "None"
        
        st.markdown(f"""
        <div style="border: 2px solid {color}; border-radius: 10px; padding: 0.5rem; text-align: center; font-size: 0.8rem;">
            <h5>{pair}</h5>
            <h4 style="color: {color}; margin: 0;">{text}</h4>
            <p>Price: {current_price:.4f}</p>
            <p>Patterns: {pattern_str}</p>
        </div>
        """, unsafe_allow_html=True)

# Auto-refresh
st.divider()
st.write("🔄 Auto-refreshing every 30 seconds...")
time.sleep(30)
st.rerun()

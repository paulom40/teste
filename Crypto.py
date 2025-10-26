import streamlit as st
import pandas as pd
import numpy as np
import requests
import time
from datetime import datetime, timedelta
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

st.markdown("""
    <style>
    #MainMenu {visibility: hidden;}
    header {visibility: hidden;}
    footer {visibility: hidden;}
    </style>
""", unsafe_allow_html=True)

# Page configuration with modern theme
st.set_page_config(
    page_title="Crypto Auto Trading Bot - 15min Strategy",
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
    .risk-warning {
        background: linear-gradient(135deg, #ff4444 0%, #cc0000 100%);
        color: white;
        padding: 0.5rem;
        border-radius: 5px;
        margin: 0.2rem 0;
        text-align: center;
        font-weight: bold;
    }
    .timeframe-badge {
        background: rgba(255,255,255,0.2);
        padding: 0.2rem 0.5rem;
        border-radius: 10px;
        font-size: 0.7em;
        margin-left: 0.5rem;
    }
</style>
""", unsafe_allow_html=True)

# CoinGecko coin IDs mapping - Using working pairs
coin_map = {
    "BNB/USDT": "binancecoin", 
    "XRP/USDT": "ripple",
    "SOL/USDT": "solana",
    "ADA/USDT": "cardano",
    "DOT/USDT": "polkadot",
    "DOGE/USDT": "dogecoin"
}

# Default trading parameters
DEFAULT_PARAMS = {
    'initial_bank': 1000,
    'profit_target': 15,  # pips
    'stop_loss': 10,      # pips
    'ma_fast': 9,
    'ma_slow': 21,
    'rsi_period': 14,
    'rsi_overbought': 70,
    'rsi_oversold': 30,
    'macd_fast': 12,
    'macd_slow': 26,
    'macd_signal': 9,
    'required_indicators': 2,
    # Risk Management Parameters
    'max_open_trades': 3,
    'max_risk_percent': 2.0,
    'daily_loss_limit': 5.0,
    'max_drawdown': 10.0,
    'candles_to_analyze': 4
}

# Initialize session state with parameters
if 'trading_params' not in st.session_state:
    st.session_state.trading_params = DEFAULT_PARAMS.copy()

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

# Risk Management Session State
if 'peak_balance' not in st.session_state:
    st.session_state.peak_balance = st.session_state.bank_balance
if 'daily_start_balance' not in st.session_state:
    st.session_state.daily_start_balance = st.session_state.bank_balance
if 'current_date' not in st.session_state:
    st.session_state.current_date = datetime.now().date()
if 'daily_pnl' not in st.session_state:
    st.session_state.daily_pnl = 0.0
if 'risk_halt' not in st.session_state:
    st.session_state.risk_halt = False

# Performance monitoring
if 'performance_stats' not in st.session_state:
    st.session_state.performance_stats = {
        'api_calls': 0,
        'errors': 0,
        'last_success': None
    }

# Pip sizes for pairs
pip_sizes = {
    "BNB/USDT": 0.1,
    "XRP/USDT": 0.0001,
    "SOL/USDT": 0.1,
    "ADA/USDT": 0.0001,
    "DOT/USDT": 0.1,
    "DOGE/USDT": 0.0001
}

# Trading pairs
trading_pairs = list(coin_map.keys())

# Initial prices (fallback)
initial_prices = {
    "BNB/USDT": 500,
    "XRP/USDT": 0.5,
    "SOL/USDT": 150,
    "ADA/USDT": 0.4,
    "DOT/USDT": 7.0,
    "DOGE/USDT": 0.15
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
            'agreement': 'NONE',
            'timeframe': '15min'
        }

# Create session with retry strategy
def create_session():
    session = requests.Session()
    retry_strategy = Retry(
        total=3,
        backoff_factor=1,
        status_forcelist=[429, 500, 502, 503, 504],
    )
    adapter = HTTPAdapter(max_retries=retry_strategy)
    session.mount("http://", adapter)
    session.mount("https://", adapter)
    return session

# Function to check and update daily reset
def check_daily_reset():
    today = datetime.now().date()
    if today > st.session_state.current_date:
        st.session_state.daily_start_balance = st.session_state.bank_balance
        st.session_state.daily_pnl = 0.0
        st.session_state.current_date = today
        st.rerun()

# Function to fetch market data using a different CoinGecko endpoint
def fetch_market_data(coin_id, days=7):
    """
    Fetch market data using CoinGecko's market_chart endpoint which is more reliable
    """
    st.session_state.performance_stats['api_calls'] += 1
    url = f"https://api.coingecko.com/api/v3/coins/{coin_id}/market_chart"
    params = {"vs_currency": "usd", "days": days, "interval": "daily"}
    
    try:
        session = create_session()
        response = session.get(url, params=params, timeout=10)
        
        if response.status_code != 200:
            st.warning(f"API returned status {response.status_code} for {coin_id}. Using simulated data.")
            return generate_15min_simulated_data(pair, 200)
            
        data = response.json()
        
        if not data or 'prices' not in data:
            st.warning(f"No price data returned for {coin_id}. Using simulated data.")
            return generate_15min_simulated_data(pair, 200)
            
        # Convert to DataFrame with OHLC format
        prices = data['prices']
        df = pd.DataFrame(prices, columns=['timestamp', 'close'])
        df['date'] = pd.to_datetime(df['timestamp'], unit='ms')
        
        # Generate OHLC data from close prices
        df_15min = generate_15min_from_close_prices(df, 200)
        
        st.session_state.performance_stats['last_success'] = datetime.now()
        return df_15min
        
    except Exception as e:
        st.session_state.performance_stats['errors'] += 1
        # Don't show error messages for each pair to avoid spam
        return generate_15min_simulated_data(pair, 200)

def generate_15min_from_close_prices(df, periods=200):
    """Generate 15-minute OHLC data from daily close prices"""
    if df.empty:
        return generate_15min_simulated_data(None, periods)
    
    base_price = df['close'].iloc[0]
    prices = []
    current_time = datetime.now()
    
    for i in range(periods):
        date = current_time - timedelta(minutes=15 * (periods - i - 1))
        
        open_price = base_price
        # Realistic 15-minute volatility
        change = np.random.normal(0, 0.0012)
        close_price = base_price * (1 + change)
        high_price = max(open_price, close_price) * (1 + abs(np.random.normal(0, 0.0006)))
        low_price = min(open_price, close_price) * (1 - abs(np.random.normal(0, 0.0006)))
        
        prices.append({
            "date": date,
            "open": open_price,
            "high": high_price,
            "low": low_price,
            "close": close_price
        })
        
        base_price = close_price
    
    return pd.DataFrame(prices)

def generate_15min_simulated_data(pair, periods=200):
    """Generate simulated 15-minute candle data"""
    if pair and pair in initial_prices:
        base_price = initial_prices[pair]
    else:
        base_price = 100  # Default base price
        
    prices = []
    current_time = datetime.now()
    
    for i in range(periods):
        date = current_time - timedelta(minutes=15 * (periods - i - 1))
        
        open_price = base_price
        change = np.random.normal(0, 0.0012)
        close_price = base_price * (1 + change)
        high_price = max(open_price, close_price) * (1 + abs(np.random.normal(0, 0.0006)))
        low_price = min(open_price, close_price) * (1 - abs(np.random.normal(0, 0.0006)))
        
        prices.append({
            "date": date,
            "open": open_price,
            "high": high_price,
            "low": low_price,
            "close": close_price
        })
        
        base_price = close_price
    
    return pd.DataFrame(prices)

# Technical Indicator Calculations
def calculate_rsi(prices, period=14):
    """Calculate RSI manually"""
    delta = prices.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    return rsi

def calculate_macd(prices, fast=12, slow=26, signal=9):
    """Calculate MACD manually"""
    exp1 = prices.ewm(span=fast).mean()
    exp2 = prices.ewm(span=slow).mean()
    macd_line = exp1 - exp2
    signal_line = macd_line.ewm(span=signal).mean()
    histogram = macd_line - signal_line
    return macd_line, signal_line, histogram

def calculate_indicators(df):
    try:
        if df.empty or len(df) < 50:
            return df
            
        df_indicators = df.copy()
        params = st.session_state.trading_params
        
        # Moving Averages
        df_indicators['MA_Fast'] = df_indicators['close'].rolling(window=params['ma_fast']).mean()
        df_indicators['MA_Slow'] = df_indicators['close'].rolling(window=params['ma_slow']).mean()
        
        # RSI
        df_indicators['RSI'] = calculate_rsi(df_indicators['close'], params['rsi_period'])
        
        # MACD
        macd_line, signal_line, histogram = calculate_macd(
            df_indicators['close'], 
            params['macd_fast'], 
            params['macd_slow'], 
            params['macd_signal']
        )
        df_indicators['MACD'] = macd_line
        df_indicators['MACD_Signal'] = signal_line
        df_indicators['MACD_Histogram'] = histogram
        
        return df_indicators
        
    except Exception as e:
        return df

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
    # Reset risk state
    st.session_state.peak_balance = st.session_state.bank_balance
    st.session_state.daily_start_balance = st.session_state.bank_balance
    st.session_state.daily_pnl = 0.0
    st.session_state.current_date = datetime.now().date()
    st.session_state.risk_halt = False
    
    # Re-initialize signal history
    for pair in trading_pairs:
        st.session_state.signal_history[pair] = {
            'signals': [],
            'buy_indicators': [],
            'sell_indicators': [],
            'time': datetime.now(),
            'buy_count': 0,
            'sell_count': 0,
            'agreement': 'NONE',
            'timeframe': '15min'
        }

# Function to detect trading signals with 4-candle analysis
def detect_trading_signals(df):
    buy_indicators = []
    sell_indicators = []
    params = st.session_state.trading_params
    candles_to_analyze = params['candles_to_analyze']
    
    try:
        # We need enough data for the indicators plus candles for analysis
        min_data_required = max(params['ma_slow'], params['macd_slow'], params['rsi_period']) + candles_to_analyze
        if len(df) < min_data_required:
            return [], [], [], 'NONE'
        
        # Analyze last N candles
        recent_data = df.tail(candles_to_analyze).reset_index(drop=True)
        
        # Get individual candles for analysis
        candles = []
        for i in range(candles_to_analyze):
            candles.append(recent_data.iloc[-(i+1)])
        
        latest = candles[0]  # Current candle
        previous_candles = candles[1:]  # Previous candles
        
        has_valid_data = all(pd.notna(latest.get(col, np.nan)) for col in 
                           ['MA_Fast', 'MA_Slow', 'RSI', 'MACD', 'MACD_Signal'])
        
        if not has_valid_data:
            return [], [], [], 'NONE'
        
        # 1. Moving Average Crossover Signal with multi-candle confirmation
        if all(pd.notna(candle['MA_Fast']) and pd.notna(candle['MA_Slow']) for candle in candles):
            # Check for recent crossover
            current_bullish = latest['MA_Fast'] > latest['MA_Slow']
            previous_bullish = all(candle['MA_Fast'] > candle['MA_Slow'] for candle in previous_candles[:2])
            
            current_bearish = latest['MA_Fast'] < latest['MA_Slow']
            previous_bearish = all(candle['MA_Fast'] < candle['MA_Slow'] for candle in previous_candles[:2])
            
            # Bullish crossover confirmation
            if current_bullish and not previous_bullish:
                buy_indicators.append(f"MA Bullish Crossover ({candles_to_analyze}c)")
            # Bearish crossover confirmation
            elif current_bearish and not previous_bearish:
                sell_indicators.append(f"MA Bearish Crossover ({candles_to_analyze}c)")
        
        # 2. RSI Signals with multi-candle analysis
        if pd.notna(latest['RSI']):
            # Check RSI levels across multiple candles
            oversold_candles = sum(1 for candle in candles if pd.notna(candle['RSI']) and candle['RSI'] < params['rsi_oversold'])
            overbought_candles = sum(1 for candle in candles if pd.notna(candle['RSI']) and candle['RSI'] > params['rsi_overbought'])
            
            # RSI recovering from oversold
            if (oversold_candles >= 2 and latest['RSI'] > params['rsi_oversold'] + 5):
                buy_indicators.append(f"RSI Recovery ({oversold_candles}/{candles_to_analyze}c)")
            # RSI declining from overbought
            elif (overbought_candles >= 2 and latest['RSI'] < params['rsi_overbought'] - 5):
                sell_indicators.append(f"RSI Decline ({overbought_candles}/{candles_to_analyze}c)")
            
            # Current extreme levels
            if latest['RSI'] < params['rsi_oversold']:
                buy_indicators.append("RSI Oversold")
            elif latest['RSI'] > params['rsi_overbought']:
                sell_indicators.append("RSI Overbought")
        
        # 3. MACD Signals with momentum analysis
        if all(pd.notna(candle['MACD']) and pd.notna(candle['MACD_Signal']) for candle in candles[:3]):
            # MACD crossover analysis
            current_bullish_cross = latest['MACD'] > latest['MACD_Signal']
            previous_bullish_cross = any(candle['MACD'] > candle['MACD_Signal'] for candle in previous_candles[:2])
            
            current_bearish_cross = latest['MACD'] < latest['MACD_Signal']
            previous_bearish_cross = any(candle['MACD'] < candle['MACD_Signal'] for candle in previous_candles[:2])
            
            # MACD momentum (trend)
            macd_trend_up = True
            macd_trend_down = True
            
            for i in range(min(3, len(candles)-1)):
                if candles[i]['MACD'] <= candles[i+1]['MACD']:
                    macd_trend_up = False
                if candles[i]['MACD'] >= candles[i+1]['MACD']:
                    macd_trend_down = False
            
            if current_bullish_cross and not previous_bullish_cross and macd_trend_up:
                buy_indicators.append(f"MACD Bullish Cross ({candles_to_analyze}c)")
            elif current_bearish_cross and not previous_bearish_cross and macd_trend_down:
                sell_indicators.append(f"MACD Bearish Cross ({candles_to_analyze}c)")
        
        # 4. Price Action Analysis
        price_trend_up = True
        price_trend_down = True
        
        for i in range(len(candles)-1):
            if candles[i]['close'] <= candles[i+1]['close']:
                price_trend_up = False
            if candles[i]['close'] >= candles[i+1]['close']:
                price_trend_down = False
        
        if price_trend_up and len(buy_indicators) > 0:
            buy_indicators.append(f"Uptrend ({candles_to_analyze}c)")
        if price_trend_down and len(sell_indicators) > 0:
            sell_indicators.append(f"Downtrend ({candles_to_analyze}c)")
        
        # 5. Multi-indicator confirmation
        if len(buy_indicators) >= 2:
            buy_indicators.append("Multi-Indicator Confirmation")
        if len(sell_indicators) >= 2:
            sell_indicators.append("Multi-Indicator Confirmation")
        
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

def scan_all_pairs_signals():
    all_signals = {}
    params = st.session_state.trading_params
    
    for pair in trading_pairs:
        coin_id = coin_map[pair]
        # Fetch market data
        df = fetch_market_data(coin_id, days=7)
        if df.empty:
            continue
            
        st.session_state.price_history[pair] = df
        df_with_indicators = calculate_indicators(df)
        
        signals, buy_indicators, sell_indicators, agreement = detect_trading_signals(df_with_indicators)
        
        current_price = df_with_indicators['close'].iloc[-1]
        st.session_state.current_prices[pair] = current_price
        
        all_signals[pair] = {
            'signals': signals,
            'buy_indicators': buy_indicators,
            'sell_indicators': sell_indicators,
            'time': datetime.now(),
            'buy_count': len(buy_indicators),
            'sell_count': len(sell_indicators),
            'agreement': agreement,
            'current_price': current_price,
            'price_change': calculate_price_change_15min(pair),
            'timeframe': '15min',
            'candles_analyzed': params['candles_to_analyze']
        }
        
        st.session_state.signal_history[pair] = all_signals[pair]
    
    return all_signals

def calculate_price_change_15min(pair):
    """Calculate price change based on 15-minute data"""
    if pair in st.session_state.price_history and len(st.session_state.price_history[pair]) > 1:
        df = st.session_state.price_history[pair]
        current_price = df.iloc[-1]['close']
        previous_price = df.iloc[-2]['close']
        change = ((current_price - previous_price) / previous_price) * 100
        return change
    return 0

# Risk Management Functions
def check_risk_rules():
    params = st.session_state.trading_params
    
    # Daily loss limit
    daily_loss_percent = (st.session_state.daily_pnl / st.session_state.daily_start_balance) * 100
    if daily_loss_percent <= -params['daily_loss_limit']:
        return False, f"Daily loss limit exceeded: {daily_loss_percent:.2f}%"
    
    # Max drawdown
    total_open_pnl = sum(t['profit_loss'] for t in st.session_state.open_trades)
    current_equity = st.session_state.bank_balance + total_open_pnl
    drawdown = ((current_equity - st.session_state.peak_balance) / st.session_state.peak_balance) * 100
    if drawdown <= -params['max_drawdown']:
        return False, f"Max drawdown exceeded: {drawdown:.2f}%"
    
    # Max open trades
    if len(st.session_state.open_trades) >= params['max_open_trades']:
        return False, f"Max open trades limit reached: {params['max_open_trades']}"
    
    return True, "OK"

def can_open_trade(pair, direction):
    if st.session_state.risk_halt:
        return False
    
    params = st.session_state.trading_params
    
    # Check global risk rules
    can_trade, reason = check_risk_rules()
    if not can_trade:
        return False
    
    # Same direction trades check
    same_direction_trades = [t for t in st.session_state.open_trades 
                           if t['pair'] == pair and t['direction'] == direction]
    
    if same_direction_trades:
        return False
    
    # Dynamic stake based on risk %
    risk_amount = (params['max_risk_percent'] / 100) * st.session_state.bank_balance
    if st.session_state.bank_balance < risk_amount:
        return False
    
    return True

def execute_auto_trades():
    if not st.session_state.auto_trading or st.session_state.risk_halt:
        return []
    
    auto_trades_executed = []
    params = st.session_state.trading_params
    
    try:
        all_signals = scan_all_pairs_signals()
        
        for pair, signal_info in all_signals.items():
            signals = signal_info.get('signals', [])
            agreement = signal_info.get('agreement', 'NONE')
            
            for signal_type, count, indicators in signals:
                try:
                    if agreement == 'BUY' and signal_type == "BUY" and can_open_trade(pair, 'BUY'):
                        current_price = st.session_state.current_prices.get(pair, initial_prices[pair])
                        risk_amount = (params['max_risk_percent'] / 100) * st.session_state.bank_balance
                        
                        # Minimum stake check
                        if risk_amount < 1:
                            continue
                            
                        if execute_trade(pair, 'BUY', current_price, risk_amount):
                            auto_trades_executed.append(f"AUTO BUY {pair} ({count} indicators: {', '.join(indicators)})")
                            st.session_state.last_auto_trade[pair] = datetime.now()
                    
                    elif agreement == 'SELL' and signal_type == "SELL" and can_open_trade(pair, 'SELL'):
                        current_price = st.session_state.current_prices.get(pair, initial_prices[pair])
                        risk_amount = (params['max_risk_percent'] / 100) * st.session_state.bank_balance
                        
                        if risk_amount < 1:
                            continue
                            
                        if execute_trade(pair, 'SELL', current_price, risk_amount):
                            auto_trades_executed.append(f"AUTO SELL {pair} ({count} indicators: {', '.join(indicators)})")
                            st.session_state.last_auto_trade[pair] = datetime.now()
                            
                except Exception as e:
                    continue
                    
    except Exception as e:
        pass
    
    return auto_trades_executed

def execute_trade(pair, direction, entry_price, stake_amount):
    try:
        params = st.session_state.trading_params
        
        if st.session_state.bank_balance >= stake_amount:
            trade = {
                'id': len(st.session_state.trade_history) + 1,
                'pair': pair,
                'direction': direction,
                'entry_price': entry_price,
                'stake': stake_amount,
                'time': datetime.now(),
                'status': 'open',
                'profit_loss': 0,
                'type': 'AUTO' if st.session_state.auto_trading else 'MANUAL'
            }
            st.session_state.open_trades.append(trade)
            st.session_state.bank_balance -= stake_amount
            st.session_state.peak_balance = max(st.session_state.peak_balance, st.session_state.bank_balance + sum(t['stake'] + t['profit_loss'] for t in st.session_state.open_trades))
            return True
        return False
    except Exception as e:
        return False

def update_trades():
    try:
        params = st.session_state.trading_params
        profit_target = params['profit_target']
        stop_loss = params['stop_loss']
        
        trades_to_remove = []
        for i, trade in enumerate(st.session_state.open_trades):
            if trade['status'] == 'open':
                current_price = st.session_state.current_prices.get(trade['pair'], trade['entry_price'])
                pip_size = pip_sizes.get(trade['pair'], 0.0001)
                
                if trade['direction'] == 'BUY':
                    pips = (current_price - trade['entry_price']) / pip_size
                else:
                    pips = (trade['entry_price'] - current_price) / pip_size
                
                # P&L calculation
                profit_loss = pips * (trade['stake'] / 10)
                trade['profit_loss'] = profit_loss
                trade['current_price'] = current_price
                
                if profit_loss >= profit_target:
                    trade['status'] = 'closed'
                    trade['close_time'] = datetime.now()
                    trade['close_price'] = current_price
                    st.session_state.bank_balance += trade['stake'] + profit_loss
                    st.session_state.daily_pnl += profit_loss
                    st.session_state.trade_history.append(trade.copy())
                    trades_to_remove.append(i)
                elif profit_loss <= -stop_loss:
                    trade['status'] = 'closed'
                    trade['close_time'] = datetime.now()
                    trade['close_price'] = current_price
                    st.session_state.bank_balance += trade['stake'] + profit_loss
                    st.session_state.daily_pnl += profit_loss
                    st.session_state.trade_history.append(trade.copy())
                    trades_to_remove.append(i)
        
        for i in sorted(trades_to_remove, reverse=True):
            if i < len(st.session_state.open_trades):
                st.session_state.open_trades.pop(i)
        
        # Update peak balance
        total_open_pnl = sum(t['profit_loss'] for t in st.session_state.open_trades)
        current_equity = st.session_state.bank_balance + total_open_pnl
        st.session_state.peak_balance = max(st.session_state.peak_balance, current_equity)
        
        # Check for risk halt
        drawdown = ((current_equity - st.session_state.peak_balance) / st.session_state.peak_balance) * 100
        if drawdown <= -params['max_drawdown']:
            st.session_state.risk_halt = True
                
    except Exception as e:
        pass

# Main application layout
st.markdown('<h1 class="main-header">🤖 Crypto 15min 4-Candle Strategy</h1>', unsafe_allow_html=True)

# Check daily reset
check_daily_reset()

# Sidebar
with st.sidebar:
    st.markdown("## 🎯 Trading Controls")
    
    col1, col2 = st.columns(2)
    with col1:
        if st.button("🚀 Start Auto Trading", use_container_width=True, type="primary"):
            st.session_state.auto_trading = True
            st.session_state.risk_halt = False
            st.success("Auto Trading Started!")
    with col2:
        if st.button("🛑 Stop Auto Trading", use_container_width=True, type="secondary"):
            st.session_state.auto_trading = False
            st.warning("Auto Trading Stopped!")
    
    if st.session_state.risk_halt:
        st.markdown('<div class="risk-warning">🚨 RISK HALT ACTIVE - Trading Paused</div>', unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Trading Parameters
    st.markdown("## ⚙️ Trading Parameters")
    
    with st.expander("💰 Money Management", expanded=True):
        st.markdown('<div class="param-section">', unsafe_allow_html=True)
        st.session_state.trading_params['initial_bank'] = st.number_input(
            "Initial Bank Balance (USDT)", 
            min_value=100, 
            max_value=10000, 
            value=st.session_state.trading_params['initial_bank'],
            step=100
        )
        st.session_state.trading_params['max_risk_percent'] = st.number_input(
            "Max Risk per Trade (%)", 
            min_value=0.5, 
            max_value=5.0, 
            value=st.session_state.trading_params['max_risk_percent'],
            step=0.5
        )
        st.markdown('</div>', unsafe_allow_html=True)
    
    with st.expander("🎯 Trade Settings", expanded=True):
        st.markdown('<div class="param-section">', unsafe_allow_html=True)
        st.session_state.trading_params['profit_target'] = st.number_input(
            "Profit Target (pips)", 
            min_value=1, 
            max_value=100, 
            value=st.session_state.trading_params['profit_target'],
            step=1
        )
        st.session_state.trading_params['stop_loss'] = st.number_input(
            "Stop Loss (pips)", 
            min_value=1, 
            max_value=100, 
            value=st.session_state.trading_params['stop_loss'],
            step=1
        )
        st.session_state.trading_params['required_indicators'] = st.selectbox(
            "Required Indicators Agreement",
            options=[2, 3],
            index=0 if st.session_state.trading_params['required_indicators'] == 2 else 1
        )
        st.session_state.trading_params['candles_to_analyze'] = st.number_input(
            "Candles to Analyze", 
            min_value=2, 
            max_value=10, 
            value=st.session_state.trading_params['candles_to_analyze'],
            step=1,
            help="Number of recent candles to analyze for signals"
        )
        st.session_state.trading_params['max_open_trades'] = st.number_input(
            "Max Open Trades", 
            min_value=1, 
            max_value=10, 
            value=st.session_state.trading_params['max_open_trades'],
            step=1
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
        st.markdown('</div>', unsafe_allow_html=True)
    
    # Apply Parameters Button
    if st.button("🔄 Apply Parameters & Reset", type="primary", use_container_width=True):
        reset_trading_system()
        st.success("Parameters applied and system reset!")
    
    st.markdown("---")
    st.markdown("## 📊 Current Parameters")
    params = st.session_state.trading_params
    st.write(f"**Bank:** ${st.session_state.bank_balance:.2f}")
    st.write(f"**Risk/Trade:** {params['max_risk_percent']}%")
    st.write(f"**TP/SL:** ±{params['profit_target']} pips")
    st.write(f"**Candles:** {params['candles_to_analyze']}")
    st.write(f"**Agreement:** {params['required_indicators']}/3 indicators")
    st.write(f"**Timeframe:** 15min")
    
    st.markdown("---")
    st.markdown("## 📈 Monitoring Pairs")
    for pair in trading_pairs:
        st.write(f"• {pair}")
    
    st.markdown("---")
    st.markdown("## 🎯 Trading Rules")
    st.write(f"• **15-minute timeframe**")
    st.write(f"• **{params['candles_to_analyze']}-candle analysis**")
    st.write(f"• **{params['required_indicators']} indicator agreement** required")
    st.write(f"• **No duplicate trades** per pair")

# Main execution
try:
    st.session_state.all_signals = scan_all_pairs_signals()
    auto_trades_executed = execute_auto_trades()
    update_trades()
except Exception as e:
    pass

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
        <h2>${st.session_state.bank_balance:.2f}</h2>
    </div>
    """, unsafe_allow_html=True)

with col3:
    total_profit = sum(trade['profit_loss'] for trade in st.session_state.trade_history)
    profit_class = "profit-positive" if total_profit >= 0 else "profit-negative"
    st.markdown(f"""
    <div class="metric-card">
        <h3>📊 Total P&L</h3>
        <h2 class="{profit_class}">${total_profit:.2f}</h2>
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
st.markdown("## 📊 All Pairs 15min Dashboard")

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
        candles_analyzed = signal_info.get('candles_analyzed', 4)
        
        current_trades = [t for t in st.session_state.open_trades if t['pair'] == pair]
        buy_trades = len([t for t in current_trades if t['direction'] == 'BUY'])
        sell_trades = len([t for t in current_trades if t['direction'] == 'SELL'])
        
        pip_size = pip_sizes[pair]
        if pip_size >= 1:
            price_display = f"{current_price:.0f}"
        elif pip_size >= 0.1:
            price_display = f"{current_price:.1f}"
        else:
            price_display = f"{current_price:.4f}"
        
        if agreement == 'BUY':
            signal_class = "signal-strong-buy"
            signal_text = "STRONG BUY"
            signal_emoji = "🟢"
            border_class = "agreement-buy"
        elif agreement == 'SELL':
            signal_class = "signal-strong-sell"
            signal_text = "STRONG SELL"
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
            <h3>{pair} {signal_emoji} <span class="timeframe-badge">15min</span></h3>
            <div class="{signal_class}">
                {signal_text}<br>
                Indicators: {buy_count}B/{sell_count}S
            </div>
            <div style="margin-top: 0.5rem;">
                <strong>Price: {price_display}</strong><br>
                <span style="color: {change_color};">
                    {change_emoji} {price_change:+.2f}%
                </span><br>
                <small>Analysis: {candles_analyzed} candles • Trades: {buy_trades}B/{sell_trades}S</small>
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
                st.success(f"🎯 **STRONG BUY SIGNAL** - {len(buy_indicators)} indicators across {candles_analyzed} candles agree!")
            elif agreement == 'SELL':
                st.error(f"🎯 **STRONG SELL SIGNAL** - {len(sell_indicators)} indicators across {candles_analyzed} candles agree!")
            elif agreement == 'MIXED':
                st.warning(f"⚠️ **MIXED SIGNALS** - Indicators conflict across {candles_analyzed} candles")
            else:
                st.info(f"⏸️ **NO CLEAR SIGNAL** - Analyzing {candles_analyzed} recent candles")

# Trade Summary Table
st.markdown("---")
st.markdown("## 📋 Trade Summary")

tab1, tab2 = st.tabs(["Closed Trades", "Open Trades"])

with tab1:
    if st.session_state.trade_history:
        closed_df = pd.DataFrame(st.session_state.trade_history)
        if not closed_df.empty:
            closed_df_display = closed_df[['id', 'pair', 'direction', 'entry_price', 'close_price', 'profit_loss', 'stake', 'time', 'type']].copy()
            closed_df_display['profit_loss'] = closed_df_display['profit_loss'].round(2)
            closed_df_display['entry_price'] = closed_df_display['entry_price'].round(4)
            closed_df_display['close_price'] = closed_df_display['close_price'].round(4)
            closed_df_display = closed_df_display.rename(columns={
                'id': 'ID', 'pair': 'Pair', 'direction': 'Direction', 'entry_price': 'Entry',
                'close_price': 'Exit', 'profit_loss': 'P&L', 'stake': 'Stake', 'time': 'Time', 'type': 'Type'
            })
            st.dataframe(closed_df_display, use_container_width=True, hide_index=True)
    else:
        st.info("No closed trades yet.")

with tab2:
    if st.session_state.open_trades:
        open_df = pd.DataFrame(st.session_state.open_trades)
        open_df_display = open_df[['id', 'pair', 'direction', 'entry_price', 'current_price', 'profit_loss', 'stake', 'time', 'type']].copy()
        open_df_display['profit_loss'] = open_df_display['profit_loss'].round(2)
        open_df_display['entry_price'] = open_df_display['entry_price'].round(4)
        open_df_display['current_price'] = open_df_display['current_price'].round(4)
        open_df_display = open_df_display.rename(columns={
            'id': 'ID', 'pair': 'Pair', 'direction': 'Direction', 'entry_price': 'Entry',
            'current_price': 'Current', 'profit_loss': 'P&L', 'stake': 'Stake', 'time': 'Time', 'type': 'Type'
        })
        st.dataframe(open_df_display, use_container_width=True, hide_index=True)
    else:
        st.info("No open trades.")

# Auto-refresh
st.markdown("---")
st.markdown("🔄 Auto-refreshing every 30 seconds...")

try:
    time.sleep(30)
    st.rerun()
except Exception as e:
    pass

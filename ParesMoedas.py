import streamlit as st
import pandas as pd
import numpy as np
import requests
import time
from datetime import datetime, timedelta
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Try to import yfinance, if not available, use simulated data
try:
    import yfinance as yf
    YFINANCE_AVAILABLE = True
except ImportError:
    YFINANCE_AVAILABLE = False
    st.warning("yfinance not installed. Using simulated data. Install with: pip install yfinance")

st.markdown("""
    <style>
    #MainMenu {visibility: hidden;}
    header {visibility: hidden;}
    footer {visibility: hidden;}
    </style>
""", unsafe_allow_html=True)

# Page configuration with modern theme
st.set_page_config(
    page_title="Forex Auto Trading Bot - 4 Signal Agreement",
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
    .engulfing-buy {
        background: linear-gradient(135deg, #32CD32 0%, #228B22 100%);
        color: white;
        padding: 0.5rem;
        border-radius: 5px;
        margin: 0.2rem 0;
        text-align: center;
        font-weight: bold;
        border: 2px solid #00FF00;
    }
    .engulfing-sell {
        background: linear-gradient(135deg, #FF4500 0%, #B22222 100%);
        color: white;
        padding: 0.5rem;
        border-radius: 5px;
        margin: 0.2rem 0;
        text-align: center;
        font-weight: bold;
        border: 2px solid #FF0000;
    }
    .data-source-real {
        background: linear-gradient(135deg, #00ff88 0%, #00cc66 100%);
        color: white;
        padding: 0.5rem;
        border-radius: 5px;
        text-align: center;
        font-weight: bold;
    }
    .data-source-simulated {
        background: linear-gradient(135deg, #FFA500 0%, #FF8C00 100%);
        color: white;
        padding: 0.5rem;
        border-radius: 5px;
        text-align: center;
        font-weight: bold;
    }
    @keyframes pulse {
        0% { transform: scale(1); }
        50% { transform: scale(1.05); }
        100% { transform: scale(1); }
    }
</style>
""", unsafe_allow_html=True)

# Technical indicator functions (same as before)
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

def detect_engulfing_pattern(df, lookback=2):
    """
    Detect Bullish and Bearish Engulfing candlestick patterns
    Returns: 1 for Bullish Engulfing, -1 for Bearish Engulfing, 0 for no pattern
    """
    engulfing_signals = []
    
    for i in range(len(df)):
        if i < lookback:
            engulfing_signals.append(0)
            continue
            
        current_open = df['Open'].iloc[i]
        current_close = df['Close'].iloc[i]
        current_high = df['High'].iloc[i]
        current_low = df['Low'].iloc[i]
        
        prev_open = df['Open'].iloc[i-1]
        prev_close = df['Close'].iloc[i-1]
        prev_high = df['High'].iloc[i-1]
        prev_low = df['Low'].iloc[i-1]
        
        # Calculate candle sizes
        current_body = abs(current_close - current_open)
        prev_body = abs(prev_close - prev_open)
        
        # Bullish Engulfing Pattern
        bullish_engulfing = (
            prev_close < prev_open and  # Previous red candle
            current_close > current_open and  # Current green candle
            current_open < prev_close and  # Current opens below previous close
            current_close > prev_open and  # Current closes above previous open
            current_body > prev_body * 1.1  # Current body is significantly larger
        )
        
        # Bearish Engulfing Pattern
        bearish_engulfing = (
            prev_close > prev_open and  # Previous green candle
            current_close < current_open and  # Current red candle
            current_open > prev_close and  # Current opens above previous close
            current_close < prev_open and  # Current closes below previous open
            current_body > prev_body * 1.1  # Current body is significantly larger
        )
        
        if bullish_engulfing:
            engulfing_signals.append(1)
        elif bearish_engulfing:
            engulfing_signals.append(-1)
        else:
            engulfing_signals.append(0)
    
    return pd.Series(engulfing_signals, index=df.index)

def get_forex_data_yahoo(pair, period="60d", interval="1h"):
    """
    Get Forex data from Yahoo Finance using yfinance library
    """
    if not YFINANCE_AVAILABLE:
        return generate_forex_data(pair)
    
    # Convert pair format (EUR/USD -> EURUSD=X)
    yahoo_symbol = pair.replace("/", "") + "=X"
    
    try:
        ticker = yf.Ticker(yahoo_symbol)
        data = ticker.history(period=period, interval=interval)
        
        if data.empty:
            st.warning(f"No data found for {pair} from Yahoo Finance. Using simulated data.")
            return generate_forex_data(pair)
            
        # Ensure we have enough data points
        if len(data) < 50:
            st.warning(f"Insufficient data for {pair} ({len(data)} points). Using simulated data.")
            return generate_forex_data(pair)
        
        # Reset index to make DateTime a column
        data = data.reset_index()
        data = data.rename(columns={'Date': 'Date'})
        
        # Set Date as index
        data = data.set_index('Date')
        
        return data[['Open', 'High', 'Low', 'Close']]
        
    except Exception as e:
        st.warning(f"Error fetching {pair} from Yahoo Finance: {str(e)}. Using simulated data.")
        return generate_forex_data(pair)

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
    
    # Generate realistic Forex price movements with some trends for engulfing patterns
    returns = np.random.randn(len(dates)) * volatility
    # Add some momentum for more realistic engulfing patterns
    momentum = np.convolve(returns, np.ones(5)/5, mode='same')
    returns = returns * 0.7 + momentum * 0.3
    
    prices = base_price * (1 + returns).cumprod()
    
    # Generate high/low based on price with typical Forex spreads
    # Create more realistic candle patterns for engulfing detection
    opens = []
    highs = []
    lows = []
    closes = []
    
    for i in range(len(prices)):
        if i == 0:
            open_price = prices[i] * (1 + np.random.randn() * 0.0001)
        else:
            open_price = closes[i-1]
        
        close_price = prices[i]
        
        # Determine if bullish or bearish candle
        if close_price > open_price:
            # Bullish candle
            high_price = close_price * (1 + abs(np.random.randn()) * 0.0005)
            low_price = open_price * (1 - abs(np.random.randn()) * 0.0003)
        else:
            # Bearish candle
            high_price = open_price * (1 + abs(np.random.randn()) * 0.0003)
            low_price = close_price * (1 - abs(np.random.randn()) * 0.0005)
        
        opens.append(open_price)
        highs.append(high_price)
        lows.append(low_price)
        closes.append(close_price)
    
    df = pd.DataFrame({
        'Date': dates,
        'Open': opens,
        'High': highs,
        'Low': lows,
        'Close': closes
    })
    
    df = df.set_index('Date')
    return df

def fetch_forex_data(pair, days=80, data_source='yahoo'):
    """
    Main function to fetch Forex data from various sources
    """
    if data_source == 'yahoo' and YFINANCE_AVAILABLE:
        return get_forex_data_yahoo(pair, period=f"{days}d", interval="1h")
    else:
        return generate_forex_data(pair, days)

# Initialize session state (same as before)
if 'trade_history' not in st.session_state:
    st.session_state.trade_history = pd.DataFrame({
        'Date': [], 'Pair': [], 'Direction': [], 'Entry Price': [], 'Exit Price': [],
        'Quantity': [], 'P&L': [], 'P&L (€)': [], 'Status': [], 'Signal Strength': [],
        'Signal Count': [], 'Stake (€)': [], 'Target Profit': [], 'Stop Loss': [],
        'Engulfing Pattern': [], 'Data Source': []
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

if 'data_source' not in st.session_state:
    st.session_state.data_source = 'yahoo' if YFINANCE_AVAILABLE else 'simulated'

# Initialize indicator parameters in session state
for param in ['rsi_period', 'ma_fast', 'ma_slow', 'bb_period']:
    if param not in st.session_state:
        st.session_state[param] = 14 if param == 'rsi_period' else 20 if param == 'ma_fast' else 50 if param == 'ma_slow' else 20

def add_trade_to_history(pair, direction, entry_price, exit_price, quantity, pnl, pnl_eur, status, signal_strength, signal_count, stake_eur, target_profit, stop_loss, engulfing_pattern, data_source):
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
        'Stop Loss': [float(stop_loss)],
        'Engulfing Pattern': [engulfing_pattern],
        'Data Source': [data_source]
    })
    
    st.session_state.trade_history = pd.concat([st.session_state.trade_history, new_trade], ignore_index=True)

def analyze_pair(pair, rsi_period=14, ma_fast=20, ma_slow=50, bb_period=20, data_source='yahoo'):
    """Analyze a single Forex pair and return trading signals"""
    try:
        # Fetch data
        df = fetch_forex_data(pair, days=80, data_source=data_source)
        current_data_source = 'yahoo' if YFINANCE_AVAILABLE and data_source == 'yahoo' else 'simulated'
        
        # Calculate all indicators
        df['RSI'] = calculate_rsi(df['Close'], period=rsi_period)
        df['MA_Fast'], df['MA_Slow'] = calculate_moving_averages(df['Close'], ma_fast, ma_slow)
        df['MACD'], df['MACD_Signal'], df['MACD_Hist'] = calculate_macd(df['Close'])
        df['BB_Upper'], df['BB_Middle'], df['BB_Lower'] = calculate_bollinger_bands(df['Close'], bb_period)
        df['Stoch_K'], df['Stoch_D'] = calculate_stochastic(df['High'], df['Low'], df['Close'])
        df['Engulfing'] = detect_engulfing_pattern(df)
        
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
        current_engulfing = current_data['Engulfing']
        
        # Calculate signals (1 for buy, -1 for sell, 0 for neutral)
        signals = {
            'RSI': 1 if current_rsi < 30 else -1 if current_rsi > 70 else 0,
            'MACrossover': 1 if current_ma_fast > current_ma_slow else -1,
            'MACD': 1 if current_macd > current_macd_signal else -1,
            'Bollinger': 1 if current_price < current_bb_lower else -1 if current_price > current_bb_upper else 0,
            'Stochastic': 1 if current_stoch_k < 20 and current_stoch_k > current_stoch_d else -1 if current_stoch_k > 80 and current_stoch_k < current_stoch_d else 0,
            'Engulfing': current_engulfing
        }
        
        # Count buy/sell signals (Engulfing gets double weight)
        buy_signals = sum(1 for signal in signals.values() if signal == 1)
        sell_signals = sum(1 for signal in signals.values() if signal == -1)
        
        # Add extra weight for engulfing pattern
        if current_engulfing == 1:
            buy_signals += 1
        elif current_engulfing == -1:
            sell_signals += 1
        
        # Determine final signal
        if buy_signals >= 4:
            return {
                'pair': pair,
                'signal': 'BUY',
                'strength': 'Strong' if buy_signals >= 5 else 'Moderate',
                'signal_count': buy_signals,
                'price': current_price,
                'signals': signals,
                'engulfing_pattern': 'Bullish Engulfing' if current_engulfing == 1 else 'None',
                'data_source': current_data_source
            }
        elif sell_signals >= 4:
            return {
                'pair': pair,
                'signal': 'SELL',
                'strength': 'Strong' if sell_signals >= 5 else 'Moderate',
                'signal_count': sell_signals,
                'price': current_price,
                'signals': signals,
                'engulfing_pattern': 'Bearish Engulfing' if current_engulfing == -1 else 'None',
                'data_source': current_data_source
            }
        else:
            return {
                'pair': pair,
                'signal': 'HOLD',
                'strength': 'Weak',
                'signal_count': max(buy_signals, sell_signals),
                'price': current_price,
                'signals': signals,
                'engulfing_pattern': 'Bullish Engulfing' if current_engulfing == 1 else 'Bearish Engulfing' if current_engulfing == -1 else 'None',
                'data_source': current_data_source
            }
            
    except Exception as e:
        st.error(f"Error analyzing {pair}: {str(e)}")
        return {
            'pair': pair,
            'signal': 'ERROR',
            'strength': 'Error',
            'signal_count': 0,
            'price': 0,
            'signals': {},
            'engulfing_pattern': 'None',
            'data_source': 'error'
        }

def execute_auto_trade(signal_data, lot_size, risk_percent, stake_eur, target_profit_pips, stop_loss_pips):
    """Execute an automated trade based on signal data"""
    pair = signal_data['pair']
    direction = signal_data['signal']
    signal_count = signal_data['signal_count']
    current_price = signal_data['price']
    engulfing_pattern = signal_data['engulfing_pattern']
    data_source = signal_data['data_source']
    
    # Check maximum simultaneous trades limit (3)
    if len(st.session_state.open_positions) >= 3:
        return f"❌ Maximum of 3 simultaneous trades reached. Cannot open new position for {pair}"
    
    # Check if we already have an open position for this pair
    if pair in st.session_state.open_positions:
        return f"Position already open for {pair}"
    
    # Determine pip size
    pip_size = 0.01 if 'JPY' in pair else 0.0001
    pip_value = 10  # Approximate USD per pip per standard lot
    
    # Calculate position size based on risk
    risk_amount = (risk_percent / 100) * 10000
    quantity = min(float(lot_size), risk_amount / (stop_loss_pips * pip_value))
    
    # Higher success probability when engulfing pattern is present
    engulfing_bonus = 0.15 if engulfing_pattern != 'None' else 0
    base_success_rate = 0.7 + engulfing_bonus
    
    # Simulate whether trade hits target profit or stop loss
    hit_target = np.random.random() > (1 - base_success_rate)
    
    if hit_target:
        # Trade hits target profit
        pnl = target_profit_pips * pip_value * quantity
        if direction == "BUY":
            exit_price = current_price + (target_profit_pips * pip_size)
        else:
            exit_price = current_price - (target_profit_pips * pip_size)
    else:
        # Trade hits stop loss
        pnl = -stop_loss_pips * pip_value * quantity
        if direction == "BUY":
            exit_price = current_price - (stop_loss_pips * pip_size)
        else:
            exit_price = current_price + (stop_loss_pips * pip_size)
    
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
        stop_loss=stop_loss_pips,
        engulfing_pattern=engulfing_pattern,
        data_source=data_source
    )
    
    # Add to open positions
    st.session_state.open_positions[pair] = {
        'direction': direction,
        'entry_price': current_price,
        'quantity': quantity,
        'entry_time': datetime.now(),
        'stake_eur': stake_eur,
        'target_profit': target_profit_pips,
        'stop_loss': stop_loss_pips,
        'engulfing_pattern': engulfing_pattern,
        'data_source': data_source
    }
    
    result_type = "Target Profit" if hit_target else "Stop Loss"
    pattern_info = f" ({engulfing_pattern})" if engulfing_pattern != 'None' else ""
    source_info = " (Real Data)" if data_source == 'yahoo' else " (Simulated Data)"
    return f"Auto trade executed: {direction} {pair} with {signal_count} signals{pattern_info}{source_info} (Stake: €{stake_eur:.2f}, {result_type})"

def scan_all_pairs(data_source='yahoo'):
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
            bb_period=st.session_state.bb_period,
            data_source=data_source
        )
        
        if signal_data['signal'] in ['BUY', 'SELL']:
            trading_opportunities.append(signal_data)
    
    return trading_opportunities

# ... (rest of the functions like calculate_trade_statistics remain the same)

def main():
    # Header
    st.markdown('<h1 class="main-header">🌍 Forex Auto Trading Bot</h1>', unsafe_allow_html=True)
    st.markdown('<h3 style="text-align: center; color: #666;">Fully Automated 4-Signal Agreement System with Engulfing Patterns</h3>', unsafe_allow_html=True)
    
    # Data Source Status
    if YFINANCE_AVAILABLE:
        st.markdown('<div class="data-source-real">✅ Real Data: Yahoo Finance Active</div>', unsafe_allow_html=True)
    else:
        st.markdown('<div class="data-source-simulated">⚠️ Simulated Data: Install yfinance for real data</div>', unsafe_allow_html=True)
        st.info("To get real Forex data, install: `pip install yfinance`")
    
    # Sidebar
    st.sidebar.title("⚙️ Forex Trading Configuration")
    
    # Data Source Selection
    st.sidebar.subheader("🌐 Data Source")
    if YFINANCE_AVAILABLE:
        data_source = st.sidebar.radio(
            "Select Data Source",
            ["yahoo", "simulated"],
            index=0,
            format_func=lambda x: "Yahoo Finance (Real)" if x == "yahoo" else "Simulated Data"
        )
        st.session_state.data_source = data_source
    else:
        st.sidebar.warning("yfinance not installed")
        st.sidebar.info("Using simulated data. Install: pip install yfinance")
        st.session_state.data_source = 'simulated'
    
    # Trading parameters (same as before)
    st.sidebar.subheader("💰 Stake Configuration")
    stake_euros = st.sidebar.number_input(
        "Stake Amount (€)", 
        value=st.session_state.stake_euros, 
        min_value=10.0, 
        max_value=10000.0, 
        step=50.0
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
            step=5
        )
        st.session_state.target_profit_pips = target_profit_pips
    with col2:
        stop_loss_pips = st.sidebar.number_input(
            "Stop Loss (pips)",
            value=st.session_state.stop_loss_pips,
            min_value=5,
            max_value=50,
            step=5
        )
        st.session_state.stop_loss_pips = stop_loss_pips
    
    risk_reward_ratio = target_profit_pips / stop_loss_pips if stop_loss_pips > 0 else 0
    st.sidebar.metric("Risk/Reward Ratio", f"{risk_reward_ratio:.2f}:1")
    
    st.sidebar.subheader("Trading Parameters")
    initial_balance = st.sidebar.number_input("Account Balance ($)", value=10000.0, min_value=1000.0, step=1000.0)
    risk_per_trade = st.sidebar.slider("Risk per Trade (%)", 0.5, 5.0, 1.0)
    lot_size = st.sidebar.selectbox("Lot Size", ["0.01", "0.1", "1.0", "10.0"])
    
    # Indicator settings
    st.sidebar.subheader("Indicator Settings")
    st.session_state.rsi_period = st.sidebar.slider("RSI Period", 5, 30, 14)
    st.session_state.ma_fast = st.sidebar.slider("Fast MA Period", 5, 50, 20)
    st.session_state.ma_slow = st.sidebar.slider("Slow MA Period", 20, 200, 50)
    st.session_state.bb_period = st.sidebar.slider("Bollinger Bands Period", 10, 30, 20)
    
    # Auto trading controls (same as before)
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
    
    # ... (rest of the main function remains the same)

if __name__ == "__main__":
    main()

import streamlit as st
import pandas as pd
import numpy as np
import time
from datetime import datetime, timedelta

st.set_page_config(
    page_title="Forex Auto Trading Bot - 15min Strategy",
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
    .pair-card {
        background: linear-gradient(135deg, #2c3e50 0%, #3498db 100%);
        padding: 1rem;
        border-radius: 10px;
        color: white;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        margin-bottom: 1rem;
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
    .indicator-buy {
        background: rgba(0, 255, 136, 0.2);
        border-left: 4px solid #00ff88;
        padding: 0.5rem;
        margin: 0.2rem 0;
        border-radius: 4px;
        color: white;
    }
    .indicator-sell {
        background: rgba(255, 68, 68, 0.2);
        border-left: 4px solid #ff4444;
        padding: 0.5rem;
        margin: 0.2rem 0;
        border-radius: 4px;
        color: white;
    }
    .trade-row {
        background: rgba(248, 249, 250, 0.8);
        padding: 0.5rem;
        margin: 0.2rem 0;
        border-radius: 5px;
        border-left: 4px solid #667eea;
    }
    .trade-buy {
        border-left: 4px solid #00ff88 !important;
    }
    .trade-sell {
        border-left: 4px solid #ff4444 !important;
    }
    .forex-pip {
        font-size: 0.8em;
        color: #ccc;
    }
</style>
""", unsafe_allow_html=True)

# Forex pairs with realistic base prices and volatility
FOREX_PAIRS = {
    "EUR/USD": {"base_price": 1.0850, "volatility": 0.0008, "pip_value": 0.0001},
    "GBP/USD": {"base_price": 1.2650, "volatility": 0.0010, "pip_value": 0.0001},
    "USD/JPY": {"base_price": 148.50, "volatility": 0.15, "pip_value": 0.01},
    "USD/CHF": {"base_price": 0.8800, "volatility": 0.0009, "pip_value": 0.0001},
    "USD/CAD": {"base_price": 1.3550, "volatility": 0.0010, "pip_value": 0.0001},
    "AUD/USD": {"base_price": 0.6550, "volatility": 0.0012, "pip_value": 0.0001},
    "NZD/USD": {"base_price": 0.6100, "volatility": 0.0012, "pip_value": 0.0001},
    "EUR/GBP": {"base_price": 0.8570, "volatility": 0.0007, "pip_value": 0.0001}
}

# Default trading parameters for Forex
DEFAULT_PARAMS = {
    'initial_bank': 10000,  # Higher for Forex
    'profit_target': 0.8,   # In pips
    'stop_loss': 0.5,      # In pips
    'ma_fast': 9,
    'ma_slow': 21,
    'rsi_period': 14,
    'rsi_overbought': 70,
    'rsi_oversold': 30,
    'macd_fast': 12,
    'macd_slow': 26,
    'macd_signal': 9,
    'fib_swing': 20,
    'fib_tolerance': 0.5,
    'required_indicators': 3,
    'max_open_trades': 3,
    'max_risk_percent': 2.0,
    'daily_loss_limit': 5.0,
    'max_drawdown': 10.0,
    'candles_to_analyze': 4,
    'lot_size': 10000,     # Mini lots
    'leverage': 30         # Common Forex leverage
}

# Initialize session state
if 'trading_params' not in st.session_state:
    st.session_state.trading_params = DEFAULT_PARAMS.copy()
else:
    # Ensure all default keys are present in case of version updates
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
if 'last_auto_trade' not in st.session_state:
    st.session_state.last_auto_trade = {}
if 'trade_counter' not in st.session_state:
    st.session_state.trade_counter = 0

# Trading pairs list
trading_pairs = list(FOREX_PAIRS.keys())

# Technical Indicator Calculations for Forex
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
        
        # Moving Averages
        df_indicators['MA_Fast'] = df_indicators['close'].rolling(window=params['ma_fast']).mean()
        df_indicators['MA_Slow'] = df_indicators['close'].rolling(window=params['ma_slow']).mean()
        
        # RSI
        df_indicators['RSI'] = calculate_rsi(df_indicators['close'], params['rsi_period'])
        
        # MACD
        macd_line, signal_line = calculate_macd(
            df_indicators['close'], 
            params['macd_fast'], 
            params['macd_slow'], 
            params['macd_signal']
        )
        df_indicators['MACD'] = macd_line
        df_indicators['MACD_Signal'] = signal_line
        
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
        # More realistic Forex price movements
        change = np.random.normal(0, volatility * 0.1)  # Smaller intra-15min moves
        close_price = base_price * (1 + change)
        
        # Ensure realistic high/low based on typical Forex spreads
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

def detect_trading_signals(df):
    buy_indicators = []
    sell_indicators = []
    params = st.session_state.trading_params
    
    try:
        if len(df) < 20:
            return [], [], [], 'NONE'
        
        latest = df.iloc[-1]
        
        # 1. Moving Average Signals
        if pd.notna(latest['MA_Fast']) and pd.notna(latest['MA_Slow']):
            if latest['MA_Fast'] > latest['MA_Slow']:
                buy_indicators.append("MA Bullish")
            else:
                sell_indicators.append("MA Bearish")
        
        # 2. RSI Signals
        if pd.notna(latest['RSI']):
            if latest['RSI'] < params['rsi_oversold']:
                buy_indicators.append("RSI Oversold")
            elif latest['RSI'] > params['rsi_overbought']:
                sell_indicators.append("RSI Overbought")
        
        # 3. MACD Signals
        if pd.notna(latest['MACD']) and pd.notna(latest['MACD_Signal']):
            if latest['MACD'] > latest['MACD_Signal']:
                buy_indicators.append("MACD Bullish")
            else:
                sell_indicators.append("MACD Bearish")
        
        # 4. Price Action Signal (Support/Resistance)
        if len(df) >= 20:
            recent_high = df['high'].tail(20).max()
            recent_low = df['low'].tail(20).min()
            current_close = latest['close']
            
            # If price is near recent high, potential resistance
            if abs(current_close - recent_high) / current_close < 0.001:  # 0.1% tolerance
                sell_indicators.append("Near Resistance")
            # If price is near recent low, potential support
            elif abs(current_close - recent_low) / current_close < 0.001:  # 0.1% tolerance
                buy_indicators.append("Near Support")
        
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
            
        return signals, buy_indicators, sell_indicators, agreement
        
    except Exception as e:
        return [], [], [], 'NONE'

def scan_all_pairs_signals():
    all_signals = {}
    
    for pair in trading_pairs:
        # Generate simulated Forex data
        df = generate_15min_forex_data(pair, 200)
        df_with_indicators = calculate_indicators(df)
        
        signals, buy_indicators, sell_indicators, agreement = detect_trading_signals(df_with_indicators)
        
        current_price = df_with_indicators['close'].iloc[-1]
        st.session_state.current_prices[pair] = current_price
        
        # Simulate realistic Forex price movement
        volatility = FOREX_PAIRS[pair]['volatility']
        price_change = np.random.normal(0, volatility) * 100  # Convert to percentage
        
        all_signals[pair] = {
            'signals': signals,
            'buy_indicators': buy_indicators,
            'sell_indicators': sell_indicators,
            'time': datetime.now(),
            'buy_count': len(buy_indicators),
            'sell_count': len(sell_indicators),
            'agreement': agreement,
            'current_price': current_price,
            'price_change': price_change,
            'timeframe': '15min',
            'candles_analyzed': st.session_state.trading_params['candles_to_analyze'],
            'pip_value': FOREX_PAIRS[pair]['pip_value']
        }
    
    return all_signals

def calculate_position_size(pair, risk_amount):
    """Calculate position size based on risk and stop loss"""
    params = st.session_state.trading_params
    pip_value = FOREX_PAIRS[pair]['pip_value']
    stop_loss_pips = params['stop_loss']
    
    # Position size = Risk amount / (Stop loss in pips * Pip value)
    if pip_value > 0 and stop_loss_pips > 0:
        position_size = risk_amount / (stop_loss_pips * pip_value * 100)  # Adjusted for lot size
        return min(position_size, params['lot_size'])  # Don't exceed lot size
    return params['lot_size']

def can_open_trade(pair, direction):
    params = st.session_state.trading_params
    
    # Check if we already have a trade for this pair in the same direction
    existing_trades = [t for t in st.session_state.open_trades if t['pair'] == pair and t['direction'] == direction]
    if existing_trades:
        return False
    
    # Check max open trades
    if len(st.session_state.open_trades) >= params['max_open_trades']:
        return False
    
    # Check bank balance
    risk_amount = (params['max_risk_percent'] / 100) * st.session_state.bank_balance
    if st.session_state.bank_balance < risk_amount:
        return False
    
    return True

def execute_trade(pair, direction, entry_price, stake_amount):
    try:
        st.session_state.trade_counter += 1
        
        # Calculate position size based on risk
        risk_amount = (st.session_state.trading_params['max_risk_percent'] / 100) * st.session_state.bank_balance
        position_size = calculate_position_size(pair, risk_amount)
        
        trade = {
            'id': st.session_state.trade_counter,
            'pair': pair,
            'direction': direction,
            'entry_price': entry_price,
            'stake': stake_amount,
            'position_size': position_size,
            'time': datetime.now(),
            'status': 'open',
            'profit_loss': 0,
            'profit_loss_pips': 0,
            'current_price': entry_price,
            'type': 'AUTO',
            'leverage': st.session_state.trading_params['leverage']
        }
        st.session_state.open_trades.append(trade)
        st.session_state.bank_balance -= stake_amount
        return True
    except Exception as e:
        return False

def close_trade(trade_id, close_price=None):
    """Close a specific trade manually"""
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
            
            # Calculate dollar P&L (simplified)
            profit_loss_dollar = pips * pip_value * trade['position_size'] * 0.1  # Simplified calculation
            
            # Update trade details
            trade['status'] = 'closed'
            trade['close_time'] = datetime.now()
            trade['close_price'] = close_price
            trade['profit_loss'] = profit_loss_dollar
            trade['profit_loss_pips'] = pips
            trade['close_reason'] = 'MANUAL'
            
            # Move to trade history and return stake + P&L
            st.session_state.trade_history.append(trade.copy())
            st.session_state.bank_balance += trade['stake'] + profit_loss_dollar
            
            # Remove from open trades
            st.session_state.open_trades.pop(i)
            return True
    return False

def execute_auto_trades():
    if not st.session_state.auto_trading:
        return []
    
    auto_trades_executed = []
    params = st.session_state.trading_params
    
    try:
        all_signals = scan_all_pairs_signals()
        
        for pair, signal_info in all_signals.items():
            signals = signal_info.get('signals', [])
            agreement = signal_info.get('agreement', 'NONE')
            
            for signal_type, count, indicators in signals:
                if agreement == 'BUY' and signal_type == "BUY" and can_open_trade(pair, 'BUY'):
                    current_price = st.session_state.current_prices.get(pair, FOREX_PAIRS[pair]['base_price'])
                    risk_amount = (params['max_risk_percent'] / 100) * st.session_state.bank_balance
                    
                    if execute_trade(pair, 'BUY', current_price, risk_amount):
                        auto_trades_executed.append(f"✅ AUTO BUY {pair} - {count} indicators: {', '.join(indicators)}")
                        st.session_state.last_auto_trade[pair] = datetime.now()
                
                elif agreement == 'SELL' and signal_type == "SELL" and can_open_trade(pair, 'SELL'):
                    current_price = st.session_state.current_prices.get(pair, FOREX_PAIRS[pair]['base_price'])
                    risk_amount = (params['max_risk_percent'] / 100) * st.session_state.bank_balance
                    
                    if execute_trade(pair, 'SELL', current_price, risk_amount):
                        auto_trades_executed.append(f"❌ AUTO SELL {pair} - {count} indicators: {', '.join(indicators)}")
                        st.session_state.last_auto_trade[pair] = datetime.now()
                        
    except Exception as e:
        st.error(f"Error in auto trading: {e}")
    
    return auto_trades_executed

def update_trades():
    params = st.session_state.trading_params
    profit_target_pips = params['profit_target']
    stop_loss_pips = params['stop_loss']
    
    trades_to_remove = []
    for i, trade in enumerate(st.session_state.open_trades):
        if trade['status'] == 'open':
            current_price = st.session_state.current_prices.get(trade['pair'], trade['entry_price'])
            pip_value = FOREX_PAIRS[trade['pair']]['pip_value']
            
            # Calculate P&L in pips
            if trade['direction'] == 'BUY':
                pips = (current_price - trade['entry_price']) / pip_value
            else:
                pips = (trade['entry_price'] - current_price) / pip_value
            
            # Calculate dollar P&L (simplified)
            profit_loss_dollar = pips * pip_value * trade['position_size'] * 0.1
            
            trade['profit_loss'] = profit_loss_dollar
            trade['profit_loss_pips'] = pips
            trade['current_price'] = current_price
            
            # Check if trade should be closed
            if pips >= profit_target_pips or pips <= -stop_loss_pips:
                trade['status'] = 'closed'
                trade['close_time'] = datetime.now()
                trade['close_price'] = current_price
                trade['close_reason'] = 'TP' if pips >= profit_target_pips else 'SL'
                st.session_state.bank_balance += trade['stake'] + profit_loss_dollar
                st.session_state.trade_history.append(trade.copy())
                trades_to_remove.append(i)
    
    # Remove closed trades
    for i in sorted(trades_to_remove, reverse=True):
        st.session_state.open_trades.pop(i)

def reset_trading_system():
    st.session_state.bank_balance = st.session_state.trading_params['initial_bank']
    st.session_state.open_trades = []
    st.session_state.trade_history = []
    st.session_state.auto_trading = False
    st.session_state.all_signals = {}
    st.session_state.current_prices = {pair: data['base_price'] for pair, data in FOREX_PAIRS.items()}
    st.session_state.trade_counter = 0

# MAIN APP LAYOUT
st.markdown('<h1 class="main-header">🤖 Forex 15min Trading Bot</h1>', unsafe_allow_html=True)

# Sidebar
with st.sidebar:
    st.header("🎯 Trading Controls")
    
    col1, col2 = st.columns(2)
    with col1:
        if st.button("🚀 Start Auto Trading", use_container_width=True, type="primary"):
            st.session_state.auto_trading = True
            st.success("Auto Trading Started!")
    with col2:
        if st.button("🛑 Stop Auto Trading", use_container_width=True, type="secondary"):
            st.session_state.auto_trading = False
            st.warning("Auto Trading Stopped!")
    
    # Show auto-trading status prominently
    if st.session_state.auto_trading:
        st.markdown('<div class="auto-trade-active">AUTO TRADING ACTIVE</div>', unsafe_allow_html=True)
    else:
        st.info("Auto Trading: INACTIVE")
    
    st.divider()
    
    # Trading Parameters
    st.header("⚙️ Forex Trading Parameters")
    
    # Money Management
    st.subheader("💰 Money Management")
    st.session_state.trading_params['initial_bank'] = st.number_input(
        "Initial Bank Balance (USD)", 
        value=st.session_state.trading_params['initial_bank'],
        min_value=1000, 
        max_value=50000,
        step=1000,
        help="Starting capital for trading"
    )
    
    st.session_state.trading_params['max_risk_percent'] = st.number_input(
        "Max Risk Per Trade (%)", 
        value=st.session_state.trading_params['max_risk_percent'],
        min_value=0.5, 
        max_value=5.0,
        step=0.5,
        help="Maximum percentage of balance to risk per trade"
    )
    
    st.session_state.trading_params['lot_size'] = st.number_input(
        "Lot Size", 
        value=st.session_state.trading_params['lot_size'],
        min_value=1000, 
        max_value=100000,
        step=1000,
        help="Standard lot size (10000 = mini lot)"
    )
    
    st.session_state.trading_params['leverage'] = st.number_input(
        "Leverage", 
        value=st.session_state.trading_params['leverage'],
        min_value=1, 
        max_value=100,
        step=1,
        help="Trading leverage ratio"
    )
    
    st.session_state.trading_params['max_open_trades'] = st.number_input(
        "Max Open Trades", 
        value=st.session_state.trading_params['max_open_trades'],
        min_value=1, 
        max_value=10,
        step=1,
        help="Maximum number of concurrent open trades"
    )
    
    st.divider()
    
    # Trade Settings
    st.subheader("🎯 Trade Settings (Pips)")
    st.session_state.trading_params['profit_target'] = st.number_input(
        "Profit Target (Pips)", 
        value=st.session_state.trading_params['profit_target'],
        min_value=0.1, 
        max_value=10.0,
        step=0.1,
        help="Take profit level in pips"
    )
    
    st.session_state.trading_params['stop_loss'] = st.number_input(
        "Stop Loss (Pips)", 
        value=st.session_state.trading_params['stop_loss'],
        min_value=0.1, 
        max_value=10.0,
        step=0.1,
        help="Stop loss level in pips"
    )
    
    current_req = st.session_state.trading_params.get('required_indicators', 3)
    options = [2, 3, 4]
    index_val = options.index(current_req) if current_req in options else 1
    st.session_state.trading_params['required_indicators'] = st.selectbox(
        "Required Indicators Agreement",
        options=options,
        index=index_val,
        help="Number of indicators that must agree for trade entry"
    )
    
    st.session_state.trading_params['candles_to_analyze'] = st.number_input(
        "Candles to Analyze", 
        value=st.session_state.trading_params['candles_to_analyze'],
        min_value=2, 
        max_value=10,
        step=1,
        help="Number of recent candles to analyze for signals"
    )
    
    st.divider()
    
    # INDICATOR PARAMETERS
    st.subheader("📊 Indicator Parameters")
    
    st.write("**Moving Averages**")
    col1, col2 = st.columns(2)
    with col1:
        st.session_state.trading_params['ma_fast'] = st.number_input(
            "MA Fast Period", 
            value=st.session_state.trading_params['ma_fast'],
            min_value=5, 
            max_value=50,
            step=1
        )
    with col2:
        st.session_state.trading_params['ma_slow'] = st.number_input(
            "MA Slow Period", 
            value=st.session_state.trading_params['ma_slow'],
            min_value=10, 
            max_value=100,
            step=1
        )
    
    st.write("**RSI Settings**")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.session_state.trading_params['rsi_period'] = st.number_input(
            "RSI Period", 
            value=st.session_state.trading_params['rsi_period'],
            min_value=5, 
            max_value=30,
            step=1
        )
    with col2:
        st.session_state.trading_params['rsi_overbought'] = st.number_input(
            "Overbought", 
            value=st.session_state.trading_params['rsi_overbought'],
            min_value=60, 
            max_value=90,
            step=1
        )
    with col3:
        st.session_state.trading_params['rsi_oversold'] = st.number_input(
            "Oversold", 
            value=st.session_state.trading_params['rsi_oversold'],
            min_value=10, 
            max_value=40,
            step=1
        )
    
    st.write("**MACD Settings**")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.session_state.trading_params['macd_fast'] = st.number_input(
            "MACD Fast", 
            value=st.session_state.trading_params['macd_fast'],
            min_value=5, 
            max_value=20,
            step=1
        )
    with col2:
        st.session_state.trading_params['macd_slow'] = st.number_input(
            "MACD Slow", 
            value=st.session_state.trading_params['macd_slow'],
            min_value=15, 
            max_value=50,
            step=1
        )
    with col3:
        st.session_state.trading_params['macd_signal'] = st.number_input(
            "MACD Signal", 
            value=st.session_state.trading_params['macd_signal'],
            min_value=5, 
            max_value=20,
            step=1
        )
    
    st.divider()
    
    # Risk Management
    st.subheader("🛡️ Risk Management")
    st.session_state.trading_params['daily_loss_limit'] = st.number_input(
        "Daily Loss Limit (%)", 
        value=st.session_state.trading_params['daily_loss_limit'],
        min_value=1.0, 
        max_value=20.0,
        step=1.0
    )
    
    st.session_state.trading_params['max_drawdown'] = st.number_input(
        "Max Drawdown (%)", 
        value=st.session_state.trading_params['max_drawdown'],
        min_value=5.0, 
        max_value=30.0,
        step=1.0
    )
    
    st.divider()
    
    # Current Parameters Summary
    st.subheader("📈 Current Settings")
    params = st.session_state.trading_params
    st.write(f"**Bank:** ${st.session_state.bank_balance:.2f}")
    st.write(f"**Risk/Trade:** {params['max_risk_percent']}%")
    st.write(f"**TP/SL:** {params['profit_target']}p / {params['stop_loss']}p")
    st.write(f"**Leverage:** {params['leverage']}:1")
    st.write(f"**Lot Size:** {params['lot_size']:,.0f}")
    st.write(f"**Indicators:** {params['required_indicators']}/4 required")
    
    st.divider()
    
    # System Controls
    st.subheader("🔧 System Controls")
    if st.button("🔄 Apply Parameters & Reset", use_container_width=True, type="primary"):
        reset_trading_system()
        st.success("Parameters applied and system reset!")
    
    if st.button("🗑️ Clear All Trades", use_container_width=True):
        st.session_state.open_trades = []
        st.session_state.trade_history = []
        st.success("All trades cleared!")
    
    st.divider()
    
    # Forex Pairs Info
    st.subheader("💱 Monitoring Pairs")
    for pair in trading_pairs[:4]:  # Show first 4 pairs
        current_price = st.session_state.current_prices.get(pair, 0)
        pip_value = FOREX_PAIRS[pair]['pip_value']
        st.write(f"• {pair}: {current_price:.4f}")

# Execute auto trading
auto_trades_executed = execute_auto_trades()
update_trades()

# Scan for signals
st.session_state.all_signals = scan_all_pairs_signals()

# Main Dashboard - TOP METRICS
st.subheader("📊 Forex Trading Dashboard")
col1, col2, col3, col4 = st.columns(4)

with col1:
    status_color = "auto-trade-active" if st.session_state.auto_trading else "metric-card"
    status_text = "ACTIVE" if st.session_state.auto_trading else "INACTIVE"
    st.markdown(f"""
    <div class="{status_color}">
        <h3>Auto Trading</h3>
        <h2>{status_text}</h2>
    </div>
    """, unsafe_allow_html=True)

with col2:
    st.markdown(f"""
    <div class="metric-card">
        <h3>Bank Balance</h3>
        <h2>${st.session_state.bank_balance:.2f}</h2>
    </div>
    """, unsafe_allow_html=True)

with col3:
    total_profit = sum(trade.get('profit_loss', 0) for trade in st.session_state.trade_history)
    profit_class = "profit-positive" if total_profit >= 0 else "profit-negative"
    st.markdown(f"""
    <div class="metric-card">
        <h3>Total P&L</h3>
        <h2 class="{profit_class}">${total_profit:.2f}</h2>
    </div>
    """, unsafe_allow_html=True)

with col4:
    st.markdown(f"""
    <div class="metric-card">
        <h3>Open Trades</h3>
        <h2>{len(st.session_state.open_trades)}</h2>
    </div>
    """, unsafe_allow_html=True)

# Show auto trade executions
if auto_trades_executed:
    st.subheader("🤖 Auto Trade Executions")
    for trade in auto_trades_executed:
        if "BUY" in trade:
            st.success(trade)
        else:
            st.error(trade)

# TRADING PAIRS SIGNALS
st.subheader("🎯 Forex Trading Signals - 15min Timeframe")
st.write(f"**Required Agreement:** {st.session_state.trading_params['required_indicators']} out of 4 indicators (MA, RSI, MACD, Price Action)")

# Display pairs in a grid
cols = st.columns(3)
for idx, pair in enumerate(trading_pairs):
    with cols[idx % 3]:
        signal_info = st.session_state.all_signals.get(pair, {})
        buy_count = signal_info.get('buy_count', 0)
        sell_count = signal_info.get('sell_count', 0)
        agreement = signal_info.get('agreement', 'NONE')
        current_price = signal_info.get('current_price', st.session_state.current_prices.get(pair, 0))
        price_change = signal_info.get('price_change', 0)
        buy_indicators = signal_info.get('buy_indicators', [])
        sell_indicators = signal_info.get('sell_indicators', [])
        pip_value = signal_info.get('pip_value', 0.0001)
        
        if agreement == 'BUY':
            signal_class = "signal-strong-buy"
            signal_text = "BUY SIGNAL"
            signal_emoji = "🟢"
            border_class = "agreement-buy"
        elif agreement == 'SELL':
            signal_class = "signal-strong-sell"
            signal_text = "SELL SIGNAL"
            signal_emoji = "🔴"
            border_class = "agreement-sell"
        elif agreement == 'MIXED':
            signal_class = "signal-mixed"
            signal_text = "MIXED"
            signal_emoji = "🟡"
            border_class = "no-agreement"
        else:
            signal_class = "signal-no-trade"
            signal_text = "NO SIGNAL"
            signal_emoji = "⚪"
            border_class = "no-agreement"
        
        change_color = "#00ff88" if price_change >= 0 else "#ff4444"
        
        # Format price based on pair type
        if 'JPY' in pair:
            price_format = f"{current_price:.2f}"
        else:
            price_format = f"{current_price:.4f}"
        
        st.markdown(f"""
        <div class="pair-card {border_class}">
            <h3>{pair} {signal_emoji}</h3>
            <div class="{signal_class}">
                {signal_text}<br>
                Buy: {buy_count} | Sell: {sell_count}
            </div>
            <div style="margin-top: 0.5rem;">
                <strong>Price: {price_format}</strong><br>
                <span style="color: {change_color};">
                    {price_change:+.2f}%
                </span>
                <div class="forex-pip">Pip: {pip_value:.4f}</div>
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        # Show indicators directly on the card
        if buy_indicators:
            st.write("**Buy Indicators:**")
            for indicator in buy_indicators:
                st.markdown(f'<div class="indicator-buy">✅ {indicator}</div>', unsafe_allow_html=True)
        
        if sell_indicators:
            st.write("**Sell Indicators:**")
            for indicator in sell_indicators:
                st.markdown(f'<div class="indicator-sell">❌ {indicator}</div>', unsafe_allow_html=True)

# TRADE HISTORY
st.subheader("📋 Trade History")

tab1, tab2 = st.tabs(["Open Trades", "Closed Trades"])

with tab1:
    if st.session_state.open_trades:
        # Display each open trade with a stop button
        for trade in st.session_state.open_trades:
            trade_class = "trade-buy" if trade['direction'] == 'BUY' else "trade-sell"
            profit_class = "profit-positive" if trade['profit_loss'] >= 0 else "profit-negative"
            
            col1, col2, col3 = st.columns([3, 2, 1])
            
            with col1:
                pip_value = FOREX_PAIRS[trade['pair']]['pip_value']
                current_pips = trade['profit_loss_pips']
                
                st.markdown(f"""
                <div class="trade-row {trade_class}">
                    <strong>{trade['pair']} {trade['direction']}</strong><br>
                    Entry: {trade['entry_price']:.4f} | Current: {trade['current_price']:.4f}<br>
                    P&L: <span class="{profit_class}">${trade['profit_loss']:.2f}</span> | Pips: {current_pips:.1f}<br>
                    Size: {trade['position_size']:,.0f} | Leverage: {trade['leverage']}:1
                </div>
                """, unsafe_allow_html=True)
            
            with col2:
                st.metric(
                    "P&L Pips", 
                    f"{current_pips:.1f}",
                    delta=f"{current_pips:+.1f}pips"
                )
            
            with col3:
                # Stop trading button for this specific trade
                if st.button(f"🛑 Stop", key=f"stop_{trade['id']}", use_container_width=True):
                    if close_trade(trade['id']):
                        st.success(f"Trade {trade['id']} closed manually!")
                        st.rerun()
                    else:
                        st.error("Failed to close trade")
        
        # Also show as dataframe for overview
        st.subheader("Open Trades Overview")
        open_df = pd.DataFrame(st.session_state.open_trades)
        if not open_df.empty:
            display_df = open_df[['id', 'pair', 'direction', 'entry_price', 'current_price', 'profit_loss', 'profit_loss_pips', 'position_size', 'time']].copy()
            display_df['profit_loss'] = display_df['profit_loss'].round(2)
            display_df['profit_loss_pips'] = display_df['profit_loss_pips'].round(1)
            display_df['entry_price'] = display_df['entry_price'].round(4)
            display_df['current_price'] = display_df['current_price'].round(4)
            st.dataframe(display_df, use_container_width=True)
    else:
        st.info("No open trades")

with tab2:
    if st.session_state.trade_history:
        closed_df = pd.DataFrame(st.session_state.trade_history)
        # Format the display
        display_df = closed_df[['id', 'pair', 'direction', 'entry_price', 'close_price', 'profit_loss', 'profit_loss_pips', 'position_size', 'time', 'close_time', 'close_reason']].copy()
        display_df['profit_loss'] = display_df['profit_loss'].round(2)
        display_df['profit_loss_pips'] = display_df['profit_loss_pips'].round(1)
        display_df['entry_price'] = display_df['entry_price'].round(4)
        display_df['close_price'] = display_df['close_price'].round(4)
        st.dataframe(display_df, use_container_width=True)
        
        # Performance summary
        st.subheader("📊 Performance Summary")
        col1, col2, col3, col4 = st.columns(4)
        
        total_trades = len(closed_df)
        winning_trades = len(closed_df[closed_df['profit_loss'] > 0])
        win_rate = (winning_trades / total_trades * 100) if total_trades > 0 else 0
        total_profit = closed_df['profit_loss'].sum()
        avg_profit = closed_df['profit_loss'].mean()
        
        with col1:
            st.metric("Total Trades", total_trades)
        with col2:
            st.metric("Win Rate", f"{win_rate:.1f}%")
        with col3:
            st.metric("Total P&L", f"${total_profit:.2f}")
        with col4:
            st.metric("Avg P&L", f"${avg_profit:.2f}")
            
    else:
        st.info("No trade history")

# Auto-refresh
st.divider()
st.write("🔄 Auto-refreshing every 30 seconds...")
time.sleep(30)
st.rerun()

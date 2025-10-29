import streamlit as st
import pandas as pd
import numpy as np
import time
from datetime import datetime, timedelta

st.set_page_config(
    page_title="Forex Auto Trading Bot - Advanced SL/TP",
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
    .sl-tp-indicator {
        font-size: 0.8em;
        padding: 0.2rem 0.5rem;
        border-radius: 3px;
        margin: 0.1rem;
    }
    .sl-indicator {
        background: rgba(255, 68, 68, 0.3);
        color: #ff4444;
        border: 1px solid #ff4444;
    }
    .tp-indicator {
        background: rgba(0, 255, 136, 0.3);
        color: #00ff88;
        border: 1px solid #00ff88;
    }
    .close-to-tp {
        background: linear-gradient(135deg, #00ff88 0%, #00cc66 100%);
        color: white;
        animation: glow-green 1s infinite alternate;
    }
    .close-to-sl {
        background: linear-gradient(135deg, #ff4444 0%, #cc0000 100%);
        color: white;
        animation: glow-red 1s infinite alternate;
    }
    @keyframes glow-green {
        from { box-shadow: 0 0 5px #00ff88; }
        to { box-shadow: 0 0 15px #00ff88; }
    }
    @keyframes glow-red {
        from { box-shadow: 0 0 5px #ff4444; }
        to { box-shadow: 0 0 15px #ff4444; }
    }
    .risk-meter {
        height: 10px;
        border-radius: 5px;
        margin: 0.2rem 0;
        background: linear-gradient(90deg, #00ff88 0%, #ffaa00 50%, #ff4444 100%);
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
    "AUD/USD": {"base_price": 0.6550, "volatility": 0.0012, "pip_value": 0.0001}
}

# Default trading parameters for Forex with advanced SL/TP
DEFAULT_PARAMS = {
    'initial_bank': 10000,
    'profit_target_pips': 15.0,      # Take Profit in pips
    'stop_loss_pips': 10.0,          # Stop Loss in pips
    'trailing_stop': False,          # Trailing stop feature
    'trailing_stop_activation': 5.0, # Activate trailing after X pips profit
    'break_even': False,             # Move SL to breakeven
    'break_even_activation': 8.0,    # Move SL to breakeven after X pips
    'risk_reward_ratio': 1.5,        # Minimum R:R ratio
    'ma_fast': 9,
    'ma_slow': 21,
    'rsi_period': 14,
    'rsi_overbought': 70,
    'rsi_oversold': 30,
    'macd_fast': 12,
    'macd_slow': 26,
    'macd_signal': 9,
    'required_indicators': 3,
    'max_open_trades': 3,
    'max_risk_percent': 2.0,
    'daily_loss_limit': 5.0,
    'max_drawdown': 10.0,
    'candles_to_analyze': 4,
    'lot_size': 10000,
    'leverage': 30
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
if 'last_auto_trade' not in st.session_state:
    st.session_state.last_auto_trade = {}
if 'trade_counter' not in st.session_state:
    st.session_state.trade_counter = 0

# Trading pairs list
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

def detect_trading_signals(df):
    buy_indicators = []
    sell_indicators = []
    params = st.session_state.trading_params
    
    try:
        if len(df) < 20:
            return [], [], [], 'NONE'
        
        latest = df.iloc[-1]
        
        # Moving Average Signals
        if pd.notna(latest['MA_Fast']) and pd.notna(latest['MA_Slow']):
            if latest['MA_Fast'] > latest['MA_Slow']:
                buy_indicators.append("MA Bullish")
            else:
                sell_indicators.append("MA Bearish")
        
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
        
        # Price Action Signal
        if len(df) >= 20:
            recent_high = df['high'].tail(20).max()
            recent_low = df['low'].tail(20).min()
            current_close = latest['close']
            
            if abs(current_close - recent_high) / current_close < 0.001:
                sell_indicators.append("Near Resistance")
            elif abs(current_close - recent_low) / current_close < 0.001:
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

def calculate_sl_tp_prices(entry_price, direction, sl_pips, tp_pips, pair):
    """Calculate Stop Loss and Take Profit prices"""
    pip_value = FOREX_PAIRS[pair]['pip_value']
    
    if direction == 'BUY':
        stop_loss_price = entry_price - (sl_pips * pip_value)
        take_profit_price = entry_price + (tp_pips * pip_value)
    else:  # SELL
        stop_loss_price = entry_price + (sl_pips * pip_value)
        take_profit_price = entry_price - (tp_pips * pip_value)
    
    return stop_loss_price, take_profit_price

def calculate_position_size(pair, risk_amount, sl_pips):
    """Calculate position size based on risk and stop loss"""
    params = st.session_state.trading_params
    pip_value = FOREX_PAIRS[pair]['pip_value']
    
    if pip_value > 0 and sl_pips > 0:
        position_size = risk_amount / (sl_pips * pip_value)
        return min(position_size, params['lot_size'] * 3)  # Allow up to 3 lots
    return params['lot_size']

def can_open_trade(pair, direction):
    params = st.session_state.trading_params
    
    # Check existing trades
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

def execute_trade(pair, direction, entry_price):
    try:
        st.session_state.trade_counter += 1
        params = st.session_state.trading_params
        
        # Calculate risk amount
        risk_amount = (params['max_risk_percent'] / 100) * st.session_state.bank_balance
        
        # Calculate position size
        position_size = calculate_position_size(pair, risk_amount, params['stop_loss_pips'])
        
        # Calculate SL and TP prices
        stop_loss_price, take_profit_price = calculate_sl_tp_prices(
            entry_price, direction, params['stop_loss_pips'], params['profit_target_pips'], pair
        )
        
        trade = {
            'id': st.session_state.trade_counter,
            'pair': pair,
            'direction': direction,
            'entry_price': entry_price,
            'stop_loss_price': stop_loss_price,
            'take_profit_price': take_profit_price,
            'original_sl_price': stop_loss_price,  # Keep original for trailing stop
            'stake': risk_amount,
            'position_size': position_size,
            'time': datetime.now(),
            'status': 'open',
            'profit_loss': 0,
            'profit_loss_pips': 0,
            'current_price': entry_price,
            'type': 'AUTO',
            'leverage': params['leverage'],
            'sl_pips': params['stop_loss_pips'],
            'tp_pips': params['profit_target_pips'],
            'trailing_stop_active': False,
            'breakeven_active': False,
            'max_profit_pips': 0,  # Track maximum profit for trailing stop
            'close_reason': None
        }
        
        st.session_state.open_trades.append(trade)
        st.session_state.bank_balance -= risk_amount
        return True
    except Exception as e:
        return False

def update_trailing_stop(trade, current_price):
    """Update trailing stop loss"""
    params = st.session_state.trading_params
    pip_value = FOREX_PAIRS[trade['pair']]['pip_value']
    
    if trade['direction'] == 'BUY':
        current_pips = (current_price - trade['entry_price']) / pip_value
    else:
        current_pips = (trade['entry_price'] - current_price) / pip_value
    
    # Update max profit
    trade['max_profit_pips'] = max(trade['max_profit_pips'], current_pips)
    
    # Activate trailing stop if conditions met
    if params['trailing_stop'] and current_pips >= params['trailing_stop_activation']:
        trade['trailing_stop_active'] = True
        
        # Calculate new stop loss (trailing by activation distance)
        trailing_distance = params['trailing_stop_activation']
        if trade['direction'] == 'BUY':
            new_sl = current_price - (trailing_distance * pip_value)
            trade['stop_loss_price'] = max(trade['stop_loss_price'], new_sl)
        else:
            new_sl = current_price + (trailing_distance * pip_value)
            trade['stop_loss_price'] = min(trade['stop_loss_price'], new_sl)
    
    # Move to breakeven if conditions met
    if params['break_even'] and not trade['breakeven_active'] and current_pips >= params['break_even_activation']:
        trade['breakeven_active'] = True
        if trade['direction'] == 'BUY':
            trade['stop_loss_price'] = trade['entry_price']  # Move SL to entry
        else:
            trade['stop_loss_price'] = trade['entry_price']

def check_sl_tp(trade, current_price):
    """Check if Stop Loss or Take Profit is hit"""
    pip_value = FOREX_PAIRS[trade['pair']]['pip_value']
    
    if trade['direction'] == 'BUY':
        # Check Take Profit
        if current_price >= trade['take_profit_price']:
            return 'TP'
        # Check Stop Loss
        elif current_price <= trade['stop_loss_price']:
            return 'SL'
    else:  # SELL
        # Check Take Profit
        if current_price <= trade['take_profit_price']:
            return 'TP'
        # Check Stop Loss
        elif current_price >= trade['stop_loss_price']:
            return 'SL'
    
    return None

def close_trade(trade_id, close_price=None, reason='MANUAL'):
    """Close a specific trade"""
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
            
            # Calculate dollar P&L
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
    """Update all open trades and check SL/TP"""
    trades_to_remove = []
    
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
            
            # Update trailing stop and breakeven
            update_trailing_stop(trade, current_price)
            
            # Check SL/TP
            sl_tp_hit = check_sl_tp(trade, current_price)
            if sl_tp_hit:
                close_trade(trade['id'], current_price, sl_tp_hit)
                trades_to_remove.append(i)

def scan_all_pairs_signals():
    all_signals = {}
    
    for pair in trading_pairs:
        df = generate_15min_forex_data(pair, 200)
        df_with_indicators = calculate_indicators(df)
        
        signals, buy_indicators, sell_indicators, agreement = detect_trading_signals(df_with_indicators)
        
        current_price = df_with_indicators['close'].iloc[-1]
        st.session_state.current_prices[pair] = current_price
        
        volatility = FOREX_PAIRS[pair]['volatility']
        price_change = np.random.normal(0, volatility) * 100
        
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
            'pip_value': FOREX_PAIRS[pair]['pip_value']
        }
    
    return all_signals

def execute_auto_trades():
    if not st.session_state.auto_trading:
        return []
    
    auto_trades_executed = []
    
    try:
        all_signals = scan_all_pairs_signals()
        
        for pair, signal_info in all_signals.items():
            signals = signal_info.get('signals', [])
            agreement = signal_info.get('agreement', 'NONE')
            
            for signal_type, count, indicators in signals:
                if agreement == 'BUY' and signal_type == "BUY" and can_open_trade(pair, 'BUY'):
                    current_price = st.session_state.current_prices.get(pair, FOREX_PAIRS[pair]['base_price'])
                    
                    if execute_trade(pair, 'BUY', current_price):
                        sl_price, tp_price = calculate_sl_tp_prices(
                            current_price, 'BUY', 
                            st.session_state.trading_params['stop_loss_pips'],
                            st.session_state.trading_params['profit_target_pips'],
                            pair
                        )
                        auto_trades_executed.append(f"✅ BUY {pair} | SL: {sl_price:.4f} | TP: {tp_price:.4f}")
                        st.session_state.last_auto_trade[pair] = datetime.now()
                
                elif agreement == 'SELL' and signal_type == "SELL" and can_open_trade(pair, 'SELL'):
                    current_price = st.session_state.current_prices.get(pair, FOREX_PAIRS[pair]['base_price'])
                    
                    if execute_trade(pair, 'SELL', current_price):
                        sl_price, tp_price = calculate_sl_tp_prices(
                            current_price, 'SELL', 
                            st.session_state.trading_params['stop_loss_pips'],
                            st.session_state.trading_params['profit_target_pips'],
                            pair
                        )
                        auto_trades_executed.append(f"❌ SELL {pair} | SL: {sl_price:.4f} | TP: {tp_price:.4f}")
                        st.session_state.last_auto_trade[pair] = datetime.now()
                        
    except Exception as e:
        st.error(f"Error in auto trading: {e}")
    
    return auto_trades_executed

def reset_trading_system():
    st.session_state.bank_balance = st.session_state.trading_params['initial_bank']
    st.session_state.open_trades = []
    st.session_state.trade_history = []
    st.session_state.auto_trading = False
    st.session_state.all_signals = {}
    st.session_state.current_prices = {pair: data['base_price'] for pair, data in FOREX_PAIRS.items()}
    st.session_state.trade_counter = 0

# MAIN APP LAYOUT
st.markdown('<h1 class="main-header">🤖 Forex Bot - Advanced SL/TP</h1>', unsafe_allow_html=True)

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
    
    if st.session_state.auto_trading:
        st.markdown('<div class="auto-trade-active">AUTO TRADING ACTIVE</div>', unsafe_allow_html=True)
    else:
        st.info("Auto Trading: INACTIVE")
    
    st.divider()
    
    # STOP LOSS & TAKE PROFIT SETTINGS
    st.header("🛡️ Stop Loss & Take Profit")
    
    st.subheader("🎯 Basic SL/TP Settings")
    col1, col2 = st.columns(2)
    with col1:
        st.session_state.trading_params['stop_loss_pips'] = st.number_input(
            "Stop Loss (Pips)", 
            value=st.session_state.trading_params['stop_loss_pips'],
            min_value=1.0, 
            max_value=50.0,
            step=1.0,
            help="Stop loss distance in pips"
        )
    with col2:
        st.session_state.trading_params['profit_target_pips'] = st.number_input(
            "Take Profit (Pips)", 
            value=st.session_state.trading_params['profit_target_pips'],
            min_value=1.0, 
            max_value=100.0,
            step=1.0,
            help="Take profit distance in pips"
        )
    
    # Risk Reward Ratio
    current_rr = st.session_state.trading_params['profit_target_pips'] / st.session_state.trading_params['stop_loss_pips']
    st.write(f"**Risk/Reward Ratio:** {current_rr:.2f}:1")
    
    if current_rr < 1:
        st.warning("⚠️ Risk/Reward ratio below 1:1")
    elif current_rr >= 1.5:
        st.success("✅ Good Risk/Reward ratio")
    
    st.divider()
    
    # ADVANCED SL/TP FEATURES
    st.subheader("⚡ Advanced Features")
    
    st.session_state.trading_params['trailing_stop'] = st.checkbox(
        "Enable Trailing Stop",
        value=st.session_state.trading_params['trailing_stop'],
        help="Stop loss follows price when in profit"
    )
    
    if st.session_state.trading_params['trailing_stop']:
        st.session_state.trading_params['trailing_stop_activation'] = st.number_input(
            "Trailing Stop Activation (Pips)", 
            value=st.session_state.trading_params['trailing_stop_activation'],
            min_value=1.0, 
            max_value=20.0,
            step=1.0,
            help="Profit level to activate trailing stop"
        )
    
    st.session_state.trading_params['break_even'] = st.checkbox(
        "Enable Break-Even Stop",
        value=st.session_state.trading_params['break_even'],
        help="Move SL to entry price when in profit"
    )
    
    if st.session_state.trading_params['break_even']:
        st.session_state.trading_params['break_even_activation'] = st.number_input(
            "Break-Even Activation (Pips)", 
            value=st.session_state.trading_params['break_even_activation'],
            min_value=1.0, 
            max_value=20.0,
            step=1.0,
            help="Profit level to move SL to breakeven"
        )
    
    st.divider()
    
    # MONEY MANAGEMENT
    st.subheader("💰 Money Management")
    st.session_state.trading_params['initial_bank'] = st.number_input(
        "Initial Bank Balance (USD)", 
        value=st.session_state.trading_params['initial_bank'],
        min_value=1000, 
        max_value=50000,
        step=1000
    )
    
    st.session_state.trading_params['max_risk_percent'] = st.number_input(
        "Max Risk Per Trade (%)", 
        value=st.session_state.trading_params['max_risk_percent'],
        min_value=0.5, 
        max_value=5.0,
        step=0.5
    )
    
    # Risk per trade calculation
    risk_per_trade = (st.session_state.trading_params['max_risk_percent'] / 100) * st.session_state.bank_balance
    st.write(f"**Risk per trade:** ${risk_per_trade:.2f}")
    
    st.divider()
    
    # SYSTEM CONTROLS
    st.subheader("🔧 System Controls")
    if st.button("🔄 Apply & Reset", use_container_width=True, type="primary"):
        reset_trading_system()
        st.success("System reset with new parameters!")
    
    if st.button("🗑️ Clear Trades", use_container_width=True):
        st.session_state.open_trades = []
        st.session_state.trade_history = []
        st.success("All trades cleared!")

# Execute trading logic
auto_trades_executed = execute_auto_trades()
update_trades()
st.session_state.all_signals = scan_all_pairs_signals()

# Main Dashboard
st.subheader("📊 Trading Dashboard")
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

# OPEN TRADES WITH SL/TP VISUALIZATION
st.subheader("📈 Open Trades - SL/TP Monitoring")

if st.session_state.open_trades:
    for trade in st.session_state.open_trades:
        # Calculate distance to SL/TP
        pip_value = FOREX_PAIRS[trade['pair']]['pip_value']
        current_price = trade['current_price']
        
        if trade['direction'] == 'BUY':
            distance_to_sl = (current_price - trade['stop_loss_price']) / pip_value
            distance_to_tp = (trade['take_profit_price'] - current_price) / pip_value
            sl_percentage = (distance_to_sl / trade['sl_pips']) * 100
            tp_percentage = (distance_to_tp / trade['tp_pips']) * 100
        else:
            distance_to_sl = (trade['stop_loss_price'] - current_price) / pip_value
            distance_to_tp = (current_price - trade['take_profit_price']) / pip_value
            sl_percentage = (distance_to_sl / trade['sl_pips']) * 100
            tp_percentage = (distance_to_tp / trade['tp_pips']) * 100
        
        # Determine if close to SL/TP
        row_class = ""
        if tp_percentage < 20:  # Within 20% of TP
            row_class = "close-to-tp"
        elif sl_percentage < 20:  # Within 20% of SL
            row_class = "close-to-sl"
        
        trade_class = "trade-buy" if trade['direction'] == 'BUY' else "trade-sell"
        profit_class = "profit-positive" if trade['profit_loss'] >= 0 else "profit-negative"
        
        col1, col2, col3, col4 = st.columns([3, 2, 2, 1])
        
        with col1:
            st.markdown(f"""
            <div class="trade-row {trade_class} {row_class}">
                <strong>{trade['pair']} {trade['direction']}</strong><br>
                Entry: {trade['entry_price']:.4f} | Current: {trade['current_price']:.4f}<br>
                P&L: <span class="{profit_class}">${trade['profit_loss']:.2f}</span> | Pips: {trade['profit_loss_pips']:.1f}<br>
                <div class="sl-tp-indicator sl-indicator">SL: {trade['stop_loss_price']:.4f} ({distance_to_sl:.1f}p)</div>
                <div class="sl-tp-indicator tp-indicator">TP: {trade['take_profit_price']:.4f} ({distance_to_tp:.1f}p)</div>
                {

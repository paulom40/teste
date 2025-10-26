import streamlit as st
import pandas as pd
import numpy as np
import requests
import time
from datetime import datetime, timedelta
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

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
</style>
""", unsafe_allow_html=True)

# CoinGecko coin IDs mapping
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
    'profit_target': 15,
    'stop_loss': 10,
    'ma_fast': 9,
    'ma_slow': 21,
    'rsi_period': 14,
    'rsi_overbought': 70,
    'rsi_oversold': 30,
    'macd_fast': 12,
    'macd_slow': 26,
    'macd_signal': 9,
    'required_indicators': 2,
    'max_open_trades': 3,
    'max_risk_percent': 2.0,
    'daily_loss_limit': 5.0,
    'max_drawdown': 10.0,
    'candles_to_analyze': 4
}

# Initialize session state
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
if 'all_signals' not in st.session_state:
    st.session_state.all_signals = {}
if 'current_prices' not in st.session_state:
    st.session_state.current_prices = {
        "BNB/USDT": 500, "XRP/USDT": 0.5, "SOL/USDT": 150, 
        "ADA/USDT": 0.4, "DOT/USDT": 7.0, "DOGE/USDT": 0.15
    }

# Trading pairs
trading_pairs = list(coin_map.keys())

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

def generate_15min_simulated_data(pair, periods=200):
    base_price = st.session_state.current_prices.get(pair, 100)
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

def detect_trading_signals(df):
    buy_indicators = []
    sell_indicators = []
    params = st.session_state.trading_params
    candles_to_analyze = params['candles_to_analyze']
    
    try:
        if len(df) < candles_to_analyze + 10:
            return [], [], [], 'NONE'
        
        # Analyze last N candles
        recent_data = df.tail(candles_to_analyze).reset_index(drop=True)
        candles = [recent_data.iloc[i] for i in range(len(recent_data))]
        
        latest = candles[-1]
        
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
        # Generate simulated data for demonstration
        df = generate_15min_simulated_data(pair, 200)
        df_with_indicators = calculate_indicators(df)
        
        signals, buy_indicators, sell_indicators, agreement = detect_trading_signals(df_with_indicators)
        
        current_price = df_with_indicators['close'].iloc[-1]
        st.session_state.current_prices[pair] = current_price
        
        # Simulate some price movement
        price_change = np.random.uniform(-2, 2)
        
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
            'candles_analyzed': st.session_state.trading_params['candles_to_analyze']
        }
    
    return all_signals

def reset_trading_system():
    st.session_state.bank_balance = st.session_state.trading_params['initial_bank']
    st.session_state.open_trades = []
    st.session_state.trade_history = []
    st.session_state.auto_trading = False
    st.session_state.all_signals = {}
    st.session_state.current_prices = {
        "BNB/USDT": 500, "XRP/USDT": 0.5, "SOL/USDT": 150, 
        "ADA/USDT": 0.4, "DOT/USDT": 7.0, "DOGE/USDT": 0.15
    }

# MAIN APP LAYOUT
st.markdown('<h1 class="main-header">🤖 Crypto 15min Trading Bot</h1>', unsafe_allow_html=True)

# Sidebar
with st.sidebar:
    st.header("🎯 Trading Controls")
    
    col1, col2 = st.columns(2)
    with col1:
        if st.button("🚀 Start Auto Trading", use_container_width=True):
            st.session_state.auto_trading = True
            st.success("Auto Trading Started!")
    with col2:
        if st.button("🛑 Stop Auto Trading", use_container_width=True):
            st.session_state.auto_trading = False
            st.warning("Auto Trading Stopped!")
    
    st.divider()
    
    # Trading Parameters
    st.header("⚙️ Parameters")
    
    with st.expander("Money Management"):
        st.session_state.trading_params['initial_bank'] = st.number_input(
            "Initial Balance", value=1000, min_value=100, max_value=10000
        )
        st.session_state.trading_params['max_risk_percent'] = st.number_input(
            "Risk per Trade %", value=2.0, min_value=0.5, max_value=5.0
        )
    
    with st.expander("Trade Settings"):
        st.session_state.trading_params['profit_target'] = st.number_input(
            "Profit Target (pips)", value=15, min_value=1, max_value=100
        )
        st.session_state.trading_params['stop_loss'] = st.number_input(
            "Stop Loss (pips)", value=10, min_value=1, max_value=100
        )
        st.session_state.trading_params['candles_to_analyze'] = st.number_input(
            "Candles to Analyze", value=4, min_value=2, max_value=10
        )
    
    if st.button("🔄 Reset System", use_container_width=True):
        reset_trading_system()
        st.success("System Reset!")

# Scan for signals
st.session_state.all_signals = scan_all_pairs_signals()

# Main Dashboard - TOP METRICS
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

# TRADING PAIRS SIGNALS
st.subheader("🎯 Trading Signals - 15min Timeframe")
st.write(f"Analyzing last **{st.session_state.trading_params['candles_to_analyze']} candles** for each pair")

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
        
        st.markdown(f"""
        <div class="pair-card {border_class}">
            <h3>{pair} {signal_emoji}</h3>
            <div class="{signal_class}">
                {signal_text}<br>
                Buy: {buy_count} | Sell: {sell_count}
            </div>
            <div style="margin-top: 0.5rem;">
                <strong>Price: ${current_price:.4f}</strong><br>
                <span style="color: {change_color};">
                    {price_change:+.2f}%
                </span>
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        # Show indicator details
        with st.expander(f"View {pair} indicators"):
            buy_indicators = signal_info.get('buy_indicators', [])
            sell_indicators = signal_info.get('sell_indicators', [])
            
            if buy_indicators:
                st.write("**Buy Indicators:**")
                for indicator in buy_indicators:
                    st.success(f"✅ {indicator}")
            
            if sell_indicators:
                st.write("**Sell Indicators:**")
                for indicator in sell_indicators:
                    st.error(f"❌ {indicator}")
            
            if not buy_indicators and not sell_indicators:
                st.info("No clear signals detected")

# TRADE HISTORY
st.subheader("📋 Trade History")

tab1, tab2 = st.tabs(["Open Trades", "Closed Trades"])

with tab1:
    if st.session_state.open_trades:
        open_df = pd.DataFrame(st.session_state.open_trades)
        st.dataframe(open_df, use_container_width=True)
    else:
        st.info("No open trades")

with tab2:
    if st.session_state.trade_history:
        closed_df = pd.DataFrame(st.session_state.trade_history)
        st.dataframe(closed_df, use_container_width=True)
    else:
        st.info("No trade history")

# Add some sample trades for demonstration
if not st.session_state.trade_history and not st.session_state.open_trades:
    st.info("💡 **Demo Mode**: Add some sample trades to see the interface")
    if st.button("Add Sample Trades"):
        # Add sample open trade
        st.session_state.open_trades.append({
            'id': 1,
            'pair': 'BNB/USDT',
            'direction': 'BUY',
            'entry_price': 510.25,
            'stake': 50,
            'time': datetime.now(),
            'status': 'open',
            'profit_loss': 12.50,
            'type': 'MANUAL'
        })
        # Add sample closed trade
        st.session_state.trade_history.append({
            'id': 1,
            'pair': 'XRP/USDT',
            'direction': 'SELL',
            'entry_price': 0.52,
            'close_price': 0.48,
            'stake': 30,
            'time': datetime.now() - timedelta(hours=2),
            'close_time': datetime.now() - timedelta(hours=1),
            'status': 'closed',
            'profit_loss': 24.00,
            'type': 'AUTO'
        })
        st.rerun()

# Auto-refresh
st.divider()
st.write("🔄 Auto-refreshing every 30 seconds...")
time.sleep(30)
st.rerun()

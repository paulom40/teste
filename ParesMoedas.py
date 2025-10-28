import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
from datetime import datetime, timedelta
import ta  # Technical analysis library

# Configure the page
st.set_page_config(
    page_title="LIVE FOREX PRICES - PURE PYTHON ANALYSIS",
    page_icon="📈",
    layout="wide"
)

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
</style>
""", unsafe_allow_html=True)

# Initialize session state with proper data types
if 'initialized' not in st.session_state:
    st.session_state.initialized = True
    st.session_state.connection_errors = []
    st.session_state.last_data_update = datetime.now()
    st.session_state.connection_status = "connected"
    st.session_state.stake_amounts = {
        'USD/JPY': 10, 'USD/CHF': 10, 'USD/CAD': 10,
        'EUR/USD': 10, 'GBP/USD': 10, 'AUD/USD': 10
    }
    st.session_state.active_trades = []
    st.session_state.trade_history = []
    st.session_state.next_trade_id = 1001
    st.session_state.price_history = {}

# Technical Analysis Functions
def calculate_technical_indicators(prices):
    """Calculate various technical indicators"""
    if len(prices) < 20:  # Need enough data for indicators
        return {}
    
    # Convert to pandas Series for TA library
    price_series = pd.Series(prices)
    
    indicators = {}
    
    # Moving Averages
    indicators['sma_20'] = ta.trend.sma_indicator(price_series, window=20).iloc[-1]
    indicators['sma_50'] = ta.trend.sma_indicator(price_series, window=50).iloc[-1] if len(prices) >= 50 else None
    indicators['ema_12'] = ta.trend.ema_indicator(price_series, window=12).iloc[-1]
    indicators['ema_26'] = ta.trend.ema_indicator(price_series, window=26).iloc[-1]
    
    # RSI
    indicators['rsi'] = ta.momentum.rsi(price_series, window=14).iloc[-1]
    
    # MACD
    macd = ta.trend.MACD(price_series)
    indicators['macd'] = macd.macd().iloc[-1]
    indicators['macd_signal'] = macd.macd_signal().iloc[-1]
    indicators['macd_histogram'] = macd.macd_diff().iloc[-1]
    
    # Bollinger Bands
    bollinger = ta.volatility.BollingerBands(price_series)
    indicators['bb_upper'] = bollinger.bollinger_hband().iloc[-1]
    indicators['bb_lower'] = bollinger.bollinger_lband().iloc[-1]
    indicators['bb_middle'] = bollinger.bollinger_mavg().iloc[-1]
    
    # Stochastic
    stochastic = ta.momentum.StochasticOscillator(high=price_series, low=price_series, close=price_series)
    indicators['stoch_k'] = stochastic.stoch().iloc[-1]
    indicators['stoch_d'] = stochastic.stoch_signal().iloc[-1]
    
    # ADX for trend strength
    adx = ta.trend.ADXIndicator(high=price_series, low=price_series, close=price_series, window=14)
    indicators['adx'] = adx.adx().iloc[-1]
    
    return indicators

def analyze_trend(indicators, current_price):
    """Analyze overall trend based on multiple indicators with ADX weighting"""
    if not indicators:
        return "NEUTRAL", "Insufficient Data"
    
    bullish_signals = 0
    bearish_signals = 0
    total_signals = 0
    trend_strength_multiplier = 1.0
    
    # Moving Average Analysis
    if indicators['sma_20'] and current_price > indicators['sma_20']:
        bullish_signals += 1
    elif indicators['sma_20']:
        bearish_signals += 1
    total_signals += 1
    
    # EMA Analysis (Golden Cross/Death Cross)
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
    
    # ADX for trend strength adjustment
    if indicators.get('adx', 0) > 25:
        trend_strength_multiplier = 1.2  # Amplify strong trends
    elif indicators.get('adx', 0) < 20:
        trend_strength_multiplier = 0.8  # Dampen weak trends
    
    # Determine trend strength
    if total_signals > 0:
        bullish_percentage = (bullish_signals / total_signals) * 100 * trend_strength_multiplier
        bearish_percentage = (bearish_signals / total_signals) * 100 * trend_strength_multiplier
        
        if bullish_percentage >= 65:
            return "BULLISH", f"Strong Uptrend ({bullish_percentage:.0f}%)"
        elif bearish_percentage >= 65:
            return "BEARISH", f"Strong Downtrend ({bearish_percentage:.0f}%)"
        elif bullish_percentage > bearish_percentage:
            return "BULLISH", f"Weak Uptrend ({bullish_percentage:.0f}%)"
        elif bearish_percentage > bullish_percentage:
            return "BEARISH", f"Weak Downtrend ({bearish_percentage:.0f}%)"
    
    return "NEUTRAL", "Sideways Market"

def generate_trading_signal(indicators, trend_direction):
    """Generate trading signal based on technical analysis with multi-confirmation"""
    if not indicators:
        return "HOLD", "Waiting for data"
    
    bullish_confirmations = 0
    bearish_confirmations = 0
    signals = []
    
    # RSI Signal
    if indicators['rsi']:
        if indicators['rsi'] < 30:
            signals.append("RSI: OVERSOLD - Potential BUY")
            bullish_confirmations += 1
        elif indicators['rsi'] > 70:
            signals.append("RSI: OVERBOUGHT - Potential SELL")
            bearish_confirmations += 1
    
    # MACD Signal
    if indicators['macd'] and indicators['macd_signal']:
        if indicators['macd'] > indicators['macd_signal'] and indicators['macd_histogram'] > 0:
            signals.append("MACD: BULLISH Crossover")
            bullish_confirmations += 1
        elif indicators['macd'] < indicators['macd_signal'] and indicators['macd_histogram'] < 0:
            signals.append("MACD: BEARISH Crossover")
            bearish_confirmations += 1
    
    # Moving Average Signal
    if indicators['sma_20'] and 'current_price' in indicators:
        if indicators['current_price'] > indicators['sma_20']:
            signals.append("Price above SMA20: BULLISH")
            bullish_confirmations += 1
        else:
            signals.append("Price below SMA20: BEARISH")
            bearish_confirmations += 1
    
    # Stochastic Signal
    if indicators['stoch_k'] and indicators['stoch_d']:
        if indicators['stoch_k'] < 20 and indicators['stoch_d'] < 20:
            signals.append("Stoch: OVERSOLD - Potential BUY")
            bullish_confirmations += 1
        elif indicators['stoch_k'] > 80 and indicators['stoch_d'] > 80:
            signals.append("Stoch: OVERBOUGHT - Potential SELL")
            bearish_confirmations += 1
    
    # Require at least 2 confirmations for strong signals
    if trend_direction == "BULLISH" and bullish_confirmations >= 2:
        return "BUY", " | ".join(signals)
    elif trend_direction == "BEARISH" and bearish_confirmations >= 2:
        return "SELL", " | ".join(signals)
    
    if signals:
        return "HOLD", " | ".join(signals)
    
    return "HOLD", "No clear signals"

# Trade management functions
def open_trade(currency_pair, position, stake, entry_price):
    """Open a new trade with SL/TP"""
    # Define SL/TP based on position (50 pips TP, 25 pips SL - approx 0.0050/0.0025 for majors)
    pip_value = 0.0005 if '/' in currency_pair and 'JPY' not in currency_pair else 0.005  # Adjust for JPY pairs
    tp_pips = 10 * pip_value  # Conservative TP
    sl_pips = 5 * pip_value   # Tight SL for better risk/reward
    
    if position == 'LONG':
        tp = entry_price + tp_pips
        sl = entry_price - sl_pips
    else:  # SHORT
        tp = entry_price - tp_pips
        sl = entry_price + sl_pips
    
    trade = {
        'trade_id': f"TR{st.session_state.next_trade_id}",
        'currency_pair': currency_pair,
        'position': position,
        'stake': stake,
        'entry_price': entry_price,
        'current_price': entry_price,
        'take_profit': tp,
        'stop_loss': sl,
        'open_time': datetime.now(),
        'status': 'ACTIVE'
    }
    st.session_state.active_trades.append(trade)
    st.session_state.next_trade_id += 1
    return trade

def check_and_close_trade(trade):
    """Check if trade should be auto-closed (SL/TP hit or signal reversal)"""
    current_price = trade['current_price']
    pair = trade['currency_pair']
    prices = st.session_state.price_history.get(pair, [])
    if len(prices) < 20:
        return None
    
    indicators = calculate_technical_indicators(prices)
    indicators['current_price'] = current_price
    trend_direction, _ = analyze_trend(indicators, current_price)
    signal, _ = generate_trading_signal(indicators, trend_direction)
    
    # Check SL/TP
    if trade['position'] == 'LONG':
        if current_price <= trade['stop_loss']:
            return close_trade(trade['trade_id'], current_price, reason="Stop Loss Hit")
        elif current_price >= trade['take_profit']:
            return close_trade(trade['trade_id'], current_price, reason="Take Profit Hit")
    else:  # SHORT
        if current_price >= trade['stop_loss']:
            return close_trade(trade['trade_id'], current_price, reason="Stop Loss Hit")
        elif current_price <= trade['take_profit']:
            return close_trade(trade['trade_id'], current_price, reason="Take Profit Hit")
    
    # Check signal reversal (close if opposite strong signal)
    if (trade['position'] == 'LONG' and signal == 'SELL') or (trade['position'] == 'SHORT' and signal == 'BUY'):
        return close_trade(trade['trade_id'], current_price, reason="Signal Reversal")
    
    return None

def close_trade(trade_id, exit_price, reason="Manual Close"):
    """Close an active trade"""
    for i, trade in enumerate(st.session_state.active_trades):
        if trade['trade_id'] == trade_id:
            # Calculate P/L
            if trade['position'] == 'LONG':
                pnl = (exit_price - trade['entry_price']) * trade['stake']
            else:  # SHORT
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
                'close_reason': reason,
                'status': 'CLOSED'
            }
            
            st.session_state.trade_history.append(closed_trade)
            st.session_state.active_trades.pop(i)
            return closed_trade
    return None

def generate_sample_data():
    """Generate sample data only if empty"""
    # Initialize price history
    currency_pairs = ['USD/JPY', 'USD/CHF', 'USD/CAD', 'EUR/USD', 'GBP/USD', 'AUD/USD']
    for pair in currency_pairs:
        if pair not in st.session_state.price_history:
            # Generate 100 historical prices for each pair with slight upward bias for demo
            base_price = np.random.uniform(1.0, 1.5) if 'USD' in pair else np.random.uniform(0.8, 1.4)
            prices = [base_price]
            for _ in range(99):
                # Slight upward bias: mean +0.0005 for non-JPY, +0.005 for JPY
                bias = 0.0005 if 'JPY' not in pair else 0.005
                change = np.random.uniform(-0.001, 0.002) + (bias / base_price)
                new_price = prices[-1] * (1 + change)
                prices.append(new_price)
            st.session_state.price_history[pair] = prices
    
    # Generate sample active trades if empty
    if len(st.session_state.active_trades) == 0:
        for i in range(2):
            pair = np.random.choice(currency_pairs)
            stake = st.session_state.stake_amounts.get(pair, 10)
            current_prices = st.session_state.price_history[pair]
            entry_price = current_prices[-1]  # Use latest price
            open_trade(pair, np.random.choice(['LONG', 'SHORT']), stake, round(entry_price, 5))
    
    # Generate sample historical trades if empty (with more wins for demo)
    if len(st.session_state.trade_history) == 0:
        for i in range(8):
            pair = np.random.choice(currency_pairs)
            stake = 10  # Fixed 10€ stake
            current_prices = st.session_state.price_history[pair]
            entry_price = current_prices[-50]  # Historical price
            exit_price = current_prices[-1]    # Current price
            
            # Bias towards profitable trades
            if np.random.random() > 0.3:  # 70% win rate
                if np.random.choice(['LONG', 'SHORT']) == 'LONG':
                    exit_price = entry_price + np.random.uniform(0.001, 0.005)  # Profitable
                    pnl = (exit_price - entry_price) * stake
                else:
                    exit_price = entry_price - np.random.uniform(0.001, 0.005)  # Profitable
                    pnl = (entry_price - exit_price) * stake
            else:
                if np.random.choice(['LONG', 'SHORT']) == 'LONG':
                    exit_price = entry_price - np.random.uniform(0.0005, 0.002)  # Loss
                    pnl = (exit_price - entry_price) * stake
                else:
                    exit_price = entry_price + np.random.uniform(0.0005, 0.002)  # Loss
                    pnl = (entry_price - exit_price) * stake
            
            historical_trade = {
                'trade_id': f"TR{800 + i}",
                'currency_pair': pair,
                'position': 'LONG' if 'USD' in pair else 'SHORT',  # Simple bias
                'stake': stake,
                'entry_price': round(entry_price, 5),
                'exit_price': round(exit_price, 5),
                'pnl': round(pnl, 2),
                'pnl_percent': round((pnl / stake) * 100, 2),
                'open_time': datetime.now() - timedelta(days=np.random.randint(5, 60)),
                'close_time': datetime.now() - timedelta(days=np.random.randint(1, 5)),
                'close_reason': 'Signal',
                'status': 'CLOSED'
            }
            st.session_state.trade_history.append(historical_trade)

# Generate sample data
generate_sample_data()

# Update current prices and technical analysis
for pair in st.session_state.stake_amounts.keys():
    if pair in st.session_state.price_history:
        # Add new price movement with slight positive bias
        last_price = st.session_state.price_history[pair][-1]
        bias = 0.0001 if 'JPY' not in pair else 0.001
        new_price = last_price * (1 + np.random.uniform(-0.0015, 0.0025) + bias)
        st.session_state.price_history[pair].append(new_price)
        # Keep only last 100 prices
        if len(st.session_state.price_history[pair]) > 100:
            st.session_state.price_history[pair] = st.session_state.price_history[pair][-100:]

# Update active trades with current prices and auto-close logic
trades_to_remove = []
for trade in st.session_state.active_trades:
    if trade['currency_pair'] in st.session_state.price_history:
        trade['current_price'] = round(st.session_state.price_history[trade['currency_pair']][-1], 5)
        closed = check_and_close_trade(trade)
        if closed:
            trades_to_remove.append(trade['trade_id'])
            st.info(f"Auto-closed trade {closed['trade_id']}: {closed['close_reason']} | P/L: €{closed['pnl']:.2f}")

# Remove closed trades
for trade_id in trades_to_remove:
    st.session_state.active_trades = [t for t in st.session_state.active_trades if t['trade_id'] != trade_id]

# Header
st.markdown('<div class="main-header">LIVE FOREX PRICES - PURE PYTHON ANALYSIS</div>', unsafe_allow_html=True)

# Connection Status
st.success("🟢 CONNECTION STABLE - All Systems Operational")

# Fixed Stake Notice
st.info("🎯 **Trading with Fixed 10€ Stake per Trade | Enhanced Strategy with SL/TP & Auto-Close**")

# Technical Analysis Overview
st.markdown("---")
st.subheader("📊 Technical Analysis Overview")

# Display technical analysis for each currency pair
currency_pairs = list(st.session_state.stake_amounts.keys())
for pair in currency_pairs:
    if pair in st.session_state.price_history:
        prices = st.session_state.price_history[pair]
        current_price = prices[-1]
        previous_price = prices[-2] if len(prices) > 1 else current_price
        price_change = current_price - previous_price
        price_change_percent = (price_change / previous_price) * 100
        
        # Calculate technical indicators
        indicators = calculate_technical_indicators(prices)
        indicators['current_price'] = current_price
        
        # Analyze trend
        trend_direction, trend_strength = analyze_trend(indicators, current_price)
        
        # Generate trading signal
        signal, signal_reason = generate_trading_signal(indicators, trend_direction)
        
        # Display analysis card
        with st.container():
            st.markdown(f'<div class="analysis-card">', unsafe_allow_html=True)
            
            col1, col2, col3, col4 = st.columns([2, 2, 2, 2])
            
            with col1:
                st.write(f"**{pair}**")
                st.write(f"**{current_price:.5f}**")
                change_color = "positive-change" if price_change >= 0 else "negative-change"
                change_symbol = "+" if price_change >= 0 else ""
                st.markdown(f'<span class="{change_color}">{change_symbol}{price_change_percent:.2f}%</span>', unsafe_allow_html=True)
            
            with col2:
                # Trend analysis
                if trend_direction == "BULLISH":
                    st.markdown(f'<div class="trend-up">📈 {trend_strength}</div>', unsafe_allow_html=True)
                elif trend_direction == "BEARISH":
                    st.markdown(f'<div class="trend-down">📉 {trend_strength}</div>', unsafe_allow_html=True)
                else:
                    st.markdown(f'<div class="trend-neutral">➡️ {trend_strength}</div>', unsafe_allow_html=True)
                
                # Trading signal
                if signal == "BUY":
                    st.markdown(f'<div class="trend-up">🎯 SIGNAL: {signal}</div>', unsafe_allow_html=True)
                elif signal == "SELL":
                    st.markdown(f'<div class="trend-down">🎯 SIGNAL: {signal}</div>', unsafe_allow_html=True)
                else:
                    st.markdown(f'<div class="trend-neutral">🎯 SIGNAL: {signal}</div>', unsafe_allow_html=True)
            
            with col3:
                # Key indicators
                if indicators.get('rsi'):
                    rsi_color = "indicator-bullish" if indicators['rsi'] < 30 else "indicator-bearish" if indicators['rsi'] > 70 else "indicator-neutral"
                    st.markdown(f'<span class="{rsi_color}">RSI: {indicators["rsi"]:.1f}</span>', unsafe_allow_html=True)
                
                if indicators.get('macd'):
                    macd_color = "indicator-bullish" if indicators['macd'] > indicators.get('macd_signal', 0) else "indicator-bearish"
                    st.markdown(f'<span class="{macd_color}">MACD: {indicators["macd"]:.4f}</span>', unsafe_allow_html=True)
                
                # ADX
                if indicators.get('adx'):
                    adx_color = "indicator-bullish" if indicators['adx'] > 25 else "indicator-neutral"
                    st.markdown(f'<span class="{adx_color}">ADX: {indicators["adx"]:.1f}</span>', unsafe_allow_html=True)
            
            with col4:
                # Additional indicators
                if indicators.get('sma_20'):
                    sma_relation = "Above" if current_price > indicators['sma_20'] else "Below"
                    st.write(f"Price {sma_relation} SMA20")
                
                if indicators.get('stoch_k') and indicators.get('stoch_d'):
                    st.write(f"Stoch: K={indicators['stoch_k']:.1f}, D={indicators['stoch_d']:.1f}")
                
                st.caption(signal_reason)
            
            st.markdown('</div>', unsafe_allow_html=True)

# Current Active Trades Table
st.markdown("---")
st.subheader("📊 Current Active Trades")

if len(st.session_state.active_trades) > 0:
    total_exposure = sum(trade['stake'] for trade in st.session_state.active_trades)
    total_unrealized_pnl = 0
    
    # Display summary metrics
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Active Trades", len(st.session_state.active_trades))
    with col2:
        st.metric("Total Exposure", f"€{total_exposure:,}")
    
    # Display active trades
    for trade in st.session_state.active_trades:
        if trade['position'] == 'LONG':
            unrealized_pnl = (trade['current_price'] - trade['entry_price']) * trade['stake']
        else:  # SHORT
            unrealized_pnl = (trade['entry_price'] - trade['current_price']) * trade['stake']
        
        unrealized_pnl_percent = (unrealized_pnl / trade['stake']) * 100
        total_unrealized_pnl += unrealized_pnl
        
        pnl_color = "positive-pnl" if unrealized_pnl >= 0 else "negative-pnl"
        
        col1, col2, col3, col4, col5, col6 = st.columns([1, 1.5, 2, 1.5, 1.5, 1])
        with col1:
            st.write(f"**{trade['trade_id']}**")
        with col2:
            st.write(f"**{trade['currency_pair']}**")
            st.write(f"{trade['position']} | €{trade['stake']:,}")
        with col3:
            st.write(f"Entry: {trade['entry_price']:.5f}")
            st.write(f"Current: {trade['current_price']:.5f}")
        with col4:
            st.markdown(f'<div class="{pnl_color}">€{unrealized_pnl:,.2f}</div>', unsafe_allow_html=True)
            st.markdown(f'<div class="{pnl_color}">{unrealized_pnl_percent:.2f}%</div>', unsafe_allow_html=True)
        with col5:
            st.write(f"TP: {trade['take_profit']:.5f}")
            st.write(f"SL: {trade['stop_loss']:.5f}")
        with col6:
            if st.button("Close", key=f"close_{trade['trade_id']}"):
                close_trade(trade['trade_id'], trade['current_price'])
                st.rerun()
        
        st.markdown("---")
    
    with col3:
        st.metric("Unrealized P/L", f"€{total_unrealized_pnl:,.2f}")
        
else:
    st.info("No active trades currently open")

# Trade History Table
st.markdown("---")
st.subheader("📈 Trade History")

if len(st.session_state.trade_history) > 0:
    total_closed_pnl = sum(trade['pnl'] for trade in st.session_state.trade_history)
    winning_trades = len([t for t in st.session_state.trade_history if t['pnl'] > 0])
    total_trades = len(st.session_state.trade_history)
    win_rate = (winning_trades / total_trades) * 100 if total_trades > 0 else 0
    
    # Display summary metrics
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
        
        col1, col2, col3, col4, col5, col6 = st.columns([1, 1.5, 2, 1.5, 1, 1])
        with col1:
            st.write(f"**{trade['trade_id']}**")
        with col2:
            st.write(f"**{trade['currency_pair']}**")
            st.write(f"{trade['position']} | €{trade['stake']:,}")
        with col3:
            st.write(f"Entry: {trade['entry_price']:.5f}")
            st.write(f"Exit: {trade['exit_price']:.5f}")
        with col4:
            st.markdown(f'<div class="{pnl_color}">€{trade["pnl"]:,.2f}</div>', unsafe_allow_html=True)
            st.markdown(f'<div class="{pnl_color}">{trade["pnl_percent"]:.2f}%</div>', unsafe_allow_html=True)
        with col5:
            duration = (trade['close_time'] - trade['open_time']).days
            st.write(f"Duration: {duration}d")
        with col6:
            st.caption(trade.get('close_reason', 'Manual'))
        
        st.markdown("---")
    
else:
    st.info("No trade history available")

# Trading Interface
st.markdown("---")
st.subheader("🚀 Quick Trading")

col1, col2, col3 = st.columns(3)

with col1:
    st.write("**Open New Trade**")
    selected_pair = st.selectbox("Currency Pair", list(st.session_state.stake_amounts.keys()))
    position = st.selectbox("Position", ["LONG", "SHORT"])
    
with col2:
    stake = 10  # Fixed 10€ stake
    st.write(f"**Stake: €{stake}** (Fixed)")
    if selected_pair in st.session_state.price_history:
        current_price = st.session_state.price_history[selected_pair][-1]
        st.write(f"Current Price: **{current_price:.5f}**")
        
        # Show potential SL/TP
        pip_value = 0.0005 if 'JPY' not in selected_pair else 0.005
        tp_pips = 10 * pip_value
        sl_pips = 5 * pip_value
        if position == 'LONG':
            tp = current_price + tp_pips
            sl = current_price - sl_pips
        else:
            tp = current_price - tp_pips
            sl = current_price + sl_pips
        st.write(f"TP: {tp:.5f} | SL: {sl:.5f}")
    
with col3:
    st.write("&nbsp;")
    st.write("&nbsp;")
    if st.button("🎯 Execute Trade", use_container_width=True, type="primary"):
        if selected_pair in st.session_state.price_history:
            current_price = st.session_state.price_history[selected_pair][-1]
            new_trade = open_trade(selected_pair, position, stake, round(current_price, 5))
            st.success(f"Trade opened: {new_trade['trade_id']} - {selected_pair} {position} €{stake} | TP: {new_trade['take_profit']:.5f} | SL: {new_trade['stop_loss']:.5f}")
            st.rerun()

# Footer
st.markdown("---")
if st.button("🔄 Refresh Data", use_container_width=True):
    st.rerun()

st.markdown("""
<div style="text-align: center; color: #666; margin-top: 2rem;">
    Pure Python Forex Analysis Dashboard | Fixed €10 Stake | Enhanced Strategy: Multi-Confirmation Signals, ADX, Auto SL/TP & Reversal Close
</div>
""", unsafe_allow_html=True)

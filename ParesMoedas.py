import streamlit as st
import pandas as pd
import numpy as np
import time
from datetime import datetime, timedelta

st.set_page_config(
    page_title="Forex Pro Bot - Complete Trading",
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
    .trade-table {
        background: white;
        border-radius: 10px;
        padding: 1rem;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        margin: 1rem 0;
    }
    .profit-positive {
        color: #00ff88;
        font-weight: bold;
    }
    .profit-negative {
        color: #ff4444;
        font-weight: bold;
    }
    .status-open {
        background: #e3f2fd;
        color: #1976d2;
        padding: 0.3rem 0.8rem;
        border-radius: 15px;
        font-size: 0.8rem;
        font-weight: bold;
    }
    .status-closed {
        background: #e8f5e9;
        color: #2e7d32;
        padding: 0.3rem 0.8rem;
        border-radius: 15px;
        font-size: 0.8rem;
        font-weight: bold;
    }
    .direction-buy {
        background: #e8f5e9;
        color: #2e7d32;
        padding: 0.3rem 0.8rem;
        border-radius: 15px;
        font-size: 0.8rem;
        font-weight: bold;
    }
    .direction-sell {
        background: #ffebee;
        color: #c62828;
        padding: 0.3rem 0.8rem;
        border-radius: 15px;
        font-size: 0.8rem;
        font-weight: bold;
    }
    .reason-tp {
        background: #e8f5e9;
        color: #2e7d32;
        padding: 0.3rem 0.8rem;
        border-radius: 15px;
        font-size: 0.8rem;
    }
    .reason-sl {
        background: #ffebee;
        color: #c62828;
        padding: 0.3rem 0.8rem;
        border-radius: 15px;
        font-size: 0.8rem;
    }
    .reason-manual {
        background: #fff3e0;
        color: #ef6c00;
        padding: 0.3rem 0.8rem;
        border-radius: 15px;
        font-size: 0.8rem;
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

# Default trading parameters
DEFAULT_PARAMS = {
    'initial_bank': 10000,
    'profit_target_pips': 20.0,
    'stop_loss_pips': 12.0,
    'trailing_stop': True,
    'trailing_stop_activation': 8.0,
    'break_even': True,
    'break_even_activation': 10.0,
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
    'selected_strategy': 'PROFESSIONAL_COMBO'
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
    pip_value = FOREX_PAIRS[pair]['pip_value']
    
    if direction == 'BUY':
        stop_loss_price = entry_price - (sl_pips * pip_value)
        take_profit_price = entry_price + (tp_pips * pip_value)
    else:
        stop_loss_price = entry_price + (sl_pips * pip_value)
        take_profit_price = entry_price - (tp_pips * pip_value)
    
    return stop_loss_price, take_profit_price

def execute_trade(pair, direction, entry_price):
    try:
        st.session_state.trade_counter += 1
        params = st.session_state.trading_params
        
        # Calculate risk amount
        risk_amount = (params['max_risk_percent'] / 100) * st.session_state.bank_balance
        
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
            'stake': risk_amount,
            'time': datetime.now(),
            'status': 'open',
            'profit_loss': 0,
            'profit_loss_pips': 0,
            'current_price': entry_price,
            'type': 'AUTO',
            'close_reason': None
        }
        
        st.session_state.open_trades.append(trade)
        st.session_state.bank_balance -= risk_amount
        return True
    except Exception as e:
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
            
            # Calculate dollar P&L
            profit_loss_dollar = pips * pip_value * 10000  # Simplified calculation
            
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
    
    for i, trade in enumerate(st.session_state.open_trades):
        if trade['status'] == 'open':
            current_price = st.session_state.current_prices.get(trade['pair'], trade['entry_price'])
            pip_value = FOREX_PAIRS[trade['pair']]['pip_value']
            
            # Calculate current P&L
            if trade['direction'] == 'BUY':
                pips = (current_price - trade['entry_price']) / pip_value
            else:
                pips = (trade['entry_price'] - current_price) / pip_value
            
            profit_loss_dollar = pips * pip_value * 10000
            
            trade['profit_loss'] = profit_loss_dollar
            trade['profit_loss_pips'] = pips
            trade['current_price'] = current_price
            
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
        
        signals, buy_indicators, sell_indicators, agreement = detect_trading_signals(df_with_indicators)
        
        current_price = df_with_indicators['close'].iloc[-1]
        st.session_state.current_prices[pair] = current_price
        
        all_signals[pair] = {
            'signals': signals,
            'buy_indicators': buy_indicators,
            'sell_indicators': sell_indicators,
            'agreement': agreement,
            'current_price': current_price
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
                if agreement == 'BUY' and signal_type == "BUY" and len(st.session_state.open_trades) < st.session_state.trading_params['max_open_trades']:
                    current_price = st.session_state.current_prices.get(pair, FOREX_PAIRS[pair]['base_price'])
                    
                    if execute_trade(pair, 'BUY', current_price):
                        auto_trades_executed.append(f"✅ BUY {pair} - {count} indicators")
                
                elif agreement == 'SELL' and signal_type == "SELL" and len(st.session_state.open_trades) < st.session_state.trading_params['max_open_trades']:
                    current_price = st.session_state.current_prices.get(pair, FOREX_PAIRS[pair]['base_price'])
                    
                    if execute_trade(pair, 'SELL', current_price):
                        auto_trades_executed.append(f"❌ SELL {pair} - {count} indicators")
                        
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
st.markdown('<h1 class="main-header">🤖 Forex Pro Bot - Complete Trading</h1>', unsafe_allow_html=True)

# Sidebar
with st.sidebar:
    st.header("🎯 Trading Controls")
    
    # Strategy Selection
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
    col1, col2 = st.columns(2)
    with col1:
        if st.button("🚀 Start Auto", use_container_width=True, type="primary"):
            st.session_state.auto_trading = True
            st.success("Auto Trading Started!")
    with col2:
        if st.button("🛑 Stop Auto", use_container_width=True):
            st.session_state.auto_trading = False
            st.warning("Auto Trading Stopped!")
    
    st.divider()
    
    # Quick Stats
    total_profit = sum(trade.get('profit_loss', 0) for trade in st.session_state.trade_history)
    st.write(f"**Bank:** ${st.session_state.bank_balance:.2f}")
    st.write(f"**Total P&L:** ${total_profit:.2f}")
    st.write(f"**Open Trades:** {len(st.session_state.open_trades)}")
    st.write(f"**Total Trades:** {len(st.session_state.trade_history)}")

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
        open_trades_data.append({
            'ID': trade['id'],
            'Pair': trade['pair'],
            'Direction': trade['direction'],
            'Entry Price': f"{trade['entry_price']:.4f}",
            'Current Price': f"{trade['current_price']:.4f}",
            'Stop Loss': f"{trade['stop_loss_price']:.4f}",
            'Take Profit': f"{trade['take_profit_price']:.4f}",
            'P&L ($)': f"${trade['profit_loss']:.2f}",
            'P&L (Pips)': f"{trade['profit_loss_pips']:.1f}",
            'Time Opened': trade['time'].strftime('%H:%M:%S'),
            'Actions': trade['id']
        })
    
    open_df = pd.DataFrame(open_trades_data)
    
    # Display the table with custom formatting
    st.markdown('<div class="trade-table">', unsafe_allow_html=True)
    
    # Convert DataFrame to HTML with custom styling
    def style_trade_dataframe(df):
        styled_df = df.copy()
        
        # Apply styling
        styles = []
        for _, row in styled_df.iterrows():
            # Color P&L column
            pnl_value = float(row['P&L ($)'].replace('$', ''))
            pnl_color = 'color: #00ff88;' if pnl_value >= 0 else 'color: #ff4444;'
            styles.append(['', '', '', '', '', '', '', pnl_color, '', '', ''])
        
        return styled_df.style.apply(lambda x: styles[df.index.get_loc(x.name)] if df.index.get_loc(x.name) < len(styles) else [''] * len(x), axis=1)
    
    # Display styled dataframe
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
    
    st.markdown('</div>', unsafe_allow_html=True)
else:
    st.info("No open trades currently")

# TRADE HISTORY TABLE
st.subheader("📈 Trade History & Performance")

if st.session_state.trade_history:
    # Create DataFrame for trade history
    history_data = []
    for trade in st.session_state.trade_history:
        # Determine close reason styling
        if trade['close_reason'] == 'TP':
            reason_class = "reason-tp"
            reason_text = "TAKE PROFIT"
        elif trade['close_reason'] == 'SL':
            reason_class = "reason-sl"
            reason_text = "STOP LOSS"
        else:
            reason_class = "reason-manual"
            reason_text = "MANUAL"
        
        history_data.append({
            'ID': trade['id'],
            'Pair': trade['pair'],
            'Direction': trade['direction'],
            'Entry Price': f"{trade['entry_price']:.4f}",
            'Exit Price': f"{trade['close_price']:.4f}",
            'P&L ($)': f"${trade['profit_loss']:.2f}",
            'P&L (Pips)': f"{trade['profit_loss_pips']:.1f}",
            'Close Reason': trade['close_reason'],
            'Duration': f"{(trade['close_time'] - trade['time']).seconds // 60}min",
            'Open Time': trade['time'].strftime('%H:%M'),
            'Close Time': trade['close_time'].strftime('%H:%M')
        })
    
    history_df = pd.DataFrame(history_data)
    
    st.markdown('<div class="trade-table">', unsafe_allow_html=True)
    
    # Display trade history with filters
    col1, col2, col3 = st.columns(3)
    with col1:
        show_trades = st.selectbox("Show", ["All Trades", "Winning Trades", "Losing Trades"])
    with col2:
        sort_by = st.selectbox("Sort By", ["Most Recent", "Highest Profit", "Lowest Profit"])
    with col3:
        items_per_page = st.selectbox("Items per page", [10, 25, 50])
    
    # Apply filters
    filtered_df = history_df.copy()
    
    if show_trades == "Winning Trades":
        filtered_df = filtered_df[filtered_df['P&L ($)'].str.contains(r'\$[0-9]', regex=True)]
        filtered_df = filtered_df[pd.to_numeric(filtered_df['P&L ($)'].str.replace('$', '')) > 0]
    elif show_trades == "Losing Trades":
        filtered_df = filtered_df[filtered_df['P&L ($)'].str.contains(r'\$[0-9]', regex=True)]
        filtered_df = filtered_df[pd.to_numeric(filtered_df['P&L ($)'].str.replace('$', '')) < 0]
    
    # Apply sorting
    if sort_by == "Most Recent":
        filtered_df = filtered_df.iloc[::-1]
    elif sort_by == "Highest Profit":
        filtered_df = filtered_df.iloc[pd.to_numeric(filtered_df['P&L ($)'].str.replace('$', '')).argsort()[::-1]]
    elif sort_by == "Lowest Profit":
        filtered_df = filtered_df.iloc[pd.to_numeric(filtered_df['P&L ($)'].str.replace('$', '')).argsort()]
    
    # Pagination
    total_trades = len(filtered_df)
    page_number = st.number_input("Page", min_value=1, max_value=max(1, (total_trades // items_per_page) + 1), value=1)
    start_idx = (page_number - 1) * items_per_page
    end_idx = min(start_idx + items_per_page, total_trades)
    
    st.write(f"Showing {start_idx + 1}-{end_idx} of {total_trades} trades")
    
    # Display the table
    st.dataframe(filtered_df.iloc[start_idx:end_idx], use_container_width=True, hide_index=True)
    
    st.markdown('</div>', unsafe_allow_html=True)
    
    # PERFORMANCE SUMMARY
    st.subheader("📊 Performance Summary")
    
    total_trades_count = len(st.session_state.trade_history)
    winning_trades = len([t for t in st.session_state.trade_history if t['profit_loss'] > 0])
    losing_trades = len([t for t in st.session_state.trade_history if t['profit_loss'] < 0])
    win_rate = (winning_trades / total_trades_count * 100) if total_trades_count > 0 else 0
    
    total_profit = sum(trade['profit_loss'] for trade in st.session_state.trade_history)
    avg_profit = total_profit / total_trades_count if total_trades_count > 0 else 0
    
    # SL/TP Statistics
    tp_trades = len([t for t in st.session_state.trade_history if t['close_reason'] == 'TP'])
    sl_trades = len([t for t in st.session_state.trade_history if t['close_reason'] == 'SL'])
    manual_trades = len([t for t in st.session_state.trade_history if t['close_reason'] == 'MANUAL'])
    
    col1, col2, col3, col4, col5 = st.columns(5)
    
    with col1:
        st.metric("Total Trades", total_trades_count)
    with col2:
        st.metric("Win Rate", f"{win_rate:.1f}%")
    with col3:
        st.metric("Total P&L", f"${total_profit:.2f}")
    with col4:
        st.metric("Avg Trade", f"${avg_profit:.2f}")
    with col5:
        st.metric("Best Trade", f"${max([t['profit_loss'] for t in st.session_state.trade_history], default=0):.2f}")
    
    # Detailed Statistics
    st.write("**Detailed Breakdown:**")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.write(f"**Winning Trades:** {winning_trades}")
        st.write(f"**Losing Trades:** {losing_trades}")
    with col2:
        st.write(f"**TP Hits:** {tp_trades} ({tp_trades/total_trades_count*100:.1f}%)")
        st.write(f"**SL Hits:** {sl_trades} ({sl_trades/total_trades_count*100:.1f}%)")
    with col3:
        st.write(f"**Manual Closes:** {manual_trades} ({manual_trades/total_trades_count*100:.1f}%)")
        st.write(f"**Largest Win:** ${max([t['profit_loss'] for t in st.session_state.trade_history if t['profit_loss'] > 0], default=0):.2f}")
    
else:
    st.info("No trade history yet. Trades will appear here once they are closed.")

# TRADING ACTIVITY
st.subheader("🎯 Recent Trading Activity")

# Show recent auto trade executions
if auto_trades_executed:
    st.write("**Recent Auto Trades:**")
    for trade in auto_trades_executed[-5:]:  # Show last 5
        if "BUY" in trade:
            st.success(trade)
        else:
            st.error(trade)

# Current signals
st.write("**Current Market Signals:**")
cols = st.columns(3)
for idx, pair in enumerate(trading_pairs[:3]):  # Show first 3 pairs
    with cols[idx]:
        signal_info = st.session_state.all_signals.get(pair, {})
        agreement = signal_info.get('agreement', 'NONE')
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
        
        st.markdown(f"""
        <div style="border: 2px solid {color}; border-radius: 10px; padding: 1rem; text-align: center;">
            <h4>{pair}</h4>
            <h3 style="color: {color};">{text}</h3>
            <p>Price: {current_price:.4f}</p>
        </div>
        """, unsafe_allow_html=True)

# Export functionality
if st.session_state.trade_history:
    st.divider()
    st.subheader("💾 Export Trade Data")
    
    # Convert trade history to CSV
    export_data = []
    for trade in st.session_state.trade_history:
        export_data.append({
            'Trade ID': trade['id'],
            'Pair': trade['pair'],
            'Direction': trade['direction'],
            'Entry Price': trade['entry_price'],
            'Exit Price': trade.get('close_price', ''),
            'P&L ($)': trade['profit_loss'],
            'P&L (Pips)': trade['profit_loss_pips'],
            'Stop Loss': trade['stop_loss_price'],
            'Take Profit': trade['take_profit_price'],
            'Open Time': trade['time'],
            'Close Time': trade.get('close_time', ''),
            'Close Reason': trade.get('close_reason', ''),
            'Status': trade['status']
        })
    
    export_df = pd.DataFrame(export_data)
    csv = export_df.to_csv(index=False)
    
    st.download_button(
        label="📥 Download Trade History as CSV",
        data=csv,
        file_name=f"forex_trade_history_{datetime.now().strftime('%Y%m%d_%H%M')}.csv",
        mime="text/csv",
        use_container_width=True
    )

# Auto-refresh
st.divider()
st.write("🔄 Auto-refreshing every 30 seconds...")
time.sleep(30)
st.rerun()

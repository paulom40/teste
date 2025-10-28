import streamlit as st
import pandas as pd
import numpy as np
import requests
import time
from datetime import datetime, timedelta
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Try to import yfinance
try:
    import yfinance as yf
    YFINANCE_AVAILABLE = True
except ImportError:
    YFINANCE_AVAILABLE = False

st.set_page_config(
    page_title="Forex Auto Trading Bot",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-align: center;
        margin-bottom: 1rem;
        font-weight: bold;
    }
    .prices-table {
        background: white;
        border-radius: 10px;
        padding: 1rem;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        margin: 1rem 0;
        border: 1px solid #e0e0e0;
    }
    .price-up {
        color: #00ff88;
        font-weight: bold;
    }
    .price-down {
        color: #ff4444;
        font-weight: bold;
    }
    .price-neutral {
        color: #666666;
    }
    .signal-buy {
        background: linear-gradient(135deg, #00ff88 0%, #00cc66 100%);
        color: white;
        padding: 0.3rem 0.8rem;
        border-radius: 20px;
        font-weight: bold;
        text-align: center;
        font-size: 0.9rem;
    }
    .signal-sell {
        background: linear-gradient(135deg, #ff4444 0%, #cc0000 100%);
        color: white;
        padding: 0.3rem 0.8rem;
        border-radius: 20px;
        font-weight: bold;
        text-align: center;
        font-size: 0.9rem;
    }
    .signal-hold {
        background: linear-gradient(135deg, #808080 0%, #a0a0a0 100%);
        color: white;
        padding: 0.3rem 0.8rem;
        border-radius: 20px;
        font-weight: bold;
        text-align: center;
        font-size: 0.9rem;
    }
    .engulfing-buy {
        background: linear-gradient(135deg, #32CD32 0%, #228B22 100%);
        color: white;
        padding: 0.3rem 0.8rem;
        border-radius: 20px;
        font-weight: bold;
        text-align: center;
        font-size: 0.9rem;
        border: 2px solid #00FF00;
    }
    .engulfing-sell {
        background: linear-gradient(135deg, #FF4500 0%, #B22222 100%);
        color: white;
        padding: 0.3rem 0.8rem;
        border-radius: 20px;
        font-weight: bold;
        text-align: center;
        font-size: 0.9rem;
        border: 2px solid #FF0000;
    }
    .active-trade-profit {
        color: #00ff88;
        font-weight: bold;
        font-size: 1.1em;
    }
    .active-trade-loss {
        color: #ff4444;
        font-weight: bold;
        font-size: 1.1em;
    }
    .active-trade-neutral {
        color: #666666;
        font-weight: bold;
    }
    .tab-container {
        background: white;
        border-radius: 10px;
        padding: 1rem;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'trade_history' not in st.session_state:
    st.session_state.trade_history = pd.DataFrame(columns=[
        'Date', 'Pair', 'Direction', 'Entry Price', 'Exit Price', 
        'Quantity', 'P&L', 'P&L (€)', 'Status', 'Signal Strength', 
        'Signal Count', 'Stake (€)', 'Target Profit', 'Stop Loss', 'Engulfing Pattern'
    ])

if 'auto_trading' not in st.session_state:
    st.session_state.auto_trading = False

if 'open_positions' not in st.session_state:
    st.session_state.open_positions = {}

if 'scan_count' not in st.session_state:
    st.session_state.scan_count = 0

if 'last_scan_time' not in st.session_state:
    st.session_state.last_scan_time = datetime.now()

# Default settings
if 'stake_euros' not in st.session_state:
    st.session_state.stake_euros = 100.0
if 'target_profit_pips' not in st.session_state:
    st.session_state.target_profit_pips = 30
if 'stop_loss_pips' not in st.session_state:
    st.session_state.stop_loss_pips = 20

# Realistic base prices for Forex pairs with current variations
FOREX_BASE_PRICES = {
    'EUR/USD': 1.08542, 'GBP/USD': 1.26518, 'USD/JPY': 148.53,
    'USD/CHF': 0.88325, 'AUD/USD': 0.65532, 'USD/CAD': 1.35567,
    'NZD/USD': 0.61045, 'EUR/GBP': 0.85792, 'EUR/JPY': 161.28,
    'GBP/JPY': 187.85, 'AUD/JPY': 97.32, 'USD/CNY': 7.25580
}

def get_realistic_price_data():
    """Generate realistic price data for all pairs"""
    pairs_data = []
    
    for pair, base_price in FOREX_BASE_PRICES.items():
        # Generate small random price movement
        movement = (np.random.random() - 0.5) * 0.002  # ±0.1%
        current_price = base_price * (1 + movement)
        change_percent = movement * 100
        
        # Generate realistic signals
        signal_weights = [0.35, 0.35, 0.3]  # BUY, SELL, HOLD probabilities
        signal = np.random.choice(['BUY', 'SELL', 'HOLD'], p=signal_weights)
        
        strength_weights = [0.2, 0.5, 0.3]  # STRONG, MODERATE, WEAK probabilities
        strength = np.random.choice(['STRONG', 'MODERATE', 'WEAK'], p=strength_weights)
        
        engulfing_weights = [0.15, 0.15, 0.7]  # BULLISH, BEARISH, NONE probabilities
        engulfing = np.random.choice(['BULLISH', 'BEARISH', 'NONE'], p=engulfing_weights)
        
        signal_count = np.random.randint(2, 5)
        
        pairs_data.append({
            'pair': pair,
            'price': float(current_price),  # Ensure it's a Python float
            'change_percent': float(change_percent),  # Ensure it's a Python float
            'signal': signal,
            'strength': strength,
            'engulfing': engulfing,
            'signal_count': signal_count
        })
    
    return pairs_data

def display_real_time_prices():
    """Display real-time prices in a beautiful table"""
    st.markdown("### 📊 LIVE FOREX PRICES")
    
    # Refresh button
    col1, col2, col3 = st.columns([3, 1, 1])
    with col2:
        if st.button("🔄 Refresh Prices", use_container_width=True, key="refresh_prices"):
            st.rerun()
    with col3:
        if st.button("📊 Scan Signals", use_container_width=True, key="scan_signals"):
            st.session_state.scan_count += 1
            st.rerun()
    
    # Get price data
    with st.spinner("🔄 Updating market data..."):
        pairs_data = get_realistic_price_data()
    
    # Create the prices table using Streamlit components instead of raw HTML
    st.markdown('<div class="prices-table">', unsafe_allow_html=True)
    
    # Create a dataframe for display
    display_data = []
    for data in pairs_data:
        # Determine change arrow and color
        change = data['change_percent']
        if change > 0:
            change_display = f"↗ +{change:.2f}%"
            change_class = "price-up"
        elif change < 0:
            change_display = f"↘ {change:.2f}%"
            change_class = "price-down"
        else:
            change_display = f"→ {change:.2f}%"
            change_class = "price-neutral"
        
        # Format price
        if 'JPY' in data['pair'] or 'CNY' in data['pair']:
            price_str = f"{data['price']:.2f}"
        else:
            price_str = f"{data['price']:.5f}"
        
        display_data.append({
            'Pair': data['pair'],
            'Current Price': price_str,
            'Change %': change_display,
            'Signal': data['signal'],
            'Engulfing': data['engulfing'],
            'Change Class': change_class
        })
    
    # Display each row with proper styling
    for row in display_data:
        col1, col2, col3, col4, col5, col6 = st.columns([2, 2, 2, 2, 2, 2])
        
        with col1:
            st.markdown(f"**{row['Pair']}**")
        
        with col2:
            st.markdown(f"`{row['Current Price']}`")
        
        with col3:
            st.markdown(f"<span class='{row['Change Class']}'>{row['Change %']}</span>", unsafe_allow_html=True)
        
        with col4:
            if row['Signal'] == 'BUY':
                st.markdown('<div class="signal-buy">BUY</div>', unsafe_allow_html=True)
            elif row['Signal'] == 'SELL':
                st.markdown('<div class="signal-sell">SELL</div>', unsafe_allow_html=True)
            else:
                st.markdown('<div class="signal-hold">HOLD</div>', unsafe_allow_html=True)
        
        with col5:
            if row['Engulfing'] == 'BULLISH':
                st.markdown('<div class="engulfing-buy">BULLISH</div>', unsafe_allow_html=True)
            elif row['Engulfing'] == 'BEARISH':
                st.markdown('<div class="engulfing-sell">BEARISH</div>', unsafe_allow_html=True)
            else:
                st.markdown('No Pattern')
        
        with col6:
            if st.button("TRADE", key=f"trade_{row['Pair']}", use_container_width=True):
                # Find the matching data for this pair
                pair_data = next((p for p in pairs_data if p['pair'] == row['Pair']), None)
                if pair_data:
                    result = execute_trade(pair_data, st.session_state.stake_euros)
                    st.success(result)
    
    st.markdown('</div>', unsafe_allow_html=True)
    
    return pairs_data

def execute_trade(signal_data, stake_eur):
    """Execute a trade based on signal"""
    pair = signal_data['pair']
    direction = signal_data['signal']
    price = signal_data['price']
    
    # Simple trade simulation
    pip_value = 10
    pip_size = 0.0001
    quantity = stake_eur / 100
    
    # Higher win rate for strong signals
    base_win_rate = 0.7 if signal_data['strength'] == 'STRONG' else 0.6
    is_win = np.random.random() < base_win_rate
    
    if is_win:
        pnl = st.session_state.target_profit_pips * pip_value * quantity
        exit_price = price + (st.session_state.target_profit_pips * pip_size if direction == 'BUY' 
                            else -st.session_state.target_profit_pips * pip_size)
    else:
        pnl = -st.session_state.stop_loss_pips * pip_value * quantity
        exit_price = price - (st.session_state.stop_loss_pips * pip_size if direction == 'BUY' 
                            else -st.session_state.stop_loss_pips * pip_size)
    
    # Add to trade history
    new_trade = pd.DataFrame([{
        'Date': datetime.now(),
        'Pair': pair,
        'Direction': direction,
        'Entry Price': price,
        'Exit Price': exit_price,
        'Quantity': quantity,
        'P&L': pnl,
        'P&L (€)': pnl,
        'Status': 'CLOSED',
        'Signal Strength': signal_data['strength'],
        'Signal Count': signal_data['signal_count'],
        'Stake (€)': stake_eur,
        'Target Profit': st.session_state.target_profit_pips,
        'Stop Loss': st.session_state.stop_loss_pips,
        'Engulfing Pattern': signal_data['engulfing']
    }])
    
    st.session_state.trade_history = pd.concat([st.session_state.trade_history, new_trade], ignore_index=True)
    
    return f"✅ Trade executed: {direction} {pair} at {price:.5f}"

def open_trade_position(signal_data, stake_eur):
    """Open a new trade position (active trade)"""
    pair = signal_data['pair']
    
    # Ensure all required fields are present with default values
    position_data = {
        'pair': pair,
        'direction': signal_data.get('signal', 'BUY'),
        'entry_price': signal_data.get('price', 1.0),
        'entry_time': datetime.now(),
        'stake_eur': stake_eur,
        'target_profit': st.session_state.target_profit_pips,
        'stop_loss': st.session_state.stop_loss_pips,
        'signal_strength': signal_data.get('strength', 'MODERATE'),
        'engulfing_pattern': signal_data.get('engulfing', 'NONE'),
        'current_pnl': 0.0,
        'current_pnl_percent': 0.0,
        'current_price': signal_data.get('price', 1.0)
    }
    
    # Store active trade in open_positions
    st.session_state.open_positions[pair] = position_data
    
    return f"🟢 Position opened: {signal_data['signal']} {pair} at {signal_data['price']:.5f}"

def close_trade_position(pair, current_price):
    """Close an active trade position"""
    if pair in st.session_state.open_positions:
        position = st.session_state.open_positions[pair]
        
        # Ensure all required fields exist with safe defaults
        stake_eur = position.get('stake_eur', st.session_state.stake_euros)
        entry_price = position.get('entry_price', 1.0)
        direction = position.get('direction', 'BUY')
        
        # Calculate final P&L
        pip_value = 10
        quantity = stake_eur / 100
        
        if direction == 'BUY':
            pnl_pips = (current_price - entry_price) / 0.0001
        else:
            pnl_pips = (entry_price - current_price) / 0.0001
        
        pnl = pnl_pips * pip_value * quantity
        
        # Add to trade history
        new_trade = pd.DataFrame([{
            'Date': datetime.now(),
            'Pair': pair,
            'Direction': direction,
            'Entry Price': entry_price,
            'Exit Price': current_price,
            'Quantity': quantity,
            'P&L': pnl,
            'P&L (€)': pnl,
            'Status': 'CLOSED',
            'Signal Strength': position.get('signal_strength', 'MODERATE'),
            'Signal Count': 4,  # Assuming 4 signals for active trades
            'Stake (€)': stake_eur,
            'Target Profit': position.get('target_profit', st.session_state.target_profit_pips),
            'Stop Loss': position.get('stop_loss', st.session_state.stop_loss_pips),
            'Engulfing Pattern': position.get('engulfing_pattern', 'NONE')
        }])
        
        st.session_state.trade_history = pd.concat([st.session_state.trade_history, new_trade], ignore_index=True)
        
        # Remove from open positions
        del st.session_state.open_positions[pair]
        
        return f"🔴 Position closed: {direction} {pair} | P&L: €{pnl:.2f}"
    
    return f"❌ No active position found for {pair}"

def update_active_trades_pnl():
    """Update P&L for active trades based on current prices"""
    for pair, position in st.session_state.open_positions.items():
        try:
            # Get current price for the pair
            current_price_data = next((p for p in get_realistic_price_data() if p['pair'] == pair), None)
            if current_price_data:
                current_price = current_price_data['price']
                
                # Ensure all required fields exist with safe defaults
                stake_eur = position.get('stake_eur', st.session_state.stake_euros)
                entry_price = position.get('entry_price', 1.0)
                direction = position.get('direction', 'BUY')
                
                # Calculate current P&L
                pip_value = 10
                quantity = stake_eur / 100
                
                if direction == 'BUY':
                    pnl_pips = (current_price - entry_price) / 0.0001
                else:
                    pnl_pips = (entry_price - current_price) / 0.0001
                
                pnl = pnl_pips * pip_value * quantity
                pnl_percent = (pnl / stake_eur) * 100 if stake_eur > 0 else 0
                
                # Update position with safe field access
                position['current_price'] = current_price
                position['current_pnl'] = pnl
                position['current_pnl_percent'] = pnl_percent
                position['duration'] = datetime.now() - position.get('entry_time', datetime.now())
                
                # Ensure stake_eur is set
                if 'stake_eur' not in position:
                    position['stake_eur'] = stake_eur
                    
        except Exception as e:
            # If there's any error, set default values
            position['current_pnl'] = 0.0
            position['current_pnl_percent'] = 0.0
            position['duration'] = timedelta(0)
            continue

def display_active_trades():
    """Display active trades in a tab"""
    update_active_trades_pnl()
    
    if not st.session_state.open_positions:
        st.info("📭 No active trades. Open positions will appear here.")
        return
    
    st.subheader(f"🟢 ACTIVE TRADES ({len(st.session_state.open_positions)})")
    
    for pair, position in st.session_state.open_positions.items():
        # Use safe field access with defaults
        direction = position.get('direction', 'BUY')
        stake_eur = position.get('stake_eur', st.session_state.stake_euros)
        entry_price = position.get('entry_price', 1.0)
        current_price = position.get('current_price', entry_price)
        current_pnl = position.get('current_pnl', 0.0)
        current_pnl_percent = position.get('current_pnl_percent', 0.0)
        target_profit = position.get('target_profit', st.session_state.target_profit_pips)
        stop_loss = position.get('stop_loss', st.session_state.stop_loss_pips)
        duration = position.get('duration', timedelta(0))
        
        col1, col2, col3, col4, col5, col6, col7, col8 = st.columns([2, 1, 2, 2, 2, 2, 2, 2])
        
        with col1:
            st.markdown(f"**{pair}**")
            st.markdown(f"*{direction}*")
        
        with col2:
            if current_pnl > 0:
                st.markdown(f"<div class='active-trade-profit'>€{current_pnl:.2f}</div>", unsafe_allow_html=True)
                st.markdown(f"<div class='active-trade-profit'>{current_pnl_percent:.1f}%</div>", unsafe_allow_html=True)
            elif current_pnl < 0:
                st.markdown(f"<div class='active-trade-loss'>€{current_pnl:.2f}</div>", unsafe_allow_html=True)
                st.markdown(f"<div class='active-trade-loss'>{current_pnl_percent:.1f}%</div>", unsafe_allow_html=True)
            else:
                st.markdown(f"<div class='active-trade-neutral'>€{current_pnl:.2f}</div>", unsafe_allow_html=True)
                st.markdown(f"<div class='active-trade-neutral'>{current_pnl_percent:.1f}%</div>", unsafe_allow_html=True)
        
        with col3:
            st.markdown(f"**Entry**")
            st.markdown(f"`{entry_price:.5f}`")
        
        with col4:
            st.markdown(f"**Current**")
            st.markdown(f"`{current_price:.5f}`")
        
        with col5:
            st.markdown(f"**Stake**")
            st.markdown(f"€{stake_eur:.0f}")
        
        with col6:
            st.markdown(f"**TP/SL**")
            st.markdown(f"{target_profit}/{stop_loss}")
        
        with col7:
            st.markdown(f"**Duration**")
            hours = duration.total_seconds() // 3600
            minutes = (duration.total_seconds() % 3600) // 60
            st.markdown(f"{int(hours)}h {int(minutes)}m")
        
        with col8:
            if st.button("CLOSE", key=f"close_{pair}", use_container_width=True):
                result = close_trade_position(pair, current_price)
                st.success(result)
                st.rerun()
        
        st.markdown("---")

def display_trade_history():
    """Display trade history in a tab"""
    if st.session_state.trade_history.empty:
        st.info("📊 No trade history yet. Closed trades will appear here.")
        return
    
    # Display summary metrics
    total_trades = len(st.session_state.trade_history)
    winning_trades = len(st.session_state.trade_history[st.session_state.trade_history['P&L'] > 0])
    win_rate = (winning_trades / total_trades * 100) if total_trades > 0 else 0
    total_pnl = st.session_state.trade_history['P&L'].sum()
    
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Total Trades", total_trades)
    with col2:
        st.metric("Win Rate", f"{win_rate:.1f}%")
    with col3:
        st.metric("Total P&L", f"€{total_pnl:.2f}")
    with col4:
        st.metric("Avg P&L/Trade", f"€{(total_pnl/total_trades):.2f}" if total_trades > 0 else "€0.00")
    
    # Display trade history table
    st.dataframe(
        st.session_state.trade_history.sort_values('Date', ascending=False).head(15),
        use_container_width=True
    )

def main():
    # Header
    st.markdown('<h1 class="main-header">🌍 FOREX AUTO TRADING BOT</h1>', unsafe_allow_html=True)
    st.markdown('<p style="text-align: center; color: #666; font-size: 1.2rem;">Advanced 4-Signal Agreement System with Engulfing Patterns</p>', unsafe_allow_html=True)
    
    # Data source info
    if YFINANCE_AVAILABLE:
        st.success("✅ Connected to Yahoo Finance - Using Real Market Data")
    else:
        st.warning("⚠️ Using Simulated Data - Install yfinance for real market data: `pip install yfinance`")
    
    # Display real-time prices table
    analysis_data = display_real_time_prices()
    
    # Sidebar
    with st.sidebar:
        st.title("⚙️ Trading Configuration")
        
        st.subheader("💰 Trade Settings")
        st.session_state.stake_euros = st.number_input("Stake Amount (€)", 10.0, 10000.0, 100.0, 50.0)
        
        col1, col2 = st.columns(2)
        with col1:
            st.session_state.target_profit_pips = st.number_input("Target (pips)", 5, 100, 30, 5)
        with col2:
            st.session_state.stop_loss_pips = st.number_input("Stop Loss (pips)", 5, 50, 20, 5)
        
        risk_reward = st.session_state.target_profit_pips / st.session_state.stop_loss_pips
        st.metric("Risk/Reward Ratio", f"{risk_reward:.1f}:1")
        
        st.subheader("🤖 Auto Trading")
        col1, col2 = st.columns(2)
        with col1:
            if st.button("🚀 START" if not st.session_state.auto_trading else "🛑 STOP", 
                        use_container_width=True, type="primary" if not st.session_state.auto_trading else "secondary"):
                st.session_state.auto_trading = not st.session_state.auto_trading
                st.rerun()
        
        with col2:
            if st.button("🔍 SCAN NOW", use_container_width=True):
                st.session_state.scan_count += 1
                st.session_state.last_scan_time = datetime.now()
                st.rerun()
        
        if st.session_state.auto_trading:
            st.markdown('<div style="background: linear-gradient(135deg, #00ff88 0%, #00cc66 100%); color: white; padding: 1rem; border-radius: 10px; text-align: center; font-weight: bold;">AUTO TRADING ACTIVE</div>', unsafe_allow_html=True)
            st.info(f"Scans: {st.session_state.scan_count}")
        else:
            st.warning("Auto Trading: OFF")
        
        # Quick stats
        st.subheader("📈 Quick Stats")
        total_trades = len(st.session_state.trade_history)
        active_trades = len(st.session_state.open_positions)
        
        if total_trades > 0 or active_trades > 0:
            winning_trades = len(st.session_state.trade_history[st.session_state.trade_history['P&L'] > 0])
            win_rate = (winning_trades / total_trades) * 100 if total_trades > 0 else 0
            total_pnl = st.session_state.trade_history['P&L'].sum()
            
            st.metric("Active Trades", active_trades)
            st.metric("Total Trades", total_trades)
            st.metric("Win Rate", f"{win_rate:.1f}%")
            st.metric("Total P&L", f"€{total_pnl:.2f}")
        else:
            st.info("No trades yet")
    
    # Trading Opportunities Section
    st.markdown("---")
    st.subheader("🎯 ACTIVE TRADING OPPORTUNITIES")
    
    # Filter only BUY/SELL signals
    trading_opportunities = [data for data in analysis_data if data['signal'] in ['BUY', 'SELL']]
    
    if trading_opportunities:
        st.success(f"🎯 Found {len(trading_opportunities)} trading opportunities!")
        
        for opportunity in trading_opportunities:
            col1, col2, col3, col4, col5, col6, col7 = st.columns([2, 1, 1, 1, 1, 2, 2])
            
            with col1:
                st.write(f"**{opportunity['pair']}**")
                st.write(f"Price: `{opportunity['price']:.5f}`")
                st.write(f"Change: `{opportunity['change_percent']:+.2f}%`")
            
            with col2:
                if opportunity['signal'] == 'BUY':
                    st.markdown('<div class="signal-buy">BUY</div>', unsafe_allow_html=True)
                else:
                    st.markdown('<div class="signal-sell">SELL</div>', unsafe_allow_html=True)
            
            with col3:
                st.write(f"**{opportunity['strength']}**")
            
            with col4:
                st.write(f"Signals: **{opportunity['signal_count']}/4**")
            
            with col5:
                if opportunity['engulfing'] == 'BULLISH':
                    st.markdown('<div class="engulfing-buy">ENGULFING</div>', unsafe_allow_html=True)
                elif opportunity['engulfing'] == 'BEARISH':
                    st.markdown('<div class="engulfing-sell">ENGULFING</div>', unsafe_allow_html=True)
                else:
                    st.write("No Pattern")
            
            with col6:
                if st.button(f"OPEN TRADE", key=f"open_{opportunity['pair']}", use_container_width=True):
                    result = open_trade_position(opportunity, st.session_state.stake_euros)
                    st.success(result)
                    st.rerun()
            
            with col7:
                if st.button(f"QUICK TRADE", key=f"quick_{opportunity['pair']}", use_container_width=True, type="primary"):
                    result = execute_trade(opportunity, st.session_state.stake_euros)
                    st.success(result)
                    st.rerun()
            
            st.markdown("---")
    else:
        st.info("No strong trading opportunities detected at the moment. The system will continue monitoring...")
    
    # Trade History & Active Trades Section with Tabs
    st.markdown("---")
    st.subheader("📋 TRADE MANAGEMENT")
    
    # Create tabs
    tab1, tab2 = st.tabs(["🟢 ACTIVE TRADES", "📊 TRADE HISTORY"])
    
    with tab1:
        display_active_trades()
        
        # Clear all positions button
        if st.session_state.open_positions:
            if st.button("🗑️ Close All Positions", use_container_width=True, type="secondary"):
                for pair in list(st.session_state.open_positions.keys()):
                    position = st.session_state.open_positions[pair]
                    current_price = position.get('current_price', FOREX_BASE_PRICES.get(pair, 1.0))
                    close_trade_position(pair, current_price)
                st.success("All positions closed!")
                st.rerun()
    
    with tab2:
        display_trade_history()
        
        # Clear history button
        if not st.session_state.trade_history.empty:
            if st.button("🗑️ Clear Trade History", use_container_width=True):
                st.session_state.trade_history = pd.DataFrame(columns=st.session_state.trade_history.columns)
                st.success("Trade history cleared!")
                st.rerun()

if __name__ == "__main__":
    main()

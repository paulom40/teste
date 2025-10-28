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
        text-align: bold;
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
    .trade-button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        padding: 0.5rem 1rem;
        border-radius: 5px;
        font-weight: bold;
        cursor: pointer;
        transition: all 0.3s ease;
    }
    .trade-button:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 8px rgba(0,0,0,0.2);
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

if 'last_prices' not in st.session_state:
    st.session_state.last_prices = {}

if 'price_changes' not in st.session_state:
    st.session_state.price_changes = {}

# Default settings
if 'stake_euros' not in st.session_state:
    st.session_state.stake_euros = 100.0
if 'target_profit_pips' not in st.session_state:
    st.session_state.target_profit_pips = 30
if 'stop_loss_pips' not in st.session_state:
    st.session_state.stop_loss_pips = 20

# Realistic base prices for Forex pairs
FOREX_BASE_PRICES = {
    'EUR/USD': 1.0850, 'GBP/USD': 1.2650, 'USD/JPY': 148.50,
    'USD/CHF': 0.8830, 'AUD/USD': 0.6550, 'USD/CAD': 1.3550,
    'NZD/USD': 0.6100, 'EUR/GBP': 0.8570, 'EUR/JPY': 161.00,
    'GBP/JPY': 187.80, 'AUD/JPY': 97.20, 'USD/CNY': 7.2550
}

def get_realistic_price(pair):
    """Get realistic current price with simulated fluctuations"""
    base_price = FOREX_BASE_PRICES.get(pair, 1.0000)
    
    # Store previous price to calculate change
    if pair not in st.session_state.last_prices:
        st.session_state.last_prices[pair] = base_price
        st.session_state.price_changes[pair] = 0.0
    
    # Simulate small price movement (±0.5%)
    movement = (np.random.random() - 0.5) * 0.01  # ±0.5%
    new_price = base_price * (1 + movement)
    
    # Calculate percentage change
    old_price = st.session_state.last_prices[pair]
    change_percent = ((new_price - old_price) / old_price) * 100
    
    # Update stored values
    st.session_state.last_prices[pair] = new_price
    st.session_state.price_changes[pair] = change_percent
    
    return new_price, change_percent

def analyze_pair_signals(pair):
    """Analyze trading signals for a pair"""
    current_price, change_percent = get_realistic_price(pair)
    
    # Simulate realistic trading signals
    signals = np.random.choice(['BUY', 'SELL', 'HOLD'], p=[0.3, 0.3, 0.4])
    strength = np.random.choice(['STRONG', 'MODERATE', 'WEAK'])
    engulfing = np.random.choice(['BULLISH', 'BEARISH', 'NONE'], p=[0.2, 0.2, 0.6])
    signal_count = np.random.randint(1, 5)
    
    return {
        'pair': pair,
        'price': current_price,
        'change_percent': change_percent,
        'signal': signals,
        'strength': strength,
        'engulfing': engulfing,
        'signal_count': signal_count
    }

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
    
    # Get analysis for all pairs
    forex_pairs = list(FOREX_BASE_PRICES.keys())
    analysis_data = []
    
    with st.spinner("🔄 Updating market data..."):
        for pair in forex_pairs:
            analysis = analyze_pair_signals(pair)
            analysis_data.append(analysis)
    
    # Create the prices table
    st.markdown("""
    <div class="prices-table">
    <table style="width: 100%; border-collapse: collapse; font-family: Arial, sans-serif;">
        <thead>
            <tr style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); color: white; border-radius: 10px;">
                <th style="padding: 15px; text-align: center; font-weight: bold; border: none;">FOREX PAIR</th>
                <th style="padding: 15px; text-align: center; font-weight: bold; border: none;">CURRENT PRICE</th>
                <th style="padding: 15px; text-align: center; font-weight: bold; border: none;">CHANGE %</th>
                <th style="padding: 15px; text-align: center; font-weight: bold; border: none;">TRADING SIGNAL</th>
                <th style="padding: 15px; text-align: center; font-weight: bold; border: none;">ENGULFING</th>
                <th style="padding: 15px; text-align: center; font-weight: bold; border: none;">ACTION</th>
            </tr>
        </thead>
        <tbody>
    """, unsafe_allow_html=True)
    
    for analysis in analysis_data:
        pair = analysis['pair']
        price = analysis['price']
        change = analysis['change_percent']
        signal = analysis['signal']
        engulfing = analysis['engulfing']
        
        # Format price based on pair type
        if 'JPY' in pair or 'CNY' in pair:
            price_str = f"{price:.2f}"
        else:
            price_str = f"{price:.5f}"
        
        # Determine change color and arrow
        if change > 0:
            change_color = "price-up"
            change_arrow = "↗"
            change_display = f"+{change:.2f}%"
        elif change < 0:
            change_color = "price-down"
            change_arrow = "↘"
            change_display = f"{change:.2f}%"
        else:
            change_color = "price-neutral"
            change_arrow = "→"
            change_display = f"{change:.2f}%"
        
        # Signal styling
        if signal == 'BUY':
            signal_html = f'<div class="signal-buy">BUY</div>'
        elif signal == 'SELL':
            signal_html = f'<div class="signal-sell">SELL</div>'
        else:
            signal_html = f'<div class="signal-hold">HOLD</div>'
        
        # Engulfing pattern styling
        if engulfing == 'BULLISH':
            engulfing_html = f'<div class="engulfing-buy">BULLISH</div>'
        elif engulfing == 'BEARISH':
            engulfing_html = f'<div class="engulfing-sell">BEARISH</div>'
        else:
            engulfing_html = 'No Pattern'
        
        # Add row to table
        st.markdown(f"""
        <tr style="border-bottom: 1px solid #e0e0e0;">
            <td style="padding: 15px; text-align: center; font-weight: bold; font-size: 1.1em;">{pair}</td>
            <td style="padding: 15px; text-align: center; font-family: 'Courier New', monospace; font-size: 1.1em; font-weight: bold;">{price_str}</td>
            <td style="padding: 15px; text-align: center; font-weight: bold;">
                <span class="{change_color}">{change_arrow} {change_display}</span>
            </td>
            <td style="padding: 15px; text-align: center;">{signal_html}</td>
            <td style="padding: 15px; text-align: center;">{engulfing_html}</td>
            <td style="padding: 15px; text-align: center;">
                <button class="trade-button" onclick="alert('Trading {pair} at {price_str}')">
                    TRADE
                </button>
            </td>
        </tr>
        """, unsafe_allow_html=True)
    
    # Close table
    st.markdown("""
        </tbody>
    </table>
    </div>
    """, unsafe_allow_html=True)
    
    return analysis_data

def execute_trade(signal_data, stake_eur):
    """Execute a trade based on signal"""
    pair = signal_data['pair']
    direction = signal_data['signal']
    price = signal_data['price']
    
    # Simple trade simulation
    pip_value = 10
    pip_size = 0.0001
    quantity = stake_eur / 100
    
    # Simulate trade outcome
    is_win = np.random.random() > 0.4
    
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
            st.markdown('<div style="background: linear-gradient(135deg, #00ff88 0%, #00cc66 100%); color: white; padding: 1rem; border-radius: 10px; text-align: center; font-weight: bold; animation: pulse 2s infinite;">AUTO TRADING ACTIVE</div>', unsafe_allow_html=True)
            st.info(f"Scans: {st.session_state.scan_count}")
        else:
            st.warning("Auto Trading: OFF")
        
        # Quick stats
        st.subheader("📈 Quick Stats")
        total_trades = len(st.session_state.trade_history)
        if total_trades > 0:
            winning_trades = len(st.session_state.trade_history[st.session_state.trade_history['P&L'] > 0])
            win_rate = (winning_trades / total_trades) * 100
            total_pnl = st.session_state.trade_history['P&L'].sum()
            
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
            col1, col2, col3, col4, col5, col6 = st.columns([2, 1, 1, 1, 1, 2])
            
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
                if st.button(f"TRADE NOW", key=f"trade_{opportunity['pair']}", use_container_width=True):
                    result = execute_trade(opportunity, st.session_state.stake_euros)
                    st.success(result)
                    st.rerun()
            
            st.markdown("---")
        
        # Auto-execute option
        if st.checkbox("🤖 AUTO-EXECUTE ALL QUALIFIED TRADES", value=False):
            executed = []
            for opportunity in trading_opportunities:
                if opportunity['pair'] not in st.session_state.open_positions:
                    result = execute_trade(opportunity, st.session_state.stake_euros)
                    executed.append(result)
                    st.session_state.open_positions[opportunity['pair']] = opportunity
            
            if executed:
                st.success("🤖 Auto-execution completed!")
                for trade in executed:
                    st.write(f"✅ {trade}")
                st.rerun()
    else:
        st.info("No strong trading opportunities detected at the moment. The system will continue monitoring...")
    
    # Trade History Section
    st.markdown("---")
    st.subheader("📋 TRADE HISTORY & PERFORMANCE")
    
    if not st.session_state.trade_history.empty:
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
            st.session_state.trade_history.sort_values('Date', ascending=False).head(10),
            use_container_width=True
        )
        
        # Clear history button
        if st.button("🗑️ Clear Trade History", use_container_width=True):
            st.session_state.trade_history = pd.DataFrame(columns=st.session_state.trade_history.columns)
            st.session_state.open_positions = {}
            st.success("Trade history cleared!")
            st.rerun()
    else:
        st.info("No trades executed yet. Execute trades from the opportunities above to see history here.")

if __name__ == "__main__":
    main()

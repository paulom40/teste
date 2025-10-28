import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
from datetime import datetime, timedelta
import time

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
    .stake-card {
        background-color: #e8f5e9;
        border-radius: 10px;
        padding: 1rem;
        margin: 0.5rem 0;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        border-left: 4px solid #4CAF50;
    }
    .error-card {
        background-color: #ffebee;
        border-radius: 10px;
        padding: 1rem;
        margin: 0.5rem 0;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        border-left: 4px solid #ef553b;
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
    .trade-button {
        background-color: #4CAF50;
        color: white;
        padding: 0.5rem 1rem;
        border: none;
        border-radius: 4px;
        cursor: pointer;
        font-weight: bold;
    }
    .close-button {
        background-color: #ff6b6b;
        color: white;
        padding: 0.3rem 0.8rem;
        border: none;
        border-radius: 4px;
        cursor: pointer;
        font-weight: bold;
        font-size: 0.8rem;
    }
    .table-container {
        background-color: white;
        border-radius: 10px;
        padding: 1rem;
        margin: 1rem 0;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }
    .active-trade-row {
        background-color: #e3f2fd !important;
    }
    .closed-trade-row {
        background-color: #f5f5f5 !important;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'connection_errors' not in st.session_state:
    st.session_state.connection_errors = []
    
if 'last_data_update' not in st.session_state:
    st.session_state.last_data_update = datetime.now()
    
if 'connection_status' not in st.session_state:
    st.session_state.connection_status = "connected"

if 'stake_amounts' not in st.session_state:
    st.session_state.stake_amounts = {
        'USD/JPY': 10000, 'USD/CHF': 10000, 'USD/CAD': 10000,
        'EUR/USD': 10000, 'GBP/USD': 10000, 'AUD/USD': 10000
    }

if 'active_trades' not in st.session_state:
    st.session_state.active_trades = []

if 'trade_history' not in st.session_state:
    st.session_state.trade_history = []

if 'next_trade_id' not in st.session_state:
    st.session_state.next_trade_id = 1001

# Trade management functions
def open_trade(currency_pair, position, stake, entry_price):
    """Open a new trade"""
    trade = {
        'trade_id': f"TR{st.session_state.next_trade_id}",
        'currency_pair': currency_pair,
        'position': position,
        'stake': stake,
        'entry_price': entry_price,
        'current_price': entry_price,
        'open_time': datetime.now(),
        'status': 'ACTIVE'
    }
    st.session_state.active_trades.append(trade)
    st.session_state.next_trade_id += 1
    return trade

def close_trade(trade_id, exit_price):
    """Close an active trade"""
    for i, trade in enumerate(st.session_state.active_trades):
        if trade['trade_id'] == trade_id:
            closed_trade = trade.copy()
            closed_trade['exit_price'] = exit_price
            closed_trade['close_time'] = datetime.now()
            closed_trade['status'] = 'CLOSED'
            
            # Calculate P/L
            if trade['position'] == 'LONG':
                pnl = (exit_price - trade['entry_price']) * trade['stake']
            else:  # SHORT
                pnl = (trade['entry_price'] - exit_price) * trade['stake']
            
            closed_trade['pnl'] = pnl
            closed_trade['pnl_percent'] = (pnl / trade['stake']) * 100
            
            # Move to history
            st.session_state.trade_history.append(closed_trade)
            st.session_state.active_trades.pop(i)
            return closed_trade
    return None

def generate_sample_trades():
    """Generate sample active and historical trades for demo"""
    if not st.session_state.active_trades:
        # Create some sample active trades
        pairs = ['USD/JPY', 'EUR/USD', 'GBP/USD', 'USD/CAD']
        for i in range(3):
            pair = np.random.choice(pairs)
            stake = st.session_state.stake_amounts.get(pair, 10000)
            entry_price = np.random.uniform(1.0, 1.5) if 'USD' in pair else np.random.uniform(0.8, 1.4)
            
            open_trade(
                currency_pair=pair,
                position=np.random.choice(['LONG', 'SHORT']),
                stake=stake,
                entry_price=round(entry_price, 5)
            )
    
    if not st.session_state.trade_history:
        # Create sample historical trades
        pairs = ['USD/JPY', 'EUR/USD', 'GBP/USD', 'USD/CHF', 'AUD/USD']
        for i in range(15):
            pair = np.random.choice(pairs)
            stake = np.random.choice([5000, 10000, 15000, 20000])
            entry_price = np.random.uniform(1.0, 1.5) if 'USD' in pair else np.random.uniform(0.8, 1.4)
            exit_price = entry_price * np.random.uniform(0.95, 1.08)
            
            if np.random.choice(['LONG', 'SHORT']) == 'LONG':
                pnl = (exit_price - entry_price) * stake
            else:
                pnl = (entry_price - exit_price) * stake
            
            historical_trade = {
                'trade_id': f"TR{800 + i}",
                'currency_pair': pair,
                'position': np.random.choice(['LONG', 'SHORT']),
                'stake': stake,
                'entry_price': round(entry_price, 5),
                'exit_price': round(exit_price, 5),
                'pnl': round(pnl, 2),
                'pnl_percent': round((pnl / stake) * 100, 2),
                'open_time': datetime.now() - timedelta(days=np.random.randint(5, 60)),
                'close_time': datetime.now() - timedelta(days=np.random.randint(1, 5)),
                'status': 'CLOSED'
            }
            st.session_state.trade_history.append(historical_trade)

# Generate sample data
generate_sample_trades()

# Update current prices for active trades
for trade in st.session_state.active_trades:
    price_change = np.random.uniform(-0.02, 0.02)
    trade['current_price'] = round(trade['entry_price'] * (1 + price_change), 5)

# Header
st.markdown('<div class="main-header">LIVE FOREX PRICES - PURE PYTHON ANALYSIS</div>', unsafe_allow_html=True)

# Connection Status
if st.session_state.connection_status == "error":
    st.error("🔴 CONNECTION ERROR - Trading Disabled")
else:
    st.success("🟢 CONNECTION STABLE - All Systems Operational")

# Manual Stake Management Section
st.markdown("---")
st.subheader("🎯 Manual Stake Management")

col1, col2, col3 = st.columns([1, 1, 1])

with col1:
    st.markdown('<div class="stake-card">', unsafe_allow_html=True)
    st.write("**Quick Stake Presets**")
    preset_col1, preset_col2 = st.columns(2)
    with preset_col1:
        if st.button("$1,000", use_container_width=True):
            for pair in st.session_state.stake_amounts:
                st.session_state.stake_amounts[pair] = 1000
        if st.button("$5,000", use_container_width=True):
            for pair in st.session_state.stake_amounts:
                st.session_state.stake_amounts[pair] = 5000
    with preset_col2:
        if st.button("$10,000", use_container_width=True):
            for pair in st.session_state.stake_amounts:
                st.session_state.stake_amounts[pair] = 10000
        if st.button("$25,000", use_container_width=True):
            for pair in st.session_state.stake_amounts:
                st.session_state.stake_amounts[pair] = 25000
    st.markdown('</div>', unsafe_allow_html=True)

with col2:
    st.markdown('<div class="stake-card">', unsafe_allow_html=True)
    st.write("**Individual Pair Stakes**")
    pairs = list(st.session_state.stake_amounts.keys())[:3]
    for pair in pairs:
        current_stake = st.session_state.stake_amounts.get(pair, 10000)
        new_stake = st.number_input(
            f"{pair} Stake ($)",
            min_value=100,
            max_value=100000,
            value=current_stake,
            step=1000,
            key=f"stake_{pair}"
        )
        st.session_state.stake_amounts[pair] = new_stake
    st.markdown('</div>', unsafe_allow_html=True)

with col3:
    st.markdown('<div class="stake-card">', unsafe_allow_html=True)
    st.write("**Individual Pair Stakes (Cont.)**")
    pairs = list(st.session_state.stake_amounts.keys())[3:]
    for pair in pairs:
        current_stake = st.session_state.stake_amounts.get(pair, 10000)
        new_stake = st.number_input(
            f"{pair} Stake ($)",
            min_value=100,
            max_value=100000,
            value=current_stake,
            step=1000,
            key=f"stake_{pair}"
        )
        st.session_state.stake_amounts[pair] = new_stake
    st.markdown('</div>', unsafe_allow_html=True)

# Current Active Trades Table
st.markdown("---")
st.subheader("📊 Current Active Trades")

if st.session_state.active_trades:
    # Calculate totals
    total_exposure = sum(trade['stake'] for trade in st.session_state.active_trades)
    total_unrealized_pnl = 0
    
    # Prepare data for display
    active_trades_data = []
    for trade in st.session_state.active_trades:
        if trade['position'] == 'LONG':
            unrealized_pnl = (trade['current_price'] - trade['entry_price']) * trade['stake']
        else:  # SHORT
            unrealized_pnl = (trade['entry_price'] - trade['current_price']) * trade['stake']
        
        unrealized_pnl_percent = (unrealized_pnl / trade['stake']) * 100
        total_unrealized_pnl += unrealized_pnl
        
        active_trades_data.append({
            'Trade ID': trade['trade_id'],
            'Currency Pair': trade['currency_pair'],
            'Position': trade['position'],
            'Stake': f"${trade['stake']:,}",
            'Entry Price': trade['entry_price'],
            'Current Price': trade['current_price'],
            'P/L ($)': f"${unrealized_pnl:,.2f}",
            'P/L (%)': f"{unrealized_pnl_percent:.2f}%",
            'Open Time': trade['open_time'].strftime('%Y-%m-%d %H:%M'),
            'Actions': trade['trade_id']
        })
    
    # Display summary metrics
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Active Trades", len(st.session_state.active_trades))
    with col2:
        st.metric("Total Exposure", f"${total_exposure:,}")
    with col3:
        st.metric("Unrealized P/L", f"${total_unrealized_pnl:,.2f}")
    with col4:
        avg_holding = np.mean([(datetime.now() - trade['open_time']).days 
                              for trade in st.session_state.active_trades])
        st.metric("Avg Holding Time", f"{avg_holding:.1f} days")
    
    # Display the table
    active_df = pd.DataFrame(active_trades_data)
    
    # Create a styled dataframe with close buttons
    for i, trade in enumerate(st.session_state.active_trades):
        col1, col2, col3, col4 = st.columns([1, 2, 1, 1])
        with col1:
            st.write(f"**{trade['trade_id']}**")
        with col2:
            st.write(f"{trade['currency_pair']} - {trade['position']}")
            st.write(f"Entry: {trade['entry_price']} | Current: {trade['current_price']}")
        with col3:
            if trade['position'] == 'LONG':
                pnl = (trade['current_price'] - trade['entry_price']) * trade['stake']
            else:
                pnl = (trade['entry_price'] - trade['current_price']) * trade['stake']
            
            pnl_color = "positive-pnl" if pnl >= 0 else "negative-pnl"
            st.markdown(f'<div class="{pnl_color}">${pnl:,.2f}</div>', unsafe_allow_html=True)
        with col4:
            if st.button("Close Trade", key=f"close_{trade['trade_id']}", use_container_width=True):
                close_trade(trade['trade_id'], trade['current_price'])
                st.rerun()
    
    # Detailed table view
    with st.expander("Detailed Active Trades View"):
        st.dataframe(
            active_df,
            use_container_width=True,
            hide_index=True
        )
else:
    st.info("No active trades currently open")

# Trade History Table
st.markdown("---")
st.subheader("📈 Trade History")

if st.session_state.trade_history:
    # Calculate performance metrics
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
        st.metric("Total P/L", f"${total_closed_pnl:,.2f}")
    with col4:
        avg_trade_pnl = total_closed_pnl / total_trades if total_trades > 0 else 0
        st.metric("Avg Trade P/L", f"${avg_trade_pnl:.2f}")
    
    # Prepare historical data for display
    historical_trades_data = []
    for trade in st.session_state.trade_history[-20:]:  # Show last 20 trades
        historical_trades_data.append({
            'Trade ID': trade['trade_id'],
            'Currency Pair': trade['currency_pair'],
            'Position': trade['position'],
            'Stake': f"${trade['stake']:,}",
            'Entry Price': trade['entry_price'],
            'Exit Price': trade['exit_price'],
            'P/L ($)': f"${trade['pnl']:,.2f}",
            'P/L (%)': f"{trade['pnl_percent']:.2f}%",
            'Open Time': trade['open_time'].strftime('%Y-%m-%d %H:%M'),
            'Close Time': trade['close_time'].strftime('%Y-%m-%d %H:%M'),
            'Duration': f"{(trade['close_time'] - trade['open_time']).days}d"
        })
    
    # Display the historical trades table
    historical_df = pd.DataFrame(historical_trades_data)
    
    # Add filters
    col1, col2 = st.columns(2)
    with col1:
        show_count = st.selectbox("Show last trades:", [10, 20, 50, "All"])
    with col2:
        sort_by = st.selectbox("Sort by:", ["Most Recent", "Highest P/L", "Lowest P/L", "Largest Stake"])
    
    # Apply filters
    if show_count != "All":
        display_df = historical_df.head(show_count)
    else:
        display_df = historical_df
    
    # Apply sorting
    if sort_by == "Most Recent":
        display_df = display_df.iloc[::-1]  # Reverse to show most recent first
    elif sort_by == "Highest P/L":
        display_df = display_df.iloc[display_df['P/L ($)'].str.replace('$', '').str.replace(',', '').astype(float).argsort()[::-1]]
    elif sort_by == "Lowest P/L":
        display_df = display_df.iloc[display_df['P/L ($)'].str.replace('$', '').str.replace(',', '').astype(float).argsort()]
    
    st.dataframe(
        display_df,
        use_container_width=True,
        hide_index=True
    )
    
    # Export functionality
    col1, col2 = st.columns([3, 1])
    with col2:
        if st.button("Export Trade History to CSV"):
            export_df = pd.DataFrame([{
                'Trade ID': t['trade_id'],
                'Currency Pair': t['currency_pair'],
                'Position': t['position'],
                'Stake': t['stake'],
                'Entry Price': t['entry_price'],
                'Exit Price': t['exit_price'],
                'P/L ($)': t['pnl'],
                'P/L (%)': t['pnl_percent'],
                'Open Time': t['open_time'],
                'Close Time': t['close_time']
            } for t in st.session_state.trade_history])
            
            csv = export_df.to_csv(index=False)
            st.download_button(
                label="Download CSV",
                data=csv,
                file_name=f"trade_history_{datetime.now().strftime('%Y%m%d_%H%M')}.csv",
                mime="text/csv"
            )
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
    stake = st.number_input(
        "Stake Amount ($)",
        min_value=100,
        max_value=100000,
        value=st.session_state.stake_amounts[selected_pair],
        step=1000
    )
    current_price = np.random.uniform(1.0, 1.5) if 'USD' in selected_pair else np.random.uniform(0.8, 1.4)
    st.write(f"Current Price: **{current_price:.5f}**")
    
with col3:
    st.write("&nbsp;")  # Spacer
    st.write("&nbsp;")
    if st.button("🎯 Execute Trade", use_container_width=True, type="primary"):
        new_trade = open_trade(selected_pair, position, stake, round(current_price, 5))
        st.success(f"Trade opened: {new_trade['trade_id']} - {selected_pair} {position} ${stake:,}")
        st.rerun()

# Footer
st.markdown("---")
if st.button("🔄 Refresh All Data", use_container_width=True):
    st.rerun()

st.markdown("""
<div style="text-align: center; color: #666; margin-top: 2rem;">
    Pure Python Forex Analysis Dashboard | Complete Trade Management System
</div>
""", unsafe_allow_html=True)

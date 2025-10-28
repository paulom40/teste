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
    .stake-button {
        background-color: #2196F3;
        color: white;
        padding: 0.5rem 1rem;
        border: none;
        border-radius: 4px;
        cursor: pointer;
        font-weight: bold;
        width: 100%;
    }
    .error-message {
        background-color: #ffebee;
        color: #c62828;
        padding: 0.5rem;
        border-radius: 5px;
        text-align: center;
        margin: 0.5rem 0;
    }
    .success-message {
        background-color: #e8f5e9;
        color: #2e7d32;
        padding: 0.5rem;
        border-radius: 5px;
        text-align: center;
        margin: 0.5rem 0;
    }
    .warning-message {
        background-color: #fff3e0;
        color: #ef6c00;
        padding: 0.5rem;
        border-radius: 5px;
        text-align: center;
        margin: 0.5rem 0;
    }
    .connection-status {
        padding: 0.5rem;
        border-radius: 5px;
        text-align: center;
        margin: 0.5rem 0;
        font-weight: bold;
    }
    .connection-good {
        background-color: #e8f5e9;
        color: #2e7d32;
        border: 1px solid #2e7d32;
    }
    .connection-error {
        background-color: #ffebee;
        color: #c62828;
        border: 1px solid #c62828;
    }
    .connection-warning {
        background-color: #fff3e0;
        color: #ef6c00;
        border: 1px solid #ef6c00;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state for connection management
if 'connection_errors' not in st.session_state:
    st.session_state.connection_errors = []
    
if 'last_data_update' not in st.session_state:
    st.session_state.last_data_update = datetime.now()
    
if 'connection_status' not in st.session_state:
    st.session_state.connection_status = "connected"  # connected, warning, error
    
if 'stake_amounts' not in st.session_state:
    st.session_state.stake_amounts = {
        'USD/JPY': 10000,
        'USD/CHF': 10000,
        'USD/CAD': 10000,
        'EUR/USD': 10000,
        'GBP/USD': 10000,
        'AUD/USD': 10000
    }

if 'trade_history' not in st.session_state:
    st.session_state.trade_history = []

# Connection management functions
def check_connection_status():
    """Check and update connection status"""
    now = datetime.now()
    time_since_update = (now - st.session_state.last_data_update).total_seconds()
    
    if time_since_update > 30:  # 30 seconds without update
        st.session_state.connection_status = "error"
        if len(st.session_state.connection_errors) < 2:
            st.session_state.connection_errors.append({
                'timestamp': now,
                'message': 'Data feed timeout - No updates for 30+ seconds'
            })
    elif time_since_update > 15:  # 15 seconds without update
        st.session_state.connection_status = "warning"
    else:
        st.session_state.connection_status = "connected"

def simulate_connection_issues():
    """Simulate random connection issues for demo"""
    if np.random.random() < 0.1:  # 10% chance of error
        error_type = np.random.choice(['timeout', 'api_limit', 'authentication'])
        error_messages = {
            'timeout': 'Data feed timeout - Connection lost',
            'api_limit': 'API rate limit exceeded - Please wait',
            'authentication': 'Authentication failed - Reconnecting'
        }
        
        if len(st.session_state.connection_errors) < 5:  # Limit errors stored
            st.session_state.connection_errors.append({
                'timestamp': datetime.now(),
                'message': error_messages[error_type]
            })

def clear_errors():
    """Clear connection errors"""
    st.session_state.connection_errors = []
    st.session_state.connection_status = "connected"
    st.session_state.last_data_update = datetime.now()

# Header
st.markdown('<div class="main-header">LIVE FOREX PRICES - PURE PYTHON ANALYSIS</div>', unsafe_allow_html=True)

# Connection Status Banner
check_connection_status()

if st.session_state.connection_status == "error":
    st.markdown('<div class="connection-status connection-error">🔴 CONNECTION ERROR - Trading Disabled</div>', unsafe_allow_html=True)
elif st.session_state.connection_status == "warning":
    st.markdown('<div class="connection-status connection-warning">🟡 CONNECTION WARNING - Limited Functionality</div>', unsafe_allow_html=True)
else:
    st.markdown('<div class="connection-status connection-good">🟢 CONNECTION STABLE - All Systems Operational</div>', unsafe_allow_html=True)

# Error Details Section
if st.session_state.connection_errors:
    st.markdown("---")
    st.subheader("🔧 Connection Issues & Solutions")
    
    with st.container():
        st.markdown('<div class="error-card">', unsafe_allow_html=True)
        
        col1, col2 = st.columns([3, 1])
        with col1:
            st.write("**Recent Connection Errors:**")
            for i, error in enumerate(st.session_state.connection_errors[-3:]):  # Show last 3 errors
                error_time = error['timestamp'].strftime('%H:%M:%S')
                st.write(f"• **{error_time}**: {error['message']}")
        
        with col2:
            st.write("**Quick Actions:**")
            if st.button("🔄 Reconnect", use_container_width=True):
                clear_errors()
                st.rerun()
            
            if st.button("🗑️ Clear Errors", use_container_width=True):
                st.session_state.connection_errors = []
                st.rerun()
        
        st.markdown('</div>', unsafe_allow_html=True)

# Sample data generation functions with error simulation
def generate_forex_data():
    """Generate forex data with simulated connection issues"""
    simulate_connection_issues()
    
    currencies = ['USD/JPY', 'USD/CHF', 'USD/CAD', 'EUR/USD', 'GBP/USD', 'AUD/USD']
    data = []
    
    # Simulate data delay if there are connection issues
    if st.session_state.connection_status == "error":
        # Return stale data
        for currency in currencies:
            data.append({
                'Currency': currency,
                'Price': 0.00000,
                'Change': 0.000,
                'ChangePercent': 0.00,
                'Signals': 0,
                'Status': 'ERROR',
                'DataStatus': 'STALE'
            })
    else:
        # Generate fresh data
        for currency in currencies:
            base_price = np.random.uniform(0.8, 1.4) if 'USD' not in currency else np.random.uniform(1.0, 1.5)
            change = np.random.uniform(-0.001, 0.001)
            change_percent = change * 100
            
            data.append({
                'Currency': currency,
                'Price': base_price,
                'Change': change,
                'ChangePercent': change_percent,
                'Signals': np.random.randint(0, 3),
                'Status': 'HOLD' if abs(change_percent) < 0.05 else 'TRADE',
                'DataStatus': 'LIVE'
            })
        
        st.session_state.last_data_update = datetime.now()
    
    return pd.DataFrame(data)

def generate_current_trades():
    trades = []
    currency_pairs = ['USD/JPY', 'EUR/USD', 'GBP/USD', 'USD/CAD', 'AUD/USD']
    
    for i in range(2):  # Reduced number for demo
        pair = np.random.choice(currency_pairs)
        entry_price = np.random.uniform(1.0, 1.5) if 'USD' in pair else np.random.uniform(0.8, 1.4)
        current_price = entry_price * np.random.uniform(0.98, 1.03)
        pnl = (current_price - entry_price) * st.session_state.stake_amounts.get(pair, 10000)
        
        trades.append({
            'Trade ID': f"TR{1000 + i}",
            'Currency Pair': pair,
            'Position': np.random.choice(['LONG', 'SHORT']),
            'Stake': st.session_state.stake_amounts.get(pair, 10000),
            'Entry Price': round(entry_price, 5),
            'Current Price': round(current_price, 5),
            'P/L ($)': round(pnl, 2),
            'P/L (%)': round((current_price - entry_price) / entry_price * 100, 2),
            'Open Time': (datetime.now() - timedelta(hours=np.random.randint(1, 72))).strftime('%Y-%m-%d %H:%M'),
            'Status': 'ACTIVE'
        })
    
    return pd.DataFrame(trades)

def generate_past_trades():
    trades = []
    currency_pairs = ['USD/JPY', 'EUR/USD', 'GBP/USD', 'USD/CHF', 'USD/CAD', 'AUD/USD', 'NZD/USD']
    
    for i in range(8):
        pair = np.random.choice(currency_pairs)
        entry_price = np.random.uniform(1.0, 1.5) if 'USD' in pair else np.random.uniform(0.8, 1.4)
        exit_price = entry_price * np.random.uniform(0.95, 1.08)
        stake = st.session_state.stake_amounts.get(pair, 10000)
        pnl = (exit_price - entry_price) * stake
        
        trades.append({
            'Trade ID': f"TR{800 + i}",
            'Currency Pair': pair,
            'Position': np.random.choice(['LONG', 'SHORT']),
            'Stake': stake,
            'Entry Price': round(entry_price, 5),
            'Exit Price': round(exit_price, 5),
            'P/L ($)': round(pnl, 2),
            'P/L (%)': round((exit_price - entry_price) / entry_price * 100, 2),
            'Open Time': (datetime.now() - timedelta(days=np.random.randint(5, 30))).strftime('%Y-%m-%d %H:%M'),
            'Close Time': (datetime.now() - timedelta(days=np.random.randint(1, 5))).strftime('%Y-%m-%d %H:%M'),
            'Status': 'CLOSED'
        })
    
    return pd.DataFrame(trades)

# Manual Stake Management Section
st.markdown("---")
st.subheader("🎯 Manual Stake Management")

col1, col2, col3 = st.columns([1, 1, 1])

with col1:
    st.markdown('<div class="stake-card">', unsafe_allow_html=True)
    st.write("**Quick Stake Presets**")
    
    preset_col1, preset_col2 = st.columns(2)
    with preset_col1:
        if st.button("$1,000", use_container_width=True, disabled=st.session_state.connection_status == "error"):
            for pair in st.session_state.stake_amounts:
                st.session_state.stake_amounts[pair] = 1000
            st.success("Stake set to $1,000 for all pairs")
        
        if st.button("$5,000", use_container_width=True, disabled=st.session_state.connection_status == "error"):
            for pair in st.session_state.stake_amounts:
                st.session_state.stake_amounts[pair] = 5000
            st.success("Stake set to $5,000 for all pairs")
    
    with preset_col2:
        if st.button("$10,000", use_container_width=True, disabled=st.session_state.connection_status == "error"):
            for pair in st.session_state.stake_amounts:
                st.session_state.stake_amounts[pair] = 10000
            st.success("Stake set to $10,000 for all pairs")
        
        if st.button("$25,000", use_container_width=True, disabled=st.session_state.connection_status == "error"):
            for pair in st.session_state.stake_amounts:
                st.session_state.stake_amounts[pair] = 25000
            st.success("Stake set to $25,000 for all pairs")
    st.markdown('</div>', unsafe_allow_html=True)

with col2:
    st.markdown('<div class="stake-card">', unsafe_allow_html=True)
    st.write("**Individual Pair Stakes**")
    
    pairs = ['USD/JPY', 'USD/CHF', 'USD/CAD', 'EUR/USD', 'GBP/USD', 'AUD/USD']
    for pair in pairs[:3]:
        current_stake = st.session_state.stake_amounts.get(pair, 10000)
        new_stake = st.number_input(
            f"{pair} Stake ($)",
            min_value=100,
            max_value=100000,
            value=current_stake,
            step=1000,
            key=f"stake_{pair}",
            disabled=st.session_state.connection_status == "error"
        )
        st.session_state.stake_amounts[pair] = new_stake
    st.markdown('</div>', unsafe_allow_html=True)

with col3:
    st.markdown('<div class="stake-card">', unsafe_allow_html=True)
    st.write("**Individual Pair Stakes (Cont.)**")
    
    for pair in pairs[3:]:
        current_stake = st.session_state.stake_amounts.get(pair, 10000)
        new_stake = st.number_input(
            f"{pair} Stake ($)",
            min_value=100,
            max_value=100000,
            value=current_stake,
            step=1000,
            key=f"stake_{pair}",
            disabled=st.session_state.connection_status == "error"
        )
        st.session_state.stake_amounts[pair] = new_stake
    st.markdown('</div>', unsafe_allow_html=True)

# Generate data with current stakes
forex_data = generate_forex_data()
current_trades = generate_current_trades()
past_trades = generate_past_trades()

# Create layout with columns
col1, col2, col3 = st.columns([1, 2, 1])

with col1:
    st.subheader("Portfolio Overview")
    
    total_stake = sum(st.session_state.stake_amounts.values())
    active_exposure = current_trades['Stake'].sum() if not current_trades.empty else 0
    
    st.metric(label="Total Stake", value=f"${total_stake:,}")
    st.metric(label="Active Exposure", value=f"${active_exposure:,}")
    st.metric(label="Available Balance", value=f"${total_stake - active_exposure:,}")
    
    st.subheader("Trading Signals")
    for index, row in forex_data.iterrows():
        signal_class = "signal-low" if row['Signals'] == 0 else "signal-high"
        status_text = f"LOW - Signals: {row['Signals']}" if row['Signals'] == 0 else f"ACTIVE - Signals: {row['Signals']}"
        
        if st.session_state.connection_status == "error":
            st.markdown(f'<div class="error-message">ERROR - No Data</div>', unsafe_allow_html=True)
        else:
            st.markdown(f'<div class="{signal_class}">{status_text}</div>', unsafe_allow_html=True)

with col2:
    st.subheader("Currency Pairs")
    
    # Display currency cards with connection status
    for index, row in forex_data.iterrows():
        current_stake = st.session_state.stake_amounts.get(row['Currency'], 10000)
        
        if st.session_state.connection_status == "error":
            # Show error state
            with st.container():
                st.markdown(f'<div class="error-card">', unsafe_allow_html=True)
                st.write(f"**{row['Currency']}**")
                st.markdown('<div class="error-message">🔴 CONNECTION ERROR - No live data</div>', unsafe_allow_html=True)
                st.write(f"Last Stake: ${current_stake:,}")
                st.markdown('</div>', unsafe_allow_html=True)
        else:
            # Show normal state
            change_color = "positive-change" if row['Change'] >= 0 else "negative-change"
            change_symbol = "+" if row['Change'] >= 0 else ""
            
            with st.container():
                st.markdown(f'<div class="currency-card">', unsafe_allow_html=True)
                col_a, col_b, col_c, col_d = st.columns([2, 1, 1, 1])
                
                with col_a:
                    st.write(f"**{row['Currency']}**")
                    st.write(f"**{row['Price']:.5f}**")
                    st.write(f"Stake: ${current_stake:,}")
                
                with col_b:
                    st.markdown(f'<span class="{change_color}">{change_symbol}{row["ChangePercent"]:.2f}%</span>', unsafe_allow_html=True)
                
                with col_c:
                    new_stake = st.number_input(
                        "Adjust Stake",
                        min_value=100,
                        max_value=100000,
                        value=current_stake,
                        step=1000,
                        key=f"quick_stake_{row['Currency']}",
                        label_visibility="collapsed",
                        disabled=st.session_state.connection_status == "error"
                    )
                    if new_stake != current_stake:
                        st.session_state.stake_amounts[row['Currency']] = new_stake
                        st.rerun()
                
                with col_d:
                    if row['Status'] == 'TRADE' and st.session_state.connection_status != "error":
                        if st.button("TRADE", key=f"trade_{index}", use_container_width=True):
                            trade_data = {
                                'timestamp': datetime.now(),
                                'currency': row['Currency'],
                                'action': 'OPEN',
                                'stake': current_stake,
                                'price': row['Price'],
                                'position': 'LONG' if row['Change'] >= 0 else 'SHORT'
                            }
                            st.session_state.trade_history.append(trade_data)
                            st.success(f"Trade executed: {row['Currency']} ${current_stake:,}")
                    else:
                        status_class = "error-message" if st.session_state.connection_status == "error" else "warning-message"
                        status_text = "DISABLED" if st.session_state.connection_status == "error" else "HOLD"
                        st.markdown(f'<div class="{status_class}">{status_text}</div>', unsafe_allow_html=True)
                
                st.markdown('</div>', unsafe_allow_html=True)

with col3:
    st.subheader("Performance")
    
    # Create performance chart
    dates = [datetime.now() - timedelta(days=x) for x in range(30, 0, -1)]
    
    if st.session_state.connection_status == "error":
        # Show placeholder chart for error state
        fig = go.Figure()
        fig.add_annotation(
            text="📡 Connection Error<br>Chart data unavailable",
            xref="paper", yref="paper",
            x=0.5, y=0.5,
            showarrow=False,
            font=dict(size=16, color="red")
        )
        fig.update_layout(
            height=300,
            margin=dict(l=0, r=0, t=0, b=0),
            xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            yaxis=dict(showgrid=False, zeroline=False, showticklabels=False)
        )
    else:
        performance = [10000 + x * np.random.uniform(-50, 100) for x in range(30)]
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=dates, 
            y=performance,
            mode='lines',
            line=dict(color='#1f77b4', width=3),
            name='Portfolio Value'
        ))
        fig.update_layout(
            height=300,
            margin=dict(l=0, r=0, t=0, b=0),
            xaxis_title="Date",
            yaxis_title="Value ($)",
            showlegend=False
        )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Risk metrics
    if st.session_state.connection_status == "error":
        st.markdown('<div class="error-message">Risk Level: UNAVAILABLE</div>', unsafe_allow_html=True)
        st.markdown('<div class="error-message">Today\'s P/L: UNAVAILABLE</div>', unsafe_allow_html=True)
    else:
        st.metric(label="Risk Level", value="MEDIUM", delta="-2%")
        st.metric(label="Today's P/L", value="+$142", delta="+1.4%")

# System Status Section
st.markdown("---")
st.subheader("🔧 System Status & Connection Management")

status_col1, status_col2, status_col3 = st.columns(3)

with status_col1:
    if st.session_state.connection_status == "error":
        st.error("❌ Data Feed: DISCONNECTED")
        st.error(f"❌ Connection: {len(st.session_state.connection_errors)} Active Errors")
    elif st.session_state.connection_status == "warning":
        st.warning("⚠️ Data Feed: UNSTABLE")
        st.warning(f"⚠️ Connection: {len(st.session_state.connection_errors)} Warnings")
    else:
        st.success("✅ Data Feed: ACTIVE")
        st.success("✅ Connection: STABLE")

with status_col2:
    if st.session_state.connection_status == "error":
        st.error("⚠️ API Limit: UNAVAILABLE")
    else:
        st.warning("⚠️ API Limit: 78% Used")
    
    time_since_update = (datetime.now() - st.session_state.last_data_update).total_seconds()
    st.info(f"ℹ️ Last Update: {int(time_since_update)}s ago")

with status_col3:
    if st.session_state.connection_status == "error":
        st.error("❌ Analysis: PAUSED")
        st.error("❌ Alerts: DISABLED")
    else:
        st.success("✅ Analysis: RUNNING")
        st.warning("⚠️ Alerts: 1 Pending")

# Connection Management Actions
st.markdown("### Connection Controls")
col1, col2, col3 = st.columns(3)

with col1:
    if st.button("🔄 Force Reconnect", use_container_width=True):
        clear_errors()
        st.session_state.last_data_update = datetime.now()
        st.rerun()

with col2:
    if st.button("🗑️ Clear All Errors", use_container_width=True):
        st.session_state.connection_errors = []
        st.session_state.connection_status = "connected"
        st.rerun()

with col3:
    if st.button("📊 Test Data Feed", use_container_width=True):
        st.session_state.last_data_update = datetime.now()
        if np.random.random() < 0.7:  # 70% success rate
            st.success("✅ Data feed test: SUCCESS")
            st.session_state.connection_status = "connected"
        else:
            st.error("❌ Data feed test: FAILED")
            st.session_state.connection_status = "error"
        st.rerun()

# Footer
st.markdown("---")
col1, col2 = st.columns([3, 1])
with col2:
    if st.button("🔄 Refresh All Data", use_container_width=True):
        st.session_state.last_data_update = datetime.now()
        st.rerun()

st.markdown(f"""
<div style="text-align: center; color: #666; margin-top: 2rem;">
    Pure Python Forex Analysis Dashboard | 
    Connection: <strong>{st.session_state.connection_status.upper()}</strong> | 
    Last Update: {st.session_state.last_data_update.strftime('%H:%M:%S')}
</div>
""", unsafe_allow_html=True)

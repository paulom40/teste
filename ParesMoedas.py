import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
from datetime import datetime, timedelta

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
    .error-message {
        background-color: #ffebee;
        color: #c62828;
        padding: 0.5rem;
        border-radius: 5px;
        text-align: center;
        margin: 0.5rem 0;
    }
    .table-header {
        background-color: #1f77b4;
        color: white;
        font-weight: bold;
    }
    .active-trade {
        background-color: #e3f2fd;
    }
    .closed-trade {
        background-color: #f5f5f5;
    }
</style>
""", unsafe_allow_html=True)

# Header
st.markdown('<div class="main-header">LIVE FOREX PRICES - PURE PYTHON ANALYSIS</div>', unsafe_allow_html=True)

# Sample data generation functions
def generate_forex_data():
    currencies = ['USD/JPY', 'USD/CHF', 'USD/CAD', 'EUR/USD', 'GBP/USD', 'AUD/USD']
    data = []
    
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
            'Status': 'HOLD' if abs(change_percent) < 0.05 else 'TRADE'
        })
    
    return pd.DataFrame(data)

def generate_current_trades():
    trades = []
    currency_pairs = ['USD/JPY', 'EUR/USD', 'GBP/USD', 'USD/CAD', 'AUD/USD']
    
    for i in range(5):
        pair = np.random.choice(currency_pairs)
        entry_price = np.random.uniform(1.0, 1.5) if 'USD' in pair else np.random.uniform(0.8, 1.4)
        current_price = entry_price * np.random.uniform(0.98, 1.03)
        pnl = (current_price - entry_price) * 10000
        
        trades.append({
            'Trade ID': f"TR{1000 + i}",
            'Currency Pair': pair,
            'Position': np.random.choice(['LONG', 'SHORT']),
            'Size': np.random.choice([10000, 25000, 50000, 100000]),
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
    
    for i in range(15):
        pair = np.random.choice(currency_pairs)
        entry_price = np.random.uniform(1.0, 1.5) if 'USD' in pair else np.random.uniform(0.8, 1.4)
        exit_price = entry_price * np.random.uniform(0.95, 1.08)
        pnl = (exit_price - entry_price) * 10000
        
        trades.append({
            'Trade ID': f"TR{800 + i}",
            'Currency Pair': pair,
            'Position': np.random.choice(['LONG', 'SHORT']),
            'Size': np.random.choice([10000, 25000, 50000, 100000]),
            'Entry Price': round(entry_price, 5),
            'Exit Price': round(exit_price, 5),
            'P/L ($)': round(pnl, 2),
            'P/L (%)': round((exit_price - entry_price) / entry_price * 100, 2),
            'Open Time': (datetime.now() - timedelta(days=np.random.randint(5, 30))).strftime('%Y-%m-%d %H:%M'),
            'Close Time': (datetime.now() - timedelta(days=np.random.randint(1, 5))).strftime('%Y-%m-%d %H:%M'),
            'Status': 'CLOSED'
        })
    
    return pd.DataFrame(trades)

# Generate data
forex_data = generate_forex_data()
current_trades = generate_current_trades()
past_trades = generate_past_trades()

# Create layout with columns
col1, col2, col3 = st.columns([1, 2, 1])

with col1:
    st.subheader("Portfolio Overview")
    st.metric(label="USD Balance", value="$10,250", delta="+2.1%")
    st.metric(label="Hold Position", value="30%")
    
    st.subheader("Trading Signals")
    for index, row in forex_data.iterrows():
        if row['Signals'] == 0:
            st.markdown(f'<div class="signal-low">LOW - Signals: {row["Signals"]}</div>', unsafe_allow_html=True)
        else:
            st.markdown(f'<div class="signal-high">ACTIVE - Signals: {row["Signals"]}</div>', unsafe_allow_html=True)

with col2:
    st.subheader("Currency Pairs")
    
    # Display currency cards
    for index, row in forex_data.iterrows():
        change_color = "positive-change" if row['Change'] >= 0 else "negative-change"
        change_symbol = "+" if row['Change'] >= 0 else ""
        
        with st.container():
            st.markdown(f'<div class="currency-card">', unsafe_allow_html=True)
            col_a, col_b, col_c = st.columns([2, 1, 1])
            
            with col_a:
                st.write(f"**{row['Currency']}**")
                st.write(f"**{row['Price']:.5f}**")
            
            with col_b:
                st.markdown(f'<span class="{change_color}">{change_symbol}{row["ChangePercent"]:.2f}%</span>', unsafe_allow_html=True)
            
            with col_c:
                if row['Status'] == 'TRADE':
                    st.button("TRADE", key=f"trade_{index}", use_container_width=True)
                else:
                    st.markdown('<div class="error-message">HOLD</div>', unsafe_allow_html=True)
            
            st.markdown('</div>', unsafe_allow_html=True)

with col3:
    st.subheader("Performance")
    
    # Create a simple performance chart
    dates = [datetime.now() - timedelta(days=x) for x in range(30, 0, -1)]
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
    st.metric(label="Risk Level", value="MEDIUM", delta="-2%")
    st.metric(label="Today's P/L", value="+$142", delta="+1.4%")

# Current Trades Table
st.markdown("---")
st.subheader("📊 Current Open Trades")

# Style the current trades table
def style_current_trades(df):
    styled_df = df.copy()
    
    # Apply styling to P/L columns
    def color_pnl(val):
        color = 'color: #2e7d32;' if val > 0 else 'color: #c62828;'
        return color
    
    # Hide the index
    styled_df = styled_df.reset_index(drop=True)
    
    return styled_df.style.applymap(color_pnl, subset=['P/L ($)', 'P/L (%)'])

# Display current trades with better formatting
if not current_trades.empty:
    # Calculate totals
    total_pnl = current_trades['P/L ($)'].sum()
    total_pnl_percent = current_trades['P/L (%)'].sum()
    
    # Display summary metrics
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Open Trades", len(current_trades))
    with col2:
        st.metric("Total Exposure", f"${current_trades['Size'].sum():,}")
    with col3:
        st.metric("Total P/L ($)", f"${total_pnl:,.2f}", delta=f"{total_pnl_percent:.2f}%")
    with col4:
        avg_holding = np.mean([(datetime.now() - datetime.strptime(t, '%Y-%m-%d %H:%M')).days 
                              for t in current_trades['Open Time']])
        st.metric("Avg Holding Time", f"{avg_holding:.1f} days")
    
    # Display the table
    st.dataframe(
        current_trades,
        column_config={
            "P/L ($)": st.column_config.NumberColumn(
                "P/L ($)",
                format="$%.2f",
            ),
            "P/L (%)": st.column_config.NumberColumn(
                "P/L (%)",
                format="%.2f%%",
            ),
            "Size": st.column_config.NumberColumn(
                "Size",
                format="%d",
            )
        },
        use_container_width=True
    )
else:
    st.info("No current open trades")

# Past Trades Table
st.markdown("---")
st.subheader("📈 Trade History (Last 30 Days)")

# Style the past trades table
def style_past_trades(df):
    styled_df = df.copy()
    
    # Apply styling to P/L columns
    def color_pnl(val):
        color = 'color: #2e7d32;' if val > 0 else 'color: #c62828;'
        return color
    
    # Hide the index
    styled_df = styled_df.reset_index(drop=True)
    
    return styled_df.style.applymap(color_pnl, subset=['P/L ($)', 'P/L (%)'])

# Display past trades with better formatting
if not past_trades.empty:
    # Calculate totals for past trades
    total_closed_pnl = past_trades['P/L ($)'].sum()
    winning_trades = len(past_trades[past_trades['P/L ($)'] > 0])
    total_trades = len(past_trades)
    win_rate = (winning_trades / total_trades) * 100
    
    # Display summary metrics
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Total Closed Trades", total_trades)
    with col2:
        st.metric("Win Rate", f"{win_rate:.1f}%")
    with col3:
        st.metric("Total P/L ($)", f"${total_closed_pnl:,.2f}")
    with col4:
        avg_trade_pnl = total_closed_pnl / total_trades
        st.metric("Avg Trade P/L", f"${avg_trade_pnl:.2f}")
    
    # Add filters for past trades
    col1, col2 = st.columns(2)
    with col1:
        selected_pairs = st.multiselect(
            "Filter by Currency Pair",
            options=past_trades['Currency Pair'].unique(),
            default=past_trades['Currency Pair'].unique()
        )
    with col2:
        date_range = st.selectbox(
            "Time Period",
            ["Last 7 days", "Last 30 days", "Last 90 days", "All time"]
        )
    
    # Filter the data
    filtered_past_trades = past_trades[past_trades['Currency Pair'].isin(selected_pairs)]
    
    # Display the table
    st.dataframe(
        filtered_past_trades,
        column_config={
            "P/L ($)": st.column_config.NumberColumn(
                "P/L ($)",
                format="$%.2f",
            ),
            "P/L (%)": st.column_config.NumberColumn(
                "P/L (%)",
                format="%.2f%%",
            ),
            "Size": st.column_config.NumberColumn(
                "Size",
                format="%d",
            )
        },
        use_container_width=True
    )
else:
    st.info("No trade history available")

# Error handling section
st.markdown("---")
st.subheader("System Status")
status_col1, status_col2, status_col3 = st.columns(3)

with status_col1:
    st.success("✅ Data Feed: Active")
    st.error("❌ Connection: 2 Errors")

with status_col2:
    st.warning("⚠️ API Limit: 78% Used")
    st.info("ℹ️ Last Update: Just Now")

with status_col3:
    st.success("✅ Analysis: Running")
    st.error("❌ Alerts: 1 Pending")

# Footer with refresh button
st.markdown("---")
if st.button("Refresh Data", use_container_width=True):
    st.rerun()

st.markdown("""
<div style="text-align: center; color: #666; margin-top: 2rem;">
    Pure Python Forex Analysis Dashboard | Live prices update every 10 seconds
</div>
""", unsafe_allow_html=True)

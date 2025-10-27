import streamlit as st
import pandas as pd
import numpy as np
import requests
import time
from datetime import datetime, timedelta
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

st.markdown("""
    <style>
    #MainMenu {visibility: hidden;}
    header {visibility: hidden;}
    footer {visibility: hidden;}
    </style>
""", unsafe_allow_html=True)

# Page configuration with modern theme
st.set_page_config(
    page_title="Auto Trading Bot - 2 Indicator Agreement",
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
    .signal-buy {
        background: linear-gradient(135deg, #87CEEB 0%, #1E90FF 100%);
        color: white;
        padding: 0.5rem;
        border-radius: 5px;
        margin: 0.2rem 0;
        text-align: center;
        font-weight: bold;
    }
    .signal-sell {
        background: linear-gradient(135deg, #FFA500 0%, #FF8C00 100%);
        color: white;
        padding: 0.5rem;
        border-radius: 5px;
        margin: 0.2rem 0;
        text-align: center;
        font-weight: bold;
    }
    .signal-neutral {
        background: linear-gradient(135deg, #808080 0%, #A9A9A9 100%);
        color: white;
        padding: 0.5rem;
        border-radius: 5px;
        margin: 0.2rem 0;
        text-align: center;
        font-weight: bold;
    }
    .stButton>button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        padding: 0.5rem 1rem;
        border-radius: 5px;
        font-weight: bold;
        width: 100%;
    }
    .stButton>button:hover {
        background: linear-gradient(135deg, #764ba2 0%, #667eea 100%);
        color: white;
    }
    .sidebar .sidebar-content {
        background: linear-gradient(180deg, #2d3748 0%, #4a5568 100%);
    }
</style>
""", unsafe_allow_html=True)

# Main application
def main():
    # Header
    st.markdown('<h1 class="main-header">🤖 Auto Trading Bot</h1>', unsafe_allow_html=True)
    st.markdown('<h3 style="text-align: center; color: #666;">2-Indicator Agreement System</h3>', unsafe_allow_html=True)
    
    # Sidebar
    st.sidebar.title("⚙️ Trading Configuration")
    
    # Trading parameters
    st.sidebar.subheader("Trading Parameters")
    initial_balance = st.sidebar.number_input("Initial Balance ($)", value=10000.0, min_value=1000.0, step=1000.0)
    risk_per_trade = st.sidebar.slider("Risk per Trade (%)", 1, 10, 2)
    trading_pair = st.sidebar.selectbox("Trading Pair", ["BTC/USD", "ETH/USD", "AAPL", "GOOGL", "MSFT"])
    
    # Indicator settings
    st.sidebar.subheader("Indicator Settings")
    rsi_period = st.sidebar.slider("RSI Period", 5, 30, 14)
    ma_fast = st.sidebar.slider("Fast MA Period", 5, 50, 20)
    ma_slow = st.sidebar.slider("Slow MA Period", 20, 200, 50)
    
    # Strategy selection
    strategy = st.sidebar.selectbox(
        "Trading Strategy",
        ["RSI + Moving Average Crossover", "MACD + Bollinger Bands", "Stochastic + Volume Profile"]
    )
    
    # Main content area
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        st.metric("Current Balance", f"${initial_balance:,.2f}", delta="+2.3%")
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col2:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        st.metric("Total Trades", "156", delta="12 today")
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col3:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        st.metric("Win Rate", "68.2%", delta="+2.1%")
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col4:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        st.metric("Current P&L", "$2,345.67", delta="+5.8%", delta_color="normal")
        st.markdown('</div>', unsafe_allow_html=True)
    
    # Trading signals section
    st.subheader("📊 Current Trading Signals")
    
    signal_col1, signal_col2, signal_col3, signal_col4 = st.columns(4)
    
    with signal_col1:
        st.markdown('<div class="signal-strong-buy">STRONG BUY</div>', unsafe_allow_html=True)
        st.write("RSI: Oversold")
        st.write("MA: Bullish Crossover")
    
    with signal_col2:
        st.markdown('<div class="signal-buy">BUY</div>', unsafe_allow_html=True)
        st.write("MACD: Positive")
        st.write("Volume: Increasing")
    
    with signal_col3:
        st.markdown('<div class="signal-neutral">NEUTRAL</div>', unsafe_allow_html=True)
        st.write("Stochastic: Neutral")
        st.write("Trend: Sideways")
    
    with signal_col4:
        st.markdown('<div class="signal-sell">SELL</div>', unsafe_allow_html=True)
        st.write("Bollinger: Upper Band")
        st.write("RSI: Overbought")
    
    # Charts section
    st.subheader("📈 Technical Analysis")
    
    # Generate sample data for demonstration
    dates = pd.date_range(start='2024-01-01', end='2024-03-20', freq='D')
    prices = 100 + np.cumsum(np.random.randn(len(dates)) * 2)
    
    # Create sample DataFrame
    df = pd.DataFrame({
        'Date': dates,
        'Price': prices,
        'MA_Fast': prices.rolling(window=ma_fast).mean(),
        'MA_Slow': prices.rolling(window=ma_slow).mean(),
        'RSI': 50 + np.random.randn(len(dates)) * 10  # Sample RSI values
    })
    
    # Price chart with moving averages
    fig_price = go.Figure()
    fig_price.add_trace(go.Scatter(x=df['Date'], y=df['Price'], mode='lines', name='Price', line=dict(color='#667eea')))
    fig_price.add_trace(go.Scatter(x=df['Date'], y=df['MA_Fast'], mode='lines', name=f'MA {ma_fast}', line=dict(color='#ff4444')))
    fig_price.add_trace(go.Scatter(x=df['Date'], y=df['MA_Slow'], mode='lines', name=f'MA {ma_slow}', line=dict(color='#00ff88')))
    
    fig_price.update_layout(
        title=f'{trading_pair} Price Chart with Moving Averages',
        xaxis_title='Date',
        yaxis_title='Price',
        template='plotly_dark',
        height=400
    )
    
    # RSI chart
    fig_rsi = go.Figure()
    fig_rsi.add_trace(go.Scatter(x=df['Date'], y=df['RSI'], mode='lines', name='RSI', line=dict(color='#ffa500')))
    fig_rsi.add_hline(y=70, line_dash="dash", line_color="red", annotation_text="Overbought")
    fig_rsi.add_hline(y=30, line_dash="dash", line_color="green", annotation_text="Oversold")
    
    fig_rsi.update_layout(
        title='RSI Indicator',
        xaxis_title='Date',
        yaxis_title='RSI',
        template='plotly_dark',
        height=300
    )
    
    # Display charts
    st.plotly_chart(fig_price, use_container_width=True)
    st.plotly_chart(fig_rsi, use_container_width=True)
    
    # Trading history
    st.subheader("📋 Recent Trading Activity")
    
    # Sample trading history
    trades_data = {
        'Date': ['2024-03-20 10:30', '2024-03-20 09:15', '2024-03-19 14:20', '2024-03-19 11:45'],
        'Pair': [trading_pair, trading_pair, trading_pair, trading_pair],
        'Action': ['BUY', 'SELL', 'BUY', 'SELL'],
        'Quantity': [1.5, 1.5, 2.0, 2.0],
        'Price': [51200, 51500, 50800, 51000],
        'P&L': [450, 300, -150, 400]
    }
    
    trades_df = pd.DataFrame(trades_data)
    
    # Style the P&L column
    def style_pnl(val):
        color = 'color: #00ff88' if val > 0 else 'color: #ff4444'
        return color
    
    styled_trades = trades_df.style.applymap(style_pnl, subset=['P&L'])
    st.dataframe(styled_trades, use_container_width=True)
    
    # Control panel
    st.subheader("🎮 Trading Controls")
    
    control_col1, control_col2, control_col3, control_col4 = st.columns(4)
    
    with control_col1:
        if st.button("🔄 Start Live Trading"):
            st.success("Live trading started!")
    
    with control_col2:
        if st.button("⏸️ Pause Trading"):
            st.warning("Trading paused")
    
    with control_col3:
        if st.button("🛑 Emergency Stop"):
            st.error("EMERGENCY STOP ACTIVATED!")
    
    with control_col4:
        if st.button("📊 Generate Report"):
            st.info("Performance report generated!")

if __name__ == "__main__":
    main()

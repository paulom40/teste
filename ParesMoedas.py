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

# Technical indicator functions
def calculate_rsi(prices, period=14):
    """Calculate RSI indicator"""
    delta = prices.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    return rsi

def calculate_moving_averages(prices, fast_period=20, slow_period=50):
    """Calculate moving averages"""
    ma_fast = prices.rolling(window=fast_period).mean()
    ma_slow = prices.rolling(window=slow_period).mean()
    return ma_fast, ma_slow

def generate_sample_data(days=80, initial_price=100, volatility=2):
    """Generate realistic sample price data"""
    np.random.seed(42)  # For reproducible results
    dates = pd.date_range(start=datetime.now() - timedelta(days=days), 
                         end=datetime.now(), freq='D')
    
    # Generate more realistic price data with trends
    returns = np.random.randn(len(dates)) * volatility / 100
    prices = initial_price * (1 + returns).cumprod()
    
    return pd.DataFrame({'Date': dates, 'Price': prices})

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
    
    # Generate sample data
    df = generate_sample_data()
    
    # Calculate indicators
    df['RSI'] = calculate_rsi(df['Price'], period=rsi_period)
    df['MA_Fast'], df['MA_Slow'] = calculate_moving_averages(df['Price'], ma_fast, ma_slow)
    
    # Main content area
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        current_price = df['Price'].iloc[-1]
        st.metric("Current Price", f"${current_price:.2f}", delta="+2.3%")
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
    
    # Determine current signal based on indicators
    current_rsi = df['RSI'].iloc[-1]
    current_ma_fast = df['MA_Fast'].iloc[-1]
    current_ma_slow = df['MA_Slow'].iloc[-1]
    current_price = df['Price'].iloc[-1]
    
    # Simple signal logic
    if (current_rsi < 30 and current_price > current_ma_fast and current_ma_fast > current_ma_slow):
        signal_class = "signal-strong-buy"
        signal_text = "STRONG BUY"
        signal_reason = "RSI Oversold + Bullish MA Alignment"
    elif (current_rsi > 70 and current_price < current_ma_fast and current_ma_fast < current_ma_slow):
        signal_class = "signal-strong-sell"
        signal_text = "STRONG SELL"
        signal_reason = "RSI Overbought + Bearish MA Alignment"
    elif (current_rsi < 40 and current_price > current_ma_fast):
        signal_class = "signal-buy"
        signal_text = "BUY"
        signal_reason = "RSI Approaching Oversold + Price above Fast MA"
    elif (current_rsi > 60 and current_price < current_ma_fast):
        signal_class = "signal-sell"
        signal_text = "SELL"
        signal_reason = "RSI Approaching Overbought + Price below Fast MA"
    else:
        signal_class = "signal-neutral"
        signal_text = "NEUTRAL"
        signal_reason = "Waiting for clearer signals"
    
    signal_col1, signal_col2, signal_col3, signal_col4 = st.columns(4)
    
    with signal_col1:
        st.markdown(f'<div class="{signal_class}">{signal_text}</div>', unsafe_allow_html=True)
        st.write(signal_reason)
        st.write(f"RSI: {current_rsi:.1f}")
    
    with signal_col2:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        st.metric("RSI", f"{current_rsi:.1f}", 
                 delta="Oversold" if current_rsi < 30 else "Overbought" if current_rsi > 70 else "Neutral")
        st.markdown('</div>', unsafe_allow_html=True)
    
    with signal_col3:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        ma_signal = "Bullish" if current_ma_fast > current_ma_slow else "Bearish"
        st.metric("MA Signal", ma_signal)
        st.markdown('</div>', unsafe_allow_html=True)
    
    with signal_col4:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        price_vs_ma = "Above MA" if current_price > current_ma_fast else "Below MA"
        st.metric("Price Position", price_vs_ma)
        st.markdown('</div>', unsafe_allow_html=True)
    
    # Charts section
    st.subheader("📈 Technical Analysis")
    
    # Price chart with moving averages
    fig_price = go.Figure()
    fig_price.add_trace(go.Scatter(x=df['Date'], y=df['Price'], mode='lines', name='Price', line=dict(color='#667eea', width=2)))
    fig_price.add_trace(go.Scatter(x=df['Date'], y=df['MA_Fast'], mode='lines', name=f'MA {ma_fast}', line=dict(color='#ff4444', width=1.5)))
    fig_price.add_trace(go.Scatter(x=df['Date'], y=df['MA_Slow'], mode='lines', name=f'MA {ma_slow}', line=dict(color='#00ff88', width=1.5)))
    
    fig_price.update_layout(
        title=f'{trading_pair} Price Chart with Moving Averages',
        xaxis_title='Date',
        yaxis_title='Price ($)',
        template='plotly_dark',
        height=400,
        showlegend=True
    )
    
    # RSI chart
    fig_rsi = go.Figure()
    fig_rsi.add_trace(go.Scatter(x=df['Date'], y=df['RSI'], mode='lines', name='RSI', line=dict(color='#ffa500', width=2)))
    fig_rsi.add_hline(y=70, line_dash="dash", line_color="red", annotation_text="Overbought")
    fig_rsi.add_hline(y=30, line_dash="dash", line_color="green", annotation_text="Oversold")
    fig_rsi.add_hline(y=50, line_dash="dot", line_color="gray")
    
    fig_rsi.update_layout(
        title='RSI Indicator',
        xaxis_title='Date',
        yaxis_title='RSI',
        template='plotly_dark',
        height=300,
        yaxis_range=[0, 100]
    )
    
    # Display charts
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.plotly_chart(fig_price, use_container_width=True)
    
    with col2:
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

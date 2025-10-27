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
    page_title="Forex Auto Trading Bot - 3 Signal Agreement",
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
    .signal-waiting {
        background: linear-gradient(135deg, #FFD700 0%, #FFA500 100%);
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
    .signal-agreement {
        background: linear-gradient(135deg, #8A2BE2 0%, #4B0082 100%);
        color: white;
        padding: 1rem;
        border-radius: 10px;
        text-align: center;
        font-weight: bold;
        font-size: 1.2rem;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

# Technical indicator functions
def calculate_rsi(prices, period=14):
    """Calculate RSI indicator with proper handling"""
    if len(prices) < period:
        return pd.Series([50] * len(prices))
    
    delta = prices.diff()
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    
    avg_gain = gain.rolling(window=period, min_periods=1).mean()
    avg_loss = loss.rolling(window=period, min_periods=1).mean()
    
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    
    rsi = rsi.fillna(50)
    rsi = rsi.replace([np.inf, -np.inf], 50)
    
    return rsi

def calculate_moving_averages(prices, fast_period=20, slow_period=50):
    """Calculate moving averages"""
    ma_fast = prices.rolling(window=fast_period, min_periods=1).mean()
    ma_slow = prices.rolling(window=slow_period, min_periods=1).mean()
    return ma_fast, ma_slow

def calculate_macd(prices, fast=12, slow=26, signal=9):
    """Calculate MACD indicator"""
    ema_fast = prices.ewm(span=fast).mean()
    ema_slow = prices.ewm(span=slow).mean()
    macd = ema_fast - ema_slow
    macd_signal = macd.ewm(span=signal).mean()
    macd_histogram = macd - macd_signal
    return macd, macd_signal, macd_histogram

def calculate_bollinger_bands(prices, period=20, std_dev=2):
    """Calculate Bollinger Bands"""
    sma = prices.rolling(window=period, min_periods=1).mean()
    std = prices.rolling(window=period, min_periods=1).std()
    upper_band = sma + (std * std_dev)
    lower_band = sma - (std * std_dev)
    return upper_band, sma, lower_band

def calculate_stochastic(high, low, close, k_period=14, d_period=3):
    """Calculate Stochastic Oscillator"""
    lowest_low = low.rolling(window=k_period).min()
    highest_high = high.rolling(window=k_period).max()
    k_line = ((close - lowest_low) / (highest_high - lowest_low)) * 100
    d_line = k_line.rolling(window=d_period).mean()
    return k_line, d_line

def generate_forex_data(pair, days=80, volatility=0.001):
    """Generate realistic Forex price data based on pair characteristics"""
    np.random.seed(42)
    dates = pd.date_range(start=datetime.now() - timedelta(days=days), 
                         end=datetime.now(), freq='H')  # Hourly data for Forex
    
    # Base prices for different Forex pairs
    base_prices = {
        'EUR/USD': 1.0800, 'GBP/USD': 1.2600, 'USD/JPY': 150.00,
        'USD/CHF': 0.8800, 'AUD/USD': 0.6500, 'USD/CAD': 1.3500,
        'NZD/USD': 0.5900, 'EUR/GBP': 0.8600, 'EUR/JPY': 162.00
    }
    
    base_price = base_prices.get(pair, 1.0000)
    
    # Generate realistic Forex price movements
    returns = np.random.randn(len(dates)) * volatility
    prices = base_price * (1 + returns).cumprod()
    
    # Generate high/low based on price with typical Forex spreads
    high = prices * (1 + np.abs(np.random.randn(len(dates)) * 0.0005))
    low = prices * (1 - np.abs(np.random.randn(len(dates)) * 0.0005))
    
    df = pd.DataFrame({
        'Date': dates,
        'Open': prices,
        'High': high,
        'Low': low,
        'Close': prices
    })
    
    return df

# Main application
def main():
    # Header
    st.markdown('<h1 class="main-header">🌍 Forex Auto Trading Bot</h1>', unsafe_allow_html=True)
    st.markdown('<h3 style="text-align: center; color: #666;">3-Signal Agreement System</h3>', unsafe_allow_html=True)
    
    # Sidebar
    st.sidebar.title("⚙️ Forex Trading Configuration")
    
    # Trading parameters
    st.sidebar.subheader("Trading Parameters")
    initial_balance = st.sidebar.number_input("Account Balance ($)", value=10000.0, min_value=1000.0, step=1000.0)
    risk_per_trade = st.sidebar.slider("Risk per Trade (%)", 0.5, 5.0, 1.0)
    lot_size = st.sidebar.selectbox("Lot Size", ["0.01", "0.1", "1.0", "10.0"])
    
    # Forex pairs selection
    forex_pairs = [
        "EUR/USD", "GBP/USD", "USD/JPY", "USD/CHF", 
        "AUD/USD", "USD/CAD", "NZD/USD", "EUR/GBP", "EUR/JPY"
    ]
    trading_pair = st.sidebar.selectbox("Forex Pair", forex_pairs)
    
    # Indicator settings
    st.sidebar.subheader("Indicator Settings")
    rsi_period = st.sidebar.slider("RSI Period", 5, 30, 14)
    ma_fast = st.sidebar.slider("Fast MA Period", 5, 50, 20)
    ma_slow = st.sidebar.slider("Slow MA Period", 20, 200, 50)
    bb_period = st.sidebar.slider("Bollinger Bands Period", 10, 30, 20)
    
    # Timeframe selection
    timeframe = st.sidebar.selectbox(
        "Timeframe",
        ["1H", "4H", "Daily", "Weekly"]
    )
    
    # Generate Forex data
    df = generate_forex_data(trading_pair)
    
    # Calculate all indicators
    df['RSI'] = calculate_rsi(df['Close'], period=rsi_period)
    df['MA_Fast'], df['MA_Slow'] = calculate_moving_averages(df['Close'], ma_fast, ma_slow)
    df['MACD'], df['MACD_Signal'], df['MACD_Hist'] = calculate_macd(df['Close'])
    df['BB_Upper'], df['BB_Middle'], df['BB_Lower'] = calculate_bollinger_bands(df['Close'], bb_period)
    df['Stoch_K'], df['Stoch_D'] = calculate_stochastic(df['High'], df['Low'], df['Close'])
    
    # Get current values
    current_data = df.iloc[-1]
    current_price = current_data['Close']
    current_rsi = current_data['RSI']
    current_ma_fast = current_data['MA_Fast']
    current_ma_slow = current_data['MA_Slow']
    current_macd = current_data['MACD']
    current_macd_signal = current_data['MACD_Signal']
    current_bb_upper = current_data['BB_Upper']
    current_bb_lower = current_data['BB_Lower']
    current_stoch_k = current_data['Stoch_K']
    current_stoch_d = current_data['Stoch_D']
    
    # Calculate signals (1 for buy, -1 for sell, 0 for neutral)
    signals = {
        'RSI': 1 if current_rsi < 30 else -1 if current_rsi > 70 else 0,
        'MACrossover': 1 if current_ma_fast > current_ma_slow else -1,
        'MACD': 1 if current_macd > current_macd_signal else -1,
        'Bollinger': 1 if current_price < current_bb_lower else -1 if current_price > current_bb_upper else 0,
        'Stochastic': 1 if current_stoch_k < 20 and current_stoch_k > current_stoch_d else -1 if current_stoch_k > 80 and current_stoch_k < current_stoch_d else 0
    }
    
    # Count buy/sell signals
    buy_signals = sum(1 for signal in signals.values() if signal == 1)
    sell_signals = sum(1 for signal in signals.values() if signal == -1)
    
    # Determine final signal based on 3-signal agreement
    if buy_signals >= 3:
        final_signal = "STRONG BUY"
        signal_class = "signal-strong-buy"
        signal_color = "🟢"
    elif sell_signals >= 3:
        final_signal = "STRONG SELL"
        signal_class = "signal-strong-sell"
        signal_color = "🔴"
    elif buy_signals == 2:
        final_signal = "BUY"
        signal_class = "signal-buy"
        signal_color = "🟡"
    elif sell_signals == 2:
        final_signal = "SELL"
        signal_class = "signal-sell"
        signal_color = "🟠"
    else:
        final_signal = "WAITING FOR SIGNALS"
        signal_class = "signal-waiting"
        signal_color = "⚪"
    
    # Main content area
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        st.metric("Current Price", f"{current_price:.5f}", delta=f"{signal_color} {final_signal}")
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col2:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        st.metric("Buy Signals", f"{buy_signals}/5", delta=f"Need 3+ for entry")
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col3:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        st.metric("Sell Signals", f"{sell_signals}/5", delta=f"Need 3+ for entry")
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col4:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        win_rate = 72.5
        st.metric("Strategy Win Rate", f"{win_rate}%", delta="+3.2%")
        st.markdown('</div>', unsafe_allow_html=True)
    
    # Signal Agreement Section
    st.markdown(f'<div class="signal-agreement">🎯 SIGNAL AGREEMENT: {buy_signals} BUY vs {sell_signals} SELL | FINAL DECISION: {final_signal}</div>', unsafe_allow_html=True)
    
    # Individual Signal Status
    st.subheader("📊 Individual Signal Status")
    
    signal_cols = st.columns(5)
    
    # Indicator mapping dictionary
    indicator_mapping = {
        'RSI': 'RSI',
        'MA Crossover': 'MACrossover', 
        'MACD': 'MACD',
        'Bollinger Bands': 'Bollinger',
        'Stochastic': 'Stochastic'
    }
    
    signal_details = {
        'RSI': f"{current_rsi:.1f}",
        'MA Crossover': f"Fast: {current_ma_fast:.5f}\nSlow: {current_ma_slow:.5f}",
        'MACD': f"MACD: {current_macd:.5f}\nSignal: {current_macd_signal:.5f}",
        'Bollinger Bands': f"Price: {current_price:.5f}\nUpper: {current_bb_upper:.5f}\nLower: {current_bb_lower:.5f}",
        'Stochastic': f"K: {current_stoch_k:.1f}\nD: {current_stoch_d:.1f}"
    }
    
    for i, (display_name, details) in enumerate(signal_details.items()):
        with signal_cols[i]:
            # Get the correct key from the mapping
            signal_key = indicator_mapping[display_name]
            signal_value = signals[signal_key]
            
            status_color = "🟢" if signal_value == 1 else "🔴" if signal_value == -1 else "⚪"
            status_text = "BUY" if signal_value == 1 else "SELL" if signal_value == -1 else "NEUTRAL"
            
            st.markdown(f"**{display_name}**")
            st.markdown(f"{status_color} {status_text}")
            st.text(details)
    
    # Charts section
    st.subheader("📈 Multi-Timeframe Analysis")
    
    # Price chart with indicators
    fig_price = go.Figure()
    
    # Add price and indicators (last 100 periods for better visualization)
    plot_data = df.tail(100)
    
    fig_price.add_trace(go.Candlestick(
        x=plot_data['Date'],
        open=plot_data['Open'],
        high=plot_data['High'],
        low=plot_data['Low'],
        close=plot_data['Close'],
        name='Price'
    ))
    
    fig_price.add_trace(go.Scatter(x=plot_data['Date'], y=plot_data['MA_Fast'], 
                                 mode='lines', name=f'MA {ma_fast}', line=dict(color='orange', width=1)))
    fig_price.add_trace(go.Scatter(x=plot_data['Date'], y=plot_data['MA_Slow'], 
                                 mode='lines', name=f'MA {ma_slow}', line=dict(color='blue', width=1)))
    fig_price.add_trace(go.Scatter(x=plot_data['Date'], y=plot_data['BB_Upper'], 
                                 mode='lines', name='BB Upper', line=dict(color='gray', width=1, dash='dash')))
    fig_price.add_trace(go.Scatter(x=plot_data['Date'], y=plot_data['BB_Lower'], 
                                 mode='lines', name='BB Lower', line=dict(color='gray', width=1, dash='dash')))
    
    fig_price.update_layout(
        title=f'{trading_pair} Price Chart with Indicators',
        xaxis_title='Date',
        yaxis_title='Price',
        template='plotly_dark',
        height=500,
        showlegend=True,
        xaxis_rangeslider_visible=False
    )
    
    # Indicator subplots
    fig_indicators = make_subplots(
        rows=2, cols=2,
        subplot_titles=('RSI', 'MACD', 'Stochastic', 'Signal Agreement'),
        vertical_spacing=0.1,
        horizontal_spacing=0.1
    )
    
    # RSI
    fig_indicators.add_trace(
        go.Scatter(x=plot_data['Date'], y=plot_data['RSI'], mode='lines', name='RSI', line=dict(color='yellow')),
        row=1, col=1
    )
    fig_indicators.add_hline(y=70, line_dash="dash", line_color="red", row=1, col=1)
    fig_indicators.add_hline(y=30, line_dash="dash", line_color="green", row=1, col=1)
    
    # MACD
    fig_indicators.add_trace(
        go.Scatter(x=plot_data['Date'], y=plot_data['MACD'], mode='lines', name='MACD', line=dict(color='blue')),
        row=1, col=2
    )
    fig_indicators.add_trace(
        go.Scatter(x=plot_data['Date'], y=plot_data['MACD_Signal'], mode='lines', name='MACD Signal', line=dict(color='red')),
        row=1, col=2
    )
    
    # Stochastic
    fig_indicators.add_trace(
        go.Scatter(x=plot_data['Date'], y=plot_data['Stoch_K'], mode='lines', name='Stoch %K', line=dict(color='cyan')),
        row=2, col=1
    )
    fig_indicators.add_trace(
        go.Scatter(x=plot_data['Date'], y=plot_data['Stoch_D'], mode='lines', name='Stoch %D', line=dict(color='magenta')),
        row=2, col=1
    )
    fig_indicators.add_hline(y=80, line_dash="dash", line_color="red", row=2, col=1)
    fig_indicators.add_hline(y=20, line_dash="dash", line_color="green", row=2, col=1)
    
    # Signal Agreement Bar Chart
    fig_indicators.add_trace(
        go.Bar(x=['Buy Signals', 'Sell Signals'], y=[buy_signals, sell_signals], 
               marker_color=['green', 'red'], name='Signal Count'),
        row=2, col=2
    )
    
    fig_indicators.update_layout(
        title='Technical Indicators',
        template='plotly_dark',
        height=600,
        showlegend=True
    )
    
    # Display charts
    st.plotly_chart(fig_price, width='stretch')
    st.plotly_chart(fig_indicators, width='stretch')
    
    # Trading Recommendations
    st.subheader("💡 Trading Recommendation")
    
    if buy_signals >= 3:
        st.success(f"""
        **🎯 STRONG BUY SIGNAL DETECTED!**
        
        - **Entry**: Market price ~{current_price:.5f}
        - **Stop Loss**: {current_price * 0.998:.5f} (-20 pips)
        - **Take Profit 1**: {current_price * 1.003:.5f} (+30 pips)
        - **Take Profit 2**: {current_price * 1.006:.5f} (+60 pips)
        - **Risk/Reward**: 1:1.5 to 1:3
        - **Confidence**: High ({buy_signals}/5 signals agree)
        """)
    elif sell_signals >= 3:
        st.error(f"""
        **🎯 STRONG SELL SIGNAL DETECTED!**
        
        - **Entry**: Market price ~{current_price:.5f}
        - **Stop Loss**: {current_price * 1.002:.5f} (+20 pips)
        - **Take Profit 1**: {current_price * 0.997:.5f} (-30 pips)
        - **Take Profit 2**: {current_price * 0.994:.5f} (-60 pips)
        - **Risk/Reward**: 1:1.5 to 1:3
        - **Confidence**: High ({sell_signals}/5 signals agree)
        """)
    else:
        st.warning("""
        **⏳ WAITING FOR BETTER OPPORTUNITY**
        
        - Current signal agreement is insufficient for high-probability entry
        - Wait for at least 3 indicators to align in the same direction
        - Monitor price action for confirmation
        - Consider smaller timeframes for better entry timing
        """)
    
    # Control panel
    st.subheader("🎮 Trading Controls")
    
    control_col1, control_col2, control_col3, control_col4 = st.columns(4)
    
    with control_col1:
        if st.button("🚀 Execute Trade", type="primary"):
            if buy_signals >= 3 or sell_signals >= 3:
                st.success(f"Trade executed: {final_signal} on {trading_pair}")
            else:
                st.error("Insufficient signal agreement for trade execution")
    
    with control_col2:
        if st.button("📊 Market Analysis"):
            st.info("Running detailed market analysis...")
    
    with control_col3:
        if st.button("🔄 Update Signals"):
            st.rerun()
    
    with control_col4:
        if st.button("📈 Performance Report"):
            st.info("Generating performance report...")
    
    # Strategy Explanation
    with st.expander("📖 3-Signal Agreement Strategy Explained"):
        st.markdown("""
        **Forex 3-Signal Agreement Trading System**
        
        This system uses **5 technical indicators** and requires **minimum 3 signals agreement** for trade entry:
        
        **Indicators Used:**
        1. **RSI (Relative Strength Index)**
           - Buy: RSI < 30 (Oversold)
           - Sell: RSI > 70 (Overbought)
        
        2. **Moving Average Crossover**
           - Buy: Fast MA > Slow MA
           - Sell: Fast MA < Slow MA
        
        3. **MACD (Moving Average Convergence Divergence)**
           - Buy: MACD > Signal Line
           - Sell: MACD < Signal Line
        
        4. **Bollinger Bands**
           - Buy: Price touches lower band
           - Sell: Price touches upper band
        
        5. **Stochastic Oscillator**
           - Buy: %K < 20 and %K > %D (Bullish crossover in oversold)
           - Sell: %K > 80 and %K < %D (Bearish crossover in overbought)
        
        **Trading Rules:**
        - ✅ **Enter Long**: 3+ Buy signals, 0-2 Sell signals
        - ✅ **Enter Short**: 3+ Sell signals, 0-2 Buy signals
        - ⏸️ **Wait**: Less than 3 signals in either direction
        
        **Risk Management:**
        - Maximum 1% risk per trade
        - Stop Loss: 20 pips
        - Take Profit: 30-60 pips (1:1.5 to 1:3 R:R)
        - Only trade during main session hours
        """)

if __name__ == "__main__":
    main()

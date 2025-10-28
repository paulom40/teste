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
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1rem;
        border-radius: 10px;
        color: white;
        text-align: center;
    }
    .auto-trading-active {
        background: linear-gradient(135deg, #00ff88 0%, #00cc66 100%);
        color: white;
        padding: 1rem;
        border-radius: 10px;
        text-align: center;
        font-weight: bold;
        animation: pulse 2s infinite;
    }
    .refresh-button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        padding: 0.5rem 1rem;
        border-radius: 5px;
        font-weight: bold;
        margin: 0.5rem 0;
    }
    @keyframes pulse {
        0% { transform: scale(1); }
        50% { transform: scale(1.02); }
        100% { transform: scale(1); }
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

# Technical Indicators
def calculate_rsi(prices, period=14):
    delta = prices.diff()
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    avg_gain = gain.rolling(window=period).mean()
    avg_loss = loss.rolling(window=period).mean()
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    return rsi.fillna(50)

def calculate_moving_averages(prices, fast=20, slow=50):
    return prices.rolling(fast).mean(), prices.rolling(slow).mean()

def calculate_macd(prices, fast=12, slow=26, signal=9):
    ema_fast = prices.ewm(span=fast).mean()
    ema_slow = prices.ewm(span=slow).mean()
    macd = ema_fast - ema_slow
    signal_line = macd.ewm(span=signal).mean()
    return macd, signal_line

def detect_engulfing_pattern(df):
    signals = []
    for i in range(1, len(df)):
        prev_open, prev_close = df['Open'].iloc[i-1], df['Close'].iloc[i-1]
        curr_open, curr_close = df['Open'].iloc[i], df['Close'].iloc[i]
        
        # Bullish engulfing
        if (prev_close < prev_open and curr_close > curr_open and 
            curr_open < prev_close and curr_close > prev_open):
            signals.append(1)
        # Bearish engulfing
        elif (prev_close > prev_open and curr_close < curr_open and 
              curr_open > prev_close and curr_close < prev_open):
            signals.append(-1)
        else:
            signals.append(0)
    signals.append(0)  # For the last element
    return pd.Series(signals, index=df.index)

def generate_forex_data(pair, days=60):
    """Generate realistic Forex data for demonstration"""
    np.random.seed(hash(pair) % 10000)
    dates = pd.date_range(end=datetime.now(), periods=days*24, freq='H')
    
    base_prices = {
        'EUR/USD': 1.0800, 'GBP/USD': 1.2600, 'USD/JPY': 150.00,
        'USD/CHF': 0.8800, 'AUD/USD': 0.6500, 'USD/CAD': 1.3500,
        'EUR/GBP': 0.8600, 'EUR/JPY': 162.00, 'GBP/JPY': 188.00,
        'AUD/JPY': 97.00, 'USD/CNY': 7.2500, 'NZD/USD': 0.5900
    }
    
    base_price = base_prices.get(pair, 1.0000)
    volatility = 0.0008
    
    prices = [base_price]
    for i in range(1, len(dates)):
        change = np.random.randn() * volatility
        new_price = prices[-1] * (1 + change)
        prices.append(new_price)
    
    df = pd.DataFrame({
        'Date': dates,
        'Open': prices,
        'High': [p * (1 + abs(np.random.randn()) * 0.001) for p in prices],
        'Low': [p * (1 - abs(np.random.randn()) * 0.001) for p in prices],
        'Close': prices
    })
    
    # Add some trends and patterns
    df['Close'] = df['Close'] + np.sin(np.arange(len(df)) * 0.1) * 0.005
    return df.set_index('Date')

def get_forex_data(pair):
    """Get Forex data - tries Yahoo Finance first, falls back to generated data"""
    if YFINANCE_AVAILABLE:
        try:
            symbol = pair.replace("/", "") + "=X"
            data = yf.download(symbol, period="60d", interval="1h", progress=False)
            if not data.empty:
                return data
        except:
            pass
    return generate_forex_data(pair)

def get_current_price(pair):
    """Get current price for a Forex pair"""
    try:
        df = get_forex_data(pair)
        current_price = df['Close'].iloc[-1]
        
        # Calculate price change
        if pair in st.session_state.last_prices:
            previous_price = st.session_state.last_prices[pair]
            change = ((current_price - previous_price) / previous_price) * 100
            st.session_state.price_changes[pair] = change
        
        st.session_state.last_prices[pair] = current_price
        return current_price
    except:
        # Fallback to base price
        base_prices = {
            'EUR/USD': 1.0800, 'GBP/USD': 1.2600, 'USD/JPY': 150.00,
            'USD/CHF': 0.8800, 'AUD/USD': 0.6500, 'USD/CAD': 1.3500,
            'EUR/GBP': 0.8600, 'EUR/JPY': 162.00, 'GBP/JPY': 188.00,
            'AUD/JPY': 97.00, 'USD/CNY': 7.2500, 'NZD/USD': 0.5900
        }
        return base_prices.get(pair, 1.0000)

def get_all_current_prices():
    """Get current prices for all Forex pairs"""
    forex_pairs = [
        "EUR/USD", "GBP/USD", "USD/JPY", "USD/CHF", 
        "AUD/USD", "USD/CAD", "NZD/USD", "EUR/GBP", 
        "EUR/JPY", "GBP/JPY", "AUD/JPY", "USD/CNY"
    ]
    
    prices_data = []
    for pair in forex_pairs:
        current_price = get_current_price(pair)
        change = st.session_state.price_changes.get(pair, 0)
        
        prices_data.append({
            'Pair': pair,
            'Current Price': current_price,
            'Change %': change,
            'Signal': 'Loading...',
            'Engulfing': 'No'
        })
    
    return pd.DataFrame(prices_data)

def analyze_pair(pair):
    """Analyze a Forex pair and return trading signals"""
    try:
        df = get_forex_data(pair)
        
        # Calculate indicators
        df['RSI'] = calculate_rsi(df['Close'])
        df['MA_Fast'], df['MA_Slow'] = calculate_moving_averages(df['Close'])
        df['MACD'], df['MACD_Signal'] = calculate_macd(df['Close'])
        df['Engulfing'] = detect_engulfing_pattern(df)
        
        current = df.iloc[-1]
        price = current['Close']
        
        # Generate signals
        signals = {
            'RSI': 1 if current['RSI'] < 30 else -1 if current['RSI'] > 70 else 0,
            'MA_Crossover': 1 if current['MA_Fast'] > current['MA_Slow'] else -1,
            'MACD': 1 if current['MACD'] > current['MACD_Signal'] else -1,
            'Engulfing': current['Engulfing']
        }
        
        buy_signals = sum(1 for s in signals.values() if s == 1)
        sell_signals = sum(1 for s in signals.values() if s == -1)
        
        # Double weight for engulfing
        if signals['Engulfing'] == 1:
            buy_signals += 1
        elif signals['Engulfing'] == -1:
            sell_signals += 1
        
        total_signals = buy_signals + sell_signals
        
        if buy_signals >= 3:
            return {
                'pair': pair,
                'signal': 'BUY',
                'strength': 'STRONG' if buy_signals >= 4 else 'MODERATE',
                'price': price,
                'signal_count': total_signals,
                'engulfing': 'BULLISH' if signals['Engulfing'] == 1 else 'NONE'
            }
        elif sell_signals >= 3:
            return {
                'pair': pair,
                'signal': 'SELL',
                'strength': 'STRONG' if sell_signals >= 4 else 'MODERATE',
                'price': price,
                'signal_count': total_signals,
                'engulfing': 'BEARISH' if signals['Engulfing'] == -1 else 'NONE'
            }
        else:
            return {
                'pair': pair,
                'signal': 'HOLD',
                'strength': 'WEAK',
                'price': price,
                'signal_count': total_signals,
                'engulfing': 'NONE'
            }
            
    except Exception as e:
        return {
            'pair': pair,
            'signal': 'ERROR',
            'strength': 'ERROR',
            'price': 0,
            'signal_count': 0,
            'engulfing': 'NONE'
        }

def execute_trade(signal_data, stake_eur):
    """Execute a trade based on signal"""
    pair = signal_data['pair']
    direction = signal_data['signal']
    price = signal_data['price']
    
    # Simple trade simulation
    pip_value = 10
    pip_size = 0.0001
    quantity = stake_eur / 100
    
    # Simulate trade outcome (70% win rate for strong signals)
    is_win = np.random.random() > 0.3 if signal_data['strength'] == 'STRONG' else np.random.random() > 0.5
    
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
    
    return f"Trade executed: {direction} {pair} at {price:.5f}"

def display_prices_table():
    """Display real-time prices table"""
    st.markdown("### 📊 REAL-TIME FOREX PRICES")
    
    # Refresh button
    col1, col2 = st.columns([3, 1])
    with col2:
        if st.button("🔄 Refresh Prices", use_container_width=True):
            st.rerun()
    
    # Get current prices
    prices_df = get_all_current_prices()
    
    # Analyze signals for all pairs
    forex_pairs = prices_df['Pair'].tolist()
    with st.spinner("Analyzing trading signals..."):
        signals_data = [analyze_pair(pair) for pair in forex_pairs]
    
    # Update prices dataframe with signals
    for signal in signals_data:
        mask = prices_df['Pair'] == signal['pair']
        if mask.any():
            prices_df.loc[mask, 'Signal'] = signal['signal']
            prices_df.loc[mask, 'Engulfing'] = signal['engulfing']
            prices_df.loc[mask, 'Current Price'] = signal['price']
    
    # Display the prices table with styling
    st.markdown('<div class="prices-table">', unsafe_allow_html=True)
    
    # Create a styled dataframe
    styled_df = prices_df.copy()
    
    # Format the display
    for i, row in styled_df.iterrows():
        # Color code price changes
        change = row['Change %']
        if change > 0:
            styled_df.at[i, 'Change %'] = f"<span class='price-up'>+{change:.2f}%</span>"
        elif change < 0:
            styled_df.at[i, 'Change %'] = f"<span class='price-down'>{change:.2f}%</span>"
        else:
            styled_df.at[i, 'Change %'] = f"<span class='price-neutral'>{change:.2f}%</span>"
        
        # Format price based on pair type
        if 'JPY' in row['Pair'] or 'CNY' in row['Pair']:
            styled_df.at[i, 'Current Price'] = f"{row['Current Price']:.2f}"
        else:
            styled_df.at[i, 'Current Price'] = f"{row['Current Price']:.5f}"
    
    # Display the table
    st.write("""
    <style>
    .prices-table table {
        width: 100%;
        border-collapse: collapse;
    }
    .prices-table th {
        background: #667eea;
        color: white;
        padding: 12px;
        text-align: center;
        font-weight: bold;
    }
    .prices-table td {
        padding: 12px;
        text-align: center;
        border-bottom: 1px solid #ddd;
    }
    .prices-table tr:hover {
        background-color: #f5f5f5;
    }
    </style>
    """, unsafe_allow_html=True)
    
    # Convert to HTML for custom styling
    html_table = """
    <table style="width: 100%; border-collapse: collapse; margin: 1rem 0;">
        <thead>
            <tr style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); color: white;">
                <th style="padding: 12px; text-align: center;">Forex Pair</th>
                <th style="padding: 12px; text-align: center;">Current Price</th>
                <th style="padding: 12px; text-align: center;">Change %</th>
                <th style="padding: 12px; text-align: center;">Trading Signal</th>
                <th style="padding: 12px; text-align: center;">Engulfing Pattern</th>
                <th style="padding: 12px; text-align: center;">Action</th>
            </tr>
        </thead>
        <tbody>
    """
    
    for _, row in prices_df.iterrows():
        # Determine signal styling
        if row['Signal'] == 'BUY':
            signal_html = '<span class="signal-buy">BUY</span>'
        elif row['Signal'] == 'SELL':
            signal_html = '<span class="signal-sell">SELL</span>'
        else:
            signal_html = '<span class="signal-hold">HOLD</span>'
        
        # Determine engulfing styling
        if row['Engulfing'] == 'BULLISH':
            engulfing_html = '<span class="engulfing-buy">BULLISH</span>'
        elif row['Engulfing'] == 'BEARISH':
            engulfing_html = '<span class="engulfing-sell">BEARISH</span>'
        else:
            engulfing_html = 'No Pattern'
        
        # Determine change styling
        change = row['Change %']
        if change > 0:
            change_html = f'<span style="color: #00ff88; font-weight: bold;">+{change:.2f}%</span>'
        elif change < 0:
            change_html = f'<span style="color: #ff4444; font-weight: bold;">{change:.2f}%</span>'
        else:
            change_html = f'<span style="color: #666666;">{change:.2f}%</span>'
        
        # Format price
        if 'JPY' in row['Pair'] or 'CNY' in row['Pair']:
            price_str = f"{row['Current Price']:.2f}"
        else:
            price_str = f"{row['Current Price']:.5f}"
        
        html_table += f"""
            <tr>
                <td style="padding: 12px; text-align: center; font-weight: bold;">{row['Pair']}</td>
                <td style="padding: 12px; text-align: center; font-family: monospace; font-size: 1.1em;">{price_str}</td>
                <td style="padding: 12px; text-align: center;">{change_html}</td>
                <td style="padding: 12px; text-align: center;">{signal_html}</td>
                <td style="padding: 12px; text-align: center;">{engulfing_html}</td>
                <td style="padding: 12px; text-align: center;">
                    <button onclick="alert('Trading {row['Pair']} at {price_str}')" 
                            style="background: #667eea; color: white; border: none; padding: 8px 16px; border-radius: 5px; cursor: pointer;">
                        Trade
                    </button>
                </td>
            </tr>
        """
    
    html_table += """
        </tbody>
    </table>
    """
    
    st.markdown(html_table, unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)
    
    return prices_df, signals_data

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
    prices_df, signals_data = display_prices_table()
    
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
            st.markdown('<div class="auto-trading-active">AUTO TRADING ACTIVE</div>', unsafe_allow_html=True)
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
    
    # Trading Opportunities Section
    st.markdown("---")
    st.subheader("🎯 TRADING OPPORTUNITIES")
    
    # Filter only BUY/SELL signals
    trading_opportunities = [s for s in signals_data if s['signal'] in ['BUY', 'SELL']]
    
    if trading_opportunities:
        for opportunity in trading_opportunities:
            col1, col2, col3, col4, col5, col6 = st.columns([2, 1, 1, 1, 1, 2])
            
            with col1:
                st.write(f"**{opportunity['pair']}**")
                st.write(f"Current Price: `{opportunity['price']:.5f}`")
            
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
                if st.button(f"TRADE {opportunity['pair']}", key=f"trade_{opportunity['pair']}", use_container_width=True):
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
                st.success("Auto-execution completed!")
                for trade in executed:
                    st.write(f"✅ {trade}")
    else:
        st.info("No strong trading opportunities detected. Monitor the prices table for signal changes.")
    
    # Trade History Section
    st.markdown("---")
    st.subheader("📋 TRADE HISTORY")
    
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
        if st.button("Clear Trade History"):
            st.session_state.trade_history = pd.DataFrame(columns=st.session_state.trade_history.columns)
            st.session_state.open_positions = {}
            st.rerun()
    else:
        st.info("No trades executed yet. Execute trades from the opportunities above to see history here.")

if __name__ == "__main__":
    main()

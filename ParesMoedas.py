import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import io

st.set_page_config(
    page_title="Forex Pro Bot - Manual Stake",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Estilo visual moderno
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
    .stake-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 1rem;
        border-radius: 10px;
        margin: 0.5rem 0;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }
    .manual-trade-card {
        background: linear-gradient(135deg, #00b09b 0%, #96c93d 100%);
        color: white;
        padding: 1rem;
        border-radius: 10px;
        margin: 1rem 0;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }
</style>
""", unsafe_allow_html=True)

# Pares de Forex
FOREX_PAIRS = {
    "EUR/USD": {"base_price": 1.0850, "volatility": 0.0008, "pip_value": 0.0001},
    "GBP/USD": {"base_price": 1.2650, "volatility": 0.0010, "pip_value": 0.0001},
    "USD/JPY": {"base_price": 148.50, "volatility": 0.15, "pip_value": 0.01},
    "USD/CHF": {"base_price": 0.8800, "volatility": 0.0009, "pip_value": 0.0001},
    "USD/CAD": {"base_price": 1.3550, "volatility": 0.0010, "pip_value": 0.0001},
    "AUD/USD": {"base_price": 0.6550, "volatility": 0.0012, "pip_value": 0.0001},
    "NZD/USD": {"base_price": 0.6050, "volatility": 0.0010, "pip_value": 0.0001},
    "EUR/GBP": {"base_price": 0.8350, "volatility": 0.0005, "pip_value": 0.0001},
    "EUR/JPY": {"base_price": 161.00, "volatility": 0.20, "pip_value": 0.01},
    "GBP/JPY": {"base_price": 188.50, "volatility": 0.25, "pip_value": 0.01}
}

DEFAULT_PARAMS = {
    'initial_bank': 10000.0,
    'profit_target_pips': 20.0,
    'stop_loss_pips': 12.0,
    'ma_fast': 7,
    'ma_slow': 25,
    'rsi_period': 14,
    'rsi_overbought': 72,
    'rsi_oversold': 28,
    'macd_fast': 10,
    'macd_slow': 22,
    'macd_signal': 7,
    'required_indicators': 3,
    'manual_stake_amount': 100.0,
    'use_candlestick_patterns': True,
    'min_pattern_confidence': 0.7
}
# Inicialização do estado
for key, value in DEFAULT_PARAMS.items():
    st.session_state.setdefault('trading_params', {}).setdefault(key, value)

st.session_state.setdefault('bank_balance', DEFAULT_PARAMS['initial_bank'])
st.session_state.setdefault('open_trades', [])
st.session_state.setdefault('trade_history', [])
st.session_state.setdefault('current_prices', {pair: data['base_price'] for pair, data in FOREX_PAIRS.items()})
st.session_state.setdefault('trade_counter', 0)
st.session_state.setdefault('auto_trading', False)

trading_pairs = list(FOREX_PAIRS.keys())

# Indicadores técnicos
def calculate_rsi(prices, period=14):
    delta = prices.diff()
    gain = delta.where(delta > 0, 0).rolling(window=period).mean()
    loss = -delta.where(delta < 0, 0).rolling(window=period).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))

def calculate_macd(prices, fast=12, slow=26, signal=9):
    exp1 = prices.ewm(span=fast).mean()
    exp2 = prices.ewm(span=slow).mean()
    macd_line = exp1 - exp2
    signal_line = macd_line.ewm(span=signal).mean()
    return macd_line, signal_line

def calculate_indicators(df):
    if df.empty or len(df) < 50:
        return df
    params = st.session_state.trading_params
    df['MA_Fast'] = df['close'].rolling(window=params['ma_fast']).mean()
    df['MA_Slow'] = df['close'].rolling(window=params['ma_slow']).mean()
    df['RSI'] = calculate_rsi(df['close'], params['rsi_period'])
    macd_line, signal_line = calculate_macd(df['close'], params['macd_fast'], params['macd_slow'], params['macd_signal'])
    df['MACD'] = macd_line
    df['MACD_Signal'] = signal_line
    return df
def generate_15min_forex_data(pair, periods=200):
    pair_data = FOREX_PAIRS[pair]
    base_price = st.session_state.current_prices.get(pair, pair_data['base_price'])
    volatility = pair_data['volatility']
    prices = []
    current_time = datetime.now()

    for i in range(periods):
        date = current_time - timedelta(minutes=15 * (periods - i - 1))
        change = np.random.normal(0, volatility * 0.1)
        close_price = base_price * (1 + change)
        high_price = max(base_price, close_price) * (1 + abs(np.random.normal(0, volatility * 0.05)))
        low_price = min(base_price, close_price) * (1 - abs(np.random.normal(0, volatility * 0.05)))
        prices.append({
            "date": date,
            "open": base_price,
            "high": high_price,
            "low": low_price,
            "close": close_price
        })
        base_price = close_price

    return pd.DataFrame(prices)

def detect_trading_signals(df):
    buy_indicators, sell_indicators = [], []
    params = st.session_state.trading_params
    latest = df.iloc[-1]

    if latest['MA_Fast'] > latest['MA_Slow']:
        buy_indicators.append("MA Bullish Crossover")
    else:
        sell_indicators.append("MA Bearish Crossover")

    if latest['RSI'] < params['rsi_oversold']:
        buy_indicators.append("RSI Oversold")
    elif latest['RSI'] > params['rsi_overbought']:
        sell_indicators.append("RSI Overbought")

    if latest['MACD'] > latest['MACD_Signal']:
        buy_indicators.append("MACD Bullish Crossover")
    elif latest['MACD'] < latest['MACD_Signal']:
        sell_indicators.append("MACD Bearish Crossover")

    if len(buy_indicators) >= params['required_indicators']:
        return buy_indicators, sell_indicators, 'BUY'
    elif len(sell_indicators) >= params['required_indicators']:
        return buy_indicators, sell_indicators, 'SELL'
    else:
        return buy_indicators, sell_indicators, 'HOLD'
st.markdown("## 🎯 Execução Manual de Trade")

col1, col2 = st.columns([2, 1])
with col1:
    selected_pair = st.selectbox("📌 Selecione o par para análise", trading_pairs)
with col2:
    stake = st.number_input("💰 Valor do stake manual (€)", min_value=10.0, max_value=10000.0,
                            value=st.session_state.trading_params['manual_stake_amount'], step=10.0)

# Gerar dados e calcular indicadores
df_pair = generate_15min_forex_data(selected_pair)
df_ind = calculate_indicators(df_pair)
buy_signals, sell_signals, signal = detect_trading_signals(df_ind)

# Exibir sinais
st.markdown(f"### 📡 Sinal atual para `{selected_pair}`: **:blue[{signal}]**")
st.write("📈 Indicadores de Compra:", buy_signals)
st.write("📉 Indicadores de Venda:", sell_signals)

# Botão de execução manual
if st.button("🚀 Executar Trade Manual"):
    trade = {
        "id": st.session_state.trade_counter + 1,
        "pair": selected_pair,
        "stake": stake,
        "signal": signal,
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "auto": False
    }
    st.session_state.trade_counter += 1
    st.session_state.open_trades.append(trade)
    st.success(f"Trade manual executado para {selected_pair} com stake de €{stake:.2f} ({signal})")

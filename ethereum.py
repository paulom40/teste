import streamlit as st
import pandas as pd
import numpy as np
import requests
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Configuração da página
st.set_page_config(page_title="📈 Live Trading RSI + MACD", layout="wide")

# Chave da API e símbolo
API_KEY = "O6DP7BY7OQ10I0G2"
SYMBOL = "ETHUSD"

# Estado persistente
if 'capital' not in st.session_state:
    st.session_state.capital = 1000
    st.session_state.trades = pd.DataFrame()
    st.session_state.position = 0
    st.session_state.entry_price = 0

# Função para obter dados intradiários
@st.cache_data(ttl=60)
def get_intraday_data(symbol):
    url = "https://www.alphavantage.co/query"
    params = {
        "function": "TIME_SERIES_INTRADAY",
        "symbol": symbol,
        "interval": "1min",
        "apikey": API_KEY,
        "outputsize": "compact"
    }
    response = requests.get(url, params=params)
    data = response.json()
    if "Time Series (1min)" in data:
        df = pd.DataFrame.from_dict(data["Time Series (1min)"], orient="index")
        df = df.rename(columns={"4. close": "price"})
        df.index = pd.to_datetime(df.index)
        df = df.sort_index()
        df = df[['price']].astype(float)
        return df
    else:
        return pd.DataFrame()

# Indicadores
def calculate_rsi(prices, period=14):
    delta = prices.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.rolling(window=period).mean()
    avg_loss = loss.rolling(window=period).mean()
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    return rsi.fillna(50)

def calculate_macd(prices, fast=12, slow=26, signal=9):
    ema_fast = prices.ewm(span=fast).mean()
    ema_slow = prices.ewm(span=slow).mean()
    macd_line = ema_fast - ema_slow
    signal_line = macd_line.ewm(span=signal).mean()
    return macd_line, signal_line

# Simulação de trading
def simulate_trading(df):
    df['rsi'] = calculate_rsi(df['price'])
    df['macd'], df['macd_signal'] = calculate_macd(df['price'])
    df['signal'] = 'HOLD'
    for i in range(1, len(df)):
        rsi = df['rsi'].iloc[i]
        macd = df['macd'].iloc[i]
        signal = df['macd_signal'].iloc[i]
        price = df['price'].iloc[i]
        prev_macd = df['macd'].iloc[i-1]
        prev_signal = df['macd_signal'].iloc[i-1]
        bullish = macd > signal and prev_macd <= prev_signal
        bearish = macd < signal and prev_macd >= prev_signal
        if st.session_state.position == 0 and rsi < 30 and bullish:
            st.session_state.position = 1
            st.session_state.entry_price = price
            st.session_state.capital -= 10
            df.at[df.index[i], 'signal'] = 'BUY'
        elif st.session_state.position == 1:
            take_profit = st.session_state.entry_price * 1.05
            stop_loss = st.session_state.entry_price * 0.95
            if rsi > 70 or bearish or price >= take_profit or price <= stop_loss:
                pnl = price - st.session_state.entry_price
                st.session_state.capital += 10 + pnl
                df.at[df.index[i], 'signal'] = 'SELL'
                st.session_state.trades = pd.concat([
                    st.session_state.trades,
                    pd.DataFrame({
                        'time': [df.index[i]],
                        'signal': ['SELL'],
                        'price': [price],
                        'pnl': [pnl]
                    })
                ])
                st.session_state.position = 0
                st.session_state.entry_price = 0
    return df

# Interface
st.title("📊 Live Trading RSI + MACD")
df = get_intraday_data(SYMBOL)

if not df.empty:
    df = simulate_trading(df)
    st.metric("💰 Preço Atual", f"${df['price'].iloc[-1]:.2f}")
    st.metric("💳 Banca Atual", f"€{st.session_state.capital:.2f}")

    # Gráfico interativo
    fig = make_subplots(rows=3, cols=1, shared_xaxes=True, subplot_titles=["Preço", "RSI", "MACD"])
    fig.add_trace(go.Scatter(x=df.index, y=df['price'], name='Preço'), row=1, col=1)
    fig.add_trace(go.Scatter(x=df.index, y=df['rsi'], name='RSI'), row=2, col=1)
    fig.add_hline(y=70, line_dash="dash", line_color="red", row=2, col=1)
    fig.add_hline(y=30, line_dash="dash", line_color="green", row=2, col=1)
    fig.add_trace(go.Scatter(x=df.index, y=df['macd'], name='MACD'), row=3, col=1)
    fig.add_trace(go.Scatter(x=df.index, y=df['macd_signal'], name='Sinal'), row=3, col=1)
    fig.update_layout(height=800, template="plotly_dark")
    st.plotly_chart(fig, use_container_width=True)

    # Tabela de trades
    if not st.session_state.trades.empty:
        st.subheader("💼 Trades Executados")
        st.dataframe(st.session_state.trades.round(2))
else:
    st.warning("Sem dados disponíveis. Verifica o símbolo ou a chave da API.")

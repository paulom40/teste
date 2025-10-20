import streamlit as st
import pandas as pd
import numpy as np
import requests
import datetime

st.set_page_config(page_title="Finnhub Forex", layout="centered")
st.title("💱 Simulador com RSI + MACD (Finnhub)")

# Chave da API
API_KEY = "d3r3tbpr01qopgh6rrtgd3r3tbpr01qopgh6rru0"

# Pares suportados pela Finnhub
pares = ["EUR/USD", "USD/BRL", "GBP/USD", "USD/JPY", "AUD/USD"]
par = st.selectbox("Seleciona o par de moedas", pares)

# Extrair símbolo Finnhub
symbol = par.replace("/", "")

# Função para obter dados históricos
@st.cache_data(ttl=600)
def get_finnhub_data(symbol):
    end = int(datetime.datetime.now().timestamp())
    start = end - 60 * 60 * 24 * 30  # últimos 30 dias
    url = f"https://finnhub.io/api/v1/forex/candle?symbol=OANDA:{symbol}&resolution=D&from={start}&to={end}&token={API_KEY}"
    r = requests.get(url)
    if r.status_code == 200:
        data = r.json()
        if data.get("s") == "ok":
            df = pd.DataFrame({
                "timestamp": pd.to_datetime(data["t"], unit="s"),
                "price": data["c"]
            })
            df.set_index("timestamp", inplace=True)
            return df
        else:
            st.error("❌ Dados inválidos ou vazios retornados pela Finnhub.")
            return pd.DataFrame()
    else:
        st.error(f"❌ Erro na API Finnhub: {r.status_code}")
        return pd.DataFrame()

# Indicadores técnicos
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

# Executar
df = get_finnhub_data(symbol)
if not df.empty:
    df["rsi"] = calculate_rsi(df["price"])
    df["macd"], df["macd_signal"] = calculate_macd(df["price"])
    st.line_chart(df[["price", "rsi", "macd", "macd_signal"]])
    st.success("✅ Dados carregados com sucesso.")
else:
    st.warning("Nenhum dado disponível para este par.")

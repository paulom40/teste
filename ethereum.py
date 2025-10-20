import streamlit as st
import pandas as pd
import numpy as np
import requests
import io

# 🔒 Inicialização segura
if 'capital' not in st.session_state:
    st.session_state.capital = 1000
if 'trades' not in st.session_state:
    st.session_state.trades = pd.DataFrame()
if 'position' not in st.session_state:
    st.session_state.position = 0
if 'entry_price' not in st.session_state:
    st.session_state.entry_price = 0

# 📊 Interface principal
st.set_page_config(page_title="Live Forex Trading", layout="wide")
st.title("💱 Simulador de Trading com RSI + MACD (Alpha Vantage)")

# 🔧 Configurações
API_KEY = "O6DP7BY7OQ10I0G2"
INTERVAL = "5min"
pair = st.sidebar.selectbox("Seleciona o par de moedas", ["USD/JPY", "EUR/USD", "GBP/JPY", "AUD/CAD"])
from_symbol, to_symbol = pair.split("/")

# 📈 Obter dados da Alpha Vantage
@st.cache_data(ttl=300)
def get_forex_data(from_symbol, to_symbol, interval):
    url = "https://www.alphavantage.co/query"
    params = {
        "function": "FX_INTRADAY",
        "from_symbol": from_symbol,
        "to_symbol": to_symbol,
        "interval": interval,
        "apikey": API_KEY,
        "outputsize": "compact"
    }
    response = requests.get(url, params=params)
    data = response.json()
    key = f"Time Series FX ({interval})"
    if key in data:
        df = pd.DataFrame.from_dict(data[key], orient="index")
        df.index = pd.to_datetime(df.index)
        df = df.sort_index()
        df["price"] = df["4. close"].astype(float)
        return df[["price"]]
    else:
        return pd.DataFrame()

# 📊 Indicadores técnicos
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

# 🧠 Simulação de trading
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
            take_profit = st.session_state.entry_price * 1.005
            stop_loss = st.session_state.entry_price * 0.995
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

# 🚀 Executar simulação
df = get_forex_data(from_symbol, to_symbol, INTERVAL)
if df.empty:
    st.error("❌ Erro na API. Verifica o par selecionado ou o limite de chamadas.")
else:
    df = simulate_trading(df)
    st.metric("💰 Preço Atual", f"{df['price'].iloc[-1]:.4f}")
    st.metric("💳 Banca Atual", f"€{st.session_state.capital:.2f}")
    st.line_chart(df['price'])

# 📤 Exportação para Excel com gráficos e métricas
if not st.session_state.trades.empty:
    trades = st.session_state.trades.copy()
    trades['Período'] = trades['time'].apply(lambda dt: 'Manhã' if dt.hour < 12 else 'Tarde' if dt.hour < 18 else 'Noite')
    trades['capital'] = 1000 + trades['pnl'].cumsum()

    total_trades = len(trades)
    total_pnl = trades['pnl'].sum()
    win_rate = len(trades[trades['pnl'] > 0]) / total_trades * 100 if total_trades > 0 else 0
    avg_pnl = trades['pnl'].mean()
    std_pnl = trades['pnl'].std()
    sharpe_ratio = avg_pnl / std_pnl if std_pnl != 0 else 0
    drawdown = (trades['capital'].cummax() - trades['capital']).max()

    metrics_df = pd.DataFrame({
        'Métrica': [
            'Capital Final', 'Lucro Líquido', 'Total de Trades', 'Win Rate (%)',
            'Lucro Médio por Trade', 'Desvio Padrão do PnL', 'Índice de Sharpe', 'Drawdown Máximo'
        ],
        'Valor': [
            st.session_state.capital, total_pnl, total_trades, round(win_rate, 2),
            round(avg_pnl, 2), round(std_pnl, 2), round(sharpe_ratio, 2), round(drawdown, 2)
        ]
    })

    df_chart = df[['rsi', 'macd', 'macd_signal']].copy()
    df_chart.reset_index(inplace=True)

    output = io.BytesIO()
    with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
        trades.to_excel(writer, index=False, sheet_name='Trades')
        metrics_df.to_excel(writer, index=False, sheet_name='Métricas')
        df_chart.to_excel(writer, index=False, sheet_name='Gráfico Técnico')

        pnl_counts = trades['pnl'].apply(lambda x: 'Lucro' if x > 0 else 'Prejuízo').value_counts().reset_index()
        pnl_counts.columns = ['Resultado', 'Quantidade']
        pnl_counts.to_excel(writer, index=False, sheet_name='Distribuição PnL')

        signal_counts = trades['signal'].value_counts().reset_index()
        signal_counts.columns = ['Sinal', 'Quantidade']
        signal_counts.to_excel(writer, index=False, sheet_name='Sinais')

        period_counts = trades['Período'].value_counts().reset_index()
        period_counts.columns = ['Período', 'Quantidade']
        period_counts.to_excel(writer, index=False, sheet_name='Segmentação Horária')

        writer.save()

    st.download_button(
        label="📥 Exportar para Excel com Segmentação e Gráficos",
        data=output.getvalue(),
        file_name="trades_forex_segmentado.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
    )

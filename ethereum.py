import streamlit as st
import pandas as pd
import numpy as np
import requests
import io

# Inicialização segura
if 'capital' not in st.session_state:
    st.session_state.capital = 1000
if 'trades' not in st.session_state:
    st.session_state.trades = pd.DataFrame()
if 'position' not in st.session_state:
    st.session_state.position = 0
if 'entry_price' not in st.session_state:
    st.session_state.entry_price = 0

# Interface principal
st.set_page_config(page_title="Live Forex Trading", layout="wide")
st.title("💱 Simulador de Trading com RSI + MACD (AwesomeAPI)")

# Pares confirmados pela AwesomeAPI
pares_suportados = [
    "USD-BRL", "EUR-BRL", "GBP-BRL", "JPY-BRL", "CAD-BRL",
    "AUD-BRL", "CHF-BRL", "BTC-BRL", "ETH-BRL", "LTC-BRL",
    "ARS-BRL", "CNY-BRL", "ILS-BRL"
]
pair = st.sidebar.selectbox("Seleciona o par de moedas", pares_suportados)

# Obter dados da AwesomeAPI
@st.cache_data(ttl=300)
def get_forex_data(pair):
    url = f"https://economia.awesomeapi.com.br/json/daily/{pair}/30"
    response = requests.get(url)
    if response.status_code == 200:
        try:
            data = response.json()
            if isinstance(data, list) and len(data) > 0 and 'bid' in data[0]:
                df = pd.DataFrame(data)
                df['timestamp'] = pd.to_datetime(df['timestamp'], unit='s')
                df = df.sort_values('timestamp')
                df['price'] = df['bid'].astype(float)
                df.set_index('timestamp', inplace=True)
                return df[['price']]
            else:
                st.error("❌ Dados inválidos ou vazios retornados pela API.")
                return pd.DataFrame()
        except Exception as e:
            st.error(f"❌ Erro ao processar os dados: {e}")
            return pd.DataFrame()
    else:
        st.error("❌ Erro na API AwesomeAPI. Verifica o par selecionado.")
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

# Executar simulação
df = get_forex_data(pair)
if not df.empty:
    df = simulate_trading(df)
    st.metric("💰 Preço Atual", f"{df['price'].iloc[-1]:.4f}")
    st.metric("💳 Banca Atual", f"€{st.session_state.capital:.2f}")
    st.line_chart(df['price'])

# Filtros de análise
if not st.session_state.trades.empty:
    st.sidebar.markdown("### 🎯 Filtros de análise")
    min_date = st.session_state.trades['time'].min().date()
    max_date = st.session_state.trades['time'].max().date()
    start_date = st.sidebar.date_input("Data inicial", min_value=min_date, max_value=max_date, value=min_date)
    end_date = st.sidebar.date_input("Data final", min_value=min_date, max_value=max_date, value=max_date)
    signal_filter = st.sidebar.multiselect("Tipo de sinal", ["BUY", "SELL"], default=["BUY", "SELL"])

    filtered_trades = st.session_state.trades[
        (st.session_state.trades['time'].dt.date >= start_date) &
        (st.session_state.trades['time'].dt.date <= end_date) &
        (st.session_state.trades['signal'].isin(signal_filter))
    ]

    trades = filtered_trades.copy()
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

    st.download_button(
    label="📥 Exportar para Excel com Segmentação e Gráficos",
    data=output.getvalue(),
    file_name="trades_forex_awesomeapi.xlsx",
    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
)


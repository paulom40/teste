import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import requests
import io

st.set_page_config(page_title="Forex Pro Bot", layout="wide")

FOREX_PAIRS = {
    "EUR/USD": {"base_price": 1.0850, "volatility": 0.0008, "pip_value": 0.0001},
    "GBP/USD": {"base_price": 1.2650, "volatility": 0.0010, "pip_value": 0.0001},
    "USD/JPY": {"base_price": 148.50, "volatility": 0.15, "pip_value": 0.01},
    "AUD/USD": {"base_price": 0.6550, "volatility": 0.0012, "pip_value": 0.0001}
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
    'manual_stake_amount': 100.0
}
for key, value in DEFAULT_PARAMS.items():
    st.session_state.setdefault('trading_params', {}).setdefault(key, value)

st.session_state.setdefault('bank_balance', DEFAULT_PARAMS['initial_bank'])
st.session_state.setdefault('open_trades', [])
st.session_state.setdefault('trade_history', [])
st.session_state.setdefault('trade_counter', 0)
st.session_state.setdefault('auto_trading', False)

trading_pairs = list(FOREX_PAIRS.keys())

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
    params = st.session_state.trading_params
    df['MA_Fast'] = df['close'].rolling(window=params['ma_fast']).mean()
    df['MA_Slow'] = df['close'].rolling(window=params['ma_slow']).mean()
    df['RSI'] = calculate_rsi(df['close'], params['rsi_period'])
    macd_line, signal_line = calculate_macd(df['close'], params['macd_fast'], params['macd_slow'], params['macd_signal'])
    df['MACD'] = macd_line
    df['MACD_Signal'] = signal_line
    return df
def detectar_padrao_price_action(df):
    if len(df) < 3:
        return None
    ultima = df.iloc[-1]
    anterior = df.iloc[-2]

    if ultima["close"] > ultima["open"] and anterior["close"] < anterior["open"]:
        if ultima["open"] < anterior["close"] and ultima["close"] > anterior["open"]:
            return "Bullish Engulfing"
    elif ultima["close"] < ultima["open"] and anterior["close"] > anterior["open"]:
        if ultima["open"] > anterior["close"] and ultima["close"] < anterior["open"]:
            return "Bearish Engulfing"

    corpo = abs(ultima["close"] - ultima["open"])
    sombra_sup = ultima["high"] - max(ultima["close"], ultima["open"])
    sombra_inf = min(ultima["close"], ultima["open"]) - ultima["low"]
    if corpo < sombra_sup * 0.3 or corpo < sombra_inf * 0.3:
        return "Pin Bar"

    if ultima["high"] < anterior["high"] and ultima["low"] > anterior["low"]:
        return "Inside Bar"

    return None

def detectar_suporte_resistencia(df, margem=0.002):
    zonas = []
    for i in range(10, len(df)):
        janela = df.iloc[i-10:i]
        zonas.append((janela["low"].min(), janela["high"].max()))
    return zonas

def esta_proximo_de_zona(preco, zonas, margem=0.002):
    for suporte, resistencia in zonas:
        if abs(preco - suporte) / preco < margem or abs(preco - resistencia) / preco < margem:
            return True
    return False

def detectar_breakout(df, zonas):
    preco = df.iloc[-1]["close"]
    suporte, resistencia = zonas[-1] if zonas else (None, None)
    if resistencia and preco > resistencia:
        return "Breakout Acima"
    elif suporte and preco < suporte:
        return "Breakout Abaixo"
    return None
def confirmar_padrao_por_candles(df, padrao, n=2):
    contagem = 0
    for i in range(-n, 0):
        sub_padrao = detectar_padrao_price_action(df.iloc[:i+1])
        if sub_padrao == padrao:
            contagem += 1
    return contagem >= n

def simular_volume_impulso(df):
    ultima = df.iloc[-1]
    amplitude = ultima["high"] - ultima["low"]
    media = df["high"].sub(df["low"]).rolling(window=10).mean().iloc[-1]
    return amplitude > media * 1.2

def generate_15min_forex_data(pair, periods=200):
    base = FOREX_PAIRS[pair]["base_price"]
    vol = FOREX_PAIRS[pair]["volatility"]
    data = []
    now = datetime.now()
    for i in range(periods):
        t = now - timedelta(minutes=15 * (periods - i - 1))
        change = np.random.normal(0, vol * 0.1)
        close = base * (1 + change)
        high = max(base, close) * (1 + abs(np.random.normal(0, vol * 0.05)))
        low = min(base, close) * (1 - abs(np.random.normal(0, vol * 0.05)))
        data.append({"timestamp": t, "open": base, "high": high, "low": low, "close": close})
        base = close
    return pd.DataFrame(data)

def obter_dados_polygon(pair, api_key, fallback=True):
    origem = "API"
    try:
        symbol = pair.replace("/", "")
        url = f"https://api.polygon.io/v2/aggs/ticker/C:{symbol}/range/15/minute/1/day?adjusted=true&sort=asc&apiKey={api_key}"
        response = requests.get(url)
        response.raise_for_status()
        dados = response.json()["results"]
        df = pd.DataFrame([{
            "timestamp": datetime.fromtimestamp(item["t"] / 1000),
            "open": item["o"],
            "high": item["h"],
            "low": item["l"],
            "close": item["c"]
        } for item in dados])
    except Exception:
        origem = "Simulação"
        st.warning(f"⚠️ API da Polygon falhou para {pair}. Usando dados simulados.")
        df = generate_15min_forex_data(pair)
    return df, origem
def detect_trading_signals(df):
    params = st.session_state.trading_params
    latest = df.iloc[-1]
    buy, sell = [], []
    if latest['MA_Fast'] > latest['MA_Slow']: buy.append("MA Bullish")
    else: sell.append("MA Bearish")
    if latest['RSI'] < params['rsi_oversold']: buy.append("RSI Oversold")
    elif latest['RSI'] > params['rsi_overbought']: sell.append("RSI Overbought")
    if latest['MACD'] > latest['MACD_Signal']: buy.append("MACD Bullish")
    elif latest['MACD'] < latest['MACD_Signal']: sell.append("MACD Bearish")
    if len(buy) >= params['required_indicators']: return buy, sell, 'BUY'
    elif len(sell) >= params['required_indicators']: return buy, sell, 'SELL'
    return buy, sell, 'HOLD'

COMBINACOES_CRITICAS = [
    {"indicadores": {"RSI Oversold", "MACD Bullish"}, "padrao": "Pin Bar", "tipo": "BUY"},
    {"indicadores": {"RSI Overbought", "MACD Bearish"}, "padrao": "Bearish Engulfing", "tipo": "SELL"},
    {"indicadores": {"MA Bullish", "MACD Bullish"}, "padrao": "Inside Bar", "tipo": "BUY"},
]

def verificar_alerta_combinado(indicadores, padrao, tipo):
    ativos = set(indicadores)
    for regra in COMBINACOES_CRITICAS:
        if regra["tipo"] == tipo and regra["padrao"] == padrao:
            if regra["indicadores"].issubset(ativos):
                return True
    return False
def registrar_log(pair, signal, padrao, breakout, confirmado, impulso, resultado, origem_dados, indicadores_ativos, alerta_critico=False):
    log = {
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "pair": pair,
        "signal": signal,
        "padrao": padrao,
        "breakout": breakout,
        "confirmado": confirmado,
        "impulso": impulso,
        "resultado (€)": round(resultado, 2),
        "origem_dados": origem_dados,
        "indicadores_ativos": ", ".join(indicadores_ativos),
        "alerta_critico": "Sim" if alerta_critico else "Não"
    }
    if "logs_tecnicos" not in st.session_state:
        st.session_state.logs_tecnicos = []
    st.session_state.logs_tecnicos.append(log)

st.markdown("## 🤖 Trader Automático com Price Action")

if st.checkbox("🔁 Ativar trade automático", value=st.session_state.auto_trading):
    st.session_state.auto_trading = True

    for pair in trading_pairs:
        df, origem_dados = obter_dados_polygon(pair, api_key="ZACYNJQZmDFZV0B92pErxGfiF60iUuZ_")
        df_ind = calculate_indicators(df)
        buy, sell, signal = detect_trading_signals(df_ind)
        indicadores_ativos = buy if signal == "BUY" else sell if signal == "SELL" else []

        padrao = detectar_padrao_price_action(df)
        zonas = detectar_suporte_resistencia(df)
        preco_atual = df.iloc[-1]["close"]
        breakout = detectar_breakout(df, zonas)
        confirmado = confirmar_padrao_por_candles(df, padrao, n=2)
        impulso = simular_volume_impulso(df)
        alerta_critico = verificar_alerta_combinado(indicadores_ativos, padrao, signal)

        condicoes_ideais = (
            signal in ["BUY", "SELL"] and
            padrao in ["Bullish Engulfing", "Bearish Engulfing", "Pin Bar", "Inside Bar"] and
            esta_proximo_de_zona(preco_atual, zonas) and
            confirmado
        ) or (breakout in ["Breakout Acima", "Breakout Abaixo"] and impulso)

        if condicoes_ideais:
            stake = st.session_state.trading_params['manual_stake_amount']
            pip_value = FOREX_PAIRS[pair]["pip_value"]
            movimento = np.random.choice(["profit", "loss"], p=[0.55, 0.45])
            pips = DEFAULT_PARAMS["profit_target_pips"] if movimento == "profit" else DEFAULT_PARAMS["stop_loss_pips"]
            resultado = stake * (pips * pip_value) * (1 if movimento == "profit" else -1)
            status = "Lucro" if resultado > 0 else "Prejuízo"

            trade = {
                "id": st.session_state.trade_counter + 1,
                "pair": pair,
                "stake": stake,
                "signal": signal,
                "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "auto": True
            }
            st.session_state.trade_counter += 1
            st.session_state.open_trades.append(trade)

            resultado_trade = {
                "pair": pair,
                "direção": signal,
                "stake": stake,
                "resultado (€)": round(resultado, 2),
                "status": status,
                "timestamp": trade["timestamp"]
            }

            st.session_state.trade_history.append(resultado_trade)
            st.session_state.bank_balance += resultado
            registrar_log(pair, signal, padrao, breakout, confirmado, impulso, resultado, origem_dados, indicadores_ativos, alerta_critico)

            if alerta_critico:
                st.error(f"🚨 Alerta crítico: {signal} com {padrao} e {', '.join(indicadores_ativos)}")
            else:
                st.write(f"{pair} → {signal} → {padrao or breakout} → {status} (€{resultado:.2f})")
st.markdown("## 📊 Painel de Performance")
df_resultados = pd.DataFrame(st.session_state.trade_history)
if not df_resultados.empty:
    df_resultados["saldo"] = DEFAULT_PARAMS["initial_bank"] + df_resultados["resultado (€)"].cumsum()
    lucro_total = df_resultados["resultado (€)"].sum()
    taxa_acerto = (df_resultados["resultado (€)"] > 0).mean() * 100
    drawdown = (df_resultados["saldo"].cummax() - df_resultados["saldo"]).max()

    st.write(f"Lucro Líquido: €{lucro_total:.2f}")
    st.write(f"Taxa de Acerto: {taxa_acerto:.1f}%")
    st.write(f"Drawdown Máximo: €{drawdown:.2f}")
    st.line_chart(df_resultados.set_index("timestamp")["saldo"])

    st.subheader("🚨 Alertas por Par")
    META_LUCRO = 150.0
    LIMITE_DRAW = 100.0
    for pair in df_resultados["pair"].unique():
        df_par = df_resultados[df_resultados["pair"] == pair].copy()
        df_par["saldo"] = DEFAULT_PARAMS["initial_bank"] + df_par["resultado (€)"].cumsum()
        lucro = df_par["resultado (€)"].sum()
        draw = (df_par["saldo"].cummax() - df_par["saldo"]).max()
        if lucro >= META_LUCRO:
            st.success(f"{pair}: Meta de lucro atingida (€{lucro:.2f})")
        elif draw >= LIMITE_DRAW:
            st.warning(f"{pair}: Drawdown elevado (€{draw:.2f})")
        else:
            st.info(f"{pair}: Dentro dos parâmetros operacionais")
st.subheader("🧾 Logs Técnicos")
if "logs_tecnicos" in st.session_state and st.session_state.logs_tecnicos:
    df_logs = pd.DataFrame(st.session_state.logs_tecnicos)
    par_log = st.selectbox("Filtrar por par", ["Todos"] + df_logs["pair"].unique().tolist())
    if par_log != "Todos":
        df_logs = df_logs[df_logs["pair"] == par_log]
    st.dataframe(df_logs, use_container_width=True)

    output_logs = io.BytesIO()
    with pd.ExcelWriter(output_logs, engine="xlsxwriter") as writer:
        df_logs.to_excel(writer, index=False, sheet_name="Logs Técnicos")
    st.download_button("📥 Baixar Logs Técnicos", output_logs.getvalue(), "logs_tecnicos_trader.xlsx", "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")

st.subheader("📤 Exportar por Par com Métricas e Gráfico")
output = io.BytesIO()
with pd.ExcelWriter(output, engine="xlsxwriter") as writer:
    df_logs.to_excel(writer, index=False, sheet_name="Logs Técnicos")


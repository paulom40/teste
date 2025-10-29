import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
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
        data.append({"date": t, "open": base, "high": high, "low": low, "close": close})
        base = close
    return pd.DataFrame(data)

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
st.markdown("## 🤖 Trader Automático com Price Action")

if st.toggle("🔁 Ativar trade automático", value=st.session_state.auto_trading):
    st.session_state.auto_trading = True

    for pair in trading_pairs:
        df = generate_15min_forex_data(pair)
        df_ind = calculate_indicators(df)
        buy, sell, signal = detect_trading_signals(df_ind)

        padrao = detectar_padrao_price_action(df)
        zonas = detectar_suporte_resistencia(df)
        preco_atual = df.iloc[-1]["close"]
        breakout = detectar_breakout(df, zonas)
        confirmado = confirmar_padrao_por_candles(df, padrao, n=2)
        impulso = simular_volume_impulso(df)

        condicoes_ideais = (
            signal in ["BUY", "SELL"] and
            padrao in ["Bullish Engulfing", "Bearish Engulfing", "Pin Bar", "Inside Bar"] and
            esta_proximo_de_zona(preco_atual, zonas) and
            confirmado
        ) or (breakout in ["Breakout Acima", "Breakout Abaixo"] and impulso)

        if condicoes_ideais:
            trade = {
                "id": st.session_state.trade_counter + 1,
                "pair": pair,
                "stake": st.session_state.trading_params['manual_stake_amount'],
                "signal": signal,
                "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "auto": True
            }
            st.session_state.trade_counter += 1
            st.session_state.open_trades.append(trade)
            pip_value = FOREX_PAIRS[pair]["pip_value"]
            movimento = np.random.choice(["profit", "loss"], p=[0.55, 0.45])

            if movimento == "profit":
                pips = DEFAULT_PARAMS["profit_target_pips"]
                resultado = trade["stake"] * (pips * pip_value)
                status = "✅ Lucro"
            else:
                pips = DEFAULT_PARAMS["stop_loss_pips"]
                resultado = -trade["stake"] * (pips * pip_value)
                status = "❌ Prejuízo"

            resultado_trade = {
                "pair": pair,
                "direção": signal,
                "stake": trade["stake"],
                "resultado (€)": round(resultado, 2),
                "status": status,
                "timestamp": trade["timestamp"]
            }

            st.session_state.trade_history.append(resultado_trade)
            st.session_state.bank_balance += resultado
            st.success(f"{pair} → {signal} → {padrao or breakout} → {status} (€{resultado:.2f})")

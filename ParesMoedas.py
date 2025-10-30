import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timedelta
import io
import time
import talib

# Configuração da página
st.set_page_config(
    page_title="Advanced Trading Dashboard", 
    layout="wide",
    page_icon="📈"
)

# CSS personalizado para melhorar a aparência
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 10px;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        border-left: 4px solid #1f77b4;
    }
    .alert-high {
        background-color: #ffcccc;
        padding: 0.5rem;
        border-radius: 5px;
        border-left: 4px solid #ff4444;
    }
    .stMetric {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 10px;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        border-left: 4px solid #1f77b4;
    }
    .live-indicator {
        background-color: #ff4444;
        color: white;
        padding: 2px 8px;
        border-radius: 12px;
        font-size: 0.8rem;
        animation: blink 1s infinite;
    }
    .pattern-bullish {
        background-color: #90EE90 !important;
        color: black;
    }
    .pattern-bearish {
        background-color: #FFB6C1 !important;
        color: black;
    }
    .pattern-neutral {
        background-color: #F0F0F0 !important;
        color: black;
    }
    @keyframes blink {
        0% { opacity: 1; }
        50% { opacity: 0.5; }
        100% { opacity: 1; }
    }
</style>
""", unsafe_allow_html=True)

# Limites de volatilidade por par
LIMITE_VOLATILIDADE = {
    "EUR/USD": 0.0020,
    "GBP/USD": 0.0025,
    "USD/JPY": 0.25,
    "AUD/USD": 0.0022,
    "USD/CAD": 0.0023,
    "USD/CHF": 0.0021
}

# Parâmetros simulados expandidos
PARES = {
    "EUR/USD": {"base": 1.10, "vol": 0.005},
    "GBP/USD": {"base": 1.26, "vol": 0.006},
    "USD/JPY": {"base": 148.0, "vol": 0.004},
    "AUD/USD": {"base": 0.66, "vol": 0.007},
    "USD/CAD": {"base": 1.35, "vol": 0.005},
    "USD/CHF": {"base": 0.88, "vol": 0.004}
}

# Estratégia para candles de 15 minutos
STRATEGY_15MIN = {
    "timeframe": "15min",
    "indicators": ["EMA_9", "EMA_21", "RSI", "MACD"],
    "entry_conditions": {
        "trend": "EMA_9 > EMA_21",
        "oversold": "RSI < 30",
        "overbought": "RSI > 70",
        "momentum": "MACD > 0"
    },
    "exit_conditions": {
        "take_profit": 0.0015,  # 15 pips
        "stop_loss": 0.0010,    # 10 pips
        "trailing_stop": True
    }
}

# Session state para dados em tempo real
if "dados_live" not in st.session_state:
    st.session_state.dados_live = {}
if "ultima_atualizacao" not in st.session_state:
    st.session_state.ultima_atualizacao = datetime.now()
if "contador_updates" not in st.session_state:
    st.session_state.contador_updates = 0

def detectar_padroes_candles(df):
    """Detecta padrões de candles usando lógica de price action"""
    patterns = []
    
    for i in range(2, len(df)):
        current = df.iloc[i]
        prev1 = df.iloc[i-1]
        prev2 = df.iloc[i-2]
        
        # Hammer (Martelo)
        body = abs(current['close'] - current['open'])
        lower_wick = current['low'] - min(current['open'], current['close'])
        upper_wick = max(current['open'], current['close']) - current['high']
        total_range = current['high'] - current['low']
        
        if (lower_wick >= 2 * body and 
            upper_wick <= body * 0.1 and 
            total_range > 0):
            patterns.append({"timestamp": current['timestamp'], "pattern": "Hammer", "type": "bullish", "strength": "medium"})
        
        # Shooting Star (Estrela Cadente)
        if (upper_wick >= 2 * body and 
            lower_wick <= body * 0.1 and 
            total_range > 0):
            patterns.append({"timestamp": current['timestamp'], "pattern": "Shooting Star", "type": "bearish", "strength": "medium"})
        
        # Engulfing Bullish
        if (prev1['close'] < prev1['open'] and  # Candle anterior de baixa
            current['close'] > current['open'] and  # Candle atual de alta
            current['open'] < prev1['close'] and 
            current['close'] > prev1['open']):
            patterns.append({"timestamp": current['timestamp'], "pattern": "Bullish Engulfing", "type": "bullish", "strength": "strong"})
        
        # Engulfing Bearish
        if (prev1['close'] > prev1['open'] and  # Candle anterior de alta
            current['close'] < current['open'] and  # Candle atual de baixa
            current['open'] > prev1['close'] and 
            current['close'] < prev1['open']):
            patterns.append({"timestamp": current['timestamp'], "pattern": "Bearish Engulfing", "type": "bearish", "strength": "strong"})
        
        # Doji
        if body <= total_range * 0.1 and total_range > 0:
            patterns.append({"timestamp": current['timestamp'], "pattern": "Doji", "type": "neutral", "strength": "weak"})
        
        # Morning Star
        if (i >= 2 and
            prev2['close'] < prev2['open'] and  # Primeiro candle de baixa
            abs(prev1['close'] - prev1['open']) <= (prev1['high'] - prev1['low']) * 0.3 and  # Segundo candle pequeno
            current['close'] > current['open'] and  # Terceiro candle de alta
            current['close'] > (prev2['open'] + prev2['close']) / 2):
            patterns.append({"timestamp": current['timestamp'], "pattern": "Morning Star", "type": "bullish", "strength": "strong"})
        
        # Evening Star
        if (i >= 2 and
            prev2['close'] > prev2['open'] and  # Primeiro candle de alta
            abs(prev1['close'] - prev1['open']) <= (prev1['high'] - prev1['low']) * 0.3 and  # Segundo candle pequeno
            current['close'] < current['open'] and  # Terceiro candle de baixa
            current['close'] < (prev2['open'] + prev2['close']) / 2):
            patterns.append({"timestamp": current['timestamp'], "pattern": "Evening Star", "type": "bearish", "strength": "strong"})
    
    return patterns

def calcular_suporte_resistencia(df, window=20):
    """Calcula níveis de suporte e resistência"""
    df = df.copy()
    df['resistance'] = df['high'].rolling(window=window).max()
    df['support'] = df['low'].rolling(window=window).min()
    return df

def analisar_tendencia(df):
    """Analisa a tendência baseada em MME e price action"""
    if len(df) < 20:
        return "Indefinida"
    
    # Tendência por MME
    ema_9 = df['close'].ewm(span=9).mean().iloc[-1]
    ema_21 = df['close'].ewm(span=21).mean().iloc[-1]
    ema_50 = df['close'].ewm(span=50).mean().iloc[-1]
    
    # Price Action - Máximas e Mínimas crescentes/decrescentes
    high_5 = df['high'].tail(5)
    low_5 = df['low'].tail(5)
    
    max_increasing = all(high_5.iloc[i] > high_5.iloc[i-1] for i in range(1, len(high_5)))
    min_increasing = all(low_5.iloc[i] > low_5.iloc[i-1] for i in range(1, len(low_5)))
    max_decreasing = all(high_5.iloc[i] < high_5.iloc[i-1] for i in range(1, len(high_5)))
    min_decreasing = all(low_5.iloc[i] < low_5.iloc[i-1] for i in range(1, len(low_5)))
    
    if (ema_9 > ema_21 > ema_50) and (max_increasing and min_increasing):
        return "Forte Alta"
    elif (ema_9 < ema_21 < ema_50) and (max_decreasing and min_decreasing):
        return "Forte Baixa"
    elif ema_9 > ema_21:
        return "Alta"
    elif ema_9 < ema_21:
        return "Baixa"
    else:
        return "Lateral"

def calcular_volume_profile(df):
    """Calcula perfil de volume para análise de price action"""
    if len(df) == 0:
        return {}
    
    price_levels = np.linspace(df['low'].min(), df['high'].max(), 20)
    volume_at_price = {}
    
    for level in price_levels:
        # Volume próximo a este nível de preço
        mask = (df['low'] <= level) & (df['high'] >= level)
        volume_at_price[round(level, 4)] = df[mask]['volume'].sum()
    
    return volume_at_price

def gerar_dado_live(par, base_price):
    """Gera um novo dado em tempo real para um par"""
    agora = datetime.now()
    
    # Simular preço atual com movimento realista
    variacao = np.random.normal(0, 0.0002)
    spread = np.random.uniform(0.0001, 0.0003)
    
    bid = base_price * (1 + variacao)
    ask = bid + spread
    
    return {
        "timestamp": agora,
        "par": par,
        "bid": round(bid, 5),
        "ask": round(ask, 5),
        "spread": round(spread, 5),
        "volume": np.random.randint(50, 500)
    }

def atualizar_dados_live():
    """Atualiza dados em tempo real para todos os pares selecionados"""
    agora = datetime.now()
    st.session_state.ultima_atualizacao = agora
    st.session_state.contador_updates += 1
    
    for par in st.session_state.get('pares_selecionados', []):
        if par in PARES:
            novo_dado = gerar_dado_live(par, PARES[par]["base"])
            
            if par not in st.session_state.dados_live:
                st.session_state.dados_live[par] = []
            
            # Manter apenas os últimos 100 pontos
            st.session_state.dados_live[par].append(novo_dado)
            if len(st.session_state.dados_live[par]) > 100:
                st.session_state.dados_live[par] = st.session_state.dados_live[par][-100:]

def simular_dados_15min(par, base_price, days=7):
    """Simula dados de 15 minutos para análise"""
    timeframe_data = []
    current_time = datetime.now().replace(hour=0, minute=0, second=0, microsecond=0)
    
    for day in range(days):
        for hour in range(24):
            for minute in range(0, 60, 15):
                timestamp = current_time - timedelta(days=day) + timedelta(hours=hour, minutes=minute)
                
                # Simular preços OHLC
                open_price = base_price * (1 + np.random.normal(0, 0.0005))
                high = open_price * (1 + abs(np.random.normal(0, 0.001)))
                low = open_price * (1 - abs(np.random.normal(0, 0.001)))
                close = (high + low) / 2 + np.random.normal(0, 0.0002)
                
                # Calcular indicadores
                volume = np.random.randint(100, 1000)
                spread = np.random.uniform(0.0001, 0.0003)
                
                timeframe_data.append({
                    "timestamp": timestamp,
                    "par": par,
                    "timeframe": "15min",
                    "open": round(open_price, 5),
                    "high": round(high, 5),
                    "low": round(low, 5),
                    "close": round(close, 5),
                    "volume": volume,
                    "spread": round(spread, 5)
                })
    
    return pd.DataFrame(timeframe_data)

def calcular_indicadores_tecnicos(df):
    """Calcula indicadores técnicos para estratégia de 15min"""
    df = df.sort_values('timestamp')
    
    # EMA
    df['EMA_9'] = df['close'].ewm(span=9).mean()
    df['EMA_21'] = df['close'].ewm(span=21).mean()
    df['EMA_50'] = df['close'].ewm(span=50).mean()
    
    # RSI
    delta = df['close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    df['RSI'] = 100 - (100 / (1 + rs))
    
    # MACD
    exp1 = df['close'].ewm(span=12).mean()
    exp2 = df['close'].ewm(span=26).mean()
    df['MACD'] = exp1 - exp2
    df['MACD_Signal'] = df['MACD'].ewm(span=9).mean()
    df['MACD_Histogram'] = df['MACD'] - df['MACD_Signal']
    
    # Bollinger Bands
    df['BB_Middle'] = df['close'].rolling(window=20).mean()
    bb_std = df['close'].rolling(window=20).std()
    df['BB_Upper'] = df['BB_Middle'] + (bb_std * 2)
    df['BB_Lower'] = df['BB_Middle'] - (bb_std * 2)
    
    # Price Action
    df['Body'] = abs(df['close'] - df['open'])
    df['Total_Range'] = df['high'] - df['low']
    df['Body_Ratio'] = df['Body'] / df['Total_Range']
    df['Upper_Shadow'] = df['high'] - np.maximum(df['open'], df['close'])
    df['Lower_Shadow'] = np.minimum(df['open'], df['close']) - df['low']
    
    # Suporte e Resistência
    df = calcular_suporte_resistencia(df)
    
    return df

def gerar_sinais_15min(df):
    """Gera sinais de trading baseados na estratégia de 15min"""
    df['sinal'] = 'NEUTRO'
    
    # Detectar padrões de candles
    padroes = detectar_padroes_candles(df)
    df['padrao_candle'] = ''
    df['tipo_padrao'] = ''
    
    for padrao in padroes:
        mask = df['timestamp'] == padrao['timestamp']
        df.loc[mask, 'padrao_candle'] = padrao['pattern']
        df.loc[mask, 'tipo_padrao'] = padrao['type']
    
    # Condições de compra
    buy_condition = (
        (df['EMA_9'] > df['EMA_21']) & 
        (df['RSI'] < 35) & 
        (df['MACD'] > df['MACD_Signal']) &
        (df['tipo_padrao'].isin(['bullish', 'strong']))
    )
    
    # Condições de venda
    sell_condition = (
        (df['EMA_9'] < df['EMA_21']) & 
        (df['RSI'] > 65) & 
        (df['MACD'] < df['MACD_Signal']) &
        (df['tipo_padrao'].isin(['bearish', 'strong']))
    )
    
    df.loc[buy_condition, 'sinal'] = 'COMPRA'
    df.loc[sell_condition, 'sinal'] = 'VENDA'
    
    return df

# Função alternativa para style_metric_cards
def custom_metric_style():
    """Aplica estilo personalizado às métricas"""
    st.markdown("""
    <style>
    div[data-testid="metric-container"] {
        background-color: #f0f2f6;
        border: 1px solid #ccc;
        padding: 5% 5% 5% 10%;
        border-radius: 10px;
        border-left: 4px solid #1f77b4;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }
    </style>
    """, unsafe_allow_html=True)

# Simular dados de 15 minutos
st.markdown('<h1 class="main-header">🎯 Advanced Trading Dashboard - Price Action & 15min Strategy</h1>', unsafe_allow_html=True)

# Sidebar modernizada
with st.sidebar:
    st.header("⚙️ Configurações de Trading")
    
    # Indicador de atualização em tempo real
    col_live1, col_live2 = st.columns([2, 1])
    with col_live1:
        st.markdown(f"<span class='live-indicator'>LIVE</span>", unsafe_allow_html=True)
    with col_live2:
        if st.button("🔄 Atualizar"):
            st.rerun()
    
    st.subheader("💰 Gestão de Capital")
    stake_manual = st.radio("Stake:", ["Automático", "€5", "€10", "€25", "€50"])
    meta_lucro = st.number_input("🎯 Meta de Lucro (€)", min_value=10, max_value=5000, value=500, step=50)
    limite_perda = st.number_input("📉 Limite de Perda (€)", min_value=-2000, max_value=0, value=-200, step=50)
    
    st.subheader("📊 Parâmetros da Estratégia")
    rsi_oversold = st.slider("RSI Oversold", 20, 40, 30)
    rsi_overbought = st.slider("RSI Overbought", 60, 80, 70)
    take_profit = st.number_input("Take Profit (pips)", value=15, min_value=5, max_value=50)
    stop_loss = st.number_input("Stop Loss (pips)", value=10, min_value=5, max_value=30)
    
    st.subheader("📈 Filtros")
    pares_selecionados = st.multiselect(
        "Pares para análise:",
        list(PARES.keys()),
        default=list(PARES.keys())[:3]
    )
    
    # Configuração de atualização automática
    st.subheader("🔄 Configuração Live")
    auto_refresh = st.checkbox("Atualização Automática", value=True)
    refresh_interval = st.slider("Intervalo (segundos)", 1, 60, 5)

# Atualizar pares selecionados no session state
st.session_state.pares_selecionados = pares_selecionados

# Inicializar session state
if "logs_tecnicos" not in st.session_state:
    st.session_state.logs_tecnicos = []
if "stake_valor" not in st.session_state:
    st.session_state.stake_valor = 10

# Atualizar stake
if stake_manual == "€5":
    st.session_state.stake_valor = 5
elif stake_manual == "€10":
    st.session_state.stake_valor = 10
elif stake_manual == "€25":
    st.session_state.stake_valor = 25
elif stake_manual == "€50":
    st.session_state.stake_valor = 50

# Atualizar dados em tempo real
atualizar_dados_live()

# Simular dados de 15 minutos
if pares_selecionados:
    dados_15min = pd.concat([
        simular_dados_15min(par, PARES[par]["base"]) 
        for par in pares_selecionados
    ])

    # Calcular indicadores técnicos
    dados_15min = dados_15min.groupby('par').apply(calcular_indicadores_tecnicos).reset_index(drop=True)
    dados_15min = dados_15min.groupby('par').apply(gerar_sinais_15min).reset_index(drop=True)
else:
    dados_15min = pd.DataFrame()

# Layout principal em abas
tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs(["📊 Dashboard Live", "🎯 Estratégia 15min", "📈 Price Action", "💰 Preços Live", "📋 Relatórios", "🔍 Análise Candles"])

with tab1:
    # Header com informações de atualização
    col_update1, col_update2, col_update3 = st.columns(3)
    with col_update1:
        st.metric("Última Atualização", st.session_state.ultima_atualizacao.strftime("%H:%M:%S"))
    with col_update2:
        st.metric("Total Updates", st.session_state.contador_updates)
    with col_update3:
        tempo_decorrido = (datetime.now() - st.session_state.ultima_atualizacao).seconds
        st.metric("Segundos desde update", tempo_decorrido)
    
    # KPIs principais
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        total_trades = len(st.session_state.logs_tecnicos)
        st.metric("Total Trades", total_trades)
    
    with col2:
        trades_ativos = len([log for log in st.session_state.logs_tecnicos if log.get('status') == 'Ativo'])
        st.metric("Trades Ativos", trades_ativos)
    
    with col3:
        win_rate = np.random.uniform(65, 75) if st.session_state.logs_tecnicos else 0
        st.metric("Win Rate", f"{win_rate:.1f}%")
    
    with col4:
        stake_atual = st.session_state.stake_valor
        st.metric("Stake Atual", f"€{stake_atual}")
    
    # Aplicar estilo personalizado às métricas
    custom_metric_style()
    
    # Gráfico de preços em tempo real
    st.subheader("📈 Preços em Tempo Real")
    
    if pares_selecionados and st.session_state.dados_live:
        fig_live = go.Figure()
        
        for par in pares_selecionados[:3]:  # Mostrar apenas 3 pares para não poluir
            if par in st.session_state.dados_live and st.session_state.dados_live[par]:
                df_par = pd.DataFrame(st.session_state.dados_live[par])
                fig_live.add_trace(go.Scatter(
                    x=df_par['timestamp'], 
                    y=df_par['bid'],
                    mode='lines',
                    name=par,
                    line=dict(width=2)
                ))
        
        fig_live.update_layout(
            title="Evolução de Preços em Tempo Real",
            xaxis_title="Tempo",
            yaxis_title="Preço (Bid)",
            height=400,
            showlegend=True
        )
        st.plotly_chart(fig_live, width='stretch')
    
    # Alertas de volatilidade
    st.subheader("🚨 Alertas de Volatilidade")
    
    if not dados_15min.empty:
        # Calcular volatilidade dos últimos dados
        volatilidade_recente = dados_15min.groupby('par').apply(
            lambda x: (x['high'].max() - x['low'].min()) / x['close'].mean()
        ).round(5)
        
        alertas_ativos = []
        for par, vol in volatilidade_recente.items():
            if vol > LIMITE_VOLATILIDADE.get(par, 0.0020):
                alertas_ativos.append((par, vol))
        
        if alertas_ativos:
            for par, vol in alertas_ativos:
                st.markdown(f'<div class="alert-high"><b>{par}</b>: Volatilidade {vol:.4f} - ALERTA!</div>', unsafe_allow_html=True)
        else:
            st.info("✅ Nenhum alerta de volatilidade no momento")

with tab2:
    st.header("🎯 Estratégia de 15 Minutos")
    
    if not dados_15min.empty:
        col1, col2 = st.columns([2, 1])
        
        with col1:
            # Gráfico de candlesticks com sinais
            par_selecionado = st.selectbox("Selecione o par:", pares_selecionados, key="strategy_select")
            
            if par_selecionado:
                df_par = dados_15min[dados_15min['par'] == par_selecionado].tail(50)
                
                # Criar gráfico de candlestick
                fig_candle = go.Figure()
                
                # Candlesticks
                fig_candle.add_trace(go.Candlestick(
                    x=df_par['timestamp'],
                    open=df_par['open'],
                    high=df_par['high'],
                    low=df_par['low'],
                    close=df_par['close'],
                    name="Price"
                ))
                
                # EMAs
                fig_candle.add_trace(go.Scatter(
                    x=df_par['timestamp'], y=df_par['EMA_9'],
                    mode='lines', name='EMA 9',
                    line=dict(color='orange', width=1)
                ))
                
                fig_candle.add_trace(go.Scatter(
                    x=df_par['timestamp'], y=df_par['EMA_21'],
                    mode='lines', name='EMA 21',
                    line=dict(color='blue', width=1)
                ))
                
                # Bollinger Bands
                fig_candle.add_trace(go.Scatter(
                    x=df_par['timestamp'], y=df_par['BB_Upper'],
                    mode='lines', name='BB Upper',
                    line=dict(color='gray', width=1, dash='dash')
                ))
                
                fig_candle.add_trace(go.Scatter(
                    x=df_par['timestamp'], y=df_par['BB_Lower'],
                    mode='lines', name='BB Lower',
                    line=dict(color='gray', width=1, dash='dash')
                ))
                
                # Sinais de trading
                compras = df_par[df_par['sinal'] == 'COMPRA']
                vendas = df_par[df_par['sinal'] == 'VENDA']
                
                if not compras.empty:
                    fig_candle.add_trace(go.Scatter(
                        x=compras['timestamp'], y=compras['low'] * 0.998,
                        mode='markers', name='Compra',
                        marker=dict(color='green', size=10, symbol='triangle-up')
                    ))
                
                if not vendas.empty:
                    fig_candle.add_trace(go.Scatter(
                        x=vendas['timestamp'], y=vendas['high'] * 1.002,
                        mode='markers', name='Venda',
                        marker=dict(color='red', size=10, symbol='triangle-down')
                    ))
                
                fig_candle.update_layout(
                    title=f"{par_selecionado} - Candlestick 15min com Sinais",
                    xaxis_title="Data/Hora",
                    yaxis_title="Preço",
                    height=500
                )
                
                st.plotly_chart(fig_candle, width='stretch')
        
        with col2:
            st.subheader("📋 Sinais Atuais")
            
            # Últimos sinais por par
            ultimos_sinais = []
            for par in pares_selecionados:
                df_par = dados_15min[dados_15min['par'] == par]
                if not df_par.empty:
                    ultimo_sinal = df_par.iloc[-1]
                    tendencia = analisar_tendencia(df_par.tail(20))
                    
                    ultimos_sinais.append({
                        'Par': par,
                        'Sinal': ultimo_sinal['sinal'],
                        'Padrão': ultimo_sinal['padrao_candle'],
                        'Preço': ultimo_sinal['close'],
                        'RSI': f"{ultimo_sinal['RSI']:.1f}" if not pd.isna(ultimo_sinal['RSI']) else "N/A",
                        'Tendência': tendencia
                    })
            
            if ultimos_sinais:
                df_sinais = pd.DataFrame(ultimos_sinais)
                
                # Colorir os sinais
                def colorir_sinal(val):
                    if val == 'COMPRA':
                        return 'background-color: #90EE90'
                    elif val == 'VENDA':
                        return 'background-color: #FFB6C1'
                    else:
                        return ''
                
                st.dataframe(
                    df_sinais.style.applymap(colorir_sinal, subset=['Sinal']),
                    width='stretch'
                )
            else:
                st.info("Nenhum sinal disponível")
            
            # Botão de execução
            st.subheader("🤖 Executar Trades")
            if st.button("🚀 Executar Estratégia 15min", type="primary"):
                trades_executados = 0
                for par in pares_selecionados:
                    df_par = dados_15min[dados_15min['par'] == par]
                    if not df_par.empty:
                        ultimo_sinal = df_par.iloc[-1]
                        
                        if ultimo_sinal['sinal'] in ['COMPRA', 'VENDA']:
                            stake = st.session_state.stake_valor
                            risco = round(stake * 0.01, 2)
                            lucro = round(stake * 0.02, 2)
                            
                            st.session_state.logs_tecnicos.append({
                                "timestamp": datetime.now(),
                                "data": str(datetime.now().date()),
                                "pair": par,
                                "sinal": ultimo_sinal['sinal'],
                                "preco_entrada": ultimo_sinal['close'],
                                "timeframe": "15min",
                                "trade_executado": "Sim",
                                "status": "Ativo",
                                "log_stake": stake,
                                "log_risco_estimado": risco,
                                "log_lucro_estimado": lucro,
                                "rsi_entrada": round(ultimo_sinal['RSI'], 1) if not pd.isna(ultimo_sinal['RSI']) else 0,
                                "ema_tendencia": "Alta" if ultimo_sinal['EMA_9'] > ultimo_sinal['EMA_21'] else "Baixa",
                                "padrao_candle": ultimo_sinal['padrao_candle']
                            })
                            trades_executados += 1
                
                if trades_executados > 0:
                    st.success(f"✅ {trades_executados} trades executados!")
                else:
                    st.info("Nenhum sinal de trade válido encontrado")
    else:
        st.info("Selecione pares para análise na sidebar")

with tab3:
    st.header("📈 Análise de Price Action")
    
    if not dados_15min.empty:
        par_selecionado_pa = st.selectbox("Selecione o par para análise:", pares_selecionados, key="pa_select")
        
        if par_selecionado_pa:
            df_par = dados_15min[dados_15min['par'] == par_selecionado_pa].tail(30)
            
            col1, col2 = st.columns(2)
            
            with col1:
                # Análise de Tendência
                st.subheader("📊 Análise de Tendência")
                tendencia = analisar_tendencia(df_par)
                st.metric("Tendência Atual", tendencia)
                
                # Suporte e Resistência
                st.subheader("🛡️ Suporte e Resistência")
                ultimo = df_par.iloc[-1]
                col_sup_res1, col_sup_res2 = st.columns(2)
                with col_sup_res1:
                    st.metric("Resistência", f"{ultimo['resistance']:.5f}")
                with col_sup_res2:
                    st.metric("Suporte", f"{ultimo['support']:.5f}")
                
                # Volume Profile
                st.subheader("📈 Perfil de Volume")
                volume_profile = calcular_volume_profile(df_par)
                if volume_profile:
                    df_volume = pd.DataFrame(list(volume_profile.items()), columns=['Preço', 'Volume'])
                    df_volume = df_volume.sort_values('Volume', ascending=False).head(5)
                    st.dataframe(df_volume, width='stretch')
            
            with col2:
                # Padrões de Candles Detectados
                st.subheader("🕯️ Padrões de Candles")
                padroes_recentes = df_par[df_par['padrao_candle'] != ''].tail(5)
                
                if not padroes_recentes.empty:
                    df_padroes = padroes_recentes[['timestamp', 'padrao_candle', 'tipo_padrao']].copy()
                    df_padroes['timestamp'] = df_padroes['timestamp'].dt.strftime('%H:%M')
                    
                    # Aplicar cores baseadas no tipo de padrão
                    def color_pattern(row):
                        if row['tipo_padrao'] == 'bullish':
                            return ['pattern-bullish'] * len(row)
                        elif row['tipo_padrao'] == 'bearish':
                            return ['pattern-bearish'] * len(row)
                        else:
                            return ['pattern-neutral'] * len(row)
                    
                    st.dataframe(
                        df_padroes.style.apply(color_pattern, axis=1),
                        width='stretch'
                    )
                else:
                    st.info("Nenhum padrão de candle detectado recentemente")
                
                # Análise de Momentum
                st.subheader("⚡ Momentum")
                col_mom1, col_mom2 = st.columns(2)
                with col_mom1:
                    rsi_atual = df_par['RSI'].iloc[-1]
                    st.metric("RSI", f"{rsi_atual:.1f}")
                with col_mom2:
                    macd_atual = df_par['MACD'].iloc[-1]
                    st.metric("MACD", f"{macd_atual:.4f}")
            
            # Gráfico de Price Action Avançado
            st.subheader("🎯 Gráfico de Price Action")
            
            fig_pa = go.Figure()
            
            # Candlesticks
            fig_pa.add_trace(go.Candlestick(
                x=df_par['timestamp'],
                open=df_par['open'],
                high=df_par['high'],
                low=df_par['low'],
                close=df_par['close'],
                name="Price"
            ))
            
            # Suporte e Resistência
            fig_pa.add_trace(go.Scatter(
                x=df_par['timestamp'], y=df_par['resistance'],
                mode='lines', name='Resistência',
                line=dict(color='red', width=2, dash='dash')
            ))
            
            fig_pa.add_trace(go.Scatter(
                x=df_par['timestamp'], y=df_par['support'],
                mode='lines', name='Suporte',
                line=dict(color='green', width=2, dash='dash')
            ))
            
            # Destacar padrões de candles
            padroes = df_par[df_par['padrao_candle'] != '']
            for _, padrao in padroes.iterrows():
                color = 'green' if padrao['tipo_padrao'] == 'bullish' else 'red' if padrao['tipo_padrao'] == 'bearish' else 'yellow'
                fig_pa.add_trace(go.Scatter(
                    x=[padrao['timestamp']], y=[padrao['high'] * 1.001],
                    mode='markers',
                    marker=dict(color=color, size=12, symbol='star'),
                    name=f"{padrao['padrao_candle']}",
                    showlegend=False
                ))
            
            fig_pa.update_layout(
                title=f"{par_selecionado_pa} - Análise de Price Action",
                xaxis_title="Data/Hora",
                yaxis_title="Preço",
                height=500
            )
            
            st.plotly_chart(fig_pa, width='stretch')
    else:
        st.info("Selecione pares para análise na sidebar")

with tab4:
    st.header("💰 Preços em Tempo Real")
    
    if pares_selecionados and st.session_state.dados_live:
        # Tabela de preços atualizados
        st.subheader("📊 Cotações Atuais")
        
        precos_atuais = []
        for par in pares_selecionados:
            if par in st.session_state.dados_live and st.session_state.dados_live[par]:
                ultimo_dado = st.session_state.dados_live[par][-1]
                precos_atuais.append({
                    'Par': par,
                    'Bid': ultimo_dado['bid'],
                    'Ask': ultimo_dado['ask'],
                    'Spread': ultimo_dado['spread'],
                    'Volume': ultimo_dado['volume'],
                    'Última Atualização': ultimo_dado['timestamp'].strftime("%H:%M:%S")
                })
        
        if precos_atuais:
            df_precos = pd.DataFrame(precos_atuais)
            st.dataframe(df_precos, width='stretch')
        
        # Gráficos individuais por par
        st.subheader("📈 Evolução Individual por Par")
        
        for par in pares_selecionados[:4]:  # Mostrar até 4 pares
            if par in st.session_state.dados_live and st.session_state.dados_live[par]:
                df_par = pd.DataFrame(st.session_state.dados_live[par])
                
                fig_individual = go.Figure()
                fig_individual.add_trace(go.Scatter(
                    x=df_par['timestamp'], 
                    y=df_par['bid'],
                    mode='lines+markers',
                    name=f'{par} Bid',
                    line=dict(color='blue', width=2),
                    marker=dict(size=4)
                ))
                
                fig_individual.update_layout(
                    title=f"{par} - Preço em Tempo Real",
                    xaxis_title="Tempo",
                    yaxis_title="Preço",
                    height=300
                )
                
                st.plotly_chart(fig_individual, width='stretch')
    else:
        st.info("Selecione pares para ver dados em tempo real")

with tab5:
    st.header("📋 Relatórios e Exportação")
    
    # Painel de trades
    if st.session_state.logs_tecnicos:
        df_logs = pd.DataFrame(st.session_state.logs_tecnicos)
        
        # KPIs de performance
        col1, col2, col3, col4 = st.columns(4)
        
        lucro_total = df_logs['log_lucro_estimado'].sum()
        risco_total = df_logs['log_risco_estimado'].sum()
        trades_vencedores = len(df_logs[df_logs['log_lucro_estimado'] > 0])
        win_rate = (trades_vencedores / len(df_logs)) * 100 if len(df_logs) > 0 else 0
        
        with col1:
            st.metric("Lucro Total Estimado", f"€{lucro_total:.2f}")
        with col2:
            st.metric("Risco Total", f"€{risco_total:.2f}")
        with col3:
            st.metric("Trades com Lucro", trades_vencedores)
        with col4:
            st.metric("Win Rate", f"{win_rate:.1f}%")
        
        # Aplicar estilo às métricas
        custom_metric_style()
        
        # Tabela de trades
        st.subheader("📋 Histórico de Trades")
        
        # Formatar colunas
        df_display = df_logs.copy()
        if 'timestamp' in df_display.columns:
            df_display['Timestamp'] = pd.to_datetime(df_display['timestamp']).dt.strftime('%Y-%m-%d %H:%M')
        else:
            df_display['Timestamp'] = df_display.get('data', 'N/A')
        
        df_display['Par'] = df_display['pair']
        df_display['Stake (€)'] = df_display['log_stake']
        df_display['Lucro (€)'] = df_display['log_lucro_estimado']
        df_display['Risco (€)'] = df_display['log_risco_estimado']
        
        colunas_display = ['Timestamp', 'Par', 'sinal', 'Stake (€)', 'Lucro (€)', 'Risco (€)', 'status', 'padrao_candle']
        if 'rsi_entrada' in df_display.columns:
            colunas_display.extend(['rsi_entrada', 'ema_tendencia'])
        
        st.dataframe(df_display[colunas_display], width='stretch')
        
        # Gráfico de lucro acumulado
        st.subheader("💰 Evolução do Lucro")
        
        if 'timestamp' in df_logs.columns:
            df_logs['timestamp'] = pd.to_datetime(df_logs['timestamp'])
            df_logs = df_logs.sort_values('timestamp')
            df_logs['Lucro Acumulado'] = df_logs['log_lucro_estimado'].cumsum()
            
            fig_lucro = px.line(df_logs, x='timestamp', y='Lucro Acumulado', 
                              title='Lucro Acumulado ao Longo do Tempo')
            fig_lucro.update_layout(height=400)
            st.plotly_chart(fig_lucro, width='stretch')
    else:
        st.info("Nenhum trade executado ainda")
    
    # Exportação de dados
    st.subheader("📥 Exportar Dados")
    
    col_exp1, col_exp2 = st.columns(2)
    
    with col_exp1:
        # Exportar dados de trading
        if st.button("📊 Exportar Relatório Completo"):
            buffer = io.BytesIO()
            with pd.ExcelWriter(buffer, engine='xlsxwriter') as writer:
                if st.session_state.logs_tecnicos:
                    df_logs = pd.DataFrame(st.session_state.logs_tecnicos)
                    df_logs.to_excel(writer, sheet_name='Trades Executados', index=False)
                
                if not dados_15min.empty:
                    dados_15min.to_excel(writer, sheet_name='Dados 15min', index=False)
                
                # Dados live
                dados_live_export = []
                for par, dados in st.session_state.dados_live.items():
                    for dado in dados:
                        dados_live_export.append(dado)
                if dados_live_export:
                    pd.DataFrame(dados_live_export).to_excel(writer, sheet_name='Dados Live', index=False)
                
                # Resumo estatístico
                resumo = {
                    'Metrica': ['Total Trades', 'Lucro Total', 'Risco Total', 'Win Rate', 'Total Updates Live'],
                    'Valor': [
                        len(st.session_state.logs_tecnicos), 
                        lucro_total if st.session_state.logs_tecnicos else 0, 
                        risco_total if st.session_state.logs_tecnicos else 0, 
                        f"{win_rate:.1f}%" if st.session_state.logs_tecnicos else "0%",
                        st.session_state.contador_updates
                    ]
                }
                pd.DataFrame(resumo).to_excel(writer, sheet_name='Resumo', index=False)
            
            st.download_button(
                label="⬇️ Baixar Relatório Excel",
                data=buffer.getvalue(),
                file_name=f"relatorio_trading_{datetime.now().strftime('%Y%m%d_%H%M')}.xlsx",
                mime="application/vnd.ms-excel"
            )

with tab6:
    st.header("🔍 Análise Detalhada de Candles")
    
    if not dados_15min.empty:
        par_selecionado_candle = st.selectbox("Selecione o par:", pares_selecionados, key="candle_analysis")
        
        if par_selecionado_candle:
            df_par = dados_15min[dados_15min['par'] == par_selecionado_candle].tail(50)
            
            col1, col2 = st.columns(2)
            
            with col1:
                # Estatísticas de Candles
                st.subheader("📊 Estatísticas de Candles")
                
                # Tipos de candles
                df_par['candle_type'] = np.where(df_par['close'] > df_par['open'], 'Alta', 'Baixa')
                candle_stats = df_par['candle_type'].value_counts()
                
                fig_candle_types = px.pie(
                    values=candle_stats.values,
                    names=candle_stats.index,
                    title="Distribuição de Candles de Alta/Baixa"
                )
                st.plotly_chart(fig_candle_types, width='stretch')
                
                # Médias de tamanho
                st.metric("Tamanho Médio do Corpo", f"{(df_par['Body'].mean() * 10000):.1f} pips")
                st.metric("Tamanho Médio da Sombra Superior", f"{(df_par['Upper_Shadow'].mean() * 10000):.1f} pips")
                st.metric("Tamanho Médio da Sombra Inferior", f"{(df_par['Lower_Shadow'].mean() * 10000):.1f} pips")
            
            with col2:
                # Padrões Detectados
                st.subheader("🕯️ Resumo de Padrões")
                
                padroes_count = df_par['padrao_candle'].value_counts()
                if not padroes_count.empty:
                    fig_patterns = px.bar(
                        x=padroes_count.values,
                        y=padroes_count.index,
                        orientation='h',
                        title="Frequência de Padrões de Candles"
                    )
                    st.plotly_chart(fig_patterns, width='stretch')
                else:
                    st.info("Nenhum padrão detectado no período")
                
                # Eficiência dos padrões
                st.subheader("📈 Eficiência dos Sinais")
                if 'padrao_candle' in df_par.columns and 'sinal' in df_par.columns:
                    eficiencia = df_par.groupby('padrao_candle').agg({
                        'sinal': lambda x: (x != 'NEUTRO').mean() * 100
                    }).reset_index()
                    eficiencia.columns = ['Padrão', 'Taxa de Sinal (%)']
                    st.dataframe(eficiencia, width='stretch')
            
            # Heatmap de Volume por Preço
            st.subheader("🔥 Heatmap de Volume por Preço")
            
            # Criar heatmap simplificado
            price_levels = np.linspace(df_par['low'].min(), df_par['high'].max(), 10)
            volume_heatmap = []
            
            for i in range(len(price_levels)-1):
                low_level = price_levels[i]
                high_level = price_levels[i+1]
                mask = (df_par['low'] >= low_level) & (df_par['high'] <= high_level)
                total_volume = df_par[mask]['volume'].sum()
                volume_heatmap.append({
                    'price_range': f"{low_level:.4f}-{high_level:.4f}",
                    'volume': total_volume
                })
            
            df_heatmap = pd.DataFrame(volume_heatmap)
            fig_heatmap = px.bar(df_heatmap, x='volume', y='price_range', orientation='h',
                               title='Distribuição de Volume por Faixa de Preço')
            st.plotly_chart(fig_heatmap, width='stretch')

# Sistema de atualização automática
if auto_refresh:
    time.sleep(refresh_interval)
    st.rerun()

# Sistema de gestão de risco
st.sidebar.header("🛡️ Gestão de Risco")

if st.session_state.logs_tecnicos:
    df_logs = pd.DataFrame(st.session_state.logs_tecnicos)
    lucro_total = df_logs['log_lucro_estimado'].sum()
    
    if lucro_total >= meta_lucro:
        st.sidebar.success(f"🎯 Meta de lucro atingida: €{lucro_total:.2f}")
        st.sidebar.info("✅ Operações podem ser continuadas")
    elif lucro_total <= limite_perda:
        st.sidebar.error(f"📉 Limite de perda atingido: €{lucro_total:.2f}")
        st.sidebar.warning("⛔ Operações devem ser pausadas")
    else:
        st.sidebar.info(f"📊 Lucro atual: €{lucro_total:.2f}")

# Footer
st.markdown("---")
st.markdown(
    "<div style='text-align: center; color: gray;'>"
    "📊 Advanced Trading Dashboard - Price Action & Live 15min Strategy | Análise Completa de Candles"
    "</div>", 
    unsafe_allow_html=True
)

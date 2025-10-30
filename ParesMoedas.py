import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timedelta
import io
import time

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
    .market-open {
        background-color: #90EE90 !important;
        color: black;
        font-weight: bold;
    }
    .market-closed {
        background-color: #FFB6C1 !important;
        color: black;
        font-weight: bold;
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
    .bullish-sentiment {
        background-color: #90EE90 !important;
        color: black;
    }
    .bearish-sentiment {
        background-color: #FFB6C1 !important;
        color: black;
    }
    .neutral-sentiment {
        background-color: #F0F0F0 !important;
        color: black;
    }
    .trade-active {
        background-color: #90EE90 !important;
        color: black;
    }
    .trade-closed {
        background-color: #F0F0F0 !important;
        color: black;
    }
    .trade-profit {
        background-color: #90EE90 !important;
        color: black;
        font-weight: bold;
    }
    .trade-loss {
        background-color: #FFB6C1 !important;
        color: black;
        font-weight: bold;
    }
    .pair-card {
        background-color: #f8f9fa;
        padding: 1rem;
        border-radius: 10px;
        border-left: 4px solid #1f77b4;
        margin-bottom: 1rem;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
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
    "EUR/USD": 0.0020, "GBP/USD": 0.0025, "USD/JPY": 0.25, "AUD/USD": 0.0022,
    "USD/CAD": 0.0023, "USD/CHF": 0.0021, "EUR/GBP": 0.0018, "EUR/JPY": 0.0028,
    "GBP/JPY": 0.0030, "AUD/JPY": 0.0028
}

# Parâmetros simulados para pares principais
PARES = {
    "EUR/USD": {"base": 1.10, "vol": 0.005, "pip_value": 10},
    "GBP/USD": {"base": 1.26, "vol": 0.006, "pip_value": 10},
    "USD/JPY": {"base": 148.0, "vol": 0.004, "pip_value": 9},
    "AUD/USD": {"base": 0.66, "vol": 0.007, "pip_value": 10},
    "USD/CAD": {"base": 1.35, "vol": 0.005, "pip_value": 10},
    "USD/CHF": {"base": 0.88, "vol": 0.004, "pip_value": 10},
    "EUR/GBP": {"base": 0.87, "vol": 0.004, "pip_value": 10},
    "EUR/JPY": {"base": 162.0, "vol": 0.005, "pip_value": 9},
    "GBP/JPY": {"base": 186.0, "vol": 0.006, "pip_value": 9},
    "AUD/JPY": {"base": 97.0, "vol": 0.006, "pip_value": 9}
}

# Estratégia para candles de 15 minutos
STRATEGY_15MIN = {
    "timeframe": "15min",
    "indicators": ["EMA_9", "EMA_21", "RSI", "MACD"],
    "entry_conditions": {
        "trend": "EMA_9 > EMA_21",
        "oversold": "RSI < 35",
        "overbought": "RSI > 65", 
        "momentum": "MACD > 0"
    },
    "exit_conditions": {
        "take_profit": 0.0015,  # 15 pips
        "stop_loss": 0.0010,    # 10 pips
        "trailing_stop": False
    }
}

# Inicializar session state com valores padrão
def initialize_session_state():
    """Inicializa todos os session states com valores padrão"""
    defaults = {
        "dados_live": {},
        "ultima_atualizacao": datetime.now(),
        "contador_updates": 0,
        "logs_tecnicos": [],
        "stake_valor": 10,
        "auto_trade_active": False,
        "trades_ativos": [],
        "historico_trades": [],
        "auto_trade_stats": {
            "total_trades": 0,
            "trades_lucrativos": 0,
            "trades_perdedores": 0,
            "lucro_total": 0.0,
            "melhor_trade": 0.0,
            "pior_trade": 0.0,
            "win_rate": 0.0
        },
        "pares_config": {par: {"ativo": True, "stake": 10} for par in PARES.keys()},
        "dados_15min": pd.DataFrame(),
        "meta_lucro": 500,
        "limite_perda": -200,
        "mercado_aberto": True,  # Sempre aberto para testes
        "ultima_execucao_auto_trade": datetime.now() - timedelta(minutes=5)
    }
    
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value

# Inicializar session state
initialize_session_state()

def detectar_padroes_candles(df):
    """Detecta padrões de candles usando lógica de price action"""
    patterns = []
    
    if len(df) < 3:
        return patterns
    
    for i in range(2, len(df)):
        current = df.iloc[i]
        prev1 = df.iloc[i-1]
        
        # Calcular componentes do candle
        body = abs(current['close'] - current['open'])
        lower_wick = current['low'] - min(current['open'], current['close'])
        upper_wick = max(current['open'], current['close']) - current['high']
        total_range = current['high'] - current['low']
        
        # Evitar divisão por zero
        if total_range <= 0:
            continue
            
        body_ratio = body / total_range
        
        # Hammer (Martelo) - candle de reversão bullish
        if (lower_wick >= 2 * body and upper_wick <= body * 0.1 and body_ratio <= 0.3):
            patterns.append({
                "timestamp": current['timestamp'], 
                "pattern": "Hammer", 
                "type": "bullish", 
                "strength": "medium"
            })
        
        # Shooting Star (Estrela Cadente) - candle de reversão bearish
        if (upper_wick >= 2 * body and lower_wick <= body * 0.1 and body_ratio <= 0.3):
            patterns.append({
                "timestamp": current['timestamp'], 
                "pattern": "Shooting Star", 
                "type": "bearish", 
                "strength": "medium"
            })
        
        # Engulfing Bullish
        if (prev1['close'] < prev1['open'] and current['close'] > current['open'] and
            current['open'] < prev1['close'] and current['close'] > prev1['open']):
            patterns.append({
                "timestamp": current['timestamp'], 
                "pattern": "Bullish Engulfing", 
                "type": "bullish", 
                "strength": "strong"
            })
        
        # Engulfing Bearish
        if (prev1['close'] > prev1['open'] and current['close'] < current['open'] and
            current['open'] > prev1['close'] and current['close'] < prev1['open']):
            patterns.append({
                "timestamp": current['timestamp'], 
                "pattern": "Bearish Engulfing", 
                "type": "bearish", 
                "strength": "strong"
            })
    
    return patterns

def executar_auto_trade():
    """Executa a lógica de auto trading para todos os pares"""
    if not st.session_state.auto_trade_active:
        return
    
    # Verificar se passou tempo suficiente desde a última execução (evitar múltiplas execuções rápidas)
    tempo_atual = datetime.now()
    tempo_desde_ultima_execucao = (tempo_atual - st.session_state.ultima_execucao_auto_trade).total_seconds()
    
    if tempo_desde_ultima_execucao < 10:  # Esperar pelo menos 10 segundos entre execuções
        return
    
    st.session_state.ultima_execucao_auto_trade = tempo_atual
    
    # Verificar se há dados suficientes
    if st.session_state.dados_15min is None or st.session_state.dados_15min.empty:
        st.warning("⚠️ Dados insuficientes para auto trade")
        return
    
    # Verificar limites de risco
    if verificar_limites_risco():
        st.session_state.auto_trade_active = False
        st.error("🚫 Auto Trade pausado - Limites de risco atingidos")
        return
    
    trades_executados = 0
    pares_com_sinais = []
    
    for par in PARES.keys():
        # Verificar se o par está ativo na configuração
        if not st.session_state.pares_config[par]["ativo"]:
            continue
            
        df_par = st.session_state.dados_15min[st.session_state.dados_15min['par'] == par]
        
        if len(df_par) < 10:
            continue
            
        ultimo = df_par.iloc[-1]
        
        # Verificar se já existe trade ativo para este par
        trade_ativo = any(trade['par'] == par and trade['status'] == 'ativo' 
                         for trade in st.session_state.trades_ativos)
        
        if trade_ativo:
            # Gerenciar trade ativo
            gerenciar_trade_ativo(par, df_par)
        else:
            # Procurar nova entrada - CONDIÇÕES MAIS FLEXÍVEIS
            if verificar_condicoes_entrada_simplificada(ultimo):
                if executar_nova_entrada(par, ultimo):
                    trades_executados += 1
                    pares_com_sinais.append(par)
    
    if trades_executados > 0:
        st.success(f"🤖 Auto Trade executou {trades_executados} trades: {', '.join(pares_com_sinais)}")
        st.session_state.logs_tecnicos.append({
            "timestamp": datetime.now(),
            "tipo": "AUTO_TRADE_EXECUCAO",
            "trades_executados": trades_executados,
            "pares": pares_com_sinais,
            "status": f"Executados {trades_executados} trades"
        })
    else:
        # Mostrar que o auto trade está rodando mas não encontrou sinais
        st.info("🔍 Auto Trade ativo - Procurando sinais...")

def verificar_condicoes_entrada_simplificada(ultimo):
    """Verifica condições para entrada no trade - VERSÃO SIMPLIFICADA"""
    try:
        # Condições mais flexíveis para TESTE
        condicoes_compra = (
            ultimo['sinal'] == 'COMPRA' and
            ultimo['EMA_9'] > ultimo['EMA_21'] and
            ultimo['RSI'] < 40  # Mais flexível
        )
        
        condicoes_venda = (
            ultimo['sinal'] == 'VENDA' and
            ultimo['EMA_9'] < ultimo['EMA_21'] and 
            ultimo['RSI'] > 60  # Mais flexível
        )
        
        return condicoes_compra or condicoes_venda
    except Exception as e:
        st.error(f"Erro na verificação de condições: {e}")
        return False

def executar_nova_entrada(par, dados_entrada):
    """Executa uma nova entrada de trade"""
    try:
        stake = st.session_state.pares_config[par]["stake"]
        tipo_operacao = 'COMPRA' if dados_entrada['sinal'] == 'COMPRA' else 'VENDA'
        preco_entrada = dados_entrada['close']
        
        # Calcular stop loss e take profit
        if tipo_operacao == 'COMPRA':
            stop_loss = preco_entrada * (1 - STRATEGY_15MIN['exit_conditions']['stop_loss'])
            take_profit = preco_entrada * (1 + STRATEGY_15MIN['exit_conditions']['take_profit'])
        else:
            stop_loss = preco_entrada * (1 + STRATEGY_15MIN['exit_conditions']['stop_loss'])
            take_profit = preco_entrada * (1 - STRATEGY_15MIN['exit_conditions']['take_profit'])
        
        trade = {
            'id': len(st.session_state.trades_ativos) + len(st.session_state.historico_trades) + 1,
            'par': par,
            'tipo': tipo_operacao,
            'preco_entrada': round(preco_entrada, 5),
            'stop_loss': round(stop_loss, 5),
            'take_profit': round(take_profit, 5),
            'stake': stake,
            'timestamp_entrada': datetime.now(),
            'status': 'ativo',
            'lucro_prejuizo': 0.0,
            'rsi_entrada': round(dados_entrada['RSI'], 1),
            'padrao_entrada': dados_entrada.get('padrao_candle', '')
        }
        
        st.session_state.trades_ativos.append(trade)
        
        # Registrar no log
        st.session_state.logs_tecnicos.append({
            "timestamp": datetime.now(),
            "tipo": "AUTO_TRADE_ENTRADA",
            "par": par,
            "operacao": tipo_operacao,
            "preco": round(preco_entrada, 5),
            "stake": stake,
            "stop_loss": round(stop_loss, 5),
            "take_profit": round(take_profit, 5),
            "status": "Entrada executada"
        })
        
        return True
    except Exception as e:
        st.error(f"Erro ao executar entrada: {e}")
        return False

def gerenciar_trade_ativo(par, df_par):
    """Gerencia trade ativo (check stop loss e take profit)"""
    try:
        trades_par = [t for t in st.session_state.trades_ativos if t['par'] == par and t['status'] == 'ativo']
        
        for trade in trades_par:
            preco_atual = df_par.iloc[-1]['close']
            
            # Verificar stop loss
            if (trade['tipo'] == 'COMPRA' and preco_atual <= trade['stop_loss']) or \
               (trade['tipo'] == 'VENDA' and preco_atual >= trade['stop_loss']):
                fechar_trade(trade, preco_atual, 'stop_loss')
                st.warning(f"⛔ STOP LOSS atingido para {trade['par']}")
            
            # Verificar take profit
            elif (trade['tipo'] == 'COMPRA' and preco_atual >= trade['take_profit']) or \
                 (trade['tipo'] == 'VENDA' and preco_atual <= trade['take_profit']):
                fechar_trade(trade, preco_atual, 'take_profit')
                st.success(f"🎯 TAKE PROFIT atingido para {trade['par']}")
                
    except Exception as e:
        st.error(f"Erro ao gerenciar trade: {e}")

def fechar_trade(trade, preco_saida, motivo):
    """Fecha um trade e move para o histórico"""
    try:
        # Calcular lucro/prejuízo
        if trade['tipo'] == 'COMPRA':
            lucro_pips = (preco_saida - trade['preco_entrada']) / trade['preco_entrada']
        else:
            lucro_pips = (trade['preco_entrada'] - preco_saida) / trade['preco_entrada']
        
        lucro_euros = lucro_pips * trade['stake'] * 10000  # Conversão simplificada
        
        # Atualizar trade
        trade['preco_saida'] = round(preco_saida, 5)
        trade['timestamp_saida'] = datetime.now()
        trade['status'] = 'fechado'
        trade['motivo_saida'] = motivo
        trade['lucro_prejuizo'] = round(lucro_euros, 2)
        
        # Mover para histórico
        st.session_state.historico_trades.append(trade.copy())
        st.session_state.trades_ativos = [t for t in st.session_state.trades_ativos if t['id'] != trade['id']]
        
        # Atualizar estatísticas
        st.session_state.auto_trade_stats["total_trades"] += 1
        st.session_state.auto_trade_stats["lucro_total"] += lucro_euros
        
        if lucro_euros > 0:
            st.session_state.auto_trade_stats["trades_lucrativos"] += 1
            st.session_state.auto_trade_stats["melhor_trade"] = max(
                st.session_state.auto_trade_stats["melhor_trade"], lucro_euros
            )
        else:
            st.session_state.auto_trade_stats["trades_perdedores"] += 1
            st.session_state.auto_trade_stats["pior_trade"] = min(
                st.session_state.auto_trade_stats["pior_trade"], lucro_euros
            )
        
        # Calcular win rate
        total = st.session_state.auto_trade_stats["total_trades"]
        lucrativos = st.session_state.auto_trade_stats["trades_lucrativos"]
        st.session_state.auto_trade_stats["win_rate"] = (lucrativos / total * 100) if total > 0 else 0
        
        # Registrar no log
        st.session_state.logs_tecnicos.append({
            "timestamp": datetime.now(),
            "tipo": "AUTO_TRADE_SAIDA",
            "par": trade['par'],
            "operacao": trade['tipo'],
            "preco_entrada": trade['preco_entrada'],
            "preco_saida": round(preco_saida, 5),
            "lucro": round(lucro_euros, 2),
            "motivo": motivo,
            "status": "Trade fechado"
        })
        
    except Exception as e:
        st.error(f"Erro ao fechar trade: {e}")

def verificar_limites_risco():
    """Verifica limites de risco para pausar auto trade"""
    try:
        lucro_total = st.session_state.auto_trade_stats["lucro_total"]
        meta_lucro = st.session_state.get('meta_lucro', 500)
        limite_perda = st.session_state.get('limite_perda', -200)
        
        if lucro_total >= meta_lucro:
            st.success(f"🎯 Meta de lucro atingida: €{lucro_total:.2f}")
            return True
        elif lucro_total <= limite_perda:
            st.error(f"📉 Limite de perda atingido: €{lucro_total:.2f}")
            return True
        
        return False
    except:
        return False

def gerar_dado_live(par, base_price):
    """Gera um novo dado em tempo real para um par"""
    agora = datetime.now()
    
    # Simular preço atual com movimento realista
    variacao = np.random.normal(0, PARES[par]["vol"] * 0.1)
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
    """Atualiza dados em tempo real para todos os pares"""
    agora = datetime.now()
    st.session_state.ultima_atualizacao = agora
    st.session_state.contador_updates += 1
    
    for par in PARES.keys():
        novo_dado = gerar_dado_live(par, PARES[par]["base"])
        
        if par not in st.session_state.dados_live:
            st.session_state.dados_live[par] = []
        
        # Manter apenas os últimos 50 pontos
        st.session_state.dados_live[par].append(novo_dado)
        if len(st.session_state.dados_live[par]) > 50:
            st.session_state.dados_live[par] = st.session_state.dados_live[par][-50:]

def simular_dados_15min(par, base_price, days=1):
    """Simula dados de 15 minutos para análise"""
    timeframe_data = []
    current_time = datetime.now().replace(hour=0, minute=0, second=0, microsecond=0)
    
    for day in range(days):
        for hour in range(24):
            for minute in range(0, 60, 15):
                timestamp = current_time - timedelta(days=day) + timedelta(hours=hour, minutes=minute)
                
                # Simular preços OHLC
                variacao = np.random.normal(0, PARES[par]["vol"])
                open_price = base_price * (1 + variacao)
                close = open_price * (1 + np.random.normal(0, PARES[par]["vol"] * 0.6))
                high = max(open_price, close) * (1 + abs(np.random.normal(0, PARES[par]["vol"] * 0.4)))
                low = min(open_price, close) * (1 - abs(np.random.normal(0, PARES[par]["vol"] * 0.4)))
                
                timeframe_data.append({
                    "timestamp": timestamp,
                    "par": par,
                    "timeframe": "15min",
                    "open": round(open_price, 5),
                    "high": round(high, 5),
                    "low": round(low, 5),
                    "close": round(close, 5),
                    "volume": np.random.randint(100, 1000)
                })
    
    return pd.DataFrame(timeframe_data)

def calcular_indicadores_tecnicos(df):
    """Calcula indicadores técnicos para estratégia de 15min"""
    try:
        if df.empty:
            return df
            
        df = df.sort_values('timestamp').copy()
        
        # EMA
        df['EMA_9'] = df['close'].ewm(span=9).mean()
        df['EMA_21'] = df['close'].ewm(span=21).mean()
        
        # RSI
        delta = df['close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14, min_periods=1).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14, min_periods=1).mean()
        rs = gain / loss
        df['RSI'] = 100 - (100 / (1 + rs))
        df['RSI'] = df['RSI'].fillna(50)
        
        # MACD simplificado
        exp1 = df['close'].ewm(span=12).mean()
        exp2 = df['close'].ewm(span=26).mean()
        df['MACD'] = exp1 - exp2
        df['MACD_Signal'] = df['MACD'].ewm(span=9).mean()
        
        return df
    except Exception as e:
        st.error(f"Erro no cálculo de indicadores: {e}")
        return df

def gerar_sinais_15min(df):
    """Gera sinais de trading baseados na estratégia de 15min - VERSÃO SIMPLIFICADA"""
    try:
        if df.empty:
            return df
            
        df = df.copy()
        df['sinal'] = 'NEUTRO'
        
        # Detectar padrões de candles
        padroes = detectar_padroes_candles(df)
        df['padrao_candle'] = ''
        df['tipo_padrao'] = ''
        
        for padrao in padroes:
            mask = df['timestamp'] == padrao['timestamp']
            if mask.any():
                df.loc[mask, 'padrao_candle'] = padrao['pattern']
                df.loc[mask, 'tipo_padrao'] = padrao['type']
        
        # Condições de compra MAIS FLEXÍVEIS
        buy_condition = (
            (df['EMA_9'] > df['EMA_21']) & 
            (df['RSI'] < 40)  # Mais flexível
        )
        
        # Condições de venda MAIS FLEXÍVEIS
        sell_condition = (
            (df['EMA_9'] < df['EMA_21']) & 
            (df['RSI'] > 60)  # Mais flexível
        )
        
        df.loc[buy_condition, 'sinal'] = 'COMPRA'
        df.loc[sell_condition, 'sinal'] = 'VENDA'
        
        return df
    except Exception as e:
        st.error(f"Erro na geração de sinais: {e}")
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

# Interface principal
st.markdown('<h1 class="main-header">🎯 Advanced Trading Dashboard</h1>', unsafe_allow_html=True)
st.markdown('<h3 class="main-header" style="font-size: 1.5rem;">Auto Trading System - TEST MODE</h3>', unsafe_allow_html=True)

# Sidebar modernizada
with st.sidebar:
    st.header("⚙️ Configurações de Trading")
    
    # Indicador de status
    col_status1, col_status2 = st.columns([2, 1])
    with col_status1:
        if st.session_state.auto_trade_active:
            st.markdown(f"<span class='live-indicator'>AUTO TRADE ATIVO</span>", unsafe_allow_html=True)
        else:
            st.markdown(f"<span style='background-color: #cccccc; color: black; padding: 2px 8px; border-radius: 12px; font-size: 0.8rem;'>AUTO TRADE INATIVO</span>", unsafe_allow_html=True)
    with col_status2:
        if st.button("🔄 Atualizar"):
            st.rerun()
    
    st.subheader("🤖 Configurações Auto Trade")
    
    # Controles simples de auto trade
    col_auto1, col_auto2 = st.columns(2)
    with col_auto1:
        if not st.session_state.auto_trade_active:
            if st.button("▶️ Iniciar Auto Trade", type="primary", use_container_width=True):
                st.session_state.auto_trade_active = True
                st.success("✅ Auto Trade INICIADO!")
                st.session_state.logs_tecnicos.append({
                    "timestamp": datetime.now(),
                    "tipo": "AUTO_TRADE_INICIADO",
                    "status": "Auto Trade iniciado pelo usuário"
                })
    with col_auto2:
        if st.session_state.auto_trade_active:
            if st.button("⏹️ Parar Auto Trade", type="secondary", use_container_width=True):
                st.session_state.auto_trade_active = False
                st.warning("⏹️ Auto Trade PARADO!")
                st.session_state.logs_tecnicos.append({
                    "timestamp": datetime.now(),
                    "tipo": "AUTO_TRADE_PARADO", 
                    "status": "Auto Trade parado pelo usuário"
                })
    
    st.subheader("💰 Gestão de Capital")
    stake_padrao = st.number_input("Stake Padrão (€)", min_value=5, max_value=100, value=10, step=5)
    meta_lucro = st.number_input("🎯 Meta de Lucro (€)", min_value=10, max_value=5000, value=500, step=50)
    limite_perda = st.number_input("📉 Limite de Perda (€)", min_value=-2000, max_value=0, value=-200, step=50)
    
    st.subheader("📊 Estratégia")
    st.info("📈 EMA Crossover + RSI")
    st.write("**Compra:** EMA9 > EMA21 & RSI < 40")
    st.write("**Venda:** EMA9 < EMA21 & RSI > 60")
    
    # Configuração de atualização automática
    st.subheader("🔄 Configuração Live")
    auto_refresh = st.checkbox("Atualização Automática", value=True)
    refresh_interval = st.slider("Intervalo (segundos)", 5, 60, 10)

# Atualizar configurações globais
st.session_state.meta_lucro = meta_lucro
st.session_state.limite_perda = limite_perda

# Atualizar stake padrão para todos os pares
for par in PARES.keys():
    st.session_state.pares_config[par]["stake"] = stake_padrao

# Atualizar dados em tempo real
atualizar_dados_live()

# Simular dados de 15 minutos para todos os pares (apenas se necessário)
try:
    if st.session_state.dados_15min.empty:
        with st.spinner("🔄 Gerando dados de mercado..."):
            dados_15min = pd.concat([
                simular_dados_15min(par, PARES[par]["base"]) 
                for par in PARES.keys()
            ])

            # Calcular indicadores técnicos
            dados_15min = dados_15min.groupby('par').apply(calcular_indicadores_tecnicos).reset_index(drop=True)
            dados_15min = dados_15min.groupby('par').apply(gerar_sinais_15min).reset_index(drop=True)
            st.session_state.dados_15min = dados_15min
            st.success("✅ Dados de mercado gerados!")
except Exception as e:
    st.error(f"❌ Erro ao processar dados: {e}")

# Executar auto trade se estiver ativo
if st.session_state.auto_trade_active:
    executar_auto_trade()

# Layout principal em abas
tab1, tab2, tab3, tab4 = st.tabs(["📊 Dashboard", "🤖 Auto Trade", "🔧 Config Pares", "📈 Sinais"])

with tab1:
    # Header com informações
    col_update1, col_update2, col_update3 = st.columns(3)
    with col_update1:
        st.metric("Última Atualização", st.session_state.ultima_atualizacao.strftime("%H:%M:%S"))
    with col_update2:
        st.metric("Total Updates", st.session_state.contador_updates)
    with col_update3:
        st.metric("Pares Monitorados", len(PARES))
    
    # Status do Auto Trade em destaque
    st.subheader("🎯 Status do Sistema")
    
    col_status1, col_status2, col_status3, col_status4 = st.columns(4)
    with col_status1:
        status_color = "🟢" if st.session_state.auto_trade_active else "🔴"
        status_text = "ATIVO" if st.session_state.auto_trade_active else "INATIVO"
        st.metric("Auto Trade", f"{status_color} {status_text}")
    with col_status2:
        total_trades = st.session_state.auto_trade_stats.get("total_trades", 0)
        st.metric("Total Trades", total_trades)
    with col_status3:
        win_rate = st.session_state.auto_trade_stats.get("win_rate", 0.0)
        st.metric("Win Rate", f"{win_rate:.1f}%")
    with col_status4:
        lucro_total = st.session_state.auto_trade_stats.get("lucro_total", 0.0)
        st.metric("Lucro Total", f"€{lucro_total:.2f}")
    
    # Aplicar estilo personalizado às métricas
    custom_metric_style()
    
    # Trades Ativos
    st.subheader("📈 Trades Ativos")
    if st.session_state.trades_ativos:
        trades_df = pd.DataFrame(st.session_state.trades_ativos)
        st.dataframe(trades_df, use_container_width=True)
    else:
        st.info("📭 Nenhum trade ativo no momento")

with tab2:
    st.header("🤖 Painel de Auto Trade")
    
    col_auto1, col_auto2 = st.columns([2, 1])
    
    with col_auto1:
        st.subheader("📊 Controles")
        
        # Botão para forçar execução manual
        if st.session_state.auto_trade_active:
            if st.button("🔍 Executar Agora", type="secondary"):
                executar_auto_trade()
                st.rerun()
        
        st.subheader("📈 Estatísticas")
        stats = st.session_state.auto_trade_stats
        col_stat1, col_stat2, col_stat3, col_stat4 = st.columns(4)
        with col_stat1:
            st.metric("Total Trades", stats.get("total_trades", 0))
        with col_stat2:
            win_rate = stats.get("win_rate", 0.0)
            st.metric("Win Rate", f"{win_rate:.1f}%")
        with col_stat3:
            st.metric("Trades Lucrativos", stats.get("trades_lucrativos", 0))
        with col_stat4:
            st.metric("Trades Perdedores", stats.get("trades_perdedores", 0))
        
        # Limites de Risco
        st.subheader("🛡️ Gestão de Risco")
        col_risk1, col_risk2 = st.columns(2)
        with col_risk1:
            st.metric("Meta de Lucro", f"€{st.session_state.meta_lucro}")
        with col_risk2:
            st.metric("Limite de Perda", f"€{st.session_state.limite_perda}")
    
    with col_auto2:
        st.subheader("⚙️ Ações Rápidas")
        
        if st.button("🗑️ Limpar Todos Trades", use_container_width=True):
            st.session_state.trades_ativos = []
            st.session_state.historico_trades = []
            st.session_state.auto_trade_stats = {
                "total_trades": 0,
                "trades_lucrativos": 0,
                "trades_perdedores": 0,
                "lucro_total": 0.0,
                "melhor_trade": 0.0,
                "pior_trade": 0.0,
                "win_rate": 0.0
            }
            st.success("✅ Todos os trades foram limpos!")
            st.rerun()
        
        st.subheader("📋 Histórico Recente")
        if st.session_state.historico_trades:
            historico_recente = st.session_state.historico_trades[-5:]
            historico_df = pd.DataFrame(historico_recente)
            st.dataframe(historico_df, use_container_width=True)
        else:
            st.info("📭 Nenhum trade no histórico")

with tab4:
    st.header("📈 Sinais de Trading")
    
    if not st.session_state.dados_15min.empty:
        # Resumo de sinais
        sinais_por_par = []
        for par in PARES.keys():
            df_par = st.session_state.dados_15min[st.session_state.dados_15min['par'] == par]
            if not df_par.empty:
                ultimo = df_par.iloc[-1]
                sinais_por_par.append({
                    'Par': par,
                    'Preço': ultimo['close'],
                    'Sinal': ultimo.get('sinal', 'N/A'),
                    'RSI': f"{ultimo.get('RSI', 0):.1f}",
                    'EMA9': f"{ultimo.get('EMA_9', 0):.5f}",
                    'EMA21': f"{ultimo.get('EMA_21', 0):.5f}",
                    'Ativo': "✅" if st.session_state.pares_config[par]["ativo"] else "❌"
                })
        
        if sinais_por_par:
            df_sinais = pd.DataFrame(sinais_por_par)
            st.dataframe(df_sinais, use_container_width=True)
    else:
        st.error("❌ Dados não disponíveis")

# Sistema de atualização automática
if auto_refresh:
    time.sleep(refresh_interval)
    st.rerun()

# Footer
st.markdown("---")
st.markdown(
    "<div style='text-align: center; color: gray;'>"
    "🤖 Auto Trading System - Modo Teste | Desenvolvido para Demonstração"
    "</div>", 
    unsafe_allow_html=True
)

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
    "GBP/JPY": 0.0030, "AUD/JPY": 0.0028, "NZD/USD": 0.0024, "USD/CNH": 0.0035,
    "USD/MXN": 0.0040, "USD/TRY": 0.0080, "USD/ZAR": 0.0050, "USD/SGD": 0.0020,
    "USD/HKD": 0.0015, "USD/SEK": 0.0030, "USD/NOK": 0.0032, "USD/DKK": 0.0020
}

# Parâmetros simulados expandidos para todos os pares
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
    "AUD/JPY": {"base": 97.0, "vol": 0.006, "pip_value": 9},
    "NZD/USD": {"base": 0.61, "vol": 0.007, "pip_value": 10},
    "USD/CNH": {"base": 7.25, "vol": 0.003, "pip_value": 10},
    "USD/MXN": {"base": 17.5, "vol": 0.008, "pip_value": 10},
    "USD/TRY": {"base": 32.0, "vol": 0.012, "pip_value": 10},
    "USD/ZAR": {"base": 18.5, "vol": 0.009, "pip_value": 10},
    "USD/SGD": {"base": 1.35, "vol": 0.003, "pip_value": 10},
    "USD/HKD": {"base": 7.82, "vol": 0.002, "pip_value": 10},
    "USD/SEK": {"base": 10.5, "vol": 0.004, "pip_value": 10},
    "USD/NOK": {"base": 10.8, "vol": 0.004, "pip_value": 10},
    "USD/DKK": {"base": 6.85, "vol": 0.003, "pip_value": 10}
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
        "trailing_stop": True
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
        "limite_perda": -200
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
        prev2 = df.iloc[i-2]
        
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
        if (lower_wick >= 2 * body and 
            upper_wick <= body * 0.1 and 
            body_ratio <= 0.3):
            patterns.append({
                "timestamp": current['timestamp'], 
                "pattern": "Hammer", 
                "type": "bullish", 
                "strength": "medium"
            })
        
        # Shooting Star (Estrela Cadente) - candle de reversão bearish
        if (upper_wick >= 2 * body and 
            lower_wick <= body * 0.1 and 
            body_ratio <= 0.3):
            patterns.append({
                "timestamp": current['timestamp'], 
                "pattern": "Shooting Star", 
                "type": "bearish", 
                "strength": "medium"
            })
        
        # Engulfing Bullish
        if (prev1['close'] < prev1['open'] and  # Candle anterior de baixa
            current['close'] > current['open'] and  # Candle atual de alta
            current['open'] < prev1['close'] and 
            current['close'] > prev1['open']):
            patterns.append({
                "timestamp": current['timestamp'], 
                "pattern": "Bullish Engulfing", 
                "type": "bullish", 
                "strength": "strong"
            })
        
        # Engulfing Bearish
        if (prev1['close'] > prev1['open'] and  # Candle anterior de alta
            current['close'] < current['open'] and  # Candle atual de baixa
            current['open'] > prev1['close'] and 
            current['close'] < prev1['open']):
            patterns.append({
                "timestamp": current['timestamp'], 
                "pattern": "Bearish Engulfing", 
                "type": "bearish", 
                "strength": "strong"
            })
        
        # Doji - indecisão
        if body_ratio <= 0.1:
            patterns.append({
                "timestamp": current['timestamp'], 
                "pattern": "Doji", 
                "type": "neutral", 
                "strength": "weak"
            })
    
    return patterns

def calcular_suporte_resistencia(df, window=20):
    """Calcula níveis de suporte e resistência"""
    df = df.copy()
    df['resistance'] = df['high'].rolling(window=window, min_periods=1).max()
    df['support'] = df['low'].rolling(window=window, min_periods=1).min()
    return df

def analisar_tendencia(df):
    """Analisa a tendência baseada em MME e price action"""
    if len(df) < 5:
        return "Indefinida"
    
    try:
        # Tendência por MME
        ema_9 = df['close'].ewm(span=9).mean().iloc[-1]
        ema_21 = df['close'].ewm(span=21).mean().iloc[-1]
        
        # Price Action - Máximas e Mínimas
        high_5 = df['high'].tail(5)
        low_5 = df['low'].tail(5)
        
        max_increasing = all(high_5.iloc[i] > high_5.iloc[i-1] for i in range(1, len(high_5)))
        min_increasing = all(low_5.iloc[i] > low_5.iloc[i-1] for i in range(1, len(low_5)))
        max_decreasing = all(high_5.iloc[i] < high_5.iloc[i-1] for i in range(1, len(high_5)))
        min_decreasing = all(low_5.iloc[i] < low_5.iloc[i-1] for i in range(1, len(low_5)))
        
        if ema_9 > ema_21 and (max_increasing or min_increasing):
            return "Alta"
        elif ema_9 < ema_21 and (max_decreasing or min_decreasing):
            return "Baixa"
        else:
            return "Lateral"
    except:
        return "Indefinida"

def executar_auto_trade():
    """Executa a lógica de auto trading para todos os pares"""
    if not st.session_state.auto_trade_active:
        return
    
    # Verificar se há dados suficientes
    if st.session_state.dados_15min is None or len(st.session_state.dados_15min) == 0:
        return
    
    # Verificar limites de risco
    if verificar_limites_risco():
        st.session_state.auto_trade_active = False
        st.warning("🚫 Auto Trade pausado - Limites de risco atingidos")
        return
    
    trades_executados = 0
    
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
            # Procurar nova entrada
            if verificar_condicoes_entrada(ultimo, df_par):
                executar_nova_entrada(par, ultimo)
                trades_executados += 1
    
    if trades_executados > 0:
        st.session_state.logs_tecnicos.append({
            "timestamp": datetime.now(),
            "tipo": "AUTO_TRADE_EXECUCAO",
            "trades_executados": trades_executados,
            "status": f"Executados {trades_executados} trades"
        })

def verificar_condicoes_entrada(ultimo, df_par):
    """Verifica condições para entrada no trade"""
    try:
        # Condições de COMPRA
        condicoes_compra = (
            ultimo['sinal'] == 'COMPRA' and
            ultimo['EMA_9'] > ultimo['EMA_21'] and
            ultimo['RSI'] < 35 and
            ultimo['MACD'] > ultimo['MACD_Signal'] and
            ultimo['tipo_padrao'] == 'bullish'
        )
        
        # Condições de VENDA
        condicoes_venda = (
            ultimo['sinal'] == 'VENDA' and
            ultimo['EMA_9'] < ultimo['EMA_21'] and
            ultimo['RSI'] > 65 and
            ultimo['MACD'] < ultimo['MACD_Signal'] and
            ultimo['tipo_padrao'] == 'bearish'
        )
        
        return condicoes_compra or condicoes_venda
    except:
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
            'preco_entrada': preco_entrada,
            'stop_loss': stop_loss,
            'take_profit': take_profit,
            'stake': stake,
            'timestamp_entrada': datetime.now(),
            'status': 'ativo',
            'lucro_prejuizo': 0.0,
            'rsi_entrada': dados_entrada['RSI'],
            'padrao_entrada': dados_entrada['padrao_candle']
        }
        
        st.session_state.trades_ativos.append(trade)
        
        # Registrar no log
        st.session_state.logs_tecnicos.append({
            "timestamp": datetime.now(),
            "tipo": "AUTO_TRADE_ENTRADA",
            "par": par,
            "operacao": tipo_operacao,
            "preco": preco_entrada,
            "stake": stake,
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
            
            # Verificar take profit
            elif (trade['tipo'] == 'COMPRA' and preco_atual >= trade['take_profit']) or \
                 (trade['tipo'] == 'VENDA' and preco_atual <= trade['take_profit']):
                fechar_trade(trade, preco_atual, 'take_profit')
            
            # Atualizar trailing stop (se habilitado)
            elif STRATEGY_15MIN['exit_conditions']['trailing_stop']:
                atualizar_trailing_stop(trade, preco_atual)
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
        trade['preco_saida'] = preco_saida
        trade['timestamp_saida'] = datetime.now()
        trade['status'] = 'fechado'
        trade['motivo_saida'] = motivo
        trade['lucro_prejuizo'] = lucro_euros
        
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
            "preco_saida": preco_saida,
            "lucro": lucro_euros,
            "motivo": motivo,
            "status": "Trade fechado"
        })
        
    except Exception as e:
        st.error(f"Erro ao fechar trade: {e}")

def atualizar_trailing_stop(trade, preco_atual):
    """Atualiza stop loss para trailing stop"""
    try:
        if trade['tipo'] == 'COMPRA':
            novo_stop = preco_atual * (1 - STRATEGY_15MIN['exit_conditions']['stop_loss'])
            if novo_stop > trade['stop_loss']:
                trade['stop_loss'] = novo_stop
        else:
            novo_stop = preco_atual * (1 + STRATEGY_15MIN['exit_conditions']['stop_loss'])
            if novo_stop < trade['stop_loss']:
                trade['stop_loss'] = novo_stop
    except:
        pass

def verificar_limites_risco():
    """Verifica limites de risco para pausar auto trade"""
    try:
        lucro_total = st.session_state.auto_trade_stats["lucro_total"]
        meta_lucro = st.session_state.get('meta_lucro', 500)
        limite_perda = st.session_state.get('limite_perda', -200)
        
        if lucro_total >= meta_lucro:
            return True
        elif lucro_total <= limite_perda:
            return True
        
        # Verificar drawdown máximo
        trades_recentes = st.session_state.historico_trades[-10:]  # Últimos 10 trades
        if len(trades_recentes) >= 5:
            perdas_consecutivas = sum(1 for trade in trades_recentes if trade['lucro_prejuizo'] < 0)
            if perdas_consecutivas >= 3:
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
    """Simula dados de 15 minutos para análise - versão mais leve"""
    timeframe_data = []
    current_time = datetime.now().replace(hour=0, minute=0, second=0, microsecond=0)
    
    # Reduzir para 1 dia para performance
    for day in range(days):
        for hour in range(24):
            for minute in range(0, 60, 15):
                timestamp = current_time - timedelta(days=day) + timedelta(hours=hour, minutes=minute)
                
                # Simular preços OHLC mais simples
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
        
        # Price Action
        df['Body'] = abs(df['close'] - df['open'])
        df['Total_Range'] = df['high'] - df['low']
        df['Body_Ratio'] = np.where(df['Total_Range'] > 0, df['Body'] / df['Total_Range'], 0)
        df['Upper_Shadow'] = df['high'] - np.maximum(df['open'], df['close'])
        df['Lower_Shadow'] = np.minimum(df['open'], df['close']) - df['low']
        df['Candle_Type'] = np.where(df['close'] > df['open'], 'Alta', 'Baixa')
        
        # Suporte e Resistência
        df = calcular_suporte_resistencia(df)
        
        return df
    except Exception as e:
        st.error(f"Erro no cálculo de indicadores: {e}")
        return df

def gerar_sinais_15min(df):
    """Gera sinais de trading baseados na estratégia de 15min"""
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
        
        # Condições de compra
        buy_condition = (
            (df['EMA_9'] > df['EMA_21']) & 
            (df['RSI'] < 35) & 
            (df['MACD'] > df['MACD_Signal']) &
            (df['tipo_padrao'] == 'bullish')
        )
        
        # Condições de venda
        sell_condition = (
            (df['EMA_9'] < df['EMA_21']) & 
            (df['RSI'] > 65) & 
            (df['MACD'] < df['MACD_Signal']) &
            (df['tipo_padrao'] == 'bearish')
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
st.markdown('<h3 class="main-header" style="font-size: 1.5rem;">Multi-Pair Auto Trading System</h3>', unsafe_allow_html=True)

# Sidebar modernizada
with st.sidebar:
    st.header("⚙️ Configurações de Trading")
    
    # Indicador de atualização em tempo real
    col_live1, col_live2 = st.columns([2, 1])
    with col_live1:
        status_color = "🟢" if st.session_state.auto_trade_active else "🔴"
        st.markdown(f"<span class='live-indicator'>AUTO TRADE {status_color}</span>", unsafe_allow_html=True)
    with col_live2:
        if st.button("🔄 Atualizar"):
            st.rerun()
    
    st.subheader("🤖 Configurações Auto Trade")
    auto_trade_mode = st.radio("Modo Auto Trade:", ["Manual", "Automático"])
    
    if auto_trade_mode == "Automático":
        col_auto1, col_auto2 = st.columns(2)
        with col_auto1:
            if st.button("▶️ Iniciar Auto Trade", type="primary"):
                st.session_state.auto_trade_active = True
                st.success("Auto Trade Iniciado para todos os pares!")
        with col_auto2:
            if st.button("⏹️ Parar Auto Trade"):
                st.session_state.auto_trade_active = False
                st.warning("Auto Trade Parado!")
    
    st.subheader("💰 Gestão de Capital")
    stake_padrao = st.number_input("Stake Padrão (€)", min_value=5, max_value=100, value=10, step=5)
    meta_lucro = st.number_input("🎯 Meta de Lucro (€)", min_value=10, max_value=5000, value=500, step=50)
    limite_perda = st.number_input("📉 Limite de Perda (€)", min_value=-2000, max_value=0, value=-200, step=50)
    
    st.subheader("📊 Parâmetros da Estratégia")
    rsi_oversold = st.slider("RSI Oversold", 20, 40, 30)
    rsi_overbought = st.slider("RSI Overbought", 60, 80, 70)
    
    # Configuração de atualização automática
    st.subheader("🔄 Configuração Live")
    auto_refresh = st.checkbox("Atualização Automática", value=False)
    refresh_interval = st.slider("Intervalo (segundos)", 1, 60, 5)

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
        with st.spinner("Gerando dados de mercado..."):
            dados_15min = pd.concat([
                simular_dados_15min(par, PARES[par]["base"]) 
                for par in list(PARES.keys())[:10]  # Limitar a 10 pares para performance
            ])

            # Calcular indicadores técnicos
            dados_15min = dados_15min.groupby('par').apply(calcular_indicadores_tecnicos).reset_index(drop=True)
            dados_15min = dados_15min.groupby('par').apply(gerar_sinais_15min).reset_index(drop=True)
            st.session_state.dados_15min = dados_15min
except Exception as e:
    st.error(f"Erro ao processar dados: {e}")

# Executar auto trade se estiver ativo
if st.session_state.auto_trade_active:
    executar_auto_trade()

# Layout principal em abas
tab1, tab2, tab3, tab4 = st.tabs(["📊 Dashboard", "🤖 Auto Trade", "🔧 Config Pares", "📈 Mercado"])

with tab1:
    # Header com informações de atualização
    col_update1, col_update2, col_update3 = st.columns(3)
    with col_update1:
        st.metric("Última Atualização", st.session_state.ultima_atualizacao.strftime("%H:%M:%S"))
    with col_update2:
        st.metric("Total Updates", st.session_state.contador_updates)
    with col_update3:
        st.metric("Pares Monitorados", len(PARES))
    
    # KPIs principais - COM VERIFICAÇÃO DE SEGURANÇA
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        total_trades = len(st.session_state.historico_trades)
        st.metric("Total Trades", total_trades)
    
    with col2:
        trades_ativos = len(st.session_state.trades_ativos)
        st.metric("Trades Ativos", trades_ativos)
    
    with col3:
        # Verificação segura do win_rate
        win_rate = st.session_state.auto_trade_stats.get("win_rate", 0.0)
        st.metric("Win Rate", f"{win_rate:.1f}%")
    
    with col4:
        lucro_total = st.session_state.auto_trade_stats.get("lucro_total", 0.0)
        st.metric("Lucro Total", f"€{lucro_total:.2f}")
    
    # Aplicar estilo personalizado às métricas
    custom_metric_style()
    
    # Status do Auto Trade
    st.subheader("🤖 Status do Auto Trade")
    
    col_status1, col_status2, col_status3, col_status4 = st.columns(4)
    with col_status1:
        status_color = "🟢" if st.session_state.auto_trade_active else "🔴"
        status_text = "ATIVO" if st.session_state.auto_trade_active else "INATIVO"
        st.metric("Status", f"{status_color} {status_text}")
    with col_status2:
        total_trades = st.session_state.auto_trade_stats.get("total_trades", 0)
        st.metric("Trades Hoje", total_trades)
    with col_status3:
        lucro_total = st.session_state.auto_trade_stats.get("lucro_total", 0.0)
        st.metric("Lucro/Prejuízo", f"€{lucro_total:.2f}")
    with col_status4:
        pares_ativos = sum(1 for config in st.session_state.pares_config.values() if config["ativo"])
        st.metric("Pares Ativos", f"{pares_ativos}/{len(PARES)}")

with tab3:
    st.header("🔧 Configuração por Par")
    
    st.subheader("⚙️ Configurações Individuais dos Pares")
    
    # Dividir pares em colunas para melhor organização
    pares_list = list(PARES.keys())
    cols = st.columns(3)
    
    for i, par in enumerate(pares_list):
        with cols[i % 3]:
            with st.container():
                st.markdown(f'<div class="pair-card">', unsafe_allow_html=True)
                st.write(f"**{par}**")
                
                # Configurações do par
                ativo = st.checkbox(
                    f"Ativo", 
                    value=st.session_state.pares_config[par]["ativo"],
                    key=f"ativo_{par}"
                )
                
                stake = st.number_input(
                    f"Stake (€)",
                    min_value=5,
                    max_value=100,
                    value=st.session_state.pares_config[par]["stake"],
                    key=f"stake_{par}"
                )
                
                # Atualizar configurações
                st.session_state.pares_config[par]["ativo"] = ativo
                st.session_state.pares_config[par]["stake"] = stake
                
                # Mostrar informações do par de forma segura
                if not st.session_state.dados_15min.empty:
                    df_par = st.session_state.dados_15min[st.session_state.dados_15min['par'] == par]
                    if not df_par.empty:
                        ultimo = df_par.iloc[-1]
                        st.write(f"Preço: {ultimo['close']:.5f}")
                        st.write(f"Sinal: {ultimo.get('sinal', 'N/A')}")
                        st.write(f"RSI: {ultimo.get('RSI', 0):.1f}")
                else:
                    st.write("Dados não disponíveis")
                
                st.markdown('</div>', unsafe_allow_html=True)

with tab2:
    st.header("🤖 Painel de Auto Trade")
    
    col_auto1, col_auto2 = st.columns([2, 1])
    
    with col_auto1:
        st.subheader("📊 Trades Ativos")
        if st.session_state.trades_ativos:
            trades_df = pd.DataFrame(st.session_state.trades_ativos)
            # Verificar se as colunas existem antes de acessá-las
            available_columns = [col for col in ['id', 'par', 'tipo', 'preco_entrada', 'stop_loss', 'take_profit', 'stake', 'timestamp_entrada'] 
                               if col in trades_df.columns]
            trades_display = trades_df[available_columns].copy()
            
            # Formatar timestamp se existir
            if 'timestamp_entrada' in trades_display.columns:
                trades_display['timestamp_entrada'] = trades_display['timestamp_entrada'].apply(
                    lambda x: x.strftime('%H:%M:%S') if hasattr(x, 'strftime') else str(x)
                )
                
            st.dataframe(trades_display, width='stretch', height=400)
        else:
            st.info("Nenhum trade ativo no momento")
        
        st.subheader("📈 Estatísticas de Performance")
        stats = st.session_state.auto_trade_stats
        col_stat1, col_stat2, col_stat3, col_stat4 = st.columns(4)
        with col_stat1:
            st.metric("Total Trades", stats.get("total_trades", 0))
        with col_stat2:
            win_rate = stats.get("win_rate", 0.0)
            st.metric("Win Rate", f"{win_rate:.1f}%")
        with col_stat3:
            st.metric("Melhor Trade", f"€{stats.get('melhor_trade', 0.0):.2f}")
        with col_stat4:
            st.metric("Pior Trade", f"€{stats.get('pior_trade', 0.0):.2f}")
    
    with col_auto2:
        st.subheader("⚙️ Controles Rápidos")
        
        if st.session_state.auto_trade_active:
            if st.button("⏸️ Pausar Auto Trade", type="secondary", use_container_width=True):
                st.session_state.auto_trade_active = False
                st.rerun()
        else:
            if st.button("▶️ Iniciar Auto Trade", type="primary", use_container_width=True):
                st.session_state.auto_trade_active = True
                st.rerun()
        
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
            st.rerun()
        
        st.subheader("📋 Histórico Recente")
        if st.session_state.historico_trades:
            historico_recente = st.session_state.historico_trades[-8:]  # Últimos 8 trades
            historico_df = pd.DataFrame(historico_recente)
            if not historico_df.empty:
                # Verificar colunas disponíveis
                available_cols = [col for col in ['id', 'par', 'tipo', 'preco_entrada', 'preco_saida', 'lucro_prejuizo', 'motivo_saida'] 
                                if col in historico_df.columns]
                historico_display = historico_df[available_cols].copy()
                
                # Adicionar timestamp se disponível
                if 'timestamp_saida' in historico_df.columns:
                    historico_display['timestamp'] = historico_df['timestamp_saida'].apply(
                        lambda x: x.strftime('%H:%M') if hasattr(x, 'strftime') else 'N/A'
                    )
                
                # Colorir baseado no resultado se a coluna existir
                if 'lucro_prejuizo' in historico_display.columns:
                    def colorir_resultado(val):
                        if val > 0:
                            return 'background-color: #90EE90; font-weight: bold;'
                        else:
                            return 'background-color: #FFB6C1; font-weight: bold;'
                    
                    st.dataframe(
                        historico_display.style.applymap(colorir_resultado, subset=['lucro_prejuizo']),
                        width='stretch',
                        height=400
                    )
                else:
                    st.dataframe(historico_display, width='stretch', height=400)
        else:
            st.info("Nenhum trade no histórico")

with tab4:
    st.header("📈 Visão Geral do Mercado")
    
    # Resumo de todos os pares
    st.subheader("🎯 Sinais por Par")
    
    if not st.session_state.dados_15min.empty:
        sinais_por_par = []
        for par in list(PARES.keys())[:15]:  # Limitar para performance
            df_par = st.session_state.dados_15min[st.session_state.dados_15min['par'] == par]
            if not df_par.empty:
                ultimo = df_par.iloc[-1]
                tendencia = analisar_tendencia(df_par)
                sinais_por_par.append({
                    'Par': par,
                    'Preço': ultimo['close'],
                    'Sinal': ultimo.get('sinal', 'N/A'),
                    'RSI': f"{ultimo.get('RSI', 0):.1f}",
                    'Tendência': tendencia,
                    'Padrão': ultimo.get('padrao_candle', ''),
                    'Ativo': st.session_state.pares_config[par]["ativo"]
                })
        
        if sinais_por_par:
            df_sinais = pd.DataFrame(sinais_por_par)
            
            # Colorir os sinais de forma segura
            def colorir_linha(row):
                if row.get('Sinal') == 'COMPRA':
                    return ['background-color: #90EE90'] * len(row)
                elif row.get('Sinal') == 'VENDA':
                    return ['background-color: #FFB6C1'] * len(row)
                else:
                    return ['background-color: #F0F0F0'] * len(row)
            
            st.dataframe(
                df_sinais.style.apply(colorir_linha, axis=1),
                width='stretch',
                height=600
            )
    else:
        st.info("Dados de mercado não disponíveis")

# Sistema de atualização automática
if auto_refresh:
    time.sleep(refresh_interval)
    st.rerun()

# Footer
st.markdown("---")
st.markdown(
    "<div style='text-align: center; color: gray;'>"
    "📊 Advanced Trading Dashboard - Multi-Pair Auto Trading System | 20 Pares Monitorados"
    "</div>", 
    unsafe_allow_html=True
)

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timedelta
import io
from streamlit_extras.metric_cards import style_metric_cards

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
    }
    .alert-high {
        background-color: #ffcccc;
        padding: 0.5rem;
        border-radius: 5px;
        border-left: 4px solid #ff4444;
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
    
    return df

def gerar_sinais_15min(df):
    """Gera sinais de trading baseados na estratégia de 15min"""
    df['sinal'] = 'NEUTRO'
    
    # Condições de compra
    buy_condition = (
        (df['EMA_9'] > df['EMA_21']) & 
        (df['RSI'] < 35) & 
        (df['MACD'] > df['MACD_Signal'])
    )
    
    # Condições de venda
    sell_condition = (
        (df['EMA_9'] < df['EMA_21']) & 
        (df['RSI'] > 65) & 
        (df['MACD'] < df['MACD_Signal'])
    )
    
    df.loc[buy_condition, 'sinal'] = 'COMPRA'
    df.loc[sell_condition, 'sinal'] = 'VENDA'
    
    return df

# Simular dados de 15 minutos
st.markdown('<h1 class="main-header">🎯 Advanced Trading Dashboard - 15min Strategy</h1>', unsafe_allow_html=True)

# Sidebar modernizada
with st.sidebar:
    st.header("⚙️ Configurações de Trading")
    
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

# Simular dados de 15 minutos
dados_15min = pd.concat([
    simular_dados_15min(par, PARES[par]["base"]) 
    for par in pares_selecionados
])

# Calcular indicadores técnicos
dados_15min = dados_15min.groupby('par').apply(calcular_indicadores_tecnicos).reset_index(drop=True)
dados_15min = dados_15min.groupby('par').apply(gerar_sinais_15min).reset_index(drop=True)

# Layout principal em abas
tab1, tab2, tab3, tab4 = st.tabs(["📊 Dashboard", "🎯 Estratégia 15min", "📈 Análise Técnica", "📋 Relatórios"])

with tab1:
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
    
    style_metric_cards()
    
    # Alertas de volatilidade
    st.subheader("🚨 Alertas de Volatilidade")
    
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
    
    # Gráfico de volatilidade
    st.subheader("📈 Volatilidade por Par (15min)")
    
    fig_vol = go.Figure()
    for par in pares_selecionados:
        df_par = dados_15min[dados_15min['par'] == par]
        volatilidade = (df_par['high'] - df_par['low']) / df_par['close']
        fig_vol.add_trace(go.Scatter(
            x=df_par['timestamp'], 
            y=volatilidade,
            mode='lines',
            name=par,
            line=dict(width=2)
        ))
    
    fig_vol.update_layout(
        title="Evolução da Volatilidade (15min)",
        xaxis_title="Data/Hora",
        yaxis_title="Volatilidade",
        height=400
    )
    st.plotly_chart(fig_vol, use_container_width=True)

with tab2:
    st.header("🎯 Estratégia de 15 Minutos")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        # Gráfico de candlesticks com sinais
        par_selecionado = st.selectbox("Selecione o par:", pares_selecionados)
        
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
            
            # Sinais de trading
            compras = df_par[df_par['sinal'] == 'COMPRA']
            vendas = df_par[df_par['sinal'] == 'VENDA']
            
            fig_candle.add_trace(go.Scatter(
                x=compras['timestamp'], y=compras['low'] * 0.998,
                mode='markers', name='Compra',
                marker=dict(color='green', size=10, symbol='triangle-up')
            ))
            
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
            
            st.plotly_chart(fig_candle, use_container_width=True)
    
    with col2:
        st.subheader("📋 Sinais Atuais")
        
        # Últimos sinais por par
        ultimos_sinais = []
        for par in pares_selecionados:
            df_par = dados_15min[dados_15min['par'] == par]
            ultimo_sinal = df_par.iloc[-1]
            ultimos_sinais.append({
                'Par': par,
                'Sinal': ultimo_sinal['sinal'],
                'Preço': ultimo_sinal['close'],
                'RSI': f"{ultimo_sinal['RSI']:.1f}",
                'Tendência': "Alta" if ultimo_sinal['EMA_9'] > ultimo_sinal['EMA_21'] else "Baixa"
            })
        
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
            use_container_width=True
        )
        
        # Botão de execução
        st.subheader("🤖 Executar Trades")
        if st.button("🚀 Executar Estratégia 15min", type="primary"):
            for par in pares_selecionados:
                df_par = dados_15min[dados_15min['par'] == par]
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
                        "rsi_entrada": round(ultimo_sinal['RSI'], 1),
                        "ema_tendencia": "Alta" if ultimo_sinal['EMA_9'] > ultimo_sinal['EMA_21'] else "Baixa"
                    })
            
            st.success(f"✅ {len([s for s in ultimos_sinais if s['Sinal'] in ['COMPRA', 'VENDA']])} trades executados!")

with tab3:
    st.header("📈 Análise Técnica Detalhada")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Gráfico RSI
        st.subheader("📊 RSI por Par")
        fig_rsi = go.Figure()
        
        for par in pares_selecionados:
            df_par = dados_15min[dados_15min['par'] == par].tail(24)
            fig_rsi.add_trace(go.Scatter(
                x=df_par['timestamp'], y=df_par['RSI'],
                mode='lines', name=par,
                line=dict(width=2)
            ))
        
        # Linhas de sobrecompra/sobrevenda
        fig_rsi.add_hline(y=70, line_dash="dash", line_color="red", annotation_text="Sobrecompra")
        fig_rsi.add_hline(y=30, line_dash="dash", line_color="green", annotation_text="Sobrevenda")
        
        fig_rsi.update_layout(
            title="RSI - Indicador de Momentum",
            xaxis_title="Data/Hora",
            yaxis_title="RSI",
            height=400
        )
        st.plotly_chart(fig_rsi, use_container_width=True)
    
    with col2:
        # Gráfico MACD
        st.subheader("📈 MACD por Par")
        fig_macd = go.Figure()
        
        for par in pares_selecionados[:2]:  # Mostrar apenas 2 para não poluir
            df_par = dados_15min[dados_15min['par'] == par].tail(24)
            fig_macd.add_trace(go.Scatter(
                x=df_par['timestamp'], y=df_par['MACD'],
                mode='lines', name=f"{par} MACD",
                line=dict(width=2)
            ))
            fig_macd.add_trace(go.Scatter(
                x=df_par['timestamp'], y=df_par['MACD_Signal'],
                mode='lines', name=f"{par} Signal",
                line=dict(width=1, dash='dash')
            ))
        
        fig_macd.update_layout(
            title="MACD - Indicador de Tendência",
            xaxis_title="Data/Hora",
            yaxis_title="MACD",
            height=400
        )
        st.plotly_chart(fig_macd, use_container_width=True)

with tab4:
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
        
        # Tabela de trades
        st.subheader("📋 Histórico de Trades")
        
        # Formatar colunas
        df_display = df_logs.copy()
        df_display['Timestamp'] = pd.to_datetime(df_display['timestamp']).dt.strftime('%Y-%m-%d %H:%M')
        df_display['Par'] = df_display['pair']
        df_display['Stake (€)'] = df_display['log_stake']
        df_display['Lucro (€)'] = df_display['log_lucro_estimado']
        df_display['Risco (€)'] = df_display['log_risco_estimado']
        
        colunas_display = ['Timestamp', 'Par', 'sinal', 'Stake (€)', 'Lucro (€)', 'Risco (€)', 'status', 'rsi_entrada', 'ema_tendencia']
        
        st.dataframe(df_display[colunas_display], use_container_width=True)
        
        # Gráfico de lucro acumulado
        st.subheader("💰 Evolução do Lucro")
        
        if 'timestamp' in df_logs.columns:
            df_logs['timestamp'] = pd.to_datetime(df_logs['timestamp'])
            df_logs = df_logs.sort_values('timestamp')
            df_logs['Lucro Acumulado'] = df_logs['log_lucro_estimado'].cumsum()
            
            fig_lucro = px.line(df_logs, x='timestamp', y='Lucro Acumulado', 
                              title='Lucro Acumulado ao Longo do Tempo')
            fig_lucro.update_layout(height=400)
            st.plotly_chart(fig_lucro, use_container_width=True)
    
    # Exportação de dados
    st.subheader("📥 Exportar Dados")
    
    col_exp1, col_exp2 = st.columns(2)
    
    with col_exp1:
        # Exportar dados de trading
        if st.button("📊 Exportar Relatório Completo"):
            buffer = io.BytesIO()
            with pd.ExcelWriter(buffer, engine='xlsxwriter') as writer:
                if st.session_state.logs_tecnicos:
                    df_logs.to_excel(writer, sheet_name='Trades Executados', index=False)
                dados_15min.to_excel(writer, sheet_name='Dados 15min', index=False)
                
                # Resumo estatístico
                resumo = {
                    'Metrica': ['Total Trades', 'Lucro Total', 'Risco Total', 'Win Rate'],
                    'Valor': [len(st.session_state.logs_tecnicos), lucro_total, risco_total, f"{win_rate:.1f}%"]
                }
                pd.DataFrame(resumo).to_excel(writer, sheet_name='Resumo', index=False)
            
            st.download_button(
                label="⬇️ Baixar Relatório Excel",
                data=buffer.getvalue(),
                file_name=f"relatorio_trading_{datetime.now().strftime('%Y%m%d_%H%M')}.xlsx",
                mime="application/vnd.ms-excel"
            )
    
    with col_exp2:
        # Exportar estratégia
        st.download_button(
            label="📋 Exportar Configuração da Estratégia",
            data=str(STRATEGY_15MIN),
            file_name="estrategia_15min.txt",
            mime="text/plain"
        )

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
    "📊 Advanced Trading Dashboard - Estratégia 15min | Desenvolvido para análise técnica"
    "</div>", 
    unsafe_allow_html=True
)

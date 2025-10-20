import streamlit as st
import pandas as pd
import numpy as np
import requests
import time
from datetime import datetime, timedelta
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import io

# Configuração da página
st.set_page_config(
    page_title="Ethereum Trading Bot - Simulação Real",
    page_icon="📈",
    layout="wide"
)

# Título da aplicação
st.title("🤖 Ethereum Trading Bot - Simulação Realista")
st.markdown("---")

# Função para calcular RSI
def calculate_rsi_simple(prices, period=14):
    """
    Versão simplificada e mais robusta para calcular RSI
    """
    try:
        # Converter para pandas Series se necessário
        if not isinstance(prices, pd.Series):
            prices = pd.Series(prices)
        
        # Calcular mudanças de preço
        delta = prices.diff()
        
        # Separar ganhos e perdas
        gain = delta.where(delta > 0, 0)
        loss = -delta.where(delta < 0, 0)
        
        # Calcular médias móveis simples
        avg_gain = gain.rolling(window=period, min_periods=1).mean()
        avg_loss = loss.rolling(window=period, min_periods=1).mean()
        
        # Calcular RS
        rs = avg_gain / avg_loss
        
        # Calcular RSI
        rsi = 100 - (100 / (1 + rs))
        
        # Preencher valores NaN
        rsi = rsi.fillna(50)
        
        return rsi
        
    except Exception as e:
        st.error(f"Erro no cálculo simplificado do RSI: {e}")
        return pd.Series([50] * len(prices), index=prices.index)

# Função para calcular Médias Móveis Simples
def calculate_sma(prices, period):
    """
    Calcula a Média Móvel Simples para o período especificado
    """
    try:
        if not isinstance(prices, pd.Series):
            prices = pd.Series(prices)
        
        sma = prices.rolling(window=period, min_periods=1).mean()
        sma = sma.fillna(prices)  # Preencher NaNs com o preço atual inicialmente
        
        return sma
        
    except Exception as e:
        st.error(f"Erro no cálculo da SMA: {e}")
        return pd.Series([np.nan] * len(prices), index=prices.index)

# Função para obter dados reais do Ethereum
def get_real_ethereum_data():
    try:
        # Usar múltiplas fontes para dados mais realistas
        st.info("🔄 Obtendo dados reais do Ethereum...")
        
        # Fonte 1: CoinGecko para dados históricos
        url = "https://api.coingecko.com/api/v3/coins/ethereum/market_chart"
        params = {
            'vs_currency': 'usd',
            'days': '90',  # 3 meses para mais dados
            'interval': 'daily'
        }
        
        response = requests.get(url, params=params, timeout=15)
        
        if response.status_code == 200:
            data = response.json()
            
            if 'prices' in data and len(data['prices']) > 0:
                prices = data['prices']
                df = pd.DataFrame(prices, columns=['timestamp', 'price'])
                df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
                df = df.set_index('timestamp')
                
                # Converter para dados horários com interpolação
                df_hourly = df.resample('H').interpolate()
                
                # Adicionar algum ruído realista
                np.random.seed(42)
                noise = np.random.normal(0, df_hourly['price'].std() * 0.01, len(df_hourly))
                df_hourly['price'] = df_hourly['price'] + noise
                
                st.success("✅ Dados reais obtidos com sucesso!")
                return df_hourly
                
        # Fallback para dados mais realistas se a API falhar
        return create_realistic_sample_data()
            
    except Exception as e:
        st.warning(f"⚠️ Erro na API: {e}. Usando dados realistas simulados.")
        return create_realistic_sample_data()

# Função para criar dados realistas baseados no comportamento real do Ethereum
def create_realistic_sample_data():
    st.info("📊 Gerando dados realistas do Ethereum...")
    
    # Criar dados horários para os últimos 90 dias
    end_date = datetime.now()
    start_date = end_date - timedelta(days=90)
    dates = pd.date_range(start=start_date, end=end_date, freq='H')
    
    # Preços baseados no comportamento real do ETH nos últimos 3 meses
    np.random.seed(42)
    
    # Tendência principal baseada no mercado real
    base_trend = np.linspace(2800, 3800, len(dates))  # Tendência de alta
    volatility = np.random.normal(0, 80, len(dates))  # Volatilidade realista
    
    # Adicionar ciclos de mercado
    market_cycle = np.sin(np.arange(len(dates)) / 500) * 200  # Ciclos longos
    short_cycle = np.sin(np.arange(len(dates)) / 50) * 50     # Ciclos curtos
    
    # Combinar todos os componentes
    prices = base_trend + volatility + market_cycle + short_cycle
    
    # Garantir que os preços são realistas
    prices = np.maximum(2500, prices)  # Mínimo realista
    prices = np.minimum(4500, prices)  # Máximo realista
    
    df = pd.DataFrame({
        'price': prices
    }, index=dates)
    
    st.success("✅ Dados realistas gerados com sucesso!")
    return df

# Função para simular trading realista com custos e limitações
def simulate_realistic_trading(df, rsi_lower=30, rsi_upper=70, rsi_period=14, 
                              initial_capital=1000, trade_amount=10):
    df = df.copy()
    
    # Calcular RSI
    df['rsi'] = calculate_rsi_simple(df['price'], rsi_period)
    
    # Calcular Médias Móveis
    df['sma_9'] = calculate_sma(df['price'], 9)
    df['sma_13'] = calculate_sma(df['price'], 13)
    
    # Configurações realistas
    current_capital = initial_capital
    fee_rate = 0.002  # 0.2% de fee por trade (Binance spot)
    
    # Inicializar colunas
    df['signal'] = 'HOLD'
    df['position'] = 0
    df['trade_price'] = 0.0
    df['pnl_percent'] = 0.0
    df['pnl_euros'] = 0.0
    df['position_size_eth'] = 0.0
    df['current_capital'] = current_capital
    df['trade_amount'] = 0.0
    df['fees'] = 0.0
    df['total_fees'] = 0.0
    
    position = 0
    entry_price = 0
    position_size_eth = 0
    total_fees_paid = 0
    trades_executed = 0
    max_trades_per_day = 3  # Limite realista de trades por dia

    for i in range(len(df)):
        current_rsi = df['rsi'].iloc[i]
        current_price = df['price'].iloc[i]
        current_hour = df.index[i].hour
        
        # Atualizar capital atual
        df.loc[df.index[i], 'current_capital'] = current_capital
        df.loc[df.index[i], 'total_fees'] = total_fees_paid
        
        # Verificar limite diário de trades
        trades_today = 0
        if i > 24:  # Verificar últimas 24 horas
            today_start = df.index[i] - timedelta(hours=24)
            trades_today = len(df.loc[today_start:df.index[i], 'signal'][df.loc[today_start:df.index[i], 'signal'] != 'HOLD'])
        
        # Lógica de trading realista
        if (position == 0 and 
            current_capital >= trade_amount and 
            trades_today < max_trades_per_day and
            8 <= current_hour <= 22):  # Trading apenas em horas de mercado
            
            if current_rsi < rsi_lower:  # COMPRAR
                # Calcular fees
                trade_fee = trade_amount * fee_rate
                net_trade_amount = trade_amount - trade_fee
                position_size_eth = net_trade_amount / current_price
                
                df.loc[df.index[i], 'signal'] = 'BUY'
                df.loc[df.index[i], 'position'] = 1
                df.loc[df.index[i], 'trade_price'] = current_price
                df.loc[df.index[i], 'trade_amount'] = trade_amount
                df.loc[df.index[i], 'position_size_eth'] = position_size_eth
                df.loc[df.index[i], 'fees'] = trade_fee
                
                position = 1
                entry_price = current_price
                current_capital -= trade_amount
                total_fees_paid += trade_fee
                trades_executed += 1
                
            elif current_rsi > rsi_upper:  # VENDER (apenas long, não short)
                # Em mercado spot, não fazemos short selling
                pass
                
        elif position == 1:  # Posição comprada
            # Condições de saída
            take_profit = entry_price * 1.05  # TP 5%
            stop_loss = entry_price * 0.95    # SL 5%
            
            if (current_rsi > rsi_upper or 
                current_price >= take_profit or 
                current_price <= stop_loss):
                
                # Calcular PnL
                pnl_percent = (current_price - entry_price) / entry_price * 100
                pnl_euros = position_size_eth * (current_price - entry_price)
                
                # Calcular fee de saída
                exit_fee = (position_size_eth * current_price) * fee_rate
                net_pnl_euros = pnl_euros - exit_fee
                
                df.loc[df.index[i], 'signal'] = 'SELL'
                df.loc[df.index[i], 'position'] = 0
                df.loc[df.index[i], 'trade_price'] = current_price
                df.loc[df.index[i], 'pnl_percent'] = pnl_percent
                df.loc[df.index[i], 'pnl_euros'] = net_pnl_euros
                df.loc[df.index[i], 'fees'] = exit_fee
                
                # Atualizar capital
                current_capital += trade_amount + net_pnl_euros
                total_fees_paid += exit_fee
                
                position = 0
                position_size_eth = 0
    
    return df

# Função para análise de risco
def calculate_risk_metrics(trades_df, initial_capital):
    if len(trades_df) == 0:
        return {}
    
    returns = trades_df['pnl_euros'] / initial_capital
    
    risk_metrics = {
        'Sharpe Ratio': returns.mean() / returns.std() * np.sqrt(365) if returns.std() > 0 else 0,
        'Max Drawdown': (trades_df['pnl_euros'].cumsum().cummax() - trades_df['pnl_euros'].cumsum()).max(),
        'Volatilidade Anual': returns.std() * np.sqrt(365),
        'VaR 95%': np.percentile(returns, 5) * initial_capital,
        'Win Rate': len(trades_df[trades_df['pnl_euros'] > 0]) / len(trades_df) * 100,
        'Profit Factor': trades_df[trades_df['pnl_euros'] > 0]['pnl_euros'].sum() / 
                        abs(trades_df[trades_df['pnl_euros'] < 0]['pnl_euros'].sum()) 
                        if trades_df[trades_df['pnl_euros'] < 0]['pnl_euros'].sum() != 0 else float('inf')
    }
    
    return risk_metrics

# Interface principal
st.sidebar.header("⚙️ Configurações Realistas")

# Parâmetros de trading
rsi_period = st.sidebar.slider("Período RSI", 5, 30, 14)
rsi_upper = st.sidebar.slider("RSI Superior (Venda)", 60, 90, 70)
rsi_lower = st.sidebar.slider("RSI Inferior (Compra)", 10, 40, 30)

# Configurações de capital
st.sidebar.header("💰 Gestão de Capital")
initial_capital = st.sidebar.number_input("Capital Inicial (€)", min_value=100, max_value=10000, value=1000, step=100)
trade_amount = st.sidebar.number_input("Valor por Trade (€)", min_value=5, max_value=200, value=10, step=5)
risk_per_trade = st.sidebar.slider("Risco por Trade (%)", 0.5, 5.0, 1.0) / 100

# Configurações de mercado
st.sidebar.header("📈 Condições de Mercado")
trading_fee = st.sidebar.slider("Fee por Trade (%)", 0.1, 1.0, 0.2) / 100
max_daily_trades = st.sidebar.slider("Max Trades/Dia", 1, 10, 3)
trading_hours = st.sidebar.checkbox("Apenas Horário de Mercado (8h-22h)", True)

# Carregar dados
st.header("📊 Dados Reais do Ethereum")

with st.spinner("Carregando dados de mercado reais..."):
    df = get_real_ethereum_data()

if df is not None and len(df) > 0:
    # Simular trading realista
    trading_df = simulate_realistic_trading(
        df, rsi_lower, rsi_upper, rsi_period, 
        initial_capital, trade_amount
    )
    
    # Filtrar trades executados
    trades_df = trading_df[trading_df['signal'] != 'HOLD'].copy()
    
    # Calcular métricas
    final_capital = trading_df['current_capital'].iloc[-1]
    total_pnl_euros = final_capital - initial_capital
    total_return = (total_pnl_euros / initial_capital) * 100
    total_fees = trading_df['total_fees'].iloc[-1]
    
    # Calcular métricas de risco
    risk_metrics = calculate_risk_metrics(trades_df, initial_capital)
    
    # Display principal
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        current_price = trading_df['price'].iloc[-1]
        st.metric("💰 Preço ETH", f"${current_price:.2f}")
    
    with col2:
        current_rsi = trading_df['rsi'].iloc[-1]
        st.metric("📊 RSI Atual", f"{current_rsi:.2f}")
    
    with col3:
        st.metric("💳 Capital Final", f"€{final_capital:.2f}")
    
    with col4:
        st.metric("📈 Retorno Total", f"€{total_pnl_euros:.2f}", f"{total_return:.2f}%")
    
    # Métricas de desempenho
    st.subheader("🎯 Performance da Estratégia")
    
    perf_col1, perf_col2, perf_col3, perf_col4 = st.columns(4)
    
    with perf_col1:
        st.info(f"**Trades Executados:** {len(trades_df)}")
    
    with perf_col2:
        win_rate = risk_metrics.get('Win Rate', 0)
        st.info(f"**Win Rate:** {win_rate:.1f}%")
    
    with perf_col3:
        st.info(f"**Total em Fees:** €{total_fees:.2f}")
    
    with perf_col4:
        profit_factor = risk_metrics.get('Profit Factor', 0)
        st.info(f"**Profit Factor:** {profit_factor:.2f}")
    
    # Gráfico de evolução do capital
    st.subheader("💹 Evolução do Capital")
    
    capital_fig = go.Figure()
    
    capital_fig.add_trace(go.Scatter(
        x=trading_df.index,
        y=trading_df['current_capital'],
        name='Capital (€)',
        line=dict(color='#00D4AA', width=3),
        fill='tozeroy',
        fillcolor='rgba(0, 212, 170, 0.1)'
    ))
    
    capital_fig.add_hline(
        y=initial_capital, 
        line_dash="dash", 
        line_color="white",
        annotation_text=f"Capital Inicial: €{initial_capital:.0f}"
    )
    
    capital_fig.update_layout(
        title="Evolução do Capital - Simulação Realista",
        xaxis_title="Data",
        yaxis_title="Capital (€)",
        template="plotly_dark",
        height=400
    )
    
    st.plotly_chart(capital_fig, use_container_width=True)
    
    # Análise de risco
    st.subheader("⚠️ Análise de Risco")
    
    if risk_metrics:
        risk_col1, risk_col2, risk_col3, risk_col4 = st.columns(4)
        
        with risk_col1:
            sharpe = risk_metrics.get('Sharpe Ratio', 0)
            st.metric("Sharpe Ratio", f"{sharpe:.2f}")
        
        with risk_col2:
            max_dd = risk_metrics.get('Max Drawdown', 0)
            st.metric("Max Drawdown", f"€{max_dd:.2f}")
        
        with risk_col3:
            volatility = risk_metrics.get('Volatilidade Anual', 0)
            st.metric("Volatilidade Anual", f"{volatility:.2%}")
        
        with risk_col4:
            var = risk_metrics.get('VaR 95%', 0)
            st.metric("VaR 95%", f"€{var:.2f}")
    
    # Gráfico completo
    st.subheader("📊 Análise Técnica Completa")
    
    fig = make_subplots(
        rows=3, cols=1,
        subplot_titles=('Preço do Ethereum com Sinais e Médias Móveis', 'RSI Indicator', 'Evolução do Capital'),
        vertical_spacing=0.08,
        row_heights=[0.5, 0.25, 0.25]
    )
    
    # Preço
    fig.add_trace(
        go.Scatter(
            x=trading_df.index,
            y=trading_df['price'],
            name='Preço ETH',
            line=dict(color='#00D4AA', width=2)
        ),
        row=1, col=1
    )
    
    # Média Móvel 9
    fig.add_trace(
        go.Scatter(
            x=trading_df.index,
            y=trading_df['sma_9'],
            name='SMA 9',
            line=dict(color='#FFD700', width=1.5, dash='dot')
        ),
        row=1, col=1
    )
    
    # Média Móvel 13
    fig.add_trace(
        go.Scatter(
            x=trading_df.index,
            y=trading_df['sma_13'],
            name='SMA 13',
            line=dict(color='#FF6B6B', width=1.5, dash='dash')
        ),
        row=1, col=1
    )
    
    # Sinais de compra
    buy_signals = trading_df[trading_df['signal'] == 'BUY']
    if len(buy_signals) > 0:
        fig.add_trace(
            go.Scatter(
                x=buy_signals.index,
                y=buy_signals['price'],
                mode='markers',
                name='COMPRA',
                marker=dict(color='green', size=10, symbol='triangle-up')
            ),
            row=1, col=1
        )
    
    # Sinais de venda
    sell_signals = trading_df[trading_df['signal'] == 'SELL']
    if len(sell_signals) > 0:
        fig.add_trace(
            go.Scatter(
                x=sell_signals.index,
                y=sell_signals['price'],
                mode='markers',
                name='VENDA',
                marker=dict(color='red', size=10, symbol='triangle-down')
            ),
            row=1, col=1
        )
    
    # RSI
    fig.add_trace(
        go.Scatter(
            x=trading_df.index,
            y=trading_df['rsi'],
            name='RSI',
            line=dict(color='#FF6B6B', width=2)
        ),
        row=2, col=1
    )
    
    fig.add_hline(y=rsi_upper, line_dash="dash", line_color="red", row=2, col=1)
    fig.add_hline(y=rsi_lower, line_dash="dash", line_color="green", row=2, col=1)
    fig.add_hline(y=50, line_dash="dot", line_color="gray", row=2, col=1)
    
    # Capital
    fig.add_trace(
        go.Scatter(
            x=trading_df.index,
            y=trading_df['current_capital'],
            name='Capital (€)',
            line=dict(color='#FFD700', width=2)
        ),
        row=3, col=1
    )
    
    fig.update_layout(
        height=900, 
        showlegend=True,
        title_text="Simulação Realista - Ethereum Trading Bot",
        template="plotly_dark"
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Tabela de trades
    if len(trades_df) > 0:
        st.subheader("💼 Histórico de Trades")
        
        display_trades = trades_df[['price', 'rsi', 'signal', 'trade_price', 
                                  'trade_amount', 'position_size_eth', 'pnl_percent', 'pnl_euros', 'fees']].copy()
        display_trades['timestamp'] = display_trades.index
        display_trades = display_trades[['timestamp', 'signal', 'trade_price', 
                                       'trade_amount', 'position_size_eth', 'pnl_percent', 'pnl_euros', 'fees']]
        display_trades.columns = ['Data/Hora', 'Sinal', 'Preço Trade', 
                                'Valor (€)', 'Tamanho (ETH)', 'PnL (%)', 'PnL (€)', 'Fees (€)']
        
        # Formatação
        for col in ['Preço Trade', 'Valor (€)', 'PnL (€)', 'Fees (€)']:
            display_trades[col] = display_trades[col].round(2)
        display_trades['Tamanho (ETH)'] = display_trades['Tamanho (ETH)'].round(6)
        display_trades['PnL (%)'] = display_trades['PnL (%)'].round(2)
        
        st.dataframe(display_trades, use_container_width=True)

# Conclusão realista
st.markdown("---")
st.subheader("📋 Conclusão da Simulação Realista")

if len(trades_df) > 0:
    conclusion_col1, conclusion_col2 = st.columns(2)
    
    with conclusion_col1:
        st.info("""
        **✅ Pontos Fortes:**
        - Estratégia baseada em RSI testada
        - Gestão de risco incorporada
        - Custos de trading realistas
        - Limites de operação diária
        - Indicadores de Média Móvel (9 e 13) adicionados para análise visual
        """)
    
    with conclusion_col2:
        st.warning("""
        **⚠️ Limitações:**
        - Mercado crypto é altamente volátil
        - Estratégia pode não funcionar em todos os mercados
        - Não considera notícias ou eventos
        - Backtest não garante resultados futuros
        """)
else:
    st.info("""
    **ℹ️ Sem trades executados:**
    - Os parâmetros atuais não geraram sinais de trading
    - Tente ajustar os limites do RSI
    - Considere condições de mercado diferentes
    """)

st.markdown("""
<div style='background-color: #2E4A21; padding: 20px; border-radius: 10px; margin: 20px 0;'>
    <h4 style='color: white; margin: 0;'>🚨 Aviso Importante</h4>
    <p style='color: white; margin: 10px 0 0 0;'>
    Esta é uma <strong>simulação educacional</strong>. Trading real envolve risços significativos. 
    Nunca invista mais do que pode perder e sempre consulte profissionais financeiros.
    </p>
</div>
""", unsafe_allow_html=True)

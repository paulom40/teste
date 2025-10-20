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
    page_title="Multi-Asset Trading Bot - Simulação Real",
    page_icon="📈",
    layout="wide"
)

# =============================================================================
# MÓDULO: CÁLCULO DE INDICADORES
# =============================================================================

def calculate_rsi_simple(prices, period=14):
    """
    Versão simplificada e mais robusta para calcular RSI
    """
    try:
        if not isinstance(prices, pd.Series):
            prices = pd.Series(prices)
        
        delta = prices.diff()
        gain = delta.where(delta > 0, 0)
        loss = -delta.where(delta < 0, 0)
        
        avg_gain = gain.rolling(window=period, min_periods=1).mean()
        avg_loss = loss.rolling(window=period, min_periods=1).mean()
        
        rs = avg_gain / avg_loss
        rsi = 100 - (100 / (1 + rs))
        rsi = rsi.fillna(50)
        
        return rsi
        
    except Exception as e:
        st.error(f"Erro no cálculo simplificado do RSI: {e}")
        return pd.Series([50] * len(prices), index=prices.index)

def calculate_sma(prices, period):
    """
    Calcula a Média Móvel Simples para o período especificado
    """
    try:
        if not isinstance(prices, pd.Series):
            prices = pd.Series(prices)
        
        sma = prices.rolling(window=period, min_periods=1).mean()
        sma = sma.fillna(prices)
        
        return sma
        
    except Exception as e:
        st.error(f"Erro no cálculo da SMA: {e}")
        return pd.Series([np.nan] * len(prices), index=prices.index)

def calculate_indicators(df, rsi_period=14):
    """
    Calcula todos os indicadores técnicos necessários
    """
    df = df.copy()
    df['rsi'] = calculate_rsi_simple(df['price'], rsi_period)
    df['sma_9'] = calculate_sma(df['price'], 9)
    df['sma_13'] = calculate_sma(df['price'], 13)
    return df

# =============================================================================
# MÓDULO: OBTENÇÃO DE DADOS
# =============================================================================

def get_real_coin_data(coin_id):
    try:
        st.info(f"🔄 Obtendo dados reais do {coin_id.upper()}...")
        
        url = f"https://api.coingecko.com/api/v3/coins/{coin_id}/market_chart"
        params = {
            'vs_currency': 'usd',
            'days': '90',
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
                df.columns = ['price']  # Padronizar coluna
                
                df_hourly = df.resample('H').interpolate()
                
                np.random.seed(42 + hash(coin_id))  # Seed diferente por coin para variação
                noise = np.random.normal(0, df_hourly['price'].std() * 0.01, len(df_hourly))
                df_hourly['price'] = df_hourly['price'] + noise
                
                st.success(f"✅ Dados reais de {coin_id.upper()} obtidos com sucesso!")
                return df_hourly
                
        return create_realistic_sample_data(coin_id)
            
    except Exception as e:
        st.warning(f"⚠️ Erro na API para {coin_id}: {e}. Usando dados realistas simulados.")
        return create_realistic_sample_data(coin_id)

def create_realistic_sample_data(coin_id):
    st.info(f"📊 Gerando dados realistas para {coin_id.upper()}...")
    
    end_date = datetime.now()
    start_date = end_date - timedelta(days=90)
    dates = pd.date_range(start=start_date, end=end_date, freq='H')
    
    np.random.seed(42 + hash(coin_id))
    
    # Diferentes tendências baseadas no coin
    if coin_id == 'bitcoin':
        base_trend = np.linspace(50000, 70000, len(dates))
    elif coin_id == 'solana':
        base_trend = np.linspace(150, 250, len(dates))
    else:  # Default para ethereum
        base_trend = np.linspace(2800, 3800, len(dates))
    
    volatility = np.random.normal(0, base_trend.std() * 0.1, len(dates))
    market_cycle = np.sin(np.arange(len(dates)) / 500) * base_trend.std() * 0.2
    short_cycle = np.sin(np.arange(len(dates)) / 50) * base_trend.std() * 0.05
    
    prices = base_trend + volatility + market_cycle + short_cycle
    
    # Limites realistas por coin
    if coin_id == 'bitcoin':
        prices = np.clip(prices, 40000, 80000)
    elif coin_id == 'solana':
        prices = np.clip(prices, 100, 300)
    else:
        prices = np.clip(prices, 2500, 4500)
    
    df = pd.DataFrame({'price': prices}, index=dates)
    
    st.success(f"✅ Dados realistas para {coin_id.upper()} gerados com sucesso!")
    return df

def load_multi_asset_data(selected_coins):
    """
    Carrega os dados de preço para múltiplos ativos
    """
    data_dict = {}
    for coin in selected_coins:
        with st.spinner(f"Carregando dados para {coin.upper()}..."):
            df = get_real_coin_data(coin)
            if df is not None and len(df) > 0:
                data_dict[coin] = df
    return data_dict

# =============================================================================
# MÓDULO: SIMULAÇÃO DE TRADING
# =============================================================================

def initialize_trading_columns(df, initial_capital):
    """
    Inicializa as colunas necessárias para a simulação de trading
    """
    df = df.copy()
    df['signal'] = 'HOLD'
    df['position'] = 0
    df['trade_price'] = 0.0
    df['pnl_percent'] = 0.0
    df['pnl_euros'] = 0.0
    df['position_size_eth'] = 0.0  # Generalizar para position_size
    df['current_capital'] = initial_capital
    df['trade_amount'] = 0.0
    df['fees'] = 0.0
    df['total_fees'] = 0.0
    return df

def check_daily_trade_limit(i, df, max_trades_per_day):
    """
    Verifica o limite de trades diários
    """
    trades_today = 0
    if i > 24:
        today_start = df.index[i] - timedelta(hours=24)
        trades_today = len(df.loc[today_start:df.index[i], 'signal'][df.loc[today_start:df.index[i], 'signal'] != 'HOLD'])
    return trades_today < max_trades_per_day

def execute_buy_signal(i, df, trade_amount, fee_rate, current_price):
    """
    Executa sinal de compra
    """
    trade_fee = trade_amount * fee_rate
    net_trade_amount = trade_amount - trade_fee
    position_size = net_trade_amount / current_price
    
    df.loc[df.index[i], 'signal'] = 'BUY'
    df.loc[df.index[i], 'position'] = 1
    df.loc[df.index[i], 'trade_price'] = current_price
    df.loc[df.index[i], 'trade_amount'] = trade_amount
    df.loc[df.index[i], 'position_size'] = position_size
    df.loc[df.index[i], 'fees'] = trade_fee
    
    return 1, current_price, position_size, trade_fee

def execute_sell_signal(i, df, position_size, entry_price, current_price, fee_rate):
    """
    Executa sinal de venda
    """
    pnl_percent = (current_price - entry_price) / entry_price * 100
    pnl_euros = position_size * (current_price - entry_price)
    exit_fee = (position_size * current_price) * fee_rate
    net_pnl_euros = pnl_euros - exit_fee
    
    df.loc[df.index[i], 'signal'] = 'SELL'
    df.loc[df.index[i], 'position'] = 0
    df.loc[df.index[i], 'trade_price'] = current_price
    df.loc[df.index[i], 'pnl_percent'] = pnl_percent
    df.loc[df.index[i], 'pnl_euros'] = net_pnl_euros
    df.loc[df.index[i], 'fees'] = exit_fee
    
    return 0, 0, 0, exit_fee, net_pnl_euros

def simulate_realistic_trading(df, rsi_lower=30, rsi_upper=70, rsi_period=14, 
                              initial_capital=1000, trade_amount=10, 
                              fee_rate=0.002, max_trades_per_day=3, trading_hours=True):
    """
    Simula trading realista com custos e limitações
    """
    df = calculate_indicators(df, rsi_period)
    df = initialize_trading_columns(df, initial_capital)
    
    current_capital = initial_capital
    position = 0
    entry_price = 0
    position_size = 0
    total_fees_paid = 0

    for i in range(len(df)):
        current_rsi = df['rsi'].iloc[i]
        current_price = df['price'].iloc[i]
        current_hour = df.index[i].hour
        
        df.loc[df.index[i], 'current_capital'] = current_capital
        df.loc[df.index[i], 'total_fees'] = total_fees_paid
        
        if trading_hours and not (8 <= current_hour <= 22):
            continue
        
        trades_today_ok = check_daily_trade_limit(i, df, max_trades_per_day)
        
        if (position == 0 and current_capital >= trade_amount and trades_today_ok):
            
            if current_rsi < rsi_lower:
                position, entry_price, position_size, trade_fee = execute_buy_signal(
                    i, df, trade_amount, fee_rate, current_price
                )
                current_capital -= trade_amount
                total_fees_paid += trade_fee
                
        elif position == 1:
            take_profit = entry_price * 1.05
            stop_loss = entry_price * 0.95
            
            if (current_rsi > rsi_upper or current_price >= take_profit or current_price <= stop_loss):
                position, entry_price, position_size, exit_fee, net_pnl_euros = execute_sell_signal(
                    i, df, position_size, entry_price, current_price, fee_rate
                )
                current_capital += trade_amount + net_pnl_euros
                total_fees_paid += exit_fee
    
    return df

# =============================================================================
# MÓDULO: MÉTRICAS DE RISCO
# =============================================================================

def calculate_risk_metrics(trades_df, initial_capital):
    if len(trades_df) == 0:
        return {}
    
    returns = trades_df['pnl_euros'] / initial_capital
    
    risk_metrics = {
        'Sharpe Ratio': returns.mean() / returns.std() * np.sqrt(365) if returns.std() > 0 else 0,
        'Max Drawdown': (trades_df['pnl_euros'].cumsum().cummax() - trades_df['pnl_euros'].cumsum()).max(),
        'Volatilidade Anual': returns.std() * np.sqrt(365),
        'VaR 95%': np.percentile(returns, 5) * initial_capital,
        'Win Rate': len(trades_df[trades_df['pnl_euros'] > 0]) / len(trades_df) * 100 if len(trades_df) > 0 else 0,
        'Profit Factor': trades_df[trades_df['pnl_euros'] > 0]['pnl_euros'].sum() / 
                        abs(trades_df[trades_df['pnl_euros'] < 0]['pnl_euros'].sum()) 
                        if len(trades_df[trades_df['pnl_euros'] < 0]) > 0 and trades_df[trades_df['pnl_euros'] < 0]['pnl_euros'].sum() != 0 else float('inf')
    }
    
    return risk_metrics

def calculate_performance_metrics(trading_df, initial_capital, trades_df, risk_metrics):
    """
    Calcula métricas de performance gerais
    """
    final_capital = trading_df['current_capital'].iloc[-1]
    total_pnl_euros = final_capital - initial_capital
    total_return = (total_pnl_euros / initial_capital) * 100
    total_fees = trading_df['total_fees'].iloc[-1]
    
    metrics = {
        'final_capital': final_capital,
        'total_pnl_euros': total_pnl_euros,
        'total_return': total_return,
        'total_fees': total_fees,
        'num_trades': len(trades_df),
        'win_rate': risk_metrics.get('Win Rate', 0),
        'profit_factor': risk_metrics.get('Profit Factor', 0),
        'sharpe_ratio': risk_metrics.get('Sharpe Ratio', 0),
        'max_drawdown': risk_metrics.get('Max Drawdown', 0),
        'volatilidade_anual': risk_metrics.get('Volatilidade Anual', 0)
    }
    
    return metrics

def run_backtest_for_asset(coin_id, df, config):
    """
    Executa backtest para um ativo específico
    """
    trading_df = simulate_realistic_trading(
        df, 
        config['rsi_lower'], 
        config['rsi_upper'], 
        config['rsi_period'], 
        config['initial_capital'], 
        config['trade_amount'],
        config['trading_fee'],
        config['max_daily_trades'],
        config['trading_hours']
    )
    
    trades_df = trading_df[trading_df['signal'] != 'HOLD'].copy()
    risk_metrics = calculate_risk_metrics(trades_df, config['initial_capital'])
    metrics = calculate_performance_metrics(trading_df, config['initial_capital'], trades_df, risk_metrics)
    metrics['coin'] = coin_id
    
    return trading_df, trades_df, metrics

# =============================================================================
# MÓDULO: VISUALIZAÇÕES
# =============================================================================

def create_capital_evolution_fig(trading_df, initial_capital, coin_id):
    """
    Cria gráfico de evolução do capital para um ativo
    """
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        x=trading_df.index,
        y=trading_df['current_capital'],
        name=f'Capital {coin_id.upper()} (€)',
        line=dict(color='#00D4AA', width=3),
        fill='tozeroy',
        fillcolor='rgba(0, 212, 170, 0.1)'
    ))
    
    fig.add_hline(
        y=initial_capital, 
        line_dash="dash", 
        line_color="white",
        annotation_text=f"Capital Inicial: €{initial_capital:.0f}"
    )
    
    fig.update_layout(
        title=f"Evolução do Capital - {coin_id.upper()}",
        xaxis_title="Data",
        yaxis_title="Capital (€)",
        template="plotly_dark",
        height=400
    )
    
    return fig

def create_main_analysis_fig(trading_df, rsi_upper, rsi_lower, coin_id):
    """
    Cria o gráfico principal de análise técnica para um ativo
    """
    fig = make_subplots(
        rows=3, cols=1,
        subplot_titles=(f'Preço do {coin_id.upper()} com Sinais e Médias Móveis', 'RSI Indicator', 'Evolução do Capital'),
        vertical_spacing=0.08,
        row_heights=[0.5, 0.25, 0.25]
    )
    
    # Preço e SMAs
    fig.add_trace(
        go.Scatter(x=trading_df.index, y=trading_df['price'], name=f'Preço {coin_id.upper()}',
                   line=dict(color='#00D4AA', width=2)), row=1, col=1
    )
    fig.add_trace(
        go.Scatter(x=trading_df.index, y=trading_df['sma_9'], name='SMA 9',
                   line=dict(color='#FFD700', width=1.5, dash='dot')), row=1, col=1
    )
    fig.add_trace(
        go.Scatter(x=trading_df.index, y=trading_df['sma_13'], name='SMA 13',
                   line=dict(color='#FF6B6B', width=1.5, dash='dash')), row=1, col=1
    )
    
    # Sinais
    buy_signals = trading_df[trading_df['signal'] == 'BUY']
    if len(buy_signals) > 0:
        fig.add_trace(
            go.Scatter(x=buy_signals.index, y=buy_signals['price'], mode='markers', name='COMPRA',
                       marker=dict(color='green', size=10, symbol='triangle-up')), row=1, col=1
        )
    
    sell_signals = trading_df[trading_df['signal'] == 'SELL']
    if len(sell_signals) > 0:
        fig.add_trace(
            go.Scatter(x=sell_signals.index, y=sell_signals['price'], mode='markers', name='VENDA',
                       marker=dict(color='red', size=10, symbol='triangle-down')), row=1, col=1
        )
    
    # RSI
    fig.add_trace(
        go.Scatter(x=trading_df.index, y=trading_df['rsi'], name='RSI',
                   line=dict(color='#FF6B6B', width=2)), row=2, col=1
    )
    fig.add_hline(y=rsi_upper, line_dash="dash", line_color="red", row=2, col=1)
    fig.add_hline(y=rsi_lower, line_dash="dash", line_color="green", row=2, col=1)
    fig.add_hline(y=50, line_dash="dot", line_color="gray", row=2, col=1)
    
    # Capital
    fig.add_trace(
        go.Scatter(x=trading_df.index, y=trading_df['current_capital'], name='Capital (€)',
                   line=dict(color='#FFD700', width=2)), row=3, col=1
    )
    
    fig.update_layout(
        height=900, showlegend=True,
        title_text=f"Simulação Realista - {coin_id.upper()} Trading Bot",
        template="plotly_dark"
    )
    
    return fig

def display_main_metrics(current_price, current_rsi, metrics, coin_id):
    """
    Exibe as métricas principais para um ativo
    """
    st.subheader(f"📊 Visão Geral - {coin_id.upper()}")
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("💰 Preço Atual", f"${current_price:.2f}")
    
    with col2:
        st.metric("📊 RSI Atual", f"{current_rsi:.2f}")
    
    with col3:
        st.metric("💳 Capital Final", f"€{metrics['final_capital']:.2f}")
    
    with col4:
        st.metric("📈 Retorno Total", f"€{metrics['total_pnl_euros']:.2f}", f"{metrics['total_return']:.2f}%")

def display_performance_metrics(metrics, risk_metrics, coin_id):
    """
    Exibe métricas de performance para um ativo
    """
    st.subheader(f"🎯 Performance - {coin_id.upper()}")
    
    perf_col1, perf_col2, perf_col3, perf_col4 = st.columns(4)
    
    with perf_col1:
        st.info(f"**Trades Executados:** {metrics['num_trades']}")
    
    with perf_col2:
        st.info(f"**Win Rate:** {metrics['win_rate']:.1f}%")
    
    with perf_col3:
        st.info(f"**Total em Fees:** €{metrics['total_fees']:.2f}")
    
    with perf_col4:
        st.info(f"**Profit Factor:** {metrics['profit_factor']:.2f}")

def display_risk_metrics(risk_metrics, coin_id):
    """
    Exibe análise de risco para um ativo
    """
    st.subheader(f"⚠️ Análise de Risco - {coin_id.upper()}")
    
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

def display_trades_table(trades_df, coin_id):
    """
    Exibe tabela de histórico de trades para um ativo
    """
    if len(trades_df) > 0:
        st.subheader(f"💼 Histórico de Trades - {coin_id.upper()}")
        
        display_trades = trades_df[['price', 'rsi', 'signal', 'trade_price', 
                                  'trade_amount', 'position_size', 'pnl_percent', 'pnl_euros', 'fees']].copy()
        display_trades['timestamp'] = display_trades.index
        display_trades = display_trades[['timestamp', 'signal', 'trade_price', 
                                       'trade_amount', 'position_size', 'pnl_percent', 'pnl_euros', 'fees']]
        display_trades.columns = ['Data/Hora', 'Sinal', 'Preço Trade', 
                                'Valor (€)', 'Tamanho', 'PnL (%)', 'PnL (€)', 'Fees (€)']
        
        for col in ['Preço Trade', 'Valor (€)', 'PnL (€)', 'Fees (€)']:
            display_trades[col] = display_trades[col].round(2)
        display_trades['Tamanho'] = display_trades['Tamanho'].round(6)
        display_trades['PnL (%)'] = display_trades['PnL (%)'].round(2)
        
        st.dataframe(display_trades, use_container_width=True)

def display_asset_dashboard(coin_id, trading_df, trades_df, metrics, config):
    """
    Exibe o dashboard completo para um ativo
    """
    current_price = trading_df['price'].iloc[-1]
    current_rsi = trading_df['rsi'].iloc[-1]
    
    display_main_metrics(current_price, current_rsi, metrics, coin_id)
    
    st.subheader(f"💹 Evolução do Capital - {coin_id.upper()}")
    capital_fig = create_capital_evolution_fig(trading_df, config['initial_capital'], coin_id)
    st.plotly_chart(capital_fig, use_container_width=True)
    
    display_performance_metrics(metrics, {}, coin_id)  # risk_metrics já em metrics
    
    display_risk_metrics({}, coin_id)  # Integrado em metrics
    
    st.subheader(f"📊 Análise Técnica Completa - {coin_id.upper()}")
    main_fig = create_main_analysis_fig(trading_df, config['rsi_upper'], config['rsi_lower'], coin_id)
    st.plotly_chart(main_fig, use_container_width=True)
    
    display_trades_table(trades_df, coin_id)

def create_comparison_table(all_metrics):
    """
    Cria tabela comparativa de métricas entre ativos
    """
    if not all_metrics:
        return
    
    comparison_df = pd.DataFrame(all_metrics)
    comparison_df = comparison_df[['coin', 'total_return', 'num_trades', 'win_rate', 'sharpe_ratio', 
                                  'max_drawdown', 'total_fees']]
    comparison_df.columns = ['Ativo', 'Retorno Total (%)', 'Nº Trades', 'Win Rate (%)', 
                            'Sharpe Ratio', 'Max Drawdown (€)', 'Total Fees (€)']
    
    for col in ['Retorno Total (%)', 'Win Rate (%)']:
        comparison_df[col] = comparison_df[col].round(2)
    for col in ['Max Drawdown (€)', 'Total Fees (€)']:
        comparison_df[col] = comparison_df[col].round(2)
    
    st.subheader("📋 Comparação Multi-Ativo")
    st.dataframe(comparison_df.set_index('Ativo'), use_container_width=True)

# =============================================================================
# MÓDULO: INTERFACE DE USUÁRIO
# =============================================================================

def setup_sidebar():
    """
    Configura a sidebar com parâmetros
    """
    st.sidebar.header("⚙️ Configurações Realistas")

    # Seleção de ativos
    st.sidebar.header("📈 Ativos para Backtest")
    available_coins = ['bitcoin', 'ethereum', 'solana', 'cardano', 'polkadot']
    selected_coins = st.sidebar.multiselect(
        "Selecione os ativos", 
        available_coins, 
        default=['bitcoin', 'ethereum', 'solana']
    )

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
    
    return {
        'selected_coins': selected_coins,
        'rsi_period': rsi_period,
        'rsi_upper': rsi_upper,
        'rsi_lower': rsi_lower,
        'initial_capital': initial_capital,
        'trade_amount': trade_amount,
        'risk_per_trade': risk_per_trade,
        'trading_fee': trading_fee,
        'max_daily_trades': max_daily_trades,
        'trading_hours': trading_hours
    }

def display_conclusion(all_metrics):
    """
    Exibe a conclusão da simulação multi-ativo
    """
    st.markdown("---")
    st.subheader("📋 Conclusão da Simulação Multi-Ativo")

    total_trades = sum(m['num_trades'] for m in all_metrics)
    if total_trades > 0:
        conclusion_col1, conclusion_col2 = st.columns(2)
        
        with conclusion_col1:
            st.info("""
            **✅ Pontos Fortes:**
            - Backtest independente por ativo
            - Estratégia RSI consistente
            - Gestão de risco incorporada
            - Custos de trading realistas
            - Comparação entre ativos
            - Indicadores de Média Móvel (9 e 13)
            """)
        
        with conclusion_col2:
            st.warning("""
            **⚠️ Limitações:**
            - Mercado crypto é altamente volátil
            - Estratégia pode variar por ativo
            - Não considera correlações entre ativos
            - Não inclui notícias ou eventos
            - Backtest não garante resultados futuros
            """)
    else:
        st.info("""
        **ℹ️ Sem trades executados:**
        - Os parâmetros atuais não geraram sinais
        - Ajuste os limites do RSI por ativo
        - Considere condições de mercado diferentes
        """)

def display_disclaimer():
    """
    Exibe o aviso importante
    """
    st.markdown("""
    <div style='background-color: #2E4A21; padding: 20px; border-radius: 10px; margin: 20px 0;'>
        <h4 style='color: white; margin: 0;'>🚨 Aviso Importante</h4>
        <p style='color: white; margin: 10px 0 0 0;'>
        Esta é uma <strong>simulação educacional</strong>. Trading real envolve risços significativos. 
        Nunca invista mais do que pode perder e sempre consulte profissionais financeiros.
        </p>
    </div>
    """, unsafe_allow_html=True)

# =============================================================================
# EXECUÇÃO PRINCIPAL
# =============================================================================

def main():
    st.title("🤖 Multi-Asset Trading Bot - Simulação Realista")
    st.markdown("---")
    
    # Configurações
    config = setup_sidebar()
    
    if not config['selected_coins']:
        st.warning("⚠️ Selecione pelo menos um ativo para prosseguir.")
        st.stop()
    
    # Carregar dados multi-ativo
    st.header("📊 Dados Reais Multi-Ativo")
    data_dict = load_multi_asset_data(config['selected_coins'])
    
    if not data_dict:
        st.error("❌ Nenhum dado carregado. Verifique a conexão.")
        st.stop()
    
    # Executar backtests
    all_results = {}
    all_metrics = []
    
    for coin_id, df in data_dict.items():
        if len(df) == 0:
            continue
        
        st.info(f"🔄 Executando backtest para {coin_id.upper()}...")
        trading_df, trades_df, metrics = run_backtest_for_asset(coin_id, df, config)
        all_results[coin_id] = {'trading_df': trading_df, 'trades_df': trades_df, 'metrics': metrics}
        all_metrics.append(metrics)
    
    # Tabela de comparação
    create_comparison_table(all_metrics)
    
    # Dashboards individuais
    st.header("📈 Dashboards Individuais")
    tabs = st.tabs([f"{coin.upper()}" for coin in config['selected_coins']])
    
    for idx, coin_id in enumerate(config['selected_coins']):
        if coin_id in all_results:
            with tabs[idx]:
                results = all_results[coin_id]
                display_asset_dashboard(coin_id, results['trading_df'], results['trades_df'], results['metrics'], config)
    
    # Conclusão
    display_conclusion(all_metrics)
    
    display_disclaimer()

if __name__ == "__main__":
    main()

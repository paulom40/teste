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
    page_title="Ethereum Trading Bot",
    page_icon="📈",
    layout="wide"
)

# Título da aplicação
st.title("🤖 Ethereum Trading Bot - RSI Strategy")
st.markdown("---")

# Função para calcular RSI
def calculate_rsi(prices, period=14):
    deltas = np.diff(prices)
    gains = np.where(deltas > 0, deltas, 0)
    losses = np.where(deltas < 0, -deltas, 0)
    
    avg_gains = pd.Series(gains).rolling(window=period).mean()
    avg_losses = pd.Series(losses).rolling(window=period).mean()
    
    rs = avg_gains / avg_losses
    rsi = 100 - (100 / (1 + rs))
    
    # Preencher valores NaN
    rsi = rsi.fillna(50)
    
    return rsi

# Função para obter dados do Ethereum
def get_ethereum_data():
    try:
        # Usando CoinGecko API
        url = "https://api.coingecko.com/api/v3/coins/ethereum/market_chart"
        params = {
            'vs_currency': 'usd',
            'days': '30',
            'interval': 'hourly'
        }
        
        response = requests.get(url, timeout=10)
        data = response.json()
        
        # Processar dados
        prices = data['prices']
        df = pd.DataFrame(prices, columns=['timestamp', 'price'])
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
        df = df.set_index('timestamp')
        
        return df
    
    except Exception as e:
        st.error(f"Erro ao obter dados: {e}")
        # Retornar dados de exemplo em caso de erro
        return create_sample_data()

# Função para criar dados de exemplo
def create_sample_data():
    dates = pd.date_range(start='2024-01-01', end=datetime.now(), freq='H')
    np.random.seed(42)
    prices = [2000 + i * 0.1 + np.random.normal(0, 50) for i in range(len(dates))]
    
    df = pd.DataFrame({
        'price': prices
    }, index=dates)
    
    return df

# Função para simular trading
def simulate_trading(df):
    df = df.copy()
    
    # Calcular RSI
    df['rsi'] = calculate_rsi(df['price'].values)
    
    # Inicializar colunas de sinal
    df['signal'] = 'HOLD'
    df['position'] = 0
    df['trade_price'] = 0.0
    df['pnl'] = 0.0
    
    position = 0  # 0: sem posição, 1: comprado, -1: vendido
    entry_price = 0
    
    for i in range(1, len(df)):
        current_rsi = df['rsi'].iloc[i]
        current_price = df['price'].iloc[i]
        
        # Lógica de trading
        if position == 0:  # Sem posição
            if current_rsi < 30:  # RSI abaixo de 30 - COMPRAR
                df.loc[df.index[i], 'signal'] = 'BUY'
                df.loc[df.index[i], 'position'] = 1
                df.loc[df.index[i], 'trade_price'] = current_price
                position = 1
                entry_price = current_price
                
            elif current_rsi > 70:  # RSI acima de 70 - VENDER
                df.loc[df.index[i], 'signal'] = 'SELL'
                df.loc[df.index[i], 'position'] = -1
                df.loc[df.index[i], 'trade_price'] = current_price
                position = -1
                entry_price = current_price
                
        elif position == 1:  # Posição comprada
            if current_rsi > 70:  # Fechar posição quando RSI > 70
                df.loc[df.index[i], 'signal'] = 'SELL'
                df.loc[df.index[i], 'position'] = 0
                df.loc[df.index[i], 'trade_price'] = current_price
                df.loc[df.index[i], 'pnl'] = (current_price - entry_price) / entry_price * 100
                position = 0
                
        elif position == -1:  # Posição vendida
            if current_rsi < 30:  # Fechar posição quando RSI < 30
                df.loc[df.index[i], 'signal'] = 'BUY'
                df.loc[df.index[i], 'position'] = 0
                df.loc[df.index[i], 'trade_price'] = current_price
                df.loc[df.index[i], 'pnl'] = (entry_price - current_price) / entry_price * 100
                position = 0
    
    return df

# Função para exportar para Excel
def export_to_excel(df, trades_df):
    output = io.BytesIO()
    
    with pd.ExcelWriter(output, engine='openpyxl') as writer:
        # Dados completos
        df.to_excel(writer, sheet_name='Dados Completos')
        
        # Trades executados
        trades_df.to_excel(writer, sheet_name='Trades Executados')
        
        # Resumo de performance
        summary_data = {
            'Métrica': [
                'Total de Trades',
                'Trades Lucrativos',
                'Trades Prejudiciais',
                'Taxa de Sucesso (%)',
                'Lucro Total (%)',
                'Melhor Trade (%)',
                'Pior Trade (%)'
            ],
            'Valor': [
                len(trades_df),
                len(trades_df[trades_df['pnl'] > 0]),
                len(trades_df[trades_df['pnl'] < 0]),
                len(trades_df[trades_df['pnl'] > 0]) / len(trades_df) * 100 if len(trades_df) > 0 else 0,
                trades_df['pnl'].sum(),
                trades_df['pnl'].max(),
                trades_df['pnl'].min()
            ]
        }
        summary_df = pd.DataFrame(summary_data)
        summary_df.to_excel(writer, sheet_name='Resumo Performance', index=False)
    
    output.seek(0)
    return output

# Sidebar para configurações
st.sidebar.header("⚙️ Configurações do Trading")

# Parâmetros RSI
rsi_period = st.sidebar.slider("Período RSI", 5, 30, 14)
rsi_upper = st.sidebar.slider("RSI Superior (Venda)", 60, 90, 70)
rsi_lower = st.sidebar.slider("RSI Inferior (Compra)", 10, 40, 30)

# Atualização automática
auto_update = st.sidebar.checkbox("Atualização Automática", value=False)
update_interval = st.sidebar.selectbox("Intervalo de Atualização", [5, 15, 30, 60], index=3)

# Botão para atualizar dados manualmente
if st.sidebar.button("🔄 Atualizar Dados Agora"):
    st.rerun()

# Carregar dados
st.header("📊 Dados do Ethereum em Tempo Real")

with st.spinner("Carregando dados do Ethereum..."):
    df = get_ethereum_data()

if df is not None:
    # Simular trading
    trading_df = simulate_trading(df)
    
    # Filtrar apenas trades executados
    trades_df = trading_df[trading_df['signal'] != 'HOLD'].copy()
    
    # Layout de colunas para métricas
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        current_price = trading_df['price'].iloc[-1]
        st.metric("Preço Atual ETH", f"${current_price:.2f}")
    
    with col2:
        current_rsi = trading_df['rsi'].iloc[-1]
        st.metric("RSI Atual", f"{current_rsi:.2f}")
    
    with col3:
        total_trades = len(trades_df)
        st.metric("Total de Trades", total_trades)
    
    with col4:
        if len(trades_df) > 0:
            win_rate = len(trades_df[trades_df['pnl'] > 0]) / len(trades_df) * 100
            st.metric("Taxa de Sucesso", f"{win_rate:.1f}%")
        else:
            st.metric("Taxa de Sucesso", "0%")
    
    # Gráfico de preço e RSI
    st.header("📈 Gráfico de Preço e RSI")
    
    fig = make_subplots(
        rows=2, cols=1,
        subplot_titles=('Preço do Ethereum', 'RSI Indicator'),
        vertical_spacing=0.1,
        row_heights=[0.7, 0.3]
    )
    
    # Gráfico de preço
    fig.add_trace(
        go.Scatter(
            x=trading_df.index,
            y=trading_df['price'],
            name='Preço ETH',
            line=dict(color='#00D4AA')
        ),
        row=1, col=1
    )
    
    # Adicionar sinais de compra
    buy_signals = trading_df[trading_df['signal'] == 'BUY']
    if len(buy_signals) > 0:
        fig.add_trace(
            go.Scatter(
                x=buy_signals.index,
                y=buy_signals['price'],
                mode='markers',
                name='Compra',
                marker=dict(color='green', size=10, symbol='triangle-up')
            ),
            row=1, col=1
        )
    
    # Adicionar sinais de venda
    sell_signals = trading_df[trading_df['signal'] == 'SELL']
    if len(sell_signals) > 0:
        fig.add_trace(
            go.Scatter(
                x=sell_signals.index,
                y=sell_signals['price'],
                mode='markers',
                name='Venda',
                marker=dict(color='red', size=10, symbol='triangle-down')
            ),
            row=1, col=1
        )
    
    # Gráfico RSI
    fig.add_trace(
        go.Scatter(
            x=trading_df.index,
            y=trading_df['rsi'],
            name='RSI',
            line=dict(color='#FF6B6B')
        ),
        row=2, col=1
    )
    
    # Linhas de sobrecompra/sobrevenda
    fig.add_hline(y=rsi_upper, line_dash="dash", line_color="red", row=2, col=1)
    fig.add_hline(y=rsi_lower, line_dash="dash", line_color="green", row=2, col=1)
    
    fig.update_layout(height=600, showlegend=True)
    st.plotly_chart(fig, use_container_width=True)
    
    # Tabela de trades
    st.header("💼 Trades Executados")
    
    if len(trades_df) > 0:
        display_trades = trades_df[['price', 'rsi', 'signal', 'trade_price', 'pnl']].copy()
        display_trades['timestamp'] = display_trades.index
        display_trades = display_trades[['timestamp', 'price', 'rsi', 'signal', 'trade_price', 'pnl']]
        display_trades.columns = ['Data/Hora', 'Preço ETH', 'RSI', 'Sinal', 'Preço Trade', 'PnL (%)']
        display_trades['PnL (%)'] = display_trades['PnL (%)'].round(2)
        
        st.dataframe(display_trades, use_container_width=True)
        
        # Métricas de performance
        st.header("📊 Performance do Trading")
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            total_pnl = trades_df['pnl'].sum()
            st.metric("Lucro/Prejuízo Total", f"{total_pnl:.2f}%")
        
        with col2:
            winning_trades = len(trades_df[trades_df['pnl'] > 0])
            st.metric("Trades Lucrativos", winning_trades)
        
        with col3:
            losing_trades = len(trades_df[trades_df['pnl'] < 0])
            st.metric("Trades Prejudiciais", losing_trades)
        
        with col4:
            best_trade = trades_df['pnl'].max()
            st.metric("Melhor Trade", f"{best_trade:.2f}%")
    
    else:
        st.info("Nenhum trade executado ainda com os parâmetros atuais.")
    
    # Exportação para Excel
    st.header("📤 Exportar Dados")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.download_button(
            label="📥 Baixar Relatório Excel",
            data=export_to_excel(trading_df, trades_df),
            file_name=f"ethereum_trading_report_{datetime.now().strftime('%Y%m%d_%H%M')}.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
        )
    
    with col2:
        # Botão para simular próximo período
        if st.button("🔄 Simular Próxima Hora"):
            st.info("Funcionalidade de simulação em tempo real será implementada")
    
    # Informações adicionais
    st.sidebar.header("ℹ️ Sobre a Estratégia")
    st.sidebar.info("""
    **Estratégia RSI:**
    - RSI < 30: Sinal de COMPRA (sobrevendido)
    - RSI > 70: Sinal de VENDA (sobrecomprado)
    - Fechamento automático de posições
    """)
    
    # Atualização automática
    if auto_update:
        st.sidebar.info(f"Próxima atualização em {update_interval} minutos")
        time.sleep(update_interval * 60)
        st.rerun()

else:
    st.error("Não foi possível carregar os dados do Ethereum.")

# Rodapé
st.markdown("---")
st.markdown(
    """
    <div style='text-align: center'>
        <p><strong>⚠️ Aviso:</strong> Esta é uma ferramenta educacional. Trading envolve riscos.</p>
    </div>
    """,
    unsafe_allow_html=True
)

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

# Função para obter dados do Ethereum - VERSÃO CORRIGIDA
def get_ethereum_data():
    try:
        # Tentativa 1: CoinGecko API
        st.info("🔄 Obtendo dados da CoinGecko API...")
        url = "https://api.coingecko.com/api/v3/coins/ethereum/market_chart"
        params = {
            'vs_currency': 'usd',
            'days': '30',
            'interval': 'daily'  # Mudado para daily para mais estabilidade
        }
        
        response = requests.get(url, params=params, timeout=15)
        
        if response.status_code == 200:
            data = response.json()
            
            # Verificar se a chave 'prices' existe
            if 'prices' in data and len(data['prices']) > 0:
                prices = data['prices']
                df = pd.DataFrame(prices, columns=['timestamp', 'price'])
                df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
                df = df.set_index('timestamp')
                df = df.resample('H').ffill()  # Converter para horário
                st.success("✅ Dados obtidos com sucesso da CoinGecko!")
                return df
            else:
                st.warning("⚠️ Estrutura de dados inesperada da API. Usando dados de exemplo.")
                return create_sample_data()
        else:
            st.warning(f"⚠️ API CoinGecko retornou status {response.status_code}. Usando dados de exemplo.")
            return create_sample_data()
            
    except Exception as e:
        st.warning(f"⚠️ Erro na API: {e}. Usando dados de exemplo.")
        return create_sample_data()

# Função para criar dados de exemplo mais realistas
def create_sample_data():
    st.info("📊 Gerando dados de exemplo realistas...")
    
    # Criar dados horários para os últimos 30 dias
    end_date = datetime.now()
    start_date = end_date - timedelta(days=30)
    dates = pd.date_range(start=start_date, end=end_date, freq='H')
    
    # Preços mais realistas do Ethereum
    np.random.seed(42)
    base_price = 3500  # Preço base do ETH
    
    # Criar tendência com alguma volatilidade
    prices = []
    current_price = base_price
    
    for i in range(len(dates)):
        # Adicionar alguma tendência e volatilidade
        change = np.random.normal(0, 25)  # Mudança de preço
        current_price += change
        current_price = max(1000, current_price)  # Preço mínimo
        prices.append(current_price)
    
    df = pd.DataFrame({
        'price': prices
    }, index=dates)
    
    st.success("✅ Dados de exemplo gerados com sucesso!")
    return df

# Função para simular trading
def simulate_trading(df, rsi_lower=30, rsi_upper=70):
    df = df.copy()
    
    # Calcular RSI
    df['rsi'] = calculate_rsi(df['price'].values)
    
    # Inicializar colunas de sinal
    df['signal'] = 'HOLD'
    df['position'] = 0
    df['trade_price'] = 0.0
    df['pnl'] = 0.0
    df['cumulative_pnl'] = 0.0
    
    position = 0  # 0: sem posição, 1: comprado, -1: vendido
    entry_price = 0
    cumulative_pnl = 0

    for i in range(1, len(df)):
        current_rsi = df['rsi'].iloc[i]
        current_price = df['price'].iloc[i]
        
        # Lógica de trading
        if position == 0:  # Sem posição
            if current_rsi < rsi_lower:  # RSI abaixo do limite inferior - COMPRAR
                df.loc[df.index[i], 'signal'] = 'BUY'
                df.loc[df.index[i], 'position'] = 1
                df.loc[df.index[i], 'trade_price'] = current_price
                position = 1
                entry_price = current_price
                
            elif current_rsi > rsi_upper:  # RSI acima do limite superior - VENDER
                df.loc[df.index[i], 'signal'] = 'SELL'
                df.loc[df.index[i], 'position'] = -1
                df.loc[df.index[i], 'trade_price'] = current_price
                position = -1
                entry_price = current_price
                
        elif position == 1:  # Posição comprada
            if current_rsi > rsi_upper:  # Fechar posição quando RSI > limite superior
                df.loc[df.index[i], 'signal'] = 'SELL'
                df.loc[df.index[i], 'position'] = 0
                df.loc[df.index[i], 'trade_price'] = current_price
                pnl = (current_price - entry_price) / entry_price * 100
                df.loc[df.index[i], 'pnl'] = pnl
                cumulative_pnl += pnl
                df.loc[df.index[i], 'cumulative_pnl'] = cumulative_pnl
                position = 0
                
        elif position == -1:  # Posição vendida
            if current_rsi < rsi_lower:  # Fechar posição quando RSI < limite inferior
                df.loc[df.index[i], 'signal'] = 'BUY'
                df.loc[df.index[i], 'position'] = 0
                df.loc[df.index[i], 'trade_price'] = current_price
                pnl = (entry_price - current_price) / entry_price * 100
                df.loc[df.index[i], 'pnl'] = pnl
                cumulative_pnl += pnl
                df.loc[df.index[i], 'cumulative_pnl'] = cumulative_pnl
                position = 0
    
    return df

# Função para exportar para Excel
def export_to_excel(df, trades_df):
    output = io.BytesIO()
    
    with pd.ExcelWriter(output, engine='openpyxl') as writer:
        # Dados completos
        df.to_excel(writer, sheet_name='Dados Completos')
        
        # Trades executados
        if len(trades_df) > 0:
            trades_export = trades_df.copy()
            trades_export['timestamp'] = trades_export.index
            trades_export = trades_export[['timestamp', 'price', 'rsi', 'signal', 'trade_price', 'pnl']]
            trades_export.columns = ['Data/Hora', 'Preço ETH', 'RSI', 'Sinal', 'Preço Trade', 'PnL (%)']
            trades_export.to_excel(writer, sheet_name='Trades Executados', index=False)
        else:
            pd.DataFrame({'Info': ['Nenhum trade executado']}).to_excel(writer, sheet_name='Trades Executados', index=False)
        
        # Resumo de performance
        if len(trades_df) > 0:
            summary_data = {
                'Métrica': [
                    'Total de Trades',
                    'Trades Lucrativos',
                    'Trades Prejudiciais',
                    'Taxa de Sucesso (%)',
                    'Lucro Total (%)',
                    'Melhor Trade (%)',
                    'Pior Trade (%)',
                    'Lucro Médio por Trade (%)'
                ],
                'Valor': [
                    len(trades_df),
                    len(trades_df[trades_df['pnl'] > 0]),
                    len(trades_df[trades_df['pnl'] < 0]),
                    len(trades_df[trades_df['pnl'] > 0]) / len(trades_df) * 100 if len(trades_df) > 0 else 0,
                    trades_df['pnl'].sum(),
                    trades_df['pnl'].max(),
                    trades_df['pnl'].min(),
                    trades_df['pnl'].mean()
                ]
            }
        else:
            summary_data = {
                'Métrica': [
                    'Total de Trades',
                    'Trades Lucrativos', 
                    'Trades Prejudiciais',
                    'Taxa de Sucesso (%)',
                    'Lucro Total (%)',
                    'Melhor Trade (%)',
                    'Pior Trade (%)',
                    'Lucro Médio por Trade (%)'
                ],
                'Valor': [0, 0, 0, 0, 0, 0, 0, 0]
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
update_interval = st.sidebar.selectbox("Intervalo (minutos)", [5, 15, 30, 60], index=3)

# Botão para atualizar dados manualmente
if st.sidebar.button("🔄 Atualizar Dados Agora"):
    st.rerun()

# Carregar dados
st.header("📊 Dados do Ethereum")

with st.spinner("Carregando dados do Ethereum..."):
    df = get_ethereum_data()

if df is not None and len(df) > 0:
    # Simular trading
    trading_df = simulate_trading(df, rsi_lower, rsi_upper)
    
    # Filtrar apenas trades executados
    trades_df = trading_df[trading_df['signal'] != 'HOLD'].copy()
    
    # Layout de colunas para métricas
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        current_price = trading_df['price'].iloc[-1]
        st.metric("Preço Atual ETH", f"${current_price:.2f}")
    
    with col2:
        current_rsi = trading_df['rsi'].iloc[-1]
        rsi_color = "red" if current_rsi > 70 else "green" if current_rsi < 30 else "gray"
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
        subplot_titles=('Preço do Ethereum com Sinais de Trading', 'RSI Indicator'),
        vertical_spacing=0.1,
        row_heights=[0.7, 0.3]
    )
    
    # Gráfico de preço
    fig.add_trace(
        go.Scatter(
            x=trading_df.index,
            y=trading_df['price'],
            name='Preço ETH',
            line=dict(color='#00D4AA', width=2)
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
                name='COMPRA',
                marker=dict(color='green', size=12, symbol='triangle-up', line=dict(width=2, color='darkgreen'))
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
                name='VENDA',
                marker=dict(color='red', size=12, symbol='triangle-down', line=dict(width=2, color='darkred'))
            ),
            row=1, col=1
        )
    
    # Gráfico RSI
    fig.add_trace(
        go.Scatter(
            x=trading_df.index,
            y=trading_df['rsi'],
            name='RSI',
            line=dict(color='#FF6B6B', width=2)
        ),
        row=2, col=1
    )
    
    # Linhas de sobrecompra/sobrevenda
    fig.add_hline(y=rsi_upper, line_dash="dash", line_color="red", row=2, col=1, annotation_text=f"Venda ({rsi_upper})")
    fig.add_hline(y=rsi_lower, line_dash="dash", line_color="green", row=2, col=1, annotation_text=f"Compra ({rsi_lower})")
    fig.add_hline(y=50, line_dash="dot", line_color="gray", row=2, col=1)
    
    fig.update_layout(
        height=800, 
        showlegend=True,
        title_text="Ethereum Trading Strategy - RSI",
        template="plotly_dark"
    )
    
    fig.update_xaxes(title_text="Data", row=2, col=1)
    fig.update_yaxes(title_text="Preço (USD)", row=1, col=1)
    fig.update_yaxes(title_text="RSI", row=2, col=1)
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Tabela de trades
    st.header("💼 Trades Executados")
    
    if len(trades_df) > 0:
        display_trades = trades_df[['price', 'rsi', 'signal', 'trade_price', 'pnl']].copy()
        display_trades['timestamp'] = display_trades.index
        display_trades = display_trades[['timestamp', 'price', 'rsi', 'signal', 'trade_price', 'pnl']]
        display_trades.columns = ['Data/Hora', 'Preço ETH', 'RSI', 'Sinal', 'Preço Trade', 'PnL (%)']
        display_trades['PnL (%)'] = display_trades['PnL (%)'].round(2)
        display_trades['Preço ETH'] = display_trades['Preço ETH'].round(2)
        display_trades['Preço Trade'] = display_trades['Preço Trade'].round(2)
        display_trades['RSI'] = display_trades['RSI'].round(2)
        
        # Colorir a coluna PnL
        def color_pnl(val):
            color = 'green' if val > 0 else 'red' if val < 0 else 'gray'
            return f'color: {color}'
        
        styled_df = display_trades.style.applymap(color_pnl, subset=['PnL (%)'])
        st.dataframe(styled_df, use_container_width=True)
        
        # Métricas de performance
        st.header("📊 Performance do Trading")
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            total_pnl = trades_df['pnl'].sum()
            pnl_color = "green" if total_pnl > 0 else "red"
            st.metric("Lucro/Prejuízo Total", f"{total_pnl:.2f}%", delta_color="off")
        
        with col2:
            winning_trades = len(trades_df[trades_df['pnl'] > 0])
            st.metric("Trades Lucrativos", winning_trades)
        
        with col3:
            losing_trades = len(trades_df[trades_df['pnl'] < 0])
            st.metric("Trades Prejudiciais", losing_trades)
        
        with col4:
            best_trade = trades_df['pnl'].max()
            st.metric("Melhor Trade", f"{best_trade:.2f}%")
        
        # Gráfico de performance acumulada
        st.subheader("📈 Performance Acumulada")
        
        if 'cumulative_pnl' in trading_df.columns:
            perf_fig = go.Figure()
            perf_fig.add_trace(go.Scatter(
                x=trading_df.index,
                y=trading_df['cumulative_pnl'],
                name='Performance Acumulada',
                line=dict(color='#00D4AA', width=3)
            ))
            
            perf_fig.update_layout(
                title="Performance Acumulada da Estratégia",
                xaxis_title="Data",
                yaxis_title="Retorno Acumulado (%)",
                template="plotly_dark",
                height=400
            )
            st.plotly_chart(perf_fig, use_container_width=True)
    
    else:
        st.info("ℹ️ Nenhum trade executado ainda com os parâmetros atuais.")
        st.info("💡 Tente ajustar os limites do RSI na sidebar para gerar mais sinais.")
    
    # Exportação para Excel
    st.header("📤 Exportar Dados")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.download_button(
            label="📥 Baixar Relatório Excel Completo",
            data=export_to_excel(trading_df, trades_df),
            file_name=f"ethereum_trading_report_{datetime.now().strftime('%Y%m%d_%H%M')}.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            use_container_width=True
        )
    
    with col2:
        if st.button("🔄 Simular com Novos Dados", use_container_width=True):
            st.rerun()
    
    # Informações adicionais
    st.sidebar.header("ℹ️ Sobre a Estratégia")
    st.sidebar.info("""
    **Estratégia RSI:**
    - RSI < 30: Sinal de COMPRA (sobrevendido)
    - RSI > 70: Sinal de VENDA (sobrecomprado)
    - Fechamento automático de posições
    
    **⚠️ Aviso:**
    Esta é uma ferramenta educacional
    para backtesting. Trading real
    envolve riscos significativos.
    """)
    
    # Status do sistema
    st.sidebar.header("🖥️ Status do Sistema")
    st.sidebar.success(f"✅ Dados carregados: {len(df)} registros")
    st.sidebar.success(f"✅ Período: {df.index[0].strftime('%d/%m/%Y')} - {df.index[-1].strftime('%d/%m/%Y')}")
    
    # Atualização automática
    if auto_update:
        st.sidebar.info(f"⏰ Próxima atualização em {update_interval} minutos")
        time.sleep(update_interval * 60)
        st.rerun()

else:
    st.error("❌ Não foi possível carregar os dados do Ethereum. Tente novamente.")

# Rodapé
st.markdown("---")
st.markdown(
    """
    <div style='text-align: center; color: gray;'>
        <p><strong>⚠️ AVISO LEGAL:</strong> Esta aplicação é para fins educacionais e de demonstração apenas. 
        Não constitui aconselhamento financeiro. Cryptomoedas são investimentos de alto risco.</p>
        <p>Desenvolvido para análise técnica e aprendizado de estratégias de trading.</p>
    </div>
    """,
    unsafe_allow_html=True
)

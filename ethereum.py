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

# Função para obter dados do Ethereum
def get_ethereum_data():
    try:
        # Tentativa 1: CoinGecko API
        st.info("🔄 Obtendo dados da CoinGecko API...")
        url = "https://api.coingecko.com/api/v3/coins/ethereum/market_chart"
        params = {
            'vs_currency': 'usd',
            'days': '30',
            'interval': 'daily'
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
    
    # Preços mais realistas do Ethereum com mais volatilidade
    np.random.seed(42)
    base_price = 3500  # Preço base do ETH
    
    # Criar tendência com alguma volatilidade
    prices = []
    current_price = base_price
    
    for i in range(len(dates)):
        # Adicionar alguma tendência e volatilidade mais pronunciada
        change = np.random.normal(0, 50)  # Mais volatilidade para RSI funcionar melhor
        # Adicionar uma tendência suave
        trend = np.sin(i / 100) * 10
        current_price += change + trend
        current_price = max(2800, min(4200, current_price))  # Manter em range realista
        prices.append(current_price)
    
    df = pd.DataFrame({
        'price': prices
    }, index=dates)
    
    st.success("✅ Dados de exemplo gerados com sucesso!")
    return df

# Função para simular trading com banca de 1000€ e trades de 10€
def simulate_trading(df, rsi_lower=30, rsi_upper=70, rsi_period=14):
    df = df.copy()
    
    # Calcular RSI usando a versão simplificada
    df['rsi'] = calculate_rsi_simple(df['price'], rsi_period)
    
    # Configurações de capital
    initial_capital = 1000.0  # 1000€ de banca inicial
    trade_amount = 10.0       # 10€ por trade
    current_capital = initial_capital
    
    # Inicializar colunas de sinal
    df['signal'] = 'HOLD'
    df['position'] = 0
    df['trade_price'] = 0.0
    df['pnl_percent'] = 0.0
    df['pnl_euros'] = 0.0
    df['position_size_eth'] = 0.0
    df['current_capital'] = current_capital
    df['trade_amount'] = 0.0
    
    position = 0  # 0: sem posição, 1: comprado, -1: vendido
    entry_price = 0
    position_size_eth = 0

    for i in range(len(df)):
        current_rsi = df['rsi'].iloc[i]
        current_price = df['price'].iloc[i]
        
        # Atualizar capital atual
        df.loc[df.index[i], 'current_capital'] = current_capital
        
        # Lógica de trading
        if position == 0 and current_capital >= trade_amount:  # Sem posição e com capital
            if current_rsi < rsi_lower:  # RSI abaixo do limite inferior - COMPRAR
                df.loc[df.index[i], 'signal'] = 'BUY'
                df.loc[df.index[i], 'position'] = 1
                df.loc[df.index[i], 'trade_price'] = current_price
                df.loc[df.index[i], 'trade_amount'] = trade_amount
                
                # Calcular tamanho da posição em ETH
                position_size_eth = trade_amount / current_price
                df.loc[df.index[i], 'position_size_eth'] = position_size_eth
                
                position = 1
                entry_price = current_price
                current_capital -= trade_amount  # Deduzir o valor do trade do capital
                
            elif current_rsi > rsi_upper:  # RSI acima do limite superior - VENDER
                df.loc[df.index[i], 'signal'] = 'SELL_SHORT'
                df.loc[df.index[i], 'position'] = -1
                df.loc[df.index[i], 'trade_price'] = current_price
                df.loc[df.index[i], 'trade_amount'] = trade_amount
                
                # Para venda a descoberto, assumimos que podemos vender 10€ em ETH
                position_size_eth = trade_amount / current_price
                df.loc[df.index[i], 'position_size_eth'] = position_size_eth
                
                position = -1
                entry_price = current_price
                current_capital -= trade_amount  # Margem para short
                
        elif position == 1:  # Posição comprada
            if current_rsi > rsi_upper:  # Fechar posição quando RSI > limite superior
                df.loc[df.index[i], 'signal'] = 'SELL'
                df.loc[df.index[i], 'position'] = 0
                df.loc[df.index[i], 'trade_price'] = current_price
                
                # Calcular PnL
                pnl_percent = (current_price - entry_price) / entry_price * 100
                pnl_euros = position_size_eth * (current_price - entry_price)
                
                df.loc[df.index[i], 'pnl_percent'] = pnl_percent
                df.loc[df.index[i], 'pnl_euros'] = pnl_euros
                df.loc[df.index[i], 'trade_amount'] = trade_amount
                
                # Atualizar capital
                current_capital += trade_amount + pnl_euros  # Devolver capital + lucro/prejuízo
                
                position = 0
                position_size_eth = 0
                
        elif position == -1:  # Posição vendida (short)
            if current_rsi < rsi_lower:  # Fechar posição quando RSI < limite inferior
                df.loc[df.index[i], 'signal'] = 'BUY_COVER'
                df.loc[df.index[i], 'position'] = 0
                df.loc[df.index[i], 'trade_price'] = current_price
                
                # Calcular PnL (invertido para short)
                pnl_percent = (entry_price - current_price) / entry_price * 100
                pnl_euros = position_size_eth * (entry_price - current_price)
                
                df.loc[df.index[i], 'pnl_percent'] = pnl_percent
                df.loc[df.index[i], 'pnl_euros'] = pnl_euros
                df.loc[df.index[i], 'trade_amount'] = trade_amount
                
                # Atualizar capital
                current_capital += trade_amount + pnl_euros  # Devolver margem + lucro/prejuízo
                
                position = 0
                position_size_eth = 0
    
    return df

# Função para exportar para Excel
def export_to_excel(df, trades_df, initial_capital=1000):
    output = io.BytesIO()
    
    with pd.ExcelWriter(output, engine='openpyxl') as writer:
        # Dados completos
        df.to_excel(writer, sheet_name='Dados Completos')
        
        # Trades executados
        if len(trades_df) > 0:
            trades_export = trades_df.copy()
            trades_export['timestamp'] = trades_export.index
            trades_export = trades_export[['timestamp', 'price', 'rsi', 'signal', 'trade_price', 
                                         'trade_amount', 'position_size_eth', 'pnl_percent', 'pnl_euros']]
            trades_export.columns = ['Data/Hora', 'Preço ETH', 'RSI', 'Sinal', 'Preço Trade', 
                                   'Valor Trade (€)', 'Tamanho Posição (ETH)', 'PnL (%)', 'PnL (€)']
            trades_export.to_excel(writer, sheet_name='Trades Executados', index=False)
        else:
            pd.DataFrame({'Info': ['Nenhum trade executado']}).to_excel(writer, sheet_name='Trades Executados', index=False)
        
        # Resumo de performance
        if len(trades_df) > 0:
            final_capital = df['current_capital'].iloc[-1]
            total_pnl_euros = final_capital - initial_capital
            total_return = (total_pnl_euros / initial_capital) * 100
            
            summary_data = {
                'Métrica': [
                    'Capital Inicial (€)',
                    'Capital Final (€)',
                    'Lucro/Prejuízo Total (€)',
                    'Retorno Total (%)',
                    'Total de Trades',
                    'Trades Lucrativos',
                    'Trades Prejudiciais',
                    'Taxa de Sucesso (%)',
                    'Lucro Total (€)',
                    'Prejuízo Total (€)',
                    'Melhor Trade (€)',
                    'Pior Trade (€)',
                    'Trade Médio (€)'
                ],
                'Valor': [
                    initial_capital,
                    final_capital,
                    total_pnl_euros,
                    total_return,
                    len(trades_df),
                    len(trades_df[trades_df['pnl_euros'] > 0]),
                    len(trades_df[trades_df['pnl_euros'] < 0]),
                    len(trades_df[trades_df['pnl_euros'] > 0]) / len(trades_df) * 100 if len(trades_df) > 0 else 0,
                    trades_df[trades_df['pnl_euros'] > 0]['pnl_euros'].sum(),
                    trades_df[trades_df['pnl_euros'] < 0]['pnl_euros'].sum(),
                    trades_df['pnl_euros'].max(),
                    trades_df['pnl_euros'].min(),
                    trades_df['pnl_euros'].mean()
                ]
            }
        else:
            summary_data = {
                'Métrica': [
                    'Capital Inicial (€)',
                    'Capital Final (€)',
                    'Lucro/Prejuízo Total (€)',
                    'Retorno Total (%)',
                    'Total de Trades',
                    'Trades Lucrativos',
                    'Trades Prejudiciais',
                    'Taxa de Sucesso (%)'
                ],
                'Valor': [initial_capital, initial_capital, 0, 0, 0, 0, 0, 0]
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

# Configurações de capital
st.sidebar.header("💰 Configurações de Capital")
initial_capital = st.sidebar.number_input("Capital Inicial (€)", min_value=100, max_value=10000, value=1000, step=100)
trade_amount = st.sidebar.number_input("Valor por Trade (€)", min_value=5, max_value=100, value=10, step=5)

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
    trading_df = simulate_trading(df, rsi_lower, rsi_upper, rsi_period)
    
    # Filtrar apenas trades executados
    trades_df = trading_df[trading_df['signal'] != 'HOLD'].copy()
    
    # Calcular métricas finais
    final_capital = trading_df['current_capital'].iloc[-1]
    total_pnl_euros = final_capital - initial_capital
    total_return = (total_pnl_euros / initial_capital) * 100
    
    # Layout de colunas para métricas
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        current_price = trading_df['price'].iloc[-1]
        st.metric("Preço Atual ETH", f"${current_price:.2f}")
    
    with col2:
        current_rsi = trading_df['rsi'].iloc[-1]
        st.metric("RSI Atual", f"{current_rsi:.2f}")
    
    with col3:
        st.metric("Capital Atual", f"€{final_capital:.2f}")
    
    with col4:
        st.metric("Retorno Total", f"€{total_pnl_euros:.2f}", f"{total_return:.2f}%")
    
    # Métricas de capital e performance
    st.subheader("💰 Performance do Capital")
    
    cap_col1, cap_col2, cap_col3, cap_col4 = st.columns(4)
    
    with cap_col1:
        st.info(f"**Capital Inicial:** €{initial_capital:.2f}")
    
    with cap_col2:
        st.info(f"**Capital Final:** €{final_capital:.2f}")
    
    with cap_col3:
        return_color = "🟢" if total_return > 0 else "🔴" if total_return < 0 else "⚪"
        st.info(f"**Retorno Total:** {return_color} {total_return:.2f}%")
    
    with cap_col4:
        st.info(f"**Total de Trades:** {len(trades_df)}")
    
    # Gráfico de evolução do capital
    st.subheader("📈 Evolução do Capital")
    
    capital_fig = go.Figure()
    
    capital_fig.add_trace(go.Scatter(
        x=trading_df.index,
        y=trading_df['current_capital'],
        name='Capital (€)',
        line=dict(color='#00D4AA', width=3),
        fill='tozeroy',
        fillcolor='rgba(0, 212, 170, 0.1)'
    ))
    
    # Adicionar linha do capital inicial
    capital_fig.add_hline(
        y=initial_capital, 
        line_dash="dash", 
        line_color="white",
        annotation_text=f"Capital Inicial: €{initial_capital:.0f}"
    )
    
    capital_fig.update_layout(
        title="Evolução do Capital de Trading",
        xaxis_title="Data",
        yaxis_title="Capital (€)",
        template="plotly_dark",
        height=400
    )
    
    st.plotly_chart(capital_fig, use_container_width=True)
    
    # Gráfico de preço e RSI
    st.header("📊 Gráfico de Preço e RSI")
    
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
    buy_signals = trading_df[trading_df['signal'].str.contains('BUY')]
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
    sell_signals = trading_df[trading_df['signal'].str.contains('SELL')]
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
        display_trades = trades_df[['price', 'rsi', 'signal', 'trade_price', 
                                  'trade_amount', 'position_size_eth', 'pnl_percent', 'pnl_euros']].copy()
        display_trades['timestamp'] = display_trades.index
        display_trades = display_trades[['timestamp', 'price', 'rsi', 'signal', 'trade_price', 
                                       'trade_amount', 'position_size_eth', 'pnl_percent', 'pnl_euros']]
        display_trades.columns = ['Data/Hora', 'Preço ETH', 'RSI', 'Sinal', 'Preço Trade', 
                                'Valor Trade (€)', 'Tamanho Posição (ETH)', 'PnL (%)', 'PnL (€)']
        
        # Arredondar valores
        display_trades['PnL (%)'] = display_trades['PnL (%)'].round(2)
        display_trades['PnL (€)'] = display_trades['PnL (€)'].round(2)
        display_trades['Preço ETH'] = display_trades['Preço ETH'].round(2)
        display_trades['Preço Trade'] = display_trades['Preço Trade'].round(2)
        display_trades['RSI'] = display_trades['RSI'].round(2)
        display_trades['Tamanho Posição (ETH)'] = display_trades['Tamanho Posição (ETH)'].round(6)
        display_trades['Valor Trade (€)'] = display_trades['Valor Trade (€)'].round(2)
        
        # Colorir a coluna PnL (€)
        def color_pnl_euros(val):
            color = 'green' if val > 0 else 'red' if val < 0 else 'gray'
            return f'color: {color}; font-weight: bold;'
        
        styled_df = display_trades.style.applymap(color_pnl_euros, subset=['PnL (€)'])
        st.dataframe(styled_df, use_container_width=True)
        
    else:
        st.info("ℹ️ Nenhum trade executado ainda com os parâmetros atuais.")
        st.info("💡 Tente ajustar os limites do RSI na sidebar para gerar mais sinais.")
    
    # Exportação para Excel
    st.header("📤 Exportar Dados")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.download_button(
            label="📥 Baixar Relatório Excel Completo",
            data=export_to_excel(trading_df, trades_df, initial_capital),
            file_name=f"ethereum_trading_report_{datetime.now().strftime('%Y%m%d_%H%M')}.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            use_container_width=True
        )
    
    with col2:
        if st.button("🔄 Simular com Novos Dados", use_container_width=True):
            st.rerun()

else:
    st.error("❌ Não foi possível carregar os dados do Ethereum. Tente novamente.")

# Informações na sidebar
st.sidebar.header("💰 Configuração Atual")
st.sidebar.info(f"""
**Capital Inicial:** €{initial_capital:.2f}
**Valor por Trade:** €{trade_amount:.2f}
**Trades Possíveis:** {initial_capital // trade_amount}
**Capital por Trade:** {(trade_amount / initial_capital * 100):.1f}%
""")

st.sidebar.header("ℹ️ Sobre a Estratégia")
st.sidebar.info("""
**Estratégia RSI:**
- RSI < 30: COMPRA (10€)
- RSI > 70: VENDA (10€)
- Fechamento automático

**⚠️ Aviso:**
Ferramenta educacional para backtesting.
Trading real envolve riscos.
""")

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

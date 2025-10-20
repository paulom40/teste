import streamlit as st
import pandas as pd
import numpy as np
import requests
from datetime import datetime, timedelta
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Configuração da página
st.set_page_config(
    page_title="Simulação RSI + MACD - 1000€ Banca",
    page_icon="📈",
    layout="wide"
)

# =============================================================================
# MÓDULO: CÁLCULO DE INDICADORES
# =============================================================================

def calculate_rsi_simple(prices, period=14):
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
        st.error(f"Erro no RSI: {e}")
        return pd.Series([50] * len(prices), index=prices.index)

def calculate_macd(prices, fast=12, slow=26, signal=9):
    try:
        if not isinstance(prices, pd.Series):
            prices = pd.Series(prices)
        ema_fast = prices.ewm(span=fast).mean()
        ema_slow = prices.ewm(span=slow).mean()
        macd_line = ema_fast - ema_slow
        signal_line = macd_line.ewm(span=signal).mean()
        histogram = macd_line - signal_line
        return macd_line, signal_line, histogram
    except Exception as e:
        st.error(f"Erro no MACD: {e}")
        return pd.Series([0] * len(prices), index=prices.index), pd.Series([0] * len(prices), index=prices.index), pd.Series([0] * len(prices), index=prices.index)

def calculate_indicators(df, rsi_period=14, macd_fast=12, macd_slow=26, macd_signal=9):
    df = df.copy()
    df['rsi'] = calculate_rsi_simple(df['price'], rsi_period)
    df['macd'], df['macd_signal'], df['histogram'] = calculate_macd(df['price'], macd_fast, macd_slow, macd_signal)
    return df

# =============================================================================
# MÓDULO: OBTENÇÃO DE DADOS
# =============================================================================

@st.cache_data
def get_real_coin_data(coin_id):
    try:
        url = f"https://api.coingecko.com/api/v3/coins/{coin_id}/market_chart"
        params = {'vs_currency': 'usd', 'days': '90', 'interval': 'daily'}
        response = requests.get(url, params=params, timeout=15)
        if response.status_code == 200:
            data = response.json()
            if 'prices' in data and len(data['prices']) > 0:
                prices = data['prices']
                df = pd.DataFrame(prices, columns=['timestamp', 'price'])
                df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
                df = df.set_index('timestamp')
                df_hourly = df.resample('H').interpolate()
                np.random.seed(abs(hash(coin_id)) % (1 << 32))
                noise = np.random.normal(0, df_hourly['price'].std() * 0.01, len(df_hourly))
                df_hourly['price'] = df_hourly['price'] + noise
                return df_hourly
        return create_realistic_sample_data(coin_id)
    except Exception as e:
        st.warning(f"Erro na API: {e}. Usando dados simulados.")
        return create_realistic_sample_data(coin_id)

def create_realistic_sample_data(coin_id):
    end_date = datetime.now()
    start_date = end_date - timedelta(days=90)
    dates = pd.date_range(start=start_date, end=end_date, freq='H')
    np.random.seed(abs(hash(coin_id)) % (1 << 32))
    if coin_id == 'ethereum':
        base_trend = np.linspace(2800, 3800, len(dates))
    else:
        base_trend = np.linspace(2800, 3800, len(dates))  # Default
    volatility = np.random.normal(0, base_trend.std() * 0.1, len(dates))
    market_cycle = np.sin(np.arange(len(dates)) / 500) * base_trend.std() * 0.2
    short_cycle = np.sin(np.arange(len(dates)) / 50) * base_trend.std() * 0.05
    prices = base_trend + volatility + market_cycle + short_cycle
    prices = np.clip(prices, 2500, 4500)
    return pd.DataFrame({'price': prices}, index=dates)

# =============================================================================
# MÓDULO: SIMULAÇÃO DE TRADING
# =============================================================================

def initialize_trading_columns(df, initial_capital):
    df = df.copy()
    df['signal'] = 'HOLD'
    df['position'] = 0
    df['trade_price'] = 0.0
    df['pnl_percent'] = 0.0
    df['pnl_euros'] = 0.0
    df['position_size'] = 0.0
    df['current_capital'] = initial_capital
    df['trade_amount'] = 0.0
    df['fees'] = 0.0
    df['total_fees'] = 0.0
    return df

def check_daily_trade_limit(i, df, max_trades_per_day):
    trades_today = 0
    if i > 24:
        today_start = df.index[i] - timedelta(hours=24)
        trades_today = len(df.loc[today_start:df.index[i], 'signal'][df.loc[today_start:df.index[i], 'signal'] != 'HOLD'])
    return trades_today < max_trades_per_day

def execute_buy_signal(i, df, trade_amount, fee_rate, current_price):
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
                              macd_fast=12, macd_slow=26, macd_signal=9,
                              initial_capital=1000, trade_amount=10, 
                              fee_rate=0.002, max_trades_per_day=3, trading_hours=True):
    df = calculate_indicators(df, rsi_period, macd_fast, macd_slow, macd_signal)
    df = initialize_trading_columns(df, initial_capital)
    current_capital = initial_capital
    position = 0
    entry_price = 0
    position_size = 0
    total_fees_paid = 0
    for i in range(1, len(df)):
        current_rsi = df['rsi'].iloc[i]
        current_macd = df['macd'].iloc[i]
        current_macd_signal = df['macd_signal'].iloc[i]
        current_price = df['price'].iloc[i]
        current_hour = df.index[i].hour
        df.loc[df.index[i], 'current_capital'] = current_capital
        df.loc[df.index[i], 'total_fees'] = total_fees_paid
        if trading_hours and not (8 <= current_hour <= 22):
            continue
        trades_today_ok = check_daily_trade_limit(i, df, max_trades_per_day)
        prev_macd = df['macd'].iloc[i-1]
        prev_macd_signal = df['macd_signal'].iloc[i-1]
        macd_bullish_cross = (current_macd > current_macd_signal) and (prev_macd <= prev_macd_signal)
        macd_bearish_cross = (current_macd < current_macd_signal) and (prev_macd >= prev_macd_signal)
        if (position == 0 and current_capital >= trade_amount and trades_today_ok):
            if current_rsi < rsi_lower and macd_bullish_cross:
                position, entry_price, position_size, trade_fee = execute_buy_signal(
                    i, df, trade_amount, fee_rate, current_price
                )
                current_capital -= trade_amount
                total_fees_paid += trade_fee
        elif position == 1:
            take_profit = entry_price * 1.05
            stop_loss = entry_price * 0.95
            if (current_rsi > rsi_upper or macd_bearish_cross or 
                current_price >= take_profit or current_price <= stop_loss):
                position, entry_price, position_size, exit_fee, net_pnl_euros = execute_sell_signal(
                    i, df, position_size, entry_price, current_price, fee_rate
                )
                current_capital += trade_amount + net_pnl_euros
                total_fees_paid += exit_fee
    return df

# =============================================================================
# MÓDULO: MÉTRICAS
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
        'volatilidade_anual': risk_metrics.get('Volatilidade Anual', 0),
        'var_95': risk_metrics.get('VaR 95%', 0)
    }
    return metrics

# =============================================================================
# EXECUÇÃO PRINCIPAL
# =============================================================================

def main():
    st.title("🤖 Simulação RSI + MACD - Banca 1000€")
    st.markdown("---")
    
    # Sidebar
    st.sidebar.header("⚙️ Configurações")
    coin_id = st.sidebar.selectbox("Ativo", ['ethereum'], index=0)
    rsi_period = st.sidebar.slider("RSI Período", 5, 30, 14)
    rsi_lower = st.sidebar.slider("RSI Compra (<)", 10, 40, 30)
    rsi_upper = st.sidebar.slider("RSI Venda (>)", 60, 90, 70)
    macd_fast = st.sidebar.slider("MACD Rápido", 5, 20, 12)
    macd_slow = st.sidebar.slider("MACD Lento", 20, 40, 26)
    macd_signal = st.sidebar.slider("MACD Sinal", 5, 15, 9)
    initial_capital = st.sidebar.number_input("Banca Inicial (€)", value=1000, min_value=100)
    trade_amount = st.sidebar.number_input("Stake por Trade (€)", value=10, min_value=5)
    fee_rate = st.sidebar.slider("Fee (%)", 0.1, 1.0, 0.2) / 100
    max_daily_trades = st.sidebar.slider("Max Trades/Dia", 1, 10, 3)
    trading_hours = st.sidebar.checkbox("Horário Mercado (8h-22h)", True)
    
    # Carregar dados
    with st.spinner("Carregando dados..."):
        df = get_real_coin_data(coin_id)
    
    if len(df) > 0:
        # Simular
        trading_df = simulate_realistic_trading(
            df, rsi_lower, rsi_upper, rsi_period, macd_fast, macd_slow, macd_signal,
            initial_capital, trade_amount, fee_rate, max_daily_trades, trading_hours
        )
        trades_df = trading_df[trading_df['signal'] != 'HOLD'].copy()
        risk_metrics = calculate_risk_metrics(trades_df, initial_capital)
        metrics = calculate_performance_metrics(trading_df, initial_capital, trades_df, risk_metrics)
        
        # Métricas principais
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("💰 Preço Atual", f"${trading_df['price'].iloc[-1]:.2f}")
        with col2:
            st.metric("📊 RSI Atual", f"{trading_df['rsi'].iloc[-1]:.2f}")
        with col3:
            st.metric("💳 Banca Final", f"€{metrics['final_capital']:.2f}")
        with col4:
            st.metric("📈 Retorno", f"{metrics['total_return']:.2f}%")
        
        # Gráfico de Capital
        fig_capital = go.Figure()
        fig_capital.add_trace(go.Scatter(x=trading_df.index, y=trading_df['current_capital'], 
                                         line=dict(color='#00D4AA', width=3), fill='tozeroy'))
        fig_capital.add_hline(y=initial_capital, line_dash="dash", line_color="gray")
        fig_capital.update_layout(title="Evolução da Banca", xaxis_title="Data", yaxis_title="€", template="plotly_dark")
        st.plotly_chart(fig_capital, use_container_width=True)
        
        # Análise Técnica
        fig = make_subplots(rows=3, cols=1, subplot_titles=('Preço', 'RSI', 'MACD'))
        fig.add_trace(go.Scatter(x=trading_df.index, y=trading_df['price'], name='Preço'), row=1, col=1)
        fig.add_trace(go.Scatter(x=trading_df.index, y=trading_df['rsi'], name='RSI'), row=2, col=1)
        fig.add_hline(y=rsi_upper, line_dash="dash", line_color="red", row=2, col=1)
        fig.add_hline(y=rsi_lower, line_dash="dash", line_color="green", row=2, col=1)
        fig.add_trace(go.Scatter(x=trading_df.index, y=trading_df['macd'], name='MACD'), row=3, col=1)
        fig.add_trace(go.Scatter(x=trading_df.index, y=trading_df['macd_signal'], name='Sinal'), row=3, col=1)
        fig.add_trace(go.Bar(x=trading_df.index, y=trading_df['histogram'], name='Histograma'), row=3, col=1)
        fig.update_layout(height=800, template="plotly_dark")
        st.plotly_chart(fig, use_container_width=True)
        
        # Tabela de Trades
        if len(trades_df) > 0:
            st.subheader("💼 Trades Executados")
            display_trades = trades_df[['signal', 'trade_price', 'pnl_euros']].round(2)
            st.dataframe(display_trades)
        else:
            st.info("ℹ️ Nenhum trade gerado. Ajuste os parâmetros para mais sensibilidade!")
        
        # Métricas de Risco
        st.subheader("⚠️ Métricas de Risco")
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Win Rate", f"{metrics['win_rate']:.1f}%")
        with col2:
            st.metric("Sharpe Ratio", f"{metrics['sharpe_ratio']:.2f}")
        with col3:
            st.metric("Max Drawdown", f"€{metrics['max_drawdown']:.2f}")
        with col4:
            st.metric("Total Fees", f"€{metrics['total_fees']:.2f}")
    
    st.markdown("---")
    st.info("🚨 **Aviso**: Simulação educacional. Trading real tem riscos!")

if __name__ == "__main__":
    main()

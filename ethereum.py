import io

if not st.session_state.trades.empty:
    st.subheader("💼 Trades Executados")
    st.dataframe(st.session_state.trades.round(4))

    # Análise de risco
    trades = st.session_state.trades.copy()
    trades['Período'] = trades['time'].apply(lambda dt: 'Manhã' if dt.hour < 12 else 'Tarde' if dt.hour < 18 else 'Noite')
    total_trades = len(trades)
    total_pnl = trades['pnl'].sum()
    win_rate = len(trades[trades['pnl'] > 0]) / total_trades * 100 if total_trades > 0 else 0
    avg_pnl = trades['pnl'].mean()
    std_pnl = trades['pnl'].std()
    sharpe_ratio = avg_pnl / std_pnl if std_pnl != 0 else 0
    trades['capital'] = 1000 + trades['pnl'].cumsum()
    drawdown = (trades['capital'].cummax() - trades['capital']).max()

    metrics_df = pd.DataFrame({
        'Métrica': [
            'Capital Final', 'Lucro Líquido', 'Total de Trades', 'Win Rate (%)',
            'Lucro Médio por Trade', 'Desvio Padrão do PnL', 'Índice de Sharpe', 'Drawdown Máximo'
        ],
        'Valor': [
            st.session_state.capital, total_pnl, total_trades, round(win_rate, 2),
            round(avg_pnl, 2), round(std_pnl, 2), round(sharpe_ratio, 2), round(drawdown, 2)
        ]
    })

    # Gráficos técnicos
    df_chart = df[['rsi', 'macd', 'macd_signal']].copy()
    df_chart.reset_index(inplace=True)

    # Exportação
    output = io.BytesIO()
    with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
        wb = writer.book

        # Trades
        trades.to_excel(writer, index=False, sheet_name='Trades')
        sheet = writer.sheets['Trades']
        for i, col in enumerate(trades.columns):
            sheet.set_column(i, i, 18)

        # Métricas
        metrics_df.to_excel(writer, index=False, sheet_name='Métricas')
        sheet = writer.sheets['Métricas']
        sheet.set_column(0, 1, 25)

        # Gráfico Técnico
        df_chart.to_excel(writer, index=False, sheet_name='Gráfico Técnico')
        sheet = writer.sheets['Gráfico Técnico']
        chart_rsi = wb.add_chart({'type': 'line'})
        chart_rsi.add_series({
            'name': 'RSI',
            'categories': f'=Gráfico Técnico!$A$2:$A${len(df_chart)+1}',
            'values': f'=Gráfico Técnico!$B$2:$B${len(df_chart)+1}',
        })
        chart_rsi.set_title({'name': 'RSI'})
        chart_rsi.set_y_axis({'min': 0, 'max': 100})
        sheet.insert_chart('F2', chart_rsi)

        chart_macd = wb.add_chart({'type': 'line'})
        chart_macd.add_series({
            'name': 'MACD',
            'categories': f'=Gráfico Técnico!$A$2:$A${len(df_chart)+1}',
            'values': f'=Gráfico Técnico!$C$2:$C${len(df_chart)+1}',
        })
        chart_macd.add_series({
            'name': 'Sinal',
            'categories': f'=Gráfico Técnico!$A$2:$A${len(df_chart)+1}',
            'values': f'=Gráfico Técnico!$D$2:$D${len(df_chart)+1}',
        })
        chart_macd.set_title({'name': 'MACD vs Sinal'})
        sheet.insert_chart('F20', chart_macd)

        # Distribuição de PnL
        pnl_counts = trades['pnl'].apply(lambda x: 'Lucro' if x > 0 else 'Prejuízo').value_counts().reset_index()
        pnl_counts.columns = ['Resultado', 'Quantidade']
        pnl_counts.to_excel(writer, index=False, sheet_name='Distribuição PnL')
        sheet = writer.sheets['Distribuição PnL']
        chart = wb.add_chart({'type': 'column'})
        chart.add_series({
            'name': 'Distribuição de PnL',
            'categories': f'=Distribuição PnL!$A$2:$A${len(pnl_counts)+1}',
            'values': f'=Distribuição PnL!$B$2:$B${len(pnl_counts)+1}',
        })
        chart.set_title({'name': 'Lucros vs Prejuízos'})
        sheet.insert_chart('D2', chart)

        # Sinais
        signal_counts = trades['signal'].value_counts().reset_index()
        signal_counts.columns = ['Sinal', 'Quantidade']
        signal_counts.to_excel(writer, index=False, sheet_name='Sinais')
        sheet = writer.sheets['Sinais']
        chart = wb.add_chart({'type': 'pie'})
        chart.add_series({
            'name': 'Sinais',
            'categories': f'=Sinais!$A$2:$A${len(signal_counts)+1}',
            'values': f'=Sinais!$B$2:$B${len(signal_counts)+1}',
        })
        chart.set_title({'name': 'Distribuição de Sinais'})
        sheet.insert_chart('D2', chart)

        # Segmentação Horária
        period_counts = trades['Período'].value_counts().reset_index()
        period_counts.columns = ['Período', 'Quantidade']
        period_counts.to_excel(writer, index=False, sheet_name='Segmentação Horária')
        sheet = writer.sheets['Segmentação Horária']
        chart = wb.add_chart({'type': 'column'})
        chart.add_series({
            'name': 'Trades por Período',
            'categories': f'=Segmentação Horária!$A$2:$A${len(period_counts)+1}',
            'values': f'=Segmentação Horária!$B$2:$B${len(period_counts)+1}',
        })
        chart.set_title({'name': 'Distribuição por Horário'})
        sheet.insert_chart('D2', chart)

        writer.save()

    st.download_button(
        label="📥 Exportar para Excel com Segmentação e Gráficos",
        data=output.getvalue(),
        file_name="trades_forex_segmentado.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
    )

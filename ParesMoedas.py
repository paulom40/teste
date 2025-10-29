import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import io

# Limites de volatilidade por par
LIMITE_VOLATILIDADE = {
    "EUR/USD": 0.0020,
    "GBP/USD": 0.0025,
    "USD/JPY": 0.25,
    "AUD/USD": 0.0022
}

# Parâmetros simulados
PARES = {
    "EUR/USD": {"base": 1.10, "vol": 0.005},
    "GBP/USD": {"base": 1.26, "vol": 0.006},
    "USD/JPY": {"base": 148.0, "vol": 0.004},
    "AUD/USD": {"base": 0.66, "vol": 0.007}
}

# Simula candles diários
def simular_high_low(par, base, vol):
    dias = [datetime.now().date() - timedelta(days=i) for i in range(6, -1, -1)]
    dados = []
    for dia in dias:
        variacao = np.random.normal(0, vol)
        high = base * (1 + abs(variacao))
        low = base * (1 - abs(variacao))
        volatilidade = high - low
        alerta = volatilidade > LIMITE_VOLATILIDADE[par]
        dados.append({
            "data": dia,
            "par": par,
            "high": round(high, 5),
            "low": round(low, 5),
            "volatilidade": round(volatilidade, 5),
            "alerta_volatilidade": "Sim" if alerta else "Não"
        })
    return pd.DataFrame(dados)

# Junta todos os pares
df_volatilidade = pd.concat([
    simular_high_low(par, info["base"], info["vol"])
    for par, info in PARES.items()
])

# Inicializa logs técnicos
if "logs_tecnicos" not in st.session_state:
    st.session_state.logs_tecnicos = []

# Painel de destaque
st.subheader("🚨 Pares com Volatilidade Diária Elevada")
pares_alerta = df_volatilidade[df_volatilidade["alerta_volatilidade"] == "Sim"]["par"].unique()

if len(pares_alerta) > 0:
    for par in pares_alerta:
        df_par = df_volatilidade[df_volatilidade["par"] == par]
        st.warning(f"{par}: {df_par['volatilidade'].max():.5f} de volatilidade máxima")
else:
    st.info("Nenhum par excedeu o limite de volatilidade nos últimos 7 dias.")

# Gráfico de volatilidade
st.subheader("📈 Gráfico de Volatilidade Diária (Apenas Alertas)")
fig, ax = plt.subplots(figsize=(10, 6))
cores = {"EUR/USD": "blue", "GBP/USD": "green", "USD/JPY": "red", "AUD/USD": "purple"}

for par in pares_alerta:
    df_par = df_volatilidade[df_volatilidade["par"] == par]
    ax.plot(df_par["data"], df_par["volatilidade"], label=par, color=cores[par])

ax.set_title("Volatilidade Diária por Par (com alerta)")
ax.set_xlabel("Data")
ax.set_ylabel("Volatilidade")
ax.legend()
ax.grid(True)
st.pyplot(fig)

# Botão para iniciar trade automático
st.subheader("🤖 Iniciar Trade Automático")
if st.button("🚀 Executar Trades com Volatilidade Alta"):
    def executar_trades_automaticos():
        if st.session_state.get("pausa_operacoes", False):
            st.info("⏸️ Operações pausadas com base na gestão de risco.")
            return

        stake = st.session_state.get("stake_valor", 10)
        st.write(f"💸 Executando trade com stake de €{stake}")

        for pair in PARES.keys():
            df_hoje = df_volatilidade[
                (df_volatilidade["par"] == pair) &
                (df_volatilidade["data"] == datetime.now().date())
            ]

            if df_hoje.empty or df_hoje["alerta_volatilidade"].values[0] != "Sim":
                st.info(f"{pair}: volatilidade insuficiente hoje, trade ignorado.")
                continue

            entrada_valida = np.random.choice([True, False], p=[0.7, 0.3])
            risco_pct = 0.01
            retorno_pct = 0.02
            risco_estimado = round(stake * risco_pct, 2)
            lucro_estimado = round(stake * retorno_pct, 2)
            status = np.random.choice(["Ativo", "Fechado"], p=[0.4, 0.6])

            if entrada_valida:
                st.success(f"{pair}: trade executado com volatilidade {df_hoje['volatilidade'].values[0]:.5f}")
                log = {
                    "pair": pair,
                    "data": str(datetime.now().date()),
                    "volatilidade_dia": df_hoje["volatilidade"].values[0],
                    "alerta_volatilidade": "Sim",
                    "trade_executado": "Sim",
                    "status": status,
                    "log_stake": stake,
                    "log_risco_estimado": risco_estimado,
                    "log_lucro_estimado": lucro_estimado
                }
            else:
                st.warning(f"{pair}: condições de entrada não atendidas.")
                log = {
                    "pair": pair,
                    "data": str(datetime.now().date()),
                    "volatilidade_dia": df_hoje["volatilidade"].values[0],
                    "alerta_volatilidade": "Sim",
                    "trade_executado": "Não",
                    "status": "Fechado",
                    "log_stake": stake,
                    "log_risco_estimado": 0,
                    "log_lucro_estimado": 0
                }

            st.session_state.logs_tecnicos.append(log)

    executar_trades_automaticos()

# Filtros interativos
st.subheader("🔍 Filtrar Trades Registrados")
if st.session_state.logs_tecnicos:
    df_logs = pd.DataFrame(st.session_state.logs_tecnicos)
    df_logs["Data"] = pd.to_datetime(df_logs["data"]).dt.date

    datas = sorted(df_logs["Data"].unique())
    pares = sorted(df_logs["pair"].unique())
    status_opcoes = sorted(df_logs["status"].unique())

    data_selecionada = st.selectbox("📅 Selecionar Data", options=datas)
    par_selecionado = st.selectbox("💱 Selecionar Par", options=["Todos"] + pares)
    status_selecionado = st.selectbox("🔄 Selecionar Status", options=["Todos"] + status_opcoes)

    df_filtrado = df_logs[df_logs["Data"] == data_selecionada]
    if par_selecionado != "Todos":
        df_filtrado = df_filtrado[df_filtrado["pair"] == par_selecionado]
    if status_selecionado != "Todos":
        df_filtrado = df_filtrado[df_filtrado["status"] == status_selecionado]

    df_exibir = df_filtrado[["Data", "pair", "volatilidade_dia", "alerta_volatilidade", "trade_executado", "status", "log_stake", "log_risco_estimado", "log_lucro_estimado"]]
    df_exibir.columns = ["Data", "Par", "Volatilidade", "Alerta", "Executado", "Status", "Stake", "Risco (€)", "Lucro (€)"]
    st.dataframe(df_exibir, use_container_width=True)

    # Exportar Excel com gráfico e resumo
    df_grafico = df_volatilidade[df_volatilidade["alerta_volatilidade"] == "Sim"][["data", "par", "volatilidade"]]
    df_resumo = pd.DataFrame()
    total_trades = len(df_logs)
    executados = df_logs[df_logs["trade_executado"] == "Sim"]
    ativos = df_logs[df_logs["status"] == "Ativo"]
    fechados = df_logs[df_logs["status"] == "Fechado"]
    media_volatilidade = round(df_logs["volatilidade_dia"].mean(), 5)
    resumo = {
        "Total de Trades": total_trades,
        "Trades Executados": len(executados),
        "Trades Ativos": len(ativos),
        "Trades Fechados": len(fechados),
        "Média de Volatilidade": media,
        resumo = {
        "Total de Trades": total_trades,
        "Trades Executados": len(executados),
        "Trades Ativos": len(ativos),
        "Trades Fechados": len(fechados),
        "Média de Volatilidade": media_volatilidade
    }
    df_resumo = pd.DataFrame(list(resumo.items()), columns=["Indicador", "Valor"])

    # Lucro acumulado
    df_lucro = df_logs[df_logs["trade_executado"] == "Sim"].copy()
    df_lucro["Lucro"] = df_lucro["log_stake"].apply(lambda x: 0.10 if x == 5 else 0.20)
    lucro_diario = df_lucro.groupby("Data")["Lucro"].sum().reset_index()
    lucro_diario["Lucro Acumulado"] = lucro_diario["Lucro"].cumsum()
    lucro_total = round(df_lucro["Lucro"].sum(), 2)

    # Gestão de risco com limites configuráveis
    st.subheader("⚙️ Configuração de Limites de Risco")
    meta_lucro = st.number_input("🎯 Meta de Lucro (€)", min_value=10, max_value=1000, value=100, step=10)
    limite_perda = st.number_input("📉 Limite de Perda (€)", min_value=-500, max_value=0, value=0, step=10)
    st.session_state.meta_lucro_config = meta_lucro
    st.session_state.limite_perda_config = limite_perda

    if lucro_total >= meta_lucro:
        st.success(f"🎯 Meta de lucro (€{meta_lucro}) atingida! Operações pausadas.")
        st.session_state.pausa_operacoes = True
    elif lucro_total <= limite_perda:
        st.error(f"📉 Lucro abaixo do limite (€{limite_perda}). Operações pausadas.")
        st.session_state.pausa_operacoes = True
    else:
        st.session_state.pausa_operacoes = False

    # Escolha manual de stake
    st.subheader("🎛️ Escolher Stake Manual")
    stake_manual = st.radio("Seleciona o valor do stake:", options=["Automático", "€5", "€10"])
    if stake_manual == "€5":
        st.session_state.stake_valor = 5
        st.success("Stake manual definido para €5.")
    elif stake_manual == "€10":
        st.session_state.stake_valor = 10
        st.success("Stake manual definido para €10.")
    else:
        st.info(f"Stake automático mantido: €{st.session_state.get('stake_valor', 10)}")

    # KPIs
    st.subheader("📌 Resumo Estatístico dos Trades")
    col1, col2, col3 = st.columns(3)
    col1.metric("📊 Total de Trades", total_trades)
    col2.metric("✅ Executados", len(executados))
    col3.metric("📈 Média Volatilidade", media_volatilidade)
    col4, col5 = st.columns(2)
    col4.metric("🔄 Ativos", len(ativos))
    col5.metric("📥 Fechados", len(fechados))
    st.metric("💹 Lucro Total", f"€{lucro_total}")

    # Gráfico de lucro acumulado
    st.subheader("📊 Lucro Acumulado por Dia")
    fig_lucro, ax_lucro = plt.subplots(figsize=(10, 5))
    ax_lucro.plot(lucro_diario["Data"], lucro_diario["Lucro Acumulado"], marker="o", color="green")
    ax_lucro.set_title("Lucro Acumulado por Dia")
    ax_lucro.set_xlabel("Data")
    ax_lucro.set_ylabel("Lucro (€)")
    ax_lucro.grid(True)
    st.pyplot(fig_lucro)

    # Exportar tudo para Excel
    export_buffer = io.BytesIO()
    with pd.ExcelWriter(export_buffer, engine="xlsxwriter") as writer:
        df_exibir.to_excel(writer, index=False, sheet_name="Trades Filtrados")
        df_logs.to_excel(writer, index=False, sheet_name="Logs Técnicos")
        df_grafico.to_excel(writer, index=False, sheet_name="Volatilidade Gráfico")
        df_resumo.to_excel(writer, index=False, sheet_name="Resumo Estatístico")
        lucro_diario.to_excel(writer, index=False, sheet_name="Lucro Acumulado")

        workbook = writer.book

        # Gráfico de volatilidade
        worksheet_vol = writer.sheets["Volatilidade Gráfico"]
        chart_vol = workbook.add_chart({'type': 'line'})
        for i, par in enumerate(df_grafico["par"].unique()):
            df_par = df_grafico[df_grafico["par"] == par].reset_index(drop=True)
            chart_vol.add_series({
                'name': par,
                'categories': ['Volatilidade Gráfico', 1, 0, len(df_par), 0],
                'values':     ['Volatilidade Gráfico', 1, 2, len(df_par), 2],
            })
        chart_vol.set_title({'name': 'Volatilidade Diária com Alertas'})
        chart_vol.set_x_axis({'name': 'Data'})
        chart_vol.set_y_axis({'name': 'Volatilidade'})
        worksheet_vol.insert_chart('E2', chart_vol)

        # Gráfico de lucro acumulado
        worksheet_lucro = writer.sheets["Lucro Acumulado"]
        chart_lucro = workbook.add_chart({'type': 'line'})
        chart_lucro.add_series({
            'name': 'Lucro Acumulado',
            'categories': ['Lucro Acumulado', 1, 0, len(lucro_diario), 0],
            'values':     ['Lucro Acumulado', 1, 2, len(lucro_diario), 2],
        })
        chart_lucro.set_title({'name': 'Lucro Acumulado por Dia'})
        chart_lucro.set_x_axis({'name': 'Data'})
        chart_lucro.set_y_axis({'name': 'Lucro (€)'})
        worksheet_lucro.insert_chart('E2', chart_lucro)

    st.download_button(
        label="📥 Exportar Painel Completo para Excel",
        data=export_buffer.getvalue(),
        file_name="painel_trades_completo.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
    )
else:
    st.info("Nenhum trade registrado ainda.")

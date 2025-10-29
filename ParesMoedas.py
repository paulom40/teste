import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import io

st.set_page_config(page_title="Painel de Trades", layout="wide")

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

df_volatilidade = pd.concat([
    simular_high_low(par, info["base"], info["vol"])
    for par, info in PARES.items()
])
if "logs_tecnicos" not in st.session_state:
    st.session_state.logs_tecnicos = []

st.title("📊 Painel de Monitoramento de Trades")

st.subheader("🚨 Pares com Volatilidade Diária Elevada")
pares_alerta = df_volatilidade[df_volatilidade["alerta_volatilidade"] == "Sim"]["par"].unique()
if len(pares_alerta) > 0:
    for par in pares_alerta:
        df_par = df_volatilidade[df_volatilidade["par"] == par]
        st.warning(f"{par}: {df_par['volatilidade'].max():.5f} de volatilidade máxima")
else:
    st.info("Nenhum par excedeu o limite de volatilidade nos últimos 7 dias.")
st.subheader("📈 Gráfico de Volatilidade Diária (com alertas)")
fig, ax = plt.subplots(figsize=(10, 6))
for par in pares_alerta:
    df_par = df_volatilidade[df_volatilidade["par"] == par]
    ax.plot(df_par["data"], df_par["volatilidade"], label=par)
ax.set_title("Volatilidade Diária por Par")
ax.set_xlabel("Data")
ax.set_ylabel("Volatilidade")
ax.legend()
ax.grid(True)
st.pyplot(fig)
st.sidebar.header("⚙️ Configurações")
stake_manual = st.sidebar.radio("Stake:", ["Automático", "€5", "€10"])
meta_lucro = st.sidebar.number_input("🎯 Meta de Lucro (€)", min_value=10, max_value=1000, value=100, step=10)
limite_perda = st.sidebar.number_input("📉 Limite de Perda (€)", min_value=-500, max_value=0, value=0, step=10)

if stake_manual == "€5":
    st.session_state.stake_valor = 5
elif stake_manual == "€10":
    st.session_state.stake_valor = 10
else:
    st.session_state.stake_valor = 10  # padrão automático
st.subheader("🤖 Executar Trades com Volatilidade Alta")
if st.button("🚀 Iniciar Execução"):
    for pair in PARES.keys():
        df_hoje = df_volatilidade[
            (df_volatilidade["par"] == pair) &
            (df_volatilidade["data"] == datetime.now().date())
        ]
        if df_hoje.empty or df_hoje["alerta_volatilidade"].values[0] != "Sim":
            continue

        stake = st.session_state.get("stake_valor", 10)
        risco = round(stake * 0.01, 2)
        lucro = round(stake * 0.02, 2)
        status = np.random.choice(["Ativo", "Fechado"], p=[0.4, 0.6])
        executado = np.random.choice(["Sim", "Não"], p=[0.7, 0.3])

        st.session_state.logs_tecnicos.append({
            "data": str(datetime.now().date()),
            "pair": pair,
            "volatilidade_dia": df_hoje["volatilidade"].values[0],
            "alerta_volatilidade": "Sim",
            "trade_executado": executado,
            "status": status,
            "log_stake": stake,
            "log_risco_estimado": risco if executado == "Sim" else 0,
            "log_lucro_estimado": lucro if executado == "Sim" else 0
        })
if st.session_state.logs_tecnicos:
    df_logs = pd.DataFrame(st.session_state.logs_tecnicos)
    df_logs["Data"] = pd.to_datetime(df_logs["data"]).dt.date
    df_logs["Lucro"] = df_logs["log_stake"].apply(lambda x: 0.10 if x == 5 else 0.20)
    df_logs = df_logs[df_logs["trade_executado"] == "Sim"]
    lucro_total = round(df_logs["Lucro"].sum(), 2)
    lucro_diario = df_logs.groupby("Data")["Lucro"].sum().reset_index()
    lucro_diario["Lucro Acumulado"] = lucro_diario["Lucro"].cumsum()

    st.subheader("📌 Indicadores")
    col1, col2, col3 = st.columns(3)
    col1.metric("Trades Executados", len(df_logs))
    col2.metric("Lucro Total", f"€{lucro_total}")
    col3.metric("Stake Atual", f"€{st.session_state.stake_valor}")

    st.subheader("📊 Lucro Acumulado")
    fig2, ax2 = plt.subplots()
    ax2.plot(lucro_diario["Data"], lucro_diario["Lucro Acumulado"], marker="o")
    ax2.set_title("Lucro Acumulado por Dia")
    ax2.set_xlabel("Data")
    ax2.set_ylabel("Lucro (€)")
    st.pyplot(fig2)

    st.subheader("📋 Tabela de Trades Ativos e Concluídos")

if "logs_tecnicos" in st.session_state and st.session_state.logs_tecnicos:
    df_logs = pd.DataFrame(st.session_state.logs_tecnicos)
    df_logs["Data"] = pd.to_datetime(df_logs["data"]).dt.date

    # Separar por status
    df_ativos = df_logs[df_logs["status"] == "Ativo"]
    df_fechados = df_logs[df_logs["status"] == "Fechado"]

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("### ✅ Trades Ativos")
        if not df_ativos.empty:
            st.dataframe(df_ativos[["Data", "pair", "volatilidade_dia", "log_stake", "log_lucro_estimado"]].rename(columns={
                "pair": "Par",
                "volatilidade_dia": "Volatilidade",
                "log_stake": "Stake (€)",
                "log_lucro_estimado": "Lucro Estimado (€)"
            }), use_container_width=True)
        else:
            st.info("Nenhum trade ativo no momento.")

    with col2:
        st.markdown("### 📦 Trades Concluídos")
        if not df_fechados.empty:
            st.dataframe(df_fechados[["Data", "pair", "volatilidade_dia", "log_stake", "log_lucro_estimado"]].rename(columns={
                "pair": "Par",
                "volatilidade_dia": "Volatilidade",
                "log_stake": "Stake (€)",
                "log_lucro_estimado": "Lucro Estimado (€)"
            }), use_container_width=True)
        else:
            st.info("Nenhum trade concluído ainda.")
else:
    st.info("Nenhum trade registrado ainda.")

    
    # Exportar Excel
    export_buffer = io.BytesIO()
    with pd.ExcelWriter(export_buffer, engine="xlsxwriter") as writer:
        df_logs.to_excel(writer, index=False, sheet_name="Logs Técnicos")
        lucro_diario.to_excel(writer, index=False, sheet_name="Lucro Acumulado")
        worksheet = writer.sheets["Lucro Acumulado"]
        chart = writer.book.add_chart({'type': 'line'})
        chart.add_series({
            'name': 'Lucro Acumulado',
            'categories': ['Lucro Acumulado', 1, 0, len(lucro_diario), 0],
            'values': ['Lucro Acumulado', 1, 1, len(lucro_diario), 1],
        })
        chart.set_title({'name': 'Lucro Acumulado por Dia'})
        chart.set_x_axis({'name': 'Data'})
        chart.set_y_axis({'name': 'Lucro (€)'})
        worksheet.insert_chart('E2', chart)

    st.download_button(
        label="📥 Exportar para Excel",
        data=export_buffer.getvalue(),
        file_name="painel_trades.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
    )

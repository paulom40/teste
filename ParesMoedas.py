import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import io

st.subheader("🤖 Iniciar Trade Automático")

if st.button("🚀 Executar Trades com Volatilidade Alta"):
    executar_trades_automaticos()


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
def executar_trades_automaticos():
    st.success("🔄 Iniciando execução automática de trades...")

    for pair in PARES.keys():
        df_hoje = df_volatilidade[
            (df_volatilidade["par"] == pair) &
            (df_volatilidade["data"] == datetime.now().date())
        ]

        if df_hoje.empty or df_hoje["alerta_volatilidade"].values[0] != "Sim":
            st.info(f"{pair}: volatilidade insuficiente hoje, trade ignorado.")
            continue

        # Simula condições de entrada (substitui por tua lógica real)
        entrada_valida = np.random.choice([True, False], p=[0.7, 0.3])

        if entrada_valida:
            st.success(f"{pair}: trade executado com volatilidade {df_hoje['volatilidade'].values[0]:.5f}")
            log = {
                "pair": pair,
                "data": str(datetime.now().date()),
                "volatilidade_dia": df_hoje["volatilidade"].values[0],
                "alerta_volatilidade": "Sim",
                "trade_executado": "Sim"
            }
        else:
            st.warning(f"{pair}: condições de entrada não atendidas.")
            log = {
                "pair": pair,
                "data": str(datetime.now().date()),
                "volatilidade_dia": df_hoje["volatilidade"].values[0],
                "alerta_volatilidade": "Sim",
                "trade_executado": "Não"
            }

        st.session_state.logs_tecnicos.append(log)


# Junta todos os pares
df_volatilidade = pd.concat([
    simular_high_low(par, info["base"], info["vol"])
    for par, info in PARES.items()
])
st.subheader("🚨 Pares com Volatilidade Diária Elevada")
pares_alerta = df_volatilidade[df_volatilidade["alerta_volatilidade"] == "Sim"]["par"].unique()

if len(pares_alerta) > 0:
    for par in pares_alerta:
        df_par = df_volatilidade[df_volatilidade["par"] == par]
        st.warning(f"{par}: {df_par['volatilidade'].max():.5f} de volatilidade máxima")
else:
    st.info("Nenhum par excedeu o limite de volatilidade nos últimos 7 dias.")
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
output = io.BytesIO()
with pd.ExcelWriter(output, engine="xlsxwriter") as writer:
    df_volatilidade.to_excel(writer, index=False, sheet_name="Volatilidade Diária")
    workbook = writer.book
    worksheet = writer.sheets["Volatilidade Diária"]

    chart = workbook.add_chart({'type': 'line'})
    for i, par in enumerate(pares_alerta):
        df_par = df_volatilidade[df_volatilidade["par"] == par].reset_index(drop=True)
        chart.add_series({
            'name': par,
            'categories': ['Volatilidade Diária', i * 7 + 1, 0, i * 7 + 7, 0],
            'values':     ['Volatilidade Diária', i * 7 + 1, 3, i * 7 + 7, 3],
        })

    chart.set_title({'name': 'Volatilidade Diária com Alertas'})
    chart.set_x_axis({'name': 'Data'})
    chart.set_y_axis({'name': 'Volatilidade'})
    worksheet.insert_chart('G2', chart)
st.download_button(
    label="📥 Baixar Excel com Volatilidade",
    data=output.getvalue(),
    file_name="volatilidade_diaria_alertas.xlsx",
    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
)
for pair in PARES.keys():
    df_hoje = df_volatilidade[
        (df_volatilidade["par"] == pair) &
        (df_volatilidade["data"] == datetime.now().date())
    ]

    if df_hoje.empty or df_hoje["alerta_volatilidade"].values[0] != "Sim":
        st.info(f"{pair}: volatilidade insuficiente hoje, trade ignorado.")
        continue

    # Aqui entra tua lógica de entrada no trade
    st.success(f"{pair}: volatilidade OK, pronto para executar trade.")
if "logs_tecnicos" not in st.session_state:
    st.session_state.logs_tecnicos = []

for pair in PARES.keys():
    df_hoje = df_volatilidade[
        (df_volatilidade["par"] == pair) &
        (df_volatilidade["data"] == datetime.now().date())
    ]
    if df_hoje.empty:
        continue

    log = {
        "pair": pair,
        "data": str(datetime.now().date()),
        "volatilidade_dia": df_hoje["volatilidade"].values[0],
        "alerta_volatilidade": df_hoje["alerta_volatilidade"].values[0]
    }
    st.session_state.logs_tecnicos.append(log)

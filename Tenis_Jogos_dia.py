import streamlit as st
import requests
import pandas as pd
from datetime import date

st.set_page_config(page_title="Jogos de Tênis do Dia", layout="centered")
st.title("🎾 Jogos de Tênis do Dia (ATP & WTA)")

HEADERS = {
    "User-Agent": "Mozilla/5.0",
    "X-Requested-With": "XMLHttpRequest"
}

def obter_jogos_do_dia():
    hoje = date.today().strftime("%Y%m%d")

    url = (
        "https://www.flashscore.com/x/feed/"
        f"f_2_{hoje}_1_en_1"
    )

    r = requests.get(url, headers=HEADERS, timeout=15)
    r.raise_for_status()

    linhas = r.text.split("\n")
    jogos = []

    for linha in linhas:
        if linha.startswith("~"):
            partes = linha.split("¬")

            dados = {p.split("÷")[0]: p.split("÷")[1]
                     for p in partes if "÷" in p}

            if "AD" in dados and "AE" in dados:
                jogos.append({
                    "data": date.today().isoformat(),
                    "jogador1": dados.get("AD"),
                    "jogador2": dados.get("AE"),
                    "horario": dados.get("HH", ""),
                    "torneio": dados.get("CT", ""),
                    "tour": "ATP/WTA"
                })

    return pd.DataFrame(jogos)

if st.button("🔄 Carregar jogos do dia"):
    with st.spinner("Buscando jogos..."):
        try:
            df = obter_jogos_do_dia()

            if df.empty:
                st.warning("Nenhum jogo encontrado para hoje.")
            else:
                st.success(f"{len(df)} jogos encontrados")
                st.dataframe(df, use_container_width=True)

                csv = df.to_csv(index=False).encode("utf-8")
                st.download_button(
                    "⬇️ Baixar CSV",
                    csv,
                    f"jogos_tenis_{date.today()}.csv",
                    "text/csv"
                )

        except Exception as e:
            st.error(f"Erro: {e}")

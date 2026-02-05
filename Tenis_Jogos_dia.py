import streamlit as st
import requests
import pandas as pd
from bs4 import BeautifulSoup
from datetime import date

st.set_page_config(page_title="Jogos de Tênis do Dia")
st.title("🎾 Jogos de Tênis do Dia (ATP & WTA)")

HEADERS = {
    "User-Agent": "Mozilla/5.0"
}

def obter_atp():
    url = "https://www.atptour.com/en/scores/current"
    html = requests.get(url, headers=HEADERS, timeout=15).text
    soup = BeautifulSoup(html, "html.parser")

    jogos = []
    for match in soup.select(".day-table-match"):
        players = match.select(".day-table-name")
        if len(players) != 2:
            continue

        jogos.append({
            "data": date.today().isoformat(),
            "tour": "ATP",
            "jogador1": players[0].get_text(strip=True),
            "jogador2": players[1].get_text(strip=True),
        })
    return jogos

def obter_wta():
    url = "https://www.wtatennis.com/scores"
    html = requests.get(url, headers=HEADERS, timeout=15).text
    soup = BeautifulSoup(html, "html.parser")

    jogos = []
    for match in soup.select(".match"):
        players = match.select(".player")
        if len(players) != 2:
            continue

        jogos.append({
            "data": date.today().isoformat(),
            "tour": "WTA",
            "jogador1": players[0].get_text(strip=True),
            "jogador2": players[1].get_text(strip=True),
        })
    return jogos

if st.button("🔄 Carregar jogos do dia"):
    try:
        jogos = obter_atp() + obter_wta()
        df = pd.DataFrame(jogos)

        if df.empty:
            st.warning("Nenhum jogo encontrado para hoje.")
        else:
            st.success(f"{len(df)} jogos encontrados")
            st.dataframe(df, use_container_width=True)

            st.download_button(
                "⬇️ Baixar CSV",
                df.to_csv(index=False).encode("utf-8"),
                f"jogos_tenis_{date.today()}.csv",
                "text/csv"
            )

    except Exception as e:
        st.error(f"Erro: {e}")

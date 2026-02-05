import streamlit as st
import requests
import pandas as pd
from bs4 import BeautifulSoup
from datetime import date

st.set_page_config(page_title="Jogos de Tênis do Dia", layout="centered")

st.title("🎾 Jogos de Tênis do Dia (ATP & WTA)")
st.write("Fonte: Flashscore")

URL = "https://www.flashscore.com/tennis/"
HEADERS = {
    "User-Agent": "Mozilla/5.0"
}

def obter_jogos_do_dia():
    response = requests.get(URL, headers=HEADERS, timeout=15)
    response.raise_for_status()

    soup = BeautifulSoup(response.text, "html.parser")
    jogos = []

    for match in soup.select("div.event__match"):
        players = match.select("div.event__participant")
        time_el = match.select_one("div.event__time")

        if len(players) != 2 or not time_el:
            continue

        jogos.append({
            "data": date.today().isoformat(),
            "tour": "ATP/WTA",
            "jogador1": players[0].text.strip(),
            "jogador2": players[1].text.strip(),
            "horario": time_el.text.strip()
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
                    label="⬇️ Baixar CSV",
                    data=csv,
                    file_name=f"jogos_tenis_{date.today()}.csv",
                    mime="text/csv"
                )

        except Exception as e:
            st.error(f"Erro ao obter jogos: {e}")

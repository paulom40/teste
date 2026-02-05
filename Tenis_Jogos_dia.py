import streamlit as st
import requests
import pandas as pd
from datetime import date

API_KEY = "SUA_API_KEY"

HEADERS = {
    "X-RapidAPI-Key": bba6af0e8dmsh6350139b0f77a4ap16b6fajsn219553636a44,
    "X-RapidAPI-Host": "api-tennis.p.rapidapi.com"
}

st.title("🎾 Jogos de Tênis do Dia (ATP & WTA)")

def obter_jogos():
    hoje = date.today().strftime("%Y-%m-%d")

    url = "https://api-tennis.p.rapidapi.com/fixtures"
    params = {"date": hoje}

    r = requests.get(url, headers=HEADERS, params=params)
    r.raise_for_status()
    data = r.json()

    jogos = []
    for jogo in data.get("response", []):
        league = jogo["league"]["name"]
        if "ATP" not in league and "WTA" not in league:
            continue

        jogos.append({
            "data": hoje,
            "tour": "ATP" if "ATP" in league else "WTA",
            "torneio": league,
            "jogador1": jogo["players"]["home"]["name"],
            "jogador2": jogo["players"]["away"]["name"],
            "horario": jogo["time"]
        })

    return pd.DataFrame(jogos)

if st.button("🔄 Carregar jogos do dia"):
    df = obter_jogos()

    if df.empty:
        st.warning("Nenhum jogo encontrado para hoje.")
    else:
        st.dataframe(df, use_container_width=True)
        st.download_button(
            "⬇️ Baixar CSV",
            df.to_csv(index=False).encode(),
            f"jogos_tenis_{date.today()}.csv",
            "text/csv"
        )

import streamlit as st
import requests
import pandas as pd
from datetime import date

st.set_page_config(page_title="Jogos de Tênis do Dia", layout="centered")
st.title("🎾 Jogos de Tênis do Dia (ATP & WTA)")
st.write("Fonte: SofaScore")

HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)",
    "Accept": "application/json, text/plain, */*",
    "Accept-Language": "pt-PT,pt;q=0.9,en-US;q=0.8,en;q=0.7",
    "Referer": "https://www.sofascore.com/",
    "Origin": "https://www.sofascore.com"
}

def obter_jogos_do_dia():
    hoje = date.today().strftime("%Y-%m-%d")
    url = f"https://api.sofascore.com/api/v1/sport/tennis/scheduled-events/{hoje}"

    r = requests.get(url, headers=HEADERS, timeout=15)
    r.raise_for_status()

    data = r.json()
    jogos = []

    for event in data.get("events", []):
        tournament = event.get("tournament", {})
        category = tournament.get("category", {})

        tour = category.get("name")
        if tour not in ["ATP", "WTA"]:
            continue

        jogos.append({
            "data": hoje,
            "tour": tour,
            "torneio": tournament.get("name"),
            "jogador1": event["homeTeam"]["name"],
            "jogador2": event["awayTeam"]["name"],
            "horario": pd.to_datetime(
                event["startTimestamp"], unit="s"
            ).strftime("%H:%M")
        })

    return pd.DataFrame(jogos)

if st.button("🔄 Carregar jogos do dia"):
    with st.spinner("Buscando jogos..."):
        try:
            df = obter_jogos_do_dia()

            if df.empty:
                st.warning("Nenhum jogo ATP/WTA encontrado para hoje.")
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

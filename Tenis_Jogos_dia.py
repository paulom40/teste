import streamlit as st
import requests
import pandas as pd
from datetime import date

st.set_page_config(page_title="Jogos de Tênis do Dia", layout="centered")
st.title("🎾 Jogos de Tênis do Dia (ATP & WTA)")
st.write("Fonte: SofaScore")

HEADERS = {
    "User-Agent": "Mozilla/5.0"
}

def obter_jogos_do_dia():
    hoje = date.today().strftime("%Y-%m-%d")

    url = (
        "https://api.sofascore.com/api/v1/sport/tennis/"
        f"scheduled-events/{hoje}"
    )

    r = requests.get(url, headers=HEADERS, timeout=15)
    r.raise_for_status()

    data = r.json()
    jogos = []

    for event in data.get("events", []):
        tournament = event.get("tournament", {})
        category = tournament.get("category", {})

        # Filtrar apenas ATP e WTA
        tour = category.get("name", "")
        if tour not in ["ATP", "WTA"]:
            continue

        home = event.get("homeTeam", {}).get("name")
        away = event.get("awayTeam", {}).get("name")
        start = event.get("startTimestamp")

        jogos.append({
            "data": hoje,
            "tour": tour,
            "torneio": tournament.get("name"),
            "jogador1": home,
            "jogador2": away,
            "horario": pd.to_datetime(start, unit="s").strftime("%H:%M")
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

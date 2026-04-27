import streamlit as st
import pandas as pd
import numpy as np
import requests
from io import BytesIO
from bs4 import BeautifulSoup


# ---------- NORMALIZAÇÃO DE NOMES ----------
def normalize_name(name):
    if not name:
        return ""
    name = name.lower().strip()
    name = name.replace(".", "").replace(",", "").replace("-", " ")
    return " ".join(name.split())


def find_player(name, all_players):
    name_norm = normalize_name(name)

    # Matching perfeito
    for p in all_players:
        if normalize_name(p) == name_norm:
            return p

    # Matching parcial
    for p in all_players:
        if normalize_name(p).startswith(name_norm.split()[0]):
            return p

    return None


# ---------- CARREGAR JOGADORES DO EXCEL ----------
@st.cache_data
def load_players_from_excel(uploaded_file):
    df = pd.read_excel(uploaded_file)

    if "winner_name" not in df.columns or "loser_name" not in df.columns:
        st.error("O Excel deve conter as colunas 'winner_name' e 'loser_name'.")
        return []

    winners = df["winner_name"].dropna().unique().tolist()
    losers = df["loser_name"].dropna().unique().tolist()

    players = sorted(list(set(winners + losers)))
    return players
# ---------- MODELO DUMMY (para correr no Streamlit Cloud) ----------
class DummyModel:
    def predict_proba(self, X):
        X = np.array(X)
        if X.ndim == 1:
            X = X.reshape(1, -1)
        return np.tile(np.array([[0.5, 0.5]]), (X.shape[0], 1))


model = DummyModel()


# ---------- BUILD_MATCH_SUMMARY (substitui pelo teu depois) ----------
def build_match_summary(player1, player2, surface):
    features = {
        "surface_hard": 1 if surface.lower() == "hard" else 0,
        "surface_clay": 1 if surface.lower() == "clay" else 0,
        "surface_grass": 1 if surface.lower() == "grass" else 0,
        "dummy_strength": 0.5
    }
    return pd.Series(features)
# ---------- SCRAPER TENNIS24 ----------
def scrape_tennis24():
    logs = ["📅 Procurando jogos de HOJE no Tennis24 (ATP + Challenger)"]

    url = "https://www.tennis24.com/"
    headers = {"User-Agent": "Mozilla/5.0"}

    try:
        r = requests.get(url, headers=headers, timeout=10)
        if r.status_code != 200:
            logs.append(f"❌ HTTP {r.status_code} no Tennis24")
            return [], logs

        soup = BeautifulSoup(r.text, "html.parser")
        matches = []

        for m in soup.select(".event__match"):

            tournament = m.get("data-competition-name", "")

            # Filtrar apenas ATP + Challenger
            if not any(x in tournament.upper() for x in ["ATP", "CHALLENGER"]):
                continue

            p1 = m.select_one(".event__participant--home")
            p2 = m.select_one(".event__participant--away")

            if not p1 or not p2:
                continue

            p1_name = p1.text.strip()
            p2_name = p2.text.strip()

            odd_home = m.get("data-odd-home")
            odd_away = m.get("data-odd-away")

            try:
                odd_home = float(odd_home) if odd_home else None
                odd_away = float(odd_away) if odd_away else None
            except:
                odd_home = None
                odd_away = None

            matches.append({
                "tournament": tournament,
                "player1": p1_name,
                "player2": p2_name,
                "surface": "Hard",  # Tennis24 não mostra superfície
                "odd1": odd_home,
                "odd2": odd_away
            })

        logs.append(f"🎾 TOTAL: {len(matches)} jogos encontrados no Tennis24")
        return matches, logs

    except Exception as e:
        logs.append(f"💥 Erro no scraper Tennis24: {e}")
        return [], logs


# ---------- PREPARAR MATCH ----------
def prepare_match_for_prediction(match, all_players):
    p1_hist = find_player(match["player1"], all_players)
    p2_hist = find_player(match["player2"], all_players)

    if not p1_hist or not p2_hist:
        return None, f"❌ Matching falhou: {match['player1']} vs {match['player2']}"

    summary = build_match_summary(p1_hist, p2_hist, match["surface"])
    return summary, f"✅ Matching OK: {match['player1']} vs {match['player2']}"


# ---------- EXPORTAR EXCEL ----------
def export_daily_predictions_to_excel(df):
    output = BytesIO()
    writer = pd.ExcelWriter(output, engine="xlsxwriter")

    df.to_excel(writer, index=False, sheet_name="Previsoes")

    workbook = writer.book
    worksheet = writer.sheets["Previsoes"]

    fmt_green = workbook.add_format({"bg_color": "#C6EFCE", "font_color": "#006100"})
    fmt_red = workbook.add_format({"bg_color": "#FFC7CE", "font_color": "#9C0006"})

    if "VALUE BET" in df.columns:
        col_idx = df.columns.get_loc("VALUE BET")
        for row in range(1, len(df) + 1):
            value = df.iloc[row - 1, col_idx]
            worksheet.write(row, col_idx, value, fmt_green if value == "YES" else fmt_red)

    writer.close()
    output.seek(0)
    return output
# ---------- PIPELINE ----------
def run_daily_predictions(all_players):
    matches, logs = scrape_tennis24()

    st.subheader("Logs do scraper")
    for l in logs:
        st.text(l)

    if not matches:
        st.error("❌ Nenhum jogo encontrado hoje.")
        return

    resultados = []

    for match in matches:
        summary, log_match = prepare_match_for_prediction(match, all_players)
        st.text(log_match)

        if summary is None:
            continue

        X = summary.values.reshape(1, -1)
        prob = model.predict_proba(X)[0][1]

        odd1 = match["odd1"]
        ev1 = prob * odd1 - 1 if odd1 else None
        value_bet = "YES" if ev1 and ev1 > 0 else "NO"

        resultados.append({
            "Torneio": match["tournament"],
            "Jogador 1": match["player1"],
            "Jogador 2": match["player2"],
            "Odd J1": odd1,
            "Odd J2": match["odd2"],
            "Prob J1": prob,
            "EV J1": ev1,
            "VALUE BET": value_bet
        })

    df_res = pd.DataFrame(resultados)
    st.subheader("Previsões Jogos do Dia")
    st.dataframe(df_res)

    excel_file = export_daily_predictions_to_excel(df_res)
    st.download_button(
        label="📥 Baixar Excel com Previsões",
        data=excel_file,
        file_name="previsoes_tennis24.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
    )


# ---------- STREAMLIT APP ----------
def main():
    st.set_page_config(page_title="Challenger Predictions", layout="wide")
    st.title("🎾 Challenger / ATP — Previsões (Tennis24)")

    st.sidebar.header("Configuração")

    uploaded_file = st.sidebar.file_uploader(
        "Carregar histórico (Excel .xlsx com winner_name / loser_name)",
        type=["xlsx"]
    )

    if uploaded_file is None:
        st.warning("Carrega um Excel com histórico para construir a lista de jogadores.")
        return

    all_players = load_players_from_excel(uploaded_file)
    st.sidebar.success(f"{len(all_players)} jogadores carregados.")

    if st.button("Buscar jogos de hoje e prever"):
        run_daily_predictions(all_players)


if __name__ == "__main__":
    main()

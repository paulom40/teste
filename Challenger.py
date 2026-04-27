import requests
import pandas as pd
from io import BytesIO
from datetime import datetime
from email.utils import parsedate_to_datetime

def normalize_name(name):
    if not name:
        return ""
    name = name.lower().strip()
    name = name.replace(".", "")
    name = name.replace(",", "")
    name = name.replace("-", " ")
    parts = name.split()
    return " ".join(parts)

def find_player(name, all_players):
    name_norm = normalize_name(name)

    # Matching perfeito
    for p in all_players:
        if normalize_name(p) == name_norm:
            return p

    # Matching parcial (ex: "T Longacre" → "Trevor Longacre")
    for p in all_players:
        if normalize_name(p).startswith(name_norm.split()[0]):
            return p

    return None

def get_real_date():
    try:
        r = requests.get("https://www.google.com", timeout=5)
        date_str = r.headers["Date"]
        dt = parsedate_to_datetime(date_str)
        return dt.strftime("%Y-%m-%d")
    except:
        return datetime.utcnow().strftime("%Y-%m-%d")
def scrape_matches():
    logs = ["📅 Procurando jogos de HOJE (RapidAPI — ATP/WTA/ITF)"]

    today = get_real_date()
    logs.append(f"📅 Data real usada: {today}")

    API_KEY = "bba6af0e8dmsh6350139b0f77a4ap16b6fajsn219553636a44"
    API_HOST = "tennis-api-atp-wta-itf.p.rapidapi.com"

    headers = {
        "x-rapidapi-key": API_KEY,
        "x-rapidapi-host": API_HOST,
        "Content-Type": "application/json"
    }

    url = f"https://tennis-api-atp-wta-itf.p.rapidapi.com/tennis/v2/atp/matches-by-date/{today}"
    logs.append(f"🔎 Endpoint: {url}")

    try:
        r = requests.get(url, headers=headers, timeout=15)

        if r.status_code != 200:
            logs.append(f"❌ HTTP {r.status_code}")
            return [], logs

        data = r.json()
        matches = data.get("matches", [])

        if not matches:
            logs.append("⚠️ Nenhum jogo encontrado hoje")
            return [], logs

        logs.append(f"🎾 Encontrados {len(matches)} jogos (antes do filtro)")

        all_matches = []

        for m in matches:
            try:
                p1_api = m["homePlayer"]["name"]
                p2_api = m["awayPlayer"]["name"]

                tournament = m["tournament"]["name"]
                surface = m["tournament"].get("surface", "Hard")
                match_id = m["id"]

                odds_home = m.get("odds", {}).get("home")
                odds_away = m.get("odds", {}).get("away")

                # Corrigir ordem: favorito = Jogador 1
                if odds_home and odds_away and odds_home > odds_away:
                    p1_api, p2_api = p2_api, p1_api
                    odds_home, odds_away = odds_away, odds_home

                all_matches.append({
                    "tournament": tournament,
                    "player1": p1_api,
                    "player2": p2_api,
                    "surface": surface,
                    "match_id": match_id,
                    "odd1": odds_home,
                    "odd2": odds_away
                })

            except Exception as e:
                logs.append(f"Erro num match: {e}")
                continue

        logs.append(f"🎾 Após correção de ordem: {len(all_matches)} jogos")
        return all_matches, logs

    except Exception as e:
        logs.append(f"💥 Erro: {e}")
        return [], logs
def prepare_match_for_prediction(match, all_players, surface_default="Hard"):
    p1_name_api = match["player1"]
    p2_name_api = match["player2"]

    p1_hist = find_player(p1_name_api, all_players)
    p2_hist = find_player(p2_name_api, all_players)

    if not p1_hist or not p2_hist:
        return None, f"❌ Matching falhou: API({p1_name_api} vs {p2_name_api}) → HIST({p1_hist} vs {p2_hist})"

    surface = match.get("surface") or surface_default

    summary = build_match_summary(
        player1=p1_hist,
        player2=p2_hist,
        surface=surface
    )

    return summary, f"✅ Matching OK: API({p1_name_api} vs {p2_name_api}) → HIST({p1_hist} vs {p2_hist})"


def export_daily_predictions_to_excel(df):
    output = BytesIO()
    writer = pd.ExcelWriter(output, engine="xlsxwriter")

    df.to_excel(writer, index=False, sheet_name="Previsoes")

    workbook = writer.book
    worksheet = writer.sheets["Previsoes"]

    fmt_num = workbook.add_format({"num_format": "0.00"})
    fmt_green = workbook.add_format({"bg_color": "#C6EFCE", "font_color": "#006100"})
    fmt_red = workbook.add_format({"bg_color": "#FFC7CE", "font_color": "#9C0006"})

    for col in range(len(df.columns)):
        worksheet.set_column(col, col, 18, fmt_num)

    col_idx = df.columns.get_loc("VALUE BET")

    for row in range(1, len(df) + 1):
        value = df.iloc[row - 1, col_idx]
        if value == "YES":
            worksheet.write(row, col_idx, value, fmt_green)
        else:
            worksheet.write(row, col_idx, value, fmt_red)

    writer.close()
    output.seek(0)
    return output
import streamlit as st

def run_daily_predictions(all_players, model):
    matches, logs = scrape_matches()

    if not matches:
        st.write("Nenhum jogo encontrado hoje.")
        for l in logs:
            st.text(l)
        return

    st.subheader("Logs do scraper")
    for l in logs:
        st.text(l)

    resultados = []

    for match in matches:
        summary, log_match = prepare_match_for_prediction(match, all_players)
        st.text(log_match)

        if summary is None:
            continue

        X = summary.values.reshape(1, -1) if hasattr(summary, "values") else summary
        prob = model.predict_proba(X)[0][1]

        odd1 = match.get("odd1")
        odd2 = match.get("odd2")

        ev1 = prob * odd1 - 1 if odd1 else None
        value_bet = "YES" if ev1 and ev1 > 0 else "NO"

        resultados.append({
            "Torneio": match["tournament"],
            "Jogador 1": match["player1"],
            "Jogador 2": match["player2"],
            "Superfície": match["surface"],
            "Odd J1": odd1,
            "Odd J2": odd2,
            "Prob J1": prob,
            "EV J1": ev1,
            "VALUE BET": value_bet
        })

    if not resultados:
        st.write("Nenhum jogo com jogadores encontrados no histórico.")
        return

    df_res = pd.DataFrame(resultados)

    st.subheader("Previsões Jogos do Dia")
    st.dataframe(df_res)

    excel_file = export_daily_predictions_to_excel(df_res)
    st.download_button(
        label="📥 Baixar Excel com Previsões",
        data=excel_file,
        file_name=f"previsoes_{get_real_date()}.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
    )

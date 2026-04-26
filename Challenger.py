import warnings
from collections import defaultdict
from datetime import datetime, timedelta
import io
import numpy as np
import pandas as pd
import streamlit as st
import requests
from lightgbm import LGBMClassifier
import matplotlib.pyplot as plt
import re

warnings.filterwarnings('ignore')

st.set_page_config(page_title="ATP Predictor PRO", page_icon="🎾", layout="wide")

# ==============================================================================
# CONFIG
# ==============================================================================
WINNER_SMOOTH = 0.35
MIN_CONFIDENCE_STRONG = 0.65
MIN_CONFIDENCE_GOOD = 0.56
MIN_CONFIDENCE_WEAK = 0.50

# ==============================================================================
# SURFACE DETECTION
# ==============================================================================
def detect_surface(tournament_name):
    if pd.isna(tournament_name):
        return 'Hard'
    t = str(tournament_name).lower()
    clay = ['clay', 'monte carlo', 'madrid', 'rome', 'barcelona', 'munich', 'roland garros']
    grass = ['grass', 'wimbledon', 'queens', 'halle']
    if any(k in t for k in clay):
        return 'Clay'
    if any(k in t for k in grass):
        return 'Grass'
    return 'Hard'

# ==============================================================================
# PROCESS DATA
# ==============================================================================
def process_historical_data(df):
    df.columns = [str(c).strip().lower().replace(' ', '_').replace('-', '_') for c in df.columns]
    
    winner_col = None
    loser_col = None
    
    for col in df.columns:
        if 'winner' in col or 'vencedor' in col:
            winner_col = col
        elif 'loser' in col or 'perdedor' in col:
            loser_col = col
    
    if not winner_col or not loser_col:
        return None, None
    
    df = df.rename(columns={winner_col: 'winner', loser_col: 'loser'})
    
    if 'date' not in df.columns:
        df['date'] = pd.Timestamp.now()
    else:
        df['date'] = pd.to_datetime(df['date'], errors='coerce')
    
    if 'total_games' not in df.columns:
        df['total_games'] = 22
    
    if 'surface' not in df.columns:
        df['surface'] = 'Hard'
    
    df['winner'] = df['winner'].astype(str).str.strip()
    df['loser'] = df['loser'].astype(str).str.strip()
    
    df = df[df['winner'].notna() & df['loser'].notna()]
    df = df[df['winner'] != 'nan']
    df = df[df['loser'] != 'nan']
    df = df[df['winner'] != '']
    df = df[df['loser'] != '']
    
    all_players = list(set(df['winner'].unique()) | set(df['loser'].unique()))
    
    return df, all_players

# ==============================================================================
# PLAYER STATISTICS
# ==============================================================================
def calculate_player_stats(df, all_players):
    stats = {}
    
    for player in all_players:
        player_matches = df[(df['winner'] == player) | (df['loser'] == player)]
        
        if len(player_matches) == 0:
            stats[player] = {
                'matches': 0, 'wins': 0, 'win_rate': 0.5,
                'recent_form': 0.5, 'very_recent_form': 0.5, 'avg_games': 22
            }
            continue
        
        wins = len(player_matches[player_matches['winner'] == player])
        total = len(player_matches)
        win_rate = wins / total if total > 0 else 0.5
        
        recent = player_matches.sort_values('date', ascending=False).head(10)
        recent_wins = len(recent[recent['winner'] == player])
        recent_form = recent_wins / len(recent) if len(recent) > 0 else 0.5
        
        very_recent = player_matches.sort_values('date', ascending=False).head(5)
        very_recent_wins = len(very_recent[very_recent['winner'] == player])
        very_recent_form = very_recent_wins / len(very_recent) if len(very_recent) > 0 else 0.5
        
        avg_games = player_matches['total_games'].mean() if 'total_games' in player_matches.columns else 22
        
        stats[player] = {
            'matches': total,
            'wins': wins,
            'win_rate': win_rate,
            'recent_form': recent_form,
            'very_recent_form': very_recent_form,
            'avg_games': avg_games
        }
    
    return stats

# ==============================================================================
# H2H
# ==============================================================================
def calculate_h2h(df):
    h2h = defaultdict(lambda: {'wins': 0, 'total': 0})
    for _, row in df.iterrows():
        w, l = row['winner'], row['loser']
        h2h[(w, l)]['wins'] += 1
        h2h[(w, l)]['total'] += 1
    return h2h

# ==============================================================================
# ELO GLOBAL
# ==============================================================================
def calculate_elo(df, all_players, k=32):
    elo = {p: 1500 for p in all_players}
    
    for _, row in df.sort_values('date').iterrows():
        w, l = row['winner'], row['loser']
        if w in elo and l in elo:
            exp_w = 1 / (1 + 10 ** ((elo[l] - elo[w]) / 400))
            elo[w] += k * (1 - exp_w)
            elo[l] += k * (0 - (1 - exp_w))
    
    return elo

# ==============================================================================
# ELO20 GLOBAL (últimos 20 jogos)
# ==============================================================================
def calculate_elo_last20(df, all_players, k=32):
    elo = {p: 1500 for p in all_players}
    history = {p: [] for p in all_players}

    df_sorted = df.sort_values("date")

    for _, row in df_sorted.iterrows():
        w, l = row["winner"], row["loser"]

        history[w].append((w, l))
        history[l].append((w, l))

        if len(history[w]) > 20:
            history[w].pop(0)
        if len(history[l]) > 20:
            history[l].pop(0)

        if len(history[w]) >= 5 and len(history[l]) >= 5:
            exp_w = 1 / (1 + 10 ** ((elo[l] - elo[w]) / 400))
            elo[w] += k * (1 - exp_w)
            elo[l] += k * (0 - (1 - exp_w))

    return elo

# ==============================================================================
# ELO por superfície últimos 20 jogos
# ==============================================================================
def calculate_surface_elo_last20(df, all_players, surface, k=32):
    elo = {p: 1500 for p in all_players}
    history = {p: [] for p in all_players}

    df_surf = df[df["surface"] == surface].sort_values("date")

    for _, row in df_surf.iterrows():
        w = row["winner"]
        l = row["loser"]

        history[w].append((w, l))
        history[l].append((w, l))

        if len(history[w]) > 20:
            history[w].pop(0)
        if len(history[l]) > 20:
            history[l].pop(0)

        if len(history[w]) >= 5 and len(history[l]) >= 5:
            exp_w = 1 / (1 + 10 ** ((elo[l] - elo[w]) / 400))
            elo[w] += k * (1 - exp_w)
            elo[l] += k * (0 - (1 - exp_w))

    return elo

# ==============================================================================
# W‑ELO (ponderado pela forma)
# ==============================================================================
def calculate_welo(elo_value, recent_form):
    return round(elo_value * 0.7 + (recent_form * 1000) * 0.3, 1)

# ==============================================================================
# NAME MATCHER
# ==============================================================================
def find_player(name, historical_names):
    if not name:
        return None
    
    name_str = str(name).strip()
    name_lower = name_str.lower()
    
    if name_str in historical_names:
        return name_str
    
    for player in historical_names:
        if player.lower() == name_lower:
            return player
    
    parts = name_lower.split()
    if parts:
        last_name = parts[-1]
        for player in historical_names:
            if player.lower().endswith(last_name):
                return player
    
    for player in historical_names:
        if name_lower in player.lower():
            return player
    
    return None

# ==============================================================================
# SCRAPER RAPIDAPI — ATP + CHALLENGER + ODDS
# ==============================================================================
def scrape_matches():
    API_KEY = "bba6af0e8dmsh6350139b0f77a4ap16b6fajsn219553636a44"
    API_HOST = "tennisapi1.p.rapidapi.com"

    headers = {
        "x-rapidapi-key": API_KEY,
        "x-rapidapi-host": API_HOST,
        "Content-Type": "application/json"
    }

    today = datetime.now().strftime("%Y-%m-%d")
    tomorrow = (datetime.now() + timedelta(days=1)).strftime("%Y-%m-%d")

    dates_to_check = [today, tomorrow]

    all_matches = []
    logs = []

    for d in dates_to_check:
        url = f"https://tennisapi1.p.rapidapi.com/api/tennis/matches/date/{d}"
        logs.append(f"📅 Procurando jogos para {d}")
        logs.append(f"🔎 Endpoint: {url}")

        try:
            r = requests.get(url, headers=headers, timeout=15)

            if r.status_code != 200:
                logs.append(f"❌ HTTP {r.status_code}")
                continue

            data = r.json()
            matches = data.get("data", [])

            if not matches:
                logs.append("⚠️ Nenhum jogo encontrado nesta data")
                continue

            logs.append(f"✅ Encontrados {len(matches)} jogos (antes do filtro)")

            for m in matches:
                try:
                    tournament = m["tournament"]["name"]
                    category = m["tournament"].get("category", "").upper()

                    if not any(x in category for x in ["ATP", "CHALLENGER"]):
                        continue

                    match_id = m["id"]
                    p1 = m["home_player"]["name"]
                    p2 = m["away_player"]["name"]
                    surface = m["tournament"].get("surface", "Hard")

                    odds_url = f"https://tennisapi1.p.rapidapi.com/api/tennis/odds/{match_id}"
                    odds_home = None
                    odds_away = None

                    try:
                        r_odds = requests.get(odds_url, headers=headers, timeout=10)
                        if r_odds.status_code == 200:
                            odds_data = r_odds.json().get("data", {})
                            if "odds" in odds_data:
                                odds_home = odds_data["odds"].get("home")
                                odds_away = odds_data["odds"].get("away")
                    except:
                        pass

                    all_matches.append({
                        "tournament": tournament,
                        "player1": p1,
                        "player2": p2,
                        "surface": surface,
                        "match_id": match_id,
                        "odd1": odds_home,
                        "odd2": odds_away
                    })

                except Exception:
                    continue

            logs.append(f"🎾 Após filtro ATP/Challenger: {len(all_matches)} jogos")

        except Exception as e:
            logs.append(f"💥 Erro: {e}")

    return all_matches, logs
# ==============================================================================
# TRAIN MODEL
# ==============================================================================
def train_model(df, stats, h2h, elo_global):
    X = []
    y = []

    for _, row in df.iterrows():
        p1 = row["winner"]
        p2 = row["loser"]

        if p1 not in stats or p2 not in stats:
            continue

        s1 = stats[p1]
        s2 = stats[p2]

        h2h_wins = h2h.get((p1, p2), {"wins": 0})["wins"]
        h2h_losses = h2h.get((p2, p1), {"wins": 0})["wins"]

        X.append([
            s1["win_rate"], s2["win_rate"],
            s1["recent_form"], s2["recent_form"],
            s1["very_recent_form"], s2["very_recent_form"],
            s1["avg_games"], s2["avg_games"],
            h2h_wins, h2h_losses,
            elo_global.get(p1, 1500), elo_global.get(p2, 1500)
        ])

        y.append(1)

    model = LGBMClassifier(
        n_estimators=300,
        learning_rate=0.05,
        max_depth=5,
        subsample=0.8,
        colsample_bytree=0.8
    )

    model.fit(X, y)
    return model

# ==============================================================================
# PREDICT MATCH
# ==============================================================================
def predict_match(model, p1, p2, surface, stats, h2h, elo_global, all_players, tournament):
    if p1 not in all_players or p2 not in all_players:
        return None, f"Jogador não encontrado: {p1} ou {p2}"

    s1 = stats[p1]
    s2 = stats[p2]

    h2h_wins = h2h.get((p1, p2), {"wins": 0})["wins"]
    h2h_losses = h2h.get((p2, p1), {"wins": 0})["wins"]

    X = [[
        s1["win_rate"], s2["win_rate"],
        s1["recent_form"], s2["recent_form"],
        s1["very_recent_form"], s2["very_recent_form"],
        s1["avg_games"], s2["avg_games"],
        h2h_wins, h2h_losses,
        elo_global.get(p1, 1500), elo_global.get(p2, 1500)
    ]]

    prob = model.predict_proba(X)[0][1]
    prob = WINNER_SMOOTH * prob + (1 - WINNER_SMOOTH) * 0.5

    prob_p1 = prob
    prob_p2 = 1 - prob

    if prob_p1 > MIN_CONFIDENCE_STRONG:
        rec = "STRONG"
    elif prob_p1 > MIN_CONFIDENCE_GOOD:
        rec = "GOOD"
    elif prob_p1 > MIN_CONFIDENCE_WEAK:
        rec = "WEAK"
    else:
        rec = "NO BET"

    return {
        "Torneio": tournament,
        "Jogador1": p1,
        "Jogador2": p2,
        "Superficie": surface,
        "Prob_P1": f"{prob_p1*100:.1f}%",
        "Prob_P2": f"{prob_p2*100:.1f}%",
        "Recomendacao": rec
    }, None

# ==============================================================================
# VALUE BET CLASSIFICATION
# ==============================================================================
def classify_value(ev):
    if ev is None:
        return "NO BET"
    if ev >= 0.08:
        return "STRONG VALUE"
    if ev >= 0.04:
        return "GOOD VALUE"
    if ev >= 0.01:
        return "WEAK VALUE"
    return "NO VALUE"

# ==============================================================================
# COLOR EV
# ==============================================================================
def color_ev(val):
    if val is None:
        return ""
    if val > 0:
        return "background-color:#c6f6c6;color:#004d00;"
    if val < 0:
        return "background-color:#f6c6c6;color:#8b0000;"
    return ""

# ==============================================================================
# SUMMARY BUILDER (Winner + O/U + ELO + W‑ELO + Odds + EV)
# ==============================================================================
def build_match_summary(p1, p2, surface, stats, h2h, elo_global, elo_surface, model_prob_p1, odds_p1, odds_p2):
    s1 = stats[p1]
    s2 = stats[p2]

    elo1 = elo_global.get(p1, 1500)
    elo2 = elo_global.get(p2, 1500)

    selo1 = elo_surface.get(p1, 1500)
    selo2 = elo_surface.get(p2, 1500)

    welo1 = calculate_welo(elo1, s1["recent_form"])
    welo2 = calculate_welo(elo2, s2["recent_form"])

    avg_games = (s1["avg_games"] + s2["avg_games"]) / 2
    prob_over = min(max((avg_games - 21.5) / 10 + 0.5, 0.05), 0.95)
    prob_under = 1 - prob_over

    prob_p1 = model_prob_p1
    prob_p2 = 1 - model_prob_p1

    ev_p1 = prob_p1 - (1 / odds_p1) if odds_p1 and odds_p1 > 1 else None
    ev_p2 = prob_p2 - (1 / odds_p2) if odds_p2 and odds_p2 > 1 else None

    ev_over = prob_over - (1 / odds_p1) if odds_p1 else None
    ev_under = prob_under - (1 / odds_p2) if odds_p2 else None

    summary = {
        "Jogador 1": p1,
        "Jogador 2": p2,
        "Superfície": surface,

        "Prob P1 (%)": round(prob_p1 * 100, 1),
        "Prob P2 (%)": round(prob_p2 * 100, 1),

        "Over 21.5 (%)": round(prob_over * 100, 1),
        "Under 21.5 (%)": round(prob_under * 100, 1),

        "ELO P1": round(elo1, 1),
        "ELO P2": round(elo2, 1),

        "ELO Superfície P1": round(selo1, 1),
        "ELO Superfície P2": round(selo2, 1),

        "W‑ELO P1": welo1,
        "W‑ELO P2": welo2,

        "Odd P1": odds_p1,
        "Odd P2": odds_p2,

        "EV P1": round(ev_p1, 3) if ev_p1 is not None else None,
        "EV P2": round(ev_p2, 3) if ev_p2 is not None else None,

        "EV Over": round(ev_over, 3) if ev_over is not None else None,
        "EV Under": round(ev_under, 3) if ev_under is not None else None,

        "VALUE P1": classify_value(ev_p1),
        "VALUE P2": classify_value(ev_p2),
        "VALUE Over": classify_value(ev_over),
        "VALUE Under": classify_value(ev_under)
    }

    return summary

# ==============================================================================
# RANKING DIÁRIO COMPLETO
# ==============================================================================
def build_value_ranking(summaries):
    df = pd.DataFrame(summaries)
    df["EV_MAX"] = df[["EV P1", "EV P2", "EV Over", "EV Under"]].max(axis=1)

    def best_value(row):
        tags = [
            row["VALUE P1"],
            row["VALUE P2"],
            row["VALUE Over"],
            row["VALUE Under"]
        ]
        if "STRONG VALUE" in tags:
            return "STRONG VALUE"
        if "GOOD VALUE" in tags:
            return "GOOD VALUE"
        if "WEAK VALUE" in tags:
            return "WEAK VALUE"
        return "NO VALUE"

    df["VALUE_TAG"] = df.apply(best_value, axis=1)
    df = df.sort_values("EV_MAX", ascending=False)
    return df

# ==============================================================================
# RANKING COMPACTO
# ==============================================================================
def build_compact_value_table(summaries):
    rows = []

    for s in summaries:
        jogo = f"{s['Jogador 1']} vs {s['Jogador 2']}"

        ev_max = max([
            s.get("EV P1") or -999,
            s.get("EV P2") or -999,
            s.get("EV Over") or -999,
            s.get("EV Under") or -999
        ])

        tags = [
            s["VALUE P1"],
            s["VALUE P2"],
            s["VALUE Over"],
            s["VALUE Under"]
        ]

        if "STRONG VALUE" in tags:
            value_tag = "STRONG VALUE"
        elif "GOOD VALUE" in tags:
            value_tag = "GOOD VALUE"
        elif "WEAK VALUE" in tags:
            value_tag = "WEAK VALUE"
        else:
            value_tag = "NO VALUE"

        prob = max(s["Prob P1 (%)"], s["Prob P2 (%)"])

        if s["Prob P1 (%)"] > s["Prob P2 (%)"]:
            odd = s["Odd P1"]
        else:
            odd = s["Odd P2"]

        rows.append({
            "Jogo": jogo,
            "EV_MAX": round(ev_max, 3),
            "VALUE_TAG": value_tag,
            "Odd": odd,
            "Prob (%)": prob,
            "W‑ELO Jogador 1": s["W‑ELO P1"],
            "W‑ELO Jogador 2": s["W‑ELO P2"]
        })

    df = pd.DataFrame(rows)
    df = df.sort_values("EV_MAX", ascending=False)
    return df

# ==============================================================================
# EXPORT EXCEL
# ==============================================================================
def export_excel(df):
    buffer = io.BytesIO()
    df.to_excel(buffer, index=False)
    return buffer.getvalue()
# ==============================================================================
# COMPARAÇÃO 1v1 — ELO, ELO20, W‑ELO, W‑ELO20
# ==============================================================================
def build_elo_comparison_1v1(p1, p2, stats, elo_global, elo20_global, elo_surface, elo20_surface):
    def welo(elo, form):
        return round(elo * 0.7 + (form * 1000) * 0.3, 1)

    form1 = stats[p1]["recent_form"]
    form2 = stats[p2]["recent_form"]

    comp = {
        "Métrica": [
            "ELO Global",
            "ELO20 Global",
            "W‑ELO Global",
            "W‑ELO20 Global",
            "ELO Superfície",
            "ELO20 Superfície",
            "W‑ELO Superfície",
            "W‑ELO20 Superfície"
        ],
        p1: [
            round(elo_global.get(p1, 1500), 1),
            round(elo20_global.get(p1, 1500), 1),
            welo(elo_global.get(p1, 1500), form1),
            welo(elo20_global.get(p1, 1500), form1),
            round(elo_surface.get(p1, 1500), 1),
            round(elo20_surface.get(p1, 1500), 1),
            welo(elo_surface.get(p1, 1500), form1),
            welo(elo20_surface.get(p1, 1500), form1)
        ],
        p2: [
            round(elo_global.get(p2, 1500), 1),
            round(elo20_global.get(p2, 1500), 1),
            welo(elo_global.get(p2, 1500), form2),
            welo(elo20_global.get(p2, 1500), form2),
            round(elo_surface.get(p2, 1500), 1),
            round(elo20_surface.get(p2, 1500), 1),
            welo(elo_surface.get(p2, 1500), form2),
            welo(elo20_surface.get(p2, 1500), form2)
        ]
    }

    return pd.DataFrame(comp)

# ==============================================================================
# GRÁFICO 1v1
# ==============================================================================
def plot_elo_comparison_1v1(df, p1, p2):
    fig, ax = plt.subplots(figsize=(10, 5))

    x = range(len(df))
    ax.bar([i - 0.2 for i in x], df[p1], width=0.4, label=p1, color="#4c72b0")
    ax.bar([i + 0.2 for i in x], df[p2], width=0.4, label=p2, color="#c44e52")

    ax.set_xticks(x)
    ax.set_xticklabels(df["Métrica"], rotation=45, ha="right")
    ax.set_ylabel("Rating")
    ax.set_title(f"Comparação 1v1 — {p1} vs {p2}")
    ax.legend()

    plt.tight_layout()
    return fig

# ==============================================================================
# INTERFACE STREAMLIT
# ==============================================================================
def main():
    st.title("🎾 ATP Predictor PRO — v7.4")

    st.sidebar.header("📁 Carregar Base de Dados")
file = st.sidebar.file_uploader("Carregar Excel", type=["xlsx", "xls"])

if file is None:
    st.info("Carrega um ficheiro Excel para começar.")
    return

df = pd.read_excel(file)

    df, all_players = process_historical_data(df)

    if df is None:
        st.error("Erro ao processar o ficheiro.")
        return

    stats = calculate_player_stats(df, all_players)
    h2h = calculate_h2h(df)

    elo_global = calculate_elo(df, all_players)
    elo20_global = calculate_elo_last20(df, all_players)

    elo_clay = calculate_surface_elo_last20(df, all_players, "Clay")
    elo_hard = calculate_surface_elo_last20(df, all_players, "Hard")
    elo_grass = calculate_surface_elo_last20(df, all_players, "Grass")

    st.session_state.stats = stats
    st.session_state.h2h = h2h
    st.session_state.elo = elo_global
    st.session_state.elo20_global = elo20_global
    st.session_state.elo_clay = elo_clay
    st.session_state.elo_hard = elo_hard
    st.session_state.elo_grass = elo_grass

    model = train_model(df, stats, h2h, elo_global)

    tabs = st.tabs(["🎾 Previsão Individual", "📋 Previsão Jogos do Dia", "🏆 Ranking VALUE BETS", "⚔️ Comparação 1v1"])

    # ==============================================================================
    # TAB 1 — PREVISÃO INDIVIDUAL
    # ==============================================================================
    with tabs[0]:
        st.subheader("🎾 Previsão Individual")

        col1, col2 = st.columns(2)
        with col1:
            p1 = st.selectbox("Jogador 1", [""] + sorted(all_players))
        with col2:
            p2 = st.selectbox("Jogador 2", [""] + sorted(all_players))

        surface = st.selectbox("Superfície", ["Clay", "Hard", "Grass"])
        tournament = st.text_input("Torneio", "ATP 250")

        if st.button("Prever"):
            if not p1 or not p2 or p1 == p2:
                st.error("Jogadores inválidos.")
            else:
                result, err = predict_match(model, p1, p2, surface, stats, h2h, elo_global, all_players, tournament)
                if err:
                    st.error(err)
                else:
                    st.success(f"Probabilidade {p1}: {result['Prob_P1']}")
                    st.success(f"Probabilidade {p2}: {result['Prob_P2']}")

    # ==============================================================================
    # TAB 2 — PREVISÃO JOGOS DO DIA
    # ==============================================================================
    with tabs[1]:
        st.subheader("📋 Previsão Jogos do Dia")

        if st.button("Buscar Jogos de Hoje"):
            matches, logs = scrape_matches()

            st.text("\n".join(logs))

            summaries = []

            for match in matches:
                p1 = find_player(match["player1"], all_players)
                p2 = find_player(match["player2"], all_players)

                if not p1 or not p2:
                    continue

                result, err = predict_match(
                    model, p1, p2, match["surface"], stats, h2h, elo_global, all_players, match["tournament"]
                )

                if err:
                    continue

                prob_p1 = float(result["Prob_P1"].replace("%", "")) / 100

                summary = build_match_summary(
                    p1, p2, match["surface"], stats, h2h, elo_global,
                    elo_clay if match["surface"] == "Clay" else
                    elo_hard if match["surface"] == "Hard" else
                    elo_grass,
                    prob_p1,
                    match["odd1"],
                    match["odd2"]
                )

                summaries.append(summary)

            st.session_state.summaries = summaries

            st.dataframe(pd.DataFrame(summaries), use_container_width=True)

    # ==============================================================================
    # TAB 3 — RANKING VALUE BETS
    # ==============================================================================
    with tabs[2]:
        st.subheader("🏆 Ranking Diário — VALUE BETS")

        if "summaries" not in st.session_state:
            st.info("Primeiro gera previsões no separador anterior.")
        else:
            ranking_df = build_value_ranking(st.session_state.summaries)

            st.dataframe(
                ranking_df.style.applymap(color_ev, subset=["EV P1", "EV P2", "EV Over", "EV Under", "EV_MAX"]),
                use_container_width=True
            )

            st.download_button(
                "📥 Download Ranking Completo (Excel)",
                export_excel(ranking_df),
                file_name="ranking_valuebets.xlsx"
            )

            st.subheader("📌 Ranking Compacto")
            compact_df = build_compact_value_table(st.session_state.summaries)

            st.dataframe(
                compact_df.style.applymap(color_ev, subset=["EV_MAX"]),
                use_container_width=True
            )

            st.download_button(
                "📥 Download Ranking Compacto (Excel)",
                export_excel(compact_df),
                file_name="ranking_compacto.xlsx"
            )

    # ==============================================================================
    # TAB 4 — COMPARAÇÃO 1v1
    # ==============================================================================
    with tabs[3]:
        st.subheader("⚔️ Comparação 1v1 — ELO vs ELO20 vs W‑ELO vs W‑ELO20")

        col1, col2 = st.columns(2)
        with col1:
            p1 = st.selectbox("Jogador 1", [""] + sorted(all_players), key="cmp1")
        with col2:
            p2 = st.selectbox("Jogador 2", [""] + sorted(all_players), key="cmp2")

        surface = st.selectbox("Superfície", ["Clay", "Hard", "Grass"], key="cmp_surface")

        if surface == "Clay":
            elo_surf = elo_clay
            elo20_surf = elo_clay
        elif surface == "Hard":
            elo_surf = elo_hard
            elo20_surf = elo_hard
        else:
            elo_surf = elo_grass
            elo20_surf = elo_grass

        if p1 and p2 and p1 != p2:
            df_comp = build_elo_comparison_1v1(
                p1, p2, stats,
                elo_global, elo20_global,
                elo_surf, elo20_surf
            )

            st.dataframe(df_comp, use_container_width=True)

            fig = plot_elo_comparison_1v1(df_comp, p1, p2)
            st.pyplot(fig)

# ==============================================================================
# RUN
# ==============================================================================
if __name__ == "__main__":
    main()

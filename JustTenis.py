import os
import re
import time
import warnings
from collections import defaultdict
from datetime import datetime, timedelta
import io

import numpy as np
import pandas as pd
import streamlit as st

from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score, f1_score, log_loss, accuracy_score

import requests

warnings.filterwarnings('ignore')

st.set_page_config(page_title="🎾 ATP & Challenger Predictor", page_icon="🎾", layout="wide")

st.markdown("""
    <style>
    .prediction-card {
        background: linear-gradient(135deg, #1e3c72 0%, #2a5298 100%);
        color: white;
        padding: 20px;
        border-radius: 15px;
        margin: 15px 0;
        box-shadow: 0 8px 16px rgba(0,0,0,0.3);
    }
    .confidence-high {color: #00ff88; font-weight: bold; font-size: 1.15em;}
    .confidence-medium {color: #ffd700; font-weight: bold; font-size: 1.15em;}
    .confidence-low {color: #ff6b6b; font-weight: bold; font-size: 1.15em;}
    </style>
""", unsafe_allow_html=True)


# ==============================================================================
# NORMALIZAÇÃO DE SUPERFÍCIES
# ==============================================================================

def normalize_surface(s):
    if pd.isna(s):
        return "Hard"
    s = str(s).lower()
    if "clay" in s:
        return "Clay"
    if "grass" in s:
        return "Grass"
    return "Hard"


# ==============================================================================
# ELO RECENTE (ÚLTIMOS 20 JOGOS)
# ==============================================================================

def calculate_recent_elo(df, k=32, surface_k=25, window=20):
    players = set(df['winner'].dropna().unique()) | set(df['loser'].dropna().unique())

    elo = {p: 1500.0 for p in players}
    welo = {p: 1500.0 for p in players}
    surface_elo = {p: {'Hard':1500.0, 'Clay':1500.0, 'Grass':1500.0} for p in players}
    history = {p: [] for p in players}

    df_sorted = df.sort_values('date').copy()

    for _, row in df_sorted.iterrows():
        w = row['winner']
        l = row['loser']
        surf = row.get('surface', 'Hard')

        if surf not in ['Hard', 'Clay', 'Grass']:
            surf = 'Hard'

        if pd.isna(w) or pd.isna(l):
            continue

        history[w].append(1)
        history[l].append(0)

        w_recent = history[w][-window:]
        l_recent = history[l][-window:]

        w_form = sum(w_recent) / len(w_recent)
        l_form = sum(l_recent) / len(l_recent)

        r1 = elo[w] + 50 * (w_form - 0.5)
        r2 = elo[l] + 50 * (l_form - 0.5)

        exp1 = 1 / (1 + 10 ** ((r2 - r1) / 400))

        elo[w] += k * (1 - exp1)
        elo[l] += k * (0 - (1 - exp1))

        s1 = surface_elo[w][surf]
        s2 = surface_elo[l][surf]
        exp_s1 = 1 / (1 + 10 ** ((s2 - s1) / 400))

        surface_elo[w][surf] += surface_k * (1 - exp_s1)
        surface_elo[l][surf] += surface_k * (0 - (1 - exp_s1))

        welo[w] = welo[w] * 0.90 + elo[w] * 0.10
        welo[l] = welo[l] * 0.90 + elo[l] * 0.10

    return elo, welo, surface_elo


# ==============================================================================
# LOAD HISTÓRICO
# ==============================================================================

@st.cache_data(ttl=3600)
def load_historical_data(file_path):
    df = pd.read_excel(file_path)
    df.columns = [str(c).strip().lower().replace(' ', '_') for c in df.columns]

    rename_map = {
        'winner_name':'winner',
        'loser_name':'loser',
        'tourney_date':'date',
        'winner_rank':'wrank',
        'loser_rank':'lrank'
    }
    df.rename(columns={k:v for k,v in rename_map.items() if k in df.columns}, inplace=True)

    df['date'] = pd.to_datetime(df['date'], errors='coerce')

    if 'surface' in df.columns:
        df['surface'] = df['surface'].apply(normalize_surface)
    else:
        df['surface'] = "Hard"

    if 'score' in df.columns:
        def total_games_from_score(x):
            if pd.isna(x):
                return 22
            nums = [int(n) for n in re.findall(r'\d+', str(x))]
            return sum(nums) if nums else 22
        df['total_games'] = df['score'].apply(total_games_from_score)
    else:
        df['total_games'] = 22

    return df


# ==============================================================================
# PLAYER STATS
# ==============================================================================

def compute_player_stats(df):
    stats = {}
    elo, welo, surface_elo = calculate_recent_elo(df)
    
    players = set(df['winner'].dropna().unique()) | set(df['loser'].dropna().unique())
    for player in players:
        matches = df[(df['winner'] == player) | (df['loser'] == player)]
        if len(matches) == 0:
            continue
            
        wins = len(df[df['winner'] == player])
        total = wins + len(df[df['loser'] == player])
        win_rate = wins / total if total > 0 else 0.5
        
        surface_stats = {}
        for surf in ['Hard', 'Clay', 'Grass']:
            surf_matches = matches[matches['surface'] == surf]
            if len(surf_matches) > 0:
                surf_wins = len(surf_matches[surf_matches['winner'] == player])
                surface_stats[surf] = surf_wins / len(surf_matches)
            else:
                surface_stats[surf] = win_rate
        
        recent = matches.sort_values('date', ascending=False).head(10)
        recent_form = len(recent[recent['winner'] == player]) / len(recent) if len(recent) > 0 else win_rate
        
        stats[player] = {
            'win_rate': win_rate,
            'avg_rank': float(matches['wrank'].mean()) if 'wrank' in matches.columns else 150.0,
            'recent_form': recent_form,
            'avg_games': float(matches['total_games'].mean()),
            'elo': float(elo[player]),
            'welo': float(welo[player]),
            'surface_elo': surface_elo[player],
            'surface_win_rate': surface_stats,
            'matches_played': float(total)
        }
    return stats


# ==============================================================================
# H2H
# ==============================================================================

def build_h2h_dict(df):
    h2h = defaultdict(int)
    for _, row in df.iterrows():
        if pd.notna(row['winner']) and pd.notna(row['loser']):
            h2h[(row['winner'], row['loser'])] += 1
    return h2h


# ==============================================================================
# FEATURES (inclui ODDS)
# ==============================================================================

def build_features(p1, p2, surface, player_stats, h2h, match=None):
    if p1 not in player_stats or p2 not in player_stats:
        return None

    s1 = player_stats[p1]
    s2 = player_stats[p2]

    surf = surface if surface in ['Hard', 'Clay', 'Grass'] else 'Hard'

    h2h_p1 = h2h.get((p1, p2), 0)
    h2h_p2 = h2h.get((p2, p1), 0)
    h2h_ratio = (h2h_p1 + 1) / (h2h_p1 + h2h_p2 + 2)

    feat = [
        s1['elo'] / (s2['elo'] + 1),
        s1['welo'] / (s2['welo'] + 1),
        s1['surface_elo'][surf] / (s2['surface_elo'][surf] + 1),

        s1['win_rate'] - s2['win_rate'],
        s1['recent_form'] - s2['recent_form'],
        s1['surface_win_rate'][surf] - s2['surface_win_rate'][surf],

        (s2['avg_rank'] + 1) / (s1['avg_rank'] + 1),

        h2h_ratio,

        (s1['avg_games'] + s2['avg_games']) / 2,

        (s1['matches_played'] + 1) / (s2['matches_played'] + 1)
    ]

    prob1 = prob2 = 0.5
    if match is not None:
        odd1 = match.get("odd_p1")
        odd2 = match.get("odd_p2")
        if odd1 and odd2 and odd1 > 1 and odd2 > 1:
            prob1 = 1.0 / odd1
            prob2 = 1.0 / odd2

    feat.extend([prob1, prob2, prob1 - prob2])

    return feat
# ==============================================================================
# CROSS‑VALIDATION
# ==============================================================================

def cross_val_metrics(model, X, y, name=""):
    if len(np.unique(y)) < 2:
        st.write(f"⚠️ {name}: apenas uma classe presente.")
        return

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    aucs, f1s, logs, accs = [], [], [], []

    for train_idx, test_idx in skf.split(X, y):
        X_tr, X_te = X[train_idx], X[test_idx]
        y_tr, y_te = y[train_idx], y[test_idx]

        m = GradientBoostingClassifier(
            n_estimators=model.n_estimators,
            max_depth=model.max_depth,
            learning_rate=model.learning_rate,
            random_state=42
        )
        m.fit(X_tr, y_tr)

        probs = m.predict_proba(X_te)[:, 1]
        preds = (probs >= 0.5).astype(int)

        aucs.append(roc_auc_score(y_te, probs))
        logs.append(log_loss(y_te, probs))
        f1s.append(f1_score(y_te, preds))
        accs.append(accuracy_score(y_te, preds))

    st.write(
        f"🔍 {name} | AUC: {np.mean(aucs):.3f} | "
        f"F1: {np.mean(f1s):.3f} | "
        f"Acc: {np.mean(accs):.3f} | "
        f"LogLoss: {np.mean(logs):.3f}"
    )


# ==============================================================================
# FUNÇÃO PARA LIMPAR NaN / INF
# ==============================================================================

def clean_xy(X, y):
    X = np.array(X)
    y = np.array(y)
    mask = np.isfinite(X).all(axis=1)
    return X[mask], y[mask]


# ==============================================================================
# TREINO DOS MODELOS
# ==============================================================================

def train_models(df, player_stats, h2h):

    progress = st.progress(0)
    step = 0
    total_steps = 6

    X_winner, y_winner = [], []
    X_ou, y_ou = [], []
    X_sets, y_sets = [], []
    X_hcap, y_hcap = [], []

    for _, row in df.iterrows():
        w = row['winner']
        l = row['loser']
        if pd.isna(w) or pd.isna(l):
            continue

        surf = row.get('surface', 'Hard')
        total_games = row.get('total_games', 22)
        score = str(row.get('score', "")) if 'score' in df.columns else ""

        # AMOSTRA 1: p1 = winner
        feat1 = build_features(w, l, surf, player_stats, h2h, match=None)
        if feat1 is not None:
            X_winner.append(feat1)
            y_winner.append(1)

            X_ou.append(feat1)
            y_ou.append(1 if total_games > 21.5 else 0)

            sets = score.split()
            if len(sets) == 2:
                X_sets.append(feat1)
                y_sets.append(0)
            elif len(sets) >= 3:
                X_sets.append(feat1)
                y_sets.append(1)

            if total_games <= 20:
                X_hcap.append(feat1)
                y_hcap.append(1)
            elif total_games >= 24:
                X_hcap.append(feat1)
                y_hcap.append(0)

        # AMOSTRA 2: p1 = loser
        feat2 = build_features(l, w, surf, player_stats, h2h, match=None)
        if feat2 is not None:
            X_winner.append(feat2)
            y_winner.append(0)

    progress.progress((step := step + 1) / total_steps)

    # LIMPAR NaN / INF
    X_winner, y_winner = clean_xy(X_winner, y_winner)
    X_ou, y_ou = clean_xy(X_ou, y_ou)

    if len(X_sets):
        X_sets, y_sets = clean_xy(X_sets, y_sets)

    if len(X_hcap):
        X_hcap, y_hcap = clean_xy(X_hcap, y_hcap)

    model_winner = GradientBoostingClassifier(
        n_estimators=200, max_depth=4, learning_rate=0.05, random_state=42
    )
    model_winner.fit(X_winner, y_winner)
    progress.progress((step := step + 1) / total_steps)
    cross_val_metrics(model_winner, X_winner, y_winner, "Winner")
    progress.progress((step := step + 1) / total_steps)

    model_ou = GradientBoostingClassifier(
        n_estimators=150, max_depth=3, learning_rate=0.05, random_state=42
    )
    model_ou.fit(X_ou, y_ou)
    cross_val_metrics(model_ou, X_ou, y_ou, "Over/Under 21.5")
    progress.progress((step := step + 1) / total_steps)

    model_sets = None
    if len(X_sets) and len(np.unique(y_sets)) == 2:
        model_sets = GradientBoostingClassifier(
            n_estimators=150, max_depth=3, learning_rate=0.05, random_state=42
        )
        model_sets.fit(X_sets, y_sets)
        cross_val_metrics(model_sets, X_sets, y_sets, "Sets 2–0 vs 2–1")
    progress.progress((step := step + 1) / total_steps)

    model_hcap = None
    if len(X_hcap) and len(np.unique(y_hcap)) == 2:
        model_hcap = GradientBoostingClassifier(
            n_estimators=150, max_depth=3, learning_rate=0.05, random_state=42
        )
        model_hcap.fit(X_hcap, y_hcap)
        cross_val_metrics(model_hcap, X_hcap, y_hcap, "Handicap -2.5")
    progress.progress((step := step + 1) / total_steps)

    return model_winner, model_ou, model_sets, model_hcap


# ==============================================================================
# PREDIÇÃO (inclui ELO + WELO)
# ==============================================================================
def predict_match(model_winner, model_ou, model_sets, model_hcap,
                  player_stats, h2h, match):

    p1 = match['player1']
    p2 = match['player2']
    surface = match['surface']

    feat = build_features(p1, p2, surface, player_stats, h2h, match=match)
    if feat is None:
        return None

    X = np.array([feat])

    # Winner
    probs_w = model_winner.predict_proba(X)[0]
    p1_prob = probs_w[1]
    p2_prob = 1 - p1_prob

    # Over/Under
    probs_ou = model_ou.predict_proba(X)[0]
    over_prob = probs_ou[1]

    # Sets
    sets_pred = None
    if model_sets is not None:
        probs_sets = model_sets.predict_proba(X)[0]
        sets_pred = {
            'label': "2–1 / jogo longo" if probs_sets[1] > probs_sets[0] else "2–0 / vitória clara",
            'conf': max(probs_sets)
        }

    # Handicap
    hcap_pred = None
    if model_hcap is not None:
        probs_h = model_hcap.predict_proba(X)[0]
        hcap_pred = {
            'label': f"{p1} -2.5" if probs_h[1] > probs_h[0] else f"{p2} +2.5",
            'conf': max(probs_h)
        }

    # ELO + WELO
    elo_p1 = player_stats[p1]['elo']
    elo_p2 = player_stats[p2]['elo']
    welo_p1 = player_stats[p1]['welo']
    welo_p2 = player_stats[p2]['welo']

    return {
        'winner': p1 if p1_prob > p2_prob else p2,
        'winner_conf': max(p1_prob, p2_prob),
        'p1_prob': p1_prob,
        'p2_prob': p2_prob,
        'ou': "Over 21.5" if over_prob > 0.5 else "Under 21.5",
        'ou_conf': max(over_prob, 1 - over_prob),
        'sets': sets_pred,
        'handicap': hcap_pred,
        'elo_p1': elo_p1,
        'elo_p2': elo_p2,
        'welo_p1': welo_p1,
        'welo_p2': welo_p2
    }


# SCRAPER SOFASCORE (ODDS PLACEHOLDER)
# ==============================================================================

def scrape_matches_sofascore(days_ahead=0):
    try:
        # Endpoint funcional
        if days_ahead == 0:
            url = "https://api.sofascore.com/api/v1/sport/tennis/events/live"
        else:
            url = "https://api.sofascore.com/api/v1/sport/tennis/events/next"

        headers = {
            "User-Agent": "Mozilla/5.0",
            "Accept": "application/json"
        }

        r = requests.get(url, headers=headers, timeout=10)

        if r.status_code != 200:
            st.error(f"Erro SofaScore: HTTP {r.status_code}")
            return []

        data = r.json()

        if "events" not in data:
            return []

        matches = []
        now = datetime.utcnow().date()

        for ev in data["events"]:
            try:
                # Filtrar por data correta
                ts = ev["startTimestamp"]
                ev_date = datetime.utcfromtimestamp(ts).date()

                if days_ahead == 0 and ev_date != now:
                    continue
                if days_ahead == 1 and ev_date != now + timedelta(days=1):
                    continue

                tournament = ev["tournament"]["name"]
                category = ev["tournament"]["category"]["name"]

                if "WTA" in category.upper():
                    continue

                p1 = ev["homeTeam"]["name"]
                p2 = ev["awayTeam"]["name"]

                t = tournament.upper()
                surface = (
                    "Clay" if any(x in t for x in ["CLAY", "ROLAND", "MADRID", "ROME"]) else
                    "Grass" if any(x in t for x in ["WIMBLEDON", "HALLE"]) else
                    "Hard"
                )

                matches.append({
                    "tournament": tournament,
                    "player1": p1,
                    "player2": p2,
                    "surface": surface,
                    "type": category,
                    "odd_p1": None,
                    "odd_p2": None
                })

            except:
                continue

        return matches

    except Exception as e:
        st.error(f"Erro SofaScore: {e}")
        return []




# ==============================================================================
# APP
# ==============================================================================

def main():
    st.title("🎾 ATP & Challenger Tennis Predictor")
    st.markdown("**Winner, O/U, Sets, Handicap + CV + SofaScore + Export Excel + Odds + ELO/WELO**")

    with st.sidebar:
        uploaded_file = st.file_uploader("Upload Historical Data (Excel)", type=['xlsx'])
        st.markdown("---")

        col1, col2 = st.columns(2)
        with col1:
            if st.button("📅 Today"):
                st.session_state.matches = scrape_matches_sofascore(0)
        with col2:
            if st.button("📅 Tomorrow"):
                st.session_state.matches = scrape_matches_sofascore(1)

    if 'matches' not in st.session_state:
        st.session_state.matches = []

    if uploaded_file:
        with st.spinner("Training model..."):
            temp_path = "/tmp/tennis_data.xlsx"
            with open(temp_path, 'wb') as f:
                f.write(uploaded_file.read())
            
            df = load_historical_data(temp_path)
            st.write(f"📁 Loaded {len(df)} matches")

            player_stats = compute_player_stats(df)
            st.write(f"👥 Computed stats for {len(player_stats)} players")

            h2h = build_h2h_dict(df)

            model_w, model_ou, model_sets, model_hcap = train_models(df, player_stats, h2h)

            st.session_state.player_stats = player_stats
            st.session_state.h2h = h2h
            st.session_state.model_winner = model_w
            st.session_state.model_ou = model_ou
            st.session_state.model_sets = model_sets
            st.session_state.model_hcap = model_hcap
            st.session_state.models_trained = True

            st.success("✅ Model trained successfully!")

    if st.session_state.get('matches') and st.session_state.get('models_trained'):
        st.header("🎯 Predictions")

        results_export = []

        for m in st.session_state.matches:
            pred = predict_match(
                st.session_state.model_winner,
                st.session_state.model_ou,
                st.session_state.model_sets,
                st.session_state.model_hcap,
                st.session_state.player_stats,
                st.session_state.h2h,
                m
            )

            if pred is None:
                continue

            results_export.append({
                "Tournament": m['tournament'],
                "Player1": m['player1'],
                "Player2": m['player2'],
                "Surface": m['surface'],
                "Odd_P1": m.get("odd_p1"),
                "Odd_P2": m.get("odd_p2"),
                "Winner": pred['winner'],
                "Winner_Prob": pred['winner_conf'],
                "P1_Prob": pred['p1_prob'],
                "P2_Prob": pred['p2_prob'],
                "ELO_P1": pred['elo_p1'],
                "ELO_P2": pred['elo_p2'],
                "WELO_P1": pred['welo_p1'],
                "WELO_P2": pred['welo_p2'],
                "OU": pred['ou'],
                "OU_Prob": pred['ou_conf'],
                "Sets": pred['sets']['label'] if pred['sets'] else "",
                "Sets_Prob": pred['sets']['conf'] if pred['sets'] else "",
                "Handicap": pred['handicap']['label'] if pred['handicap'] else "",
                "Handicap_Prob": pred['handicap']['conf'] if pred['handicap'] else ""
            })

            conf_class = (
                "confidence-high" if pred['winner_conf'] >= 0.65 else
                "confidence-medium" if pred['winner_conf'] >= 0.55 else
                "confidence-low"
            )

            sets_line = ""
            if pred['sets'] is not None:
                sets_line = f"<br>🧩 Sets: {pred['sets']['label']} ({pred['sets']['conf']:.1%})"

            hcap_line = ""
            if pred['handicap'] is not None:
                hcap_line = f"<br>📏 Handicap -2.5: {pred['handicap']['label']} ({pred['handicap']['conf']:.1%})"

            st.markdown(f"""
            <div class="prediction-card">
                <b>{m['type']} • {m['surface']}</b><br>
                <h3>{m['player1']} vs {m['player2']}</h3>

                🏆 Winner: <span class="{conf_class}">
                    {pred['winner']} ({pred['winner_conf']:.1%})
                </span><br>

                📊 {m['player1']}: {pred['p1_prob']:.1%} |
                {m['player2']}: {pred['p2_prob']:.1%}<br>

                📈 ELO: {pred['elo_p1']:.0f} | {pred['elo_p2']:.0f}<br>
                🔥 WELO: {pred['welo_p1']:.0f} | {pred['welo_p2']:.0f}<br>

                🎲 O/U 21.5: {pred['ou']} ({pred['ou_conf']:.1%})
                {sets_line}
                {hcap_line}
            </div>
            """, unsafe_allow_html=True)

        if results_export:
            df_export = pd.DataFrame(results_export)
            buffer = io.BytesIO()
            df_export.to_excel(buffer, index=False, engine='openpyxl')
            buffer.seek(0)

            st.download_button(
                label="📥 Exportar previsões para Excel",
                data=buffer,
                file_name="tennis_predictions.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
            )


if __name__ == "__main__":
    main()

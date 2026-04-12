import os
import re
import time
import warnings
import random
from collections import defaultdict
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import streamlit as st

from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score, f1_score, log_loss, accuracy_score

import requests
from bs4 import BeautifulSoup

warnings.filterwarnings('ignore')

st.set_page_config(page_title="🎾 ATP & Challenger Predictor", page_icon="🎾", layout="wide")

st.markdown("""
    <style>
    .prediction-card {background: linear-gradient(135deg, #1e3c72 0%, #2a5298 100%); color: white; padding: 20px; border-radius: 15px; margin: 15px 0; box-shadow: 0 8px 16px rgba(0,0,0,0.3);}
    .confidence-high {color: #00ff88; font-weight: bold; font-size: 1.15em;}
    .confidence-medium {color: #ffd700; font-weight: bold; font-size: 1.15em;}
    .confidence-low {color: #ff6b6b; font-weight: bold; font-size: 1.15em;}
    </style>
""", unsafe_allow_html=True)
# ==============================================================================
# FEATURE BUILDER
# ==============================================================================

def build_features(p1, p2, surface, player_stats, h2h):
    if p1 not in player_stats or p2 not in player_stats:
        return None
    
    s1 = player_stats[p1]
    s2 = player_stats[p2]
    surf = surface if surface in ['Hard', 'Clay', 'Grass'] else 'Hard'

    h2h_p1 = h2h.get((p1, p2), 0)
    h2h_p2 = h2h.get((p2, p1), 0)
    h2h_total = h2h_p1 + h2h_p2 + 1

    feat = [
        s1['elo'] - s2['elo'],
        s1['welo'] - s2['welo'],
        s1['surface_elo'][surf] - s2['surface_elo'][surf],
        s1['win_rate'] - s2['win_rate'],
        s1['recent_form'] - s2['recent_form'],
        s1['surface_win_rate'][surf] - s2['surface_win_rate'][surf],
        s2['avg_rank'] - s1['avg_rank'],
        h2h_p1 / h2h_total,
        abs(s1['elo'] - s2['elo']),
        abs(s1['recent_form'] - s2['recent_form']),
        (s1['avg_games'] + s2['avg_games']) / 2
    ]
    if any(pd.isna(f) for f in feat):
        return None
    return feat


# ==============================================================================
# CROSS‑VALIDATION METRICS
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
# TRAINING (COM BARRA DE PROGRESSO)
# ==============================================================================

def train_models(df, player_stats, h2h):

    progress = st.progress(0)
    step = 0
    total_steps = 6

    X_winner, y_winner = [], []
    X_ou, y_ou = [], []
    X_sets, y_sets = [], []
    X_hcap, y_hcap = [], []

    # Step 1 — preparar dados
    for _, row in df.iterrows():
        w = row['winner']
        l = row['loser']
        if pd.isna(w) or pd.isna(l):
            continue

        surf = row.get('surface', 'Hard')
        total_games = row.get('total_games', 22)
        score = str(row.get('score', "")) if 'score' in df.columns else ""

        # Randomizar ordem para evitar viés
        if random.random() < 0.5:
            p1, p2 = w, l
        else:
            p1, p2 = l, w

        feat = build_features(p1, p2, surf, player_stats, h2h)
        if feat is None:
            continue

        # Winner
        y_label_winner = 1 if p1 == w else 0
        X_winner.append(feat)
        y_winner.append(y_label_winner)

        # Over/Under 21.5
        X_ou.append(feat)
        y_ou.append(1 if total_games > 21.5 else 0)

        # Sets (2–0 vs 2–1)
        sets = score.split()
        if len(sets) == 2:
            X_sets.append(feat)
            y_sets.append(0)
        elif len(sets) >= 3:
            X_sets.append(feat)
            y_sets.append(1)

        # Handicap -2.5
        if total_games <= 20:
            X_hcap.append(feat)
            y_hcap.append(1)
        elif total_games >= 24:
            X_hcap.append(feat)
            y_hcap.append(0)

    progress.progress((step := step + 1) / total_steps)

    # Convert to numpy
    X_winner = np.array(X_winner)
    X_ou = np.array(X_ou)
    X_sets = np.array(X_sets) if len(X_sets) else None
    X_hcap = np.array(X_hcap) if len(X_hcap) else None

    y_winner = np.array(y_winner)
    y_ou = np.array(y_ou)
    y_sets = np.array(y_sets) if X_sets is not None else None
    y_hcap = np.array(y_hcap) if X_hcap is not None else None

    # Step 2 — Winner
    model_winner = GradientBoostingClassifier(
        n_estimators=200, max_depth=4, learning_rate=0.05, random_state=42
    )
    model_winner.fit(X_winner, y_winner)
    progress.progress((step := step + 1) / total_steps)

    cross_val_metrics(model_winner, X_winner, y_winner, "Winner")
    progress.progress((step := step + 1) / total_steps)

    # Step 3 — Over/Under
    model_ou = GradientBoostingClassifier(
        n_estimators=150, max_depth=3, learning_rate=0.05, random_state=42
    )
    model_ou.fit(X_ou, y_ou)
    cross_val_metrics(model_ou, X_ou, y_ou, "Over/Under 21.5")
    progress.progress((step := step + 1) / total_steps)

    # Step 4 — Sets
    model_sets = None
    if X_sets is not None and len(np.unique(y_sets)) == 2:
        model_sets = GradientBoostingClassifier(
            n_estimators=150, max_depth=3, learning_rate=0.05, random_state=42
        )
        model_sets.fit(X_sets, y_sets)
        cross_val_metrics(model_sets, X_sets, y_sets, "Sets 2–0 vs 2–1")
    progress.progress((step := step + 1) / total_steps)

    # Step 5 — Handicap
    model_hcap = None
    if X_hcap is not None and len(np.unique(y_hcap)) == 2:
        model_hcap = GradientBoostingClassifier(
            n_estimators=150, max_depth=3, learning_rate=0.05, random_state=42
        )
        model_hcap.fit(X_hcap, y_hcap)
        cross_val_metrics(model_hcap, X_hcap, y_hcap, "Handicap -2.5")
    progress.progress((step := step + 1) / total_steps)

    return model_winner, model_ou, model_sets, model_hcap
# ==============================================================================
# PREDICTION
# ==============================================================================

def predict_match(model_winner, model_ou, model_sets, model_hcap,
                  player_stats, h2h, p1_name, p2_name, surface):

    feat = build_features(p1_name, p2_name, surface, player_stats, h2h)
    if feat is None:
        return None

    X = np.array([feat])

    probs_w = model_winner.predict_proba(X)[0]
    p1_prob = probs_w[1]
    p2_prob = probs_w[0]

    probs_ou = model_ou.predict_proba(X)[0]
    over_prob = probs_ou[1]
    under_prob = probs_ou[0]

    sets_pred = None
    if model_sets is not None:
        probs_sets = model_sets.predict_proba(X)[0]
        sets_pred = {
            'label': "2–1 / jogo longo" if probs_sets[1] > probs_sets[0] else "2–0 / vitória clara",
            'conf': max(probs_sets)
        }

    hcap_pred = None
    if model_hcap is not None:
        probs_h = model_hcap.predict_proba(X)[0]
        hcap_pred = {
            'label': f"{p1_name} -2.5" if probs_h[1] > probs_h[0] else f"{p2_name} +2.5",
            'conf': max(probs_h)
        }

    return {
        'winner': p1_name if p1_prob > p2_prob else p2_name,
        'winner_conf': max(p1_prob, p2_prob),
        'p1_prob': p1_prob,
        'p2_prob': p2_prob,
        'ou': "Over 21.5" if over_prob > under_prob else "Under 21.5",
        'ou_conf': max(over_prob, under_prob),
        'sets': sets_pred,
        'handicap': hcap_pred
    }


# ==============================================================================
# SCRAPER — RAPIDAPI (ATP/WTA/ITF)
# ==============================================================================

def scrape_matches_rapidapi(days_ahead=0):
    try:
        API_KEY = "bba6af0e8dmsh6350139b0f77a4ap16b6fajsn219553636a44"

        base_url = "https://tennis-api-atp-wta-itf.p.rapidapi.com/tennis/v2/events/date"
        target_date = (datetime.utcnow() + timedelta(days=days_ahead)).strftime("%Y-%m-%d")

        url = f"{base_url}/{target_date}"

        headers = {
            "x-rapidapi-key": API_KEY,
            "x-rapidapi-host": "tennis-api-atp-wta-itf.p.rapidapi.com"
        }

        r = requests.get(url, headers=headers)

        if r.status_code != 200:
            st.error(f"Erro HTTP {r.status_code}")
            return []

        data = r.json()

        if "events" not in data:
            return []

        matches = []

        for event in data["events"]:
            try:
                tournament = event["tournament"]["name"]
                category = event["tournament"]["level"]

                if "WTA" in category.upper():
                    continue

                p1 = event["home"]["name"]
                p2 = event["away"]["name"]

                t = tournament.upper()
                surface = "Clay" if any(x in t for x in ["CLAY","ROLAND","MADRID","ROME"]) else \
                          "Grass" if any(x in t for x in ["WIMBLEDON","HALLE"]) else "Hard"

                matches.append({
                    "tournament": tournament,
                    "player1": p1,
                    "player2": p2,
                    "surface": surface,
                    "type": category
                })

            except:
                continue

        return matches

    except Exception as e:
        st.error(f"Erro RapidAPI: {e}")
        return []


# ==============================================================================
# APP
# ==============================================================================

def main():
    st.title("🎾 ATP & Challenger Tennis Predictor")
    st.markdown("**Versão completa: Winner, O/U, Sets, Handicap, CV + Barra de Progresso**")

    with st.sidebar:
        uploaded_file = st.file_uploader("Upload Historical Data (Excel)", type=['xlsx'])
        st.markdown("---")

        col1, col2 = st.columns(2)
        with col1:
            if st.button("📅 Today"):
                st.session_state.matches = scrape_matches_rapidapi(0)
        with col2:
            if st.button("📅 Tomorrow"):
                st.session_state.matches = scrape_matches_rapidapi(1)

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

        for m in st.session_state.matches:
            pred = predict_match(
                st.session_state.model_winner,
                st.session_state.model_ou,
                st.session_state.model_sets,
                st.session_state.model_hcap,
                st.session_state.player_stats,
                st.session_state.h2h,
                m['player1'],
                m['player2'],
                m['surface']
            )

            if pred is None:
                continue

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

                🎲 O/U 21.5: {pred['ou']} ({pred['ou_conf']:.1%})
                {sets_line}
                {hcap_line}
            </div>
            """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()

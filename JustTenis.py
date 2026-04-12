import os
import re
import time
import warnings
import pandas as pd
import numpy as np
from collections import defaultdict
from datetime import datetime
import streamlit as st

from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score, f1_score, log_loss, accuracy_score

from bs4 import BeautifulSoup
from selenium import webdriver
from selenium.webdriver.chrome.options import Options

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
# ELO + WELO
# ==============================================================================

def calculate_elo_ratings(df, k=32, surface_k=25):
    winner_col = 'winner' if 'winner' in df.columns else 'winner_name'
    loser_col = 'loser' if 'loser' in df.columns else 'loser_name'
    players = set(df[winner_col].dropna().unique()) | set(df[loser_col].dropna().unique())
    
    elo = {p: 1500.0 for p in players}
    welo = {p: 1500.0 for p in players}
    surface_elo = {p: {'Hard':1500.0, 'Clay':1500.0, 'Grass':1500.0} for p in players}
    
    df_sorted = df.sort_values('date' if 'date' in df.columns else df.columns[0]).copy()
    
    for _, row in df_sorted.iterrows():
        p1 = row[winner_col]
        p2 = row[loser_col]
        surf = row.get('surface', 'Hard')
        if pd.isna(p1) or pd.isna(p2):
            continue
            
        r1, r2 = elo.get(p1, 1500), elo.get(p2, 1500)
        exp1 = 1 / (1 + 10 ** ((r2 - r1) / 400))
        elo[p1] = elo.get(p1, 1500) + k * (1 - exp1)
        elo[p2] = elo.get(p2, 1500) + k * (0 - (1 - exp1))
        
        if surf in surface_elo.get(p1, {}):
            s1 = surface_elo[p1][surf]
            s2 = surface_elo[p2][surf]
            exp_s1 = 1 / (1 + 10 ** ((s2 - s1) / 400))
            surface_elo[p1][surf] += surface_k * (1 - exp_s1)
            surface_elo[p2][surf] += surface_k * (0 - (1 - exp_s1))
        
        welo[p1] = welo.get(p1, 1500) * 0.96 + elo.get(p1, 1500) * 0.04
        welo[p2] = welo.get(p2, 1500) * 0.96 + elo.get(p2, 1500) * 0.04
    
    return elo, welo, surface_elo

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
    if 'date' in df.columns:
        df['date'] = pd.to_datetime(df['date'].astype(str), errors='coerce')
    if 'score' in df.columns:
        def total_games_from_score(x):
            if pd.isna(x):
                return 22
            nums = [int(n) for n in re.findall(r'\d+', str(x)) if n.isdigit()]
            return sum(nums) if nums else 22
        df['total_games'] = df['score'].apply(total_games_from_score)
    else:
        df['total_games'] = 22
    return df

def compute_player_stats(df):
    stats = {}
    elo, welo, surface_elo = calculate_elo_ratings(df)
    winner_col = 'winner'
    loser_col = 'loser'
    
    for player in set(df[winner_col].dropna().unique()) | set(df[loser_col].dropna().unique()):
        matches = df[(df[winner_col] == player) | (df[loser_col] == player)]
        if len(matches) == 0:
            continue
            
        wins = len(df[df[winner_col] == player])
        total = wins + len(df[df[loser_col] == player])
        win_rate = wins / total if total > 0 else 0.5
        
        surface_stats = {}
        for surf in ['Hard', 'Clay', 'Grass']:
            if 'surface' in matches.columns:
                surf_matches = matches[matches['surface'] == surf]
            else:
                surf_matches = pd.DataFrame()
            if len(surf_matches) > 0:
                surf_wins = len(surf_matches[surf_matches[winner_col] == player])
                surface_stats[surf] = surf_wins / len(surf_matches)
            else:
                surface_stats[surf] = win_rate
        
        recent = matches.sort_values('date' if 'date' in matches.columns else matches.columns[0],
                                     ascending=False).head(10)
        recent_form = len(recent[recent[winner_col] == player]) / len(recent) if len(recent) > 0 else win_rate
        
        stats[player] = {
            'win_rate': win_rate,
            'avg_rank': float(matches['wrank'].mean()) if 'wrank' in matches.columns else 150.0,
            'recent_form': recent_form,
            'avg_games': float(matches['total_games'].mean()) if 'total_games' in matches.columns else 22.0,
            'elo': float(elo.get(player, 1500)),
            'welo': float(welo.get(player, 1500)),
            'surface_elo': surface_elo.get(player, {'Hard':1500.0, 'Clay':1500.0, 'Grass':1500.0}),
            'surface_win_rate': surface_stats
        }
    return stats

def build_h2h_dict(df):
    h2h = defaultdict(int)
    winner_col = 'winner'
    loser_col = 'loser'
    for _, row in df.iterrows():
        if pd.notna(row[winner_col]) and pd.notna(row[loser_col]):
            h2h[(row[winner_col], row[loser_col])] += 1
    return h2h

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

    return [
        s1['elo'] - s2['elo'],
        s1['welo'] - s2['welo'],
        s1['surface_elo'].get(surf, 1500) - s2['surface_elo'].get(surf, 1500),
        s1['win_rate'] - s2['win_rate'],
        s1['recent_form'] - s2['recent_form'],
        s1['surface_win_rate'].get(surf, s1['win_rate']) - s2['surface_win_rate'].get(surf, s2['win_rate']),
        s2['avg_rank'] - s1['avg_rank'],
        h2h_p1 / h2h_total,
        abs(s1['elo'] - s2['elo']),
        abs(s1['recent_form'] - s2['recent_form']),
        (s1['avg_games'] + s2['avg_games']) / 2
    ]

# ==============================================================================
# TRAINING WITH CV + EXTRA TARGETS
# ==============================================================================

def cross_val_metrics(model, X, y, name=""):
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
        probs = m.predict_proba(X_te)[:,1]
        preds = (probs >= 0.5).astype(int)
        try:
            aucs.append(roc_auc_score(y_te, probs))
            logs.append(log_loss(y_te, probs, eps=1e-7))
        except:
            pass
        f1s.append(f1_score(y_te, preds))
        accs.append(accuracy_score(y_te, preds))
    st.write(f"🔍 {name} | AUC: {np.mean(aucs):.3f} | F1: {np.mean(f1s):.3f} | Acc: {np.mean(accs):.3f} | LogLoss: {np.mean(logs):.3f}")

def train_models(df, player_stats, h2h):
    X_winner, y_winner = [], []
    X_ou, y_ou = [], []
    X_sets, y_sets = [], []
    X_hcap, y_hcap = [], []

    winner_col = 'winner'
    loser_col = 'loser'

    for _, row in df.iterrows():
        w = row[winner_col]
        l = row[loser_col]
        surf = row.get('surface', 'Hard')
        total_games = row.get('total_games', 22)
        score = str(row.get('score', ""))

        feat = build_features(w, l, surf, player_stats, h2h)
        if feat is None:
            continue

        # Winner model: p1 = winner, p2 = loser → label 1
        X_winner.append(feat)
        y_winner.append(1)

        # Over/Under 21.5
        X_ou.append(feat)
        y_ou.append(1 if total_games > 21.5 else 0)

        # Sets model (best of 3): 2–0 vs 2–1 (apenas se score tiver 2 sets ou 3 sets)
        sets = score.split()
        if len(sets) >= 2:
            # heurística simples: se há 2 sets → 2–0, se há 3+ → 2–1 (ou 3–0 em BO5, mas ignoramos BO5)
            if len(sets) == 2:
                y_sets.append(0)  # 2–0
                X_sets.append(feat)
            elif len(sets) >= 3:
                y_sets.append(1)  # 2–1 (ou jogo longo)
                X_sets.append(feat)

        # Handicap -2.5 jogos para o vencedor
        # Se vencedor ganhou por margem >= 3 jogos → cobre -2.5
        # Aproximação: total_games e sets não dão margem exata, mas usamos heurística:
        # jogos médios por set * diferença de sets
        # Para simplificar: se total_games <= 20 → provável blowout → cobre -2.5
        # se total_games >= 24 → jogo equilibrado → não cobre
        if total_games <= 20:
            y_hcap.append(1)  # cobre -2.5
            X_hcap.append(feat)
        elif total_games >= 24:
            y_hcap.append(0)  # não cobre
            X_hcap.append(feat)
        # entre 21 e 23 ignoramos (zona cinzenta)

    X_winner = np.nan_to_num(np.array(X_winner), nan=0.0)
    X_ou = np.nan_to_num(np.array(X_ou), nan=0.0)
    X_sets = np.nan_to_num(np.array(X_sets), nan=0.0) if len(X_sets) > 0 else None
    X_hcap = np.nan_to_num(np.array(X_hcap), nan=0.0) if len(X_hcap) > 0 else None

    y_winner = np.array(y_winner)
    y_ou = np.array(y_ou)
    y_sets = np.array(y_sets) if len(X_sets) > 0 else None
    y_hcap = np.array(y_hcap) if len(X_hcap) > 0 else None

    st.write(f"📊 Amostras Winner: {len(X_winner)}")
    st.write(f"📊 Amostras O/U: {len(X_ou)}")
    if X_sets is not None:
        st.write(f"📊 Amostras Sets (2–0 vs 2–1): {len(X_sets)}")
    if X_hcap is not None:
        st.write(f"📊 Amostras Handicap -2.5: {len(X_hcap)}")

    # Winner model
    model_winner = GradientBoostingClassifier(
        n_estimators=200,
        max_depth=4,
        learning_rate=0.05,
        random_state=42
    )
    model_winner.fit(X_winner, y_winner)
    cross_val_metrics(model_winner, X_winner, y_winner, name="Winner")

    # Over/Under model
    model_ou = GradientBoostingClassifier(
        n_estimators=150,
        max_depth=3,
        learning_rate=0.05,
        random_state=42
    )
    model_ou.fit(X_ou, y_ou)
    cross_val_metrics(model_ou, X_ou, y_ou, name="Over/Under 21.5")

    # Sets model
    model_sets = None
    if X_sets is not None and len(np.unique(y_sets)) == 2:
        model_sets = GradientBoostingClassifier(
            n_estimators=150,
            max_depth=3,
            learning_rate=0.05,
            random_state=42
        )
        model_sets.fit(X_sets, y_sets)
        cross_val_metrics(model_sets, X_sets, y_sets, name="Sets 2–0 vs 2–1")

    # Handicap model
    model_hcap = None
    if X_hcap is not None and len(np.unique(y_hcap)) == 2:
        model_hcap = GradientBoostingClassifier(
            n_estimators=150,
            max_depth=3,
            learning_rate=0.05,
            random_state=42
        )
        model_hcap.fit(X_hcap, y_hcap)
        cross_val_metrics(model_hcap, X_hcap, y_hcap, name="Handicap -2.5")

    return model_winner, model_ou, model_sets, model_hcap

# ==============================================================================
# PREDICTION
# ==============================================================================

def predict_match(model_winner, model_ou, model_sets, model_hcap,
                  player_stats, h2h, p1_name, p2_name, surface):
    feat = build_features(p1_name, p2_name, surface, player_stats, h2h)
    if feat is None:
        return None

    X = np.nan_to_num(np.array([feat]), nan=0.0)

    # Winner
    probs_w = model_winner.predict_proba(X)[0]
    p1_prob = probs_w[1]
    p2_prob = probs_w[0]

    # Over/Under
    probs_ou = model_ou.predict_proba(X)[0]
    over_prob = probs_ou[1]
    under_prob = probs_ou[0]

    # Sets
    sets_pred = None
    if model_sets is not None:
        probs_sets = model_sets.predict_proba(X)[0]
        prob_21 = probs_sets[1]
        prob_20 = probs_sets[0]
        sets_pred = {
            'label': "2–1 / jogo longo" if prob_21 > prob_20 else "2–0 / vitória clara",
            'conf': max(prob_21, prob_20),
            'p20': prob_20,
            'p21': prob_21
        }

    # Handicap -2.5
    hcap_pred = None
    if model_hcap is not None:
        probs_h = model_hcap.predict_proba(X)[0]
        prob_cover = probs_h[1]
        prob_not = probs_h[0]
        hcap_pred = {
            'label': "Cobre -2.5" if prob_cover > prob_not else "Não cobre -2.5",
            'conf': max(prob_cover, prob_not),
            'p_cover': prob_cover,
            'p_not': prob_not
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
# SCRAPER
# ==============================================================================

@st.cache_data(ttl=1800)
def scrape_matches_flashscore(days_ahead=0):
    matches = []
    try:
        chrome_options = Options()
        chrome_options.add_argument("--headless")
        chrome_options.add_argument("--no-sandbox")
        chrome_options.add_argument("--disable-dev-shm-usage")
        
        driver = webdriver.Chrome(options=chrome_options)
        driver.get("https://www.flashscore.com/tennis/")
        time.sleep(12)
        for _ in range(3):
            driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
            time.sleep(4)
        soup = BeautifulSoup(driver.page_source, 'html.parser')
        driver.quit()
        
        cards = soup.find_all('div', class_=lambda x: x and 'event__match' in str(x))
        for card in cards:
            try:
                tour = card.find_previous('div', class_=lambda x: x and any(k in str(x).lower() for k in ['tournament','header']))
                tournament = tour.get_text(strip=True) if tour else ""
                if any(w in tournament.upper() for w in ['WTA', 'WOMEN', 'BILLIE']):
                    continue
                
                players = card.find_all('div', class_=lambda x: x and 'participant' in str(x))
                if len(players) < 2:
                    continue
                    
                p1 = players[0].get_text(strip=True)
                p2 = players[1].get_text(strip=True)
                
                tour_u = tournament.upper()
                surface = "Clay" if any(x in tour_u for x in ['CLAY','ROLAND','MADRID','ROME']) else \
                          "Grass" if any(x in tour_u for x in ['WIMBLEDON','HALLE']) else "Hard"
                match_type = "Challenger" if "CHALLENGER" in tour_u else "ATP"
                
                matches.append({'tournament': tournament, 'player1': p1, 'player2': p2, 'surface': surface, 'type': match_type})
            except:
                continue
        st.success(f"✅ Scraped {len(matches)} matches")
        return matches
    except Exception as e:
        st.error(f"Scraping failed: {e}")
        return []

# ==============================================================================
# APP
# ==============================================================================

def main():
    st.title("🎾 ATP & Challenger Tennis Predictor")
    st.markdown("**Versão corrigida com métricas, sets e handicap**")

    with st.sidebar:
        uploaded_file = st.file_uploader("Upload Historical Data (Excel)", type=['xlsx'])
        st.markdown("---")
        col1, col2 = st.columns(2)
        with col1:
            if st.button("📅 Today"):
                st.session_state.matches = scrape_matches_flashscore(0)
        with col2:
            if st.button("📅 Tomorrow"):
                st.session_state.matches = scrape_matches_flashscore(1)

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
            
            conf_class = "confidence-high" if pred['winner_conf'] >= 0.65 else "confidence-medium" if pred['winner_conf'] >= 0.55 else "confidence-low"
            
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
                🏆 Winner: <span class="{conf_class}">{pred['winner']} ({pred['winner_conf']:.1%})</span><br>
                📊 {m['player1']}: {pred['p1_prob']:.1%} | {m['player2']}: {pred['p2_prob']:.1%}<br>
                🎲 O/U 21.5: {pred['ou']} ({pred['ou_conf']:.1%}){sets_line}{hcap_line}
            </div>
            """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()

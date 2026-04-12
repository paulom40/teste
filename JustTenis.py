import os
import re
import time
import warnings
import pandas as pd
import numpy as np
import random
import io
from datetime import datetime, timedelta
from collections import defaultdict
import streamlit as st
from sklearn.ensemble import GradientBoostingClassifier
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
        if pd.isna(p1) or pd.isna(p2): continue
            
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
    rename_map = {'winner_name':'winner', 'loser_name':'loser', 'tourney_date':'date', 'winner_rank':'wrank', 'loser_rank':'lrank'}
    df.rename(columns={k:v for k,v in rename_map.items() if k in df.columns}, inplace=True)
    if 'date' in df.columns:
        df['date'] = pd.to_datetime(df['date'].astype(str), errors='coerce')
    if 'score' in df.columns:
        df['total_games'] = df['score'].apply(lambda x: 22 if pd.isna(x) else sum(int(n) for n in re.findall(r'\d+', str(x)) if n.isdigit()) or 22)
    return df

def compute_player_stats(df):
    stats = {}
    elo, welo, surface_elo = calculate_elo_ratings(df)
    winner_col = 'winner' if 'winner' in df.columns else 'winner_name'
    loser_col = 'loser' if 'loser' in df.columns else 'loser_name'
    
    for player in set(df[winner_col].dropna().unique()) | set(df[loser_col].dropna().unique()):
        matches = df[(df[winner_col] == player) | (df[loser_col] == player)]
        if len(matches) == 0: continue
            
        wins = len(df[df[winner_col] == player])
        total = wins + len(df[df[loser_col] == player])
        win_rate = wins / total if total > 0 else 0.5
        
        # Surface-specific win rates
        surface_stats = {}
        for surf in ['Hard', 'Clay', 'Grass']:
            surf_matches = matches[matches['surface'] == surf] if 'surface' in matches.columns else pd.DataFrame()
            if len(surf_matches) > 0:
                surf_wins = len(surf_matches[surf_matches[winner_col] == player])
                surface_stats[surf] = surf_wins / len(surf_matches)
            else:
                surface_stats[surf] = win_rate
        
        recent = matches.sort_values('date' if 'date' in matches.columns else matches.columns[0], ascending=False).head(10)
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
    winner_col = 'winner' if 'winner' in df.columns else 'winner_name'
    loser_col = 'loser' if 'loser' in df.columns else 'loser_name'
    for _, row in df.iterrows():
        if pd.notna(row[winner_col]) and pd.notna(row[loser_col]):
            h2h[(row[winner_col], row[loser_col])] += 1
    return h2h

# ==============================================================================
# FIXED TRAINING - Key changes here
# ==============================================================================
def train_models(df, player_stats, h2h):
    X_data, y_winner, y_games = [], [], []
    winner_col = 'winner' if 'winner' in df.columns else 'winner_name'
    loser_col = 'loser' if 'loser' in df.columns else 'loser_name'
    
    for _, row in df.iterrows():
        w = row.get(winner_col)
        l = row.get(loser_col)
        surf = row.get('surface', 'Hard')
        
        if w not in player_stats or l not in player_stats or pd.isna(w) or pd.isna(l):
            continue
        
        s_w = player_stats[w]
        s_l = player_stats[l]
        
        # FIXED: Calculate features as (player1 - player2) where player1 is the WINNER
        # This way, label=1 means the "better" player won (positive features)
        h2h_rate = h2h.get((w, l), 0) / (h2h.get((w, l), 0) + h2h.get((l, w), 0) + 1)
        
        features = [
            s_w['win_rate'] - s_l['win_rate'],
            (s_l['avg_rank'] - s_w['avg_rank']) / 50,  # Lower rank is better, so inverted
            s_w['recent_form'] - s_l['recent_form'],
            s_w['surface_win_rate'].get(surf, s_w['win_rate']) - s_l['surface_win_rate'].get(surf, s_l['win_rate']),
            (s_w['elo'] - s_l['elo']) / 80,
            (s_w['welo'] - s_l['welo']) / 80,
            (s_w['surface_elo'].get(surf, 1500) - s_l['surface_elo'].get(surf, 1500)) / 80,
            h2h_rate,
            s_w['elo'] / (s_l['elo'] + 100),
            abs(s_w['elo'] - s_l['elo']) / 100
        ]
        
        X_data.append(features)
        y_winner.append(1)  # Winner always gets label 1
        y_games.append(row.get('total_games', 22))
        
        # FIXED: Add the reverse matchup with inverted features and label 0
        features_rev = [
            s_l['win_rate'] - s_w['win_rate'],
            (s_w['avg_rank'] - s_l['avg_rank']) / 50,
            s_l['recent_form'] - s_w['recent_form'],
            s_l['surface_win_rate'].get(surf, s_l['win_rate']) - s_w['surface_win_rate'].get(surf, s_w['win_rate']),
            (s_l['elo'] - s_w['elo']) / 80,
            (s_l['welo'] - s_w['welo']) / 80,
            (s_l['surface_elo'].get(surf, 1500) - s_w['surface_elo'].get(surf, 1500)) / 80,
            h2h.get((l, w), 0) / (h2h.get((w, l), 0) + h2h.get((l, w), 0) + 1),
            s_l['elo'] / (s_w['elo'] + 100),
            abs(s_l['elo'] - s_w['elo']) / 100
        ]
        
        X_data.append(features_rev)
        y_winner.append(0)  # Loser gets label 0
        y_games.append(row.get('total_games', 22))
    
    X = np.nan_to_num(np.array(X_data, dtype=np.float64), nan=0.0)
    y_w = np.array(y_winner)
    y_g = np.array(y_games)
    
    model_winner = GradientBoostingClassifier(
        n_estimators=250, 
        max_depth=4, 
        learning_rate=0.1, 
        subsample=0.9, 
        min_samples_leaf=2,
        random_state=42
    )
    model_winner.fit(X, y_w)
    
    # FIXED: Train on actual game counts, not just binary over/under
    model_ou = GradientBoostingClassifier(n_estimators=150, max_depth=4, random_state=42)
    model_ou.fit(X, (y_g > 21.5).astype(int))
    
    return model_winner, model_ou

def predict_match(model_winner, model_ou, player_stats, h2h, p1, p2, surface):
    s1 = player_stats.get(p1, {
        'win_rate':0.5, 'avg_rank':150, 'recent_form':0.5, 'elo':1500, 
        'welo':1500, 'surface_elo':{'Hard':1500, 'Clay':1500, 'Grass':1500},
        'surface_win_rate':{'Hard':0.5, 'Clay':0.5, 'Grass':0.5}
    })
    s2 = player_stats.get(p2, {
        'win_rate':0.5, 'avg_rank':150, 'recent_form':0.5, 'elo':1500, 
        'welo':1500, 'surface_elo':{'Hard':1500, 'Clay':1500, 'Grass':1500},
        'surface_win_rate':{'Hard':0.5, 'Clay':0.5, 'Grass':0.5}
    })
    
    h2h_rate = h2h.get((p1, p2), 0) / (h2h.get((p1, p2), 0) + h2h.get((p2, p1), 0) + 1)
    
    # FIXED: Features represent (p1 - p2), so prediction is "will p1 win?"
    features = np.nan_to_num(np.array([[ 
        s1['win_rate'] - s2['win_rate'],
        (s2['avg_rank'] - s1['avg_rank']) / 50,  # Lower rank is better
        s1['recent_form'] - s2['recent_form'],
        s1['surface_win_rate'].get(surface, s1['win_rate']) - s2['surface_win_rate'].get(surface, s2['win_rate']),
        (s1['elo'] - s2['elo']) / 80,
        (s1['welo'] - s2['welo']) / 80,
        (s1['surface_elo'].get(surface, 1500) - s2['surface_elo'].get(surface, 1500)) / 80,
        h2h_rate,
        s1['elo'] / (s2['elo'] + 100),
        abs(s1['elo'] - s2['elo']) / 100
    ]], dtype=np.float64), nan=0.0)
    
    # FIXED: predict_proba returns [prob_class_0, prob_class_1]
    # prob_class_1 is the probability that p1 wins (label=1)
    p1_win_prob = model_winner.predict_proba(features)[0][1]
    p2_win_prob = 1 - p1_win_prob
    
    # Over/Under prediction
    over_prob = model_ou.predict_proba(features)[0][1]
    
    return {
        'winner': p1 if p1_win_prob > 0.5 else p2,
        'winner_conf': max(p1_win_prob, p2_win_prob),
        'p1_prob': p1_win_prob,
        'p2_prob': p2_win_prob,
        'ou': "Over 21.5" if over_prob > 0.5 else "Under 21.5",
        'ou_conf': max(over_prob, 1 - over_prob),
        'exp_games': (s1.get('avg_games', 22) + s2.get('avg_games', 22)) / 2
    }

# Scraper (same as before)
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
                if any(w in tournament.upper() for w in ['WTA', 'WOMEN', 'BILLIE']): continue
                
                players = card.find_all('div', class_=lambda x: x and 'participant' in str(x))
                if len(players) < 2: continue
                    
                p1 = players[0].get_text(strip=True)
                p2 = players[1].get_text(strip=True)
                
                tour_u = tournament.upper()
                surface = "Clay" if any(x in tour_u for x in ['CLAY','ROLAND','MADRID','ROME']) else "Grass" if any(x in tour_u for x in ['WIMBLEDON','HALLE']) else "Hard"
                match_type = "Challenger" if "CHALLENGER" in tour_u else "ATP"
                
                matches.append({'tournament': tournament, 'player1': p1, 'player2': p2, 'surface': surface, 'type': match_type})
            except:
                continue
        st.success(f"✅ Scraped {len(matches)} matches")
        return matches
    except Exception as e:
        st.error(f"Scraping failed: {e}")
        return []

def main():
    st.title("🎾 ATP & Challenger Tennis Predictor")
    st.markdown("**Fixed ELO Model - Corrected Predictions**")

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
            player_stats = compute_player_stats(df)
            h2h = build_h2h_dict(df)
            model_w, model_ou = train_models(df, player_stats, h2h)
            
            st.session_state.player_stats = player_stats
            st.session_state.h2h = h2h
            st.session_state.model_winner = model_w
            st.session_state.model_ou = model_ou
            st.session_state.models_trained = True
            st.success("✅ Model trained!")

    if st.session_state.get('matches') and st.session_state.get('models_trained'):
        st.header("🎯 Predictions")
        for m in st.session_state.matches:
            pred = predict_match(st.session_state.model_winner, st.session_state.model_ou, 
                               st.session_state.player_stats, st.session_state.h2h, 
                               m['player1'], m['player2'], m['surface'])
            
            conf_class = "confidence-high" if pred['winner_conf'] >= 0.65 else "confidence-medium" if pred['winner_conf'] >= 0.55 else "confidence-low"
            
            st.markdown(f"""
            <div class="prediction-card">
                <b>{m['type']} • {m['surface']}</b><br>
                <h3>{m['player1']} vs {m['player2']}</h3>
                🏆 Winner: <span class="{conf_class}">{pred['winner']} ({pred['winner_conf']:.1%})</span><br>
                📊 {m['player1']}: {pred['p1_prob']:.1%} | {m['player2']}: {pred['p2_prob']:.1%}<br>
                🎲 O/U 21.5: {pred['ou']} ({pred['ou_conf']:.1%})
            </div>
            """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()

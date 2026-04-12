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
# SIMPLE AND CLEAR APPROACH
# ==============================================================================
def train_models(df, player_stats, h2h):
    """
    Train with a simple, explicit approach:
    - Features represent strength differences
    - Positive features = stronger player
    - Label: 1 if stronger player listed first, 0 if listed second
    """
    X_data, y_winner, y_games = [], [], []
    winner_col = 'winner' if 'winner' in df.columns else 'winner_name'
    loser_col = 'loser' if 'loser' in df.columns else 'loser_name'
    
    for _, row in df.iterrows():
        winner = row.get(winner_col)
        loser = row.get(loser_col)
        surf = row.get('surface', 'Hard')
        
        if winner not in player_stats or loser not in player_stats or pd.isna(winner) or pd.isna(loser):
            continue
        
        w_stats = player_stats[winner]
        l_stats = player_stats[loser]
        total_games = row.get('total_games', 22)
        
        # H2H
        h2h_w = h2h.get((winner, loser), 0)
        h2h_l = h2h.get((loser, winner), 0)
        h2h_total = h2h_w + h2h_l + 1
        
        # Create TWO training examples from each match
        # Example 1: Winner listed first → label = 1
        features_w_first = [
            w_stats['elo'] - l_stats['elo'],
            w_stats['welo'] - l_stats['welo'],
            w_stats['surface_elo'].get(surf, 1500) - l_stats['surface_elo'].get(surf, 1500),
            w_stats['win_rate'] - l_stats['win_rate'],
            w_stats['recent_form'] - l_stats['recent_form'],
            w_stats['surface_win_rate'].get(surf, w_stats['win_rate']) - l_stats['surface_win_rate'].get(surf, l_stats['win_rate']),
            l_stats['avg_rank'] - w_stats['avg_rank'],  # Lower is better
            h2h_w / h2h_total,
        ]
        X_data.append(features_w_first)
        y_winner.append(1)  # Winner is first
        y_games.append(total_games)
        
        # Example 2: Loser listed first → label = 0
        features_l_first = [
            l_stats['elo'] - w_stats['elo'],
            l_stats['welo'] - w_stats['welo'],
            l_stats['surface_elo'].get(surf, 1500) - w_stats['surface_elo'].get(surf, 1500),
            l_stats['win_rate'] - w_stats['win_rate'],
            l_stats['recent_form'] - w_stats['recent_form'],
            l_stats['surface_win_rate'].get(surf, l_stats['win_rate']) - w_stats['surface_win_rate'].get(surf, w_stats['win_rate']),
            w_stats['avg_rank'] - l_stats['avg_rank'],
            h2h_l / h2h_total,
        ]
        X_data.append(features_l_first)
        y_winner.append(0)  # Winner is second
        y_games.append(total_games)
    
    X = np.array(X_data, dtype=np.float64)
    X = np.nan_to_num(X, nan=0.0, posinf=100, neginf=-100)
    
    y_w = np.array(y_winner)
    y_g = np.array(y_games)
    
    # Diagnostics
    st.write(f"📊 Training samples: {len(X)}")
    unique, counts = np.unique(y_w, return_counts=True)
    st.write(f"📊 Class distribution: {dict(zip(unique, counts))}")
    
    model_winner = GradientBoostingClassifier(
        n_estimators=150, 
        max_depth=4, 
        learning_rate=0.1, 
        random_state=42
    )
    model_winner.fit(X, y_w)
    
    # Training diagnostics
    train_pred = model_winner.predict(X)
    train_acc = (train_pred == y_w).mean()
    train_probs = model_winner.predict_proba(X)[:, 1]
    
    st.write(f"✅ Training accuracy: {train_acc:.1%}")
    st.write(f"📊 Prob stats - Mean: {train_probs.mean():.1%}, Std: {train_probs.std():.3f}, Min: {train_probs.min():.1%}, Max: {train_probs.max():.1%}")
    
    model_ou = GradientBoostingClassifier(
        n_estimators=100, 
        max_depth=3, 
        learning_rate=0.1,
        random_state=42
    )
    model_ou.fit(X, (y_g > 21.5).astype(int))
    
    return model_winner, model_ou

def predict_match(model_winner, model_ou, player_stats, h2h, p1_name, p2_name, surface):
    """
    Predict match outcome.
    Model predicts: prob that p1 (first player) wins
    """
    default_stats = {
        'win_rate': 0.5, 'avg_rank': 150, 'recent_form': 0.5, 'elo': 1500, 
        'welo': 1500, 'surface_elo': {'Hard': 1500, 'Clay': 1500, 'Grass': 1500},
        'surface_win_rate': {'Hard': 0.5, 'Clay': 0.5, 'Grass': 0.5}, 'avg_games': 22.0
    }
    
    p1 = player_stats.get(p1_name, default_stats)
    p2 = player_stats.get(p2_name, default_stats)
    
    # H2H
    h2h_p1 = h2h.get((p1_name, p2_name), 0)
    h2h_p2 = h2h.get((p2_name, p1_name), 0)
    h2h_total = h2h_p1 + h2h_p2 + 1
    
    # Features: p1 vs p2 (p1 listed first)
    features = [
        p1['elo'] - p2['elo'],
        p1['welo'] - p2['welo'],
        p1['surface_elo'].get(surface, 1500) - p2['surface_elo'].get(surface, 1500),
        p1['win_rate'] - p2['win_rate'],
        p1['recent_form'] - p2['recent_form'],
        p1['surface_win_rate'].get(surface, p1['win_rate']) - p2['surface_win_rate'].get(surface, p2['win_rate']),
        p2['avg_rank'] - p1['avg_rank'],
        h2h_p1 / h2h_total,
    ]
    
    X = np.array([features], dtype=np.float64)
    X = np.nan_to_num(X, nan=0.0, posinf=100, neginf=-100)
    
    # Get probabilities
    probs = model_winner.predict_proba(X)[0]
    # probs[0] = probability of class 0 (p2 wins)
    # probs[1] = probability of class 1 (p1 wins)
    
    p1_win_prob = probs[1]
    p2_win_prob = probs[0]
    
    # Over/Under
    ou_probs = model_ou.predict_proba(X)[0]
    over_prob = ou_probs[1]
    under_prob = ou_probs[0]
    
    return {
        'winner': p1_name if p1_win_prob > p2_win_prob else p2_name,
        'winner_conf': max(p1_win_prob, p2_win_prob),
        'p1_prob': p1_win_prob,
        'p2_prob': p2_win_prob,
        'ou': "Over 21.5" if over_prob > under_prob else "Under 21.5",
        'ou_conf': max(over_prob, under_prob),
        'exp_games': (p1.get('avg_games', 22) + p2.get('avg_games', 22)) / 2
    }

# Scraper
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
    st.markdown("**Clean Implementation - V4**")

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
            model_w, model_ou = train_models(df, player_stats, h2h)
            
            st.session_state.player_stats = player_stats
            st.session_state.h2h = h2h
            st.session_state.model_winner = model_w
            st.session_state.model_ou = model_ou
            st.session_state.models_trained = True
            st.success("✅ Model trained successfully!")

    if st.session_state.get('matches') and st.session_state.get('models_trained'):
        st.header("🎯 Predictions")
        for m in st.session_state.matches:
            pred = predict_match(
                st.session_state.model_winner, 
                st.session_state.model_ou, 
                st.session_state.player_stats, 
                st.session_state.h2h, 
                m['player1'], 
                m['player2'], 
                m['surface']
            )
            
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

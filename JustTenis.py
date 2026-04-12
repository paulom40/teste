import os
import re
import time
import warnings
import pandas as pd
import numpy as np
import random
import io
from datetime import datetime, timedelta
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
    .confidence-high {color: #00ff88; font-weight: bold;}
    .confidence-medium {color: #ffd700; font-weight: bold;}
    .confidence-low {color: #ff6b6b; font-weight: bold;}
    </style>
""", unsafe_allow_html=True)

# ==============================================================================
# Simple but Robust Version
# ==============================================================================
@st.cache_data(ttl=3600)
def load_historical_data(file_path):
    df = pd.read_excel(file_path)
    df.columns = [str(c).strip().lower().replace(' ', '_') for c in df.columns]
    rename = {'winner_name':'winner', 'loser_name':'loser', 'tourney_date':'date', 'winner_rank':'wrank', 'loser_rank':'lrank'}
    df.rename(columns={k:v for k,v in rename.items() if k in df.columns}, inplace=True)
    
    if 'date' in df.columns:
        df['date'] = pd.to_datetime(df['date'].astype(str), errors='coerce')
    if 'score' in df.columns:
        df['total_games'] = df['score'].apply(lambda x: 22 if pd.isna(x) else sum(int(n) for n in re.findall(r'\d+', str(x)) if n.isdigit()) or 22)
    return df

def compute_simple_stats(df):
    stats = {}
    for player in set(df['winner'].dropna().unique()) | set(df['loser'].dropna().unique()):
        matches = df[(df['winner'] == player) | (df['loser'] == player)]
        if len(matches) == 0: continue
            
        wins = len(df[df['winner'] == player])
        total = wins + len(df[df['loser'] == player])
        win_rate = wins / total if total > 0 else 0.5
        
        recent = matches.sort_values('date', ascending=False).head(10)
        recent_form = len(recent[recent['winner'] == player]) / len(recent) if len(recent) > 0 else win_rate
        
        avg_rank = matches['wrank'].mean() if 'wrank' in matches.columns else 150
        
        stats[player] = {
            'win_rate': win_rate,
            'avg_rank': float(avg_rank),
            'recent_form': recent_form,
            'avg_games': float(matches['total_games'].mean()) if 'total_games' in matches.columns else 22.0
        }
    return stats

def train_models(df, player_stats):
    X_data, y_winner, y_games = [], [], []
    
    for _, row in df.iterrows():
        w = row.get('winner')
        l = row.get('loser')
        surf = row.get('surface', 'Hard')
        
        if w not in player_stats or l not in player_stats: continue
            
        # Random order
        if random.random() < 0.5:
            p1, p2, label = w, l, 1
        else:
            p1, p2, label = l, w, 0
            
        s1 = player_stats[p1]
        s2 = player_stats[p2]
        
        features = [
            s1['win_rate'] - s2['win_rate'],
            (s1['avg_rank'] - s2['avg_rank']) / 50,
            s1['recent_form'] - s2['recent_form'],
            s1['avg_games'] - s2['avg_games']
        ]
        
        X_data.append(features)
        y_winner.append(label)
        y_games.append(row.get('total_games', 22))
    
    X = np.nan_to_num(np.array(X_data, dtype=np.float64), nan=0.0)
    y_w = np.array(y_winner)
    y_g = np.array(y_games)
    
    model_w = GradientBoostingClassifier(n_estimators=200, max_depth=3, learning_rate=0.1, random_state=42)
    model_w.fit(X, y_w)
    
    model_ou = GradientBoostingClassifier(n_estimators=150, max_depth=3, random_state=42)
    model_ou.fit(X, (y_g > 21.5).astype(int))
    
    return model_w, model_ou

def predict_match(model_w, model_ou, player_stats, p1, p2, surface):
    s1 = player_stats.get(p1, {'win_rate':0.5, 'avg_rank':150, 'recent_form':0.5, 'avg_games':22})
    s2 = player_stats.get(p2, {'win_rate':0.5, 'avg_rank':150, 'recent_form':0.5, 'avg_games':22})
    
    features = np.nan_to_num(np.array([[ 
        s1['win_rate'] - s2['win_rate'],
        (s1['avg_rank'] - s2['avg_rank']) / 50,
        s1['recent_form'] - s2['recent_form'],
        s1['avg_games'] - s2['avg_games']
    ]]), nan=0.0)
    
    p1_prob = model_w.predict_proba(features)[0][1]
    ou_prob = model_ou.predict_proba(features)[0][1]
    
    return {
        'winner': p1 if p1_prob > 0.5 else p2,
        'winner_conf': max(p1_prob, 1 - p1_prob),
        'ou': "Over 21.5" if ou_prob > 0.5 else "Under 21.5",
        'ou_conf': max(ou_prob, 1 - ou_prob)
    }

# Improved Scraper with better surface detection
@st.cache_data(ttl=1800)
def scrape_matches_flashscore(days_ahead=0):
    matches = []
    try:
        st.info(f"Scraping matches for day +{days_ahead}...")
        chrome_options = Options()
        chrome_options.add_argument("--headless")
        chrome_options.add_argument("--no-sandbox")
        chrome_options.add_argument("--disable-dev-shm-usage")
        
        driver = webdriver.Chrome(options=chrome_options)
        driver.get("https://www.flashscore.com/tennis/")
        time.sleep(10)
        
        for _ in range(4):
            driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
            time.sleep(3)
        
        soup = BeautifulSoup(driver.page_source, 'html.parser')
        driver.quit()
        
        cards = soup.find_all('div', class_=lambda x: x and 'event__match' in str(x))
        
        for card in cards:
            try:
                tour_elem = card.find_previous('div', class_=lambda x: x and any(k in str(x).lower() for k in ['tournament','header','round']))
                tournament = tour_elem.get_text(strip=True) if tour_elem else ""
                
                if any(w in tournament.upper() for w in ['WTA', 'WOMEN', 'BILLIE']): continue
                
                players = card.find_all('div', class_=lambda x: x and 'participant' in str(x))
                if len(players) < 2: continue
                
                p1 = players[0].get_text(strip=True)
                p2 = players[1].get_text(strip=True)
                
                tour_u = tournament.upper()
                if any(x in tour_u for x in ['CLAY', 'ROLAND', 'MADRID', 'ROME', 'BARCELONA', 'BUENOS', 'ESTORIL']):
                    surface = "Clay"
                elif any(x in tour_u for x in ['WIMBLEDON', 'HALLE', 'QUEEN', 'EASTBOURNE']):
                    surface = "Grass"
                else:
                    surface = "Hard"
                
                match_type = "Challenger" if "CHALLENGER" in tour_u else "ATP"
                
                matches.append({
                    'tournament': tournament,
                    'player1': p1,
                    'player2': p2,
                    'surface': surface,
                    'type': match_type
                })
            except:
                continue
                
        st.success(f"✅ Scraped {len(matches)} matches")
        return matches
    except Exception as e:
        st.error(f"Scraping error: {e}")
        return []

# ==============================================================================
# Main
# ==============================================================================
def main():
    st.title("🎾 ATP & Challenger Predictor")
    st.markdown("**Simplified Stable Model**")

    with st.sidebar:
        uploaded_file = st.file_uploader("Upload Historical Data (Excel)", type=['xlsx'])
        st.markdown("---")
        col1, col2 = st.columns(2)
        with col1:
            if st.button("📅 Scrape Today"):
                st.session_state.matches = scrape_matches_flashscore(0)
        with col2:
            if st.button("📅 Scrape Tomorrow"):
                st.session_state.matches = scrape_matches_flashscore(1)

    if 'matches' not in st.session_state:
        st.session_state.matches = []

    if uploaded_file:
        with st.spinner("Training model..."):
            temp_path = "/tmp/tennis_data.xlsx"
            with open(temp_path, 'wb') as f:
                f.write(uploaded_file.read())
            
            df = load_historical_data(temp_path)
            player_stats = compute_simple_stats(df)
            model_w, model_ou = train_models(df, player_stats)
            
            st.session_state.player_stats = player_stats
            st.session_state.model_winner = model_w
            st.session_state.model_ou = model_ou
            st.session_state.models_trained = True
            st.success("✅ Model trained!")

    if st.session_state.get('matches') and st.session_state.get('models_trained'):
        st.header("🎯 Predictions")
        for m in st.session_state.matches:
            pred = predict_match(st.session_state.model_winner, st.session_state.model_ou, 
                               st.session_state.player_stats, m['player1'], m['player2'], m['surface'])
            
            conf_class = "confidence-high" if pred['winner_conf'] >= 0.65 else "confidence-medium" if pred['winner_conf'] >= 0.55 else "confidence-low"
            
            st.markdown(f"""
            <div class="prediction-card">
                <b>{m['type']} • {m['surface']}</b><br>
                <h3>{m['player1']} vs {m['player2']}</h3>
                🏆 Winner: <span class="{conf_class}">{pred['winner']} ({pred['winner_conf']:.1%})</span><br>
                🎲 O/U 21.5: {pred['ou']} ({pred['ou_conf']:.1%})
            </div>
            """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()

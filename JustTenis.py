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

# ==============================================================================
# Streamlit Configuration
# ==============================================================================
st.set_page_config(
    page_title="🎾 ATP & Challenger Tennis Predictor",
    page_icon="🎾",
    layout="wide",
    initial_sidebar_state="expanded"
)

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
    .confidence-high {color: #00ff88; font-weight: bold; font-size: 1.1em;}
    .confidence-medium {color: #ffd700; font-weight: bold; font-size: 1.1em;}
    .confidence-low {color: #ff6b6b; font-weight: bold; font-size: 1.1em;}
    .match-header {font-size: 1.35em; font-weight: bold; margin-bottom: 10px;}
    </style>
""", unsafe_allow_html=True)

# ==============================================================================
# ELO & WELO Calculation
# ==============================================================================
def calculate_elo_ratings(df, k=32, surface_k=25):
    players = set(df['winner'].unique()) | set(df['loser'].unique())
    elo = {p: 1500.0 for p in players}
    welo = {p: 1500.0 for p in players}
    surface_elo = {p: {'Hard':1500.0, 'Clay':1500.0, 'Grass':1500.0, 'Carpet':1500.0} for p in players}
    
    df_sorted = df.sort_values('date').copy()
    
    for _, row in df_sorted.iterrows():
        p1 = row['winner']
        p2 = row['loser']
        surf = row.get('surface', 'Hard')
        
        if p1 not in elo or p2 not in elo:
            continue
            
        # Main ELO
        r1, r2 = elo[p1], elo[p2]
        exp1 = 1 / (1 + 10 ** ((r2 - r1) / 400))
        elo[p1] += k * (1 - exp1)
        elo[p2] += k * (0 - (1 - exp1))
        
        # Surface ELO
        if surf in surface_elo[p1]:
            s1, s2 = surface_elo[p1][surf], surface_elo[p2][surf]
            exp_s1 = 1 / (1 + 10 ** ((s2 - s1) / 400))
            surface_elo[p1][surf] += surface_k * (1 - exp_s1)
            surface_elo[p2][surf] += surface_k * (0 - (1 - exp_s1))
        
        # Weighted ELO (recent matches matter more)
        welo[p1] = welo[p1] * 0.97 + elo[p1] * 0.03
        welo[p2] = welo[p2] * 0.97 + elo[p2] * 0.03
    
    return elo, welo, surface_elo

# ==============================================================================
# Helper Functions
# ==============================================================================
def parse_score_to_games(score):
    if pd.isna(score) or score == "":
        return 22
    score = re.sub(r'\(\d+\)', '', str(score))
    score = re.sub(r'RET|DEF|W/O', '', score, flags=re.IGNORECASE)
    total = 0
    for s in score.split():
        if '-' in s:
            try:
                a, b = map(int, re.findall(r'\d+', s))
                total += a + b
            except:
                pass
    return total if total > 0 else 22

@st.cache_data(ttl=3600)
def load_historical_data(file_path):
    df = pd.read_excel(file_path)
    df.columns = [str(c).strip().lower().replace(' ', '_') for c in df.columns]
    
    if 'date' in df.columns:
        df['date'] = pd.to_datetime(df['date'].astype(str), format='%Y%m%d', errors='coerce')
    if 'score' in df.columns:
        df['total_games'] = df['score'].apply(parse_score_to_games)
    return df

def compute_player_stats(df):
    stats = {}
    elo, welo, surface_elo = calculate_elo_ratings(df)
    
    for player in set(df['winner'].unique()) | set(df['loser'].unique()):
        matches = df[(df['winner'] == player) | (df['loser'] == player)]
        if len(matches) == 0:
            continue
            
        wins = len(df[df['winner'] == player])
        total = wins + len(df[df['loser'] == player])
        win_rate = wins / total if total > 0 else 0.5
        
        recent = matches.sort_values('date', ascending=False).head(10)
        recent_form = len(recent[recent['winner'] == player]) / len(recent) if len(recent) > 0 else win_rate
        
        avg_rank = matches['wrank'].mean() if 'wrank' in matches.columns else 150
        avg_games = matches['total_games'].mean() if 'total_games' in matches.columns else 22
        
        surfaces = {'Hard': win_rate, 'Clay': win_rate, 'Grass': win_rate, 'Carpet': win_rate}
        if 'surface' in df.columns:
            for surf in surfaces:
                surf_m = matches[matches['surface'] == surf]
                if len(surf_m) > 0:
                    surfaces[surf] = len(surf_m[surf_m['winner'] == player]) / len(surf_m)
        
        stats[player] = {
            'win_rate': win_rate,
            'avg_rank': avg_rank,
            'recent_form': recent_form,
            'avg_games': avg_games,
            'hard': surfaces['Hard'],
            'clay': surfaces['Clay'],
            'grass': surfaces['Grass'],
            'elo': elo.get(player, 1500),
            'welo': welo.get(player, 1500),
            'surface_elo': surface_elo.get(player, {'Hard':1500,'Clay':1500,'Grass':1500,'Carpet':1500})
        }
    return stats

def build_h2h_dict(df):
    h2h = defaultdict(int)
    for _, row in df.iterrows():
        h2h[(row['winner'], row['loser'])] += 1
    return h2h

# ==============================================================================
# Model Training
# ==============================================================================
def train_models(df, player_stats, h2h):
    X_data, y_winner, y_games = [], [], []
    
    for _, row in df.iterrows():
        w, l = row['winner'], row['loser']
        surf = row.get('surface', 'Hard')
        if w not in player_stats or l not in player_stats:
            continue
            
        p1, p2, label = (w, l, 1) if random.random() < 0.5 else (l, w, 0)
        s1, s2 = player_stats[p1], player_stats[p2]
        
        h2h_rate = h2h.get((p1, p2), 0) / (h2h.get((p1, p2), 0) + h2h.get((p2, p1), 0) + 1)
        
        elo_diff = s1['elo'] - s2['elo']
        welo_diff = s1['welo'] - s2['welo']
        surf_elo_diff = s1['surface_elo'].get(surf, 1500) - s2['surface_elo'].get(surf, 1500)
        
        features = [
            s1['win_rate'] - s2['win_rate'],
            (s1['avg_rank'] - s2['avg_rank']) / 50,
            s1['recent_form'] - s2['recent_form'],
            s1.get(surf.lower(), s1['win_rate']) - s2.get(surf.lower(), s2['win_rate']),
            elo_diff / 80,
            welo_diff / 80,
            surf_elo_diff / 80,
            h2h_rate,
            s1['elo'] / (s2['elo'] + 100),
            abs(elo_diff) / 100
        ]
        
        X_data.append(features)
        y_winner.append(label)
        y_games.append(row.get('total_games', 22))
    
    X = np.array(X_data)
    
    model_winner = GradientBoostingClassifier(
        n_estimators=600, max_depth=7, learning_rate=0.04, 
        subsample=0.82, random_state=42
    )
    model_winner.fit(X, y_winner)
    
    model_ou = GradientBoostingClassifier(n_estimators=400, max_depth=6, random_state=42)
    model_ou.fit(X, (np.array(y_games) > 21.5).astype(int))
    
    return model_winner, model_ou

def predict_match(model_winner, model_ou, player_stats, h2h, p1, p2, surface):
    default = {'win_rate':0.5, 'avg_rank':150, 'recent_form':0.5, 'avg_games':22,
               'elo':1500, 'welo':1500, 'surface_elo':{'Hard':1500,'Clay':1500,'Grass':1500}}
    
    s1 = player_stats.get(p1, default)
    s2 = player_stats.get(p2, default)
    
    h2h_rate = h2h.get((p1,p2), 0) / (h2h.get((p1,p2),0) + h2h.get((p2,p1),0) + 1)
    
    features = np.array([[ 
        s1['win_rate'] - s2['win_rate'],
        (s1['avg_rank'] - s2['avg_rank']) / 50,
        s1['recent_form'] - s2['recent_form'],
        s1.get(surface.lower(), s1['win_rate']) - s2.get(surface.lower(), s2['win_rate']),
        (s1['elo'] - s2['elo']) / 80,
        (s1['welo'] - s2['welo']) / 80,
        (s1['surface_elo'].get(surface,1500) - s2['surface_elo'].get(surface,1500)) / 80,
        h2h_rate,
        s1['elo'] / (s2['elo'] + 100),
        abs(s1['elo'] - s2['elo']) / 100
    ]])
    
    p1_prob = model_winner.predict_proba(features)[0][1]
    ou_prob = model_ou.predict_proba(features)[0][1]
    
    return {
        'winner': p1 if p1_prob > 0.5 else p2,
        'winner_conf': max(p1_prob, 1 - p1_prob),
        'p1_prob': p1_prob,
        'p2_prob': 1 - p1_prob,
        'ou': "Over 21.5" if ou_prob > 0.5 else "Under 21.5",
        'ou_conf': max(ou_prob, 1 - ou_prob),
        'exp_games': (s1['avg_games'] + s2['avg_games']) / 2
    }

# ==============================================================================
# Flashscore Scraper (ATP & Challenger Only)
# ==============================================================================
@st.cache_data(ttl=1800)
def scrape_matches_flashscore(days_ahead=0):
    matches = []
    try:
        target_date = datetime.now() + timedelta(days=days_ahead)
        st.info(f"🤖 Scraping ATP & Challenger matches for {target_date.strftime('%d.%m.%Y')}...")
        
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
                tour_elem = card.find_previous('div', class_=lambda x: x and any(k in str(x).lower() for k in ['tournament','header']))
                tournament = tour_elem.get_text(strip=True) if tour_elem else ""
                
                # Skip WTA and women events
                if any(w in tournament.upper() for w in ['WTA', 'WOMEN', 'BILLIE JEAN']):
                    continue
                
                players = card.find_all('div', class_=lambda x: x and 'participant' in str(x))
                if len(players) < 2:
                    continue
                    
                p1 = players[0].get_text(strip=True)
                p2 = players[1].get_text(strip=True)
                
                # Surface detection
                tour_u = tournament.upper()
                surface = "Clay" if any(x in tour_u for x in ['CLAY','ROLAND','MADRID','ROME','BARCELONA','BUENOS','ESTORIL']) else \
                          "Grass" if any(x in tour_u for x in ['WIMBLEDON','HALLE','QUEEN','EASTBOURNE']) else "Hard"
                
                match_type = "Challenger" if "CHALLENGER" in tour_u or "CH " in tour_u else "ATP"
                
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
        st.error(f"Scraping failed: {e}")
        return []

# ==============================================================================
# Main Application
# ==============================================================================
def main():
    st.title("🎾 ATP & Challenger Tennis Predictor")
    st.markdown("**Powered by ELO + WELO Ratings**")

    with st.sidebar:
        st.header("Settings")
        uploaded_file = st.file_uploader("Upload Historical Data (Excel)", type=['xlsx'])
        
        st.markdown("---")
        st.subheader("🌐 Live Scraping")
        col1, col2 = st.columns(2)
        with col1:
            if st.button("📅 Today"):
                st.session_state.matches = scrape_matches_flashscore(0)
        with col2:
            if st.button("📅 Tomorrow"):
                st.session_state.matches = scrape_matches_flashscore(1)

    if 'matches' not in st.session_state:
        st.session_state.matches = []
    if 'models_trained' not in st.session_state:
        st.session_state.models_trained = False

    # Train Model
    if uploaded_file:
        with st.spinner("Training model with ELO & WELO..."):
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
            st.success(f"✅ Model trained on {len(df)} matches with ELO + WELO!")

    # Predictions
    if st.session_state.matches and st.session_state.get('models_trained', False):
        st.markdown("---")
        st.header("🎯 Predictions")
        
        results_data = []
        for match in st.session_state.matches:
            pred = predict_match(
                st.session_state.model_winner,
                st.session_state.model_ou,
                st.session_state.player_stats,
                st.session_state.h2h,
                match['player1'], match['player2'], match['surface']
            )
            
            conf_class = "confidence-high" if pred['winner_conf'] >= 0.65 else \
                        "confidence-medium" if pred['winner_conf'] >= 0.55 else "confidence-low"
            
            st.markdown(f"""
            <div class="prediction-card">
                <div class="match-header">
                    {match['type']} • {match['surface']}
                </div>
                <h3>{match['player1']} vs {match['player2']}</h3>
                🏆 Winner: <span class="{conf_class}">{pred['winner']} ({pred['winner_conf']:.1%})</span><br>
                🎲 Over/Under 21.5: {pred['ou']} ({pred['ou_conf']:.1%})<br>
                Expected Games: {pred['exp_games']:.1f}
            </div>
            """, unsafe_allow_html=True)
            
            results_data.append({
                'Tournament': match['tournament'],
                'Match': f"{match['player1']} vs {match['player2']}",
                'Surface': match['surface'],
                'Predicted Winner': pred['winner'],
                'Win Probability': f"{pred['winner_conf']:.1%}",
                'Over/Under': pred['ou'],
                'Expected Games': round(pred['exp_games'], 1)
            })
        
        if results_data:
            df_results = pd.DataFrame(results_data)
            col1, col2 = st.columns(2)
            with col1:
                st.download_button("📥 Download CSV", df_results.to_csv(index=False), 
                                 f"predictions_{datetime.now().strftime('%Y%m%d')}.csv", "text/csv")
            with col2:
                buffer = io.BytesIO()
                with pd.ExcelWriter(buffer, engine='openpyxl') as writer:
                    df_results.to_excel(writer, index=False)
                st.download_button("📊 Download Excel", buffer.getvalue(), 
                                 f"predictions_{datetime.now().strftime('%Y%m%d_%H%M')}.xlsx",
                                 "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")

    else:
        st.info("Upload historical data and scrape matches to get predictions.")

if __name__ == "__main__":
    main()

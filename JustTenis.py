import os
import re
import time
import warnings
import pandas as pd
import numpy as np
import random
from datetime import datetime, timedelta
import streamlit as st
from sklearn.ensemble import GradientBoostingClassifier
from bs4 import BeautifulSoup
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC

warnings.filterwarnings('ignore')

# ==============================================================================
# Streamlit Page Configuration
# ==============================================================================
st.set_page_config(
    page_title="🎾 ATP & Challenger Tennis Predictions",
    page_icon="🎾",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
    <style>
    .main {background-color: #0e1117;}
    .stApp {background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);}
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
    .match-header {font-size: 1.3em; font-weight: bold; margin-bottom: 10px; color: #ffffff;}
    .tournament-badge {
        background: rgba(255,255,255,0.2);
        padding: 5px 10px;
        border-radius: 5px;
        font-size: 0.9em;
        display: inline-block;
        margin-right: 10px;
    }
    </style>
""", unsafe_allow_html=True)

# ==============================================================================
# Helper Functions
# ==============================================================================
def parse_score_to_games(score):
    if pd.isna(score) or score == "":
        return 20
    score = re.sub(r'\(\d+\)', '', str(score))
    score = re.sub(r'RET|DEF|W/O', '', score, flags=re.IGNORECASE)
    total = 0
    for set_score in score.split():
        if '-' in set_score:
            try:
                parts = set_score.split('-')
                a = int(re.findall(r'\d+', parts[0])[0])
                b = int(re.findall(r'\d+', parts[1])[0])
                total += a + b
            except:
                continue
    return total if total > 0 else 20

@st.cache_data(ttl=3600)
def load_historical_data(file_path):
    try:
        df = pd.read_excel(file_path)
        df.columns = [str(c).strip().lower().replace(' ', '_') for c in df.columns]
        
        rename_map = {'tourney_date': 'date', 'winner_name': 'winner', 'loser_name': 'loser',
                      'winner_rank': 'wrank', 'loser_rank': 'lrank', 't_games': 'total_games'}
        df.rename(columns={k: v for k, v in rename_map.items() if k in df.columns}, inplace=True)
        
        if 'date' in df.columns:
            df['date'] = pd.to_datetime(df['date'].astype(str), format='%Y%m%d', errors='coerce')
        
        if 'total_games' not in df.columns and 'score' in df.columns:
            df['total_games'] = df['score'].apply(parse_score_to_games)
        
        return df
    except Exception as e:
        st.error(f"Error loading data: {e}")
        return None

def compute_player_stats(df):
    stats = {}
    for player in set(df['winner'].unique()) | set(df['loser'].unique()):
        player_matches = df[(df['winner'] == player) | (df['loser'] == player)]
        if len(player_matches) == 0:
            continue
            
        wins = len(df[df['winner'] == player])
        losses = len(df[df['loser'] == player])
        total = wins + losses
        if total == 0: continue
            
        win_rate = wins / total
        
        # Surface stats
        surfaces = {}
        for surf in ['Hard', 'Clay', 'Grass', 'Carpet']:
            surf_matches = player_matches[player_matches.get('surface', 'Hard') == surf]
            surf_wins = len(surf_matches[surf_matches['winner'] == player])
            surf_total = len(surf_matches)
            surfaces[surf] = surf_wins / surf_total if surf_total > 0 else win_rate
        
        # Avg rank
        ranks = []
        ranks.extend(df[df['winner'] == player]['wrank'].dropna().tolist())
        ranks.extend(df[df['loser'] == player]['lrank'].dropna().tolist())
        avg_rank = np.mean(ranks) if ranks else 200
        
        # Recent form
        recent = player_matches.sort_values('date', ascending=False).head(10)
        recent_form = len(recent[recent['winner'] == player]) / len(recent) if len(recent) > 0 else win_rate
        
        avg_games = player_matches['total_games'].mean() if 'total_games' in player_matches.columns else 22
        
        stats[player] = {
            'win_rate': win_rate, 'wins': wins, 'losses': losses, 'avg_rank': avg_rank,
            'recent_form': recent_form, 'avg_games': avg_games,
            'hard': surfaces.get('Hard', win_rate), 'clay': surfaces.get('Clay', win_rate),
            'grass': surfaces.get('Grass', win_rate), 'carpet': surfaces.get('Carpet', win_rate)
        }
    return stats

def train_models(df, player_stats):
    X_data = []
    y_winner = []
    y_games = []
    
    for _, row in df.iterrows():
        winner_name = row['winner']
        loser_name = row['loser']
        surf = row.get('surface', 'Hard')
        
        if winner_name not in player_stats or loser_name not in player_stats:
            continue
        
        # Random order to avoid bias
        if random.random() < 0.5:
            p1, p2, label = winner_name, loser_name, 1
        else:
            p1, p2, label = loser_name, winner_name, 0
        
        p1_stats = player_stats[p1]
        p2_stats = player_stats[p2]
        surf_key = surf.lower()
        
        features = [
            p1_stats['win_rate'], p2_stats['win_rate'],
            p1_stats['avg_rank'], p2_stats['avg_rank'],
            p1_stats['recent_form'], p2_stats['recent_form'],
            p1_stats.get(surf_key, p1_stats['win_rate']),
            p2_stats.get(surf_key, p2_stats['win_rate']),
            p1_stats['avg_games'], p2_stats['avg_games']
        ]
        
        X_data.append(features)
        y_winner.append(label)
        y_games.append(row.get('total_games', 22))
    
    X = np.array(X_data)
    y_w = np.array(y_winner)
    
    if len(np.unique(y_w)) < 2:
        st.error("Not enough variety in training data.")
        st.stop()
    
    model_winner = GradientBoostingClassifier(n_estimators=200, max_depth=5, random_state=42, subsample=0.9)
    model_winner.fit(X, y_w)
    
    model_ou = GradientBoostingClassifier(n_estimators=150, random_state=42)
    y_ou = (np.array(y_games) > 21.5).astype(int)
    model_ou.fit(X, y_ou)
    
    return model_winner, model_ou

def predict_match(model_winner, model_ou, player_stats, p1, p2, surface):
    default = {'win_rate': 0.5, 'avg_rank': 200, 'recent_form': 0.5, 'avg_games': 22,
               'hard':0.5, 'clay':0.5, 'grass':0.5, 'carpet':0.5}
    
    p1_stats = player_stats.get(p1, default)
    p2_stats = player_stats.get(p2, default)
    surf_key = surface.lower()
    
    features = np.array([[ 
        p1_stats['win_rate'], p2_stats['win_rate'],
        p1_stats['avg_rank'], p2_stats['avg_rank'],
        p1_stats['recent_form'], p2_stats['recent_form'],
        p1_stats.get(surf_key, p1_stats['win_rate']),
        p2_stats.get(surf_key, p2_stats['win_rate']),
        p1_stats['avg_games'], p2_stats['avg_games']
    ]])
    
    p1_prob = model_winner.predict_proba(features)[0][1]
    ou_prob = model_ou.predict_proba(features)[0][1]
    
    return {
        'winner': p1 if p1_prob > 0.5 else p2,
        'winner_conf': max(p1_prob, 1-p1_prob),
        'p1_prob': p1_prob,
        'p2_prob': 1-p1_prob,
        'ou': "Over 21.5" if ou_prob > 0.5 else "Under 21.5",
        'ou_conf': max(ou_prob, 1-ou_prob),
        'exp_games': (p1_stats['avg_games'] + p2_stats['avg_games']) / 2,
        'p1_known': p1 in player_stats,
        'p2_known': p2 in player_stats
    }

# ====================== WEB SCRAPER ======================
@st.cache_data(ttl=1800)
def scrape_matches_flashscore(days_ahead=0):
    matches = []
    try:
        target_date = datetime.now() + timedelta(days=days_ahead)
        date_str = target_date.strftime('%d.%m.%Y')
        
        st.info(f"🤖 Scraping Flashscore for {date_str}... Please wait (15-30 seconds)")
        
        chrome_options = Options()
        chrome_options.add_argument("--headless")
        chrome_options.add_argument("--no-sandbox")
        chrome_options.add_argument("--disable-dev-shm-usage")
        chrome_options.add_argument("--disable-blink-features=AutomationControlled")
        
        driver = webdriver.Chrome(options=chrome_options)
        driver.get("https://www.flashscore.com/tennis/")
        time.sleep(10)
        
        driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
        time.sleep(5)
        
        soup = BeautifulSoup(driver.page_source, 'html.parser')
        driver.quit()
        
        # Extract matches
        match_cards = soup.find_all('div', {'class': lambda x: x and 'event__match' in str(x)})
        
        current_tournament = ""
        for card in match_cards:
            try:
                tour = card.find_previous('div', class_=lambda x: x and ('tournament' in str(x).lower() or 'header' in str(x).lower()))
                if tour:
                    current_tournament = tour.get_text(strip=True)
                
                players = card.find_all('div', class_=lambda x: x and 'participant' in str(x))
                if len(players) >= 2:
                    p1 = players[0].get_text(strip=True)
                    p2 = players[1].get_text(strip=True)
                    
                    if len(p1) < 3 or len(p2) < 3:
                        continue
                        
                    surface = 'Clay' if any(x in current_tournament.upper() for x in ['CLAY','ROLAND','MADRID','ROME']) else \
                              'Grass' if 'WIMBLEDON' in current_tournament.upper() else 'Hard'
                    
                    match_type = "Challenger" if "CHALLENGER" in current_tournament.upper() or "CH" in current_tournament.upper() else "ATP"
                    
                    matches.append({
                        'tournament': current_tournament,
                        'player1': p1,
                        'player2': p2,
                        'surface': surface,
                        'type': match_type
                    })
            except:
                continue
                
        st.success(f"✅ Successfully scraped {len(matches)} matches for {date_str}")
        return matches
        
    except Exception as e:
        st.error(f"Scraping failed: {e}")
        st.info("Try again or use manual input.")
        return []

def get_confidence_class(conf):
    if conf >= 0.70: return "confidence-high"
    elif conf >= 0.55: return "confidence-medium"
    else: return "confidence-low"

# ==============================================================================
# Main Application
# ==============================================================================
def main():
    st.title("🎾 ATP & Challenger Tennis Match Predictor")
    st.markdown("### Predict winners + Over/Under with Machine Learning")

    # Sidebar
    with st.sidebar:
        st.header("⚙️ Settings")
        uploaded_file = st.file_uploader("Upload Historical Data (Excel)", type=['xlsx'])
        
        st.markdown("---")
        st.subheader("🌐 Live Scraping")
        col1, col2 = st.columns(2)
        with col1:
            if st.button("📅 Scrape Today's Matches"):
                matches = scrape_matches_flashscore(days_ahead=0)
                if matches:
                    st.session_state.matches = matches
        with col2:
            if st.button("📅 Scrape Tomorrow's Matches"):
                matches = scrape_matches_flashscore(days_ahead=1)
                if matches:
                    st.session_state.matches = matches
        
        st.markdown("---")
        st.info("Using Gradient Boosting Models")

    # Session State
    if 'models_trained' not in st.session_state:
        st.session_state.models_trained = False
    if 'matches' not in st.session_state:
        st.session_state.matches = []

    # Load & Train Model
    if uploaded_file is not None:
        with st.spinner("Training models..."):
            temp_path = "/tmp/tennis_data.xlsx"
            with open(temp_path, 'wb') as f:
                f.write(uploaded_file.read())
            
            df = load_historical_data(temp_path)
            if df is not None:
                player_stats = compute_player_stats(df)
                st.session_state.player_stats = player_stats
                model_w, model_ou = train_models(df, player_stats)
                st.session_state.model_winner = model_w
                st.session_state.model_ou = model_ou
                st.session_state.models_trained = True
                st.success(f"✅ Models trained on {len(df)} matches!")
    else:
        st.warning("Please upload your historical tennis data to train the model.")
        return

    if not st.session_state.models_trained:
        return

    # Input Tabs
    st.markdown("---")
    st.header("🎮 Get Predictions")
    tab1, tab2 = st.tabs(["Manual Input", "Paste Multiple Matches"])

    with tab1:
        col1, col2 = st.columns(2)
        with col1:
            tournament = st.text_input("Tournament", "ATP Masters")
            player1 = st.text_input("Player 1")
            surface = st.selectbox("Surface", ["Hard", "Clay", "Grass"])
        with col2:
            match_type = st.selectbox("Type", ["ATP", "Challenger"])
            player2 = st.text_input("Player 2")
        
        if st.button("Predict Single Match", type="primary"):
            if player1 and player2:
                st.session_state.matches = [{
                    'tournament': tournament,
                    'player1': player1,
                    'player2': player2,
                    'surface': surface,
                    'type': match_type
                }]

    with tab2:
        text_input = st.text_area("Paste matches (format: Tournament\nPlayer1 vs Player2)", height=150)
        if st.button("Load Pasted Matches"):
            # You can keep your parse_manual_matches function here if you want
            st.info("Manual paste parsing can be added if needed.")

    # Show Predictions
    if st.session_state.matches:
        st.markdown("---")
        st.header("🎯 Predictions")
        
        results_data = []
        for match in st.session_state.matches:
            pred = predict_match(
                st.session_state.model_winner,
                st.session_state.model_ou,
                st.session_state.player_stats,
                match['player1'], match['player2'], match['surface']
            )
            
            conf_class = get_confidence_class(pred['winner_conf'])
            
            st.markdown(f"""
            <div class="prediction-card">
                <div class="match-header">
                    <span class="tournament-badge">{match['type']}</span>
                    <span class="tournament-badge">{match['surface']}</span>
                    {match['tournament']}
                </div>
                <h3>{match['player1']} vs {match['player2']}</h3>
                <div>
                    🏆 <strong>Winner:</strong> <span class="{conf_class}">{pred['winner']} ({pred['winner_conf']:.1%})</span><br>
                    🎲 <strong>O/U 21.5:</strong> {pred['ou']} ({pred['ou_conf']:.1%})<br>
                    Expected Games: {pred['exp_games']:.1f}
                </div>
            </div>
            """, unsafe_allow_html=True)
            
            results_data.append({
                'Tournament': match['tournament'],
                'Match': f"{match['player1']} vs {match['player2']}",
                'Predicted Winner': pred['winner'],
                'Win %': f"{pred['winner_conf']:.1%}",
                'Over/Under': pred['ou'],
                'Expected Games': round(pred['exp_games'], 1)
            })
        
        if results_data:
            df_results = pd.DataFrame(results_data)
            st.download_button("Download Predictions (CSV)", df_results.to_csv(index=False), "predictions.csv", "text/csv")

if __name__ == "__main__":
    main()

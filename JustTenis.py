import os
import re
import time
import warnings
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import streamlit as st
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import requests
from bs4 import BeautifulSoup
import json

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
    .match-header {
        font-size: 1.3em;
        font-weight: bold;
        margin-bottom: 10px;
        color: #ffffff;
    }
    .tournament-badge {
        background: rgba(255,255,255,0.2);
        padding: 5px 10px;
        border-radius: 5px;
        font-size: 0.9em;
        display: inline-block;
        margin-right: 10px;
    }
    .stats-box {
        background: rgba(255,255,255,0.1);
        padding: 10px;
        border-radius: 8px;
        margin: 5px 0;
    }
    </style>
""", unsafe_allow_html=True)

# ==============================================================================
# Helper Functions
# ==============================================================================

def parse_score_to_games(score):
    """Parse tennis score string to total games count"""
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
    """Load and preprocess historical tennis data"""
    try:
        df = pd.read_excel(file_path)
        
        if df.empty:
            st.error("Historical data file is empty")
            return None
        
        # Normalize column names
        df.columns = [str(c).strip().lower().replace(' ', '_') for c in df.columns]
        
        # Rename columns for consistency
        rename_map = {
            'tourney_date': 'date',
            'winner_name': 'winner',
            'loser_name': 'loser',
            'winner_rank': 'wrank',
            'loser_rank': 'lrank',
            'winner_rank_points': 'wpts',
            'loser_rank_points': 'lpts',
            't_games': 'total_games'
        }
        
        df.rename(columns={k: v for k, v in rename_map.items() if k in df.columns}, inplace=True)
        
        # Convert date
        if 'date' in df.columns:
            df['date'] = pd.to_datetime(df['date'].astype(str), format='%Y%m%d', errors='coerce')
        
        # Calculate total games if not present
        if 'total_games' not in df.columns and 'score' in df.columns:
            df['total_games'] = df['score'].apply(parse_score_to_games)
        
        return df
        
    except Exception as e:
        st.error(f"Error loading historical data: {e}")
        return None

def compute_player_stats(df):
    """Compute player statistics from historical data"""
    stats = {}
    
    for player in set(df['winner'].unique()) | set(df['loser'].unique()):
        player_matches = df[(df['winner'] == player) | (df['loser'] == player)]
        
        if len(player_matches) == 0:
            continue
        
        wins = len(df[df['winner'] == player])
        losses = len(df[df['loser'] == player])
        total = wins + losses
        
        if total == 0:
            continue
        
        # Win rate
        win_rate = wins / total
        
        # Surface-specific stats
        surfaces = {}
        for surf in ['Hard', 'Clay', 'Grass', 'Carpet']:
            surf_matches = player_matches[player_matches['surface'] == surf]
            surf_wins = len(surf_matches[surf_matches['winner'] == player])
            surf_total = len(surf_matches)
            surfaces[surf] = surf_wins / surf_total if surf_total > 0 else win_rate
        
        # Average ranking
        player_as_winner = df[df['winner'] == player]
        player_as_loser = df[df['loser'] == player]
        
        ranks = []
        if not player_as_winner.empty:
            ranks.extend(player_as_winner['wrank'].dropna().tolist())
        if not player_as_loser.empty:
            ranks.extend(player_as_loser['lrank'].dropna().tolist())
        
        avg_rank = np.mean(ranks) if ranks else 200
        
        # Recent form (last 10 matches)
        recent = player_matches.sort_values('date', ascending=False).head(10)
        recent_wins = len(recent[recent['winner'] == player])
        recent_form = recent_wins / len(recent) if len(recent) > 0 else win_rate
        
        # Average total games
        avg_games = player_matches['total_games'].mean() if 'total_games' in player_matches.columns else 22
        
        stats[player] = {
            'win_rate': win_rate,
            'wins': wins,
            'losses': losses,
            'avg_rank': avg_rank,
            'recent_form': recent_form,
            'avg_games': avg_games,
            'hard': surfaces.get('Hard', win_rate),
            'clay': surfaces.get('Clay', win_rate),
            'grass': surfaces.get('Grass', win_rate),
            'carpet': surfaces.get('Carpet', win_rate)
        }
    
    return stats

def train_models(df, player_stats):
    """Train ML models for match outcome and over/under prediction"""
    
    # Prepare training data
    X_data = []
    y_winner = []
    y_games = []
    
    for _, row in df.iterrows():
        p1 = row['winner']
        p2 = row['loser']
        surf = row.get('surface', 'Hard')
        
        if p1 not in player_stats or p2 not in player_stats:
            continue
        
        # Features
        p1_stats = player_stats[p1]
        p2_stats = player_stats[p2]
        
        surf_key = surf.lower()
        
        features = [
            p1_stats['win_rate'],
            p2_stats['win_rate'],
            p1_stats['avg_rank'],
            p2_stats['avg_rank'],
            p1_stats['recent_form'],
            p2_stats['recent_form'],
            p1_stats.get(surf_key, p1_stats['win_rate']),
            p2_stats.get(surf_key, p2_stats['win_rate']),
            p1_stats['avg_games'],
            p2_stats['avg_games'],
        ]
        
        X_data.append(features)
        y_winner.append(1)  # Player 1 won
        y_games.append(row.get('total_games', 22))
    
    X = np.array(X_data)
    y_w = np.array(y_winner)
    y_g = np.array(y_games)
    
    # Train winner prediction model
    model_winner = GradientBoostingClassifier(n_estimators=100, random_state=42)
    model_winner.fit(X, y_w)
    
    # Train over/under model (classify as Over or Under 21.5 games)
    y_ou = (y_g > 21.5).astype(int)
    model_ou = GradientBoostingClassifier(n_estimators=100, random_state=42)
    model_ou.fit(X, y_ou)
    
    return model_winner, model_ou

def predict_match(model_winner, model_ou, player_stats, p1, p2, surface):
    """Predict match outcome and over/under"""
    
    # Default values for unknown players
    default_stats = {
        'win_rate': 0.5,
        'avg_rank': 200,
        'recent_form': 0.5,
        'avg_games': 22,
        'hard': 0.5,
        'clay': 0.5,
        'grass': 0.5,
        'carpet': 0.5
    }
    
    p1_stats = player_stats.get(p1, default_stats)
    p2_stats = player_stats.get(p2, default_stats)
    
    surf_key = surface.lower()
    
    features = np.array([[
        p1_stats['win_rate'],
        p2_stats['win_rate'],
        p1_stats['avg_rank'],
        p2_stats['avg_rank'],
        p1_stats['recent_form'],
        p2_stats['recent_form'],
        p1_stats.get(surf_key, p1_stats['win_rate']),
        p2_stats.get(surf_key, p2_stats['win_rate']),
        p1_stats['avg_games'],
        p2_stats['avg_games'],
    ]])
    
    # Predict winner
    p1_prob = model_winner.predict_proba(features)[0][1]
    p2_prob = 1 - p1_prob
    
    winner = p1 if p1_prob > p2_prob else p2
    winner_conf = max(p1_prob, p2_prob)
    
    # Predict over/under
    ou_prob = model_ou.predict_proba(features)[0][1]
    ou_result = "Over 21.5" if ou_prob > 0.5 else "Under 21.5"
    ou_conf = max(ou_prob, 1 - ou_prob)
    
    # Expected games
    exp_games = (p1_stats['avg_games'] + p2_stats['avg_games']) / 2
    
    return {
        'winner': winner,
        'winner_conf': winner_conf,
        'p1_prob': p1_prob,
        'p2_prob': p2_prob,
        'ou': ou_result,
        'ou_conf': ou_conf,
        'exp_games': exp_games,
        'p1_known': p1 in player_stats,
        'p2_known': p2 in player_stats
    }

def scrape_matches_flashscore(days_ahead=0):
    """
    Scrape tennis matches from Flashscore (ATP and Challenger only)
    days_ahead: 0 for today, 1 for tomorrow
    """
    matches = []
    
    try:
        # Flashscore tennis schedule URL
        target_date = datetime.now() + timedelta(days=days_ahead)
        date_str = target_date.strftime('%d.%m.%Y')
        
        # Using a simple requests approach (Flashscore data is in HTML)
        url = f"https://www.flashscore.com/tennis/"
        
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        }
        
        st.info(f"🔍 Attempting to fetch matches for {date_str}...")
        
        # Note: Flashscore requires JavaScript, so we provide manual input option
        st.warning("⚠️ Automatic scraping from Flashscore requires JavaScript. Please use the manual input tab or paste match data.")
        
        return []
        
    except Exception as e:
        st.error(f"Scraping error: {e}")
        return []

def parse_manual_matches(text_input):
    """Parse manually entered match data"""
    matches = []
    lines = text_input.strip().split('\n')
    
    current_tournament = ""
    current_surface = "Hard"
    
    for line in lines:
        line = line.strip()
        if not line:
            continue
        
        # Check if line is a tournament header
        if any(keyword in line.upper() for keyword in ['ATP', 'CHALLENGER', 'TOURNAMENT', 'MASTERS']):
            current_tournament = line
            # Detect surface from tournament name
            if 'CLAY' in line.upper() or 'ROLAND' in line.upper() or 'MONTE' in line.upper():
                current_surface = 'Clay'
            elif 'GRASS' in line.upper() or 'WIMBLEDON' in line.upper():
                current_surface = 'Grass'
            else:
                current_surface = 'Hard'
            continue
        
        # Parse match line (format: "Player1 vs Player2" or "Player1 - Player2")
        if ' vs ' in line or ' - ' in line:
            separator = ' vs ' if ' vs ' in line else ' - '
            parts = line.split(separator)
            
            if len(parts) == 2:
                p1 = parts[0].strip()
                p2 = parts[1].strip()
                
                # Remove odds or extra info in parentheses
                p1 = re.sub(r'\([^)]*\)', '', p1).strip()
                p2 = re.sub(r'\([^)]*\)', '', p2).strip()
                
                # Determine match type
                match_type = "ATP" if "ATP" in current_tournament.upper() else "Challenger"
                
                matches.append({
                    'tournament': current_tournament,
                    'player1': p1,
                    'player2': p2,
                    'surface': current_surface,
                    'type': match_type
                })
    
    return matches

def get_confidence_class(conf):
    """Return CSS class based on confidence level"""
    if conf >= 0.70:
        return "confidence-high"
    elif conf >= 0.55:
        return "confidence-medium"
    else:
        return "confidence-low"

# ==============================================================================
# Main Application
# ==============================================================================

def main():
    st.title("🎾 ATP & Challenger Tennis Match Predictor")
    st.markdown("### Predict match winners and Over/Under 21.5 games with ML")
    
    # Sidebar
    with st.sidebar:
        st.header("⚙️ Configuration")
        
        # File upload
        uploaded_file = st.file_uploader(
            "Upload Historical Data (Excel)",
            type=['xlsx'],
            help="Upload your Challenger/ATP historical data file"
        )
        
        st.markdown("---")
        st.markdown("### 📊 Model Info")
        st.info("Using Gradient Boosting Classifier for predictions")
        
        st.markdown("### 🎯 Predictions")
        st.write("- Match Winner")
        st.write("- Win Probability")
        st.write("- Over/Under 21.5 games")
        st.write("- Expected total games")
    
    # Initialize session state
    if 'models_trained' not in st.session_state:
        st.session_state.models_trained = False
    if 'matches' not in st.session_state:
        st.session_state.matches = []
    
    # Load and train models
    if uploaded_file is not None:
        with st.spinner("📚 Loading historical data and training models..."):
            # Save uploaded file temporarily
            temp_path = "/tmp/tennis_data.xlsx"
            with open(temp_path, 'wb') as f:
                f.write(uploaded_file.read())
            
            # Load data
            df = load_historical_data(temp_path)
            
            if df is not None:
                st.success(f"✅ Loaded {len(df)} historical matches")
                
                # Compute player stats
                player_stats = compute_player_stats(df)
                st.session_state.player_stats = player_stats
                st.success(f"✅ Computed stats for {len(player_stats)} players")
                
                # Train models
                model_winner, model_ou = train_models(df, player_stats)
                st.session_state.model_winner = model_winner
                st.session_state.model_ou = model_ou
                st.session_state.models_trained = True
                
                st.success("✅ Models trained successfully!")
                
                # Show data summary
                with st.expander("📈 View Data Summary"):
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("Total Matches", len(df))
                    with col2:
                        st.metric("Unique Players", len(player_stats))
                    with col3:
                        avg_games = df['total_games'].mean() if 'total_games' in df.columns else 0
                        st.metric("Avg Games/Match", f"{avg_games:.1f}")
    else:
        st.warning("⬆️ Please upload historical data file to begin")
        st.info("💡 The app needs historical ATP/Challenger match data to train prediction models")
        return
    
    if not st.session_state.models_trained:
        return
    
    # Match Input Section
    st.markdown("---")
    st.header("🎮 Get Match Predictions")
    
    tab1, tab2 = st.tabs(["📝 Manual Input", "📋 Paste Multiple Matches"])
    
    with tab1:
        st.subheader("Enter Match Details")
        
        col1, col2 = st.columns(2)
        
        with col1:
            tournament = st.text_input("Tournament Name", value="ATP Tournament")
            player1 = st.text_input("Player 1", value="")
            surface = st.selectbox("Surface", ["Hard", "Clay", "Grass", "Carpet"])
        
        with col2:
            match_type = st.selectbox("Type", ["ATP", "Challenger"])
            player2 = st.text_input("Player 2", value="")
            st.markdown("<br>", unsafe_allow_html=True)
        
        if st.button("🔮 Predict This Match", type="primary"):
            if player1 and player2:
                match = {
                    'tournament': tournament,
                    'player1': player1,
                    'player2': player2,
                    'surface': surface,
                    'type': match_type
                }
                st.session_state.matches = [match]
            else:
                st.error("Please enter both player names")
    
    with tab2:
        st.subheader("Paste Match Data")
        st.markdown("""
        **Format examples:**
        ```
        ATP Miami - Hard Court
        Carlos Alcaraz vs Novak Djokovic
        Jannik Sinner vs Daniil Medvedev
        
        Challenger Lyon - Clay Court
        Arthur Fils vs Hugo Gaston
        ```
        """)
        
        text_input = st.text_area(
            "Paste matches here (one per line)",
            height=200,
            placeholder="Tournament Name\nPlayer1 vs Player2\nPlayer3 vs Player4\n..."
        )
        
        col1, col2 = st.columns([1, 1])
        with col1:
            if st.button("📥 Load Matches", type="primary"):
                if text_input.strip():
                    matches = parse_manual_matches(text_input)
                    if matches:
                        st.session_state.matches = matches
                        st.success(f"✅ Loaded {len(matches)} matches")
                    else:
                        st.error("No valid matches found. Check format.")
                else:
                    st.error("Please paste match data")
        
        with col2:
            if st.button("🗑️ Clear All"):
                st.session_state.matches = []
                st.rerun()
    
    # Display Predictions
    if st.session_state.matches:
        st.markdown("---")
        st.header("🎯 Match Predictions")
        
        # Filters
        col1, col2 = st.columns([2, 1])
        with col1:
            filter_type = st.radio(
                "Filter by type:",
                ["All Matches", "ATP Only", "Challenger Only"],
                horizontal=True
            )
        with col2:
            show_unknown = st.checkbox("Show unknown players", value=True)
        
        # Filter matches
        matches = st.session_state.matches
        if filter_type == "ATP Only":
            matches = [m for m in matches if m.get('type') == 'ATP']
        elif filter_type == "Challenger Only":
            matches = [m for m in matches if m.get('type') == 'Challenger']
        
        # Generate predictions
        results_data = []
        
        for idx, match in enumerate(matches):
            p1 = match['player1']
            p2 = match['player2']
            surf = match['surface']
            
            # Check if players are known
            p1_known = p1 in st.session_state.player_stats
            p2_known = p2 in st.session_state.player_stats
            
            if not show_unknown and (not p1_known or not p2_known):
                continue
            
            # Get prediction
            try:
                pred = predict_match(
                    st.session_state.model_winner,
                    st.session_state.model_ou,
                    st.session_state.player_stats,
                    p1, p2, surf
                )
                
                # Display prediction card
                conf_class = get_confidence_class(pred['winner_conf'])
                ou_class = get_confidence_class(pred['ou_conf'])
                
                unknown_badge = "⚠️ Unknown Player(s)" if not p1_known or not p2_known else ""
                
                st.markdown(f"""
                <div class="prediction-card">
                    <div class="match-header">
                        <span class="tournament-badge">{match.get('type', 'ATP')}</span>
                        <span class="tournament-badge">{surf}</span>
                        {match.get('tournament', 'Tournament')}
                    </div>
                    <div style="font-size: 1.5em; margin: 15px 0; color: #fff;">
                        <strong>{p1}</strong> vs <strong>{p2}</strong>
                        <span style="font-size: 0.7em; color: #ffaa00;">{unknown_badge}</span>
                    </div>
                    <div class="stats-box">
                        <div style="display: flex; justify-content: space-between; margin-bottom: 10px;">
                            <div>
                                <span style="font-size: 1.2em;">🏆 Winner Prediction:</span><br>
                                <span class="{conf_class}">
                                    {pred['winner']} ({pred['winner_conf']:.1%} confidence)
                                </span>
                            </div>
                            <div style="text-align: right;">
                                <span style="font-size: 0.9em; opacity: 0.8;">
                                    {p1}: {pred['p1_prob']:.1%}<br>
                                    {p2}: {pred['p2_prob']:.1%}
                                </span>
                            </div>
                        </div>
                        <div style="display: flex; justify-content: space-between;">
                            <div>
                                <span style="font-size: 1.2em;">🎲 Over/Under 21.5:</span><br>
                                <span class="{ou_class}">
                                    {pred['ou']} ({pred['ou_conf']:.1%} confidence)
                                </span>
                            </div>
                            <div style="text-align: right;">
                                <span style="font-size: 0.9em; opacity: 0.8;">
                                    Expected: {pred['exp_games']:.1f} games
                                </span>
                            </div>
                        </div>
                    </div>
                </div>
                """, unsafe_allow_html=True)
                
                # Store for CSV export
                results_data.append({
                    'Tournament': match.get('tournament', ''),
                    'Type': match.get('type', 'ATP'),
                    'Surface': surf,
                    'Match': f"{p1} vs {p2}",
                    'Predicted Winner': pred['winner'],
                    'Win Confidence': f"{pred['winner_conf']:.1%}",
                    'P1 Win Prob': f"{pred['p1_prob']:.1%}",
                    'P2 Win Prob': f"{pred['p2_prob']:.1%}",
                    'Over/Under': pred['ou'],
                    'O/U Confidence': f"{pred['ou_conf']:.1%}",
                    'Expected Games': f"{pred['exp_games']:.1f}",
                    'Known Players': 'Yes' if p1_known and p2_known else 'Partial' if p1_known or p2_known else 'No'
                })
                
            except Exception as e:
                st.error(f"❌ Prediction error for {p1} vs {p2}: {str(e)}")
        
        # Export options
        if results_data:
            st.markdown("---")
            df_results = pd.DataFrame(results_data)
            
            col1, col2, col3 = st.columns([2, 1, 1])
            with col1:
                st.metric("Total Predictions", len(df_results))
            with col2:
                csv = df_results.to_csv(index=False)
                st.download_button(
                    "📥 Download CSV",
                    csv,
                    file_name=f"tennis_predictions_{datetime.now().strftime('%Y%m%d')}.csv",
                    mime="text/csv"
                )
            with col3:
                if st.button("📊 Show Table"):
                    st.dataframe(df_results, use_container_width=True)

if __name__ == "__main__":
    main()

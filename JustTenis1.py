import warnings
from collections import defaultdict
from datetime import datetime, timedelta
import io
import numpy as np
import pandas as pd
import streamlit as st
import requests
from lightgbm import LGBMClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.calibration import CalibratedClassifierCV
from scipy.stats import norm
import re

warnings.filterwarnings('ignore')

st.set_page_config(page_title="🎾 ATP Predictor v4.1 - Fixed Features", page_icon="🎾", layout="wide")

# ==============================================================================
# CONFIG
# ==============================================================================
ENSEMBLE_WEIGHTS = {
    'lightgbm': 0.40,
    'random_forest': 0.30,
    'gradient_boosting': 0.20,
    'cbrf': 0.10
}

WINNER_SMOOTH = 0.50  # Reduzido para permitir mais variação
OU_SMOOTH = 0.50

MIN_CONFIDENCE_STRONG = 0.70
MIN_CONFIDENCE_GOOD = 0.62
MIN_CONFIDENCE_WEAK = 0.55

CBRF_MOMENTUM_WINDOW = 5
CBRF_MOMENTUM_DECAY = 0.90

BETAMINIC_MIN_SAMPLES = 10
BETAMINIC_SURFACE_ADJUST = {'Hard': 1.0, 'Clay': 1.03, 'Grass': 0.97}

# ==============================================================================
# SURFACE DETECTION
# ==============================================================================
TOURNAMENT_SURFACE_MAP = {
    'monte carlo': 'Clay', 'madrid': 'Clay', 'rome': 'Clay', 'barcelona': 'Clay',
    'munich': 'Clay', 'estoril': 'Clay', 'geneva': 'Clay', 'oeiras': 'Clay',
    'wimbledon': 'Grass', 'queens': 'Grass', 'halle': 'Grass', 'newport': 'Grass',
    'us open': 'Hard', 'australian open': 'Hard', 'roland garros': 'Clay'
}

def detect_surface_from_tournament(tournament_name):
    if pd.isna(tournament_name):
        return 'Hard'
    t = str(tournament_name).lower()
    for key, surf in TOURNAMENT_SURFACE_MAP.items():
        if key in t:
            return surf
    if any(x in t for x in ['clay', 'terre battue']):
        return 'Clay'
    if any(x in t for x in ['grass', 'lawn']):
        return 'Grass'
    return 'Hard'

# ==============================================================================
# ELO SYSTEM
# ==============================================================================
def calculate_elo_ratings(df, k_factor=32):
    """Calculate ELO ratings for all players"""
    players = set(df['winner'].dropna()) | set(df['loser'].dropna())
    elo = {p: 1500 for p in players}
    surface_elo = {p: {'Hard': 1500, 'Clay': 1500, 'Grass': 1500} for p in players}
    
    for _, row in df.sort_values('date').iterrows():
        w, l = row['winner'], row['loser']
        if pd.isna(w) or pd.isna(l):
            continue
        
        surf = row.get('surface', 'Hard')
        
        # Update overall ELO
        exp_w = 1 / (1 + 10 ** ((elo[l] - elo[w]) / 400))
        elo[w] += k_factor * (1 - exp_w)
        elo[l] += k_factor * (0 - (1 - exp_w))
        
        # Update surface-specific ELO
        exp_w_surf = 1 / (1 + 10 ** ((surface_elo[l][surf] - surface_elo[w][surf]) / 400))
        surface_elo[w][surf] += k_factor * (1 - exp_w_surf)
        surface_elo[l][surf] += k_factor * (0 - (1 - exp_w_surf))
    
    return elo, surface_elo

# ==============================================================================
# PLAYER STATISTICS
# ==============================================================================
def calculate_player_statistics(df):
    """Calculate comprehensive player statistics"""
    players = set(df['winner'].dropna()) | set(df['loser'].dropna())
    stats = {}
    
    for player in players:
        matches = df[(df['winner'] == player) | (df['loser'] == player)]
        
        if len(matches) == 0:
            stats[player] = {
                'total_matches': 0,
                'win_rate': 0.5,
                'hard_win_rate': 0.5,
                'clay_win_rate': 0.5,
                'grass_win_rate': 0.5,
                'recent_form': 0.5,
                'very_recent_form': 0.5,
                'avg_games': 22,
                'elo': 1500
            }
            continue
        
        # Calculate win rates
        total_wins = len(matches[matches['winner'] == player])
        win_rate = total_wins / len(matches) if len(matches) > 0 else 0.5
        
        # Surface-specific win rates
        hard_matches = matches[matches['surface'] == 'Hard']
        clay_matches = matches[matches['surface'] == 'Clay']
        grass_matches = matches[matches['surface'] == 'Grass']
        
        hard_wins = len(hard_matches[hard_matches['winner'] == player])
        clay_wins = len(clay_matches[clay_matches['winner'] == player])
        grass_wins = len(grass_matches[grass_matches['winner'] == player])
        
        hard_rate = hard_wins / len(hard_matches) if len(hard_matches) > 0 else 0.5
        clay_rate = clay_wins / len(clay_matches) if len(clay_matches) > 0 else 0.5
        grass_rate = grass_wins / len(grass_matches) if len(grass_matches) > 0 else 0.5
        
        # Recent form (last 10 matches)
        recent = matches.sort_values('date', ascending=False).head(10)
        recent_wins = len(recent[recent['winner'] == player])
        recent_form = recent_wins / len(recent) if len(recent) > 0 else 0.5
        
        # Very recent form (last 3 matches)
        very_recent = matches.sort_values('date', ascending=False).head(3)
        very_recent_wins = len(very_recent[very_recent['winner'] == player])
        very_recent_form = very_recent_wins / len(very_recent) if len(very_recent) > 0 else 0.5
        
        # Average games per match
        avg_games = matches['total_games'].mean() if 'total_games' in matches.columns else 22
        
        stats[player] = {
            'total_matches': len(matches),
            'win_rate': win_rate,
            'hard_win_rate': hard_rate,
            'clay_win_rate': clay_rate,
            'grass_win_rate': grass_rate,
            'recent_form': recent_form,
            'very_recent_form': very_recent_form,
            'avg_games': avg_games,
            'matches': matches
        }
    
    return stats

# ==============================================================================
# FEATURE CONSTRUCTION
# ==============================================================================
def build_match_features(p1, p2, surface, player_stats, h2h_data, elo_ratings, surface_elo):
    """Build features for a single match"""
    
    if p1 not in player_stats or p2 not in player_stats:
        return None
    
    s1 = player_stats[p1]
    s2 = player_stats[p2]
    
    # Get surface-specific win rates
    if surface == 'Clay':
        win_rate1 = s1['clay_win_rate']
        win_rate2 = s2['clay_win_rate']
    elif surface == 'Grass':
        win_rate1 = s1['grass_win_rate']
        win_rate2 = s2['grass_win_rate']
    else:
        win_rate1 = s1['hard_win_rate']
        win_rate2 = s2['hard_win_rate']
    
    # ELO features
    elo1 = elo_ratings.get(p1, 1500)
    elo2 = elo_ratings.get(p2, 1500)
    elo_diff = (elo1 - elo2) / 400
    
    # Surface ELO
    surf_elo1 = surface_elo.get(p1, {}).get(surface, 1500)
    surf_elo2 = surface_elo.get(p2, {}).get(surface, 1500)
    surf_elo_diff = (surf_elo1 - surf_elo2) / 400
    
    # Form features
    form_diff = s1['recent_form'] - s2['recent_form']
    very_recent_diff = s1['very_recent_form'] - s2['very_recent_form']
    
    # Win rate features
    overall_win_diff = s1['win_rate'] - s2['win_rate']
    surface_win_diff = win_rate1 - win_rate2
    
    # H2H feature
    h2h_key = (p1, p2)
    h2h_advantage = 0.5
    if h2h_key in h2h_data:
        h2h_wins = h2h_data[h2h_key].get('wins', 0)
        h2h_total = h2h_data[h2h_key].get('total', 1)
        h2h_advantage = h2h_wins / h2h_total
    
    # Games feature
    avg_games = (s1['avg_games'] + s2['avg_games']) / 2
    games_norm = (avg_games - 21.5) / 10
    
    # Experience feature (number of matches)
    exp_diff = (s1['total_matches'] - s2['total_matches']) / 100
    
    # Combine all features
    features = [
        elo_diff,
        surf_elo_diff,
        form_diff,
        very_recent_diff,
        overall_win_diff,
        surface_win_diff,
        h2h_advantage,
        games_norm,
        exp_diff
    ]
    
    return features

# ==============================================================================
# TRAINING MODELS
# ==============================================================================
def train_prediction_models(df, player_stats, h2h_data, elo_ratings, surface_elo):
    """Train the prediction models"""
    
    X_winner, y_winner = [], []
    X_ou, y_ou = [], []
    
    for _, row in df.iterrows():
        if pd.isna(row.get('winner')) or pd.isna(row.get('loser')):
            continue
        
        winner = row['winner']
        loser = row['loser']
        surface = row.get('surface', 'Hard')
        total_games = row.get('total_games', 22)
        
        # Features for winner prediction
        features = build_match_features(winner, loser, surface, player_stats, h2h_data, elo_ratings, surface_elo)
        
        if features:
            X_winner.append(features)
            y_winner.append(1)  # Winner wins
            
            # Reverse features for loser
            features_rev = build_match_features(loser, winner, surface, player_stats, h2h_data, elo_ratings, surface_elo)
            if features_rev:
                X_winner.append(features_rev)
                y_winner.append(0)  # Loser loses
            
            # OU features (same features, different target)
            X_ou.append(features)
            y_ou.append(1 if total_games > 21.5 else 0)
    
    if len(X_winner) == 0:
        raise ValueError("No training data available")
    
    X_winner = np.array(X_winner)
    X_ou = np.array(X_ou)
    
    # Train winner model
    winner_model = LGBMClassifier(
        n_estimators=200,
        max_depth=5,
        learning_rate=0.03,
        num_leaves=16,
        reg_alpha=0.5,
        reg_lambda=0.5,
        random_state=42,
        verbose=-1
    )
    winner_model.fit(X_winner, y_winner)
    
    # Train OU model
    ou_model = LGBMClassifier(
        n_estimators=150,
        max_depth=4,
        learning_rate=0.04,
        num_leaves=12,
        reg_alpha=0.5,
        reg_lambda=0.5,
        random_state=42,
        verbose=-1
    )
    ou_model.fit(X_ou, y_ou)
    
    return winner_model, ou_model

# ==============================================================================
# PREDICTION FUNCTION
# ==============================================================================
def predict_match(winner_model, ou_model, p1, p2, surface, player_stats, h2h_data, elo_ratings, surface_elo):
    """Predict a single match"""
    
    features = build_match_features(p1, p2, surface, player_stats, h2h_data, elo_ratings, surface_elo)
    
    if features is None:
        return None
    
    features = np.array([features])
    
    # Winner probability
    winner_prob = winner_model.predict_proba(features)[0][1]
    
    # Apply smoothing
    prob_p1 = 0.5 + (winner_prob - 0.5) * WINNER_SMOOTH
    prob_p1 = np.clip(prob_p1, 0.15, 0.85)
    prob_p2 = 1 - prob_p1
    
    # OU probability
    ou_prob_raw = ou_model.predict_proba(features)[0][1]
    ou_prob = 0.5 + (ou_prob_raw - 0.5) * OU_SMOOTH
    ou_prob = np.clip(ou_prob, 0.30, 0.70)
    
    # Calculate confidence
    confidence = abs(prob_p1 - 0.5) * 2
    
    winner = p1 if prob_p1 > 0.5 else p2
    
    # Recommendation
    if confidence >= MIN_CONFIDENCE_STRONG:
        rec = f"🔥 STRONG {winner}"
    elif confidence >= MIN_CONFIDENCE_GOOD:
        rec = f"✅ GOOD {winner}"
    elif confidence >= MIN_CONFIDENCE_WEAK:
        rec = f"🟡 WEAK {winner}"
    else:
        rec = f"⚪ AVOID {winner}"
    
    # Calculate expected games
    expected_games = 21.5 + (ou_prob - 0.5) * 8
    expected_games = np.clip(expected_games, 18, 35)
    
    # Momentum edge (based on recent form difference)
    s1 = player_stats.get(p1, {})
    s2 = player_stats.get(p2, {})
    momentum_edge = s1.get('recent_form', 0.5) - s2.get('recent_form', 0.5)
    
    return {
        'Player1': p1,
        'Player2': p2,
        'Surface': surface,
        'Prob_P1': prob_p1,
        'Prob_P2': prob_p2,
        'Predicted_Winner': winner,
        'Confidence': confidence,
        'Recommendation': rec,
        'Momentum_Edge': round(momentum_edge * 100, 1),
        'Expected_Games': round(expected_games, 1),
        'OU': "Over 21.5" if ou_prob > 0.5 else "Under 21.5",
        'OU_Prob': ou_prob
    }

# ==============================================================================
# SCRAPER
# ==============================================================================
def scrape_matches_sofascore(days_ahead=0):
    """Scrape matches from Sofascore API"""
    try:
        target_date = (datetime.now() + timedelta(days=days_ahead)).strftime("%Y-%m-%d")
        url = f"https://api.sofascore.com/api/v1/sport/tennis/scheduled-events/{target_date}"
        headers = {"User-Agent": "Mozilla/5.0"}
        r = requests.get(url, headers=headers, timeout=10)
        if r.status_code != 200:
            return []
        
        data = r.json()
        matches = []
        for ev in data.get("events", []):
            try:
                category = ev.get("tournament", {}).get("category", {}).get("name", "")
                if "WTA" in str(category).upper():
                    continue
                
                tournament_name = ev["tournament"]["name"]
                surface = detect_surface_from_tournament(tournament_name)
                
                matches.append({
                    "tournament": tournament_name,
                    "player1": ev["homeTeam"]["name"],
                    "player2": ev["awayTeam"]["name"],
                    "surface": surface
                })
            except Exception:
                continue
        return matches
    except Exception:
        return []

# ==============================================================================
# MAIN APP
# ==============================================================================
def main():
    st.title("🎾 ATP Predictor v4.1 - Fixed Features")
    st.caption("Modelo corrigido com features variáveis e previsões realistas")
    
    uploaded_file = st.file_uploader("📁 Upload do ficheiro histórico (Excel/CSV)", type=['xlsx', 'csv'])
    
    if uploaded_file and 'winner_model' not in st.session_state:
        with st.spinner("🔄 Treinando modelo..."):
            try:
                # Load data
                if uploaded_file.name.endswith('.csv'):
                    df = pd.read_csv(uploaded_file)
                else:
                    df = pd.read_excel(uploaded_file)
                
                df.columns = [str(c).strip().lower().replace(' ', '_').replace('-', '_') for c in df.columns]
                
                # Column mapping
                if 'tourney_date' in df.columns:
                    df.rename(columns={'tourney_date': 'date'}, inplace=True)
                if 'winner_name' in df.columns:
                    df.rename(columns={'winner_name': 'winner'}, inplace=True)
                if 'loser_name' in df.columns:
                    df.rename(columns={'loser_name': 'loser'}, inplace=True)
                if 'tourney_name' in df.columns:
                    df.rename(columns={'tourney_name': 'tournament'}, inplace=True)
                
                df['date'] = pd.to_datetime(df['date'], errors='coerce')
                
                # Calculate total games
                if 'total_games' not in df.columns and 'score' in df.columns:
                    def get_games(s):
                        nums = [int(n) for n in re.findall(r'\d+', str(s)) if int(n) < 20]
                        return sum(nums) if nums else 22
                    df['total_games'] = df['score'].apply(get_games)
                elif 'total_games' not in df.columns:
                    df['total_games'] = 22
                
                # Detect surface
                df['surface'] = df['tournament'].apply(detect_surface_from_tournament)
                
                st.info(f"📊 Dados carregados: {len(df)} jogos")
                
                # Calculate ELO ratings
                elo_ratings, surface_elo = calculate_elo_ratings(df)
                
                # Calculate player statistics
                player_stats = calculate_player_statistics(df)
                
                # Build H2H data
                h2h_data = defaultdict(lambda: {'wins': 0, 'total': 0})
                for _, row in df.iterrows():
                    if pd.notna(row.get('winner')) and pd.notna(row.get('loser')):
                        w, l = row['winner'], row['loser']
                        h2h_data[(w, l)]['wins'] += 1
                        h2h_data[(w, l)]['total'] += 1
                
                # Train models
                winner_model, ou_model = train_prediction_models(df, player_stats, h2h_data, elo_ratings, surface_elo)
                
                # Store in session
                st.session_state.winner_model = winner_model
                st.session_state.ou_model = ou_model
                st.session_state.player_stats = player_stats
                st.session_state.h2h_data = h2h_data
                st.session_state.elo_ratings = elo_ratings
                st.session_state.surface_elo = surface_elo
                st.session_state.models_ready = True
                
                st.success(f"✅ Modelo treinado! {len(player_stats)} jogadores no histórico")
                
            except Exception as e:
                st.error(f"Erro: {str(e)}")
                import traceback
                st.code(traceback.format_exc())
    
    if st.session_state.get('models_ready'):
        col1, col2 = st.columns(2)
        with col1:
            if st.button("📅 HOJE", use_container_width=True):
                with st.spinner("Buscando jogos..."):
                    st.session_state.matches = scrape_matches_sofascore(0)
        with col2:
            if st.button("📅 AMANHÃ", use_container_width=True):
                with st.spinner("Buscando jogos..."):
                    st.session_state.matches = scrape_matches_sofascore(1)
        
        # Manual input
        with st.expander("✏️ Previsão Manual"):
            col_a, col_b, col_c = st.columns(3)
            with col_a:
                manual_p1 = st.text_input("Jogador 1")
            with col_b:
                manual_p2 = st.text_input("Jogador 2")
            with col_c:
                manual_surface = st.selectbox("Superfície", ["Hard", "Clay", "Grass"])
            
            if st.button("🔮 Prever") and manual_p1 and manual_p2:
                result = predict_match(
                    st.session_state.winner_model,
                    st.session_state.ou_model,
                    manual_p1, manual_p2, manual_surface,
                    st.session_state.player_stats,
                    st.session_state.h2h_data,
                    st.session_state.elo_ratings,
                    st.session_state.surface_elo
                )
                if result:
                    st.dataframe(pd.DataFrame([result]).style.format({
                        'Prob_P1': '{:.1%}', 'Prob_P2': '{:.1%}',
                        'Confidence': '{:.1%}', 'OU_Prob': '{:.1%}'
                    }), use_container_width=True)
                else:
                    st.warning("Jogador não encontrado no histórico")
        
        # Show predictions
        if st.session_state.get('matches'):
            st.subheader("🎯 Previsões")
            
            results = []
            for match in st.session_state.matches:
                result = predict_match(
                    st.session_state.winner_model,
                    st.session_state.ou_model,
                    match['player1'], match['player2'], match['surface'],
                    st.session_state.player_stats,
                    st.session_state.h2h_data,
                    st.session_state.elo_ratings,
                    st.session_state.surface_elo
                )
                if result:
                    result['Tournament'] = match['tournament']
                    results.append(result)
            
            if results:
                df_results = pd.DataFrame(results)
                
                # Reorder columns
                cols = ['Tournament', 'Player1', 'Player2', 'Surface', 'Prob_P1', 'Prob_P2', 
                       'Predicted_Winner', 'Confidence', 'Recommendation', 'Momentum_Edge', 
                       'Expected_Games', 'OU', 'OU_Prob']
                df_results = df_results[cols]
                
                styled = df_results.style.format({
                    'Prob_P1': '{:.1%}', 'Prob_P2': '{:.1%}',
                    'Confidence': '{:.1%}', 'OU_Prob': '{:.1%}'
                })
                
                st.dataframe(styled, use_container_width=True, hide_index=True)
                
                # Download
                buffer = io.BytesIO()
                df_results.to_excel(buffer, index=False)
                st.download_button(
                    "📥 Download Excel",
                    buffer.getvalue(),
                    f"predictions_{datetime.now().strftime('%Y%m%d_%H%M')}.xlsx",
                    use_container_width=True
                )
                
                # Summary
                st.subheader("📊 Resumo")
                col_s1, col_s2, col_s3 = st.columns(3)
                with col_s1:
                    strong = sum(1 for r in results if 'STRONG' in r['Recommendation'])
                    st.metric("STRONG Picks", strong)
                with col_s2:
                    avg_conf = df_results['Confidence'].mean()
                    st.metric("Confiança Média", f"{avg_conf:.1%}")
                with col_s3:
                    over = sum(1 for r in results if r['OU'] == 'Over 21.5')
                    st.metric("Over 21.5", f"{over}/{len(results)}")
    
    elif not uploaded_file:
        st.info("📂 Faça upload do ficheiro Excel/CSV com dados históricos")
        st.markdown("""
        ### Colunas esperadas:
        - `winner_name` / `vencedor`
        - `loser_name` / `perdedor`  
        - `tourney_name` / `torneio`
        - `tourney_date` / `data`
        - `score` / `placar` (opcional)
        """)

if __name__ == "__main__":
    main()

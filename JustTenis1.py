import warnings
from collections import defaultdict
from datetime import datetime, timedelta
import io
import numpy as np
import pandas as pd
import streamlit as st
import requests
from lightgbm import LGBMClassifier
from scipy.stats import beta
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings('ignore')

st.set_page_config(page_title="🎾 ATP Predictor v3.0 - CBRF + Betaminic", page_icon="🎾", layout="wide")

# ==============================================================================
# CONFIG - CBRF + BETAMINIC INTEGRATION
# ==============================================================================
# CBRF (Momentum-based) parameters
CBRF_MOMENTUM_WINDOW = 5
CBRF_MOMENTUM_DECAY = 0.85
CBRF_SET_WEIGHT = 0.3
CBRF_GAME_WEIGHT = 0.2

# Betaminic parameters
BETAMINIC_OVER_BASELINE = 0.52
BETAMINIC_UNDER_BASELINE = 0.48
BETAMINIC_MIN_SAMPLES = 10

# Calibration
WINNER_SMOOTH = 0.48     # Ajustado para CBRF
OU_SMOOTH = 0.52
MIN_CONFIDENCE_STRONG = 0.68
MIN_CONFIDENCE_GOOD = 0.61

# ==============================================================================
# CBRF MODEL (Momentum-based)
# ==============================================================================
class CBRFModel:
    """
    Case-Based Reasoning with Recency Frequency
    Detects momentum shifts in tennis matches
    """
    
    def __init__(self, momentum_window=5, decay_factor=0.85):
        self.momentum_window = momentum_window
        self.decay_factor = decay_factor
        self.player_momentum = defaultdict(lambda: {
            'recent_games': [],
            'momentum_score': 0.5,
            'set_streak': 0,
            'game_streak': 0,
            'breaking_points': 0
        })
    
    def update_player_history(self, df):
        """Update player history with match results"""
        for _, row in df.sort_values('date').iterrows():
            winner = row.get('winner')
            loser = row.get('loser')
            if pd.isna(winner) or pd.isna(loser):
                continue
            
            # Update winner momentum
            self._update_momentum(winner, True, row)
            # Update loser momentum
            self._update_momentum(loser, False, row)
    
    def _update_momentum(self, player, won, match_row):
        """Update momentum metrics for a player"""
        momentum_data = self.player_momentum[player]
        
        # Add result to recent games
        momentum_data['recent_games'].append(1 if won else 0)
        if len(momentum_data['recent_games']) > self.momentum_window:
            momentum_data['recent_games'].pop(0)
        
        # Calculate momentum with decay
        weighted_sum = 0
        weight_sum = 0
        for i, result in enumerate(reversed(momentum_data['recent_games'])):
            weight = self.decay_factor ** i
            weighted_sum += result * weight
            weight_sum += weight
        
        momentum_data['momentum_score'] = weighted_sum / weight_sum if weight_sum > 0 else 0.5
        
        # Update streaks
        if won:
            momentum_data['set_streak'] = max(1, momentum_data['set_streak'] + 1)
            momentum_data['game_streak'] = max(1, momentum_data['game_streak'] + 1)
        else:
            momentum_data['set_streak'] = min(-1, momentum_data['set_streak'] - 1)
            momentum_data['game_streak'] = min(-1, momentum_data['game_streak'] - 1)
        
        # Track breaking points (close games)
        if 'total_games' in match_row and not pd.isna(match_row['total_games']):
            if abs(match_row['total_games'] - 21.5) < 3:
                if won:
                    momentum_data['breaking_points'] += 1
                else:
                    momentum_data['breaking_points'] -= 1
    
    def get_momentum_features(self, player1, player2, surface):
        """Extract CBRF features for match prediction"""
        p1 = self.player_momentum[player1]
        p2 = self.player_momentum[player2]
        
        # Momentum difference
        momentum_diff = p1['momentum_score'] - p2['momentum_score']
        
        # Streak features
        set_streak_diff = p1['set_streak'] - p2['set_streak']
        game_streak_diff = p1['game_streak'] - p2['game_streak']
        
        # Recent form (last 5 games)
        p1_recent = sum(p1['recent_games']) / len(p1['recent_games']) if p1['recent_games'] else 0.5
        p2_recent = sum(p2['recent_games']) / len(p2['recent_games']) if p2['recent_games'] else 0.5
        recent_diff = p1_recent - p2_recent
        
        # Breaking point advantage
        bp_diff = p1['breaking_points'] - p2['breaking_points']
        
        # CBRF probability calculation
        cbrf_prob = 0.5 + (momentum_diff * 0.3) + (recent_diff * 0.25) + (set_streak_diff * 0.025) + (bp_diff * 0.02)
        cbrf_prob = np.clip(cbrf_prob, 0.05, 0.95)
        
        return {
            'cbrf_probability': cbrf_prob,
            'momentum_diff': momentum_diff,
            'recent_form_diff': recent_diff,
            'set_streak_diff': set_streak_diff,
            'game_streak_diff': game_streak_diff,
            'breaking_points_diff': bp_diff
        }

# ==============================================================================
# BETAMINIC MODEL (Over/Under 21.5)
# ==============================================================================
class BetaminicModel:
    """
    Betaminic-style statistical model for Over/Under 21.5 games
    Uses player-specific serve/hold rates and set distribution
    """
    
    def __init__(self, min_samples=10):
        self.min_samples = min_samples
        self.player_stats = defaultdict(lambda: {
            'avg_games_per_match': 22.0,
            'over_21_5_rate': 0.5,
            'serve_hold_rate': 0.65,
            'tiebreak_frequency': 0.25,
            'three_set_rate': 0.35,
            'games_std': 5.0
        })
    
    def update_player_stats(self, df):
        """Update player statistics from historical data"""
        for player in set(df['winner'].dropna()) | set(df['loser'].dropna()):
            matches = df[(df['winner'] == player) | (df['loser'] == player)]
            if len(matches) < self.min_samples:
                continue
            
            # Average games per match
            if 'total_games' in matches.columns:
                self.player_stats[player]['avg_games_per_match'] = matches['total_games'].mean()
                self.player_stats[player]['games_std'] = matches['total_games'].std()
                self.player_stats[player]['over_21_5_rate'] = (matches['total_games'] > 21.5).mean()
            
            # Estimate serve hold rate from games won/lost
            # Simplified: if player wins, they likely held serve more
            wins = matches[matches['winner'] == player]
            if len(wins) > 0:
                self.player_stats[player]['serve_hold_rate'] = 0.65 + (len(wins) / len(matches)) * 0.15
            
            # Estimate three-set frequency
            if 'best_of' in matches.columns:
                three_set_matches = matches[matches.get('best_of', 3) == 3]
                if len(three_set_matches) > 0:
                    self.player_stats[player]['three_set_rate'] = len(three_set_matches) / len(matches)
    
    def predict_over_under(self, player1, player2, surface='Hard'):
        """Predict probability of Over 21.5 games"""
        p1 = self.player_stats[player1]
        p2 = self.player_stats[player2]
        
        # Combined expected games
        expected_games = (p1['avg_games_per_match'] + p2['avg_games_per_match']) / 2
        
        # Surface adjustment
        surface_multiplier = 1.0
        if surface == 'Clay':
            surface_multiplier = 1.08  # Clay has longer rallies, more games
        elif surface == 'Grass':
            surface_multiplier = 0.92  # Grass has shorter points
        
        expected_games *= surface_multiplier
        
        # Historical over rate
        historical_over = (p1['over_21_5_rate'] + p2['over_21_5_rate']) / 2
        
        # Serve hold impact - higher hold rates = more games
        serve_factor = (p1['serve_hold_rate'] + p2['serve_hold_rate']) / 2
        serve_adjustment = (serve_factor - 0.65) * 0.5
        
        # Three-set probability
        three_set_prob = (p1['three_set_rate'] + p2['three_set_rate']) / 2
        three_set_adjustment = (three_set_prob - 0.35) * 0.3
        
        # Calculate final probability using Bayesian approach
        base_prob = 0.5
        expected_diff = (expected_games - 21.5) / 8
        over_prob = base_prob + expected_diff * 0.4 + serve_adjustment + three_set_adjustment
        over_prob = 0.3 * over_prob + 0.7 * historical_over  # Blend with historical
        over_prob = np.clip(over_prob, 0.15, 0.85)
        
        return over_prob, expected_games

# ==============================================================================
# SURFACE DETECTION
# ==============================================================================
TOURNAMENT_SURFACE_MAP = {
    'monte carlo': 'Clay', 'madrid': 'Clay', 'rome': 'Clay', 'barcelona': 'Clay',
    'munich': 'Clay', 'estoril': 'Clay', 'geneva': 'Clay', 'oeiras': 'Clay',
    'santa cruz': 'Clay', 'tallahassee': 'Clay', 'busan': 'Hard', 'wuning': 'Hard',
    'wimbledon': 'Grass', 'queens': 'Grass', 'halle': 'Grass', 'newport': 'Grass'
}

def detect_surface_from_tournament(tournament_name, surface_hint=None):
    if pd.isna(tournament_name):
        return surface_hint if surface_hint in ['Clay', 'Grass', 'Hard'] else 'Hard'
    t = str(tournament_name).lower()
    for key, surf in TOURNAMENT_SURFACE_MAP.items():
        if key in t:
            return surf
    if any(x in t for x in ['clay', 'terre', 'antuka']):
        return 'Clay'
    if any(x in t for x in ['grass', 'lawn']):
        return 'Grass'
    return surface_hint if surface_hint in ['Clay', 'Grass', 'Hard'] else 'Hard'

# ==============================================================================
# ENHANCED ELO WITH MOMENTUM
# ==============================================================================
def calculate_enhanced_elo(df, recent_matches=20):
    players = set(df['winner'].dropna().unique()) | set(df['loser'].dropna().unique())
    surface_elo = {p: {'Hard':1500.0, 'Clay':1500.0, 'Grass':1500.0} for p in players}
    recent_form = {p: [] for p in players}
    elo_history = {p: [] for p in players}
    
    for _, row in df.sort_values('date').iterrows():
        w, l = row['winner'], row['loser']
        surf = row.get('surface', 'Hard')
        if surf not in ['Hard', 'Clay', 'Grass']:
            surf = 'Hard'
        if pd.isna(w) or pd.isna(l):
            continue
        
        recent_form[w].append(1)
        recent_form[l].append(0)
        if len(recent_form[w]) > recent_matches:
            recent_form[w].pop(0)
        if len(recent_form[l]) > recent_matches:
            recent_form[l].pop(0)
        
        w_recent = sum(recent_form[w]) / len(recent_form[w]) if recent_form[w] else 0.5
        l_recent = sum(recent_form[l]) / len(recent_form[l]) if recent_form[l] else 0.5
        
        # Add momentum boost
        momentum_boost = (w_recent - 0.5) * 15
        
        r1 = surface_elo[w][surf] + 45 * (w_recent - 0.5) + momentum_boost
        r2 = surface_elo[l][surf] + 45 * (l_recent - 0.5)
        
        exp = 1 / (1 + 10 ** ((r2 - r1) / 400))
        
        # Adaptive K-factor
        k_factor = 28 + 12 * (1 - abs(exp - 0.5) * 2)
        
        surface_elo[w][surf] += k_factor * (1 - exp)
        surface_elo[l][surf] += k_factor * (0 - (1 - exp))
        
        # Store history for trend analysis
        elo_history[w].append(surface_elo[w][surf])
        elo_history[l].append(surface_elo[l][surf])
    
    return surface_elo, recent_form, elo_history

def compute_player_stats_enhanced(df, recent_matches=20):
    surface_elo, recent_form, elo_history = calculate_enhanced_elo(df, recent_matches)
    stats = {}
    
    for player in set(df['winner'].dropna()) | set(df['loser'].dropna()):
        matches = df[(df['winner'] == player) | (df['loser'] == player)].copy()
        if len(matches) == 0:
            stats[player] = {
                'surface_elo': {'Hard':1500,'Clay':1500,'Grass':1500},
                'surface_win_rate': {'Hard':0.5,'Clay':0.5,'Grass':0.5},
                'very_recent_form': 0.5, 'recent_20_form': 0.5, 'avg_games': 22,
                'elo_trend': 0, 'volatility': 0.05
            }
            continue
        
        recent = matches.sort_values('date', ascending=False).head(recent_matches)
        very_recent = matches.sort_values('date', ascending=False).head(5)
        
        surface_stats = {}
        for surf in ['Hard', 'Clay', 'Grass']:
            m = matches[matches['surface'] == surf]
            surface_stats[surf] = len(m[m['winner'] == player]) / len(m) if len(m) > 0 else 0.5
        
        # Calculate ELO trend
        elo_vals = elo_history.get(player, [1500] * 5)
        elo_trend = (elo_vals[-1] - elo_vals[0]) / 100 if len(elo_vals) > 1 else 0
        volatility = np.std(elo_vals[-10:]) / 100 if len(elo_vals) >= 10 else 0.05
        
        stats[player] = {
            'surface_elo': surface_elo[player],
            'surface_win_rate': surface_stats,
            'very_recent_form': len(very_recent[very_recent['winner'] == player]) / len(very_recent) if len(very_recent) > 0 else 0.5,
            'recent_20_form': len(recent[recent['winner'] == player]) / len(recent) if len(recent) > 0 else 0.5,
            'avg_games': float(matches.get('total_games', pd.Series([22])).mean()),
            'elo_trend': elo_trend,
            'volatility': volatility
        }
    
    return stats

# ==============================================================================
# ENHANCED FEATURES WITH CBRF + BETAMINIC
# ==============================================================================
def build_features_enhanced(p1, p2, surface, player_stats, h2h_surface, cbrf_model, betaminic_model):
    if p1 not in player_stats or p2 not in player_stats:
        return None
    
    s1 = player_stats[p1]
    s2 = player_stats[p2]
    surf = surface if surface in ['Hard','Clay','Grass'] else 'Hard'
    
    # H2H features
    h2h_surf_ratio = 0.5
    pair = (p1, p2)
    if pair in h2h_surface:
        total = h2h_surface[pair].get(surf, 0) + h2h_surface.get((p2,p1), {}).get(surf, 0) + 1
        h2h_surf_ratio = (h2h_surface[pair].get(surf, 0) + 0.5) / total
    
    # CBRF momentum features
    cbrf_features = cbrf_model.get_momentum_features(p1, p2, surf)
    
    # Betaminic OU features
    betaminic_over_prob, expected_games = betaminic_model.predict_over_under(p1, p2, surf)
    
    # Combined features array
    features = [
        # ELO based (4 features)
        s1['surface_elo'][surf] / (s2['surface_elo'][surf] + 1),
        s1['surface_win_rate'][surf] - s2['surface_win_rate'][surf],
        abs(s1['surface_elo'][surf] - s2['surface_elo'][surf]) / 180,
        s1['elo_trend'] - s2['elo_trend'],
        
        # Form based (3 features)
        s1.get('recent_20_form', 0.5) - s2.get('recent_20_form', 0.5),
        s1.get('very_recent_form', 0.5) - s2.get('very_recent_form', 0.5),
        s1['volatility'] - s2['volatility'],
        
        # CBRF features (5 features)
        cbrf_features['momentum_diff'],
        cbrf_features['recent_form_diff'],
        cbrf_features['set_streak_diff'] / 5,
        cbrf_features['breaking_points_diff'] / 10,
        cbrf_features['cbrf_probability'],
        
        # H2H (1 feature)
        h2h_surf_ratio,
        
        # Games/OU (2 features)
        (s1.get('avg_games', 22) + s2.get('avg_games', 22)) / 44,
        betaminic_over_prob
    ]
    
    return features

# ==============================================================================
# TRAINING WITH ENHANCED MODELS
# ==============================================================================
def train_models_enhanced(df, player_stats, h2h_surface, cbrf_model, betaminic_model):
    X, y_winner, y_ou = [], [], []
    
    for _, row in df.iterrows():
        if pd.isna(row.get('winner')) or pd.isna(row.get('loser')):
            continue
        surf = row.get('surface', 'Hard')
        total_games = row.get('total_games', 22)
        
        feat = build_features_enhanced(row['winner'], row['loser'], surf, player_stats, h2h_surface, cbrf_model, betaminic_model)
        if feat:
            X.append(feat)
            y_winner.append(1)
            y_ou.append(1 if total_games > 21.5 else 0)
            
            # Add reverse match for symmetry
            feat_rev = build_features_enhanced(row['loser'], row['winner'], surf, player_stats, h2h_surface, cbrf_model, betaminic_model)
            if feat_rev:
                X.append(feat_rev)
                y_winner.append(0)
                y_ou.append(1 if total_games > 21.5 else 0)
    
    X = np.array(X)
    
    # Scale features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Train winner prediction model (with CBRF focus)
    model_winner = LGBMClassifier(
        n_estimators=220, max_depth=5, learning_rate=0.03,
        num_leaves=18, reg_alpha=2.5, reg_lambda=2.5,
        subsample=0.85, colsample_bytree=0.8,
        random_state=42, verbose=-1
    )
    
    # Train OU prediction model (with Betaminic focus)
    model_ou = LGBMClassifier(
        n_estimators=180, max_depth=4, learning_rate=0.035,
        num_leaves=14, reg_alpha=2.0, reg_lambda=2.0,
        subsample=0.8, colsample_bytree=0.75,
        random_state=42, verbose=-1
    )
    
    model_winner.fit(X_scaled, y_winner)
    model_ou.fit(X_scaled, y_ou)
    
    return model_winner, model_ou, scaler

# ==============================================================================
# PREDICTION WITH MODEL ENSEMBLE
# ==============================================================================
def predict_match_enhanced(model_winner, model_ou, scaler, player_stats, h2h_surface, 
                           cbrf_model, betaminic_model, match):
    p1 = match['player1']
    p2 = match['player2']
    surface = match['surface']
    
    feat = build_features_enhanced(p1, p2, surface, player_stats, h2h_surface, cbrf_model, betaminic_model)
    if feat is None:
        return None
    
    feat_scaled = scaler.transform([feat])
    
    # Winner prediction with CBRF ensemble
    raw_p = model_winner.predict_proba(feat_scaled)[0][1]
    
    # Blend with CBRF probability
    cbrf_features = cbrf_model.get_momentum_features(p1, p2, surface)
    cbrf_prob = cbrf_features['cbrf_probability']
    
    # Weighted ensemble (70% ML, 30% CBRF)
    blended_prob = 0.7 * raw_p + 0.3 * cbrf_prob
    
    # Apply calibration
    prob_p1 = 0.5 + (blended_prob - 0.5) * WINNER_SMOOTH
    prob_p1 = max(0.08, min(0.92, prob_p1))
    prob_p2 = 1 - prob_p1
    
    # Over/Under with Betaminic ensemble
    raw_ou = model_ou.predict_proba(feat_scaled)[0][1]
    betaminic_over_prob, expected_games = betaminic_model.predict_over_under(p1, p2, surface)
    
    # Weighted ensemble (60% ML, 40% Betaminic)
    blended_ou = 0.6 * raw_ou + 0.4 * betaminic_over_prob
    ou_prob = 0.5 + (blended_ou - 0.5) * OU_SMOOTH
    ou_prob = max(0.20, min(0.80, ou_prob))
    
    winner_pred = p1 if prob_p1 > prob_p2 else p2
    confidence = max(prob_p1, prob_p2)
    
    # Enhanced recommendation with CBRF confidence boost
    cbrf_confidence_boost = abs(cbrf_features['momentum_diff']) * 0.1
    final_confidence = min(0.95, confidence + cbrf_confidence_boost)
    
    if final_confidence >= MIN_CONFIDENCE_STRONG:
        rec = f"✅ STRONG {winner_pred} (CBRF: {cbrf_features['momentum_diff']:.2f})"
    elif final_confidence >= MIN_CONFIDENCE_GOOD:
        rec = f"🟢 {winner_pred}"
    else:
        rec = f"🟡 {winner_pred}"
    
    return {
        'Tournament': match['tournament'],
        'Player1': p1,
        'Player2': p2,
        'Surface': surface,
        'Prob_P1': prob_p1,
        'Prob_P2': prob_p2,
        'Predicted_Winner': winner_pred,
        'Confidence': final_confidence,
        'Recommendation': rec,
        'CBRF_Momentum': cbrf_features['momentum_diff'],
        'Expected_Games': round(expected_games, 1),
        'OU': "Over 21.5" if ou_prob > 0.5 else "Under 21.5",
        'OU_Prob': ou_prob
    }

# ==============================================================================
# SCRAPER
# ==============================================================================
def scrape_matches_sofascore(days_ahead=0):
    try:
        target_date = (datetime.utcnow() + timedelta(days=days_ahead)).strftime("%Y-%m-%d")
        url = f"https://api.sofascore.com/api/v1/sport/tennis/scheduled-events/{target_date}"
        headers = {"User-Agent": "Mozilla/5.0"}
        r = requests.get(url, headers=headers, timeout=12)
        if r.status_code != 200:
            return []
        
        data = r.json()
        matches = []
        for ev in data.get("events", []):
            try:
                if "WTA" in str(ev.get("tournament", {}).get("category", {}).get("name", "")).upper():
                    continue
                matches.append({
                    "tournament": ev["tournament"]["name"],
                    "player1": ev["homeTeam"]["name"],
                    "player2": ev["awayTeam"]["name"],
                    "surface": detect_surface_from_tournament(ev["tournament"]["name"], ev.get("groundType"))
                })
            except:
                continue
        return matches
    except:
        return []

# ==============================================================================
# MAIN APP
# ==============================================================================
def main():
    st.title("🎾 ATP Predictor v3.0 - CBRF + Betaminic Integration")
    st.caption("Modelos: CBRF (Momentum Detection) | Betaminic (Over/Under) | ELO Avançado")
    
    with st.expander("📊 Sobre os Modelos"):
        st.markdown("""
        **CBRF (Case-Based Reasoning with Recency Frequency)**
        - Detecta mudanças de momentum em jogos recentes
        - Analisa streaks de sets e games
        - Avalia pontos de quebra e viradas
        
        **Betaminic Statistics**
        - Especializado em Over/Under 21.5 games
        - Baseado em taxas de saque e hold
        - Considera distribuição de sets e superfície
        
        **ELO Avançado**
        - Fator K adaptativo
        - Histórico de tendência
        - Volatilidade do jogador
        """)

    uploaded_file = st.file_uploader("📁 Upload do teu ficheiro histórico (Excel)", type=['xlsx'])
    
    RECENT_MATCHES = st.slider("📊 Número de jogos recentes para análise", 10, 30, 20, 5)
    
    if uploaded_file and 'model_winner' not in st.session_state:
        with st.spinner("🔄 A treinar modelos CBRF + Betaminic..."):
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
            
            # Calculate total games if not present
            if 'total_games' not in df.columns and 'score' in df.columns:
                def get_games(s):
                    nums = [int(n) for n in str(s).replace(' ', '').split('-') if n.isdigit()]
                    return sum(nums) if nums else 22
                df['total_games'] = df['score'].apply(get_games)
            elif 'total_games' not in df.columns:
                df['total_games'] = 22
            
            # Limit training data
            max_rows = st.slider("Máximo de jogos para treino", 3000, len(df), min(8000, len(df)), 1000)
            if len(df) > max_rows:
                df = df.sort_values('date', ascending=False).head(max_rows).copy()
            
            # Detect surfaces
            df['surface'] = df.apply(lambda row: detect_surface_from_tournament(row.get('tournament'), row.get('surface')), axis=1)
            
            # Initialize CBRF and Betaminic models
            cbrf_model = CBRFModel(momentum_window=CBRF_MOMENTUM_WINDOW, decay_factor=CBRF_MOMENTUM_DECAY)
            betaminic_model = BetaminicModel(min_samples=BETAMINIC_MIN_SAMPLES)
            
            # Update models with historical data
            cbrf_model.update_player_history(df)
            betaminic_model.update_player_stats(df)
            
            # Compute enhanced player stats
            player_stats = compute_player_stats_enhanced(df, RECENT_MATCHES)
            
            # Build H2H surface data
            h2h_surface = defaultdict(lambda: {'Hard':0, 'Clay':0, 'Grass':0})
            for _, row in df.iterrows():
                if pd.notna(row.get('winner')) and pd.notna(row.get('loser')):
                    pair = (row['winner'], row['loser'])
                    h2h_surface[pair][row.get('surface', 'Hard')] += 1
            
            # Train models
            model_winner, model_ou, scaler = train_models_enhanced(df, player_stats, h2h_surface, cbrf_model, betaminic_model)
            
            # Store in session
            st.session_state.model_winner = model_winner
            st.session_state.model_ou = model_ou
            st.session_state.scaler = scaler
            st.session_state.player_stats = player_stats
            st.session_state.h2h_surface = h2h_surface
            st.session_state.cbrf_model = cbrf_model
            st.session_state.betaminic_model = betaminic_model
            
            st.success("✅ Modelos treinados com sucesso! (CBRF + Betaminic integrados)")
            
            # Show model statistics
            st.info(f"📈 Dados: {len(df)} jogos | {len(player_stats)} jogadores | CBRF window: {CBRF_MOMENTUM_WINDOW}")

    # Prediction interface
    if st.session_state.get('model_winner'):
        col1, col2 = st.columns(2)
        with col1:
            if st.button("📅 HOJE", use_container_width=True):
                st.session_state.current_matches = scrape_matches_sofascore(0)
        with col2:
            if st.button("📅 AMANHÃ", use_container_width=True):
                st.session_state.current_matches = scrape_matches_sofascore(1)
        
        # Manual match input
        with st.expander("✏️ Ou insere manualmente um jogo"):
            col_a, col_b, col_c = st.columns(3)
            with col_a:
                manual_p1 = st.text_input("Jogador 1")
            with col_b:
                manual_p2 = st.text_input("Jogador 2")
            with col_c:
                manual_surface = st.selectbox("Superfície", ["Hard", "Clay", "Grass"])
            
            if st.button("🔮 Prever Jogo Manual") and manual_p1 and manual_p2:
                manual_match = {
                    'tournament': 'Manual Entry',
                    'player1': manual_p1,
                    'player2': manual_p2,
                    'surface': manual_surface
                }
                result = predict_match_enhanced(
                    st.session_state.model_winner,
                    st.session_state.model_ou,
                    st.session_state.scaler,
                    st.session_state.player_stats,
                    st.session_state.h2h_surface,
                    st.session_state.cbrf_model,
                    st.session_state.betaminic_model,
                    manual_match
                )
                if result:
                    st.dataframe(pd.DataFrame([result]).style.format({
                        'Prob_P1': '{:.1%}',
                        'Prob_P2': '{:.1%}',
                        'Confidence': '{:.1%}',
                        'OU_Prob': '{:.1%}'
                    }), use_container_width=True)
        
        # Show predictions
        if st.session_state.get('current_matches'):
            st.subheader("🎯 Previsões do Dia")
            
            results = []
            progress_bar = st.progress(0)
            for i, match in enumerate(st.session_state.current_matches):
                result = predict_match_enhanced(
                    st.session_state.model_winner,
                    st.session_state.model_ou,
                    st.session_state.scaler,
                    st.session_state.player_stats,
                    st.session_state.h2h_surface,
                    st.session_state.cbrf_model,
                    st.session_state.betaminic_model,
                    match
                )
                if result:
                    results.append(result)
                progress_bar.progress((i + 1) / len(st.session_state.current_matches))
            
            progress_bar.empty()
            
            if results:
                df_show = pd.DataFrame(results)
                
                # Color coding for recommendations
                def color_recommendation(val):
                    if 'STRONG' in str(val):
                        return 'background-color: #2e7d32; color: white'
                    elif '🟢' in str(val):
                        return 'background-color: #4caf50; color: white'
                    elif '🟡' in str(val):
                        return 'background-color: #ff9800; color: black'
                    return ''
                
                styled = df_show.style.format({
                    'Prob_P1': '{:.1%}',
                    'Prob_P2': '{:.1%}',
                    'Confidence': '{:.1%}',
                    'OU_Prob': '{:.1%}'
                }).applymap(color_recommendation, subset=['Recommendation'])
                
                st.dataframe(styled, use_container_width=True, hide_index=True, height=700)
                
                # Download button
                buffer = io.BytesIO()
                df_show.to_excel(buffer, index=False)
                st.download_button(
                    "📥 Baixar Excel com Previsões",
                    buffer.getvalue(),
                    f"previsoes_atp_{datetime.now().strftime('%Y-%m-%d_%H%M')}.xlsx",
                    use_container_width=True
                )
                
                # Summary statistics
                st.subheader("📊 Resumo das Previsões")
                col_s1, col_s2, col_s3, col_s4 = st.columns(4)
                with col_s1:
                    st.metric("Total Jogos", len(results))
                with col_s2:
                    strong_count = sum(1 for r in results if 'STRONG' in r['Recommendation'])
                    st.metric("STRONG Picks", strong_count)
                with col_s3:
                    avg_conf = df_show['Confidence'].mean()
                    st.metric("Confiança Média", f"{avg_conf:.1%}")
                with col_s4:
                    over_count = sum(1 for r in results if r['OU'] == 'Over 21.5')
                    st.metric("Over 21.5", f"{over_count}/{len(results)}")

if __name__ == "__main__":
    main()

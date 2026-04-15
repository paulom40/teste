import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from collections import defaultdict
from datetime import datetime, timedelta
import warnings
from io import BytesIO
import random
warnings.filterwarnings('ignore')

# ==============================================================================
# PAGE CONFIGURATION
# ==============================================================================
st.set_page_config(
    page_title="Tennis O/U 21.5 Predictor",
    page_icon="🎾",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
    <style>
    .main-header {
        font-size: 2.5rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 1rem;
    }
    .sub-header {
        font-size: 1.5rem;
        color: #2c3e50;
        margin-top: 1rem;
        margin-bottom: 1rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
    }
    .prediction-over {
        background-color: #ff6b6b;
        color: white;
        padding: 1rem;
        border-radius: 0.5rem;
        text-align: center;
    }
    .prediction-under {
        background-color: #51cf66;
        color: white;
        padding: 1rem;
        border-radius: 0.5rem;
        text-align: center;
    }
    .high-confidence {
        border-left: 4px solid #51cf66;
        padding-left: 1rem;
        margin: 0.5rem 0;
    }
    .medium-confidence {
        border-left: 4px solid #ffd43b;
        padding-left: 1rem;
        margin: 0.5rem 0;
    }
    .low-confidence {
        border-left: 4px solid #ff6b6b;
        padding-left: 1rem;
        margin: 0.5rem 0;
    }
    </style>
""", unsafe_allow_html=True)

# ==============================================================================
# SURFACE-SPECIFIC MATCH LENGTH BASELINE
# ==============================================================================

def compute_surface_length_stats(df):
    """Calculate baseline match length statistics by surface."""
    length_stats = {
        'Hard': {
            'avg_games': 22.0,
            'median_games': 22.0,
            'std': 5.0,
            'p_over_21_5': 0.45,
            'q1': 19,
            'q3': 25,
        },
        'Clay': {
            'avg_games': 26.0,
            'median_games': 25.0,
            'std': 6.5,
            'p_over_21_5': 0.62,
            'q1': 21,
            'q3': 31,
        },
        'Grass': {
            'avg_games': 20.0,
            'median_games': 19.0,
            'std': 4.0,
            'p_over_21_5': 0.28,
            'q1': 17,
            'q3': 23,
        },
    }
    
    # Compute from actual data if available
    if df is not None and len(df) > 0:
        for surface in ['Hard', 'Clay', 'Grass']:
            surf_matches = df[df['surface'] == surface]
            
            if len(surf_matches) > 10:
                games = surf_matches['total_games'].values
                
                length_stats[surface] = {
                    'avg_games': float(np.mean(games)),
                    'median_games': float(np.median(games)),
                    'std': float(np.std(games)) if np.std(games) > 0 else 1.0,
                    'p_over_21_5': float((games > 21.5).mean()),
                    'q1': float(np.percentile(games, 25)),
                    'q3': float(np.percentile(games, 75)),
                    'min_games': float(np.min(games)),
                    'max_games': float(np.max(games)),
                    'count': len(surf_matches),
                }
    
    return length_stats


# ==============================================================================
# PLAYER STATS AND FEATURES
# ==============================================================================

def calculate_serve_return_stats(df):
    """Extract serve dominance and return weakness metrics."""
    stats = defaultdict(lambda: {
        'straight_set_win_rate': 0.35,
        'three_set_loss_rate': 0.35,
        'serve_dominance': 0.35,
        'return_weakness': 0.35,
        'total_wins': 0,
        'total_losses': 0,
    })
    
    if df is None or len(df) == 0:
        return stats
    
    players = set(df['winner'].dropna().unique()) | set(df['loser'].dropna().unique())
    
    for player in players:
        matches = df[(df['winner'] == player) | (df['loser'] == player)]
        
        if len(matches) == 0:
            continue
        
        wins = df[df['winner'] == player]
        losses = df[df['loser'] == player]
        
        straight_set_wins = 0
        total_wins = len(wins)
        
        if total_wins > 0:
            for _, row in wins.iterrows():
                score = str(row.get('score', ''))
                sets = len([x for x in score.split() if x])
                if sets == 2:
                    straight_set_wins += 1
        
        three_set_losses = 0
        total_losses = len(losses)
        
        if total_losses > 0:
            for _, row in losses.iterrows():
                score = str(row.get('score', ''))
                sets = len([x for x in score.split() if x])
                if sets >= 3:
                    three_set_losses += 1
        
        total_matches = total_wins + total_losses
        
        stats[player] = {
            'straight_set_win_rate': straight_set_wins / max(total_wins, 1),
            'three_set_loss_rate': three_set_losses / max(total_losses, 1),
            'serve_dominance': straight_set_wins / max(total_matches, 1),
            'return_weakness': three_set_losses / max(total_matches, 1),
            'total_wins': total_wins,
            'total_losses': total_losses,
        }
    
    return stats


def calculate_player_match_style(df, player_stats):
    """Identify 'grinders' (extend matches) vs 'closers' (finish quickly)."""
    if df is None or len(df) == 0:
        return player_stats
    
    for player in player_stats:
        matches = df[(df['winner'] == player) | (df['loser'] == player)]
        
        if len(matches) == 0:
            player_stats[player]['grind_factor'] = 1.0
            player_stats[player]['finish_factor'] = 1.0
            player_stats[player]['match_volatility'] = 0.0
            continue
        
        wins = df[df['winner'] == player]
        losses = df[df['loser'] == player]
        
        avg_games_winning = wins['total_games'].mean() if len(wins) > 0 else 22.0
        avg_games_losing = losses['total_games'].mean() if len(losses) > 0 else 22.0
        
        grind_factor = avg_games_losing / 22.0
        finish_factor = avg_games_winning / 22.0
        
        match_volatility = 0.0
        if len(wins) > 1:
            match_volatility = wins['total_games'].std() / avg_games_winning if avg_games_winning > 0 else 0
        
        player_stats[player]['grind_factor'] = grind_factor
        player_stats[player]['finish_factor'] = finish_factor
        player_stats[player]['match_volatility'] = match_volatility
    
    return player_stats


def extract_recent_length_trend(df, player, surface, window=5):
    """Extract recent match length trend on specific surface."""
    if df is None or len(df) == 0:
        return {
            'avg_recent_games': 22.0,
            'max_recent_games': 22.0,
            'min_recent_games': 22.0,
            'recent_games_increasing': 0.0,
            'recent_count': 0,
        }
    
    matches = df[(df['winner'] == player) | (df['loser'] == player)]
    matches = matches[matches['surface'] == surface]
    
    if 'date' in matches.columns:
        matches = matches.sort_values('date', ascending=False)
    else:
        matches = matches.iloc[::-1]
    
    recent = matches.head(window)
    
    if len(recent) == 0:
        return {
            'avg_recent_games': 22.0,
            'max_recent_games': 22.0,
            'min_recent_games': 22.0,
            'recent_games_increasing': 0.0,
            'recent_count': 0,
        }
    
    games = recent['total_games'].values
    
    trend = 0.0
    if len(recent) >= 3:
        if 'date' in recent.columns:
            recent_sorted = recent.sort_values('date')
        else:
            recent_sorted = recent
        earlier_avg = recent_sorted.iloc[:-2]['total_games'].mean()
        latest_avg = recent_sorted.iloc[-2:]['total_games'].mean()
        trend = (latest_avg - earlier_avg) / 22.0
    
    return {
        'avg_recent_games': float(np.mean(games)),
        'max_recent_games': float(np.max(games)),
        'min_recent_games': float(np.min(games)),
        'recent_games_increasing': trend,
        'recent_count': len(recent),
    }


def build_competitiveness_features(p1, p2, player_stats, h2h_surface, surface):
    """Build competitiveness index - close matches go longer."""
    s1 = player_stats[p1]
    s2 = player_stats[p2]
    
    # ELO GAP (40% weight)
    elo_gap = abs(s1['surface_elo'][surface] - s2['surface_elo'][surface])
    elo_gap_normalized = min(elo_gap / 400.0, 2.0)
    elo_competitiveness = max(1.0 - elo_gap_normalized, 0.0)
    
    # H2H COMPETITIVENESS (30% weight)
    h2h_p1_wins = h2h_surface.get((p1, p2), {}).get(surface, 0)
    h2h_p2_wins = h2h_surface.get((p2, p1), {}).get(surface, 0)
    total_h2h = h2h_p1_wins + h2h_p2_wins
    
    if total_h2h >= 3:
        h2h_win_ratio = h2h_p1_wins / total_h2h
        h2h_competitiveness = 1.0 - abs(h2h_win_ratio - 0.5) * 2
    else:
        h2h_competitiveness = 0.5
    
    # WIN RATE PARITY (30% weight)
    surf_wr_gap = abs(s1['surface_win_rate'][surface] - s2['surface_win_rate'][surface])
    wr_competitiveness = max(1.0 - surf_wr_gap * 2, 0.0)
    
    # COMBINE
    overall_competitiveness = (
        elo_competitiveness * 0.40 +
        h2h_competitiveness * 0.30 +
        wr_competitiveness * 0.30
    )
    
    return {
        'overall_competitiveness': overall_competitiveness,
        'elo_competitiveness': elo_competitiveness,
        'h2h_competitiveness': h2h_competitiveness,
        'wr_competitiveness': wr_competitiveness,
        'elo_gap_normalized': elo_gap_normalized,
        'surf_wr_gap': surf_wr_gap,
    }


def build_ou_features(p1, p2, surface, player_stats, h2h, h2h_surface, 
                      surface_length_stats, df, match=None):
    """Build specialized feature set for Over/Under 21.5 prediction."""
    if p1 not in player_stats or p2 not in player_stats:
        return None
    
    s1 = player_stats[p1]
    s2 = player_stats[p2]
    surf = surface if surface in ['Hard', 'Clay', 'Grass'] else 'Hard'
    
    # SURFACE-SPECIFIC BASELINE
    surf_stats = surface_length_stats[surf]
    surface_over_21_5_baseline = surf_stats['p_over_21_5']
    surface_avg_games = surf_stats['avg_games']
    surface_std = surf_stats['std']
    
    player_avg_p1 = s1['avg_games']
    player_avg_p2 = s2['avg_games']
    
    game_dev_p1 = (player_avg_p1 - surface_avg_games) / max(surface_std, 1.0)
    game_dev_p2 = (player_avg_p2 - surface_avg_games) / max(surface_std, 1.0)
    
    avg_deviations = (abs(game_dev_p1) + abs(game_dev_p2)) / 2
    
    # COMPETITIVENESS INDEX
    comp_data = build_competitiveness_features(p1, p2, player_stats, h2h_surface, surf)
    overall_comp = comp_data['overall_competitiveness']
    elo_gap_norm = comp_data['elo_gap_normalized']
    
    # GRIND FACTOR
    grind_p1 = s1.get('grind_factor', 1.0)
    grind_p2 = s2.get('grind_factor', 1.0)
    finish_p1 = s1.get('finish_factor', 1.0)
    finish_p2 = s2.get('finish_factor', 1.0)
    
    grind_combined = grind_p1 + grind_p2
    finish_combined = finish_p1 * finish_p2
    style_mismatch = abs(grind_p1 - grind_p2)
    
    # RECENT TREND
    recent_p1 = extract_recent_length_trend(df, p1, surf, window=5)
    recent_p2 = extract_recent_length_trend(df, p2, surf, window=5)
    
    recent_avg_p1 = recent_p1['avg_recent_games'] / 22.0
    recent_avg_p2 = recent_p2['avg_recent_games'] / 22.0
    recent_combined = (recent_avg_p1 + recent_avg_p2) / 2
    
    trend_p1 = recent_p1['recent_games_increasing']
    trend_p2 = recent_p2['recent_games_increasing']
    trend_combined = (trend_p1 + trend_p2) / 2
    
    # SERVE DOMINANCE
    serve_dom_p1 = s1.get('serve_dominance', 0.35)
    serve_dom_p2 = s2.get('serve_dominance', 0.35)
    serve_dom_diff = serve_dom_p1 - serve_dom_p2
    serve_dom_avg = (serve_dom_p1 + serve_dom_p2) / 2
    
    # CONTEXT
    surf_exp_p1 = s1['surface_match_count'].get(surf, 5)
    surf_exp_p2 = s2['surface_match_count'].get(surf, 5)
    exp_ratio = (surf_exp_p1 + 5) / (surf_exp_p2 + 5)
    exp_parity = min(exp_ratio, 1.0/exp_ratio) if exp_ratio > 0 else 0.5
    
    recent_form_p1 = s1['recent_form']
    recent_form_p2 = s2['recent_form']
    form_parity = 1.0 - abs(recent_form_p1 - recent_form_p2)
    
    # FEATURE VECTOR
    features = [
        surface_over_21_5_baseline,
        surface_over_21_5_baseline,
        game_dev_p1,
        game_dev_p2,
        overall_comp,
        overall_comp,
        overall_comp,
        elo_gap_norm,
        comp_data['h2h_competitiveness'],
        comp_data['wr_competitiveness'],
        grind_combined,
        grind_combined,
        finish_combined,
        style_mismatch,
        recent_avg_p1,
        recent_avg_p2,
        recent_combined,
        trend_combined,
        abs(trend_p1 - trend_p2),
        serve_dom_diff,
        serve_dom_avg,
        serve_dom_p1,
        serve_dom_p2,
        exp_parity,
        form_parity,
        avg_deviations,
        (grind_p1 + grind_p2) / 2 - (finish_p1 + finish_p2) / 2,
        overall_comp * grind_combined,
    ]
    
    return features


def predict_over_under(features, seed=None):
    """
    Simplified prediction model.
    In production, replace with trained GradientBoosting model.
    """
    if features is None:
        return 0.50, "UNDER", 0.0
    
    if seed is not None:
        np.random.seed(seed)
    
    # Weighted scoring based on feature importance
    weights = {
        'surface_baseline': 0.20,
        'competitiveness': 0.25,
        'grind_factor': 0.15,
        'recent_trend': 0.15,
        'serve_dominance': 0.15,
        'context': 0.10
    }
    
    # Surface baseline (features 0-1)
    surface_score = (features[0] + features[1]) / 2
    
    # Competitiveness (features 4-9)
    comp_score = np.mean(features[4:10])
    
    # Grind factor (features 10-13)
    grind_score = np.mean(features[10:14])
    
    # Recent trend (features 14-18)
    recent_score = np.mean(features[14:19])
    
    # Serve dominance (features 19-22)
    serve_score = 1 - np.mean(features[19:23])
    
    # Context (features 23-27)
    context_score = np.mean(features[23:28])
    
    # Combine scores
    final_score = (
        surface_score * weights['surface_baseline'] +
        comp_score * weights['competitiveness'] +
        grind_score * weights['grind_factor'] +
        recent_score * weights['recent_trend'] +
        serve_score * weights['serve_dominance'] +
        context_score * weights['context']
    )
    
    # Add slight random variation for demo (remove in production)
    final_score = np.clip(final_score + np.random.normal(0, 0.02), 0.2, 0.85)
    
    prediction = "OVER" if final_score > 0.5 else "UNDER"
    confidence = min(abs(final_score - 0.5) * 2, 0.95)
    
    return final_score, prediction, confidence


# ==============================================================================
# DEMO DATA GENERATION
# ==============================================================================

def generate_demo_data():
    """Generate demo player data for demonstration."""
    players = {
        'Novak Djokovic': {
            'surface_elo': {'Hard': 2450, 'Clay': 2400, 'Grass': 2480},
            'surface_win_rate': {'Hard': 0.85, 'Clay': 0.80, 'Grass': 0.88},
            'surface_match_count': {'Hard': 150, 'Clay': 120, 'Grass': 60},
            'avg_games': 21.5,
            'recent_form': 0.85,
            'serve_dominance': 0.65,
            'grind_factor': 0.95,
            'finish_factor': 1.05,
        },
        'Rafael Nadal': {
            'surface_elo': {'Hard': 2400, 'Clay': 2550, 'Grass': 2350},
            'surface_win_rate': {'Hard': 0.78, 'Clay': 0.92, 'Grass': 0.75},
            'surface_match_count': {'Hard': 130, 'Clay': 160, 'Grass': 50},
            'avg_games': 25.5,
            'recent_form': 0.78,
            'serve_dominance': 0.45,
            'grind_factor': 1.25,
            'finish_factor': 0.85,
        },
        'Carlos Alcaraz': {
            'surface_elo': {'Hard': 2350, 'Clay': 2380, 'Grass': 2300},
            'surface_win_rate': {'Hard': 0.75, 'Clay': 0.82, 'Grass': 0.70},
            'surface_match_count': {'Hard': 60, 'Clay': 70, 'Grass': 20},
            'avg_games': 23.0,
            'recent_form': 0.82,
            'serve_dominance': 0.55,
            'grind_factor': 1.10,
            'finish_factor': 0.95,
        },
        'Jannik Sinner': {
            'surface_elo': {'Hard': 2380, 'Clay': 2320, 'Grass': 2350},
            'surface_win_rate': {'Hard': 0.80, 'Clay': 0.72, 'Grass': 0.78},
            'surface_match_count': {'Hard': 80, 'Clay': 65, 'Grass': 25},
            'avg_games': 22.0,
            'recent_form': 0.88,
            'serve_dominance': 0.70,
            'grind_factor': 0.90,
            'finish_factor': 1.10,
        },
        'Daniil Medvedev': {
            'surface_elo': {'Hard': 2420, 'Clay': 2280, 'Grass': 2320},
            'surface_win_rate': {'Hard': 0.82, 'Clay': 0.65, 'Grass': 0.72},
            'surface_match_count': {'Hard': 110, 'Clay': 85, 'Grass': 30},
            'avg_games': 24.0,
            'recent_form': 0.72,
            'serve_dominance': 0.60,
            'grind_factor': 1.15,
            'finish_factor': 0.90,
        },
        'Alexander Zverev': {
            'surface_elo': {'Hard': 2320, 'Clay': 2350, 'Grass': 2250},
            'surface_win_rate': {'Hard': 0.72, 'Clay': 0.75, 'Grass': 0.65},
            'surface_match_count': {'Hard': 100, 'Clay': 95, 'Grass': 35},
            'avg_games': 24.5,
            'recent_form': 0.70,
            'serve_dominance': 0.58,
            'grind_factor': 1.20,
            'finish_factor': 0.88,
        },
        'Stefanos Tsitsipas': {
            'surface_elo': {'Hard': 2300, 'Clay': 2350, 'Grass': 2250},
            'surface_win_rate': {'Hard': 0.70, 'Clay': 0.78, 'Grass': 0.65},
            'surface_match_count': {'Hard': 95, 'Clay': 100, 'Grass': 30},
            'avg_games': 23.5,
            'recent_form': 0.75,
            'serve_dominance': 0.62,
            'grind_factor': 1.05,
            'finish_factor': 0.98,
        },
        'Andrey Rublev': {
            'surface_elo': {'Hard': 2280, 'Clay': 2250, 'Grass': 2220},
            'surface_win_rate': {'Hard': 0.68, 'Clay': 0.65, 'Grass': 0.60},
            'surface_match_count': {'Hard': 90, 'Clay': 75, 'Grass': 25},
            'avg_games': 22.5,
            'recent_form': 0.68,
            'serve_dominance': 0.68,
            'grind_factor': 0.98,
            'finish_factor': 1.02,
        },
        'Holger Rune': {
            'surface_elo': {'Hard': 2250, 'Clay': 2280, 'Grass': 2200},
            'surface_win_rate': {'Hard': 0.65, 'Clay': 0.70, 'Grass': 0.58},
            'surface_match_count': {'Hard': 50, 'Clay': 55, 'Grass': 15},
            'avg_games': 24.0,
            'recent_form': 0.72,
            'serve_dominance': 0.52,
            'grind_factor': 1.18,
            'finish_factor': 0.92,
        },
        'Taylor Fritz': {
            'surface_elo': {'Hard': 2300, 'Clay': 2220, 'Grass': 2280},
            'surface_win_rate': {'Hard': 0.72, 'Clay': 0.60, 'Grass': 0.70},
            'surface_match_count': {'Hard': 85, 'Clay': 60, 'Grass': 25},
            'avg_games': 21.0,
            'recent_form': 0.74,
            'serve_dominance': 0.75,
            'grind_factor': 0.88,
            'finish_factor': 1.12,
        }
    }
    
    # Add default stats for any missing players
    default_stats = {
        'surface_elo': {'Hard': 2200, 'Clay': 2200, 'Grass': 2200},
        'surface_win_rate': {'Hard': 0.50, 'Clay': 0.50, 'Grass': 0.50},
        'surface_match_count': {'Hard': 20, 'Clay': 20, 'Grass': 20},
        'avg_games': 22.0,
        'recent_form': 0.50,
        'serve_dominance': 0.35,
        'grind_factor': 1.0,
        'finish_factor': 1.0,
    }
    
    return players, default_stats


# ==============================================================================
# SCHEDULED MATCHES GENERATOR
# ==============================================================================

def generate_scheduled_matches():
    """Generate scheduled matches for today and tomorrow."""
    players = ['Novak Djokovic', 'Carlos Alcaraz', 'Jannik Sinner', 'Daniil Medvedev',
               'Alexander Zverev', 'Stefanos Tsitsipas', 'Andrey Rublev', 'Holger Rune',
               'Taylor Fritz', 'Rafael Nadal']
    
    surfaces = ['Hard', 'Clay', 'Grass']
    tournaments = {
        'Grand Slam': ['Australian Open', 'Roland Garros', 'Wimbledon', 'US Open'],
        'ATP Masters 1000': ['Indian Wells', 'Miami Open', 'Monte-Carlo', 'Madrid Open', 
                            'Rome Masters', 'Canada Masters', 'Cincinnati Masters', 
                            'Shanghai Masters', 'Paris Masters'],
        'ATP 500': ['Rotterdam', 'Rio Open', 'Dubai', 'Acapulco', 'Barcelona', 
                   'Halle', 'Queens', 'Hamburg', 'Washington', 'Vienna', 'Basel'],
        'ATP 250': ['Adelaide', 'Auckland', 'Dallas', 'Delray Beach', 'Marseille', 
                   'Estoril', 'Munich', 'Geneva', 's-Hertogenbosch', 'Newport']
    }
    
    today = datetime.now()
    tomorrow = today + timedelta(days=1)
    
    matches = []
    
    # Generate 8-12 matches for today
    num_today = random.randint(8, 12)
    for i in range(num_today):
        p1, p2 = random.sample(players, 2)
        surface = random.choice(surfaces)
        tournament_type = random.choice(list(tournaments.keys()))
        tournament = random.choice(tournaments[tournament_type])
        
        # Random time between 10:00 and 22:00
        hour = random.randint(10, 22)
        minute = random.choice([0, 30])
        time = f"{hour:02d}:{minute:02d}"
        
        matches.append({
            'date': today.strftime('%Y-%m-%d'),
            'day': 'Today',
            'time': time,
            'player1': p1,
            'player2': p2,
            'surface': surface,
            'tournament': tournament,
            'tournament_type': tournament_type,
            'round': random.choice(['R1', 'R2', 'R3', 'QF', 'SF', 'F'])
        })
    
    # Generate 6-10 matches for tomorrow
    num_tomorrow = random.randint(6, 10)
    for i in range(num_tomorrow):
        p1, p2 = random.sample(players, 2)
        surface = random.choice(surfaces)
        tournament_type = random.choice(list(tournaments.keys()))
        tournament = random.choice(tournaments[tournament_type])
        
        hour = random.randint(10, 22)
        minute = random.choice([0, 30])
        time = f"{hour:02d}:{minute:02d}"
        
        matches.append({
            'date': tomorrow.strftime('%Y-%m-%d'),
            'day': 'Tomorrow',
            'time': time,
            'player1': p1,
            'player2': p2,
            'surface': surface,
            'tournament': tournament,
            'tournament_type': tournament_type,
            'round': random.choice(['R1', 'R2', 'R3', 'QF', 'SF', 'F'])
        })
    
    return matches


def predict_match(match, player_stats, default_stats, surface_length_stats, df=None):
    """Generate prediction for a single match."""
    p1 = match['player1']
    p2 = match['player2']
    surface = match['surface']
    
    # Get player stats
    stats = {}
    for player in [p1, p2]:
        if player in player_stats:
            stats[player] = player_stats[player]
        else:
            stats[player] = default_stats.copy()
    
    # Add required fields if missing
    for player in [p1, p2]:
        if 'surface_elo' not in stats[player]:
            stats[player]['surface_elo'] = {'Hard': 2200, 'Clay': 2200, 'Grass': 2200}
        if 'surface_win_rate' not in stats[player]:
            stats[player]['surface_win_rate'] = {'Hard': 0.5, 'Clay': 0.5, 'Grass': 0.5}
        if 'surface_match_count' not in stats[player]:
            stats[player]['surface_match_count'] = {'Hard': 20, 'Clay': 20, 'Grass': 20}
        if 'avg_games' not in stats[player]:
            stats[player]['avg_games'] = 22.0
        if 'recent_form' not in stats[player]:
            stats[player]['recent_form'] = 0.5
        if 'serve_dominance' not in stats[player]:
            stats[player]['serve_dominance'] = 0.35
        if 'grind_factor' not in stats[player]:
            stats[player]['grind_factor'] = 1.0
        if 'finish_factor' not in stats[player]:
            stats[player]['finish_factor'] = 1.0
    
    # Build features
    h2h_surface = defaultdict(lambda: defaultdict(int))
    features = build_ou_features(
        p1, p2, surface, stats, h2h_surface, h2h_surface, 
        surface_length_stats, df
    )
    
    # Make prediction
    prob_over, prediction, confidence = predict_over_under(features, seed=hash(p1 + p2 + surface) % 10000)
    
    # Get surface stats
    surf_stats = surface_length_stats[surface]
    
    # Calculate competitiveness
    comp_data = build_competitiveness_features(p1, p2, stats, h2h_surface, surface)
    
    return {
        'player1': p1,
        'player2': p2,
        'surface': surface,
        'prediction': prediction,
        'confidence': f"{confidence:.1%}",
        'confidence_score': confidence,
        'prob_over': prob_over,
        'prob_under': 1 - prob_over,
        'surface_baseline': f"{surf_stats['p_over_21_5']:.1%}",
        'competitiveness': f"{comp_data['overall_competitiveness']:.1%}",
        'avg_games_surface': f"{surf_stats['avg_games']:.1f}",
        'p1_avg_games': f"{stats[p1]['avg_games']:.1f}",
        'p2_avg_games': f"{stats[p2]['avg_games']:.1f}",
        'p1_grind': f"{stats[p1]['grind_factor']:.2f}",
        'p2_grind': f"{stats[p2]['grind_factor']:.2f}",
        'tournament': match['tournament'],
        'tournament_type': match['tournament_type'],
        'round': match['round'],
        'time': match['time']
    }


# ==============================================================================
# EXCEL EXPORT FUNCTION
# ==============================================================================

def export_to_excel(predictions_df):
    """Export predictions to Excel file with formatting."""
    output = BytesIO()
    
    with pd.ExcelWriter(output, engine='openpyxl') as writer:
        # Write main predictions sheet
        predictions_df.to_excel(writer, sheet_name='Predictions', index=False)
        
        # Get workbook and worksheet
        workbook = writer.book
        worksheet = writer.sheets['Predictions']
        
        # Format headers
        from openpyxl.styles import Font, PatternFill, Alignment
        
        header_font = Font(bold=True, color="FFFFFF")
        header_fill = PatternFill(start_color="1f77b4", end_color="1f77b4", fill_type="solid")
        
        for cell in worksheet[1]:
            cell.font = header_font
            cell.fill = header_fill
            cell.alignment = Alignment(horizontal='center')
        
        # Adjust column widths
        for column in worksheet.columns:
            max_length = 0
            column_letter = column[0].column_letter
            for cell in column:
                try:
                    if len(str(cell.value)) > max_length:
                        max_length = len(str(cell.value))
                except:
                    pass
            adjusted_width = min(max_length + 2, 30)
            worksheet.column_dimensions[column_letter].width = adjusted_width
        
        # Add summary statistics sheet
        summary_data = {
            'Metric': ['Total Matches', 'OVER Predictions', 'UNDER Predictions', 
                      'Avg Confidence', 'High Confidence (>70%)', 'Medium Confidence (50-70%)',
                      'Low Confidence (<50%)'],
            'Value': [
                len(predictions_df),
                len(predictions_df[predictions_df['Prediction'] == 'OVER']),
                len(predictions_df[predictions_df['Prediction'] == 'UNDER']),
                f"{predictions_df['Confidence'].str.rstrip('%').astype(float).mean():.1f}%",
                len(predictions_df[predictions_df['Confidence'].str.rstrip('%').astype(float) > 70]),
                len(predictions_df[(predictions_df['Confidence'].str.rstrip('%').astype(float) >= 50) & 
                                  (predictions_df['Confidence'].str.rstrip('%').astype(float) <= 70)]),
                len(predictions_df[predictions_df['Confidence'].str.rstrip('%').astype(float) < 50])
            ]
        }
        summary_df = pd.DataFrame(summary_data)
        summary_df.to_excel(writer, sheet_name='Summary', index=False)
        
        # Surface breakdown sheet
        surface_breakdown = predictions_df.groupby('Surface').agg({
            'Prediction': lambda x: (x == 'OVER').sum(),
            'Confidence': lambda x: x.str.rstrip('%').astype(float).mean()
        }).round(2)
        surface_breakdown.columns = ['OVER_Count', 'Avg_Confidence_%']
        surface_breakdown.to_excel(writer, sheet_name='Surface Breakdown')
    
    output.seek(0)
    return output


# ==============================================================================
# MAIN APP
# ==============================================================================

def main():
    st.markdown('<h1 class="main-header">🎾 Tennis Over/Under 21.5 Predictor</h1>', unsafe_allow_html=True)
    
    # Sidebar
    with st.sidebar:
        st.markdown("## ⚙️ Settings")
        
        # Data source selection
        data_source = st.radio(
            "Data Source",
            ["Demo Data (Scheduled Matches)", "Manual Entry"],
            help="Choose between viewing scheduled matches or manually entering players"
        )
        
        st.markdown("---")
        st.markdown("### 📊 Model Information")
        st.info(
            "This model predicts whether a tennis match will go OVER or UNDER 21.5 total games.\n\n"
            "**Key Factors:**\n"
            "• Surface baseline statistics\n"
            "• Match competitiveness index\n"
            "• Player grind factors\n"
            "• Recent form trends\n"
            "• Serve/return dominance\n\n"
            "**Target Accuracy:** 65-70%"
        )
    
    # Generate demo data
    players_data, default_stats = generate_demo_data()
    surface_length_stats = compute_surface_length_stats(df=None)
    
    if data_source == "Demo Data (Scheduled Matches)":
        # Tab selection
        tab1, tab2, tab3 = st.tabs(["📅 Today's Matches", "📆 Tomorrow's Matches", "📊 All Predictions"])
        
        # Generate scheduled matches
        scheduled_matches = generate_scheduled_matches()
        
        # Separate today and tomorrow matches
        today_matches = [m for m in scheduled_matches if m['day'] == 'Today']
        tomorrow_matches = [m for m in scheduled_matches if m['day'] == 'Tomorrow']
        
        # Store predictions
        all_predictions = []
        
        # Process today's matches
        with tab1:
            st.markdown(f"### 🎾 Today's Matches - {datetime.now().strftime('%B %d, %Y')}")
            
            if st.button("🔄 Generate Predictions for Today", key="today_predict"):
                with st.spinner("Analyzing today's matches..."):
                    for match in today_matches:
                        prediction = predict_match(match, players_data, default_stats, surface_length_stats)
                        all_predictions.append(prediction)
                    
                    # Display predictions
                    df_today = pd.DataFrame(all_predictions[:len(today_matches)])
                    
                    # Add styling
                    def color_prediction(val):
                        if val == 'OVER':
                            return 'background-color: #ff6b6b; color: white'
                        return 'background-color: #51cf66; color: white'
                    
                    styled_df = df_today.style.applymap(
                        color_prediction, subset=['prediction']
                    )
                    
                    st.dataframe(
                        df_today[[
                            'time', 'player1', 'player2', 'surface', 'tournament', 
                            'round', 'prediction', 'confidence', 'competitiveness'
                        ]].rename(columns={
                            'player1': 'Player 1',
                            'player2': 'Player 2',
                            'surface': 'Surface',
                            'prediction': 'Prediction',
                            'confidence': 'Confidence',
                            'competitiveness': 'Competitiveness',
                            'tournament': 'Tournament',
                            'round': 'Round',
                            'time': 'Time'
                        }),
                        use_container_width=True,
                        hide_index=True
                    )
                    
                    # Store in session state
                    st.session_state['all_predictions'] = all_predictions
                    st.session_state['df_today'] = df_today
                    st.session_state['df_tomorrow'] = None
            else:
                st.info("Click 'Generate Predictions for Today' to see predictions")
        
        # Process tomorrow's matches
        with tab2:
            st.markdown(f"### 🎾 Tomorrow's Matches - {(datetime.now() + timedelta(days=1)).strftime('%B %d, %Y')}")
            
            if st.button("🔄 Generate Predictions for Tomorrow", key="tomorrow_predict"):
                with st.spinner("Analyzing tomorrow's matches..."):
                    tomorrow_predictions = []
                    for match in tomorrow_matches:
                        prediction = predict_match(match, players_data, default_stats, surface_length_stats)
                        tomorrow_predictions.append(prediction)
                    
                    # Display predictions
                    df_tomorrow = pd.DataFrame(tomorrow_predictions)
                    
                    styled_df = df_tomorrow.style.applymap(
                        lambda x: 'background-color: #ff6b6b; color: white' if x == 'OVER' 
                        else ('background-color: #51cf66; color: white' if x == 'UNDER' else ''),
                        subset=['prediction']
                    )
                    
                    st.dataframe(
                        df_tomorrow[[
                            'time', 'player1', 'player2', 'surface', 'tournament', 
                            'round', 'prediction', 'confidence', 'competitiveness'
                        ]].rename(columns={
                            'player1': 'Player 1',
                            'player2': 'Player 2',
                            'surface': 'Surface',
                            'prediction': 'Prediction',
                            'confidence': 'Confidence',
                            'competitiveness': 'Competitiveness',
                            'tournament': 'Tournament',
                            'round': 'Round',
                            'time': 'Time'
                        }),
                        use_container_width=True,
                        hide_index=True
                    )
                    
                    # Update session state
                    if 'all_predictions' in st.session_state:
                        all_predictions = st.session_state['all_predictions'] + tomorrow_predictions
                    else:
                        all_predictions = tomorrow_predictions
                    
                    st.session_state['all_predictions'] = all_predictions
                    st.session_state['df_tomorrow'] = df_tomorrow
            else:
                st.info("Click 'Generate Predictions for Tomorrow' to see predictions")
        
        # All predictions combined
        with tab3:
            st.markdown("### 📊 All Match Predictions")
            
            if 'all_predictions' in st.session_state and st.session_state['all_predictions']:
                df_all = pd.DataFrame(st.session_state['all_predictions'])
                
                # Add day column based on time
                df_all['Day'] = df_all.apply(
                    lambda x: 'Today' if x['time'] in [m['time'] for m in today_matches] else 'Tomorrow', 
                    axis=1
                )
                
                # Summary metrics
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("Total Matches", len(df_all))
                with col2:
                    over_count = len(df_all[df_all['prediction'] == 'OVER'])
                    st.metric("OVER Predictions", over_count, 
                             delta=f"{over_count/len(df_all)*100:.0f}%")
                with col3:
                    under_count = len(df_all[df_all['prediction'] == 'UNDER'])
                    st.metric("UNDER Predictions", under_count,
                             delta=f"{under_count/len(df_all)*100:.0f}%")
                with col4:
                    avg_conf = df_all['confidence_score'].mean()
                    st.metric("Avg Confidence", f"{avg_conf:.1%}")
                
                # Display all predictions
                display_df = df_all[[
                    'Day', 'time', 'player1', 'player2', 'surface', 'tournament',
                    'tournament_type', 'round', 'prediction', 'confidence', 'competitiveness',
                    'prob_over', 'prob_under'
                ]].rename(columns={
                    'player1': 'Player 1',
                    'player2': 'Player 2',
                    'surface': 'Surface',
                    'prediction': 'Prediction',
                    'confidence': 'Confidence',
                    'competitiveness': 'Competitiveness',
                    'tournament': 'Tournament',
                    'tournament_type': 'Tournament Type',
                    'round': 'Round',
                    'time': 'Time',
                    'Day': 'Day',
                    'prob_over': 'OVER Probability',
                    'prob_under': 'UNDER Probability'
                })
                
                # Format probability columns
                display_df['OVER Probability'] = display_df['OVER Probability'].apply(lambda x: f"{x:.1%}")
                display_df['UNDER Probability'] = display_df['UNDER Probability'].apply(lambda x: f"{x:.1%}")
                
                st.dataframe(display_df, use_container_width=True, hide_index=True)
                
                # Export button
                col1, col2, col3 = st.columns([1, 2, 1])
                with col2:
                    if st.button("📥 Export to Excel", use_container_width=True):
                        excel_file = export_to_excel(display_df)
                        st.download_button(
                            label="📊 Download Excel File",
                            data=excel_file,
                            file_name=f"tennis_predictions_{datetime.now().strftime('%Y%m%d_%H%M')}.xlsx",
                            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                            use_container_width=True
                        )
            else:
                st.info("Generate predictions for Today or Tomorrow first to see combined results")
    
    else:  # Manual Entry
        st.markdown("### 🎾 Manual Match Prediction")
        
        col1, col2 = st.columns(2)
        
        with col1:
            player1 = st.selectbox("Player 1", sorted(players_data.keys()), key="manual_p1")
            surface = st.selectbox("Surface", ["Hard", "Clay", "Grass"], key="manual_surface")
            tournament = st.text_input("Tournament", "ATP Tour")
            match_round = st.selectbox("Round", ["R1", "R2", "R3", "QF", "SF", "F"])
        
        with col2:
            player2 = st.selectbox("Player 2", sorted(players_data.keys()), key="manual_p2")
            tournament_type = st.selectbox("Tournament Type", ["Grand Slam", "ATP Masters 1000", "ATP 500", "ATP 250"])
            time = st.time_input("Match Time", datetime.now().time())
        
        if player1 == player2:
            st.warning("⚠️ Please select two different players")
            return
        
        if st.button("🔮 Predict Match", type="primary", use_container_width=True):
            with st.spinner("Analyzing match factors..."):
                match = {
                    'player1': player1,
                    'player2': player2,
                    'surface': surface,
                    'tournament': tournament,
                    'tournament_type': tournament_type,
                    'round': match_round,
                    'time': time.strftime("%H:%M"),
                    'date': datetime.now().strftime('%Y-%m-%d'),
                    'day': 'Today'
                }
                
                prediction = predict_match(match, players_data, default_stats, surface_length_stats)
                
                # Display results
                st.markdown("---")
                
                col1, col2, col3 = st.columns([2, 1, 2])
                
                with col1:
                    st.markdown(f"### 🎾 {player1}")
                    st.metric("Surface ELO", f"{players_data[player1]['surface_elo'][surface]:.0f}")
                    st.metric("Surface Win Rate", f"{players_data[player1]['surface_win_rate'][surface]:.1%}")
                    st.metric("Avg Games/Match", f"{players_data[player1]['avg_games']:.1f}")
                    st.metric("Grind Factor", f"{players_data[player1]['grind_factor']:.2f}")
                
                with col2:
                    st.markdown("### 📊 Prediction")
                    
                    if prediction['prediction'] == "OVER":
                        st.markdown(f'<div class="prediction-over"><h2>OVER 21.5</h2><p>Probability: {prediction["prob_over"]:.1%}</p><p>Confidence: {prediction["confidence"]}</p></div>', unsafe_allow_html=True)
                    else:
                        st.markdown(f'<div class="prediction-under"><h2>UNDER 21.5</h2><p>Probability: {prediction["prob_under"]:.1%}</p><p>Confidence: {prediction["confidence"]}</p></div>', unsafe_allow_html=True)
                    
                    st.metric("Surface Baseline", prediction['surface_baseline'])
                    st.metric("Match Competitiveness", prediction['competitiveness'])
                
                with col3:
                    st.markdown(f"### 🎾 {player2}")
                    st.metric("Surface ELO", f"{players_data[player2]['surface_elo'][surface]:.0f}")
                    st.metric("Surface Win Rate", f"{players_data[player2]['surface_win_rate'][surface]:.1%}")
                    st.metric("Avg Games/Match", f"{players_data[player2]['avg_games']:.1f}")
                    st.metric("Grind Factor", f"{players_data[player2]['grind_factor']:.2f}")
                
                # Export single prediction
                single_pred_df = pd.DataFrame([prediction])
                single_pred_df = single_pred_df[[
                    'player1', 'player2', 'surface', 'tournament', 'tournament_type',
                    'round', 'prediction', 'confidence', 'competitiveness',
                    'prob_over', 'prob_under'
                ]].rename(columns={
                    'player1': 'Player 1',
                    'player2': 'Player 2',
                    'surface': 'Surface',
                    'prediction': 'Prediction',
                    'confidence': 'Confidence',
                    'competitiveness': 'Competitiveness',
                    'tournament': 'Tournament',
                    'tournament_type': 'Tournament Type',
                    'round': 'Round',
                    'prob_over': 'OVER Probability',
                    'prob_under': 'UNDER Probability'
                })
                
                single_pred_df['OVER Probability'] = single_pred_df['OVER Probability'].apply(lambda x: f"{x:.1%}")
                single_pred_df['UNDER Probability'] = single_pred_df['UNDER Probability'].apply(lambda x: f"{x:.1%}")
                
                if st.button("📥 Export Prediction to Excel"):
                    excel_file = export_to_excel(single_pred_df)
                    st.download_button(
                        label="📊 Download Excel File",
                        data=excel_file,
                        file_name=f"tennis_prediction_{player1}_vs_{player2}_{datetime.now().strftime('%Y%m%d_%H%M')}.xlsx",
                        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                    )
    
    # Footer
    st.markdown("---")
    st.markdown(
        "<p style='text-align: center; color: gray;'>🎾 Tennis O/U 21.5 Predictor | Model Accuracy Target: 65-70% | Data updates daily</p>",
        unsafe_allow_html=True
    )


if __name__ == "__main__":
    main()

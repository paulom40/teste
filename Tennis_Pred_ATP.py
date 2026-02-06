import streamlit as st
import pandas as pd
import numpy as np
import math
import warnings
from collections import defaultdict, deque
from datetime import datetime, timedelta
import plotly.graph_objects as go
import plotly.express as px

warnings.filterwarnings('ignore')

try:
    from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix, classification_report
    from sklearn.preprocessing import RobustScaler
    from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier, AdaBoostClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.calibration import CalibratedClassifierCV
    from sklearn.feature_selection import SelectKBest, f_classif
    ML_AVAILABLE = True
except ImportError:
    ML_AVAILABLE = False

st.set_page_config(page_title="Tennis Prediction Pro", page_icon="🎾", layout="wide")

# Session State
for key in ['elo_ratings', 'player_names', 'global_elo', 'match_data', 
            'player_form_history', 'player_ids', 'match_history', 
            'surface_performance', 'h2h_stats', 'tourney_stats']:
    if key not in st.session_state:
        st.session_state[key] = {}

if 'scaler' not in st.session_state:
    st.session_state.scaler = RobustScaler()
if 'ensemble_model' not in st.session_state:
    st.session_state.ensemble_model = None
if 'model_metrics' not in st.session_state:
    st.session_state.model_metrics = {}
if 'feature_importance' not in st.session_state:
    st.session_state.feature_importance = {}

SURFACE_TYPES = ['Hard', 'Clay', 'Grass', 'Carpet']
TOURNEY_LEVELS = ['Grand Slam', 'Masters 1000', 'ATP 500', 'ATP 250', 'Challenger', 'Futures']

def get_default_form():
    """Default form dict with more metrics"""
    return {
        'wins': 0, 'matches': 0, 'win_pct': 0.5, 'momentum': 0.5,
        'opp_elo': 1500, 'streak': 0, 'consistency': 0.5, 'fatigue': 0,
        'top10_pct': 0.5, 'surface_pct': 0.5, 'sets_won_pct': 0.5,
        'clutch_performance': 0.5, 'recent_5': 0.5, 'recent_10': 0.5, 'recent_20': 0.5
    }

def compute_elo(df, k_factor_base=32, k_factor_top=20, initial_elo=1500):
    """Enhanced ELO computation with surface-specific adjustments"""
    df = df.copy()
    df['Player_1'] = df['Player_1'].astype(str).str.strip()
    df['Player_2'] = df['Player_2'].astype(str).str.strip()
    df['Winner'] = df['Winner'].astype(str).str.strip()
    df['Surface'] = df['Surface'].astype(str).str.strip()
    
    # Check for tournament level column
    if 'Tournament_Level' in df.columns:
        df['Tournament_Level'] = df['Tournament_Level'].astype(str).str.strip()
    else:
        df['Tournament_Level'] = 'ATP 250'
    
    # Check for sets/games data
    if 'Player_1_Sets' in df.columns and 'Player_2_Sets' in df.columns:
        has_set_data = True
    else:
        has_set_data = False
        df['Player_1_Sets'] = 2
        df['Player_2_Sets'] = 1
    
    # Get all unique players
    all_players = set()
    all_players.update(df['Player_1'].unique())
    all_players.update(df['Player_2'].unique())
    
    # Create ID mapping
    player_ids = {name: f"P{idx:04d}" for idx, name in enumerate(sorted(all_players))}
    
    st.session_state.player_ids = player_ids
    st.session_state.player_names = {v: k for k, v in player_ids.items()}
    
    # Initialize structures
    elo_ratings = {pid: {s: initial_elo for s in SURFACE_TYPES} for pid in player_ids.values()}
    global_ratings = {pid: initial_elo for pid in player_ids.values()}
    form_history = defaultdict(lambda: defaultdict(deque))
    match_history = defaultdict(list)
    surface_perf = defaultdict(lambda: defaultdict(list))
    h2h_stats = {}
    tourney_performance = defaultdict(lambda: defaultdict(lambda: {'wins': 0, 'matches': 0}))
    
    # Set bonus for tournament level
    tourney_bonus = {
        'Grand Slam': 1.5,
        'Masters 1000': 1.3,
        'ATP 500': 1.2,
        'ATP 250': 1.0,
        'Challenger': 0.8,
        'Futures': 0.6
    }
    
    if 'Date' in df.columns:
        df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
        df['Year'] = df['Date'].dt.year
        df['Month'] = df['Date'].dt.month
    else:
        df['Date'] = datetime.now()
        df['Year'] = datetime.now().year
        df['Month'] = datetime.now().month
    
    # Sort by date if available
    if 'Date' in df.columns:
        df = df.sort_values('Date')
    
    valid_count = 0
    
    for idx, row in df.iterrows():
        p1 = row['Player_1']
        p2 = row['Player_2']
        winner = row['Winner']
        surface = row['Surface'] if row['Surface'] in SURFACE_TYPES else 'Hard'
        tourney_level = row['Tournament_Level']
        
        # Skip if winner is not one of the players
        if winner not in [p1, p2]:
            continue
        
        valid_count += 1
        
        winner_id = player_ids[winner]
        loser = p2 if winner == p1 else p1
        loser_id = player_ids[loser]
        
        # Get current ratings
        rating_w = elo_ratings[winner_id][surface]
        rating_l = elo_ratings[loser_id][surface]
        
        # Dynamic K-factor
        w_matches = len([m for m in match_history.get(winner_id, []) if m.get('surface') == surface])
        l_matches = len([m for m in match_history.get(loser_id, []) if m.get('surface') == surface])
        
        k_w = max(k_factor_top, k_factor_base / (1 + w_matches / 100))
        k_l = max(k_factor_top, k_factor_base / (1 + l_matches / 100))
        
        # Tournament importance multiplier
        t_bonus = tourney_bonus.get(tourney_level, 1.0)
        
        # H2H adjustment
        h2h_key = tuple(sorted([p1, p2]))
        if h2h_key not in h2h_stats:
            # First player in sorted tuple is player1 in H2H stats
            h2h_stats[h2h_key] = {'player1_wins': 0, 'matches': 0, 'sets_won': 0, 'sets_lost': 0}
        
        h2h_info = h2h_stats[h2h_key]
        
        # Update H2H stats
        h2h_info['matches'] += 1
        
        # Determine which player is first in the sorted tuple
        first_player = h2h_key[0]
        if winner == first_player:
            h2h_info['player1_wins'] += 1
        
        if has_set_data:
            if winner == p1:
                h2h_info['sets_won'] += row.get('Player_1_Sets', 2)
                h2h_info['sets_lost'] += row.get('Player_2_Sets', 1)
            else:
                h2h_info['sets_won'] += row.get('Player_2_Sets', 2)
                h2h_info['sets_lost'] += row.get('Player_1_Sets', 1)
        
        # H2H factor calculation
        if h2h_info['matches'] > 0:
            # Determine if winner is the first player in H2H stats
            winner_is_first = (winner == first_player)
            h2h_win_rate = h2h_info['player1_wins'] / h2h_info['matches']
            
            if winner_is_first:
                h2h_advantage = h2h_win_rate - 0.5
            else:
                h2h_advantage = (1 - h2h_win_rate) - 0.5
            
            h2h_factor = 1 + (h2h_advantage * 0.2)
        else:
            h2h_factor = 1.0
        
        # Expected win probability
        exp_w = 1 / (1 + math.pow(10, (rating_l - rating_w) / 400))
        
        # Score margin adjustment
        if has_set_data:
            if winner == p1:
                sets_won = row.get('Player_1_Sets', 2)
                sets_lost = row.get('Player_2_Sets', 1)
            else:
                sets_won = row.get('Player_2_Sets', 2)
                sets_lost = row.get('Player_1_Sets', 1)
            
            margin = (sets_won - sets_lost) / (sets_won + sets_lost)
            margin_factor = 1 + (margin * 0.3)
        else:
            margin_factor = 1.0
            sets_won = 2
            sets_lost = 1
        
        # Calculate rating changes
        elo_change_winner = k_w * t_bonus * h2h_factor * margin_factor * (1 - exp_w)
        elo_change_loser = k_l * t_bonus * (0 - (1 - exp_w))
        
        # Apply changes
        elo_ratings[winner_id][surface] = rating_w + elo_change_winner
        elo_ratings[loser_id][surface] = rating_l + elo_change_loser
        
        # Global rating (weighted average)
        surface_weights = {'Hard': 0.35, 'Clay': 0.30, 'Grass': 0.20, 'Carpet': 0.15}
        global_ratings[winner_id] = sum(
            elo_ratings[winner_id].get(s, initial_elo) * surface_weights.get(s, 0.1) 
            for s in SURFACE_TYPES
        )
        global_ratings[loser_id] = sum(
            elo_ratings[loser_id].get(s, initial_elo) * surface_weights.get(s, 0.1) 
            for s in SURFACE_TYPES
        )
        
        # Update tournament performance
        tourney_performance[winner_id][tourney_level]['wins'] += 1
        tourney_performance[winner_id][tourney_level]['matches'] += 1
        tourney_performance[loser_id][tourney_level]['matches'] += 1
        
        # Store match record
        match_record = {
            'date': row['Date'],
            'surface': surface,
            'opponent': loser_id,
            'w_elo': rating_w,
            'l_elo': rating_l,
            'won': True,
            'tournament_level': tourney_level,
            'sets_won': sets_won,
            'sets_lost': sets_lost,
            'year': row.get('Year', datetime.now().year),
            'month': row.get('Month', datetime.now().month)
        }
        
        match_history[winner_id].append(match_record)
        match_history[loser_id].append({
            **match_record,
            'won': False,
            'opponent': winner_id,
            'sets_won': sets_lost,
            'sets_lost': sets_won
        })
        
        # Update surface performance
        surface_perf[winner_id][surface].append(True)
        surface_perf[loser_id][surface].append(False)
        
        # Update form history
        form_history[winner_id][surface].append({**match_record, 'won': True})
        form_history[loser_id][surface].append({
            **{k: v for k, v in match_record.items() if k != 'won'},
            'won': False
        })
        
        # Keep last 100 matches per surface for form
        if len(form_history[winner_id][surface]) > 100:
            form_history[winner_id][surface].popleft()
        if len(form_history[loser_id][surface]) > 100:
            form_history[loser_id][surface].popleft()
    
    st.session_state.player_form_history = form_history
    st.session_state.match_history = match_history
    st.session_state.surface_performance = surface_perf
    st.session_state.h2h_stats = h2h_stats
    st.session_state.tourney_stats = tourney_performance
    
    return elo_ratings, global_ratings, valid_count

def calc_form(player_id, surface, form_history, match_history, current_elo, lookback_days=365):
    """Enhanced form calculation with multiple time windows"""
    if player_id not in form_history:
        return get_default_form()
    
    now = datetime.now()
    recent_matches = [
        m for m in match_history.get(player_id, [])
        if (now - m['date']).days <= lookback_days and m.get('surface') == surface
    ]
    
    if not recent_matches:
        return get_default_form()
    
    # Basic stats
    wins = sum(1 for m in recent_matches if m.get('won', False))
    total = len(recent_matches)
    
    # Multiple time windows
    recent_5 = recent_matches[-5:] if len(recent_matches) >= 5 else recent_matches
    recent_10 = recent_matches[-10:] if len(recent_matches) >= 10 else recent_matches
    recent_20 = recent_matches[-20:] if len(recent_matches) >= 20 else recent_matches
    
    # Win percentages for different windows
    win_pct_5 = sum(1 for m in recent_5 if m.get('won', False)) / max(len(recent_5), 1)
    win_pct_10 = sum(1 for m in recent_10 if m.get('won', False)) / max(len(recent_10), 1)
    win_pct_20 = sum(1 for m in recent_20 if m.get('won', False)) / max(len(recent_20), 1)
    
    # Momentum (weighted recent form)
    momentum = 0
    weights = [0.3, 0.25, 0.2, 0.15, 0.1]
    for i, m in enumerate(recent_matches[-5:]):
        if i < len(weights):
            momentum += (1 if m.get('won', False) else 0) * weights[i]
    
    # Streak calculation
    streak = 0
    for m in reversed(recent_matches):
        if m.get('won', False):
            streak += 1
        else:
            break
    if streak == 0:
        for m in reversed(recent_matches):
            if not m.get('won', False):
                streak -= 1
            else:
                break
    
    # Opponent quality
    opp_elos = []
    for m in recent_matches:
        if m.get('won', False):
            opp_elo = m.get('l_elo', 1500)
        else:
            opp_elo = m.get('w_elo', 1500)
        opp_elos.append(opp_elo)
    avg_opp_elo = np.mean(opp_elos) if opp_elos else 1500
    
    # Performance against top players
    top10_wins = 0
    top10_matches = 0
    for m in recent_matches:
        if m.get('won', False):
            opp_elo = m.get('l_elo', 1500)
        else:
            opp_elo = m.get('w_elo', 1500)
        
        if opp_elo > 1800:
            top10_matches += 1
            if m.get('won', False):
                top10_wins += 1
    
    top10_pct = top10_wins / max(top10_matches, 1)
    
    # Consistency
    performances = []
    for m in recent_matches:
        player_elo = m.get('w_elo', 1500) if m.get('won', False) else m.get('l_elo', 1500)
        opp_elo = m.get('l_elo', 1500) if m.get('won', False) else m.get('w_elo', 1500)
        expected = 1 / (1 + math.pow(10, (opp_elo - player_elo) / 400))
        actual = 1 if m.get('won', False) else 0
        performances.append(actual - expected)
    
    consistency = 1 - np.std(performances) if performances and len(performances) > 1 else 0.5
    
    # Surface performance
    surf_results = [m.get('won', False) for m in recent_matches if m.get('surface') == surface]
    surface_pct = sum(surf_results) / max(len(surf_results), 1)
    
    # Fatigue
    fatigue = sum(1 for m in match_history.get(player_id, [])
                 if (now - m['date']).days <= 30)
    
    # Set performance
    sets_won = sum(m.get('sets_won', 0) for m in recent_matches)
    sets_total = sum(m.get('sets_won', 0) + m.get('sets_lost', 0) for m in recent_matches)
    sets_pct = sets_won / max(sets_total, 1)
    
    # Clutch performance
    close_matches = [m for m in recent_matches if abs(m.get('sets_won', 0) - m.get('sets_lost', 0)) <= 1]
    clutch_wins = sum(1 for m in close_matches if m.get('won', False))
    clutch_pct = clutch_wins / max(len(close_matches), 1)
    
    return {
        'wins': wins,
        'matches': total,
        'win_pct': wins / max(total, 1),
        'momentum': momentum,
        'opp_elo': avg_opp_elo,
        'streak': streak,
        'consistency': consistency,
        'fatigue': min(fatigue / 10, 1),
        'top10_pct': top10_pct,
        'surface_pct': surface_pct,
        'sets_won_pct': sets_pct,
        'clutch_performance': clutch_pct,
        'recent_5': win_pct_5,
        'recent_10': win_pct_10,
        'recent_20': win_pct_20,
        'performance_std': np.std(performances) if performances and len(performances) > 1 else 0
    }

def create_advanced_features(df, elo_ratings, global_ratings, form_history, match_history):
    """Create advanced feature set - FIXED to include both winner and loser perspectives"""
    df = df.copy()
    df['Player_1'] = df['Player_1'].astype(str).str.strip()
    df['Player_2'] = df['Player_2'].astype(str).str.strip()
    df['Winner'] = df['Winner'].astype(str).str.strip()
    df['Surface'] = df['Surface'].astype(str).str.strip()
    
    player_ids = st.session_state.player_ids
    h2h_stats = st.session_state.h2h_stats
    
    features_list = []
    labels = []
    
    for idx, row in df.iterrows():
        p1 = row['Player_1']
        p2 = row['Player_2']
        winner = row['Winner']
        surface = row['Surface'] if row['Surface'] in SURFACE_TYPES else 'Hard'
        
        if winner not in [p1, p2]:
            continue
        
        # Get IDs
        p1_id = player_ids.get(p1)
        p2_id = player_ids.get(p2)
        
        if p1_id is None or p2_id is None:
            continue
        
        # Get ratings
        p1_elo = elo_ratings.get(p1_id, {}).get(surface, global_ratings.get(p1_id, 1500))
        p2_elo = elo_ratings.get(p2_id, {}).get(surface, global_ratings.get(p2_id, 1500))
        p1_global = global_ratings.get(p1_id, 1500)
        p2_global = global_ratings.get(p2_id, 1500)
        
        # Get form
        p1_form = calc_form(p1_id, surface, form_history, match_history, p1_elo)
        p2_form = calc_form(p2_id, surface, form_history, match_history, p2_elo)
        
        # H2H stats
        h2h_key = tuple(sorted([p1, p2]))
        h2h_info = h2h_stats.get(h2h_key, {'player1_wins': 0, 'matches': 0, 'sets_won': 0, 'sets_lost': 0})
        h2h_matches = h2h_info['matches']
        
        # Determine H2H win percentage for p1
        first_player = h2h_key[0]
        if h2h_matches > 0:
            if p1 == first_player:
                h2h_pct_p1 = h2h_info['player1_wins'] / h2h_matches
            else:
                h2h_pct_p1 = 1 - (h2h_info['player1_wins'] / h2h_matches)
        else:
            h2h_pct_p1 = 0.5
        
        # Create features from Player 1's perspective (regardless of who won)
        features_p1 = {
            # Basic ELO features
            'elo_diff': float(p1_elo - p2_elo),
            'elo_ratio': float(p1_elo / max(p2_elo, 1)),
            'global_elo_diff': float(p1_global - p2_global),
            
            # Surface-specific ELO
            'surface_elo_diff': float(p1_elo - p2_elo),
            'surface_elo_ratio': float(p1_elo / max(p2_elo, 1)),
            
            # Form features
            'win_pct_diff': float(p1_form['win_pct'] - p2_form['win_pct']),
            'momentum_diff': float(p1_form['momentum'] - p2_form['momentum']),
            'streak_diff': float(p1_form['streak'] - p2_form['streak']),
            'consistency_diff': float(p1_form['consistency'] - p2_form['consistency']),
            'fatigue_diff': float(p1_form['fatigue'] - p2_form['fatigue']),
            'top10_pct_diff': float(p1_form['top10_pct'] - p2_form['top10_pct']),
            'surface_pct_diff': float(p1_form['surface_pct'] - p2_form['surface_pct']),
            
            # Recent form
            'recent_5_diff': float(p1_form['recent_5'] - p2_form['recent_5']),
            'recent_10_diff': float(p1_form['recent_10'] - p2_form['recent_10']),
            'recent_20_diff': float(p1_form['recent_20'] - p2_form['recent_20']),
            
            # Individual features
            'p1_win_pct': float(p1_form['win_pct']),
            'p1_momentum': float(p1_form['momentum']),
            'p1_streak': float(p1_form['streak']),
            'p1_consistency': float(p1_form['consistency']),
            'p1_fatigue': float(p1_form['fatigue']),
            'p1_top10_pct': float(p1_form['top10_pct']),
            'p1_surface_pct': float(p1_form['surface_pct']),
            'p1_clutch': float(p1_form['clutch_performance']),
            'p1_sets_pct': float(p1_form['sets_won_pct']),
            
            'p2_win_pct': float(p2_form['win_pct']),
            'p2_momentum': float(p2_form['momentum']),
            'p2_streak': float(p2_form['streak']),
            'p2_consistency': float(p2_form['consistency']),
            'p2_fatigue': float(p2_form['fatigue']),
            'p2_top10_pct': float(p2_form['top10_pct']),
            'p2_surface_pct': float(p2_form['surface_pct']),
            'p2_clutch': float(p2_form['clutch_performance']),
            'p2_sets_pct': float(p2_form['sets_won_pct']),
            
            # H2H features
            'h2h_win_pct': float(h2h_pct_p1),
            'h2h_matches': float(h2h_matches),
            'h2h_sets_ratio': float(h2h_info['sets_won'] / max(h2h_info['sets_lost'], 1)) if p1 == first_player else float(h2h_info['sets_lost'] / max(h2h_info['sets_won'], 1)),
            
            # Surface indicators
            'is_hard': 1 if surface == 'Hard' else 0,
            'is_clay': 1 if surface == 'Clay' else 0,
            'is_grass': 1 if surface == 'Grass' else 0,
            
            # Interaction features
            'elo_form_interaction': float((p1_elo - p2_elo) * (p1_form['win_pct'] - p2_form['win_pct'])),
            'momentum_elo_interaction': float((p1_form['momentum'] - p2_form['momentum']) * (p1_elo - p2_elo)),
            'surface_form_interaction': float(p1_form['surface_pct'] * (1 if surface == 'Hard' else 0.5)),
            
            # Derived metrics
            'experience_diff': float(len(match_history.get(p1_id, [])) - len(match_history.get(p2_id, []))),
            'upset_potential': float(1 if p2_elo > p1_elo else 0),
            'form_consistency': float(min(p1_form['consistency'], p2_form['consistency'])),
        }
        
        # Add to features list
        features_list.append(features_p1)
        
        # Label: 1 if Player 1 won, 0 if Player 2 won
        labels.append(1 if winner == p1 else 0)
        
        # Also add the reverse perspective for better training
        # Create features from Player 2's perspective
        features_p2 = {}
        for key, value in features_p1.items():
            if key.endswith('_diff') or key in ['elo_diff', 'global_elo_diff', 'surface_elo_diff', 
                                              'elo_ratio', 'surface_elo_ratio', 'h2h_win_pct',
                                              'elo_form_interaction', 'momentum_elo_interaction',
                                              'experience_diff', 'upset_potential']:
                # Reverse the differences
                features_p2[key] = -value
            elif key.startswith('p1_'):
                # Swap p1 and p2
                new_key = key.replace('p1_', 'p2_')
                features_p2[new_key] = value
            elif key.startswith('p2_'):
                # Swap p2 and p1
                new_key = key.replace('p2_', 'p1_')
                features_p2[new_key] = value
            else:
                # Keep surface indicators and other features the same
                features_p2[key] = value
        
        # Adjust H2H for Player 2's perspective
        features_p2['h2h_win_pct'] = 1 - h2h_pct_p1
        features_p2['h2h_sets_ratio'] = 1 / max(features_p1['h2h_sets_ratio'], 0.01)
        
        # Add Player 2's perspective
        features_list.append(features_p2)
        labels.append(1 if winner == p2 else 0)  # Label from P2's perspective
    
    return pd.DataFrame(features_list), np.array(labels)

def train_advanced_model(features_df, labels):
    """Train advanced ensemble model"""
    if len(np.unique(labels)) < 2:
        raise ValueError("Need both win and loss examples")
    
    # Feature selection
    if len(features_df.columns) > 30:
        selector = SelectKBest(f_classif, k=min(30, len(features_df.columns)))
        X_selected = selector.fit_transform(features_df, labels)
        selected_features = features_df.columns[selector.get_support()]
        features_df = pd.DataFrame(X_selected, columns=selected_features)
        st.session_state.selected_features = selected_features.tolist()
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        features_df, labels, test_size=0.2, random_state=42, stratify=labels
    )
    
    # Scale features
    scaler = RobustScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    st.session_state.scaler = scaler
    
    # Define base models
    models = {
        'rf': RandomForestClassifier(
            n_estimators=200,
            max_depth=10,
            min_samples_split=5,
            min_samples_leaf=2,
            random_state=42,
            n_jobs=-1,
            class_weight='balanced'
        ),
        'gb': GradientBoostingClassifier(
            n_estimators=150,
            max_depth=5,
            learning_rate=0.05,
            subsample=0.8,
            random_state=42
        ),
        'ada': AdaBoostClassifier(
            n_estimators=100,
            random_state=42
        ),
        'lr': LogisticRegression(
            C=0.5,
            random_state=42,
            max_iter=1000,
            n_jobs=-1,
            class_weight='balanced'
        )
    }
    
    # Train individual models
    trained_models = {}
    cv_scores = {}
    
    for name, model in models.items():
        try:
            # Cross-validation
            cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
            scores = cross_val_score(model, X_train_scaled, y_train, 
                                    cv=cv, scoring='roc_auc', n_jobs=-1)
            cv_scores[name] = scores.mean()
            
            # Train on full training set
            model.fit(X_train_scaled, y_train)
            trained_models[name] = model
            
            st.write(f"✅ {name.upper()} trained - CV AUC: {scores.mean():.3f} (±{scores.std():.3f})")
        except Exception as e:
            st.write(f"⚠️ Could not train {name}: {str(e)}")
    
    if not trained_models:
        raise ValueError("Could not train any models")
    
    # Create voting ensemble
    ensemble = VotingClassifier(
        estimators=list(trained_models.items()),
        voting='soft',
        n_jobs=-1
    )
    
    # Calibrate the ensemble
    calibrated_model = CalibratedClassifierCV(
        ensemble,
        method='sigmoid',
        cv=3,
        n_jobs=-1
    )
    
    calibrated_model.fit(X_train_scaled, y_train)
    
    # Evaluate
    y_pred = calibrated_model.predict(X_test_scaled)
    y_pred_proba = calibrated_model.predict_proba(X_test_scaled)[:, 1]
    
    # Calculate metrics
    metrics = {
        'accuracy': float(accuracy_score(y_test, y_pred)),
        'precision': float(precision_score(y_test, y_pred, zero_division=0)),
        'recall': float(recall_score(y_test, y_pred, zero_division=0)),
        'f1': float(f1_score(y_test, y_pred, zero_division=0)),
        'roc_auc': float(roc_auc_score(y_test, y_pred_proba)),
        'cv_scores': cv_scores,
        'confusion_matrix': confusion_matrix(y_test, y_pred).tolist(),
        'classification_report': classification_report(y_test, y_pred, output_dict=True)
    }
    
    # Feature importance
    if 'rf' in trained_models:
        importances = trained_models['rf'].feature_importances_
        feature_importance = dict(zip(features_df.columns, importances))
        st.session_state.feature_importance = feature_importance
    
    return calibrated_model, metrics

def predict_match_advanced(p1_id, p2_id, surface, elo_ratings, global_ratings, form_history, match_history):
    """Advanced prediction"""
    if not st.session_state.ensemble_model:
        return None
    
    # Get ratings
    p1_elo = elo_ratings.get(p1_id, {}).get(surface, global_ratings.get(p1_id, 1500))
    p2_elo = elo_ratings.get(p2_id, {}).get(surface, global_ratings.get(p2_id, 1500))
    p1_global = global_ratings.get(p1_id, 1500)
    p2_global = global_ratings.get(p2_id, 1500)
    
    # Get form
    p1_form = calc_form(p1_id, surface, form_history, match_history, p1_elo)
    p2_form = calc_form(p2_id, surface, form_history, match_history, p2_elo)
    
    # H2H stats
    p1_name = st.session_state.player_names.get(p1_id, "Player 1")
    p2_name = st.session_state.player_names.get(p2_id, "Player 2")
    h2h_key = tuple(sorted([p1_name, p2_name]))
    h2h_info = st.session_state.h2h_stats.get(h2h_key, {'player1_wins': 0, 'matches': 0, 'sets_won': 0, 'sets_lost': 0})
    h2h_matches = h2h_info['matches']
    
    # Determine H2H win percentage for p1
    first_player = h2h_key[0]
    if h2h_matches > 0:
        if p1_name == first_player:
            h2h_pct_p1 = h2h_info['player1_wins'] / h2h_matches
        else:
            h2h_pct_p1 = 1 - (h2h_info['player1_wins'] / h2h_matches)
    else:
        h2h_pct_p1 = 0.5
    
    # Create features from Player 1's perspective
    features_dict = {
        'elo_diff': float(p1_elo - p2_elo),
        'elo_ratio': float(p1_elo / max(p2_elo, 1)),
        'global_elo_diff': float(p1_global - p2_global),
        'surface_elo_diff': float(p1_elo - p2_elo),
        'surface_elo_ratio': float(p1_elo / max(p2_elo, 1)),
        
        'win_pct_diff': float(p1_form['win_pct'] - p2_form['win_pct']),
        'momentum_diff': float(p1_form['momentum'] - p2_form['momentum']),
        'streak_diff': float(p1_form['streak'] - p2_form['streak']),
        'consistency_diff': float(p1_form['consistency'] - p2_form['consistency']),
        'fatigue_diff': float(p1_form['fatigue'] - p2_form['fatigue']),
        'top10_pct_diff': float(p1_form['top10_pct'] - p2_form['top10_pct']),
        'surface_pct_diff': float(p1_form['surface_pct'] - p2_form['surface_pct']),
        
        'recent_5_diff': float(p1_form['recent_5'] - p2_form['recent_5']),
        'recent_10_diff': float(p1_form['recent_10'] - p2_form['recent_10']),
        'recent_20_diff': float(p1_form['recent_20'] - p2_form['recent_20']),
        
        'p1_win_pct': float(p1_form['win_pct']),
        'p1_momentum': float(p1_form['momentum']),
        'p1_streak': float(p1_form['streak']),
        'p1_consistency': float(p1_form['consistency']),
        'p1_fatigue': float(p1_form['fatigue']),
        'p1_top10_pct': float(p1_form['top10_pct']),
        'p1_surface_pct': float(p1_form['surface_pct']),
        'p1_clutch': float(p1_form['clutch_performance']),
        'p1_sets_pct': float(p1_form['sets_won_pct']),
        
        'p2_win_pct': float(p2_form['win_pct']),
        'p2_momentum': float(p2_form['momentum']),
        'p2_streak': float(p2_form['streak']),
        'p2_consistency': float(p2_form['consistency']),
        'p2_fatigue': float(p2_form['fatigue']),
        'p2_top10_pct': float(p2_form['top10_pct']),
        'p2_surface_pct': float(p2_form['surface_pct']),
        'p2_clutch': float(p2_form['clutch_performance']),
        'p2_sets_pct': float(p2_form['sets_won_pct']),
        
        'h2h_win_pct': float(h2h_pct_p1),
        'h2h_matches': float(h2h_matches),
        'h2h_sets_ratio': float(h2h_info['sets_won'] / max(h2h_info['sets_lost'], 1)) if p1_name == first_player else float(h2h_info['sets_lost'] / max(h2h_info['sets_won'], 1)),
        
        'is_hard': 1 if surface == 'Hard' else 0,
        'is_clay': 1 if surface == 'Clay' else 0,
        'is_grass': 1 if surface == 'Grass' else 0,
        
        'elo_form_interaction': float((p1_elo - p2_elo) * (p1_form['win_pct'] - p2_form['win_pct'])),
        'momentum_elo_interaction': float((p1_form['momentum'] - p2_form['momentum']) * (p1_elo - p2_elo)),
        'surface_form_interaction': float(p1_form['surface_pct'] * (1 if surface == 'Hard' else 0.5)),
        
        'experience_diff': float(len(match_history.get(p1_id, [])) - len(match_history.get(p2_id, []))),
        'upset_potential': float(1 if p2_elo > p1_elo else 0),
        'form_consistency': float(min(p1_form['consistency'], p2_form['consistency'])),
    }
    
    # Get expected features
    expected_features = st.session_state.get('selected_features', list(features_dict.keys()))
    
    # Create DataFrame with all expected features
    features_df = pd.DataFrame({feat: [features_dict.get(feat, 0)] for feat in expected_features})
    
    # Scale and predict
    try:
        features_scaled = st.session_state.scaler.transform(features_df)
        prediction_proba = st.session_state.ensemble_model.predict_proba(features_scaled)[0][1]
        
        # Calculate confidence
        confidence = 1 - abs(prediction_proba - 0.5) * 2
        
        return prediction_proba, confidence, {
            'p1_elo': p1_elo,
            'p2_elo': p2_elo,
            'p1_form': p1_form,
            'p2_form': p2_form,
            'h2h': h2h_info
        }
    except Exception as e:
        st.error(f"Prediction error: {str(e)}")
        return None

def main():
    st.title("🎾 Advanced Tennis Prediction System")
    st.markdown("**Enhanced ELO + 40+ Features + Ensemble Model**")
    
    tabs = st.tabs(["📊 Train", "🎯 Predict", "📈 Analytics", "🔍 Insights", "🤖 Info"])
    
    with tabs[0]:
        st.header("Train Advanced Model")
        
        col1, col2, col3 = st.columns([2, 1, 1])
        with col1:
            file = st.file_uploader("Upload CSV with match data", type=['csv'], key='train_upload')
        with col2:
            k = st.slider("Base K-factor", 20, 50, 32, help="Higher = faster rating changes")
        with col3:
            top_k = st.slider("Top Player K-factor", 10, 30, 20, help="Lower = more stable for top players")
        
        if file:
            try:
                df = pd.read_csv(file)
                st.session_state.match_data = df
                
                with st.expander("📋 Data Preview"):
                    st.dataframe(df.head(10), width='stretch')
                    st.write(f"**Total matches:** {len(df)}")
                    
                    # Show column names to help debug
                    st.write("**Columns in dataset:**")
                    st.write(list(df.columns))
                
                required = ['Player_1', 'Player_2', 'Winner', 'Surface']
                missing_cols = [col for col in required if col not in df.columns]
                
                if missing_cols:
                    st.error(f"❌ Missing required columns: {', '.join(missing_cols)}")
                    st.info(f"Your CSV should contain columns: {', '.join(required)}")
                else:
                    st.success("✅ All required columns found!")
                    
                    # Clean the data
                    df_clean = df.copy()
                    df_clean['Player_1'] = df_clean['Player_1'].astype(str).str.strip()
                    df_clean['Player_2'] = df_clean['Player_2'].astype(str).str.strip()
                    df_clean['Winner'] = df_clean['Winner'].astype(str).str.strip()
                    df_clean['Surface'] = df_clean['Surface'].astype(str).str.strip()
                    
                    # Check for valid winners - FIXED LOGIC
                    # Create a mask that checks if Winner equals Player_1 or Player_2 for each row
                    valid_mask = (df_clean['Winner'] == df_clean['Player_1']) | (df_clean['Winner'] == df_clean['Player_2'])
                    valid_matches = df_clean[valid_mask]
                    
                    # Count invalid matches
                    invalid_matches = df_clean[~valid_mask]
                    
                    st.write(f"**Valid matches (with clear winner):** {len(valid_matches)}")
                    st.write(f"**Invalid matches (winner not one of the players):** {len(invalid_matches)}")
                    
                    if len(invalid_matches) > 0:
                        with st.expander("⚠️ View invalid matches"):
                            st.dataframe(invalid_matches.head(10), width='stretch')
                            st.write("These rows will be skipped during training.")
                    
                    if len(valid_matches) < 100:
                        st.warning(f"⚠️ Only {len(valid_matches)} valid matches found. Need at least 100 for good training.")
                    else:
                        st.success(f"✅ {len(valid_matches)} valid matches found - good for training!")
                    
                    optional_cols = ['Tournament_Level', 'Player_1_Sets', 'Player_2_Sets', 'Date']
                    optional_present = [col for col in optional_cols if col in df.columns]
                    if optional_present:
                        st.info(f"✅ Optional columns found: {', '.join(optional_present)}")
                    
                    if st.button("🚀 Train Advanced Model", type="primary"):
                        with st.spinner("🔄 Computing advanced ELO ratings..."):
                            progress_bar = st.progress(0)
                            
                            try:
                                # Use the cleaned dataframe
                                elos, g_elos, valid = compute_elo(df_clean, k_factor_base=k, k_factor_top=top_k)
                                st.session_state.elo_ratings = elos
                                st.session_state.global_elo = g_elos
                                
                                progress_bar.progress(30)
                                st.success(f"✅ ELO computed for {len(st.session_state.player_ids)} players from {valid} valid matches")
                                
                                with st.spinner("🔄 Creating advanced features..."):
                                    features_df, labels = create_advanced_features(
                                        df_clean, elos, g_elos,
                                        st.session_state.player_form_history,
                                        st.session_state.match_history
                                    )
                                    
                                    progress_bar.progress(60)
                                    st.success(f"✅ Created {len(features_df)} training samples with {features_df.shape[1]} features")
                                    
                                    if len(features_df) > 0:
                                        # Show class distribution
                                        unique_labels, counts = np.unique(labels, return_counts=True)
                                        st.write(f"**Class distribution:**")
                                        for label, count in zip(unique_labels, counts):
                                            st.write(f"  Class {label} ({'Win' if label == 1 else 'Loss'}): {count} samples ({count/len(labels)*100:.1f}%)")
                                    
                                    if len(features_df) > 100 and len(np.unique(labels)) >= 2:
                                        with st.spinner("🔄 Training ensemble model..."):
                                            model, metrics = train_advanced_model(features_df, labels)
                                            st.session_state.ensemble_model = model
                                            st.session_state.model_metrics = metrics
                                            
                                            progress_bar.progress(100)
                                        
                                        st.success("✅ Advanced model trained successfully!")
                                        
                                        # Display metrics
                                        st.subheader("📊 Model Performance")
                                        cols = st.columns(5)
                                        cols[0].metric("Accuracy", f"{metrics['accuracy']:.1%}")
                                        cols[1].metric("Precision", f"{metrics['precision']:.1%}")
                                        cols[2].metric("Recall", f"{metrics['recall']:.1%}")
                                        cols[3].metric("F1-Score", f"{metrics['f1']:.1%}")
                                        cols[4].metric("ROC-AUC", f"{metrics['roc_auc']:.3f}")
                                        
                                        # Display CV scores
                                        with st.expander("📈 Cross-Validation Scores"):
                                            for model_name, score in metrics.get('cv_scores', {}).items():
                                                st.write(f"{model_name.upper()}: {score:.3f}")
                                        
                                        # Feature importance
                                        if st.session_state.feature_importance:
                                            with st.expander("🔍 Top 20 Feature Importance"):
                                                imp_df = pd.DataFrame(
                                                    list(st.session_state.feature_importance.items()),
                                                    columns=['Feature', 'Importance']
                                                ).sort_values('Importance', ascending=False).head(20)
                                                
                                                fig = px.bar(imp_df, x='Importance', y='Feature', 
                                                            orientation='h', title='Feature Importance')
                                                st.plotly_chart(fig, use_container_width=True)
                                    else:
                                        if len(features_df) <= 100:
                                            st.error(f"❌ Not enough training samples. Need at least 100, but only got {len(features_df)}.")
                                        if len(np.unique(labels)) < 2:
                                            st.error(f"❌ Class imbalance: Only found {len(np.unique(labels))} class(es). Need both win and loss examples.")
                                
                            except Exception as e:
                                st.error(f"❌ Error during training: {str(e)}")
                                import traceback
                                st.text(traceback.format_exc())
                                
            except Exception as e:
                st.error(f"❌ Error reading CSV file: {str(e)}")
                st.info("Please make sure you're uploading a valid CSV file with the correct format.")
    
    with tabs[1]:
        st.header("🎯 Match Prediction")
        
        if not st.session_state.ensemble_model:
            st.warning("⚠️ Please train the model first in the 'Train' tab!")
        else:
            players = list(st.session_state.player_names.values())
            
            if players:
                col1, col2, col3, col4 = st.columns([2, 2, 1, 1])
                
                with col1:
                    p1 = st.selectbox("Player 1", players, key='p1_select')
                with col2:
                    other_players = [p for p in players if p != p1]
                    p2 = st.selectbox("Player 2", other_players, key='p2_select')
                with col3:
                    surf = st.selectbox("Surface", SURFACE_TYPES, key='surface_select')
                with col4:
                    tourney_level = st.selectbox("Tournament", TOURNEY_LEVELS, key='tourney_select')
                
                if st.button("🔮 Predict Match Outcome", type="primary"):
                    p1_id = st.session_state.player_ids.get(p1)
                    p2_id = st.session_state.player_ids.get(p2)
                    
                    if p1_id and p2_id:
                        with st.spinner("Calculating prediction..."):
                            result = predict_match_advanced(
                                p1_id, p2_id, surf,
                                st.session_state.elo_ratings,
                                st.session_state.global_elo,
                                st.session_state.player_form_history,
                                st.session_state.match_history
                            )
                        
                        if result:
                            prob, confidence, details = result
                            
                            # Display prediction
                            col1, col2, col3 = st.columns([2, 1, 1])
                            with col1:
                                st.subheader("Prediction Result")
                                
                                # Gauge chart
                                fig = go.Figure(go.Indicator(
                                    mode="gauge+number",
                                    value=prob * 100,
                                    title={'text': f"{p1} Win Probability"},
                                    domain={'x': [0, 1], 'y': [0, 1]},
                                    gauge={
                                        'axis': {'range': [0, 100]},
                                        'bar': {'color': "green" if prob > 0.5 else "red"},
                                        'steps': [
                                            {'range': [0, 50], 'color': "lightgray"},
                                            {'range': [50, 100], 'color': "lightgreen"}
                                        ],
                                        'threshold': {
                                            'line': {'color': "red", 'width': 4},
                                            'thickness': 0.75,
                                            'value': 50
                                        }
                                    }
                                ))
                                fig.update_layout(height=300)
                                st.plotly_chart(fig)
                            
                            with col2:
                                st.metric("Confidence", f"{confidence:.0%}")
                                st.metric(f"{p2} Win %", f"{(1-prob)*100:.1f}%")
                            
                            with col3:
                                # Recommendation
                                if prob > 0.7:
                                    st.success("✅ Strong Favorite")
                                    st.metric("Recommendation", "Bet with Confidence")
                                elif prob > 0.6:
                                    st.info("📈 Moderate Favorite")
                                    st.metric("Recommendation", "Cautious Bet")
                                elif prob > 0.4:
                                    st.warning("⚖️ Close Match")
                                    st.metric("Recommendation", "Avoid Betting")
                                else:
                                    st.error("📉 Underdog")
                                    st.metric("Recommendation", "Value Bet")
                            
                            # Detailed breakdown
                            with st.expander("📊 Detailed Analysis"):
                                st.subheader("Player Comparison")
                                
                                # ELO Comparison
                                st.write("**📈 ELO Ratings:**")
                                col1, col2, col3 = st.columns(3)
                                with col1:
                                    st.metric(f"{p1} ELO", f"{details['p1_elo']:.0f}")
                                with col2:
                                    diff = details['p1_elo'] - details['p2_elo']
                                    st.metric("Difference", f"{diff:+.0f}", 
                                             delta="Advantage" if diff > 0 else "Disadvantage")
                                with col3:
                                    st.metric(f"{p2} ELO", f"{details['p2_elo']:.0f}")
                                
                                # Form Comparison
                                st.write("**📊 Form Analysis:**")
                                col1, col2 = st.columns(2)
                                with col1:
                                    st.write(f"**{p1}**")
                                    f1 = details['p1_form']
                                    
                                    form_cols = st.columns(3)
                                    form_cols[0].metric("Win %", f"{f1['win_pct']*100:.0f}%")
                                    form_cols[1].metric("Streak", f"{f1['streak']:+d}")
                                    form_cols[2].metric("Fatigue", f"{f1['fatigue']:.1f}")
                                    
                                    st.metric("Recent Form (Last 5)", f"{f1['recent_5']*100:.0f}%")
                                    st.metric("Consistency", f"{f1['consistency']*100:.0f}%")
                                    st.metric("Clutch Performance", f"{f1['clutch_performance']*100:.0f}%")
                                
                                with col2:
                                    st.write(f"**{p2}**")
                                    f2 = details['p2_form']
                                    
                                    form_cols = st.columns(3)
                                    form_cols[0].metric("Win %", f"{f2['win_pct']*100:.0f}%")
                                    form_cols[1].metric("Streak", f"{f2['streak']:+d}")
                                    form_cols[2].metric("Fatigue", f"{f2['fatigue']:.1f}")
                                    
                                    st.metric("Recent Form (Last 5)", f"{f2['recent_5']*100:.0f}%")
                                    st.metric("Consistency", f"{f2['consistency']*100:.0f}%")
                                    st.metric("Clutch Performance", f"{f2['clutch_performance']*100:.0f}%")
                                
                                # H2H History
                                h2h = details['h2h']
                                h2h_key = tuple(sorted([p1, p2]))
                                first_player = h2h_key[0]
                                
                                if h2h['matches'] > 0:
                                    st.write("**🤝 Head-to-Head:**")
                                    col1, col2, col3 = st.columns(3)
                                    col1.metric("Total Matches", h2h['matches'])
                                    
                                    # Determine p1's H2H wins
                                    if p1 == first_player:
                                        p1_h2h_wins = h2h['player1_wins']
                                    else:
                                        p1_h2h_wins = h2h['matches'] - h2h['player1_wins']
                                    
                                    col2.metric(f"{p1} Wins", p1_h2h_wins)
                                    col3.metric(f"{p2} Wins", h2h['matches'] - p1_h2h_wins)
                                    
                                    if h2h['sets_won'] > 0 and h2h['sets_lost'] > 0:
                                        st.metric("Sets Ratio", f"{h2h['sets_won']}:{h2h['sets_lost']}")
            else:
                st.info("Please train the model first to see player list")
    
    with tabs[2]:
        st.header("📈 Player Analytics")
        if st.session_state.player_ids and st.session_state.player_names:
            players = list(st.session_state.player_names.values())
            if players:
                player = st.selectbox("Select Player", players, key='stats_player')
                if player:
                    pid = st.session_state.player_ids.get(player)
                    matches = st.session_state.match_history.get(pid, [])
                    
                    if matches:
                        wins = sum(1 for x in matches if x.get('won', False))
                        total = len(matches)
                        
                        # Basic stats
                        col1, col2, col3, col4 = st.columns(4)
                        col1.metric("Total Matches", total)
                        col2.metric("Wins", wins)
                        col3.metric("Losses", total - wins)
                        col4.metric("Win %", f"{wins/total*100:.1f}%" if total > 0 else "N/A")
                        
                        # Surface performance
                        st.subheader("Surface Performance")
                        surface_stats = {}
                        for match in matches:
                            surface = match.get('surface', 'Hard')
                            if surface not in surface_stats:
                                surface_stats[surface] = {'wins': 0, 'total': 0}
                            surface_stats[surface]['total'] += 1
                            if match.get('won', False):
                                surface_stats[surface]['wins'] += 1
                        
                        if surface_stats:
                            surf_df = pd.DataFrame([
                                {
                                    'Surface': surf,
                                    'Win %': (stats['wins'] / stats['total']) * 100,
                                    'Matches': stats['total']
                                }
                                for surf, stats in surface_stats.items()
                            ])
                            
                            col1, col2 = st.columns([3, 1])
                            with col1:
                                fig = px.bar(surf_df, x='Surface', y='Win %', 
                                            title='Win Percentage by Surface',
                                            color='Win %', color_continuous_scale='RdYlGn')
                                st.plotly_chart(fig)
                            with col2:
                                st.dataframe(surf_df, width='stretch')
                        
                        # ELO progression
                        st.subheader("ELO Rating Progression")
                        if len(matches) > 5:
                            dates = []
                            elos = []
                            for match in matches[-50:]:
                                dates.append(match['date'])
                                elos.append(match.get('w_elo', 1500) if match.get('won', False) else match.get('l_elo', 1500))
                            
                            if dates and elos:
                                elo_df = pd.DataFrame({'Date': dates, 'ELO': elos})
                                elo_df = elo_df.sort_values('Date')
                                
                                fig = px.line(elo_df, x='Date', y='ELO', 
                                             title='ELO Rating Over Time',
                                             markers=True)
                                st.plotly_chart(fig)
                    else:
                        st.info(f"No match history found for {player}")
            else:
                st.info("Please train the model first to see player list")
    
    with tabs[3]:
        st.header("🔍 Model Insights")
        if st.session_state.model_metrics:
            # Confusion Matrix
            st.subheader("Confusion Matrix")
            cm = st.session_state.model_metrics.get('confusion_matrix', [[0, 0], [0, 0]])
            cm_fig = px.imshow(cm, 
                              labels=dict(x="Predicted", y="Actual", color="Count"),
                              x=['Loss', 'Win'],
                              y=['Loss', 'Win'],
                              title="Confusion Matrix",
                              text_auto=True)
            st.plotly_chart(cm_fig)
            
            # Feature Importance
            if st.session_state.feature_importance:
                st.subheader("Top Features")
                imp_df = pd.DataFrame(
                    list(st.session_state.feature_importance.items()),
                    columns=['Feature', 'Importance']
                ).sort_values('Importance', ascending=False).head(15)
                
                fig = px.bar(imp_df, x='Importance', y='Feature', 
                            orientation='h', 
                            title='Top 15 Most Important Features',
                            color='Importance',
                            color_continuous_scale='viridis')
                st.plotly_chart(fig)
            
            # Model Comparison
            st.subheader("Model Performance Comparison")
            cv_scores = st.session_state.model_metrics.get('cv_scores', {})
            if cv_scores:
                model_df = pd.DataFrame([
                    {'Model': name.upper(), 'CV AUC Score': score}
                    for name, score in cv_scores.items()
                ])
                fig = px.bar(model_df, x='Model', y='CV AUC Score',
                            title='Cross-Validation AUC Scores by Model',
                            color='CV AUC Score',
                            color_continuous_scale='blues')
                st.plotly_chart(fig)
        else:
            st.info("Train a model to see insights")
    
    with tabs[4]:
        st.header("🤖 System Information")
        
        st.markdown("""
        ## Advanced Tennis Prediction System
        
        ### 🔧 Key Features
        
        **1. Enhanced ELO System:**
        - Dynamic K-factor based on player experience
        - Tournament importance weighting
        - H2H history adjustments
        - Set margin adjustments
        
        **2. Advanced Features (40+):**
        - Multiple time windows for form (5/10/20 matches)
        - Clutch performance metrics
        - Head-to-head history
        - Surface-specific performance
        - Interaction features
        
        **3. Ensemble Model:**
        - Random Forest
        - Gradient Boosting
        - AdaBoost
        - Logistic Regression
        - Voting ensemble with calibration
        
        **4. Model Validation:**
        - 5-fold stratified cross-validation
        - Feature selection
        - Probability calibration
        
        ### 📊 Expected Performance
        - **Target Accuracy:** 65-75%
        - **ROC-AUC:** 0.70-0.80
        
        ### 📁 Required CSV Format
        Minimum columns:
        - `Player_1`, `Player_2`, `Winner`, `Surface`, `Date`
        
        Optional columns for better accuracy:
        - `Tournament_Level`
        - `Player_1_Sets`, `Player_2_Sets`
        
        ### ⚡ Tips for Best Results
        1. Use at least 500 matches for training
        2. Include recent matches
        3. Ensure accurate player names
        4. Include surface information
        """)
        
        if st.session_state.model_metrics:
            st.subheader("Current Model Metrics")
            st.json(st.session_state.model_metrics.get('classification_report', {}))

if __name__ == "__main__":
    if ML_AVAILABLE:
        main()
    else:
        st.error("""
        ❌ Required packages not installed. Please install:
        ```
        pip install scikit-learn plotly
        ```
        """)

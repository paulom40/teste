import streamlit as st
import pandas as pd
import numpy as np
import math
import warnings
from collections import defaultdict, deque, Counter
from datetime import datetime, timedelta
import plotly.graph_objects as go
import plotly.express as px

warnings.filterwarnings('ignore')

try:
    from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV, StratifiedKFold
    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix, classification_report
    from sklearn.preprocessing import RobustScaler, StandardScaler
    from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier, AdaBoostClassifier, StackingClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.calibration import CalibratedClassifierCV
    from sklearn.svm import SVC
    from xgboost import XGBClassifier
    from lightgbm import LGBMClassifier
    from catboost import CatBoostClassifier
    from sklearn.feature_selection import SelectKBest, f_classif, RFECV
    from sklearn.pipeline import Pipeline
    import joblib
    ML_AVAILABLE = True
except ImportError:
    ML_AVAILABLE = False
    st.error("Please install required packages: scikit-learn, xgboost, lightgbm, catboost, plotly")

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
        'games_won_pct': 0.5, 'clutch_performance': 0.5,
        'recent_5': 0.5, 'recent_10': 0.5, 'recent_20': 0.5
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
    
    # Initialize structures with correct defaultdict usage
    elo_ratings = {pid: {s: initial_elo for s in SURFACE_TYPES} for pid in player_ids.values()}
    global_ratings = {pid: initial_elo for pid in player_ids.values()}
    form_history = defaultdict(lambda: defaultdict(deque))
    match_history = defaultdict(list)
    surface_perf = defaultdict(lambda: defaultdict(list))
    
    # Fix: Properly initialize h2h_stats with nested defaultdict
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
        
        if winner not in [p1, p2]:
            continue
        
        valid_count += 1
        
        winner_id = player_ids[winner]
        loser = p2 if winner == p1 else p1
        loser_id = player_ids[loser]
        
        # Get current ratings
        rating_w = elo_ratings[winner_id][surface]
        rating_l = elo_ratings[loser_id][surface]
        
        # Dynamic K-factor based on player reliability and tournament importance
        w_matches = len([m for m in match_history.get(winner_id, []) if m.get('surface') == surface])
        l_matches = len([m for m in match_history.get(loser_id, []) if m.get('surface') == surface])
        
        # More stable K-factor for established players
        k_w = max(k_factor_top, k_factor_base / (1 + w_matches / 100))
        k_l = max(k_factor_top, k_factor_base / (1 + l_matches / 100))
        
        # Tournament importance multiplier
        t_bonus = tourney_bonus.get(tourney_level, 1.0)
        
        # H2H adjustment - initialize if not exists
        h2h_key = tuple(sorted([p1, p2]))
        if h2h_key not in h2h_stats:
            h2h_stats[h2h_key] = {'wins': 0, 'matches': 0, 'sets_won': 0, 'sets_lost': 0}
        
        # Update H2H stats
        h2h_info = h2h_stats[h2h_key]
        if winner == p1:
            h2h_info['wins'] += 1
            if has_set_data:
                h2h_info['sets_won'] += row.get('Player_1_Sets', 2)
                h2h_info['sets_lost'] += row.get('Player_2_Sets', 1)
        else:
            # When p2 wins, we need to track who won the H2H
            # For simplicity, we'll just count matches
            if has_set_data:
                h2h_info['sets_won'] += row.get('Player_2_Sets', 2)
                h2h_info['sets_lost'] += row.get('Player_1_Sets', 1)
        h2h_info['matches'] += 1
        
        # H2H factor calculation
        if h2h_info['matches'] > 0:
            # Determine which player has the advantage
            if winner == p1:
                h2h_advantage = (h2h_info['wins'] / h2h_info['matches']) - 0.5
            else:
                # If p2 won, p1's H2H win rate is (matches - wins)/matches
                p1_wins = h2h_info['matches'] - h2h_info['wins']
                h2h_advantage = (p1_wins / h2h_info['matches']) - 0.5
            h2h_factor = 1 + (h2h_advantage * 0.2)  # 20% adjustment max
        else:
            h2h_factor = 1.0
        
        # Expected win probability
        exp_w = 1 / (1 + math.pow(10, (rating_l - rating_w) / 400))
        
        # Score margin adjustment (if set data available)
        if has_set_data:
            if winner == p1:
                sets_won = row.get('Player_1_Sets', 2)
                sets_lost = row.get('Player_2_Sets', 1)
            else:
                sets_won = row.get('Player_2_Sets', 2)
                sets_lost = row.get('Player_1_Sets', 1)
            
            margin = (sets_won - sets_lost) / (sets_won + sets_lost)
            margin_factor = 1 + (margin * 0.3)  # 30% adjustment for dominant wins
        else:
            margin_factor = 1.0
        
        # Calculate rating changes
        elo_change_winner = k_w * t_bonus * h2h_factor * margin_factor * (1 - exp_w)
        elo_change_loser = k_l * t_bonus * (0 - (1 - exp_w))
        
        # Apply changes
        elo_ratings[winner_id][surface] = rating_w + elo_change_winner
        elo_ratings[loser_id][surface] = rating_l + elo_change_loser
        
        # Global rating (weighted average of surface ratings)
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
        
        # Store match record with detailed info
        match_record = {
            'date': row['Date'],
            'surface': surface,
            'opponent': loser_id if winner_id == winner_id else winner_id,
            'w_elo': rating_w,
            'l_elo': rating_l,
            'won': True,
            'tournament_level': tourney_level,
            'sets_won': sets_won if has_set_data else 2,
            'sets_lost': sets_lost if has_set_data else 1,
            'year': row.get('Year', datetime.now().year),
            'month': row.get('Month', datetime.now().month)
        }
        
        match_history[winner_id].append(match_record)
        match_history[loser_id].append({
            **match_record,
            'won': False,
            'opponent': winner_id,
            'sets_won': sets_lost if has_set_data else 1,
            'sets_lost': sets_won if has_set_data else 2
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
    weights = [0.3, 0.25, 0.2, 0.15, 0.1]  # Last 5 matches weighted
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
    
    # Consistency (performance vs expectation)
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
    
    # Fatigue (matches in last 30 days)
    fatigue = sum(1 for m in match_history.get(player_id, [])
                 if (now - m['date']).days <= 30)
    
    # Set performance
    sets_won = sum(m.get('sets_won', 0) for m in recent_matches)
    sets_total = sum(m.get('sets_won', 0) + m.get('sets_lost', 0) for m in recent_matches)
    sets_pct = sets_won / max(sets_total, 1)
    
    # Clutch performance (close matches)
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
        'fatigue': min(fatigue / 10, 1),  # Normalized
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
    """Create advanced feature set with interaction terms"""
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
        if winner == p1:
            w_id = player_ids.get(p1)
            l_id = player_ids.get(p2)
            w_name, l_name = p1, p2
        else:
            w_id = player_ids.get(p2)
            l_id = player_ids.get(p1)
            w_name, l_name = p2, p1
        
        if w_id is None or l_id is None:
            continue
        
        # Get ratings
        w_elo = elo_ratings.get(w_id, {}).get(surface, global_ratings.get(w_id, 1500))
        l_elo = elo_ratings.get(l_id, {}).get(surface, global_ratings.get(l_id, 1500))
        w_global = global_ratings.get(w_id, 1500)
        l_global = global_ratings.get(l_id, 1500)
        
        # Get form
        w_form = calc_form(w_id, surface, form_history, match_history, w_elo)
        l_form = calc_form(l_id, surface, form_history, match_history, l_elo)
        
        # H2H stats
        h2h_key = tuple(sorted([w_name, l_name]))
        h2h_info = h2h_stats.get(h2h_key, {'wins': 0, 'matches': 0, 'sets_won': 0, 'sets_lost': 0})
        h2h_matches = h2h_info['matches']
        
        # Determine who is p1 in features (always the winner for feature creation)
        if w_name == p1:
            # Winner is Player 1
            features = {
                # Basic ELO features
                'elo_diff': float(w_elo - l_elo),
                'elo_ratio': float(w_elo / max(l_elo, 1)),
                'global_elo_diff': float(w_global - l_global),
                
                # Surface-specific ELO
                'surface_elo_diff': float(w_elo - l_elo),
                'surface_elo_ratio': float(w_elo / max(l_elo, 1)),
                
                # Form features
                'win_pct_diff': float(w_form['win_pct'] - l_form['win_pct']),
                'momentum_diff': float(w_form['momentum'] - l_form['momentum']),
                'streak_diff': float(w_form['streak'] - l_form['streak']),
                'consistency_diff': float(w_form['consistency'] - l_form['consistency']),
                'fatigue_diff': float(w_form['fatigue'] - l_form['fatigue']),
                'top10_pct_diff': float(w_form['top10_pct'] - l_form['top10_pct']),
                'surface_pct_diff': float(w_form['surface_pct'] - l_form['surface_pct']),
                
                # Recent form
                'recent_5_diff': float(w_form['recent_5'] - l_form['recent_5']),
                'recent_10_diff': float(w_form['recent_10'] - l_form['recent_10']),
                'recent_20_diff': float(w_form['recent_20'] - l_form['recent_20']),
                
                # Individual features
                'p1_win_pct': float(w_form['win_pct']),
                'p1_momentum': float(w_form['momentum']),
                'p1_streak': float(w_form['streak']),
                'p1_consistency': float(w_form['consistency']),
                'p1_fatigue': float(w_form['fatigue']),
                'p1_top10_pct': float(w_form['top10_pct']),
                'p1_surface_pct': float(w_form['surface_pct']),
                'p1_clutch': float(w_form['clutch_performance']),
                'p1_sets_pct': float(w_form['sets_won_pct']),
                
                'p2_win_pct': float(l_form['win_pct']),
                'p2_momentum': float(l_form['momentum']),
                'p2_streak': float(l_form['streak']),
                'p2_consistency': float(l_form['consistency']),
                'p2_fatigue': float(l_form['fatigue']),
                'p2_top10_pct': float(l_form['top10_pct']),
                'p2_surface_pct': float(l_form['surface_pct']),
                'p2_clutch': float(l_form['clutch_performance']),
                'p2_sets_pct': float(l_form['sets_won_pct']),
                
                # H2H features
                'h2h_win_pct': float(h2h_info['wins'] / max(h2h_matches, 1)) if w_name == p1 else float(1 - (h2h_info['wins'] / max(h2h_matches, 1))),
                'h2h_matches': float(h2h_matches),
                'h2h_sets_ratio': float(h2h_info['sets_won'] / max(h2h_info['sets_lost'], 1)),
                
                # Surface indicators
                'is_hard': 1 if surface == 'Hard' else 0,
                'is_clay': 1 if surface == 'Clay' else 0,
                'is_grass': 1 if surface == 'Grass' else 0,
                
                # Interaction features
                'elo_form_interaction': float((w_elo - l_elo) * (w_form['win_pct'] - l_form['win_pct'])),
                'momentum_elo_interaction': float((w_form['momentum'] - l_form['momentum']) * (w_elo - l_elo)),
                'surface_form_interaction': float(w_form['surface_pct'] * (1 if surface == 'Hard' else 0.5)),
                
                # Derived metrics
                'experience_diff': float(len(match_history.get(w_id, [])) - len(match_history.get(l_id, []))),
                'upset_potential': float(1 if l_elo > w_elo else 0),
                'form_consistency': float(min(w_form['consistency'], l_form['consistency'])),
            }
        else:
            # Winner is Player 2, we'll create symmetric features
            features = {
                # Basic ELO features (reversed)
                'elo_diff': float(l_elo - w_elo),
                'elo_ratio': float(l_elo / max(w_elo, 1)),
                'global_elo_diff': float(l_global - w_global),
                
                # Surface-specific ELO
                'surface_elo_diff': float(l_elo - w_elo),
                'surface_elo_ratio': float(l_elo / max(w_elo, 1)),
                
                # Form features (reversed)
                'win_pct_diff': float(l_form['win_pct'] - w_form['win_pct']),
                'momentum_diff': float(l_form['momentum'] - w_form['momentum']),
                'streak_diff': float(l_form['streak'] - w_form['streak']),
                'consistency_diff': float(l_form['consistency'] - w_form['consistency']),
                'fatigue_diff': float(l_form['fatigue'] - w_form['fatigue']),
                'top10_pct_diff': float(l_form['top10_pct'] - w_form['top10_pct']),
                'surface_pct_diff': float(l_form['surface_pct'] - w_form['surface_pct']),
                
                # Recent form
                'recent_5_diff': float(l_form['recent_5'] - w_form['recent_5']),
                'recent_10_diff': float(l_form['recent_10'] - w_form['recent_10']),
                'recent_20_diff': float(l_form['recent_20'] - w_form['recent_20']),
                
                # Individual features (p1 is now the original loser)
                'p1_win_pct': float(l_form['win_pct']),
                'p1_momentum': float(l_form['momentum']),
                'p1_streak': float(l_form['streak']),
                'p1_consistency': float(l_form['consistency']),
                'p1_fatigue': float(l_form['fatigue']),
                'p1_top10_pct': float(l_form['top10_pct']),
                'p1_surface_pct': float(l_form['surface_pct']),
                'p1_clutch': float(l_form['clutch_performance']),
                'p1_sets_pct': float(l_form['sets_won_pct']),
                
                'p2_win_pct': float(w_form['win_pct']),
                'p2_momentum': float(w_form['momentum']),
                'p2_streak': float(w_form['streak']),
                'p2_consistency': float(w_form['consistency']),
                'p2_fatigue': float(w_form['fatigue']),
                'p2_top10_pct': float(w_form['top10_pct']),
                'p2_surface_pct': float(w_form['surface_pct']),
                'p2_clutch': float(w_form['clutch_performance']),
                'p2_sets_pct': float(w_form['sets_won_pct']),
                
                # H2H features (reversed perspective)
                'h2h_win_pct': float(1 - (h2h_info['wins'] / max(h2h_matches, 1))) if w_name == p1 else float(h2h_info['wins'] / max(h2h_matches, 1)),
                'h2h_matches': float(h2h_matches),
                'h2h_sets_ratio': float(h2h_info['sets_lost'] / max(h2h_info['sets_won'], 1)),
                
                # Surface indicators (same)
                'is_hard': 1 if surface == 'Hard' else 0,
                'is_clay': 1 if surface == 'Clay' else 0,
                'is_grass': 1 if surface == 'Grass' else 0,
                
                # Interaction features (reversed)
                'elo_form_interaction': float((l_elo - w_elo) * (l_form['win_pct'] - w_form['win_pct'])),
                'momentum_elo_interaction': float((l_form['momentum'] - w_form['momentum']) * (l_elo - w_elo)),
                'surface_form_interaction': float(l_form['surface_pct'] * (1 if surface == 'Hard' else 0.5)),
                
                # Derived metrics
                'experience_diff': float(len(match_history.get(l_id, [])) - len(match_history.get(w_id, []))),
                'upset_potential': float(1 if w_elo > l_elo else 0),
                'form_consistency': float(min(l_form['consistency'], w_form['consistency'])),
            }
        
        features_list.append(features)
        labels.append(1)  # Winner perspective always gets label 1
    
    return pd.DataFrame(features_list), np.array(labels)

def train_advanced_model(features_df, labels):
    """Train advanced ensemble with hyperparameter tuning"""
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
        features_df, labels, test_size=0.15, random_state=42, stratify=labels
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
            n_estimators=200,
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
    
    # Try to add XGBoost if available
    try:
        models['xgb'] = XGBClassifier(
            n_estimators=200,
            max_depth=6,
            learning_rate=0.05,
            random_state=42,
            n_jobs=-1,
            use_label_encoder=False,
            eval_metric='logloss'
        )
    except:
        pass
    
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
    
    # Feature importance from best model
    if 'rf' in trained_models:
        importances = trained_models['rf'].feature_importances_
        feature_importance = dict(zip(features_df.columns, importances))
        st.session_state.feature_importance = feature_importance
    
    return calibrated_model, metrics, X_test, y_test, y_pred_proba

def predict_match_advanced(p1_id, p2_id, surface, elo_ratings, global_ratings, form_history, match_history):
    """Advanced prediction with confidence intervals"""
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
    h2h_info = st.session_state.h2h_stats.get(h2h_key, {'wins': 0, 'matches': 0, 'sets_won': 0, 'sets_lost': 0})
    h2h_matches = h2h_info['matches']
    
    # Determine H2H win percentage for p1
    if h2h_matches > 0:
        # Determine if p1 is the first player in the sorted tuple
        first_player = h2h_key[0]
        if p1_name == first_player:
            h2h_pct_p1 = h2h_info['wins'] / h2h_matches
        else:
            h2h_pct_p1 = 1 - (h2h_info['wins'] / h2h_matches)
    else:
        h2h_pct_p1 = 0.5
    
    # Get expected features from session state or use all available
    expected_features = st.session_state.get('selected_features', [])
    if not expected_features:
        # Create a default feature set based on available data
        features_dict = {
            'elo_diff': float(p1_elo - p2_elo),
            'elo_ratio': float(p1_elo / max(p2_elo, 1)),
            'global_elo_diff': float(p1_global - p2_global),
            'surface_elo_diff': float(p1_elo - p2_elo),
            
            'win_pct_diff': float(p1_form['win_pct'] - p2_form['win_pct']),
            'momentum_diff': float(p1_form['momentum'] - p2_form['momentum']),
            'streak_diff': float(p1_form['streak'] - p2_form['streak']),
            
            'p1_win_pct': float(p1_form['win_pct']),
            'p1_momentum': float(p1_form['momentum']),
            'p1_streak': float(p1_form['streak']),
            
            'p2_win_pct': float(p2_form['win_pct']),
            'p2_momentum': float(p2_form['momentum']),
            'p2_streak': float(p2_form['streak']),
            
            'h2h_win_pct': float(h2h_pct_p1),
            'h2h_matches': float(h2h_matches),
            
            'is_hard': 1 if surface == 'Hard' else 0,
            'is_clay': 1 if surface == 'Clay' else 0,
            'is_grass': 1 if surface == 'Grass' else 0,
        }
        expected_features = list(features_dict.keys())
    else:
        # Create features dictionary with all expected features
        features_dict = {}
        # Add all possible features
        all_possible_features = {
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
            'h2h_sets_ratio': float(h2h_info['sets_won'] / max(h2h_info['sets_lost'], 1)) if p1_name == h2h_key[0] else float(h2h_info['sets_lost'] / max(h2h_info['sets_won'], 1)),
            
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
        
        # Only include features that are in expected_features
        features_dict = {k: v for k, v in all_possible_features.items() if k in expected_features}
    
    # Create DataFrame
    features_df = pd.DataFrame([features_dict])
    
    # Ensure all expected features are present
    for feat in expected_features:
        if feat not in features_df.columns:
            features_df[feat] = 0
    
    # Scale and predict
    try:
        features_scaled = st.session_state.scaler.transform(features_df)
        prediction_proba = st.session_state.ensemble_model.predict_proba(features_scaled)[0][1]
        
        # Calculate confidence based on feature extremity
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
            df = pd.read_csv(file)
            st.session_state.match_data = df
            
            with st.expander("📋 Data Preview"):
                st.dataframe(df.head(10), use_container_width=True)
                st.write(f"**Total matches:** {len(df)}")
            
            required = ['Player_1', 'Player_2', 'Winner', 'Surface']
            if all(col in df.columns for col in required):
                st.success("✅ Valid CSV format detected")
                
                optional_cols = ['Tournament_Level', 'Player_1_Sets', 'Player_2_Sets']
                optional_present = [col for col in optional_cols if col in df.columns]
                if optional_present:
                    st.info(f"✅ Optional columns found: {', '.join(optional_present)}")
                
                if st.button("🚀 Train Advanced Model", type="primary", use_container_width=True):
                    with st.spinner("🔄 Computing advanced ELO ratings..."):
                        progress_bar = st.progress(0)
                        
                        try:
                            elos, g_elos, valid = compute_elo(df, k_factor_base=k, k_factor_top=top_k)
                            st.session_state.elo_ratings = elos
                            st.session_state.global_elo = g_elos
                            
                            progress_bar.progress(30)
                            st.success(f"✅ ELO computed for {len(st.session_state.player_ids)} players from {valid} matches")
                            
                            with st.spinner("🔄 Creating advanced features..."):
                                features_df, labels = create_advanced_features(
                                    df, elos, g_elos,
                                    st.session_state.player_form_history,
                                    st.session_state.match_history
                                )
                                
                                progress_bar.progress(60)
                                st.success(f"✅ Created {len(features_df)} training samples with {features_df.shape[1]} features")
                                
                                if len(features_df) > 100:
                                    with st.spinner("🔄 Training ensemble model (this may take a minute)..."):
                                        model, metrics, X_test, y_test, y_proba = train_advanced_model(features_df, labels)
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
                                    st.error("❌ Not enough training data. Need at least 100 valid matches.")
                            
                        except Exception as e:
                            st.error(f"❌ Error during training: {str(e)}")
                            st.info("Please check your CSV format and ensure it has the required columns: Player_1, Player_2, Winner, Surface")
            else:
                st.error("❌ CSV must contain columns: Player_1, Player_2, Winner, Surface")
    
    with tabs[1]:
        st.header("🎯 Match Prediction")
        
        if not st.session_state.ensemble_model:
            st.warning("⚠️ Please train the model first in the 'Train' tab!")
        else:
            col1, col2, col3, col4 = st.columns([2, 2, 1, 1])
            players = list(st.session_state.player_names.values())
            
            if players:
                with col1:
                    p1 = st.selectbox("Player 1", players, key='p1_select')
                with col2:
                    # Filter out p1 from the list
                    other_players = [p for p in players if p != p1]
                    p2 = st.selectbox("Player 2", other_players, key='p2_select')
                with col3:
                    surf = st.selectbox("Surface", SURFACE_TYPES, key='surface_select')
                with col4:
                    tourney_level = st.selectbox("Tournament", TOURNEY_LEVELS, key='tourney_select')
                
                if st.button("🔮 Predict Match Outcome", type="primary", use_container_width=True):
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
                                st.plotly_chart(fig, use_container_width=True)
                            
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
                                    
                                    # Create form metrics
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
                                if h2h['matches'] > 0:
                                    st.write("**🤝 Head-to-Head:**")
                                    col1, col2, col3 = st.columns(3)
                                    col1.metric("Total Matches", h2h['matches'])
                                    
                                    # Determine p1's H2H wins
                                    p1_h2h_wins = h2h['wins'] if p1 == h2h_key[0] else (h2h['matches'] - h2h['wins'])
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
                                st.plotly_chart(fig, use_container_width=True)
                            with col2:
                                st.dataframe(surf_df, use_container_width=True)
                        
                        # ELO progression
                        st.subheader("ELO Rating Progression")
                        if len(matches) > 5:
                            dates = []
                            elos = []
                            for match in matches[-50:]:  # Last 50 matches
                                dates.append(match['date'])
                                elos.append(match.get('w_elo', 1500) if match.get('won', False) else match.get('l_elo', 1500))
                            
                            if dates and elos:
                                elo_df = pd.DataFrame({'Date': dates, 'ELO': elos})
                                elo_df = elo_df.sort_values('Date')
                                
                                fig = px.line(elo_df, x='Date', y='ELO', 
                                             title='ELO Rating Over Time',
                                             markers=True)
                                st.plotly_chart(fig, use_container_width=True)
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
            st.plotly_chart(cm_fig, use_container_width=True)
            
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
                st.plotly_chart(fig, use_container_width=True)
            
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
                st.plotly_chart(fig, use_container_width=True)
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
        
        **2. Advanced Features (30+):**
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
        - XGBoost (if available)
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
        pip install scikit-learn xgboost plotly
        ```
        For basic functionality without XGBoost:
        ```
        pip install scikit-learn plotly
        ```
        """)

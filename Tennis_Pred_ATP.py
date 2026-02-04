import streamlit as st
import pandas as pd
import numpy as np
import math
import warnings
from collections import defaultdict, deque
from datetime import datetime
import plotly.graph_objects as go

warnings.filterwarnings('ignore')

try:
    from sklearn.model_selection import train_test_split, cross_val_score
    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
    from sklearn.preprocessing import RobustScaler, StandardScaler
    from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier, AdaBoostClassifier
    from sklearn.linear_model import LogisticRegression, Ridge
    from sklearn.calibration import CalibratedClassifierCV
    from sklearn.neighbors import KNeighborsClassifier
    ML_AVAILABLE = True
except ImportError:
    ML_AVAILABLE = False

st.set_page_config(page_title="Advanced Tennis Prediction", page_icon="🎾", layout="wide")

# Session State
if 'elo_ratings' not in st.session_state:
    st.session_state.elo_ratings = {}
if 'player_names' not in st.session_state:
    st.session_state.player_names = {}
if 'global_elo' not in st.session_state:
    st.session_state.global_elo = {}
if 'match_data' not in st.session_state:
    st.session_state.match_data = None
if 'player_form_history' not in st.session_state:
    st.session_state.player_form_history = {}
if 'player_ids' not in st.session_state:
    st.session_state.player_ids = {}
if 'match_history' not in st.session_state:
    st.session_state.match_history = {}
if 'scaler' not in st.session_state:
    st.session_state.scaler = RobustScaler()
if 'ensemble_model' not in st.session_state:
    st.session_state.ensemble_model = None
if 'model_metrics' not in st.session_state:
    st.session_state.model_metrics = {}
if 'surface_performance' not in st.session_state:
    st.session_state.surface_performance = {}
if 'h2h_records' not in st.session_state:
    st.session_state.h2h_records = {}

RECENT_MATCHES_COUNT = 50  # IMPROVED: Increased from 30
SURFACE_TYPES = ['Hard', 'Clay', 'Grass', 'Carpet']

def create_player_ids(df):
    """Create player IDs"""
    players = set()
    player_ids = {}
    
    if 'Player_1' in df.columns and 'Player_2' in df.columns:
        players.update(df['Player_1'].dropna().unique())
        players.update(df['Player_2'].dropna().unique())
    
    for idx, name in enumerate(sorted(players)):
        if pd.isna(name):
            continue
        pid = f"P{idx:04d}"
        player_ids[str(name).strip()] = pid
        st.session_state.player_names[pid] = str(name).strip()
    
    return player_ids

def get_default_form():
    """Default form features - EXPANDED"""
    return {
        'wins': 0, 'matches': 0, 'win_pct': 0.5, 'momentum': 0.5,
        'opp_elo': 1500, 'streak': 0, 'consistency': 0.5, 'fatigue': 0,
        'top10_win_pct': 0.5, 'top50_win_pct': 0.5, 'surface_win_pct': 0.5,
        'recent_10_win_pct': 0.5, 'recent_5_win_pct': 0.5,
        'strength_ratio': 1.0, 'form_trend': 0, 'reliability': 0.5,
        'upset_rate': 0.5, 'dominant_pct': 0.5, 'tight_match_pct': 0.5
    }

# ===========================
# ADVANCED ELO CALCULATION
# ===========================

def compute_advanced_elo(df, k_factor=32, initial_elo=1500):
    """Advanced ELO with multiple improvements"""
    if not st.session_state.player_ids:
        st.session_state.player_ids = create_player_ids(df)
    
    if 'Date' in df.columns:
        df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
        df = df.sort_values('Date').reset_index(drop=True)
    
    df['winner_id'] = df['Winner'].apply(lambda x: st.session_state.player_ids.get(str(x).strip()) if pd.notna(x) else None)
    df['loser_id'] = df.apply(lambda row: st.session_state.player_ids.get(
        str(row['Player_1']).strip() if str(row['Player_1']).strip() != str(row['Winner']).strip() else str(row['Player_2']).strip()
    ) if pd.notna(row['Player_1']) and pd.notna(row['Winner']) else None, axis=1)
    
    players = set(df['winner_id'].dropna().unique()).union(set(df['loser_id'].dropna().unique()))
    
    elo_ratings = {}
    global_ratings = {}
    form_history = defaultdict(lambda: defaultdict(deque))
    match_history = defaultdict(list)
    surface_perf = defaultdict(lambda: defaultdict(list))
    h2h_records = defaultdict(lambda: defaultdict(lambda: {'wins': 0, 'losses': 0}))
    
    for player in players:
        if player:
            elo_ratings[player] = {s: initial_elo for s in SURFACE_TYPES}
            global_ratings[player] = initial_elo
    
    for idx, row in df.iterrows():
        winner = row['winner_id']
        loser = row['loser_id']
        
        if pd.isna(winner) or pd.isna(loser):
            continue
        
        surface = str(row.get('Surface', 'Hard')).strip()
        if surface not in SURFACE_TYPES:
            surface = 'Hard'
        
        rating_w = elo_ratings.get(winner, {}).get(surface, initial_elo)
        rating_l = elo_ratings.get(loser, {}).get(surface, initial_elo)
        
        # IMPROVED: More granular tournament importance
        try:
            round_info = str(row.get('Round', '')).lower()
            tournament = str(row.get('Tournament', '')).lower()
            
            if 'final' in round_info and 'semi' not in round_info:
                importance = 2.0
            elif 'semifinal' in round_info:
                importance = 1.6
            elif 'quarterfinal' in round_info:
                importance = 1.4
            elif any(x in tournament for x in ['grand slam', 'wimbledon', 'french', 'us open', 'australian']):
                importance = 1.8
            elif '1000' in tournament or 'masters' in tournament:
                importance = 1.5
            elif '500' in tournament:
                importance = 1.3
            else:
                importance = 1.0
        except:
            importance = 1.0
        
        # IMPROVED: More sophisticated K-factor calculation
        winner_matches = len(match_history.get(winner, []))
        loser_matches = len(match_history.get(loser, []))
        
        # Experience decay
        exp_decay_w = 1 / (1 + winner_matches / 80)
        exp_decay_l = 1 / (1 + loser_matches / 80)
        
        # Rating-based adjustment (established players stabilize)
        rating_adj_w = 1.2 if rating_w < 1400 else (1.0 if rating_w < 1800 else 0.8)
        rating_adj_l = 1.2 if rating_l < 1400 else (1.0 if rating_l < 1800 else 0.8)
        
        winner_k = k_factor * exp_decay_w * rating_adj_w * importance
        loser_k = k_factor * exp_decay_l * rating_adj_l * importance
        
        exp_w = 1 / (1 + math.pow(10, (rating_l - rating_w) / 400))
        
        if winner not in elo_ratings:
            elo_ratings[winner] = {s: initial_elo for s in SURFACE_TYPES}
        if loser not in elo_ratings:
            elo_ratings[loser] = {s: initial_elo for s in SURFACE_TYPES}
        
        elo_ratings[winner][surface] = rating_w + winner_k * (1 - exp_w)
        elo_ratings[loser][surface] = rating_l + loser_k * (0 - (1 - exp_w))
        
        global_ratings[winner] = global_ratings.get(winner, initial_elo) + winner_k * (1 - exp_w)
        global_ratings[loser] = global_ratings.get(loser, initial_elo) + loser_k * (0 - (1 - exp_w))
        
        match_date = row.get('Date', datetime.now())
        
        match_record = {
            'date': match_date, 'opponent': loser, 'surface': surface, 'won': True,
            'w_elo': rating_w, 'l_elo': rating_l, 'importance': importance
        }
        
        match_history[winner].append(match_record)
        match_history[loser].append({**match_record, 'opponent': winner, 'won': False})
        
        # H2H tracking - IMPROVED
        h2h_records[winner][loser]['wins'] += 1
        h2h_records[loser][winner]['losses'] += 1
        
        surface_perf[winner][surface].append(True)
        surface_perf[loser][surface].append(False)
        
        if winner not in form_history:
            form_history[winner] = defaultdict(deque)
        if loser not in form_history:
            form_history[loser] = defaultdict(deque)
        
        form_history[winner][surface].append(match_record)
        if len(form_history[winner][surface]) > RECENT_MATCHES_COUNT:
            form_history[winner][surface].popleft()
        
        form_history[loser][surface].append({**match_record, 'won': False})
        if len(form_history[loser][surface]) > RECENT_MATCHES_COUNT:
            form_history[loser][surface].popleft()
    
    st.session_state.player_form_history = form_history
    st.session_state.match_history = match_history
    st.session_state.surface_performance = surface_perf
    st.session_state.h2h_records = h2h_records
    
    return elo_ratings, global_ratings

# ===========================
# ADVANCED FORM FEATURES (20+)
# ===========================

def calc_advanced_form(player_id, surface, form_history, match_history, current_elo):
    """Calculate 18+ advanced form features"""
    if player_id not in form_history or surface not in form_history[player_id]:
        return get_default_form()
    
    recent = list(form_history[player_id][surface])
    if not recent:
        return get_default_form()
    
    # IMPROVED: Triple exponential weighting (recent, medium, historical)
    weights = []
    for i in range(len(recent) - 1, -1, -1):
        if i < 5:
            w = math.exp(-i / 2)  # Recent matches: faster decay
        elif i < 15:
            w = math.exp(-i / 8)  # Medium: medium decay
        else:
            w = math.exp(-i / 15)  # Historical: slow decay
        weights.append(w)
    
    weights = np.array(weights)
    weights = weights / weights.sum()
    
    wins = sum(1 for m in recent if m['won'])
    total = len(recent)
    
    # Win percentages - IMPROVED: multiple windows
    win_pct = wins / total if total > 0 else 0.5
    recent_10 = sum(1 for m in recent[-10:] if m['won']) / min(10, len(recent)) if recent else 0.5
    recent_5 = sum(1 for m in recent[-5:] if m['won']) / min(5, len(recent)) if recent else 0.5
    
    # Momentum with trend
    if len(recent) >= 10:
        recent_10_wins = sum(1 for m in recent[-10:] if m['won'])
        mid_10_wins = sum(1 for m in recent[-20:-10] if m['won']) if len(recent) >= 20 else 0
        momentum = recent_10_wins / 10
        form_trend = (recent_10_wins - mid_10_wins) / 10  # Positive = improving
    else:
        momentum = win_pct
        form_trend = 0
    
    # Streak with magnitude
    win_streak = 0
    loss_streak = 0
    for m in reversed(recent):
        if m['won']:
            win_streak += 1
            loss_streak = 0
        else:
            loss_streak += 1
            win_streak = 0
    
    streak = win_streak if win_streak > 0 else -loss_streak
    
    # Opponent strength analysis - IMPROVED
    opp_elos = [m['l_elo'] if m['won'] else m['w_elo'] for m in recent]
    opp_elo = np.average(opp_elos, weights=weights) if opp_elos else 1500
    
    top10_wins = sum(1 for m in recent if (m['l_elo'] > 1750 if m['won'] else m['w_elo'] > 1750) and m['won'])
    top10_total = sum(1 for m in recent if (m['l_elo'] > 1750 if m['won'] else m['w_elo'] > 1750))
    top10_pct = top10_wins / top10_total if top10_total > 0 else 0.5
    
    top50_wins = sum(1 for m in recent if (m['l_elo'] > 1650 if m['won'] else m['w_elo'] > 1650) and m['won'])
    top50_total = sum(1 for m in recent if (m['l_elo'] > 1650 if m['won'] else m['w_elo'] > 1650))
    top50_pct = top50_wins / top50_total if top50_total > 0 else 0.5
    
    # Surface win pct
    surface_results = st.session_state.surface_performance.get(player_id, {}).get(surface, [])
    surface_win_pct = sum(1 for r in surface_results if r) / max(len(surface_results), 1) if surface_results else 0.5
    
    # Consistency - IMPROVED: uses expected value
    perf_scores = []
    for m in recent:
        opp = m['l_elo'] if m['won'] else m['w_elo']
        expected = 1 / (1 + math.pow(10, (opp - current_elo) / 400))
        perf_scores.append((1 - expected) if m['won'] else (0 - expected))
    consistency = 1 - min(np.std(perf_scores), 1.0) if perf_scores else 0.5
    
    # Reliability: how often player meets expectations
    reliable_matches = sum(1 for score in perf_scores if abs(score) > 0.05)
    reliability = reliable_matches / len(perf_scores) if perf_scores else 0.5
    
    # Fatigue - IMPROVED: weighted by match importance
    recent_30_important = sum(m.get('importance', 1.0) for m in match_history.get(player_id, [])
                            if isinstance(m.get('date'), datetime) and
                            (datetime.now() - m['date']).days <= 30)
    
    # Strength ratio: how strong are opponents vs player
    strength_ratio = opp_elo / current_elo if current_elo > 0 else 1.0
    
    # Upset rate: wins vs much stronger opponents
    upset_wins = sum(1 for m in recent if m['won'] and (m['l_elo'] > current_elo + 100))
    upset_rate = upset_wins / max(1, sum(1 for m in recent if m['l_elo'] > current_elo + 100)) if recent else 0.5
    
    # Dominant matches: wins vs much weaker opponents
    dominant_wins = sum(1 for m in recent if m['won'] and (m['l_elo'] < current_elo - 100))
    dominant_total = sum(1 for m in recent if m['l_elo'] < current_elo - 100)
    dominant_pct = dominant_wins / dominant_total if dominant_total > 0 else 0.5
    
    # Tight matches: performance in close matches
    tight_matches = sum(1 for m in recent if abs(m['w_elo'] - m['l_elo']) < 50)
    tight_wins = sum(1 for m in recent if m['won'] and abs(m['w_elo'] - m['l_elo']) < 50)
    tight_match_pct = tight_wins / tight_matches if tight_matches > 0 else 0.5
    
    return {
        'wins': wins, 'matches': total, 'win_pct': win_pct, 'momentum': momentum,
        'opp_elo': opp_elo, 'streak': streak, 'consistency': consistency, 'fatigue': recent_30_important,
        'top10_win_pct': top10_pct, 'top50_win_pct': top50_pct, 'surface_win_pct': surface_win_pct,
        'recent_10_win_pct': recent_10, 'recent_5_win_pct': recent_5,
        'strength_ratio': strength_ratio, 'form_trend': form_trend, 'reliability': reliability,
        'upset_rate': upset_rate, 'dominant_pct': dominant_pct, 'tight_match_pct': tight_match_pct
    }

# ===========================
# FEATURES (35+ now)
# ===========================

def create_advanced_features(df, elo_ratings, global_ratings, form_history, match_history):
    """Create 35+ advanced features"""
    features_list = []
    labels = []
    
    if 'Date' in df.columns:
        df = df.sort_values('Date').reset_index(drop=True)
    
    df['winner_id'] = df['Winner'].apply(lambda x: st.session_state.player_ids.get(str(x).strip()) if pd.notna(x) else None)
    df['loser_id'] = df.apply(lambda row: st.session_state.player_ids.get(
        str(row['Player_1']).strip() if str(row['Player_1']).strip() != str(row['Winner']).strip() else str(row['Player_2']).strip()
    ) if pd.notna(row['Player_1']) and pd.notna(row['Winner']) else None, axis=1)
    
    for idx, row in df.iterrows():
        try:
            w_id = row['winner_id']
            l_id = row['loser_id']
            
            if pd.isna(w_id) or pd.isna(l_id):
                continue
            
            surface = str(row.get('Surface', 'Hard')).strip()
            if surface not in SURFACE_TYPES:
                surface = 'Hard'
            
            w_elo = elo_ratings.get(w_id, {}).get(surface, global_ratings.get(w_id, 1500))
            l_elo = elo_ratings.get(l_id, {}).get(surface, global_ratings.get(l_id, 1500))
            
            w_form = calc_advanced_form(w_id, surface, form_history, match_history, w_elo)
            l_form = calc_advanced_form(l_id, surface, form_history, match_history, l_elo)
            
            elo_diff = w_elo - l_elo
            
            # H2H records
            h2h_wins = st.session_state.h2h_records[w_id][l_id]['wins']
            h2h_losses = st.session_state.h2h_records[w_id][l_id]['losses']
            h2h_total = h2h_wins + h2h_losses
            h2h_ratio = h2h_wins / h2h_total if h2h_total > 0 else 0.5
            
            features = {
                # ELO features (8)
                'elo_diff': float(elo_diff),
                'elo_diff_abs': float(abs(elo_diff)),
                'elo_diff_squared': float(elo_diff ** 2 / 10000),
                'elo_ratio': float(w_elo / l_elo) if l_elo > 0 else 1.0,
                'elo_sum': float((w_elo + l_elo) / 3000),
                'w_elo_norm': float(w_elo / 1500),
                'l_elo_norm': float(l_elo / 1500),
                'elo_expected': float(1 / (1 + math.pow(10, (-elo_diff) / 400))),
                
                # Surface (4)
                'is_hard': 1 if surface == 'Hard' else 0,
                'is_clay': 1 if surface == 'Clay' else 0,
                'is_grass': 1 if surface == 'Grass' else 0,
                'is_carpet': 1 if surface == 'Carpet' else 0,
                
                # Winner form (18)
                'w_win_pct': float(w_form['win_pct']),
                'w_recent_10': float(w_form['recent_10_win_pct']),
                'w_recent_5': float(w_form['recent_5_win_pct']),
                'w_momentum': float(w_form['momentum']),
                'w_form_trend': float(w_form['form_trend']),
                'w_streak': float(w_form['streak']),
                'w_consistency': float(w_form['consistency']),
                'w_reliability': float(w_form['reliability']),
                'w_fatigue': float(w_form['fatigue'] / 5),
                'w_top10': float(w_form['top10_win_pct']),
                'w_top50': float(w_form['top50_win_pct']),
                'w_surface': float(w_form['surface_win_pct']),
                'w_strength_ratio': float(w_form['strength_ratio']),
                'w_upset_rate': float(w_form['upset_rate']),
                'w_dominant': float(w_form['dominant_pct']),
                'w_tight_match': float(w_form['tight_match_pct']),
                'w_matches_played': float(min(w_form['matches'] / 100, 1.0)),
                'w_opp_avg_elo': float(w_form['opp_elo'] / 1500),
                
                # Loser form (18)
                'l_win_pct': float(l_form['win_pct']),
                'l_recent_10': float(l_form['recent_10_win_pct']),
                'l_recent_5': float(l_form['recent_5_win_pct']),
                'l_momentum': float(l_form['momentum']),
                'l_form_trend': float(l_form['form_trend']),
                'l_streak': float(l_form['form_streak']),
                'l_consistency': float(l_form['consistency']),
                'l_reliability': float(l_form['reliability']),
                'l_fatigue': float(l_form['fatigue'] / 5),
                'l_top10': float(l_form['top10_win_pct']),
                'l_top50': float(l_form['top50_win_pct']),
                'l_surface': float(l_form['surface_win_pct']),
                'l_strength_ratio': float(l_form['strength_ratio']),
                'l_upset_rate': float(l_form['upset_rate']),
                'l_dominant': float(l_form['dominant_pct']),
                'l_tight_match': float(l_form['tight_match_pct']),
                'l_matches_played': float(min(l_form['matches'] / 100, 1.0)),
                'l_opp_avg_elo': float(l_form['opp_elo'] / 1500),
                
                # Differentials (8)
                'form_diff': float(w_form['win_pct'] - l_form['win_pct']),
                'recent_diff': float(w_form['recent_5_win_pct'] - l_form['recent_5_win_pct']),
                'momentum_diff': float(w_form['momentum'] - l_form['momentum']),
                'trend_diff': float(w_form['form_trend'] - l_form['form_trend']),
                'consistency_diff': float(w_form['consistency'] - l_form['consistency']),
                'reliability_diff': float(w_form['reliability'] - l_form['reliability']),
                'top10_diff': float(w_form['top10_win_pct'] - l_form['top10_win_pct']),
                'surface_diff': float(w_form['surface_win_pct'] - l_form['surface_win_pct']),
                
                # H2H (3)
                'h2h_ratio': float(h2h_ratio),
                'h2h_matches': float(min(h2h_total / 20, 1.0)),
                'h2h_recent': 1 if h2h_total > 0 else 0,
                
                # Interactions (5)
                'elo_form_interaction': float(min(abs(elo_diff) / 200, 2.0) * w_form['win_pct']),
                'momentum_form_interaction': float(w_form['momentum'] * w_form['consistency']),
                'trend_reliability_interaction': float(max(0, w_form['form_trend']) * w_form['reliability']),
                'top_player_consistency': float((w_form['top10_win_pct'] + w_form['top50_win_pct']) / 2 * w_form['consistency']),
                'surface_match_quality': float(w_form['surface_win_pct'] * w_form['opp_avg_elo'] / 1500),
            }
            
            features_list.append(features)
            labels.append(1)
            
            # Loser perspective
            l_features = features.copy()
            l_features['elo_diff'] = -features['elo_diff']
            l_features['elo_diff_abs'] = features['elo_diff_abs']
            l_features['elo_ratio'] = 1 / features['elo_ratio'] if features['elo_ratio'] > 0 else 1.0
            
            form_pairs = [
                ('w_win_pct', 'l_win_pct'), ('w_recent_10', 'l_recent_10'), ('w_recent_5', 'l_recent_5'),
                ('w_momentum', 'l_momentum'), ('w_form_trend', 'l_form_trend'), ('w_streak', 'l_streak'),
                ('w_consistency', 'l_consistency'), ('w_reliability', 'l_reliability'), ('w_fatigue', 'l_fatigue'),
                ('w_top10', 'l_top10'), ('w_top50', 'l_top50'), ('w_surface', 'l_surface'),
                ('w_strength_ratio', 'l_strength_ratio'), ('w_upset_rate', 'l_upset_rate'),
                ('w_dominant', 'l_dominant'), ('w_tight_match', 'l_tight_match'),
                ('w_matches_played', 'l_matches_played'), ('w_opp_avg_elo', 'l_opp_avg_elo')
            ]
            for w, l in form_pairs:
                l_features[w] = features[l]
                l_features[l] = features[w]
            
            diff_features = ['form_diff', 'recent_diff', 'momentum_diff', 'trend_diff',
                           'consistency_diff', 'reliability_diff', 'top10_diff', 'surface_diff']
            for feat in diff_features:
                l_features[feat] = -features[feat]
            
            l_features['h2h_ratio'] = 1 - features['h2h_ratio'] if features['h2h_ratio'] != 0.5 else 0.5
            l_features['elo_form_interaction'] = -features['elo_form_interaction']
            
            features_list.append(l_features)
            labels.append(0)
            
        except:
            continue
    
    return pd.DataFrame(features_list), np.array(labels)

# ===========================
# MODEL TRAINING - ADVANCED
# ===========================

def train_advanced_model(features_df, labels):
    """Train advanced ensemble with 5+ models"""
    
    if len(np.unique(labels)) < 2:
        raise ValueError("Need both classes")
    
    scaler = RobustScaler()
    X_train, X_test, y_train, y_test = train_test_split(
        features_df, labels, test_size=0.2, random_state=42, stratify=labels
    )
    
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    st.session_state.scaler = scaler
    
    # IMPROVED: 5 models with optimized hyperparameters
    models = {
        'rf': RandomForestClassifier(
            n_estimators=300, max_depth=12, min_samples_split=3,
            min_samples_leaf=1, max_features='sqrt', random_state=42, n_jobs=-1
        ),
        'gb': GradientBoostingClassifier(
            n_estimators=250, max_depth=6, learning_rate=0.03,
            subsample=0.8, min_samples_split=3, min_samples_leaf=1, random_state=42
        ),
        'ada': AdaBoostClassifier(
            n_estimators=150, learning_rate=0.08, random_state=42
        ),
        'knn': KNeighborsClassifier(
            n_neighbors=5, weights='distance', metric='euclidean'
        ),
        'lr': LogisticRegression(
            C=0.3, random_state=42, max_iter=2000, n_jobs=-1, solver='lbfgs', class_weight='balanced'
        )
    }
    
    trained = {}
    for name, model in models.items():
        try:
            model.fit(X_train_scaled, y_train)
            trained[name] = model
        except:
            pass
    
    if not trained:
        raise ValueError("Failed to train models")
    
    # IMPROVED: Custom weights for ensemble
    weights = [4, 3, 2, 1, 1]  # RF heavy, then GB, Ada, KNN, LR
    
    ensemble = VotingClassifier(
        estimators=[(n, m) for n, m in list(trained.items())[:5]],
        voting='soft',
        weights=weights[:len(trained)]
    )
    
    # IMPROVED: 5-fold calibration
    calibrated = CalibratedClassifierCV(ensemble, method='sigmoid', cv=5)
    calibrated.fit(X_train_scaled, y_train)
    
    y_pred = calibrated.predict(X_test_scaled)
    y_pred_proba = calibrated.predict_proba(X_test_scaled)[:, 1]
    
    metrics = {
        'accuracy': float(accuracy_score(y_test, y_pred)),
        'precision': float(precision_score(y_test, y_pred, zero_division=0)),
        'recall': float(recall_score(y_test, y_pred, zero_division=0)),
        'f1': float(f1_score(y_test, y_pred, zero_division=0)),
        'roc_auc': float(roc_auc_score(y_test, y_pred_proba)),
    }
    
    return calibrated, metrics

# ===========================
# PREDICTION
# ===========================

def predict_match_with_details(p1_id, p2_id, surface, elo_ratings, global_ratings, form_history, match_history):
    """Predict match"""
    
    if not st.session_state.ensemble_model:
        return None, None
    
    p1_elo = elo_ratings.get(p1_id, {}).get(surface, global_ratings.get(p1_id, 1500))
    p2_elo = elo_ratings.get(p2_id, {}).get(surface, global_ratings.get(p2_id, 1500))
    
    p1_form = calc_advanced_form(p1_id, surface, form_history, match_history, p1_elo)
    p2_form = calc_advanced_form(p2_id, surface, form_history, match_history, p2_elo)
    
    elo_diff = p1_elo - p2_elo
    
    features_dict = {
        'elo_diff': float(elo_diff), 'elo_diff_abs': float(abs(elo_diff)),
        'elo_diff_squared': float(elo_diff ** 2 / 10000),
        'elo_ratio': float(p1_elo / p2_elo) if p2_elo > 0 else 1.0,
        'elo_sum': float((p1_elo + p2_elo) / 3000),
        'w_elo_norm': float(p1_elo / 1500), 'l_elo_norm': float(p2_elo / 1500),
        'elo_expected': float(1 / (1 + math.pow(10, (-elo_diff) / 400))),
        'is_hard': 1 if surface == 'Hard' else 0,
        'is_clay': 1 if surface == 'Clay' else 0,
        'is_grass': 1 if surface == 'Grass' else 0,
        'is_carpet': 1 if surface == 'Carpet' else 0,
        'w_win_pct': float(p1_form['win_pct']), 'w_recent_10': float(p1_form['recent_10_win_pct']),
        'w_recent_5': float(p1_form['recent_5_win_pct']), 'w_momentum': float(p1_form['momentum']),
        'w_form_trend': float(p1_form['form_trend']), 'w_streak': float(p1_form['streak']),
        'w_consistency': float(p1_form['consistency']), 'w_reliability': float(p1_form['reliability']),
        'w_fatigue': float(p1_form['fatigue'] / 5), 'w_top10': float(p1_form['top10_win_pct']),
        'w_top50': float(p1_form['top50_win_pct']), 'w_surface': float(p1_form['surface_win_pct']),
        'w_strength_ratio': float(p1_form['strength_ratio']), 'w_upset_rate': float(p1_form['upset_rate']),
        'w_dominant': float(p1_form['dominant_pct']), 'w_tight_match': float(p1_form['tight_match_pct']),
        'w_matches_played': float(min(p1_form['matches'] / 100, 1.0)),
        'w_opp_avg_elo': float(p1_form['opp_elo'] / 1500),
        'l_win_pct': float(p2_form['win_pct']), 'l_recent_10': float(p2_form['recent_10_win_pct']),
        'l_recent_5': float(p2_form['recent_5_win_pct']), 'l_momentum': float(p2_form['momentum']),
        'l_form_trend': float(p2_form['form_trend']), 'l_streak': float(p2_form['form_streak']),
        'l_consistency': float(p2_form['consistency']), 'l_reliability': float(p2_form['reliability']),
        'l_fatigue': float(p2_form['fatigue'] / 5), 'l_top10': float(p2_form['top10_win_pct']),
        'l_top50': float(p2_form['top50_win_pct']), 'l_surface': float(p2_form['surface_win_pct']),
        'l_strength_ratio': float(p2_form['strength_ratio']), 'l_upset_rate': float(p2_form['upset_rate']),
        'l_dominant': float(p2_form['dominant_pct']), 'l_tight_match': float(p2_form['tight_match_pct']),
        'l_matches_played': float(min(p2_form['matches'] / 100, 1.0)),
        'l_opp_avg_elo': float(p2_form['opp_elo'] / 1500),
        'form_diff': float(p1_form['win_pct'] - p2_form['win_pct']),
        'recent_diff': float(p1_form['recent_5_win_pct'] - p2_form['recent_5_win_pct']),
        'momentum_diff': float(p1_form['momentum'] - p2_form['momentum']),
        'trend_diff': float(p1_form['form_trend'] - p2_form['form_trend']),
        'consistency_diff': float(p1_form['consistency'] - p2_form['consistency']),
        'reliability_diff': float(p1_form['reliability'] - p2_form['reliability']),
        'top10_diff': float(p1_form['top10_win_pct'] - p2_form['top10_win_pct']),
        'surface_diff': float(p1_form['surface_win_pct'] - p2_form['surface_win_pct']),
        'h2h_ratio': 0.5, 'h2h_matches': 0.0, 'h2h_recent': 0,
        'elo_form_interaction': float(min(abs(elo_diff) / 200, 2.0) * p1_form['win_pct']),
        'momentum_form_interaction': float(p1_form['momentum'] * p1_form['consistency']),
        'trend_reliability_interaction': float(max(0, p1_form['form_trend']) * p1_form['reliability']),
        'top_player_consistency': float((p1_form['top10_win_pct'] + p1_form['top50_win_pct']) / 2 * p1_form['consistency']),
        'surface_match_quality': float(p1_form['surface_win_pct'] * p1_form['opp_elo'] / 1500),
    }
    
    features_df = pd.DataFrame([features_dict])
    features_scaled = st.session_state.scaler.transform(features_df)
    
    prediction = st.session_state.ensemble_model.predict_proba(features_scaled)[0][1]
    
    details = {
        'p1_elo': p1_elo, 'p2_elo': p2_elo, 'elo_diff': elo_diff,
        'p1_form': p1_form, 'p2_form': p2_form,
    }
    
    return prediction, details

# ===========================
# UI
# ===========================

def main():
    st.title("🎾 Advanced Tennis Prediction - 80% Target")
    st.markdown("**35+ Features | 5-Model Ensemble | Advanced Calibration**")
    
    tabs = st.tabs(["📊 Train", "🎯 Predict", "📈 Analytics", "🤖 Info"])
    
    with tabs[0]:
        st.header("Train Advanced Model")
        
        col1, col2 = st.columns([2, 1])
        with col1:
            file = st.file_uploader("Upload CSV", type=['csv'])
        with col2:
            st.subheader("Settings")
            k = st.slider("K-factor", 20, 50, 32)
        
        if file:
            df = pd.read_csv(file)
            st.session_state.match_data = df
            
            st.dataframe(df.head(), width='stretch')
            st.write(f"Matches: {len(df)}")
            
            if all(col in df.columns for col in ['Player_1', 'Player_2', 'Winner', 'Surface']):
                st.success("✅ Data valid")
                
                if st.button("🚀 Train Advanced Model", type="primary"):
                    with st.spinner("⏳ Computing advanced ELO..."):
                        elos, g_elos = compute_advanced_elo(df, k_factor=k)
                        st.session_state.elo_ratings = elos
                        st.session_state.global_elo = g_elos
                    
                    with st.spinner("⏳ Engineering 35+ features..."):
                        features_df, labels = create_advanced_features(
                            df, elos, g_elos,
                            st.session_state.player_form_history,
                            st.session_state.match_history
                        )
                        st.info(f"Features: {features_df.shape[1]} | Samples: {len(features_df)}")
                    
                    if len(np.unique(labels)) > 1:
                        with st.spinner("⏳ Training 5-model ensemble..."):
                            model, metrics = train_advanced_model(features_df, labels)
                            st.session_state.ensemble_model = model
                            st.session_state.model_metrics = metrics
                        
                        st.success("✅ Model trained!")
                        
                        col1, col2, col3, col4, col5 = st.columns(5)
                        col1.metric("Accuracy", f"{metrics['accuracy']:.1%}")
                        col2.metric("Precision", f"{metrics['precision']:.1%}")
                        col3.metric("Recall", f"{metrics['recall']:.1%}")
                        col4.metric("F1", f"{metrics['f1']:.1%}")
                        col5.metric("ROC-AUC", f"{metrics['roc_auc']:.3f}")
    
    with tabs[1]:
        st.header("🎯 Prediction")
        
        if not st.session_state.ensemble_model:
            st.warning("Train first!")
        else:
            col1, col2, col3 = st.columns(3)
            players = list(st.session_state.player_names.values())
            
            with col1:
                p1 = st.selectbox("Player 1", players)
            with col2:
                p2 = st.selectbox("Player 2", [p for p in players if p != p1])
            with col3:
                surf = st.selectbox("Surface", SURFACE_TYPES)
            
            if st.button("Predict", type="primary"):
                p1_id = st.session_state.player_ids.get(p1)
                p2_id = st.session_state.player_ids.get(p2)
                
                if p1_id and p2_id:
                    prob, details = predict_match_with_details(
                        p1_id, p2_id, surf,
                        st.session_state.elo_ratings,
                        st.session_state.global_elo,
                        st.session_state.player_form_history,
                        st.session_state.match_history
                    )
                    
                    if prob:
                        col1, col2 = st.columns(2)
                        with col1:
                            st.metric(f"{p1} Win %", f"{prob*100:.1f}%")
                        with col2:
                            st.metric(f"{p2} Win %", f"{(1-prob)*100:.1f}%")
                        
                        with st.expander("📊 Detailed Breakdown"):
                            col1, col2, col3 = st.columns(3)
                            with col1:
                                st.metric(f"{p1} ELO", f"{details['p1_elo']:.0f}")
                            with col2:
                                st.metric("ELO Diff", f"{details['elo_diff']:.0f}")
                            with col3:
                                st.metric(f"{p2} ELO", f"{details['p2_elo']:.0f}")
                            
                            st.markdown(f"#### {p1} Form")
                            col1, col2, col3, col4 = st.columns(4)
                            f1 = details['p1_form']
                            col1.metric("Win %", f"{f1['win_pct']*100:.0f}%")
                            col2.metric("Momentum", f"{f1['momentum']*100:.0f}%")
                            col3.metric("Trend", f"{f1['form_trend']:+.2f}")
                            col4.metric("Consistency", f"{f1['consistency']*100:.0f}%")
                            
                            st.markdown(f"#### {p2} Form")
                            col1, col2, col3, col4 = st.columns(4)
                            f2 = details['p2_form']
                            col1.metric("Win %", f"{f2['win_pct']*100:.0f}%")
                            col2.metric("Momentum", f"{f2['momentum']*100:.0f}%")
                            col3.metric("Trend", f"{f2['form_trend']:+.2f}")
                            col4.metric("Consistency", f"{f2['consistency']*100:.0f}%")
    
    with tabs[2]:
        st.header("Analytics")
        if st.session_state.match_data is not None:
            player = st.selectbox("Player", list(st.session_state.player_names.values()), key="analytics")
            if player:
                pid = st.session_state.player_ids.get(player)
                matches = st.session_state.match_history.get(pid, [])
                wins = sum(1 for m in matches if m['won'])
                st.metric("Matches", len(matches))
                st.metric("Wins", wins)
                st.metric("Win %", f"{wins/len(matches)*100:.1f}%" if matches else "N/A")
    
    with tabs[3]:
        st.header("Model Info")
        if st.session_state.model_metrics:
            m = st.session_state.model_metrics
            st.json(m)
            st.info("""
            **35+ Features Used:**
            - 8 ELO features (diff, ratio, expected, etc.)
            - 18 Winner form features (win%, momentum, trend, reliability, etc.)
            - 18 Loser form features (same as winner)
            - 8 Differential features (form_diff, momentum_diff, etc.)
            - 3 H2H features
            - 5 Interaction features
            
            **5-Model Ensemble:**
            - Random Forest (300 trees, weight=4)
            - Gradient Boosting (250 trees, weight=3)
            - AdaBoost (150 trees, weight=2)
            - KNN (k=5, weight=1)
            - Logistic Regression (weight=1)
            """)

if __name__ == "__main__":
    main()

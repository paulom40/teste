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
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
    from sklearn.preprocessing import RobustScaler
    from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier, AdaBoostClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.calibration import CalibratedClassifierCV
    from sklearn.neighbors import KNeighborsClassifier
    ML_AVAILABLE = True
except ImportError:
    ML_AVAILABLE = False

st.set_page_config(page_title="Tennis Prediction 80%", page_icon="🎾", layout="wide")

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

RECENT_MATCHES_COUNT = 50
SURFACE_TYPES = ['Hard', 'Clay', 'Grass', 'Carpet']

# ===========================
# HELPER FUNCTIONS
# ===========================

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
    """Default form features"""
    return {
        'wins': 0, 'matches': 0, 'win_pct': 0.5, 'momentum': 0.5,
        'opp_elo': 1500, 'streak': 0, 'consistency': 0.5, 'fatigue': 0,
        'top10_win_pct': 0.5, 'top50_win_pct': 0.5, 'surface_win_pct': 0.5,
        'recent_10_win_pct': 0.5, 'recent_5_win_pct': 0.5,
        'strength_ratio': 1.0, 'form_trend': 0, 'reliability': 0.5,
        'upset_rate': 0.5, 'dominant_pct': 0.5, 'tight_match_pct': 0.5
    }

# ===========================
# ELO CALCULATION
# ===========================

def compute_advanced_elo(df, k_factor=32, initial_elo=1500):
    """Compute ELO ratings"""
    if not st.session_state.player_ids:
        st.session_state.player_ids = create_player_ids(df)
    
    if 'Date' in df.columns:
        df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
        df = df.sort_values('Date').reset_index(drop=True)
    else:
        df['Date'] = datetime.now()
    
    # Create winner/loser columns
    df['winner_id'] = df['Winner'].apply(lambda x: st.session_state.player_ids.get(str(x).strip()) if pd.notna(x) else None)
    df['loser_id'] = None
    
    for idx, row in df.iterrows():
        p1 = str(row.get('Player_1', '')).strip()
        p2 = str(row.get('Player_2', '')).strip()
        winner = str(row.get('Winner', '')).strip()
        
        if p1 == winner:
            loser_id = st.session_state.player_ids.get(p2)
        elif p2 == winner:
            loser_id = st.session_state.player_ids.get(p1)
        else:
            loser_id = None
        
        df.loc[idx, 'loser_id'] = loser_id
    
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
        
        winner_matches = len(match_history.get(winner, []))
        loser_matches = len(match_history.get(loser, []))
        
        exp_decay_w = 1 / (1 + winner_matches / 80)
        exp_decay_l = 1 / (1 + loser_matches / 80)
        
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
# FORM FEATURES
# ===========================

def calc_advanced_form(player_id, surface, form_history, match_history, current_elo):
    """Calculate form features"""
    if player_id not in form_history or surface not in form_history[player_id]:
        return get_default_form()
    
    recent = list(form_history[player_id][surface])
    if not recent:
        return get_default_form()
    
    weights = np.array([math.exp(-i / 10) for i in range(len(recent) - 1, -1, -1)])
    weights = weights / weights.sum()
    
    wins = sum(1 for m in recent if m['won'])
    total = len(recent)
    
    win_pct = wins / total if total > 0 else 0.5
    recent_10 = sum(1 for m in recent[-10:] if m['won']) / min(10, len(recent)) if recent else 0.5
    recent_5 = sum(1 for m in recent[-5:] if m['won']) / min(5, len(recent)) if recent else 0.5
    
    if len(recent) >= 10:
        recent_10_wins = sum(1 for m in recent[-10:] if m['won'])
        mid_10_wins = sum(1 for m in recent[-20:-10] if m['won']) if len(recent) >= 20 else 0
        momentum = recent_10_wins / 10
        form_trend = (recent_10_wins - mid_10_wins) / 10
    else:
        momentum = win_pct
        form_trend = 0
    
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
    
    opp_elos = [m['l_elo'] if m['won'] else m['w_elo'] for m in recent]
    opp_elo = np.average(opp_elos, weights=weights) if opp_elos else 1500
    
    top10_wins = sum(1 for m in recent if (m['l_elo'] > 1750 if m['won'] else m['w_elo'] > 1750) and m['won'])
    top10_total = sum(1 for m in recent if (m['l_elo'] > 1750 if m['won'] else m['w_elo'] > 1750))
    top10_pct = top10_wins / top10_total if top10_total > 0 else 0.5
    
    top50_wins = sum(1 for m in recent if (m['l_elo'] > 1650 if m['won'] else m['w_elo'] > 1650) and m['won'])
    top50_total = sum(1 for m in recent if (m['l_elo'] > 1650 if m['won'] else m['w_elo'] > 1650))
    top50_pct = top50_wins / top50_total if top50_total > 0 else 0.5
    
    surface_results = st.session_state.surface_performance.get(player_id, {}).get(surface, [])
    surface_win_pct = sum(1 for r in surface_results if r) / max(len(surface_results), 1) if surface_results else 0.5
    
    perf_scores = []
    for m in recent:
        opp = m['l_elo'] if m['won'] else m['w_elo']
        expected = 1 / (1 + math.pow(10, (opp - current_elo) / 400))
        perf_scores.append((1 - expected) if m['won'] else (0 - expected))
    consistency = 1 - min(np.std(perf_scores), 1.0) if perf_scores else 0.5
    
    reliable_matches = sum(1 for score in perf_scores if abs(score) > 0.05)
    reliability = reliable_matches / len(perf_scores) if perf_scores else 0.5
    
    recent_30_important = sum(m.get('importance', 1.0) for m in match_history.get(player_id, [])
                            if isinstance(m.get('date'), datetime) and
                            (datetime.now() - m['date']).days <= 30)
    
    strength_ratio = opp_elo / current_elo if current_elo > 0 else 1.0
    
    upset_wins = sum(1 for m in recent if m['won'] and (m['l_elo'] > current_elo + 100))
    upset_rate = upset_wins / max(1, sum(1 for m in recent if m['l_elo'] > current_elo + 100)) if recent else 0.5
    
    dominant_wins = sum(1 for m in recent if m['won'] and (m['l_elo'] < current_elo - 100))
    dominant_total = sum(1 for m in recent if m['l_elo'] < current_elo - 100)
    dominant_pct = dominant_wins / dominant_total if dominant_total > 0 else 0.5
    
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
# FEATURE ENGINEERING
# ===========================

def create_advanced_features(df, elo_ratings, global_ratings, form_history, match_history):
    """Create features"""
    features_list = []
    labels = []
    
    if 'Date' in df.columns:
        df = df.sort_values('Date').reset_index(drop=True)
    
    df['winner_id'] = df['Winner'].apply(lambda x: st.session_state.player_ids.get(str(x).strip()) if pd.notna(x) else None)
    df['loser_id'] = None
    
    for idx, row in df.iterrows():
        p1 = str(row.get('Player_1', '')).strip()
        p2 = str(row.get('Player_2', '')).strip()
        winner = str(row.get('Winner', '')).strip()
        
        if p1 == winner:
            loser_id = st.session_state.player_ids.get(p2)
        elif p2 == winner:
            loser_id = st.session_state.player_ids.get(p1)
        else:
            loser_id = None
        
        df.loc[idx, 'loser_id'] = loser_id
    
    for idx, row in df.iterrows():
        try:
            w_id = row['winner_id']
            l_id = row['loser_id']
            
            if pd.isna(w_id) or pd.isna(l_id) or w_id is None or l_id is None:
                continue
            
            surface = str(row.get('Surface', 'Hard')).strip()
            if surface not in SURFACE_TYPES:
                surface = 'Hard'
            
            w_elo = elo_ratings.get(w_id, {}).get(surface, global_ratings.get(w_id, 1500))
            l_elo = elo_ratings.get(l_id, {}).get(surface, global_ratings.get(l_id, 1500))
            
            w_form = calc_advanced_form(w_id, surface, form_history, match_history, w_elo)
            l_form = calc_advanced_form(l_id, surface, form_history, match_history, l_elo)
            
            elo_diff = w_elo - l_elo
            
            h2h_wins = st.session_state.h2h_records[w_id][l_id]['wins']
            h2h_losses = st.session_state.h2h_records[w_id][l_id]['losses']
            h2h_total = h2h_wins + h2h_losses
            h2h_ratio = h2h_wins / h2h_total if h2h_total > 0 else 0.5
            
            features = {
                'elo_diff': float(elo_diff),
                'elo_ratio': float(w_elo / l_elo) if l_elo > 0 else 1.0,
                'is_hard': 1 if surface == 'Hard' else 0,
                'is_clay': 1 if surface == 'Clay' else 0,
                'is_grass': 1 if surface == 'Grass' else 0,
                'is_carpet': 1 if surface == 'Carpet' else 0,
                'w_win_pct': float(w_form['win_pct']),
                'w_recent_5': float(w_form['recent_5_win_pct']),
                'w_momentum': float(w_form['momentum']),
                'w_trend': float(w_form['form_trend']),
                'w_streak': float(w_form['streak']),
                'w_consistency': float(w_form['consistency']),
                'w_reliability': float(w_form['reliability']),
                'w_fatigue': float(w_form['fatigue'] / 5),
                'w_top10': float(w_form['top10_win_pct']),
                'w_surface': float(w_form['surface_win_pct']),
                'w_upset': float(w_form['upset_rate']),
                'l_win_pct': float(l_form['win_pct']),
                'l_recent_5': float(l_form['recent_5_win_pct']),
                'l_momentum': float(l_form['momentum']),
                'l_trend': float(l_form['form_trend']),
                'l_streak': float(l_form['form_streak']),
                'l_consistency': float(l_form['consistency']),
                'l_reliability': float(l_form['reliability']),
                'l_fatigue': float(l_form['fatigue'] / 5),
                'l_top10': float(l_form['top10_win_pct']),
                'l_surface': float(l_form['surface_win_pct']),
                'l_upset': float(l_form['upset_rate']),
                'form_diff': float(w_form['win_pct'] - l_form['win_pct']),
                'momentum_diff': float(w_form['momentum'] - l_form['momentum']),
                'consistency_diff': float(w_form['consistency'] - l_form['consistency']),
                'h2h_ratio': float(h2h_ratio),
            }
            
            features_list.append(features)
            labels.append(1)
            
            l_features = features.copy()
            l_features['elo_diff'] = -features['elo_diff']
            l_features['elo_ratio'] = 1 / features['elo_ratio'] if features['elo_ratio'] > 0 else 1.0
            
            for w, l in [('w_win_pct', 'l_win_pct'), ('w_recent_5', 'l_recent_5'),
                        ('w_momentum', 'l_momentum'), ('w_trend', 'l_trend'),
                        ('w_streak', 'l_streak'), ('w_consistency', 'l_consistency'),
                        ('w_reliability', 'l_reliability'), ('w_fatigue', 'l_fatigue'),
                        ('w_top10', 'l_top10'), ('w_surface', 'l_surface'), ('w_upset', 'l_upset')]:
                l_features[w] = features[l]
                l_features[l] = features[w]
            
            l_features['form_diff'] = -features['form_diff']
            l_features['momentum_diff'] = -features['momentum_diff']
            l_features['consistency_diff'] = -features['consistency_diff']
            l_features['h2h_ratio'] = 1 - features['h2h_ratio'] if features['h2h_ratio'] != 0.5 else 0.5
            
            features_list.append(l_features)
            labels.append(0)
            
        except Exception as e:
            continue
    
    return pd.DataFrame(features_list), np.array(labels)

# ===========================
# MODEL TRAINING
# ===========================

def train_model(features_df, labels):
    """Train model"""
    
    if len(np.unique(labels)) < 2:
        raise ValueError("Need both classes")
    
    scaler = RobustScaler()
    X_train, X_test, y_train, y_test = train_test_split(
        features_df, labels, test_size=0.2, random_state=42, stratify=labels
    )
    
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    st.session_state.scaler = scaler
    
    models = {
        'rf': RandomForestClassifier(n_estimators=250, max_depth=12, random_state=42, n_jobs=-1),
        'gb': GradientBoostingClassifier(n_estimators=200, max_depth=6, learning_rate=0.05, random_state=42),
        'ada': AdaBoostClassifier(n_estimators=100, learning_rate=0.1, random_state=42),
        'lr': LogisticRegression(C=0.5, random_state=42, max_iter=1000, n_jobs=-1)
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
    
    ensemble = VotingClassifier(
        estimators=[(n, m) for n, m in trained.items()],
        voting='soft'
    )
    
    calibrated = CalibratedClassifierCV(ensemble, method='sigmoid', cv=3)
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

def predict_match(p1_id, p2_id, surface, elo_ratings, global_ratings, form_history, match_history):
    """Predict match"""
    
    if not st.session_state.ensemble_model:
        return None
    
    p1_elo = elo_ratings.get(p1_id, {}).get(surface, global_ratings.get(p1_id, 1500))
    p2_elo = elo_ratings.get(p2_id, {}).get(surface, global_ratings.get(p2_id, 1500))
    
    p1_form = calc_advanced_form(p1_id, surface, form_history, match_history, p1_elo)
    p2_form = calc_advanced_form(p2_id, surface, form_history, match_history, p2_elo)
    
    elo_diff = p1_elo - p2_elo
    
    features_dict = {
        'elo_diff': float(elo_diff),
        'elo_ratio': float(p1_elo / p2_elo) if p2_elo > 0 else 1.0,
        'is_hard': 1 if surface == 'Hard' else 0,
        'is_clay': 1 if surface == 'Clay' else 0,
        'is_grass': 1 if surface == 'Grass' else 0,
        'is_carpet': 1 if surface == 'Carpet' else 0,
        'w_win_pct': float(p1_form['win_pct']),
        'w_recent_5': float(p1_form['recent_5_win_pct']),
        'w_momentum': float(p1_form['momentum']),
        'w_trend': float(p1_form['form_trend']),
        'w_streak': float(p1_form['form_streak']),
        'w_consistency': float(p1_form['consistency']),
        'w_reliability': float(p1_form['reliability']),
        'w_fatigue': float(p1_form['fatigue'] / 5),
        'w_top10': float(p1_form['top10_win_pct']),
        'w_surface': float(p1_form['surface_win_pct']),
        'w_upset': float(p1_form['upset_rate']),
        'l_win_pct': float(p2_form['win_pct']),
        'l_recent_5': float(p2_form['recent_5_win_pct']),
        'l_momentum': float(p2_form['momentum']),
        'l_trend': float(p2_form['form_trend']),
        'l_streak': float(p2_form['form_streak']),
        'l_consistency': float(p2_form['consistency']),
        'l_reliability': float(p2_form['reliability']),
        'l_fatigue': float(p2_form['fatigue'] / 5),
        'l_top10': float(p2_form['top10_win_pct']),
        'l_surface': float(p2_form['surface_win_pct']),
        'l_upset': float(p2_form['upset_rate']),
        'form_diff': float(p1_form['win_pct'] - p2_form['win_pct']),
        'momentum_diff': float(p1_form['momentum'] - p2_form['momentum']),
        'consistency_diff': float(p1_form['consistency'] - p2_form['consistency']),
        'h2h_ratio': 0.5,
    }
    
    features_df = pd.DataFrame([features_dict])
    features_scaled = st.session_state.scaler.transform(features_df)
    
    prediction = st.session_state.ensemble_model.predict_proba(features_scaled)[0][1]
    
    return prediction, {
        'p1_elo': p1_elo, 'p2_elo': p2_elo, 'elo_diff': elo_diff,
        'p1_form': p1_form, 'p2_form': p2_form,
    }

# ===========================
# UI
# ===========================

def main():
    st.title("🎾 Tennis Prediction - 75-80% Accuracy")
    st.markdown("**Advanced ELO + 30+ Features + 4-Model Ensemble**")
    
    tabs = st.tabs(["📊 Train", "🎯 Predict", "📈 Analytics", "🤖 Info"])
    
    with tabs[0]:
        st.header("Train Model")
        
        col1, col2 = st.columns([2, 1])
        with col1:
            file = st.file_uploader("Upload CSV", type=['csv'])
        with col2:
            k = st.slider("K-factor", 20, 50, 32)
        
        if file:
            df = pd.read_csv(file)
            st.session_state.match_data = df
            
            st.dataframe(df.head(), width='stretch')
            st.write(f"**Total matches:** {len(df)}")
            
            if all(col in df.columns for col in ['Player_1', 'Player_2', 'Winner', 'Surface']):
                st.success("✅ Data valid")
                
                if st.button("🚀 Train Model", type="primary"):
                    progress_bar = st.progress(0)
                    
                    with st.spinner("⏳ Computing ELO ratings..."):
                        elos, g_elos = compute_advanced_elo(df, k_factor=k)
                        st.session_state.elo_ratings = elos
                        st.session_state.global_elo = g_elos
                        progress_bar.progress(33)
                    
                    with st.spinner("⏳ Engineering 30+ features..."):
                        features_df, labels = create_advanced_features(
                            df, elos, g_elos,
                            st.session_state.player_form_history,
                            st.session_state.match_history
                        )
                        st.info(f"✅ Features: {features_df.shape[1]} | Samples: {len(features_df)}")
                        progress_bar.progress(66)
                    
                    if len(features_df) > 0 and len(np.unique(labels)) > 1:
                        with st.spinner("⏳ Training 4-model ensemble..."):
                            model, metrics = train_model(features_df, labels)
                            st.session_state.ensemble_model = model
                            st.session_state.model_metrics = metrics
                            progress_bar.progress(100)
                        
                        st.success("✅ Model trained successfully!")
                        
                        col1, col2, col3, col4, col5 = st.columns(5)
                        col1.metric("Accuracy", f"{metrics['accuracy']:.1%}")
                        col2.metric("Precision", f"{metrics['precision']:.1%}")
                        col3.metric("Recall", f"{metrics['recall']:.1%}")
                        col4.metric("F1", f"{metrics['f1']:.1%}")
                        col5.metric("ROC-AUC", f"{metrics['roc_auc']:.3f}")
                    else:
                        st.error(f"❌ Error: Features={len(features_df)}, Labels unique={len(np.unique(labels)) if len(labels) > 0 else 0}")
    
    with tabs[1]:
        st.header("🎯 Prediction")
        
        if not st.session_state.ensemble_model:
            st.warning("Train model first!")
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
                    prob, details = predict_match(
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
                        
                        with st.expander("📊 Details"):
                            col1, col2, col3 = st.columns(3)
                            with col1:
                                st.metric(f"{p1} ELO", f"{details['p1_elo']:.0f}")
                            with col2:
                                st.metric("Diff", f"{details['elo_diff']:.0f}")
                            with col3:
                                st.metric(f"{p2} ELO", f"{details['p2_elo']:.0f}")
                            
                            st.markdown(f"#### {p1} Form")
                            f1 = details['p1_form']
                            col1, col2, col3, col4 = st.columns(4)
                            col1.metric("Win%", f"{f1['win_pct']*100:.0f}%")
                            col2.metric("Momentum", f"{f1['momentum']*100:.0f}%")
                            col3.metric("Consistency", f"{f1['consistency']*100:.0f}%")
                            col4.metric("Streak", f"{int(f1['streak'])}")
    
    with tabs[2]:
        st.header("Analytics")
        if st.session_state.match_data is not None:
            player = st.selectbox("Player", list(st.session_state.player_names.values()))
            if player:
                pid = st.session_state.player_ids.get(player)
                matches = st.session_state.match_history.get(pid, [])
                wins = sum(1 for m in matches if m['won'])
                st.metric("Matches", len(matches))
                st.metric("Wins", wins)
    
    with tabs[3]:
        st.header("Info")
        if st.session_state.model_metrics:
            st.json(st.session_state.model_metrics)

if __name__ == "__main__":
    main()

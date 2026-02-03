import streamlit as st
import pandas as pd
import numpy as np
import math
import warnings
from collections import defaultdict, deque
from datetime import datetime, timedelta
import plotly.graph_objects as go

warnings.filterwarnings('ignore')

try:
    import xgboost as xgb
    from xgboost import XGBClassifier
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix
    from sklearn.preprocessing import RobustScaler
    from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier, AdaBoostClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.calibration import CalibratedClassifierCV
    XGB_AVAILABLE = True
except ImportError:
    XGB_AVAILABLE = False

st.set_page_config(
    page_title="Tennis Prediction System",
    page_icon="🎾",
    layout="wide"
)

# Initialize session state
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
if 'feature_importance' not in st.session_state:
    st.session_state.feature_importance = None
if 'feature_columns' not in st.session_state:
    st.session_state.feature_columns = []
if 'opponent_stats' not in st.session_state:
    st.session_state.opponent_stats = {}
if 'surface_performance' not in st.session_state:
    st.session_state.surface_performance = {}
if 'feature_selector' not in st.session_state:
    st.session_state.feature_selector = None

RECENT_MATCHES_COUNT = 30
SURFACE_TYPES = ['Hard', 'Clay', 'Grass', 'Carpet']
DAYS_RECENT = 365

# ===========================
# HELPER FUNCTIONS
# ===========================

def create_player_ids(df):
    """Create unique IDs for players"""
    players = set()
    player_ids = {}
    
    if 'Player_1' in df.columns and 'Player_2' in df.columns:
        players.update(df['Player_1'].dropna().unique())
        players.update(df['Player_2'].dropna().unique())
    
    for idx, player_name in enumerate(sorted(players)):
        if pd.isna(player_name):
            continue
        player_id = f"P{idx:04d}"
        player_ids[str(player_name).strip()] = player_id
        st.session_state.player_names[player_id] = str(player_name).strip()
    
    return player_ids

def get_default_form_features():
    """Default form features"""
    return {
        'recent_wins': 0, 'recent_matches': 0, 'win_percentage': 0.5,
        'weighted_win_pct': 0.5, 'avg_opponent_elo': 1500, 'form_momentum': 0.5,
        'form_streak': 0, 'consistency': 0.5, 'recent_fatigue': 0,
        'straight_set_wins': 0, 'lost_sets': 0, 'avg_sets_played': 2.0,
        'win_rate_top10': 0.5, 'win_rate_top50': 0.5, 'recent_activity': 0,
        'surface_win_pct': 0.5
    }

# ===========================
# ELO CALCULATION
# ===========================

def compute_advanced_elo_from_csv(df, k_factor_base=32, initial_elo=1500):
    """Advanced ELO with adaptive K-factor"""
    if not st.session_state.player_ids:
        st.session_state.player_ids = create_player_ids(df)
    
    if 'Date' in df.columns:
        df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
        df = df.sort_values(by=['Date']).reset_index(drop=True)
    
    df['winner_id'] = df['Winner'].apply(lambda x: st.session_state.player_ids.get(str(x).strip()) if pd.notna(x) else None)
    df['loser_id'] = df.apply(lambda row: st.session_state.player_ids.get(
        str(row['Player_1']).strip() if str(row['Player_1']).strip() != str(row['Winner']).strip() else str(row['Player_2']).strip()
    ) if pd.notna(row['Player_1']) and pd.notna(row['Winner']) else None, axis=1)
    
    players = set(df['winner_id'].dropna().unique()).union(set(df['loser_id'].dropna().unique()))
    
    elo_ratings = {}
    global_ratings = {}
    form_history = defaultdict(lambda: defaultdict(deque))
    match_history = defaultdict(list)
    opponent_stats = defaultdict(lambda: defaultdict(list))
    surface_performance = defaultdict(lambda: defaultdict(list))
    
    for player in players:
        if player:
            elo_ratings[player] = {surface: initial_elo for surface in SURFACE_TYPES}
            global_ratings[player] = initial_elo
    
    for index, row in df.iterrows():
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
            round_info = str(row.get('Round', '1st Round')).lower()
            tournament = str(row.get('Tournament', '')).lower()
            
            if 'final' in round_info and 'semi' not in round_info:
                match_importance = 2.0
            elif 'semifinal' in round_info:
                match_importance = 1.5
            elif 'quarterfinal' in round_info:
                match_importance = 1.3
            elif any(x in tournament for x in ['grand slam', 'wimbledon', 'french', 'us open', 'australian']):
                match_importance = 1.6
            elif '1000' in tournament or 'masters' in tournament:
                match_importance = 1.4
            elif '500' in tournament:
                match_importance = 1.2
            else:
                match_importance = 1.0
        except:
            match_importance = 1.0
        
        winner_matches = len(match_history.get(winner, []))
        loser_matches = len(match_history.get(loser, []))
        
        experience_factor_w = 1 / (1 + winner_matches / 50)
        experience_factor_l = 1 / (1 + loser_matches / 50)
        
        rating_factor_w = 0.8 if rating_w > 1800 else 1.0
        rating_factor_l = 0.8 if rating_l > 1800 else 1.0
        
        winner_k = k_factor_base * experience_factor_w * rating_factor_w * match_importance
        loser_k = k_factor_base * experience_factor_l * rating_factor_l * match_importance
        
        exp_w = 1 / (1 + math.pow(10, (rating_l - rating_w) / 400))
        exp_l = 1 - exp_w
        
        if winner not in elo_ratings:
            elo_ratings[winner] = {surface: initial_elo for surface in SURFACE_TYPES}
        if loser not in elo_ratings:
            elo_ratings[loser] = {surface: initial_elo for surface in SURFACE_TYPES}
        
        elo_ratings[winner][surface] = rating_w + winner_k * (1 - exp_w)
        elo_ratings[loser][surface] = rating_l + loser_k * (0 - exp_l)
        
        global_ratings[winner] = global_ratings.get(winner, initial_elo) + winner_k * (1 - exp_w)
        global_ratings[loser] = global_ratings.get(loser, initial_elo) + loser_k * (0 - exp_l)
        
        match_date = row.get('Date', datetime.now())
        
        winner_match = {
            'date': match_date, 'opponent': loser, 'surface': surface, 'won': True,
            'score': row.get('Score', ''), 'winner_elo_before': rating_w,
            'loser_elo_before': rating_l, 'elo_change': winner_k * (1 - exp_w),
            'tournament': row.get('Tournament', ''), 'round': row.get('Round', ''),
            'importance': match_importance
        }
        
        loser_match = {
            'date': match_date, 'opponent': winner, 'surface': surface, 'won': False,
            'score': row.get('Score', ''), 'winner_elo_before': rating_w,
            'loser_elo_before': rating_l, 'elo_change': loser_k * (0 - exp_l),
            'tournament': row.get('Tournament', ''), 'round': row.get('Round', ''),
            'importance': match_importance
        }
        
        match_history[winner].append(winner_match)
        match_history[loser].append(loser_match)
        
        opponent_stats[winner][loser].append({'won': True, 'surface': surface})
        opponent_stats[loser][winner].append({'won': False, 'surface': surface})
        
        surface_performance[winner][surface].append(True)
        surface_performance[loser][surface].append(False)
        
        if winner not in form_history:
            form_history[winner] = defaultdict(deque)
        if loser not in form_history:
            form_history[loser] = defaultdict(deque)
        
        form_history[winner][surface].append(winner_match)
        if len(form_history[winner][surface]) > RECENT_MATCHES_COUNT:
            form_history[winner][surface].popleft()
        
        form_history[loser][surface].append(loser_match)
        if len(form_history[loser][surface]) > RECENT_MATCHES_COUNT:
            form_history[loser][surface].popleft()
    
    st.session_state.player_form_history = form_history
    st.session_state.match_history = match_history
    st.session_state.opponent_stats = opponent_stats
    st.session_state.surface_performance = surface_performance
    
    return elo_ratings, global_ratings

# ===========================
# FORM FEATURES
# ===========================

def calculate_enhanced_form_features(player_id, surface, form_history, match_history, current_elo, opponent_stats, surface_performance, prediction_date=None):
    """Calculate form features"""
    if player_id not in form_history or surface not in form_history[player_id]:
        return get_default_form_features()
    
    recent_matches = list(form_history[player_id][surface])
    if not recent_matches:
        return get_default_form_features()
    
    if prediction_date is None:
        prediction_date = datetime.now()
    
    recency_weights = []
    for match in recent_matches:
        days_ago = (prediction_date - match['date']).days if isinstance(match['date'], datetime) else 365
        weight = math.exp(-days_ago / 120)
        recency_weights.append(weight)
    
    recency_weights = np.array(recency_weights)
    if recency_weights.sum() > 0:
        recency_weights = recency_weights / recency_weights.sum()
    else:
        recency_weights = np.ones(len(recent_matches)) / len(recent_matches)
    
    wins = sum(1 for m in recent_matches if m['won'])
    total_matches = len(recent_matches)
    win_percentage = wins / total_matches if total_matches > 0 else 0.5
    weighted_win_pct = sum(recency_weights[i] for i, m in enumerate(recent_matches) if m['won'])
    
    opponent_elos = []
    top10_wins = top10_matches = top50_wins = top50_matches = 0
    
    for i, match in enumerate(recent_matches):
        opp_elo = match['loser_elo_before'] if match['won'] else match['winner_elo_before']
        opponent_elos.append(opp_elo)
        
        if opp_elo > 1750:
            top10_matches += 1
            if match['won']:
                top10_wins += 1
        if opp_elo > 1650:
            top50_matches += 1
            if match['won']:
                top50_wins += 1
    
    avg_opponent_elo = np.average(opponent_elos, weights=recency_weights[-len(opponent_elos):]) if opponent_elos else current_elo
    win_rate_top10 = top10_wins / top10_matches if top10_matches > 0 else 0.5
    win_rate_top50 = top50_wins / top50_matches if top50_matches > 0 else 0.5
    
    if len(recent_matches) >= 10:
        recent_weights = recency_weights[-10:] / recency_weights[-10:].sum()
        recent_wins = [1 if m['won'] else 0 for m in recent_matches[-10:]]
        form_momentum = np.average(recent_wins, weights=recent_weights)
    else:
        form_momentum = weighted_win_pct
    
    win_streak = 0
    for match in reversed(recent_matches):
        if match['won']:
            win_streak += 1
        else:
            break
    if win_streak == 0:
        loss_streak = 0
        for match in reversed(recent_matches):
            if not match['won']:
                loss_streak += 1
            else:
                break
        form_streak = -loss_streak
    else:
        form_streak = win_streak
    
    performance_scores = []
    for match in recent_matches:
        opponent_elo = match['loser_elo_before'] if match['won'] else match['winner_elo_before']
        expected = 1 / (1 + math.pow(10, (opponent_elo - current_elo) / 400))
        performance = (1 - expected) if match['won'] else (0 - expected)
        performance_scores.append(performance)
    
    consistency = 1 - min(np.std(performance_scores), 1.0) if performance_scores else 0.5
    
    recent_days = 30
    recent_match_count = sum(1 for m in match_history.get(player_id, []) 
                           if isinstance(m['date'], datetime) and 
                           (prediction_date - m['date']).days <= recent_days)
    
    surface_wins = sum(1 for r in surface_performance.get(player_id, {}).get(surface, []) if r)
    surface_matches = len(surface_performance.get(player_id, {}).get(surface, []))
    surface_win_pct = surface_wins / surface_matches if surface_matches > 0 else 0.5
    
    return {
        'recent_wins': wins,
        'recent_matches': total_matches,
        'win_percentage': win_percentage,
        'weighted_win_pct': weighted_win_pct,
        'avg_opponent_elo': avg_opponent_elo,
        'form_momentum': form_momentum,
        'form_streak': form_streak,
        'consistency': consistency,
        'recent_fatigue': recent_match_count,
        'straight_set_wins': 0,
        'lost_sets': 0,
        'avg_sets_played': 2.0,
        'win_rate_top10': win_rate_top10,
        'win_rate_top50': win_rate_top50,
        'recent_activity': min(recent_match_count / 4, 4),
        'surface_win_pct': surface_win_pct
    }

# ===========================
# FEATURE ENGINEERING
# ===========================

def create_enhanced_features(df, elo_ratings, global_ratings, form_history, match_history, opponent_stats, surface_performance):
    """Create features"""
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
            winner_id = row['winner_id']
            loser_id = row['loser_id']
            
            if pd.isna(winner_id) or pd.isna(loser_id):
                continue
                
            surface = str(row.get('Surface', 'Hard')).strip()
            match_date = row.get('Date', datetime.now())
            
            winner_elo = elo_ratings.get(winner_id, {}).get(surface, global_ratings.get(winner_id, 1500))
            loser_elo = elo_ratings.get(loser_id, {}).get(surface, global_ratings.get(loser_id, 1500))
            
            winner_form = calculate_enhanced_form_features(winner_id, surface, form_history, match_history, winner_elo, opponent_stats, surface_performance, match_date)
            loser_form = calculate_enhanced_form_features(loser_id, surface, form_history, match_history, loser_elo, opponent_stats, surface_performance, match_date)
            
            elo_diff = winner_elo - loser_elo
            elo_expected = 1 / (1 + math.pow(10, (-elo_diff) / 400))
            
            try:
                winner_rank = float(row.get('Rank_1', 100)) if str(row['Player_1']).strip() == str(row['Winner']).strip() else float(row.get('Rank_2', 100))
                loser_rank = float(row.get('Rank_2', 100)) if str(row['Player_2']).strip() != str(row['Winner']).strip() else float(row.get('Rank_1', 100))
            except:
                winner_rank = 100.0
                loser_rank = 100.0
            
            h2h_wins = h2h_matches = 0
            if winner_id in opponent_stats and loser_id in opponent_stats[winner_id]:
                for m in opponent_stats[winner_id][loser_id]:
                    h2h_matches += 1
                    if m['won']:
                        h2h_wins += 1
            
            h2h_ratio = h2h_wins / h2h_matches if h2h_matches > 0 else 0.5
            
            features = {
                'elo_diff': float(elo_diff),
                'winner_elo': float(winner_elo),
                'loser_elo': float(loser_elo),
                'elo_expected': float(elo_expected),
                'elo_ratio': float(winner_elo / loser_elo) if loser_elo > 0 else 1.0,
                'is_hard': 1 if surface == 'Hard' else 0,
                'is_clay': 1 if surface == 'Clay' else 0,
                'is_grass': 1 if surface == 'Grass' else 0,
                'is_carpet': 1 if surface == 'Carpet' else 0,
                'winner_rank': float(winner_rank),
                'loser_rank': float(loser_rank),
                'rank_diff': float(winner_rank - loser_rank),
                'rank_ratio': float(winner_rank / loser_rank) if loser_rank > 0 else 1.0,
                'w_win_pct': float(winner_form['win_percentage']),
                'w_weighted_pct': float(winner_form['weighted_win_pct']),
                'w_momentum': float(winner_form['form_momentum']),
                'w_streak': float(winner_form['form_streak']),
                'w_opp_elo': float(winner_form['avg_opponent_elo']),
                'w_consistency': float(winner_form['consistency']),
                'w_fatigue': float(winner_form['recent_fatigue']),
                'w_vs_top10': float(winner_form['win_rate_top10']),
                'w_vs_top50': float(winner_form['win_rate_top50']),
                'w_activity': float(winner_form['recent_activity']),
                'w_surface_pct': float(winner_form['surface_win_pct']),
                'l_win_pct': float(loser_form['win_percentage']),
                'l_weighted_pct': float(loser_form['weighted_win_pct']),
                'l_momentum': float(loser_form['form_momentum']),
                'l_streak': float(loser_form['form_streak']),
                'l_opp_elo': float(loser_form['avg_opponent_elo']),
                'l_consistency': float(loser_form['consistency']),
                'l_fatigue': float(loser_form['recent_fatigue']),
                'l_vs_top10': float(loser_form['win_rate_top10']),
                'l_vs_top50': float(loser_form['win_rate_top50']),
                'l_activity': float(loser_form['recent_activity']),
                'l_surface_pct': float(loser_form['surface_win_pct']),
                'form_diff': float(winner_form['win_percentage'] - loser_form['win_percentage']),
                'momentum_diff': float(winner_form['form_momentum'] - loser_form['form_momentum']),
                'consistency_diff': float(winner_form['consistency'] - loser_form['consistency']),
                'h2h_ratio': float(h2h_ratio),
                'h2h_matches': float(h2h_matches),
                'is_final': 1 if 'Final' in str(row.get('Round', '')) else 0,
                'best_of': float(row.get('Best of', 3)),
            }
            
            features_list.append(features)
            labels.append(1)
            
            loser_features = features.copy()
            loser_features['elo_diff'] = -features['elo_diff']
            loser_features['winner_elo'] = features['loser_elo']
            loser_features['loser_elo'] = features['winner_elo']
            loser_features['elo_ratio'] = 1 / features['elo_ratio'] if features['elo_ratio'] > 0 else 1.0
            
            for w, l in [('w_win_pct', 'l_win_pct'), ('w_weighted_pct', 'l_weighted_pct'),
                        ('w_momentum', 'l_momentum'), ('w_streak', 'l_streak'),
                        ('w_opp_elo', 'l_opp_elo'), ('w_consistency', 'l_consistency'),
                        ('w_fatigue', 'l_fatigue'), ('w_vs_top10', 'l_vs_top10'),
                        ('w_vs_top50', 'l_vs_top50'), ('w_activity', 'l_activity'),
                        ('w_surface_pct', 'l_surface_pct')]:
                loser_features[w] = features[l]
                loser_features[l] = features[w]
            
            loser_features['form_diff'] = -features['form_diff']
            loser_features['momentum_diff'] = -features['momentum_diff']
            loser_features['consistency_diff'] = -features['consistency_diff']
            loser_features['rank_diff'] = -features['rank_diff']
            loser_features['rank_ratio'] = 1 / features['rank_ratio'] if features['rank_ratio'] > 0 else 1.0
            loser_features['winner_rank'] = features['loser_rank']
            loser_features['loser_rank'] = features['winner_rank']
            loser_features['h2h_ratio'] = 1 - features['h2h_ratio'] if features['h2h_ratio'] != 0.5 else 0.5
            
            features_list.append(loser_features)
            labels.append(0)
            
        except Exception as e:
            continue
    
    return pd.DataFrame(features_list), np.array(labels)

# ===========================
# MODEL TRAINING
# ===========================

def train_ensemble_model(features_df, labels):
    """Train ensemble with proper feature handling"""
    
    unique_labels = np.unique(labels)
    if len(unique_labels) < 2:
        raise ValueError(f"Need both classes, got: {unique_labels}")
    
    scaler = RobustScaler()
    features_scaled = scaler.fit_transform(features_df)
    st.session_state.scaler = scaler
    
    X_train, X_test, y_train, y_test = train_test_split(
        features_scaled, labels, test_size=0.2, random_state=42, stratify=labels
    )
    
    models = {}
    
    if XGB_AVAILABLE:
        models['xgb'] = XGBClassifier(
            n_estimators=200, max_depth=6, learning_rate=0.05,
            subsample=0.8, colsample_bytree=0.8, min_child_weight=3,
            gamma=1, reg_alpha=0.5, reg_lambda=1.0, random_state=42,
            n_jobs=-1, eval_metric='logloss'
        )
    
    models['rf'] = RandomForestClassifier(
        n_estimators=200, max_depth=10, min_samples_split=5,
        min_samples_leaf=2, max_features='sqrt', random_state=42, n_jobs=-1
    )
    
    models['gb'] = GradientBoostingClassifier(
        n_estimators=150, max_depth=5, learning_rate=0.05,
        subsample=0.8, min_samples_split=5, random_state=42
    )
    
    models['lr'] = LogisticRegression(
        C=0.5, random_state=42, max_iter=1000, n_jobs=-1, solver='lbfgs'
    )
    
    trained_models = {}
    for name, model in models.items():
        try:
            model.fit(X_train, y_train)
            trained_models[name] = model
        except:
            continue
    
    if not trained_models:
        raise ValueError("Could not train any models")
    
    ensemble = VotingClassifier(
        estimators=[(name, model) for name, model in trained_models.items()],
        voting='soft'
    )
    
    calibrated_ensemble = CalibratedClassifierCV(ensemble, method='sigmoid', cv=3)
    calibrated_ensemble.fit(X_train, y_train)
    
    y_pred = calibrated_ensemble.predict(X_test)
    y_pred_proba = calibrated_ensemble.predict_proba(X_test)[:, 1]
    
    metrics = {
        'accuracy': float(accuracy_score(y_test, y_pred)),
        'precision': float(precision_score(y_test, y_pred, zero_division=0)),
        'recall': float(recall_score(y_test, y_pred, zero_division=0)),
        'f1': float(f1_score(y_test, y_pred, zero_division=0)),
        'roc_auc': float(roc_auc_score(y_test, y_pred_proba)),
        'confusion_matrix': confusion_matrix(y_test, y_pred).tolist()
    }
    
    if XGB_AVAILABLE and 'xgb' in trained_models:
        feature_importance = pd.DataFrame({
            'feature': features_df.columns,
            'importance': trained_models['xgb'].feature_importances_
        }).sort_values('importance', ascending=False)
        st.session_state.feature_importance = feature_importance
    
    return calibrated_ensemble, metrics, features_df.columns.tolist()

# ===========================
# PREDICTION
# ===========================

def predict_match(player1_id, player2_id, surface, elo_ratings, global_ratings, form_history, match_history, opponent_stats, surface_performance):
    """Predict match outcome"""
    
    if not st.session_state.ensemble_model:
        return None
    
    p1_elo = elo_ratings.get(player1_id, {}).get(surface, global_ratings.get(player1_id, 1500))
    p2_elo = elo_ratings.get(player2_id, {}).get(surface, global_ratings.get(player2_id, 1500))
    
    p1_form = calculate_enhanced_form_features(player1_id, surface, form_history, match_history, p1_elo, opponent_stats, surface_performance)
    p2_form = calculate_enhanced_form_features(player2_id, surface, form_history, match_history, p2_elo, opponent_stats, surface_performance)
    
    elo_diff = p1_elo - p2_elo
    elo_expected = 1 / (1 + math.pow(10, (-elo_diff) / 400))
    
    features_dict = {
        'elo_diff': float(elo_diff),
        'winner_elo': float(p1_elo),
        'loser_elo': float(p2_elo),
        'elo_expected': float(elo_expected),
        'elo_ratio': float(p1_elo / p2_elo) if p2_elo > 0 else 1.0,
        'is_hard': 1 if surface == 'Hard' else 0,
        'is_clay': 1 if surface == 'Clay' else 0,
        'is_grass': 1 if surface == 'Grass' else 0,
        'is_carpet': 1 if surface == 'Carpet' else 0,
        'winner_rank': 1.0,
        'loser_rank': 1.0,
        'rank_diff': 0.0,
        'rank_ratio': 1.0,
        'w_win_pct': float(p1_form['win_percentage']),
        'w_weighted_pct': float(p1_form['weighted_win_pct']),
        'w_momentum': float(p1_form['form_momentum']),
        'w_streak': float(p1_form['form_streak']),
        'w_opp_elo': float(p1_form['avg_opponent_elo']),
        'w_consistency': float(p1_form['consistency']),
        'w_fatigue': float(p1_form['recent_fatigue']),
        'w_vs_top10': float(p1_form['win_rate_top10']),
        'w_vs_top50': float(p1_form['win_rate_top50']),
        'w_activity': float(p1_form['recent_activity']),
        'w_surface_pct': float(p1_form['surface_win_pct']),
        'l_win_pct': float(p2_form['win_percentage']),
        'l_weighted_pct': float(p2_form['weighted_win_pct']),
        'l_momentum': float(p2_form['form_momentum']),
        'l_streak': float(p2_form['form_streak']),
        'l_opp_elo': float(p2_form['avg_opponent_elo']),
        'l_consistency': float(p2_form['consistency']),
        'l_fatigue': float(p2_form['recent_fatigue']),
        'l_vs_top10': float(p2_form['win_rate_top10']),
        'l_vs_top50': float(p2_form['win_rate_top50']),
        'l_activity': float(p2_form['recent_activity']),
        'l_surface_pct': float(p2_form['surface_win_pct']),
        'form_diff': float(p1_form['win_percentage'] - p2_form['win_percentage']),
        'momentum_diff': float(p1_form['form_momentum'] - p2_form['form_momentum']),
        'consistency_diff': float(p1_form['consistency'] - p2_form['consistency']),
        'h2h_ratio': 0.5,
        'h2h_matches': 0.0,
        'is_final': 0,
        'best_of': 3.0,
    }
    
    # CRITICAL: Use same feature columns as training
    features_df = pd.DataFrame([features_dict])
    features_scaled = st.session_state.scaler.transform(features_df)
    
    prediction = st.session_state.ensemble_model.predict_proba(features_scaled)[0][1]
    
    return prediction

# ===========================
# STREAMLIT UI
# ===========================

def main():
    st.title("🎾 Tennis Prediction System - Final")
    st.markdown("**Advanced ELO + Ensemble ML + Calibration**")
    
    tabs = st.tabs(["📊 Training", "🎯 Predictions", "📈 Analytics", "🤖 Model"])
    
    with tabs[0]:
        st.header("Train Model")
        
        col1, col2 = st.columns([2, 1])
        with col1:
            uploaded_file = st.file_uploader("Upload CSV", type=['csv'])
        with col2:
            elo_k = st.slider("K-factor", 20, 50, 32)
            initial_elo = st.slider("Init ELO", 1400, 1600, 1500)
        
        if uploaded_file:
            df = pd.read_csv(uploaded_file)
            st.session_state.match_data = df
            
            st.dataframe(df.head(10), width='stretch')
            st.write(f"Matches: {len(df)}")
            
            if all(col in df.columns for col in ['Player_1', 'Player_2', 'Winner', 'Surface']):
                st.success("✅ Data valid")
                
                if st.button("🚀 Train", type="primary"):
                    with st.spinner("Training..."):
                        elo_ratings, global_ratings = compute_advanced_elo_from_csv(
                            df, k_factor_base=elo_k, initial_elo=initial_elo
                        )
                        st.session_state.elo_ratings = elo_ratings
                        st.session_state.global_elo = global_ratings
                        
                        features_df, labels = create_enhanced_features(
                            df, elo_ratings, global_ratings,
                            st.session_state.player_form_history,
                            st.session_state.match_history,
                            st.session_state.opponent_stats,
                            st.session_state.surface_performance
                        )
                        
                        st.info(f"Samples: {len(features_df)} | Wins: {sum(labels)}")
                        
                        if len(np.unique(labels)) > 1:
                            model, metrics, cols = train_ensemble_model(features_df, labels)
                            st.session_state.ensemble_model = model
                            st.session_state.feature_columns = cols
                            st.session_state.model_metrics = metrics
                            
                            st.success("✅ Done!")
                            col1, col2, col3, col4, col5 = st.columns(5)
                            col1.metric("Acc", f"{metrics['accuracy']:.1%}")
                            col2.metric("Prec", f"{metrics['precision']:.1%}")
                            col3.metric("Rec", f"{metrics['recall']:.1%}")
                            col4.metric("F1", f"{metrics['f1']:.1%}")
                            col5.metric("AUC", f"{metrics['roc_auc']:.3f}")
    
    with tabs[1]:
        st.header("Prediction")
        
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
                surface = st.selectbox("Surface", SURFACE_TYPES)
            
            if st.button("Predict", type="primary"):
                p1_id = st.session_state.player_ids.get(p1)
                p2_id = st.session_state.player_ids.get(p2)
                
                if p1_id and p2_id:
                    prob = predict_match(p1_id, p2_id, surface,
                                        st.session_state.elo_ratings,
                                        st.session_state.global_elo,
                                        st.session_state.player_form_history,
                                        st.session_state.match_history,
                                        st.session_state.opponent_stats,
                                        st.session_state.surface_performance)
                    
                    st.metric(f"{p1} Win %", f"{prob:.1%}")
                    st.metric(f"{p2} Win %", f"{1-prob:.1%}")
    
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
        st.header("Model Info")
        if st.session_state.model_metrics:
            st.json({k: v for k, v in st.session_state.model_metrics.items() if k != 'confusion_matrix'})

if __name__ == "__main__":
    main()

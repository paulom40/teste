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

# Try to import ML libraries
try:
    import xgboost as xgb
    from xgboost import XGBClassifier
    from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix
    from sklearn.preprocessing import StandardScaler, MinMaxScaler
    from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.calibration import CalibratedClassifierCV
    XGB_AVAILABLE = True
except ImportError:
    XGB_AVAILABLE = False

# Page config
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
    st.session_state.scaler = StandardScaler()
if 'ensemble_model' not in st.session_state:
    st.session_state.ensemble_model = None
if 'xgb_model' not in st.session_state:
    st.session_state.xgb_model = None
if 'model_metrics' not in st.session_state:
    st.session_state.model_metrics = {}
if 'feature_importance' not in st.session_state:
    st.session_state.feature_importance = None
if 'feature_columns' not in st.session_state:
    st.session_state.feature_columns = []

# Constants
RECENT_MATCHES_COUNT = 20
SURFACE_TYPES = ['Hard', 'Clay', 'Grass', 'Carpet']
DAYS_RECENT = 365

# ===========================
# PLAYER ID & FORM MANAGEMENT
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
        'recent_wins': 0,
        'recent_matches': 0,
        'win_percentage': 0.5,
        'weighted_win_pct': 0.5,
        'avg_opponent_elo': 1500,
        'form_momentum': 0.5,
        'form_streak': 0,
        'similar_level_pct': 0.5,
        'consistency': 0.5,
        'recent_fatigue': 0,
        'straight_set_wins': 0,
        'lost_sets': 0,
        'comeback_wins': 0,
        'avg_sets_played': 2.0,
        'tournament_experience': 0
    }

# ===========================
# ELO CALCULATION
# ===========================

def compute_advanced_elo_from_csv(df, k_factor_base=32, initial_elo=1500):
    """Advanced ELO with match importance weighting"""
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
        
        # Match importance
        try:
            round_info = str(row.get('Round', '1st Round')).lower()
            if 'final' in round_info:
                match_importance = 1.5
            elif 'semifinal' in round_info or 'quarterfinal' in round_info:
                match_importance = 1.3
            elif 'grand slam' in str(row.get('Tournament', '')).lower():
                match_importance = 1.4
            else:
                match_importance = 1.0
        except:
            match_importance = 1.0
        
        winner_matches = len(match_history.get(winner, []))
        loser_matches = len(match_history.get(loser, []))
        
        winner_k = (k_factor_base / (1 + winner_matches / 100)) * match_importance
        loser_k = (k_factor_base / (1 + loser_matches / 100)) * match_importance
        
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
            'date': match_date,
            'opponent': loser,
            'surface': surface,
            'won': True,
            'score': row.get('Score', ''),
            'winner_elo_before': rating_w,
            'loser_elo_before': rating_l,
            'elo_change': winner_k * (1 - exp_w),
            'tournament': row.get('Tournament', ''),
            'round': row.get('Round', ''),
            'importance': match_importance
        }
        
        loser_match = {
            'date': match_date,
            'opponent': winner,
            'surface': surface,
            'won': False,
            'score': row.get('Score', ''),
            'winner_elo_before': rating_w,
            'loser_elo_before': rating_l,
            'elo_change': loser_k * (0 - exp_l),
            'tournament': row.get('Tournament', ''),
            'round': row.get('Round', ''),
            'importance': match_importance
        }
        
        match_history[winner].append(winner_match)
        match_history[loser].append(loser_match)
        
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
    return elo_ratings, global_ratings

# ===========================
# FORM FEATURES
# ===========================

def analyze_set_performance(matches):
    """Analyze set performance"""
    straight_set_wins = 0
    lost_sets = 0
    comeback_wins = 0
    total_sets = 0
    
    for match in matches:
        if match['won'] and match.get('score'):
            score = match['score']
            if isinstance(score, str) and '-' in score:
                sets = score.split()
                sets_played = len(sets)
                total_sets += sets_played
                
                if sets_played == 2 and match['won']:
                    straight_set_wins += 1
                
                if sets_played == 3 and match['won']:
                    lost_sets += 1
    
    avg_sets = total_sets / len(matches) if matches else 2.0
    
    return {
        'straight_set_wins': straight_set_wins,
        'lost_sets': lost_sets,
        'comeback_wins': comeback_wins,
        'avg_sets_played': avg_sets
    }

def calculate_enhanced_form_features(player_id, surface, form_history, match_history, current_elo, prediction_date=None):
    """Calculate enhanced form features"""
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
        weight = math.exp(-days_ago / 90)
        recency_weights.append(weight)
    
    recency_weights = np.array(recency_weights)
    if recency_weights.sum() > 0:
        recency_weights = recency_weights / recency_weights.sum()
    else:
        recency_weights = np.ones(len(recent_matches)) / len(recent_matches)
    
    wins = sum(1 for match in recent_matches if match['won'])
    total_matches = len(recent_matches)
    win_percentage = wins / total_matches if total_matches > 0 else 0.5
    
    weighted_wins = sum(recency_weights[i] for i, match in enumerate(recent_matches) if match['won'])
    weighted_win_pct = weighted_wins / recency_weights.sum() if recency_weights.sum() > 0 else 0.5
    
    opponent_elos = []
    opponent_weights = []
    for i, match in enumerate(recent_matches):
        if match['won']:
            opponent_elos.append(match['loser_elo_before'])
        else:
            opponent_elos.append(match['winner_elo_before'])
        opponent_weights.append(recency_weights[i] * match.get('importance', 1.0))
    
    avg_opponent_elo = np.average(opponent_elos, weights=opponent_weights) if opponent_elos else current_elo
    
    if len(recent_matches) >= 10:
        recent_weights = recency_weights[-10:] / recency_weights[-10:].sum() if recency_weights[-10:].sum() > 0 else np.ones(10) / 10
        recent_wins = [1 if match['won'] else 0 for match in recent_matches[-10:]]
        form_momentum = np.average(recent_wins, weights=recent_weights)
    else:
        form_momentum = weighted_win_pct
    
    streak = 0
    for match in reversed(recent_matches):
        if match['won']:
            streak += 1
        else:
            break
    if streak == 0:
        for match in reversed(recent_matches):
            if not match['won']:
                streak -= 1
            else:
                break
    
    similar_level_wins = 0
    similar_level_matches = 0
    for match in recent_matches:
        opponent_elo = match['loser_elo_before'] if match['won'] else match['winner_elo_before']
        if abs(current_elo - opponent_elo) < 100:
            similar_level_matches += 1
            if match['won']:
                similar_level_wins += 1
    
    similar_level_pct = similar_level_wins / similar_level_matches if similar_level_matches > 0 else 0.5
    
    performance_scores = []
    for match in recent_matches:
        if match['won']:
            opponent_elo = match['loser_elo_before']
            expected = 1 / (1 + math.pow(10, (opponent_elo - current_elo) / 400))
            performance = 1 - expected
        else:
            opponent_elo = match['winner_elo_before']
            expected = 1 / (1 + math.pow(10, (opponent_elo - current_elo) / 400))
            performance = 0 - expected
        performance_scores.append(performance)
    
    consistency = 1 - np.std(performance_scores) if performance_scores else 0.5
    
    recent_days = 30
    recent_match_count = sum(1 for match in match_history.get(player_id, []) 
                           if isinstance(match['date'], datetime) and 
                           (prediction_date - match['date']).days <= recent_days)
    
    set_stats = analyze_set_performance(recent_matches)
    
    return {
        'recent_wins': wins,
        'recent_matches': total_matches,
        'win_percentage': win_percentage,
        'weighted_win_pct': weighted_win_pct,
        'avg_opponent_elo': avg_opponent_elo,
        'form_momentum': form_momentum,
        'form_streak': streak,
        'similar_level_pct': similar_level_pct,
        'consistency': consistency,
        'recent_fatigue': recent_match_count,
        'straight_set_wins': set_stats['straight_set_wins'],
        'lost_sets': set_stats['lost_sets'],
        'comeback_wins': set_stats['comeback_wins'],
        'avg_sets_played': set_stats['avg_sets_played'],
        'tournament_experience': 0
    }

# ===========================
# FEATURE ENGINEERING
# ===========================

def create_enhanced_features(df, elo_ratings, global_ratings, form_history, match_history):
    """Create comprehensive features"""
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
            
            winner_form = calculate_enhanced_form_features(winner_id, surface, form_history, match_history, winner_elo, match_date)
            loser_form = calculate_enhanced_form_features(loser_id, surface, form_history, match_history, loser_elo, match_date)
            
            elo_diff = winner_elo - loser_elo
            elo_expected = 1 / (1 + math.pow(10, (-elo_diff) / 400))
            
            try:
                winner_rank = float(row.get('Rank_1', 100)) if str(row['Player_1']).strip() == str(row['Winner']).strip() else float(row.get('Rank_2', 100))
                loser_rank = float(row.get('Rank_2', 100)) if str(row['Player_2']).strip() != str(row['Winner']).strip() else float(row.get('Rank_1', 100))
            except:
                winner_rank = 100.0
                loser_rank = 100.0
            
            h2h_wins = 0
            h2h_matches = 0
            for match in match_history.get(winner_id, []):
                if match['opponent'] == loser_id:
                    h2h_matches += 1
                    if match['won']:
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
                'winner_rank': winner_rank,
                'loser_rank': loser_rank,
                'rank_diff': float(winner_rank - loser_rank),
                'rank_ratio': float(winner_rank / loser_rank) if loser_rank > 0 else 1.0,
                'winner_win_pct': float(winner_form['win_percentage']),
                'winner_weighted_pct': float(winner_form['weighted_win_pct']),
                'winner_form_momentum': float(winner_form['form_momentum']),
                'winner_streak': float(winner_form['form_streak']),
                'winner_avg_opp_elo': float(winner_form['avg_opponent_elo']),
                'winner_similarity': float(winner_form['similar_level_pct']),
                'winner_consistency': float(winner_form['consistency']),
                'winner_fatigue': float(winner_form['recent_fatigue']),
                'loser_win_pct': float(loser_form['win_percentage']),
                'loser_weighted_pct': float(loser_form['weighted_win_pct']),
                'loser_form_momentum': float(loser_form['form_momentum']),
                'loser_streak': float(loser_form['form_streak']),
                'loser_avg_opp_elo': float(loser_form['avg_opponent_elo']),
                'loser_similarity': float(loser_form['similar_level_pct']),
                'loser_consistency': float(loser_form['consistency']),
                'loser_fatigue': float(loser_form['recent_fatigue']),
                'form_diff': float(winner_form['win_percentage'] - loser_form['win_percentage']),
                'momentum_diff': float(winner_form['form_momentum'] - loser_form['form_momentum']),
                'consistency_diff': float(winner_form['consistency'] - loser_form['consistency']),
                'h2h_ratio': float(h2h_ratio),
                'h2h_matches': float(h2h_matches),
                'best_of': float(row.get('Best of', 3)),
                'is_final': 1 if 'Final' in str(row.get('Round', '')) else 0,
            }
            
            features_list.append(features)
            labels.append(1)
            
        except Exception as e:
            continue
    
    features_df = pd.DataFrame(features_list)
    return features_df, np.array(labels)

# ===========================
# MODEL TRAINING
# ===========================

def train_ensemble_model(features_df, labels):
    """Train ensemble model"""
    
    X_train, X_test, y_train, y_test = train_test_split(
        features_df, labels, test_size=0.2, random_state=42, stratify=labels
    )
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    st.session_state.scaler = scaler
    
    models = {}
    
    if XGB_AVAILABLE:
        models['xgb'] = XGBClassifier(
            n_estimators=150,
            max_depth=7,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
            n_jobs=-1
        )
    
    models['rf'] = RandomForestClassifier(
        n_estimators=200,
        max_depth=10,
        random_state=42,
        n_jobs=-1
    )
    
    models['gb'] = GradientBoostingClassifier(
        n_estimators=150,
        max_depth=5,
        learning_rate=0.1,
        random_state=42
    )
    
    models['lr'] = LogisticRegression(
        C=0.1,
        random_state=42,
        max_iter=1000,
        n_jobs=-1
    )
    
    trained_models = {}
    for name, model in models.items():
        model.fit(X_train_scaled, y_train)
        trained_models[name] = model
    
    ensemble = VotingClassifier(
        estimators=[(name, model) for name, model in trained_models.items()],
        voting='soft'
    )
    
    calibrated_ensemble = CalibratedClassifierCV(ensemble, method='sigmoid', cv=3)
    calibrated_ensemble.fit(X_train_scaled, y_train)
    
    y_pred = calibrated_ensemble.predict(X_test_scaled)
    y_pred_proba = calibrated_ensemble.predict_proba(X_test_scaled)[:, 1]
    
    metrics = {
        'accuracy': accuracy_score(y_test, y_pred),
        'precision': precision_score(y_test, y_pred),
        'recall': recall_score(y_test, y_pred),
        'f1': f1_score(y_test, y_pred),
        'roc_auc': roc_auc_score(y_test, y_pred_proba),
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

def predict_match(player1_id, player2_id, surface, elo_ratings, global_ratings, form_history, match_history):
    """Predict match outcome"""
    
    if not st.session_state.ensemble_model:
        return None
    
    p1_elo = elo_ratings.get(player1_id, {}).get(surface, global_ratings.get(player1_id, 1500))
    p2_elo = elo_ratings.get(player2_id, {}).get(surface, global_ratings.get(player2_id, 1500))
    
    p1_form = calculate_enhanced_form_features(player1_id, surface, form_history, match_history, p1_elo)
    p2_form = calculate_enhanced_form_features(player2_id, surface, form_history, match_history, p2_elo)
    
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
        'winner_win_pct': float(p1_form['win_percentage']),
        'winner_weighted_pct': float(p1_form['weighted_win_pct']),
        'winner_form_momentum': float(p1_form['form_momentum']),
        'winner_streak': float(p1_form['form_streak']),
        'winner_avg_opp_elo': float(p1_form['avg_opponent_elo']),
        'winner_similarity': float(p1_form['similar_level_pct']),
        'winner_consistency': float(p1_form['consistency']),
        'winner_fatigue': float(p1_form['recent_fatigue']),
        'loser_win_pct': float(p2_form['win_percentage']),
        'loser_weighted_pct': float(p2_form['weighted_win_pct']),
        'loser_form_momentum': float(p2_form['form_momentum']),
        'loser_streak': float(p2_form['form_streak']),
        'loser_avg_opp_elo': float(p2_form['avg_opponent_elo']),
        'loser_similarity': float(p2_form['similar_level_pct']),
        'loser_consistency': float(p2_form['consistency']),
        'loser_fatigue': float(p2_form['recent_fatigue']),
        'form_diff': float(p1_form['win_percentage'] - p2_form['win_percentage']),
        'momentum_diff': float(p1_form['form_momentum'] - p2_form['form_momentum']),
        'consistency_diff': float(p1_form['consistency'] - p2_form['consistency']),
        'h2h_ratio': 0.5,
        'h2h_matches': 0.0,
        'best_of': 3.0,
        'is_final': 0,
    }
    
    features_df = pd.DataFrame([features_dict])
    features_scaled = st.session_state.scaler.transform(features_df)
    
    prediction = st.session_state.ensemble_model.predict_proba(features_scaled)[0][1]
    
    return prediction

# ===========================
# STREAMLIT UI
# ===========================

def main():
    st.title("🎾 Advanced Tennis Prediction System")
    st.markdown("Enhanced ELO + Ensemble ML + Advanced Form Analysis")
    
    tabs = st.tabs(["📊 Data & Training", "🎯 Predictions", "📈 Analytics", "🤖 Model Info"])
    
    with tabs[0]:
        st.header("Data Upload & Model Training")
        
        col1, col2 = st.columns([2, 1])
        with col1:
            uploaded_file = st.file_uploader("Upload Tennis Match CSV", type=['csv'])
        
        with col2:
            elo_k = st.slider("ELO K-factor", 10, 50, 32)
            initial_elo = st.slider("Initial ELO", 1200, 1800, 1500)
        
        if uploaded_file is not None:
            try:
                df = pd.read_csv(uploaded_file)
                st.session_state.match_data = df
                
                st.subheader("Data Preview")
                st.dataframe(df.head(), width='stretch')
                st.write(f"Total matches: {len(df)}")
                
                required_cols = ['Player_1', 'Player_2', 'Winner', 'Surface']
                missing_cols = [col for col in required_cols if col not in df.columns]
                
                if missing_cols:
                    st.error(f"Missing: {missing_cols}")
                else:
                    st.success("✅ Data valid")
                    
                    if st.button("🚀 Train Models", type="primary"):
                        with st.spinner("Computing ELO ratings and features..."):
                            elo_ratings, global_ratings = compute_advanced_elo_from_csv(
                                df, k_factor_base=elo_k, initial_elo=initial_elo
                            )
                            st.session_state.elo_ratings = elo_ratings
                            st.session_state.global_elo = global_ratings
                            
                            features_df, labels = create_enhanced_features(
                                df, elo_ratings, global_ratings, 
                                st.session_state.player_form_history,
                                st.session_state.match_history
                            )
                            
                            if len(features_df) > 0:
                                with st.spinner("Training ensemble..."):
                                    model, metrics, feature_cols = train_ensemble_model(features_df, labels)
                                    st.session_state.ensemble_model = model
                                    st.session_state.feature_columns = feature_cols
                                    st.session_state.model_metrics = metrics
                                
                                st.success("✅ Models trained!")
                                
                                col1, col2, col3, col4, col5 = st.columns(5)
                                col1.metric("Accuracy", f"{metrics['accuracy']:.2%}")
                                col2.metric("Precision", f"{metrics['precision']:.2%}")
                                col3.metric("Recall", f"{metrics['recall']:.2%}")
                                col4.metric("F1", f"{metrics['f1']:.2%}")
                                col5.metric("ROC-AUC", f"{metrics['roc_auc']:.3f}")
                                
                                st.subheader("Confusion Matrix")
                                cm = metrics['confusion_matrix']
                                cm_df = pd.DataFrame(cm, 
                                                   columns=['Loss', 'Win'],
                                                   index=['Actual Loss', 'Actual Win'])
                                st.dataframe(cm_df)
            
            except Exception as e:
                st.error(f"Error: {str(e)}")
    
    with tabs[1]:
        st.header("Match Prediction")
        
        if not st.session_state.ensemble_model:
            st.warning("Train model first!")
        else:
            col1, col2, col3 = st.columns(3)
            
            player_list = list(st.session_state.player_names.values())
            
            with col1:
                player1 = st.selectbox("Player 1", player_list)
            with col2:
                player2 = st.selectbox("Player 2", [p for p in player_list if p != player1])
            with col3:
                surface = st.selectbox("Surface", SURFACE_TYPES)
            
            if st.button("Predict Match", type="primary"):
                p1_id = st.session_state.player_ids.get(player1)
                p2_id = st.session_state.player_ids.get(player2)
                
                if p1_id and p2_id:
                    with st.spinner("Predicting..."):
                        p1_win_prob = predict_match(p1_id, p2_id, surface,
                                                   st.session_state.elo_ratings,
                                                   st.session_state.global_elo,
                                                   st.session_state.player_form_history,
                                                   st.session_state.match_history)
                        
                        p2_win_prob = 1 - p1_win_prob
                        
                        st.subheader("Prediction Results")
                        
                        col1, col2 = st.columns(2)
                        with col1:
                            st.metric(f"{player1} Win %", f"{p1_win_prob:.1%}")
                        with col2:
                            st.metric(f"{player2} Win %", f"{p2_win_prob:.1%}")
                        
                        # Create chart
                        fig = go.Figure(data=[
                            go.Bar(x=[player1, player2], y=[p1_win_prob, p2_win_prob],
                                   marker=dict(color=['#1f77b4', '#ff7f0e']))
                        ])
                        fig.update_layout(title="Win Probability", yaxis_title="Probability")
                        st.plotly_chart(fig, use_container_width=True)
    
    with tabs[2]:
        st.header("Analytics & Insights")
        
        if st.session_state.match_data is not None:
            st.subheader("Player Statistics")
            
            player = st.selectbox("Select Player", list(st.session_state.player_names.values()))
            
            if player:
                player_id = st.session_state.player_ids.get(player)
                
                col1, col2, col3 = st.columns(3)
                
                # Calculate stats
                player_matches = st.session_state.match_history.get(player_id, [])
                wins = sum(1 for m in player_matches if m['won'])
                losses = len(player_matches) - wins
                
                with col1:
                    st.metric("Total Matches", len(player_matches))
                with col2:
                    st.metric("Wins", wins)
                with col3:
                    st.metric("Losses", losses)
                
                # ELO by surface
                st.subheader("ELO Rating by Surface")
                elo_data = []
                for surface in SURFACE_TYPES:
                    elo = st.session_state.elo_ratings.get(player_id, {}).get(surface, 1500)
                    elo_data.append({'Surface': surface, 'ELO': elo})
                
                elo_df = pd.DataFrame(elo_data)
                st.bar_chart(elo_df.set_index('Surface'))
    
    with tabs[3]:
        st.header("Model Information")
        
        if st.session_state.model_metrics:
            st.subheader("Model Metrics")
            st.json(st.session_state.model_metrics)
        
        if st.session_state.feature_importance is not None:
            st.subheader("Top 15 Important Features")
            top_15 = st.session_state.feature_importance.head(15)
            st.bar_chart(top_15.set_index('feature')['importance'])

if __name__ == "__main__":
    main()

import streamlit as st
import pandas as pd
import numpy as np
import math
import warnings
from collections import defaultdict, deque
from datetime import datetime, timedelta
import re

warnings.filterwarnings('ignore')

try:
    from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold, GridSearchCV
    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix, classification_report
    from sklearn.preprocessing import RobustScaler, StandardScaler
    from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier, StackingClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.calibration import CalibratedClassifierCV
    from sklearn.svm import SVC
    from sklearn.feature_selection import SelectKBest, f_classif, RFE
    from sklearn.decomposition import PCA
    from sklearn.pipeline import Pipeline
    from xgboost import XGBClassifier
    from lightgbm import LGBMClassifier
    import joblib
    ML_AVAILABLE = True
except ImportError:
    ML_AVAILABLE = False

st.set_page_config(page_title="Tennis Prediction Pro", page_icon="🎾", layout="wide")

# Initialize session state
if 'player_ids' not in st.session_state:
    st.session_state.player_ids = {}
if 'player_names' not in st.session_state:
    st.session_state.player_names = {}
if 'elo_ratings' not in st.session_state:
    st.session_state.elo_ratings = {}
if 'global_elo' not in st.session_state:
    st.session_state.global_elo = {}
if 'match_history' not in st.session_state:
    st.session_state.match_history = defaultdict(list)
if 'h2h_stats' not in st.session_state:
    st.session_state.h2h_stats = {}
if 'ensemble_model' not in st.session_state:
    st.session_state.ensemble_model = None
if 'scaler' not in st.session_state:
    st.session_state.scaler = RobustScaler()
if 'model_trained' not in st.session_state:
    st.session_state.model_trained = False
if 'match_data' not in st.session_state:
    st.session_state.match_data = None
if 'feature_importance' not in st.session_state:
    st.session_state.feature_importance = {}
if 'model_metrics' not in st.session_state:
    st.session_state.model_metrics = {}
if 'selected_features' not in st.session_state:
    st.session_state.selected_features = None
if 'feature_selector' not in st.session_state:
    st.session_state.feature_selector = None

SURFACE_TYPES = ['Hard', 'Clay', 'Grass', 'Carpet']

def get_default_form():
    """Default form dict"""
    return {
        'wins': 0, 'matches': 0, 'win_pct': 0.5, 'momentum': 0.5,
        'opp_elo': 1500, 'streak': 0, 'consistency': 0.5, 'fatigue': 0,
        'sets_played_avg': 3.0, 'games_played_avg': 22.0,
        'clutch_performance': 0.5, 'top10_performance': 0.5,
        'recent_5': 0.5, 'recent_10': 0.5, 'recent_20': 0.5
    }

def parse_score(score_str):
    """Parse tennis score to extract sets and games"""
    if pd.isna(score_str):
        return 3, 18
    
    score_str = str(score_str).strip()
    
    # Remove retirements, walkovers
    if any(x in score_str.upper() for x in ['RET', 'W/O', 'WO', 'DEF']):
        return 2, 12  # Default for incomplete matches
    
    # Split into sets
    sets = re.findall(r'\d+-\d+', score_str)
    if not sets:
        return 3, 18
    
    total_sets = len(sets)
    total_games = 0
    
    for set_score in sets:
        try:
            games = set_score.split('-')
            if len(games) == 2:
                total_games += int(games[0]) + int(games[1])
        except:
            total_games += 6  # Default per set
    
    return total_sets, total_games

def compute_advanced_elo(df, k_factor_base=32, k_factor_top=20, initial_elo=1500):
    """Advanced ELO computation with more features"""
    df = df.copy()
    
    # Map column names
    player1_col = None
    player2_col = None
    winner_col = None
    surface_col = None
    
    for col in df.columns:
        col_lower = str(col).lower()
        if 'player' in col_lower and ('1' in col_lower or 'player1' in col_lower):
            player1_col = col
        elif 'player' in col_lower and ('2' in col_lower or 'player2' in col_lower):
            player2_col = col
        elif 'winner' in col_lower:
            winner_col = col
        elif 'surface' in col_lower:
            surface_col = col
    
    # Rename columns
    if player1_col:
        df = df.rename(columns={player1_col: 'Player_1'})
    if player2_col:
        df = df.rename(columns={player2_col: 'Player_2'})
    if winner_col:
        df = df.rename(columns={winner_col: 'Winner'})
    if surface_col:
        df = df.rename(columns={surface_col: 'Surface'})
    else:
        df['Surface'] = 'Hard'
    
    # Clean data
    for col in ['Player_1', 'Player_2', 'Winner', 'Surface']:
        if col in df.columns:
            df[col] = df[col].astype(str).str.strip()
    
    # Get unique players
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
    match_history = defaultdict(list)
    h2h_stats = {}
    
    # Tournament importance (if available)
    tourney_importance = {
        'Grand Slam': 1.5, 'Masters 1000': 1.3, 'ATP 500': 1.2,
        'ATP 250': 1.0, 'Challenger': 0.8, 'Futures': 0.6
    }
    
    # Process matches
    valid_count = 0
    
    for idx, row in df.iterrows():
        p1 = row['Player_1']
        p2 = row['Player_2']
        winner = row['Winner']
        surface = row['Surface'] if row['Surface'] in SURFACE_TYPES else 'Hard'
        
        # Tournament level
        tourney_level = 'ATP 250'
        for col in df.columns:
            if any(x in str(col).lower() for x in ['tournament', 'series', 'level']):
                if col in row and pd.notna(row[col]):
                    tourney_val = str(row[col]).lower()
                    for key in tourney_importance:
                        if key.lower() in tourney_val:
                            tourney_level = key
                            break
        
        # Parse score
        if 'Score' in df.columns and pd.notna(row.get('Score')):
            sets_played, games_played = parse_score(row['Score'])
        else:
            sets_played, games_played = 3, 18
        
        if winner not in [p1, p2]:
            continue
            
        valid_count += 1
        winner_id = player_ids[winner]
        loser_id = player_ids[p2 if winner == p1 else p1]
        
        # Get current ratings
        rating_w = elo_ratings[winner_id][surface]
        rating_l = elo_ratings[loser_id][surface]
        
        # Dynamic K-factor based on experience
        w_matches = len([m for m in match_history.get(winner_id, []) if m.get('surface') == surface])
        l_matches = len([m for m in match_history.get(loser_id, []) if m.get('surface') == surface])
        
        k_w = max(k_factor_top, k_factor_base / (1 + w_matches / 100))
        k_l = max(k_factor_top, k_factor_base / (1 + l_matches / 100))
        
        # Tournament importance multiplier
        t_bonus = tourney_importance.get(tourney_level, 1.0)
        
        # H2H stats
        h2h_key = tuple(sorted([p1, p2]))
        if h2h_key not in h2h_stats:
            h2h_stats[h2h_key] = {'player1_wins': 0, 'matches': 0}
        
        h2h_info = h2h_stats[h2h_key]
        h2h_info['matches'] += 1
        
        # Determine which player is first in sorted tuple
        first_player = h2h_key[0]
        if winner == first_player:
            h2h_info['player1_wins'] += 1
        
        # H2H factor (if they've played before)
        if h2h_info['matches'] > 1:
            winner_is_first = (winner == first_player)
            h2h_win_rate = h2h_info['player1_wins'] / h2h_info['matches']
            
            if winner_is_first:
                h2h_advantage = h2h_win_rate - 0.5
            else:
                h2h_advantage = (1 - h2h_win_rate) - 0.5
            
            h2h_factor = 1 + (h2h_advantage * 0.2)  # Max 20% adjustment
        else:
            h2h_factor = 1.0
        
        # Score margin adjustment
        if sets_played > 0:
            # Determine sets won by winner (simplified - winner always won match)
            winner_sets = (sets_played + 1) // 2
            loser_sets = sets_played - winner_sets
            margin = (winner_sets - loser_sets) / sets_played
            margin_factor = 1 + (margin * 0.3)  # Up to 30% adjustment for dominant wins
        else:
            margin_factor = 1.0
        
        # Expected win probability
        exp_w = 1 / (1 + math.pow(10, (rating_l - rating_w) / 400))
        
        # Calculate rating changes with all factors
        elo_change_winner = k_w * t_bonus * h2h_factor * margin_factor * (1 - exp_w)
        elo_change_loser = k_l * t_bonus * h2h_factor * (0 - (1 - exp_w))
        
        # Apply changes
        elo_ratings[winner_id][surface] = rating_w + elo_change_winner
        elo_ratings[loser_id][surface] = rating_l + elo_change_loser
        
        # Update global rating (weighted average by surface frequency)
        surface_weights = {'Hard': 0.35, 'Clay': 0.30, 'Grass': 0.20, 'Carpet': 0.15}
        for pid in [winner_id, loser_id]:
            total_weight = sum(surface_weights.get(s, 0.1) for s in SURFACE_TYPES)
            global_ratings[pid] = sum(
                elo_ratings[pid].get(s, initial_elo) * surface_weights.get(s, 0.1) 
                for s in SURFACE_TYPES
            ) / total_weight
        
        # Store detailed match history
        match_history[winner_id].append({
            'date': pd.Timestamp.now(),
            'surface': surface,
            'opponent': loser_id,
            'won': True,
            'elo': rating_w,
            'opp_elo': rating_l,
            'sets_played': sets_played,
            'games_played': games_played,
            'tournament_level': tourney_level,
            'score_margin': margin
        })
        
        match_history[loser_id].append({
            'date': pd.Timestamp.now(),
            'surface': surface,
            'opponent': winner_id,
            'won': False,
            'elo': rating_l,
            'opp_elo': rating_w,
            'sets_played': sets_played,
            'games_played': games_played,
            'tournament_level': tourney_level,
            'score_margin': -margin
        })
    
    st.session_state.elo_ratings = elo_ratings
    st.session_state.global_elo = global_ratings
    st.session_state.match_history = match_history
    st.session_state.h2h_stats = h2h_stats
    
    return valid_count

def calc_advanced_form(player_id, surface, match_history, lookback_days=365):
    """Calculate advanced form metrics"""
    matches = match_history.get(player_id, [])
    
    # Get recent matches
    now = pd.Timestamp.now()
    recent_matches = [
        m for m in matches[-50:]  # Last 50 matches max
        if (now - m.get('date', now)).days <= lookback_days and m.get('surface') == surface
    ]
    
    if not recent_matches:
        return get_default_form()
    
    wins = sum(1 for m in recent_matches if m.get('won', False))
    total = len(recent_matches)
    
    # Multiple time windows
    recent_5 = recent_matches[-5:] if len(recent_matches) >= 5 else recent_matches
    recent_10 = recent_matches[-10:] if len(recent_matches) >= 10 else recent_matches
    recent_20 = recent_matches[-20:] if len(recent_matches) >= 20 else recent_matches
    
    win_pct_5 = sum(1 for m in recent_5 if m.get('won', False)) / max(len(recent_5), 1)
    win_pct_10 = sum(1 for m in recent_10 if m.get('won', False)) / max(len(recent_10), 1)
    win_pct_20 = sum(1 for m in recent_20 if m.get('won', False)) / max(len(recent_20), 1)
    
    # Momentum (exponential weighting)
    momentum = 0
    weights = [0.35, 0.25, 0.20, 0.12, 0.08]  # More weight to recent matches
    for i, m in enumerate(recent_matches[-5:]):
        if i < len(weights):
            momentum += (1 if m.get('won', False) else 0) * weights[i]
    
    # Streak
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
    top10_wins = 0
    top10_matches = 0
    
    for m in recent_matches:
        opp_elo = m.get('opp_elo', 1500)
        opp_elos.append(opp_elo)
        
        if opp_elo > 1800:  # Top 10 level
            top10_matches += 1
            if m.get('won', False):
                top10_wins += 1
    
    avg_opp_elo = np.mean(opp_elos) if opp_elos else 1500
    top10_performance = top10_wins / max(top10_matches, 1)
    
    # Consistency (performance vs expectation)
    performances = []
    for m in recent_matches:
        player_elo = m.get('elo', 1500)
        opp_elo = m.get('opp_elo', 1500)
        expected = 1 / (1 + math.pow(10, (opp_elo - player_elo) / 400))
        actual = 1 if m.get('won', False) else 0
        performances.append(actual - expected)
    
    consistency = 1 - np.std(performances) if performances and len(performances) > 1 else 0.5
    
    # Clutch performance (close matches)
    close_matches = [m for m in recent_matches if abs(m.get('score_margin', 0)) < 0.3]
    clutch_wins = sum(1 for m in close_matches if m.get('won', False))
    clutch_performance = clutch_wins / max(len(close_matches), 1)
    
    # Fatigue
    fatigue = sum(1 for m in matches if (now - m.get('date', now)).days <= 30)
    
    # Average sets and games
    avg_sets = np.mean([m.get('sets_played', 3) for m in recent_matches])
    avg_games = np.mean([m.get('games_played', 18) for m in recent_matches])
    
    return {
        'wins': wins,
        'matches': total,
        'win_pct': wins / max(total, 1),
        'momentum': momentum,
        'opp_elo': avg_opp_elo,
        'streak': streak,
        'consistency': consistency,
        'fatigue': min(fatigue / 10, 1),
        'sets_played_avg': avg_sets,
        'games_played_avg': avg_games,
        'clutch_performance': clutch_performance,
        'top10_performance': top10_performance,
        'recent_5': win_pct_5,
        'recent_10': win_pct_10,
        'recent_20': win_pct_20,
        'performance_std': np.std(performances) if performances and len(performances) > 1 else 0
    }

def create_all_features(p1, p2, surface, p1_form, p2_form, p1_elo, p2_elo, p1_global, p2_global, h2h_info):
    """Create all possible features for a match"""
    # Determine H2H win percentage for p1
    h2h_key = tuple(sorted([p1, p2]))
    h2h_matches = h2h_info['matches']
    first_player = h2h_key[0]
    
    if h2h_matches > 0:
        if p1 == first_player:
            h2h_pct_p1 = h2h_info['player1_wins'] / h2h_matches
        else:
            h2h_pct_p1 = 1 - (h2h_info['player1_wins'] / h2h_matches)
    else:
        h2h_pct_p1 = 0.5
    
    # Create all possible features
    features = {
        # ELO features
        'elo_diff': p1_elo - p2_elo,
        'elo_ratio': p1_elo / max(p2_elo, 1),
        'global_elo_diff': p1_global - p2_global,
        
        # Form features
        'win_pct_diff': p1_form['win_pct'] - p2_form['win_pct'],
        'momentum_diff': p1_form['momentum'] - p2_form['momentum'],
        'streak_diff': p1_form['streak'] - p2_form['streak'],
        'consistency_diff': p1_form['consistency'] - p2_form['consistency'],
        'fatigue_diff': p1_form['fatigue'] - p2_form['fatigue'],
        
        # Recent form
        'recent_5_diff': p1_form['recent_5'] - p2_form['recent_5'],
        'recent_10_diff': p1_form['recent_10'] - p2_form['recent_10'],
        'recent_20_diff': p1_form['recent_20'] - p2_form['recent_20'],
        
        # Advanced metrics
        'clutch_diff': p1_form['clutch_performance'] - p2_form['clutch_performance'],
        'top10_diff': p1_form['top10_performance'] - p2_form['top10_performance'],
        'opp_elo_diff': p1_form['opp_elo'] - p2_form['opp_elo'],
        
        # Individual features
        'p1_win_pct': p1_form['win_pct'],
        'p1_momentum': p1_form['momentum'],
        'p1_streak': p1_form['streak'],
        'p1_consistency': p1_form['consistency'],
        'p1_clutch': p1_form['clutch_performance'],
        
        'p2_win_pct': p2_form['win_pct'],
        'p2_momentum': p2_form['momentum'],
        'p2_streak': p2_form['streak'],
        'p2_consistency': p2_form['consistency'],
        'p2_clutch': p2_form['clutch_performance'],
        
        # H2H features
        'h2h_win_pct': h2h_pct_p1,
        'h2h_matches': h2h_matches,
        
        # Surface features
        'is_hard': 1 if surface == 'Hard' else 0,
        'is_clay': 1 if surface == 'Clay' else 0,
        'is_grass': 1 if surface == 'Grass' else 0,
        
        # Interaction features
        'elo_momentum_interaction': (p1_elo - p2_elo) * (p1_form['momentum'] - p2_form['momentum']),
        'elo_form_interaction': (p1_elo - p2_elo) * (p1_form['win_pct'] - p2_form['win_pct']),
        'momentum_streak_interaction': (p1_form['momentum'] - p2_form['momentum']) * (p1_form['streak'] - p2_form['streak']),
        
        # Derived features
        'experience_diff': len(st.session_state.match_history.get(st.session_state.player_ids.get(p1, ''), [])) - 
                         len(st.session_state.match_history.get(st.session_state.player_ids.get(p2, ''), [])),
        'upset_potential': 1 if p2_elo > p1_elo else 0,
        'form_consistency_product': p1_form['consistency'] * p2_form['consistency'],
    }
    
    return features

def create_advanced_features(df):
    """Create advanced feature set for higher accuracy"""
    features_list = []
    labels = []
    
    player_ids = st.session_state.player_ids
    match_history = st.session_state.match_history
    h2h_stats = st.session_state.h2h_stats
    
    for idx, row in df.iterrows():
        p1 = row['Player_1'] if 'Player_1' in row else None
        p2 = row['Player_2'] if 'Player_2' in row else None
        winner = row['Winner'] if 'Winner' in row else None
        surface = row['Surface'] if 'Surface' in row and row['Surface'] in SURFACE_TYPES else 'Hard'
        
        if not p1 or not p2 or not winner or winner not in [p1, p2]:
            continue
            
        p1_id = player_ids.get(p1)
        p2_id = player_ids.get(p2)
        
        if not p1_id or not p2_id:
            continue
        
        # Get ELO ratings
        p1_elo = st.session_state.elo_ratings.get(p1_id, {}).get(surface, 1500)
        p2_elo = st.session_state.elo_ratings.get(p2_id, {}).get(surface, 1500)
        p1_global = st.session_state.global_elo.get(p1_id, 1500)
        p2_global = st.session_state.global_elo.get(p2_id, 1500)
        
        # Get advanced form
        p1_form = calc_advanced_form(p1_id, surface, match_history)
        p2_form = calc_advanced_form(p2_id, surface, match_history)
        
        # H2H stats
        h2h_key = tuple(sorted([p1, p2]))
        h2h_info = h2h_stats.get(h2h_key, {'player1_wins': 0, 'matches': 0})
        
        # Create features
        features = create_all_features(p1, p2, surface, p1_form, p2_form, p1_elo, p2_elo, p1_global, p2_global, h2h_info)
        
        features_list.append(features)
        labels.append(1 if winner == p1 else 0)
    
    return pd.DataFrame(features_list), np.array(labels)

def train_advanced_model(features_df, labels, use_advanced=True):
    """Train advanced ensemble model for higher accuracy"""
    if len(np.unique(labels)) < 2:
        raise ValueError("Need both win and loss examples")
    
    # Store all feature names before selection
    all_feature_names = features_df.columns.tolist()
    
    # Feature selection
    if len(features_df.columns) > 30 and use_advanced:
        selector = SelectKBest(f_classif, k=min(30, len(features_df.columns)))
        X_selected = selector.fit_transform(features_df, labels)
        selected_features = features_df.columns[selector.get_support()].tolist()
        st.session_state.selected_features = selected_features
        st.session_state.feature_selector = selector
        
        # Keep only selected features
        features_df = pd.DataFrame(X_selected, columns=selected_features)
    else:
        st.session_state.selected_features = all_feature_names
        st.session_state.feature_selector = None
    
    # Split data with stratification
    X_train, X_test, y_train, y_test = train_test_split(
        features_df, labels, test_size=0.15, random_state=42, stratify=labels
    )
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    st.session_state.scaler = scaler
    
    if use_advanced:
        # Advanced ensemble with hyperparameter tuning
        models = {
            'xgb': XGBClassifier(
                n_estimators=300,
                max_depth=7,
                learning_rate=0.05,
                subsample=0.8,
                colsample_bytree=0.8,
                random_state=42,
                n_jobs=-1,
                use_label_encoder=False,
                eval_metric='logloss'
            ),
            'rf': RandomForestClassifier(
                n_estimators=300,
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
            'lr': LogisticRegression(
                C=0.5,
                random_state=42,
                max_iter=1000,
                n_jobs=-1,
                class_weight='balanced'
            )
        }
        
        # Train with cross-validation
        trained_models = {}
        cv_scores = {}
        
        for name, model in models.items():
            try:
                cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
                scores = cross_val_score(model, X_train_scaled, y_train, 
                                        cv=cv, scoring='roc_auc', n_jobs=-1)
                cv_scores[name] = scores.mean()
                
                model.fit(X_train_scaled, y_train)
                trained_models[name] = model
                
                st.write(f"✅ {name.upper()} - CV AUC: {scores.mean():.3f} (±{scores.std():.3f})")
            except Exception as e:
                st.write(f"⚠️ Could not train {name}: {str(e)}")
        
        # Create stacking ensemble
        estimators = list(trained_models.items())
        
        # Meta-learner
        meta_learner = LogisticRegression(
            C=0.5,
            random_state=42,
            max_iter=1000,
            n_jobs=-1
        )
        
        # Stacking classifier
        stack_model = StackingClassifier(
            estimators=estimators,
            final_estimator=meta_learner,
            cv=5,
            stack_method='predict_proba',
            n_jobs=-1
        )
        
        # Train stacking model
        stack_model.fit(X_train_scaled, y_train)
        
        # Calibrate
        calibrated_model = CalibratedClassifierCV(
            stack_model,
            method='isotonic',
            cv=3,
            n_jobs=-1
        )
        
    else:
        # Simple model for comparison
        model = RandomForestClassifier(
            n_estimators=200,
            max_depth=10,
            random_state=42,
            n_jobs=-1
        )
        
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        scores = cross_val_score(model, X_train_scaled, y_train, 
                                cv=cv, scoring='roc_auc', n_jobs=-1)
        cv_scores = {'rf': scores.mean()}
        
        calibrated_model = CalibratedClassifierCV(model, method='sigmoid', cv=3)
    
    calibrated_model.fit(X_train_scaled, y_train)
    
    # Evaluate
    y_pred = calibrated_model.predict(X_test_scaled)
    y_pred_proba = calibrated_model.predict_proba(X_test_scaled)[:, 1]
    
    # Calculate comprehensive metrics
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
    if use_advanced and 'rf' in trained_models:
        importances = trained_models['rf'].feature_importances_
    else:
        importances = calibrated_model.base_estimator.feature_importances_ if hasattr(calibrated_model.base_estimator, 'feature_importances_') else None
    
    if importances is not None:
        feature_importance = dict(zip(features_df.columns, importances))
        st.session_state.feature_importance = feature_importance
    
    st.session_state.model_metrics = metrics
    return calibrated_model, metrics

def predict_advanced_match(p1_name, p2_name, surface):
    """Advanced prediction"""
    if not st.session_state.ensemble_model:
        return None
    
    p1_id = st.session_state.player_ids.get(p1_name)
    p2_id = st.session_state.player_ids.get(p2_name)
    
    if not p1_id or not p2_id:
        return None
    
    # Get ratings
    p1_elo = st.session_state.elo_ratings.get(p1_id, {}).get(surface, 1500)
    p2_elo = st.session_state.elo_ratings.get(p2_id, {}).get(surface, 1500)
    p1_global = st.session_state.global_elo.get(p1_id, 1500)
    p2_global = st.session_state.global_elo.get(p2_id, 1500)
    
    # Get advanced form
    p1_form = calc_advanced_form(p1_id, surface, st.session_state.match_history)
    p2_form = calc_advanced_form(p2_id, surface, st.session_state.match_history)
    
    # H2H stats
    h2h_key = tuple(sorted([p1_name, p2_name]))
    h2h_info = st.session_state.h2h_stats.get(h2h_key, {'player1_wins': 0, 'matches': 0})
    
    # Create all features
    all_features = create_all_features(p1_name, p2_name, surface, p1_form, p2_form, 
                                      p1_elo, p2_elo, p1_global, p2_global, h2h_info)
    
    # Create DataFrame with all features
    all_features_df = pd.DataFrame([all_features])
    
    # Apply feature selection if used during training
    if st.session_state.selected_features is not None:
        # Ensure we only use the features that were selected during training
        features_df = all_features_df[st.session_state.selected_features]
    else:
        features_df = all_features_df
    
    # Ensure all expected features are present (fill missing with 0)
    for feature in st.session_state.selected_features if st.session_state.selected_features else features_df.columns:
        if feature not in features_df.columns:
            features_df[feature] = 0
    
    # Scale and predict
    try:
        features_scaled = st.session_state.scaler.transform(features_df)
        prediction_proba = st.session_state.ensemble_model.predict_proba(features_scaled)[0]
        p1_win_prob = float(prediction_proba[1])
        p2_win_prob = float(prediction_proba[0])
        
        # Calculate expected score (sets)
        expected_score = 3 * p1_win_prob
        
        return {
            'p1_win_prob': p1_win_prob,
            'p2_win_prob': p2_win_prob,
            'expected_score': expected_score,
            'features': all_features,
            'p1_elo': p1_elo,
            'p2_elo': p2_elo,
            'p1_form': p1_form,
            'p2_form': p2_form,
            'h2h': h2h_info
        }
    except Exception as e:
        st.error(f"Prediction error: {e}")
        return None

def display_match_prediction(p1_name, p2_name, surface):
    """Display match prediction with detailed analysis"""
    if p1_name not in st.session_state.player_ids or p2_name not in st.session_state.player_ids:
        st.error("One or both players not found in dataset")
        return
    
    prediction = predict_advanced_match(p1_name, p2_name, surface)
    
    if not prediction:
        st.error("Could not generate prediction")
        return
    
    # Create columns for display
    col1, col2, col3 = st.columns([2, 1, 2])
    
    with col1:
        st.markdown(f"### {p1_name}")
        st.metric("Win Probability", f"{prediction['p1_win_prob']*100:.1f}%")
        st.metric("ELO Rating", f"{prediction['p1_elo']:.0f}")
        st.metric("Current Form", f"{prediction['p1_form']['win_pct']*100:.1f}%")
        st.metric("Win Streak", prediction['p1_form']['streak'])
        
    with col2:
        st.markdown("### VS")
        # Expected sets visualization
        if prediction['p1_win_prob'] > 0.5:
            st.markdown(f"**Favorite**: {p1_name}")
            confidence = abs(prediction['p1_win_prob'] - 0.5) * 2
            st.progress(confidence)
            st.caption(f"Confidence: {confidence*100:.0f}%")
        else:
            st.markdown(f"**Favorite**: {p2_name}")
            confidence = abs(prediction['p2_win_prob'] - 0.5) * 2
            st.progress(confidence)
            st.caption(f"Confidence: {confidence*100:.0f}%")
            
        # Expected score
        st.metric("Expected Sets", f"{prediction['expected_score']:.1f}")
    
    with col3:
        st.markdown(f"### {p2_name}")
        st.metric("Win Probability", f"{prediction['p2_win_prob']*100:.1f}%")
        st.metric("ELO Rating", f"{prediction['p2_elo']:.0f}")
        st.metric("Current Form", f"{prediction['p2_form']['win_pct']*100:.1f}%")
        st.metric("Win Streak", prediction['p2_form']['streak'])
    
    # Detailed analysis
    st.markdown("---")
    st.markdown("### Detailed Analysis")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("#### Head-to-Head")
        h2h = prediction['h2h']
        if h2h['matches'] > 0:
            p1_wins = h2h['player1_wins'] if p1_name == tuple(sorted([p1_name, p2_name]))[0] else h2h['matches'] - h2h['player1_wins']
            p2_wins = h2h['matches'] - p1_wins
            st.write(f"**Matches Played**: {h2h['matches']}")
            st.write(f"**{p1_name} Wins**: {p1_wins}")
            st.write(f"**{p2_name} Wins**: {p2_wins}")
            st.write(f"**{p1_name} Win %**: {p1_wins/h2h['matches']*100:.1f}%")
        else:
            st.write("No previous matches")
    
    with col2:
        st.markdown("#### Key Factors")
        
        factors = []
        # ELO advantage
        elo_diff = prediction['p1_elo'] - prediction['p2_elo']
        if abs(elo_diff) > 50:
            factors.append(f"{'Strong' if elo_diff > 0 else 'Weak'} ELO advantage ({abs(elo_diff):.0f} points)")
        
        # Form advantage
        form_diff = prediction['p1_form']['win_pct'] - prediction['p2_form']['win_pct']
        if abs(form_diff) > 0.2:
            factors.append(f"{'Better' if form_diff > 0 else 'Worse'} recent form ({form_diff*100:.0f}%)")
        
        # Momentum
        momentum_diff = prediction['p1_form']['momentum'] - prediction['p2_form']['momentum']
        if abs(momentum_diff) > 0.3:
            factors.append(f"{'Positive' if momentum_diff > 0 else 'Negative'} momentum")
        
        # Clutch performance
        clutch_diff = prediction['p1_form']['clutch_performance'] - prediction['p2_form']['clutch_performance']
        if abs(clutch_diff) > 0.25:
            factors.append(f"{'Better' if clutch_diff > 0 else 'Worse'} in close matches")
        
        for factor in factors:
            st.write(f"• {factor}")
    
    # Advanced metrics
    st.markdown("#### Advanced Metrics")
    adv_col1, adv_col2, adv_col3, adv_col4 = st.columns(4)
    
    with adv_col1:
        st.metric("Consistency", 
                 f"{prediction['p1_form']['consistency']*100:.0f}%",
                 f"{prediction['p2_form']['consistency']*100:.0f}%",
                 delta_color="off")
    
    with adv_col2:
        st.metric("Clutch Performance",
                 f"{prediction['p1_form']['clutch_performance']*100:.0f}%",
                 f"{prediction['p2_form']['clutch_performance']*100:.0f}%",
                 delta_color="off")
    
    with adv_col3:
        st.metric("Top 10 Performance",
                 f"{prediction['p1_form']['top10_performance']*100:.0f}%",
                 f"{prediction['p2_form']['top10_performance']*100:.0f}%",
                 delta_color="off")
    
    with adv_col4:
        st.metric("Opponent Quality",
                 f"{prediction['p1_form']['opp_elo']:.0f}",
                 f"{prediction['p2_form']['opp_elo']:.0f}",
                 delta_color="off")

def main():
    """Main Streamlit application"""
    st.title("🎾 Tennis Prediction Pro")
    st.markdown("Advanced tennis match prediction using ELO ratings and machine learning")
    
    # Sidebar
    with st.sidebar:
        st.header("Settings")
        
        # Data upload
        st.subheader("Upload Match Data")
        uploaded_file = st.file_uploader("Upload CSV with match data", type=['csv'])
        
        if uploaded_file:
            try:
                df = pd.read_csv(uploaded_file)
                st.session_state.match_data = df
                st.success(f"Loaded {len(df)} matches")
                
                # Process data
                if st.button("Process Data & Calculate ELO"):
                    with st.spinner("Processing match data..."):
                        valid_count = compute_advanced_elo(df)
                        st.success(f"Processed {valid_count} valid matches")
                        
                        # Display summary
                        st.write(f"**Players in system**: {len(st.session_state.player_ids)}")
                        st.write(f"**ELO range**: {min(st.session_state.global_elo.values()):.0f} - {max(st.session_state.global_elo.values()):.0f}")
            
            except Exception as e:
                st.error(f"Error loading file: {e}")
        
        st.markdown("---")
        
        # Model training
        st.subheader("Model Training")
        use_advanced = st.checkbox("Use Advanced Ensemble", value=True)
        
        if st.session_state.match_data is not None and st.button("Train Prediction Model"):
            with st.spinner("Training advanced model..."):
                try:
                    # Create features
                    features_df, labels = create_advanced_features(st.session_state.match_data)
                    
                    if len(features_df) > 0:
                        # Train model
                        model, metrics = train_advanced_model(features_df, labels, use_advanced)
                        st.session_state.ensemble_model = model
                        st.session_state.model_trained = True
                        
                        st.success(f"Model trained successfully!")
                        st.write(f"**Test Accuracy**: {metrics['accuracy']*100:.1f}%")
                        st.write(f"**ROC AUC**: {metrics['roc_auc']:.3f}")
                        
                        # Show selected features count
                        if st.session_state.selected_features:
                            st.write(f"**Features used**: {len(st.session_state.selected_features)}")
                    else:
                        st.error("Could not create features from data")
                except Exception as e:
                    st.error(f"Training error: {e}")
        
        st.markdown("---")
        
        # Player rankings
        if st.session_state.global_elo:
            st.subheader("Top 10 Players")
            sorted_players = sorted(st.session_state.global_elo.items(), 
                                   key=lambda x: x[1], reverse=True)[:10]
            
            for pid, rating in sorted_players:
                name = st.session_state.player_names.get(pid, "Unknown")
                st.write(f"**{name}**: {rating:.0f}")
    
    # Main content area
    if st.session_state.match_data is not None and st.session_state.player_ids:
        
        # Tab layout
        tab1, tab2, tab3, tab4 = st.tabs(["Match Prediction", "Player Analysis", "Model Performance", "Data Exploration"])
        
        with tab1:
            st.header("Match Prediction")
            
            col1, col2 = st.columns(2)
            
            with col1:
                # Player 1 selection
                p1_name = st.selectbox(
                    "Select Player 1",
                    options=list(st.session_state.player_ids.keys()),
                    key="p1_select"
                )
                
                if p1_name:
                    p1_id = st.session_state.player_ids[p1_name]
                    p1_elo = st.session_state.global_elo.get(p1_id, 1500)
                    st.metric("Global ELO", f"{p1_elo:.0f}")
                    
                    # Surface-specific ELO
                    surface_elos = st.session_state.elo_ratings.get(p1_id, {})
                    for surface in SURFACE_TYPES:
                        if surface in surface_elos:
                            st.caption(f"{surface}: {surface_elos[surface]:.0f}")
            
            with col2:
                # Player 2 selection
                p2_name = st.selectbox(
                    "Select Player 2",
                    options=[p for p in st.session_state.player_ids.keys() if p != p1_name],
                    key="p2_select"
                )
                
                if p2_name:
                    p2_id = st.session_state.player_ids[p2_name]
                    p2_elo = st.session_state.global_elo.get(p2_id, 1500)
                    st.metric("Global ELO", f"{p2_elo:.0f}")
                    
                    # Surface-specific ELO
                    surface_elos = st.session_state.elo_ratings.get(p2_id, {})
                    for surface in SURFACE_TYPES:
                        if surface in surface_elos:
                            st.caption(f"{surface}: {surface_elos[surface]:.0f}")
            
            # Surface selection
            surface = st.selectbox("Select Surface", SURFACE_TYPES, index=0)
            
            # Prediction button
            if st.button("Predict Match", type="primary"):
                if p1_name and p2_name:
                    if st.session_state.model_trained:
                        display_match_prediction(p1_name, p2_name, surface)
                    else:
                        st.warning("Please train the model first in the sidebar")
                else:
                    st.error("Please select both players")
        
        with tab2:
            st.header("Player Analysis")
            
            player_name = st.selectbox(
                "Select Player",
                options=list(st.session_state.player_ids.keys()),
                key="analysis_player"
            )
            
            if player_name:
                player_id = st.session_state.player_ids[player_name]
                
                col1, col2 = st.columns(2)
                
                with col1:
                    st.subheader("ELO Ratings by Surface")
                    surface_data = []
                    for surface in SURFACE_TYPES:
                        elo = st.session_state.elo_ratings.get(player_id, {}).get(surface, 1500)
                        surface_data.append({"Surface": surface, "ELO": elo})
                    
                    if surface_data:
                        df_surface = pd.DataFrame(surface_data)
                        st.bar_chart(df_surface.set_index("Surface"))
                
                with col2:
                    st.subheader("Recent Form")
                    matches = st.session_state.match_history.get(player_id, [])[-10:]
                    
                    if matches:
                        recent_results = []
                        for match in matches:
                            result = "W" if match.get('won', False) else "L"
                            opponent_id = match.get('opponent', '')
                            opponent = st.session_state.player_names.get(opponent_id, "Unknown")
                            surface = match.get('surface', 'Unknown')
                            recent_results.append({
                                "Result": result,
                                "Opponent": opponent,
                                "Surface": surface,
                                "ELO": match.get('elo', 1500)
                            })
                        
                        df_results = pd.DataFrame(recent_results)
                        st.dataframe(df_results, width='stretch')
                    else:
                        st.write("No recent matches")
                
                # Form metrics
                st.subheader("Form Metrics")
                form_cols = st.columns(4)
                
                for surface in SURFACE_TYPES:
                    form = calc_advanced_form(player_id, surface, st.session_state.match_history)
                    
                    with form_cols[SURFACE_TYPES.index(surface)]:
                        st.markdown(f"**{surface}**")
                        st.metric("Win %", f"{form['win_pct']*100:.1f}%")
                        st.metric("Momentum", f"{form['momentum']:.2f}")
                        st.metric("Streak", form['streak'])
                        st.metric("Consistency", f"{form['consistency']*100:.0f}%")
        
        with tab3:
            st.header("Model Performance")
            
            if st.session_state.model_metrics:
                metrics = st.session_state.model_metrics
                
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    st.metric("Accuracy", f"{metrics['accuracy']*100:.1f}%")
                with col2:
                    st.metric("Precision", f"{metrics['precision']*100:.1f}%")
                with col3:
                    st.metric("Recall", f"{metrics['recall']*100:.1f}%")
                with col4:
                    st.metric("F1 Score", f"{metrics['f1']*100:.1f}%")
                
                st.metric("ROC AUC", f"{metrics['roc_auc']:.3f}")
                
                # Cross-validation scores
                if 'cv_scores' in metrics:
                    st.subheader("Cross-Validation Scores")
                    cv_df = pd.DataFrame(list(metrics['cv_scores'].items()), 
                                        columns=['Model', 'AUC Score'])
                    st.bar_chart(cv_df.set_index('Model'))
                
                # Feature importance
                if st.session_state.feature_importance:
                    st.subheader("Top 10 Important Features")
                    fi_df = pd.DataFrame(list(st.session_state.feature_importance.items()),
                                        columns=['Feature', 'Importance'])
                    fi_df = fi_df.sort_values('Importance', ascending=False).head(10)
                    st.bar_chart(fi_df.set_index('Feature'))
                
                # Show selected features
                if st.session_state.selected_features:
                    st.subheader(f"Selected Features ({len(st.session_state.selected_features)})")
                    with st.expander("View all features"):
                        for i, feature in enumerate(st.session_state.selected_features[:50]):
                            st.write(f"{i+1}. {feature}")
                        if len(st.session_state.selected_features) > 50:
                            st.write(f"... and {len(st.session_state.selected_features) - 50} more")
                
                # Confusion matrix
                st.subheader("Confusion Matrix")
                if 'confusion_matrix' in metrics:
                    cm = metrics['confusion_matrix']
                    cm_df = pd.DataFrame(cm, 
                                        index=['Actual Loss', 'Actual Win'],
                                        columns=['Predicted Loss', 'Predicted Win'])
                    st.dataframe(cm_df.style.background_gradient(cmap='Blues'), width='stretch')
            else:
                st.info("Train the model to see performance metrics")
        
        with tab4:
            st.header("Data Exploration")
            
            if st.session_state.match_data is not None:
                st.subheader("Match Data Preview")
                st.dataframe(st.session_state.match_data.head(20), width='stretch')
                
                st.subheader("Dataset Statistics")
                st.write(f"**Total Matches**: {len(st.session_state.match_data)}")
                st.write(f"**Unique Players**: {len(st.session_state.player_ids)}")
                st.write(f"**Surface Distribution**:")
                if 'Surface' in st.session_state.match_data.columns:
                    surface_counts = st.session_state.match_data['Surface'].value_counts()
                    st.bar_chart(surface_counts)
                
                # Player statistics
                st.subheader("Player Statistics")
                player_stats = []
                for name, pid in list(st.session_state.player_ids.items())[:20]:
                    matches = len(st.session_state.match_history.get(pid, []))
                    global_elo = st.session_state.global_elo.get(pid, 1500)
                    player_stats.append({
                        "Player": name,
                        "Matches": matches,
                        "ELO": global_elo
                    })
                
                if player_stats:
                    stats_df = pd.DataFrame(player_stats)
                    st.dataframe(stats_df.sort_values('ELO', ascending=False), width='stretch')
    
    else:
        # Welcome screen
        st.markdown("""
        ## Welcome to Tennis Prediction Pro
        
        To get started:
        1. **Upload your match data** (CSV format) in the sidebar
        2. **Process the data** to calculate ELO ratings
        3. **Train the prediction model**
        4. **Start predicting matches!**
        
        ### Expected CSV Format:
        Your CSV should contain at minimum:
        - Player 1 name
        - Player 2 name  
        - Winner name
        - Surface (Hard/Clay/Grass/Carpet)
        
        Optional columns:
        - Score
        - Tournament level
        - Date
        
        ### Example Data:
        You can start with a sample dataset from ATP match results.
        """)
        
        # Example data
        example_data = {
            'Player_1': ['Djokovic', 'Nadal', 'Federer', 'Murray'],
            'Player_2': ['Nadal', 'Federer', 'Murray', 'Djokovic'],
            'Winner': ['Djokovic', 'Nadal', 'Federer', 'Djokovic'],
            'Surface': ['Hard', 'Clay', 'Grass', 'Hard'],
            'Score': ['6-4 6-3', '6-2 6-3 6-1', '7-6 6-4', '6-3 7-5']
        }
        
        st.dataframe(pd.DataFrame(example_data), width='stretch')

if __name__ == "__main__":
    main()

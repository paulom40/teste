import streamlit as st
import pandas as pd
import numpy as np
import math
import warnings
from collections import defaultdict, deque
from datetime import datetime, timedelta
warnings.filterwarnings('ignore')

# Try to import additional libraries
try:
    import xgboost as xgb
    from xgboost import XGBClassifier
    from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix
    from sklearn.preprocessing import StandardScaler, MinMaxScaler
    from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.neural_network import MLPClassifier
    from sklearn.calibration import CalibratedClassifierCV
    XGB_AVAILABLE = True
except ImportError:
    XGB_AVAILABLE = False

# Set page config
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
if 'xgb_model' not in st.session_state:
    st.session_state.xgb_model = None
if 'feature_columns' not in st.session_state:
    st.session_state.feature_columns = []
if 'match_data' not in st.session_state:
    st.session_state.match_data = None
if 'player_form_history' not in st.session_state:
    st.session_state.player_form_history = {}
if 'player_ids' not in st.session_state:
    st.session_state.player_ids = {}
if 'scaler' not in st.session_state:
    st.session_state.scaler = StandardScaler()
if 'ensemble_model' not in st.session_state:
    st.session_state.ensemble_model = None
if 'model_metrics' not in st.session_state:
    st.session_state.model_metrics = {}
if 'feature_importance' not in st.session_state:
    st.session_state.feature_importance = {}

# Constants
RECENT_MATCHES_COUNT = 20
SURFACE_TYPES = ['Hard', 'Clay', 'Grass', 'Carpet']
DAYS_RECENT = 365  # Consider matches within last year for "recent" form

# Function to create player IDs from names
def create_player_ids(df):
    """Create unique IDs for players based on their names"""
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

# Advanced ELO calculation with decay
def compute_advanced_elo_from_csv(df, k_factor_base=32, initial_elo=1500, elo_decay=0.97):
    """Advanced ELO with time decay and match importance weighting"""
    if not st.session_state.player_ids:
        st.session_state.player_ids = create_player_ids(df)
    
    if 'Date' in df.columns:
        df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
        df = df.sort_values(by=['Date']).reset_index(drop=True)
    
    # Map player names to IDs
    df['winner_id'] = df['Winner'].apply(lambda x: st.session_state.player_ids.get(str(x).strip()) if pd.notna(x) else None)
    df['loser_id'] = df.apply(lambda row: st.session_state.player_ids.get(
        str(row['Player_1']).strip() if str(row['Player_1']).strip() != str(row['Winner']).strip() else str(row['Player_2']).strip()
    ) if pd.notna(row['Player_1']) and pd.notna(row['Winner']) else None, axis=1)
    
    players = set(df['winner_id'].dropna().unique()).union(set(df['loser_id'].dropna().unique()))
    
    elo_ratings = {}
    global_ratings = {}
    form_history = defaultdict(lambda: defaultdict(deque))
    match_history = defaultdict(list)  # Store all matches for each player
    
    # Initialize ratings
    for player in players:
        if player:
            elo_ratings[player] = {}
            for surface in SURFACE_TYPES:
                elo_ratings[player][surface] = initial_elo
            global_ratings[player] = initial_elo
    
    # Process matches
    for index, row in df.iterrows():
        winner = row['winner_id']
        loser = row['loser_id']
        
        if pd.isna(winner) or pd.isna(loser):
            continue
        
        surface = str(row.get('Surface', 'Hard')).strip()
        if pd.isna(surface) or surface not in SURFACE_TYPES:
            surface = 'Hard'
        
        # Get current ratings
        rating_w = elo_ratings.get(winner, {}).get(surface, global_ratings.get(winner, initial_elo))
        rating_l = elo_ratings.get(loser, {}).get(surface, global_ratings.get(loser, initial_elo))
        
        # Calculate dynamic K-factor based on match importance
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
        
        # Adjust K-factor based on player experience (more matches = smaller K)
        winner_matches = len(match_history.get(winner, []))
        loser_matches = len(match_history.get(loser, []))
        
        winner_k = k_factor_base / (1 + winner_matches / 100)
        loser_k = k_factor_base / (1 + loser_matches / 100)
        
        # Apply match importance
        winner_k *= match_importance
        loser_k *= match_importance
        
        # Calculate ELO update
        exp_w = 1 / (1 + math.pow(10, (rating_l - rating_w) / 400))
        exp_l = 1 - exp_w
        
        # Update ratings
        if winner not in elo_ratings:
            elo_ratings[winner] = {}
        if loser not in elo_ratings:
            elo_ratings[loser] = {}
        
        elo_ratings[winner][surface] = rating_w + winner_k * (1 - exp_w)
        elo_ratings[loser][surface] = rating_l + loser_k * (0 - exp_l)
        
        global_ratings[winner] = global_ratings.get(winner, initial_elo) + winner_k * (1 - exp_w)
        global_ratings[loser] = global_ratings.get(loser, initial_elo) + loser_k * (0 - exp_l)
        
        # Store match for history
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
        
        # Keep recent matches for form (with time decay)
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

# Enhanced form feature calculation with recency weighting
def calculate_enhanced_form_features(player_id, surface, form_history, match_history, current_elo, prediction_date=None):
    """Calculate enhanced form features with recency weighting"""
    if player_id not in form_history or surface not in form_history[player_id]:
        return get_default_form_features()
    
    recent_matches = list(form_history[player_id][surface])
    if not recent_matches:
        return get_default_form_features()
    
    # Calculate recency weights (exponential decay)
    if prediction_date is None:
        prediction_date = datetime.now()
    
    recency_weights = []
    for match in recent_matches:
        days_ago = (prediction_date - match['date']).days if isinstance(match['date'], datetime) else 365
        weight = math.exp(-days_ago / 90)  # 90-day half-life
        recency_weights.append(weight)
    
    recency_weights = np.array(recency_weights)
    recency_weights = recency_weights / recency_weights.sum() if recency_weights.sum() > 0 else np.ones(len(recent_matches)) / len(recent_matches)
    
    # Basic metrics
    wins = sum(1 for match in recent_matches if match['won'])
    total_matches = len(recent_matches)
    win_percentage = wins / total_matches if total_matches > 0 else 0.5
    
    # Weighted metrics
    weighted_wins = sum(recency_weights[i] for i, match in enumerate(recent_matches) if match['won'])
    weighted_win_pct = weighted_wins / recency_weights.sum() if recency_weights.sum() > 0 else 0.5
    
    # Opponent strength
    opponent_elos = []
    opponent_weights = []
    for i, match in enumerate(recent_matches):
        if match['won']:
            opponent_elos.append(match['loser_elo_before'])
        else:
            opponent_elos.append(match['winner_elo_before'])
        opponent_weights.append(recency_weights[i] * match.get('importance', 1.0))
    
    avg_opponent_elo = np.average(opponent_elos, weights=opponent_weights) if opponent_elos else current_elo
    
    # Momentum (last 10 matches with recency)
    if len(recent_matches) >= 10:
        recent_weights = recency_weights[-10:] / recency_weights[-10:].sum() if recency_weights[-10:].sum() > 0 else np.ones(10) / 10
        recent_wins = [1 if match['won'] else 0 for match in recent_matches[-10:]]
        form_momentum = np.average(recent_wins, weights=recent_weights)
    else:
        form_momentum = weighted_win_pct
    
    # Streak calculation
    streak = 0
    for match in reversed(recent_matches):
        if match['won']:
            streak += 1
        else:
            break
    if streak == 0:  # If not winning streak, check losing streak
        for match in reversed(recent_matches):
            if not match['won']:
                streak -= 1
            else:
                break
    
    # Performance against similar level opponents
    similar_level_wins = 0
    similar_level_matches = 0
    for i, match in enumerate(recent_matches):
        opponent_elo = match['loser_elo_before'] if match['won'] else match['winner_elo_before']
        if abs(current_elo - opponent_elo) < 100:  # Within 100 ELO points
            similar_level_matches += 1
            if match['won']:
                similar_level_wins += 1
    
    similar_level_pct = similar_level_wins / similar_level_matches if similar_level_matches > 0 else 0.5
    
    # Tournament performance
    tournament_wins = {}
    for match in recent_matches:
        tournament = match.get('tournament', '')
        if tournament not in tournament_wins:
            tournament_wins[tournament] = {'wins': 0, 'matches': 0}
        tournament_wins[tournament]['matches'] += 1
        if match['won']:
            tournament_wins[tournament]['wins'] += 1
    
    # Calculate consistency (variance in performance)
    performance_scores = []
    for match in recent_matches:
        if match['won']:
            # Higher score for beating stronger opponents
            opponent_elo = match['loser_elo_before']
            expected = 1 / (1 + math.pow(10, (opponent_elo - current_elo) / 400))
            performance = 1 - expected  # How much better than expected
        else:
            opponent_elo = match['winner_elo_before']
            expected = 1 / (1 + math.pow(10, (opponent_elo - current_elo) / 400))
            performance = 0 - expected  # How much worse than expected
        performance_scores.append(performance)
    
    consistency = 1 - np.std(performance_scores) if performance_scores else 0.5
    
    # Fatigue indicator (matches in last 30 days)
    recent_days = 30
    recent_match_count = sum(1 for match in match_history.get(player_id, []) 
                           if isinstance(match['date'], datetime) and 
                           (prediction_date - match['date']).days <= recent_days)
    
    # Set performance analysis
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
        'tournament_experience': len(tournament_wins)
    }

def get_default_form_features():
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

def analyze_set_performance(matches):
    """Analyze set performance from match scores"""
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
                
                # Check for straight set win
                if sets_played == 2 and match['won']:
                    straight_set_wins += 1
                
                # Check for lost sets
                if sets_played == 3 and match['won']:
                    lost_sets += 1
                    
                    # Check for comeback win (lost first set but won)
                    try:
                        first_set = sets[0].split('-')
                        if len(first_set) == 2:
                            w, l = map(int, first_set)
                            if w < l:  # Lost first set
                                comeback_wins += 1
                    except:
                        pass
    
    avg_sets = total_sets / len(matches) if matches else 2.0
    
    return {
        'straight_set_wins': straight_set_wins,
        'lost_sets': lost_sets,
        'comeback_wins': comeback_wins,
        'avg_sets_played': avg_sets
    }

# Enhanced feature engineering
def create_enhanced_features(df, elo_ratings, global_ratings, form_history, match_history):
    """Create comprehensive feature set for model training"""
    features_list = []
    labels = []
    
    if 'Date' in df.columns:
        df = df.sort_values('Date').reset_index(drop=True)
    
    # Map player names to IDs
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
            
            # Get ELO ratings
            winner_elo = elo_ratings.get(winner_id, {}).get(surface, global_ratings.get(winner_id, 1500))
            loser_elo = elo_ratings.get(loser_id, {}).get(surface, global_ratings.get(loser_id, 1500))
            
            # Calculate enhanced form features
            winner_form = calculate_enhanced_form_features(winner_id, surface, form_history, match_history, winner_elo, match_date)
            loser_form = calculate_enhanced_form_features(loser_id, surface, form_history, match_history, loser_elo, match_date)
            
            # Advanced ELO calculations
            elo_diff = winner_elo - loser_elo
            elo_expected = 1 / (1 + math.pow(10, (-elo_diff) / 400))
            
            # Get rankings
            try:
                winner_rank = float(row.get('Rank_1', 100)) if str(row['Player_1']).strip() == str(row['Winner']).strip() else float(row.get('Rank_2', 100))
                loser_rank = float(row.get('Rank_2', 100)) if str(row['Player_2']).strip() != str(row['Winner']).strip() else float(row.get('Rank_1', 100))
            except:
                winner_rank = 100.0
                loser_rank = 100.0
            
            # Head-to-head history (if available)
            h2h_wins = 0
            h2h_matches = 0
            for match in match_history.get(winner_id, []):
                if match['opponent'] == loser_id:
                    h2h_matches += 1
                    if match['won']:
                        h2h_wins += 1
            
            h2h_ratio = h2h_wins / h2h_matches if h2h_matches > 0 else 0.5
            
            # Tournament experience
            tournament = str(row.get('Tournament', ''))
            winner_tourney_exp = sum(1 for m in match_history.get(winner_id, []) if m.get('tournament') == tournament)
            loser_tourney_exp = sum(1 for m in match_history.get(loser_id, []) if m.get('tournament') == tournament)
            
            # Create comprehensive feature set
            features = {
                # ELO Features
                'elo_diff': float(elo_diff),
                'winner_elo': float(winner_elo),
                'loser_elo': float(loser_elo),
                'elo_expected': float(elo_expected),
                'elo_ratio': float(winner_elo / loser_elo) if loser_elo > 0 else 1.0,
                
                # Surface Features
                'is_hard': 1 if surface == 'Hard' else 0,
                'is_clay': 1 if surface == 'Clay' else 0,
                'is_grass': 1 if surface == 'Grass' else 0,
                'is_carpet': 1 if surface == 'Carpet' else 0,
                'surface_advantage': float(winner_form.get('surface_win_pct', 0.5) - loser_form.get('surface_win_pct', 0.5)),
                
                # Ranking Features
                'winner_rank': winner_rank,
                'loser_rank': loser_rank,
                'rank_diff': float(winner_rank - loser_rank),
                'rank_ratio': float(winner_rank / loser_rank) if loser_rank > 0 else 1.0,
                
                # Winner Enhanced Form Features
                'winner_win_pct': float(winner_form['win_percentage']),
                'winner_weighted_pct': float(winner_form['weighted_win_pct']),
                'winner_form_momentum': float(winner_form['form_momentum']),
                'winner_streak': float(winner_form['form_streak']),
                'winner_avg_opp_elo': float(winner_form['avg_opponent_elo']),
                'winner_similar_level': float(winner_form['similar_level_pct']),
                'winner_consistency': float(winner_form['consistency']),
                'winner_fatigue': float(winner_form['recent_fatigue']),
                'winner_straight_sets': float(winner_form['straight_set_wins']),
                'winner_comebacks': float(winner_form['comeback_wins']),
                'winner_avg_sets': float(winner_form['avg_sets_played']),
                
                # Loser Enhanced Form Features
                'loser_win_pct': float(loser_form['win_percentage']),
                'loser_weighted_pct': float(loser_form['weighted_win_pct']),
                'loser_form_momentum': float(loser_form['form_momentum']),
                'loser_streak': float(loser_form['form_streak']),
                'loser_avg_opp_elo': float(loser_form['avg_opponent_elo']),
                'loser_similar_level': float(loser_form['similar_level_pct']),
                'loser_consistency': float(loser_form['consistency']),
                'loser_fatigue': float(loser_form['recent_fatigue']),
                'loser_straight_sets': float(loser_form['straight_set_wins']),
                'loser_comebacks': float(loser_form['comeback_wins']),
                'loser_avg_sets': float(loser_form['avg_sets_played']),
                
                # Form Differentials
                'form_diff': float(winner_form['win_percentage'] - loser_form['win_percentage']),
                'weighted_form_diff': float(winner_form['weighted_win_pct'] - loser_form['weighted_win_pct']),
                'momentum_diff': float(winner_form['form_momentum'] - loser_form['form_momentum']),
                'streak_diff': float(winner_form['form_streak'] - loser_form['form_streak']),
                'opp_strength_diff': float(winner_form['avg_opponent_elo'] - loser_form['avg_opponent_elo']),
                'consistency_diff': float(winner_form['consistency'] - loser_form['consistency']),
                'fatigue_diff': float(winner_form['recent_fatigue'] - loser_form['recent_fatigue']),
                
                # Match Context Features
                'h2h_ratio': float(h2h_ratio),
                'h2h_matches': float(h2h_matches),
                'tourney_exp_diff': float(winner_tourney_exp - loser_tourney_exp),
                'best_of': float(row.get('Best of', 3)),
                'is_final': 1 if 'Final' in str(row.get('Round', '')) else 0,
                'is_grand_slam': 1 if 'Grand Slam' in str(row.get('Tournament', '')) else 0,
                
                # Interaction Features
                'elo_form_interaction': float(elo_diff * (winner_form['win_percentage'] - loser_form['win_percentage'])),
                'rank_form_interaction': float((winner_rank - loser_rank) * (winner_form['win_percentage'] - loser_form['win_percentage'])),
                
                # Statistical Features
                'winner_variance': float(1 - winner_form['consistency']),
                'loser_variance': float(1 - loser_form['consistency']),
            }
            
            # Add odds if available
            if 'Odd_1' in row and pd.notna(row['Odd_1']) and row['Odd_1'] != -1:
                features['winner_odds'] = float(row['Odd_1']) if str(row['Player_1']).strip() == str(row['Winner']).strip() else float(row.get('Odd_2', 2.0))
                features['loser_odds'] = float(row['Odd_2']) if str(row['Player_2']).strip() != str(row['Winner']).strip() else float(row.get('Odd_1', 2.0))
                features['odds_ratio'] = float(features['loser_odds'] / features['winner_odds']) if features['winner_odds'] > 0 else 1.0
            else:
                features['winner_odds'] = 2.0
                features['loser_odds'] = 2.0
                features['odds_ratio'] = 1.0
            
            features_list.append(features)
            labels.append(1)  # Winner perspective
            
            # Create reverse perspective
            features_reverse = create_reverse_features(features)
            features_list.append(features_reverse)
            labels.append(0)  # Loser perspective
            
        except Exception as e:
            continue
    
    features_df = pd.DataFrame(features_list)
    return features_df, np.array(labels)

def create_reverse_features(features):
    """Create reverse perspective features"""
    reverse_features = {}
    
    # Swap ELO features
    reverse_features['elo_diff'] = -features['elo_diff']
    reverse_features['winner_elo'] = features['loser_elo']
    reverse_features['loser_elo'] = features['winner_elo']
    reverse_features['elo_expected'] = 1 - features['elo_expected']
    reverse_features['elo_ratio'] = 1 / features['elo_ratio'] if features['elo_ratio'] > 0 else 1.0
    
    # Surface features remain the same
    for sf in ['is_hard', 'is_clay', 'is_grass', 'is_carpet']:
        reverse_features[sf] = features[sf]
    reverse_features['surface_advantage'] = -features['surface_advantage']
    
    # Swap rankings
    reverse_features['winner_rank'] = features['loser_rank']
    reverse_features['loser_rank'] = features['winner_rank']
    reverse_features['rank_diff'] = -features['rank_diff']
    reverse_features['rank_ratio'] = 1 / features['rank_ratio'] if features['rank_ratio'] > 0 else 1.0
    
    # Swap form features
    form_pairs = [
        ('winner_win_pct', 'loser_win_pct'),
        ('winner_weighted_pct', 'loser_weighted_pct'),
        ('winner_form_momentum', 'loser_form_momentum'),
        ('winner_streak', 'loser_streak'),
        ('winner_avg_opp_elo', 'loser_avg_opp_elo'),
        ('winner_similar_level', 'loser_similar_level'),
        ('winner_consistency', 'loser_consistency'),
        ('winner_fatigue', 'loser_fatigue'),
        ('winner_straight_sets', 'loser_straight_sets'),
        ('winner_comebacks', 'loser_comebacks'),
        ('winner_avg_sets', 'loser_avg_sets'),
        ('winner_variance', 'loser_variance')
    ]
    
    for w_feat, l_feat in form_pairs:
        reverse_features[w_feat] = features[l_feat]
        reverse_features[l_feat] = features[w_feat]
    
    # Reverse differentials
    diff_features = ['form_diff', 'weighted_form_diff', 'momentum_diff', 'streak_diff', 
                    'opp_strength_diff', 'consistency_diff', 'fatigue_diff']
    for feat in diff_features:
        reverse_features[feat] = -features[feat]
    
    # Match context (mostly swap)
    reverse_features['h2h_ratio'] = 1 - features['h2h_ratio'] if features['h2h_ratio'] != 0.5 else 0.5
    reverse_features['h2h_matches'] = features['h2h_matches']
    reverse_features['tourney_exp_diff'] = -features['tourney_exp_diff']
    reverse_features['best_of'] = features['best_of']
    reverse_features['is_final'] = features['is_final']
    reverse_features['is_grand_slam'] = features['is_grand_slam']
    
    # Interaction features
    reverse_features['elo_form_interaction'] = -features['elo_form_interaction']
    reverse_features['rank_form_interaction'] = -features['rank_form_interaction']
    
    # Odds
    reverse_features['winner_odds'] = features['loser_odds']
    reverse_features['loser_odds'] = features['winner_odds']
    reverse_features['odds_ratio'] = 1 / features['odds_ratio'] if features['odds_ratio'] > 0 else 1.0
    
    return reverse_features

# Ensemble model training
def train_ensemble_model(features_df, labels):
    """Train ensemble model for better accuracy"""
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        features_df, labels, test_size=0.2, random_state=42, stratify=labels
    )
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    st.session_state.scaler = scaler
    
    # Define models
    models = {
        'xgb': XGBClassifier(
            n_estimators=150,
            max_depth=7,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
            n_jobs=-1
        ),
        'rf': RandomForestClassifier(
            n_estimators=200,
            max_depth=10,
            random_state=42,
            n_jobs=-1
        ),
        'gb': GradientBoostingClassifier(
            n_estimators=150,
            max_depth=5,
            learning_rate=0.1,
            random_state=42
        ),
        'lr': LogisticRegression(
            C=0.1,
            random_state=42,
            max_iter=1000,
            n_jobs=-1
        )
    }
    
    # Train individual models
    trained_models = {}
    for name, model in models.items():
        model.fit(X_train_scaled, y_train)
        trained_models[name] = model
    
    # Create ensemble
    ensemble = VotingClassifier(
        estimators=[(name, model) for name, model in trained_models.items()],
        voting='soft',
        weights=[3, 2, 2, 1]  # Weight XGBoost highest
    )
    
    # Calibrate the ensemble
    calibrated_ensemble = CalibratedClassifierCV(ensemble, method='sigmoid', cv=3)
    calibrated_ensemble.fit(X_train_scaled, y_train)
    
    # Evaluate
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
    
    # Get feature importance from XGBoost
    xgb_model = trained_models['xgb']
    feature_importance = pd.DataFrame({
        'feature': features_df.columns,
        'importance': xgb_model.feature_importances_
    }).sort_values('importance', ascending=False)
    
    st.session_state.feature_importance = feature_importance
    
    return calibrated_ensemble, metrics, features_df.columns.tolist()

# Cross-validation for hyperparameter tuning
def tune_hyperparameters(features_df, labels):
    """Perform hyperparameter tuning using GridSearchCV"""
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        features_df, labels, test_size=0.2, random_state=42, stratify=labels
    )
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Define parameter grid for XGBoost
    param_grid = {
        'n_estimators': [100, 150, 200],
        'max_depth': [5, 7, 9],
        'learning_rate': [0.01, 0.05, 0.1],
        'subsample': [0.7, 0.8, 0.9],
        'colsample_bytree': [0.7, 0.8, 0.9]
    }
    
    # Perform grid search
    grid_search = GridSearchCV(
        XGBClassifier(random_state=42),
        param_grid,
        cv=5,
        scoring='roc_auc',
        n_jobs=-1,
        verbose=0
    )
    
    grid_search.fit(X_train_scaled, y_train)
    
    # Get best model
    best_model = grid_search.best_estimator_
    best_params = grid_search.best_params_
    
    # Evaluate
    y_pred = best_model.predict(X_test_scaled)
    y_pred_proba = best_model.predict_proba(X_test_scaled)[:, 1]
    
    metrics = {
        'best_params': best_params,
        'accuracy': accuracy_score(y_test, y_pred),
        'precision': precision_score(y_test, y_pred),
        'recall': recall_score(y_test, y_pred),
        'f1': f1_score(y_test, y_pred),
        'roc_auc': roc_auc_score(y_test, y_pred_proba),
        'best_score': grid_search.best_score_
    }
    
    # Feature importance
    feature_importance = pd.DataFrame({
        'feature': features_df.columns,
        'importance': best_model.feature_importances_
    }).sort_values('importance', ascending=False)
    
    return best_model, metrics, features_df.columns.tolist(), feature_importance

# Streamlit App Interface
st.title("🎾 Advanced Tennis Prediction System")
st.markdown(f"Enhanced ELO + Ensemble Models + Advanced Form Analysis (last {RECENT_MATCHES_COUNT} matches)")

# Sidebar navigation
st.sidebar.title("Navigation")
app_mode = st.sidebar.radio(
    "Choose a section:",
    ["📊 Data & Model Training", "🎯 Match Prediction", "📈 Player Analysis", "🤖 Model Insights", "⚙️ Model Optimization"]
)

# Main app logic
if app_mode == "📊 Data & Model Training":
    st.header("Data Upload & Advanced Model Training")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        uploaded_file = st.file_uploader(
            "Upload ATP Match Data CSV",
            type=['csv'],
            help="Upload match data with Tournament, Date, Player_1, Player_2, Winner, Surface, etc."
        )
    
    with col2:
        st.subheader("Training Options")
        
        model_type = st.selectbox(
            "Model Type",
            ["Ensemble Model", "XGBoost Only", "Hyperparameter Tuned"]
        )
        
        elo_k = st.slider("ELO K-factor", 10, 50, 32)
        initial_elo = st.slider("Initial ELO", 1200, 1800, 1500)
        
        if model_type == "Hyperparameter Tuned":
            st.info("GridSearchCV will optimize parameters (takes longer)")
    
    if uploaded_file is not None:
        try:
            df = pd.read_csv(uploaded_file)
            st.session_state.match_data = df
            
            # Show data info
            st.subheader("Data Preview")
            st.dataframe(df.head(), use_container_width=True)
            
            # Check required columns
            required_cols = ['Player_1', 'Player_2', 'Winner', 'Surface']
            missing_cols = [col for col in required_cols if col not in df.columns]
            
            if missing_cols:
                st.error(f"Missing required columns: {missing_cols}")
            else:
                st.success(f"✅ Data loaded successfully: {len(df)} matches")
                
                if st.button("Train Advanced Models", type="primary"):
                    with st.spinner("Training advanced models..."):
                        # Calculate advanced ELO
                        elo_ratings, global_ratings = compute_advanced_elo_from_csv(
                            df, k_factor_base=elo_k, initial_elo=initial_elo
                        )
                        st.session_state.elo_ratings = elo_ratings
                        st.session_state.global_elo = global_ratings
                        
                        # Create enhanced features
                        features_df, labels = create_enhanced_features(
                            df, elo_ratings, global_ratings, 
                            st.session_state.player_form_history,
                            st.session_state.match_history
                        )
                        
                        if len(features_df) > 0 and len(labels) > 0:
                            # Train selected model type
                            if model_type == "Ensemble Model":
                                model, metrics, feature_cols = train_ensemble_model(features_df, labels)
                                st.session_state.ensemble_model = model
                            elif model_type == "Hyperparameter Tuned":
                                model, metrics, feature_cols, feature_imp = tune_hyperparameters(features_df, labels)
                                st.session_state.xgb_model = model
                                st.session_state.feature_importance = feature_imp
                            else:  # XGBoost Only
                                model, metrics, feature_cols, feature_imp = tune_hyperparameters(features_df, labels)
                                st.session_state.xgb_model = model
                                st.session_state.feature_importance = feature_imp
                            
                            st.session_state.feature_columns = feature_cols
                            st.session_state.model_metrics = metrics
                            
                            # Display results
                            st.success(f"✅ {model_type} trained successfully!")
                            
                            # Show metrics
                            st.subheader("Model Performance Metrics")
                            metrics_cols = st.columns(4)
                            with metrics_cols[0]:
                                st.metric("Accuracy", f"{metrics['accuracy']:.2%}")
                            with metrics_cols[1]:
                                st.metric("Precision", f"{metrics['precision']:.2%}")
                            with metrics_cols[2]:
                                st.metric("Recall", f"{metrics['recall']:.2%}")
                            with metrics_cols[3]:
                                st.metric("F1 Score", f"{metrics['f1']:.2%}")
                            
                            if 'roc_auc' in metrics:
                                st.metric("ROC AUC", f"{metrics['roc_auc']:.3f}")
                            
                            # Show confusion matrix
                            if 'confusion_matrix' in metrics:
                                st.subheader("Confusion Matrix")
                                cm = metrics['confusion_matrix']
                                cm_df = pd.DataFrame(cm, 
                                                   columns=['Predicted Loss', 'Predicted Win'],
                                                   index=['Actual Loss', 'Actual Win'])
                                st.dataframe(cm_df)
                            
                            # Show feature importance
                            st.subheader("Top 15 Feature Importances")
                            if hasattr(st.session_state, 'feature_importance'):
                                top_features = st.session_state.feature_importance.head(15)
                                st.bar_chart(top_features.set_index('feature')['importance'])
                                
                                # Categorize top features
                                st.subheader("Feature Categories")
                                form_features = [f for f in top_features['feature'] if any(x in f for x in ['pct', 'momentum', 'streak', 'consistency', 'fatigue'])]
                                elo_features = [f for f in top_features['feature'] if 'elo' in f.lower()]
                                rank_features = [f for f in top_features['feature'] if 'rank' in f.lower()]
                                context_features = [f for f in top_features['feature'] if any(x in f for x in ['h2h', 'tourney', 'final', 'slam'])]
                                
                                if form_features:
                                    st.info(f"**Form Features:** {', '.join(form_features[:3])}")
                                if elo_features:
                                    st.info(f"**ELO Features:** {', '.join(elo_features[:3])}")
                                if rank_features:
                                    st.info(f"**Rank Features:** {', '.join(rank_features[:3])}")
                                if context_features:
                                    st.info(f"**Context Features:** {', '.join(context_features[:3])}")
                                
                        else:
                            st.warning("Could not create features from data")
                
        except Exception as e:
            st.error(f"Error: {str(e)}")

elif app_mode == "⚙️ Model Optimization":
    st.header("Model Optimization & Feature Engineering")
    
    if not st.session_state.match_data:
        st.warning("Please upload data first.")
    else:
        st.subheader("Feature Selection")
        
        # Manual feature selection
        if hasattr(st.session_state, 'feature_columns') and st.session_state.feature_columns:
            selected_features = st.multiselect(
                "Select features to include in model",
                options=st.session_state.feature_columns,
                default=st.session_state.feature_columns[:20]  # Top 20 by default
            )
            
            if st.button("Retrain with Selected Features"):
                with st.spinner("Retraining model..."):
                    # Recreate features with selected subset
                    features_df, labels = create_enhanced_features(
                        st.session_state.match_data,
                        st.session_state.elo_ratings,
                        st.session_state.global_elo,
                        st.session_state.player_form_history,
                        st.session_state.match_history
                    )
                    
                    if len(selected_features) > 0:
                        features_df = features_df[selected_features]
                        
                        # Retrain model
                        model, metrics, feature_cols = train_ensemble_model(features_df, labels)
                        st.session_state.ensemble_model = model
                        st.session_state.feature_columns = feature_cols
                        st.session_state.model_metrics = metrics
                        
                        st.success(f"✅ Model retrained with {len(selected_features)} features!")
                        st.metric("New Accuracy", f"{metrics['accuracy']:.2%}")
        
        # Feature correlation analysis
        st.subheader("Feature Correlation Analysis")
        if hasattr(st.session_state, 'feature_columns') and len(st.session_state.feature_columns) > 0:
            if st.button("Analyze Feature Correlations"):
                with st.spinner("Calculating correlations..."):
                    features_df, _ = create_enhanced_features(
                        st.session_state.match_data,
                        st.session_state.elo_ratings,
                        st.session_state.global_elo,
                        st.session_state.player_form_history,
                        st.session_state.match_history
                    )
                    
                    # Calculate correlation matrix
                    corr_matrix = features_df.corr().abs()
                    
                    # Find highly correlated features
                    high_corr = []
                    for i in range(len(corr_matrix.columns)):
                        for j in range(i+1, len(corr_matrix.columns)):
                            if corr_matrix.iloc[i, j] > 0.8:
                                high_corr.append((corr_matrix.columns[i], corr_matrix.columns[j], corr_matrix.iloc[i, j]))
                    
                    if high_corr:
                        st.warning(f"Found {len(high_corr)} pairs of highly correlated features (>0.8):")
                        for feat1, feat2, corr in high_corr[:10]:  # Show top 10
                            st.write(f"{feat1} - {feat2}: {corr:.3f}")
                    else:
                        st.success("No highly correlated feature pairs found")

# The rest of the app modes (🎯 Match Prediction, 📈 Player Analysis, 🤖 Model Insights)
# would be updated similarly with the enhanced features

# ... (Rest of the app code would follow similar pattern)

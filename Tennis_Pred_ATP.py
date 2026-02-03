import streamlit as st
import pandas as pd
import numpy as np
import math
import warnings
from collections import defaultdict, deque
warnings.filterwarnings('ignore')

# Try to import XGBoost
try:
    import xgboost as xgb
    from xgboost import XGBClassifier
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import accuracy_score
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
    st.session_state.player_form_history = {}  # Store recent match history
if 'player_ids' not in st.session_state:
    st.session_state.player_ids = {}  # Map player names to IDs

# Function to create player IDs from names
def create_player_ids(df):
    """Create unique IDs for players based on their names"""
    players = set()
    player_ids = {}
    
    # Collect all unique players
    if 'Player_1' in df.columns and 'Player_2' in df.columns:
        players.update(df['Player_1'].dropna().unique())
        players.update(df['Player_2'].dropna().unique())
    
    # Create ID mapping
    for idx, player_name in enumerate(sorted(players)):
        if pd.isna(player_name):
            continue
        player_id = f"P{idx:04d}"
        player_ids[str(player_name).strip()] = player_id
        st.session_state.player_names[player_id] = str(player_name).strip()
    
    return player_ids

# Function to compute surface-aware ELO ratings and track form for new CSV format
def compute_surface_elo_from_csv(df, k_factor=32, initial_elo=1500):
    # Create player IDs if not already created
    if not st.session_state.player_ids:
        st.session_state.player_ids = create_player_ids(df)
    
    # Sort chronologically
    if 'Date' in df.columns:
        df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
        df = df.sort_values(by=['Date']).reset_index(drop=True)
    
    # Map player names to IDs
    df['winner_id'] = df['Winner'].apply(lambda x: st.session_state.player_ids.get(str(x).strip()) if pd.notna(x) else None)
    df['loser_id'] = df.apply(lambda row: st.session_state.player_ids.get(
        str(row['Player_1']).strip() if str(row['Player_1']).strip() != str(row['Winner']).strip() else str(row['Player_2']).strip()
    ) if pd.notna(row['Player_1']) and pd.notna(row['Winner']) else None, axis=1)
    
    # Get all unique player ids
    players = set(df['winner_id'].dropna().unique()).union(set(df['loser_id'].dropna().unique()))
    
    # Initialize structures
    elo_ratings = {}
    global_ratings = {}
    form_history = defaultdict(lambda: defaultdict(deque))  # player_id -> surface -> deque of last 5 matches
    
    surfaces = ['Hard', 'Clay', 'Grass', 'Carpet']
    
    for player in players:
        if player:  # Skip None values
            elo_ratings[player] = {}
            for surface in surfaces:
                elo_ratings[player][surface] = initial_elo
            global_ratings[player] = initial_elo
    
    # Process matches chronologically
    for index, row in df.iterrows():
        winner = row['winner_id']
        loser = row['loser_id']
        
        if pd.isna(winner) or pd.isna(loser):
            continue
        
        # Get surface
        surface = str(row.get('Surface', 'Hard')).strip()
        if pd.isna(surface) or surface not in surfaces:
            surface = 'Hard'
        
        # Get current ratings
        rating_w = elo_ratings.get(winner, {}).get(surface, global_ratings.get(winner, initial_elo))
        rating_l = elo_ratings.get(loser, {}).get(surface, global_ratings.get(loser, initial_elo))
        
        # Update ELO
        exp_w = 1 / (1 + math.pow(10, (rating_l - rating_w) / 400))
        exp_l = 1 - exp_w
        
        # Initialize if not present
        if winner not in elo_ratings:
            elo_ratings[winner] = {}
        if loser not in elo_ratings:
            elo_ratings[loser] = {}
        
        elo_ratings[winner][surface] = rating_w + k_factor * (1 - exp_w)
        elo_ratings[loser][surface] = rating_l + k_factor * (0 - exp_l)
        
        global_ratings[winner] = global_ratings.get(winner, initial_elo) + k_factor * (1 - exp_w)
        global_ratings[loser] = global_ratings.get(loser, initial_elo) + k_factor * (0 - exp_l)
        
        # Track form (recent matches)
        winner_result = {
            'date': row.get('Date', 0),
            'opponent': loser,
            'surface': surface,
            'won': True,
            'score': row.get('Score', ''),
            'winner_elo_before': rating_w,
            'loser_elo_before': rating_l,
            'elo_change': k_factor * (1 - exp_w)
        }
        
        loser_result = {
            'date': row.get('Date', 0),
            'opponent': winner,
            'surface': surface,
            'won': False,
            'score': row.get('Score', ''),
            'winner_elo_before': rating_w,
            'loser_elo_before': rating_l,
            'elo_change': k_factor * (0 - exp_l)
        }
        
        # Add to form history (keep last 5 matches per surface)
        if winner not in form_history:
            form_history[winner] = defaultdict(deque)
        if loser not in form_history:
            form_history[loser] = defaultdict(deque)
        
        form_history[winner][surface].append(winner_result)
        if len(form_history[winner][surface]) > 5:
            form_history[winner][surface].popleft()
        
        form_history[loser][surface].append(loser_result)
        if len(form_history[loser][surface]) > 5:
            form_history[loser][surface].popleft()
    
    st.session_state.player_form_history = form_history
    return elo_ratings, global_ratings

# Function to calculate form features from recent matches
def calculate_form_features(player_id, surface, form_history, current_elo):
    """Calculate form features from last 5 matches on same surface"""
    if player_id not in form_history or surface not in form_history[player_id]:
        return {
            'recent_wins': 0,
            'recent_matches': 0,
            'win_percentage': 0.5,
            'avg_opponent_elo': current_elo,
            'form_momentum': 0,
            'avg_elo_change': 0,
            'straight_set_wins': 0,
            'lost_sets': 0,
            'recent_opponent_rank_avg': 100
        }
    
    recent_matches = list(form_history[player_id][surface])
    if not recent_matches:
        return {
            'recent_wins': 0,
            'recent_matches': 0,
            'win_percentage': 0.5,
            'avg_opponent_elo': current_elo,
            'form_momentum': 0,
            'avg_elo_change': 0,
            'straight_set_wins': 0,
            'lost_sets': 0,
            'recent_opponent_rank_avg': 100
        }
    
    wins = sum(1 for match in recent_matches if match['won'])
    total_matches = len(recent_matches)
    win_percentage = wins / total_matches if total_matches > 0 else 0.5
    
    # Calculate opponent strength
    opponent_elos = []
    for match in recent_matches:
        if match['won']:
            opponent_elos.append(match['loser_elo_before'])
        else:
            opponent_elos.append(match['winner_elo_before'])
    
    avg_opponent_elo = np.mean(opponent_elos) if opponent_elos else current_elo
    
    # Calculate form momentum (recent performance trend)
    if len(recent_matches) >= 3:
        recent_wins = [1 if match['won'] else 0 for match in recent_matches[-3:]]
        form_momentum = sum(recent_wins) / 3
    else:
        form_momentum = win_percentage
    
    # Average ELO change
    elo_changes = [match['elo_change'] for match in recent_matches]
    avg_elo_change = np.mean(elo_changes) if elo_changes else 0
    
    # Analyze set scores
    straight_set_wins = 0
    lost_sets = 0
    
    for match in recent_matches:
        if match['won'] and match.get('score'):
            score = match['score']
            # Simple check for straight set win (no sets lost by winner)
            if isinstance(score, str) and '-' in score:
                sets = score.split()
                # Check if it's a best of 3 match
                if len(sets) <= 3:  # Assuming best of 3
                    if len(sets) == 2:  # Won in straight sets
                        straight_set_wins += 1
                    elif len(sets) == 3:  # Went to 3 sets
                        # Check if winner lost any sets
                        winner_sets = 0
                        for set_score in sets:
                            try:
                                w, l = map(int, set_score.split('-'))
                                if w > l:
                                    winner_sets += 1
                            except:
                                pass
                        if winner_sets == 2:  # Won 2-1
                            lost_sets += 1
    
    return {
        'recent_wins': wins,
        'recent_matches': total_matches,
        'win_percentage': win_percentage,
        'avg_opponent_elo': avg_opponent_elo,
        'form_momentum': form_momentum,
        'avg_elo_change': avg_elo_change,
        'straight_set_wins': straight_set_wins,
        'lost_sets': lost_sets,
        'recent_opponent_rank_avg': avg_opponent_elo  # Using ELO as proxy for rank
    }

# Enhanced feature preparation with form data for new CSV format
def prepare_features_with_form(df, elo_ratings, global_ratings, form_history):
    """Prepare match data with ELO and form features"""
    features_list = []
    labels = []
    
    # Sort by date to ensure chronological order
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
            
            # Get current ELO ratings
            winner_elo = elo_ratings.get(winner_id, {}).get(surface, global_ratings.get(winner_id, 1500))
            loser_elo = elo_ratings.get(loser_id, {}).get(surface, global_ratings.get(loser_id, 1500))
            
            # Calculate form features
            winner_form = calculate_form_features(winner_id, surface, form_history, winner_elo)
            loser_form = calculate_form_features(loser_id, surface, form_history, loser_elo)
            
            # Calculate ELO difference and probability
            elo_diff = winner_elo - loser_elo
            elo_expected = 1 / (1 + math.pow(10, (-elo_diff) / 400))
            
            # Get rankings (handle missing values)
            try:
                winner_rank = float(row.get('Rank_1', 100)) if str(row['Player_1']).strip() == str(row['Winner']).strip() else float(row.get('Rank_2', 100))
                loser_rank = float(row.get('Rank_2', 100)) if str(row['Player_2']).strip() != str(row['Winner']).strip() else float(row.get('Rank_1', 100))
            except:
                winner_rank = 100.0
                loser_rank = 100.0
            
            # Get Best of value
            try:
                best_of = int(row.get('Best of', 3))
            except:
                best_of = 3
            
            # Get Round information
            round_info = str(row.get('Round', '1st Round'))
            is_final = 1 if 'Final' in round_info else 0
            is_semifinal = 1 if 'Semifinal' in round_info else 0
            is_quarterfinal = 1 if 'Quarterfinal' in round_info else 0
            
            # Create feature vector with form data
            features = {
                # Basic ELO features
                'elo_diff': float(elo_diff),
                'winner_elo': float(winner_elo),
                'loser_elo': float(loser_elo),
                'elo_expected': float(elo_expected),
                
                # Surface encoding
                'is_hard': 1 if surface == 'Hard' else 0,
                'is_clay': 1 if surface == 'Clay' else 0,
                'is_grass': 1 if surface == 'Grass' else 0,
                'is_carpet': 1 if surface == 'Carpet' else 0,
                
                # Player rankings
                'winner_rank': winner_rank,
                'loser_rank': loser_rank,
                
                # Winner form features
                'winner_recent_wins': float(winner_form['recent_wins']),
                'winner_win_pct': float(winner_form['win_percentage']),
                'winner_form_momentum': float(winner_form['form_momentum']),
                'winner_avg_opp_elo': float(winner_form['avg_opponent_elo']),
                'winner_avg_elo_change': float(winner_form['avg_elo_change']),
                'winner_straight_set_wins': float(winner_form['straight_set_wins']),
                'winner_lost_sets': float(winner_form['lost_sets']),
                
                # Loser form features
                'loser_recent_wins': float(loser_form['recent_wins']),
                'loser_win_pct': float(loser_form['win_percentage']),
                'loser_form_momentum': float(loser_form['form_momentum']),
                'loser_avg_opp_elo': float(loser_form['avg_opponent_elo']),
                'loser_avg_elo_change': float(loser_form['avg_elo_change']),
                'loser_straight_set_wins': float(loser_form['straight_set_wins']),
                'loser_lost_sets': float(loser_form['lost_sets']),
                
                # Form differentials
                'form_diff': float(winner_form['win_percentage'] - loser_form['win_percentage']),
                'momentum_diff': float(winner_form['form_momentum'] - loser_form['form_momentum']),
                'opp_strength_diff': float(winner_form['avg_opponent_elo'] - loser_form['avg_opponent_elo']),
                
                # Match format and round
                'best_of': float(best_of),
                'is_final': is_final,
                'is_semifinal': is_semifinal,
                'is_quarterfinal': is_quarterfinal,
            }
            
            # Add odds if available
            if 'Odd_1' in row and pd.notna(row['Odd_1']) and row['Odd_1'] != -1:
                features['winner_odds'] = float(row['Odd_1']) if str(row['Player_1']).strip() == str(row['Winner']).strip() else float(row.get('Odd_2', 2.0))
                features['loser_odds'] = float(row['Odd_2']) if str(row['Player_2']).strip() != str(row['Winner']).strip() else float(row.get('Odd_1', 2.0))
            else:
                features['winner_odds'] = 2.0
                features['loser_odds'] = 2.0
            
            features_list.append(features)
            labels.append(1)  # Winner perspective
            
            # Also create reverse perspective for balanced training
            features_reverse = features.copy()
            
            # Swap ELO features
            features_reverse['elo_diff'] = -features['elo_diff']
            features_reverse['winner_elo'], features_reverse['loser_elo'] = features['loser_elo'], features['winner_elo']
            features_reverse['elo_expected'] = 1 - features['elo_expected']
            
            # Swap rankings
            features_reverse['winner_rank'], features_reverse['loser_rank'] = features['loser_rank'], features['winner_rank']
            
            # Swap form features
            form_swap_pairs = [
                ('winner_recent_wins', 'loser_recent_wins'),
                ('winner_win_pct', 'loser_win_pct'),
                ('winner_form_momentum', 'loser_form_momentum'),
                ('winner_avg_opp_elo', 'loser_avg_opp_elo'),
                ('winner_avg_elo_change', 'loser_avg_elo_change'),
                ('winner_straight_set_wins', 'loser_straight_set_wins'),
                ('winner_lost_sets', 'loser_lost_sets')
            ]
            
            for w_feat, l_feat in form_swap_pairs:
                features_reverse[w_feat], features_reverse[l_feat] = features[l_feat], features[w_feat]
            
            # Reverse differentials
            features_reverse['form_diff'] = -features['form_diff']
            features_reverse['momentum_diff'] = -features['momentum_diff']
            features_reverse['opp_strength_diff'] = -features['opp_strength_diff']
            
            # Swap odds
            features_reverse['winner_odds'], features_reverse['loser_odds'] = features['loser_odds'], features['winner_odds']
            
            features_list.append(features_reverse)
            labels.append(0)  # Loser perspective
            
        except Exception as e:
            continue
    
    features_df = pd.DataFrame(features_list)
    return features_df, np.array(labels)

# Train XGBoost model with form features
def train_xgboost_model_with_form(features_df, labels, params=None):
    """Train XGBoost classifier with form features"""
    if params is None:
        params = {
            'max_depth': 6,
            'learning_rate': 0.1,
            'n_estimators': 100,
            'objective': 'binary:logistic',
            'eval_metric': 'logloss',
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'random_state': 42
        }
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        features_df, labels, test_size=0.2, random_state=42, stratify=labels
    )
    
    # Train model
    model = XGBClassifier(**params)
    model.fit(
        X_train, y_train,
        eval_set=[(X_train, y_train), (X_test, y_test)],
        verbose=False
    )
    
    # Evaluate
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    
    return model, accuracy, features_df.columns.tolist()

# Enhanced prediction with form analysis
def hybrid_prediction_with_form(player_a_id, player_b_id, surface, elo_ratings, global_ratings, form_history, xgb_model, feature_columns):
    """Make prediction using ELO, XGBoost, and recent form"""
    
    # Get current ELO ratings
    player_a_elo = elo_ratings.get(player_a_id, {}).get(surface, global_ratings.get(player_a_id, 1500))
    player_b_elo = elo_ratings.get(player_b_id, {}).get(surface, global_ratings.get(player_b_id, 1500))
    
    # Calculate form features
    player_a_form = calculate_form_features(player_a_id, surface, form_history, player_a_elo)
    player_b_form = calculate_form_features(player_b_id, surface, form_history, player_b_elo)
    
    # Calculate ELO-based prediction
    elo_diff = float(player_a_elo - player_b_elo)
    elo_prob = float(1 / (1 + math.pow(10, (-elo_diff) / 400)))
    
    # Prepare features for XGBoost
    features = {
        # Basic ELO features
        'elo_diff': elo_diff,
        'winner_elo': float(player_a_elo),
        'loser_elo': float(player_b_elo),
        'elo_expected': elo_prob,
        
        # Surface encoding
        'is_hard': 1 if surface == 'Hard' else 0,
        'is_clay': 1 if surface == 'Clay' else 0,
        'is_grass': 1 if surface == 'Grass' else 0,
        'is_carpet': 1 if surface == 'Carpet' else 0,
        
        # Player rankings (using average)
        'winner_rank': 50.0,
        'loser_rank': 50.0,
        
        # Player A form features (as winner)
        'winner_recent_wins': float(player_a_form['recent_wins']),
        'winner_win_pct': float(player_a_form['win_percentage']),
        'winner_form_momentum': float(player_a_form['form_momentum']),
        'winner_avg_opp_elo': float(player_a_form['avg_opponent_elo']),
        'winner_avg_elo_change': float(player_a_form['avg_elo_change']),
        'winner_straight_set_wins': float(player_a_form['straight_set_wins']),
        'winner_lost_sets': float(player_a_form['lost_sets']),
        
        # Player B form features (as loser)
        'loser_recent_wins': float(player_b_form['recent_wins']),
        'loser_win_pct': float(player_b_form['win_percentage']),
        'loser_form_momentum': float(player_b_form['form_momentum']),
        'loser_avg_opp_elo': float(player_b_form['avg_opponent_elo']),
        'loser_avg_elo_change': float(player_b_form['avg_elo_change']),
        'loser_straight_set_wins': float(player_b_form['straight_set_wins']),
        'loser_lost_sets': float(player_b_form['lost_sets']),
        
        # Form differentials
        'form_diff': float(player_a_form['win_percentage'] - player_b_form['win_percentage']),
        'momentum_diff': float(player_a_form['form_momentum'] - player_b_form['form_momentum']),
        'opp_strength_diff': float(player_a_form['avg_opponent_elo'] - player_b_form['avg_opponent_elo']),
        
        # Match format (using defaults)
        'best_of': 3.0,
        'is_final': 0,
        'is_semifinal': 0,
        'is_quarterfinal': 0,
        
        # Odds (using defaults)
        'winner_odds': 2.0,
        'loser_odds': 2.0
    }
    
    # Create feature DataFrame
    features_df = pd.DataFrame([features])
    
    # Ensure all training columns are present
    if feature_columns:
        for col in feature_columns:
            if col not in features_df.columns:
                features_df[col] = 0.0
        
        features_df = features_df[feature_columns]
    
    # Get XGBoost prediction
    if xgb_model is not None and not features_df.empty:
        try:
            xgb_prob = float(xgb_model.predict_proba(features_df)[0, 1])
            
            # Weighted combination with emphasis on form
            elo_weight = 0.25
            xgb_weight = 0.75  # Higher weight for model with form features
            
            final_prob = float(elo_weight * elo_prob + xgb_weight * xgb_prob)
        except:
            xgb_prob = None
            final_prob = elo_prob
    else:
        xgb_prob = None
        final_prob = elo_prob
    
    return {
        'elo_probability': elo_prob,
        'xgb_probability': xgb_prob,
        'final_probability': final_prob,
        'player_a_elo': player_a_elo,
        'player_b_elo': player_b_elo,
        'elo_difference': elo_diff,
        'player_a_form': player_a_form,
        'player_b_form': player_b_form
    }

# Streamlit App Interface
st.title("🎾 Tennis Prediction System with Form Analysis")
st.markdown("ELO ratings + XGBoost + Recent Form Analysis (last 5 matches on same surface)")

# Sidebar navigation
st.sidebar.title("Navigation")
app_mode = st.sidebar.radio(
    "Choose a section:",
    ["📊 Data & Model Training", "🎯 Match Prediction", "📈 Player Analysis", "🤖 Model Insights"]
)

# Check XGBoost availability
if not XGB_AVAILABLE and app_mode in ["🤖 Model Insights", "🎯 Match Prediction"]:
    st.error("""
    **XGBoost not installed!**
    
    Please install XGBoost to use machine learning features:
    ```
    pip install xgboost scikit-learn
    ```
    """)

# Main app logic
if app_mode == "📊 Data & Model Training":
    st.header("Data Upload & Model Training")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        uploaded_file = st.file_uploader(
            "Upload ATP Match Data CSV",
            type=['csv'],
            help="Upload match data with Tournament, Date, Player_1, Player_2, Winner, Surface, etc."
        )
    
    with col2:
        st.subheader("Model Parameters")
        
        elo_k = st.slider("ELO K-factor", 10, 50, 32)
        initial_elo = st.slider("Initial ELO", 1200, 1800, 1500)
        
        if XGB_AVAILABLE:
            use_xgb = st.checkbox("Train XGBoost Model", value=True)
            xgb_depth = st.slider("XGBoost Max Depth", 3, 10, 6)
            xgb_estimators = st.slider("Number of Trees", 50, 200, 100)
        else:
            use_xgb = False
    
    if uploaded_file is not None:
        try:
            df = pd.read_csv(uploaded_file)
            
            # Clean and prepare data
            st.session_state.match_data = df
            
            # Show data info
            st.subheader("Data Preview")
            st.dataframe(df.head(), use_container_width=True)
            
            # Show column information
            st.subheader("Data Information")
            col_info1, col_info2 = st.columns(2)
            
            with col_info1:
                st.write(f"**Total Matches:** {len(df)}")
                st.write(f"**Total Tournaments:** {df['Tournament'].nunique() if 'Tournament' in df.columns else 'N/A'}")
                if 'Date' in df.columns:
                    st.write(f"**Date Range:** {df['Date'].min()} to {df['Date'].max()}")
                else:
                    st.write("**Date Range:** N/A")
            
            with col_info2:
                if 'Player_1' in df.columns and 'Player_2' in df.columns:
                    unique_players = len(set(df['Player_1'].dropna().unique()) | set(df['Player_2'].dropna().unique()))
                    st.write(f"**Unique Players:** {unique_players}")
                else:
                    st.write("**Unique Players:** N/A")
                
                if 'Surface' in df.columns:
                    surfaces = df['Surface'].unique()
                    st.write(f"**Surfaces:** {', '.join(map(str, surfaces))}")
                else:
                    st.write("**Surfaces:** N/A")
                
                if 'Round' in df.columns:
                    rounds = df['Round'].unique()[:5]  # Show first 5 unique rounds
                    st.write(f"**Rounds:** {', '.join(map(str, rounds))}...")
                else:
                    st.write("**Rounds:** N/A")
            
            # Check required columns
            required_cols = ['Player_1', 'Player_2', 'Winner', 'Surface']
            missing_cols = [col for col in required_cols if col not in df.columns]
            
            if missing_cols:
                st.error(f"Missing required columns: {missing_cols}")
            else:
                st.success(f"✅ Data loaded successfully: {len(df)} matches")
                
                # Calculate ELO ratings and track form
                if st.button("Calculate ELO Ratings & Train Models", type="primary"):
                    with st.spinner("Calculating ELO ratings and tracking form..."):
                        elo_ratings, global_ratings = compute_surface_elo_from_csv(
                            df, k_factor=elo_k, initial_elo=initial_elo
                        )
                        st.session_state.elo_ratings = elo_ratings
                        st.session_state.global_elo = global_ratings
                    
                    st.success(f"✅ ELO calculated for {len(elo_ratings)} players")
                    st.info(f"📊 Form history tracked for {len(st.session_state.player_form_history)} players")
                    
                    # Show top players
                    top_players = sorted(
                        [(pid, st.session_state.global_elo.get(pid, 1500)) 
                         for pid in elo_ratings.keys() if pid],
                        key=lambda x: x[1], reverse=True
                    )[:10]
                    
                    st.subheader("Top 10 Players (Global ELO)")
                    if top_players:
                        top_df = pd.DataFrame(top_players, columns=['Player ID', 'ELO Rating'])
                        top_df['Player Name'] = top_df['Player ID'].map(st.session_state.player_names)
                        top_df = top_df[['Player Name', 'ELO Rating', 'Player ID']]
                        st.dataframe(top_df, use_container_width=True)
                    else:
                        st.warning("No players found in ELO ratings")
                    
                    # Train XGBoost model with form features
                    if XGB_AVAILABLE and use_xgb and st.session_state.elo_ratings:
                        with st.spinner("Training XGBoost model with form features..."):
                            # Prepare features with form data
                            features_df, labels = prepare_features_with_form(
                                df, elo_ratings, global_ratings, st.session_state.player_form_history
                            )
                            
                            if len(features_df) > 0 and len(labels) > 0:
                                # Train model
                                xgb_params = {
                                    'max_depth': xgb_depth,
                                    'n_estimators': xgb_estimators,
                                    'learning_rate': 0.1,
                                    'objective': 'binary:logistic',
                                    'random_state': 42
                                }
                                
                                model, accuracy, feature_cols = train_xgboost_model_with_form(
                                    features_df, labels, xgb_params
                                )
                                
                                st.session_state.xgb_model = model
                                st.session_state.feature_columns = feature_cols
                                
                                st.success(f"✅ XGBoost trained with form features! Accuracy: {accuracy:.2%}")
                                
                                # Show feature importance
                                st.subheader("Top Feature Importances")
                                importance_df = pd.DataFrame({
                                    'feature': feature_cols,
                                    'importance': model.feature_importances_
                                }).sort_values('importance', ascending=False).head(15)
                                
                                st.bar_chart(importance_df.set_index('feature')['importance'])
                                
                                # Show which form features are most important
                                form_features = [f for f in importance_df['feature'] if 'recent' in f or 'form' in f or 'win_pct' in f or 'momentum' in f]
                                if form_features:
                                    st.info(f"**Key form features in model:** {', '.join(form_features[:5])}")
                            else:
                                st.warning("Could not prepare features for training. Check data quality.")
                
        except Exception as e:
            st.error(f"Error loading data: {str(e)}")

# Rest of the code remains the same as before...
elif app_mode == "🎯 Match Prediction":
    st.header("Match Prediction with Form Analysis")
    
    if not st.session_state.elo_ratings:
        st.warning("Please upload data and train models first in the 'Data & Model Training' section.")
    else:
        col1, col2 = st.columns(2)
        
        with col1:
            # Surface selection
            surface = st.selectbox("Court Surface", ['Hard', 'Clay', 'Grass', 'Carpet'])
            
            # Player A selection
            player_options = [(pid, name) for pid, name in st.session_state.player_names.items() 
                            if pid in st.session_state.elo_ratings]
            
            player_a_tuple = st.selectbox(
                "Player A",
                options=player_options,
                format_func=lambda x: x[1],
                key="player_a_select"
            )
            if player_a_tuple:
                player_a = player_a_tuple[0]
        
        with col2:
            # Player B selection
            player_b_options = [(pid, name) for pid, name in player_options if pid != player_a]
            
            player_b_tuple = st.selectbox(
                "Player B",
                options=player_b_options,
                format_func=lambda x: x[1],
                key="player_b_select"
            )
            if player_b_tuple:
                player_b = player_b_tuple[0]
            
            # Prediction options
            st.subheader("Prediction Options")
            use_xgb = st.checkbox("Use XGBoost with Form Analysis", 
                                 value=st.session_state.xgb_model is not None,
                                 help="Use machine learning model with recent form data")
            show_form_details = st.checkbox("Show form analysis", value=True)
            
            # Prediction button
            if st.button("Run Prediction with Form Analysis", type="primary", use_container_width=True):
                # Store prediction in session state
                st.session_state.prediction_result = hybrid_prediction_with_form(
                    player_a, player_b, surface,
                    st.session_state.elo_ratings,
                    st.session_state.global_elo,
                    st.session_state.player_form_history,
                    st.session_state.xgb_model if use_xgb else None,
                    st.session_state.feature_columns
                )
        
        # Display results if prediction exists
        if hasattr(st.session_state, 'prediction_result') and st.session_state.prediction_result:
            player_a_name = st.session_state.player_names.get(player_a, player_a)
            player_b_name = st.session_state.player_names.get(player_b, player_b)
            
            prediction = st.session_state.prediction_result
            
            st.subheader(f"Prediction: {player_a_name} vs {player_b_name}")
            st.markdown(f"**Surface:** {surface}")
            
            # Show probabilities
            col_prob_a, col_prob_b = st.columns(2)
            
            with col_prob_a:
                prob_a = float(prediction['final_probability'])
                st.metric(
                    label=player_a_name,
                    value=f"{prob_a:.1%}",
                    delta=f"Win Probability"
                )
                st.progress(min(1.0, max(0.0, prob_a)))
            
            with col_prob_b:
                prob_b = float(1 - prob_a)
                st.metric(
                    label=player_b_name,
                    value=f"{prob_b:.1%}",
                    delta=f"Win Probability"
                )
                st.progress(min(1.0, max(0.0, prob_b)))
            
            # Form analysis section
            if show_form_details:
                st.subheader("📊 Recent Form Analysis (Last 5 matches on same surface)")
                
                form_cols = st.columns(2)
                
                with form_cols[0]:
                    st.markdown(f"**{player_a_name} Form:**")
                    form_a = prediction['player_a_form']
                    st.write(f"Recent Wins: {form_a['recent_wins']}/5")
                    st.write(f"Win %: {form_a['win_percentage']:.1%}")
                    st.write(f"Form Momentum: {form_a['form_momentum']:.1%}")
                    st.write(f"Avg Opponent ELO: {form_a['avg_opponent_elo']:.0f}")
                    st.write(f"Straight Set Wins: {form_a['straight_set_wins']}")
                    
                    # Form indicator
                    if form_a['win_percentage'] >= 0.7:
                        st.success("🔥 Excellent Form")
                    elif form_a['win_percentage'] >= 0.5:
                        st.info("📈 Good Form")
                    else:
                        st.warning("⚠️ Needs Improvement")
                
                with form_cols[1]:
                    st.markdown(f"**{player_b_name} Form:**")
                    form_b = prediction['player_b_form']
                    st.write(f"Recent Wins: {form_b['recent_wins']}/5")
                    st.write(f"Win %: {form_b['win_percentage']:.1%}")
                    st.write(f"Form Momentum: {form_b['form_momentum']:.1%}")
                    st.write(f"Avg Opponent ELO: {form_b['avg_opponent_elo']:.0f}")
                    st.write(f"Straight Set Wins: {form_b['straight_set_wins']}")
                    
                    # Form indicator
                    if form_b['win_percentage'] >= 0.7:
                        st.success("🔥 Excellent Form")
                    elif form_b['win_percentage'] >= 0.5:
                        st.info("📈 Good Form")
                    else:
                        st.warning("⚠️ Needs Improvement")
                
                # Form comparison
                st.subheader("Form Comparison")
                form_diff = form_a['win_percentage'] - form_b['win_percentage']
                
                if abs(form_diff) > 0.3:
                    st.success(f"**Significant form advantage:** {player_a_name if form_diff > 0 else player_b_name}")
                elif abs(form_diff) > 0.15:
                    st.info(f"**Moderate form advantage:** {player_a_name if form_diff > 0 else player_b_name}")
                else:
                    st.info("**Form is relatively even**")
            
            # Prediction breakdown
            st.subheader("Prediction Breakdown")
            
            if prediction['xgb_probability'] is not None:
                cols = st.columns(3)
                with cols[0]:
                    st.metric("ELO Probability", f"{float(prediction['elo_probability']):.1%}")
                with cols[1]:
                    st.metric("XGBoost + Form", f"{float(prediction['xgb_probability']):.1%}")
                with cols[2]:
                    st.metric("Final Probability", f"{float(prediction['final_probability']):.1%}")
            else:
                st.info("Using ELO-only prediction")
            
            # ELO ratings
            st.markdown("**ELO Ratings**")
            elo_cols = st.columns(2)
            with elo_cols[0]:
                st.write(f"{player_a_name}: {float(prediction['player_a_elo']):.0f}")
            with elo_cols[1]:
                st.write(f"{player_b_name}: {float(prediction['player_b_elo']):.0f}")
                st.write(f"ELO Difference: {float(prediction['elo_difference']):.0f}")
            
            # Final verdict
            st.subheader("🎯 Final Verdict")
            if prob_a > 0.75:
                st.success(f"**Strong favorite:** {player_a_name} is heavily favored to win on {surface}!")
            elif prob_a > 0.65:
                st.info(f"**Clear favorite:** {player_a_name} is favored to win on {surface}!")
            elif prob_a > 0.55:
                st.info(f"**Slight favorite:** {player_a_name} has a small advantage on {surface}!")
            elif prob_b > 0.75:
                st.success(f"**Strong favorite:** {player_b_name} is heavily favored to win on {surface}!")
            elif prob_b > 0.65:
                st.info(f"**Clear favorite:** {player_b_name} is favored to win on {surface}!")
            elif prob_b > 0.55:
                st.info(f"**Slight favorite:** {player_b_name} has a small advantage on {surface}!")
            else:
                st.warning("**Too close to call!** Form and match conditions will be crucial.")
        else:
            st.info("👈 Select players and click 'Run Prediction with Form Analysis'")

elif app_mode == "📈 Player Analysis":
    st.header("Player Form Analysis")
    
    if not st.session_state.elo_ratings:
        st.warning("Please upload data first in the 'Data & Model Training' section.")
    else:
        # Player selection
        player_options = [(pid, name) for pid, name in st.session_state.player_names.items() 
                         if pid in st.session_state.elo_ratings]
        
        selected_tuple = st.selectbox(
            "Select Player",
            options=player_options,
            format_func=lambda x: x[1]
        )
        
        if selected_tuple:
            selected_player = selected_tuple[0]
            player_name = selected_tuple[1]
            
            # Surface selection
            selected_surface = st.selectbox(
                "Select Surface",
                ['All', 'Hard', 'Clay', 'Grass', 'Carpet']
            )
            
            # Get player's form history
            if selected_player in st.session_state.player_form_history:
                form_history = st.session_state.player_form_history[selected_player]
                
                st.subheader(f"Recent Form for {player_name}")
                
                if selected_surface == 'All':
                    surfaces_to_show = form_history.keys()
                else:
                    surfaces_to_show = [selected_surface] if selected_surface in form_history else []
                
                for surface in surfaces_to_show:
                    matches = list(form_history[surface])
                    if matches:
                        st.markdown(f"**{surface} Courts (Last {len(matches)} matches):**")
                        
                        # Calculate form metrics
                        wins = sum(1 for match in matches if match['won'])
                        win_pct = wins / len(matches)
                        
                        # Display form summary
                        col1, col2, col3 = st.columns(3)
                        with col1:
                            st.metric("Matches", len(matches))
                        with col2:
                            st.metric("Wins", wins)
                        with col3:
                            st.metric("Win %", f"{win_pct:.1%}")
                        
                        # Recent matches table
                        match_data = []
                        for match in matches:
                            opponent_id = match['opponent']
                            opponent_name = st.session_state.player_names.get(opponent_id, opponent_id)
                            match_data.append({
                                'Result': 'W' if match['won'] else 'L',
                                'Opponent': opponent_name,
                                'Score': match.get('score', ''),
                                'ELO Change': f"{match['elo_change']:+.1f}"
                            })
                        
                        if match_data:
                            st.dataframe(pd.DataFrame(match_data), use_container_width=True)
                    else:
                        st.info(f"No recent matches on {surface} courts")
            else:
                st.info("No form history available for this player")
            
            # Player ELO ratings across surfaces
            st.subheader("Surface-Specific ELO Ratings")
            if selected_player in st.session_state.elo_ratings:
                ratings = st.session_state.elo_ratings[selected_player]
                global_rating = st.session_state.global_elo.get(selected_player, 1500)
                
                ratings_data = []
                for surface in ['Hard', 'Clay', 'Grass', 'Carpet']:
                    rating = ratings.get(surface, global_rating)
                    ratings_data.append({
                        'Surface': surface,
                        'ELO Rating': float(rating),
                        'Difference from Global': float(rating - global_rating)
                    })
                
                ratings_df = pd.DataFrame(ratings_data)
                st.dataframe(ratings_df, use_container_width=True)
                
                # Specialization indicator
                if len(ratings) > 1:
                    surface_ratings = [r for r in ratings.values()]
                    max_rating = max(surface_ratings)
                    min_rating = min(surface_ratings)
                    specialization = max_rating - min_rating
                    
                    if specialization > 100:
                        st.warning(f"**Surface Specialist** ({specialization:.0f} point difference)")
                    elif specialization > 50:
                        st.info(f"**Surface Preference** ({specialization:.0f} point difference)")
                    else:
                        st.success(f"**All-Surface Player** ({specialization:.0f} point difference)")

elif app_mode == "🤖 Model Insights":
    st.header("Model Insights & Feature Analysis")
    
    if st.session_state.xgb_model is None:
        st.warning("No XGBoost model trained yet. Please train a model in the 'Data & Model Training' section.")
    else:
        st.success("✅ XGBoost model with form features is ready!")
        
        # Model information
        st.subheader("Model Configuration")
        st.json({
            "n_estimators": st.session_state.xgb_model.n_estimators,
            "max_depth": st.session_state.xgb_model.max_depth,
            "learning_rate": st.session_state.xgb_model.learning_rate,
            "n_features": len(st.session_state.feature_columns)
        })
        
        # Feature importance
        st.subheader("Feature Importance Analysis")
        
        importance_dict = st.session_state.xgb_model.get_booster().get_score(importance_type="weight")
        
        if importance_dict:
            importance_df = pd.DataFrame(
                list(importance_dict.items()),
                columns=['Feature', 'Importance']
            ).sort_values('Importance', ascending=False)
            
            # Categorize features
            form_features = []
            elo_features = []
            other_features = []
            
            for feature in importance_df['Feature']:
                if 'recent' in feature or 'form' in feature or 'win_pct' in feature or 'momentum' in feature:
                    form_features.append(feature)
                elif 'elo' in feature.lower():
                    elo_features.append(feature)
                else:
                    other_features.append(feature)
            
            # Display by category
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.markdown("**Top Form Features**")
                if form_features:
                    top_form = importance_df[importance_df['Feature'].isin(form_features)].head(5)
                    st.dataframe(top_form)
                else:
                    st.info("No form features in top features")
            
            with col2:
                st.markdown("**Top ELO Features**")
                if elo_features:
                    top_elo = importance_df[importance_df['Feature'].isin(elo_features)].head(5)
                    st.dataframe(top_elo)
                else:
                    st.info("No ELO features in top features")
            
            with col3:
                st.markdown("**Other Important Features**")
                if other_features:
                    top_other = importance_df[importance_df['Feature'].isin(other_features)].head(5)
                    st.dataframe(top_other)
                else:
                    st.info("No other features in top features")
        
        # How form analysis works
        st.subheader("How Form Analysis Works")
        st.markdown("""
        **Recent Form Tracking System:**
        
        1. **Last 5 Matches on Same Surface**
           - Tracks each player's performance on the specific surface
           - Only considers matches on the same surface as the upcoming match
        
        2. **Form Metrics Calculated:**
           - **Win Percentage**: Success rate in recent matches
           - **Form Momentum**: Performance trend (last 3 matches)
           - **Opponent Strength**: Average ELO of recent opponents
           - **Set Performance**: Straight set wins vs matches with lost sets
           - **ELO Momentum**: Average ELO change in recent matches
        
        3. **How It Improves Predictions:**
           - Identifies players in "hot streaks" or poor form
           - Accounts for surface-specific momentum
           - Considers quality of recent opponents
           - Captures recent performance better than overall ELO alone
        
        **Model Weighting:**
        - ELO Probability: 25%
        - XGBoost with Form Features: 75%
        
        This gives more weight to recent performance while still considering long-term skill level.
        """)

# Footer
st.markdown("---")
st.markdown(
    """
    **Enhanced Prediction System Features:**
    - **Form Analysis**: Last 5 matches on same surface
    - **Surface-Specific Tracking**: Separate form for each surface
    - **Momentum Calculation**: Performance trends
    - **Opponent Quality**: Strength of recent opponents
    - **Hybrid Model**: Combines ELO, form, and match statistics
    
    **Key Benefits:**
    1. Better accounts for recent player performance
    2. Surface-specific form tracking
    3. Identifies players in good/bad form
    4. More accurate predictions for current matchups
    
    **Dataset Features Used:**
    - Tournament, Date, Round
    - Player names and rankings (Rank_1, Rank_2)
    - Surface type (Hard, Clay, Grass, Carpet)
    - Match scores and results
    - Best of format
    """
)

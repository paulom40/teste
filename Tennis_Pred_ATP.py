import streamlit as st
import pandas as pd
import numpy as np
import math
import warnings
from collections import defaultdict, deque
from datetime import datetime, timedelta

warnings.filterwarnings('ignore')

try:
    from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix
    from sklearn.preprocessing import RobustScaler
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.calibration import CalibratedClassifierCV
    ML_AVAILABLE = True
except ImportError:
    ML_AVAILABLE = False

st.set_page_config(page_title="Tennis Prediction", page_icon="🎾", layout="wide")

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
if 'ensemble_model' not in st.session_state:
    st.session_state.ensemble_model = None
if 'scaler' not in st.session_state:
    st.session_state.scaler = RobustScaler()
if 'model_trained' not in st.session_state:
    st.session_state.model_trained = False
if 'match_data' not in st.session_state:
    st.session_state.match_data = None

SURFACE_TYPES = ['Hard', 'Clay', 'Grass', 'Carpet']

def get_default_form():
    """Default form dict"""
    return {
        'wins': 0, 'matches': 0, 'win_pct': 0.5, 'momentum': 0.5,
        'opp_elo': 1500, 'streak': 0, 'consistency': 0.5, 'fatigue': 0,
        'sets_played_avg': 3.0, 'games_played_avg': 22.0  # Added averages
    }

def compute_elo_simple(df, k_factor=32, initial_elo=1500):
    """Simple ELO computation - less memory intensive"""
    df = df.copy()
    
    # Map column names - your CSV has different column names
    column_mapping = {
        'Player_1': 'Player_1',
        'Player_2': 'Player_2', 
        'Winner': 'Winner',
        'Surface': 'Surface'
    }
    
    # Check if we need to rename columns
    actual_columns = set(df.columns)
    expected_columns = set(column_mapping.keys())
    
    # Try to find the right columns
    player1_col = None
    player2_col = None
    winner_col = None
    surface_col = None
    
    # Look for common column name patterns
    for col in df.columns:
        col_lower = col.lower()
        if 'player' in col_lower and ('1' in col_lower or 'player1' in col_lower):
            player1_col = col
        elif 'player' in col_lower and ('2' in col_lower or 'player2' in col_lower):
            player2_col = col
        elif 'winner' in col_lower:
            winner_col = col
        elif 'surface' in col_lower:
            surface_col = col
    
    # Use found columns or default to expected names
    if player1_col:
        df = df.rename(columns={player1_col: 'Player_1'})
    if player2_col:
        df = df.rename(columns={player2_col: 'Player_2'})
    if winner_col:
        df = df.rename(columns={winner_col: 'Winner'})
    if surface_col:
        df = df.rename(columns={surface_col: 'Surface'})
    
    # Clean data
    for col in ['Player_1', 'Player_2', 'Winner', 'Surface']:
        if col in df.columns:
            df[col] = df[col].astype(str).str.strip()
        else:
            st.error(f"Column '{col}' not found in CSV")
            st.info(f"Available columns: {list(df.columns)}")
            return 0
    
    # Get unique players
    all_players = set()
    all_players.update(df['Player_1'].unique())
    all_players.update(df['Player_2'].unique())
    
    # Create ID mapping
    player_ids = {name: f"P{idx:04d}" for idx, name in enumerate(sorted(all_players))}
    st.session_state.player_ids = player_ids
    st.session_state.player_names = {v: k for k, v in player_ids.items()}
    
    # Initialize ELO ratings
    elo_ratings = {pid: {s: initial_elo for s in SURFACE_TYPES} for pid in player_ids.values()}
    global_ratings = {pid: initial_elo for pid in player_ids.values()}
    match_history = defaultdict(list)
    
    # Process matches
    valid_count = 0
    for idx, row in df.iterrows():
        p1 = row['Player_1']
        p2 = row['Player_2']
        winner = row['Winner']
        surface = row['Surface'] if 'Surface' in row and pd.notna(row['Surface']) and str(row['Surface']).strip() in SURFACE_TYPES else 'Hard'
        
        # Try to get set/game information from Score column if available
        sets_played = 3  # Default
        games_played = 18  # Default
        
        if 'Score' in df.columns and pd.notna(row.get('Score')):
            try:
                score_str = str(row['Score'])
                # Simple parsing of score (e.g., "6-4 6-3" or "6-3 4-6 6-2")
                sets = score_str.split()
                sets_played = len(sets)
                # Estimate total games (sum of all numbers in score)
                total_games = 0
                for set_score in sets:
                    if '-' in set_score:
                        games = set_score.split('-')
                        if len(games) == 2:
                            try:
                                total_games += int(games[0]) + int(games[1])
                            except:
                                total_games += 12  # Default if parsing fails
                if total_games > 0:
                    games_played = total_games
            except:
                pass
        
        if winner not in [p1, p2]:
            continue
            
        valid_count += 1
        winner_id = player_ids[winner]
        loser_id = player_ids[p2 if winner == p1 else p1]
        
        # Get current ratings
        rating_w = elo_ratings[winner_id][surface]
        rating_l = elo_ratings[loser_id][surface]
        
        # Expected win probability
        exp_w = 1 / (1 + math.pow(10, (rating_l - rating_w) / 400))
        
        # Update ratings
        elo_ratings[winner_id][surface] = rating_w + k_factor * (1 - exp_w)
        elo_ratings[loser_id][surface] = rating_l + k_factor * (0 - (1 - exp_w))
        
        # Update global ratings (average of surface ratings)
        global_ratings[winner_id] = np.mean(list(elo_ratings[winner_id].values()))
        global_ratings[loser_id] = np.mean(list(elo_ratings[loser_id].values()))
        
        # Store match history
        match_history[winner_id].append({
            'date': pd.Timestamp.now(),
            'surface': surface,
            'opponent': loser_id,
            'won': True,
            'elo': rating_w,
            'sets_played': sets_played,
            'games_played': games_played
        })
        match_history[loser_id].append({
            'date': pd.Timestamp.now(),
            'surface': surface,
            'opponent': winner_id,
            'won': False,
            'elo': rating_l,
            'sets_played': sets_played,
            'games_played': games_played
        })
    
    st.session_state.elo_ratings = elo_ratings
    st.session_state.global_elo = global_ratings
    st.session_state.match_history = match_history
    
    return valid_count

def calc_form_simple(player_id, surface, match_history, lookback=20):
    """Calculate simple form metrics"""
    matches = match_history.get(player_id, [])
    
    # Get recent matches for this surface
    recent_matches = [m for m in matches[-lookback:] if m.get('surface') == surface]
    
    if not recent_matches:
        return get_default_form()
    
    wins = sum(1 for m in recent_matches if m.get('won', False))
    total = len(recent_matches)
    
    # Momentum (last 5 matches)
    last_5 = recent_matches[-5:] if len(recent_matches) >= 5 else recent_matches
    momentum = sum(1 for m in last_5 if m.get('won', False)) / max(len(last_5), 1)
    
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
    
    # Opponent quality (average opponent ELO)
    opp_elos = [m.get('elo', 1500) for m in recent_matches]
    avg_opp_elo = np.mean(opp_elos) if opp_elos else 1500
    
    # Average sets and games played
    avg_sets = np.mean([m.get('sets_played', 3) for m in recent_matches])
    avg_games = np.mean([m.get('games_played', 18) for m in recent_matches])
    
    # Estimate fatigue (matches in last 30 days)
    now = pd.Timestamp.now()
    fatigue = sum(1 for m in matches if (now - m.get('date', now)).days <= 30)
    
    return {
        'wins': wins,
        'matches': total,
        'win_pct': wins / max(total, 1),
        'momentum': momentum,
        'opp_elo': avg_opp_elo,
        'streak': streak,
        'consistency': 0.5,
        'fatigue': min(fatigue / 10, 1),  # Normalized
        'sets_played_avg': avg_sets,
        'games_played_avg': avg_games
    }

def create_features_simple(df):
    """Create simple features"""
    features_list = []
    labels = []
    
    player_ids = st.session_state.player_ids
    match_history = st.session_state.match_history
    
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
        
        # Get form
        p1_form = calc_form_simple(p1_id, surface, match_history)
        p2_form = calc_form_simple(p2_id, surface, match_history)
        
        # Create features from Player 1's perspective
        features = {
            'elo_diff': p1_elo - p2_elo,
            'win_pct_diff': p1_form['win_pct'] - p2_form['win_pct'],
            'momentum_diff': p1_form['momentum'] - p2_form['momentum'],
            'streak_diff': p1_form['streak'] - p2_form['streak'],
            'fatigue_diff': p1_form['fatigue'] - p2_form['fatigue'],
            'is_hard': 1 if surface == 'Hard' else 0,
            'is_clay': 1 if surface == 'Clay' else 0,
            'is_grass': 1 if surface == 'Grass' else 0,
        }
        
        features_list.append(features)
        labels.append(1 if winner == p1 else 0)
    
    return pd.DataFrame(features_list), np.array(labels)

def train_simple_model(features_df, labels):
    """Train a simple model"""
    if len(np.unique(labels)) < 2:
        raise ValueError("Need both win and loss examples")
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        features_df, labels, test_size=0.2, random_state=42, stratify=labels
    )
    
    # Scale features
    scaler = RobustScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    st.session_state.scaler = scaler
    
    # Train Random Forest
    model = RandomForestClassifier(
        n_estimators=100,
        max_depth=10,
        random_state=42,
        n_jobs=-1
    )
    
    # Calibrate the model
    calibrated_model = CalibratedClassifierCV(model, method='sigmoid', cv=3)
    calibrated_model.fit(X_train_scaled, y_train)
    
    # Evaluate
    y_pred = calibrated_model.predict(X_test_scaled)
    y_pred_proba = calibrated_model.predict_proba(X_test_scaled)[:, 1]
    
    metrics = {
        'accuracy': float(accuracy_score(y_test, y_pred)),
        'precision': float(precision_score(y_test, y_pred, zero_division=0)),
        'recall': float(recall_score(y_test, y_pred, zero_division=0)),
        'f1': float(f1_score(y_test, y_pred, zero_division=0)),
        'roc_auc': float(roc_auc_score(y_test, y_pred_proba)),
    }
    
    return calibrated_model, metrics

def predict_match_simple(p1_id, p2_id, surface):
    """Simple prediction"""
    if not st.session_state.ensemble_model:
        return None
    
    # Get ratings
    p1_elo = st.session_state.elo_ratings.get(p1_id, {}).get(surface, 1500)
    p2_elo = st.session_state.elo_ratings.get(p2_id, {}).get(surface, 1500)
    
    # Get form
    p1_form = calc_form_simple(p1_id, surface, st.session_state.match_history)
    p2_form = calc_form_simple(p2_id, surface, st.session_state.match_history)
    
    # Create features
    features = {
        'elo_diff': p1_elo - p2_elo,
        'win_pct_diff': p1_form['win_pct'] - p2_form['win_pct'],
        'momentum_diff': p1_form['momentum'] - p2_form['momentum'],
        'streak_diff': p1_form['streak'] - p2_form['streak'],
        'fatigue_diff': p1_form['fatigue'] - p2_form['fatigue'],
        'is_hard': 1 if surface == 'Hard' else 0,
        'is_clay': 1 if surface == 'Clay' else 0,
        'is_grass': 1 if surface == 'Grass' else 0,
    }
    
    features_df = pd.DataFrame([features])
    
    # Scale and predict
    try:
        features_scaled = st.session_state.scaler.transform(features_df)
        prediction_proba = st.session_state.ensemble_model.predict_proba(features_scaled)[0][1]
        
        return prediction_proba, {
            'p1_elo': p1_elo,
            'p2_elo': p2_elo,
            'p1_form': p1_form,
            'p2_form': p2_form
        }
    except Exception as e:
        st.error(f"Prediction error: {str(e)}")
        return None

def analyze_match(p1, p2, surface, prob, details):
    """Generate detailed match analysis"""
    # Determine favorite
    favorite = p1 if prob > 0.5 else p2
    underdog = p2 if prob > 0.5 else p1
    favorite_prob = prob if prob > 0.5 else 1 - prob
    
    # Calculate expected total games
    avg_games_p1 = details['p1_form']['games_played_avg']
    avg_games_p2 = details['p2_form']['games_played_avg']
    expected_games = (avg_games_p1 + avg_games_p2) / 2
    
    # Adjust based on probability difference (closer matches tend to have more games)
    closeness_factor = 1 - abs(prob - 0.5) * 2  # 1 for even match, 0 for very one-sided
    expected_games_adjusted = expected_games * (1 + closeness_factor * 0.2)
    
    # Calculate expected sets
    avg_sets_p1 = details['p1_form']['sets_played_avg']
    avg_sets_p2 = details['p2_form']['sets_played_avg']
    expected_sets = (avg_sets_p1 + avg_sets_p2) / 2
    
    # Key factors analysis
    key_factors = []
    
    # ELO difference
    elo_diff = abs(details['p1_elo'] - details['p2_elo'])
    if elo_diff > 100:
        key_factors.append(f"Significant ELO difference ({elo_diff:.0f} points)")
    elif elo_diff > 50:
        key_factors.append(f"Moderate ELO difference ({elo_diff:.0f} points)")
    else:
        key_factors.append(f"Close ELO ratings ({elo_diff:.0f} points difference)")
    
    # Form difference
    form_diff = abs(details['p1_form']['win_pct'] - details['p2_form']['win_pct'])
    if form_diff > 0.3:
        key_factors.append("Large form gap between players")
    elif form_diff > 0.15:
        key_factors.append("Noticeable form difference")
    
    # Streak
    if details['p1_form']['streak'] > 2 or details['p2_form']['streak'] > 2:
        key_factors.append("One player is on a winning streak")
    elif details['p1_form']['streak'] < -2 or details['p2_form']['streak'] < -2:
        key_factors.append("One player is on a losing streak")
    
    # Fatigue
    fatigue_diff = abs(details['p1_form']['fatigue'] - details['p2_form']['fatigue'])
    if fatigue_diff > 0.3:
        key_factors.append("Significant fatigue difference between players")
    
    # Surface preference
    surface_factor = ""
    if surface == 'Clay':
        surface_factor = "Clay specialist advantage"
    elif surface == 'Grass':
        surface_factor = "Grass court specialist advantage"
    elif surface == 'Hard':
        surface_factor = "Hard court specialist advantage"
    
    if surface_factor:
        key_factors.append(surface_factor)
    
    # Generate confidence level
    if favorite_prob > 0.7:
        confidence = "High confidence"
        match_type = "One-sided match"
    elif favorite_prob > 0.6:
        confidence = "Moderate confidence"
        match_type = "Clear favorite"
    elif favorite_prob > 0.55:
        confidence = "Low confidence"
        match_type = "Slight favorite"
    else:
        confidence = "Very low confidence"
        match_type = "Toss-up match"
    
    # Prediction summary
    summary = f"""
    ### 🎯 Match Prediction Summary
    
    **{favorite}** is favored to win with **{favorite_prob*100:.1f}%** probability
    
    #### 📊 Expected Match Characteristics:
    - **Match Type**: {match_type}
    - **Confidence Level**: {confidence}
    - **Expected Total Sets**: {expected_sets:.1f}
    - **Expected Total Games**: {expected_games_adjusted:.1f}
    
    #### 🔑 Key Factors:
    """
    
    for factor in key_factors:
        summary += f"- {factor}\n"
    
    # Betting recommendation
    summary += "\n#### 💡 Recommendation:"
    if favorite_prob > 0.65:
        summary += f"\nConsider betting on **{favorite}** - strong favorite"
    elif favorite_prob > 0.55:
        summary += f"\nCautious bet on **{favorite}** - slight edge"
    else:
        summary += "\nAvoid betting - match too close to call"
    
    # Underdog chance
    underdog_chance = 1 - favorite_prob
    if underdog_chance > 0.4:
        summary += f"\n\n⚠️ **{underdog}** has a {underdog_chance*100:.1f}% chance of an upset"
    
    return summary, {
        'favorite': favorite,
        'favorite_prob': favorite_prob,
        'underdog': underdog,
        'underdog_prob': underdog_chance,
        'expected_sets': expected_sets,
        'expected_games': expected_games_adjusted,
        'match_type': match_type,
        'confidence': confidence,
        'key_factors': key_factors
    }

def main():
    st.title("🎾 Tennis Prediction System")
    st.markdown("Simple ELO-based prediction model")
    
    tabs = st.tabs(["📊 Train Model", "🎯 Predict Match", "📈 Player Stats"])
    
    with tabs[0]:
        st.header("Train Model")
        
        col1, col2 = st.columns([2, 1])
        with col1:
            file = st.file_uploader("Upload CSV file", type=['csv'])
        with col2:
            k_factor = st.slider("K-factor", 10, 50, 32)
        
        if file is not None:
            try:
                df = pd.read_csv(file)
                st.session_state.match_data = df
                
                # Show data preview
                with st.expander("Data Preview"):
                    st.dataframe(df.head(), width='stretch')
                    st.write(f"Total rows: {len(df)}")
                    st.write("Columns found:", list(df.columns))
                    
                    # Show sample of data to help identify columns
                    st.write("### Sample of first row:")
                    for col in df.columns:
                        st.write(f"**{col}**: {df.iloc[0][col] if col in df.columns else 'N/A'}")
                
                # Check required columns - your CSV has different names
                st.write("### 🔍 Checking CSV format...")
                
                # Look for player and winner columns
                player_cols_found = []
                winner_col_found = None
                surface_col_found = None
                
                for col in df.columns:
                    col_lower = str(col).lower()
                    if 'player' in col_lower:
                        player_cols_found.append(col)
                    if 'winner' in col_lower:
                        winner_col_found = col
                    if 'surface' in col_lower:
                        surface_col_found = col
                
                if len(player_cols_found) >= 2 and winner_col_found:
                    st.success(f"✅ Found player columns: {player_cols_found}")
                    st.success(f"✅ Found winner column: {winner_col_found}")
                    if surface_col_found:
                        st.success(f"✅ Found surface column: {surface_col_found}")
                    else:
                        st.warning("⚠️ No surface column found - will use 'Hard' as default")
                else:
                    st.error("❌ Could not find required columns. Your CSV needs columns containing 'Player' and 'Winner' in their names.")
                    return
                
                # Process the data
                df_clean = df.copy()
                
                # Rename columns to standard names
                if len(player_cols_found) >= 2:
                    df_clean = df_clean.rename(columns={
                        player_cols_found[0]: 'Player_1',
                        player_cols_found[1]: 'Player_2',
                        winner_col_found: 'Winner'
                    })
                    if surface_col_found:
                        df_clean = df_clean.rename(columns={surface_col_found: 'Surface'})
                    else:
                        df_clean['Surface'] = 'Hard'
                
                # Clean the data
                for col in ['Player_1', 'Player_2', 'Winner', 'Surface']:
                    if col in df_clean.columns:
                        df_clean[col] = df_clean[col].astype(str).str.strip()
                
                # Check for valid matches
                valid_mask = (df_clean['Winner'] == df_clean['Player_1']) | (df_clean['Winner'] == df_clean['Player_2'])
                valid_matches = df_clean[valid_mask]
                
                st.write(f"**Valid matches:** {len(valid_matches)} / {len(df)}")
                
                if len(valid_matches) < 50:
                    st.warning(f"Only {len(valid_matches)} valid matches found. Need at least 50 for training.")
                    return
                
                if st.button("Train Model", type="primary"):
                    with st.spinner("Training model..."):
                        # Compute ELO
                        valid_count = compute_elo_simple(valid_matches, k_factor=k_factor)
                        st.success(f"ELO computed for {len(st.session_state.player_ids)} players from {valid_count} valid matches")
                        
                        # Create features
                        features_df, labels = create_features_simple(valid_matches)
                        st.success(f"Created {len(features_df)} training samples")
                        
                        if len(features_df) > 0:
                            # Show class distribution
                            unique_labels, counts = np.unique(labels, return_counts=True)
                            st.write("**Class distribution:**")
                            for label, count in zip(unique_labels, counts):
                                label_text = "Win" if label == 1 else "Loss"
                                percentage = (count/len(labels))*100
                                st.write(f"  {label_text}: {count} samples ({percentage:.1f}%)")
                            
                            # Train model
                            model, metrics = train_simple_model(features_df, labels)
                            st.session_state.ensemble_model = model
                            st.session_state.model_trained = True
                            
                            st.success("Model trained successfully!")
                            
                            # Show metrics - FIXED THE FORMATTING ERROR HERE
                            st.subheader("Model Performance")
                            cols = st.columns(4)
                            cols[0].metric("Accuracy", f"{metrics['accuracy']:.1%}")  # Fixed: removed extra 'f'
                            cols[1].metric("Precision", f"{metrics['precision']:.1%}")  # Fixed: removed extra 'f'
                            cols[2].metric("Recall", f"{metrics['recall']:.1%}")  # Fixed: removed extra 'f'
                            cols[3].metric("F1-Score", f"{metrics['f1']:.1%}")  # Fixed: removed extra 'f'
                            
            except Exception as e:
                st.error(f"Error: {str(e)}")
                st.info("Please make sure your CSV has the correct format with columns for players and winner")
    
    with tabs[1]:
        st.header("Predict Match")
        
        if not st.session_state.model_trained:
            st.warning("Please train the model first!")
        else:
            players = list(st.session_state.player_names.values())
            
            if len(players) == 0:
                st.info("No players loaded. Please train the model first.")
                return
            
            col1, col2, col3 = st.columns(3)
            with col1:
                p1 = st.selectbox("Player 1", players)
            with col2:
                # Filter out p1 from the list
                other_players = [p for p in players if p != p1]
                p2 = st.selectbox("Player 2", other_players)
            with col3:
                surface = st.selectbox("Surface", SURFACE_TYPES)
            
            if st.button("Predict Match Outcome", type="primary"):
                p1_id = st.session_state.player_ids.get(p1)
                p2_id = st.session_state.player_ids.get(p2)
                
                if p1_id and p2_id:
                    result = predict_match_simple(p1_id, p2_id, surface)
                    
                    if result:
                        prob, details = result
                        
                        # Generate analysis
                        summary, analysis_details = analyze_match(p1, p2, surface, prob, details)
                        
                        # Display main prediction
                        st.subheader("🎯 Match Prediction")
                        
                        col1, col2, col3 = st.columns(3)
                        with col1:
                            st.metric(f"{p1} Win %", f"{prob*100:.1f}%")
                        with col2:
                            st.metric(f"{p2} Win %", f"{(1-prob)*100:.1f}%")
                        with col3:
                            st.metric("Expected Games", f"{analysis_details['expected_games']:.1f}")
                        
                        # Display the summary
                        st.markdown(summary)
                        
                        # Detailed breakdown in expander
                        with st.expander("📊 Detailed Match Analysis"):
                            
                            # Head-to-head comparison
                            st.write("### 🤝 Player Comparison")
                            
                            col1, col2, col3 = st.columns(3)
                            with col1:
                                st.write(f"**{analysis_details['favorite']}**")
                                st.metric("Favorite Probability", f"{analysis_details['favorite_prob']*100:.1f}%")
                                if analysis_details['favorite'] == p1:
                                    st.metric("ELO Rating", f"{details['p1_elo']:.0f}")
                                    st.metric("Current Form", f"{details['p1_form']['win_pct']*100:.0f}%")
                                else:
                                    st.metric("ELO Rating", f"{details['p2_elo']:.0f}")
                                    st.metric("Current Form", f"{details['p2_form']['win_pct']*100:.0f}%")
                            
                            with col2:
                                st.write("**Match Details**")
                                st.metric("Confidence Level", analysis_details['confidence'])
                                st.metric("Match Type", analysis_details['match_type'])
                                st.metric("Expected Sets", f"{analysis_details['expected_sets']:.1f}")
                            
                            with col3:
                                st.write(f"**{analysis_details['underdog']}**")
                                st.metric("Upset Probability", f"{analysis_details['underdog_prob']*100:.1f}%")
                                if analysis_details['underdog'] == p1:
                                    st.metric("ELO Rating", f"{details['p1_elo']:.0f}")
                                    st.metric("Current Form", f"{details['p1_form']['win_pct']*100:.0f}%")
                                else:
                                    st.metric("ELO Rating", f"{details['p2_elo']:.0f}")
                                    st.metric("Current Form", f"{details['p2_form']['win_pct']*100:.0f}%")
                            
                            # Form comparison table
                            st.write("### 📈 Form Statistics")
                            
                            form_data = {
                                'Metric': ['Win Percentage', 'Momentum (Last 5)', 'Current Streak', 
                                         'Fatigue Level', 'Avg Games per Match', 'Avg Sets per Match'],
                                p1: [
                                    f"{details['p1_form']['win_pct']*100:.1f}%",
                                    f"{details['p1_form']['momentum']*100:.1f}%",
                                    f"{details['p1_form']['streak']:+d}",
                                    f"{details['p1_form']['fatigue']*100:.0f}%",
                                    f"{details['p1_form']['games_played_avg']:.1f}",
                                    f"{details['p1_form']['sets_played_avg']:.1f}"
                                ],
                                p2: [
                                    f"{details['p2_form']['win_pct']*100:.1f}%",
                                    f"{details['p2_form']['momentum']*100:.1f}%",
                                    f"{details['p2_form']['streak']:+d}",
                                    f"{details['p2_form']['fatigue']*100:.0f}%",
                                    f"{details['p2_form']['games_played_avg']:.1f}",
                                    f"{details['p2_form']['sets_played_avg']:.1f}"
                                ]
                            }
                            
                            form_df = pd.DataFrame(form_data)
                            st.dataframe(form_df, width='stretch')
                            
                            # ELO comparison
                            st.write("### 📊 ELO Rating Analysis")
                            elo_diff = details['p1_elo'] - details['p2_elo']
                            
                            col1, col2, col3 = st.columns(3)
                            col1.metric(f"{p1} ELO", f"{details['p1_elo']:.0f}")
                            col2.metric("Difference", f"{elo_diff:+.0f}", 
                                      delta="Advantage" if elo_diff > 0 else "Disadvantage")
                            col3.metric(f"{p2} ELO", f"{details['p2_elo']:.0f}")
                            
                            # Surface analysis
                            st.write("### 🏟️ Surface Analysis")
                            surface_notes = {
                                'Hard': "Fast surface favors big servers and aggressive baseline players",
                                'Clay': "Slow surface favors defensive players and grinders",
                                'Grass': "Very fast surface favors serve-and-volley players and big servers",
                                'Carpet': "Fast indoor surface (rarely used nowadays)"
                            }
                            
                            st.info(f"**{surface} Court**: {surface_notes.get(surface, '')}")
                            
                            # Fatigue analysis
                            st.write("### ⚡ Fatigue Analysis")
                            fatigue_p1 = details['p1_form']['fatigue']
                            fatigue_p2 = details['p2_form']['fatigue']
                            
                            if fatigue_p1 > 0.7 or fatigue_p2 > 0.7:
                                st.warning("⚠️ High fatigue levels detected - could impact performance")
                            elif fatigue_p1 > fatigue_p2 + 0.3:
                                st.info(f"📉 {p1} shows higher fatigue levels than {p2}")
                            elif fatigue_p2 > fatigue_p1 + 0.3:
                                st.info(f"📉 {p2} shows higher fatigue levels than {p1}")
                            else:
                                st.success("✅ Both players appear to have similar fatigue levels")
    
    with tabs[2]:
        st.header("Player Statistics")
        
        if not st.session_state.model_trained:
            st.warning("Please train the model first!")
        else:
            players = list(st.session_state.player_names.values())
            
            if len(players) == 0:
                st.info("No players loaded.")
                return
            
            player = st.selectbox("Select Player", players, key='player_select')
            
            if player:
                pid = st.session_state.player_ids.get(player)
                matches = st.session_state.match_history.get(pid, [])
                
                if matches:
                    wins = sum(1 for m in matches if m.get('won', False))
                    total = len(matches)
                    
                    st.write(f"### 📊 Statistics for {player}")
                    col1, col2, col3, col4 = st.columns(4)
                    col1.metric("Total Matches", total)
                    col2.metric("Wins", wins)
                    col3.metric("Losses", total - wins)
                    col4.metric("Win %", f"{wins/total*100:.1f}%" if total > 0 else "N/A")
                    
                    # Surface breakdown
                    surface_stats = {}
                    for match in matches:
                        surf = match.get('surface', 'Hard')
                        if surf not in surface_stats:
                            surface_stats[surf] = {'wins': 0, 'total': 0, 'games_avg': [], 'sets_avg': []}
                        surface_stats[surf]['total'] += 1
                        if match.get('won', False):
                            surface_stats[surf]['wins'] += 1
                        surface_stats[surf]['games_avg'].append(match.get('games_played', 18))
                        surface_stats[surf]['sets_avg'].append(match.get('sets_played', 3))
                    
                    if surface_stats:
                        st.write("#### 🏟️ Performance by Surface:")
                        surface_data = []
                        for surf, stats in surface_stats.items():
                            win_pct = stats['wins'] / stats['total'] * 100
                            avg_games = np.mean(stats['games_avg']) if stats['games_avg'] else 18
                            avg_sets = np.mean(stats['sets_avg']) if stats['sets_avg'] else 3
                            surface_data.append({
                                'Surface': surf,
                                'Matches': stats['total'],
                                'Wins': stats['wins'],
                                'Losses': stats['total'] - stats['wins'],
                                'Win %': f"{win_pct:.1f}%",
                                'Avg Games': f"{avg_games:.1f}",
                                'Avg Sets': f"{avg_sets:.1f}"
                            })
                        
                        surface_df = pd.DataFrame(surface_data)
                        st.dataframe(surface_df, width='stretch')
                        
                        # ELO by surface
                        st.write("#### 📈 ELO Ratings by Surface:")
                        elo_data = []
                        if pid in st.session_state.elo_ratings:
                            for surf, elo in st.session_state.elo_ratings[pid].items():
                                elo_data.append({'Surface': surf, 'ELO Rating': f"{elo:.0f}"})
                        
                        if elo_data:
                            elo_df = pd.DataFrame(elo_data)
                            st.dataframe(elo_df, width='stretch')
                else:
                    st.info(f"No match history for {player}")

if __name__ == "__main__":
    if ML_AVAILABLE:
        main()
    else:
        st.error("""
        Required packages not installed. Please install:
        ```
        pip install scikit-learn pandas numpy
        ```
        """)

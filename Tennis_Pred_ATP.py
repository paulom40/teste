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

SURFACE_TYPES = ['Hard', 'Clay', 'Grass', 'Carpet']

def get_default_form():
    """Default form dict"""
    return {
        'wins': 0, 'matches': 0, 'win_pct': 0.5, 'momentum': 0.5,
        'opp_elo': 1500, 'streak': 0, 'consistency': 0.5, 'fatigue': 0
    }

def compute_elo_simple(df, k_factor=32, initial_elo=1500):
    """Simple ELO computation - less memory intensive"""
    df = df.copy()
    
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
        surface = row['Surface'] if 'Surface' in row and row['Surface'] in SURFACE_TYPES else 'Hard'
        
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
            'elo': rating_w
        })
        match_history[loser_id].append({
            'date': pd.Timestamp.now(),
            'surface': surface,
            'opponent': winner_id,
            'won': False,
            'elo': rating_l
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
    
    return {
        'wins': wins,
        'matches': total,
        'win_pct': wins / max(total, 1),
        'momentum': momentum,
        'opp_elo': avg_opp_elo,
        'streak': streak,
        'consistency': 0.5,  # Simplified
        'fatigue': 0
    }

def create_features_simple(df):
    """Create simple features"""
    features_list = []
    labels = []
    
    player_ids = st.session_state.player_ids
    match_history = st.session_state.match_history
    
    for idx, row in df.iterrows():
        p1 = row['Player_1']
        p2 = row['Player_2']
        winner = row['Winner']
        surface = row['Surface'] if 'Surface' in row and row['Surface'] in SURFACE_TYPES else 'Hard'
        
        if winner not in [p1, p2]:
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
                
                # Show data preview
                with st.expander("Data Preview"):
                    st.dataframe(df.head(), width='stretch')
                    st.write(f"Total rows: {len(df)}")
                    st.write("Columns:", list(df.columns))
                
                # Check required columns
                required_cols = ['Player_1', 'Player_2', 'Winner']
                missing_cols = [col for col in required_cols if col not in df.columns]
                
                if missing_cols:
                    st.error(f"Missing columns: {', '.join(missing_cols)}")
                    return
                
                # Check for valid matches
                df_clean = df.copy()
                for col in required_cols:
                    df_clean[col] = df_clean[col].astype(str).str.strip()
                
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
                        st.success(f"ELO computed for {len(st.session_state.player_ids)} players")
                        
                        # Create features
                        features_df, labels = create_features_simple(valid_matches)
                        st.success(f"Created {len(features_df)} training samples")
                        
                        if len(features_df) > 0:
                            # Train model
                            model, metrics = train_simple_model(features_df, labels)
                            st.session_state.ensemble_model = model
                            st.session_state.model_trained = True
                            
                            st.success("Model trained successfully!")
                            
                            # Show metrics
                            st.subheader("Model Performance")
                            cols = st.columns(4)
                            cols[0].metric("Accuracy", f"{metrics['accuracy']:.1%}")
                            cols[1].metric("Precision", f"{metrics['precision']:.1%}")
                            cols[2].metric("Recall", f"{metrics['recall']:.1%}")
                            cols[3].metric("F1-Score", f"{metrics['f1']:.1%}")
                            
            except Exception as e:
                st.error(f"Error: {str(e)}")
                st.info("Please make sure your CSV has the correct format with columns: Player_1, Player_2, Winner, Surface")
    
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
            
            if st.button("Predict", type="primary"):
                p1_id = st.session_state.player_ids.get(p1)
                p2_id = st.session_state.player_ids.get(p2)
                
                if p1_id and p2_id:
                    result = predict_match_simple(p1_id, p2_id, surface)
                    
                    if result:
                        prob, details = result
                        
                        st.subheader("Prediction Result")
                        
                        col1, col2 = st.columns(2)
                        with col1:
                            st.metric(f"{p1} Win %", f"{prob*100:.1f}%")
                        with col2:
                            st.metric(f"{p2} Win %", f"{(1-prob)*100:.1f}%")
                        
                        with st.expander("Details"):
                            st.write("**ELO Ratings:**")
                            col1, col2 = st.columns(2)
                            col1.metric(f"{p1} ELO", f"{details['p1_elo']:.0f}")
                            col2.metric(f"{p2} ELO", f"{details['p2_elo']:.0f}")
                            
                            st.write("**Form Stats:**")
                            col1, col2 = st.columns(2)
                            with col1:
                                st.write(f"**{p1}**")
                                f1 = details['p1_form']
                                st.write(f"Win %: {f1['win_pct']*100:.0f}%")
                                st.write(f"Momentum: {f1['momentum']*100:.0f}%")
                                st.write(f"Streak: {f1['streak']:+d}")
                            with col2:
                                st.write(f"**{p2}**")
                                f2 = details['p2_form']
                                st.write(f"Win %: {f2['win_pct']*100:.0f}%")
                                st.write(f"Momentum: {f2['momentum']*100:.0f}%")
                                st.write(f"Streak: {f2['streak']:+d}")
    
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
                    
                    st.write(f"**Statistics for {player}**")
                    col1, col2, col3 = st.columns(3)
                    col1.metric("Total Matches", total)
                    col2.metric("Wins", wins)
                    col3.metric("Win %", f"{wins/total*100:.1f}%" if total > 0 else "N/A")
                    
                    # Surface breakdown
                    surface_stats = {}
                    for match in matches:
                        surf = match.get('surface', 'Hard')
                        if surf not in surface_stats:
                            surface_stats[surf] = {'wins': 0, 'total': 0}
                        surface_stats[surf]['total'] += 1
                        if match.get('won', False):
                            surface_stats[surf]['wins'] += 1
                    
                    if surface_stats:
                        st.write("**Performance by Surface:**")
                        for surf, stats in surface_stats.items():
                            win_pct = stats['wins'] / stats['total'] * 100
                            st.write(f"{surf}: {stats['wins']}W - {stats['total']-stats['wins']}L ({win_pct:.1f}%)")
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

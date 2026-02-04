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
    ML_AVAILABLE = True
except ImportError:
    ML_AVAILABLE = False

st.set_page_config(page_title="Tennis Prediction", page_icon="🎾", layout="wide")

# Session State
for key in ['elo_ratings', 'player_names', 'global_elo', 'match_data', 'player_form_history', 
            'player_ids', 'match_history', 'scaler', 'ensemble_model', 'model_metrics', 'surface_performance']:
    if key not in st.session_state:
        st.session_state[key] = {} if key not in ['match_data', 'scaler', 'ensemble_model', 'model_metrics'] else None

SURFACE_TYPES = ['Hard', 'Clay', 'Grass', 'Carpet']

def create_player_ids(df):
    """Create player IDs from match data"""
    players = set()
    
    # Collect all unique players
    if 'Player_1' in df.columns:
        players.update(df['Player_1'].dropna().astype(str).str.strip().unique())
    if 'Player_2' in df.columns:
        players.update(df['Player_2'].dropna().astype(str).str.strip().unique())
    if 'Winner' in df.columns:
        players.update(df['Winner'].dropna().astype(str).str.strip().unique())
    
    # Create IDs
    player_ids = {}
    for idx, name in enumerate(sorted(players)):
        player_ids[name] = f"P{idx:04d}"
        st.session_state.player_names[f"P{idx:04d}"] = name
    
    return player_ids

def get_default_form():
    return {
        'wins': 0, 'matches': 0, 'win_pct': 0.5, 'momentum': 0.5,
        'opp_elo': 1500, 'streak': 0, 'consistency': 0.5, 'fatigue': 0,
        'top10_pct': 0.5, 'surface_pct': 0.5, 'recent_trend': 0
    }

def compute_elo(df, k_factor=32, initial_elo=1500):
    """Compute ELO ratings"""
    st.session_state.player_ids = create_player_ids(df)
    
    if 'Date' in df.columns:
        df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
        df = df.sort_values('Date').reset_index(drop=True)
    else:
        df['Date'] = datetime.now()
    
    # Normalize names
    df['Player_1'] = df['Player_1'].astype(str).str.strip()
    df['Player_2'] = df['Player_2'].astype(str).str.strip()
    df['Winner'] = df['Winner'].astype(str).str.strip()
    
    # Map to IDs
    player_ids = st.session_state.player_ids
    df['winner_id'] = df['Winner'].map(player_ids)
    
    # Determine loser
    def get_loser_id(row):
        if pd.isna(row['winner_id']):
            return None
        p1_id = player_ids.get(row['Player_1'])
        p2_id = player_ids.get(row['Player_2'])
        w_id = row['winner_id']
        
        if p1_id == w_id:
            return p2_id
        elif p2_id == w_id:
            return p1_id
        return None
    
    df['loser_id'] = df.apply(get_loser_id, axis=1)
    
    # Filter valid matches
    valid_matches = df[df['winner_id'].notna() & df['loser_id'].notna()].copy()
    
    if len(valid_matches) == 0:
        st.error("No valid matches found! Check your CSV format.")
        return None, None
    
    # Initialize ratings
    players = set(valid_matches['winner_id'].unique()).union(set(valid_matches['loser_id'].unique()))
    elo_ratings = {p: {s: initial_elo for s in SURFACE_TYPES} for p in players}
    global_ratings = {p: initial_elo for p in players}
    form_history = defaultdict(lambda: defaultdict(deque))
    match_history = defaultdict(list)
    surface_perf = defaultdict(lambda: defaultdict(list))
    
    # Process matches
    for idx, row in valid_matches.iterrows():
        winner = row['winner_id']
        loser = row['loser_id']
        surface = str(row.get('Surface', 'Hard')).strip()
        
        if surface not in SURFACE_TYPES:
            surface = 'Hard'
        
        rating_w = elo_ratings[winner][surface]
        rating_l = elo_ratings[loser][surface]
        
        # Match importance
        importance = 1.0
        try:
            round_info = str(row.get('Round', '')).lower()
            if 'final' in round_info:
                importance = 1.5
            elif 'semi' in round_info:
                importance = 1.3
        except:
            pass
        
        k = k_factor * importance / (1 + len(match_history[winner]) / 50)
        exp_w = 1 / (1 + math.pow(10, (rating_l - rating_w) / 400))
        
        elo_ratings[winner][surface] = rating_w + k * (1 - exp_w)
        elo_ratings[loser][surface] = rating_l + k * (0 - (1 - exp_w))
        global_ratings[winner] = global_ratings.get(winner, initial_elo) + k * (1 - exp_w)
        global_ratings[loser] = global_ratings.get(loser, initial_elo) + k * (0 - (1 - exp_w))
        
        match_record = {
            'date': row['Date'], 'surface': surface, 'won': True,
            'w_elo': rating_w, 'l_elo': rating_l, 'importance': importance
        }
        
        match_history[winner].append(match_record)
        match_history[loser].append({**match_record, 'won': False})
        
        surface_perf[winner][surface].append(True)
        surface_perf[loser][surface].append(False)
        
        form_history[winner][surface].append(match_record)
        form_history[loser][surface].append({**match_record, 'won': False})
        
        if len(form_history[winner][surface]) > 50:
            form_history[winner][surface].popleft()
        if len(form_history[loser][surface]) > 50:
            form_history[loser][surface].popleft()
    
    st.session_state.player_form_history = form_history
    st.session_state.match_history = match_history
    st.session_state.surface_performance = surface_perf
    
    return elo_ratings, global_ratings

def calc_form(player_id, surface, form_history, match_history, current_elo):
    """Calculate form metrics"""
    if player_id not in form_history or surface not in form_history[player_id]:
        return get_default_form()
    
    recent = list(form_history[player_id][surface])
    if not recent:
        return get_default_form()
    
    wins = sum(1 for m in recent if m['won'])
    total = len(recent)
    
    # Momentum
    if total >= 5:
        recent_5 = sum(1 for m in recent[-5:] if m['won']) / 5
    else:
        recent_5 = wins / total if total > 0 else 0.5
    
    # Streak
    streak = 0
    for m in reversed(recent):
        if m['won']:
            streak += 1
        else:
            break
    if streak == 0:
        for m in reversed(recent):
            if not m['won']:
                streak -= 1
            else:
                break
    
    # Opponent ELO
    opp_elos = [m['l_elo'] if m['won'] else m['w_elo'] for m in recent]
    opp_elo = np.mean(opp_elos) if opp_elos else 1500
    
    # Top 10
    top10 = sum(1 for m in recent if m['won'] and m['l_elo'] > 1750)
    top10_total = sum(1 for m in recent if m['l_elo'] > 1750)
    top10_pct = top10 / top10_total if top10_total > 0 else 0.5
    
    # Consistency
    perf = []
    for m in recent:
        opp = m['l_elo'] if m['won'] else m['w_elo']
        expected = 1 / (1 + math.pow(10, (opp - current_elo) / 400))
        perf.append((1 - expected) if m['won'] else (0 - expected))
    consistency = 1 - min(np.std(perf) if perf else 0, 1.0)
    
    # Surface pct
    surf_results = st.session_state.surface_performance.get(player_id, {}).get(surface, [])
    surface_pct = sum(1 for r in surf_results if r) / max(len(surf_results), 1) if surf_results else 0.5
    
    # Trend
    if len(recent) >= 10:
        recent_10_avg = sum(1 for m in recent[-10:] if m['won']) / 10
        old_10_avg = sum(1 for m in recent[-20:-10] if m['won']) / 10 if len(recent) >= 20 else 0.5
        trend = recent_10_avg - old_10_avg
    else:
        trend = 0
    
    return {
        'wins': wins, 'matches': total, 'win_pct': wins / total if total > 0 else 0.5,
        'momentum': recent_5, 'opp_elo': opp_elo, 'streak': streak,
        'consistency': consistency, 'fatigue': len([m for m in match_history.get(player_id, [])
                                                    if (datetime.now() - m['date']).days <= 30]),
        'top10_pct': top10_pct, 'surface_pct': surface_pct, 'recent_trend': trend
    }

def create_features(df, elo_ratings, global_ratings, form_history, match_history):
    """Create training features"""
    player_ids = st.session_state.player_ids
    
    df['Player_1'] = df['Player_1'].astype(str).str.strip()
    df['Player_2'] = df['Player_2'].astype(str).str.strip()
    df['Winner'] = df['Winner'].astype(str).str.strip()
    
    df['winner_id'] = df['Winner'].map(player_ids)
    
    def get_loser_id(row):
        if pd.isna(row['winner_id']):
            return None
        p1_id = player_ids.get(row['Player_1'])
        p2_id = player_ids.get(row['Player_2'])
        w_id = row['winner_id']
        
        if p1_id == w_id:
            return p2_id
        elif p2_id == w_id:
            return p1_id
        return None
    
    df['loser_id'] = df.apply(get_loser_id, axis=1)
    
    features_list = []
    labels = []
    
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
            
            w_form = calc_form(w_id, surface, form_history, match_history, w_elo)
            l_form = calc_form(l_id, surface, form_history, match_history, l_elo)
            
            elo_diff = w_elo - l_elo
            
            features = {
                'elo_diff': float(elo_diff),
                'elo_ratio': float(w_elo / l_elo) if l_elo > 0 else 1.0,
                'is_hard': 1 if surface == 'Hard' else 0,
                'is_clay': 1 if surface == 'Clay' else 0,
                'is_grass': 1 if surface == 'Grass' else 0,
                'w_win_pct': float(w_form['win_pct']),
                'w_momentum': float(w_form['momentum']),
                'w_streak': float(w_form['streak']),
                'w_consistency': float(w_form['consistency']),
                'w_fatigue': float(w_form['fatigue'] / 5),
                'w_top10': float(w_form['top10_pct']),
                'w_surface': float(w_form['surface_pct']),
                'w_trend': float(w_form['recent_trend']),
                'l_win_pct': float(l_form['win_pct']),
                'l_momentum': float(l_form['momentum']),
                'l_streak': float(l_form['form_streak']),
                'l_consistency': float(l_form['consistency']),
                'l_fatigue': float(l_form['fatigue'] / 5),
                'l_top10': float(l_form['top10_pct']),
                'l_surface': float(l_form['surface_pct']),
                'l_trend': float(l_form['recent_trend']),
                'form_diff': float(w_form['win_pct'] - l_form['win_pct']),
                'momentum_diff': float(w_form['momentum'] - l_form['momentum']),
            }
            
            features_list.append(features)
            labels.append(1)
            
            # Loser perspective
            l_features = features.copy()
            l_features['elo_diff'] = -features['elo_diff']
            l_features['elo_ratio'] = 1 / features['elo_ratio'] if features['elo_ratio'] > 0 else 1.0
            
            for w, l in [('w_win_pct', 'l_win_pct'), ('w_momentum', 'l_momentum'),
                        ('w_streak', 'l_streak'), ('w_consistency', 'l_consistency'),
                        ('w_fatigue', 'l_fatigue'), ('w_top10', 'l_top10'),
                        ('w_surface', 'l_surface'), ('w_trend', 'l_trend')]:
                l_features[w] = features[l]
                l_features[l] = features[w]
            
            l_features['form_diff'] = -features['form_diff']
            l_features['momentum_diff'] = -features['momentum_diff']
            
            features_list.append(l_features)
            labels.append(0)
            
        except Exception as e:
            continue
    
    return pd.DataFrame(features_list), np.array(labels)

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
        'rf': RandomForestClassifier(n_estimators=200, max_depth=10, random_state=42, n_jobs=-1),
        'gb': GradientBoostingClassifier(n_estimators=150, max_depth=5, learning_rate=0.05, random_state=42),
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

def predict_match(p1_id, p2_id, surface, elo_ratings, global_ratings, form_history, match_history):
    """Predict match"""
    if not st.session_state.ensemble_model:
        return None
    
    p1_elo = elo_ratings.get(p1_id, {}).get(surface, global_ratings.get(p1_id, 1500))
    p2_elo = elo_ratings.get(p2_id, {}).get(surface, global_ratings.get(p2_id, 1500))
    
    p1_form = calc_form(p1_id, surface, form_history, match_history, p1_elo)
    p2_form = calc_form(p2_id, surface, form_history, match_history, p2_elo)
    
    elo_diff = p1_elo - p2_elo
    
    features_dict = {
        'elo_diff': float(elo_diff),
        'elo_ratio': float(p1_elo / p2_elo) if p2_elo > 0 else 1.0,
        'is_hard': 1 if surface == 'Hard' else 0,
        'is_clay': 1 if surface == 'Clay' else 0,
        'is_grass': 1 if surface == 'Grass' else 0,
        'w_win_pct': float(p1_form['win_pct']),
        'w_momentum': float(p1_form['momentum']),
        'w_streak': float(p1_form['streak']),
        'w_consistency': float(p1_form['consistency']),
        'w_fatigue': float(p1_form['fatigue'] / 5),
        'w_top10': float(p1_form['top10_pct']),
        'w_surface': float(p1_form['surface_pct']),
        'w_trend': float(p1_form['recent_trend']),
        'l_win_pct': float(p2_form['win_pct']),
        'l_momentum': float(p2_form['momentum']),
        'l_streak': float(p2_form['form_streak']),
        'l_consistency': float(p2_form['consistency']),
        'l_fatigue': float(p2_form['fatigue'] / 5),
        'l_top10': float(p2_form['top10_pct']),
        'l_surface': float(p2_form['surface_pct']),
        'l_trend': float(p2_form['recent_trend']),
        'form_diff': float(p1_form['win_pct'] - p2_form['win_pct']),
        'momentum_diff': float(p1_form['momentum'] - p2_form['momentum']),
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
    st.title("🎾 Tennis Prediction System")
    st.markdown("**Advanced ELO + 22 Features + 4-Model Ensemble**")
    
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
            
            st.dataframe(df.head(10), width='stretch')
            st.write(f"**Matches:** {len(df)}")
            
            required = ['Player_1', 'Player_2', 'Winner', 'Surface']
            if all(col in df.columns for col in required):
                st.success("✅ Valid format")
                
                if st.button("🚀 Train Model", type="primary"):
                    progress = st.progress(0)
                    
                    with st.spinner("Computing ELO..."):
                        elos, g_elos = compute_elo(df, k_factor=k)
                        if elos is None:
                            st.stop()
                        st.session_state.elo_ratings = elos
                        st.session_state.global_elo = g_elos
                        progress.progress(33)
                    
                    with st.spinner("Creating features..."):
                        features_df, labels = create_features(
                            df, elos, g_elos,
                            st.session_state.player_form_history,
                            st.session_state.match_history
                        )
                        progress.progress(66)
                    
                    st.info(f"✅ Created {len(features_df)} samples with {features_df.shape[1]} features")
                    
                    if len(features_df) > 0 and len(np.unique(labels)) > 1:
                        with st.spinner("Training ensemble..."):
                            model, metrics = train_model(features_df, labels)
                            st.session_state.ensemble_model = model
                            st.session_state.model_metrics = metrics
                            progress.progress(100)
                        
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
                    result = predict_match(
                        p1_id, p2_id, surf,
                        st.session_state.elo_ratings,
                        st.session_state.global_elo,
                        st.session_state.player_form_history,
                        st.session_state.match_history
                    )
                    
                    if result:
                        prob, details = result
                        
                        col1, col2 = st.columns(2)
                        with col1:
                            st.metric(f"{p1} Win %", f"{prob*100:.1f}%")
                        with col2:
                            st.metric(f"{p2} Win %", f"{(1-prob)*100:.1f}%")
                        
                        with st.expander("📊 Details"):
                            col1, col2, col3 = st.columns(3)
                            col1.metric(f"{p1} ELO", f"{details['p1_elo']:.0f}")
                            col2.metric("Diff", f"{details['elo_diff']:.0f}")
                            col3.metric(f"{p2} ELO", f"{details['p2_elo']:.0f}")
                            
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
                st.metric("Win%", f"{wins/len(matches)*100:.1f}%" if matches else "N/A")
    
    with tabs[3]:
        st.header("Model Info")
        if st.session_state.model_metrics:
            st.json(st.session_state.model_metrics)
            st.info("**4 Models:** Random Forest, Gradient Boosting, AdaBoost, Logistic Regression")

if __name__ == "__main__":
    main()

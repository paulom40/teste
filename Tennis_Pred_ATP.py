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
    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix
    from sklearn.preprocessing import RobustScaler
    from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.calibration import CalibratedClassifierCV
    ML_AVAILABLE = True
except ImportError:
    ML_AVAILABLE = False

st.set_page_config(page_title="Tennis Prediction", page_icon="🎾", layout="wide")

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
if 'feature_importance' not in st.session_state:
    st.session_state.feature_importance = None
if 'all_feature_columns' not in st.session_state:
    st.session_state.all_feature_columns = []
if 'surface_performance' not in st.session_state:
    st.session_state.surface_performance = {}

RECENT_MATCHES_COUNT = 30
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
        'top10_win_pct': 0.5, 'surface_win_pct': 0.5
    }

# ===========================
# ELO CALCULATION
# ===========================

def compute_elo(df, k_factor=32, initial_elo=1500):
    """Compute ELO ratings"""
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
            if 'final' in round_info:
                importance = 1.5
            elif 'semifinal' in round_info or 'quarterfinal' in round_info:
                importance = 1.3
            else:
                importance = 1.0
        except:
            importance = 1.0
        
        winner_k = k_factor * importance / (1 + len(match_history.get(winner, [])) / 50)
        loser_k = k_factor * importance / (1 + len(match_history.get(loser, [])) / 50)
        
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
    
    return elo_ratings, global_ratings

# ===========================
# FORM FEATURES
# ===========================

def calc_form(player_id, surface, form_history, match_history):
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
    
    if total >= 5:
        recent5_wins = sum(1 for m in recent[-5:] if m['won'])
        momentum = recent5_wins / 5
    else:
        momentum = wins / total if total > 0 else 0.5
    
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
    
    opp_elos = [m['l_elo'] if m['won'] else m['w_elo'] for m in recent]
    opp_elo = np.average(opp_elos, weights=weights) if opp_elos else 1500
    
    top10_wins = sum(1 for m in recent if m['won'] and (m['l_elo'] > 1750 if m['won'] else m['w_elo'] > 1750))
    top10_total = sum(1 for m in recent if (m['l_elo'] > 1750 if m['won'] else m['w_elo'] > 1750))
    top10_pct = top10_wins / top10_total if top10_total > 0 else 0.5
    
    perf_scores = []
    for m in recent:
        opp = m['l_elo'] if m['won'] else m['w_elo']
        expected = 1 / (1 + math.pow(10, (opp - 1500) / 400))
        perf_scores.append((1 - expected) if m['won'] else (0 - expected))
    consistency = 1 - min(np.std(perf_scores), 1.0) if perf_scores else 0.5
    
    recent_30 = sum(1 for m in match_history.get(player_id, [])
                   if isinstance(m.get('date'), datetime) and
                   (datetime.now() - m['date']).days <= 30)
    
    return {
        'wins': wins,
        'matches': total,
        'win_pct': wins / total if total > 0 else 0.5,
        'momentum': momentum,
        'opp_elo': opp_elo,
        'streak': streak,
        'consistency': consistency,
        'fatigue': recent_30,
        'top10_win_pct': top10_pct,
        'surface_win_pct': sum(1 for r in st.session_state.surface_performance.get(player_id, {}).get(surface, []) if r) / max(len(st.session_state.surface_performance.get(player_id, {}).get(surface, [])), 1)
    }

# ===========================
# FEATURES
# ===========================

def create_features(df, elo_ratings, global_ratings, form_history, match_history):
    """Create feature dataframe"""
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
            
            w_form = calc_form(w_id, surface, form_history, match_history)
            l_form = calc_form(l_id, surface, form_history, match_history)
            
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
                'w_fatigue': float(w_form['fatigue']),
                'w_top10': float(w_form['top10_win_pct']),
                'w_surface': float(w_form['surface_win_pct']),
                'l_win_pct': float(l_form['win_pct']),
                'l_momentum': float(l_form['momentum']),
                'l_streak': float(l_form['streak']),
                'l_consistency': float(l_form['consistency']),
                'l_fatigue': float(l_form['fatigue']),
                'l_top10': float(l_form['top10_win_pct']),
                'l_surface': float(l_form['surface_win_pct']),
                'form_diff': float(w_form['win_pct'] - l_form['win_pct']),
                'momentum_diff': float(w_form['momentum'] - l_form['momentum']),
            }
            
            features_list.append(features)
            labels.append(1)
            
            l_features = features.copy()
            l_features['elo_diff'] = -features['elo_diff']
            l_features['elo_ratio'] = 1 / features['elo_ratio'] if features['elo_ratio'] > 0 else 1.0
            
            for w, l in [('w_win_pct', 'l_win_pct'), ('w_momentum', 'l_momentum'),
                        ('w_streak', 'l_streak'), ('w_consistency', 'l_consistency'),
                        ('w_fatigue', 'l_fatigue'), ('w_top10', 'l_top10'),
                        ('w_surface', 'l_surface')]:
                l_features[w] = features[l]
                l_features[l] = features[w]
            
            l_features['form_diff'] = -features['form_diff']
            l_features['momentum_diff'] = -features['momentum_diff']
            
            features_list.append(l_features)
            labels.append(0)
            
        except:
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
    st.session_state.all_feature_columns = features_df.columns.tolist()
    
    models = {
        'rf': RandomForestClassifier(n_estimators=200, max_depth=10, random_state=42, n_jobs=-1),
        'gb': GradientBoostingClassifier(n_estimators=150, max_depth=5, learning_rate=0.05, random_state=42),
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
# PREDICTION WITH DETAILS
# ===========================

def predict_match_with_details(p1_id, p2_id, surface, elo_ratings, global_ratings, form_history, match_history):
    """Predict match and return detailed breakdown"""
    
    if not st.session_state.ensemble_model:
        return None, None
    
    # Get ELO ratings
    p1_elo = elo_ratings.get(p1_id, {}).get(surface, global_ratings.get(p1_id, 1500))
    p2_elo = elo_ratings.get(p2_id, {}).get(surface, global_ratings.get(p2_id, 1500))
    
    # Get form
    p1_form = calc_form(p1_id, surface, form_history, match_history)
    p2_form = calc_form(p2_id, surface, form_history, match_history)
    
    elo_diff = p1_elo - p2_elo
    elo_expected = 1 / (1 + math.pow(10, (-elo_diff) / 400))
    
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
        'w_fatigue': float(p1_form['fatigue']),
        'w_top10': float(p1_form['top10_win_pct']),
        'w_surface': float(p1_form['surface_win_pct']),
        'l_win_pct': float(p2_form['win_pct']),
        'l_momentum': float(p2_form['momentum']),
        'l_streak': float(p2_form['streak']),
        'l_consistency': float(p2_form['consistency']),
        'l_fatigue': float(p2_form['fatigue']),
        'l_top10': float(p2_form['top10_win_pct']),
        'l_surface': float(p2_form['surface_win_pct']),
        'form_diff': float(p1_form['win_pct'] - p2_form['win_pct']),
        'momentum_diff': float(p1_form['momentum'] - p2_form['momentum']),
    }
    
    features_df = pd.DataFrame([features_dict])
    features_scaled = st.session_state.scaler.transform(features_df)
    
    prediction = st.session_state.ensemble_model.predict_proba(features_scaled)[0][1]
    
    # Prepare detailed info
    details = {
        'p1_elo': p1_elo,
        'p2_elo': p2_elo,
        'elo_diff': elo_diff,
        'elo_expected': elo_expected,
        'p1_form': p1_form,
        'p2_form': p2_form,
    }
    
    return prediction, details

# ===========================
# UI
# ===========================

def main():
    st.title("🎾 Tennis Prediction System")
    st.markdown("**Advanced ELO + Ensemble ML + Detailed Analytics**")
    
    tabs = st.tabs(["📊 Train", "🎯 Predict", "📈 Analytics", "🤖 Info"])
    
    with tabs[0]:
        st.header("Train Model")
        
        col1, col2 = st.columns([2, 1])
        with col1:
            file = st.file_uploader("Upload CSV", type=['csv'])
        with col2:
            st.subheader("Settings")
            k = st.slider("K-factor", 20, 50, 32)
            init_elo = st.slider("Init ELO", 1400, 1600, 1500)
        
        if file:
            df = pd.read_csv(file)
            st.session_state.match_data = df
            
            st.dataframe(df.head(), width='stretch')
            st.write(f"Total Matches: {len(df)}")
            
            if all(col in df.columns for col in ['Player_1', 'Player_2', 'Winner', 'Surface']):
                st.success("✅ Valid data format")
                
                if st.button("🚀 Train Model", type="primary"):
                    with st.spinner("Computing ELO and training..."):
                        elos, g_elos = compute_elo(df, k_factor=k, initial_elo=init_elo)
                        st.session_state.elo_ratings = elos
                        st.session_state.global_elo = g_elos
                        
                        features_df, labels = create_features(
                            df, elos, g_elos,
                            st.session_state.player_form_history,
                            st.session_state.match_history
                        )
                        
                        st.info(f"📊 Created {len(features_df)} training samples")
                        st.write(f"✓ Wins: {sum(labels)} | Losses: {len(labels) - sum(labels)}")
                        
                        if len(np.unique(labels)) > 1:
                            model, metrics = train_model(features_df, labels)
                            st.session_state.ensemble_model = model
                            st.session_state.model_metrics = metrics
                            
                            st.success("✅ Model trained successfully!")
                            
                            col1, col2, col3, col4, col5 = st.columns(5)
                            col1.metric("Accuracy", f"{metrics['accuracy']:.1%}")
                            col2.metric("Precision", f"{metrics['precision']:.1%}")
                            col3.metric("Recall", f"{metrics['recall']:.1%}")
                            col4.metric("F1 Score", f"{metrics['f1']:.1%}")
                            col5.metric("ROC-AUC", f"{metrics['roc_auc']:.3f}")
    
    with tabs[1]:
        st.header("🎯 Match Prediction")
        
        if not st.session_state.ensemble_model:
            st.warning("⚠️ Please train the model first in the 'Train' tab")
        else:
            st.markdown("### Select Players and Surface")
            col1, col2, col3 = st.columns(3)
            players = list(st.session_state.player_names.values())
            
            with col1:
                p1 = st.selectbox("Player 1", players, key="p1_select")
            with col2:
                p2 = st.selectbox("Player 2", [p for p in players if p != p1], key="p2_select")
            with col3:
                surf = st.selectbox("Court Surface", SURFACE_TYPES, key="surf_select")
            
            if st.button("🔮 Calculate Prediction", type="primary", use_container_width=True):
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
                    
                    if prob is not None:
                        st.markdown("---")
                        st.markdown("### 📊 PREDICTION RESULTS")
                        
                        # Main prediction display
                        col1, col2 = st.columns(2)
                        with col1:
                            st.metric(
                                f"🎾 {p1} Win Probability",
                                f"{prob * 100:.1f}%",
                                delta=f"{(prob - 0.5) * 100:.1f}% vs 50-50",
                                delta_color="normal"
                            )
                        with col2:
                            st.metric(
                                f"🎾 {p2} Win Probability",
                                f"{(1 - prob) * 100:.1f}%",
                                delta=f"{((1 - prob) - 0.5) * 100:.1f}% vs 50-50",
                                delta_color="normal"
                            )
                        
                        # Visualization
                        st.markdown("### Visual Comparison")
                        fig = go.Figure(data=[
                            go.Bar(
                                x=[p1, p2],
                                y=[prob * 100, (1 - prob) * 100],
                                marker=dict(color=['#2E86AB', '#A23B72']),
                                text=[f"{prob * 100:.1f}%", f"{(1 - prob) * 100:.1f}%"],
                                textposition='outside',
                                hovertemplate='<b>%{x}</b><br>Win Probability: %{y:.1f}%<extra></extra>'
                            )
                        ])
                        fig.update_layout(
                            title="Match Prediction Probability",
                            yaxis_title="Win Probability (%)",
                            xaxis_title="Player",
                            showlegend=False,
                            hovermode='x unified',
                            height=400
                        )
                        st.plotly_chart(fig, use_container_width=True)
                        
                        st.markdown("---")
                        st.markdown("### 📈 HOW THE PREDICTION IS CALCULATED")
                        
                        with st.expander("📌 Click to see detailed breakdown and feature values"):
                            st.markdown("#### **Prediction Formula**")
                            st.write("""
                            The prediction combines multiple factors using an **Ensemble Model** with:
                            - **Random Forest** (200 trees)
                            - **Gradient Boosting** (150 trees)
                            - **Logistic Regression**
                            
                            Each model votes on the outcome, and the average probability is calibrated for accuracy.
                            """)
                            
                            st.markdown("#### **Key Factors Contributing to This Prediction:**")
                            
                            # ELO Section
                            with st.container(border=True):
                                st.markdown("##### 🏆 ELO Rating System")
                                col1, col2, col3 = st.columns(3)
                                with col1:
                                    st.metric(f"{p1} ELO", f"{details['p1_elo']:.0f}")
                                with col2:
                                    st.metric(f"{p2} ELO", f"{details['p2_elo']:.0f}")
                                with col3:
                                    st.metric("ELO Difference", f"{details['elo_diff']:.0f}")
                                
                                st.write(f"**ELO Expected Win Probability (based on ratings only):** {details['elo_expected'] * 100:.1f}%")
                                st.write("*ELO is adjusted by surface performance and tournament importance*")
                            
                            # Player 1 Form
                            st.markdown(f"#### 📊 {p1}'s Current Form (Last 30 matches on {surf})")
                            col1, col2, col3, col4 = st.columns(4)
                            p1_f = details['p1_form']
                            with col1:
                                st.metric("Win Rate", f"{p1_f['win_pct'] * 100:.1f}%")
                            with col2:
                                st.metric("Recent Momentum", f"{p1_f['momentum'] * 100:.1f}%")
                            with col3:
                                st.metric("Consistency", f"{p1_f['consistency'] * 100:.1f}%")
                            with col4:
                                st.metric("Current Streak", f"{int(p1_f['streak'])} matches")
                            
                            col1, col2, col3, col4 = st.columns(4)
                            with col1:
                                st.metric("vs Top 10", f"{p1_f['top10_win_pct'] * 100:.1f}%")
                            with col2:
                                st.metric("On this Surface", f"{p1_f['surface_win_pct'] * 100:.1f}%")
                            with col3:
                                st.metric("Avg Opp ELO", f"{p1_f['opp_elo']:.0f}")
                            with col4:
                                st.metric("Fatigue Index", f"{int(p1_f['fatigue'])} matches/30d")
                            
                            # Player 2 Form
                            st.markdown(f"#### 📊 {p2}'s Current Form (Last 30 matches on {surf})")
                            col1, col2, col3, col4 = st.columns(4)
                            p2_f = details['p2_form']
                            with col1:
                                st.metric("Win Rate", f"{p2_f['win_pct'] * 100:.1f}%")
                            with col2:
                                st.metric("Recent Momentum", f"{p2_f['momentum'] * 100:.1f}%")
                            with col3:
                                st.metric("Consistency", f"{p2_f['consistency'] * 100:.1f}%")
                            with col4:
                                st.metric("Current Streak", f"{int(p2_f['streak'])} matches")
                            
                            col1, col2, col3, col4 = st.columns(4)
                            with col1:
                                st.metric("vs Top 10", f"{p2_f['top10_win_pct'] * 100:.1f}%")
                            with col2:
                                st.metric("On this Surface", f"{p2_f['surface_win_pct'] * 100:.1f}%")
                            with col3:
                                st.metric("Avg Opp ELO", f"{p2_f['opp_elo']:.0f}")
                            with col4:
                                st.metric("Fatigue Index", f"{int(p2_f['fatigue'])} matches/30d")
                            
                            # Key Differences
                            st.markdown("#### ⚖️ Head-to-Head Comparison")
                            comparison_data = {
                                'Metric': ['Win Rate', 'Momentum', 'Consistency', 'vs Top 10', 'Surface Specialty'],
                                f'{p1}': [
                                    f"{p1_f['win_pct'] * 100:.1f}%",
                                    f"{p1_f['momentum'] * 100:.1f}%",
                                    f"{p1_f['consistency'] * 100:.1f}%",
                                    f"{p1_f['top10_win_pct'] * 100:.1f}%",
                                    f"{p1_f['surface_win_pct'] * 100:.1f}%"
                                ],
                                f'{p2}': [
                                    f"{p2_f['win_pct'] * 100:.1f}%",
                                    f"{p2_f['momentum'] * 100:.1f}%",
                                    f"{p2_f['consistency'] * 100:.1f}%",
                                    f"{p2_f['top10_win_pct'] * 100:.1f}%",
                                    f"{p2_f['surface_win_pct'] * 100:.1f}%"
                                ]
                            }
                            comparison_df = pd.DataFrame(comparison_data)
                            st.dataframe(comparison_df, use_container_width=True, hide_index=True)
                        
                        st.markdown("---")
                        st.markdown("### 💡 Model Confidence")
                        confidence = max(prob, 1 - prob)
                        st.progress(confidence, text=f"Confidence: {confidence * 100:.1f}%")
                        
                        if confidence > 0.65:
                            st.success("✅ High Confidence Prediction")
                        elif confidence > 0.55:
                            st.info("ℹ️ Moderate Confidence - Close Match Expected")
                        else:
                            st.warning("⚠️ Low Confidence - Very Competitive Match")
    
    with tabs[2]:
        st.header("📈 Player Analytics")
        if st.session_state.match_data is not None:
            player = st.selectbox("Select Player", list(st.session_state.player_names.values()), key="analytics_player")
            if player:
                pid = st.session_state.player_ids.get(player)
                matches = st.session_state.match_history.get(pid, [])
                wins = sum(1 for m in matches if m['won'])
                
                col1, col2, col3, col4 = st.columns(4)
                col1.metric("Total Matches", len(matches))
                col2.metric("Wins", wins)
                col3.metric("Losses", len(matches) - wins)
                col4.metric("Win Rate", f"{wins/len(matches)*100:.1f}%" if matches else "N/A")
                
                if pid in st.session_state.elo_ratings:
                    st.subheader("ELO Rating by Surface")
                    elo_data = []
                    for surface in SURFACE_TYPES:
                        elo = st.session_state.elo_ratings.get(pid, {}).get(surface, 1500)
                        elo_data.append({'Surface': surface, 'ELO': elo})
                    
                    elo_df = pd.DataFrame(elo_data)
                    st.bar_chart(elo_df.set_index('Surface')['ELO'])
        else:
            st.info("Train model first to see analytics")
    
    with tabs[3]:
        st.header("🤖 Model Information")
        if st.session_state.model_metrics:
            st.subheader("Model Performance Metrics")
            metrics_col = st.columns(5)
            m = st.session_state.model_metrics
            metrics_col[0].metric("Accuracy", f"{m['accuracy']:.1%}")
            metrics_col[1].metric("Precision", f"{m['precision']:.1%}")
            metrics_col[2].metric("Recall", f"{m['recall']:.1%}")
            metrics_col[3].metric("F1 Score", f"{m['f1']:.1%}")
            metrics_col[4].metric("ROC-AUC", f"{m['roc_auc']:.3f}")
            
            st.markdown("### How the Model Works")
            st.info("""
            **Ensemble Approach:** The prediction combines 3 different machine learning models:
            - **Random Forest:** Captures non-linear patterns in player performance
            - **Gradient Boosting:** Focuses on difficult-to-predict cases
            - **Logistic Regression:** Provides baseline probability calibration
            
            **Features Used (20 total):**
            - ELO ratings (surface-specific and global)
            - Recent form (win %, momentum, consistency)
            - Head-to-head records
            - Performance vs top players
            - Surface specialization
            - Fatigue level
            
            **Calibration:** Sigmoid calibration ensures probabilities are reliable for decision-making.
            """)
        else:
            st.warning("Train a model to see detailed information")

if __name__ == "__main__":
    main()

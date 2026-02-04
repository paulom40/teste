import streamlit as st
import pandas as pd
import numpy as np
import math
import warnings
from collections import defaultdict, deque
from datetime import datetime

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
for key in ['elo_ratings', 'player_names', 'global_elo', 'match_data', 
            'player_form_history', 'player_ids', 'match_history', 'surface_performance']:
    if key not in st.session_state:
        st.session_state[key] = {}

if 'scaler' not in st.session_state:
    st.session_state.scaler = RobustScaler()
if 'ensemble_model' not in st.session_state:
    st.session_state.ensemble_model = None
if 'model_metrics' not in st.session_state:
    st.session_state.model_metrics = {}

SURFACE_TYPES = ['Hard', 'Clay', 'Grass', 'Carpet']

def get_default_form():
    """Default form dict"""
    return {
        'wins': 0, 'matches': 0, 'win_pct': 0.5, 'momentum': 0.5,
        'opp_elo': 1500, 'streak': 0, 'consistency': 0.5, 'fatigue': 0,
        'top10_pct': 0.5, 'surface_pct': 0.5
    }

def compute_elo(df, k_factor=32, initial_elo=1500):
    """Compute ELO"""
    df = df.copy()
    df['Player_1'] = df['Player_1'].astype(str).str.strip()
    df['Player_2'] = df['Player_2'].astype(str).str.strip()
    df['Winner'] = df['Winner'].astype(str).str.strip()
    df['Surface'] = df['Surface'].astype(str).str.strip()
    
    # Get all unique players
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
    form_history = defaultdict(lambda: defaultdict(deque))
    match_history = defaultdict(list)
    surface_perf = defaultdict(lambda: defaultdict(list))
    
    if 'Date' in df.columns:
        df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
    else:
        df['Date'] = datetime.now()
    
    # Process each match
    valid_count = 0
    for idx, row in df.iterrows():
        p1 = row['Player_1']
        p2 = row['Player_2']
        winner = row['Winner']
        surface = row['Surface'] if row['Surface'] in SURFACE_TYPES else 'Hard'
        
        # Determine winner and loser
        if winner == p1:
            winner_id = player_ids[p1]
            loser_id = player_ids[p2]
        elif winner == p2:
            winner_id = player_ids[p2]
            loser_id = player_ids[p1]
        else:
            continue
        
        valid_count += 1
        
        rating_w = elo_ratings[winner_id][surface]
        rating_l = elo_ratings[loser_id][surface]
        
        k = k_factor / (1 + len(match_history[winner_id]) / 50)
        exp_w = 1 / (1 + math.pow(10, (rating_l - rating_w) / 400))
        
        elo_ratings[winner_id][surface] = rating_w + k * (1 - exp_w)
        elo_ratings[loser_id][surface] = rating_l + k * (0 - (1 - exp_w))
        global_ratings[winner_id] = global_ratings.get(winner_id, initial_elo) + k * (1 - exp_w)
        global_ratings[loser_id] = global_ratings.get(loser_id, initial_elo) + k * (0 - (1 - exp_w))
        
        match_record = {
            'date': row['Date'], 'surface': surface,
            'w_elo': rating_w, 'l_elo': rating_l
        }
        
        match_history[winner_id].append({**match_record, 'won': True})
        match_history[loser_id].append({**match_record, 'won': False})
        
        surface_perf[winner_id][surface].append(True)
        surface_perf[loser_id][surface].append(False)
        
        form_history[winner_id][surface].append({**match_record, 'won': True})
        form_history[loser_id][surface].append({**match_record, 'won': False})
        
        if len(form_history[winner_id][surface]) > 50:
            form_history[winner_id][surface].popleft()
        if len(form_history[loser_id][surface]) > 50:
            form_history[loser_id][surface].popleft()
    
    st.session_state.player_form_history = form_history
    st.session_state.match_history = match_history
    st.session_state.surface_performance = surface_perf
    
    return elo_ratings, global_ratings, valid_count

def calc_form(player_id, surface, form_history, match_history, current_elo):
    """Calculate form"""
    if player_id not in form_history or surface not in form_history[player_id]:
        return get_default_form()
    
    recent = list(form_history[player_id][surface])
    if not recent:
        return get_default_form()
    
    wins = sum(1 for m in recent if m['won'])
    total = len(recent)
    
    momentum = sum(1 for m in recent[-5:] if m['won']) / min(5, len(recent)) if recent else 0.5
    
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
    opp_elo = np.mean(opp_elos) if opp_elos else 1500
    
    top10 = sum(1 for m in recent if m['won'] and m['l_elo'] > 1750)
    top10_total = sum(1 for m in recent if m['l_elo'] > 1750)
    top10_pct = top10 / top10_total if top10_total > 0 else 0.5
    
    perf = []
    for m in recent:
        opp = m['l_elo'] if m['won'] else m['w_elo']
        exp = 1 / (1 + math.pow(10, (opp - current_elo) / 400))
        perf.append((1 - exp) if m['won'] else (0 - exp))
    consistency = 1 - min(np.std(perf) if perf else 0, 1.0)
    
    surf_results = st.session_state.surface_performance.get(player_id, {}).get(surface, [])
    surface_pct = sum(1 for r in surf_results if r) / max(len(surf_results), 1) if surf_results else 0.5
    
    fatigue = len([m for m in match_history.get(player_id, [])
                  if (datetime.now() - m['date']).days <= 30])
    
    return {
        'wins': wins, 'matches': total, 'win_pct': wins / total if total > 0 else 0.5,
        'momentum': momentum, 'opp_elo': opp_elo, 'streak': streak,
        'consistency': consistency, 'fatigue': fatigue,
        'top10_pct': top10_pct, 'surface_pct': surface_pct
    }

def create_features(df, elo_ratings, global_ratings, form_history, match_history):
    """Create features - FIXED"""
    df = df.copy()
    df['Player_1'] = df['Player_1'].astype(str).str.strip()
    df['Player_2'] = df['Player_2'].astype(str).str.strip()
    df['Winner'] = df['Winner'].astype(str).str.strip()
    df['Surface'] = df['Surface'].astype(str).str.strip()
    
    player_ids = st.session_state.player_ids
    
    features_list = []
    labels = []
    
    for idx, row in df.iterrows():
        p1 = row['Player_1']
        p2 = row['Player_2']
        winner = row['Winner']
        surface = row['Surface'] if row['Surface'] in SURFACE_TYPES else 'Hard'
        
        # Get IDs
        if winner == p1:
            w_id = player_ids.get(p1)
            l_id = player_ids.get(p2)
        elif winner == p2:
            w_id = player_ids.get(p2)
            l_id = player_ids.get(p1)
        else:
            continue
        
        if w_id is None or l_id is None:
            continue
        
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
            'l_win_pct': float(l_form['win_pct']),
            'l_momentum': float(l_form['momentum']),
            'l_streak': float(l_form['streak']),  # FIXED: was 'form_streak'
            'l_consistency': float(l_form['consistency']),
            'l_fatigue': float(l_form['fatigue'] / 5),
            'l_top10': float(l_form['top10_pct']),
            'l_surface': float(l_form['surface_pct']),
            'form_diff': float(w_form['win_pct'] - l_form['win_pct']),
        }
        
        features_list.append(features)
        labels.append(1)
        
        # Loser perspective
        l_features = {
            'elo_diff': -features['elo_diff'],
            'elo_ratio': 1 / features['elo_ratio'] if features['elo_ratio'] > 0 else 1.0,
            'is_hard': features['is_hard'],
            'is_clay': features['is_clay'],
            'is_grass': features['is_grass'],
            'w_win_pct': features['l_win_pct'],
            'w_momentum': features['l_momentum'],
            'w_streak': features['l_streak'],
            'w_consistency': features['l_consistency'],
            'w_fatigue': features['l_fatigue'],
            'w_top10': features['l_top10'],
            'w_surface': features['l_surface'],
            'l_win_pct': features['w_win_pct'],
            'l_momentum': features['w_momentum'],
            'l_streak': features['w_streak'],
            'l_consistency': features['w_consistency'],
            'l_fatigue': features['w_fatigue'],
            'l_top10': features['w_top10'],
            'l_surface': features['w_surface'],
            'form_diff': -features['form_diff'],
        }
        
        features_list.append(l_features)
        labels.append(0)
    
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
        'ada': AdaBoostClassifier(n_estimators=100, random_state=42),
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
        raise ValueError("Could not train any models")
    
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
    """Predict"""
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
        'l_win_pct': float(p2_form['win_pct']),
        'l_momentum': float(p2_form['momentum']),
        'l_streak': float(p2_form['streak']),
        'l_consistency': float(p2_form['consistency']),
        'l_fatigue': float(p2_form['fatigue'] / 5),
        'l_top10': float(p2_form['top10_pct']),
        'l_surface': float(p2_form['surface_pct']),
        'form_diff': float(p1_form['win_pct'] - p2_form['win_pct']),
    }
    
    features_df = pd.DataFrame([features_dict])
    features_scaled = st.session_state.scaler.transform(features_df)
    prediction = st.session_state.ensemble_model.predict_proba(features_scaled)[0][1]
    
    return prediction, {'p1_elo': p1_elo, 'p2_elo': p2_elo, 'p1_form': p1_form, 'p2_form': p2_form}

# ===========================
# UI
# ===========================

def main():
    st.title("🎾 Tennis Prediction System")
    st.markdown("**Advanced ELO + 19 Features + 4-Model Ensemble**")
    
    tabs = st.tabs(["📊 Train", "🎯 Predict", "📈 Stats", "🤖 Info"])
    
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
            
            st.write("**Preview:**")
            st.dataframe(df.head(5), width='stretch')
            st.write(f"**Total rows:** {len(df)}")
            
            required = ['Player_1', 'Player_2', 'Winner', 'Surface']
            if all(col in df.columns for col in required):
                st.success("✅ Valid CSV format")
                
                if st.button("🚀 Train Model", type="primary"):
                    with st.spinner("🔄 Processing ELO..."):
                        elos, g_elos, valid = compute_elo(df, k_factor=k)
                        st.session_state.elo_ratings = elos
                        st.session_state.global_elo = g_elos
                        
                        st.success(f"✅ ELO computed for {len(st.session_state.player_ids)} players from {valid} valid matches")
                    
                    with st.spinner("🔄 Creating features..."):
                        features_df, labels = create_features(
                            df, elos, g_elos,
                            st.session_state.player_form_history,
                            st.session_state.match_history
                        )
                        
                        st.success(f"✅ Created {len(features_df)} samples with {features_df.shape[1]} features")
                        
                        if len(features_df) > 0:
                            with st.spinner("🔄 Training models..."):
                                model, metrics = train_model(features_df, labels)
                                st.session_state.ensemble_model = model
                                st.session_state.model_metrics = metrics
                            
                            st.success("✅ Model trained successfully!")
                            
                            col1, col2, col3, col4, col5 = st.columns(5)
                            col1.metric("Accuracy", f"{metrics['accuracy']:.1%}")
                            col2.metric("Precision", f"{metrics['precision']:.1%}")
                            col3.metric("Recall", f"{metrics['recall']:.1%}")
                            col4.metric("F1-Score", f"{metrics['f1']:.1%}")
                            col5.metric("ROC-AUC", f"{metrics['roc_auc']:.3f}")
                        else:
                            st.error("❌ No features created. Check your CSV data.")
    
    with tabs[1]:
        st.header("🎯 Match Prediction")
        
        if not st.session_state.ensemble_model:
            st.warning("⚠️ Please train the model first!")
        else:
            col1, col2, col3 = st.columns(3)
            players = list(st.session_state.player_names.values())
            
            with col1:
                p1 = st.selectbox("Player 1", players)
            with col2:
                p2 = st.selectbox("Player 2", [p for p in players if p != p1])
            with col3:
                surf = st.selectbox("Surface", SURFACE_TYPES)
            
            if st.button("🔮 Predict Match", type="primary"):
                p1_id = st.session_state.player_ids.get(p1)
                p2_id = st.session_state.player_ids.get(p2)
                
                if p1_id and p2_id:
                    result = predict_match(p1_id, p2_id, surf,
                                          st.session_state.elo_ratings,
                                          st.session_state.global_elo,
                                          st.session_state.player_form_history,
                                          st.session_state.match_history)
                    
                    if result:
                        prob, details = result
                        
                        col1, col2 = st.columns(2)
                        with col1:
                            st.metric(f"✅ {p1} Win %", f"{prob*100:.1f}%")
                        with col2:
                            st.metric(f"❌ {p2} Win %", f"{(1-prob)*100:.1f}%")
                        
                        with st.expander("📊 Detailed Breakdown"):
                            st.write("**ELO Ratings:**")
                            col1, col2, col3 = st.columns(3)
                            with col1:
                                st.metric(f"{p1} ELO", f"{details['p1_elo']:.0f}")
                            with col2:
                                st.metric("Difference", f"{details['p1_elo'] - details['p2_elo']:+.0f}")
                            with col3:
                                st.metric(f"{p2} ELO", f"{details['p2_elo']:.0f}")
                            
                            st.write("**Form Stats:**")
                            col1, col2 = st.columns(2)
                            with col1:
                                st.write(f"**{p1}**")
                                f1 = details['p1_form']
                                st.metric("Win %", f"{f1['win_pct']*100:.0f}%")
                                st.metric("Momentum", f"{f1['momentum']*100:.0f}%")
                                st.metric("Consistency", f"{f1['consistency']*100:.0f}%")
                            with col2:
                                st.write(f"**{p2}**")
                                f2 = details['p2_form']
                                st.metric("Win %", f"{f2['win_pct']*100:.0f}%")
                                st.metric("Momentum", f"{f2['momentum']*100:.0f}%")
                                st.metric("Consistency", f"{f2['consistency']*100:.0f}%")
    
    with tabs[2]:
        st.header("📈 Player Statistics")
        if st.session_state.player_ids:
            player = st.selectbox("Select Player", list(st.session_state.player_names.values()))
            if player:
                pid = st.session_state.player_ids.get(player)
                matches = st.session_state.match_history.get(pid, [])
                wins = sum(1 for x in matches if x['won'])
                
                col1, col2, col3, col4 = st.columns(4)
                col1.metric("Total Matches", len(matches))
                col2.metric("Wins", wins)
                col3.metric("Losses", len(matches) - wins)
                col4.metric("Win %", f"{wins/len(matches)*100:.1f}%" if matches else "N/A")
    
    with tabs[3]:
        st.header("🤖 Model Information")
        if st.session_state.model_metrics:
            st.json(st.session_state.model_metrics)
            st.info("""
            **Models Used:**
            - Random Forest (200 trees)
            - Gradient Boosting (150 trees)
            - AdaBoost (100 trees)
            - Logistic Regression
            
            **Features (19 total):**
            - ELO difference & ratio
            - Win percentages
            - Momentum & consistency
            - Streak & fatigue
            - Surface performance
            - Head-to-head form
            """)
        else:
            st.warning("Train a model to see metrics")

if __name__ == "__main__":
    main()

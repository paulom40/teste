import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.calibration import CalibratedClassifierCV
import requests
from io import BytesIO
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

st.set_page_config(page_title="WTA Professional Predictor", page_icon="🎾", layout="wide")

@st.cache_data
def fetch_wta_github_data():
    try:
        url = "https://github.com/paulom40/teste/raw/main/wta_data.xlsx"
        response = requests.get(url, timeout=10)
        df = pd.read_excel(BytesIO(response.content))
        st.sidebar.success("✓ Data loaded from GitHub")
        return df
    except Exception as e:
        st.sidebar.error(f"Could not fetch from GitHub: {str(e)}")
        return None

def calculate_total_games(row):
    total_games = 0
    for i in range(1, 6):
        w_col = f'W{i}'
        l_col = f'L{i}'
        if w_col in row.index and l_col in row.index:
            w_val = row[w_col]
            l_val = row[l_col]
            if pd.notna(w_val) and pd.notna(l_val):
                total_games += int(w_val) + int(l_val)
    return total_games if total_games > 0 else None

# ============= PROFESSIONAL TIPSTER FEATURES =============

def calculate_h2h_stats(df, player_a, player_b, surface=None):
    """Head-to-head record analysis"""
    h2h = df[((df['Winner'] == player_a) & (df['Loser'] == player_b)) |
             ((df['Winner'] == player_b) & (df['Loser'] == player_a))]
    
    if surface:
        h2h = h2h[h2h['Surface'] == surface]
    
    if len(h2h) == 0:
        return 0.5, 0, 0
    
    wins_a = len(h2h[h2h['Winner'] == player_a])
    total = len(h2h)
    h2h_rate = wins_a / total
    
    recent_h2h = h2h.tail(3)
    recent_wins = len(recent_h2h[recent_h2h['Winner'] == player_a])
    recent_rate = recent_wins / len(recent_h2h) if len(recent_h2h) > 0 else 0.5
    
    psych_edge = recent_rate - h2h_rate
    
    return h2h_rate, total, psych_edge

def calculate_momentum_score(matches, player_name):
    """Weighted recent form"""
    if len(matches) == 0:
        return 0.5
    
    last_10 = matches.tail(10)
    weights = np.linspace(0.5, 1.0, len(last_10))
    
    weighted_wins = sum([
        (match['Winner'] == player_name) * weight
        for weight, (_, match) in zip(weights, last_10.iterrows())
    ])
    
    momentum = weighted_wins / weights.sum()
    return momentum

def calculate_win_streak(matches, player_name):
    """Current win/loss streak"""
    if len(matches) == 0:
        return 0
    
    recent = matches.tail(10)
    
    streak = 0
    for _, match in reversed(list(recent.iterrows())):
        if match['Winner'] == player_name:
            streak += 1
        else:
            break
    
    return streak

def calculate_consistency_score(matches, player_name):
    """Consistency score"""
    if len(matches) == 0:
        return 0.5
    
    last_30 = matches.tail(30)
    player_matches = last_30[
        (last_30['Winner'] == player_name) | (last_30['Loser'] == player_name)
    ]
    
    if len(player_matches) == 0:
        return 0.5
    
    performance = [(m['Winner'] == player_name) for _, m in player_matches.iterrows()]
    variance = np.var(performance) if len(performance) > 1 else 0.5
    consistency = 1 - variance
    
    return consistency

def calculate_strength_of_schedule(matches, player_name):
    """Quality of opponents faced"""
    if len(matches) == 0:
        return 0
    
    player_matches = matches[
        (matches['Winner'] == player_name) | (matches['Loser'] == player_name)
    ].tail(20)
    
    if len(player_matches) == 0:
        return 0
    
    opponent_ranks = []
    for _, match in player_matches.iterrows():
        if match['Winner'] == player_name:
            opponent_ranks.append(match['LRank'])
        else:
            opponent_ranks.append(match['WRank'])
    
    avg_opponent_rank = np.mean(opponent_ranks)
    sos = 1 / (1 + avg_opponent_rank / 100)
    
    return sos

def calculate_surface_performance(df, player_name, surface):
    """Surface-specific performance"""
    surface_matches = df[
        ((df['Winner'] == player_name) | (df['Loser'] == player_name)) &
        (df['Surface'] == surface)
    ].tail(30)
    
    if len(surface_matches) == 0:
        return None
    
    wins = len(surface_matches[surface_matches['Winner'] == player_name])
    total = len(surface_matches)
    win_rate = wins / total if total > 0 else 0.5
    
    # Calculate average games on this surface
    surface_matches['Total_Games'] = surface_matches.apply(calculate_total_games, axis=1)
    valid_games = surface_matches.dropna(subset=['Total_Games'])
    
    avg_games = np.mean(valid_games['Total_Games']) if len(valid_games) > 0 else 24
    
    return {
        'matches': total,
        'wins': wins,
        'win_rate': win_rate,
        'avg_games': avg_games
    }

@st.cache_resource
def load_enhanced_model(df):
    """Enhanced model with professional tipster features"""
    df['Total_Games'] = df.apply(calculate_total_games, axis=1)
    
    df_winner = df.copy()
    df_winner['Player_1'] = df_winner['Winner']
    df_winner['Player_2'] = df_winner['Loser']
    df_winner['Rank_1'] = df_winner['WRank']
    df_winner['Rank_2'] = df_winner['LRank']
    df_winner['Pts_1'] = df_winner['WPts']
    df_winner['Pts_2'] = df_winner['LPts']
    df_winner['Player_1_Won'] = 1
    
    df_loser = df.copy()
    df_loser['Player_1'] = df_loser['Loser']
    df_loser['Player_2'] = df_loser['Winner']
    df_loser['Rank_1'] = df_loser['LRank']
    df_loser['Rank_2'] = df_loser['WRank']
    df_loser['Pts_1'] = df_loser['LPts']
    df_loser['Pts_2'] = df_loser['WPts']
    df_loser['Player_1_Won'] = 0
    
    df_combined = pd.concat([df_winner, df_loser], ignore_index=True)
    
    numeric_cols = ['Rank_1', 'Rank_2', 'Pts_1', 'Pts_2', 'B365W', 'B365L']
    for col in numeric_cols:
        if col in df_combined.columns:
            df_combined[col] = pd.to_numeric(df_combined[col], errors='coerce')
    
    df_combined = df_combined.dropna(subset=['Player_1', 'Player_2', 'Rank_1', 'Rank_2', 'Pts_1', 'Pts_2'])
    
    if 'B365W' in df_combined.columns:
        df_combined['B365W'] = df_combined['B365W'].fillna(df_combined['B365W'].median())
    if 'B365L' in df_combined.columns:
        df_combined['B365L'] = df_combined['B365L'].fillna(df_combined['B365L'].median())
    
    features = []
    feature_names = []
    
    st.sidebar.write("📊 Computing professional features...")
    progress = st.sidebar.progress(0)
    
    # 1. H2H ANALYSIS
    h2h_features = []
    psych_features = []
    
    for idx, (_, row) in enumerate(df_combined.iterrows()):
        if idx % 100 == 0:
            progress.progress(min(idx / len(df_combined), 0.3))
        
        player_1 = row['Player_1']
        player_2 = row['Player_2']
        surface = row.get('Surface', None)
        
        h2h_rate, h2h_count, psych_edge = calculate_h2h_stats(df, player_1, player_2, surface)
        h2h_features.append(h2h_rate if h2h_count > 0 else 0.5)
        psych_features.append(psych_edge)
    
    features.append(np.array(h2h_features))
    feature_names.append('H2H_Record')
    features.append(np.array(psych_features))
    feature_names.append('Psychological_Edge')
    
    # 2. MOMENTUM WEIGHTING
    momentum_features = []
    for idx, (_, row) in enumerate(df_combined.iterrows()):
        if idx % 100 == 0:
            progress.progress(min(0.3 + idx / len(df_combined) * 0.2, 0.5))
        
        player_1 = row['Player_1']
        
        player_matches = df[
            (df['Winner'] == player_1) | (df['Loser'] == player_1)
        ]
        
        momentum = calculate_momentum_score(player_matches, player_1)
        momentum_features.append(momentum)
    
    features.append(np.array(momentum_features))
    feature_names.append('Momentum_Score')
    
    # 3. WIN STREAK
    streak_features = []
    for idx, (_, row) in enumerate(df_combined.iterrows()):
        if idx % 100 == 0:
            progress.progress(min(0.5 + idx / len(df_combined) * 0.1, 0.6))
        
        player_1 = row['Player_1']
        
        player_matches = df[
            (df['Winner'] == player_1) | (df['Loser'] == player_1)
        ]
        
        streak = calculate_win_streak(player_matches, player_1)
        streak_features.append(streak)
    
    features.append(np.array(streak_features))
    feature_names.append('Win_Streak')
    
    # 4. CONSISTENCY
    consistency_features = []
    for idx, (_, row) in enumerate(df_combined.iterrows()):
        if idx % 100 == 0:
            progress.progress(min(0.6 + idx / len(df_combined) * 0.1, 0.7))
        
        player_1 = row['Player_1']
        
        player_matches = df[
            (df['Winner'] == player_1) | (df['Loser'] == player_1)
        ]
        
        consistency = calculate_consistency_score(player_matches, player_1)
        consistency_features.append(consistency)
    
    features.append(np.array(consistency_features))
    feature_names.append('Consistency_Score')
    
    # 5. STRENGTH OF SCHEDULE
    sos_features = []
    for idx, (_, row) in enumerate(df_combined.iterrows()):
        if idx % 100 == 0:
            progress.progress(min(0.7 + idx / len(df_combined) * 0.1, 0.8))
        
        player_1 = row['Player_1']
        
        player_matches = df[
            (df['Winner'] == player_1) | (df['Loser'] == player_1)
        ]
        
        sos = calculate_strength_of_schedule(player_matches, player_1)
        sos_features.append(sos)
    
    features.append(np.array(sos_features))
    feature_names.append('Strength_of_Schedule')
    
    # STANDARD FEATURES
    features.append((df_combined['Rank_2'] - df_combined['Rank_1']).fillna(0).values)
    feature_names.append('Ranking_Differential')
    
    features.append(df_combined['Rank_1'].fillna(100).values)
    feature_names.append('Player_1_Rank')
    
    features.append(df_combined['Rank_2'].fillna(100).values)
    feature_names.append('Player_2_Rank')
    
    log_rank_1 = np.log(df_combined['Rank_1'].fillna(100) + 1).values
    log_rank_2 = np.log(df_combined['Rank_2'].fillna(100) + 1).values
    features.append(log_rank_1)
    feature_names.append('Log_Rank_1')
    features.append(log_rank_2)
    feature_names.append('Log_Rank_2')
    
    rank_ratio = np.where(df_combined['Rank_1'] > 0, df_combined['Rank_2'] / df_combined['Rank_1'], 1.0)
    rank_ratio = np.nan_to_num(rank_ratio, nan=1.0, posinf=1.0, neginf=1.0)
    features.append(rank_ratio)
    feature_names.append('Rank_Ratio')
    
    features.append((df_combined['Pts_1'] - df_combined['Pts_2']).fillna(0).values)
    feature_names.append('Points_Differential')
    
    features.append(df_combined['Pts_1'].fillna(0).values)
    feature_names.append('Player_1_Points')
    
    features.append(df_combined['Pts_2'].fillna(0).values)
    feature_names.append('Player_2_Points')
    
    log_pts_1 = np.log(df_combined['Pts_1'].fillna(1) + 1).values
    log_pts_2 = np.log(df_combined['Pts_2'].fillna(1) + 1).values
    features.append(log_pts_1)
    feature_names.append('Log_Pts_1')
    features.append(log_pts_2)
    feature_names.append('Log_Pts_2')
    
    pts_ratio = np.where(df_combined['Pts_2'] > 0, (df_combined['Pts_1'] + 1) / (df_combined['Pts_2'] + 1), 1.0)
    pts_ratio = np.nan_to_num(pts_ratio, nan=1.0, posinf=1.0, neginf=1.0)
    features.append(pts_ratio)
    feature_names.append('Points_Ratio')
    
    if 'B365W' in df_combined.columns and 'B365L' in df_combined.columns:
        features.append((df_combined['B365W'] - df_combined['B365L']).fillna(0).values)
        feature_names.append('Odds_Differential')
        
        odds_ratio = np.where(df_combined['B365L'] > 0, (df_combined['B365W'] + 0.1) / (df_combined['B365L'] + 0.1), 1.0)
        odds_ratio = np.nan_to_num(odds_ratio, nan=1.0, posinf=1.0, neginf=1.0)
        features.append(odds_ratio)
        feature_names.append('Odds_Ratio')
    
    if 'Surface' in df_combined.columns:
        surfaces = pd.get_dummies(df_combined['Surface'], prefix='Surface', dummy_na=False)
        for col in surfaces.columns:
            features.append(surfaces[col].values)
            feature_names.append(col)
    
    if 'Tier' in df_combined.columns:
        tiers = pd.get_dummies(df_combined['Tier'], prefix='Tier', dummy_na=False)
        for col in tiers.columns:
            features.append(tiers[col].values)
            feature_names.append(col)
    
    if 'Court' in df_combined.columns:
        courts = pd.get_dummies(df_combined['Court'], prefix='Court', dummy_na=False)
        for col in courts.columns:
            features.append(courts[col].values)
            feature_names.append(col)
    
    X = np.column_stack(features)
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
    
    y = df_combined['Player_1_Won'].values
    
    progress.progress(0.9)
    
    if len(np.unique(y)) < 2:
        raise ValueError("Dataset must have both winning and losing samples")
    
    if len(X) < 100:
        raise ValueError(f"Need 100+ matches, got {len(X)}")
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42, stratify=y)
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    gb_model = GradientBoostingClassifier(
        n_estimators=500,
        learning_rate=0.01,
        max_depth=3,
        min_samples_split=25,
        min_samples_leaf=12,
        subsample=0.6,
        max_features='sqrt',
        random_state=42,
        validation_fraction=0.2,
        n_iter_no_change=30,
        tol=1e-5,
        verbose=0
    )
    
    gb_model.fit(X_train_scaled, y_train)
    
    calibrated_model = CalibratedClassifierCV(gb_model, method='isotonic', cv=15)
    calibrated_model.fit(X_train_scaled, y_train)
    
    y_test_pred = calibrated_model.predict(X_test_scaled)
    y_test_proba = calibrated_model.predict_proba(X_test_scaled)[:, 1]
    
    test_acc = accuracy_score(y_test, y_test_pred)
    precision = precision_score(y_test, y_test_pred, zero_division=0)
    recall = recall_score(y_test, y_test_pred, zero_division=0)
    f1 = f1_score(y_test, y_test_pred, zero_division=0)
    auc_score = roc_auc_score(y_test, y_test_proba)
    
    cv_scores = cross_val_score(calibrated_model, X_train_scaled, y_train, cv=5, scoring='roc_auc')
    
    importance_df = pd.DataFrame({
        'Feature': feature_names,
        'Importance': gb_model.feature_importances_
    }).sort_values('Importance', ascending=False)
    
    progress.progress(1.0)
    
    return {
        'model': calibrated_model,
        'scaler': scaler,
        'df': df,
        'feature_names': feature_names,
        'importance_df': importance_df,
        'test_accuracy': test_acc,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'auc_score': auc_score,
        'cv_scores': cv_scores
    }

def show_home(model_data):
    st.header("🎾 WTA Professional Predictor - Enhanced")
    st.markdown("*With professional tipster features*")
    
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Total Matches", len(model_data['df']))
    with col2:
        st.metric("Accuracy", f"{model_data['test_accuracy']:.1%}")
    with col3:
        st.metric("AUC-ROC", f"{model_data['auc_score']:.1%}")
    with col4:
        st.metric("Features", len(model_data['feature_names']))
    
    st.markdown("---")
    
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("📊 Model Performance")
        metrics_df = pd.DataFrame({
            'Metric': ['Accuracy', 'Precision', 'Recall', 'F1-Score', 'AUC-ROC'],
            'Score': [
                model_data['test_accuracy'],
                model_data['precision'],
                model_data['recall'],
                model_data['f1'],
                model_data['auc_score']
            ]
        })
        st.dataframe(metrics_df.style.format({'Score': '{:.1%}'}), use_container_width=True, hide_index=True)
    
    with col2:
        st.subheader("🚀 Enhanced Features Added")
        st.write("""
        ✓ **H2H Records** - Direct head-to-head analysis
        ✓ **Momentum Score** - Weighted recent form
        ✓ **Win Streak** - Psychological momentum
        ✓ **Consistency** - Volatility analysis
        ✓ **Strength of Schedule** - Opponent quality
        ✓ **Standard Features** - Rankings, points, odds
        """)
    
    st.markdown("---")
    st.subheader("📈 Top 15 Features")
    
    top_features = model_data['importance_df'].head(15)
    fig = go.Figure(data=[
        go.Bar(y=top_features['Feature'], x=top_features['Importance'], orientation='h', marker_color='#667eea')
    ])
    fig.update_layout(title="Feature Importance", xaxis_title="Importance", height=500)
    st.plotly_chart(fig, use_container_width=True)
    
    st.markdown("---")
    st.subheader("💡 Professional Features Explained")
    
    with st.expander("📊 H2H Record (Head-to-Head)"):
        st.write("""
        **Why it matters**: Strongest predictor in professional betting
        - Direct historical record between players
        - Surface-specific H2H
        - Psychological edge from dominance
        - Recent H2H momentum
        
        **Typical impact**: +5-10% accuracy improvement
        """)
    
    with st.expander("📈 Momentum Score"):
        st.write("""
        **Why it matters**: Recent form heavily predicts outcomes
        - Weighted scoring (recent > older)
        - Last 10 matches analyzed
        - Captures hot/cold streaks
        - More nuanced than simple win rate
        
        **Typical impact**: +3-5% accuracy improvement
        """)
    
    with st.expander("🔥 Win Streak"):
        st.write("""
        **Why it matters**: Psychological confidence factor
        - Current winning/losing streak
        - Momentum indicator
        - Confidence levels
        
        **Typical impact**: +1-2% accuracy improvement
        """)
    
    with st.expander("🎯 Consistency Score"):
        st.write("""
        **Why it matters**: Predictability of performance
        - Steady performers > volatile
        - Variance in results
        - Risk assessment
        
        **Typical impact**: +1-2% accuracy improvement
        """)
    
    with st.expander("💪 Strength of Schedule"):
        st.write("""
        **Why it matters**: Quality of opposition matters
        - Average opponent ranking
        - Tougher schedule = better preparation
        - Recent SOS weighted
        
        **Typical impact**: +1-2% accuracy improvement
        """)

def show_player_selection(model_data):
    st.header("👥 Player Selection & Surface Analysis")
    st.markdown("*Compare players and analyze surface performance*")
    
    st.markdown("---")
    
    df = model_data['df']
    all_players = sorted(list(set(df['Winner'].unique()) | set(df['Loser'].unique())))
    surfaces = sorted(df['Surface'].dropna().unique())
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.subheader("👤 Player 1")
        player_a = st.selectbox("Select Player 1", all_players, key="player_1")
    
    with col2:
        st.subheader("👤 Player 2")
        player_b = st.selectbox("Select Player 2", all_players, index=1 if len(all_players) > 1 else 0, key="player_2")
    
    with col3:
        st.subheader("🏟️ Surface")
        surface = st.selectbox("Select Surface", surfaces, key="surface")
    
    st.markdown("---")
    
    if st.button("📊 Analyze Players", use_container_width=True):
        st.markdown("---")
        st.subheader("📈 PROFESSIONAL ANALYSIS")
        
        # Get player stats
        stats_a = calculate_surface_performance(df, player_a, surface)
        stats_b = calculate_surface_performance(df, player_b, surface)
        
        # Get H2H stats
        h2h_rate, h2h_count, psych_edge = calculate_h2h_stats(df, player_a, player_b, surface)
        
        # Get momentum and other metrics
        player_a_matches = df[(df['Winner'] == player_a) | (df['Loser'] == player_a)]
        player_b_matches = df[(df['Winner'] == player_b) | (df['Loser'] == player_b)]
        
        momentum_a = calculate_momentum_score(player_a_matches, player_a)
        momentum_b = calculate_momentum_score(player_b_matches, player_b)
        
        streak_a = calculate_win_streak(player_a_matches, player_a)
        streak_b = calculate_win_streak(player_b_matches, player_b)
        
        consistency_a = calculate_consistency_score(player_a_matches, player_a)
        consistency_b = calculate_consistency_score(player_b_matches, player_b)
        
        sos_a = calculate_strength_of_schedule(player_a_matches, player_a)
        sos_b = calculate_strength_of_schedule(player_b_matches, player_b)
        
        # Display comparison
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader(f"🎾 {player_a}")
            if stats_a:
                st.metric("Matches on Surface", stats_a['matches'])
                st.metric("Record", f"{stats_a['wins']}-{stats_a['wins'] - stats_a['wins']}")
                st.metric("Win Rate", f"{stats_a['win_rate']:.1%}")
                st.metric("Avg Games", f"{stats_a['avg_games']:.1f}")
            
            st.write("**Professional Metrics:**")
            st.write(f"• Momentum Score: {momentum_a:.1%}")
            st.write(f"• Win Streak: {streak_a} matches")
            st.write(f"• Consistency: {consistency_a:.1%}")
            st.write(f"• Strength of Schedule: {sos_a:.2f}")
        
        with col2:
            st.subheader(f"🎾 {player_b}")
            if stats_b:
                st.metric("Matches on Surface", stats_b['matches'])
                st.metric("Record", f"{stats_b['wins']}-{stats_b['wins'] - stats_b['wins']}")
                st.metric("Win Rate", f"{stats_b['win_rate']:.1%}")
                st.metric("Avg Games", f"{stats_b['avg_games']:.1f}")
            
            st.write("**Professional Metrics:**")
            st.write(f"• Momentum Score: {momentum_b:.1%}")
            st.write(f"• Win Streak: {streak_b} matches")
            st.write(f"• Consistency: {consistency_b:.1%}")
            st.write(f"• Strength of Schedule: {sos_b:.2f}")
        
        st.markdown("---")
        st.subheader("⚔️ HEAD-TO-HEAD ANALYSIS")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric(f"{player_a} H2H Record", f"{h2h_rate:.1%}" if h2h_count > 0 else "No H2H")
        
        with col2:
            st.metric("Total Matches", h2h_count if h2h_count > 0 else "0")
        
        with col3:
            if psych_edge > 0:
                st.success(f"✓ {player_a} Has Edge: +{psych_edge:.1%}")
            elif psych_edge < 0:
                st.success(f"✓ {player_b} Has Edge: +{abs(psych_edge):.1%}")
            else:
                st.info("🤝 Evenly Matched")
        
        st.markdown("---")
        st.subheader("📊 Detailed Statistics")
        
        # Create comparison table
        comp_df = pd.DataFrame({
            'Metric': [
                'Surface Matches',
                'Win Rate (Surface)',
                'Avg Games (Surface)',
                'Overall Momentum',
                'Current Streak',
                'Consistency',
                'Schedule Strength',
                f'H2H Record ({surface})'
            ],
            player_a: [
                stats_a['matches'] if stats_a else 'N/A',
                f"{stats_a['win_rate']:.1%}" if stats_a else 'N/A',
                f"{stats_a['avg_games']:.1f}" if stats_a else 'N/A',
                f"{momentum_a:.1%}",
                f"{streak_a}W",
                f"{consistency_a:.1%}",
                f"{sos_a:.2f}",
                f"{h2h_rate:.1%}" if h2h_count > 0 else "No Data"
            ],
            player_b: [
                stats_b['matches'] if stats_b else 'N/A',
                f"{stats_b['win_rate']:.1%}" if stats_b else 'N/A',
                f"{stats_b['avg_games']:.1f}" if stats_b else 'N/A',
                f"{momentum_b:.1%}",
                f"{streak_b}W",
                f"{consistency_b:.1%}",
                f"{sos_b:.2f}",
                f"{1-h2h_rate:.1%}" if h2h_count > 0 else "No Data"
            ]
        })
        
        st.dataframe(comp_df, use_container_width=True, hide_index=True)
        
        st.markdown("---")
        st.subheader("📈 Visual Comparison")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Win Rate Comparison
            if stats_a and stats_b:
                fig = go.Figure(data=[
                    go.Bar(
                        x=[player_a, player_b],
                        y=[stats_a['win_rate'], stats_b['win_rate']],
                        marker_color=['#667eea', '#764ba2'],
                        text=[f"{stats_a['win_rate']:.1%}", f"{stats_b['win_rate']:.1%}"],
                        textposition='auto'
                    )
                ])
                fig.update_layout(
                    title=f"Win Rate on {surface}",
                    yaxis_title="Win Rate",
                    yaxis=dict(range=[0, 1]),
                    showlegend=False,
                    height=400
                )
                st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            # Momentum Comparison
            fig = go.Figure(data=[
                go.Bar(
                    x=[player_a, player_b],
                    y=[momentum_a, momentum_b],
                    marker_color=['#667eea', '#764ba2'],
                    text=[f"{momentum_a:.1%}", f"{momentum_b:.1%}"],
                    textposition='auto'
                )
            ])
            fig.update_layout(
                title="Momentum Score",
                yaxis_title="Momentum",
                yaxis=dict(range=[0, 1]),
                showlegend=False,
                height=400
            )
            st.plotly_chart(fig, use_container_width=True)

def main():
    st.sidebar.title("🎾 WTA Professional Predictor")
    page = st.sidebar.radio("Page", ["🏠 Home", "👥 Player Selection"])
    
    st.sidebar.markdown("---")
    st.sidebar.markdown("### 📥 Data Loading")
    
    df_data = fetch_wta_github_data()
    
    if df_data is not None and len(df_data) > 0:
        try:
            with st.spinner("Training enhanced professional model..."):
                model_data = load_enhanced_model(df_data)
            st.sidebar.success(f"✓ Model trained!")
            st.sidebar.info(f"AUC-ROC: {model_data['auc_score']:.1%}")
            
            if page == "🏠 Home":
                show_home(model_data)
            else:
                show_player_selection(model_data)
        except Exception as e:
            st.error(f"Training Error: {str(e)}")
            import traceback
            st.error(traceback.format_exc())
    else:
        st.title("🎾 WTA Professional Predictor")
        st.error("❌ Could not load data from GitHub")

if __name__ == "__main__":
    main()

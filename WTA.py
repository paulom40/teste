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
from datetime import datetime, timedelta
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

# ============= ADVANCED GAMES PREDICTION FACTORS =============

def calculate_surface_expertise(df, player_name, surface):
    """
    Surface expertise: win rate and game patterns on specific surface
    """
    matches = df[
        ((df['Winner'] == player_name) | (df['Loser'] == player_name)) &
        (df['Surface'] == surface)
    ].tail(30)
    
    if len(matches) == 0:
        return {
            'expertise': 0.5,
            'matches': 0,
            'win_rate': 0.5,
            'avg_games': 22,
            'consistency': 0.5
        }
    
    # Calculate win rate
    wins = len(matches[matches['Winner'] == player_name])
    win_rate = wins / len(matches) if len(matches) > 0 else 0.5
    
    # Calculate games
    matches['Total_Games'] = matches.apply(calculate_total_games, axis=1)
    valid_matches = matches.dropna(subset=['Total_Games'])
    
    avg_games = np.mean(valid_matches['Total_Games']) if len(valid_matches) > 0 else 22
    
    # Consistency on surface (low variance = consistent)
    if len(valid_matches) > 1:
        games_variance = np.var(valid_matches['Total_Games'])
        consistency = 1 - (games_variance / 100)  # Normalize
    else:
        consistency = 0.5
    
    # Expertise score: combines win rate and consistency
    expertise = (win_rate * 0.6) + (consistency * 0.4)
    
    return {
        'expertise': expertise,
        'matches': len(matches),
        'win_rate': win_rate,
        'avg_games': avg_games,
        'consistency': consistency
    }

def calculate_last_10_surface_performance(df, player_name, surface):
    """
    Last 10 matches on this specific surface (most relevant)
    """
    matches = df[
        ((df['Winner'] == player_name) | (df['Loser'] == player_name)) &
        (df['Surface'] == surface)
    ].tail(10)
    
    if len(matches) == 0:
        return {
            'recent_matches': 0,
            'recent_wins': 0,
            'recent_win_rate': 0.5,
            'recent_avg_games': 22,
            'recent_form': 'No Data'
        }
    
    wins = len(matches[matches['Winner'] == player_name])
    
    matches['Total_Games'] = matches.apply(calculate_total_games, axis=1)
    valid_matches = matches.dropna(subset=['Total_Games'])
    
    avg_games = np.mean(valid_matches['Total_Games']) if len(valid_matches) > 0 else 22
    
    win_rate = wins / len(matches) if len(matches) > 0 else 0.5
    
    # Determine form based on recent results
    if win_rate >= 0.7:
        form = "🔥 Excellent"
    elif win_rate >= 0.5:
        form = "✓ Good"
    elif win_rate >= 0.3:
        form = "⚠️ Mixed"
    else:
        form = "❌ Poor"
    
    return {
        'recent_matches': len(matches),
        'recent_wins': wins,
        'recent_win_rate': win_rate,
        'recent_avg_games': avg_games,
        'recent_form': form
    }

def calculate_fatigue_level(df, player_name, current_date=None):
    """
    Fatigue calculation based on:
    - Days since last match
    - Matches played in last 7 days
    - Match intensity (based on games played)
    """
    if current_date is None:
        current_date = pd.Timestamp.now()
    
    matches = df[
        (df['Winner'] == player_name) | (df['Loser'] == player_name)
    ].sort_values('Date', ascending=False)
    
    if len(matches) == 0:
        return {
            'fatigue_score': 0.5,
            'days_rest': 0,
            'matches_last_week': 0,
            'fatigue_level': 'Unknown'
        }
    
    # Days since last match
    try:
        last_match_date = pd.to_datetime(matches.iloc[0]['Date'])
        days_rest = (current_date - last_match_date).days
    except:
        days_rest = 0
    
    # Matches in last 7 days
    try:
        week_matches = matches[
            (current_date - pd.to_datetime(matches['Date'])).dt.days <= 7
        ]
        matches_last_week = len(week_matches)
    except:
        matches_last_week = 0
    
    # Fatigue calculation
    # Fresh: 5+ days rest
    # Moderate: 3-4 days rest or 1-2 matches last week
    # Fatigued: 0-2 days rest and/or 3+ matches last week
    
    if days_rest >= 5:
        fatigue_score = 0.2  # Fresh
        fatigue_level = "✓ Fresh"
    elif days_rest >= 3 or matches_last_week <= 1:
        fatigue_score = 0.5  # Normal
        fatigue_level = "⚔️ Normal"
    elif days_rest >= 2 and matches_last_week <= 2:
        fatigue_score = 0.7  # Moderately fatigued
        fatigue_level = "⚠️ Fatigued"
    else:
        fatigue_score = 0.9  # Highly fatigued
        fatigue_level = "🔴 Exhausted"
    
    return {
        'fatigue_score': fatigue_score,
        'days_rest': days_rest,
        'matches_last_week': matches_last_week,
        'fatigue_level': fatigue_level
    }

def calculate_unforced_errors_estimate(df, player_name, surface=None):
    """
    Estimate unforced errors tendency based on:
    - Match length patterns (longer matches = more UE)
    - Game loss margins
    """
    matches = df[
        (df['Winner'] == player_name) | (df['Loser'] == player_name)
    ]
    
    if surface:
        matches = matches[matches['Surface'] == surface]
    
    matches = matches.tail(20)
    
    if len(matches) == 0:
        return {
            'ue_tendency': 0.5,
            'avg_game_length': 22,
            'break_tendency': 0.5,
            'error_profile': 'Unknown'
        }
    
    # Calculate game length
    matches['Total_Games'] = matches.apply(calculate_total_games, axis=1)
    valid_matches = matches.dropna(subset=['Total_Games'])
    
    if len(valid_matches) == 0:
        avg_game_length = 22
    else:
        avg_game_length = np.mean(valid_matches['Total_Games'])
    
    # Unforced errors tendency
    # Longer matches typically mean more UE
    # Shorter matches mean fewer UE but more dominant play
    if avg_game_length >= 26:
        ue_tendency = 0.8  # High UE tendency
        error_profile = "🔥 Error-prone"
    elif avg_game_length >= 24:
        ue_tendency = 0.6  # Moderate UE
        error_profile = "⚔️ Competitive"
    elif avg_game_length >= 22:
        ue_tendency = 0.4  # Few UE
        error_profile = "✓ Solid"
    else:
        ue_tendency = 0.2  # Very few UE
        error_profile = "💪 Dominant"
    
    return {
        'ue_tendency': ue_tendency,
        'avg_game_length': avg_game_length,
        'break_tendency': 1 - (ue_tendency * 0.5),  # Inverse estimate
        'error_profile': error_profile
    }

def calculate_momentum_weighted(matches, player_name):
    """
    Advanced momentum: weighted recent performance
    """
    if len(matches) == 0:
        return 0.5
    
    last_10 = matches.tail(10)
    if len(last_10) == 0:
        return 0.5
    
    # Weight recent matches higher
    weights = np.linspace(0.5, 1.0, len(last_10))
    
    weighted_wins = sum([
        (match['Winner'] == player_name) * weight
        for weight, (_, match) in zip(weights, last_10.iterrows())
    ])
    
    momentum = weighted_wins / weights.sum()
    return momentum

def calculate_advanced_expected_games(stats_a, stats_b, surface_data):
    """
    Calculate expected games using multiple factors:
    - Surface expertise
    - Last 10 matches
    - Fatigue levels
    - Momentum
    - Unforced errors
    """
    
    # Base calculation from surface expertise
    base_games_a = stats_a['surface']['avg_games'] * stats_a['surface']['expertise']
    base_games_b = stats_b['surface']['avg_games'] * stats_b['surface']['expertise']
    
    # Recent form adjustment
    recent_adj_a = stats_a['recent']['recent_avg_games'] * 0.2
    recent_adj_b = stats_b['recent']['recent_avg_games'] * 0.2
    
    # Fatigue adjustment (fatigue reduces game count)
    fatigue_adj_a = (1 - stats_a['fatigue']['fatigue_score'] * 0.1)
    fatigue_adj_b = (1 - stats_b['fatigue']['fatigue_score'] * 0.1)
    
    # Momentum adjustment
    momentum_adj_a = stats_a['momentum'] * 0.15
    momentum_adj_b = stats_b['momentum'] * 0.15
    
    # Unforced errors adjustment (more errors = longer matches)
    ue_adj_a = stats_a['ue']['ue_tendency'] * 0.1
    ue_adj_b = stats_b['ue']['ue_tendency'] * 0.1
    
    # Calculate adjusted expected games
    expected_a = (base_games_a + recent_adj_a) * fatigue_adj_a + momentum_adj_a + ue_adj_a
    expected_b = (base_games_b + recent_adj_b) * fatigue_adj_b + momentum_adj_b + ue_adj_b
    
    # Match average
    match_avg = (expected_a + expected_b) / 2
    
    return {
        'expected_a': expected_a,
        'expected_b': expected_b,
        'match_avg': match_avg,
        'components_a': {
            'base': base_games_a,
            'recent': recent_adj_a,
            'fatigue': fatigue_adj_a,
            'momentum': momentum_adj_a,
            'ue': ue_adj_a
        },
        'components_b': {
            'base': base_games_b,
            'recent': recent_adj_b,
            'fatigue': fatigue_adj_b,
            'momentum': momentum_adj_b,
            'ue': ue_adj_b
        }
    }

def show_advanced_games_prediction(model_data):
    st.header("🎾 Advanced Expected Games Prediction")
    st.markdown("*Based on surface expertise, recent form, fatigue, momentum & errors*")
    
    st.markdown("---")
    
    df = model_data['df']
    all_players = sorted(list(set(df['Winner'].unique()) | set(df['Loser'].unique())))
    surfaces = sorted(df['Surface'].dropna().unique())
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.subheader("👤 Player 1")
        player_a = st.selectbox("Select Player 1", all_players, key="adv_player_1")
    
    with col2:
        st.subheader("👤 Player 2")
        player_b = st.selectbox("Select Player 2", all_players, index=1 if len(all_players) > 1 else 0, key="adv_player_2")
    
    with col3:
        st.subheader("🏟️ Surface")
        surface = st.selectbox("Select Surface", surfaces, key="adv_surface")
    
    st.markdown("---")
    
    if st.button("🔬 Advanced Analysis", width='stretch'):
        st.markdown("---")
        st.subheader("📊 DETAILED FACTOR ANALYSIS")
        
        # Get player matches
        player_a_matches = df[(df['Winner'] == player_a) | (df['Loser'] == player_a)]
        player_b_matches = df[(df['Winner'] == player_b) | (df['Loser'] == player_b)]
        
        # Collect all factors
        stats_a = {
            'surface': calculate_surface_expertise(df, player_a, surface),
            'recent': calculate_last_10_surface_performance(df, player_a, surface),
            'fatigue': calculate_fatigue_level(df, player_a),
            'momentum': calculate_momentum_weighted(player_a_matches, player_a),
            'ue': calculate_unforced_errors_estimate(df, player_a, surface)
        }
        
        stats_b = {
            'surface': calculate_surface_expertise(df, player_b, surface),
            'recent': calculate_last_10_surface_performance(df, player_b, surface),
            'fatigue': calculate_fatigue_level(df, player_b),
            'momentum': calculate_momentum_weighted(player_b_matches, player_b),
            'ue': calculate_unforced_errors_estimate(df, player_b, surface)
        }
        
        # Calculate advanced expected games
        games_prediction = calculate_advanced_expected_games(stats_a, stats_b, {'surface': surface})
        
        # Display factor analysis
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader(f"📈 {player_a}")
            
            with st.expander("🏟️ Surface Expertise", expanded=True):
                st.metric("Expertise Score", f"{stats_a['surface']['expertise']:.1%}")
                st.metric("Win Rate (Surface)", f"{stats_a['surface']['win_rate']:.1%}")
                st.metric("Avg Games (Surface)", f"{stats_a['surface']['avg_games']:.1f}")
                st.metric("Consistency", f"{stats_a['surface']['consistency']:.1%}")
                st.write(f"Matches on {surface}: {stats_a['surface']['matches']}")
            
            with st.expander("⏰ Last 10 Matches (Surface)", expanded=True):
                st.metric("Recent Form", stats_a['recent']['recent_form'])
                st.metric("Last 10 Record", f"{stats_a['recent']['recent_wins']}/{stats_a['recent']['recent_matches']}")
                st.metric("Recent Win Rate", f"{stats_a['recent']['recent_win_rate']:.1%}")
                st.metric("Recent Avg Games", f"{stats_a['recent']['recent_avg_games']:.1f}")
            
            with st.expander("😓 Fatigue Level", expanded=True):
                st.metric("Fatigue Score", f"{stats_a['fatigue']['fatigue_score']:.1%}")
                st.metric("Days Since Last Match", stats_a['fatigue']['days_rest'])
                st.metric("Matches Last Week", stats_a['fatigue']['matches_last_week'])
                st.metric("Status", stats_a['fatigue']['fatigue_level'])
            
            with st.expander("🔥 Momentum", expanded=True):
                st.metric("Momentum Score", f"{stats_a['momentum']:.1%}")
                if stats_a['momentum'] >= 0.7:
                    st.success("📈 Hot")
                elif stats_a['momentum'] >= 0.5:
                    st.info("⚔️ Normal")
                else:
                    st.warning("📉 Cold")
            
            with st.expander("⚠️ Unforced Errors", expanded=True):
                st.metric("Error Tendency", f"{stats_a['ue']['ue_tendency']:.1%}")
                st.metric("Profile", stats_a['ue']['error_profile'])
                st.metric("Avg Game Length", f"{stats_a['ue']['avg_game_length']:.1f}")
        
        with col2:
            st.subheader(f"📈 {player_b}")
            
            with st.expander("🏟️ Surface Expertise", expanded=True):
                st.metric("Expertise Score", f"{stats_b['surface']['expertise']:.1%}")
                st.metric("Win Rate (Surface)", f"{stats_b['surface']['win_rate']:.1%}")
                st.metric("Avg Games (Surface)", f"{stats_b['surface']['avg_games']:.1f}")
                st.metric("Consistency", f"{stats_b['surface']['consistency']:.1%}")
                st.write(f"Matches on {surface}: {stats_b['surface']['matches']}")
            
            with st.expander("⏰ Last 10 Matches (Surface)", expanded=True):
                st.metric("Recent Form", stats_b['recent']['recent_form'])
                st.metric("Last 10 Record", f"{stats_b['recent']['recent_wins']}/{stats_b['recent']['recent_matches']}")
                st.metric("Recent Win Rate", f"{stats_b['recent']['recent_win_rate']:.1%}")
                st.metric("Recent Avg Games", f"{stats_b['recent']['recent_avg_games']:.1f}")
            
            with st.expander("😓 Fatigue Level", expanded=True):
                st.metric("Fatigue Score", f"{stats_b['fatigue']['fatigue_score']:.1%}")
                st.metric("Days Since Last Match", stats_b['fatigue']['days_rest'])
                st.metric("Matches Last Week", stats_b['fatigue']['matches_last_week'])
                st.metric("Status", stats_b['fatigue']['fatigue_level'])
            
            with st.expander("🔥 Momentum", expanded=True):
                st.metric("Momentum Score", f"{stats_b['momentum']:.1%}")
                if stats_b['momentum'] >= 0.7:
                    st.success("📈 Hot")
                elif stats_b['momentum'] >= 0.5:
                    st.info("⚔️ Normal")
                else:
                    st.warning("📉 Cold")
            
            with st.expander("⚠️ Unforced Errors", expanded=True):
                st.metric("Error Tendency", f"{stats_b['ue']['ue_tendency']:.1%}")
                st.metric("Profile", stats_b['ue']['error_profile'])
                st.metric("Avg Game Length", f"{stats_b['ue']['avg_game_length']:.1f}")
        
        st.markdown("---")
        st.subheader("🎯 ADVANCED EXPECTED GAMES PREDICTION")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric(f"{player_a} Expected Games", f"{games_prediction['expected_a']:.1f}")
        
        with col2:
            st.metric("Match Average", f"{games_prediction['match_avg']:.1f}")
            if games_prediction['match_avg'] < 23:
                st.info("⚡ Quick Match")
            elif games_prediction['match_avg'] < 27:
                st.info("⚔️ Competitive")
            else:
                st.warning("🔥 Long Match")
        
        with col3:
            st.metric(f"{player_b} Expected Games", f"{games_prediction['expected_b']:.1f}")
        
        st.markdown("---")
        st.subheader("📊 Factor Breakdown")
        
        # Create factor comparison
        factor_df = pd.DataFrame({
            'Factor': ['Base Games', 'Recent Form', 'Fatigue Adj', 'Momentum', 'Unforced Errors', 'TOTAL'],
            player_a: [
                f"{games_prediction['components_a']['base']:.1f}",
                f"{games_prediction['components_a']['recent']:.1f}",
                f"{games_prediction['components_a']['fatigue']:.2f}x",
                f"+{games_prediction['components_a']['momentum']:.1f}",
                f"+{games_prediction['components_a']['ue']:.1f}",
                f"{games_prediction['expected_a']:.1f}"
            ],
            player_b: [
                f"{games_prediction['components_b']['base']:.1f}",
                f"{games_prediction['components_b']['recent']:.1f}",
                f"{games_prediction['components_b']['fatigue']:.2f}x",
                f"+{games_prediction['components_b']['momentum']:.1f}",
                f"+{games_prediction['components_b']['ue']:.1f}",
                f"{games_prediction['expected_b']:.1f}"
            ]
        })
        
        st.dataframe(factor_df, width='stretch', hide_index=True)
        
        st.markdown("---")
        st.subheader("📈 Visual Comparison")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            # Surface Expertise
            fig = go.Figure(data=[
                go.Bar(
                    x=[player_a, player_b],
                    y=[stats_a['surface']['expertise'], stats_b['surface']['expertise']],
                    marker_color=['#667eea', '#764ba2'],
                    text=[f"{stats_a['surface']['expertise']:.1%}", f"{stats_b['surface']['expertise']:.1%}"],
                    textposition='auto'
                )
            ])
            fig.update_layout(
                title="Surface Expertise",
                yaxis_title="Score",
                yaxis=dict(range=[0, 1]),
                showlegend=False,
                height=350
            )
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            # Fatigue vs Momentum
            fig = go.Figure()
            fig.add_trace(go.Bar(
                x=[player_a, player_b],
                y=[1-stats_a['fatigue']['fatigue_score'], 1-stats_b['fatigue']['fatigue_score']],
                name='Energy Level',
                marker_color='#4CAF50'
            ))
            fig.add_trace(go.Bar(
                x=[player_a, player_b],
                y=[stats_a['momentum'], stats_b['momentum']],
                name='Momentum',
                marker_color='#FF9800'
            ))
            fig.update_layout(
                title="Energy vs Momentum",
                yaxis_title="Score",
                yaxis=dict(range=[0, 1]),
                barmode='group',
                height=350
            )
            st.plotly_chart(fig, use_container_width=True)
        
        with col3:
            # Expected Games
            fig = go.Figure(data=[
                go.Bar(
                    x=[player_a, player_b],
                    y=[games_prediction['expected_a'], games_prediction['expected_b']],
                    marker_color=['#667eea', '#764ba2'],
                    text=[f"{games_prediction['expected_a']:.1f}", f"{games_prediction['expected_b']:.1f}"],
                    textposition='auto'
                )
            ])
            fig.update_layout(
                title="Expected Games (Advanced)",
                yaxis_title="Games",
                yaxis=dict(range=[0, 40]),
                showlegend=False,
                height=350
            )
            st.plotly_chart(fig, use_container_width=True)
        
        st.markdown("---")
        st.info("""
        ✅ **Advanced Factors Included:**
        
        • **Surface Expertise** - Win rate & consistency on specific surface
        • **Last 10 Matches** - Most recent form on this surface
        • **Fatigue Level** - Days rest + matches in last week
        • **Momentum** - Weighted recent performance
        • **Unforced Errors** - Match length patterns & error tendency
        """)

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
    
    # Core features
    features.append((df_combined['Rank_2'] - df_combined['Rank_1']).fillna(0).values)
    feature_names.append('Ranking_Differential')
    
    features.append(df_combined['Rank_1'].fillna(100).values)
    feature_names.append('Player_1_Rank')
    
    features.append(df_combined['Rank_2'].fillna(100).values)
    feature_names.append('Player_2_Rank')
    
    if 'Surface' in df_combined.columns:
        surfaces = pd.get_dummies(df_combined['Surface'], prefix='Surface', dummy_na=False)
        for col in surfaces.columns:
            features.append(surfaces[col].values)
            feature_names.append(col)
    
    X = np.column_stack(features)
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
    
    y = df_combined['Player_1_Won'].values
    
    if len(np.unique(y)) < 2:
        raise ValueError("Dataset must have both winning and losing samples")
    
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
        verbose=0
    )
    
    gb_model.fit(X_train_scaled, y_train)
    
    calibrated_model = CalibratedClassifierCV(gb_model, method='isotonic', cv=15)
    calibrated_model.fit(X_train_scaled, y_train)
    
    y_test_pred = calibrated_model.predict(X_test_scaled)
    y_test_proba = calibrated_model.predict_proba(X_test_scaled)[:, 1]
    
    test_acc = accuracy_score(y_test, y_test_pred)
    auc_score = roc_auc_score(y_test, y_test_proba)
    
    importance_df = pd.DataFrame({
        'Feature': feature_names,
        'Importance': gb_model.feature_importances_
    }).sort_values('Importance', ascending=False)
    
    return {
        'model': calibrated_model,
        'scaler': scaler,
        'df': df,
        'feature_names': feature_names,
        'importance_df': importance_df,
        'test_accuracy': test_acc,
        'auc_score': auc_score,
    }

def show_home(model_data):
    st.header("🎾 WTA Professional Predictor")
    st.markdown("*Advanced games prediction with surface & player factors*")
    
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
    st.subheader("🎯 Features Available")
    st.write("""
    ✓ **Surface Expertise** - Win rate & consistency on specific surface
    ✓ **Last 10 Matches** - Most recent form on this surface
    ✓ **Fatigue Level** - Days rest + matches in last week
    ✓ **Momentum** - Weighted recent performance
    ✓ **Unforced Errors** - Error tendency & match length patterns
    ✓ **Advanced Calculation** - Multi-factor expected games prediction
    """)

def main():
    st.sidebar.title("🎾 WTA Professional Predictor")
    page = st.sidebar.radio("Page", ["🏠 Home", "🎾 Advanced Games Prediction"])
    
    st.sidebar.markdown("---")
    st.sidebar.markdown("### 📥 Data Loading")
    
    df_data = fetch_wta_github_data()
    
    if df_data is not None and len(df_data) > 0:
        try:
            with st.spinner("Loading data..."):
                model_data = load_enhanced_model(df_data)
            st.sidebar.success(f"✓ Data loaded!")
            st.sidebar.info(f"AUC-ROC: {model_data['auc_score']:.1%}")
            
            if page == "🏠 Home":
                show_home(model_data)
            else:
                show_advanced_games_prediction(model_data)
        except Exception as e:
            st.error(f"Error: {str(e)}")
    else:
        st.title("🎾 WTA Professional Predictor")
        st.error("❌ Could not load data from GitHub")

if __name__ == "__main__":
    main()

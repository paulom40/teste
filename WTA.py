import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, log_loss
from sklearn.calibration import CalibratedClassifierCV
import warnings
warnings.filterwarnings('ignore')

st.set_page_config(page_title="WTA Predictor", page_icon="🎾", layout="wide")

def parse_score(score_str):
    """
    Parse tennis score and calculate total games played.
    Example: '6-4 6-4' = 10 games (6+4 + 6+4)
    Example: '7-5 6-3' = 16 games (7+5 + 6+3)
    """
    if not score_str or pd.isna(score_str):
        return None
    
    try:
        score_str = str(score_str).strip()
        sets = score_str.split()
        
        total_games = 0
        for set_score in sets:
            if '-' in set_score:
                parts = set_score.split('-')
                if len(parts) == 2:
                    try:
                        games_p1 = int(parts[0])
                        games_p2 = int(parts[1])
                        total_games += games_p1 + games_p2
                    except:
                        return None
        
        return total_games if total_games > 0 else None
    except:
        return None

@st.cache_resource
def load_and_train_model(csv_file):
    df = pd.read_csv(csv_file)
    
    # Convert numeric columns
    for col in ['Rank_1', 'Rank_2', 'Pts_1', 'Pts_2', 'Odd_1', 'Odd_2']:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
    
    # Parse games from score
    if 'Score' in df.columns:
        df['Total_Games'] = df['Score'].apply(parse_score)
    else:
        df['Total_Games'] = None
    
    # Drop rows with missing critical values
    df = df.dropna(subset=['Player_1', 'Player_2', 'Winner', 'Rank_1', 'Rank_2', 'Pts_1', 'Pts_2'])
    df['Player_1_Won'] = (df['Winner'] == df['Player_1']).astype(int)
    
    features = []
    feature_names = []
    
    # Ranking features
    features.append((df['Rank_2'] - df['Rank_1']).fillna(0).values)
    feature_names.append('Ranking_Differential')
    
    features.append(df['Rank_1'].fillna(100).values)
    feature_names.append('Player_1_Rank')
    
    features.append(df['Rank_2'].fillna(100).values)
    feature_names.append('Player_2_Rank')
    
    # Log transform ranks for better scaling
    log_rank_1 = np.log(df['Rank_1'].fillna(100) + 1).values
    log_rank_2 = np.log(df['Rank_2'].fillna(100) + 1).values
    features.append(log_rank_1)
    feature_names.append('Log_Rank_1')
    features.append(log_rank_2)
    feature_names.append('Log_Rank_2')
    
    # Points features
    features.append((df['Pts_1'] - df['Pts_2']).fillna(0).values)
    feature_names.append('Points_Differential')
    
    features.append(df['Pts_1'].fillna(0).values)
    feature_names.append('Player_1_Points')
    
    features.append(df['Pts_2'].fillna(0).values)
    feature_names.append('Player_2_Points')
    
    # Log transform points
    log_pts_1 = np.log(df['Pts_1'].fillna(1) + 1).values
    log_pts_2 = np.log(df['Pts_2'].fillna(1) + 1).values
    features.append(log_pts_1)
    feature_names.append('Log_Pts_1')
    features.append(log_pts_2)
    feature_names.append('Log_Pts_2')
    
    # Ratio features with safe division
    rank_ratio = np.where(df['Rank_1'] > 0, df['Rank_2'] / df['Rank_1'], 1.0)
    rank_ratio = np.nan_to_num(rank_ratio, nan=1.0, posinf=1.0, neginf=1.0)
    features.append(rank_ratio)
    feature_names.append('Rank_Ratio')
    
    pts_ratio = np.where(df['Pts_2'] > -1, (df['Pts_1'] + 1) / (df['Pts_2'] + 1), 1.0)
    pts_ratio = np.nan_to_num(pts_ratio, nan=1.0, posinf=1.0, neginf=1.0)
    features.append(pts_ratio)
    feature_names.append('Points_Ratio')
    
    # Surface features
    if 'Surface' in df.columns:
        surfaces = pd.get_dummies(df['Surface'], prefix='Surface', dummy_na=False)
        for col in surfaces.columns:
            features.append(surfaces[col].values)
            feature_names.append(col)
    
    # Round features
    if 'Round' in df.columns:
        rounds = pd.get_dummies(df['Round'], prefix='Round', dummy_na=False)
        for col in rounds.columns:
            features.append(rounds[col].values)
            feature_names.append(col)
    
    # Court features
    if 'Court' in df.columns:
        courts = pd.get_dummies(df['Court'], prefix='Court', dummy_na=False)
        for col in courts.columns:
            features.append(courts[col].values)
            feature_names.append(col)
    
    # Odds features
    if 'Odd_1' in df.columns and 'Odd_2' in df.columns:
        odds_diff = (df['Odd_1'].fillna(1.5) - df['Odd_2'].fillna(1.5)).values
        odds_diff = np.nan_to_num(odds_diff, nan=0.0)
        features.append(odds_diff)
        feature_names.append('Odds_Differential')
        
        odds_ratio = np.where(df['Odd_2'] > 0, (df['Odd_1'].fillna(1.5) + 0.1) / (df['Odd_2'].fillna(1.5) + 0.1), 1.0)
        odds_ratio = np.nan_to_num(odds_ratio, nan=1.0, posinf=1.0, neginf=1.0)
        features.append(odds_ratio)
        feature_names.append('Odds_Ratio')
        
        # Log odds
        log_odds_1 = np.log(df['Odd_1'].fillna(1.5) + 1).values
        log_odds_2 = np.log(df['Odd_2'].fillna(1.5) + 1).values
        features.append(log_odds_1)
        feature_names.append('Log_Odds_1')
        features.append(log_odds_2)
        feature_names.append('Log_Odds_2')
    
    # Stack and clean
    X = np.column_stack(features)
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
    
    y = df['Player_1_Won'].values
    
    # Verify data
    if len(X) < 50:
        raise ValueError(f"Not enough matches. Need at least 50, got {len(X)}")
    
    if np.any(~np.isfinite(X)):
        raise ValueError("Data contains NaN or infinite values after cleaning")
    
    # Split data - use 75/25 for better training
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42, stratify=y)
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # IMPROVED MODEL: Use ensemble with better hyperparameters
    gb_model = GradientBoostingClassifier(
        n_estimators=300,           # More estimators
        learning_rate=0.02,         # Lower learning rate
        max_depth=4,                # Shallower trees
        min_samples_split=20,       # More regularization
        min_samples_leaf=10,        # More regularization
        subsample=0.7,              # Stronger stochastic boosting
        max_features='sqrt',        # Feature subsampling
        random_state=42,
        validation_fraction=0.15,
        n_iter_no_change=20,
        tol=1e-4
    )
    
    gb_model.fit(X_train_scaled, y_train)
    
    # Calibrate with isotonic regression for better probability
    calibrated_model = CalibratedClassifierCV(gb_model, method='isotonic', cv=10)
    calibrated_model.fit(X_train_scaled, y_train)
    
    # Evaluate
    y_test_pred = calibrated_model.predict(X_test_scaled)
    y_test_proba = calibrated_model.predict_proba(X_test_scaled)[:, 1]
    
    test_acc = accuracy_score(y_test, y_test_pred)
    precision = precision_score(y_test, y_test_pred, zero_division=0)
    recall = recall_score(y_test, y_test_pred, zero_division=0)
    f1 = f1_score(y_test, y_test_pred, zero_division=0)
    auc_score = roc_auc_score(y_test, y_test_proba)
    
    # Cross-validation
    cv_scores = cross_val_score(calibrated_model, X_train_scaled, y_train, cv=5, scoring='roc_auc')
    
    # Feature importance
    importance_df = pd.DataFrame({
        'Feature': feature_names,
        'Importance': gb_model.feature_importances_
    }).sort_values('Importance', ascending=False)
    
    return {
        'model': calibrated_model,
        'scaler': scaler,
        'df': df,
        'y': y,
        'X_train': X_train,
        'X_test': X_test,
        'y_test': y_test,
        'y_test_proba': y_test_proba,
        'feature_names': feature_names,
        'importance_df': importance_df,
        'test_accuracy': test_acc,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'auc_score': auc_score,
        'cv_scores': cv_scores
    }

def get_last_30_matches(df, player_name):
    p1_matches = df[df['Player_1'] == player_name].copy()
    p2_matches = df[df['Player_2'] == player_name].copy()
    
    all_matches = pd.concat([p1_matches, p2_matches], ignore_index=False)
    all_matches = all_matches.sort_index()
    
    last_30 = all_matches.tail(30)
    return last_30

def get_surface_matches(df, player_name, surface):
    """Get last 30 matches on a specific surface"""
    p1_matches = df[(df['Player_1'] == player_name) & (df['Surface'] == surface)].copy()
    p2_matches = df[(df['Player_2'] == player_name) & (df['Surface'] == surface)].copy()
    
    all_matches = pd.concat([p1_matches, p2_matches], ignore_index=False)
    all_matches = all_matches.sort_index()
    
    last_30_surface = all_matches.tail(30)
    return last_30_surface

def calculate_surface_statistics(df, player_name, surface):
    """Calculate surface-specific statistics with accurate game counting"""
    surface_matches = get_surface_matches(df, player_name, surface)
    
    if len(surface_matches) == 0:
        return None
    
    # Calculate wins/losses
    wins = len(surface_matches[surface_matches['Winner'] == player_name])
    losses = len(surface_matches) - wins
    win_rate = wins / len(surface_matches) if len(surface_matches) > 0 else 0
    
    # Analyze games accurately
    total_games_list = []
    games_when_win = []
    games_when_loss = []
    avg_games_won = 0
    avg_games_lost = 0
    
    for _, match in surface_matches.iterrows():
        total_games = parse_score(match.get('Score', ''))
        
        if total_games:
            total_games_list.append(total_games)
            
            if match['Winner'] == player_name:
                games_when_win.append(total_games)
            else:
                games_when_loss.append(total_games)
    
    # Calculate averages
    if games_when_win:
        avg_games_won = np.mean(games_when_win)
    if games_when_loss:
        avg_games_lost = np.mean(games_when_loss)
    
    avg_total_games = np.mean(total_games_list) if total_games_list else 26  # Average match ~26 games (2-0 or splits)
    
    # Expected games = (win_rate * avg_games_when_win) + ((1-win_rate) * avg_games_when_loss)
    expected_games = (win_rate * avg_games_won) + ((1 - win_rate) * avg_games_lost) if (games_when_win or games_when_loss) else avg_total_games
    
    return {
        'total_matches': len(surface_matches),
        'wins': wins,
        'losses': losses,
        'win_rate': win_rate,
        'avg_games_when_win': avg_games_won,
        'avg_games_when_loss': avg_games_lost,
        'expected_games': expected_games,
        'avg_total_games': avg_total_games,
        'total_games_samples': len(total_games_list),
        'last_matches': surface_matches
    }

def calculate_opponent_strength(last_30_matches, player_name):
    if len(last_30_matches) == 0:
        return None
    
    opponent_ranks = []
    opponent_results = []
    
    for _, match in last_30_matches.iterrows():
        if match['Player_1'] == player_name:
            opponent_rank = match['Rank_2']
            result = 1 if match['Winner'] == player_name else 0
        else:
            opponent_rank = match['Rank_1']
            result = 1 if match['Winner'] == player_name else 0
        
        opponent_ranks.append(opponent_rank)
        opponent_results.append(result)
    
    avg_opponent_rank = np.nanmean(opponent_ranks) if opponent_ranks else 100
    median_opponent_rank = np.nanmedian(opponent_ranks) if opponent_ranks else 100
    best_opponent_rank = np.nanmin(opponent_ranks) if opponent_ranks else 100
    worst_opponent_rank = np.nanmax(opponent_ranks) if opponent_ranks else 100
    
    top_10_opponents = [r for r in opponent_ranks if r and r <= 10]
    top_10_wins = sum([opponent_results[i] for i in range(len(opponent_ranks)) if opponent_ranks[i] and opponent_ranks[i] <= 10])
    top_10_rate = top_10_wins / len(top_10_opponents) if top_10_opponents else 0
    
    top_50_opponents = [r for r in opponent_ranks if r and r <= 50]
    top_50_wins = sum([opponent_results[i] for i in range(len(opponent_ranks)) if opponent_ranks[i] and opponent_ranks[i] <= 50])
    top_50_rate = top_50_wins / len(top_50_opponents) if top_50_opponents else 0
    
    lower_ranked = [r for r in opponent_ranks if r and r > 50]
    lower_wins = sum([opponent_results[i] for i in range(len(opponent_ranks)) if opponent_ranks[i] and opponent_ranks[i] > 50])
    lower_rate = lower_wins / len(lower_ranked) if lower_ranked else 0
    
    return {
        'avg_opponent_rank': avg_opponent_rank,
        'median_opponent_rank': median_opponent_rank,
        'best_opponent_rank': best_opponent_rank,
        'worst_opponent_rank': worst_opponent_rank,
        'vs_top_10': {'count': len(top_10_opponents), 'wins': top_10_wins, 'rate': top_10_rate},
        'vs_top_50': {'count': len(top_50_opponents), 'wins': top_50_wins, 'rate': top_50_rate},
        'vs_lower_50': {'count': len(lower_ranked), 'wins': lower_wins, 'rate': lower_rate},
    }

def calculate_player_stats_last_30(df, player_name):
    last_30 = get_last_30_matches(df, player_name)
    
    if len(last_30) == 0:
        return None
    
    wins = len(last_30[last_30['Winner'] == player_name])
    losses = len(last_30) - wins
    win_rate = wins / len(last_30) if len(last_30) > 0 else 0
    
    latest = last_30.iloc[-1]
    rank = latest.get('Rank_1', 100) if player_name == latest.get('Player_1') else latest.get('Rank_2', 100)
    points = latest.get('Pts_1', 0) if player_name == latest.get('Player_1') else latest.get('Pts_2', 0)
    odds = latest.get('Odd_1', 1.5) if player_name == latest.get('Player_1') else latest.get('Odd_2', 2.5)
    
    rank = float(rank) if rank else 100
    points = float(points) if points else 0
    odds = float(odds) if odds else 1.5
    
    surface_stats = {}
    if 'Surface' in last_30.columns:
        for surface in last_30['Surface'].unique():
            if pd.notna(surface):
                surface_matches = last_30[last_30['Surface'] == surface]
                surface_wins = len(surface_matches[surface_matches['Winner'] == player_name])
                surface_rate = surface_wins / len(surface_matches) if len(surface_matches) > 0 else 0
                surface_stats[surface] = {
                    'matches': len(surface_matches),
                    'wins': surface_wins,
                    'rate': surface_rate
                }
    
    opponent_strength = calculate_opponent_strength(last_30, player_name)
    
    return {
        'last_30': last_30,
        'rank': rank,
        'points': points,
        'odds': odds,
        'total_matches': len(last_30),
        'wins': wins,
        'losses': losses,
        'win_rate': win_rate,
        'surface_stats': surface_stats,
        'opponent_strength': opponent_strength
    }

def show_home(model_data):
    st.header("🎾 WTA Match Predictor")
    st.markdown("*Improved calibration with accurate game counting*")
    
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Total Matches", len(model_data['df']))
    with col2:
        st.metric("Accuracy", f"{model_data['test_accuracy']:.1%}")
    with col3:
        st.metric("AUC-ROC", f"{model_data['auc_score']:.1%}")
    with col4:
        st.metric("Status", "✓ Optimized")
    
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
        st.subheader("🔄 Cross-Validation")
        st.write(f"Mean CV Score: {np.mean(model_data['cv_scores']):.1%}")
        st.write(f"Std Dev: ±{np.std(model_data['cv_scores']):.1%}")
        st.write("\n**Model Improvements:**")
        st.write("✓ 300 estimators")
        st.write("✓ Isotonic calibration")
        st.write("✓ Log-transformed features")
        st.write("✓ Enhanced regularization")
    
    st.markdown("---")
    st.subheader("📈 Top 15 Features")
    
    top_features = model_data['importance_df'].head(15)
    fig = go.Figure(data=[
        go.Bar(y=top_features['Feature'], x=top_features['Importance'], orientation='h', marker_color='#667eea')
    ])
    fig.update_layout(title="Feature Importance", xaxis_title="Importance", height=500)
    st.plotly_chart(fig, use_container_width=True)

def show_surface_games(model_data):
    st.header("🏆 Expected Games by Surface")
    st.markdown("Predict total games based on surface-specific performance (Last 30 matches)")
    
    st.markdown("---")
    
    df = model_data['df']
    all_players = sorted(list(set(df['Player_1'].unique()) | set(df['Player_2'].unique())))
    
    col_a, col_b = st.columns(2)
    
    with col_a:
        st.subheader("👤 Player A")
        player_a_name = st.selectbox("Select Player A", all_players, key="surf_a")
    
    with col_b:
        st.subheader("👤 Player B")
        player_b_name = st.selectbox("Select Player B", all_players, index=1 if len(all_players) > 1 else 0, key="surf_b")
    
    st.markdown("---")
    
    st.subheader("🏟️ Select Surface")
    surface = st.selectbox("Surface", ["Hard", "Clay", "Grass"], key="surface_select")
    
    st.markdown("---")
    
    if st.button("📊 Calculate Expected Games", use_container_width=True):
        stats_a_surface = calculate_surface_statistics(df, player_a_name, surface)
        stats_b_surface = calculate_surface_statistics(df, player_b_name, surface)
        
        st.markdown("---")
        st.subheader("📈 SURFACE STATISTICS (Last 30 matches on this surface)")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader(f"{player_a_name} on {surface}")
            if stats_a_surface:
                st.write(f"**Matches:** {stats_a_surface['total_matches']}")
                st.write(f"**Wins:** {stats_a_surface['wins']}")
                st.write(f"**Win Rate:** {stats_a_surface['win_rate']:.1%}")
                st.write(f"\n**Game Statistics:**")
                st.write(f"• Avg Games When Wins: {stats_a_surface['avg_games_when_win']:.1f} games")
                st.write(f"• Avg Games When Loses: {stats_a_surface['avg_games_when_loss']:.1f} games")
                st.write(f"• Overall Avg: {stats_a_surface['avg_total_games']:.1f} games")
                st.write(f"• Samples: {stats_a_surface['total_games_samples']} matches with score data")
            else:
                st.warning(f"No matches found for {player_a_name} on {surface} courts")
        
        with col2:
            st.subheader(f"{player_b_name} on {surface}")
            if stats_b_surface:
                st.write(f"**Matches:** {stats_b_surface['total_matches']}")
                st.write(f"**Wins:** {stats_b_surface['wins']}")
                st.write(f"**Win Rate:** {stats_b_surface['win_rate']:.1%}")
                st.write(f"\n**Game Statistics:**")
                st.write(f"• Avg Games When Wins: {stats_b_surface['avg_games_when_win']:.1f} games")
                st.write(f"• Avg Games When Loses: {stats_b_surface['avg_games_when_loss']:.1f} games")
                st.write(f"• Overall Avg: {stats_b_surface['avg_total_games']:.1f} games")
                st.write(f"• Samples: {stats_b_surface['total_games_samples']} matches with score data")
            else:
                st.warning(f"No matches found for {player_b_name} on {surface} courts")
        
        st.markdown("---")
        st.subheader("🎯 EXPECTED GAMES ANALYSIS")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric(f"{player_a_name}", f"{stats_a_surface['expected_games']:.1f} games" if stats_a_surface else "N/A")
            if stats_a_surface:
                st.caption(f"Win Rate: {stats_a_surface['win_rate']:.1%}")
        
        with col2:
            if stats_a_surface and stats_b_surface:
                avg_games = (stats_a_surface['expected_games'] + stats_b_surface['expected_games']) / 2
                st.metric("Average Expected", f"{avg_games:.1f} games")
                if avg_games < 25:
                    st.caption("📊 Likely straight sets")
                elif avg_games < 28:
                    st.caption("📊 Competitive match")
                else:
                    st.caption("📊 Likely to go 3 sets")
        
        with col3:
            st.metric(f"{player_b_name}", f"{stats_b_surface['expected_games']:.1f} games" if stats_b_surface else "N/A")
            if stats_b_surface:
                st.caption(f"Win Rate: {stats_b_surface['win_rate']:.1%}")
        
        st.markdown("---")
        
        # Comparison table
        if stats_a_surface and stats_b_surface:
            st.subheader("📊 Detailed Comparison")
            
            comp_df = pd.DataFrame({
                'Metric': [
                    'Total Matches',
                    'Wins',
                    'Losses',
                    'Win Rate',
                    'Avg Games (Win)',
                    'Avg Games (Loss)',
                    'Expected Games',
                    'Data Samples'
                ],
                player_a_name: [
                    stats_a_surface['total_matches'],
                    stats_a_surface['wins'],
                    stats_a_surface['losses'],
                    f"{stats_a_surface['win_rate']:.1%}",
                    f"{stats_a_surface['avg_games_when_win']:.1f}",
                    f"{stats_a_surface['avg_games_when_loss']:.1f}",
                    f"{stats_a_surface['expected_games']:.1f}",
                    stats_a_surface['total_games_samples']
                ],
                player_b_name: [
                    stats_b_surface['total_matches'],
                    stats_b_surface['wins'],
                    stats_b_surface['losses'],
                    f"{stats_b_surface['win_rate']:.1%}",
                    f"{stats_b_surface['avg_games_when_win']:.1f}",
                    f"{stats_b_surface['avg_games_when_loss']:.1f}",
                    f"{stats_b_surface['expected_games']:.1f}",
                    stats_b_surface['total_games_samples']
                ]
            })
            
            st.dataframe(comp_df, use_container_width=True, hide_index=True)
            
            st.markdown("---")
            st.subheader("💡 HOW GAMES ARE CALCULATED")
            
            st.markdown("""
            **Game Count Formula:**
            - Score '6-4 6-3' = 6+4 + 6+3 = **19 games**
            - Score '7-5 7-6' = 7+5 + 7+6 = **25 games**
            - Score '6-0 6-0' = 6+0 + 6+0 = **12 games**
            
            **Expected Games Formula:**
            - Expected = (Win Rate × Avg Games When Winning) + ((1 - Win Rate) × Avg Games When Losing)
            
            **Example Calculation:**
            - Player A: 70% win rate on hard court
            - Avg games when wins: 24 games (usually 6-x 6-y)
            - Avg games when loses: 18 games (usually loses early)
            - Expected = (0.70 × 24) + (0.30 × 18) = 16.8 + 5.4 = **22.2 games**
            
            **Betting Guide:**
            - **< 22 games**: Likely quick wins (favorite dominant)
            - **22-26 games**: Competitive matches
            - **> 26 games**: Long matches (frequent 3-setters)
            """)
            
            st.markdown("---")
            st.subheader("📈 Visual Comparison")
            
            col1, col2 = st.columns(2)
            
            with col1:
                fig_games = go.Figure(data=[
                    go.Bar(
                        x=[player_a_name, player_b_name],
                        y=[stats_a_surface['expected_games'], stats_b_surface['expected_games']],
                        marker_color=['#667eea', '#764ba2'],
                        text=[f"{stats_a_surface['expected_games']:.1f}", f"{stats_b_surface['expected_games']:.1f}"],
                        textposition='auto'
                    )
                ])
                fig_games.update_layout(
                    title="Expected Games on " + surface,
                    yaxis_title="Games",
                    yaxis=dict(range=[0, 35]),
                    showlegend=False,
                    height=400
                )
                st.plotly_chart(fig_games, use_container_width=True)
            
            with col2:
                fig_wins = go.Figure(data=[
                    go.Bar(
                        x=[player_a_name, player_b_name],
                        y=[stats_a_surface['win_rate'], stats_b_surface['win_rate']],
                        marker_color=['#667eea', '#764ba2'],
                        text=[f"{stats_a_surface['win_rate']:.1%}", f"{stats_b_surface['win_rate']:.1%}"],
                        textposition='auto'
                    )
                ])
                fig_wins.update_layout(
                    title="Win Rate on " + surface,
                    yaxis_title="Win Rate",
                    yaxis=dict(range=[0, 1]),
                    showlegend=False,
                    height=400
                )
                st.plotly_chart(fig_wins, use_container_width=True)

def main():
    st.sidebar.title("🎾 WTA Predictor")
    page = st.sidebar.radio("Page", ["🏠 Home", "🏆 Expected Games by Surface"])
    
    st.sidebar.title("📁 Upload")
    uploaded_file = st.sidebar.file_uploader("CSV", type=['csv'])
    
    if uploaded_file:
        try:
            model_data = load_and_train_model(uploaded_file)
            st.sidebar.success("✓ Model Ready!")
            st.sidebar.info(f"AUC-ROC: {model_data['auc_score']:.1%}")
            
            if page == "🏠 Home":
                show_home(model_data)
            else:
                show_surface_games(model_data)
        except Exception as e:
            st.error(f"Error loading model: {str(e)}")
            st.info("Please check your CSV file format and ensure it has all required columns")
    else:
        st.title("🎾 WTA Predictor")
        st.markdown("### Improved Model with Accurate Game Counting")
        st.info("👈 Upload CSV to start!")
        st.markdown("""
        **Required CSV Columns:**
        - Tournament, Date, Surface, Court, Round
        - Player_1, Player_2, Winner
        - Rank_1, Rank_2, Pts_1, Pts_2
        - Odd_1, Odd_2, Score
        
        **Score Format:** '6-4 6-3' or '7-5 6-4'
        
        **New Features:**
        - Accurate game counting (6-4 6-4 = 20 games)
        - Expected games prediction by surface
        - Improved model calibration (Isotonic)
        - Better accuracy with more features
        """)

if __name__ == "__main__":
    main()

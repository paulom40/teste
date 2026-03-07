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
import warnings
warnings.filterwarnings('ignore')

st.set_page_config(page_title="WTA Predictor", page_icon="🎾", layout="wide")

@st.cache_resource
def load_and_train_model(csv_file):
    df = pd.read_csv(csv_file)
    
    # Convert numeric columns
    for col in ['Rank_1', 'Rank_2', 'Pts_1', 'Pts_2', 'Odd_1', 'Odd_2']:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
    
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
    
    # Points features
    features.append((df['Pts_1'] - df['Pts_2']).fillna(0).values)
    feature_names.append('Points_Differential')
    
    features.append(df['Pts_1'].fillna(0).values)
    feature_names.append('Player_1_Points')
    
    features.append(df['Pts_2'].fillna(0).values)
    feature_names.append('Player_2_Points')
    
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
    
    # Stack and clean
    X = np.column_stack(features)
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
    
    y = df['Player_1_Won'].values
    
    # Verify data
    if len(X) < 50:
        raise ValueError(f"Not enough matches. Need at least 50, got {len(X)}")
    
    if np.any(~np.isfinite(X)):
        raise ValueError("Data contains NaN or infinite values after cleaning")
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Train Gradient Boosting
    gb_model = GradientBoostingClassifier(
        n_estimators=200,
        learning_rate=0.05,
        max_depth=5,
        min_samples_split=10,
        min_samples_leaf=5,
        subsample=0.8,
        random_state=42,
        validation_fraction=0.1,
        n_iter_no_change=10,
        tol=1e-4
    )
    
    gb_model.fit(X_train_scaled, y_train)
    
    # Calibrate the model
    calibrated_model = CalibratedClassifierCV(gb_model, method='sigmoid', cv=5)
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
    """Calculate surface-specific statistics"""
    surface_matches = get_surface_matches(df, player_name, surface)
    
    if len(surface_matches) == 0:
        return None
    
    # Calculate wins/losses
    wins = len(surface_matches[surface_matches['Winner'] == player_name])
    losses = len(surface_matches) - wins
    win_rate = wins / len(surface_matches) if len(surface_matches) > 0 else 0
    
    # Analyze games (sets played)
    games_in_straight_sets = 0  # 2-0
    games_in_three_sets = 0     # 2-1
    straight_set_wins = 0
    three_set_wins = 0
    
    for _, match in surface_matches.iterrows():
        score = str(match.get('Score', ''))
        if not score or pd.isna(score):
            continue
        
        # Parse score to count sets
        try:
            sets = score.split()
            if len(sets) >= 2:
                # Check if straight sets (2-0) or three sets (2-1)
                if len(sets) == 2:
                    games_in_straight_sets += 1
                    if match['Winner'] == player_name:
                        straight_set_wins += 1
                elif len(sets) == 3:
                    games_in_three_sets += 1
                    if match['Winner'] == player_name:
                        three_set_wins += 1
        except:
            continue
    
    # Expected games calculation
    # If player has 60% win rate on hard court with 70% straight sets
    # Expected: 60% * 2 sets (straight) + 40% * 3 sets (loss) = 1.2 + 1.2 = 2.4 avg games
    
    total_games = games_in_straight_sets + games_in_three_sets
    if total_games > 0:
        straight_set_rate = games_in_straight_sets / total_games
        expected_games = (win_rate * 2) + ((1 - win_rate) * 3)
    else:
        straight_set_rate = 0
        expected_games = 2.5  # Default to 2.5 if no data
    
    return {
        'total_matches': len(surface_matches),
        'wins': wins,
        'losses': losses,
        'win_rate': win_rate,
        'games_straight_sets': games_in_straight_sets,
        'games_three_sets': games_in_three_sets,
        'straight_set_wins': straight_set_wins,
        'three_set_wins': three_set_wins,
        'straight_set_rate': straight_set_rate,
        'expected_games': expected_games,
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

def calculate_game_lines(p_a_prob, player_a_name, player_b_name):
    if p_a_prob >= 0.5:
        american_odds_fav = int(-100 / (1/p_a_prob - 1)) if p_a_prob < 1 else -9999
        american_odds_under = int(100 * (1/((1-p_a_prob)) - 1)) if p_a_prob > 0 else 9999
        favorite = player_a_name
        underdog = player_b_name
    else:
        american_odds_fav = int(-100 / (1/(1-p_a_prob) - 1)) if (1-p_a_prob) < 1 else -9999
        american_odds_under = int(100 * (1/(p_a_prob) - 1)) if p_a_prob > 0 else 9999
        favorite = player_b_name
        underdog = player_a_name
    
    spread = abs(p_a_prob - 0.5) * 20
    over_under = 2.5 + (abs(p_a_prob - 0.5) * 2)
    
    return {
        'favorite': favorite,
        'underdog': underdog,
        'spread': spread,
        'american_odds_fav': american_odds_fav,
        'american_odds_under': american_odds_under,
        'over_under': over_under
    }

def show_home(model_data):
    st.header("🎾 WTA Match Predictor & Game Lines")
    st.markdown("*Calibrated predictions with opponent strength analysis*")
    
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Total Matches", len(model_data['df']))
    with col2:
        st.metric("Accuracy", f"{model_data['test_accuracy']:.1%}")
    with col3:
        st.metric("AUC-ROC", f"{model_data['auc_score']:.1%}")
    with col4:
        st.metric("Status", "✓ Calibrated")
    
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
        st.write("\n**Model Quality:**")
        st.write("✓ Gradient Boosting")
        st.write("✓ Sigmoid Calibration")
        st.write("✓ 5-fold CV")
    
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
    st.markdown("Predict expected games based on surface-specific performance (Last 30 matches)")
    
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
        # Get surface statistics for both players
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
                st.write(f"• Straight Sets (2-0): {stats_a_surface['games_straight_sets']} ({stats_a_surface['games_straight_sets'] / (stats_a_surface['games_straight_sets'] + stats_a_surface['games_three_sets']) * 100:.0f}%)")
                st.write(f"• Three Sets (2-1): {stats_a_surface['games_three_sets']}")
                st.write(f"• Won in Straight: {stats_a_surface['straight_set_wins']}")
                st.write(f"• Won in Three: {stats_a_surface['three_set_wins']}")
            else:
                st.warning(f"No matches found for {player_a_name} on {surface} courts")
        
        with col2:
            st.subheader(f"{player_b_name} on {surface}")
            if stats_b_surface:
                st.write(f"**Matches:** {stats_b_surface['total_matches']}")
                st.write(f"**Wins:** {stats_b_surface['wins']}")
                st.write(f"**Win Rate:** {stats_b_surface['win_rate']:.1%}")
                st.write(f"\n**Game Statistics:**")
                st.write(f"• Straight Sets (2-0): {stats_b_surface['games_straight_sets']} ({stats_b_surface['games_straight_sets'] / (stats_b_surface['games_straight_sets'] + stats_b_surface['games_three_sets']) * 100:.0f}%)" if (stats_b_surface['games_straight_sets'] + stats_b_surface['games_three_sets']) > 0 else "• Straight Sets (2-0): 0")
                st.write(f"• Three Sets (2-1): {stats_b_surface['games_three_sets']}")
                st.write(f"• Won in Straight: {stats_b_surface['straight_set_wins']}")
                st.write(f"• Won in Three: {stats_b_surface['three_set_wins']}")
            else:
                st.warning(f"No matches found for {player_b_name} on {surface} courts")
        
        st.markdown("---")
        st.subheader("🎯 EXPECTED GAMES ANALYSIS")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric(f"{player_a_name}", f"{stats_a_surface['expected_games']:.2f} games" if stats_a_surface else "N/A")
            if stats_a_surface:
                st.caption(f"Avg: {stats_a_surface['win_rate']:.1%} win rate")
        
        with col2:
            if stats_a_surface and stats_b_surface:
                avg_games = (stats_a_surface['expected_games'] + stats_b_surface['expected_games']) / 2
                st.metric("Average Expected", f"{avg_games:.2f} games")
                st.caption("Both players combined")
        
        with col3:
            st.metric(f"{player_b_name}", f"{stats_b_surface['expected_games']:.2f} games" if stats_b_surface else "N/A")
            if stats_b_surface:
                st.caption(f"Avg: {stats_b_surface['win_rate']:.1%} win rate")
        
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
                    'Straight Sets',
                    'Three Sets',
                    'Straight Set %',
                    'Expected Games'
                ],
                player_a_name: [
                    stats_a_surface['total_matches'],
                    stats_a_surface['wins'],
                    stats_a_surface['losses'],
                    f"{stats_a_surface['win_rate']:.1%}",
                    stats_a_surface['games_straight_sets'],
                    stats_a_surface['games_three_sets'],
                    f"{stats_a_surface['straight_set_rate']:.1%}",
                    f"{stats_a_surface['expected_games']:.2f}"
                ],
                player_b_name: [
                    stats_b_surface['total_matches'],
                    stats_b_surface['wins'],
                    stats_b_surface['losses'],
                    f"{stats_b_surface['win_rate']:.1%}",
                    stats_b_surface['games_straight_sets'],
                    stats_b_surface['games_three_sets'],
                    f"{stats_b_surface['straight_set_rate']:.1%}",
                    f"{stats_b_surface['expected_games']:.2f}"
                ]
            })
            
            st.dataframe(comp_df, use_container_width=True, hide_index=True)
            
            st.markdown("---")
            st.subheader("💡 INTERPRETATION")
            
            st.write(f"""
            **Expected Games Meaning:**
            
            Expected games is calculated as:
            - If Player Wins: **2 games** (wins in straight sets 2-0)
            - If Player Loses: **3 games** (loses after 3 sets 1-2)
            
            **Formula:**
            Expected Games = (Win Rate × 2) + ((1 - Win Rate) × 3)
            
            **Example:**
            - Player with 60% win rate on hard court
            - Expected = (0.60 × 2) + (0.40 × 3) = 1.2 + 1.2 = **2.4 games**
            
            **Betting Interpretation:**
            - **< 2.3 games**: Favorite likely to win in straight sets
            - **2.3-2.5 games**: Competitive match, some chance of 3 sets
            - **> 2.5 games**: Likely to go to 3 sets
            """)
            
            st.markdown("---")
            st.subheader("📈 Visual Comparison")
            
            col1, col2 = st.columns(2)
            
            with col1:
                # Expected games bar chart
                fig_games = go.Figure(data=[
                    go.Bar(
                        x=[player_a_name, player_b_name],
                        y=[stats_a_surface['expected_games'], stats_b_surface['expected_games']],
                        marker_color=['#667eea', '#764ba2'],
                        text=[f"{stats_a_surface['expected_games']:.2f}", f"{stats_b_surface['expected_games']:.2f}"],
                        textposition='auto'
                    )
                ])
                fig_games.update_layout(
                    title="Expected Games on " + surface,
                    yaxis_title="Games",
                    yaxis=dict(range=[0, 3.5]),
                    showlegend=False,
                    height=400
                )
                st.plotly_chart(fig_games, use_container_width=True)
            
            with col2:
                # Win rate comparison
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
            st.sidebar.success("✓ Ready!")
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
        st.markdown("### Expected Games by Surface Prediction")
        st.info("👈 Upload CSV to start!")
        st.markdown("""
        **Required CSV Columns:**
        - Tournament, Date, Surface, Court, Round
        - Player_1, Player_2, Winner
        - Rank_1, Rank_2, Pts_1, Pts_2
        - Odd_1, Odd_2, Score
        
        **New Feature: Expected Games by Surface**
        - Analyzes last 30 matches on specific surface
        - Predicts average games expected
        - Shows win rates and set distributions
        - Helps with over/under betting
        """)

if __name__ == "__main__":
    main()

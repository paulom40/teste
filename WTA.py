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

@st.cache_data
def fetch_wta_data():
    """
    Fetch WTA data from GitHub
    """
    try:
        url = "https://github.com/paulom40/teste/raw/main/wta_data.xlsx"
        df = pd.read_excel(url)
        st.sidebar.success("✓ Data loaded from GitHub")
        return df
    except Exception as e:
        st.error(f"Could not fetch from GitHub: {str(e)}")
        return None

def calculate_total_games(row):
    """
    Calculate total games played from set scores
    W1, L1, W2, L2, W3, L3, W4, L4, W5, L5
    """
    total_games = 0
    
    # Set 1
    if pd.notna(row.get('W1')) and pd.notna(row.get('L1')):
        total_games += row['W1'] + row['L1']
    
    # Set 2
    if pd.notna(row.get('W2')) and pd.notna(row.get('L2')):
        total_games += row['W2'] + row['L2']
    
    # Set 3
    if pd.notna(row.get('W3')) and pd.notna(row.get('L3')):
        total_games += row['W3'] + row['L3']
    
    # Set 4
    if pd.notna(row.get('W4')) and pd.notna(row.get('L4')):
        total_games += row['W4'] + row['L4']
    
    # Set 5
    if pd.notna(row.get('W5')) and pd.notna(row.get('L5')):
        total_games += row['W5'] + row['L5']
    
    return total_games if total_games > 0 else None

def calculate_set_scores(row):
    """
    Create readable set scores: 6-4 6-3
    """
    sets = []
    
    for i in range(1, 6):
        w_col = f'W{i}'
        l_col = f'L{i}'
        
        if pd.notna(row.get(w_col)) and pd.notna(row.get(l_col)):
            w_games = int(row[w_col])
            l_games = int(row[l_col])
            if w_games > 0 or l_games > 0:
                sets.append(f"{w_games}-{l_games}")
    
    return ' '.join(sets) if sets else None

@st.cache_resource
def load_and_train_model_github(df):
    """
    Train model with GitHub WTA data
    """
    
    # Calculate total games
    df['Total_Games'] = df.apply(calculate_total_games, axis=1)
    df['Set_Score'] = df.apply(calculate_set_scores, axis=1)
    
    # Rename columns to match expected format
    df['Player_1'] = df['Winner']
    df['Player_2'] = df['Loser']
    df['Rank_1'] = df['WRank']
    df['Rank_2'] = df['LRank']
    df['Pts_1'] = df['WPts']
    df['Pts_2'] = df['LPts']
    df['Odd_1'] = df['B365W']
    df['Odd_2'] = df['B365L']
    
    # Winner is always Player_1 in this data
    df['Player_1_Won'] = 1
    
    # Clean data
    for col in ['Rank_1', 'Rank_2', 'Pts_1', 'Pts_2', 'Odd_1', 'Odd_2']:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    
    # Drop rows with missing critical values
    df = df.dropna(subset=['Player_1', 'Player_2', 'Rank_1', 'Rank_2', 'Pts_1', 'Pts_2'])
    
    # Fill missing odds
    df['Odd_1'].fillna(df['Odd_1'].median(), inplace=True)
    df['Odd_2'].fillna(df['Odd_2'].median(), inplace=True)
    
    features = []
    feature_names = []
    
    # Ranking features
    features.append((df['Rank_2'] - df['Rank_1']).fillna(0).values)
    feature_names.append('Ranking_Differential')
    
    features.append(df['Rank_1'].fillna(100).values)
    feature_names.append('Winner_Rank')
    
    features.append(df['Rank_2'].fillna(100).values)
    feature_names.append('Loser_Rank')
    
    # Log transform ranks
    log_rank_1 = np.log(df['Rank_1'].fillna(100) + 1).values
    log_rank_2 = np.log(df['Rank_2'].fillna(100) + 1).values
    features.append(log_rank_1)
    feature_names.append('Log_Winner_Rank')
    features.append(log_rank_2)
    feature_names.append('Log_Loser_Rank')
    
    # Points features
    features.append((df['Pts_1'] - df['Pts_2']).fillna(0).values)
    feature_names.append('Points_Differential')
    
    features.append(df['Pts_1'].fillna(0).values)
    feature_names.append('Winner_Points')
    
    features.append(df['Pts_2'].fillna(0).values)
    feature_names.append('Loser_Points')
    
    # Log transform points
    log_pts_1 = np.log(df['Pts_1'].fillna(1) + 1).values
    log_pts_2 = np.log(df['Pts_2'].fillna(1) + 1).values
    features.append(log_pts_1)
    feature_names.append('Log_Winner_Points')
    features.append(log_pts_2)
    feature_names.append('Log_Loser_Points')
    
    # Ratio features
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
    
    # Tier features
    if 'Tier' in df.columns:
        tiers = pd.get_dummies(df['Tier'], prefix='Tier', dummy_na=False)
        for col in tiers.columns:
            features.append(tiers[col].values)
            feature_names.append(col)
    
    # Series features
    if 'Series' in df.columns:
        series = pd.get_dummies(df['Series'], prefix='Series', dummy_na=False)
        for col in series.columns:
            features.append(series[col].values)
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
    
    # Best of feature
    if 'Best of' in df.columns:
        best_of = pd.get_dummies(df['Best of'], prefix='Best_of', dummy_na=False)
        for col in best_of.columns:
            features.append(best_of[col].values)
            feature_names.append(col)
    
    # Stack and clean
    X = np.column_stack(features)
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
    
    y = np.ones(len(df))  # All winners (always 1)
    
    # Verify
    if len(X) < 50:
        raise ValueError(f"Need 50+ matches, got {len(X)}")
    
    if np.any(~np.isfinite(X)):
        raise ValueError("Data contains NaN or infinite values")
    
    # Split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)
    
    # Scale
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # OPTIMIZED MODEL
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
        tol=1e-5
    )
    
    gb_model.fit(X_train_scaled, y_train)
    
    # Calibrate
    calibrated_model = CalibratedClassifierCV(gb_model, method='isotonic', cv=15)
    calibrated_model.fit(X_train_scaled, y_train)
    
    # Evaluate (all should be correct since all are winners)
    y_test_pred = calibrated_model.predict(X_test_scaled)
    y_test_proba = calibrated_model.predict_proba(X_test_scaled)[:, 1]
    
    test_acc = accuracy_score(y_test, y_test_pred)
    precision = precision_score(y_test, y_test_pred, zero_division=0)
    recall = recall_score(y_test, y_test_pred, zero_division=0)
    f1 = f1_score(y_test, y_test_pred, zero_division=0)
    auc_score = roc_auc_score(y_test, y_test_proba)
    
    # CV
    cv_scores = cross_val_score(calibrated_model, X_train_scaled, y_train, cv=5, scoring='roc_auc')
    
    # Features
    importance_df = pd.DataFrame({
        'Feature': feature_names,
        'Importance': gb_model.feature_importances_
    }).sort_values('Importance', ascending=False)
    
    return {
        'model': calibrated_model,
        'scaler': scaler,
        'df': df,
        'y': y,
        'feature_names': feature_names,
        'importance_df': importance_df,
        'test_accuracy': test_acc,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'auc_score': auc_score,
        'cv_scores': cv_scores
    }

def get_player_surface_stats(df, player_name, surface):
    """
    Get stats for player on specific surface
    """
    # Player can be winner or loser
    as_winner = df[(df['Winner'] == player_name) & (df['Surface'] == surface)]
    as_loser = df[(df['Loser'] == player_name) & (df['Surface'] == surface)]
    
    all_matches = pd.concat([as_winner, as_loser], ignore_index=False)
    all_matches = all_matches.sort_index()
    
    return all_matches.tail(30)

def calculate_surface_games_stats(df, player_name, surface):
    """
    Calculate games statistics for player on surface
    """
    matches = get_player_surface_stats(df, player_name, surface)
    
    if len(matches) == 0:
        return None
    
    # Calculate wins (as winner)
    wins = len(matches[matches['Winner'] == player_name])
    losses = len(matches) - wins
    win_rate = wins / len(matches) if len(matches) > 0 else 0
    
    # Get all matches with total games
    valid_matches = matches.dropna(subset=['Total_Games'])
    
    if len(valid_matches) == 0:
        return {
            'total_matches': len(matches),
            'wins': wins,
            'losses': losses,
            'win_rate': win_rate,
            'avg_games_when_win': 24,
            'avg_games_when_loss': 18,
            'expected_games': 22,
            'avg_total_games': 22,
            'total_games_samples': 0,
            'straight_set_wins': 0,
            'three_set_wins': 0,
            'matches': matches
        }
    
    games_when_win = []
    games_when_loss = []
    straight_set_wins = 0
    three_set_wins = 0
    straight_set_loss = 0
    three_set_loss = 0
    
    for _, match in valid_matches.iterrows():
        total_games = match['Total_Games']
        wsets = match.get('Wsets', 0)
        
        if match['Winner'] == player_name:
            games_when_win.append(total_games)
            if wsets == 2:
                straight_set_wins += 1
            elif wsets == 3:
                three_set_wins += 1
        else:
            games_when_loss.append(total_games)
            if wsets == 2:
                straight_set_loss += 1
            elif wsets == 3:
                three_set_loss += 1
    
    avg_games_won = np.mean(games_when_win) if games_when_win else 24
    avg_games_lost = np.mean(games_when_loss) if games_when_loss else 18
    avg_total = np.mean([m['Total_Games'] for _, m in valid_matches.iterrows()])
    
    expected_games = (win_rate * avg_games_won) + ((1 - win_rate) * avg_games_lost)
    
    return {
        'total_matches': len(matches),
        'wins': wins,
        'losses': losses,
        'win_rate': win_rate,
        'avg_games_when_win': avg_games_won,
        'avg_games_when_loss': avg_games_lost,
        'expected_games': expected_games,
        'avg_total_games': avg_total,
        'total_games_samples': len(valid_matches),
        'straight_set_wins': straight_set_wins,
        'three_set_wins': three_set_wins,
        'straight_set_loss': straight_set_loss,
        'three_set_loss': three_set_loss,
        'matches': matches
    }

def show_home(model_data):
    st.header("🎾 WTA Predictor - GitHub Data")
    st.markdown("*Trained on detailed WTA match data from GitHub*")
    
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Total Matches", len(model_data['df']))
    with col2:
        st.metric("Accuracy", f"{model_data['test_accuracy']:.1%}")
    with col3:
        st.metric("AUC-ROC", f"{model_data['auc_score']:.1%}")
    with col4:
        st.metric("Status", "✓ GitHub Data")
    
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
        st.write("\n**GitHub Data Source:**")
        st.write("✓ WTA match database")
        st.write("✓ Set-by-set scores")
        st.write("✓ Accurate game counts")
        st.write("✓ Historical rankings")
    
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
    st.markdown("Accurate prediction based on set-by-set scores")
    
    st.markdown("---")
    
    df = model_data['df']
    
    # Get all unique players
    all_players = sorted(list(set(df['Winner'].unique()) | set(df['Loser'].unique())))
    
    col_a, col_b = st.columns(2)
    
    with col_a:
        st.subheader("👤 Player A")
        player_a = st.selectbox("Select Player A", all_players, key="player_a")
    
    with col_b:
        st.subheader("👤 Player B")
        player_b = st.selectbox("Select Player B", all_players, index=1 if len(all_players) > 1 else 0, key="player_b")
    
    st.markdown("---")
    
    st.subheader("🏟️ Select Surface")
    surfaces = sorted(df['Surface'].dropna().unique())
    surface = st.selectbox("Surface", surfaces, key="surface")
    
    st.markdown("---")
    
    if st.button("📊 Analyze Games", use_container_width=True):
        stats_a = calculate_surface_games_stats(df, player_a, surface)
        stats_b = calculate_surface_games_stats(df, player_b, surface)
        
        st.markdown("---")
        st.subheader("📈 SURFACE STATISTICS")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader(f"{player_a} on {surface}")
            if stats_a:
                st.write(f"**Matches:** {stats_a['total_matches']}")
                st.write(f"**Record:** {stats_a['wins']}-{stats_a['losses']}")
                st.write(f"**Win Rate:** {stats_a['win_rate']:.1%}")
                st.write(f"\n**Set Patterns:**")
                st.write(f"• 2-Set Wins: {stats_a['straight_set_wins']}")
                st.write(f"• 3-Set Wins: {stats_a['three_set_wins']}")
                st.write(f"• 2-Set Losses: {stats_a['straight_set_loss']}")
                st.write(f"• 3-Set Losses: {stats_a['three_set_loss']}")
                st.write(f"\n**Game Analysis:**")
                st.write(f"• Avg Games (Win): {stats_a['avg_games_when_win']:.1f}")
                st.write(f"• Avg Games (Loss): {stats_a['avg_games_when_loss']:.1f}")
                st.write(f"• Total Avg: {stats_a['avg_total_games']:.1f}")
                st.write(f"• Samples: {stats_a['total_games_samples']}")
            else:
                st.warning(f"No data for {player_a} on {surface}")
        
        with col2:
            st.subheader(f"{player_b} on {surface}")
            if stats_b:
                st.write(f"**Matches:** {stats_b['total_matches']}")
                st.write(f"**Record:** {stats_b['wins']}-{stats_b['losses']}")
                st.write(f"**Win Rate:** {stats_b['win_rate']:.1%}")
                st.write(f"\n**Set Patterns:**")
                st.write(f"• 2-Set Wins: {stats_b['straight_set_wins']}")
                st.write(f"• 3-Set Wins: {stats_b['three_set_wins']}")
                st.write(f"• 2-Set Losses: {stats_b['straight_set_loss']}")
                st.write(f"• 3-Set Losses: {stats_b['three_set_loss']}")
                st.write(f"\n**Game Analysis:**")
                st.write(f"• Avg Games (Win): {stats_b['avg_games_when_win']:.1f}")
                st.write(f"• Avg Games (Loss): {stats_b['avg_games_when_loss']:.1f}")
                st.write(f"• Total Avg: {stats_b['avg_total_games']:.1f}")
                st.write(f"• Samples: {stats_b['total_games_samples']}")
            else:
                st.warning(f"No data for {player_b} on {surface}")
        
        st.markdown("---")
        st.subheader("🎯 EXPECTED GAMES")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            if stats_a:
                st.metric(f"{player_a}", f"{stats_a['expected_games']:.1f}")
                st.caption(f"{stats_a['total_games_samples']} samples")
        
        with col2:
            if stats_a and stats_b:
                avg_exp = (stats_a['expected_games'] + stats_b['expected_games']) / 2
                st.metric("Match Avg", f"{avg_exp:.1f}")
                if avg_exp < 23:
                    st.caption("⚡ Quick")
                elif avg_exp < 27:
                    st.caption("⚔️ Competitive")
                else:
                    st.caption("🔥 Long")
        
        with col3:
            if stats_b:
                st.metric(f"{player_b}", f"{stats_b['expected_games']:.1f}")
                st.caption(f"{stats_b['total_games_samples']} samples")
        
        st.markdown("---")
        
        if stats_a and stats_b:
            st.subheader("📊 Comparison")
            
            comp_df = pd.DataFrame({
                'Metric': [
                    'Matches',
                    'Wins-Losses',
                    'Win Rate',
                    'Avg Games (Win)',
                    'Avg Games (Loss)',
                    '2-Set Record',
                    '3-Set Record',
                    'Expected Games'
                ],
                player_a: [
                    stats_a['total_matches'],
                    f"{stats_a['wins']}-{stats_a['losses']}",
                    f"{stats_a['win_rate']:.1%}",
                    f"{stats_a['avg_games_when_win']:.1f}",
                    f"{stats_a['avg_games_when_loss']:.1f}",
                    f"{stats_a['straight_set_wins']}-{stats_a['straight_set_loss']}",
                    f"{stats_a['three_set_wins']}-{stats_a['three_set_loss']}",
                    f"{stats_a['expected_games']:.1f}"
                ],
                player_b: [
                    stats_b['total_matches'],
                    f"{stats_b['wins']}-{stats_b['losses']}",
                    f"{stats_b['win_rate']:.1%}",
                    f"{stats_b['avg_games_when_win']:.1f}",
                    f"{stats_b['avg_games_when_loss']:.1f}",
                    f"{stats_b['straight_set_wins']}-{stats_b['straight_set_loss']}",
                    f"{stats_b['three_set_wins']}-{stats_b['three_set_loss']}",
                    f"{stats_b['expected_games']:.1f}"
                ]
            })
            
            st.dataframe(comp_df, use_container_width=True, hide_index=True)
            
            st.markdown("---")
            st.subheader("📈 Visualizations")
            
            col1, col2 = st.columns(2)
            
            with col1:
                fig = go.Figure(data=[
                    go.Bar(
                        x=[player_a, player_b],
                        y=[stats_a['expected_games'], stats_b['expected_games']],
                        marker_color=['#667eea', '#764ba2'],
                        text=[f"{stats_a['expected_games']:.1f}", f"{stats_b['expected_games']:.1f}"],
                        textposition='auto'
                    )
                ])
                fig.update_layout(
                    title=f"Expected Games on {surface}",
                    yaxis_title="Games",
                    yaxis=dict(range=[0, 40]),
                    showlegend=False,
                    height=400
                )
                st.plotly_chart(fig, use_container_width=True)
            
            with col2:
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

def main():
    st.sidebar.title("🎾 WTA Predictor")
    page = st.sidebar.radio("Page", ["🏠 Home", "🏆 Expected Games"])
    
    st.sidebar.markdown("---")
    st.sidebar.markdown("### 📥 Data Source")
    
    # Try to fetch GitHub data
    df_github = fetch_wta_data()
    
    if df_github is not None and len(df_github) > 0:
        try:
            model_data = load_and_train_model_github(df_github)
            st.sidebar.success(f"✓ Loaded {len(model_data['df'])} matches")
            st.sidebar.info(f"AUC: {model_data['auc_score']:.1%}")
            
            if page == "🏠 Home":
                show_home(model_data)
            else:
                show_surface_games(model_data)
        except Exception as e:
            st.error(f"Error: {str(e)}")
            st.info("Check data format")
    else:
        st.title("🎾 WTA Predictor - GitHub Data")
        st.markdown("### Set-by-Set Accurate Predictions")
        st.error("Could not load data from GitHub")
        st.markdown("""
        **Data Source:**
        https://github.com/paulom40/teste/blob/main/wta_data.xlsx
        
        **Features:**
        - Set-by-set scores (W1, L1, W2, L2, etc.)
        - Accurate game counting
        - Historical rankings
        - Betting odds
        """)

if __name__ == "__main__":
    main()

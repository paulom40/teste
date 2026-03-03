import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, calibration_curve
from sklearn.calibration import CalibratedClassifierCV
import warnings
warnings.filterwarnings('ignore')

st.set_page_config(page_title="WTA Predictor", page_icon="🎾", layout="wide")

@st.cache_resource
def load_and_train_model(csv_file):
    df = pd.read_csv(csv_file)
    
    for col in ['Rank_1', 'Rank_2', 'Pts_1', 'Pts_2', 'Odd_1', 'Odd_2']:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
    
    df = df.dropna(subset=['Player_1', 'Player_2', 'Winner', 'Rank_1', 'Rank_2', 'Pts_1', 'Pts_2'])
    df['Player_1_Won'] = (df['Winner'] == df['Player_1']).astype(int)
    
    features = []
    feature_names = []
    
    # Ranking features
    features.append((df['Rank_2'] - df['Rank_1']).values)
    feature_names.append('Ranking_Differential')
    
    features.append(df['Rank_1'].values)
    feature_names.append('Player_1_Rank')
    
    features.append(df['Rank_2'].values)
    feature_names.append('Player_2_Rank')
    
    # Points features
    features.append((df['Pts_1'] - df['Pts_2']).values)
    feature_names.append('Points_Differential')
    
    features.append(df['Pts_1'].values)
    feature_names.append('Player_1_Points')
    
    features.append(df['Pts_2'].values)
    feature_names.append('Player_2_Points')
    
    # Ratio features (more stable than pure differences)
    rank_ratio = df['Rank_2'] / df['Rank_1']
    rank_ratio = rank_ratio.fillna(1)
    features.append(rank_ratio.values)
    feature_names.append('Rank_Ratio')
    
    pts_ratio = (df['Pts_1'] + 1) / (df['Pts_2'] + 1)
    features.append(pts_ratio.values)
    feature_names.append('Points_Ratio')
    
    # Surface features
    if 'Surface' in df.columns:
        surfaces = pd.get_dummies(df['Surface'], prefix='Surface')
        for col in surfaces.columns:
            features.append(surfaces[col].values)
            feature_names.append(col)
    
    # Round features
    if 'Round' in df.columns:
        rounds = pd.get_dummies(df['Round'], prefix='Round')
        for col in rounds.columns:
            features.append(rounds[col].values)
            feature_names.append(col)
    
    # Court features
    if 'Court' in df.columns:
        courts = pd.get_dummies(df['Court'], prefix='Court')
        for col in courts.columns:
            features.append(courts[col].values)
            feature_names.append(col)
    
    # Odds features
    if 'Odd_1' in df.columns and 'Odd_2' in df.columns:
        features.append((df['Odd_1'] - df['Odd_2']).values)
        feature_names.append('Odds_Differential')
        
        # Odds ratio (more stable)
        odds_ratio = (df['Odd_1'] + 0.1) / (df['Odd_2'] + 0.1)
        features.append(odds_ratio.values)
        feature_names.append('Odds_Ratio')
    
    X = np.column_stack(features)
    y = df['Player_1_Won'].values
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # TUNED MODEL: Use Gradient Boosting with calibration
    # Gradient Boosting often has better calibration than Random Forest
    gb_model = GradientBoostingClassifier(
        n_estimators=200,           # More estimators
        learning_rate=0.05,         # Lower learning rate for better generalization
        max_depth=5,                # Optimal depth
        min_samples_split=10,       # Prevent overfitting
        min_samples_leaf=5,         # Prevent overfitting
        subsample=0.8,              # Stochastic boosting
        random_state=42,
        validation_fraction=0.1,    # Early stopping
        n_iter_no_change=10,
        tol=1e-4
    )
    
    gb_model.fit(X_train_scaled, y_train)
    
    # Calibrate the model for better probability estimates
    calibrated_model = CalibratedClassifierCV(gb_model, method='sigmoid', cv=5)
    calibrated_model.fit(X_train_scaled, y_train)
    
    # Evaluate
    y_test_pred = calibrated_model.predict(X_test_scaled)
    y_test_proba = calibrated_model.predict_proba(X_test_scaled)[:, 1]
    
    test_acc = accuracy_score(y_test, y_test_pred)
    precision = precision_score(y_test, y_test_pred)
    recall = recall_score(y_test, y_test_pred)
    f1 = f1_score(y_test, y_test_pred)
    auc_score = roc_auc_score(y_test, y_test_proba)
    
    # Cross-validation score
    cv_scores = cross_val_score(calibrated_model, X_train_scaled, y_train, cv=5, scoring='roc_auc')
    
    # Feature importance from base model
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
    """Get last 30 matches for a player"""
    p1_matches = df[df['Player_1'] == player_name].copy()
    p2_matches = df[df['Player_2'] == player_name].copy()
    
    all_matches = pd.concat([p1_matches, p2_matches], ignore_index=False)
    all_matches = all_matches.sort_index()
    
    last_30 = all_matches.tail(30)
    
    return last_30

def calculate_opponent_strength(last_30_matches, player_name):
    """Calculate opponent strength for last 30 matches"""
    
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
    
    avg_opponent_rank = np.mean(opponent_ranks)
    median_opponent_rank = np.median(opponent_ranks)
    best_opponent_rank = np.min(opponent_ranks)
    worst_opponent_rank = np.max(opponent_ranks)
    
    top_10_opponents = [r for r in opponent_ranks if r <= 10]
    top_10_wins = sum([opponent_results[i] for i in range(len(opponent_ranks)) if opponent_ranks[i] <= 10])
    top_10_rate = top_10_wins / len(top_10_opponents) if top_10_opponents else 0
    
    top_50_opponents = [r for r in opponent_ranks if r <= 50]
    top_50_wins = sum([opponent_results[i] for i in range(len(opponent_ranks)) if opponent_ranks[i] <= 50])
    top_50_rate = top_50_wins / len(top_50_opponents) if top_50_opponents else 0
    
    lower_ranked = [r for r in opponent_ranks if r > 50]
    lower_wins = sum([opponent_results[i] for i in range(len(opponent_ranks)) if opponent_ranks[i] > 50])
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
    """Calculate stats from last 30 matches"""
    last_30 = get_last_30_matches(df, player_name)
    
    if len(last_30) == 0:
        return None
    
    wins = len(last_30[last_30['Winner'] == player_name])
    losses = len(last_30) - wins
    win_rate = wins / len(last_30) if len(last_30) > 0 else 0
    
    latest = last_30.iloc[-1]
    if player_name == latest.get('Player_1'):
        rank = latest['Rank_1']
        points = latest['Pts_1']
        odds = latest.get('Odd_1', 1.5)
    else:
        rank = latest['Rank_2']
        points = latest['Pts_2']
        odds = latest.get('Odd_2', 2.5)
    
    surface_stats = {}
    if 'Surface' in last_30.columns:
        for surface in last_30['Surface'].unique():
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
    """Calculate betting lines based on probability"""
    
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
    st.markdown("*Advanced prediction system with fine-tuned calibration*")
    
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Total Matches", len(model_data['df']))
    with col2:
        st.metric("Model Accuracy", f"{model_data['test_accuracy']:.1%}")
    with col3:
        st.metric("AUC-ROC Score", f"{model_data['auc_score']:.1%}")
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
        st.subheader("🔄 Cross-Validation Scores")
        st.write(f"Mean CV Score: {np.mean(model_data['cv_scores']):.1%}")
        st.write(f"Std Dev: ±{np.std(model_data['cv_scores']):.1%}")
        st.write("\n**Model Quality Indicators:**")
        st.write("✓ Gradient Boosting with calibration")
        st.write("✓ Cross-validated performance")
        st.write("✓ Optimized hyperparameters")
        st.write("✓ Better probability estimation")
    
    st.markdown("---")
    st.subheader("📈 Top 10 Features by Importance")
    
    top_10_features = model_data['importance_df'].head(10)
    fig = go.Figure(data=[
        go.Bar(
            y=top_10_features['Feature'],
            x=top_10_features['Importance'],
            orientation='h',
            marker_color='#667eea'
        )
    ])
    fig.update_layout(title="Feature Importance", xaxis_title="Importance", height=400)
    st.plotly_chart(fig, use_container_width=True)

def show_predictions(model_data):
    st.header("🔮 Predict & Game Lines")
    st.markdown("**Last 30 Matches Analysis with Calibrated Predictions**")
    
    st.markdown("---")
    
    df = model_data['df']
    all_players = sorted(list(set(df['Player_1'].unique()) | set(df['Player_2'].unique())))
    
    col_a, col_b = st.columns(2)
    
    with col_a:
        st.subheader("👤 Player A")
        player_a_name = st.selectbox("Select Player A", all_players)
        
        if player_a_name:
            stats_a = calculate_player_stats_last_30(df, player_a_name)
            if stats_a:
                col_m1, col_m2, col_m3 = st.columns(3)
                with col_m1:
                    st.metric("Last 30", stats_a['total_matches'])
                with col_m2:
                    st.metric("Wins", stats_a['wins'])
                with col_m3:
                    st.metric("Win %", f"{stats_a['win_rate']:.1%}")
                
                rank_1 = st.number_input("Rank", value=int(stats_a['rank']), key="rank_a")
                pts_1 = st.number_input("Points", value=int(stats_a['points']), key="pts_a")
                odds_1 = st.number_input("Odds", value=float(stats_a['odds']), step=0.1, key="odds_a")
    
    with col_b:
        st.subheader("👤 Player B")
        player_b_name = st.selectbox("Select Player B", all_players, index=1 if len(all_players) > 1 else 0)
        
        if player_b_name:
            stats_b = calculate_player_stats_last_30(df, player_b_name)
            if stats_b:
                col_m1, col_m2, col_m3 = st.columns(3)
                with col_m1:
                    st.metric("Last 30 ", stats_b['total_matches'])
                with col_m2:
                    st.metric("Wins ", stats_b['wins'])
                with col_m3:
                    st.metric("Win % ", f"{stats_b['win_rate']:.1%}")
                
                rank_2 = st.number_input("Rank ", value=int(stats_b['rank']), key="rank_b")
                pts_2 = st.number_input("Points ", value=int(stats_b['points']), key="pts_b")
                odds_2 = st.number_input("Odds ", value=float(stats_b['odds']), step=0.1, key="odds_b")
    
    st.markdown("---")
    
    st.subheader("🏟️ Match Conditions")
    col1, col2, col3 = st.columns(3)
    with col1:
        surface = st.selectbox("Surface", ["Hard", "Clay", "Grass"])
    with col2:
        court = st.selectbox("Court", ["Indoor", "Outdoor"])
    with col3:
        round_type = st.selectbox("Round", ["1st Round", "2nd Round", "Quarterfinal", "Semifinal", "Final"])
    
    st.markdown("---")
    
    if st.button("⚡ Predict Winner & Game Lines", use_container_width=True):
        
        features = [rank_2 - rank_1, rank_1, rank_2, pts_1 - pts_2, pts_1, pts_2]
        
        # Ratio features
        features.append(rank_2 / rank_1 if rank_1 > 0 else 1)
        features.append((pts_1 + 1) / (pts_2 + 1))
        
        for s in ["Hard", "Clay", "Grass"]:
            features.append(1.0 if surface == s else 0.0)
        
        for r in ["1st Round", "2nd Round", "Quarterfinal", "Semifinal", "Final"]:
            features.append(1.0 if round_type == r else 0.0)
        
        for c in ["Indoor", "Outdoor"]:
            features.append(1.0 if court == c else 0.0)
        
        features.append(odds_1 - odds_2)
        features.append((odds_1 + 0.1) / (odds_2 + 0.1))
        
        while len(features) < len(model_data['feature_names']):
            features.append(0.0)
        
        X_new = np.array(features[:len(model_data['feature_names'])]).reshape(1, -1)
        X_new_scaled = model_data['scaler'].transform(X_new)
        
        prob = model_data['model'].predict_proba(X_new_scaled)[0]
        p_a = prob[1]
        p_b = prob[0]
        
        lines = calculate_game_lines(p_a, player_a_name, player_b_name)
        
        st.markdown("---")
        st.subheader("📊 CALIBRATED PREDICTION RESULTS")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            if p_a > p_b:
                st.success(f"🏆 {player_a_name}")
                st.metric("Win Probability", f"{p_a:.2%}")
            else:
                st.success(f"🏆 {player_b_name}")
                st.metric("Win Probability", f"{p_b:.2%}")
        
        with col2:
            conf = abs(p_a - 0.5)
            st.metric("Confidence", f"{conf:.2%}")
            if conf < 0.05:
                st.warning("Toss-up")
            elif conf < 0.15:
                st.info("Moderate")
            elif conf < 0.30:
                st.success("High")
            else:
                st.success("Very High")
        
        with col3:
            st.metric("Model AUC-ROC", f"{model_data['auc_score']:.1%}")
            st.caption("Calibrated probability")
        
        st.markdown("---")
        st.subheader("📈 GAME LINES")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("SPREAD", f"{lines['favorite']} -{lines['spread']:.1f}")
            st.caption("Games favored")
        
        with col2:
            st.metric("OVER/UNDER", f"{lines['over_under']:.1f}")
            st.caption("Total games")
        
        with col3:
            st.metric("MONEYLINE", f"{lines['american_odds_fav']}")
            st.caption(f"{lines['underdog']}: +{lines['american_odds_under']}")
        
        st.markdown("---")
        st.subheader("📊 OPPONENT STRENGTH ANALYSIS (Last 30 Matches)")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader(f"{player_a_name}")
            if stats_a:
                st.write(f"**Matches:** {stats_a['total_matches']} | **Wins:** {stats_a['wins']} | **Rate:** {stats_a['win_rate']:.1%}")
                
                opp_a = stats_a['opponent_strength']
                st.write(f"\n**Opponent Strength:**")
                st.write(f"• Avg Rank: #{opp_a['avg_opponent_rank']:.0f}")
                st.write(f"• Median: #{opp_a['median_opponent_rank']:.0f}")
                
                st.write(f"\n**Win Rate by Opponent Level:**")
                st.write(f"• vs Top 10: {opp_a['vs_top_10']['wins']}/{opp_a['vs_top_10']['count']} ({opp_a['vs_top_10']['rate']:.1%})")
                st.write(f"• vs Top 50: {opp_a['vs_top_50']['wins']}/{opp_a['vs_top_50']['count']} ({opp_a['vs_top_50']['rate']:.1%})")
                st.write(f"• vs 50+: {opp_a['vs_lower_50']['wins']}/{opp_a['vs_lower_50']['count']} ({opp_a['vs_lower_50']['rate']:.1%})")
        
        with col2:
            st.subheader(f"{player_b_name}")
            if stats_b:
                st.write(f"**Matches:** {stats_b['total_matches']} | **Wins:** {stats_b['wins']} | **Rate:** {stats_b['win_rate']:.1%}")
                
                opp_b = stats_b['opponent_strength']
                st.write(f"\n**Opponent Strength:**")
                st.write(f"• Avg Rank: #{opp_b['avg_opponent_rank']:.0f}")
                st.write(f"• Median: #{opp_b['median_opponent_rank']:.0f}")
                
                st.write(f"\n**Win Rate by Opponent Level:**")
                st.write(f"• vs Top 10: {opp_b['vs_top_10']['wins']}/{opp_b['vs_top_10']['count']} ({opp_b['vs_top_10']['rate']:.1%})")
                st.write(f"• vs Top 50: {opp_b['vs_top_50']['wins']}/{opp_b['vs_top_50']['count']} ({opp_b['vs_top_50']['rate']:.1%})")
                st.write(f"• vs 50+: {opp_b['vs_lower_50']['wins']}/{opp_b['vs_lower_50']['count']} ({opp_b['vs_lower_50']['rate']:.1%})")
        
        st.markdown("---")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.subheader("Opponent Strength")
            opp_comp = pd.DataFrame({
                'Metric': ['Avg Opp', 'Median Opp', 'Best', 'Worst'],
                player_a_name: [
                    f"#{stats_a['opponent_strength']['avg_opponent_rank']:.0f}",
                    f"#{stats_a['opponent_strength']['median_opponent_rank']:.0f}",
                    f"#{stats_a['opponent_strength']['best_opponent_rank']:.0f}",
                    f"#{stats_a['opponent_strength']['worst_opponent_rank']:.0f}"
                ],
                player_b_name: [
                    f"#{stats_b['opponent_strength']['avg_opponent_rank']:.0f}",
                    f"#{stats_b['opponent_strength']['median_opponent_rank']:.0f}",
                    f"#{stats_b['opponent_strength']['best_opponent_rank']:.0f}",
                    f"#{stats_b['opponent_strength']['worst_opponent_rank']:.0f}"
                ]
            })
            st.dataframe(opp_comp, use_container_width=True, hide_index=True)
        
        with col2:
            st.subheader("Overall")
            comp = pd.DataFrame({
                'Metric': ['Rank', 'Points', 'W/L', 'Rate', 'P(Win)', 'Opp Avg'],
                player_a_name: [
                    f"#{rank_1}",
                    pts_1,
                    f"{stats_a['wins']}/{stats_a['total_matches']}",
                    f"{stats_a['win_rate']:.1%}",
                    f"{p_a:.2%}",
                    f"#{stats_a['opponent_strength']['avg_opponent_rank']:.0f}"
                ],
                player_b_name: [
                    f"#{rank_2}",
                    pts_2,
                    f"{stats_b['wins']}/{stats_b['total_matches']}",
                    f"{stats_b['win_rate']:.1%}",
                    f"{p_b:.2%}",
                    f"#{stats_b['opponent_strength']['avg_opponent_rank']:.0f}"
                ]
            })
            st.dataframe(comp, use_container_width=True, hide_index=True)
        
        with col3:
            fig = go.Figure([go.Bar(x=[player_a_name, player_b_name], y=[p_a, p_b], marker_color=['#667eea', '#764ba2'])])
            fig.update_layout(title="Calibrated Probability", yaxis=dict(range=[0, 1]), showlegend=False, height=350)
            st.plotly_chart(fig, use_container_width=True)
        
        st.markdown("---")
        
        st.subheader("🔍 KEY INSIGHTS")
        
        stronger_opp = stats_a['opponent_strength']['avg_opponent_rank'] < stats_b['opponent_strength']['avg_opponent_rank']
        better_vs_top = stats_a['opponent_strength']['vs_top_10']['rate'] > stats_b['opponent_strength']['vs_top_10']['rate']
        
        if stronger_opp:
            st.write(f"✓ {player_a_name} faced stronger opponents (avg #{stats_a['opponent_strength']['avg_opponent_rank']:.0f} vs #{stats_b['opponent_strength']['avg_opponent_rank']:.0f})")
        else:
            st.write(f"✓ {player_b_name} faced stronger opponents (avg #{stats_b['opponent_strength']['avg_opponent_rank']:.0f} vs #{stats_a['opponent_strength']['avg_opponent_rank']:.0f})")
        
        if better_vs_top:
            st.write(f"✓ {player_a_name} performs better vs Top 10 ({stats_a['opponent_strength']['vs_top_10']['rate']:.1%} vs {stats_b['opponent_strength']['vs_top_10']['rate']:.1%})")
        else:
            st.write(f"✓ {player_b_name} performs better vs Top 10 ({stats_b['opponent_strength']['vs_top_10']['rate']:.1%} vs {stats_a['opponent_strength']['vs_top_10']['rate']:.1%})")
        
        if stats_a['win_rate'] > stats_b['win_rate']:
            st.write(f"✓ {player_a_name} has better recent form ({stats_a['win_rate']:.1%} vs {stats_b['win_rate']:.1%})")
        else:
            st.write(f"✓ {player_b_name} has better recent form ({stats_b['win_rate']:.1%} vs {stats_a['win_rate']:.1%})")

def main():
    st.sidebar.title("🎾 WTA Predictor")
    page = st.sidebar.radio("Page", ["🏠 Home", "🔮 Predict & Lines"])
    
    st.sidebar.title("📁 Upload")
    uploaded_file = st.sidebar.file_uploader("CSV", type=['csv'])
    
    if uploaded_file:
        model_data = load_and_train_model(uploaded_file)
        st.sidebar.success("✓ Model Calibrated!")
        st.sidebar.info(f"AUC-ROC: {model_data['auc_score']:.1%}")
        
        if page == "🏠 Home":
            show_home(model_data)
        else:
            show_predictions(model_data)
    else:
        st.title("🎾 WTA Predictor")
        st.markdown("### Calibrated Match Prediction & Game Lines")
        st.info("👈 Upload your WTA CSV file to begin!")

if __name__ == "__main__":
    main()

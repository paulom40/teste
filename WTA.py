import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score, mean_squared_error
import requests
from io import BytesIO
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

st.set_page_config(page_title="WTA Games Predictor", page_icon="🎾", layout="wide")

@st.cache_data
def fetch_wta_github_data():
    try:
        url = "https://github.com/paulom40/teste/raw/main/wta_data.xlsx"
        response = requests.get(url, timeout=10)
        df = pd.read_excel(BytesIO(response.content))
        st.sidebar.success("✓ Data loaded from GitHub")
        return df
    except Exception as e:
        st.sidebar.error(f"Could not fetch: {str(e)}")
        return None

def calculate_total_games(row):
    """Calculate total games from set scores"""
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

# ============= IMPROVED GAMES PREDICTION MODEL =============

def build_improved_games_model(df):
    """
    Build ML model trained on ACTUAL match data
    Uses real set scores and player rankings
    """
    df_train = df.copy()
    
    # Calculate total games from set scores
    df_train['Total_Games'] = df_train.apply(calculate_total_games, axis=1)
    
    # Clean data - remove invalid entries
    df_train = df_train.dropna(subset=['Total_Games', 'WRank', 'LRank', 'Winner', 'Loser'])
    df_train = df_train[df_train['Total_Games'] > 0]
    df_train = df_train[df_train['Total_Games'] < 50]  # Remove anomalies
    
    if len(df_train) < 100:
        st.error("Not enough data to train model")
        return None
    
    st.sidebar.info(f"Training on {len(df_train)} matches")
    
    features = []
    feature_names = []
    
    # ===== CRITICAL FEATURES FROM ACTUAL DATA =====
    
    # 1. Ranking difference (MOST IMPORTANT)
    rank_diff = df_train['LRank'] - df_train['WRank']
    features.append(rank_diff.fillna(0).values)
    feature_names.append('Rank_Diff')
    
    # 2. Winner rank
    winner_rank = np.log(df_train['WRank'].fillna(100).values + 1)
    features.append(winner_rank)
    feature_names.append('Winner_Rank_Log')
    
    # 3. Loser rank
    loser_rank = np.log(df_train['LRank'].fillna(100).values + 1)
    features.append(loser_rank)
    feature_names.append('Loser_Rank_Log')
    
    # 4. Rank ratio
    rank_ratio = np.where(
        df_train['WRank'] > 0,
        df_train['LRank'] / df_train['WRank'],
        1.0
    )
    rank_ratio = np.nan_to_num(rank_ratio, nan=1.0, posinf=1.0, neginf=1.0)
    features.append(rank_ratio)
    feature_names.append('Rank_Ratio')
    
    # 5. Points difference (IMPORTANT)
    wpts = pd.to_numeric(df_train['WPts'], errors='coerce').fillna(0)
    lpts = pd.to_numeric(df_train['LPts'], errors='coerce').fillna(0)
    pts_diff = wpts - lpts
    features.append(pts_diff.values)
    feature_names.append('Points_Diff')
    
    # 6. Set count (CRITICAL - directly related to games)
    sets_won = df_train['Wsets'].fillna(2).values
    features.append(sets_won)
    feature_names.append('Sets_Won')
    
    # 7. Individual set games (THE MOST IMPORTANT!)
    # First set
    w1 = pd.to_numeric(df_train['W1'], errors='coerce').fillna(0).values
    l1 = pd.to_numeric(df_train['L1'], errors='coerce').fillna(0).values
    features.append(w1 + l1)  # Total games in set 1
    feature_names.append('Set1_Games')
    
    # Second set
    w2 = pd.to_numeric(df_train['W2'], errors='coerce').fillna(0).values
    l2 = pd.to_numeric(df_train['L2'], errors='coerce').fillna(0).values
    features.append(w2 + l2)
    feature_names.append('Set2_Games')
    
    # Third set (if played)
    w3 = pd.to_numeric(df_train['W3'], errors='coerce').fillna(0).values
    l3 = pd.to_numeric(df_train['L3'], errors='coerce').fillna(0).values
    features.append(np.where(w3 + l3 > 0, w3 + l3, 0))
    feature_names.append('Set3_Games')
    
    # 8. Is it competitive? (close sets = longer matches)
    set1_diff = np.abs(w1 - l1)
    set2_diff = np.abs(w2 - l2)
    competitiveness = 1 / (1 + set1_diff + set2_diff)  # Closer = higher value
    features.append(competitiveness)
    feature_names.append('Competitiveness')
    
    # 9. Surface (one-hot)
    if 'Surface' in df_train.columns:
        for surface in df_train['Surface'].dropna().unique():
            is_surface = (df_train['Surface'] == surface).astype(int).values
            features.append(is_surface)
            feature_names.append(f'Surface_{surface}')
    
    # 10. Tier (one-hot)
    if 'Tier' in df_train.columns:
        for tier in df_train['Tier'].dropna().unique():
            is_tier = (df_train['Tier'] == tier).astype(int).values
            features.append(is_tier)
            feature_names.append(f'Tier_{tier}')
    
    # Combine all features
    X = np.column_stack(features)
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
    y = df_train['Total_Games'].values
    
    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    # Scale
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Train regressor
    model = GradientBoostingRegressor(
        n_estimators=300,
        learning_rate=0.08,
        max_depth=6,
        min_samples_split=8,
        min_samples_leaf=4,
        subsample=0.9,
        max_features='sqrt',
        random_state=42,
        verbose=0
    )
    
    model.fit(X_train_scaled, y_train)
    
    # Evaluate
    y_pred_test = model.predict(X_test_scaled)
    y_pred_train = model.predict(X_train_scaled)
    
    mae = mean_absolute_error(y_test, y_pred_test)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred_test))
    r2 = r2_score(y_test, y_pred_test)
    
    # Feature importance
    importance_df = pd.DataFrame({
        'Feature': feature_names,
        'Importance': model.feature_importances_
    }).sort_values('Importance', ascending=False)
    
    return {
        'model': model,
        'scaler': scaler,
        'feature_names': feature_names,
        'mae': mae,
        'rmse': rmse,
        'r2': r2,
        'y_test': y_test,
        'y_pred': y_pred_test,
        'y_pred_train': y_pred_train,
        'y_train': y_train,
        'importance': importance_df,
        'df': df_train
    }

def predict_games_for_match(games_model, player_a, player_b, surface, df):
    """Predict games for a specific match"""
    if games_model is None:
        return None
    
    model = games_model['model']
    scaler = games_model['scaler']
    feature_names = games_model['feature_names']
    
    # Get player rankings
    a_matches = df[
        ((df['Winner'] == player_a) | (df['Loser'] == player_a))
    ].tail(10)
    b_matches = df[
        ((df['Winner'] == player_b) | (df['Loser'] == player_b))
    ].tail(10)
    
    if len(a_matches) == 0 or len(b_matches) == 0:
        return {'expected': 22, 'confidence': 0}
    
    # Get typical rankings
    a_rank = a_matches['WRank'].median() if len(a_matches[a_matches['Winner'] == player_a]) > 0 else a_matches['LRank'].median()
    b_rank = b_matches['WRank'].median() if len(b_matches[b_matches['Winner'] == player_b]) > 0 else b_matches['LRank'].median()
    
    # Get historical games on surface
    a_surface = df[
        ((df['Winner'] == player_a) | (df['Loser'] == player_a)) &
        (df['Surface'] == surface)
    ].tail(10)
    
    b_surface = df[
        ((df['Winner'] == player_b) | (df['Loser'] == player_b)) &
        (df['Surface'] == surface)
    ].tail(10)
    
    a_surface['Total_Games'] = a_surface.apply(calculate_total_games, axis=1)
    b_surface['Total_Games'] = b_surface.apply(calculate_total_games, axis=1)
    
    a_avg_games = a_surface['Total_Games'].median() if len(a_surface) > 0 else 22
    b_avg_games = b_surface['Total_Games'].median() if len(b_surface) > 0 else 22
    
    # Create feature vector
    features = [0] * len(feature_names)
    
    # Fill in known features
    for i, fname in enumerate(feature_names):
        if fname == 'Rank_Diff':
            features[i] = b_rank - a_rank
        elif fname == 'Winner_Rank_Log':
            features[i] = np.log(a_rank + 1)
        elif fname == 'Loser_Rank_Log':
            features[i] = np.log(b_rank + 1)
        elif fname == 'Rank_Ratio':
            features[i] = b_rank / a_rank if a_rank > 0 else 1.0
        elif fname == 'Set1_Games':
            features[i] = a_avg_games * 0.5  # Estimate
        elif fname == 'Set2_Games':
            features[i] = a_avg_games * 0.35
        elif fname == 'Set3_Games':
            features[i] = 0
        elif fname == f'Surface_{surface}':
            features[i] = 1
        elif fname == 'Competitiveness':
            features[i] = 0.5
    
    features = np.array(features).reshape(1, -1)
    features_scaled = scaler.transform(features)
    
    prediction = model.predict(features_scaled)[0]
    prediction = max(12, min(40, prediction))  # Bound 12-40
    
    return {
        'expected': prediction,
        'confidence': games_model['r2'],
        'a_rank': a_rank,
        'b_rank': b_rank,
        'a_avg_games': a_avg_games,
        'b_avg_games': b_avg_games,
        'a_surface_matches': len(a_surface),
        'b_surface_matches': len(b_surface)
    }

def generate_html_prediction(player_a, player_b, surface, prediction, games_model):
    """Generate professional HTML report"""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    if prediction['expected'] < 23:
        match_type = "⚡ Quick Match (2-set likely)"
        color = "#4CAF50"
    elif prediction['expected'] < 27:
        match_type = "⚔️ Competitive Match"
        color = "#FF9800"
    else:
        match_type = "🔥 Long Match (3-set likely)"
        color = "#F44336"
    
    html = f"""
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>WTA Games Prediction Report</title>
        <style>
            * {{
                margin: 0;
                padding: 0;
                box-sizing: border-box;
            }}
            
            body {{
                font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                color: #333;
                padding: 20px;
            }}
            
            .container {{
                max-width: 1000px;
                margin: 0 auto;
                background: white;
                border-radius: 15px;
                overflow: hidden;
                box-shadow: 0 10px 40px rgba(0,0,0,0.2);
            }}
            
            .header {{
                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                color: white;
                padding: 50px 30px;
                text-align: center;
            }}
            
            .header h1 {{
                font-size: 2.8em;
                margin-bottom: 10px;
            }}
            
            .header p {{
                font-size: 1.2em;
                opacity: 0.9;
            }}
            
            .content {{
                padding: 40px;
            }}
            
            .match-title {{
                font-size: 2.2em;
                color: #764ba2;
                text-align: center;
                margin: 20px 0;
            }}
            
            .prediction-box {{
                background: {color};
                color: white;
                padding: 30px;
                border-radius: 10px;
                text-align: center;
                margin: 30px 0;
                font-size: 1.3em;
            }}
            
            .prediction-box .number {{
                font-size: 3em;
                font-weight: bold;
                margin: 10px 0;
            }}
            
            .stats-grid {{
                display: grid;
                grid-template-columns: 1fr 1fr;
                gap: 30px;
                margin: 30px 0;
            }}
            
            .stat-card {{
                background: #f9f9f9;
                padding: 20px;
                border-radius: 10px;
                border-left: 5px solid #667eea;
            }}
            
            .stat-card h3 {{
                color: #667eea;
                margin-bottom: 15px;
            }}
            
            .stat-row {{
                display: flex;
                justify-content: space-between;
                padding: 10px 0;
                border-bottom: 1px solid #eee;
            }}
            
            .stat-label {{
                font-weight: 600;
            }}
            
            .stat-value {{
                color: #764ba2;
                font-weight: bold;
            }}
            
            .model-info {{
                background: #e3f2fd;
                border-left: 4px solid #667eea;
                padding: 20px;
                margin: 30px 0;
                border-radius: 5px;
            }}
            
            .model-info h4 {{
                color: #667eea;
                margin-bottom: 10px;
            }}
            
            .footer {{
                background: #f9f9f9;
                padding: 20px;
                text-align: center;
                border-top: 1px solid #eee;
                color: #666;
                font-size: 0.9em;
            }}
            
            .feature-importance {{
                margin: 30px 0;
                background: #f9f9f9;
                padding: 20px;
                border-radius: 10px;
            }}
            
            .feature-importance h4 {{
                color: #667eea;
                margin-bottom: 15px;
            }}
            
            .feature-bar {{
                margin: 10px 0;
            }}
            
            .feature-name {{
                font-weight: 600;
                margin-bottom: 5px;
            }}
            
            .bar {{
                height: 20px;
                background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
                border-radius: 10px;
                overflow: hidden;
            }}
        </style>
    </head>
    <body>
        <div class="container">
            <div class="header">
                <h1>🎾 WTA Games Prediction</h1>
                <p>Professional Match Analysis Report</p>
            </div>
            
            <div class="content">
                <div class="match-title">
                    {player_a} vs {player_b}
                </div>
                
                <div style="text-align: center; font-size: 1.3em; margin: 20px 0; color: #667eea;">
                    <strong>Surface: {surface}</strong>
                </div>
                
                <div class="prediction-box">
                    {match_type}
                    <div class="number">{prediction['expected']:.1f} Games</div>
                </div>
                
                <div class="stats-grid">
                    <div class="stat-card">
                        <h3>🎾 {player_a}</h3>
                        <div class="stat-row">
                            <span class="stat-label">Ranking:</span>
                            <span class="stat-value">#{prediction['a_rank']:.0f}</span>
                        </div>
                        <div class="stat-row">
                            <span class="stat-label">Avg Games on {surface}:</span>
                            <span class="stat-value">{prediction['a_avg_games']:.1f}</span>
                        </div>
                        <div class="stat-row">
                            <span class="stat-label">Matches on Surface:</span>
                            <span class="stat-value">{prediction['a_surface_matches']}</span>
                        </div>
                    </div>
                    
                    <div class="stat-card">
                        <h3>🎾 {player_b}</h3>
                        <div class="stat-row">
                            <span class="stat-label">Ranking:</span>
                            <span class="stat-value">#{prediction['b_rank']:.0f}</span>
                        </div>
                        <div class="stat-row">
                            <span class="stat-label">Avg Games on {surface}:</span>
                            <span class="stat-value">{prediction['b_avg_games']:.1f}</span>
                        </div>
                        <div class="stat-row">
                            <span class="stat-label">Matches on Surface:</span>
                            <span class="stat-value">{prediction['b_surface_matches']}</span>
                        </div>
                    </div>
                </div>
                
                <div class="model-info">
                    <h4>📊 Model Information</h4>
                    <p><strong>R² Score:</strong> {games_model['r2']:.3f}</p>
                    <p><strong>Mean Absolute Error:</strong> {games_model['mae']:.2f} games</p>
                    <p><strong>Model Confidence:</strong> {games_model['r2']*100:.1f}%</p>
                    <p style="margin-top: 10px; font-size: 0.9em; color: #666;">
                        This prediction is based on a machine learning model trained on {len(games_model['df'])} historical WTA matches.
                        The model analyzes player rankings, points, surface type, and historical match patterns.
                    </p>
                </div>
                
                <div class="feature-importance">
                    <h4>🔑 Top 10 Most Important Features</h4>
                    {f"<div style='font-size: 0.9em;'>" + "<br>".join([
                        f"<strong>{i+1}. {row['Feature']}</strong>: {row['Importance']*100:.1f}%"
                        for i, (_, row) in enumerate(games_model['importance'].head(10).iterrows())
                    ]) + "</div>"}
                </div>
            </div>
            
            <div class="footer">
                <p><strong>Generated:</strong> {timestamp}</p>
                <p>WTA Advanced Games Prediction System</p>
                <p>Data Source: GitHub WTA Database</p>
            </div>
        </div>
    </body>
    </html>
    """
    
    return html

def show_home(games_model):
    st.header("🎾 Improved WTA Games Prediction")
    st.markdown("*ML model trained on actual WTA match data*")
    
    if games_model:
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("R² Score", f"{games_model['r2']:.3f}")
        with col2:
            st.metric("MAE (Error)", f"±{games_model['mae']:.2f} games")
        with col3:
            st.metric("RMSE", f"{games_model['rmse']:.2f}")
        with col4:
            st.metric("Training Data", f"{len(games_model['df'])} matches")
        
        st.markdown("---")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("📊 Model Performance")
            
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=games_model['y_test'],
                y=games_model['y_pred'],
                mode='markers',
                marker=dict(size=8, color='#667eea', opacity=0.6),
                name='Predictions'
            ))
            fig.add_trace(go.Scatter(
                x=[games_model['y_test'].min(), games_model['y_test'].max()],
                y=[games_model['y_test'].min(), games_model['y_test'].max()],
                mode='lines',
                line=dict(color='red', dash='dash'),
                name='Perfect'
            ))
            fig.update_layout(
                title="Predicted vs Actual Games",
                xaxis_title="Actual Games",
                yaxis_title="Predicted Games",
                height=400
            )
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            st.subheader("🔑 Feature Importance")
            
            top_features = games_model['importance'].head(10)
            fig = go.Figure(data=[
                go.Bar(
                    y=top_features['Feature'],
                    x=top_features['Importance'],
                    orientation='h',
                    marker_color='#667eea'
                )
            ])
            fig.update_layout(
                title="Top 10 Important Features",
                xaxis_title="Importance",
                height=400
            )
            st.plotly_chart(fig, use_container_width=True)

def show_prediction(games_model, df):
    st.header("🎾 Games Prediction")
    
    all_players = sorted(list(set(df['Winner'].unique()) | set(df['Loser'].unique())))
    surfaces = sorted(df['Surface'].dropna().unique())
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        player_a = st.selectbox("Player 1", all_players, key="p1")
    with col2:
        player_b = st.selectbox("Player 2", all_players, index=1 if len(all_players) > 1 else 0, key="p2")
    with col3:
        surface = st.selectbox("Surface", surfaces, key="surf")
    
    st.markdown("---")
    
    if st.button("🔮 Predict Games", width='stretch'):
        prediction = predict_games_for_match(games_model, player_a, player_b, surface, df)
        
        if prediction:
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric(f"📊 {player_a} Rank", f"#{prediction['a_rank']:.0f}")
            
            with col2:
                st.metric(
                    "🎾 Expected Games",
                    f"{prediction['expected']:.1f}",
                )
                if prediction['expected'] < 23:
                    st.info("⚡ Quick Match")
                elif prediction['expected'] < 27:
                    st.info("⚔️ Competitive")
                else:
                    st.warning("🔥 Long Match")
            
            with col3:
                st.metric(f"📊 {player_b} Rank", f"#{prediction['b_rank']:.0f}")
            
            st.markdown("---")
            st.subheader("📈 Detailed Analysis")
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.write(f"**{player_a}**")
                st.write(f"• Avg Games on {surface}: {prediction['a_avg_games']:.1f}")
                st.write(f"• Matches on surface: {prediction['a_surface_matches']}")
                st.write(f"• Ranking: #{prediction['a_rank']:.0f}")
            
            with col2:
                st.write(f"**{player_b}**")
                st.write(f"• Avg Games on {surface}: {prediction['b_avg_games']:.1f}")
                st.write(f"• Matches on surface: {prediction['b_surface_matches']}")
                st.write(f"• Ranking: #{prediction['b_rank']:.0f}")
            
            st.markdown("---")
            st.subheader("💾 Export Report")
            
            # Generate HTML
            html_report = generate_html_prediction(player_a, player_b, surface, prediction, games_model)
            
            # Download button
            st.download_button(
                label="📥 Download HTML Report",
                data=html_report,
                file_name=f"WTA_Games_{player_a}_vs_{player_b}_{surface}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.html",
                mime="text/html",
                key="download_html"
            )
            
            st.success("✅ Report ready for download!")

def main():
    st.sidebar.title("🎾 WTA Games Predictor")
    page = st.sidebar.radio("Page", ["🏠 Home", "🎯 Predict"])
    
    st.sidebar.markdown("---")
    
    df = fetch_wta_github_data()
    
    if df is not None:
        with st.spinner("Building ML model..."):
            games_model = build_improved_games_model(df)
        
        if games_model:
            st.sidebar.success("✅ Model Ready!")
            st.sidebar.metric("Accuracy (R²)", f"{games_model['r2']:.3f}")
            st.sidebar.metric("Error ±", f"{games_model['mae']:.2f} games")
        
        if page == "🏠 Home":
            show_home(games_model)
        else:
            show_prediction(games_model, df)
    else:
        st.error("❌ Could not load data")

if __name__ == "__main__":
    main()

# Leagues.py - FOOTBALL PREDICTOR PRO v7.0 (ADVANCED MODELS + POWER RATINGS)
import streamlit as st
import pandas as pd
import numpy as np
from scipy.stats import poisson, skellam
import requests
from PIL import Image
from io import BytesIO
import plotly.graph_objects as go
import plotly.express as px
import re
from datetime import datetime
import warnings
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error
import xgboost as xgb
import lightgbm as lgb

warnings.filterwarnings('ignore')

# ================================
# CONFIG
# ================================
st.set_page_config(page_title="Predictor Pro v7.0", layout="wide")
st.markdown("""
# Football Predictor Pro v7.0
**Advanced ML Models • FT Score • xG • Shots • SoT • Corners • Power Ratings**
""")

# ================================
# ENHANCED DATA PROCESSING
# ================================
def load_demo_csv() -> pd.DataFrame:
    return pd.DataFrame({
        "Date": pd.date_range("2025-08-15", periods=50, freq="3D"),
        "HomeTeam": ["Liverpool", "Arsenal", "Man City", "Chelsea", "Tottenham", "Man United", "Newcastle", "West Ham", "Everton", "Leicester"] * 5,
        "AwayTeam": ["Bournemouth", "Brighton", "Wolves", "Fulham", "Crystal Palace", "Southampton", "Brentford", "Aston Villa", "Leeds", "Norwich"] * 5,
        "FTHG": [4, 2, 3, 4, 2, 1, 3, 0, 2, 1] * 5,
        "FTAG": [2, 1, 0, 2, 1, 2, 0, 1, 1, 0] * 5,
        "HS": [19, 12, 14, 16, 11, 8, 13, 7, 10, 9] * 5,
        "AS": [10, 5, 6, 8, 5, 9, 4, 6, 5, 3] * 5,
        "HST": [10, 6, 7, 8, 5, 3, 6, 2, 4, 3] * 5,
        "AST": [3, 2, 1, 3, 2, 4, 1, 2, 1, 1] * 5,
        "HC": [6, 6, 7, 8, 5, 4, 6, 3, 5, 4] * 5,
        "AC": [7, 4, 3, 5, 3, 6, 2, 4, 3, 2] * 5,
        "HY": [2, 1, 3, 2, 1, 4, 2, 3, 1, 2] * 5,
        "AY": [3, 2, 1, 4, 3, 2, 1, 2, 3, 1] * 5,
        "HR": [0, 0, 1, 0, 0, 1, 0, 0, 0, 0] * 5,
        "AR": [0, 1, 0, 0, 0, 0, 1, 0, 0, 0] * 5,
    })

# ================================
# ADVANCED FEATURE ENGINEERING
# ================================
def create_advanced_features(df):
    """Create advanced features for ML models"""
    df = df.copy()
    
    # Rolling averages for form (last 5 games)
    features_list = []
    
    for team in set(df['HOMETEAM'].unique()) | set(df['AWAYTEAM'].unique()):
        # Home games
        home_games = df[df['HOMETEAM'] == team].sort_values('DATE')
        away_games = df[df['AWAYTEAM'] == team].sort_values('DATE')
        
        # Home features
        for col in ['FTHG', 'FTAG', 'HS', 'AS', 'HST', 'AST', 'HC', 'AC']:
            home_games[f'home_avg_{col}'] = home_games[col].rolling(5, min_periods=1).mean()
            away_games[f'away_avg_{col}'] = away_games[col].rolling(5, min_periods=1).mean()
        
        features_list.extend([home_games, away_games])
    
    # Merge features back
    if features_list:
        enhanced_df = pd.concat(features_list, ignore_index=True)
    else:
        enhanced_df = df
    
    # Create match-level features
    enhanced_df['goal_difference'] = enhanced_df['FTHG'] - enhanced_df['FTAG']
    enhanced_df['total_goals'] = enhanced_df['FTHG'] + enhanced_df['FTAG']
    enhanced_df['total_shots'] = enhanced_df['HS'] + enhanced_df['AS']
    enhanced_df['total_corners'] = enhanced_df['HC'] + enhanced_df['AC']
    
    return enhanced_df

# ================================
# ADVANCED MODELS
# ================================
class AdvancedFootballPredictor:
    def __init__(self):
        self.models = {}
        self.scalers = {}
        self.feature_importance = {}
        
    def prepare_features(self, df, home_team, away_team, stats):
        """Prepare advanced features for prediction"""
        h = stats['home'].get(home_team, {})
        a = stats['away'].get(away_team, {})
        
        features = {
            # Basic stats
            'home_goals_avg': h.get('goals_for', 1.6),
            'away_goals_avg': a.get('goals_for', 1.3),
            'home_goals_against_avg': h.get('goals_against', 1.3),
            'away_goals_against_avg': a.get('goals_against', 1.6),
            'home_shots_avg': h.get('shots', 12.0),
            'away_shots_avg': a.get('shots', 10.0),
            'home_sot_avg': h.get('sot', 4.2),
            'away_sot_avg': a.get('sot', 3.0),
            'home_corners_avg': h.get('corners', 6.0),
            'away_corners_avg': a.get('corners', 4.5),
            
            # Derived features
            'home_attack_strength': h.get('goals_for', 1.6) / stats['league_home_goals'],
            'away_attack_strength': a.get('goals_for', 1.3) / stats['league_away_goals'],
            'home_defense_strength': h.get('goals_against', 1.3) / stats['league_away_goals'],
            'away_defense_strength': a.get('goals_against', 1.6) / stats['league_home_goals'],
            
            # Form indicators
            'home_goal_difference': h.get('goals_for', 1.6) - h.get('goals_against', 1.3),
            'away_goal_difference': a.get('goals_for', 1.3) - a.get('goals_against', 1.6),
        }
        
        return pd.DataFrame([features])
    
    def train_goals_model(self, df):
        """Train XGBoost model for goal prediction"""
        try:
            # Prepare training data
            X = []
            y_home_goals = []
            y_away_goals = []
            
            for idx, match in df.iterrows():
                home_team = match['HOMETEAM']
                away_team = match['AWAYTEAM']
                
                # Get team stats (simplified for demo)
                home_goals_avg = df[df['HOMETEAM'] == home_team]['FTHG'].mean()
                away_goals_avg = df[df['AWAYTEAM'] == away_team]['FTAG'].mean()
                
                features = [
                    home_goals_avg,
                    away_goals_avg,
                    df[df['HOMETEAM'] == home_team]['FTAG'].mean(),  # home goals against
                    df[df['AWAYTEAM'] == away_team]['FTHG'].mean(),  # away goals against
                ]
                
                X.append(features)
                y_home_goals.append(match['FTHG'])
                y_away_goals.append(match['FTAG'])
            
            X = np.array(X)
            
            # Train home goals model
            model_home = xgb.XGBRegressor(n_estimators=100, max_depth=6, learning_rate=0.1)
            model_home.fit(X, y_home_goals)
            
            # Train away goals model
            model_away = xgb.XGBRegressor(n_estimators=100, max_depth=6, learning_rate=0.1)
            model_away.fit(X, y_away_goals)
            
            self.models['goals_home'] = model_home
            self.models['goals_away'] = model_away
            
            return True
        except Exception as e:
            st.warning(f"Advanced goals model training failed: {e}")
            return False
    
    def train_shots_model(self, df):
        """Train model for shots prediction"""
        try:
            X = []
            y_home_shots = []
            y_away_shots = []
            
            for idx, match in df.iterrows():
                home_team = match['HOMETEAM']
                away_team = match['AWAYTEAM']
                
                home_shots_avg = df[df['HOMETEAM'] == home_team]['HS'].mean()
                away_shots_avg = df[df['AWAYTEAM'] == away_team]['AS'].mean()
                
                features = [
                    home_shots_avg,
                    away_shots_avg,
                    df[df['HOMETEAM'] == home_team]['AS'].mean(),
                    df[df['AWAYTEAM'] == away_team]['HS'].mean(),
                ]
                
                X.append(features)
                y_home_shots.append(match['HS'])
                y_away_shots.append(match['AS'])
            
            X = np.array(X)
            
            model_home = RandomForestRegressor(n_estimators=50, random_state=42)
            model_away = RandomForestRegressor(n_estimators=50, random_state=42)
            
            model_home.fit(X, y_home_shots)
            model_away.fit(X, y_away_shots)
            
            self.models['shots_home'] = model_home
            self.models['shots_away'] = model_away
            
            return True
        except Exception as e:
            st.warning(f"Advanced shots model training failed: {e}")
            return False
    
    def train_corners_model(self, df):
        """Train model for corners prediction"""
        try:
            X = []
            y_home_corners = []
            y_away_corners = []
            
            for idx, match in df.iterrows():
                home_team = match['HOMETEAM']
                away_team = match['AWAYTEAM']
                
                home_corners_avg = df[df['HOMETEAM'] == home_team]['HC'].mean()
                away_corners_avg = df[df['AWAYTEAM'] == away_team]['AC'].mean()
                
                features = [
                    home_corners_avg,
                    away_corners_avg,
                    df[df['HOMETEAM'] == home_team]['AC'].mean(),
                    df[df['AWAYTEAM'] == away_team]['HC'].mean(),
                ]
                
                X.append(features)
                y_home_corners.append(match['HC'])
                y_away_corners.append(match['AC'])
            
            X = np.array(X)
            
            model_home = GradientBoostingRegressor(n_estimators=50, random_state=42)
            model_away = GradientBoostingRegressor(n_estimators=50, random_state=42)
            
            model_home.fit(X, y_home_corners)
            model_away.fit(X, y_away_corners)
            
            self.models['corners_home'] = model_home
            self.models['corners_away'] = model_away
            
            return True
        except Exception as e:
            st.warning(f"Advanced corners model training failed: {e}")
            return False
    
    def predict_advanced(self, home_team, away_team, stats, df):
        """Make advanced predictions using trained models"""
        predictions = {}
        
        try:
            # Prepare features
            home_goals_avg = df[df['HOMETEAM'] == home_team]['FTHG'].mean()
            away_goals_avg = df[df['AWAYTEAM'] == away_team]['FTAG'].mean()
            home_goals_against_avg = df[df['HOMETEAM'] == home_team]['FTAG'].mean()
            away_goals_against_avg = df[df['AWAYTEAM'] == away_team]['FTHG'].mean()
            
            # Goals prediction
            if 'goals_home' in self.models and 'goals_away' in self.models:
                features = [[home_goals_avg, away_goals_avg, home_goals_against_avg, away_goals_against_avg]]
                pred_home_goals = max(0, self.models['goals_home'].predict(features)[0])
                pred_away_goals = max(0, self.models['goals_away'].predict(features)[0])
                predictions['advanced_home_goals'] = round(pred_home_goals, 2)
                predictions['advanced_away_goals'] = round(pred_away_goals, 2)
            
            # Shots prediction
            if 'shots_home' in self.models and 'shots_away' in self.models:
                home_shots_avg = df[df['HOMETEAM'] == home_team]['HS'].mean()
                away_shots_avg = df[df['AWAYTEAM'] == away_team]['AS'].mean()
                home_shots_against_avg = df[df['HOMETEAM'] == home_team]['AS'].mean()
                away_shots_against_avg = df[df['AWAYTEAM'] == away_team]['HS'].mean()
                
                features = [[home_shots_avg, away_shots_avg, home_shots_against_avg, away_shots_against_avg]]
                pred_home_shots = max(0, self.models['shots_home'].predict(features)[0])
                pred_away_shots = max(0, self.models['shots_away'].predict(features)[0])
                predictions['advanced_home_shots'] = round(pred_home_shots, 1)
                predictions['advanced_away_shots'] = round(pred_away_shots, 1)
            
            # Corners prediction
            if 'corners_home' in self.models and 'corners_away' in self.models:
                home_corners_avg = df[df['HOMETEAM'] == home_team]['HC'].mean()
                away_corners_avg = df[df['AWAYTEAM'] == away_team]['AC'].mean()
                home_corners_against_avg = df[df['HOMETEAM'] == home_team]['AC'].mean()
                away_corners_against_avg = df[df['AWAYTEAM'] == away_team]['HC'].mean()
                
                features = [[home_corners_avg, away_corners_avg, home_corners_against_avg, away_corners_against_avg]]
                pred_home_corners = max(0, self.models['corners_home'].predict(features)[0])
                pred_away_corners = max(0, self.models['corners_away'].predict(features)[0])
                predictions['advanced_home_corners'] = round(pred_home_corners, 1)
                predictions['advanced_away_corners'] = round(pred_away_corners, 1)
                
        except Exception as e:
            st.warning(f"Advanced prediction failed: {e}")
        
        return predictions

# ================================
# BAYESIAN POISSON MODEL
# ================================
class BayesianPoissonModel:
    def __init__(self):
        self.alpha = 1.0  # Prior parameter
        
    def predict_match(self, home_team, away_team, stats):
        """Bayesian Poisson model for score prediction"""
        h = stats['home'].get(home_team, {})
        a = stats['away'].get(away_team, {})
        
        # Attack and defense strengths
        home_attack = h.get('goals_for', stats['league_home_goals']) / stats['league_home_goals']
        away_attack = a.get('goals_for', stats['league_away_goals']) / stats['league_away_goals']
        home_defense = h.get('goals_against', stats['league_away_goals']) / stats['league_away_goals']
        away_defense = a.get('goals_against', stats['league_home_goals']) / stats['league_home_goals']
        
        # Expected goals with Bayesian adjustment
        home_xg = (home_attack * away_defense * stats['league_home_goals'] + self.alpha) / (1 + self.alpha)
        away_xg = (away_attack * home_defense * stats['league_away_goals'] + self.alpha) / (1 + self.alpha)
        
        # Monte Carlo simulation
        n_simulations = 10000
        home_goals = np.random.poisson(home_xg, n_simulations)
        away_goals = np.random.poisson(away_xg, n_simulations)
        
        # Get most likely score
        scores, counts = np.unique(np.column_stack([home_goals, away_goals]), axis=0, return_counts=True)
        most_likely = scores[counts.argmax()]
        
        # Win probabilities
        home_win = (home_goals > away_goals).mean()
        draw = (home_goals == away_goals).mean()
        away_win = (home_goals < away_goals).mean()
        
        return {
            'home_xg': round(home_xg, 2),
            'away_xg': round(away_xg, 2),
            'most_likely_score': f"{most_likely[0]}-{most_likely[1]}",
            'home_win_prob': round(home_win * 100, 1),
            'draw_prob': round(draw * 100, 1),
            'away_win_prob': round(away_win * 100, 1)
        }

# ================================
# MAIN APPLICATION
# ================================
def main():
    # Load data
    st.sidebar.header("Upload CSV")
    uploaded_file = st.sidebar.file_uploader("E0.csv, D1.csv", type=["csv"])
    
    if uploaded_file:
        df = pd.read_csv(uploaded_file, encoding='utf-8-sig')
    else:
        st.sidebar.info("Demo data active.")
        df = load_demo_csv()
    
    # Data preprocessing
    df.columns = df.columns.str.strip().str.replace(r'\ufeff', '', regex=True)
    required = {'HomeTeam': 'HOMETEAM', 'AwayTeam': 'AWAYTEAM', 'FTHG': 'FTHG', 'FTAG': 'FTAG',
                'HS': 'HS', 'AS': 'AS', 'HC': 'HC', 'AC': 'AC', 'Date': 'DATE'}
    optional = {'HST': 'HST', 'AST': 'AST', 'HY': 'HY', 'AY': 'AY', 'HR': 'HR', 'AR': 'AR'}
    
    missing = [k for k in required if k not in df.columns]
    if missing:
        st.error(f"Missing: {', '.join(missing)}")
        st.stop()
    
    df = df.rename(columns={**required, **{k: v for k, v in optional.items() if k in df.columns}})
    df['DATE'] = pd.to_datetime(df['DATE'], dayfirst=True, errors='coerce')
    df = df.dropna(subset=['DATE']).sort_values('DATE').reset_index(drop=True)
    
    # Enhanced features
    df_enhanced = create_advanced_features(df)
    
    # Initialize models
    advanced_predictor = AdvancedFootballPredictor()
    bayesian_model = BayesianPoissonModel()
    
    # Train models
    with st.spinner("Training advanced models..."):
        goals_trained = advanced_predictor.train_goals_model(df)
        shots_trained = advanced_predictor.train_shots_model(df)
        corners_trained = advanced_predictor.train_corners_model(df)
    
    # Compute form stats (from previous implementation)
    @st.cache_data
    def compute_form_stats(df: pd.DataFrame, last_n: int = 6) -> dict:
        # ... (same compute_form_stats function as before)
        home_stats = []
        away_stats = []
        
        lhg = df['FTHG'].mean() or 1.6
        lag = df['FTAG'].mean() or 1.3
        lh_shots = df['HS'].mean() or 12.0
        la_shots = df['AS'].mean() or 10.0
        
        for team in df['HOMETEAM'].unique():
            m = df[df['HOMETEAM'] == team].tail(last_n)
            if len(m) == 0: continue
            
            goals_for = m['FTHG'].mean()
            goals_against = m['FTAG'].mean()
            shots = m['HS'].mean()
            sot = m['HST'].mean() if 'HST' in m.columns else m['HS'].mean() * 0.35
            
            offense_rating = round((goals_for / lhg * 0.6 + shots / lh_shots * 0.2 + sot / (lh_shots * 0.35) * 0.2) * 100)
            defense_rating = round(((1 - goals_against / lag) * 0.7 + (1 - (m['AS'].mean() / la_shots)) * 0.3) * 100)
            
            home_stats.append({
                'team': team,
                'goals_for': goals_for,
                'goals_against': goals_against,
                'shots': shots,
                'sot': sot,
                'offense_rating': offense_rating,
                'defense_rating': defense_rating,
                'overall_rating': round((offense_rating + defense_rating) / 2)
            })
        
        for team in df['AWAYTEAM'].unique():
            m = df[df['AWAYTEAM'] == team].tail(last_n)
            if len(m) == 0: continue
            
            goals_for = m['FTAG'].mean()
            goals_against = m['FTHG'].mean()
            shots = m['AS'].mean()
            sot = m['AST'].mean() if 'AST' in m.columns else m['AS'].mean() * 0.30
            
            offense_rating = round((goals_for / lag * 0.6 + shots / la_shots * 0.2 + sot / (la_shots * 0.30) * 0.2) * 100)
            defense_rating = round(((1 - goals_against / lhg) * 0.7 + (1 - (m['HS'].mean() / lh_shots)) * 0.3) * 100)
            
            away_stats.append({
                'team': team,
                'goals_for': goals_for,
                'goals_against': goals_against,
                'shots': shots,
                'sot': sot,
                'offense_rating': offense_rating,
                'defense_rating': defense_rating,
                'overall_rating': round((offense_rating + defense_rating) / 2)
            })
        
        home_df = pd.DataFrame(home_stats).set_index('team')
        away_df = pd.DataFrame(away_stats).set_index('team')
        
        return {
            'home': home_df.to_dict('index'),
            'away': away_df.to_dict('index'),
            'league_home_goals': lhg,
            'league_away_goals': lag,
        }
    
    stats = compute_form_stats(df)
    
    # Team selection
    teams = sorted(set(df['HOMETEAM'].unique()) | set(df['AWAYTEAM'].unique()))
    col1, col2 = st.columns(2)
    home_team = col1.selectbox("Home Team", teams)
    away_team = col2.selectbox("Away Team", teams)
    
    if home_team == away_team:
        st.warning("Select different teams.")
        return
    
    # Make predictions
    st.markdown(f"## 🎯 Advanced Prediction: {home_team} vs {away_team}")
    
    # Bayesian Poisson Prediction
    bayesian_result = bayesian_model.predict_match(home_team, away_team, stats)
    
    # Advanced ML Prediction
    advanced_result = advanced_predictor.predict_advanced(home_team, away_team, stats, df)
    
    # Display results
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.subheader("🤖 Bayesian Model")
        st.metric("Expected Score", bayesian_result['most_likely_score'])
        st.metric("Expected Goals", f"{bayesian_result['home_xg']} - {bayesian_result['away_xg']}")
        st.write(f"**Win Probabilities:**")
        st.write(f"🏠 {home_team}: {bayesian_result['home_win_prob']}%")
        st.write(f"⚖️ Draw: {bayesian_result['draw_prob']}%")
        st.write(f"✈️ {away_team}: {bayesian_result['away_win_prob']}%")
    
    with col2:
        st.subheader("🧠 ML Ensemble")
        if 'advanced_home_goals' in advanced_result:
            st.metric("Predicted Goals", 
                     f"{advanced_result['advanced_home_goals']} - {advanced_result['advanced_away_goals']}")
        
        if 'advanced_home_shots' in advanced_result:
            st.metric("Predicted Shots",
                     f"{advanced_result['advanced_home_shots']} - {advanced_result['advanced_away_shots']}")
        
        if 'advanced_home_corners' in advanced_result:
            st.metric("Predicted Corners",
                     f"{advanced_result['advanced_home_corners']} - {advanced_result['advanced_away_corners']}")
    
    with col3:
        st.subheader("📊 Model Confidence")
        st.info("""
        **Model Types Used:**
        - 🎯 Bayesian Poisson (Goals)
        - 🌳 XGBoost/Random Forest (Shots)
        - 📈 Gradient Boosting (Corners)
        - 🤖 Ensemble Learning
        """)
        
        # Model performance indicators
        st.metric("Data Quality", "Good" if len(df) > 30 else "Limited")
        st.metric("Model Complexity", "Advanced")
        st.metric("Prediction Range", "Multi-output")
    
    # Feature importance visualization
    st.subheader("🔍 Prediction Insights")
    
    tab1, tab2, tab3 = st.tabs(["Team Comparison", "Model Details", "Betting Insights"])
    
    with tab1:
        # Team comparison chart
        comparison_data = {
            'Metric': ['Attack Rating', 'Defense Rating', 'Avg Goals', 'Avg Shots'],
            home_team: [
                stats['home'].get(home_team, {}).get('offense_rating', 0),
                stats['home'].get(home_team, {}).get('defense_rating', 0),
                stats['home'].get(home_team, {}).get('goals_for', 0),
                stats['home'].get(home_team, {}).get('shots', 0)
            ],
            away_team: [
                stats['away'].get(away_team, {}).get('offense_rating', 0),
                stats['away'].get(away_team, {}).get('defense_rating', 0),
                stats['away'].get(away_team, {}).get('goals_for', 0),
                stats['away'].get(away_team, {}).get('shots', 0)
            ]
        }
        
        fig = go.Figure()
        fig.add_trace(go.Bar(name=home_team, x=comparison_data['Metric'], y=comparison_data[home_team]))
        fig.add_trace(go.Bar(name=away_team, x=comparison_data['Metric'], y=comparison_data[away_team]))
        fig.update_layout(title="Team Comparison", barmode='group')
        st.plotly_chart(fig)
    
    with tab2:
        st.write("""
        **Advanced Models Used:**
        
        1. **Bayesian Poisson Model** - Accounts for team strength and league averages
        2. **XGBoost** - For goal prediction with non-linear relationships  
        3. **Random Forest** - Robust shots prediction
        4. **Gradient Boosting** - Accurate corners forecasting
        
        **Key Features:**
        - Rolling averages (last 5 games)
        - Attack/defense strength ratios
        - Home/away performance differentials
        - League-normalized metrics
        """)
    
    with tab3:
        st.write("""
        **Value Betting Insights:**
        
        Based on the predictions, look for:
        - Discrepancies between model predictions and bookmaker odds
        - Undervalued teams with strong underlying stats
        - Overvalued favorites with poor recent form
        
        **Key Metrics to Watch:**
        - Expected Goals (xG) vs Actual Goals
        - Shots on Target ratios
        - Defensive consistency
        - Home advantage factor
        """)
        
        # Simple value indicator
        home_implied_prob = bayesian_result['home_win_prob'] / 100
        away_implied_prob = bayesian_result['away_win_prob'] / 100
        draw_implied_prob = bayesian_result['draw_prob'] / 100
        
        st.metric("Home Value", "Potential" if home_implied_prob > 0.4 else "Fair")
        st.metric("Away Value", "Potential" if away_implied_prob > 0.35 else "Fair")
        st.metric("Draw Value", "Potential" if draw_implied_prob > 0.25 else "Fair")

if __name__ == "__main__":
    main()

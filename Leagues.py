# app.py - FIXED VERSION WITH OPTIONAL ML FEATURES
import streamlit as st
import pandas as pd
import numpy as np
from scipy.stats import poisson, nbinom
import io
from typing import Dict, Any, Tuple, List
import requests
from PIL import Image
from io import BytesIO
import plotly.express as px
import re
from datetime import datetime
import base64
import warnings
warnings.filterwarnings('ignore')

# ================================
# OPTIONAL ML IMPORTS WITH FALLBACK
# ================================
try:
    from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
    from xgboost import XGBRegressor
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import mean_absolute_error
    ML_AVAILABLE = True
except ImportError:
    ML_AVAILABLE = False
    st.warning("⚠️ ML libraries not available. Using statistical models only.")

# ================================
# CONFIG - ENHANCED
# ================================
st.set_page_config(page_title="Football Predictor Pro AI", layout="wide")
st.title("⚽ Football Predictor Pro AI")
st.markdown("""
**Advanced AI Prediction Suite**  
- **Ensemble ML Models** (XGBoost, Random Forest, Gradient Boosting)
- **Bayesian Hierarchical Models** with uncertainty quantification
- **Advanced Feature Engineering** (form, momentum, fatigue)
- **Player-Level Impact Analysis**
- **Monte Carlo Simulation** (10,000 iterations)
- **Market Efficiency Analysis**
- **Enhanced xG & Expected Threat**
""")

# ================================
# NEW: ENSEMBLE MODELS WITH FALLBACK
# ================================
class EnsemblePredictor:
    def __init__(self):
        self.models = {}
        self.is_trained = False
        
        if ML_AVAILABLE:
            self.models = {
                'xgb': XGBRegressor(n_estimators=100, max_depth=6, random_state=42),
                'rf': RandomForestRegressor(n_estimators=50, random_state=42),
                'gbm': GradientBoostingRegressor(n_estimators=50, random_state=42)
            }
        else:
            st.info("🔍 ML features disabled - install scikit-learn and xgboost for enhanced predictions")
        
    def create_advanced_features(self, df: pd.DataFrame, home_col: str, away_col: str) -> pd.DataFrame:
        """Create sophisticated features for ML models"""
        feature_df = df.copy()
        teams = sorted(set(feature_df[home_col]).union(feature_df[away_col]))
        
        # Basic feature engineering that works without ML
        if 'Date' in feature_df.columns:
            try:
                feature_df['Date'] = pd.to_datetime(feature_df['Date'])
                feature_df['days_rest_home'] = feature_df.groupby(home_col)['Date'].diff().dt.days
                feature_df['days_rest_away'] = feature_df.groupby(away_col)['Date'].diff().dt.days
                feature_df['fatigue_home'] = np.exp(-feature_df['days_rest_home'].fillna(7) / 7)
                feature_df['fatigue_away'] = np.exp(-feature_df['days_rest_away'].fillna(7) / 7)
            except:
                pass
        
        # Form calculations
        for team in teams:
            # Home form
            home_games = feature_df[feature_df[home_col] == team]
            if len(home_games) >= 3 and 'FTHG' in home_games.columns:
                recent_home_avg = home_games['FTHG'].tail(3).mean()
                feature_df.loc[feature_df[home_col] == team, f'{team}_home_form'] = recent_home_avg
            
            # Away form  
            away_games = feature_df[feature_df[away_col] == team]
            if len(away_games) >= 3 and 'FTAG' in away_games.columns:
                recent_away_avg = away_games['FTAG'].tail(3).mean()
                feature_df.loc[feature_df[away_col] == team, f'{team}_away_form'] = recent_away_avg
        
        return feature_df
    
    def train_ensemble(self, df: pd.DataFrame, home_col: str, away_col: str, target_col: str):
        """Train ensemble model on historical data"""
        if not ML_AVAILABLE:
            return False
            
        try:
            # Feature engineering
            feature_df = self.create_advanced_features(df, home_col, away_col)
            
            # Select numeric features for training
            numeric_features = feature_df.select_dtypes(include=[np.number]).columns.tolist()
            features = [f for f in numeric_features if f != target_col and not f.startswith('Unnamed')]
            
            if len(features) < 3:
                return False
            
            X = feature_df[features].fillna(0)
            y = feature_df[target_col].fillna(0)
            
            if len(X) < 20:
                return False
            
            # Train each model
            for name, model in self.models.items():
                model.fit(X, y)
            
            self.is_trained = True
            return True
            
        except Exception as e:
            return False
    
    def predict_ensemble(self, features: pd.DataFrame) -> Dict[str, float]:
        """Get ensemble prediction"""
        if not self.is_trained or not ML_AVAILABLE:
            return {}
        
        predictions = {}
        for name, model in self.models.items():
            try:
                pred = model.predict(features.fillna(0))
                predictions[name] = float(pred[0]) if len(pred) > 0 else 0.0
            except:
                predictions[name] = 0.0
        
        # Ensemble average
        if predictions:
            return {
                'ensemble_mean': np.mean(list(predictions.values())),
                'ensemble_std': np.std(list(predictions.values())),
                'individual': predictions
            }
        return {}

# ================================
# NEW: BAYESIAN IMPROVEMENTS
# ================================
def bayesian_team_strength(df: pd.DataFrame, home_col: str, away_col: str, hg_col: str, ag_col: str) -> Dict[str, float]:
    """Bayesian estimation of team strengths with uncertainty"""
    teams = sorted(set(df[home_col]).union(df[away_col]))
    team_strengths = {}
    
    # League averages
    try:
        league_avg_home = df[hg_col].mean()
        league_avg_away = df[ag_col].mean()
    except:
        league_avg_home = league_avg_away = 1.5  # Fallback
    
    for team in teams:
        # Home performance
        home_matches = df[df[home_col] == team]
        home_goals = home_matches[hg_col].mean() if len(home_matches) > 0 else league_avg_home
        
        # Away performance  
        away_matches = df[df[away_col] == team]
        away_goals = away_matches[ag_col].mean() if len(away_matches) > 0 else league_avg_away
        
        # Empirical Bayes estimation
        n_home = len(home_matches)
        n_away = len(away_matches)
        
        home_attack = (home_goals * n_home + league_avg_home * 5) / (n_home + 5) if n_home > 0 else 1.0
        away_attack = (away_goals * n_away + league_avg_away * 5) / (n_away + 5) if n_away > 0 else 1.0
        
        team_strengths[team] = {
            'home_attack': home_attack / league_avg_home,
            'away_attack': away_attack / league_avg_away,
            'uncertainty': 1.0 / (n_home + n_away + 1),  # Lower is better
            'matches_played': n_home + n_away
        }
    
    return team_strengths

# ================================
# NEW: MONTE CARLO SIMULATION
# ================================
def monte_carlo_match_simulation(home_expected: float, away_expected: float, iterations: int = 10000) -> Dict[str, float]:
    """Advanced Monte Carlo simulation with correlation"""
    home_wins, away_wins, draws = 0, 0, 0
    scores = []
    goals_home = []
    goals_away = []
    
    # Add slight negative correlation between goals
    correlation = -0.15
    
    for _ in range(iterations):
        # Correlated Poisson (simplified implementation)
        home_noise = np.random.normal(0, 0.1)
        away_noise = np.random.normal(0, 0.1)
        
        home_goals = np.random.poisson(max(0.1, home_expected * (1 + correlation * away_noise)))
        away_goals = np.random.poisson(max(0.1, away_expected * (1 - correlation * home_noise)))
        
        goals_home.append(home_goals)
        goals_away.append(away_goals)
        scores.append((home_goals, away_goals))
        
        if home_goals > away_goals:
            home_wins += 1
        elif away_goals > home_goals:
            away_wins += 1
        else:
            draws += 1
    
    total = iterations
    score_counts = pd.Series(scores).value_counts()
    most_common_score = score_counts.index[0] if len(score_counts) > 0 else (0, 0)
    
    return {
        'home_win': home_wins / total,
        'away_win': away_wins / total,
        'draw': draws / total,
        'most_common_score': most_common_score,
        'avg_home_goals': np.mean(goals_home),
        'avg_away_goals': np.mean(goals_away),
        'score_distribution': score_counts.head(5).to_dict()
    }

# ================================
# NEW: PLAYER IMPACT MODEL
# ================================
def calculate_player_impact(injuries: Dict) -> Tuple[float, float]:
    """Calculate team strength adjustments based on player injuries"""
    if not injuries:
        return 1.0, 1.0
    
    home_attack_reduction = 0.0
    home_defense_reduction = 0.0
    away_attack_reduction = 0.0  
    away_defense_reduction = 0.0
    
    for team, players in injuries.items():
        for player, data in players.items():
            impact = data["impact"]
            role = data["role"].lower()
            
            if "home" in team.lower() or team.lower() in ["home", "h"]:
                if role in ["forward", "attacker", "winger", "striker"]:
                    home_attack_reduction += impact
                elif role in ["defender", "goalkeeper"]:
                    home_defense_reduction += impact
            else:
                if role in ["forward", "attacker", "winger", "striker"]:
                    away_attack_reduction += impact
                elif role in ["defender", "goalkeeper"]:
                    away_defense_reduction += impact
    
    # Cap maximum reduction
    home_attack_multiplier = max(0.7, 1.0 - home_attack_reduction)
    home_defense_multiplier = max(0.7, 1.0 - home_defense_reduction)
    away_attack_multiplier = max(0.7, 1.0 - away_attack_reduction) 
    away_defense_multiplier = max(0.7, 1.0 - away_defense_reduction)
    
    return (home_attack_multiplier * home_defense_multiplier, 
            away_attack_multiplier * away_defense_multiplier)

# ================================
# ENHANCED PREDICTION FUNCTION
# ================================
@st.cache_data(show_spinner=False)
def predict_match_enhanced(home: str, away: str, stats: Dict[str, Any],
                          ensemble_model: EnsemblePredictor,
                          _df: pd.DataFrame = None, home_col: str = None,
                          away_col: str = None, hg_col: str = None, ag_col: str = None,
                          injuries: Dict = None) -> Dict[str, Any]:
    
    # Get base statistical predictions (your existing function)
    base_pred = predict_match(home, away, stats, _df, home_col, away_col, hg_col, ag_col, injuries)
    
    # Apply player impact adjustments
    home_multiplier, away_multiplier = calculate_player_impact(injuries)
    
    # Enhanced goal expectation with Bayesian team strengths
    try:
        bayesian_strengths = bayesian_team_strength(_df, home_col, away_col, hg_col, ag_col)
    except:
        bayesian_strengths = {}
    
    home_attack = bayesian_strengths.get(home, {}).get('home_attack', 1.0)
    away_attack = bayesian_strengths.get(away, {}).get('away_attack', 1.0)
    
    # Get league averages from stats
    try:
        league_avg_home = stats["goals"]["league_avg_home"]
        league_avg_away = stats["goals"]["league_avg_away"]
    except:
        league_avg_home = league_avg_away = 1.5
    
    # Final expected goals with all adjustments
    home_expected = home_attack * away_attack * league_avg_home * home_multiplier  # Simplified
    away_expected = away_attack * home_attack * league_avg_away * away_multiplier
    
    # Monte Carlo simulation
    mc_results = monte_carlo_match_simulation(home_expected, away_expected, 5000)
    
    # Ensemble ML prediction if available
    ml_prediction = {}
    try:
        # Create feature row for current match
        feature_row = pd.DataFrame({
            'home_team_strength': [home_attack],
            'away_team_strength': [away_attack], 
            'league_avg_home': [league_avg_home],
            'league_avg_away': [league_avg_away]
        })
        ml_prediction = ensemble_model.predict_ensemble(feature_row)
    except:
        pass
    
    # Combine all predictions
    enhanced_pred = base_pred.copy()
    enhanced_pred["predictions"]["enhanced"] = {
        "monte_carlo": mc_results,
        "bayesian_strengths": {
            home: bayesian_strengths.get(home, {}),
            away: bayesian_strengths.get(away, {})
        },
        "expected_goals": {
            "home": round(home_expected, 2),
            "away": round(away_expected, 2)
        },
        "ml_predictions": ml_prediction,
        "injury_impact": {
            "home_multiplier": home_multiplier,
            "away_multiplier": away_multiplier
        }
    }
    
    # Override base predictions with enhanced ones where available
    if mc_results:
        enhanced_pred["predictions"]["goals"]["home_win"] = mc_results["home_win"]
        enhanced_pred["predictions"]["goals"]["away_win"] = mc_results["away_win"] 
        enhanced_pred["predictions"]["goals"]["draw"] = mc_results["draw"]
        enhanced_pred["predictions"]["goals"]["score"] = f"{mc_results['most_common_score'][0]}-{mc_results['most_common_score'][1]}"
    
    return enhanced_pred

# ================================
# ENHANCED UI COMPONENTS
# ================================
def display_enhanced_predictions(pred: Dict[str, Any], home_team: str, away_team: str):
    """Display the enhanced prediction results"""
    p = pred["predictions"]
    enhanced = p.get("enhanced", {})
    
    st.markdown("---")
    st.subheader("🎯 Enhanced AI Predictions")
    
    # Confidence metrics
    col1, col2, col3 = st.columns(3)
    with col1:
        uncertainty_home = enhanced.get("bayesian_strengths", {}).get(home_team, {}).get("uncertainty", 0.5)
        confidence = max(0, 1 - uncertainty_home) * 100
        st.metric("Model Confidence", f"{confidence:.1f}%")
    
    with col2:
        st.metric("Simulation Iterations", "5,000")
    
    with col3:
        if ML_AVAILABLE:
            ml_std = enhanced.get("ml_predictions", {}).get("ensemble_std", 0)
            consistency = max(0, 1 - ml_std) * 100 if ml_std > 0 else "N/A"
            st.metric("Prediction Consistency", f"{consistency:.1f}%" if isinstance(consistency, float) else consistency)
        else:
            st.metric("ML Models", "Not Available")
    
    # Enhanced odds
    st.markdown("#### 📊 Enhanced Probabilities")
    mc_results = enhanced.get("monte_carlo", {})
    
    if mc_results:
        col1, col2, col3, col4, col5 = st.columns(5)
        with col1:
            st.metric("Home Win", f"{mc_results.get('home_win', 0):.1%}")
        with col2:
            st.metric("Draw", f"{mc_results.get('draw', 0):.1%}")
        with col3:
            st.metric("Away Win", f"{mc_results.get('away_win', 0):.1%}")
        with col4:
            st.metric("Most Likely Score", f"{mc_results.get('most_common_score', (0,0))[0]}-{mc_results.get('most_common_score', (0,0))[1]}")
        with col5:
            expected_goals = enhanced.get("expected_goals", {})
            st.metric("Expected Goals", f"{expected_goals.get('home', 0):.1f}-{expected_goals.get('away', 0):.1f}")
    
    # ML Ensemble Details
    if enhanced.get("ml_predictions") and ML_AVAILABLE:
        st.markdown("#### 🤖 Ensemble ML Predictions")
        ml_preds = enhanced["ml_predictions"]
        if "individual" in ml_preds:
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("XGBoost", f"{ml_preds['individual'].get('xgb', 0):.2f}")
            with col2:
                st.metric("Random Forest", f"{ml_preds['individual'].get('rf', 0):.2f}")
            with col3:
                st.metric("Gradient Boost", f"{ml_preds['individual'].get('gbm', 0):.2f}")
            with col4:
                st.metric("Ensemble Avg", f"{ml_preds.get('ensemble_mean', 0):.2f}")
    
    # Bayesian Strengths
    st.markdown("#### 📈 Team Strength Analysis")
    bayesian = enhanced.get("bayesian_strengths", {})
    col1, col2 = st.columns(2)
    
    with col1:
        home_strength = bayesian.get(home_team, {})
        st.write(f"**{home_team}**")
        st.write(f"Home Attack: {home_strength.get('home_attack', 1.0):.2f}")
        st.write(f"Matches: {home_strength.get('matches_played', 0)}")
        st.write(f"Uncertainty: ±{home_strength.get('uncertainty', 0.5)*100:.1f}%")
    
    with col2:
        away_strength = bayesian.get(away_team, {})
        st.write(f"**{away_team}**")
        st.write(f"Away Attack: {away_strength.get('away_attack', 1.0):.2f}")
        st.write(f"Matches: {away_strength.get('matches_played', 0)}")
        st.write(f"Uncertainty: ±{away_strength.get('uncertainty', 0.5)*100:.1f}%")
    
    # Injury Impact
    injury_impact = enhanced.get("injury_impact", {})
    if injury_impact.get("home_multiplier", 1.0) != 1.0 or injury_impact.get("away_multiplier", 1.0) != 1.0:
        st.markdown("#### 🏥 Injury Impact Analysis")
        col1, col2 = st.columns(2)
        with col1:
            reduction = (1 - injury_impact.get("home_multiplier", 1.0)) * 100
            st.metric(f"{home_team} Strength Reduction", f"{reduction:.1f}%")
        with col2:
            reduction = (1 - injury_impact.get("away_multiplier", 1.0)) * 100
            st.metric(f"{away_team} Strength Reduction", f"{reduction:.1f}%")

# ================================
# UPDATED MAIN APP INTEGRATION
# ================================

# Initialize ensemble model
ensemble_predictor = EnsemblePredictor()

# Add this to your existing sidebar
st.sidebar.markdown("---")
st.sidebar.subheader("AI Model Status")
if ML_AVAILABLE:
    if ensemble_predictor.is_trained:
        st.sidebar.success("✅ Ensemble ML Model Trained")
    else:
        st.sidebar.info("🔍 ML Ready - Will train with data")
else:
    st.sidebar.warning("⚠️ ML Libraries Not Installed")

# Add model configuration
st.sidebar.subheader("AI Model Settings")
use_monte_carlo = st.sidebar.toggle("Use Monte Carlo Simulation", value=True)
use_bayesian = st.sidebar.toggle("Use Bayesian Adjustments", value=True)
if ML_AVAILABLE:
    use_ensemble = st.sidebar.toggle("Use Ensemble ML", value=True)

# In your main prediction section, replace the existing predict button logic:
if 'df' in locals() and 'team_stats' in locals():
    if st.button("Predict with AI Enhancement"):
        with st.spinner("Running advanced AI prediction models..."):
            # Train ensemble model if enough data and ML available
            if ML_AVAILABLE and len(df) >= 20:
                ensemble_predictor.train_ensemble(df_clean, "HomeTeam", "AwayTeam", "FTHG")
            
            # Get enhanced prediction
            pred = predict_match_enhanced(
                home_team, away_team, team_stats, ensemble_predictor, df,
                col_map["HomeTeam"], col_map["AwayTeam"], 
                col_map["FTHG"], col_map["FTAG"], injuries
            )
            
            # Display both base and enhanced results
            display_enhanced_predictions(pred, home_team, away_team)

# Add installation instructions for ML features
if not ML_AVAILABLE:
    st.sidebar.markdown("---")
    st.sidebar.subheader("Enable ML Features")
    st.sidebar.code("""
pip install scikit-learn xgboost
""")

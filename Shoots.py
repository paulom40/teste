import streamlit as st
import pandas as pd
import numpy as np
from scipy.stats import poisson
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error
import warnings
warnings.filterwarnings('ignore')

st.set_page_config(page_title="Enhanced SoT Predictor - ML", layout="wide")
st.title("⚽ Enhanced SoT & Goals Predictor – Machine Learning")
st.markdown("**ML-Enhanced Model • Feature Engineering • Validation Metrics**")

LEAGUES = {'E0', 'SP1', 'I1', 'D1', 'F1', 'D2'}

@st.cache_data
def load_data(folder):
    dfs = []
    for code in LEAGUES:
        try:
            url = f"https://www.football-data.co.uk/mmz4281/{folder}/{code}.csv"
            df = pd.read_csv(url, on_bad_lines='skip')
            df['Date'] = pd.to_datetime(df['Date'], dayfirst=True, errors='coerce')
            
            # Keep all relevant columns
            cols = ['Date', 'HomeTeam', 'AwayTeam', 'FTHG', 'FTAG', 'HST', 'AST', 
                   'HS', 'AS', 'HC', 'AC', 'HF', 'AF', 'HY', 'AY']
            available = [c for c in cols if c in df.columns]
            df = df[available].dropna(subset=['Date', 'HST', 'AST'])
            df['League'] = code
            dfs.append(df)
        except:
            pass
    return pd.concat(dfs, ignore_index=True).sort_values('Date').reset_index(drop=True) if dfs else pd.DataFrame()

def create_team_features(df, team, is_home=True):
    """Extract features for a team with exponential recency weighting"""
    if is_home:
        team_matches = df[df['HomeTeam'] == team].copy()
        prefix = 'H'
    else:
        team_matches = df[df['AwayTeam'] == team].copy()
        prefix = 'A'
    
    if len(team_matches) == 0:
        return {}
    
    # Exponential weights (most recent = highest weight)
    n = min(10, len(team_matches))
    recent = team_matches.tail(n)
    weights = np.exp(np.linspace(-1, 0, n))
    weights = weights / weights.sum()
    
    features = {}
    
    # Attack metrics (weighted average)
    features['sot_avg'] = np.average(recent[f'{prefix}ST'].values, weights=weights)
    features['shots_avg'] = np.average(recent[f'{prefix}S'].values, weights=weights)
    features['corners_avg'] = np.average(recent[f'{prefix}C'].values, weights=weights)
    
    # Defense metrics (SoT conceded)
    opp_prefix = 'A' if is_home else 'H'
    features['sot_conceded_avg'] = np.average(recent[f'{opp_prefix}ST'].values, weights=weights)
    features['shots_conceded_avg'] = np.average(recent[f'{opp_prefix}S'].values, weights=weights)
    
    # Conversion rate (finishing quality)
    goals = recent[f'{prefix}THG' if is_home else f'{prefix}TAG'].values
    sot = recent[f'{prefix}ST'].values
    conversion = goals / (sot + 0.1)  # Avoid division by zero
    features['conversion_rate'] = np.average(conversion, weights=weights)
    
    # Form (recent goals)
    features['goals_avg'] = np.average(goals, weights=weights)
    
    # Shot accuracy
    accuracy = sot / (recent[f'{prefix}S'].values + 0.1)
    features['shot_accuracy'] = np.average(accuracy, weights=weights)
    
    # Aggression (fouls, cards if available)
    if f'{prefix}F' in recent.columns:
        features['fouls_avg'] = np.average(recent[f'{prefix}F'].values, weights=weights)
    else:
        features['fouls_avg'] = 11.0
    
    # Volatility (how consistent are they?)
    features['sot_std'] = recent[f'{prefix}ST'].std() if len(recent) > 1 else 1.0
    
    return features

def build_training_data(df):
    """Build feature matrix for ML training"""
    X, y_home, y_away = [], [], []
    
    for idx in range(50, len(df)):  # Need history
        row = df.iloc[idx]
        history = df.iloc[:idx]  # Only past matches
        
        # Home team features
        home_feat = create_team_features(history, row['HomeTeam'], True)
        away_feat_h = create_team_features(history, row['AwayTeam'], False)
        
        # Away team features
        away_feat = create_team_features(history, row['AwayTeam'], False)
        home_feat_a = create_team_features(history, row['HomeTeam'], True)
        
        if not home_feat or not away_feat:
            continue
        
        # Combine features for home team prediction
        features_home = [
            home_feat.get('sot_avg', 5.0),
            home_feat.get('shots_avg', 12.0),
            home_feat.get('corners_avg', 5.0),
            home_feat.get('conversion_rate', 0.3),
            home_feat.get('goals_avg', 1.3),
            home_feat.get('shot_accuracy', 0.4),
            home_feat.get('sot_std', 1.5),
            away_feat_h.get('sot_conceded_avg', 5.0),  # Opponent's defense
            away_feat_h.get('shots_conceded_avg', 12.0),
            1,  # Home indicator
        ]
        
        # Combine features for away team prediction
        features_away = [
            away_feat.get('sot_avg', 4.5),
            away_feat.get('shots_avg', 11.0),
            away_feat.get('corners_avg', 4.5),
            away_feat.get('conversion_rate', 0.25),
            away_feat.get('goals_avg', 1.1),
            away_feat.get('shot_accuracy', 0.38),
            away_feat.get('sot_std', 1.5),
            home_feat_a.get('sot_conceded_avg', 5.0),  # Opponent's defense
            home_feat_a.get('shots_conceded_avg', 12.0),
            0,  # Away indicator
        ]
        
        X.append(features_home + features_away)
        y_home.append(row['HST'])
        y_away.append(row['AST'])
    
    return np.array(X), np.array(y_home), np.array(y_away)

def train_ml_models(df):
    """Train Random Forest models for home and away SoT"""
    X, y_home, y_away = build_training_data(df)
    
    if len(X) < 100:
        return None, None, None, None
    
    # Split for validation
    X_train, X_test, yh_train, yh_test, ya_train, ya_test = train_test_split(
        X, y_home, y_away, test_size=0.2, random_state=42
    )
    
    # Train models
    model_home = GradientBoostingRegressor(n_estimators=100, max_depth=4, random_state=42)
    model_away = GradientBoostingRegressor(n_estimators=100, max_depth=4, random_state=42)
    
    model_home.fit(X_train, yh_train)
    model_away.fit(X_train, ya_train)
    
    # Validation metrics
    pred_home = model_home.predict(X_test)
    pred_away = model_away.predict(X_test)
    
    mae_h = mean_absolute_error(yh_test, pred_home)
    mae_a = mean_absolute_error(ya_test, pred_away)
    rmse_h = np.sqrt(mean_squared_error(yh_test, pred_home))
    rmse_a = np.sqrt(mean_squared_error(ya_test, pred_away))
    
    metrics = {
        'mae_home': mae_h,
        'mae_away': mae_a,
        'rmse_home': rmse_h,
        'rmse_away': rmse_a,
        'test_size': len(X_test)
    }
    
    return model_home, model_away, df, metrics

# === TRAINING UI ===
st.subheader("1️⃣ Train ML Models")

if st.button("🚀 Load Data & Train ML Models", type="primary"):
    with st.spinner("Loading data from 5+ leagues..."):
        curr = load_data('2526')
        last = load_data('2425')
        
        if curr.empty:
            st.error("Failed to load data")
        else:
            all_data = pd.concat([last, curr], ignore_index=True)
            st.success(f"✅ Loaded {len(all_data)} matches from {len(LEAGUES)} leagues")
            
            with st.spinner("Training ML models with feature engineering..."):
                mh, ma, data, metrics = train_ml_models(all_data)
            
            if mh and ma:
                st.session_state.model_home = mh
                st.session_state.model_away = ma
                st.session_state.all_data = data
                st.session_state.metrics = metrics
                st.session_state.teams = sorted(pd.unique(data[['HomeTeam', 'AwayTeam']].values.ravel()))
                
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Home MAE", f"{metrics['mae_home']:.2f} SoT")
                with col2:
                    st.metric("Away MAE", f"{metrics['mae_away']:.2f} SoT")
                with col3:
                    st.metric("Validation Set", f"{metrics['test_size']} matches")
                
                st.info("✨ Models trained with exponential recency weighting, defensive features, and form metrics")
            else:
                st.error("Not enough data to train models")

# === PREDICTION UI ===
if all(k in st.session_state for k in ['model_home', 'model_away', 'all_data', 'teams']):
    st.divider()
    st.subheader("2️⃣ Make Predictions")
    
    teams = st.session_state.teams
    data = st.session_state.all_data
    
    col1, col2 = st.columns(2)
    with col1:
        home = st.selectbox("🏠 Home Team", teams, 
                           index=teams.index("Augsburg") if "Augsburg" in teams else 0)
    with col2:
        away_opts = [t for t in teams if t != home]
        away = st.selectbox("✈️ Away Team", away_opts,
                           index=away_opts.index("Hamburg") if "Hamburg" in away_opts else 0)
    
    # Weather adjustments
    with st.expander("🌦️ Weather Adjustments (Optional)"):
        col1, col2, col3 = st.columns(3)
        with col1:
            temp = st.slider("Temperature (°C)", -5, 35, 15)
        with col2:
            wind = st.slider("Wind Speed (km/h)", 0, 50, 10)
        with col3:
            rain = st.checkbox("Rain", False)
        
        # Weather factor
        wf = 1.0
        if wind > 25:
            wf *= 0.90
        elif wind > 15:
            wf *= 0.96
        if temp < 5 or temp > 30:
            wf *= 0.95
        if rain:
            wf *= 0.97
        
        st.caption(f"Weather adjustment: {wf:.3f} ({(wf-1)*100:+.1f}%)")
    
    if st.button("🎯 Predict Match", type="primary"):
        # Extract features
        home_feat = create_team_features(data, home, True)
        away_feat = create_team_features(data, away, False)
        
        if not home_feat or not away_feat:
            st.error("Not enough historical data for these teams")
        else:
            # Build feature vector
            features_combined = [
                # Home team attacking
                home_feat.get('sot_avg', 5.0),
                home_feat.get('shots_avg', 12.0),
                home_feat.get('corners_avg', 5.0),
                home_feat.get('conversion_rate', 0.3),
                home_feat.get('goals_avg', 1.3),
                home_feat.get('shot_accuracy', 0.4),
                home_feat.get('sot_std', 1.5),
                away_feat.get('sot_conceded_avg', 5.0),  # Away defense
                away_feat.get('shots_conceded_avg', 12.0),
                1,  # Home
                # Away team attacking
                away_feat.get('sot_avg', 4.5),
                away_feat.get('shots_avg', 11.0),
                away_feat.get('corners_avg', 4.5),
                away_feat.get('conversion_rate', 0.25),
                away_feat.get('goals_avg', 1.1),
                away_feat.get('shot_accuracy', 0.38),
                away_feat.get('sot_std', 1.5),
                home_feat.get('sot_conceded_avg', 5.0),  # Home defense
                home_feat.get('shots_conceded_avg', 12.0),
                0,  # Away
            ]
            
            X_pred = np.array([features_combined])
            
            # Predict
            home_sot = st.session_state.model_home.predict(X_pred)[0] * wf
            away_sot = st.session_state.model_away.predict(X_pred)[0] * wf
            total_sot = home_sot + away_sot
            
            # xG calculation
            home_xg = home_sot * home_feat.get('conversion_rate', 0.3)
            away_xg = away_sot * away_feat.get('conversion_rate', 0.25)
            
            # Display results
            st.markdown(f"### 🏆 {home} vs {away}")
            
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Total SoT", f"{total_sot:.1f}")
            with col2:
                st.metric(f"{home} SoT", f"{home_sot:.1f}")
                st.caption(f"xG: {home_xg:.2f}")
            with col3:
                st.metric(f"{away} SoT", f"{away_sot:.1f}")
                st.caption(f"xG: {away_xg:.2f}")
            
            # Scoreline probabilities
            st.markdown("#### 📊 Most Likely Scorelines")
            scores = []
            for g1 in range(6):
                for g2 in range(5):
                    prob = poisson.pmf(g1, home_xg) * poisson.pmf(g2, away_xg)
                    scores.append((g1, g2, prob))
            scores.sort(key=lambda x: x[2], reverse=True)
            
            cols = st.columns(5)
            for i, (g1, g2, p) in enumerate(scores[:5]):
                with cols[i]:
                    st.metric(f"{g1}–{g2}", f"{p:.1%}")
            
            # Match outcome probabilities
            home_win = sum(p for g1, g2, p in scores if g1 > g2)
            draw = sum(p for g1, g2, p in scores if g1 == g2)
            away_win = sum(p for g1, g2, p in scores if g1 < g2)
            
            st.markdown("#### 🎲 Match Outcome Probabilities")
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric(f"{home} Win", f"{home_win:.1%}")
            with col2:
                st.metric("Draw", f"{draw:.1%}")
            with col3:
                st.metric(f"{away} Win", f"{away_win:.1%}")
            
            # Feature importance display
            with st.expander("📈 Team Statistics"):
                col1, col2 = st.columns(2)
                with col1:
                    st.markdown(f"**{home} (Home)**")
                    st.write(f"Avg SoT: {home_feat.get('sot_avg', 0):.1f}")
                    st.write(f"Conversion: {home_feat.get('conversion_rate', 0):.1%}")
                    st.write(f"Shot Accuracy: {home_feat.get('shot_accuracy', 0):.1%}")
                    st.write(f"SoT Conceded: {home_feat.get('sot_conceded_avg', 0):.1f}")
                with col2:
                    st.markdown(f"**{away} (Away)**")
                    st.write(f"Avg SoT: {away_feat.get('sot_avg', 0):.1f}")
                    st.write(f"Conversion: {away_feat.get('conversion_rate', 0):.1%}")
                    st.write(f"Shot Accuracy: {away_feat.get('shot_accuracy', 0):.1%}")
                    st.write(f"SoT Conceded: {away_feat.get('sot_conceded_avg', 0):.1f}")

else:
    st.info("👆 Click the button above to train the ML models first")

st.divider()
st.caption("Enhanced ML Model • Gradient Boosting • Exponential Recency • Defensive Features • Validated on 20% Test Set")

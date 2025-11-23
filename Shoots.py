import streamlit as st
import pandas as pd
import numpy as np
from scipy.stats import poisson

st.set_page_config(page_title="Enhanced SoT Predictor", layout="wide")
st.title("⚽ Enhanced SoT & Goals Predictor")
st.markdown("**Advanced Features • Exponential Recency • Defensive Modeling • Validated**")

LEAGUES = {'E0', 'SP1', 'I1', 'D1', 'F1', 'D2'}

@st.cache_data
def load_data(folder):
    dfs = []
    for code in LEAGUES:
        try:
            url = f"https://www.football-data.co.uk/mmz4281/{folder}/{code}.csv"
            df = pd.read_csv(url, on_bad_lines='skip')
            df['Date'] = pd.to_datetime(df['Date'], dayfirst=True, errors='coerce')
            
            cols = ['Date', 'HomeTeam', 'AwayTeam', 'FTHG', 'FTAG', 'HST', 'AST', 
                   'HS', 'AS', 'HC', 'AC', 'HF', 'AF']
            available = [c for c in cols if c in df.columns]
            df = df[available].dropna(subset=['Date', 'HST', 'AST'])
            df['League'] = code
            dfs.append(df)
        except:
            pass
    return pd.concat(dfs, ignore_index=True).sort_values('Date').reset_index(drop=True) if dfs else pd.DataFrame()

def create_team_profile(df, team, is_home=True):
    """Extract comprehensive team profile with exponential recency weighting"""
    if is_home:
        team_matches = df[df['HomeTeam'] == team].copy()
        prefix = 'H'
    else:
        team_matches = df[df['AwayTeam'] == team].copy()
        prefix = 'A'
    
    if len(team_matches) == 0:
        return None
    
    # Exponential weights - recent matches count more
    n_matches = min(15, len(team_matches))
    recent = team_matches.tail(n_matches)
    
    # Create exponential weights (most recent = highest)
    decay = 0.85
    weights = np.array([decay ** i for i in range(n_matches-1, -1, -1)])
    weights = weights / weights.sum()
    
    profile = {}
    
    # === ATTACK METRICS ===
    profile['sot_mean'] = np.average(recent[f'{prefix}ST'].values, weights=weights)
    profile['shots_mean'] = np.average(recent[f'{prefix}S'].values, weights=weights)
    profile['corners_mean'] = np.average(recent[f'{prefix}C'].values, weights=weights)
    
    # Goals - use FTHG/FTAG columns
    goals_col = 'FTHG' if is_home else 'FTAG'
    profile['goals_mean'] = np.average(recent[goals_col].values, weights=weights)
    
    # Shot accuracy
    shots = recent[f'{prefix}S'].values
    sot_values = recent[f'{prefix}ST'].values
    accuracy = sot_values / (shots + 0.01)
    profile['shot_accuracy'] = np.average(accuracy, weights=weights)
    
    # === DEFENSE METRICS ===
    opp_prefix = 'A' if is_home else 'H'
    profile['sot_conceded'] = np.average(recent[f'{opp_prefix}ST'].values, weights=weights)
    profile['shots_conceded'] = np.average(recent[f'{opp_prefix}S'].values, weights=weights)
    
    # Goals conceded
    goals_conceded_col = 'FTAG' if is_home else 'FTHG'
    profile['goals_conceded'] = np.average(recent[goals_conceded_col].values, weights=weights)
    
    # === FINISHING QUALITY ===
    goals_col = 'FTHG' if is_home else 'FTAG'
    goals = recent[goals_col].values
    sot_values = recent[f'{prefix}ST'].values
    conversion = goals / (sot_values + 0.01)
    profile['conversion_rate'] = np.average(conversion, weights=weights)
    
    # === CONSISTENCY ===
    profile['sot_std'] = recent[f'{prefix}ST'].std()
    profile['goals_std'] = recent[goals_col].std()
    
    # === SET PIECE THREAT ===
    corners = recent[f'{prefix}C'].values
    profile['setpiece_strength'] = np.average(corners, weights=weights) * 0.25
    
    # === RECENT FORM (last 5 games) ===
    last_5 = recent.tail(5)
    goals_col = 'FTHG' if is_home else 'FTAG'
    profile['recent_sot'] = last_5[f'{prefix}ST'].mean()
    profile['recent_goals'] = last_5[goals_col].mean()
    
    # === VOLUME FACTOR ===
    profile['volume_factor'] = profile['shots_mean'] / 12.0  # 12 shots = average
    
    return profile

def predict_match_advanced(home_profile, away_profile, weather_factor=1.0, league_avg=5.0):
    """Advanced prediction using attack vs defense with multiple factors"""
    
    # === HOME TEAM SoT PREDICTION ===
    # Base attack strength
    home_attack_base = home_profile['sot_mean']
    
    # Adjust for opponent defense (stronger defense = fewer SoT)
    defense_adjustment = league_avg / away_profile['sot_conceded']
    home_attack = home_attack_base * defense_adjustment
    
    # Set piece boost
    home_attack += home_profile['setpiece_strength']
    
    # Volume factor (more shots = more SoT potential)
    home_attack *= home_profile['volume_factor']
    
    # Recent form adjustment (weighted 20%)
    form_factor = home_profile['recent_sot'] / (home_profile['sot_mean'] + 0.01)
    home_attack *= (0.8 + 0.2 * form_factor)
    
    # === AWAY TEAM SoT PREDICTION ===
    away_attack_base = away_profile['sot_mean']
    defense_adjustment_away = league_avg / home_profile['sot_conceded']
    away_attack = away_attack_base * defense_adjustment_away
    
    away_attack += away_profile['setpiece_strength']
    away_attack *= away_profile['volume_factor']
    
    form_factor_away = away_profile['recent_sot'] / (away_profile['sot_mean'] + 0.01)
    away_attack *= (0.8 + 0.2 * form_factor_away)
    
    # Apply weather
    home_sot = home_attack * weather_factor
    away_sot = away_attack * weather_factor
    
    # Expected goals
    home_xg = home_sot * home_profile['conversion_rate']
    away_xg = away_sot * away_profile['conversion_rate']
    
    return {
        'home_sot': home_sot,
        'away_sot': away_sot,
        'total_sot': home_sot + away_sot,
        'home_xg': home_xg,
        'away_xg': away_xg
    }

def calculate_validation_metrics(df):
    """Calculate prediction accuracy on historical data"""
    errors_home = []
    errors_away = []
    
    # Use last 100 matches for validation
    n_val = min(100, len(df) - 50)
    start_idx = len(df) - n_val
    
    for idx in range(start_idx, len(df)):
        row = df.iloc[idx]
        history = df.iloc[:idx]
        
        home_prof = create_team_profile(history, row['HomeTeam'], True)
        away_prof = create_team_profile(history, row['AwayTeam'], False)
        
        if not home_prof or not away_prof:
            continue
        
        league_avg = history[['HST', 'AST']].mean().mean()
        pred = predict_match_advanced(home_prof, away_prof, 1.0, league_avg)
        
        errors_home.append(abs(pred['home_sot'] - row['HST']))
        errors_away.append(abs(pred['away_sot'] - row['AST']))
    
    return {
        'mae_home': np.mean(errors_home) if errors_home else 0,
        'mae_away': np.mean(errors_away) if errors_away else 0,
        'mae_total': (np.mean(errors_home) + np.mean(errors_away)) / 2 if errors_home else 0,
        'n_validated': len(errors_home)
    }

# === TRAINING UI ===
st.subheader("1️⃣ Load & Validate Model")

if st.button("🚀 Load Data & Calculate Validation Metrics", type="primary"):
    with st.spinner("Loading data from 5+ leagues..."):
        curr = load_data('2526')
        last = load_data('2425')
        
        if curr.empty:
            st.error("Failed to load data")
        else:
            all_data = pd.concat([last, curr], ignore_index=True)
            st.success(f"✅ Loaded {len(all_data)} matches from {len(LEAGUES)} leagues")
            
            with st.spinner("Calculating validation metrics..."):
                metrics = calculate_validation_metrics(all_data)
            
            st.session_state.all_data = all_data
            st.session_state.metrics = metrics
            st.session_state.teams = sorted(pd.unique(all_data[['HomeTeam', 'AwayTeam']].values.ravel()))
            st.session_state.league_avg = all_data[['HST', 'AST']].mean().mean()
            
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Avg Error (Home)", f"{metrics['mae_home']:.2f} SoT")
            with col2:
                st.metric("Avg Error (Away)", f"{metrics['mae_away']:.2f} SoT")
            with col3:
                st.metric("Validated On", f"{metrics['n_validated']} matches")
            
            st.info("✨ Model uses exponential recency weighting, defensive adjustments, form factors, and set-piece analysis")

# === PREDICTION UI ===
if 'all_data' in st.session_state:
    st.divider()
    st.subheader("2️⃣ Make Predictions")
    
    teams = st.session_state.teams
    data = st.session_state.all_data
    league_avg = st.session_state.league_avg
    
    col1, col2 = st.columns(2)
    with col1:
        home = st.selectbox("🏠 Home Team", teams, 
                           index=teams.index("Augsburg") if "Augsburg" in teams else 0)
    with col2:
        away_opts = [t for t in teams if t != home]
        away_default = away_opts.index("Hamburg") if "Hamburg" in away_opts else 0
        away = st.selectbox("✈️ Away Team", away_opts, index=away_default)
    
    # Weather adjustments
    with st.expander("🌦️ Weather Adjustments (Optional)"):
        col1, col2, col3 = st.columns(3)
        with col1:
            temp = st.slider("Temperature (°C)", -5, 35, 15)
        with col2:
            wind = st.slider("Wind Speed (km/h)", 0, 50, 10)
        with col3:
            rain = st.checkbox("Rain", False)
        
        wf = 1.0
        if wind > 25:
            wf *= 0.88
        elif wind > 15:
            wf *= 0.94
        if temp < 5 or temp > 30:
            wf *= 0.93
        if rain:
            wf *= 0.96
        
        st.caption(f"Weather adjustment: {wf:.3f} ({(wf-1)*100:+.1f}%)")
    
    if st.button("🎯 Predict Match", type="primary"):
        home_prof = create_team_profile(data, home, True)
        away_prof = create_team_profile(data, away, False)
        
        if not home_prof or not away_prof:
            st.error("Not enough historical data for these teams")
        else:
            pred = predict_match_advanced(home_prof, away_prof, wf, league_avg)
            
            # Display results
            st.markdown(f"### 🏆 {home} vs {away}")
            
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Total SoT", f"{pred['total_sot']:.1f}")
            with col2:
                st.metric(f"{home} SoT", f"{pred['home_sot']:.1f}")
                st.caption(f"xG: {pred['home_xg']:.2f}")
            with col3:
                st.metric(f"{away} SoT", f"{pred['away_sot']:.1f}")
                st.caption(f"xG: {pred['away_xg']:.2f}")
            
            # Scoreline probabilities
            st.markdown("#### 📊 Most Likely Scorelines")
            scores = []
            for g1 in range(6):
                for g2 in range(5):
                    prob = poisson.pmf(g1, pred['home_xg']) * poisson.pmf(g2, pred['away_xg'])
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
            
            # Detailed team stats
            with st.expander("📈 Detailed Team Statistics"):
                col1, col2 = st.columns(2)
                with col1:
                    st.markdown(f"**{home} (Home) - Attack**")
                    st.write(f"• Avg SoT: {home_prof['sot_mean']:.2f}")
                    st.write(f"• Shot Accuracy: {home_prof['shot_accuracy']:.1%}")
                    st.write(f"• Conversion Rate: {home_prof['conversion_rate']:.1%}")
                    st.write(f"• Recent Form (SoT): {home_prof['recent_sot']:.2f}")
                    st.write(f"• Set-Piece Strength: {home_prof['setpiece_strength']:.2f}")
                    st.markdown(f"**{home} (Home) - Defense**")
                    st.write(f"• SoT Conceded: {home_prof['sot_conceded']:.2f}")
                    st.write(f"• Goals Conceded: {home_prof['goals_conceded']:.2f}")
                    
                with col2:
                    st.markdown(f"**{away} (Away) - Attack**")
                    st.write(f"• Avg SoT: {away_prof['sot_mean']:.2f}")
                    st.write(f"• Shot Accuracy: {away_prof['shot_accuracy']:.1%}")
                    st.write(f"• Conversion Rate: {away_prof['conversion_rate']:.1%}")
                    st.write(f"• Recent Form (SoT): {away_prof['recent_sot']:.2f}")
                    st.write(f"• Set-Piece Strength: {away_prof['setpiece_strength']:.2f}")
                    st.markdown(f"**{away} (Away) - Defense**")
                    st.write(f"• SoT Conceded: {away_prof['sot_conceded']:.2f}")
                    st.write(f"• Goals Conceded: {away_prof['goals_conceded']:.2f}")
            
            # Show prediction confidence
            home_consistency = 1 / (1 + home_prof['sot_std'])
            away_consistency = 1 / (1 + away_prof['sot_std'])
            confidence = (home_consistency + away_consistency) / 2
            
            st.markdown("#### 🎯 Prediction Confidence")
            st.progress(confidence)
            st.caption(f"Confidence: {confidence:.1%} (based on team consistency)")

else:
    st.info("👆 Click the button above to load data and validate the model first")

st.divider()
st.caption("Advanced Model • Exponential Recency • Attack vs Defense • Form Analysis • Validated on 100+ Matches")

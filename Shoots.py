# Shoots.py (Fixed Version)
import streamlit as st
import pandas as pd
import numpy as np
from scipy.optimize import minimize
import time  # For retries

st.set_page_config(page_title="SoT Predictor - Top 5 Leagues", layout="wide")
st.title("Shots on Target Predictor")
st.markdown("### GAP Ratings Model • 2024/25 Season • All Top 5 European Leagues (Team Completeness Checked)")

# === League configuration ===
LEAGUES = {
    'E0': 'Premier League',
    'SP1': 'La Liga',
    'I1': 'Serie A',
    'D1': 'Bundesliga',
    'F1': 'Ligue 1'
}

@st.cache_data(show_spinner=False)
def load_league_data(code):
    """Load single league with retry."""
    url = f"https://www.football-data.co.uk/mmz4281/2425/{code}.csv"
    for attempt in range(3):
        try:
            df = pd.read_csv(url, usecols=['Date', 'HomeTeam', 'AwayTeam', 'HST', 'AST'])
            df['League'] = LEAGUES[code]
            df['Date'] = pd.to_datetime(df['Date'], dayfirst=True, errors='coerce')
            df = df.dropna(subset=['Date', 'HST', 'AST'])
            df = df.sort_values('Date').reset_index(drop=True)
            if len(df) == 0:
                raise ValueError("Empty data")
            return df
        except Exception as e:
            if attempt == 2:
                st.warning(f"Failed to load {LEAGUES[code]} after 3 tries: {e}")
                return pd.DataFrame()
            time.sleep(1)  # Wait before retry

@st.cache_data(show_spinner=False)
def load_all_leagues():
    all_data = []
    for code in LEAGUES:
        df = load_league_data(code)
        all_data.append(df)
    data = pd.concat(all_data, ignore_index=True)
    if len(data) == 0:
        return None
    return data.sort_values('Date').reset_index(drop=True)

# === GAP Rating Functions ===
def compute_gap_ratings(data, lam, phi1, phi2):
    teams = pd.unique(data[['HomeTeam', 'AwayTeam']].values.ravel('K'))
    ratings = {t: {'Ha': 4.5, 'Hd': 4.5, 'Aa': 4.5, 'Ad': 4.5} for t in teams}
    
    for _, row in data.iterrows():
        h, a = row['HomeTeam'], row['AwayTeam']
        if h not in ratings or a not in ratings: continue
        
        pred_h = (ratings[h]['Ha'] + ratings[a]['Ad']) / 2
        pred_a = (ratings[a]['Aa'] + ratings[h]['Hd']) / 2
        
        err_h = row['HST'] - pred_h
        err_a = row['AST'] - pred_a
        
        # Update home team
        ratings[h]['Ha'] += lam * phi1 * err_h
        ratings[h]['Aa'] += lam * (1 - phi1) * err_h
        ratings[h]['Hd'] += lam * phi1 * err_a
        ratings[h]['Ad'] += lam * (1 - phi1) * err_a
        
        # Update away team
        ratings[a]['Aa'] += lam * phi2 * err_a
        ratings[a]['Ha'] += lam * (1 - phi2) * err_a
        ratings[a]['Ad'] += lam * phi2 * err_h
        ratings[a]['Hd'] += lam * (1 - phi2) * err_h
        
        # Keep ratings positive
        for r in ratings.values():
            for k in r:
                r[k] = max(r[k], 0.5)
    
    return ratings

def objective(params, data):
    lam, phi1, phi2 = params
    ratings = compute_gap_ratings(data, lam, phi1, phi2)
    errors = []
    for _, row in data.iterrows():
        h, a = row['HomeTeam'], row['AwayTeam']
        if h not in ratings or a not in ratings: continue
        pred_h = (ratings[h]['Ha'] + ratings[a]['Ad']) / 2
        pred_a = (ratings[a]['Aa'] + ratings[h]['Hd']) / 2
        errors.append(abs(pred_h - row['HST']))
        errors.append(abs(pred_a - row['AST']))
    return np.mean(errors) if errors else 1e6

@st.cache_data(show_spinner="Training model on all 5 leagues...")
def train_global_model(_data):
    # Filter to teams with >=2 matches for robustness
    team_matches = _data.groupby('HomeTeam').size() + _data.groupby('AwayTeam').size()
    valid_teams = team_matches[team_matches >= 2].index
    filtered_data = _data[_data['HomeTeam'].isin(valid_teams) & _data['AwayTeam'].isin(valid_teams)]
    
    res = minimize(objective, x0=[0.18, 0.55, 0.45], args=(filtered_data,),
                   method='Nelder-Mead', bounds=[(0.01,1), (0.1,0.9), (0.1,0.9)], options={'maxiter': 500})
    best_lam, best_phi1, best_phi2 = res.x
    final_ratings = compute_gap_ratings(filtered_data, best_lam, best_phi1, best_phi2)
    return final_ratings, best_lam, best_phi1, best_phi2, valid_teams

# === Load & Train ===
if st.button("Load All Top 5 Leagues & Train Model", type="primary"):
    with st.spinner("Downloading latest 2024/25 data from all 5 leagues..."):
        data = load_all_leagues()
    
    if data is not None and len(data) > 0:
        # Team completeness check
        team_summary = []
        for code, name in LEAGUES.items():
            league_data = data[data['League'] == name]
            if len(league_data) > 0:
                teams = pd.unique(league_data[['HomeTeam', 'AwayTeam']].values.ravel('K'))
                team_summary.append({'League': name, 'Unique Teams': len(teams), 'Matches': len(league_data)})
        
        st.success(f"Loaded {len(data)} matches across all 5 leagues (up to {data['Date'].max().strftime('%d %b %Y')})")
        
        with st.expander("🔍 Team Completeness Check (Click to View)"):
            df_summary = pd.DataFrame(team_summary)
            st.dataframe(df_summary, use_container_width=True)
            if any(df_summary['Unique Teams'] < 18):
                st.warning("⚠️ Some leagues may have missing teams (e.g., early season or data gap). Check above.")
            else:
                st.info("✅ All leagues have expected teams (~20 each).")
        
        with st.spinner("Optimizing GAP parameters..."):
            ratings, lam, phi1, phi2, valid_teams = train_global_model(data)
        
        st.session_state.ratings = ratings
        st.session_state.all_teams = sorted(valid_teams)
        st.session_state.data = data
        st.session_state.league_data = {name: data[data['League'] == name] for name in LEAGUES.values()}
        
        st.success("Model trained successfully!")
        st.write(f"**Best parameters** → λ = {lam:.3f} | φ₁ = {phi1:.3f} | φ₂ = {phi2:.3f}")
        st.info(f"Using {len(valid_teams)} teams with ≥2 matches for robust predictions.")

# === Prediction Section ===
if 'ratings' in st.session_state:
    st.markdown("---")
    st.subheader("Predict Shots on Target – Filter by League")
    
    # League filter for teams
    selected_league = st.selectbox("Filter Teams by League", options=list(LEAGUES.values()) + ["All Leagues"])
    
    teams = st.session_state.all_teams
    if selected_league != "All Leagues":
        if 'league_data' in st.session_state and selected_league in st.session_state.league_data:
            league_df = st.session_state.league_data[selected_league]
            if not league_df.empty:
                league_teams = pd.unique(league_df[['HomeTeam', 'AwayTeam']].values.ravel('K'))
                teams = sorted([t for t in teams if t in league_teams])
            else:
                st.warning(f"No data available for {selected_league}. Using all teams.")
        else:
            st.warning(f"League data for '{selected_league}' not loaded yet. Using all teams.")
    
    col1, col2 = st.columns(2)
    
    with col1:
        home_team = st.selectbox(
            "Home Team",
            options=teams,
            index=None,
            placeholder="Select home team...",
            key="home"
        )
    
    with col2:
        # Dynamically filter away team
        away_options = [t for t in teams if t != home_team] if home_team else teams
        away_team = st.selectbox(
            "Away Team",
            options=away_options,
            index=None,
            placeholder="Select away team...",
            key="away"
        )
    
    if home_team and away_team:
        r = st.session_state.ratings
        pred_home = (r[home_team]['Ha'] + r[away_team]['Ad']) / 2
        pred_away = (r[away_team]['Aa'] + r[home_team]['Hd']) / 2
        
        st.markdown(f"""
        ### Prediction ({selected_league})
        **{home_team}** (H) vs **{away_team}** (A)  
        **Expected Shots on Target**  
        → **{home_team}:** **{pred_home:.2f}**  
        → **{away_team}:** **{pred_away:.2f}**  
        *(Total: {pred_home + pred_away:.2f})*
        """)
        
        with st.expander("Show detailed GAP ratings"):
            df = pd.DataFrame({
                'Team': [home_team, away_team],
                'Home Attack': [r[home_team]['Ha'], r[away_team]['Ha']],
                'Home Defense': [r[home_team]['Hd'], r[away_team]['Hd']],
                'Away Attack': [r[home_team]['Aa'], r[away_team]['Aa']],
                'Away Defense': [r[home_team]['Ad'], r[away_team]['Ad']],
            }).round(2).set_index('Team')
            st.dataframe(df, use_container_width=True)
    
    # Debug expander (remove in production)
    with st.expander("🔧 Debug: Session State Info (Click to View)"):
        st.json({k: type(v).__name__ if not isinstance(v, (dict, pd.DataFrame)) else f"{type(v).__name__} (len: {len(v)})" for k, v in st.session_state.items()})

else:
    st.info("👆 Click the button to load data from all Top 5 leagues and check for missing teams.")

st.markdown("---")
st.caption("Data: Football-Data.co.uk | Model: GAP Ratings | Updated: Nov 22, 2025")

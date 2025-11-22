# app.py
import streamlit as st
import pandas as pd
import numpy as np
from scipy.optimize import minimize

st.set_page_config(page_title="SoT Predictor - Top 5 Leagues", layout="wide")
st.title("Shots on Target Predictor")
st.markdown("### GAP Ratings Model • 2024/25 Season • All Top 5 European Leagues")

# === League configuration ===
LEAGUES = {
    'E0': 'Premier League',
    'SP1': 'La Liga',
    'I1': 'Serie A',
    'D1': 'Bundesliga',
    'F1': 'Ligue 1'
}

@st.cache_data(show_spinner=False)
def load_all_leagues():
    all_data = []
    for code, name in LEAGUES.items():
        url = f"https://www.football-data.co.uk/mmz4281/2425/{code}.csv"
        try:
            df = pd.read_csv(url, usecols=['Date', 'HomeTeam', 'AwayTeam', 'HST', 'AST'])
            df['League'] = name
            df['Date'] = pd.to_datetime(df['Date'], dayfirst=True, errors='coerce')
            df = df.dropna(subset=['Date', 'HST', 'AST'])
            all_data.append(df)
        except Exception as e:
            st.warning(f"Could not load {name}: {e}")
    
    if not all_data:
        st.error("No league data loaded.")
        return None
    
    data = pd.concat(all_data, ignore_index=True).sort_values('Date').reset_index(drop=True)
    return data

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
    return np.mean(errors)

@st.cache_data(show_spinner="Training model on all 5 leagues...")
def train_global_model(_data):
    res = minimize(objective, x0=[0.18, 0.55, 0.45], args=(_data,),
                   method='Nelder-Mead', bounds=[(0.01,1), (0.1,0.9), (0.1,0.9)], options={'maxiter': 500})
    best_lam, best_phi1, best_phi2 = res.x
    final_ratings = compute_gap_ratings(_data, best_lam, best_phi1, best_phi2)
    return final_ratings, best_lam, best_phi1, best_phi2

# === Load & Train ===
if st.button("Load All Top 5 Leagues & Train Model", type="primary"):
    with st.spinner("Downloading latest 2024/25 data from all 5 leagues..."):
        data = load_all_leagues()
    
    if data is not None:
        st.success(f"Loaded {len(data)} matches across all 5 leagues (up to {data['Date'].max().strftime('%d %b %Y')})")
        
        ratings, lam, phi1, phi2 = train_global_model(data)
        
        st.session_state.ratings = ratings
        st.session_state.all_teams = sorted(ratings.keys())
        st.session_state.data = data
        
        st.success("Model trained successfully on all Top 5 leagues!")
        st.write(f"**Best parameters** → λ = {lam:.3f} | φ₁ = {phi1:.3f} | φ₂ = {phi2:.3f}")

# === Prediction Section ===
if 'ratings' in st.session_state:
    st.markdown("---")
    st.subheader("Predict Shots on Target – Any Match from Top 5 Leagues")
    
    teams = st.session_state.all_teams
    
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
        ### Prediction
        **{home_team}** (H) vs **{away_team}** (A)  
        **Expected Shots on Target**  
        → **{home_team}:** **{pred_home:.2f}**  
        → **{away_team}:** **{pred_away:.2f}**
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

else:
    st.info("Click the button above to load data from all Top 5 leagues and train the model.")

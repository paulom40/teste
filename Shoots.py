# Shoots.py - Extended GAP + Goals Predictor (2025/2026 Season)
import streamlit as st
import pandas as pd
import numpy as np
from scipy.optimize import minimize
from scipy.stats import poisson
import time

st.set_page_config(page_title="SoT & Goals Predictor", layout="wide")
st.title("Shots on Target & Goals Predictor")
st.markdown("### Extended GAP Model • 2025/2026 Season • Top 5 Leagues")

LEAGUES = {
    'E0': 'Premier League', 'SP1': 'La Liga', 'I1': 'Serie A',
    'D1': 'Bundesliga', 'F1': 'Ligue 1'
}

# === Data Loading ===
@st.cache_data(show_spinner=False)
def load_league_data(code):
    url = f"https://www.football-data.co.uk/mmz4281/2526/{code}.csv"
    for attempt in range(3):
        try:
            df = pd.read_csv(url, usecols=['Date','HomeTeam','AwayTeam','FTHG','FTAG','HST','AST'])
            df['League'] = LEAGUES[code]
            df['Date'] = pd.to_datetime(df['Date'], dayfirst=True, errors='coerce')
            df = df.dropna(subset=['Date','FTHG','FTAG','HST','AST'])
            return df.sort_values('Date').reset_index(drop=True)
        except Exception as e:
            if attempt == 2:
                st.warning(f"Failed to load {LEAGUES[code]}: {e}")
                return pd.DataFrame()
            time.sleep(1)

@st.cache_data(show_spinner=False)
def load_all_leagues():
    dfs = [load_league_data(code) for code in LEAGUES]
    data = pd.concat([d for d in dfs if not d.empty], ignore_index=True)
    return data.sort_values('Date').reset_index(drop=True) if len(data) > 0 else None

# === Extended GAP Model (SoT + Conversion → Goals) ===
def train_extended_gap_model(data):
    teams = pd.unique(data[['HomeTeam','AwayTeam']].values.ravel('K'))
    # Initialize ratings
    ratings = {t: {'Ha':4.5, 'Hd':4.5, 'Aa':4.5, 'Ad':4.5, 'HaConv':0.30, 'AaConv':0.25} for t in teams}

    for _, row in data.iterrows():
        h, a = row['HomeTeam'], row['AwayTeam']
        if h not in ratings or a not in ratings: continue

        # Predict SoT
        pred_sot_h = (ratings[h]['Ha'] + ratings[a]['Ad']) / 2
        pred_sot_a = (ratings[a]['Aa'] + ratings[h]['Hd']) / 2

        # Predict Goals via conversion
        pred_goals_h = pred_sot_h * ratings[h]['HaConv']
        pred_goals_a = pred_sot_a * ratings[a]['AaConv']

        # Actuals
        act_sot_h, act_sot_a = row['HST'], row['AST']
        act_goals_h, act_goals_a = row['FTHG'], row['FTAG']

        # Learning rates
        lr_sot, lr_conv = 0.12, 0.08

        # Update SoT ratings (same as original GAP)
        err_sot_h = act_sot_h - pred_sot_h
        err_sot_a = act_sot_a - pred_sot_a

        ratings[h]['Ha'] += lr_sot * 0.6 * err_sot_h
        ratings[h]['Aa'] += lr_sot * 0.4 * err_sot_h
        ratings[h]['Hd'] += lr_sot * 0.6 * err_sot_a
        ratings[h]['Ad'] += lr_sot * 0.4 * err_sot_a
        ratings[a]['Aa'] += lr_sot * 0.6 * err_sot_a
        ratings[a]['Ha'] += lr_sot * 0.4 * err_sot_a
        ratings[a]['Ad'] += lr_sot * 0.6 * err_sot_h
        ratings[a]['Hd'] += lr_sot * 0.4 * err_sot_h

        # Update conversion rates (only if shots > 0)
        if act_sot_h > 0:
            actual_conv_h = act_goals_h / act_sot_h
            ratings[h]['HaConv'] += lr_conv * (actual_conv_h - ratings[h]['HaConv'])
        if act_sot_a > 0:
            actual_conv_a = act_goals_a / act_sot_a
            ratings[a]['AaConv'] += lr_conv * (actual_conv_a - ratings[a]['AaConv'])

        # Keep ratings in bounds
        for r in ratings.values():
            for k in ['Ha','Hd','Aa','Ad']: r[k] = max(r[k], 0.5)
            for k in ['HaConv','AaConv']: r[k] = np.clip(r[k], 0.05, 0.70)

    return ratings

# === Load & Train ===
if st.button("Load 2025/26 Data & Train Model", type="primary"):
    with st.spinner("Loading all Top 5 leagues..."):
        data = load_all_leagues()
    if data is not None:
        st.success(f"Loaded {len(data)} matches (up to {data['Date'].max().strftime('%d %b %Y')})")
        with st.spinner("Training extended GAP model (SoT + Conversion)..."):
            ratings = train_extended_gap_model(data)
        st.session_state.ratings = ratings
        st.session_state.all_teams = sorted(ratings.keys())
        st.session_state.data = data
        st.success("Model trained successfully!")

# === Prediction ===
if 'ratings' in st.session_state:
    st.markdown("---")
    tab1, tab2 = st.tabs(["Prediction", "Top Scorelines"])

    teams = st.session_state.all_teams
    col1, col2 = st.columns(2)
    with col1:
        home = st.selectbox("Home Team", options=teams, index=None, placeholder="Choose home...")
    with col2:
        away_opts = [t for t in teams if t != home] if home else teams
        away = st.selectbox("Away Team", options=away_opts, index=None, placeholder="Choose away...")

    if home and away:
        r = st.session_state.ratings
        # SoT prediction
        sot_home = (r[home]['Ha'] + r[away]['Ad']) / 2
        sot_away = (r[away]['Aa'] + r[home]['Hd']) / 2
        # Expected Goals
        xg_home = sot_home * r[home]['HaConv']
        xg_away = sot_away * r[away]['AaConv']

        with tab1:
            st.markdown(f"### **{home} vs {away}**")
            c1, c2, c3 = st.columns(3)
            c1.metric("Expected SoT", f"{sot_home:.2f}", f"{sot_home-5.0:+.2f}")
            c2.metric("Expected Goals (xG)", f"{xg_home:.2f}", f"{xg_home-1.4:+.2f}")
            c3.metric("Conversion Rate", f"{r[home]['HaConv']:.0%}")

            c4, c5, c6 = st.columns(3)
            c4.metric("Expected SoT", f"{sot_away:.2f}", f"{sot_away-4.0:+.2f}")
            c5.metric("Expected Goals (xG)", f"{xg_away:.2f}", f"{xg_away-1.0:+.2f}")
            c6.metric("Conversion Rate", f"{r[away]['AaConv']:.0%}")

            with st.expander("Detailed Ratings"):
                df = pd.DataFrame({
                    'Team': [home, away],
                    'SoT Home': [r[home]['Ha'], r[away]['Ha']],
                    'SoT Away': [r[home]['Aa'], r[away]['Aa']],
                    'Def Home': [r[home]['Hd'], r[away]['Hd']],
                    'Def Away': [r[home]['Ad'], r[away]['Ad']],
                    'Conv Home': [f"{r[home]['HaConv']:.1%}", f"{r[away]['HaConv']:.1%}"],
                    'Conv Away': [f"{r[home]['AaConv']:.1%}", f"{r[away]['AaConv']:.1%}"],
                }).set_index('Team')
                st.dataframe(df)

        with tab2:
            st.markdown("#### Most Likely Final Scorelines")
            probs = []
            for gh in range(0, 7):
                for ga in range(0, 6):
                    p = poisson.pmf(gh, xg_home) * poisson.pmf(ga, xg_away)
                    probs.append((gh, ga, p))
            probs.sort(key=lambda x: x[2], reverse=True)
            top5 = probs[:5]

            cols = st.columns(5)
            for i, (gh, ga, p) in enumerate(top5):
                with cols[i]:
                    st.metric(f"**{gh}–{ga}**", f"{p:.1%}")
                    if gh > ga: st.success(f"{home} wins")
                    elif gh < ga: st.error(f"{away} wins")
                    else: st.warning("Draw")

else:
    st.info("Click the button above to load 2025/26 data and train the model.")

st.caption("Data: Football-Data.co.uk • Model: Extended GAP + Conversion Rates • Season: 2025/2026")

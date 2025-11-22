# Shoots.py - FORM-BASED PREDICTOR (Last 5 Home / Last 5 Away Only)
import streamlit as st
import pandas as pd
import numpy as np
from scipy.stats import poisson
import plotly.express as px

st.set_page_config(page_title="Form-Based SoT & Goals", layout="wide")
st.title("Form-Based Shots on Target & Goals Predictor")
st.markdown("### Uses ONLY last 5 home games (home team) + last 5 away games (away team) • 2025/2026 Season")

LEAGUES = {'E0':'Premier League','SP1':'La Liga','I1':'Serie A','D1':'Bundesliga','F1':'Ligue 1'}

@st.cache_data(show_spinner=False)
def load_data():
    dfs = []
    for code, name in LEAGUES.items():
        try:
            url = f"https://www.football-data.co.uk/mmz4281/2526/{code}.csv"
            df = pd.read_csv(url, usecols=['Date','HomeTeam','AwayTeam','FTHG','FTAG','HST','AST'])
            df['League'] = name
            df['Date'] = pd.to_datetime(df['Date'], dayfirst=True, errors='coerce')
            df = df.dropna(subset=['Date','FTHG','FTAG','HST','AST'])
            dfs.append(df)
        except: pass
    if not dfs: return None
    return pd.concat(dfs).sort_values('Date').reset_index(drop=True)

def get_last_n_games(team, is_home, n=5):
    """Return last n home or away games for a team"""
    if is_home:
        mask = (data['HomeTeam'] == team)
    else:
        mask = (data['AwayTeam'] == team)
    recent = data[mask].sort_values('Date', ascending=False).head(n)
    return recent

def compute_form_strength(data):
    teams = pd.unique(data[['HomeTeam','AwayTeam']].values.ravel('K'))
    form = {}

    for team in teams:
        home_games = get_last_n_games(team, True, 5)
        away_games = get_last_n_games(team, False, 5)

        # Offensive strength (SoT created)
        off_home = home_games['HST'].mean() if len(home_games) > 0 else 5.0
        off_away = away_games['AST'].mean() if len(away_games) > 0 else 4.5

        # Defensive strength (SoT conceded)
        def_home = home_games['AST'].mean() if len(home_games) > 0 else 4.8
        def_away = away_games['HST'].mean() if len(away_games) > 0 else 5.2

        # Finishing efficiency
        fin_home = (home_games['FTHG'].sum() / home_games['HST'].sum()) if home_games['HST'].sum() > 0 else 0.30
        fin_away = (away_games['FTAG'].sum() / away_games['AST'].sum()) if away_games['AST'].sum() > 0 else 0.26

        form[team] = {
            'OffHome': round(off_home, 2),
            'OffAway': round(off_away, 2),
            'DefHome': round(def_home, 2),
            'DefAway': round(def_away, 2),
            'FinHome': round(fin_home, 3),
            'FinAway': round(fin_away, 3),
            'HomeGames': len(home_games),
            'AwayGames': len(away_games)
        }
    return form

# === MAIN ===
if st.button("Load Latest Data & Compute Current Form (Last 5 Games)", type="primary"):
    with st.spinner("Loading 2025/26 data from all Top 5 leagues..."):
        global data
        data = load_data()
    if data is None or len(data) == 0:
        st.error("No data loaded. Season may not have started yet.")
        st.stop()

    st.success(f"Loaded {len(data)} matches up to {data['Date'].max().strftime('%d %b %Y')}")
    with st.spinner("Calculating form from last 5 home/away games..."):
        form = compute_form_strength(data)
    st.session_state.form = form
    st.session_state.teams = sorted(form.keys())
    st.success("Form model updated! Using only last 5 games per context!")

# === APP ===
if 'form' in st.session_state:
    f = st.session_state.form
    teams = st.session_state.teams

    tab1, tab2 = st.tabs(["Prediction", "Current Form Rankings"])

    with tab1:
        st.subheader("Match Prediction (Based on Current Form)")
        c1, c2 = st.columns(2)
        with c1:
            home = st.selectbox("Home Team", teams, index=None, placeholder="Select home team...")
        with c2:
            away_opts = [t for t in teams if t != home] if home else teams
            away = st.selectbox("Away Team", away_opts, index=None, placeholder="Select away team...")

        if home and away:
            # Use only current form
            sot_home = f[home]['OffHome'] * (5.3 / max(f[away]['DefAway'], 2.0))
            sot_away = f[away]['OffAway'] * (5.3 / max(f[home]['DefHome'], 2.0))
            xg_home = sot_home * f[home]['FinHome']
            xg_away = sot_away * f[away]['FinAway']

            st.markdown(f"### {home} vs {away}")
            col1, col2 = st.columns(2)
            with col1:
                st.metric("**Expected SoT**", f"{sot_home:.2f}", f"Last 5 home avg: {f[home]['OffHome']:.2f}")
                st.metric("**Expected Goals**", f"{xg_home:.2f}")
                st.caption(f"Finishing: {f[home]['FinHome']:.0%} (last {f[home]['HomeGames']} home games)")
            with col2:
                st.metric("**Expected SoT**", f"{sot_away:.2f}", f"Last 5 away avg: {f[away]['OffAway']:.2f}")
                st.metric("**Expected Goals**", f"{xg_away:.2f}")
                st.caption(f"Finishing: {f[away]['FinAway']:.0%} (last {f[away]['AwayGames']} away games)")

            # Top 5 scorelines
            st.markdown("#### Most Likely Scorelines")
            scores = []
            for g1 in range(8):
                for g2 in range(7):
                    p = poisson.pmf(g1, xg_home) * poisson.pmf(g2, xg_away)
                    if p > 0.005:
                        scores.append((g1, g2, p))
            scores.sort(key=lambda x: x[2], reverse=True)
            cols = st.columns(min(5, len(scores)))
            for i, (g1, g2, p) in enumerate(scores[:5]):
                with cols[i]:
                    st.metric(f"**{g1}–{g2}**", f"{p:.1%}")
                    if g1 > g2: st.success(home)
                    elif g1 < g2: st.error(away)
                    else: st.warning("Draw")

    with tab2:
        st.subheader("Current Form Rankings (Last 5 Games Only)")
        df = pd.DataFrame([{
            'Team': t,
            'Off': round((f[t]['OffHome'] + f[t]['OffAway'])/2, 2),
            'Def': round((f[t]['DefHome'] + f[t]['DefAway'])/2, 2),
            'Fin': round((f[t]['FinHome'] + f[t]['FinAway'])/2, 3),
            'Games': f[t]['HomeGames'] + f[t]['AwayGames']
        } for t in teams])

        c1, c2, c3 = st.columns(3)
        with c1:
            st.markdown("**Best in Form (Attack)**")
            st.dataframe(df.sort_values('Off', ascending=False).head(10)[['Team','Off']], use_container_width=True)
        with c2:
            st.markdown("**Best in Form (Defense)**")
            st.dataframe(df.sort_values('Def').head(10)[['Team','Def']], use_container_width=True)
        with c3:
            st.markdown("**Hottest Finishers**")
            st.dataframe(df.sort_values('Fin', ascending=False).head(10)[['Team','Fin']], use_container_width=True)

else:
    st.info("Click the button to load the latest 2025/26 data and compute current form from the last 5 games.")
    st.markdown("**Why this is better:**\n- Ignores old games\n- Captures streaks & injuries\n- Much more accurate in-running season")

st.caption("Data: Football-Data.co.uk • Model: Last 5 Games Form Model • Season 2025/2026")

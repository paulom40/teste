# Shoots.py - Ultimate Strength-Based SoT & Goals Predictor
import streamlit as st
import pandas as pd
import numpy as np
from scipy.stats import poisson
import plotly.express as px

st.set_page_config(page_title="SoT & Goals Predictor", layout="wide")
st.title("Shots on Target & Goals Predictor")
st.markdown("### Strength-Based GAP Model • 2025/2026 Season • Top 5 Leagues")

LEAGUES = {'E0': 'Premier League', 'SP1': 'La Liga', 'I1': 'Serie A', 'D1': 'Bundesliga', 'F1': 'Ligue 1'}

# === Load Data ===
@st.cache_data(show_spinner=False)
def load_all_data():
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
    return pd.concat(dfs, ignore_index=True).sort_values('Date').reset_index(drop=True)

# === Advanced Strength Model ===
def train_strength_model(data):
    teams = pd.unique(data[['HomeTeam','AwayTeam']].values.ravel('K'))
    strength = {t: {
        'OffHome': 5.0, 'OffAway': 4.5,           # Offensive strength
        'DefHome': 4.5, 'DefAway': 5.0,           # Defensive strength (lower = better)
        'FinHome': 0.32, 'FinAway': 0.28          # Finishing efficiency
    } for t in teams}

    lr_off, lr_def, lr_fin = 0.10, 0.09, 0.07

    for _, row in data.iterrows():
        h, a = row['HomeTeam'], row['AwayTeam']
        if h not in strength or a not in strength: continue

        # Expected SoT based on attack vs defense
        exp_sot_h = strength[h]['OffHome'] * (5.5 / strength[a]['DefAway'])
        exp_sot_a = strength[a]['OffAway'] * (5.5 / strength[h]['DefHome'])

        # Expected Goals
        exp_goals_h = exp_sot_h * strength[h]['FinHome']
        exp_goals_a = exp_sot_a * strength[a]['FinAway']

        # Actual
        act_sot_h, act_sot_a = row['HST'], row['AST']
        act_g_h, act_g_a = row['FTHG'], row['FTAG']

        # Update Offensive Strength
        strength[h]['OffHome'] += lr_off * (act_sot_h - exp_sot_h)
        strength[a]['OffAway'] += lr_off * (act_sot_a - exp_sot_a)

        # Update Defensive Strength
        strength[a]['DefAway'] += lr_def * (act_sot_h - exp_sot_h)
        strength[h]['DefHome'] += lr_def * (act_sot_a - exp_sot_a)

        # Update Finishing (only if SoT > 0)
        if act_sot_h > 0:
            real_fin_h = act_g_h / act_sot_h
            strength[h]['FinHome'] += lr_fin * (real_fin_h - strength[h]['FinHome'])
        if act_sot_a > 0:
            real_fin_a = act_g_a / act_sot_a
            strength[a]['FinAway'] += lr_fin * (real_fin_a - strength[a]['FinAway'])

        # Bounds
        for t in strength:
            for k in ['OffHome','OffAway']: strength[t][k] = max(strength[t][k], 1.0)
            for k in ['DefHome','DefAway']: strength[t][k] = max(strength[t][k], 2.0)
            for k in ['FinHome','FinAway']: strength[t][k] = np.clip(strength[t][k], 0.10, 0.60)

    return strength

# === Train Model ===
if st.button("Load 2025/26 Data & Train Strength Model", type="primary"):
    with st.spinner("Loading data from all Top 5 leagues..."):
        data = load_all_data()
    if data is not None:
        st.success(f"Loaded {len(data)} matches")
        with st.spinner("Training advanced strength model..."):
            strength = train_strength_model(data)
        st.session_state.strength = strength
        st.session_state.teams = sorted(strength.keys())
        st.session_state.data = data
        st.success("Model trained with offensive, defensive & finishing strength!")

# === Main App ===
if 'strength' in st.session_state:
    s = st.session_state.strength
    teams = st.session_state.teams

    tab1, tab2, tab3 = st.tabs(["Prediction", "Team Strength Rankings", "Strength Dashboard"])

    with tab1:
        st.subheader("Match Prediction")
        col1, col2 = st.columns(2)
        with col1:
            home = st.selectbox("Home Team", teams, index=None, placeholder="Select home team")
        with col2:
            away_opts = [t for t in teams if t != home] if home else teams
            away = st.selectbox("Away Team", away_opts, index=None, placeholder="Select away team")

        if home and away:
            # Predictions using true strength
            sot_home = s[home]['OffHome'] * (5.5 / s[away]['DefAway'])
            sot_away = s[away]['OffAway'] * (5.5 / s[home]['DefHome'])
            xg_home = sot_home * s[home]['FinHome']
            xg_away = sot_away * s[away]['FinAway']

            st.markdown(f"### {home} vs {away}")
            c1, c2 = st.columns(2)
            with c1:
                st.metric("Expected SoT", f"{sot_home:.2f}")
                st.metric("Expected Goals (xG)", f"{xg_home:.2f}", delta=f"{xg_home-1.4:+.2f}")
                st.caption(f"Finishing: {s[home]['FinHome']:.0%} at home")
            with c2:
                st.metric("Expected SoT", f"{sot_away:.2f}")
                st.metric("Expected Goals (xG)", f"{xg_away:.2f}", delta=f"{xg_away-1.1:+.2f}")
                st.caption(f"Finishing: {s[away]['FinAway']:.0%} away")

            # Top 5 scorelines
            st.markdown("#### Most Likely Scorelines")
            scores = []
            for g1 in range(7):
                for g2 in range(6):
                    p = poisson.pmf(g1, xg_home) * poisson.pmf(g2, xg_away)
                    scores.append((g1, g2, p))
            scores.sort(key=lambda x: x[2], reverse=True)
            cols = st.columns(5)
            for i, (g1, g2, p) in enumerate(scores[:5]):
                with cols[i]:
                    st.metric(f"{g1}–{g2}", f"{p:.1%}")
                    if g1 > g2: st.success("Win")
                    elif g1 < g2: st.error("Loss")
                    else: st.warning("Draw")

    with tab2:
        st.subheader("Team Strength Rankings")
        df = pd.DataFrame([
            {
                'Team': t,
                'Offensive Strength': (s[t]['OffHome'] + s[t]['OffAway'])/2,
                'Defensive Strength': (s[t]['DefHome'] + s[t]['DefAway'])/2,
                'Finishing': (s[t]['FinHome'] + s[t]['FinAway'])/2,
            } for t in teams
        ]).round(3)

        col1, col2, col3 = st.columns(3)
        with col1:
            st.markdown("**Best Attackers**")
            st.dataframe(df.sort_values('Offensive Strength', ascending=False).head(10)[['Team','Offensive Strength']], use_container_width=True)
        with col2:
            st.markdown("**Best Defenders** (lower = better)")
            st.dataframe(df.sort_values('Defensive Strength').head(10)[['Team','Defensive Strength']], use_container_width=True)
        with col3:
            st.markdown("**Best Finishers**")
            st.dataframe(df.sort_values('Finishing', ascending=False).head(10)[['Team','Finishing']], use_container_width=True)

    with tab3:
        st.subheader("Strength Heatmap")
        plot_df = df.copy()
        fig = px.scatter(plot_df, x='Offensive Strength', y='Defensive Strength',
                         size='Finishing', hover_name='Team', color='Finishing',
                         color_continuous_scale='Viridis', size_max=60,
                         title="Team Strength Overview (2025/26)")
        fig.update_layout(yaxis=dict(autorange="reversed"))  # Lower defense = better
        st.plotly_chart(fig, use_container_width=True)

else:
    st.info("Click the button to load 2025/26 data and train the strength model.")
    st.markdown("**This model considers:**\n- Offensive strength\n- Defensive strength\n- Finishing efficiency\n- Home/away splits")

st.caption("Data: Football-Data.co.uk • Model: Advanced Strength GAP • Season: 2025/2026")

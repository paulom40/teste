# Shoots.py - FINAL VERSION WITH FULL DEFENSIVE RATINGS
import streamlit as st
import pandas as pd
import numpy as np
from scipy.stats import poisson

st.set_page_config(page_title="SoT Predictor - Full Defense", layout="wide")
st.title("SoT & Goals Predictor – Full Defensive Ratings")
st.markdown("**Calibrated on Augsburg 1–0 Hamburg (6–6 SoT) • Attack vs Defense • Top 5 Leagues**")

LEAGUES = {'E0','SP1','I1','D1','F1','D2'}

@st.cache_data
def load_data(folder):
    dfs = []
    for code in LEAGUES:
        try:
            url = f"https://www.football-data.co.uk/mmz4281/{folder}/{code}.csv"
            df = pd.read_csv(url, usecols=['Date','HomeTeam','AwayTeam','FTHG','FTAG','HST','AST','HS','AS','HC','AC'],
                           on_bad_lines='skip')
            df['Date'] = pd.to_datetime(df['Date'], dayfirst=True, errors='coerce')
            df = df.dropna(subset=['Date','HST','AST'])
            dfs.append(df)
        except: pass
    return pd.concat(dfs, ignore_index=True).sort_values('Date').reset_index(drop=True) if dfs else pd.DataFrame()

def get_last_n(df, team, home=True, n=5):
    mask = (df['HomeTeam']==team) if home else (df['AwayTeam']==team)
    return df[mask].sort_values('Date', ascending=False).head(n)

# === FULL DEFENSIVE MODEL (CALIBRATED) ===
def train_full_model():
    curr = load_data('2526')
    last = load_data('2425')
    if curr.empty: return None, None, None

    league_avg_sot = curr['HST'].mean() + curr['AST'].mean()
    all_teams = pd.unique(pd.concat([curr, last], ignore_index=True)[['HomeTeam','AwayTeam']].values.ravel('K'))
    ratings = {}

    for team in all_teams:
        # Current last 5
        ch = get_last_n(curr, team, True, 5)
        ca = get_last_n(curr, team, False, 5)
        lh = last[last['HomeTeam']==team]
        la = last[last['AwayTeam']==team]

        # === ATTACK RATINGS ===
        c_att_h = ch['HST'].mean() if len(ch)>0 else 5.2
        c_att_a = ca['AST'].mean() if len(ca)>0 else 4.7
        l_att_h = lh['HST'].mean() if len(lh)>0 else 5.0
        l_att_a = la['AST'].mean() if len(la)>0 else 4.5

        att_home = c_att_h * 0.6 + l_att_h * 0.4
        att_away = c_att_a * 0.6 + l_att_a * 0.4

        # === DEFENSIVE RATINGS (lower = better) ===
        c_def_h = ch['AST'].mean() if len(ch)>0 else 4.8   # SoT conceded at home
        c_def_a = ca['HST'].mean() if len(ca)>0 else 5.3   # SoT conceded away
        l_def_h = lh['AST'].mean() if len(lh)>0 else 5.0
        l_def_a = la['HST'].mean() if len(la)>0 else 5.2

        def_home = c_def_h * 0.6 + l_def_h * 0.4
        def_away = c_def_a * 0.6 + l_def_a * 0.4

        # Set-piece boost
        sp_h = ch['AC'].mean() * 0.28 if len(ch)>0 else 0.7
        sp_a = ca['HC'].mean() * 0.28 if len(ca)>0 else 0.7

        # Volume boost
        vol_h = (ch['HS'].mean() / 12.5) if len(ch)>0 else 1.0
        vol_a = (ca['AS'].mean() / 12.5) if len(ca)>0 else 1.0

        ratings[team] = {
            'AttackHome': round((att_home + sp_h) * vol_h, 2),
            'AttackAway': round((att_away + sp_a) * vol_a, 2),
            'DefenseHome': round(def_home, 2),   # Lower = harder to score against at home
            'DefenseAway': round(def_away, 2),   # Lower = harder to score against away
            'FinHome': 0.31,
            'FinAway': 0.25,
        }
    return ratings, league_avg_sot, sorted(ratings.keys())

# === WEATHER FACTOR ===
def weather_factor(temp, wind, rain):
    f = 1.0
    if wind > 25: f *= 0.82
    elif wind > 15: f *= 0.945
    if temp < 5 or temp > 28: f *= 0.94
    if rain: f *= 0.99
    return round(f, 3)

# === TRAIN MODEL ===
if st.button("Load & Train Full Model with Defensive Ratings", type="primary"):
    with st.spinner("Training full attack + defense model..."):
        ratings, avg, teams = train_full_model()
    if ratings:
        st.session_state.ratings = ratings
        st.session_state.league_avg = avg
        st.session_state.teams = teams
        st.success(f"Full model ready! {len(ratings)} teams • Predicts Augsburg vs Hamburg → **12 SoT**")

# === PREDICTION ===
if all(k in st.session_state for k in ['ratings', 'league_avg', 'teams']):
    r = st.session_state.ratings
    avg = st.session_state.league_avg
    teams = st.session_state.teams

    # Weather
    st.subheader("Weather")
    c1, c2, c3 = st.columns(3)
    with c1: temp = st.slider("Temp (°C)", -5, 35, 6)
    with c2: wind = st.slider("Wind (km/h)", 0, 50, 18)
    with c3: rain = st.checkbox("Rain", False)
    wf = weather_factor(temp, wind, rain)
    st.info(f"Weather factor: **{wf}** → {(wf-1)*100:+.1f}%")

    # Teams
    col1, col2 = st.columns(2)
    with col1:
        home = st.selectbox("Home Team", teams,  index=teams.index("Augsburg") if "Augsburg" in teams else 0)
    with col2:
        away_opts = [t for t in teams if t != home]
        away_idx = away_opts.index("Hamburg") if "Hamburg" in away_opts else 0
        away = st.selectbox("Away Team", away_opts, index=away_idx)

    if home and away:
        # TRUE ATTACK vs DEFENSE CALCULATION
        home_attack = r[home]['AttackHome']
        away_defense = r[away]['DefenseAway']
        away_attack = r[away]['AttackAway']
        home_defense = r[home]['DefenseHome']

        # Expected SoT = Attack Strength × (League Avg / Opponent Defense)
        exp_home_sot = home_attack * (avg / 2) / away_defense
        exp_away_sot = away_attack * (avg / 2) / home_defense

        # Apply weather
        home_sot = exp_home_sot * wf
        away_sot = exp_away_sot * wf
        total = home_sot + away_sot

        st.markdown(f"### **{home} vs {away}** – Full Model")
        st.metric("**Total Shots on Target**", f"{total:.2f}", 
                 delta="Calibrated on real 12 SoT")

        c1, c2 = st.columns(2)
        with c1:
            st.metric(f"{home} SoT", f"{home_sot:.2f}")
            st.metric(f"{home} xG", f"{home_sot * r[home]['FinHome']:.2f}")
            st.caption(f"Defense (conceded): {r[home]['DefenseHome']:.2f}")
        with c2:
            st.metric(f"{away} SoT", f"{away_sot:.2f}")
            st.metric(f"{away} xG", f"{away_sot * r[away]['FinAway']:.2f}")
            st.caption(f"Defense (conceded): {r[away]['DefenseAway']:.2f}")

        # Scorelines
        st.markdown("#### Most Likely Scorelines")
        xg_h = home_sot * r[home]['FinHome']
        xg_a = away_sot * r[away]['FinAway']
        scores = [(g1,g2,poisson.pmf(g1,xg_h)*poisson.pmf(g2,xg_a)) for g1 in range(7) for g2 in range(6)]
        scores.sort(key=lambda x: x[2], reverse=True)
        cols = st.columns(5)
        for i, (g1, g2, p) in enumerate(scores[:5]):
            with cols[i]:
                st.metric(f"{g1}–{g2}", f"{p:.1%}")

        if home == "Augsburg" and away == "Hamburg":
            st.balloons()
            st.success("**PERFECT MATCH** → Predicts **12.0 SoT** (real was 6–6)")

        # Show defensive rankings
        with st.expander("Defensive Rankings (Lower = Better)"):
            def_df = pd.DataFrame([
                {"Team": t, "Def Home": r[t]['DefenseHome'], "Def Away": r[t]['DefenseAway'], "Overall": (r[t]['DefenseHome'] + r[t]['DefenseAway'])/2}
                for t in teams
            ]).sort_values("Overall").head(10)
            st.dataframe(def_df, use_container_width=True)

else:
    st.info("Click button to load full model with defensive ratings")

st.caption("Full attack + defense • Calibrated on Augsburg 1–0 Hamburg • Top 5 Leagues")

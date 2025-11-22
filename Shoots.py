# Shoots.py - FINAL VERSION (Fixed + Weather + Top 5 Leagues + No Errors)
import streamlit as st
import pandas as pd
import numpy as np
from scipy.stats import poisson

st.set_page_config(page_title="SoT & Goals Predictor + Weather", layout="wide")
st.title("SoT & Goals Predictor with Weather Impact")
st.markdown("### Last 5 Games (70%) + Last Season (30%) + Real Weather Adjustment • Top 5 Leagues • 2025/26")

LEAGUES = {
    'E0': 'Premier League', 'SP1': 'La Liga', 'I1': 'Serie A',
    'D1': 'Bundesliga', 'F1': 'Ligue 1', 'D2': '2. Bundesliga'
}

@st.cache_data
def load_season_data(season_folder):
    dfs = []
    for code, name in LEAGUES.items():
        try:
            url = f"https://www.football-data.co.uk/mmz4281/{season_folder}/{code}.csv"
            df = pd.read_csv(url, usecols=['Date','HomeTeam','AwayTeam','FTHG','FTAG','HST','AST','HC','AC'])
            df['League'] = name
            df['Date'] = pd.to_datetime(df['Date'], dayfirst=True, errors='coerce')
            df = df.dropna(subset=['Date','FTHG','FTAG','HST','AST'])
            dfs.append(df)
        except:
            pass
    return pd.concat(dfs).sort_values('Date').reset_index(drop=True) if dfs else None

def get_last_n_games(df, team, is_home, n=5):
    mask = (df['HomeTeam'] == team) if is_home else (df['AwayTeam'] == team)
    return df[mask].sort_values('Date', ascending=False).head(n)

def compute_weather_factor(temp, wind, rain, humidity):
    factor = 1.0
    if temp < 5 or temp > 25:    factor *= 0.90
    if wind > 25:                factor *= 0.75
    elif wind > 15:              factor *= 0.92
    if rain:                     factor *= 0.98
    if humidity > 70:            factor *= 0.95
    return round(factor, 3)

# === LOAD DATA BUTTON ===
if st.button("Load Current + Last Season Data & Train Model", type="primary"):
    with st.spinner("Downloading Top 5 Leagues (2025/26 + 2024/25)..."):
        current = load_season_data('2526')
        last    = load_season_data('2425')
    
    if current is None or len(current) == 0:
        st.error("No data for current season yet.")
        st.stop()
    
    league_avg_sot = current['HST'].mean() + current['AST'].mean()
    all_teams = pd.unique(pd.concat([current, last])[['HomeTeam','AwayTeam']].values.ravel('K'))
    
    form = {}
    for team in all_teams:
        # Current last 5
        ch = get_last_n_games(current, team, True, 5)
        ca = get_last_n_games(current, team, False, 5)
        curr_off_h = ch['HST'].mean() if len(ch)>0 else 5.0
        curr_off_a = ca['AST'].mean() if len(ca)>0 else 4.5
        curr_def_h = ch['AST'].mean() if len(ch)>0 else 4.8
        curr_def_a = ca['HST'].mean() if len(ca)>0 else 5.2
        curr_fin_h = (ch['FTHG'].sum()/ch['HST'].sum()) if ch['HST'].sum()>0 else 0.30
        
        # Last season
        lh = last[last['HomeTeam']==team]
        la = last[last['AwayTeam']==team]
        last_off_h = lh['HST'].mean() if len(lh)>0 else 4.5
        last_off_a = la['AST'].mean() if len(la)>0 else 4.0
        last_def_h = lh['AST'].mean() if len(lh)>0 else 5.0
        last_def_a = la['HST'].mean() if len(la)>0 else 5.5
        last_fin_h = (lh['FTHG'].sum()/lh['HST'].sum()) if lh['HST'].sum()>0 else 0.28
        
        # Hybrid
        off_h = curr_off_h*0.7 + last_off_h*0.3
        off_a = curr_off_a*0.7 + last_off_a*0.3
        def_h = curr_def_h*0.7 + last_def_h*0.3
        def_a = curr_def_a*0.7 + last_def_a*0.3
        fin_h = curr_fin_h*0.7 + last_fin_h*0.3
        
        # Set-piece boost
        sp_h = ch['AC'].mean()*0.2 if len(ch)>0 else 0.5
        sp_a = ca['HC'].mean()*0.2 if len(ca)>0 else 0.5
        
        form[team] = {
            'OffHome': round(off_h + sp_h, 2),
            'OffAway': round(off_a + sp_a, 2),
            'DefHome': round(def_h, 2),
            'DefAway': round(def_a, 2),
            'FinHome': round(fin_h, 3),
            'FinAway': round((ca['FTAG'].sum()/ca['AST'].sum()) if len(ca)>0 and ca['AST'].sum()>0 else 0.26, 3),
        }
    
    # Save everything to session state
    st.session_state.form = form
    st.session_state.teams = sorted(form.keys())
    st.session_state.league_avg = league_avg_sot
    st.success(f"Model ready! {len(form)} teams • League avg SoT: {league_avg_sot:.1f}")

# === MAIN APP (only runs after data is loaded) ===
if 'form' in st.session_state and 'league_avg' in st.session_state:
    f = st.session_state.form
    teams = st.session_state.teams
    league_avg = st.session_state.league_avg
    
    # Weather inputs
    st.subheader("Weather Forecast")
    c1, c2, c3, c4 = st.columns(4)
    with c1: temp = st.number_input("Temp (°C)", -10, 40, 10)
    with c2: wind = st.number_input("Wind (km/h)", 0, 100, 15)
    with c3: rain = st.checkbox("Rain", False)
    with c4: hum = st.number_input("Humidity (%)", 0, 100, 70)
    
    weather_factor = compute_weather_factor(temp, wind, rain, hum)
    st.info(f"Weather factor: **{weather_factor}** ({(weather_factor-1)*100:+.1f}% impact)")

    # Team selection
    col1, col2 = st.columns(2)
    with col1: home = st.selectbox("Home Team", teams, index=None)
    with col2:
        away_options = [t for t in teams if t != home] if home else teams
        away = st.selectbox("Away Team", away_options, index=None)

    if home and away:
        # Base prediction
        base_home = f[home]['OffHome'] * (league_avg/2 / f[away]['DefAway'])
        base_away = f[away]['OffAway'] * (league_avg/2 / f[home]['DefHome'])
        
        # Weather-adjusted
        adj_home = base_home * weather_factor
        adj_away = base_away * weather_factor
        total_adj = adj_home + adj_away
        
        st.markdown(f"### **{home} vs {away}**")
        st.metric("Total Shots on Target (Weather-Adjusted)", f"{total_adj:.2f}", 
                 delta=f"{total_adj - (base_home+base_away):+.2f} vs no weather")
        
        c1, c2 = st.columns(2)
        with c1:
            st.metric(f"{home} SoT", f"{adj_home:.2f}")
            st.metric(f"{home} xG", f"{adj_home * f[home]['FinHome']:.2f}")
        with c2:
            st.metric(f"{away} SoT", f"{adj_away:.2f}")
            st.metric(f"{away} xG", f"{adj_away * f[away]['FinAway']:.2f}")
        
        # Top 5 scorelines
        st.markdown("#### Most Likely Scorelines")
        xg_h = adj_home * f[home]['FinHome']
        xg_a = adj_away * f[away]['FinAway']
        scores = []
        for g1 in range(7):
            for g2 in range(6):
                p = poisson.pmf(g1, xg_h) * poisson.pmf(g2, xg_a)
                if p > 0.02:
                    scores.append((g1, g2, p))
        scores.sort(key=lambda x: x[2], reverse=True)
        cols = st.columns(5)
        for i, (g1, g2, p) in enumerate(scores[:5]):
            with cols[i]:
                st.metric(f"{g1}–{g2}", f"{p:.1%}")

else:
    st.info("Click the button above to load data from all Top 5 leagues (2025/26 + 2024/25) and train the model.")

st.caption("Weather-adjusted • Hybrid form • Top 5 Leagues • Data: Football-Data.co.uk")

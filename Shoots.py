# Shoots.py - Hybrid Form Predictor with Weather Impact (Top 5 Leagues)
import streamlit as st
import pandas as pd
import numpy as np
from scipy.stats import poisson

st.set_page_config(page_title="Hybrid SoT Predictor - Weather Enhanced", layout="wide")
st.title("Hybrid Form-Based SoT & Goals Predictor")
st.markdown("### Last 5 Current (70%) + Last Season (30%) + Weather Impact • Top 5 European Leagues • 2025/2026 Season")

# Top 5 Leagues (extended for promotions)
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
            df['Season'] = season_folder
            df['Date'] = pd.to_datetime(df['Date'], dayfirst=True)
            df = df.dropna(subset=['Date','FTHG','FTAG','HST','AST'])
            dfs.append(df)
        except Exception as e:
            st.warning(f"Failed to load {name} {season_folder}: {e}")
    return pd.concat(dfs).sort_values('Date').reset_index(drop=True) if dfs else None

def get_last_n_games(df, team, is_home, n=5):
    mask = (df['HomeTeam'] == team) if is_home else (df['AwayTeam'] == team)
    return df[mask].sort_values('Date', ascending=False).head(n)

def compute_weather_factor(temp_c, wind_kmh, rain, humidity):
    """Compute SoT multiplier based on weather (research-backed). Returns factor (e.g., 0.92 for -8%)."""
    factor = 1.0
    # Temp: <5°C or >25°C: -0.10; 5-25: 0
    if temp_c < 5 or temp_c > 25:
        factor *= 0.90
    # Wind: >15 km/h: -0.15; >25: -0.25
    if wind_kmh > 15:
        factor *= 0.85 if wind_kmh > 25 else 0.92
    # Rain: Yes: +0.05 (slippery boost) but -0.05 accuracy → net 0.98
    if rain:
        factor *= 0.98
    # Humidity >70%: -0.05 stamina
    if humidity > 70:
        factor *= 0.95
    return round(factor, 3)

def compute_hybrid_form(current_df, last_df):
    all_teams = pd.unique(pd.concat([current_df, last_df])[['HomeTeam','AwayTeam']].values.ravel('K'))
    league_avg_sot = current_df['HST'].mean() + current_df['AST'].mean()
    form = {}
    for team in all_teams:
        # Current last 5
        curr_home = get_last_n_games(current_df, team, True, 5)
        curr_away = get_last_n_games(current_df, team, False, 5)
        curr_off_home = curr_home['HST'].mean() if len(curr_home) > 0 else 5.0
        curr_off_away = curr_away['AST'].mean() if len(curr_away) > 0 else 4.5
        curr_def_home = curr_home['AST'].mean() if len(curr_home) > 0 else 4.8
        curr_def_away = curr_away['HST'].mean() if len(curr_away) > 0 else 5.2
        curr_fin_home = (curr_home['FTHG'].sum() / curr_home['HST'].sum()) if curr_home['HST'].sum() > 0 else 0.30

        # Last season avg
        last_home = last_df[last_df['HomeTeam'] == team]
        last_away = last_df[last_df['AwayTeam'] == team]
        last_off_home = last_home['HST'].mean() if len(last_home) > 0 else 4.5
        last_off_away = last_away['AST'].mean() if len(last_away) > 0 else 4.0
        last_def_home = last_home['AST'].mean() if len(last_home) > 0 else 5.0
        last_def_away = last_away['HST'].mean() if len(last_away) > 0 else 5.5
        last_fin_home = (last_home['FTHG'].sum() / last_home['HST'].sum()) if last_home['HST'].sum() > 0 else 0.28

        # Hybrid weights
        off_home = curr_off_home * 0.7 + last_off_home * 0.3
        off_away = curr_off_away * 0.7 + last_off_away * 0.3
        def_home = curr_def_home * 0.7 + last_def_home * 0.3
        def_away = curr_def_away * 0.7 + last_def_away * 0.3
        fin_home = curr_fin_home * 0.7 + last_fin_home * 0.3

        # Set-piece boost
        sp_home = curr_home['AC'].mean() * 0.2 if len(curr_home) > 0 else 0.5
        sp_away = curr_away['HC'].mean() * 0.2 if len(curr_away) > 0 else 0.5

        form[team] = {
            'OffHome': round(off_home + sp_home, 2),
            'OffAway': round(off_away + sp_away, 2),
            'DefHome': round(def_home, 2),
            'DefAway': round(def_away, 2),
            'FinHome': round(fin_home, 3),
            'FinAway': round((get_last_n_games(current_df, team, False, 5)['FTAG'].sum() / get_last_n_games(current_df, team, False, 5)['AST'].sum()), 3) if len(get_last_n_games(current_df, team, False, 5)) > 0 and get_last_n_games(current_df, team, False, 5)['AST'].sum() > 0 else 0.26,
            'CurrOffHome': round(curr_off_home, 2), 'LastOffHome': round(last_off_home, 2),
            'HomeGamesCurr': len(curr_home), 'HomeGamesLast': len(last_home)
        }
    return form, league_avg_sot

if st.button("Load Current + Last Season Data & Compute Hybrid Form (Top 5 Leagues)", type="primary"):
    current_data = load_season_data('2526')
    last_data = load_season_data('2425')
    if current_data is not None and last_data is not None:
        form, league_avg = compute_hybrid_form(current_data, last_data)
        st.session_state.form = form
        st.session_state.league_avg = league_avg
        st.session_state.teams = sorted(form.keys())
        st.session_state.current_data = current_data
        st.session_state.last_data = last_data
        st.success(f"Hybrid model ready across Top 5 Leagues! League avg SoT: {league_avg:.1f} | Teams: {len(form)}")
    else:
        st.warning("Data loading issue—season may be early. Using available data.")

if 'form' in st.session_state:
    f = st.session_state.form
    teams = st.session_state.teams
    league_avg = st.session_state.league_avg
    
    # Weather Input Section
    st.subheader("Enter Weather Forecast for Match Day")
    col_w1, col_w2, col_w3, col_w4 = st.columns(4)
    with col_w1:
        temp_c = st.number_input("Temperature (°C)", min_value=-10.0, max_value=40.0, value=10.0)
    with col_w2:
        wind_kmh = st.number_input("Wind Speed (km/h)", min_value=0.0, max_value=100.0, value=15.0)
    with col_w3:
        rain = st.checkbox("Rain Expected?")
    with col_w4:
        humidity = st.number_input("Humidity (%)", min_value=0, max_value=100, value=70)
    
    weather_factor = compute_weather_factor(temp_c, wind_kmh, rain, humidity)
    st.info(f"**Weather Adjustment Factor**: {weather_factor} ({(weather_factor-1)*100:+.1f}% impact on SoT)")
    
    # Team Selection
    col1, col2 = st.columns(2)
    with col1: home = st.selectbox("Home", teams)
    with col2: away = st.selectbox("Away", [t for t in teams if t != home])
    
    if home and away:
        # Base hybrid prediction
        base_sot_home = f[home]['OffHome'] * (league_avg / 2 / f[away]['DefAway'])
        base_sot_away = f[away]['OffAway'] * (league_avg / 2 / f[home]['DefHome'])
        base_total_sot = base_sot_home + base_sot_away
        
        # Apply weather
        adj_sot_home = base_sot_home * weather_factor
        adj_sot_away = base_sot_away * weather_factor
        adj_total_sot = adj_sot_home + adj_sot_away
        xg_home = adj_sot_home * f[home]['FinHome']
        xg_away = adj_sot_away * f[away]['FinAway']
        
        st.metric("Total Predicted SoT (Weather-Adjusted)", f"{adj_total_sot:.1f}", delta=f"{adj_total_sot - base_total_sot:+.1f} vs. base")
        st.write(f"Home SoT: {adj_sot_home:.2f} (Base: {base_sot_home:.2f}) | Away SoT: {adj_sot_away:.2f} (Base: {base_sot_away:.2f})")
        st.write(f"Home SoT Breakdown: Curr: {f[home]['CurrOffHome']:.2f} + Last: {f[home]['LastOffHome']:.2f}")
        
        # Scorelines (Poisson on adjusted xG)
        scores = [(g1, g2, poisson.pmf(g1, xg_home) * poisson.pmf(g2, xg_away)) for g1 in range(5) for g2 in range(5)]
        scores.sort(key=lambda x: x[2], reverse=True)
        st.write("Top Scorelines:", [f"{s[0]}-{s[1]} ({s[2]:.1%})" for s in scores[:3]])

        # Breakdown
        with st.expander("Hybrid + Weather Breakdown"):
            st.write(f"**{home} Home Form**: Current last-{f[home]['HomeGamesCurr']}: {f[home]['CurrOffHome']:.2f} SoT | Last season ({f[home]['HomeGamesLast']} games): {f[home]['LastOffHome']:.2f} SoT")
            st.write(f"**Weather Details**: Temp {temp_c}°C, Wind {wind_kmh} km/h, Rain: {rain}, Humidity: {humidity}% → Factor: {weather_factor}")

        # League Filter
        selected_league = st.selectbox("Filter Teams by League", options=list(LEAGUES.values()) + ["All Leagues"])
        if selected_league != "All Leagues":
            league_teams = pd.unique(st.session_state.current_data[st.session_state.current_data['League'] == selected_league][['HomeTeam', 'AwayTeam']].values.ravel('K'))
            filtered_teams = [t for t in teams if t in league_teams]
            st.write(f"Teams in {selected_league}: {len(filtered_teams)} available")

else:
    st.info("Click the button to load data from Top 5 Leagues (incl. 2. Bundesliga for promotions).")

st.caption("Weather Impacts: Wind (-8-15%), Rain (±2%), Temp extremes (-5-10%), Humidity (-5%). Data: Football-Data.co.uk | Research: Soccer studies on meteorology.")

# Leagues.py - FOOTBALL PREDICTOR PRO v2.0 (FULL VERSION)
# Works with E0.csv, D1.csv, any league CSV
# Features: xG, Shots, Win %, Logos, Form Chart, PDF Export

import streamlit as st
import pandas as pd
import numpy as np
from scipy.stats import poisson
import requests
from PIL import Image
from io import BytesIO
import plotly.graph_objects as go
import plotly.express as px
import re
from datetime import datetime
import base64
from fpdf import FPDF
import io
import warnings

warnings.filterwarnings('ignore')

# ================================
# CONFIG
# ================================
st.set_page_config(page_title="Football Predictor Pro", layout="wide", initial_sidebar_state="expanded")
st.markdown("""
# Football Predictor Pro
**Realistic Match Predictions – xG, Shots, Win Probability, Form, PDF Export**
""")

# ================================
# LOGO & IMAGE HELPERS
# ================================
@st.cache_data(ttl=24*3600)
def get_team_logo(team_name: str) -> str | None:
    clean = re.sub(r"\s+FC$|CF$|SC$|AC$|United$|City$|Town$|Athletic$|Wanderers$|Hotspur$", "", team_name, flags=re.I).strip()
    url = f"https://en.wikipedia.org/w/api.php?action=query&titles={clean}&prop=pageimages&format=json&pithumbsize=120"
    try:
        resp = requests.get(url, timeout=6).json()
        pages = resp["query"]["pages"]
        page = next((p for p in pages.values() if "thumbnail" in p), None)
        if page and "thumbnail" in page:
            return page["thumbnail"]["source"]
    except Exception:
        pass
    return None

def load_image(url: str) -> Image.Image | None:
    try:
        resp = requests.get(url, timeout=6)
        return Image.open(BytesIO(resp.content))
    except Exception:
        return None

# ================================
# DEMO DATA
# ================================
def load_demo_csv() -> pd.DataFrame:
    data = {
        "Date": ["15/08/2025", "16/08/2025", "16/08/2025", "17/08/2025", "18/08/2025"],
        "HomeTeam": ["Liverpool", "Aston Villa", "Brighton", "Tottenham", "Wolves"],
        "AwayTeam": ["Bournemouth", "Newcastle", "Fulham", "Burnley", "Man City"],
        "FTHG": [4, 0, 1, 3, 0],
        "FTAG": [2, 0, 1, 0, 4],
        "HS": [19, 3, 10, 16, 9],
        "AS": [10, 16, 7, 14, 15],
        "HC": [6, 3, 4, 6, 4],
        "AC": [7, 6, 3, 5, 6],
    }
    return pd.DataFrame(data)

# ================================
# CSV LOADER + SAFE MAPPING
# ================================
st.sidebar.header("Upload League CSV")
uploaded_file = st.sidebar.file_uploader("E0.csv, D1.csv, etc.", type=["csv"])

if uploaded_file:
    try:
        df = pd.read_csv(uploaded_file, encoding='utf-8-sig')
        st.sidebar.success("CSV loaded!")
    except Exception as e:
        st.error(f"CSV read error: {e}")
        st.stop()
else:
    st.sidebar.info("Using demo data (Premier League).")
    df = load_demo_csv()

# Clean columns
df.columns = df.columns.str.strip().str.replace(r'\ufeff', '', regex=True)

# Required mapping
col_map = {
    'HomeTeam': 'HOMETEAM',
    'AwayTeam': 'AWAYTEAM',
    'FTHG': 'FTHG',
    'FTAG': 'FTAG',
    'HS': 'HS',
    'AS': 'AS',
    'HC': 'HC',
    'AC': 'AC',
    'Date': 'DATE'
}

missing = [exp for exp in col_map if exp not in df.columns]
if missing:
    st.error(f"Missing required columns: {', '.join(missing)}")
    st.stop()

df = df.rename(columns=col_map)

# Parse date
df['DATE'] = pd.to_datetime(df['DATE'], dayfirst=True, errors='coerce')
if df['DATE'].isna().all():
    st.error("All dates failed. Use DD/MM/YYYY or YYYY-MM-DD.")
    st.stop()

df = df.dropna(subset=['DATE']).sort_values('DATE').reset_index(drop=True)
st.success(f"Loaded {len(df)} matches from {df['DATE'].min().strftime('%b %Y')} to {df['DATE'].max().strftime('%b %Y')}")

# ================================
# FORM STATS – SAFE GROUPBY
# ================================
@st.cache_data
def compute_form_stats(df: pd.DataFrame, last_n: int = 5) -> dict:
    if df.empty:
        return {
            'home_attack': {}, 'home_defence': {}, 'away_attack': {}, 'away_defence': {},
            'home_shots': {}, 'away_shots': {}, 'home_corners': {}, 'away_corners': {},
            'league_home_goals': 1.6, 'league_away_goals': 1.3
        }

    # Last N home games
    home_games = []
    for team in df['HOMETEAM'].unique():
        matches = df[df['HOMETEAM'] == team].sort_values('DATE').tail(last_n)
        if len(matches) > 0:
            home_games.append({
                'team': team,
                'goals_for': matches['FTHG'].mean(),
                'goals_against': matches['FTAG'].mean(),
                'shots': matches['HS'].mean(),
                'corners': matches['HC'].mean()
            })
    home_df = pd.DataFrame(home_games)

    # Last N away games
    away_games = []
    for team in df['AWAYTEAM'].unique():
        matches = df[df['AWAYTEAM'] == team].sort_values('DATE').tail(last_n)
        if len(matches) > 0:
            away_games.append({
                'team': team,
                'goals_for': matches['FTAG'].mean(),
                'goals_against': matches['FTHG'].mean(),
                'shots': matches['AS'].mean(),
                'corners': matches['AC'].mean()
            })
    away_df = pd.DataFrame(away_games)

    lhg = df['FTHG'].mean() or 1.6
    lag = df['FTAG'].mean() or 1.3

    return {
        'home_attack': home_df.set_index('team')['goals_for'].to_dict(),
        'home_defence': home_df.set_index('team')['goals_against'].to_dict(),
        'away_attack': away_df.set_index('team')['goals_for'].to_dict(),
        'away_defence': away_df.set_index('team')['goals_against'].to_dict(),
        'home_shots': home_df.set_index('team')['shots'].to_dict(),
        'away_shots': away_df.set_index('team')['shots'].to_dict(),
        'home_corners': home_df.set_index('team')['corners'].to_dict(),
        'away_corners': away_df.set_index('team')['corners'].to_dict(),
        'league_home_goals': lhg,
        'league_away_goals': lag,
    }

stats = compute_form_stats(df)

# ================================
# PREDICTION ENGINE
# ================================
def predict_match(home: str, away: str):
    lhg = stats['league_home_goals']
    lag = stats['league_away_goals']

    # Attack/Defence ratings
    ha = stats['home_attack'].get(home, lhg) / lhg
    aa = stats['away_attack'].get(away, lag) / lag
    hd = stats['home_defence'].get(home, lag) / lag
    ad = stats['away_defence'].get(away, lhg) / lhg

    # Base shots
    base_home_shots = stats['home_shots'].get(home, 12.0)
    base_away_shots = stats['away_shots'].get(away, 10.0)

    # Adjusted shots
    home_shots = base_home_shots * ha * (2 - aa) / 2
    away_shots = base_away_shots * aa * (2 - hd) / 2

    # Bookmaker cap
    home_shots = max(3.0, min(round(home_shots, 1), 7.5))
    away_shots = max(2.0, min(round(away_shots, 1), 6.5))

    # xG
    home_xg = round(home_shots * 0.12, 2)
    away_xg = round(away_shots * 0.10, 2)

    # Simulate 10k matches
    sims = 10000
    home_goals_sim = poisson(mu=home_xg).rvs(sims)
    away_goals_sim = poisson(mu=away_xg).rvs(sims)
    home_wins = np.sum(home_goals_sim > away_goals_sim) / sims
    draws = np.sum(home_goals_sim == away_goals_sim) / sims
    away_wins = 1 - home_wins - draws

    # Corners
    home_corners = round(stats['home_corners'].get(home, 6.0) * ha * (2 - aa) / 2, 1)
    away_corners = round(stats['away_corners'].get(away, 4.5) * aa * (2 - hd) / 2, 1)
    home_corners = max(2.0, min(home_corners, 10.0))
    away_corners = max(1.0, min(away_corners, 8.0))

    return {
        'home_shots': home_shots,
        'away_shots': away_shots,
        'home_xg': home_xg,
        'away_xg': away_xg,
        'home_goals': round(home_goals_sim.mean(), 2),
        'away_goals': round(away_goals_sim.mean(), 2),
        'home_win': round(home_wins * 100, 1),
        'draw': round(draws * 100, 1),
        'away_win': round(away_wins * 100, 1),
        'home_corners': home_corners,
        'away_corners': away_corners,
    }

# ================================
# UI: TEAM SELECTOR
# ================================
teams = sorted(set(df['HOMETEAM'].unique()) | set(df['AWAYTEAM'].unique()))
if len(teams) < 2:
    st.error("Not enough teams in data.")
    st.stop()

col1, col2 = st.columns(2)
home_team = col1.selectbox("Home Team", teams, index=0)
away_team = col2.selectbox("Away Team", teams, index=min(1, len(teams)-1))

if home_team == away_team:
    st.warning("Select two different teams.")
    st.stop()

# ================================
# RUN PREDICTION
# ================================
result = predict_match(home_team, away_team)

# ================================
# DISPLAY RESULTS
# ================================
st.markdown(f"## {home_team} vs {away_team}")

colA, colB, colC = st.columns(3)
with colA:
    logo1 = get_team_logo(home_team)
    if logo1:
        st.image(load_image(logo1), width=100)
    st.metric(f"**{home_team}**", f"{result['home_goals']} goals", f"{result['home_shots']} shots")
    st.write(f"**Corners:** {result['home_corners']}")
with colB:
    st.metric("**xG**", f"{result['home_xg']} – {result['away_xg']}")
    st.metric("**Win %**", f"{result['home_win']}%", f"Draw: {result['draw']}%")
with colC:
    logo2 = get_team_logo(away_team)
    if logo2:
        st.image(load_image(logo2), width=100)
    st.metric(f"**{away_team}**", f"{result['away_goals']} goals", f"{result['away_shots']} shots")
    st.write(f"**Corners:** {result['away_corners']}")

# ================================
# FORM CHART
# ================================
def plot_form(team, is_home=True):
    if is_home:
        matches = df[df['HOMETEAM'] == team].tail(5)
        gf, ga = 'FTHG', 'FTAG'
        label = 'Home'
    else:
        matches = df[df['AWAYTEAM'] == team].tail(5)
        gf, ga = 'FTAG', 'FTHG'
        label = 'Away'
    if matches.empty:
        return go.Figure().add_annotation(text="No recent games", x=0.5, y=0.5, showarrow=False)
    fig = go.Figure()
    fig.add_trace(go.Bar(x=matches['DATE'].dt.strftime('%m/%d'), y=matches[gf], name='GF', marker_color='green'))
    fig.add_trace(go.Bar(x=matches['DATE'].dt.strftime('%m/%d'), y=matches[ga], name='GA', marker_color='red'))
    fig.update_layout(title=f"{team} – Last 5 {label} Games", barmode='relative')
    return fig

tab1, tab2 = st.tabs(["Home Form", "Away Form"])
with tab1:
    st.plotly_chart(plot_form(home_team, True), use_container_width=True)
with tab2:
    st.plotly_chart(plot_form(away_team, False), use_container_width=True)

# ================================
# PDF EXPORT
# ================================
def create_pdf():
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", size=16)
    pdf.cell(0, 10, f"{home_team} vs {away_team}", ln=1, align='C')
    pdf.ln(10)
    pdf.set_font("Arial", size=12)
    pdf.cell(0, 10, f"Predicted Score: {result['home_goals']} – {result['away_goals']}", ln=1)
    pdf.cell(0, 10, f"xG: {result['home_xg']} – {result['away_xg']}", ln=1)
    pdf.cell(0, 10, f"Shots: {result['home_shots']} – {result['away_shots']}", ln=1)
    pdf.cell(0, 10, f"Win %: {home_team} {result['home_win']}%, Draw {result['draw']}%, {away_team} {result['away_win']}%", ln=1)
    return pdf.output(dest='S').encode('latin1')

if st.button("Export to PDF"):
    pdf_bytes = create_pdf()
    b64 = base64.b64encode(pdf_bytes).decode()
    href = f'<a href="data:application/pdf;base64,{b64}" download="prediction.pdf">Download PDF</a>'
    st.markdown(href, unsafe_allow_html=True)

# ================================
# INSTALL
# ================================
st.sidebar.code("""
pip install streamlit pandas numpy scipy pillow requests plotly fpdf
streamlit run Leagues.py
""")

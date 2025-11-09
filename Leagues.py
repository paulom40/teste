# app.py - FOOTBALL PREDICTOR PRO: WORKS WITH E0.csv & D1.csv
import streamlit as st
import pandas as pd
import numpy as np
from scipy.stats import poisson
import requests
from PIL import Image
from io import BytesIO
import plotly.express as px
import plotly.graph_objects as go
import re
from datetime import datetime
import warnings

warnings.filterwarnings('ignore')

# ================================
# CONFIG
# ================================
st.set_page_config(page_title="Football Predictor Pro", layout="wide")
st.title("Football Predictor Pro")
st.markdown("**Premier League & Bundesliga Match Predictor**")

# ================================
# LOGO HELPERS
# ================================
@st.cache_data(ttl=24*3600)
def get_team_logo(team_name: str) -> str | None:
    clean = re.sub(r"\s+FC$|CF$|SC$|AC$|United$|City$|Town$|Athletic$|Wanderers$", "", team_name, flags=re.I).strip()
    url = f"https://en.wikipedia.org/w/api.php?action=query&titles={clean}&prop=pageimages&format=json&pithumbsize=100"
    try:
        resp = requests.get(url, timeout=5).json()
        pages = resp["query"]["pages"]
        page = next(iter(pages.values()))
        if "thumbnail" in page:
            return page["thumbnail"]["source"]
    except Exception:
        pass
    return None

def load_image(url: str) -> Image.Image | None:
    try:
        resp = requests.get(url, timeout=5)
        return Image.open(BytesIO(resp.content))
    except Exception:
        return None

# ================================
# DEMO DATA
# ================================
def load_demo_csv() -> pd.DataFrame:
    data = {
        "Date": pd.date_range("2025-01-01", periods=10, freq="7D"),
        "HomeTeam": ["Man City", "Liverpool", "Arsenal", "Chelsea", "Tottenham",
                     "Man United", "Newcastle", "West Ham", "Everton", "Leicester"],
        "AwayTeam": ["B-Team", "Brighton", "Wolves", "Fulham", "Crystal Palace",
                     "Southampton", "Brentford", "Aston Villa", "Leeds", "Norwich"],
        "FTHG": [3, 2, 1, 4, 2, 1, 3, 0, 2, 1],
        "FTAG": [1, 1, 0, 2, 1, 2, 0, 1, 1, 0],
        "HS": [14, 12, 9, 16, 11, 8, 13, 7, 10, 9],
        "AS": [6, 5, 4, 8, 5, 9, 4, 6, 5, 3],
        "HC": [7, 6, 5, 8, 5, 4, 6, 3, 5, 4],
        "AC": [3, 4, 2, 5, 3, 6, 2, 4, 3, 2],
    }
    return pd.DataFrame(data)

# ================================
# CSV LOADER + COLUMN MAPPING
# ================================
st.sidebar.header("Data Input")
uploaded_file = st.sidebar.file_uploader("Upload CSV (E0.csv, D1.csv)", type=["csv"])

if uploaded_file:
    try:
        df = pd.read_csv(uploaded_file, encoding='utf-8-sig')  # BOM-safe
    except Exception as e:
        st.error(f"Failed to read CSV: {e}")
        st.stop()
else:
    st.sidebar.info("Using demo data.")
    df = load_demo_csv()

# --- Clean column names ---
df.columns = df.columns.str.strip().str.replace(r'\ufeff', '', regex=True)

# --- Show original columns ---
st.sidebar.write("**Original columns:**")
st.sidebar.code("\n".join(df.columns.tolist()[:10]) + "\n...")

# --- Define expected column names ---
expected = {
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

# --- Map columns ---
col_map = {}
for exp, std in expected.items():
    if exp in df.columns:
        col_map[exp] = std
    else:
        st.error(f"Missing required column: `{exp}`")
        st.stop()

# --- Rename ---
df = df.rename(columns=col_map)

# --- Final check ---
required = ['DATE', 'HOMETEAM', 'AWAYTEAM', 'FTHG', 'FTAG', 'HS', 'AS', 'HC', 'AC']
missing = [col for col in required if col not in df.columns]
if missing:
    st.error(f"Failed to map: {missing}")
    st.stop()

# --- Parse date ---
df['DATE'] = pd.to_datetime(df['DATE'], dayfirst=True, errors='coerce')
if df['DATE'].isna().all():
    st.error("All dates failed to parse. Use DD/MM/YYYY or YYYY-MM-DD.")
    st.stop()

st.success("CSV loaded! Ready to predict.")

# ================================
# FORM STATS
# ================================
def compute_form_stats(df: pd.DataFrame, last_n: int = 5) -> dict:
    home = df.groupby('HOMETEAM').apply(lambda x: x.sort_values('DATE').tail(last_n))
    away = df.groupby('AWAYTEAM').apply(lambda x: x.sort_values('DATE').tail(last_n))

    stats = {
        'home_attack': home.groupby('HOMETEAM')['FTHG'].mean().to_dict(),
        'home_defence': home.groupby('HOMETEAM')['FTAG'].mean().to_dict(),
        'away_attack': away.groupby('AWAYTEAM')['FTAG'].mean().to_dict(),
        'away_defence': away.groupby('AWAYTEAM')['FTHG'].mean().to_dict(),
        'home_shots': home.groupby('HOMETEAM')['HS'].mean().to_dict(),
        'away_shots': away.groupby('AWAYTEAM')['AS'].mean().to_dict(),
        'league_home_goals': df['FTHG'].mean(),
        'league_away_goals': df['FTAG'].mean(),
    }
    return stats

stats = compute_form_stats(df)

# ================================
# REALISTIC SHOTS
# ================================
def predict_shots(home, away, stats):
    ha = stats['home_attack'].get(home, 1.5) / stats['league_home_goals']
    ad = stats['away_defence'].get(away, 1.2) / stats['league_away_goals']
    aa = stats['away_attack'].get(away, 1.2) / stats['league_away_goals']
    hd = stats['home_defence'].get(home, 1.5) / stats['league_home_goals']

    home_shots = round(12 * ha * (2 - aa) / 2, 1)
    away_shots = round(10 * aa * (2 - hd) / 2, 1)

    # Bookmaker cap
    home_shots = min(max(home_shots, 3.0), 7.5)
    away_shots = min(max(away_shots, 2.0), 6.5)

    return home_shots, away_shots

# ================================
# xG & PREDICTION
# ================================
def predict_match(home, away, stats):
    home_shots, away_shots = predict_shots(home, away, stats)
    home_xg = home_shots * 0.12
    away_xg = away_shots * 0.10

    home_goals = poisson(mu=home_xg).rvs(10000).mean()
    away_goals = poisson(mu=away_xg).rvs(10000).mean()

    return {
        'home_shots': home_shots,
        'away_shots': away_shots,
        'home_xg': round(home_xg, 2),
        'away_xg': round(away_xg, 2),
        'home_goals': round(home_goals, 2),
        'away_goals': round(away_goals, 2),
    }

# ================================
# DISPLAY
# ================================
teams = sorted(set(df['HOMETEAM'].unique()) | set(df['AWAYTEAM'].unique()))
col1, col2 = st.columns(2)
home_team = col1.selectbox("Home Team", teams)
away_team = col2.selectbox("Away Team", teams)

if home_team != away_team:
    result = predict_match(home_team, away_team, stats)

    colA, colB, colC = st.columns(3)
    with colA:
        st.metric(f"**{home_team}**", f"{result['home_goals']} goals", f"{result['home_shots']} shots")
    with colB:
        st.metric("xG", f"{result['home_xg']} – {result['away_xg']}")
    with colC:
        st.metric(f"**{away_team}**", f"{result['away_goals']} goals", f"{result['away_shots']} shots")

    # Logos
    colL1, colL2 = st.columns(2)
    with colL1:
        logo1 = get_team_logo(home_team)
        if logo1: st.image(load_image(logo1), width=100)
    with colL2:
        logo2 = get_team_logo(away_team)
        if logo2: st.image(load_image(logo2), width=100)

else:
    st.warning("Select two different teams.")

# ================================
# INSTALL
# ================================
st.sidebar.code("pip install streamlit pandas numpy scipy pillow requests plotly")

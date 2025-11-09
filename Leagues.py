# Leagues.py - FOOTBALL PREDICTOR PRO v5.0 (CARDS + YELLOWS ADDED)
import streamlit as st
import pandas as pd
import numpy as np
from scipy.stats import poisson
import requests
from PIL import Image
from io import BytesIO
import plotly.graph_objects as go
import re
from datetime import datetime
import warnings

warnings.filterwarnings('ignore')

# ================================
# CONFIG
# ================================
st.set_page_config(page_title="Predictor Pro v5.0", layout="wide")
st.markdown("""
# Football Predictor Pro v5.0
**Goals • xG • Shots • SoT • Corners • **Yellow Cards** • **Total Cards** • Win %**
""")

# ================================
# LOGO HELPERS
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
# DEMO DATA (WITH HY, AY, HR, AR)
# ================================
def load_demo_csv() -> pd.DataFrame:
    return pd.DataFrame({
        "Date": pd.date_range("2025-08-15", periods=10, freq="5D"),
        "HomeTeam": ["Liverpool", "Arsenal", "Man City", "Chelsea", "Tottenham", "Man United", "Newcastle", "West Ham", "Everton", "Leicester"],
        "AwayTeam": ["Bournemouth", "Brighton", "Wolves", "Fulham", "Crystal Palace", "Southampton", "Brentford", "Aston Villa", "Leeds", "Norwich"],
        "FTHG": [4, 2, 3, 4, 2, 1, 3, 0, 2, 1],
        "FTAG": [2, 1, 0, 2, 1, 2, 0, 1, 1, 0],
        "HS": [19, 12, 14, 16, 11, 8, 13, 7, 10, 9],
        "AS": [10, 5, 6, 8, 5, 9, 4, 6, 5, 3],
        "HST": [10, 6, 7, 8, 5, 3, 6, 2, 4, 3],
        "AST": [3, 2, 1, 3, 2, 4, 1, 2, 1, 1],
        "HC": [6, 6, 7, 8, 5, 4, 6, 3, 5, 4],
        "AC": [7, 4, 3, 5, 3, 6, 2, 4, 3, 2],
        "HY": [2, 1, 3, 2, 1, 4, 2, 3, 1, 2],  # YELLOW CARDS
        "AY": [3, 2, 1, 4, 3, 2, 1, 2, 3, 1],
        "HR": [0, 0, 1, 0, 0, 1, 0, 0, 0, 0],  # RED CARDS
        "AR": [0, 1, 0, 0, 0, 0, 1, 0, 0, 0],
    })

# ================================
# CSV LOADER
# ================================
st.sidebar.header("Upload CSV")
uploaded_file = st.sidebar.file_uploader("E0.csv, D1.csv", type=["csv"])

if uploaded_file:
    df = pd.read_csv(uploaded_file, encoding='utf-8-sig')
else:
    st.sidebar.info("Demo data active.")
    df = load_demo_csv()

df.columns = df.columns.str.strip().str.replace(r'\ufeff', '', regex=True)

required = {'HomeTeam': 'HOMETEAM', 'AwayTeam': 'AWAYTEAM', 'FTHG': 'FTHG', 'FTAG': 'FTAG',
            'HS': 'HS', 'AS': 'AS', 'HC': 'HC', 'AC': 'AC', 'Date': 'DATE'}
optional = {'HST': 'HST', 'AST': 'AST', 'HY': 'HY', 'AY': 'AY', 'HR': 'HR', 'AR': 'AR'}

missing = [k for k in required if k not in df.columns]
if missing:
    st.error(f"Missing: {', '.join(missing)}")
    st.stop()

df = df.rename(columns={**required, **{k: v for k, v in optional.items() if k in df.columns}})
df['DATE'] = pd.to_datetime(df['DATE'], dayfirst=True, errors='coerce')
df = df.dropna(subset=['DATE']).sort_values('DATE').reset_index(drop=True)

# ================================
# FORM STATS – WITH CARDS
# ================================
@st.cache_data
def compute_form_stats(df: pd.DataFrame, last_n: int = 6) -> dict:
    home_stats = []
    away_stats = []

    for team in df['HOMETEAM'].unique():
        m = df[df['HOMETEAM'] == team].tail(last_n)
        if len(m) == 0: continue
        home_stats.append({
            'team': team,
            'goals_for': m['FTHG'].mean(),
            'goals_against': m['FTAG'].mean(),
            'shots': m['HS'].mean(),
            'sot': m['HST'].mean() if 'HST' in m.columns else m['HS'].mean() * 0.35,
            'accuracy': (m['HST'] / m['HS']).mean() if 'HST' in m.columns and (m['HS'] > 0).all() else 0.35,
            'corners': m['HC'].mean(),
            'yellows': m['HY'].mean() if 'HY' in m.columns else 2.1,
            'reds': m['HR'].mean() if 'HR' in m.columns else 0.08,
        })

    for team in df['AWAYTEAM'].unique():
        m = df[df['AWAYTEAM'] == team].tail(last_n)
        if len(m) == 0: continue
        away_stats.append({
            'team': team,
            'goals_for': m['FTAG'].mean(),
            'goals_against': m['FTHG'].mean(),
            'shots': m['AS'].mean(),
            'sot': m['AST'].mean() if 'AST' in m.columns else m['AS'].mean() * 0.30,
            'accuracy': (m['AST'] / m['AS']).mean() if 'AST' in m.columns and (m['AS'] > 0).all() else 0.30,
            'corners': m['AC'].mean(),
            'yellows': m['AY'].mean() if 'AY' in m.columns else 2.4,
            'reds': m['AR'].mean() if 'AR' in m.columns else 0.10,
        })

    home_df = pd.DataFrame(home_stats).set_index('team')
    away_df = pd.DataFrame(away_stats).set_index('team')

    lhg = df['FTHG'].mean() or 1.6
    lag = df['FTAG'].mean() or 1.3
    league_yellows = (df['HY'].mean() + df['AY'].mean()) / 2 if 'HY' in df.columns else 4.5

    return {
        'home': home_df.to_dict('index'),
        'away': away_df.to_dict('index'),
        'league_home_goals': lhg,
        'league_away_goals': lag,
        'league_yellows': league_yellows,
    }

stats = compute_form_stats(df)

# ================================
# PREDICTION ENGINE – CARDS ADDED
# ================================
def predict_match(home: str, away: str):
    h = stats['home'].get(home, {})
    a = stats['away'].get(away, {})
    lhg = stats['league_home_goals']
    lag = stats['league_away_goals']
    ly = stats['league_yellows']

    # Attack / Defence
    ha = h.get('goals_for', lhg) / lhg
    aa = a.get('goals_for', lag) / lag
    hd = h.get('goals_against', lag) / lag
    ad = a.get('goals_against', lhg) / lhg

    # Shots
    home_shots = max(8.0, min(round(h.get('shots', 12.0) * ha * (2 - aa) / 2, 1), 22.0))
    away_shots = max(5.0, min(round(a.get('shots', 10.0) * aa * (2 - hd) / 2, 1), 20.0))

    # SoT
    home_acc = h.get('accuracy', 0.35)
    away_acc = a.get('accuracy', 0.30)
    home_sot = max(2.0, min(round(home_shots * home_acc, 1), home_shots))
    away_sot = max(1.0, min(round(away_shots * away_acc, 1), away_shots))

    # xG
    home_xg = round(home_sot * 0.28 + (home_shots - home_sot) * 0.01, 2)
    away_xg = round(away_sot * 0.25 + (away_shots - away_sot) * 0.01, 2)

    # Corners
    home_corners = max(3.0, min(round(h.get('corners', 6.0) * ha * (2 - aa) / 2, 1), 12.0))
    away_corners = max(2.0, min(round(a.get('corners', 4.5) * aa * (2 - hd) / 2, 1), 10.0))

    # **YELLOW CARDS**
    # Factors: aggression, pressure, referee, rivalry
    home_yellows = h.get('yellows', 2.1) * (1 + 0.1 * aa) * (1 + 0.05 * hd)
    away_yellows = a.get('yellows', 2.4) * (1 + 0.1 * ha) * (1 + 0.05 * ad)
    home_yellows = max(0.5, min(round(home_yellows, 1), 6.0))
    away_yellows = max(0.5, min(round(away_yellows, 1), 6.0))
    total_yellows = round(home_yellows + away_yellows, 1)

    # **RED CARDS (rare)**
    home_reds = round(h.get('reds', 0.08) * (1 + 0.3 * aa), 2)
    away_reds = round(a.get('reds', 0.10) * (1 + 0.3 * ha), 2)
    total_cards = total_yellows + home_reds + away_reds

    # Win %
    sims = 30000
    home_goals_sim = poisson(mu=home_xg).rvs(sims)
    away_goals_sim = poisson(mu=away_xg).rvs(sims)
    home_win = (home_goals_sim > away_goals_sim).mean() * 100
    draw = (home_goals_sim == away_goals_sim).mean() * 100
    away_win = 100 - home_win - draw

    return {
        'home_shots': home_shots, 'away_shots': away_shots,
        'home_sot': home_sot, 'away_sot': away_sot,
        'home_xg': home_xg, 'away_xg': away_xg,
        'home_goals': round(home_xg, 2), 'away_goals': round(away_xg, 2),
        'home_win': round(home_win, 1), 'draw': round(draw, 1), 'away_win': round(away_win, 1),
        'home_corners': home_corners, 'away_corners': away_corners,
        'home_yellows': home_yellows, 'away_yellows': away_yellows,
        'total_yellows': total_yellows, 'total_cards': round(total_cards, 1),
        'home_reds': home_reds, 'away_reds': away_reds,
    }

# ================================
# UI
# ================================
teams = sorted(set(df['HOMETEAM'].unique()) | set(df['AWAYTEAM'].unique()))
if len(teams) < 2:
    st.error("Not enough teams.")
    st.stop()

col1, col2 = st.columns(2)
home_team = col1.selectbox("Home Team", teams)
away_team = col2.selectbox("Away Team", teams)

if home_team == away_team:
    st.warning("Select different teams.")
    st.stop()

result = predict_match(home_team, away_team)

# ================================
# DISPLAY – CARDS VISIBLE
# ================================
st.markdown(f"## {home_team} vs {away_team}")

colA, colB, colC = st.columns(3)
with colA:
    logo = get_team_logo(home_team)
    if logo: st.image(load_image(logo), width=100)
    st.metric(f"**{home_team}**", f"{result['home_goals']} goals")
    st.write(f"**Shots:** {result['home_shots']} | **SoT:** {result['home_sot']} | **Corners:** {result['home_corners']}")
    st.write(f"**Yellow Cards:** {result['home_yellows']} | **Reds:** {result['home_reds']:.2f}")
with colB:
    st.metric("**xG**", f"{result['home_xg']} – {result['away_xg']}")
    st.metric("**Win %**", f"{result['home_win']}%", f"Draw: {result['draw']}%")
    st.metric("**Total Cards**", f"{result['total_cards']}", f"Yellows: {result['total_yellows']}")
with colC:
    logo = get_team_logo(away_team)
    if logo: st.image(load_image(logo), width=100)
    st.metric(f"**{away_team}**", f"{result['away_goals']} goals")
    st.write(f"**Shots:** {result['away_shots']} | **SoT:** {result['away_sot']} | **Corners:** {result['away_corners']}")
    st.write(f"**Yellow Cards:** {result['away_yellows']} | **Reds:** {result['away_reds']:.2f}")

# ================================
# FORM CHART
# ================================
def plot_form(team, home=True):
    m = (df[df['HOMETEAM'] == team] if home else df[df['AWAYTEAM'] == team]).tail(5)
    if m.empty: return go.Figure()
    fig = go.Figure()
    gf = 'FTHG' if home else 'FTAG'
    ga = 'FTAG' if home else 'FTHG'
    fig.add_trace(go.Bar(x=m['DATE'].dt.strftime('%m/%d'), y=m[gf], name='GF', marker_color='green'))
    fig.add_trace(go.Bar(x=m['DATE'].dt.strftime('%m/%d'), y=m[ga], name='GA', marker_color='red'))
    fig.update_layout(title=f"{team} Last 5", barmode='relative', height=300)
    return fig

tab1, tab2 = st.tabs(["Home Form", "Away Form"])
with tab1: st.plotly_chart(plot_form(home_team, True), width='stretch')
with tab2: st.plotly_chart(plot_form(away_team, False), width='stretch')

# ================================
# PDF EXPORT
# ================================
html = f"""
<h1>{home_team} vs {away_team}</h1>
<p><strong>Score:</strong> {result['home_goals']} – {result['away_goals']}</p>
<p><strong>xG:</strong> {result['home_xg']} – {result['away_xg']}</p>
<p><strong>Shots:</strong> {result['home_shots']} – {result['away_shots']}</p>
<p><strong>Shots on Target:</strong> {result['home_sot']} – {result['away_sot']}</p>
<p><strong>Corners:</strong> {result['home_corners']} – {result['away_corners']}</p>
<p><strong>Yellow Cards:</strong> {result['home_yellows']} – {result['away_yellows']} (Total: {result['total_yellows']})</p>
<p><strong>Total Cards:</strong> {result['total_cards']}</p>
<p><strong>Win %:</strong> {home_team} {result['home_win']}%, Draw {result['draw']}%, {away_team} {result['away_win']}%</p>
"""

if st.button("Export PDF"):
    st.download_button("Download PDF", html, "prediction.pdf", "application/pdf")

# ================================
# INSTALL
# ================================
st.sidebar.code("pip install streamlit pandas numpy scipy pillow requests plotly")

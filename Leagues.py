# Leagues.py - FOOTBALL PREDICTOR PRO v6.0 (FT SCORE + HTML EXPORT + POWER RATINGS)
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
st.set_page_config(page_title="Predictor Pro v6.0", layout="wide")
st.markdown("""
# Football Predictor Pro v6.0
**FT Score • xG • Shots • SoT • Corners • Cards • Win % • Power Ratings • HTML Export**
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
# DEMO DATA
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
        "HY": [2, 1, 3, 2, 1, 4, 2, 3, 1, 2],
        "AY": [3, 2, 1, 4, 3, 2, 1, 2, 3, 1],
        "HR": [0, 0, 1, 0, 0, 1, 0, 0, 0, 0],
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
# FORM STATS & POWER RATINGS
# ================================
@st.cache_data
def compute_form_stats(df: pd.DataFrame, last_n: int = 6) -> dict:
    home_stats = []
    away_stats = []
    
    # League averages for normalization
    lhg = df['FTHG'].mean() or 1.6
    lag = df['FTAG'].mean() or 1.3
    lh_shots = df['HS'].mean() or 12.0
    la_shots = df['AS'].mean() or 10.0
    lh_sot = df['HST'].mean() if 'HST' in df.columns else lh_shots * 0.35
    la_sot = df['AST'].mean() if 'AST' in df.columns else la_shots * 0.30

    for team in df['HOMETEAM'].unique():
        m = df[df['HOMETEAM'] == team].tail(last_n)
        if len(m) == 0: continue
        
        goals_for = m['FTHG'].mean()
        goals_against = m['FTAG'].mean()
        shots = m['HS'].mean()
        sot = m['HST'].mean() if 'HST' in m.columns else m['HS'].mean() * 0.35
        accuracy = (m['HST'] / m['HS']).mean() if 'HST' in m.columns and (m['HS'] > 0).all() else 0.35
        
        # Power Ratings Calculation
        offense_rating = round((goals_for / lhg * 0.6 + shots / lh_shots * 0.2 + sot / lh_sot * 0.2) * 100)
        defense_rating = round((1 - goals_against / lag) * 0.7 + (1 - (m['AS'].mean() / la_shots)) * 0.3) * 100)
        
        home_stats.append({
            'team': team,
            'goals_for': goals_for,
            'goals_against': goals_against,
            'shots': shots,
            'sot': sot,
            'accuracy': accuracy,
            'corners': m['HC'].mean(),
            'yellows': m['HY'].mean() if 'HY' in m.columns else 2.1,
            'reds': m['HR'].mean() if 'HR' in m.columns else 0.08,
            'offense_rating': offense_rating,
            'defense_rating': defense_rating,
            'overall_rating': round((offense_rating + defense_rating) / 2)
        })

    for team in df['AWAYTEAM'].unique():
        m = df[df['AWAYTEAM'] == team].tail(last_n)
        if len(m) == 0: continue
        
        goals_for = m['FTAG'].mean()
        goals_against = m['FTHG'].mean()
        shots = m['AS'].mean()
        sot = m['AST'].mean() if 'AST' in m.columns else m['AS'].mean() * 0.30
        accuracy = (m['AST'] / m['AS']).mean() if 'AST' in m.columns and (m['AS'] > 0).all() else 0.30
        
        # Power Ratings Calculation
        offense_rating = round((goals_for / lag * 0.6 + shots / la_shots * 0.2 + sot / la_sot * 0.2) * 100)
        defense_rating = round((1 - goals_against / lhg) * 0.7 + (1 - (m['HS'].mean() / lh_shots)) * 0.3) * 100)
        
        away_stats.append({
            'team': team,
            'goals_for': goals_for,
            'goals_against': goals_against,
            'shots': shots,
            'sot': sot,
            'accuracy': accuracy,
            'corners': m['AC'].mean(),
            'yellows': m['AY'].mean() if 'AY' in m.columns else 2.4,
            'reds': m['AR'].mean() if 'AR' in m.columns else 0.10,
            'offense_rating': offense_rating,
            'defense_rating': defense_rating,
            'overall_rating': round((offense_rating + defense_rating) / 2)
        })

    home_df = pd.DataFrame(home_stats).set_index('team')
    away_df = pd.DataFrame(away_stats).set_index('team')
    
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
# POWER RATINGS DISPLAY
# ================================
def display_power_ratings():
    st.sidebar.header("📊 Team Power Ratings")
    
    # Combine home and away stats for overall ratings
    all_teams = {}
    for team in set(stats['home'].keys()) | set(stats['away'].keys()):
        home_data = stats['home'].get(team, {})
        away_data = stats['away'].get(team, {})
        
        if home_data and away_data:
            # Average home and away performance
            overall_rating = (home_data.get('overall_rating', 0) + away_data.get('overall_rating', 0)) // 2
            offense_rating = (home_data.get('offense_rating', 0) + away_data.get('offense_rating', 0)) // 2
            defense_rating = (home_data.get('defense_rating', 0) + away_data.get('defense_rating', 0)) // 2
        elif home_data:
            overall_rating = home_data.get('overall_rating', 0)
            offense_rating = home_data.get('offense_rating', 0)
            defense_rating = home_data.get('defense_rating', 0)
        else:
            overall_rating = away_data.get('overall_rating', 0)
            offense_rating = away_data.get('offense_rating', 0)
            defense_rating = away_data.get('defense_rating', 0)
            
        all_teams[team] = {
            'overall': overall_rating,
            'offense': offense_rating,
            'defense': defense_rating
        }
    
    # Sort by overall rating
    sorted_teams = sorted(all_teams.items(), key=lambda x: x[1]['overall'], reverse=True)
    
    # Display in sidebar
    for team, ratings in sorted_teams:
        with st.sidebar.expander(f"{team} (Overall: {ratings['overall']})"):
            col1, col2 = st.columns(2)
            col1.metric("⚽ Offense", ratings['offense'])
            col2.metric("🛡️ Defense", ratings['defense'])

# ================================
# PREDICTION ENGINE – FT SCORE + HTML
# ================================
def predict_match(home: str, away: str):
    h = stats['home'].get(home, {})
    a = stats['away'].get(away, {})
    lhg = stats['league_home_goals']
    lag = stats['league_away_goals']

    ha = h.get('goals_for', lhg) / lhg
    aa = a.get('goals_for', lag) / lag
    hd = h.get('goals_against', lag) / lag
    ad = a.get('goals_against', lhg) / lhg

    home_shots = max(8.0, min(round(h.get('shots', 12.0) * ha * (2 - aa) / 2, 1), 22.0))
    away_shots = max(5.0, min(round(a.get('shots', 10.0) * aa * (2 - hd) / 2, 1), 20.0))

    home_acc = h.get('accuracy', 0.35)
    away_acc = a.get('accuracy', 0.30)
    home_sot = max(2.0, min(round(home_shots * home_acc, 1), home_shots))
    away_sot = max(1.0, min(round(away_shots * away_acc, 1), away_shots))

    home_xg = round(home_sot * 0.28 + (home_shots - home_sot) * 0.01, 2)
    away_xg = round(away_sot * 0.25 + (away_shots - away_sot) * 0.01, 2)

    home_corners = max(3.0, min(round(h.get('corners', 6.0) * ha * (2 - aa) / 2, 1), 12.0))
    away_corners = max(2.0, min(round(a.get('corners', 4.5) * aa * (2 - hd) / 2, 1), 10.0))

    home_yellows = max(0.5, min(round(h.get('yellows', 2.1) * (1 + 0.1 * aa) * (1 + 0.05 * hd), 1), 6.0))
    away_yellows = max(0.5, min(round(a.get('yellows', 2.4) * (1 + 0.1 * ha) * (1 + 0.05 * ad), 1), 6.0))
    total_yellows = round(home_yellows + away_yellows, 1)
    total_cards = round(total_yellows + h.get('reds', 0.08) * 1.3 + a.get('reds', 0.10) * 1.3, 1)

    # **FT SCORE PREDICTION (Most Likely)**
    sims = 30000
    home_goals_sim = poisson(mu=home_xg).rvs(sims)
    away_goals_sim = poisson(mu=away_xg).rvs(sims)

    # Most common score
    scores = np.column_stack([home_goals_sim, away_goals_sim])
    unique, counts = np.unique(scores, axis=0, return_counts=True)
    most_likely_idx = counts.argmax()
    ft_score = f"{unique[most_likely_idx][0]}-{unique[most_likely_idx][1]}"

    # Win %
    home_win = (home_goals_sim > away_goals_sim).mean() * 100
    draw = (home_goals_sim == away_goals_sim).mean() * 100
    away_win = 100 - home_win - draw

    return {
        'ft_score': ft_score,  # e.g. "2-1"
        'home_shots': home_shots, 'away_shots': away_shots,
        'home_sot': home_sot, 'away_sot': away_sot,
        'home_xg': home_xg, 'away_xg': away_xg,
        'home_goals': round(home_xg, 2), 'away_goals': round(away_xg, 2),
        'home_win': round(home_win, 1), 'draw': round(draw, 1), 'away_win': round(away_win, 1),
        'home_corners': home_corners, 'away_corners': away_corners,
        'home_yellows': home_yellows, 'away_yellows': away_yellows,
        'total_yellows': total_yellows, 'total_cards': total_cards,
        'home_offense_rating': h.get('offense_rating', 0),
        'home_defense_rating': h.get('defense_rating', 0),
        'home_overall_rating': h.get('overall_rating', 0),
        'away_offense_rating': a.get('offense_rating', 0),
        'away_defense_rating': a.get('defense_rating', 0),
        'away_overall_rating': a.get('overall_rating', 0),
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

# Display power ratings in sidebar
display_power_ratings()

# ================================
# DISPLAY – FT SCORE + HTML + POWER RATINGS
# ================================
st.markdown(f"## {home_team} vs {away_team}")

colA, colB, colC = st.columns(3)
with colA:
    logo = get_team_logo(home_team)
    if logo: st.image(load_image(logo), width=100)
    st.metric(f"**{home_team}**", f"**{result['ft_score'].split('-')[0]}** goals")
    st.write(f"**Shots:** {result['home_shots']} | **SoT:** {result['home_sot']} | **Corners:** {result['home_corners']}")
    st.write(f"**Yellow Cards:** {result['home_yellows']}")
    # Power Ratings for Home Team
    st.metric("⚽ Offense Rating", f"{result['home_offense_rating']}")
    st.metric("🛡️ Defense Rating", f"{result['home_defense_rating']}")
    st.metric("🌟 Overall Rating", f"{result['home_overall_rating']}")
    
with colB:
    st.metric("**FT Score**", f"**{result['ft_score']}**", "Most Likely")
    st.metric("**xG**", f"{result['home_xg']} – {result['away_xg']}")
    st.metric("**Win %**", f"{result['home_win']}%", f"Draw: {result['draw']}%")
    st.metric("**Total Cards**", f"{result['total_cards']}", f"Yellows: {result['total_yellows']}")
    
with colC:
    logo = get_team_logo(away_team)
    if logo: st.image(load_image(logo), width=100)
    st.metric(f"**{away_team}**", f"**{result['ft_score'].split('-')[1]}** goals")
    st.write(f"**Shots:** {result['away_shots']} | **SoT:** {result['away_sot']} | **Corners:** {result['away_corners']}")
    st.write(f"**Yellow Cards:** {result['away_yellows']}")
    # Power Ratings for Away Team
    st.metric("⚽ Offense Rating", f"{result['away_offense_rating']}")
    st.metric("🛡️ Defense Rating", f"{result['away_defense_rating']}")
    st.metric("🌟 Overall Rating", f"{result['away_overall_rating']}")

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
with tab1:
    st.plotly_chart(plot_form(home_team, True), width='stretch', key="home_form")
with tab2:
    st.plotly_chart(plot_form(away_team, False), width='stretch', key="away_form")

# ================================
# HTML EXPORT (Beautiful + Printable + Power Ratings)
# ================================
html_template = f"""
<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8" />
  <title>{home_team} vs {away_team} - Prediction</title>
  <style>
    body {{ font-family: 'Segoe UI', sans-serif; margin: 40px; background: #f9f9fb; color: #333; }}
    .container {{ max-width: 800px; margin: auto; background: white; padding: 30px; border-radius: 12px; box-shadow: 0 4px 12px rgba(0,0,0,0.1); }}
    h1 {{ text-align: center; color: #1a73e8; margin-bottom: 10px; }}
    .score {{ font-size: 48px; font-weight: bold; text-align: center; color: #1a73e8; margin: 20px 0; }}
    .team {{ text-align: center; font-size: 24px; margin: 10px 0; }}
    .stat {{ font-size: 18px; margin: 8px 0; display: flex; justify-content: space-between; }}
    .label {{ font-weight: 600; }}
    .ratings {{ display: flex; justify-content: space-around; margin: 20px 0; }}
    .rating-box {{ text-align: center; padding: 15px; border-radius: 8px; background: #f8f9fa; }}
    .footer {{ text-align: center; margin-top: 40px; font-size: 14px; color: #777; }}
    @media print {{ body {{ margin: 10mm; }} .no-print {{ display: none; }} }}
  </style>
</head>
<body>
  <div class="container">
    <h1>{home_team} vs {away_team}</h1>
    <div class="score">{result['ft_score']}</div>
    <div class="team">{home_team} <strong>{result['ft_score'].split('-')[0]}</strong> – <strong>{result['ft_score'].split('-')[1]}</strong> {away_team}</div>
    
    <div class="ratings">
      <div class="rating-box">
        <h3>{home_team}</h3>
        <div>⚽ Offense: {result['home_offense_rating']}</div>
        <div>🛡️ Defense: {result['home_defense_rating']}</div>
        <div>🌟 Overall: {result['home_overall_rating']}</div>
      </div>
      <div class="rating-box">
        <h3>{away_team}</h3>
        <div>⚽ Offense: {result['away_offense_rating']}</div>
        <div>🛡️ Defense: {result['away_defense_rating']}</div>
        <div>🌟 Overall: {result['away_overall_rating']}</div>
      </div>
    </div>
    
    <hr style="margin: 30px 0; border: 1px solid #eee;">
    <div class="stat"><span class="label">xG:</span> <span>{result['home_xg']} – {result['away_xg']}</span></div>
    <div class="stat"><span class="label">Shots:</span> <span>{result['home_shots']} – {result['away_shots']}</span></div>
    <div class="stat"><span class="label">Shots on Target:</span> <span>{result['home_sot']} – {result['away_sot']}</span></div>
    <div class="stat"><span class="label">Corners:</span> <span>{result['home_corners']} – {result['away_corners']}</span></div>
    <div class="stat"><span class="label">Yellow Cards:</span> <span>{result['home_yellows']} – {result['away_yellows']}</span></div>
    <div class="stat"><span class="label">Total Cards:</span> <span>{result['total_cards']}</span></div>
    <div class="stat"><span class="label">Win Probability:</span> <span>{home_team} {result['home_win']}%, Draw {result['draw']}%, {away_team} {result['away_win']}%</span></div>
    <div class="footer">Generated by Football Predictor Pro v6.0 • {datetime.now().strftime('%Y-%m-%d %H:%M')}</div>
  </div>
</body>
</html>
"""

if st.button("Export to HTML"):
    st.download_button(
        label="Download HTML Report",
        data=html_template,
        file_name=f"{home_team}_vs_{away_team}_prediction.html",
        mime="text/html"
    )
    st.success("Click above to download → Open in browser → Print/Save as PDF")

# ================================
# INSTALL
# ================================
st.sidebar.code("pip install streamlit pandas numpy scipy pillow requests plotly")

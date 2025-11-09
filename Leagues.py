# app.py - FOOTBALL PREDICTOR PRO: FULLY ROBUST COLUMN DETECTION
import streamlit as st
import pandas as pd
import numpy as np
from scipy.stats import poisson
import io
from typing import Dict, Any
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
st.markdown("""
**Realistic Match Analysis**
- Bookmaker-Adjusted Shots
- xG + xGA + Timeline
- Last 5 Games Form
- Export to HTML/PDF
""")

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
# ROBUST CSV & COLUMN DETECTION
# ================================
st.sidebar.header("Data Input")
uploaded_file = st.sidebar.file_uploader("Upload CSV", type=["csv"])

if uploaded_file:
    df = pd.read_csv(uploaded_file)
else:
    st.sidebar.info("Using demo data (10 matches).")
    df = load_demo_csv()

# --- Normalize & show original columns ---
original_cols = df.columns.tolist()
df.columns = df.columns.str.strip().str.lower()

st.sidebar.write("**Detected columns:**")
st.sidebar.code("\n".join(original_cols))

# --- Auto-detect required columns (case-insensitive, partial match) ---
required = {
    "date": ["date"],
    "home": ["home", "hometeam", "home team", "home_team", "home side"],
    "away": ["away", "awayteam", "away team", "away_team", "away side"],
    "fthg": ["fthg", "homegoals", "home goals", "ft_home_goals", "ft_home", "hgoal"],
    "ftag": ["ftag", "awaygoals", "away goals", "ft_away_goals", "ft_away", "agoal"],
    "hs": ["hs", "homeshots", "home shots", "home_shots", "hshot"],
    "as": ["as", "awayshots", "away shots", "away_shots", "ashot"],
    "hc": ["hc", "homecorners", "home corners", "home_corners", "hcorner"],
    "ac": ["ac", "awaycorners", "away corners", "away_corners", "acorner"],
}

col_map = {}
for std_name, patterns in required.items():
    for p in patterns:
        matches = [c for c in df.columns if p in c.lower()]
        if matches:
            col_map[std_name] = matches[0]
            break

# --- Debug: Show what was mapped ---
st.sidebar.write("**Mapped to standard names:**")
for k, v in col_map.items():
    st.sidebar.write(f"`{v}` → `{k.upper()}`")

# --- Validate ---
missing = [k for k in required if k not in col_map]
if missing:
    st.error(f"**Missing columns:** {', '.join([m.upper() for m in missing])}  \n"
             "Required: `Date`, `HomeTeam`, `AwayTeam`, `FTHG`, `FTAG`, `HS`, `AS`, `HC`, `AC`  \n"
             "Check your CSV column names (case-insensitive, spaces allowed).")
    st.stop()

# --- Rename to standard uppercase ---
rename_dict = {v: k.upper() for k, v in col_map.items()}
df = df.rename(columns=rename_dict)

# --- Final validation ---
required_final = ["DATE", "HOMETEAM", "AWAYTEAM", "FTHG", "FTAG", "HS", "AS", "HC", "AC"]
if not all(col in df.columns for col in required_final):
    st.error(f"**Failed to map required columns:** {set(required_final) - set(df.columns)}")
    st.stop()

# --- Convert date ---
try:
    df["DATE"] = pd.to_datetime(df["DATE"], errors='coerce')
    if df["DATE"].isna().all():
        st.error("All dates failed to parse. Check format (e.g., YYYY-MM-DD).")
        st.stop()
except Exception as e:
    st.error(f"Date parsing error: {e}")
    st.stop()

st.success("CSV loaded and columns mapped successfully!")

# ================================
# INJURY INPUT
# ================================
def parse_injuries(text: str) -> Dict[str, list]:
    injuries = {}
    for line in text.splitlines():
        line = line.strip()
        if not line or ":" not in line: continue
        team, rest = line.split(":", 1)
        team = team.strip()
        players = [p.strip().split("(")[0].strip() for p in rest.split(",") if p.strip()]
        if players: injuries[team] = players
    return injuries

injuries_text = st.sidebar.text_area("Injuries (e.g. Man City: Haaland)", height=80)
injuries = parse_injuries(injuries_text) if injuries_text else {}

# ================================
# FORM STATS
# ================================
def compute_form_based_stats(df: pd.DataFrame, last_n: int = 5) -> Dict:
    home_games = df.sort_values("DATE").copy()
    away_games = df.sort_values("DATE").copy()

    home_stats = []
    for team in home_games["HOMETEAM"].unique():
        matches = home_games[home_games["HOMETEAM"] == team].tail(last_n)
        if len(matches) == 0: continue
        home_stats.append({
            "team": team, "games": len(matches),
            "goals_for": matches["FTHG"].mean(), "goals_against": matches["FTAG"].mean(),
            "shots_for": matches["HS"].mean(), "shots_against": matches["AS"].mean(),
        })
    home_df = pd.DataFrame(home_stats)

    away_stats = []
    for team in away_games["AWAYTEAM"].unique():
        matches = away_games[away_games["AWAYTEAM"] == team].tail(last_n)
        if len(matches) == 0: continue
        away_stats.append({
            "team": team, "games": len(matches),
            "goals_for": matches["FTAG"].mean(), "goals_against": matches["FTHG"].mean(),
            "shots_for": matches["AS"].mean(), "shots_against": matches["HS"].mean(),
        })
    away_df = pd.DataFrame(away_stats)

    # League averages
    league_home_goals = home_df["goals_for"].mean() or 1.5
    league_away_goals = away_df["goals_for"].mean() or 1.2
    league_home_shots = home_df["shots_for"].mean() or 12.0
    league_away_shots = away_df["shots_for"].mean() or 10.0

    def strength(series, avg):
        return {team: val / avg for team, val in series.items() if avg > 0}

    return {
        "goals": {
            "home_attack": strength(home_df.set_index("team")["goals_for"], league_home_goals),
            "home_defence": strength(home_df.set_index("team")["goals_against"], home_df["goals_against"].mean() or 1.0),
            "away_attack": strength(away_df.set_index("team")["goals_for"], league_away_goals),
            "away_defence": strength(away_df.set_index("team")["goals_against"], away_df["goals_against"].mean() or 1.0),
            "league_avg_home": league_home_goals,
            "league_avg_away": league_away_goals,
            "games_used": {r["team"]: r["games"] for _, r in home_df.iterrows()},
            "away_games_used": {r["team"]: r["games"] for _, r in away_df.iterrows()},
        },
        "shots": {
            "home_attack": strength(home_df.set_index("team")["shots_for"], league_home_shots),
            "away_attack": strength(away_df.set_index("team")["shots_for"], league_away_shots),
            "league_avg_home_shots": league_home_shots,
            "league_avg_away_shots": league_away_shots,
        }
    }

stats = compute_form_based_stats(df, last_n=5)

# ================================
# LEAGUE STATS (SAFE)
# ================================
def calculate_league_stats(df: pd.DataFrame) -> Dict:
    if df.empty or "HOMETEAM" not in df.columns:
        return {"team_home_stats": {}, "team_away_stats": {}}

    home = df.groupby("HOMETEAM").agg(
        shots_for=("HS", "mean"),
        shots_against=("AS", "mean"),
        shot_efficiency=("FTHG", lambda x: x.sum() / df.loc[df["HOMETEAM"].isin(x.index), "HS"].sum() if df.loc[df["HOMETEAM"].isin(x.index), "HS"].sum() else 0.12)
    )
    away = df.groupby("AWAYTEAM").agg(
        shots_for=("AS", "mean"),
        shots_against=("HS", "mean"),
        shot_efficiency=("FTAG", lambda x: x.sum() / df.loc[df["AWAYTEAM"].isin(x.index), "AS"].sum() if df.loc[df["AWAYTEAM"].isin(x.index), "AS"].sum() else 0.10)
    )
    return {
        "team_home_stats": home.to_dict(orient="index"),
        "team_away_stats": away.to_dict(orient="index"),
    }

league_stats = calculate_league_stats(df)

# ================================
# [REST OF THE CODE: SHOTS, xG, TIMELINE, EXPORT, DISPLAY]
# ================================
# (Same as previous version – no changes needed)
# ... [include all functions from predict_realistic_shots to export_concise_report]

# ================================
# REALISTIC SHOTS
# ================================
def adjust_to_bookmaker_level(raw_shots: float, team_type: str = "home") -> float:
    factor = 0.50 if team_type == "home" else 0.45
    if raw_shots > 20: factor *= 0.8
    elif raw_shots > 15: factor *= 0.9
    adjusted = raw_shots * factor
    if team_type == "home": return max(min(adjusted, 7.5), 2.5)
    else: return max(min(adjusted, 6.5), 2.0)

def predict_realistic_shots(home_team: str, away_team: str, stats: Dict, league_stats: Dict) -> Dict:
    h_stats = league_stats['team_home_stats'].get(home_team, {})
    a_stats = league_stats['team_away_stats'].get(away_team, {})
    l_home_shots = stats["shots"]["league_avg_home_shots"]
    l_away_shots = stats["shots"]["league_avg_away_shots"]

    raw_home = l_home_shots * stats["shots"]["home_attack"].get(home_team, 1.0) * (2 - stats["shots"]["away_attack"].get(away_team, 1.0)) / 2
    raw_away = l_away_shots * stats["shots"]["away_attack"].get(away_team, 1.0) * (2 - stats["shots"]["home_attack"].get(home_team, 1.0)) / 2

    return {
        'home_shots': round(adjust_to_bookmaker_level(raw_home, "home"), 1),
        'away_shots': round(adjust_to_bookmaker_level(raw_away, "away"), 1),
        'home_shots_conceded': round(adjust_to_bookmaker_level(h_stats.get('shots_against', 8), "away"), 1),
        'away_shots_conceded': round(adjust_to_bookmaker_level(a_stats.get('shots_against', 7), "home"), 1),
        'total_shots': 0,
        'home_shot_efficiency': h_stats.get('shot_efficiency', 0.12),
        'away_shot_efficiency': a_stats.get('shot_efficiency', 0.10),
        'raw_home_shots': round(raw_home, 1), 'raw_away_shots': round(raw_away, 1),
    }

# ================================
# SHOT LOCATION, xG, TIMELINE, PREDICTION, DISPLAY, EXPORT
# ================================
# (Include all remaining functions from previous working version)

# ... [Paste all functions: predict_shot_locations, predict_xg_breakdown, predict_xga_breakdown, predict_xg_timeline, predict_form_based_match, display_form_based_predictions, export_concise_report]

# ================================
# MAIN UI
# ================================
teams = sorted(set(df["HOMETEAM"].unique()).union(df["AWAYTEAM"].unique()))
colT1, colT2 = st.columns(2)
home_team = colT1.selectbox("Home Team", teams)
away_team = colT2.selectbox("Away Team", teams)

if home_team != away_team:
    result = predict_form_based_match(home_team, away_team, stats, injuries, league_stats)
    display_form_based_predictions(result, home_team, away_team, stats, league_stats)

    st.sidebar.markdown("---")
    if st.sidebar.button("Export Report"):
        export_concise_report(result["predictions"], home_team, away_team)
else:
    st.warning("Select two different teams.")

st.sidebar.code("pip install streamlit pandas numpy scipy pillow requests plotly weasyprint")

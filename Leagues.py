# app.py - REALISTIC SHOT PREDICTIONS
import streamlit as st
import pandas as pd
import numpy as np
from scipy.stats import poisson
import io
from typing import Dict, Any, List
import requests
from PIL import Image
from io import BytesIO
import plotly.express as px
import plotly.graph_objects as go
import re
from datetime import datetime
import base64
import warnings

warnings.filterwarnings('ignore')

# ================================
# CONFIG
# ================================
st.set_page_config(page_title="Football Predictor - Realistic Analysis", layout="wide")
st.title("Football Predictor Pro - Realistic Analysis")
st.markdown("""
**Realistic Match Analysis**
- **Bookmaker-Adjusted Shot Predictions**
- **Realistic Shot Totals**
- **Market-Aligned Probabilities**
- **Last 5 Games Form Analysis**
""")

# ================================
# LOGO HELPERS
# ================================
@st.cache_data(ttl=24*3600)
def get_team_logo(team_name: str) -> str | None:
    """Very small logo cache – uses Wikipedia thumbnails."""
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
# CSV UPLOAD & DEMO DATA
# ================================
def load_demo_csv() -> pd.DataFrame:
    """Tiny demo dataset – 10 rows, all required columns."""
    data = {
        "Date": pd.date_range("2025-01-01", periods=10, freq="7D"),
        "HomeTeam": ["Man City", "Liverpool", "Arsenal", "Chelsea", "Tottenham",
                     "Man United", "Newcastle", "West Ham", "Everton", "Leicester"],
        "AwayTeam": ["Bournemouth", "Brighton", "Wolves", "Fulham", "Crystal Palace",
                     "Southampton", "Brentford", "Aston Villa", "Leeds", "Norwich"],
        "FTHG": [3, 2, 1, 4, 2, 1, 3, 0, 2, 1],
        "FTAG": [1, 1, 0, 2, 1, 2, 0, 1, 1, 0],
        "HS": [14, 12, 9, 16, 11, 8, 13, 7, 10, 9],
        "AS": [6, 5, 4, 8, 5, 9, 4, 6, 5, 3],
        "HC": [7, 6, 5, 8, 5, 4, 6, 3, 5, 4],
        "AC": [3, 4, 2, 5, 3, 6, 2, 4, 3, 2],
    }
    return pd.DataFrame(data)

st.sidebar.header("Data Input")
uploaded_file = st.sidebar.file_uploader("Upload your CSV (must contain Date, HomeTeam, AwayTeam, FTHG, FTAG, HS, AS, HC, AC)", type=["csv"])

if uploaded_file:
    df = pd.read_csv(uploaded_file)
else:
    st.sidebar.info("No file uploaded – using **demo data** (10 recent matches).")
    df = load_demo_csv()

# -----------------------------
# Column detection (robust)
# -----------------------------
def detect_columns(df: pd.DataFrame) -> Dict[str, str]:
    cols = df.columns.str.lower()
    mapping = {}
    for key, patterns in {
        "date": ["date"],
        "home": ["home", "hometeam"],
        "away": ["away", "awayteam"],
        "fthg": ["fthg", "homegoals"],
        "ftag": ["ftag", "awaygoals"],
        "hs": ["hs", "homeshots"],
        "as": ["as", "awayshots"],
        "hc": ["hc", "homecorners"],
        "ac": ["ac", "awaycorners"],
    }.items():
        for p in patterns:
            mask = cols.str.contains(p, regex=False)
            if mask.any():
                mapping[key] = df.columns[mask.argmax()]
                break
    return mapping

col_map = detect_columns(df)
if len(col_map) < 9:
    st.error("Could not find all required columns. Expected: Date, HomeTeam, AwayTeam, FTHG, FTAG, HS, AS, HC, AC")
    st.stop()

# Rename for internal consistency
df = df.rename(columns={v: k.upper() for k, v in col_map.items()})
df["DATE"] = pd.to_datetime(df["DATE"])

# ================================
# INJURY PARSING (optional)
# ================================
def parse_injuries(text: str) -> Dict[str, List[str]]:
    """Very simple parser – expects lines like: Man City: Haaland (out)"""
    injuries = {}
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        if ":" not in line:
            continue
        team, rest = line.split(":", 1)
        team = team.strip()
        players = [p.strip().split("(")[0].strip() for p in rest.split(",") if p.strip()]
        if players:
            injuries[team] = players
    return injuries

injuries_text = st.sidebar.text_area("Paste injury list (optional)", height=100)
injuries = parse_injuries(injuries_text) if injuries_text else {}

# ================================
# FORM-BASED STATS
# ================================
def compute_form_based_stats(df: pd.DataFrame, last_n: int = 5) -> Dict[str, Any]:
    """Calculate attack/defence ratings for the last N home/away games."""
    home_games = df.sort_values("DATE").copy()
    away_games = df.sort_values("DATE").copy()

    # ----- HOME -----
    home_stats = []
    for team in home_games["HOMETEAM"].unique():
        matches = home_games[home_games["HOMETEAM"] == team].tail(last_n)
        if len(matches) == 0:
            continue
        goals_for = matches["FTHG"].mean()
        goals_against = matches["FTAG"].mean()
        shots_for = matches["HS"].mean()
        shots_against = matches["AS"].mean()
        corners_for = matches["HC"].mean()
        corners_against = matches["AC"].mean()
        home_stats.append({
            "team": team,
            "games": len(matches),
            "goals_for": goals_for,
            "goals_against": goals_against,
            "shots_for": shots_for,
            "shots_against": shots_against,
            "corners_for": corners_for,
            "corners_against": corners_against,
        })
    home_df = pd.DataFrame(home_stats)

    # ----- AWAY -----
    away_stats = []
    for team in away_games["AWAYTEAM"].unique():
        matches = away_games[away_games["AWAYTEAM"] == team].tail(last_n)
        if len(matches) == 0:
            continue
        goals_for = matches["FTAG"].mean()
        goals_against = matches["FTHG"].mean()
        shots_for = matches["AS"].mean()
        shots_against = matches["HS"].mean()
        corners_for = matches["AC"].mean()
        corners_against = matches["HC"].mean()
        away_stats.append({
            "team": team,
            "games": len(matches),
            "goals_for": goals_for,
            "goals_against": goals_against,
            "shots_for": shots_for,
            "shots_against": shots_against,
            "corners_for": corners_for,
            "corners_against": corners_against,
        })
    away_df = pd.DataFrame(away_stats)

    # League averages
    league_home_avg = {
        "goals": home_df["goals_for"].mean(),
        "goals_conceded": home_df["goals_against"].mean(),
        "shots_for": home_df["shots_for"].mean(),
        "shots_against": home_df["shots_against"].mean(),
        "corners_for": home_df["corners_for"].mean(),
        "corners_against": home_df["corners_against"].mean(),
    }
    league_away_avg = {
        "goals": away_df["goals_for"].mean(),
        "goals_conceded": away_df["goals_against"].mean(),
        "shots_for": away_df["shots_for"].mean(),
        "shots_against": away_df["shots_against"].mean(),
        "corners_for": away_df["corners_for"].mean(),
        "corners_against": away_df["corners_against"].mean(),
    }

    # Attack / Defence strengths
    def strength(col, league_avg):
        return {row["team"]: row[col] / league_avg for _, row in col.items()}

    stats = {
        "goals": {
            "home_attack": strength(home_df.set_index("team")["goals_for"], league_home_avg["goals"]),
            "home_defence": strength(home_df.set_index("team")["goals_against"], league_home_avg["goals_conceded"]),
            "away_attack": strength(away_df.set_index("team")["goals_for"], league_away_avg["goals"]),
            "away_defence": strength(away_df.set_index("team")["goals_against"], league_away_avg["goals_conceded"]),
            "league_avg_home": league_home_avg["goals"],
            "league_avg_away": league_away_avg["goals"],
            "games_used": {row["team"]: row["games"] for _, row in home_df.iterrows()},
            "away_games_used": {row["team"]: row["games"] for _, row in away_df.iterrows()},
        },
        "corners": {
            "home_attack": strength(home_df.set_index("team")["corners_for"], league_home_avg["corners_for"]),
            "home_defence": strength(home_df.set_index("team")["corners_against"], league_home_avg["corners_against"]),
            "away_attack": strength(away_df.set_index("team")["corners_for"], league_away_avg["corners_for"]),
            "away_defence": strength(away_df.set_index("team")["corners_against"], league_away_avg["corners_against"]),
            "league_avg_home": league_home_avg["corners_for"],
            "league_avg_away": league_away_avg["corners_for"],
        },
    }
    return stats

stats = compute_form_based_stats(df, last_n=5)

# ================================
# LEAGUE STATS (for shot realism)
# ================================
def calculate_league_stats(df: pd.DataFrame) -> Dict[str, Any]:
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
    league_home_avg = {
        "shots_for": df["HS"].mean(),
        "shots_against": df["AS"].mean(),
    }
    league_away_avg = {
        "shots_for": df["AS"].mean(),
        "shots_against": df["HS"].mean(),
    }
    return {
        "team_home_stats": home.to_dict(orient="index"),
        "team_away_stats": away.to_dict(orient="index"),
        "league_home_avg": league_home_avg,
        "league_away_avg": league_away_avg,
    }

league_stats = calculate_league_stats(df)

# ================================
# REALISTIC SHOT ADJUSTMENT FUNCTIONS
# ================================
def adjust_to_bookmaker_level(raw_shots: float, team_type: str = "home") -> float:
    if team_type == "home":
        adjustment_factor = 0.50
    else:
        adjustment_factor = 0.45

    if raw_shots > 20:
        adjustment_factor *= 0.8
    elif raw_shots > 15:
        adjustment_factor *= 0.9

    adjusted_shots = raw_shots * adjustment_factor

    if team_type == "home":
        return max(min(adjusted_shots, 7.5), 2.5)
    else:
        return max(min(adjusted_shots, 6.5), 2.0)

def predict_realistic_shots(home_team: str, away_team: str, stats: Dict[str, Any], league_stats: Dict[str, Any]) -> Dict[str, Any]:
    predictions = {
        'home_shots': 0, 'away_shots': 0,
        'home_shots_conceded': 0, 'away_shots_conceded': 0,
        'total_shots': 0,
        'home_shot_efficiency': 0.12, 'away_shot_efficiency': 0.10,
        'raw_home_shots': 0, 'raw_away_shots': 0,
    }

    h_stats = league_stats['team_home_stats'].get(home_team, {})
    a_stats = league_stats['team_away_stats'].get(away_team, {})
    l_home = league_stats.get('league_home_avg', {})
    l_away = league_stats.get('league_away_avg', {})

    # ---- RAW ----
    raw_home = raw_away = 0.0
    if h_stats.get('shots_for') and l_home.get('shots_for'):
        home_factor = h_stats['shots_for'] / l_home['shots_for']
        away_def_factor = a_stats.get('shots_against', l_away.get('shots_against', 1)) / l_away.get('shots_against', 1)
        raw_home = l_home['shots_for'] * home_factor * (2 - away_def_factor) / 2

    if a_stats.get('shots_for') and l_away.get('shots_for'):
        away_factor = a_stats['shots_for'] / l_away['shots_for']
        home_def_factor = h_stats.get('shots_against', l_home.get('shots_against', 1)) / l_home.get('shots_against', 1)
        raw_away = l_away['shots_for'] * away_factor * (2 - home_def_factor) / 2

    predictions['raw_home_shots'] = raw_home
    predictions['raw_away_shots'] = raw_away

    # ---- BOOKMAKER ADJUSTED ----
    predictions['home_shots'] = round(adjust_to_bookmaker_level(raw_home, "home"), 1)
    predictions['away_shots'] = round(adjust_to_bookmaker_level(raw_away, "away"), 1)

    predictions['home_shots_conceded'] = round(adjust_to_bookmaker_level(h_stats.get('shots_against', 8), "away"), 1)
    predictions['away_shots_conceded'] = round(adjust_to_bookmaker_level(a_stats.get('shots_against', 7), "home"), 1)

    predictions['total_shots'] = round(predictions['home_shots'] + predictions['away_shots'], 1)

    # Efficiency (keep raw)
    predictions['home_shot_efficiency'] = h_stats.get('shot_efficiency', 0.12)
    predictions['away_shot_efficiency'] = a_stats.get('shot_efficiency', 0.10)

    return predictions

def get_bookmaker_comparison(home_shots: float, away_shots: float) -> Dict[str, Any]:
    total = home_shots + away_shots
    lines = {'very_low':17.5, 'low':20.5, 'medium':23.5, 'high':26.5, 'very_high':29.5}
    if total <= 19: line = 'very_low'
    elif total <= 22: line = 'low'
    elif total <= 25: line = 'medium'
    elif total <= 28: line = 'high'
    else: line = 'very_high'
    return {'aligned_line':line, 'bookmaker_line':lines[line], 'our_total':total,
            'difference':total-lines[line]}

# ================================
# SHOT PROBABILITIES
# ================================
def calculate_shot_probabilities(home_shots: float, away_shots: float,
                                 home_conceded: float, away_conceded: float,
                                 home_eff: float, away_eff: float) -> Dict[str, Any]:
    home_xg = home_shots * home_eff
    away_xg = away_shots * away_eff
    total = home_shots + away_shots
    return {
        'home_expected_goals_from_shots': home_xg,
        'away_expected_goals_from_shots': away_xg,
        'home_most_likely_shots': int(round(home_shots)),
        'away_most_likely_shots': int(round(away_shots)),
        'home_most_likely_shots_conceded': int(round(home_conceded)),
        'away_most_likely_shots_conceded': int(round(away_conceded)),
        'total_expected_goals': home_xg + away_xg,
        'both_teams_4_plus_shots_prob': (1-poisson.cdf(3.5, home_shots)) * (1-poisson.cdf(3.5, away_shots)),
        'over_total_shots_prob': 1-poisson.cdf(total-0.5, total),
        'home_under_shots_prob': poisson.cdf(5.5, home_shots),
        'away_under_shots_prob': poisson.cdf(4.5, away_shots),
    }

# ================================
# PREDICTION ENGINE
# ================================
@st.cache_data(show_spinner=False)
def predict_form_based_match(home: str, away: str, stats: Dict, injuries: Dict = None, league_stats: Dict = None) -> Dict:
    injury_summary = ""  # placeholder – you can expand later
    pred = {
        "goals": {"score":"N/A","home_win":0,"draw":0,"away_win":0,"btts_yes":0,"over_25":0},
        "xg": {"home":0.0,"away":0.0},
        "corners": {"home":0,"away":0,"total":0},
        "shots": {"home":0,"away":0,"total":0,"home_efficiency":0.0,"away_efficiency":0.0},
        "shot_probabilities": {},
        "form_based": True,
        "injury_summary": injury_summary,
        "games_used": {"home": stats["goals"]["games_used"].get(home,0),
                       "away": stats["goals"]["away_games_used"].get(away,0)}
    }

    # ---- GOALS ----
    g = stats.get("goals", {})
    if g:
        l_home = g["league_avg_home"]; l_away = g["league_avg_away"]
        att_h = g["home_attack"].get(home,1.0); def_a = g["away_defence"].get(away,1.0)
        att_a = g["away_attack"].get(away,1.0); def_h = g["home_defence"].get(home,1.0)
        lambda_h = att_h * def_a * l_home
        lambda_a = att_a * def_h * l_away
        max_g = 8
        prob = np.zeros((max_g+1, max_g+1))
        for h in range(max_g+1):
            for a in range(max_g+1):
                prob[h,a] = poisson.pmf(h, lambda_h) * poisson.pmf(a, lambda_a)
        prob /= prob.sum()
        h_idx, a_idx = np.unravel_index(np.argmax(prob), prob.shape)
        pred["goals"]["score"] = f"{h_idx}-{a_idx}"
        pred["goals"]["home_win"] = (prob[1:,:].sum() - np.trace(prob,1))
        pred["goals"]["away_win"] = (prob[:,1:].sum() - np.trace(prob,1))
        pred["goals"]["draw"] = np.trace(prob)
        pred["goals"]["btts_yes"] = prob[1:,1:].sum()
        pred["goals"]["over_25"] = (prob[3:,:].sum() + prob[:,3:].sum() - prob[3:,3:].sum())
        pred["xg"]["home"] = max(round(lambda_h,2),0.1)
        pred["xg"]["away"] = max(round(lambda_a,2),0.1)

    # ---- CORNERS ----
    c = stats.get("corners")
    if c:
        mu_hc = c["home_attack"].get(home,1.0) * c["away_defence"].get(away,1.0) * c["league_avg_home"]
        mu_ac = c["away_attack"].get(away,1.0) * c["home_defence"].get(home,1.0) * c["league_avg_away"]
        pred["corners"]["home"] = max(int(round(mu_hc)),1)
        pred["corners"]["away"] = max(int(round(mu_ac)),1)
        pred["corners"]["total"] = pred["corners"]["home"] + pred["corners"]["away"]

    # ---- REALISTIC SHOTS ----
    if league_stats:
        shot_pred = predict_realistic_shots(home, away, stats, league_stats)
        pred["shots"].update(shot_pred)
        if shot_pred['home_shots'] and shot_pred['away_shots']:
            prob_shot = calculate_shot_probabilities(
                shot_pred['home_shots'], shot_pred['away_shots'],
                shot_pred['home_shots_conceded'], shot_pred['away_shots_conceded'],
                shot_pred['home_shot_efficiency'], shot_pred['away_shot_efficiency']
            )
            pred["shot_probabilities"] = prob_shot

    return {"predictions": pred}

# ================================
# DISPLAY HELPERS
# ================================
def create_comparison_tables(league_stats: Dict, home: str, away: str):
    # Very small tables – home vs league home avg, away vs league away avg
    h = league_stats['team_home_stats'].get(home, {})
    a = league_stats['team_away_stats'].get(away, {})
    l_h = league_stats['league_home_avg']
    l_a = league_stats['league_away_avg']

    home_df = pd.DataFrame({
        "Metric": ["Shots For", "Shots Against", "Shot Efficiency"],
        f"{home}": [h.get('shots_for',0), h.get('shots_against',0), f"{h.get('shot_efficiency',0):.1%}"],
        "League Avg": [l_h.get('shots_for',0), l_h.get('shots_against',0), "-"]
    })
    away_df = pd.DataFrame({
        "Metric": ["Shots For", "Shots Against", "Shot Efficiency"],
        f"{away}": [a.get('shots_for',0), a.get('shots_against',0), f"{a.get('shot_efficiency',0):.1%}"],
        "League Avg": [l_a.get('shots_for',0), l_a.get('shots_against',0), "-"]
    })
    return home_df, away_df

def display_form_based_predictions(pred: Dict, home_team: str, away_team: str,
                                   stats: Dict, league_stats: Dict):
    p = pred["predictions"]

    st.markdown(f"### **{home_team} vs {away_team}**")
    st.markdown("#### Last 5 Games Form Analysis")

    logos = {home_team: get_team_logo(home_team), away_team: get_team_logo(away_team)}
    colA, colB, colC = st.columns([1,2,1])
    with colA:
        if logos[home_team]:
            img = load_image(logos[home_team])
            if img: st.image(img, width=80)
        st.write(f"**{home_team}**")
        st.caption(f"Last {p['games_used']['home']} home games")
    with colC:
        if logos[away_team]:
            img = load_image(logos[away_team])
            if img: st.image(img width=80)
        st.write(f"**{away_team}**")
        st.caption(f"Last {p['games_used']['away']} away games")
    with colB:
        st.markdown(f"<h2 style='text-align:center'>{p['goals']['score']}</h2>", unsafe_allow_html=True)
        st.caption("Most likely score")

    # Win probs
    c1,c2,c3 = st.columns(3)
    c1.metric("Home Win", f"{p['goals']['home_win']:.1%}")
    c2.metric("Draw", f"{p['goals']['draw']:.1%}")
    c3.metric("Away Win", f"{p['goals']['away_win']:.1%}")

    cB1,cB2 = st.columns(2)
    cB1.metric("BTTS", f"{p['goals']['btts_yes']:.1%}")
    cB2.metric("Over 2.5", f"{p['goals']['over_25']:.1%}")

    # ---------- REALISTIC SHOTS ----------
    st.markdown("---")
    st.markdown("#### Realistic Shot Predictions (Bookmaker-Adjusted)")

    if p['shots']['home_shots'] and p['shots']['away_shots']:
        comp = get_bookmaker_comparison(p['shots']['home_shots'], p['shots']['away_shots'])
        st.info(f"Market Alignment: **{comp['our_total']:.1f}** shots → bookmaker **{comp['bookmaker_line']:.1f}** line")

        s1,s2,s3 = st.columns(3)
        with s1:
            st.metric(f"{home_team} Shots", f"{p['shots']['home_shots']:.1f}")
            st.metric(f"{home_team} Conceded", f"{p['shots']['home_shots_conceded']:.1f}")
            st.metric("Efficiency", f"{p['shots']['home_shot_efficiency']:.1%}")
            if p['shot_probabilities']:
                st.metric("xG from Shots", f"{p['shot_probabilities']['home_expected_goals_from_shots']:.2f}")
            with st.expander("Raw vs Adjusted"):
                st.write(f"Raw: {p['shots']['raw_home_shots']:.1f}")
                st.write(f"Adjusted: {p['shots']['home_shots']:.1f}")
        with s2:
            st.metric(f"{away_team} Shots", f"{p['shots']['away_shots']:.1f}")
            st.metric(f"{away_team} Conceded", f"{p['shots']['away_shots_conceded']:.1f}")
            st.metric("Efficiency", f"{p['shots']['away_shot_efficiency']:.1%}")
            if p['shot_probabilities']:
                st.metric("xG from Shots", f"{p['shot_probabilities']['away_expected_goals_from_shots']:.2f}")
            with st.expander("Raw vs Adjusted"):
                st.write(f"Raw: {p['shots']['raw_away_shots']:.1f}")
                st.write(f"Adjusted: {p['shots']['away_shots']:.1f}")
        with s3:
            st.metric("Total Shots", f"{p['shots']['total_shots']:.1f}")
            st.metric("Shot Advantage", f"{p['shots']['home_shots']-p['shots']['away_shots']:+.1f}")
            if p['shot_probabilities']:
                st.metric("Both 4+ Shots", f"{p['shot_probabilities']['both_teams_4_plus_shots_prob']:.1%}")

        if p['shot_probabilities']:
            st.markdown("##### Shot Market Insights")
            i1,i2 = st.columns(2)
            with i1:
                st.metric("Home Most Likely", p['shot_probabilities']['home_most_likely_shots'])
                st.metric("Away Most Likely", p['shot_probabilities']['away_most_likely_shots'])
                st.metric("Home <5.5", f"{p['shot_probabilities']['home_under_shots_prob']:.1%}")
            with i2:
                st.metric("Away <4.5", f"{p['shot_probabilities']['away_under_shots_prob']:.1%}")
                st.metric(f"Over {comp['bookmaker_line']:.1f}", f"{p['shot_probabilities']['over_total_shots_prob']:.1%}")
                st.metric("Total xG", f"{p['shot_probabilities']['total_expected_goals']:.2f}")

        st.markdown("---")
        st.markdown("#### Why Bookmaker Totals Are Lower")
        st.markdown("""
        * Statistical models count **blocked** shots – bookmakers often exclude them.  
        * Teams that are leading **slow the game down**.  
        * Possession-heavy sides produce **fewer countable shots**.  
        * Bookmakers add a **public-bias buffer**.  

        **Our adjustment:** 50 % for home, 45 % for away → realistic 3-7 / 2-6 range.
        """)
    else:
        st.info("Shot columns (HS/AS) missing – upload a CSV with them to see realistic predictions.")

    # ---------- LEAGUE COMPARISON ----------
    st.markdown("---")
    st.markdown("#### League Performance Comparison")
    home_tbl, away_tbl = create_comparison_tables(league_stats, home_team, away_team)
    cL1, cL2 = st.columns(2)
    with cL1:
        if not home_tbl.empty:
            st.markdown(f"##### {home_team} (Home) vs League")
            st.dataframe(home_tbl, use_container_width=True, hide_index=True)
    with cL2:
        if not away_tbl.empty:
            st.markdown(f"##### {away_team} (Away) vs League")
            st.dataframe(away_tbl, use_container_width=True, hide_index=True)

# ================================
# MAIN UI – TEAM SELECTION
# ================================
teams = sorted(set(df["HOMETEAM"].unique()).union(df["AWAYTEAM"].unique()))
colT1, colT2 = st.columns(2)
home_team = colT1.selectbox("Home Team", options=teams, index=0)
away_team = colT2.selectbox("Away Team", options=teams, index=1 if len(teams)>1 else 0)

if home_team == away_team:
    st.warning("Please pick two different teams.")
else:
    result = predict_form_based_match(home_team, away_team, stats, injuries, league_stats)
    display_form_based_predictions(result, home_team, away_team, stats, league_stats)

# ================================
# INSTALL INSTRUCTIONS (run once)
# ================================
st.sidebar.markdown("---")
st.sidebar.markdown("**First-time install**")
st.sidebar.code("""
pip install streamlit pandas numpy scipy pillow requests plotly
""")

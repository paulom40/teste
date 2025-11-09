# app.py - FOOTBALL PREDICTOR PRO: Realistic Shots + xG + xGA + Timeline
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
st.set_page_config(page_title="Football Predictor Pro", layout="wide")
st.title("Football Predictor Pro - Realistic Analysis")
st.markdown("""
**Realistic Match Analysis**
- **Bookmaker-Adjusted Shot Predictions**
- **Shot Location & xG/xGA Breakdown**
- **xG Timeline Over 90 Minutes**
- **Last 5 Games Form Analysis**
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

# ================================
# CSV UPLOAD & COLUMN DETECTION
# ================================
st.sidebar.header("Data Input")
uploaded_file = st.sidebar.file_uploader("Upload CSV (Date, HomeTeam, AwayTeam, FTHG, FTAG, HS, AS, HC, AC)", type=["csv"])

if uploaded_file:
    df = pd.read_csv(uploaded_file)
else:
    st.sidebar.info("No file uploaded – using **demo data** (10 matches).")
    df = load_demo_csv()

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
    st.error("Missing columns. Need: Date, HomeTeam, AwayTeam, FTHG, FTAG, HS, AS, HC, AC")
    st.stop()

df = df.rename(columns={v: k.upper() for k, v in col_map.items()})
df["DATE"] = pd.to_datetime(df["DATE"])

# ================================
# INJURY INPUT
# ================================
def parse_injuries(text: str) -> Dict[str, List[str]]:
    injuries = {}
    for line in text.splitlines():
        line = line.strip()
        if not line or ":" not in line: continue
        team, rest = line.split(":", 1)
        team = team.strip()
        players = [p.strip().split("(")[0].strip() for p in rest.split(",") if p.strip()]
        if players: injuries[team] = players
    return injuries

injuries_text = st.sidebar.text_area("Paste injuries (e.g. Man City: Haaland (out))", height=80)
injuries = parse_injuries(injuries_text) if injuries_text else {}

# ================================
# FORM-BASED STATS
# ================================
def compute_form_based_stats(df: pd.DataFrame, last_n: int = 5) -> Dict[str, Any]:
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
            "corners_for": matches["HC"].mean(), "corners_against": matches["AC"].mean(),
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
            "corners_for": matches["AC"].mean(), "corners_against": matches["HC"].mean(),
        })
    away_df = pd.DataFrame(away_stats)

    league_home_avg = {k: home_df[f"{k}_for"].mean() for k in ["goals", "shots", "corners"]}
    league_home_avg.update({f"{k}_conceded": home_df[f"{k}_against"].mean() for k in ["goals", "shots", "corners"]})
    league_away_avg = {k: away_df[f"{k}_for"].mean() for k in ["goals", "shots", "corners"]}
    league_away_avg.update({f"{k}_conceded": away_df[f"{k}_against"].mean() for k in ["goals", "shots", "corners"]})

    def strength(col, avg): return {r["team"]: r[col] / avg for _, r in col.items()}

    stats = {
        "goals": {
            "home_attack": strength(home_df.set_index("team")["goals_for"], league_home_avg["goals"]),
            "home_defence": strength(home_df.set_index("team")["goals_against"], league_home_avg["goals_conceded"]),
            "away_attack": strength(away_df.set_index("team")["goals_for"], league_away_avg["goals"]),
            "away_defence": strength(away_df.set_index("team")["goals_against"], league_away_avg["goals_conceded"]),
            "league_avg_home": league_home_avg["goals"],
            "league_avg_away": league_away_avg["goals"],
            "games_used": {r["team"]: r["games"] for _, r in home_df.iterrows()},
            "away_games_used": {r["team"]: r["games"] for _, r in away_df.iterrows()},
        },
        "corners": {
            "home_attack": strength(home_df.set_index("team")["corners_for"], league_home_avg["corners"]),
            "home_defence": strength(home_df.set_index("team")["corners_against"], league_home_avg["corners_conceded"]),
            "away_attack": strength(away_df.set_index("team")["corners_for"], league_away_avg["corners"]),
            "away_defence": strength(away_df.set_index("team")["corners_against"], league_away_avg["corners_conceded"]),
            "league_avg_home": league_home_avg["corners"],
            "league_avg_away": league_away_avg["corners"],
        },
    }
    return stats

stats = compute_form_based_stats(df, last_n=5)

# ================================
# LEAGUE STATS (for realism)
# ================================
def calculate_league_stats(df: pd.DataFrame) -> Dict[str, Any]:
    home = df.groupby("HOMETEAM").agg(
        shots_for=("HS", "mean"), shots_against=("AS", "mean"),
        shot_efficiency=("FTHG", lambda x: x.sum() / df.loc[df["HOMETEAM"].isin(x.index), "HS"].sum() if df.loc[df["HOMETEAM"].isin(x.index), "HS"].sum() else 0.12)
    )
    away = df.groupby("AWAYTEAM").agg(
        shots_for=("AS", "mean"), shots_against=("HS", "mean"),
        shot_efficiency=("FTAG", lambda x: x.sum() / df.loc[df["AWAYTEAM"].isin(x.index), "AS"].sum() if df.loc[df["AWAYTEAM"].isin(x.index), "AS"].sum() else 0.10)
    )
    league_home_avg = {"shots_for": df["HS"].mean(), "shots_against": df["AS"].mean()}
    league_away_avg = {"shots_for": df["AS"].mean(), "shots_against": df["HS"].mean()}
    return {
        "team_home_stats": home.to_dict(orient="index"),
        "team_away_stats": away.to_dict(orient="index"),
        "league_home_avg": league_home_avg,
        "league_away_avg": league_away_avg,
    }

league_stats = calculate_league_stats(df)

# ================================
# REALISTIC SHOT ADJUSTMENT
# ================================
def adjust_to_bookmaker_level(raw_shots: float, team_type: str = "home") -> float:
    factor = 0.50 if team_type == "home" else 0.45
    if raw_shots > 20: factor *= 0.8
    elif raw_shots > 15: factor *= 0.9
    adjusted = raw_shots * factor
    if team_type == "home": return max(min(adjusted, 7.5), 2.5)
    else: return max(min(adjusted, 6.5), 2.0)

def predict_realistic_shots(home_team: str, away_team: str, stats: Dict, league_stats: Dict) -> Dict[str, Any]:
    predictions = {
        'home_shots': 0, 'away_shots': 0, 'home_shots_conceded': 0, 'away_shots_conceded': 0,
        'total_shots': 0, 'home_shot_efficiency': 0.12, 'away_shot_efficiency': 0.10,
        'raw_home_shots': 0, 'raw_away_shots': 0,
    }
    h_stats = league_stats['team_home_stats'].get(home_team, {})
    a_stats = league_stats['team_away_stats'].get(away_team, {})
    l_home = league_stats.get('league_home_avg', {})
    l_away = league_stats.get('league_away_avg', {})

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
    predictions['home_shots'] = round(adjust_to_bookmaker_level(raw_home, "home"), 1)
    predictions['away_shots'] = round(adjust_to_bookmaker_level(raw_away, "away"), 1)
    predictions['home_shots_conceded'] = round(adjust_to_bookmaker_level(h_stats.get('shots_against', 8), "away"), 1)
    predictions['away_shots_conceded'] = round(adjust_to_bookmaker_level(a_stats.get('shots_against', 7), "home"), 1)
    predictions['total_shots'] = round(predictions['home_shots'] + predictions['away_shots'], 1)
    predictions['home_shot_efficiency'] = h_stats.get('shot_efficiency', 0.12)
    predictions['away_shot_efficiency'] = a_stats.get('shot_efficiency', 0.10)
    return predictions

def get_bookmaker_comparison(home_shots: float, away_shots: float) -> Dict:
    total = home_shots + away_shots
    lines = {'very_low':17.5, 'low':20.5, 'medium':23.5, 'high':26.5, 'very_high':29.5}
    if total <= 19: line = 'very_low'
    elif total <= 22: line = 'low'
    elif total <= 25: line = 'medium'
    elif total <= 28: line = 'high'
    else: line = 'very_high'
    return {'aligned_line':line, 'bookmaker_line':lines[line], 'our_total':total, 'difference':total-lines[line]}

# ================================
# SHOT PROBABILITIES
# ================================
def calculate_shot_probabilities(home_shots: float, away_shots: float,
                                 home_conceded: float, away_conceded: float,
                                 home_eff: float, away_eff: float) -> Dict:
    home_xg = home_shots * home_eff
    away_xg = away_shots * away_eff
    total = home_shots + away_shots
    return {
        'home_expected_goals_from_shots': home_xg, 'away_expected_goals_from_shots': away_xg,
        'home_most_likely_shots': int(round(home_shots)), 'away_most_likely_shots': int(round(away_shots)),
        'total_expected_goals': home_xg + away_xg,
        'both_teams_4_plus_shots_prob': (1-poisson.cdf(3.5, home_shots)) * (1-poisson.cdf(3.5, away_shots)),
        'over_total_shots_prob': 1-poisson.cdf(total-0.5, total),
        'home_under_shots_prob': poisson.cdf(5.5, home_shots), 'away_under_shots_prob': poisson.cdf(4.5, away_shots),
    }

# ================================
# SHOT LOCATION
# ================================
def predict_shot_locations(home_shots: float, away_shots: float,
                          home_eff: float, away_eff: float,
                          league_stats: Dict, home_team: str, away_team: str) -> Dict:
    league = {"inside_box_pct": 0.68, "outside_box_pct": 0.20, "header_pct": 0.12, "set_piece_pct": 0.22}
    h_style = league_stats['team_home_stats'].get(home_team, {})
    a_style = league_stats['team_away_stats'].get(away_team, {})
    home_ib = round(home_shots * h_style.get('inside_box_ratio', league["inside_box_pct"]), 1)
    away_ib = round(away_shots * a_style.get('inside_box_ratio', league["inside_box_pct"]), 1)
    home_ob = round(home_shots * league["outside_box_pct"], 1)
    away_ob = round(away_shots * league["outside_box_pct"], 1)
    home_hdr = round(home_shots * league["header_pct"], 1)
    away_hdr = round(away_shots * league["header_pct"], 1)
    home_sp = round(home_shots * league["set_piece_pct"], 1)
    away_sp = round(away_shots * league["set_piece_pct"], 1)

    def fix_sum(team, total): 
        diff = total - sum(team.values())
        if diff != 0: team["inside_box"] += diff
        return team

    return {
        "home": fix_sum({"inside_box": home_ib, "outside_box": home_ob, "headers": home_hdr, "set_piece": home_sp}, home_shots),
        "away": fix_sum({"inside_box": away_ib, "outside_box": away_ob, "headers": away_hdr, "set_piece": away_sp}, away_shots),
        "league_avg": league
    }

# ================================
# xG PREDICTION
# ================================
def predict_xg_breakdown(home_shots: float, away_shots: float,
                        home_eff: float, away_eff: float,
                        shot_locations: Dict, league_stats: Dict,
                        home_team: str, away_team: str) -> Dict:
    xg_per_type = {"inside_box": 0.105, "outside_box": 0.038, "headers": 0.075, "set_piece": 0.080}
    home_mult = home_eff / 0.11
    away_mult = away_eff / 0.10
    loc = shot_locations

    home_xg = sum(loc["home"][k] * xg_per_type[k] for k in xg_per_type) * home_mult
    away_xg = sum(loc["away"][k] * xg_per_type[k] for k in xg_per_type) * away_mult
    total_xg = home_xg + away_xg

    market_line = 2.5
    if total_xg <= 2.1: market_line = 2.0
    elif total_xg <= 2.6: market_line = 2.5
    elif total_xg <= 3.1: market_line = 3.0
    else: market_line = 3.5

    return {
        "home_xg": round(home_xg, 2), "away_xg": round(away_xg, 2), "total_xg": round(total_xg, 2),
        "market_line": market_line, "over_25_xg_prob": 1 - poisson.cdf(2, total_xg),
        "xg_per_shot_home": round(home_xg / home_shots, 3) if home_shots > 0 else 0,
        "xg_per_shot_away": round(away_xg / away_shots, 3) if away_shots > 0 else 0,
        "breakdown": {team: {k: round(loc[team][k] * xg_per_type[k] * (home_mult if team=="home" else away_mult), 2) for k in xg_per_type} for team in ["home","away"]}
    }

# ================================
# xGA PREDICTION
# ================================
def predict_xga_breakdown(home_shots_conceded: float, away_shots_conceded: float,
                         opponent_eff_home: float, opponent_eff_away: float,
                         shot_locations: Dict, league_stats: Dict,
                         home_team: str, away_team: str) -> Dict:
    xg_per_type = {"inside_box": 0.105, "outside_box": 0.038, "headers": 0.075, "set_piece": 0.080}
    home_opp_mult = opponent_eff_away / 0.10
    away_opp_mult = opponent_eff_home / 0.11
    loc = shot_locations

    home_xga = sum(loc["away"][k] * xg_per_type[k] for k in xg_per_type) * home_opp_mult
    away_xga = sum(loc["home"][k] * xg_per_type[k] for k in xg_per_type) * away_opp_mult

    home_def_rating = round(1.0 - (home_xga / away_shots_conceded) / 0.10, 2) if away_shots_conceded > 0 else 1.0
    away_def_rating = round(1.0 - (away_xga / home_shots_conceded) / 0.11, 2) if home_shots_conceded > 0 else 1.0

    return {
        "home_xga": round(home_xga, 2), "away_xga": round(away_xga, 2), "total_xga": round(home_xga + away_xga, 2),
        "home_def_rating": max(home_def_rating, 0), "away_def_rating": max(away_def_rating, 0),
        "xga_per_shot_conceded_home": round(home_xga / away_shots_conceded, 3) if away_shots_conceded > 0 else 0,
        "xga_per_shot_conceded_away": round(away_xga / home_shots_conceded, 3) if home_shots_conceded > 0 else 0,
        "breakdown": {
            "home": {k: round(loc["away"][k] * xg_per_type[k] * home_opp_mult, 2) for k in xg_per_type},
            "away": {k: round(loc["home"][k] * xg_per_type[k] * away_opp_mult, 2) for k in xg_per_type}
        }
    }

# ================================
# xG TIMELINE
# ================================
def predict_xg_timeline(home_xg: float, away_xg: float,
                       home_shots: float, away_shots: float,
                       home_team: str, away_team: str) -> Dict:
    minutes = list(range(0, 91, 5))
    timeline = {"minutes": minutes, "home": [], "away": [], "total": []}
    home_pace = home_xg / 90
    away_pace = away_xg / 90

    def pace_modifier(m, leading, shots_so_far):
        mod = 1.0
        if m > 60: mod *= 0.92
        if m > 70: mod *= 1.08
        if leading and m > 45: mod *= 0.85
        if shots_so_far < 2 and m < 30: mod *= 1.15
        return mod

    home_cum = away_cum = 0.0
    for m in minutes:
        if m == 0:
            timeline["home"].append(0); timeline["away"].append(0); timeline["total"].append(0)
            continue
        home_goals = poisson.ppf(0.5, home_cum)
        away_goals = poisson.ppf(0.5, away_cum)
        home_mod = pace_modifier(m, home_goals > away_goals, home_shots * (m/90))
        away_mod = pace_modifier(m, away_goals > home_goals, away_shots * (m/90))
        home_cum += home_pace * 5 * home_mod
        away_cum += away_pace * 5 * away_mod
        timeline["home"].append(round(home_cum, 2))
        timeline["away"].append(round(away_cum, 2))
        timeline["total"].append(round(home_cum + away_cum, 2))
    return timeline

# ================================
# PREDICTION ENGINE
# ================================
@st.cache_data(show_spinner=False)
def predict_form_based_match(home: str, away: str, stats: Dict, injuries: Dict = None, league_stats: Dict = None) -> Dict:
    pred = {
        "goals": {"score":"N/A","home_win":0,"draw":0,"away_win":0,"btts_yes":0,"over_25":0},
        "xg": {"home":0.0,"away":0.0}, "corners": {"home":0,"away":0,"total":0},
        "shots": {"home":0,"away":0,"total":0,"home_efficiency":0.0,"away_efficiency":0.0},
        "shot_probabilities": {}, "shot_locations": {}, "xg_detailed": {}, "xga_detailed": {}, "xg_timeline": {},
        "form_based": True, "injury_summary": "", "games_used": {"home":0, "away":0}
    }

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

    c = stats.get("corners")
    if c:
        mu_hc = c["home_attack"].get(home,1.0) * c["away_defence"].get(away,1.0) * c["league_avg_home"]
        mu_ac = c["away_attack"].get(away,1.0) * c["home_defence"].get(home,1.0) * c["league_avg_away"]
        pred["corners"]["home"] = max(int(round(mu_hc)),1)
        pred["corners"]["away"] = max(int(round(mu_ac)),1)
        pred["corners"]["total"] = pred["corners"]["home"] + pred["corners"]["away"]

    if league_stats:
        shot_pred = predict_realistic_shots(home, away, stats, league_stats)
        pred["shots"].update(shot_pred)
        pred["games_used"] = {"home": stats["goals"]["games_used"].get(home,0), "away": stats["goals"]["away_games_used"].get(away,0)}

        if shot_pred['home_shots'] and shot_pred['away_shots']:
            prob_shot = calculate_shot_probabilities(
                shot_pred['home_shots'], shot_pred['away_shots'],
                shot_pred['home_shots_conceded'], shot_pred['away_shots_conceded'],
                shot_pred['home_shot_efficiency'], shot_pred['away_shot_efficiency']
            )
            pred["shot_probabilities"] = prob_shot

            locations = predict_shot_locations(
                shot_pred['home_shots'], shot_pred['away_shots'],
                shot_pred['home_shot_efficiency'], shot_pred['away_shot_efficiency'],
                league_stats, home, away
            )
            pred["shot_locations"] = locations

            xg = predict_xg_breakdown(
                shot_pred['home_shots'], shot_pred['away_shots'],
                shot_pred['home_shot_efficiency'], shot_pred['away_shot_efficiency'],
                locations, league_stats, home, away
            )
            pred["xg_detailed"] = xg
            pred["xg"]["home"] = xg["home_xg"]
            pred["xg"]["away"] = xg["away_xg"]

            xga = predict_xga_breakdown(
                shot_pred['home_shots_conceded'], shot_pred['away_shots_conceded'],
                shot_pred['home_shot_efficiency'], shot_pred['away_shot_efficiency'],
                locations, league_stats, home, away
            )
            pred["xga_detailed"] = xga

            timeline = predict_xg_timeline(
                xg["home_xg"], xg["away_xg"],
                shot_pred['home_shots'], shot_pred['away_shots'],
                home, away
            )
            pred["xg_timeline"] = timeline

    return {"predictions": pred}

# ================================
# DISPLAY
# ================================
def create_comparison_tables(league_stats: Dict, home: str, away: str):
    h = league_stats['team_home_stats'].get(home, {})
    a = league_stats['team_away_stats'].get(away, {})
    l_h = league_stats['league_home_avg']
    l_a = league_stats['league_away_avg']
    home_df = pd.DataFrame({"Metric": ["Shots For", "Shots Against", "Shot Efficiency"],
                            f"{home}": [h.get('shots_for',0), h.get('shots_against',0), f"{h.get('shot_efficiency',0):.1%}"],
                            "League Avg": [l_h.get('shots_for',0), l_h.get('shots_against',0), "-"]})
    away_df = pd.DataFrame({"Metric": ["Shots For", "Shots Against", "Shot Efficiency"],
                            f"{away}": [a.get('shots_for',0), a.get('shots_against',0), f"{a.get('shot_efficiency',0):.1%}"],
                            "League Avg": [l_a.get('shots_for',0), l_a.get('shots_against',0), "-"]})
    return home_df, away_df

def display_form_based_predictions(pred: Dict, home_team: str, away_team: str, stats: Dict, league_stats: Dict):
    p = pred["predictions"]
    st.markdown(f"### **{home_team} vs {away_team}**")
    st.markdown("#### Last 5 Games Form Analysis")

    logos = {home_team: get_team_logo(home_team), away_team: get_team_logo(away_team)}
    colA, colB, colC = st.columns([1,2,1])
    with colA:
        if logos[home_team]: img = load_image(logos[home_team]); if img: st.image(img, width=80)
        st.write(f"**{home_team}**"); st.caption(f"Last {p['games_used']['home']} home games")
    with colC:
        if logos[away_team]: img = load_image(logos[away_team]); if img: st.image(img, width=80)
        st.write(f"**{away_team}**"); st.caption(f"Last {p['games_used']['away']} away games")
    with colB:
        st.markdown(f"<h2 style='text-align:center'>{p['goals']['score']}</h2>", unsafe_allow_html=True)
        st.caption("Most likely score")

    c1,c2,c3 = st.columns(3)
    c1.metric("Home Win", f"{p['goals']['home_win']:.1%}")
    c2.metric("Draw", f"{p['goals']['draw']:.1%}")
    c3.metric("Away Win", f"{p['goals']['away_win']:.1%}")
    cB1,cB2 = st.columns(2)
    cB1.metric("BTTS", f"{p['goals']['btts_yes']:.1%}")
    cB2.metric("Over 2.5", f"{p['goals']['over_25']:.1%}")

    # SHOTS
    st.markdown("---")
    st.markdown("#### Realistic Shot Predictions (Bookmaker-Adjusted)")
    if p['shots']['home_shots'] and p['shots']['away_shots']:
        comp = get_bookmaker_comparison(p['shots']['home_shots'], p['shots']['away_shots'])
        st.info(f"Market: **{comp['our_total']:.1f}** → bookmaker **{comp['bookmaker_line']:.1f}** line")
        s1,s2,s3 = st.columns(3)
        with s1:
            st.metric(f"{home_team} Shots", f"{p['shots']['home_shots']:.1f}")
            st.metric(f"Conceded", f"{p['shots']['home_shots_conceded']:.1f}")
            st.metric("Efficiency", f"{p['shots']['home_shot_efficiency']:.1%}")
            with st.expander("Raw vs Adjusted"):
                st.write(f"Raw: {p['shots']['raw_home_shots']:.1f} → Adjusted: {p['shots']['home_shots']:.1f}")
        with s2:
            st.metric(f"{away_team} Shots", f"{p['shots']['away_shots']:.1f}")
            st.metric(f"Conceded", f"{p['shots']['away_shots_conceded']:.1f}")
            st.metric("Efficiency", f"{p['shots']['away_shot_efficiency']:.1%}")
            with st.expander("Raw vs Adjusted"):
                st.write(f"Raw: {p['shots']['raw_away_shots']:.1f} → Adjusted: {p['shots']['away_shots']:.1f}")
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

    # SHOT LOCATIONS
    if p.get("shot_locations"):
        st.markdown("---")
        st.markdown("#### Shot Location Breakdown")
        loc = p["shot_locations"]
        col_loc1, col_loc2 = st.columns(2)
        with col_loc1:
            st.markdown(f"**{home_team}**")
            df = pd.DataFrame({"Type": ["Inside Box", "Outside Box", "Headers", "Set-Piece"],
                               "Shots": [loc["home"][k] for k in ["inside_box","outside_box","headers","set_piece"]]})
            fig = px.bar(df, x="Type", y="Shots", text="Shots", color="Type", title="Home")
            fig.update_traces(textposition='outside'); fig.update_layout(showlegend=False, height=300)
            st.plotly_chart(fig, use_container_width=True)
        with col_loc2:
            st.markdown(f"**{away_team}**")
            df = pd.DataFrame({"Type": ["Inside Box", "Outside Box", "Headers", "Set-Piece"],
                               "Shots": [loc["away"][k] for k in ["inside_box","outside_box","headers","set_piece"]]})
            fig = px.bar(df, x="Type", y="Shots", text="Shots", color="Type", title="Away")
            fig.update_traces(textposition='outside'); fig.update_layout(showlegend=False, height=300)
            st.plotly_chart(fig, use_container_width=True)
        ib_total = loc["home"]["inside_box"] + loc["away"]["inside_box"]
        st.info(f"**Inside-Box Shots Total**: {ib_total:.1f} → Typical line: **12.5–16.5**")

    # xG
    if p.get("xg_detailed"):
        st.markdown("---")
        st.markdown("#### Expected Goals (xG)")
        xg = p["xg_detailed"]
        col_xg1, col_xg2, col_xg3 = st.columns(3)
        with col_xg1: st.metric(f"**{home_team} xG**", f"{xg['home_xg']:.2f}"); st.caption(f"xG/Shot: {xg['xg_per_shot_home']:.3f}")
        with col_xg2: st.metric(f"**{away_team} xG**", f"{xg['away_xg']:.2f}"); st.caption(f"xG/Shot: {xg['xg_per_shot_away']:.3f}")
        with col_xg3: st.metric("**Total xG**", f"{xg['total_xg']:.2f}"); st.caption(f"Market Line: **{xg['market_line']:.1f}**")
        st.markdown("##### xG by Shot Type")
        df_xg = pd.DataFrame({"Type": list(xg["breakdown"]["home"].keys()), home_team: list(xg["breakdown"]["home"].values()), away_team: list(xg["breakdown"]["away"].values())}).set_index("Type")
        fig = go.Figure()
        fig.add_trace(go.Bar(name=home_team, x=df_xg.index, y=df_xg[home_team], text=df_xg[home_team], textposition='outside'))
        fig.add_trace(go.Bar(name=away_team, x=df_xg.index, y=df_xg[away_team], text=df_xg[away_team], textposition='outside'))
        fig.update_layout(barmode='stack', height=350, title="xG Contribution")
        st.plotly_chart(fig, use_container_width=True)
        st.info(f"**Over {xg['market_line']:.1f} Goals**: {xg['over_25_xg_prob']:.1%} chance")

    # xGA
    if p.get("xga_detailed"):
        st.markdown("---")
        st.markdown("#### Expected Goals Against (xGA)")
        xga = p["xga_detailed"]
        col_xga1, col_xga2, col_xga3 = st.columns(3)
        with col_xga1:
            st.metric(f"**{home_team} xGA**", f"{xga['home_xga']:.2f}")
            st.caption(f"xGA/Shot Allowed: {xga['xga_per_shot_conceded_home']:.3f}")
            st.caption(f"Def Rating: {xga['home_def_rating']:.2f}")
        with col_xga2:
            st.metric(f"**{away_team} xGA**", f"{xga['away_xga']:.2f}")
            st.caption(f"xGA/Shot Allowed: {xga['xga_per_shot_conceded_away']:.3f}")
            st.caption(f"Def Rating: {xga['away_def_rating']:.2f}")
        with col_xga3:
            st.metric("**Total xGA**", f"{xga['total_xga']:.2f}")
        st.markdown("##### xGA Allowed by Opponent")
        df_xga = pd.DataFrame({"Type": list(xga["breakdown"]["home"].keys()), home_team: list(xga["breakdown"]["home"].values()), away_team: list(xga["breakdown"]["away"].values())}).set_index("Type")
        fig = go.Figure()
        fig.add_trace(go.Bar(name=f"{home_team} xGA", x=df_xga.index, y=df_xga[home_team], text=df_xga[home_team], textposition='outside'))
        fig.add_trace(go.Bar(name=f"{away_team} xGA", x=df_xga.index, y=df_xga[away_team], text=df_xga[away_team], textposition='outside'))
        fig.update_layout(barmode='stack', height=350, title="xGA Allowed")
        st.plotly_chart(fig, use_container_width=True)
        if xga['home_def_rating'] > 1.1: st.success(f"**{home_team}** has **elite defense**")
        if xga['away_def_rating'] > 1.1: st.success(f"**{away_team}** has **elite defense**")

    # xG TIMELINE
    if p.get("xg_timeline"):
        st.markdown("---")
        st.markdown("#### xG Timeline")
        tl = p["xg_timeline"]
        df_tl = pd.DataFrame({"Minute": tl["minutes"], home_team: tl["home"], away_team: tl["away"], "Total xG": tl["total"]})
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=df_tl["Minute"], y=df_tl[home_team], mode='lines+markers', name=home_team, line=dict(width=3), marker=dict(size=6)))
        fig.add_trace(go.Scatter(x=df_tl["Minute"], y=df_tl[away_team], mode='lines+markers', name=away_team, line=dict(width=3, dash='dot'), marker=dict(size=6)))
        fig.add_trace(go.Scatter(x=df_tl["Minute"], y=df_tl["Total xG"], mode='lines', name="Total", line=dict(color='gray', width=2)))
        fig.update_layout(title="xG Accumulation", xaxis_title="Minute", yaxis_title="xG", hovermode="x unified", height=400)
        fig.add_vrect(x0=60, x1=90, fillcolor="lightgray", opacity=0.2, annotation_text="Fatigue + Subs")
        st.plotly_chart(fig, use_container_width=True)
        col_t1, col_t2 = st.columns(2)
        with col_t1:
            peak = max(enumerate(tl["home"]), key=lambda x: x[1])
            st.metric("Peak Home xG", f"{peak[1]:.2f}", f"@ {peak[0]*5}'")
        with col_t2:
            peak = max(enumerate(tl["away"]), key=lambda x: x[1])
            st.metric("Peak Away xG", f"{peak[1]:.2f}", f"@ {peak[0]*5}'")

    # LEAGUE COMPARISON
    st.markdown("---")
    st.markdown("#### League Performance")
    home_tbl, away_tbl = create_comparison_tables(league_stats, home_team, away_team)
    cL1, cL2 = st.columns(2)
    with cL1:
        if not home_tbl.empty: st.markdown(f"##### {home_team} (Home)"); st.dataframe(home_tbl, use_container_width=True, hide_index=True)
    with cL2:
        if not away_tbl.empty: st.markdown(f"##### {away_team} (Away)"); st.dataframe(away_tbl, use_container_width=True, hide_index=True)

# ================================
# MAIN UI
# ================================
teams = sorted(set(df["HOMETEAM"].unique()).union(df["AWAYTEAM"].unique()))
colT1, colT2 = st.columns(2)
home_team = colT1.selectbox("Home Team", options=teams, index=0)
away_team = colT2.selectbox("Away Team", options=teams, index=1 if len(teams)>1 else 0)

if home_team == away_team:
    st.warning("Please select two different teams.")
else:
    result = predict_form_based_match(home_team, away_team, stats, injuries, league_stats)
    display_form_based_predictions(result, home_team, away_team, stats, league_stats)

# INSTALL
st.sidebar.markdown("---")
st.sidebar.code("pip install streamlit pandas numpy scipy pillow requests plotly")

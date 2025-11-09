# app.py - FOOTBALL PREDICTOR PRO: FULLY FUNCTIONAL + EXPORT
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
# CSV & COLUMN DETECTION
# ================================
st.sidebar.header("Data Input")
uploaded_file = st.sidebar.file_uploader("Upload CSV", type=["csv"])

if uploaded_file:
    df = pd.read_csv(uploaded_file)
else:
    st.sidebar.info("Using demo data (10 matches).")
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

    league_home_avg = {"goals": home_df["goals_for"].mean(), "shots": home_df["shots_for"].mean()}
    league_away_avg = {"goals": away_df["goals_for"].mean(), "shots": away_df["shots_for"].mean()}

    def strength(col, avg): return {r["team"]: r[col] / avg for _, r in col.items()}

    stats = {
        "goals": {
            "home_attack": strength(home_df.set_index("team")["goals_for"], league_home_avg["goals"]),
            "home_defence": strength(home_df.set_index("team")["goals_against"], home_df["goals_against"].mean()),
            "away_attack": strength(away_df.set_index("team")["goals_for"], league_away_avg["goals"]),
            "away_defence": strength(away_df.set_index("team")["goals_against"], away_df["goals_against"].mean()),
            "league_avg_home": league_home_avg["goals"],
            "league_avg_away": league_away_avg["goals"],
            "games_used": {r["team"]: r["games"] for _, r in home_df.iterrows()},
            "away_games_used": {r["team"]: r["games"] for _, r in away_df.iterrows()},
        },
    }
    return stats

stats = compute_form_based_stats(df, last_n=5)

# ================================
# LEAGUE STATS
# ================================
def calculate_league_stats(df: pd.DataFrame) -> Dict:
    home = df.groupby("HOMETEAM").agg(
        shots_for=("HS", "mean"), shots_against=("AS", "mean"),
        shot_efficiency=("FTHG", lambda x: x.sum() / df.loc[df["HOMETEAM"].isin(x.index), "HS"].sum() if df.loc[df["HOMETEAM"].isin(x.index), "HS"].sum() else 0.12)
    )
    away = df.groupby("AWAYTEAM").agg(
        shots_for=("AS", "mean"), shots_against=("HS", "mean"),
        shot_efficiency=("FTAG", lambda x: x.sum() / df.loc[df["AWAYTEAM"].isin(x.index), "AS"].sum() if df.loc[df["AWAYTEAM"].isin(x.index), "AS"].sum() else 0.10)
    )
    return {
        "team_home_stats": home.to_dict(orient="index"),
        "team_away_stats": away.to_dict(orient="index"),
    }

league_stats = calculate_league_stats(df)

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
    l_home = {"shots_for": df["HS"].mean(), "shots_against": df["AS"].mean()}
    l_away = {"shots_for": df["AS"].mean(), "shots_against": df["HS"].mean()}

    raw_home = l_home['shots_for'] * (h_stats.get('shots_for', 10) / l_home['shots_for']) * (2 - a_stats.get('shots_against', 8) / l_away['shots_against']) / 2
    raw_away = l_away['shots_for'] * (a_stats.get('shots_for', 8) / l_away['shots_for']) * (2 - h_stats.get('shots_against', 10) / l_home['shots_against']) / 2

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
# SHOT LOCATION
# ================================
def predict_shot_locations(home_shots: float, away_shots: float, **kwargs) -> Dict:
    league = {"inside_box_pct": 0.68, "outside_box_pct": 0.20, "header_pct": 0.12, "set_piece_pct": 0.22}
    home_ib = round(home_shots * league["inside_box_pct"], 1)
    away_ib = round(away_shots * league["inside_box_pct"], 1)
    home_ob = round(home_shots * league["outside_box_pct"], 1)
    away_ob = round(away_shots * league["outside_box_pct"], 1)
    home_hdr = round(home_shots * league["header_pct"], 1)
    away_hdr = round(away_shots * league["header_pct"], 1)
    home_sp = round(home_shots * league["set_piece_pct"], 1)
    away_sp = round(away_shots * league["set_piece_pct"], 1)

    def fix(team, total):
        diff = total - sum(team.values())
        if diff != 0: team["inside_box"] += diff
        return team

    return {
        "home": fix({"inside_box": home_ib, "outside_box": home_ob, "headers": home_hdr, "set_piece": home_sp}, home_shots),
        "away": fix({"inside_box": away_ib, "outside_box": away_ob, "headers": away_hdr, "set_piece": away_sp}, away_shots),
    }

# ================================
# xG & xGA
# ================================
def predict_xg_breakdown(home_shots: float, away_shots: float, home_eff: float, away_eff: float, shot_locations: Dict, **kwargs) -> Dict:
    xg_per_type = {"inside_box": 0.105, "outside_box": 0.038, "headers": 0.075, "set_piece": 0.080}
    home_mult = home_eff / 0.11
    away_mult = away_eff / 0.10
    loc = shot_locations

    home_xg = sum(loc["home"][k] * xg_per_type[k] for k in xg_per_type) * home_mult
    away_xg = sum(loc["away"][k] * xg_per_type[k] for k in xg_per_type) * away_mult
    total_xg = home_xg + away_xg

    return {
        "home_xg": round(home_xg, 2), "away_xg": round(away_xg, 2), "total_xg": round(total_xg, 2),
        "xg_per_shot_home": round(home_xg / home_shots, 3) if home_shots > 0 else 0,
        "xg_per_shot_away": round(away_xg / away_shots, 3) if away_shots > 0 else 0,
    }

def predict_xga_breakdown(home_shots_conceded: float, away_shots_conceded: float,
                         opponent_eff_home: float, opponent_eff_away: float,
                         shot_locations: Dict, **kwargs) -> Dict:
    xg_per_type = {"inside_box": 0.105, "outside_box": 0.038, "headers": 0.075, "set_piece": 0.080}
    home_opp_mult = opponent_eff_away / 0.10
    away_opp_mult = opponent_eff_home / 0.11
    loc = shot_locations

    home_xga = sum(loc["away"][k] * xg_per_type[k] for k in xg_per_type) * home_opp_mult
    away_xga = sum(loc["home"][k] * xg_per_type[k] for k in xg_per_type) * away_opp_mult

    return {
        "home_xga": round(home_xga, 2), "away_xga": round(away_xga, 2),
    }

# ================================
# xG TIMELINE
# ================================
def predict_xg_timeline(home_xg: float, away_xg: float, home_shots: float, away_shots: float, **kwargs) -> Dict:
    minutes = list(range(0, 91, 5))
    timeline = {"minutes": minutes, "home": [], "away": [], "total": []}
    home_pace = home_xg / 90
    away_pace = away_xg / 90

    home_cum = away_cum = 0.0
    for m in minutes:
        if m == 0:
            timeline["home"].append(0); timeline["away"].append(0); timeline["total"].append(0)
            continue
        mod = 1.0
        if m > 60: mod *= 0.92
        if m > 70: mod *= 1.08
        home_cum += home_pace * 5 * mod
        away_cum += away_pace * 5 * mod
        timeline["home"].append(round(home_cum, 2))
        timeline["away"].append(round(away_cum, 2))
        timeline["total"].append(round(home_cum + away_cum, 2))
    return timeline

# ================================
# PREDICTION ENGINE
# ================================
@st.cache_data(show_spinner=False)
def predict_form_based_match(home: str, away: str, stats: Dict, injuries: Dict, league_stats: Dict) -> Dict:
    pred = {
        "goals": {"score":"N/A","home_win":0,"draw":0,"away_win":0,"btts_yes":0,"over_25":0},
        "xg": {"home":0.0,"away":0.0}, "shots": {}, "shot_locations": {}, "xg_detailed": {}, "xga_detailed": {}, "xg_timeline": {},
        "games_used": {"home":0, "away":0}
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
        pred["goals"]["draw"] = np.trace(prob)
        pred["goals"]["away_win"] = (prob[:,1:].sum() - np.trace(prob,1))
        pred["goals"]["btts_yes"] = prob[1:,1:].sum()
        pred["goals"]["over_25"] = (prob[3:,:].sum() + prob[:,3:].sum() - prob[3:,3:].sum())
        pred["xg"]["home"] = max(round(lambda_h,2),0.1)
        pred["xg"]["away"] = max(round(lambda_a,2),0.1)

    if league_stats:
        shot_pred = predict_realistic_shots(home, away, stats, league_stats)
        shot_pred['total_shots'] = round(shot_pred['home_shots'] + shot_pred['away_shots'], 1)
        pred["shots"] = shot_pred
        pred["games_used"] = {"home": stats["goals"]["games_used"].get(home,0), "away": stats["goals"]["away_games_used"].get(away,0)}

        locations = predict_shot_locations(**shot_pred)
        pred["shot_locations"] = locations

        xg = predict_xg_breakdown(**shot_pred, shot_locations=locations)
        pred["xg_detailed"] = xg
        pred["xg"]["home"] = xg["home_xg"]
        pred["xg"]["away"] = xg["away_xg"]

        xga = predict_xga_breakdown(**shot_pred, shot_locations=locations)
        pred["xga_detailed"] = xga

        timeline = predict_xg_timeline(xg["home_xg"], xg["away_xg"], **shot_pred)
        pred["xg_timeline"] = timeline

    return {"predictions": pred}

# ================================
# DISPLAY
# ================================
def display_form_based_predictions(pred: Dict, home_team: str, away_team: str, stats: Dict, league_stats: Dict):
    p = pred["predictions"]
    st.markdown(f"### **{home_team} vs {away_team}**")
    st.markdown("#### Last 5 Games Form")

    logos = {home_team: get_team_logo(home_team), away_team: get_team_logo(away_team)}
    colA, colB, colC = st.columns([1,2,1])
    with colA:
        if logos[home_team]:
            img = load_image(logos[home_team])
            if img:
                st.image(img, width=80)  # FIXED: comma added
        st.write(f"**{home_team}**")
        st.caption(f"Last {p['games_used']['home']} home games")
    with colC:
        if logos[away_team]:
            img = load_image(logos[away_team])
            if img:
                st.image(img, width=80)  # FIXED: comma added
        st.write(f"**{away_team}**")
        st.caption(f"Last {p['games_used']['away']} away games")
    with colB:
        st.markdown(f"<h2 style='text-align:center'>{p['goals']['score']}</h2>", unsafe_allow_html=True)

    c1,c2,c3 = st.columns(3)
    c1.metric("Home Win", f"{p['goals']['home_win']:.1%}")
    c2.metric("Draw", f"{p['goals']['draw']:.1%}")
    c3.metric("Away Win", f"{p['goals']['away_win']:.1%}")

    # Shots
    if p['shots']:
        st.markdown("---")
        st.markdown("#### Shots")
        s1,s2,s3 = st.columns(3)
        with s1:
            st.metric(f"{home_team} Shots", f"{p['shots']['home_shots']:.1f}")
            st.metric("xG", f"{p['xg']['home']:.2f}")
        with s2:
            st.metric(f"{away_team} Shots", f"{p['shots']['away_shots']:.1f}")
            st.metric("xG", f"{p['xg']['away']:.2f}")
        with s3:
            st.metric("Total Shots", f"{p['shots']['total_shots']:.1f}")

    # xG Timeline
    if p.get("xg_timeline"):
        st.markdown("---")
        st.markdown("#### xG Timeline")
        tl = p["xg_timeline"]
        df_tl = pd.DataFrame({"Minute": tl["minutes"], home_team: tl["home"], away_team: tl["away"]})
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=df_tl["Minute"], y=df_tl[home_team], mode='lines+markers', name=home_team))
        fig.add_trace(go.Scatter(x=df_tl["Minute"], y=df_tl[away_team], mode='lines+markers', name=away_team))
        fig.update_layout(height=350)
        st.plotly_chart(fig, use_container_width=True)

# ================================
# CONCISE EXPORT
# ================================
def export_concise_report(p, home, away):
    score = p["goals"]["score"]
    home_win = f"{p['goals']['home_win']:.0%}"
    draw = f"{p['goals']['draw']:.0%}"
    away_win = f"{p['goals']['away_win']:.0%}"
    shots_h = f"{p['shots']['home_shots']:.1f}"
    shots_a = f"{p['shots']['away_shots']:.1f}"
    xg_h = f"{p['xg']['home']:.2f}"
    xg_a = f"{p['xg']['away']:.2f}"
    peak_h = max(p["xg_timeline"]["home"])
    peak_a = max(p["xg_timeline"]["away"])
    min_h = p["xg_timeline"]["minutes"][p["xg_timeline"]["home"].index(peak_h)]
    min_a = p["xg_timeline"]["minutes"][p["xg_timeline"]["away"].index(peak_a)]

    html = f"""
    <html><head><meta charset="utf-8"><style>
      body{{font-family:Helvetica;margin:1cm;line-height:1.4}}
      h1{{text-align:center;color:#1f77b4}}
      table{{width:100%;border-collapse:collapse}}
      th,td{{border:1px solid #aaa;padding:4px;text-align:center}}
      th{{background:#f0f0f0}}
    </style></head><body>
    <h1>{home} vs {away}</h1>
    <p style="text-align:center"><b>{score}</b> | {datetime.now():%Y-%m-%d %H:%M}</p>
    <table>
      <tr><th>Home Win</th><th>Draw</th><th>Away Win</th></tr>
      <tr><td>{home_win}</td><td>{draw}</td><td>{away_win}</td></tr>
    </table>
    <table style="margin-top:10px">
      <tr><th></th><th>{home}</th><th>{away}</th></tr>
      <tr><td>Shots</td><td>{shots_h}</td><td>{shots_a}</td></tr>
      <tr><td>xG</td><td>{xg_h}</td><td>{xg_a}</td></tr>
      <tr><td>Peak xG</td><td>{peak_h:.2f} @ {min_h}'</td><td>{peak_a:.2f} @ {min_a}'</td></tr>
    </table>
    </body></html>
    """
    html_io = io.BytesIO(html.encode())
    st.download_button("HTML Report", html_io.getvalue(), f"{home}_vs_{away}.html", "text/html")

    try:
        from weasyprint import HTML, CSS
        css = CSS(string="@page {size:A5 landscape; margin:.5cm}")
        pdf = HTML(string=html).write_pdf(stylesheets=[css])
        st.download_button("PDF Report", pdf, f"{home}_vs_{away}.pdf", "application/pdf")
    except Exception:
        st.caption("Install `weasyprint` for PDF")

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

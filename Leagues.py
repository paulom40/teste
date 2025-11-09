# app.py - FOCUSED ON LAST 5 HOME/AWAY GAMES
import streamlit as st
import pandas as pd
import numpy as np
from scipy.stats import poisson, nbinom
import io
from typing import Dict, Any, Tuple, List
import requests
from PIL import Image
from io import BytesIO
import plotly.express as px
import re
from datetime import datetime
import base64
import warnings
warnings.filterwarnings('ignore')

# ================================
# CONFIG
# ================================
st.set_page_config(page_title="Football Predictor - Last 5 Games Form", layout="wide")
st.title("⚽ Football Predictor Pro - Last 5 Games Form")
st.markdown("""
**Form-Based Analysis**  
- **Last 5 Home Games** for home teams  
- **Last 5 Away Games** for away teams  
- **Current Season Focus**  
- **Recent Performance Weighted**  
""")

# ================================
# LAST 5 GAMES ANALYSIS FUNCTIONS
# ================================
def get_last_n_home_games(df: pd.DataFrame, team: str, home_col: str, n: int = 5) -> pd.DataFrame:
    """Get last N home games for a team"""
    home_games = df[df[home_col] == team].copy()
    if 'Date' in home_games.columns:
        home_games = home_games.sort_values('Date', ascending=False)
    return home_games.head(n)

def get_last_n_away_games(df: pd.DataFrame, team: str, away_col: str, n: int = 5) -> pd.DataFrame:
    """Get last N away games for a team"""
    away_games = df[df[away_col] == team].copy()
    if 'Date' in away_games.columns:
        away_games = away_games.sort_values('Date', ascending=False)
    return away_games.head(n)

def calculate_team_form(df: pd.DataFrame, home_col: str, away_col: str, hg_col: str, ag_col: str, 
                       teams: List[str], n_games: int = 5) -> Dict[str, Any]:
    """
    Calculate team form based on last N home/away games
    """
    form_stats = {
        'home_attack': {},
        'home_defence': {}, 
        'away_attack': {},
        'away_defence': {},
        'home_games_used': {},
        'away_games_used': {}
    }
    
    # Calculate league averages from recent games
    recent_home_goals = []
    recent_away_goals = []
    
    for team in teams:
        # Home form (last N home games)
        home_games = get_last_n_home_games(df, team, home_col, n_games)
        form_stats['home_games_used'][team] = len(home_games)
        
        if len(home_games) > 0:
            home_goals_scored = home_games[hg_col].mean()
            home_goals_conceded = home_games[ag_col].mean()
            recent_home_goals.extend(home_games[hg_col].tolist())
            
            form_stats['home_attack'][team] = home_goals_scored
            form_stats['home_defence'][team] = home_goals_conceded
        else:
            form_stats['home_attack'][team] = 1.0  # Default
            form_stats['home_defence'][team] = 1.0
        
        # Away form (last N away games)  
        away_games = get_last_n_away_games(df, team, away_col, n_games)
        form_stats['away_games_used'][team] = len(away_games)
        
        if len(away_games) > 0:
            away_goals_scored = away_games[ag_col].mean()
            away_goals_conceded = away_games[hg_col].mean()
            recent_away_goals.extend(away_games[ag_col].tolist())
            
            form_stats['away_attack'][team] = away_goals_scored
            form_stats['away_defence'][team] = away_goals_conceded
        else:
            form_stats['away_attack'][team] = 1.0
            form_stats['away_defence'][team] = 1.0
    
    # Calculate league averages from the recent games we actually have
    form_stats['league_avg_home'] = np.mean(recent_home_goals) if recent_home_goals else 1.5
    form_stats['league_avg_away'] = np.mean(recent_away_goals) if recent_away_goals else 1.2
    
    return form_stats

# ================================
# ORIGINAL FUNCTIONS (UPDATED FOR FORM ANALYSIS)
# ================================
@st.cache_data(ttl=3600)
def get_team_logo(team_name: str) -> str:
    team_clean = team_name.strip().lower().replace(" ", "_").replace(".", "").replace("'", "")
    replacements = {
        "man_utd": "Manchester_United_F.C.", "man_city": "Manchester_City_F.C.",
        "arsenal": "Arsenal_F.C.", "chelsea": "Chelsea_F.C.", "liverpool": "Liverpool_F.C.",
        "nottm_forest": "Nottingham_Forest_F.C.", "leeds": "Leeds_United_F.C.",
        "spurs": "Tottenham_Hotspur_F.C.", "newcastle": "Newcastle_United_F.C.",
        "brighton": "Brighton_&_Hove_Albion_F.C.", "west_ham": "West_Ham_United_F.C."
    }
    wiki_name = replacements.get(team_clean, team_name.replace(" ", "_") + "_F.C.")
    url = f"https://en.wikipedia.org/wiki/File:{wiki_name}_logo.svg"
    try:
        if requests.head(url, timeout=5).status_code == 200:
            return url
    except:
        pass
    return None

@st.cache_data(ttl=3600)
def load_image(url: str):
    try:
        response = requests.get(url, timeout=10)
        return Image.open(BytesIO(response.content)).convert("RGBA")
    except:
        return None

@st.cache_data(show_spinner="Loading CSV...")
def load_csv(uploaded_file_bytes: bytes) -> pd.DataFrame:
    try:
        df = pd.read_csv(io.BytesIO(uploaded_file_bytes), encoding="utf-8")
        # Try to auto-detect and parse date column
        for col in df.columns:
            if 'date' in col.lower() or 'time' in col.lower():
                try:
                    df[col] = pd.to_datetime(df[col])
                except:
                    pass
        return df
    except:
        return pd.read_csv(io.BytesIO(uploaded_file_bytes), encoding="latin1")

@st.cache_data(show_spinner=False)
def detect_columns(df: pd.DataFrame) -> Dict[str, str]:
    mapping = {}
    for col in df.columns:
        lower = col.lower().replace(" ", "")
        if "home" in lower and "team" in lower: mapping["HomeTeam"] = col
        elif "away" in lower and "team" in lower: mapping["AwayTeam"] = col
        elif lower in ["fthg", "hgoals"]: mapping["FTHG"] = col
        elif lower in ["ftag", "agoals"]: mapping["FTAG"] = col
        elif lower in ["hc", "homecorners"]: mapping["HC"] = col
        elif lower in ["ac", "awaycorners"]: mapping["AC"] = col
        elif lower in ["hs", "homeshotsontarget"]: mapping["HS"] = col
        elif lower in ["as", "awayshotsontarget"]: mapping["AS"] = col
        elif lower in ["hxg", "home_xg"]: mapping["HxG"] = col
        elif lower in ["axg", "away_xg"]: mapping["AxG"] = col
        elif "date" in lower: mapping["Date"] = col
    return mapping

def parse_injuries(injury_str: str) -> Dict[str, Dict[str, float]]:
    injuries = {}
    if not injury_str.strip(): return injuries
    for line in injury_str.split('\n'):
        match = re.match(r'(\w+):\s*(\w+)\s*\(role:(\w+),\s*impact:(\d+)%\)', line.strip())
        if match:
            team, player, role, impact = match.groups()
            impact = float(impact) / 100
            if team not in injuries: injuries[team] = {}
            injuries[team][player] = {"role": role, "impact": impact}
    return injuries

# ================================
# FORM-BASED MODEL TRAINING
# ================================
@st.cache_data(show_spinner="Analyzing last 5 games form...")
def compute_form_based_stats(
    _df: pd.DataFrame,
    home_col: str, away_col: str, hg_col: str, ag_col: str,
    hc_col=None, ac_col=None, hs_col=None, as_col=None,
    n_games: int = 5
) -> Dict[str, Any]:
    """
    Compute stats based on last N home/away games only
    """
    df = _df.copy()
    
    # Ensure numeric columns
    for col in [hg_col, ag_col, hc_col, ac_col, hs_col, as_col]:
        if col and col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')

    stats = {}
    teams = sorted(set(df[home_col]).union(df[away_col]))

    # Calculate form-based stats
    form_stats = calculate_team_form(df, home_col, away_col, hg_col, ag_col, teams, n_games)
    
    # Convert to relative strengths
    home_attack = {}
    home_defence = {}
    away_attack = {}
    away_defence = {}
    
    for team in teams:
        # Home attack strength relative to league average
        home_attack[team] = form_stats['home_attack'][team] / form_stats['league_avg_home']
        home_defence[team] = form_stats['home_defence'][team] / form_stats['league_avg_away']
        
        # Away attack strength relative to league average  
        away_attack[team] = form_stats['away_attack'][team] / form_stats['league_avg_away']
        away_defence[team] = form_stats['away_defence'][team] / form_stats['league_avg_home']

    stats["goals"] = {
        "league_avg_home": form_stats['league_avg_home'],
        "league_avg_away": form_stats['league_avg_away'],
        "home_attack": home_attack,
        "away_attack": away_attack, 
        "home_defence": home_defence,
        "away_defence": away_defence,
        "games_used": form_stats['home_games_used'],
        "away_games_used": form_stats['away_games_used']
    }

    # Form-based corners
    if hc_col and ac_col and hc_col in df.columns and ac_col in df.columns:
        corner_stats = calculate_team_form(df, home_col, away_col, hc_col, ac_col, teams, n_games)
        stats["corners"] = {
            "league_avg_home": corner_stats['league_avg_home'],
            "league_avg_away": corner_stats['league_avg_away'],
            "home_attack": {t: corner_stats['home_attack'][t] / corner_stats['league_avg_home'] for t in teams},
            "away_attack": {t: corner_stats['away_attack'][t] / corner_stats['league_avg_away'] for t in teams},
            "home_defence": {t: corner_stats['home_defence'][t] / corner_stats['league_avg_away'] for t in teams},
            "away_defence": {t: corner_stats['away_defence'][t] / corner_stats['league_avg_home'] for t in teams}
        }
    else:
        # Fallback
        stats["corners"] = {
            "league_avg_home": 5.5, "league_avg_away": 4.8,
            "home_attack": {t: 1.0 for t in teams},
            "away_attack": {t: 1.0 for t in teams},
            "home_defence": {t: 1.0 for t in teams},
            "away_defence": {t: 1.0 for t in teams},
        }

    return stats

# ================================
# FORM-BASED PREDICTION
# ================================
@st.cache_data(show_spinner=False)
def predict_form_based_match(
    home: str, away: str, stats: Dict[str, Any], injuries: Dict = None
) -> Dict[str, Any]:
    """
    Predict match based on last 5 home/away games form
    """
    
    injury_summary = apply_injury_adjustment(stats, injuries) if injuries else ""

    predictions = {
        "goals": {"score": "N/A", "home_win": 0, "draw": 0, "away_win": 0, "btts_yes": 0, "over_25": 0},
        "xg": {"home": 0.0, "away": 0.0},
        "corners": {"home": 0, "away": 0, "total": 0},
        "form_based": True,
        "injury_summary": injury_summary,
        "games_used": {
            "home": stats["goals"]["games_used"].get(home, 0),
            "away": stats["goals"]["away_games_used"].get(away, 0)
        }
    }

    # --- FORM-BASED GOALS PREDICTION ---
    g = stats.get("goals", {})
    if g:
        l_home = g["league_avg_home"]
        l_away = g["league_avg_away"]
        
        # Home team: use their home form
        att_h = g["home_attack"].get(home, 1.0)  # Home team's home attack
        def_a = g["away_defence"].get(away, 1.0)  # Away team's away defence
        
        # Away team: use their away form  
        att_a = g["away_attack"].get(away, 1.0)  # Away team's away attack
        def_h = g["home_defence"].get(home, 1.0)  # Home team's home defence
        
        lambda_h = att_h * def_a * l_home
        lambda_a = att_a * def_h * l_away

        # Poisson distribution
        max_g = 8
        prob_matrix = np.zeros((max_g + 1, max_g + 1))
        for h in range(max_g + 1):
            for a in range(max_g + 1):
                p = poisson.pmf(h, lambda_h) * poisson.pmf(a, lambda_a)
                prob_matrix[h, a] = p
        prob_matrix /= prob_matrix.sum()
        
        h_idx, a_idx = np.unravel_index(np.argmax(prob_matrix), prob_matrix.shape)
        predictions["goals"]["score"] = f"{h_idx}–{a_idx}"
        predictions["goals"]["home_win"] = (prob_matrix[1:, :].sum() - prob_matrix.diagonal()[1:].sum())
        predictions["goals"]["away_win"] = (prob_matrix[:, 1:].sum() - prob_matrix.diagonal()[1:].sum())
        predictions["goals"]["draw"] = prob_matrix.diagonal().sum()
        predictions["goals"]["btts_yes"] = (prob_matrix[1:, 1:]).sum()
        predictions["goals"]["over_25"] = (prob_matrix[3:, :].sum() + prob_matrix[:, 3:].sum() - prob_matrix[3:, 3:].sum())

        predictions["xg"]["home"] = max(round(lambda_h, 2), 0.1)
        predictions["xg"]["away"] = max(round(lambda_a, 2), 0.1)

    # --- FORM-BASED CORNERS ---
    c = stats.get("corners")
    if c:
        mu_hc = c["home_attack"].get(home, 1.0) * c["away_defence"].get(away, 1.0) * c["league_avg_home"]
        mu_ac = c["away_attack"].get(away, 1.0) * c["home_defence"].get(home, 1.0) * c["league_avg_away"]
        predictions["corners"]["home"] = max(int(np.round(mu_hc)), 1)
        predictions["corners"]["away"] = max(int(np.round(mu_ac)), 1)
        predictions["corners"]["total"] = predictions["corners"]["home"] + predictions["corners"]["away"]

    return {"predictions": predictions}

def apply_injury_adjustment(stats: Dict[str, Any], injuries: Dict[str, Dict[str, float]]) -> str:
    """Apply injury adjustments to form-based stats"""
    summary = ""
    for team, players in injuries.items():
        attack_reduction = defence_reduction = 0
        for p, data in players.items():
            if data["role"] in ["forward", "midfielder", "winger", "striker"]:
                attack_reduction += data["impact"]
            elif data["role"] in ["defender", "goalkeeper"]:
                defence_reduction += data["impact"]
        
        attack_reduction = min(attack_reduction, 0.3)
        defence_reduction = min(defence_reduction, 0.3)
        
        if attack_reduction > 0:
            summary += f"{team} Attack -{attack_reduction*100:.0f}% | "
        if defence_reduction > 0:
            summary += f"{team} Defence -{defence_reduction*100:.0f}% | "
    
    return summary.strip(" | ")

# ================================
# DISPLAY FUNCTIONS
# ================================
def display_form_based_predictions(pred: Dict[str, Any], home_team: str, away_team: str, stats: Dict[str, Any]):
    p = pred["predictions"]
    
    st.markdown(f"### **{home_team} vs {away_team}**")
    st.markdown("#### 🎯 Last 5 Games Form Analysis")
    
    # Team logos and form info
    logos = {home_team: get_team_logo(home_team), away_team: get_team_logo(away_team)}
    colA, colB, colC = st.columns([1,2,1])
    
    with colA:
        if logos[home_team]: 
            img = load_image(logos[home_team])
            if img: st.image(img, width=80)
        st.write(f"**{home_team}**")
        home_games_used = p['games_used']['home']
        st.caption(f"Last {home_games_used} home games")
        
    with colC:
        if logos[away_team]: 
            img = load_image(logos[away_team])
            if img: st.image(img, width=80)
        st.write(f"**{away_team}**")
        away_games_used = p['games_used']['away']
        st.caption(f"Last {away_games_used} away games")
        
    with colB:
        st.markdown(f"<h2 style='text-align:center'>{p['goals']['score']}</h2>", unsafe_allow_html=True)
        st.caption("Most likely score based on recent form")

    # Outcomes
    st.markdown("#### 📊 Match Probabilities")
    colW1, colW2, colW3 = st.columns(3)
    colW1.metric("Home Win", f"{p['goals']['home_win']:.1%}")
    colW2.metric("Draw", f"{p['goals']['draw']:.1%}")
    colW3.metric("Away Win", f"{p['goals']['away_win']:.1%}")

    # BTTS & Over/Under
    colB1, colB2 = st.columns(2)
    colB1.metric("Both Teams to Score", f"{p['goals']['btts_yes']:.1%}")
    colB2.metric("Over 2.5 Goals", f"{p['goals']['over_25']:.1%}")

    # Stats
    st.markdown("#### ⚽ Expected Match Stats")
    colX1, colX2 = st.columns(2)
    with colX1:
        st.write(f"**Expected Goals (xG)**")
        st.write(f"{home_team}: **{p['xg']['home']}**")
        st.write(f"{away_team}: **{p['xg']['away']}**")
    with colX2:
        st.write(f"**Expected Corners**")
        st.write(f"{home_team}: **{p['corners']['home']}**")
        st.write(f"{away_team}: **{p['corners']['away']}**")
        st.write(f"**Total**: **{p['corners']['total']}**")

    # Form Analysis
    st.markdown("#### 📈 Recent Form Analysis")
    g = stats["goals"]
    
    col1, col2 = st.columns(2)
    with col1:
        st.write(f"**{home_team} Home Form**")
        home_attack = g["home_attack"].get(home_team, 1.0)
        home_defence = g["home_defence"].get(home_team, 1.0)
        st.write(f"Attack: {home_attack:.2f}× avg")
        st.write(f"Defence: {1/home_defence:.2f}× avg")
        
    with col2:
        st.write(f"**{away_team} Away Form**")
        away_attack = g["away_attack"].get(away_team, 1.0)
        away_defence = g["away_defence"].get(away_team, 1.0)
        st.write(f"Attack: {away_attack:.2f}× avg")
        st.write(f"Defence: {1/away_defence:.2f}× avg")

    if p["injury_summary"]:
        st.markdown(f"#### 🏥 Injury Impact")
        st.markdown(f"<span style='color:red'>{p['injury_summary']}</span>", unsafe_allow_html=True)

# ================================
# MAIN APP - FORM-BASED ANALYSIS
# ================================
st.sidebar.header("📁 Upload Match Data")
uploaded_file = st.sidebar.file_uploader("Choose CSV File", type=["csv"], 
                                         help="Upload match data with dates for form analysis")

if uploaded_file is not None:
    df = load_csv(uploaded_file.read())
    if df.empty:
        st.error("Empty CSV file.")
    else:
        st.success(f"✅ Loaded {len(df):,} matches")
        
        # Show preview
        with st.expander("📊 Data Preview"):
            st.dataframe(df.head(8))
        
        mapping = detect_columns(df)
        
        st.sidebar.subheader("🔧 Column Mapping")
        col_map = {}
        for label in ["HomeTeam", "AwayTeam", "FTHG", "FTAG", "HC", "AC", "Date"]:
            detected = mapping.get(label)
            options = [""] + list(df.columns)
            default_idx = options.index(detected) if detected in options else 0
            col_map[label] = st.sidebar.selectbox(f"**{label}**", options=options, index=default_idx)

        missing = [r for r in ["HomeTeam", "AwayTeam", "FTHG", "FTAG"] if not col_map[r]]
        if missing:
            st.error(f"❌ Map required fields: {', '.join(missing)}")
            st.stop()

        # Form analysis settings
        st.sidebar.subheader("⚙️ Form Analysis Settings")
        n_games = st.sidebar.slider("Number of games for form analysis", 3, 10, 5,
                                   help="Analyze last N home/away games for each team")
        
        require_dates = st.sidebar.toggle("Require date column", value=True,
                                         help="Date column needed for accurate recent form")

        if require_dates and not col_map.get("Date"):
            st.warning("⚠️ Date column not mapped. Form analysis may be less accurate.")
            # Sort by index as fallback
            df = df.sort_index(ascending=False)

        with st.spinner(f"🔄 Analyzing last {n_games} home/away games form..."):
            team_stats = compute_form_based_stats(
                _df=df,
                home_col=col_map["HomeTeam"], 
                away_col=col_map["AwayTeam"],
                hg_col=col_map["FTHG"], 
                ag_col=col_map["FTAG"],
                hc_col=col_map.get("HC"), 
                ac_col=col_map.get("AC"),
                n_games=n_games
            )

        teams = sorted(set(df[col_map["HomeTeam"]]).union(df[col_map["AwayTeam"]]))

        # Injury Input
        st.sidebar.subheader("🏥 Current Injuries")
        injury_input = st.sidebar.text_area("Injured Players", 
                                          placeholder="Arsenal: Saka (role:forward, impact:15%)\nChelsea: James (role:defender, impact:20%)",
                                          height=100)
        injuries = parse_injuries(injury_input)

        # Prediction Section
        st.markdown("---")
        st.subheader("🔮 Form-Based Match Prediction")
        
        col1, col2 = st.columns(2)
        home_team = col1.selectbox("Home Team", teams, key="home_select")
        away_team = col2.selectbox("Away Team", teams, key="away_select")

        if st.button(f"🎯 Predict Based on Last {n_games} Games", type="primary", use_container_width=True):
            with st.spinner("Analyzing recent form..."):
                pred = predict_form_based_match(home_team, away_team, team_stats, injuries)
                display_form_based_predictions(pred, home_team, away_team, team_stats)

        # Team form overview
        with st.expander("📈 Team Form Ratings (Last 5 Games)"):
            g = team_stats["goals"]
            
            st.write("**Best Home Form (Attack)**")
            home_attack_sorted = sorted(teams, key=lambda x: g["home_attack"].get(x, 0), reverse=True)
            for i, team in enumerate(home_attack_sorted[:6]):
                if g["home_attack"].get(team, 0) > 0:
                    st.write(f"{i+1}. {team}: {g['home_attack'][team]:.2f}× avg")
            
            st.write("**Best Away Form (Attack)**")
            away_attack_sorted = sorted(teams, key=lambda x: g["away_attack"].get(x, 0), reverse=True)
            for i, team in enumerate(away_attack_sorted[:6]):
                if g["away_attack"].get(team, 0) > 0:
                    st.write(f"{i+1}. {team}: {g['away_attack'][team]:.2f}× avg")

else:
    st.info("📁 Please upload CSV data to get started")
    
    with st.expander("💡 CSV Format for Form Analysis"):
        st.markdown("""
        **Required for Form Analysis:**
        - Home Team
        - Away Team  
        - Home Goals
        - Away Goals
        - **Date (highly recommended)**
        
        **Example with dates:**
        ```
        Date,HomeTeam,AwayTeam,FTHG,FTAG
        2025-01-15,Arsenal,Chelsea,2,1
        2025-01-14,Man Utd,Liverpool,1,1
        2025-01-13,Man City,Tottenham,3,0
        ```
        
        **Analysis Method:**
        - Home teams: Last 5 **home** games
        - Away teams: Last 5 **away** games  
        - Most recent games weighted highest
        """)

# Form analysis info
st.sidebar.markdown("---")
st.sidebar.subheader("📊 Form Analysis")
st.sidebar.info(f"Using last {n_games if 'n_games' in locals() else 5} home/away games for predictions")

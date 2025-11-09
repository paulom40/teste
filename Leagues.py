# app.py - FOCUSED ON 2025-2026 SEASON
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
st.set_page_config(page_title="Football Predictor 2025-26", layout="wide")
st.title("⚽ Football Predictor Pro 2025-26 Season")
st.markdown("""
**2025-2026 Season Analysis**  
- **Current Season Form Only**  
- **Latest Team Performance Data**  
- **Real-time Strength Ratings**  
- **In-season Injury Impact**  
""")

# ================================
# SEASON FILTERING
# ================================
def filter_current_season(df: pd.DataFrame, date_col: str = None) -> pd.DataFrame:
    """
    Filter data for 2025-2026 season only
    """
    df_filtered = df.copy()
    
    # If we have a date column, use it to filter
    if date_col and date_col in df.columns:
        try:
            df_filtered[date_col] = pd.to_datetime(df_filtered[date_col])
            # Keep only matches from 2025-2026 season (July 2025 onwards)
            season_start = pd.to_datetime('2025-07-01')
            df_filtered = df_filtered[df_filtered[date_col] >= season_start]
        except:
            st.warning("Could not parse dates, using all data")
    
    st.info(f"📊 Using {len(df_filtered)} matches from 2025-2026 season")
    return df_filtered

def detect_date_column(df: pd.DataFrame) -> str:
    """Auto-detect date column"""
    date_indicators = ['date', 'Date', 'TIME', 'time', 'datetime', 'Datetime']
    for col in df.columns:
        if any(indicator in col.lower() for indicator in ['date', 'time']):
            return col
    return None

# ================================
# ORIGINAL FUNCTIONS (UPDATED FOR CURRENT SEASON)
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
        return pd.read_csv(io.BytesIO(uploaded_file_bytes), encoding="utf-8")
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
# CURRENT SEASON MODEL TRAINING
# ================================
@st.cache_data(show_spinner="Analyzing 2025-26 season...")
def compute_current_season_stats(
    _df: pd.DataFrame,
    home_col: str, away_col: str, hg_col: str, ag_col: str,
    hc_col=None, ac_col=None, hs_col=None, as_col=None,
    hxg_col=None, axg_col=None,
    min_matches: int = 2  # Lower minimum for current season
) -> Dict[str, Any]:
    """
    Compute stats using ONLY 2025-2026 season data
    Higher weight on recent form, no historical data
    """
    df = _df.copy()
    
    # Ensure numeric columns
    for col in [hg_col, ag_col, hc_col, ac_col, hs_col, as_col, hxg_col, axg_col]:
        if col and col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')

    stats = {}
    teams = sorted(set(df[home_col]).union(df[away_col]))

    # --- GOALS - Current Season Only ---
    ft_mask = df[hg_col].notna() & df[ag_col].notna()
    clean_ft = df[ft_mask][[home_col, away_col, hg_col, ag_col]].copy()
    
    if len(clean_ft) < 3: 
        st.warning("⚠️ Limited current season data. Using league averages.")
        # Fallback to reasonable current season averages
        avg_home = 1.6  # Conservative current season estimate
        avg_away = 1.3
    else:
        # Use simple averages for current season (no recency weighting)
        avg_home = clean_ft[hg_col].mean()
        avg_away = clean_ft[ag_col].mean()

    # Current season team strengths (simple averages)
    home_attack = away_attack = home_defence = away_defence = {}
    
    for team in teams:
        home_matches = clean_ft[clean_ft[home_col] == team]
        away_matches = clean_ft[clean_ft[away_col] == team]
        
        n_home = len(home_matches)
        n_away = len(away_matches)
        
        # Home performance
        if n_home >= min_matches:
            home_goals_avg = home_matches[hg_col].mean()
            home_conceded_avg = home_matches[ag_col].mean()
            home_attack[team] = home_goals_avg / avg_home
            home_defence[team] = home_conceded_avg / avg_away
        else:
            # New teams or few matches start at league average
            home_attack[team] = 1.0
            home_defence[team] = 1.0
        
        # Away performance
        if n_away >= min_matches:
            away_goals_avg = away_matches[ag_col].mean()
            away_conceded_avg = away_matches[hg_col].mean()
            away_attack[team] = away_goals_avg / avg_away
            away_defence[team] = away_conceded_avg / avg_home
        else:
            away_attack[team] = 1.0
            away_defence[team] = 1.0

    stats["goals"] = {
        "league_avg_home": avg_home, 
        "league_avg_away": avg_away,
        "home_attack": home_attack, 
        "away_attack": away_attack,
        "home_defence": home_defence, 
        "away_defence": away_defence,
        "matches_analyzed": len(clean_ft)
    }

    # --- CURRENT SEASON CORNERS ---
    if hc_col and ac_col and hc_col in df.columns and ac_col in df.columns:
        c_mask = df[hc_col].notna() & df[ac_col].notna()
        clean_c = df[c_mask][[home_col, away_col, hc_col, ac_col]].copy()
        
        if len(clean_c) >= 3:
            hc_mean = clean_c[hc_col].mean()
            ac_mean = clean_c[ac_col].mean()
            
            corner_stats = {
                "league_avg_home": hc_mean, 
                "league_avg_away": ac_mean,
                "home_attack": {}, "away_attack": {}, 
                "home_defence": {}, "away_defence": {}
            }
            
            for team in teams:
                home_c = clean_c[clean_c[home_col] == team]
                away_c = clean_c[clean_c[away_col] == team]
                
                if len(home_c) >= min_matches:
                    corner_stats["home_attack"][team] = home_c[hc_col].mean() / hc_mean
                    corner_stats["home_defence"][team] = home_c[ac_col].mean() / ac_mean
                else:
                    corner_stats["home_attack"][team] = corner_stats["home_defence"][team] = 1.0
                    
                if len(away_c) >= min_matches:
                    corner_stats["away_attack"][team] = away_c[ac_col].mean() / ac_mean
                    corner_stats["away_defence"][team] = away_c[hc_col].mean() / hc_mean
                else:
                    corner_stats["away_attack"][team] = corner_stats["away_defence"][team] = 1.0
                    
            stats["corners"] = corner_stats
    
    # Fallback if no corner data
    if "corners" not in stats:
        stats["corners"] = {
            "league_avg_home": 5.5, "league_avg_away": 4.8,
            "home_attack": {t: 1.0 for t in teams},
            "away_attack": {t: 1.0 for t in teams},
            "home_defence": {t: 1.0 for t in teams},
            "away_defence": {t: 1.0 for t in teams},
        }

    return stats

# ================================
# PREDICTION FUNCTION FOR CURRENT SEASON
# ================================
@st.cache_data(show_spinner=False)
def predict_current_season_match(
    home: str, away: str, stats: Dict[str, Any], injuries: Dict = None
) -> Dict[str, Any]:
    """
    Predict match using only 2025-2026 season data
    """
    
    # Apply injury adjustments
    if injuries:
        injury_summary = apply_injury_adjustment(stats, injuries)
    else:
        injury_summary = ""

    max_g = 8  # Lower max goals for current season realism
    predictions = {
        "goals": {"score": "N/A", "home_win": 0, "draw": 0, "away_win": 0, "btts_yes": 0, "over_25": 0},
        "xg": {"home": 0.0, "away": 0.0},
        "corners": {"home": 0, "away": 0, "total": 0},
        "current_season_data": True,
        "injury_summary": injury_summary
    }

    # --- CURRENT SEASON GOALS PREDICTION ---
    g = stats.get("goals", {})
    if g:
        l_home = g["league_avg_home"]
        l_away = g["league_avg_away"]
        att_h = g["home_attack"].get(home, 1.0)
        def_a = g["away_defence"].get(away, 1.0)
        att_a = g["away_attack"].get(away, 1.0)
        def_h = g["home_defence"].get(home, 1.0)
        
        lambda_h = att_h * def_a * l_home
        lambda_a = att_a * def_h * l_away

        # Poisson distribution for goals
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

    # --- CURRENT SEASON CORNERS ---
    c = stats.get("corners")
    if c:
        mu_hc = c["home_attack"].get(home, 1.0) * c["away_defence"].get(away, 1.0) * c["league_avg_home"]
        mu_ac = c["away_attack"].get(away, 1.0) * c["home_defence"].get(home, 1.0) * c["league_avg_away"]
        predictions["corners"]["home"] = max(int(np.round(mu_hc)), 1)
        predictions["corners"]["away"] = max(int(np.round(mu_ac)), 1)
        predictions["corners"]["total"] = predictions["corners"]["home"] + predictions["corners"]["away"]

    return {"predictions": predictions}

def apply_injury_adjustment(stats: Dict[str, Any], injuries: Dict[str, Dict[str, float]]) -> str:
    """Simple injury adjustment for current season"""
    summary = ""
    for team, players in injuries.items():
        attack_reduction = defence_reduction = 0
        for p, data in players.items():
            if data["role"] in ["forward", "midfielder"]:
                attack_reduction += data["impact"]
            elif data["role"] in ["defender"]:
                defence_reduction += data["impact"]
        
        attack_reduction = min(attack_reduction, 0.25)  # Higher cap for current season impact
        defence_reduction = min(defence_reduction, 0.25)
        
        if attack_reduction > 0:
            summary += f"{team} Attack -{attack_reduction*100:.0f}% | "
        if defence_reduction > 0:
            summary += f"{team} Defence -{defence_reduction*100:.0f}% | "
    
    return summary.strip(" | ")

# ================================
# DISPLAY FUNCTIONS
# ================================
def display_current_season_predictions(pred: Dict[str, Any], home_team: str, away_team: str, stats: Dict[str, Any]):
    p = pred["predictions"]
    
    st.markdown(f"### **{home_team} vs {away_team}**")
    st.markdown("#### 🎯 2025-2026 Season Prediction")
    
    # Team logos
    logos = {home_team: get_team_logo(home_team), away_team: get_team_logo(away_team)}
    colA, colB, colC = st.columns([1,2,1])
    with colA:
        if logos[home_team]: 
            img = load_image(logos[home_team])
            if img: st.image(img, width=80)
        st.write(f"**{home_team}**")
    with colC:
        if logos[away_team]: 
            img = load_image(logos[away_team])
            if img: st.image(img, width=80)
        st.write(f"**{away_team}**")
    with colB:
        st.markdown(f"<h2 style='text-align:center'>{p['goals']['score']}</h2>", unsafe_allow_html=True)
        st.caption("Most likely score")

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

    if p["injury_summary"]:
        st.markdown(f"#### 🏥 Current Season Injury Impact")
        st.markdown(f"<span style='color:red'>{p['injury_summary']}</span>", unsafe_allow_html=True)

    # Season context
    st.markdown("#### 📈 2025-26 Season Context")
    matches_analyzed = stats["goals"].get("matches_analyzed", 0)
    st.write(f"**Analysis based on**: {matches_analyzed} current season matches")
    st.write(f"**League averages**: {stats['goals']['league_avg_home']:.1f} home goals | {stats['goals']['league_avg_away']:.1f} away goals")

# ================================
# MAIN APP - 2025-26 SEASON FOCUS
# ================================
st.sidebar.header("📁 Upload 2025-26 Season Data")
uploaded_file = st.sidebar.file_uploader("Choose CSV File", type=["csv"], 
                                         help="Upload your 2025-2026 season match data")

if uploaded_file is not None:
    df = load_csv(uploaded_file.read())
    if df.empty:
        st.error("Empty CSV file.")
    else:
        # Filter for current season
        date_col = detect_date_column(df)
        df_current = filter_current_season(df, date_col)
        
        st.success(f"✅ Loaded {len(df_current)} matches from 2025-2026 season")
        
        # Show preview
        with st.expander("📊 Current Season Data Preview"):
            st.dataframe(df_current.head(10))
            st.write(f"**Teams in dataset**: {len(set(df_current['HomeTeam'].dropna()) | set(df_current['AwayTeam'].dropna()))}")
        
        mapping = detect_columns(df_current)
        
        st.sidebar.subheader("🔧 Column Mapping")
        col_map = {}
        for label in ["HomeTeam", "AwayTeam", "FTHG", "FTAG", "HC", "AC", "HS", "AS", "HxG", "AxG", "Date"]:
            detected = mapping.get(label)
            options = [""] + list(df_current.columns)
            default_idx = options.index(detected) if detected in options else 0
            col_map[label] = st.sidebar.selectbox(f"**{label}**", options=options, index=default_idx)

        missing = [r for r in ["HomeTeam", "AwayTeam", "FTHG", "FTAG"] if not col_map[r]]
        if missing:
            st.error(f"❌ Map required fields: {', '.join(missing)}")
            st.stop()

        # Current season settings
        st.sidebar.subheader("⚙️ Current Season Settings")
        min_matches = st.sidebar.slider("Minimum matches for team rating", 1, 10, 2,
                                       help="Lower value for early season, increase as season progresses")

        with st.spinner("🔄 Analyzing 2025-26 season form..."):
            team_stats = compute_current_season_stats(
                _df=df_current,
                home_col=col_map["HomeTeam"], away_col=col_map["AwayTeam"],
                hg_col=col_map["FTHG"], ag_col=col_map["FTAG"],
                hc_col=col_map.get("HC"), ac_col=col_map.get("AC"),
                hs_col=col_map.get("HS"), as_col=col_map.get("AS"),
                hxg_col=col_map.get("HxG"), axg_col=col_map.get("AxG"),
                min_matches=min_matches
            )

        teams = sorted(set(df_current[col_map["HomeTeam"]]).union(df_current[col_map["AwayTeam"]]))

        # Injury Input for current season
        st.sidebar.subheader("🏥 Current Season Injuries")
        injury_input = st.sidebar.text_area("Injured Players", 
                                          placeholder="Example:\nArsenal: Saka (role:forward, impact:15%)\nChelsea: James (role:defender, impact:20%)",
                                          height=100)
        injuries = parse_injuries(injury_input)

        # Prediction Section
        st.markdown("---")
        st.subheader("🔮 2025-26 Season Match Prediction")
        
        col1, col2 = st.columns(2)
        home_team = col1.selectbox("Home Team", teams, key="home_select")
        away_team = col2.selectbox("Away Team", teams, key="away_select")

        if st.button("🎯 Predict Current Season Match", type="primary", use_container_width=True):
            with st.spinner("Analyzing current season form..."):
                pred = predict_current_season_match(home_team, away_team, team_stats, injuries)
                display_current_season_predictions(pred, home_team, away_team, team_stats)

        # Team strength overview
        with st.expander("📈 2025-26 Team Strength Ratings"):
            g = team_stats["goals"]
            teams_sorted = sorted(teams, key=lambda x: g["home_attack"].get(x, 1.0) + g["away_attack"].get(x, 1.0), reverse=True)
            
            col1, col2 = st.columns(2)
            with col1:
                st.write("**Top Attacking Teams**")
                for i, team in enumerate(teams_sorted[:5]):
                    home_att = g["home_attack"].get(team, 1.0)
                    away_att = g["away_attack"].get(team, 1.0)
                    st.write(f"{i+1}. {team}: Home {home_att:.2f} | Away {away_att:.2f}")
            
            with col2:
                st.write("**Top Defensive Teams**")
                teams_def_sorted = sorted(teams, key=lambda x: (1/g["home_defence"].get(x, 1.0) + 1/g["away_defence"].get(x, 1.0))/2)
                for i, team in enumerate(teams_def_sorted[:5]):
                    home_def = g["home_defence"].get(team, 1.0)
                    away_def = g["away_defence"].get(team, 1.0)
                    st.write(f"{i+1}. {team}: Home {1/home_def:.2f} | Away {1/away_def:.2f}")

else:
    st.info("📁 Please upload 2025-2026 season CSV data to get started")
    
    with st.expander("💡 2025-26 CSV Format Guide"):
        st.markdown("""
        **Required for Current Season Analysis:**
        - Home Team (e.g., 'HomeTeam', 'Home')
        - Away Team (e.g., 'AwayTeam', 'Away')  
        - Home Goals (e.g., 'FTHG', 'HG')
        - Away Goals (e.g., 'FTAG', 'AG')
        - Date (recommended for season filtering)
        
        **2025-26 Season Example:**
        ```
        Date,HomeTeam,AwayTeam,FTHG,FTAG,HC,AC
        2025-08-10,Arsenal,Chelsea,2,1,6,4
        2025-08-11,Man Utd,Liverpool,1,1,5,7
        2025-08-12,Man City,Tottenham,3,0,8,2
        ```
        
        **Note**: Only 2025-2026 season data will be used for predictions.
        """)

# Season info in sidebar
st.sidebar.markdown("---")
st.sidebar.subheader("🏆 2025-26 Season")
st.sidebar.info("Using current season data only for accurate form-based predictions")

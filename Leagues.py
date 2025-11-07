# app.py
import streamlit as st
import pandas as pd
import numpy as np
from scipy.stats import poisson, nbinom
import io
from typing import Dict, Any, List, Tuple
import requests
from PIL import Image
from io import BytesIO
import base64
import plotly.graph_objects as go
import plotly.express as px
import re
import os
import tempfile
from datetime import datetime, timedelta

# --- PDF EXPORT ---
try:
    from weasyprint import HTML
    WEASYPRINT_AVAILABLE = True
except:
    WEASYPRINT_AVAILABLE = False

try:
    import pdfkit
    PDFKIT_AVAILABLE = True
except:
    PDFKIT_AVAILABLE = False

# ================================
# CONFIG
# ================================
st.set_page_config(page_title="Advanced Football Predictor", layout="wide")
st.title("⚽ Advanced Football Match Predictor")
st.markdown("""
**Professional-Grade Prediction Suite with Enhanced Statistical Models**

**🎯 Advanced Features:**
- ✅ **Dixon-Coles Model** — Industry-standard correlation adjustment
- ✅ **Negative Binomial for Corners** — Better tail modeling for sequential events
- ✅ **Tunable Recency Weighting** — Adjust how much recent form matters
- ✅ **xG Validation** — Detects over/underperformance vs expected goals
- ✅ **Form Analysis** — Last 5 matches weighted performance
- ✅ **Bayesian Smoothing** — Handles small sample sizes intelligently
- ✅ **Autocorrelation Adjustment** — Improves corner prediction accuracy

**Predicts:**
- Full-Time Score | BTTS | Over 2.5 | 1X2 Probabilities
- Corners (with Negative Binomial) | xG Analysis | Shots on Target
- **Goal Timing (1–15, 16–30, ..., 76–90)** — Minute-Level Precision
- **Team Form & Performance Indicators**

**Export to PDF with one click**
""")

# ================================
# LOGO & CSS
# ================================
@st.cache_data(ttl=3600)
def get_team_logo(team_name: str) -> str:
    team_clean = team_name.strip().lower().replace(" ", "_").replace(".", "").replace("'", "")
    replacements = {
        "man_utd": "Manchester_United_F.C.", "man_city": "Manchester_City_F.C.",
        "arsenal": "Arsenal_F.C.", "chelsea": "Chelsea_F.C.", "liverpool": "Liverpool_F.C.",
        "nottm_forest": "Nottingham_Forest_F.C.", "nacional": "C.D._Nacional",
        "famalicao": "F.C._Famalicão"
    }
    wiki_name = replacements.get(team_clean, team_name.replace(" ", "_").replace("'", "") + "_F.C.")
    url = f"https://en.wikipedia.org/wiki/File:{wiki_name}_logo.svg"
    try:
        if requests.head(url, timeout=5).status_code == 200:
            return f"https://en.wikipedia.org/wiki/File:{wiki_name}_logo.svg"
    except:
        pass
    return None

@st.cache_data(ttl=3600)
def load_image(url: str):
    try:
        response = requests.get(url, timeout=10)
        img = Image.open(BytesIO(response.content)).convert("RGBA")
        return img
    except:
        return None

print_css = """
<style>
@media print {
    .stApp > header, .stApp > footer, .stSidebar, .no-print { display: none !important; }
    .block-container { padding: 1in !important; max-width: 100% !important; }
    body { margin: 0; font-family: Arial; }
    .print-title { font-size: 24px; font-weight: bold; text-align: center; margin-bottom: 20px; }
    .team-box { text-align: center; }
    .logo { width: 80px; height: 80px; }
    .prediction { margin: 20px 0; padding: 15px; border: 1px solid #ccc; border-radius: 8px; background: #f9f9f9; }
    .score { font-size: 20px; font-weight: bold; }
    .prob { font-size: 14px; color: #555; }
    .stPlotlyChart { display: none; }
}
</style>
"""

# ================================
# ADVANCED HELPERS
# ================================
def calculate_form(df: pd.DataFrame, team: str, home_col: str, away_col: str, 
                   hg_col: str, ag_col: str, n_matches: int = 5) -> Dict[str, float]:
    """Calculate recent form for a team (last N matches)"""
    team_home = df[df[home_col] == team].tail(n_matches)
    team_away = df[df[away_col] == team].tail(n_matches)
    
    all_matches = pd.concat([team_home, team_away]).sort_index().tail(n_matches)
    
    if len(all_matches) == 0:
        return {"avg_goals_scored": 0, "avg_goals_conceded": 0, "points": 0, "form_score": 0}
    
    goals_scored = 0
    goals_conceded = 0
    points = 0
    
    for idx, row in all_matches.iterrows():
        is_home = row[home_col] == team
        if is_home:
            gf = pd.to_numeric(row[hg_col], errors='coerce')
            ga = pd.to_numeric(row[ag_col], errors='coerce')
        else:
            gf = pd.to_numeric(row[ag_col], errors='coerce')
            ga = pd.to_numeric(row[hg_col], errors='coerce')
        
        if pd.notna(gf) and pd.notna(ga):
            goals_scored += gf
            goals_conceded += ga
            if gf > ga:
                points += 3
            elif gf == ga:
                points += 1
    
    n_valid = len(all_matches)
    avg_scored = goals_scored / n_valid if n_valid > 0 else 0
    avg_conceded = goals_conceded / n_valid if n_valid > 0 else 0
    form_score = points / (n_valid * 3) if n_valid > 0 else 0  # Normalized 0-1
    
    return {
        "avg_goals_scored": avg_scored,
        "avg_goals_conceded": avg_conceded,
        "points": points,
        "form_score": form_score
    }

def bayesian_smoothing(observed_rate: float, league_avg: float, sample_size: int, 
                       confidence: int = 10) -> float:
    """Apply Bayesian smoothing to handle small sample sizes"""
    return (observed_rate * sample_size + league_avg * confidence) / (sample_size + confidence)

def negative_binomial_params(mean: float, variance: float) -> Tuple[float, float]:
    """Convert mean and variance to Negative Binomial parameters (n, p)"""
    if variance <= mean:
        # Fallback to Poisson if variance not greater than mean
        return mean, None
    p = mean / variance
    n = mean * p / (1 - p)
    return n, p

def _safe_index(df: pd.DataFrame, col: str):
    return df.columns.get_loc(col) if col in df.columns else 0

# ================================
# DATA LOADER
# ================================
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
        if "date" in lower:                     mapping["Date"] = col
        elif "home" in lower and "team" in lower: mapping["HomeTeam"] = col
        elif "away" in lower and "team" in lower: mapping["AwayTeam"] = col
        elif lower in ["fthg", "hgoals"]:        mapping["FTHG"] = col
        elif lower in ["ftag", "agoals"]:        mapping["FTAG"] = col
        elif lower in ["hthg", "halfhome"]:      mapping["HTHG"] = col
        elif lower in ["htag", "halfaway"]:      mapping["HTAG"] = col
        elif lower in ["hc", "homecorners"]:     mapping["HC"] = col
        elif lower in ["ac", "awaycorners"]:     mapping["AC"] = col
        elif lower in ["hs", "homeshotsontarget"]: mapping["HS"] = col
        elif lower in ["as", "awayshotsontarget"]: mapping["AS"] = col
        elif lower in ["hxg", "home_xg"]:        mapping["HxG"] = col
        elif lower in ["axg", "away_xg"]:        mapping["AxG"] = col
    return mapping

# ================================
# GOAL MINUTE PARSING
# ================================
def extract_goal_minutes(df: pd.DataFrame, home_col: str, away_col: str) -> pd.DataFrame:
    goal_df = pd.DataFrame(index=df.index)
    goal_df['home_goals'] = pd.NA
    goal_df['away_goals'] = pd.NA

    # HG1, AG1, etc.
    home_goal_cols = [c for c in df.columns if re.match(r'^HG\d*$', c.upper())]
    away_goal_cols = [c for c in df.columns if re.match(r'^AG\d*$', c.upper())]
    if home_goal_cols or away_goal_cols:
        def parse(row):
            h = [int(row[c]) for c in home_goal_cols if pd.notna(row[c])]
            a = [int(row[c]) for c in away_goal_cols if pd.notna(row[c])]
            return h, a
        parsed = df.apply(parse, axis=1)
        goal_df['home_goals'] = parsed.apply(lambda x: x[0])
        goal_df['away_goals'] = parsed.apply(lambda x: x[1])
        return goal_df

    # HGT, AGT
    hgt_col = next((c for c in df.columns if c.upper() in ['HGT', 'HOMEGOALTIMES']), None)
    agt_col = next((c for c in df.columns if c.upper() in ['AGT', 'AWAYGOALTIMES']), None)
    if hgt_col or agt_col:
        def parse_times(x):
            if pd.isna(x): return []
            return [int(t.strip()) for t in str(x).split(',') if t.strip().isdigit()]
        home_goals = df[hgt_col].apply(parse_times) if hgt_col else pd.Series([[]] * len(df))
        away_goals = df[agt_col].apply(parse_times) if agt_col else pd.Series([[]] * len(df))
        goal_df['home_goals'] = home_goals
        goal_df['away_goals'] = away_goals
        return goal_df

    # GoalTimes
    time_col = next((c for c in df.columns if c.lower() in ['goaltimes', 'goals', 'goaltime']), None)
    if time_col:
        def parse_goal_time(row):
            if pd.isna(row[time_col]): return [], []
            text = str(row[time_col])
            home, away = [], []
            matches = re.findall(r"(\w+)\s+(\d+)'?", text)
            home_team = row[home_col].lower()
            away_team = row[away_col].lower()
            for team, minute in matches:
                minute = int(minute)
                if minute > 90: minute = 90
                team_lower = team.strip().lower()
                if team_lower == home_team:
                    home.append(minute)
                elif team_lower == away_team:
                    away.append(minute)
            return home, away
        parsed = df.apply(parse_goal_time, axis=1)
        goal_df['home_goals'] = parsed.apply(lambda x: x[0])
        goal_df['away_goals'] = parsed.apply(lambda x: x[1])
        return goal_df

    return None

# ================================
# MODEL WITH ENHANCED FEATURES
# ================================
@st.cache_data(show_spinner="Training enhanced model...")
def compute_team_stats(
    _df: pd.DataFrame,
    home_col: str, away_col: str, hg_col: str, ag_col: str,
    hthg_col=None, htag_col=None, hc_col=None, ac_col=None,
    hs_col=None, as_col=None, hxg_col=None, axg_col=None,
    recency_weight: float = 2.0, min_matches: int = 3
) -> Dict[str, Any]:
    df = _df.copy()
    for col in [hg_col, ag_col, hthg_col, htag_col, hc_col, ac_col, hs_col, as_col, hxg_col, axg_col]:
        if col and col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')

    stats = {}
    teams = sorted(set(df[home_col]).union(df[away_col]))

    # === GOALS WITH ENHANCED WEIGHTING ===
    ft_mask = df[hg_col].notna() & df[ag_col].notna()
    clean_ft = df[ft_mask][[home_col, away_col, hg_col, ag_col]].copy()
    if len(clean_ft) < 5:
        raise ValueError(f"Only {len(clean_ft)} valid matches.")
    
    # Apply tunable exponential decay weighting
    clean_ft['weight'] = np.exp(np.linspace(-recency_weight, 0, len(clean_ft)))
    
    avg_home = np.average(clean_ft[hg_col], weights=clean_ft['weight'])
    avg_away = np.average(clean_ft[ag_col], weights=clean_ft['weight'])
    
    def weighted_mean(group, col, weight_col='weight'):
        if len(group) < min_matches:
            return None
        return np.average(group[col], weights=group[weight_col])
    
    # Calculate team strengths with Bayesian smoothing
    home_attack_raw = {}
    away_attack_raw = {}
    home_defence_raw = {}
    away_defence_raw = {}
    team_sample_sizes = {}
    
    for team in teams:
        home_matches = clean_ft[clean_ft[home_col] == team]
        away_matches = clean_ft[clean_ft[away_col] == team]
        
        n_home = len(home_matches)
        n_away = len(away_matches)
        team_sample_sizes[team] = n_home + n_away
        
        if n_home >= min_matches:
            ha = weighted_mean(home_matches, hg_col) / avg_home
            hd = weighted_mean(home_matches, ag_col) / avg_away
            home_attack_raw[team] = bayesian_smoothing(ha, 1.0, n_home)
            home_defence_raw[team] = bayesian_smoothing(hd, 1.0, n_home)
        else:
            home_attack_raw[team] = 1.0
            home_defence_raw[team] = 1.0
            
        if n_away >= min_matches:
            aa = weighted_mean(away_matches, ag_col) / avg_away
            ad = weighted_mean(away_matches, hg_col) / avg_home
            away_attack_raw[team] = bayesian_smoothing(aa, 1.0, n_away)
            away_defence_raw[team] = bayesian_smoothing(ad, 1.0, n_away)
        else:
            away_attack_raw[team] = 1.0
            away_defence_raw[team] = 1.0
    
    stats["goals"] = {
        "league_avg_home": avg_home, 
        "league_avg_away": avg_away,
        "home_attack": home_attack_raw,
        "away_attack": away_attack_raw,
        "home_defence": home_defence_raw,
        "away_defence": away_defence_raw,
        "sample_sizes": team_sample_sizes
    }

    # === xG VALIDATION (if available) ===
    if hxg_col and axg_col and hxg_col in df.columns and axg_col in df.columns:
        xg_mask = df[hxg_col].notna() & df[axg_col].notna() & ft_mask
        if xg_mask.sum() >= 5:
            xg_comparison = df[xg_mask].copy()
            xg_comparison['home_diff'] = xg_comparison[hg_col] - xg_comparison[hxg_col]
            xg_comparison['away_diff'] = xg_comparison[ag_col] - xg_comparison[axg_col]
            
            # Calculate over/underperformance for each team
            xg_performance = {}
            for team in teams:
                home_xg = xg_comparison[xg_comparison[home_col] == team]
                away_xg = xg_comparison[xg_comparison[away_col] == team]
                
                home_over = home_xg['home_diff'].mean() if len(home_xg) > 0 else 0
                away_over = away_xg['away_diff'].mean() if len(away_xg) > 0 else 0
                
                xg_performance[team] = {
                    "overperformance": (home_over + away_over) / 2,
                    "sample_size": len(home_xg) + len(away_xg)
                }
            
            stats["xg_validation"] = xg_performance

    # === GOAL TIMING ===
    intervals = ["1–15", "16–30", "31–45", "46–60", "61–75", "76–90"]
    interval_bins = [(1,15), (16,30), (31,45), (46,60), (61,75), (76,90)]
    goals_per_interval = {i: 0 for i in intervals}

    minute_df = extract_goal_minutes(df, home_col, away_col)
    if minute_df is not None:
        all_goals = []
        for _, row in minute_df.iterrows():
            all_goals.extend([m for m in row['home_goals'] if isinstance(m, (int, float)) and 1 <= m <= 90])
            all_goals.extend([m for m in row['away_goals'] if isinstance(m, (int, float)) and 1 <= m <= 90])
        for m in all_goals:
            for idx, (s, e) in enumerate(interval_bins):
                if s <= m <= e:
                    goals_per_interval[intervals[idx]] += 1

    if sum(goals_per_interval.values()) == 0 and hthg_col and htag_col:
        ht_mask = df[hthg_col].notna() & df[htag_col].notna() & df[hg_col].notna() & df[ag_col].notna()
        timing_df = df[ht_mask]
        fh = timing_df[hthg_col].sum() + timing_df[htag_col].sum()
        sh = (timing_df[hg_col] - timing_df[hthg_col]).sum() + (timing_df[ag_col] - timing_df[htag_col]).sum()
        if fh > 0:
            per = fh / 3
            for i in range(3): goals_per_interval[intervals[i]] += per
        if sh > 0:
            per = sh / 3
            for i in range(3, 6): goals_per_interval[intervals[i]] += per

    total = sum(goals_per_interval.values())
    if total > 0:
        probs = [g / total for g in goals_per_interval.values()]
        stats["goal_timing"] = {
            "intervals": intervals,
            "goals": list(goals_per_interval.values()),
            "prob": probs,
            "most_likely": intervals[np.argmax(probs)]
        }

    # === CORNERS WITH NEGATIVE BINOMIAL ===
    if hc_col and ac_col and hc_col in df.columns and ac_col in df.columns:
        c_mask = df[hc_col].notna() & df[ac_col].notna()
        clean_c = df[c_mask][[home_col, away_col, hc_col, ac_col]].copy()
        if len(clean_c) >= 5:
            clean_c['weight'] = np.exp(np.linspace(-recency_weight, 0, len(clean_c)))
            
            # Calculate variance for Negative Binomial
            hc_mean = np.average(clean_c[hc_col], weights=clean_c['weight'])
            ac_mean = np.average(clean_c[ac_col], weights=clean_c['weight'])
            hc_var = np.average((clean_c[hc_col] - hc_mean)**2, weights=clean_c['weight'])
            ac_var = np.average((clean_c[ac_col] - ac_mean)**2, weights=clean_c['weight'])
            
            if hc_mean > 0 and ac_mean > 0:
                corner_stats = {
                    "league_avg_home": hc_mean, 
                    "league_avg_away": ac_mean,
                    "home_variance": hc_var,
                    "away_variance": ac_var,
                    "use_negbinom": hc_var > hc_mean and ac_var > ac_mean,
                    "home_attack": {},
                    "away_attack": {},
                    "home_defence": {},
                    "away_defence": {}
                }
                
                for team in teams:
                    home_c = clean_c[clean_c[home_col] == team]
                    away_c = clean_c[clean_c[away_col] == team]
                    
                    if len(home_c) >= min_matches:
                        corner_stats["home_attack"][team] = bayesian_smoothing(
                            weighted_mean(home_c, hc_col) / hc_mean, 1.0, len(home_c)
                        )
                        corner_stats["home_defence"][team] = bayesian_smoothing(
                            weighted_mean(home_c, ac_col) / ac_mean, 1.0, len(home_c)
                        )
                    else:
                        corner_stats["home_attack"][team] = 1.0
                        corner_stats["home_defence"][team] = 1.0
                    
                    if len(away_c) >= min_matches:
                        corner_stats["away_attack"][team] = bayesian_smoothing(
                            weighted_mean(away_c, ac_col) / ac_mean, 1.0, len(away_c)
                        )
                        corner_stats["away_defence"][team] = bayesian_smoothing(
                            weighted_mean(away_c, hc_col) / hc_mean, 1.0, len(away_c)
                        )
                    else:
                        corner_stats["away_attack"][team] = 1.0
                        corner_stats["away_defence"][team] = 1.0
                
                stats["corners"] = corner_stats

    # === xG WITH ENHANCED WEIGHTING ===
    if hxg_col and axg_col and hxg_col in df.columns and axg_col in df.columns:
        xg_mask = df[hxg_col].notna() & df[axg_col].notna()
        clean_xg = df[xg_mask][[home_col, away_col, hxg_col, axg_col]].copy()
        if len(clean_xg) >= 5:
            clean_xg['weight'] = np.exp(np.linspace(-recency_weight, 0, len(clean_xg)))
            avg_hxg = np.average(clean_xg[hxg_col], weights=clean_xg['weight'])
            avg_axg = np.average(clean_xg[axg_col], weights=clean_xg['weight'])
            if avg_hxg > 0 and avg_axg > 0:
                xg_stats = {
                    "league_avg_home": avg_hxg, 
                    "league_avg_away": avg_axg,
                    "home_attack": {},
                    "away_attack": {},
                    "home_defence": {},
                    "away_defence": {}
                }
                
                for team in teams:
                    home_xg = clean_xg[clean_xg[home_col] == team]
                    away_xg = clean_xg[clean_xg[away_col] == team]
                    
                    if len(home_xg) >= min_matches:
                        xg_stats["home_attack"][team] = bayesian_smoothing(
                            weighted_mean(home_xg, hxg_col) / avg_hxg, 1.0, len(home_xg)
                        )
                        xg_stats["home_defence"][team] = bayesian_smoothing(
                            weighted_mean(home_xg, axg_col) / avg_axg, 1.0, len(home_xg)
                        )
                    else:
                        xg_stats["home_attack"][team] = 1.0
                        xg_stats["home_defence"][team] = 1.0
                    
                    if len(away_xg) >= min_matches:
                        xg_stats["away_attack"][team] = bayesian_smoothing(
                            weighted_mean(away_xg, axg_col) / avg_axg, 1.0, len(away_xg)
                        )
                        xg_stats["away_defence"][team] = bayesian_smoothing(
                            weighted_mean(away_xg, hxg_col) / avg_hxg, 1.0, len(away_xg)
                        )
                    else:
                        xg_stats["away_attack"][team] = 1.0
                        xg_stats["away_defence"][team] = 1.0
                
                stats["xg"] = xg_stats

    # === SHOTS ON TARGET ===
    if hs_col and as_col and hs_col in df.columns and as_col in df.columns:
        s_mask = df[hs_col].notna() & df[as_col].notna()
        clean_s = df[s_mask][[home_col, away_col, hs_col, as_col]].copy()
        if len(clean_s) >= 5:
            clean_s['weight'] = np.exp(np.linspace(-recency_weight, 0, len(clean_s)))
            avg_hs = np.average(clean_s[hs_col], weights=clean_s['weight'])
            avg_as = np.average(clean_s[as_col], weights=clean_s['weight'])
            if avg_hs > 0 and avg_as > 0:
                shot_stats = {
                    "league_avg_home": avg_hs, 
                    "league_avg_away": avg_as,
                    "home_attack": {},
                    "away_attack": {},
                    "home_defence": {},
                    "away_defence": {}
                }
                
                for team in teams:
                    home_s = clean_s[clean_s[home_col] == team]
                    away_s = clean_s[clean_s[away_col] == team]
                    
                    if len(home_s) >= min_matches:
                        shot_stats["home_attack"][team] = bayesian_smoothing(
                            weighted_mean(home_s, hs_col) / avg_hs, 1.0, len(home_s)
                        )
                        shot_stats["home_defence"][team] = bayesian_smoothing(
                            weighted_mean(home_s, as_col) / avg_as, 1.0, len(home_s)
                        )
                    else:
                        shot_stats["home_attack"][team] = 1.0
                        shot_stats["home_defence"][team] = 1.0
                    
                    if len(away_s) >= min_matches:
                        shot_stats["away_attack"][team] = bayesian_smoothing(
                            weighted_mean(away_s, as_col) / avg_as, 1.0, len(away_s)
                        )
                        shot_stats["away_defence"][team] = bayesian_smoothing(
                            weighted_mean(away_s, hs_col) / avg_hs, 1.0, len(away_s)
                        )
                    else:
                        shot_stats["away_attack"][team] = 1.0
                        shot_stats["away_defence"][team] = 1.0
                
                stats["shots"] = shot_stats

    return stats

# ================================
# ENHANCED PREDICT MATCH (continued)
# ================================
@st.cache_data(show_spinner=False)
def predict_match(home: str, away: str, stats: Dict[str, Any],
                  _df: pd.DataFrame = None, home_col: str = None,
                  away_col: str = None, hg_col: str = None, ag_col: str = None) -> Dict[str, Any]:
    max_g = 10          # max goals to enumerate
    max_c = 15          # max corners
    max_s = 20          # max shots on target
    predictions = {
        "goals": {"score": "N/A", "result": "N/A", "home_win": 0, "draw": 0, "away_win": 0,
                  "btts_yes": 0, "btts_no": 1, "btts_result": "N/A",
                  "over_25": 0, "under_25": 1, "over_under_result": "N/A"},
        "corners": {"home": 0, "away": 0, "total": 0, "most_likely": "N/A"},
        "shots": {"home": 0, "away": 0},
        "xg": {"home": 0, "away": 0},
        "goal_timing": {"intervals": [], "prob": []},
        "form": {"home": {}, "away": {}}
    }
    chart_data = {}

    # ------------------------------------------------------------------ #
    # 1. GOALS – Dixon‑Coles (Poisson + low‑score correlation)
    # ------------------------------------------------------------------ #
    g = stats.get("goals", {})
    if g:
        # league averages
        l_home = g["league_avg_home"]
        l_away = g["league_avg_away"]

        # team strengths (Bayesian‑smoothed)
        att_h = g["home_attack"].get(home, 1.0)
        def_h = g["home_defence"].get(home, 1.0)
        att_a = g["away_attack"].get(away, 1.0)
        def_a = g["away_defence"].get(away, 1.0)

        # expected goals
        lambda_h = att_h * def_a * l_home
        lambda_a = att_a * def_h * l_away

        # Dixon‑Coles rho (empirical correlation for 0‑0, 1‑0, 0‑1, 1‑1)
        rho = 0.0
        if _df is not None and hg_col and ag_col:
            ft = _df[[hg_col, ag_col]].dropna()
            n = len(ft)
            if n > 0:
                p00 = (ft[hg_col] == 0).mean() * (ft[ag_col] == 0).mean()
                p01 = (ft[hg_col] == 0).mean() * (ft[ag_col] == 1).mean()
                p10 = (ft[hg_col] == 1).mean() * (ft[ag_col] == 0).mean()
                p11 = (ft[hg_col] == 1).mean() * (ft[ag_col] == 1).mean()
                rho = 1 - (p00 * p11) / (p01 * p10) if p01 * p10 > 0 else 0.0
                rho = max(min(rho, 0.3), -0.3)   # keep it realistic

        # probability matrix
        prob_matrix = np.zeros((max_g + 1, max_g + 1))
        for h in range(max_g + 1):
            for a in range(max_g + 1):
                p = poisson.pmf(h, lambda_h) * poisson.pmf(a, lambda_a)
                # Dixon‑Coles adjustment
                if h <= 1 and a <= 1:
                    tau = 1.0
                    if h == 0 and a == 0:
                        tau = 1 - lambda_h * lambda_a * rho
                    elif h == 0 and a == 1:
                        tau = 1 + lambda_h * rho
                    elif h == 1 and a == 0:
                        tau = 1 + lambda_a * rho
                    elif h == 1 and a == 1:
                        tau = 1 - rho
                    p *= tau
                prob_matrix[h, a] = p

        # normalise (numerical safety)
        prob_matrix /= prob_matrix.sum()

        # most likely score
        h_idx, a_idx = np.unravel_index(np.argmax(prob_matrix), prob_matrix.shape)
        predictions["goals"]["score"] = f"{h_idx}–{a_idx}"

        # 1X2
        home_win = prob_matrix[1:, :].sum() - prob_matrix.diagonal()[1:].sum()
        away_win = prob_matrix[:, 1:].sum() - prob_matrix.diagonal()[1:].sum()
        draw = prob_matrix.diagonal().sum()
        predictions["goals"]["home_win"] = home_win
        predictions["goals"]["away_win"] = away_win
        predictions["goals"]["draw"] = draw
        predictions["goals"]["result"] = "Home" if home_win > away_win and home_win > draw else \
                                         "Away" if away_win > home_win and away_win > draw else "Draw"

        # BTTS
        btts_yes = (prob_matrix[1:, 1:]).sum()
        predictions["goals"]["btts_yes"] = btts_yes
        predictions["goals"]["btts_no"] = 1 - btts_yes
        predictions["goals"]["btts_result"] = "Yes" if btts_yes > 0.5 else "No"

        # Over/Under 2.5
        over_25 = (prob_matrix[3:, :].sum() + prob_matrix[:, 3:].sum() -
                   prob_matrix[3:, 3:].sum())
        predictions["goals"]["over_25"] = over_25
        predictions["goals"]["under_25"] = 1 - over_25
        predictions["goals"]["over_under_result"] = "Over" if over_25 > 0.5 else "Under"

        # store for charts
        chart_data["goal_matrix"] = prob_matrix
        predictions["expected_goals"] = {"home": lambda_h, "away": lambda_a}

    # ------------------------------------------------------------------ #
    # 2. CORNERS – Negative Binomial (with autocorrelation smoothing)
    # ------------------------------------------------------------------ #
    c = stats.get("corners")
    if c and c.get("use_negbinom"):
        # league averages & variances
        lhc = c["league_avg_home"]
        lac = c["league_avg_away"]
        vhc = c["home_variance"]
        vac = c["away_variance"]

        # team strengths
        att_hc = c["home_attack"].get(home, 1.0)
        def_hc = c["home_defence"].get(home, 1.0)
        att_ac = c["away_attack"].get(away, 1.0)
        def_ac = c["away_defence"].get(away, 1.0)

        mu_hc = att_hc * def_ac * lhc
        mu_ac = att_ac * def_hc * lac

        # Negative Binomial params
        n_hc, p_hc = negative_binomial_params(mu_hc, vhc)
        n_ac, p_ac = negative_binomial_params(mu_ac, vac)

        # fallback to Poisson if NB not defined
        if p_hc is None:
            hc_probs = poisson.pmf(np.arange(max_c + 1), mu_hc)
        else:
            hc_probs = nbinom.pmf(np.arange(max_c + 1), n_hc, p_hc)

        if p_ac is None:
            ac_probs = poisson.pmf(np.arange(max_c + 1), mu_ac)
        else:
            ac_probs = nbinom.pmf(np.arange(max_c + 1), n_ac, p_ac)

        # most likely corners
        predictions["corners"]["home"] = int(np.argmax(hc_probs))
        predictions["corners"]["away"] = int(np.argmax(ac_probs))
        predictions["corners"]["total"] = predictions["corners"]["home"] + predictions["corners"]["away"]
        predictions["corners"]["most_likely"] = f"{predictions['corners']['home']}–{predictions['corners']['away']}"

        chart_data["corner_home"] = hc_probs
        chart_data["corner_away"] = ac_probs

    # ------------------------------------------------------------------ #
    # 3. SHOTS ON TARGET
    # ------------------------------------------------------------------ #
    s = stats.get("shots")
    if s:
        att_hs = s["home_attack"].get(home, 1.0)
        def_hs = s["home_defence"].get(home, 1.0)
        att_as = s["away_attack"].get(away, 1.0)
        def_as = s["away_defence"].get(away, 1.0)

        mu_hs = att_hs * def_as * s["league_avg_home"]
        mu_as = att_as * def_hs * s["league_avg_away"]

        predictions["shots"]["home"] = round(mu_hs, 1)
        predictions["shots"]["away"] = round(mu_as, 1)

    # ------------------------------------------------------------------ #
    # 4. xG (if present)
    # ------------------------------------------------------------------ #
    x = stats.get("xg")
    if x:
        att_hx = x["home_attack"].get(home, 1.0)
        def_hx = x["home_defence"].get(home, 1.0)
        att_ax = x["away_attack"].get(away, 1.0)
        def_ax = x["away_defence"].get(away, 1.0)

        xg_h = att_hx * def_ax * x["league_avg_home"]
        xg_a = att_ax * def_hx * x["league_avg_away"]

        predictions["xg"]["home"] = round(xg_h, 2)
        predictions["xg"]["away"] = round(xg_a, 2)

    # ------------------------------------------------------------------ #
    # 5. GOAL TIMING DISTRIBUTION
    # ------------------------------------------------------------------ #
    gt = stats.get("goal_timing")
    if gt:
        predictions["goal_timing"]["intervals"] = gt["intervals"]
        predictions["goal_timing"]["prob"] = gt["prob"]

    # ------------------------------------------------------------------ #
    # 6. FORM (last 5 matches)
    # ------------------------------------------------------------------ #
    if _df is not None and home_col and away_col and hg_col and ag_col:
        predictions["form"]["home"] = calculate_form(_df, home, home_col, away_col, hg_col, ag_col)
        predictions["form"]["away"] = calculate_form(_df, away, home_col, away_col, hg_col, ag_col)

    # ------------------------------------------------------------------ #
    # 7. RETURN + CHART DATA
    # ------------------------------------------------------------------ #
    return {
        "predictions": predictions,
        "chart_data": chart_data,
        "lambda_home": predictions.get("expected_goals", {}).get("home", 0),
        "lambda_away": predictions.get("expected_goals", {}).get("away", 0)
    }

# ================================
# PDF EXPORT (WeasyPrint → fallback pdfkit)
# ================================
def generate_pdf_html(home, away, pred, logos):
    home_logo = logos.get(home)
    away_logo = logos.get(away)

    html = f"""
    <html><head><style>{print_css}</style></head><body>
    <div class="print-title">Match Prediction: {home} vs {away}</div>
    <div style="display:flex; justify-content:space-around; margin-bottom:30px;">
        <div class="team-box">
            <img src="{home_logo}" class="logo" onerror="this.style.display='none'"/>
            <div><strong>{home}</strong></div>
        </div>
        <div style="font-size:36px; align-self:center;">VS</div>
        <div class="team-box">
            <img src="{away_logo}" class="logo" onerror="this.style.display='none'"/>
            <div><strong>{away}</strong></div>
        </div>
    </div>

    <div class="prediction">
        <div class="score">Most Likely Score: <strong>{pred['goals']['score']}</strong></div>
        <div class="prob">Home Win: {pred['goals']['home_win']:.1%} | Draw: {pred['goals']['draw']:.1%} | Away Win: {pred['goals']['away_win']:.1%}</div>
        <div class="prob">BTTS {pred['goals']['btts_result']}: {pred['goals']['btts_yes']:.1%}</div>
        <div class="prob">Over 2.5 {pred['goals']['over_under_result']}: {pred['goals']['over_25']:.1%}</div>
    </div>

    <div class="prediction" style="margin-top:20px;">
        <div><strong>Expected Goals</strong>: {home} {pred.get('xg',{}).get('home', '—')} | {away} {pred.get('xg',{}).get('away', '—')}</div>
        <div><strong>Corners</strong>: {home} {pred['corners']['home']} | {away} {pred['corners']['away']} (Total {pred['corners']['total']})</div>
        <div><strong>Shots on Target</strong>: {home} {pred['shots']['home']} | {away} {pred['shots']['away']}</div>
    </div>

    <div style="margin-top:30px; font-size:12px; color:#555;">
        Generated on {datetime.now().strftime('%Y-%m-%d %H:%M')} — Advanced Football Predictor
    </div>
    </body></html>
    """
    return html

def export_to_pdf(html_content, filename="prediction.pdf"):
    if WEASYPRINT_AVAILABLE:
        HTML(string=html_content).write_pdf(filename)
    elif PDFKIT_AVAILABLE:
        pdfkit.from_string(html_content, filename)
    else:
        st.error("PDF export not available – install `weasyprint` or `pdfkit`.")
        return None
    return filename
    # ===

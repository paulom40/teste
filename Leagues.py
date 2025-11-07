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
import time

# --- PDF EXPORT (WeasyPrint fallback) ---
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
st.title("Advanced Football Match Predictor")
st.markdown("""
**Professional-Grade Prediction Suite with Enhanced Statistical Models**  
**Advanced Features:**  
- Dixon‑Coles Model — Correlation adjustment  
- Negative Binomial for Corners — Better tail modeling  
- Tunable Recency Weighting  
- xG Validation & Form Analysis  
- Bayesian Smoothing  
- **Goal Timing (1–15, 16–30, ...)**  
- **Live Match Prediction**  
**Predicts:** Full‑Time Score | BTTS | Over 2.5 | 1X2 | Corners | xG | Shots | Goal Timing  
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
            return url
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
    team_home = df[df[home_col] == team].tail(n_matches)
    team_away = df[df[away_col] == team].tail(n_matches)
    all_matches = pd.concat([team_home, team_away]).sort_index().tail(n_matches)
    if len(all_matches) == 0:
        return {"avg_goals_scored": 0, "avg_goals_conceded": 0, "points": 0, "form_score": 0}
    goals_scored = goals_conceded = points = 0
    for _, row in all_matches.iterrows():
        is_home = row[home_col] == team
        gf = pd.to_numeric(row[hg_col] if is_home else row[ag_col], errors='coerce')
        ga = pd.to_numeric(row[ag_col] if is_home else row[hg_col], errors='coerce')
        if pd.notna(gf) and pd.notna(ga):
            goals_scored += gf
            goals_conceded += ga
            if gf > ga: points += 3
            elif gf == ga: points += 1
    n_valid = len(all_matches)
    return {
        "avg_goals_scored": goals_scored / n_valid if n_valid > 0 else 0,
        "avg_goals_conceded": goals_conceded / n_valid if n_valid > 0 else 0,
        "points": points,
        "form_score": points / (n_valid * 3) if n_valid > 0 else 0
    }

def bayesian_smoothing(observed_rate: float, league_avg: float, sample_size: int, confidence: int = 10) -> float:
    return (observed_rate * sample_size + league_avg * confidence) / (sample_size + confidence)

def negative_binomial_params(mean: float, variance: float) -> Tuple[float, float]:
    if variance <= mean: return mean, None
    p = mean / variance
    n = mean * p / (1 - p)
    return n, p

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
        if "date" in lower: mapping["Date"] = col
        elif "home" in lower and "team" in lower: mapping["HomeTeam"] = col
        elif "away" in lower and "team" in lower: mapping["AwayTeam"] = col
        elif lower in ["fthg", "hgoals"]: mapping["FTHG"] = col
        elif lower in ["ftag", "agoals"]: mapping["FTAG"] = col
        elif lower in ["hthg", "halfhome"]: mapping["HTHG"] = col
        elif lower in ["htag", "halfaway"]: mapping["HTAG"] = col
        elif lower in ["hc", "homecorners"]: mapping["HC"] = col
        elif lower in ["ac", "awaycorners"]: mapping["AC"] = col
        elif lower in ["hs", "homeshotsontarget"]: mapping["HS"] = col
        elif lower in ["as", "awayshotsontarget"]: mapping["AS"] = col
        elif lower in ["hxg", "home_xg"]: mapping["HxG"] = col
        elif lower in ["axg", "away_xg"]: mapping["AxG"] = col
    return mapping

# ================================
# GOAL MINUTE PARSING
# ================================
def extract_goal_minutes(df: pd.DataFrame, home_col: str, away_col: str) -> pd.DataFrame:
    goal_df = pd.DataFrame(index=df.index)
    goal_df['home_goals'] = pd.NA
    goal_df['away_goals'] = pd.NA
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
                if team_lower == home_team: home.append(minute)
                elif team_lower == away_team: away.append(minute)
            return home, away
        parsed = df.apply(parse_goal_time, axis=1)
        goal_df['home_goals'] = parsed.apply(lambda x: x[0])
        goal_df['away_goals'] = parsed.apply(lambda x: x[1])
        return goal_df
    return None

# ================================
# MODEL TRAINING
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
    ft_mask = df[hg_col].notna() & df[ag_col].notna()
    clean_ft = df[ft_mask][[home_col, away_col, hg_col, ag_col]].copy()
    if len(clean_ft) < 5:
        raise ValueError(f"Only {len(clean_ft)} valid matches.")
    clean_ft['weight'] = np.exp(np.linspace(-recency_weight, 0, len(clean_ft)))
    avg_home = np.average(clean_ft[hg_col], weights=clean_ft['weight'])
    avg_away = np.average(clean_ft[ag_col], weights=clean_ft['weight'])
    def weighted_mean(group, col, weight_col='weight'):
        if len(group) < min_matches: return None
        return np.average(group[col], weights=group[weight_col])
    home_attack_raw = away_attack_raw = home_defence_raw = away_defence_raw = {}
    team_sample_sizes = {}
    for team in teams:
        home_matches = clean_ft[clean_ft[home_col] == team]
        away_matches = clean_ft[clean_ft[away_col] == team]
        n_home = len(home_matches); n_away = len(away_matches)
        team_sample_sizes[team] = n_home + n_away
        if n_home >= min_matches:
            ha = weighted_mean(home_matches, hg_col) / avg_home
            hd = weighted_mean(home_matches, ag_col) / avg_away
            home_attack_raw[team] = bayesian_smoothing(ha, 1.0, n_home)
            home_defence_raw[team] = bayesian_smoothing(hd, 1.0, n_home)
        else:
            home_attack_raw[team] = home_defence_raw[team] = 1.0
        if n_away >= min_matches:
            aa = weighted_mean(away_matches, ag_col) / avg_away
            ad = weighted_mean(away_matches, hg_col) / avg_home
            away_attack_raw[team] = bayesian_smoothing(aa, 1.0, n_away)
            away_defence_raw[team] = bayesian_smoothing(ad, 1.0, n_away)
        else:
            away_attack_raw[team] = away_defence_raw[team] = 1.0
    stats["goals"] = {
        "league_avg_home": avg_home, "league_avg_away": avg_away,
        "home_attack": home_attack_raw, "away_attack": away_attack_raw,
        "home_defence": home_defence_raw, "away_defence": away_defence_raw,
        "sample_sizes": team_sample_sizes
    }
    # xG, corners, shots, goal timing (same as before)
    if hxg_col and axg_col and hxg_col in df.columns and axg_col in df.columns:
        xg_mask = df[hxg_col].notna() & df[axg_col].notna() & ft_mask
        if xg_mask.sum() >= 5:
            xg_comparison = df[xg_mask].copy()
            xg_comparison['home_diff'] = xg_comparison[hg_col] - xg_comparison[hxg_col]
            xg_comparison['away_diff'] = xg_comparison[ag_col] - xg_comparison[axg_col]
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
    if hc_col and ac_col and hc_col in df.columns and ac_col in df.columns:
        c_mask = df[hc_col].notna() & df[ac_col].notna()
        clean_c = df[c_mask][[home_col, away_col, hc_col, ac_col]].copy()
        if len(clean_c) >= 5:
            clean_c['weight'] = np.exp(np.linspace(-recency_weight, 0, len(clean_c)))
            hc_mean = np.average(clean_c[hc_col], weights=clean_c['weight'])
            ac_mean = np.average(clean_c[ac_col], weights=clean_c['weight'])
            hc_var = np.average((clean_c[hc_col] - hc_mean)**2, weights=clean_c['weight'])
            ac_var = np.average((clean_c[ac_col] - ac_mean)**2, weights=clean_c['weight'])
            if hc_mean > 0 and ac_mean > 0:
                corner_stats = {
                    "league_avg_home": hc_mean, "league_avg_away": ac_mean,
                    "home_variance": hc_var, "away_variance": ac_var,
                    "use_negbinom": hc_var > hc_mean and ac_var > ac_mean,
                    "home_attack": {}, "away_attack": {}, "home_defence": {}, "away_defence": {}
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
                        corner_stats["home_attack"][team] = corner_stats["home_defence"][team] = 1.0
                    if len(away_c) >= min_matches:
                        corner_stats["away_attack"][team] = bayesian_smoothing(
                            weighted_mean(away_c, ac_col) / ac_mean, 1.0, len(away_c)
                        )
                        corner_stats["away_defence"][team] = bayesian_smoothing(
                            weighted_mean(away_c, hc_col) / hc_mean, 1.0, len(away_c)
                        )
                    else:
                        corner_stats["away_attack"][team] = corner_stats["away_defence"][team] = 1.0
                stats["corners"] = corner_stats
    if hxg_col and axg_col and hxg_col in df.columns and axg_col in df.columns:
        xg_mask = df[hxg_col].notna() & df[axg_col].notna()
        clean_xg = df[xg_mask][[home_col, away_col, hxg_col, axg_col]].copy()
        if len(clean_xg) >= 5:
            clean_xg['weight'] = np.exp(np.linspace(-recency_weight, 0, len(clean_xg)))
            avg_hxg = np.average(clean_xg[hxg_col], weights=clean_xg['weight'])
            avg_axg = np.average(clean_xg[axg_col], weights=clean_xg['weight'])
            if avg_hxg > 0 and avg_axg > 0:
                xg_stats = {
                    "league_avg_home": avg_hxg, "league_avg_away": avg_axg,
                    "home_attack": {}, "away_attack": {}, "home_defence": {}, "away_defence": {}
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
                        xg_stats["home_attack"][team] = xg_stats["home_defence"][team] = 1.0
                    if len(away_xg) >= min_matches:
                        xg_stats["away_attack"][team] = bayesian_smoothing(
                            weighted_mean(away_xg, axg_col) / avg_axg, 1.0, len(away_xg)
                        )
                        xg_stats["away_defence"][team] = bayesian_smoothing(
                            weighted_mean(away_xg, hxg_col) / avg_hxg, 1.0, len(away_xg)
                        )
                    else:
                        xg_stats["away_attack"][team] = xg_stats["away_defence"][team] = 1.0
                stats["xg"] = xg_stats
    if hs_col and as_col and hs_col in df.columns and as_col in df.columns:
        s_mask = df[hs_col].notna() & df[as_col].notna()
        clean_s = df[s_mask][[home_col, away_col, hs_col, as_col]].copy()
        if len(clean_s) >= 5:
            clean_s['weight'] = np.exp(np.linspace(-recency_weight, 0, len(clean_s)))
            avg_hs = np.average(clean_s[hs_col], weights=clean_s['weight'])
            avg_as = np.average(clean_s[as_col], weights=clean_s['weight'])
            if avg_hs > 0 and avg_as > 0:
                shot_stats = {
                    "league_avg_home": avg_hs, "league_avg_away": avg_as,
                    "home_attack": {}, "away_attack": {}, "home_defence": {}, "away_defence": {}
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
                        shot_stats["home_attack"][team] = shot_stats["home_defence"][team] = 1.0
                    if len(away_s) >= min_matches:
                        shot_stats["away_attack"][team] = bayesian_smoothing(
                            weighted_mean(away_s, as_col) / avg_as, 1.0, len(away_s)
                        )
                        shot_stats["away_defence"][team] = bayesian_smoothing(
                            weighted_mean(away_s, hs_col) / avg_hs, 1.0, len(away_s)
                        )
                    else:
                        shot_stats["away_attack"][team] = shot_stats["away_defence"][team] = 1.0
                stats["shots"] = shot_stats
    return stats

# ================================
# PREDICT MATCH (PRE-MATCH)
# ================================
@st.cache_data(show_spinner=False)
def predict_match(home: str, away: str, stats: Dict[str, Any],
                  _df: pd.DataFrame = None, home_col: str = None,
                  away_col: str = None, hg_col: str = None, ag_col: str = None) -> Dict[str, Any]:
    max_g = 10; max_c = 15; max_s = 20
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
    g = stats.get("goals", {})
    if g:
        l_home = g["league_avg_home"]; l_away = g["league_avg_away"]
        att_h = g["home_attack"].get(home, 1.0); def_h = g["home_defence"].get(home, 1.0)
        att_a = g["away_attack"].get(away, 1.0); def_a = g["away_defence"].get(away, 1.0)
        lambda_h = att_h * def_a * l_home; lambda_a = att_a * def_h * l_away
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
                rho = max(min(rho, 0.3), -0.3)
        prob_matrix = np.zeros((max_g + 1, max_g + 1))
        for h in range(max_g + 1):
            for a in range(max_g + :max_g + 1):
                p = poisson.pmf(h, lambda_h) * poisson.pmf(a, lambda_a)
                if h <= 1 and a <= 1:
                    tau = 1.0
                    if h == 0 and a == 0: tau = 1 - lambda_h * lambda_a * rho
                    elif h == 0 and a == 1: tau = 1 + lambda_h * rho
                    elif h == 1 and a == 0: tau = 1 + lambda_a * rho
                    elif h == 1 and a == 1: tau = 1 - rho
                    p *= tau
                prob_matrix[h, a] = p
        prob_matrix /= prob_matrix.sum()
        h_idx, a_idx = np.unravel_index(np.argmax(prob_matrix), prob_matrix.shape)
        predictions["goals"]["score"] = f"{h_idx}–{a_idx}"
        home_win = prob_matrix[1:, :].sum() - prob_matrix.diagonal()[1:].sum()
        away_win = prob_matrix[:, 1:].sum() - prob_matrix.diagonal()[1:].sum()
        draw = prob_matrix.diagonal().sum()
        predictions["goals"]["home_win"] = home_win
        predictions["goals"]["away_win"] = away_win
        predictions["goals"]["draw"] = draw
        predictions["goals"]["result"] = "Home" if home_win > away_win and home_win > draw else \
                                         "Away" if away_win > home_win and away_win > draw else "Draw"
        btts_yes = (prob_matrix[1:, 1:]).sum()
        predictions["goals"]["btts_yes"] = btts_yes
        predictions["goals"]["btts_no"] = 1 - btts_yes
        predictions["goals"]["btts_result"] = "Yes" if btts_yes > 0.5 else "No"
        over_25 = (prob_matrix[3:, :].sum() + prob_matrix[:, 3:].sum() - prob_matrix[3:, 3:].sum())
        predictions["goals"]["over_25"] = over_25
        predictions["goals"]["under_25"] = 1 - over_25
        predictions["goals"]["over_under_result"] = "Over" if over_25 > 0.5 else "Under"
        chart_data["goal_matrix"] = prob_matrix
        predictions["expected_goals"] = {"home": lambda_h, "away": lambda_a}
    c = stats.get("corners")
    if c and c.get("use_negbinom"):
        lhc = c["league_avg_home"]; lac = c["league_avg_away"]
        vhc = c["home_variance"]; vac = c["away_variance"]
        att_hc = c["home_attack"].get(home, 1.0); def_hc = c["home_defence"].get(home, 1.0)
        att_ac = c["away_attack"].get(away, 1.0); def_ac = c["away_defence"].get(away, 1.0)
        mu_hc = att_hc * def_ac * lhc; mu_ac = att_ac * def_hc * lac
        n_hc, p_hc = negative_binomial_params(mu_hc, vhc)
        n_ac, p_ac = negative_binomial_params(mu_ac, vac)
        if p_hc is None:
            hc_probs = poisson.pmf(np.arange(max_c + 1), mu_hc)
        else:
            hc_probs = nbinom.pmf(np.arange(max_c + 1), n_hc, p_hc)
        if p_ac is None:
            ac_probs = poisson.pmf(np.arange(max_c + 1), mu_ac)
        else:
            ac_probs = nbinom.pmf(np.arange(max_c + 1), n_ac, p_ac)
        predictions["corners"]["home"] = int(np.argmax(hc_probs))
        predictions["corners"]["away"] = int(np.argmax(ac_probs))
        predictions["corners"]["total"] = predictions["corners"]["home"] + predictions["corners"]["away"]
        predictions["corners"]["most_likely"] = f"{predictions['corners']['home']}–{predictions['corners']['away']}"
        chart_data["corner_home"] = hc_probs
        chart_data["corner_away"] = ac_probs
    s = stats.get("shots")
    if s:
        att_hs = s["home_attack"].get(home, 1.0); def_hs = s["home_defence"].get(home, 1.0)
        att_as = s["away_attack"].get(away, 1.0); def_as = s["away_defence"].get(away, 1.0)
        mu_hs = att_hs * def_as * s["league_avg_home"]
        mu_as = att_as * def_hs * s["league_avg_away"]
        predictions["shots"]["home"] = round(mu_hs, 1)
        predictions["shots"]["away"] = round(mu_as, 1)
    x = stats.get("xg")
    if x:
        att_hx = x["home_attack"].get(home, 1.0); def_hx = x["home_defence"].get(home, 1.0)
        att_ax = x["away_attack"].get(away, 1.0); def_ax = x["away_defence"].get(away, 1.0)
        xg_h = att_hx * def_ax * x["league_avg_home"]
        xg_a = att_ax * def_hx * x["league_avg_away"]
        predictions["xg"]["home"] = round(xg_h, 2)
        predictions["xg"]["away"] = round(xg_a, 2)
    gt = stats.get("goal_timing")
    if gt:
        predictions["goal_timing"]["intervals"] = gt["intervals"]
        predictions["goal_timing"]["prob"] = gt["prob"]
    if _df is not None and home_col and away_col and hg_col and ag_col:
        predictions["form"]["home"] = calculate_form(_df, home, home_col, away_col, hg_col, ag_col)
        predictions["form"]["away"] = calculate_form(_df, away, home_col, away_col, hg_col, ag_col)
    return {
        "predictions": predictions,
        "chart_data": chart_data,
        "lambda_home": predictions.get("expected_goals", {}).get("home", 0),
        "lambda_away": predictions.get("expected_goals", {}).get("away", 0)
    }

# ================================
# PDF EXPORT WITH HTML BUTTON
# ================================
def generate_pdf_html(home, away, pred, logos, subtitle: str = None):
    home_logo = logos.get(home, "")
    away_logo = logos.get(away, "")
    title = f"Match Prediction: {home} vs {away}"
    if subtitle:
        title = f"{title}<br><small style='font-weight:normal;'>{subtitle}</small>"
    html = f"""
    <html><head><style>{print_css}</style></head><body>
    <div class="print-title">{title}</div>
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
        <div><strong>Expected Goals</strong>: {home} {pred.get('xg',{}).get('home','—')} | {away} {pred.get('xg',{}).get('away','—')}</div>
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
        return filename
    elif PDFKIT_AVAILABLE:
        pdfkit.from_string(html_content, filename)
        return filename
    else:
        st.warning("Install `weasyprint` or `pdfkit` for PDF export.")
        return None

# ================================
# CSV UPLOAD & MODEL TRAINING
# ================================
st.sidebar.header("Upload Match Data")
uploaded_file = st.sidebar.file_uploader("Choose a CSV file", type=["csv"], help="Required: HomeTeam, AwayTeam, FTHG, FTAG")
df = None; mapping = {}; team_stats = {}; teams = []

if uploaded_file is not None:
    csv_bytes = uploaded_file.read()
    df = load_csv(csv_bytes)
    if df.empty:
        st.error("CSV is empty.")
    else:
        st.success(f"Loaded {len(df):,} matches")
        mapping = detect_columns(df)
        st.sidebar.subheader("Column Mapping")
        col_map = {}
        for label, col in mapping.items():
            col_map[label] = st.sidebar.selectbox(f"{label}", options=[""] + list(df.columns), index=1 if col in df.columns else 0, key=f"map_{label}")
        required = ["HomeTeam", "AwayTeam", "FTHG", "FTAG"]
        for r in required:
            if r not in col_map or not col_map[r]:
                st.error(f"Please map **{r}**")
                st.stop()
        df_clean = df.rename(columns={v: k for k, v in col_map.items() if v})
        for c in ["FTHG", "FTAG"]:
            df_clean[c] = pd.to_numeric(df_clean[c], errors="coerce")
        with st.spinner("Training model..."):
            team_stats = compute_team_stats(
                _df=df_clean,
                home_col=col_map["HomeTeam"], away_col=col_map["AwayTeam"],
                hg_col=col_map["FTHG"], ag_col=col_map["FTAG"],
                hthg_col=col_map.get("HTHG"), htag_col=col_map.get("HTAG"),
                hc_col=col_map.get("HC"), ac_col=col_map.get("AC"),
                hs_col=col_map.get("HS"), as_col=col_map.get("AS"),
                hxg_col=col_map.get("HxG"), axg_col=col_map.get("AxG"),
                recency_weight=st.sidebar.slider("Recency Weight", 0.5, 5.0, 2.0, 0.1),
                min_matches=st.sidebar.number_input("Min matches per team", 1, 20, 3)
            )
        st.success("Model ready!")
        teams = sorted(set(df_clean[col_map["HomeTeam"]]).union(df_clean[col_map["AwayTeam"]]))
else:
    st.info("Upload a CSV to start.")

# ================================
# PRE-MATCH PREDICTION
# ================================
if uploaded_file and teams:
    st.markdown("---")
    st.subheader("Pre-Match Prediction")
    col1, col2 = st.columns(2)
    with col1:
        home_team = st.selectbox("Home Team", options=teams)
    with col2:
        away_team = st.selectbox("Away Team", options=teams)

    if st.button("Predict Match"):
        pred = predict_match(home_team, away_team, team_stats, _df=df_clean,
                             home_col=col_map["HomeTeam"], away_col=col_map["AwayTeam"],
                             hg_col=col_map["FTHG"], ag_col=col_map["FTAG"])
        p = pred["predictions"]
        st.markdown(f"### **{home_team} vs {away_team}**")
        colA, colB, colC = st.columns([1,2,1])
        with colA:
            h_logo = get_team_logo(home_team)
            if h_logo: st.image(load_image(h_logo), width=80)
            st.write(f"**{home_team}**")
        with colC:
            a_logo = get_team_logo(away_team)
            if a_logo: st.image(load_image(a_logo), width=80)
            st.write(f"**{away_team}**")
        with colB:
            st.markdown(f"<h2 style='text-align:center;'>{p['goals']['score']}</h2>", unsafe_allow_html=True)
        st.markdown("#### Win / Draw")
        colW1, colW2, colW3 = st.columns(3)
        with colW1: st.metric("Home Win", f"{p['goals']['home_win']:.1%}")
        with colW2: st.metric("Draw", f"{p['goals']['draw']:.1%}")
        with colW3: st.metric("Away Win", f"{p['goals']['away_win']:.1%}")
        colB1, colB2 = st.columns(2)
        with colB1: st.metric("BTTS", p['goals']['btts_result'], f"{p['goals']['btts_yes']:.1%}")
        with colB2: st.metric("Over 2.5", p['goals']['over_under_result'], f"{p['goals']['over_25']:.1%}")
        st.markdown("#### Expected Stats")
        colX1, colX2 = st.columns(2)
        with colX1:
            st.write(f"**xG** – {home_team}: {p['xg']['home']} | {away_team}: {p['xg']['away']}")
            st.write(f"**Corners** – {home_team}: {p['corners']['home']} | {away_team}: {p['corners']['away']} (Total {p['corners']['total']})")
        with colX2:
            st.write(f"**Shots on Target** – {home_team}: {p['shots']['home']} | {away_team}: {p['shots']['away']}")
        if p["goal_timing"]["intervals"]:
            fig = px.bar(x=p["goal_timing"]["intervals"], y=p["goal_timing"]["prob"],
                         labels={"x":"Interval","y":"Probability"}, title="Goal Timing")
            st.plotly_chart(fig, use_container_width=True)

        # HTML BUTTON FOR PDF
        logos = {home_team: get_team_logo(home_team), away_team: get_team_logo(away_team)}
        pdf_html = generate_pdf_html(home_team, away_team, p, logos)
        st.markdown("### Export to PDF")
        st.markdown(f"""
        <button onclick="window.print()">Download PDF Report</button>
        <style>
        button {{ background:#0066cc; color:white; padding:10px 20px; border:none; border-radius:5px; cursor:pointer; font-size:16px; }}
        button:hover {{ background:#0055aa; }}
        @media print {{ .no-print, button {{ display:none !important; }} }}
        </style>
        """, unsafe_allow_html=True)
        st.markdown("<small>Click button → Print → Save as PDF</small>", unsafe_allow_html=True)

# ================================
# LIVE MATCH PREDICTION (same as before)
# ================================
if uploaded_file and team_stats:
    st.markdown("---")
    st.subheader("Live Match Prediction")
    col_live1, col_live2 = st.columns([3, 2])
    with col_live1:
        live_home = st.selectbox("Live – Home Team", options=teams, key="live_home")
        live_away = st.selectbox("Live – Away Team", options=teams, key="live_away")
    with col_live2:
        live_minute = st.number_input("Current Minute", 0, 90, 0, step=1)
        col_score1, col_score2 = st.columns(2)
        with col_score1: live_home_score = st.number_input("Home Score", 0, 15, 0, step=1)
        with col_score2: live_away_score = st.number_input("Away Score", 0, 15, 0, step=1)
    if st.button("Update Live Prediction"):
        remaining = max((90 - live_minute) / 90.0, 0.0)
        if remaining == 0: remaining = 1e-9
        g = team_stats["goals"]
        lambda_full_h = g["home_attack"].get(live_home, 1.0) * g["away_defence"].get(live_away, 1.0) * g["league_avg_home"]
        lambda_full_a = g["away_attack"].get(live_away, 1.0) * g["home_defence"].get(live_home, 1.0) * g["league_avg_away"]
        lambda_rem_h = lambda_full_h * remaining; lambda_rem_a = lambda_full_a * remaining
        max_g_rem = 8
        prob_rem = np.zeros((max_g_rem + 1, max_g_rem + 1))
        rho = 0.0
        if hg_col and ag_col and df_clean is not None:
            ft = df_clean[[col_map["FTHG"], col_map["FTAG"]]].dropna()
            if len(ft) > 0:
                p00 = (ft[col_map["FTHG"]] == 0).mean() * (ft[col_map["FTAG"]] == 0).mean()
                p01 = (ft[col_map["FTHG"]] == 0).mean() * (ft[col_map["FTAG"]] == 1).mean()
                p10 = (ft[col_map["FTHG"]] == 1).mean() * (ft[col_map["FTAG"]] == 0).mean()
                p11 = (ft[col_map["FTHG"]] == 1).mean() * (ft[col_map["FTAG"]] == 1).mean()
                rho = 1 - (p00 * p11) / (p01 * p10) if p01 * p10 > 0 else 0.0
                rho = np.clip(rho, -0.3, 0.3)
        for h in range(max_g_rem + 1):
            for a in range(max_g_rem + 1):
                p = poisson.pmf(h, lambda_rem_h) * poisson.pmf(a, lambda_rem_a)
                if h <= 1 and a <= 1:
                    tau = 1.0
                    if h == 0 and a == 0: tau = 1 - lambda_rem_h * lambda_rem_a * rho
                    if h == 0 and a == 1: tau = 1 + lambda_rem_h * rho
                    if h == 1 and a == 0: tau = 1 + lambda_rem_a * rho
                    if h == 1 and a == 1: tau = 1 - rho
                    p *= tau
                prob_rem[h, a] = p
        prob_rem /= prob_rem.sum() or 1.0
        final_home_goals = live_home_score + np.arange(max_g_rem + 1)
        final_away_goals = live_away_score + np.arange(max_g_rem + 1)
        final_matrix = np.zeros((len(final_home_goals), len(final_away_goals)))
        for i, gh in enumerate(final_home_goals):
            for j, ga in enumerate(final_away_goals):
                rem_h = gh - live_home_score
                rem_a = ga - live_away_score
                if 0 <= rem_h <= max_g_rem and 0 <= rem_a <= max_g_rem:
                    final_matrix[i, j] = prob_rem[rem_h, rem_a]
        home_win_live = draw_live = away_win_live = btts_yes_live = over25_live = 0
        for i, gh in enumerate(final_home_goals):
            for j, ga in enumerate(final_away_goals):
                pr = final_matrix[i, j]
                if gh > ga: home_win_live += pr
                elif gh == ga: draw_live += pr
                else: away_win_live += pr
                if gh > 0 and ga > 0: btts_yes_live += pr
                if gh + ga > 2: over25_live += pr
        most_likely = np.unravel_index(np.argmax(final_matrix), final_matrix.shape)
        most_likely_score = f"{final_home_goals[most_likely[0]]}–{final_away_goals[most_likely[1]]}"
        st.markdown(f"### **{live_home} {live_home_score} – {live_away_score} {live_away}** (Minute {live_minute})")
        colL1, colL2, colL3 = st.columns(3)
        with colL1: st.metric("Home Win", f"{home_win_live:.1%}")
        with colL2: st.metric("Draw", f"{draw_live:.1%}")
        with colL3: st.metric("Away Win", f"{away_win_live:.1%}")
        colB1, colB2 = st.columns(2)
        with colB1: st.metric("BTTS", "Yes" if btts_yes_live > 0.5 else "No", f"{btts_yes_live:.1%}")
        with colB2: st.metric("Over 2.5", "Over" if over25_live > 0.5 else "Under", f"{over25_live:.1%}")
        st.write(f"**Most Likely Final Score:** {most_likely_score}")
        st.write(f"**Expected Additional Goals:** Home {lambda_rem_h:.2f} | Away {lambda_rem_a:.2f}")
        gt = team_stats.get("goal_timing")
        if gt and remaining > 0:
            interval_mins = [15]*6
            remaining_mins = 90 - live_minute
            timing_probs = []
            cum = 0
            for idx, mins in enumerate(interval_mins):
                start = sum(interval_mins[:idx])
                end = start + mins
                if live_minute >= end:
                    timing_probs.append(0.0)
                else:
                    overlap = min(end, 90) - max(start, live_minute)
                    weight = overlap / remaining_mins
                    timing_probs.append(gt["prob"][idx] * weight)
            total = sum(timing_probs) or 1.0
            timing_probs = [p/total for p in timing_probs]
            fig_timing = px.bar(x=gt["intervals"], y=timing_probs, title="Next Goal Timing")
            st.plotly_chart(fig_timing, use_container_width=True)
        live_pred = {
            "goals": {"score": most_likely_score, "home_win": home_win_live, "draw": draw_live, "away_win": away_win_live,
                      "btts_yes": btts_yes_live, "btts_result": "Yes" if btts_yes_live > 0.5 else "No",
                      "over_25": over25_live, "over_under_result": "Over" if over25_live > 0.5 else "Under"},
            "xg": {"home": round(lambda_full_h, 2), "away": round(lambda_full_a, 2)},
            "current": f"{live_home_score}–{live_away_score}", "minute": live_minute
        }
        live_html = generate_pdf_html(f"{live_home} {live_home_score}–{live_away_score} {live_away}",
                                      f"Minute {live_minute}", live_pred,
                                      {live_home: get_team_logo(live_home), live_away: get_team_logo(live_away)})
        st.markdown("### Live PDF Export")
        st.markdown(f"""
        <button onclick="window.print()">Download Live PDF</button>
        <style>
        button {{ background:#cc6600; color:white; padding:10px 20px; border:none; border-radius:5px; cursor:pointer; }}
        button:hover {{ background:#bb5500; }}
        @media print {{ .no-print, button {{ display:none !important; }} }}
        </style>
        """, unsafe_allow_html=True)

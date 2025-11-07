# app.py
import streamlit as st
import pandas as pd
import numpy as np
from scipy.stats import poisson, nbinom
import io
from typing import Dict, Any, Tuple
import requests
from PIL import Image
from io import BytesIO
import plotly.express as px
import re
from datetime import datetime

# --- PDF (optional) ---
try:
    from weasyprint import HTML
    WEASYPRINT_AVAILABLE = True
except:
    WEASYPRINT_AVAILABLE = False

# ================================
# CONFIG
# ================================
st.set_page_config(page_title="Football Predictor Pro", layout="wide")
st.title("Football Match Predictor Pro")
st.markdown("""
**Advanced Prediction Suite**  
- Dixon-Coles + Negative Binomial  
- **Corners, Shots, Goal Timing**  
- Player Injury Impact  
- Live & Pre-match  
- **One-Click PDF Export**  
""")

# ================================
# LOGO & CSS
# ================================
@st.cache_data(ttl=3600)
def get_team_logo(team_name: str) -> str:
    team_clean = team_name.strip().lower().replace(" ", "_").replace(".", "").replace("'", "")
    replacements = {
        "man_utd": "Manchester_United_F.C.", "man_city": "Manchester_City_F.C.",
        "arsenal": "Arsenal_F.C.", "chelsea": "Chelsea_F.C.", "liverpool": "Liverpool_F.C."
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

print_css = """
<style>
@media print {
    .stApp > header, .stApp > footer, .stSidebar, .no-print { display: none !important; }
    .block-container { padding: 1in !important; max-width: 100% !important; }
    .print-title { font-size: 24px; font-weight: bold; text-align: center; margin-bottom: 20px; }
    .team-box { text-align: center; }
    .logo { width: 80px; height: 80px; }
    .prediction { margin: 20px 0; padding: 15px; border: 1px solid #ccc; border-radius: 8px; background: #f9f9f9; }
    .score { font-size: 20px; font-weight: bold; }
    .prob { font-size: 14px; color: #555; }
    .injury { color: #d00; font-weight: bold; }
    .stPlotlyChart { display: none; }
}
</style>
"""

# ================================
# HELPERS
# ================================
def bayesian_smoothing(observed_rate: float, league_avg: float, sample_size: int, confidence: int = 10) -> float:
    return (observed_rate * sample_size + league_avg * confidence) / (sample_size + confidence)

def negative_binomial_params(mean: float, variance: float) -> Tuple[float, float]:
    if variance <= mean: return mean, None
    p = mean / variance
    n = mean * p / (1 - p)
    return n, p

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

def apply_injury_adjustment(stats: Dict[str, Any], injuries: Dict[str, Dict[str, float]]) -> Tuple[Dict[str, Any], str]:
    adjusted = stats.copy()
    summary = ""
    for section in ["goals", "xg", "corners", "shots"]:
        if section in adjusted:
            s = adjusted[section]
            for team, players in injuries.items():
                attack_reduction = defence_reduction = 0
                for p, data in players.items():
                    if data["role"] in ["forward", "midfielder"]:
                        attack_reduction += data["impact"]
                    elif data["role"] in ["defender"]:
                        defence_reduction += data["impact"]
                attack_reduction = min(attack_reduction, 0.20)
                defence_reduction = min(defence_reduction, 0.20)
                if attack_reduction > 0:
                    s["home_attack"][team] *= (1 - attack_reduction)
                    s["away_attack"][team] *= (1 - attack_reduction)
                    summary += f"{team} Attack -{attack_reduction*100:.0f}% | "
                if defence_reduction > 0:
                    s["home_defence"][team] *= (1 - defence_reduction)
                    s["away_defence"][team] *= (1 - defence_reduction)
                    summary += f"{team} Defence -{defence_reduction*100:.0f}% | "
    return adjusted, summary.strip(" | ")

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
        if "home" in lower and "team" in lower: mapping["HomeTeam"] = col
        elif "away" in lower and "team" in lower: mapping["AwayTeam"] = col
        elif lower in ["fthg", "hgoals"]: mapping["FTHG"] = col
        elif lower in ["ftag", "agoals"]: mapping["FTAG"] = col
        elif lower in ["hc", "homecorners"]: mapping["HC"] = col
        elif lower in ["ac", "awaycorners"]: mapping["AC"] = col
        elif lower in ["hs", "homeshotsontarget"]: mapping["HS"] = col
        elif lower in ["as", "awayshotsontarget"]: mapping["AS"] = col
    return mapping

# ================================
# GOAL TIMING PARSING
# ================================
def extract_goal_minutes(df: pd.DataFrame, home_col: str, away_col: str) -> pd.DataFrame:
    goal_df = pd.DataFrame(index=df.index)
    goal_df['home_goals'] = pd.NA
    goal_df['away_goals'] = pd.NA
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
@st.cache_data(show_spinner="Training model...")
def compute_team_stats(
    _df: pd.DataFrame,
    home_col: str, away_col: str, hg_col: str, ag_col: str,
    hc_col=None, ac_col=None, hs_col=None, as_col=None,
    recency_weight: float = 2.0, min_matches: int = 3
) -> Dict[str, Any]:
    df = _df.copy()
    for col in [hg_col, ag_col, hc_col, ac_col, hs_col, as_col]:
        if col and col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')

    stats = {}
    teams = sorted(set(df[home_col]).union(df[away_col]))

    # --- GOALS ---
    ft_mask = df[hg_col].notna() & df[ag_col].notna()
    clean_ft = df[ft_mask][[home_col, away_col, hg_col, ag_col]].copy()
    if len(clean_ft) < 5: raise ValueError("Not enough matches.")
    clean_ft['weight'] = np.exp(np.linspace(-recency_weight, 0, len(clean_ft)))
    avg_home = np.average(clean_ft[hg_col], weights=clean_ft['weight'])
    avg_away = np.average(clean_ft[ag_col], weights=clean_ft['weight'])

    def weighted_mean(group, col):
        if len(group) < min_matches: return None
        return np.average(group[col], weights=group['weight'])

    home_attack = away_attack = home_defence = away_defence = {}
    for team in teams:
        home_matches = clean_ft[clean_ft[home_col] == team]
        away_matches = clean_ft[clean_ft[away_col] == team]
        n_home = len(home_matches); n_away = len(away_matches)
        if n_home >= min_matches:
            ha = weighted_mean(home_matches, hg_col) / avg_home
            hd = weighted_mean(home_matches, ag_col) / avg_away
            home_attack[team] = bayesian_smoothing(ha, 1.0, n_home)
            home_defence[team] = bayesian_smoothing(hd, 1.0, n_home)
        else:
            home_attack[team] = home_defence[team] = 1.0
        if n_away >= min_matches:
            aa = weighted_mean(away_matches, ag_col) / avg_away
            ad = weighted_mean(away_matches, hg_col) / avg_home
            away_attack[team] = bayesian_smoothing(aa, 1.0, n_away)
            away_defence[team] = bayesian_smoothing(ad, 1.0, n_away)
        else:
            away_attack[team] = away_defence[team] = 1.0

    stats["goals"] = {
        "league_avg_home": avg_home, "league_avg_away": avg_away,
        "home_attack": home_attack, "away_attack": away_attack,
        "home_defence": home_defence, "away_defence": away_defence
    }

    # --- CORNERS ---
    if hc_col and ac_col and hc_col in df.columns and ac_col in df.columns:
        c_mask = df[hc_col].notna() & df[ac_col].notna()
        clean_c = df[c_mask][[home_col, away_col, hc_col, ac_col]].copy()
        if len(clean_c) >= 5:
            clean_c['weight'] = np.exp(np.linspace(-recency_weight, 0, len(clean_c)))
            hc_mean = np.average(clean_c[hc_col], weights=clean_c['weight'])
            ac_mean = np.average(clean_c[ac_col], weights=clean_c['weight'])
            hc_var = np.var(clean_c[hc_col]) * len(clean_c) / (len(clean_c) - 1)
            ac_var = np.var(clean_c[ac_col]) * len(clean_c) / (len(clean_c) - 1)
            use_nb = hc_var > hc_mean and ac_var > ac_mean
            corner_stats = {
                "league_avg_home": hc_mean, "league_avg_away": ac_mean,
                "home_attack": {}, "away_attack": {}, "home_defence": {}, "away_defence": {},
                "use_negbinom": use_nb
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

    # --- SHOTS ---
    if hs_col and as_col and hs_col in df.columns and as_col in df.columns:
        s_mask = df[hs_col].notna() & df[as_col].notna()
        clean_s = df[s_mask][[home_col, away_col, hs_col, as_col]].copy()
        if len(clean_s) >= 5:
            clean_s['weight'] = np.exp(np.linspace(-recency_weight, 0, len(clean_s)))
            hs_mean = np.average(clean_s[hs_col], weights=clean_s['weight'])
            as_mean = np.average(clean_s[as_col], weights=clean_s['weight'])
            shot_stats = {
                "league_avg_home": hs_mean, "league_avg_away": as_mean,
                "home_attack": {}, "away_attack": {}, "home_defence": {}, "away_defence": {}
            }
            for team in teams:
                home_s = clean_s[clean_s[home_col] == team]
                away_s = clean_s[clean_s[away_col] == team]
                if len(home_s) >= min_matches:
                    shot_stats["home_attack"][team] = bayesian_smoothing(
                        weighted_mean(home_s, hs_col) / hs_mean, 1.0, len(home_s)
                    )
                    shot_stats["home_defence"][team] = bayesian_smoothing(
                        weighted_mean(home_s, as_col) / as_mean, 1.0, len(home_s)
                    )
                else:
                    shot_stats["home_attack"][team] = shot_stats["home_defence"][team] = 1.0
                if len(away_s) >= min_matches:
                    shot_stats["away_attack"][team] = bayesian_smoothing(
                        weighted_mean(away_s, as_col) / as_mean, 1.0, len(away_s)
                    )
                    shot_stats["away_defence"][team] = bayesian_smoothing(
                        weighted_mean(away_s, hs_col) / hs_mean, 1.0, len(away_s)
                    )
                else:
                    shot_stats["away_attack"][team] = shot_stats["away_defence"][team] = 1.0
            stats["shots"] = shot_stats

    # --- GOAL TIMING ---
    intervals = ["1–15", "16–30", "31–45", "46–60", "61–75", "76–90"]
    goals_per_interval = {i: 0 for i in intervals}
    minute_df = extract_goal_minutes(df, home_col, away_col)
    if minute_df is not None:
        all_goals = []
        for _, row in minute_df.iterrows():
            all_goals.extend([m for m in row['home_goals'] if 1 <= m <= 90])
            all_goals.extend([m for m in row['away_goals'] if 1 <= m <= 90])
        for m in all_goals:
            for idx, (s, e) in enumerate([(1,15),(16,30),(31,45),(46,60),(61,75),(76,90)]):
                if s <= m <= e:
                    goals_per_interval[intervals[idx]] += 1
    total = sum(goals_per_interval.values())
    if total > 0:
        probs = [g / total for g in goals_per_interval.values()]
        stats["goal_timing"] = {"intervals": intervals, "prob": probs}

    return stats

# ================================
# PREDICT MATCH
# ================================
@st.cache_data(show_spinner=False)
def predict_match(home: str, away: str, stats: Dict[str, Any],
                  _df: pd.DataFrame = None, home_col: str = None,
                  away_col: str = None, hg_col: str = None, ag_col: str = None,
                  injuries: Dict = None) -> Dict[str, Any]:
    if injuries:
        stats, injury_summary = apply_injury_adjustment(stats, injuries)
    else:
        injury_summary = ""

    max_g = 10; max_c = 15
    predictions = {
        "goals": {"score": "N/A", "home_win": 0, "draw": 0, "away_win": 0, "btts_yes": 0, "over_25": 0},
        "corners": {"home": 0, "away": 0, "total": 0},
        "shots": {"home": 0, "away": 0},
        "goal_timing": {"intervals": [], "prob": []},
        "injury_summary": injury_summary
    }

    # --- GOALS ---
    g = stats.get("goals", {})
    if g:
        l_home = g["league_avg_home"]; l_away = g["league_avg_away"]
        att_h = g["home_attack"].get(home, 1.0); def_a = g["away_defence"].get(away, 1.0)
        att_a = g["away_attack"].get(away, 1.0); def_h = g["home_defence"].get(home, 1.0)
        lambda_h = att_h * def_a * l_home
        lambda_a = att_a * def_h * l_away
        rho = 0.0
        if _df is not None and hg_col and ag_col:
            ft = _df[[hg_col, ag_col]].dropna()
            if len(ft) > 0:
                p00 = (ft[hg_col] == 0).mean() * (ft[ag_col] == 0).mean()
                p01 = (ft[hg_col] == 0).mean() * (ft[ag_col] == 1).mean()
                p10 = (ft[hg_col] == 1).mean() * (ft[ag_col] == 0).mean()
                p11 = (ft[hg_col] == 1).mean() * (ft[ag_col] == 1).mean()
                rho = 1 - (p00 * p11) / (p01 * p10) if p01 * p10 > 0 else 0.0
                rho = np.clip(rho, -0.3, 0.3)
        prob_matrix = np.zeros((max_g + 1, max_g + 1))
        for h in range(max_g + 1):
            for a in range(max_g + 1):
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
        predictions["goals"]["home_win"] = (prob_matrix[1:, :].sum() - prob_matrix.diagonal()[1:].sum())
        predictions["goals"]["away_win"] = (prob_matrix[:, 1:].sum() - prob_matrix.diagonal()[1:].sum())
        predictions["goals"]["draw"] = prob_matrix.diagonal().sum()
        predictions["goals"]["btts_yes"] = (prob_matrix[1:, 1:]).sum()
        predictions["goals"]["over_25"] = (prob_matrix[3:, :].sum() + prob_matrix[:, 3:].sum() - prob_matrix[3:, 3:].sum())

    # --- CORNERS ---
    c = stats.get("corners")
    if c:
        lhc = c["league_avg_home"]; lac = c["league_avg_away"]
        att_hc = c["home_attack"].get(home, 1.0); def_ac = c["away_defence"].get(away, 1.0)
        att_ac = c["away_attack"].get(away, 1.0); def_hc = c["home_defence"].get(home, 1.0)
        mu_hc = att_hc * def_ac * lhc
        mu_ac = att_ac * def_hc * lac
        if c.get("use_negbinom"):
            n_hc, p_hc = negative_binomial_params(mu_hc, c["home_variance"] if "home_variance" in c else mu_hc*1.5)
            n_ac, p_ac = negative_binomial_params(mu_ac, c["away_variance"] if "away_variance" in c else mu_ac*1.5)
            hc_probs = nbinom.pmf(np.arange(max_c + 1), n_hc, p_hc) if p_hc else poisson.pmf(np.arange(max_c + 1), mu_hc)
            ac_probs = nbinom.pmf(np.arange(max_c + 1), n_ac, p_ac) if p_ac else poisson.pmf(np.arange(max_c + 1), mu_ac)
        else:
            hc_probs = poisson.pmf(np.arange(max_c + 1), mu_hc)
            ac_probs = poisson.pmf(np.arange(max_c + 1), mu_ac)
        predictions["corners"]["home"] = int(np.argmax(hc_probs))
        predictions["corners"]["away"] = int(np.argmax(ac_probs))
        predictions["corners"]["total"] = predictions["corners"]["home"] + predictions["corners"]["away"]

    # --- SHOTS ---
    s = stats.get("shots")
    if s:
        att_hs = s["home_attack"].get(home, 1.0); def_as = s["away_defence"].get(away, 1.0)
        att_as = s["away_attack"].get(away, 1.0); def_hs = s["home_defence"].get(home, 1.0)
        mu_hs = att_hs * def_as * s["league_avg_home"]
        mu_as = att_as * def_hs * s["league_avg_away"]
        predictions["shots"]["home"] = round(mu_hs, 1)
        predictions["shots"]["away"] = round(mu_as, 1)

    # --- GOAL TIMING ---
    gt = stats.get("goal_timing")
    if gt:
        predictions["goal_timing"] = gt

    return {"predictions": predictions}

# ================================
# PDF EXPORT
# ================================
def generate_pdf_html(home, away, pred, logos):
    p = pred["predictions"]
    injury = f'<div class="injury">{p.get("injury_summary","")}</div>' if p.get("injury_summary") else ""
    html = f"""
    <html><head><style>{print_css}</style></head><body>
    <div class="print-title">Prediction: {home} vs {away}</div>
    <div style="display:flex; justify-content:space-around;">
        <div class="team-box"><img src="{logos.get(home,'')}" class="logo" onerror="this.style.display='none'"/><br><strong>{home}</strong></div>
        <div style="font-size:36px; align-self:center;">VS</div>
        <div class="team-box"><img src="{logos.get(away,'')}" class="logo" onerror="this.style.display='none'"/><br><strong>{away}</strong></div>
    </div>
    <div class="prediction">
        <div class="score">Score: <strong>{p['goals']['score']}</strong></div>
        <div>Home Win: {p['goals']['home_win']:.1%} | Draw: {p['goals']['draw']:.1%} | Away Win: {p['goals']['away_win']:.1%}</div>
        <div>BTTS: {p['goals']['btts_yes']:.1%} | Over 2.5: {p['goals']['over_25']:.1%}</div>
        <div><strong>Corners:</strong> {home} {p['corners']['home']} | {away} {p['corners']['away']} (Total: {p['corners']['total']})</div>
        <div><strong>Shots on Target:</strong> {home} {p['shots']['home']} | {away} {p['shots']['away']}</div>
        {injury}
    </div>
    <div style="margin-top:30px; font-size:12px; color:#555;">Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}</div>
    </body></html>
    """
    return html

# ================================
# MAIN APP
# ================================
st.sidebar.header("Upload Match Data")
uploaded_file = st.sidebar.file_uploader("CSV File", type=["csv"])

if uploaded_file is not None:
    df = load_csv(uploaded_file.read())
    if df.empty:
        st.error("Empty CSV.")
    else:
        st.success(f"Loaded {len(df):,} matches")
        mapping = detect_columns(df)

        st.sidebar.subheader("Confirm Column Mapping")
        col_map = {}
        for label in ["HomeTeam", "AwayTeam", "FTHG", "FTAG", "HC", "AC", "HS", "AS"]:
            detected = mapping.get(label)
            options = [""] + [c for c in df.columns if c.lower() != "date"]
            default_idx = options.index(detected) if detected in options else 0
            col_map[label] = st.sidebar.selectbox(f"**{label}**", options=options, index=default_idx)

        missing = [r for r in ["HomeTeam", "AwayTeam", "FTHG", "FTAG"] if not col_map[r]]
        if missing:
            st.error(f"Map: {', '.join(missing)}")
            st.stop()

        with st.spinner("Training..."):
            team_stats = compute_team_stats(
                _df=df,
                home_col=col_map["HomeTeam"], away_col=col_map["AwayTeam"],
                hg_col=col_map["FTHG"], ag_col=col_map["FTAG"],
                hc_col=col_map.get("HC"), ac_col=col_map.get("AC"),
                hs_col=col_map.get("HS"), as_col=col_map.get("AS"),
                recency_weight=st.sidebar.slider("Recency", 0.5, 5.0, 2.0, 0.1),
                min_matches=st.sidebar.number_input("Min matches", 1, 20, 3)
            )

        rename_dict = {v: k for k, v in col_map.items() if v}
        df_clean = df.rename(columns=rename_dict).copy()
        teams = sorted(set(df_clean["HomeTeam"]).union(df_clean["AwayTeam"]))

        # Injury Input
        injury_input = st.sidebar.text_area("Injuries", placeholder="Arsenal: Saka (role:forward, impact:15%)", height=100)
        injuries = parse_injuries(injury_input)

        # Pre-match
        st.markdown("---")
        st.subheader("Pre-Match Prediction")
        col1, col2 = st.columns(2)
        home_team = col1.selectbox("Home", teams)
        away_team = col2.selectbox("Away", teams)

        if st.button("Predict"):
            pred = predict_match(home_team, away_team, team_stats, df,
                                 col_map["HomeTeam"], col_map["AwayTeam"],
                                 col_map["FTHG"], col_map["FTAG"], injuries)
            p = pred["predictions"]

            st.markdown(f"### **{home_team} vs {away_team}**")
            colA, colB, colC = st.columns([1,2,1])
            with colA:
                logo = get_team_logo(home_team)
                if logo: st.image(load_image(logo), width=80)
                st.write(f"**{home_team}**")
            with colC:
                logo = get_team_logo(away_team)
                if logo: st.image(load_image(logo), width=80)
                st.write(f"**{away_team}**")
            with colB:
                st.markdown(f"<h2 style='text-align:center'>{p['goals']['score']}</h2>", unsafe_allow_html=True)

            # Win/Draw
            colW1, colW2, colW3 = st.columns(3)
            colW1.metric("Home Win", f"{p['goals']['home_win']:.1%}")
            colW2.metric("Draw", f"{p['goals']['draw']:.1%}")
            colW3.metric("Away Win", f"{p['goals']['away_win']:.1%}")

            # BTTS & Over 2.5
            colB1, colB2 = st.columns(2)
            colB1.metric("BTTS", f"{p['goals']['btts_yes']:.1%}")
            colB2.metric("Over 2.5", f"{p['goals']['over_25']:.1%}")

            # Corners & Shots
            st.markdown("#### Expected Stats")
            colC1, colC2 = st.columns(2)
            with colC1:
                st.write(f"**Corners** – {home_team}: **{p['corners']['home']}** | {away_team}: **{p['corners']['away']}** (Total: {p['corners']['total']})")
            with colC2:
                st.write(f"**Shots on Target** – {home_team}: **{p['shots']['home']}** | {away_team}: **{p['shots']['away']}**")

            # Goal Timing
            if p["goal_timing"]["intervals"]:
                fig = px.bar(x=p["goal_timing"]["intervals"], y=p["goal_timing"]["prob"],
                             labels={"x": "Interval", "y": "Probability"}, title="Goal Timing Distribution")
                st.plotly_chart(fig, use_container_width=True)

            # Injuries
            if p["injury_summary"]:
                st.markdown(f"**Injury Adjustments:** <span style='color:red'>{p['injury_summary']}</span>", unsafe_allow_html=True)

            # PDF Export
            logos = {home_team: get_team_logo(home_team), away_team: get_team_logo(away_team)}
            pdf_html = generate_pdf_html(home_team, away_team, pred, logos)
            st.markdown("### Export PDF")
            st.markdown(f"<button onclick='window.print()'>Download PDF</button>", unsafe_allow_html=True)
            st.markdown("<small>Click → Print → Save as PDF</small>", unsafe_allow_html=True)

else:
    st.info("Upload a CSV to start.")

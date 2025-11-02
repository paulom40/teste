# app.py
import streamlit as st
import pandas as pd
import numpy as np
from scipy.stats import poisson
import io
from typing import Dict, Any

# ================================
# CONFIG & PAGE SETUP
# ================================
st.set_page_config(page_title="Football Predictor", layout="wide")
st.title("Football Match Outcome Predictor")
st.markdown("""
Upload any **football-data.co.uk** CSV and get **instant predictions** using **Poisson modeling**.

**Now predicts:**
- **Goals** (FTHG/FTAG)
- **Corners** (HC/AC)
- **Shots on Target** (HS/AS)
- **Expected Goals (xG)** (HxG/AxG)

""")

# ================================
# HELPER FUNCTION
# ================================
def _safe_index(df: pd.DataFrame, col: str):
    return df.columns.get_loc(col) if col in df.columns else 0

# ================================
# UTILS: DATA LOADER
# ================================
@st.cache_data(show_spinner="Loading CSV...")
def load_csv(uploaded_file_bytes: bytes) -> pd.DataFrame:
    try:
        return pd.read_csv(io.BytesIO(uploaded_file_bytes), encoding="utf-8")
    except UnicodeDecodeError:
        return pd.read_csv(io.BytesIO(uploaded_file_bytes), encoding="latin1")

@st.cache_data(show_spinner=False)
def detect_columns(df: pd.DataFrame) -> Dict[str, str]:
    mapping = {}
    for col in df.columns:
        lower = col.lower().replace(" ", "")
        if "date" in lower:
            mapping["Date"] = col
        elif "home" in lower and "team" in lower:
            mapping["HomeTeam"] = col
        elif "away" in lower and "team" in lower:
            mapping["AwayTeam"] = col
        elif lower in ["fthg", "hgoals", "homegoals"]:
            mapping["FTHG"] = col
        elif lower in ["ftag", "agoals", "awaygoals"]:
            mapping["FTAG"] = col
        elif lower in ["hcr", "homecorners", "hc"]:
            mapping["HC"] = col
        elif lower in ["acr", "awaycorners", "ac"]:
            mapping["AC"] = col
        elif lower in ["hs", "homeshotsontarget"]:
            mapping["HS"] = col
        elif lower in ["as", "awayshotsontarget"]:
            mapping["AS"] = col
        elif lower in ["hxg", "home_xg", "homeexpectedgoals"]:
            mapping["HxG"] = col
        elif lower in ["axg", "away_xg", "awayexpectedgoals"]:
            mapping["AxG"] = col
    return mapping

# ================================
# UTILS: MODEL
# ================================
@st.cache_data(show_spinner="Training Poisson model...")
def compute_team_stats(
    _df: pd.DataFrame,
    home_col: str,
    away_col: str,
    hg_col: str,
    ag_col: str,
    hc_col: str = None,
    ac_col: str = None,
    hs_col: str = None,
    as_col: str = None,
    hxg_col: str = None,
    axg_col: str = None
) -> Dict[str, Any]:
    stats = {}

    # --- Goals ---
    league_avg_home_goals = _df[hg_col].mean()
    league_avg_away_goals = _df[ag_col].mean()
    stats["goals"] = {
        "league_avg_home": league_avg_home_goals,
        "league_avg_away": league_avg_away_goals,
        "home_attack": (_df.groupby(home_col)[hg_col].mean() / league_avg_home_goals).to_dict(),
        "away_attack": (_df.groupby(away_col)[ag_col].mean() / league_avg_away_goals).to_dict(),
        "home_defence": (_df.groupby(home_col)[ag_col].mean() / league_avg_away_goals).to_dict(),
        "away_defence": (_df.groupby(away_col)[hg_col].mean() / league_avg_home_goals).to_dict(),
    }

    # --- Corners ---
    if hc_col and ac_col and hc_col in _df.columns and ac_col in _df.columns:
        stats["corners"] = {
            "league_avg_home": _df[hc_col].mean(),
            "league_avg_away": _df[ac_col].mean(),
            "home_attack": (_df.groupby(home_col)[hc_col].mean() / _df[hc_col].mean()).to_dict(),
            "away_attack": (_df.groupby(away_col)[ac_col].mean() / _df[ac_col].mean()).to_dict(),
            "home_defence": (_df.groupby(home_col)[ac_col].mean() / _df[ac_col].mean()).to_dict(),
            "away_defence": (_df.groupby(away_col)[hc_col].mean() / _df[hc_col].mean()).to_dict(),
        }

    # --- Shots on Target ---
    if hs_col and as_col and hs_col in _df.columns and as_col in _df.columns:
        stats["shots"] = {
            "league_avg_home": _df[hs_col].mean(),
            "league_avg_away": _df[as_col].mean(),
            "home_attack": (_df.groupby(home_col)[hs_col].mean() / _df[hs_col].mean()).to_dict(),
            "away_attack": (_df.groupby(away_col)[as_col].mean() / _df[as_col].mean()).to_dict(),
            "home_defence": (_df.groupby(home_col)[as_col].mean() / _df[as_col].mean()).to_dict(),
            "away_defence": (_df.groupby(away_col)[hs_col].mean() / _df[hs_col].mean()).to_dict(),
        }

    # --- Expected Goals (xG) ---
    if hxg_col and axg_col and hxg_col in _df.columns and axg_col in _df.columns:
        stats["xg"] = {
            "league_avg_home": _df[hxg_col].mean(),
            "league_avg_away": _df[axg_col].mean(),
            "home_attack": (_df.groupby(home_col)[hxg_col].mean() / _df[hxg_col].mean()).to_dict(),
            "away_attack": (_df.groupby(away_col)[axg_col].mean() / _df[axg_col].mean()).to_dict(),
            "home_defence": (_df.groupby(home_col)[axg_col].mean() / _df[axg_col].mean()).to_dict(),
            "away_defence": (_df.groupby(away_col)[hxg_col].mean() / _df[hxg_col].mean()).to_dict(),
        }

    return stats

@st.cache_data(show_spinner=False)
def predict_match(
    home: str,
    away: str,
    stats: Dict[str, Any],
    max_goals: int = 10
) -> Dict[str, Any]:
    predictions = {}

    # --- Goals ---
    g = stats["goals"]
    lambda_home = g["home_attack"].get(home, 1.0) * g["away_defence"].get(away, 1.0) * g["league_avg_home"]
    lambda_away = g["away_attack"].get(away, 1.0) * g["home_defence"].get(home, 1.0) * g["league_avg_away"]

    home_probs = poisson.pmf(np.arange(max_goals + 1), lambda_home)
    away_probs = poisson.pmf(np.arange(max_goals + 1), lambda_away)

    prob_home = prob_draw = prob_away = 0.0
    best_score = (0, 0)
    best_prob = 0.0

    for h in range(max_goals + 1):
        for a in range(max_goals + 1):
            p = home_probs[h] * away_probs[a]
            if h > a: prob_home += p
            elif h == a: prob_draw += p
            else: prob_away += p
            if p > best_prob:
                best_prob = p
                best_score = (h, a)

    result = "H" if prob_home > max(prob_draw, prob_away) else "D" if prob_draw > max(prob_home, prob_away) else "A"
    predictions["goals"] = {
        "score": f"{best_score[0]}-{best_score[1]}",
        "home_win": prob_home,
        "draw": prob_draw,
        "away_win": prob_away,
        "result": result
    }

    # --- Generic Poisson Predictor (for corners, shots, xG) ---
    def predict_stat(stat_name, over_under_threshold=None):
        if stat_name not in stats:
            return None
        s = stats[stat_name]
        lambda_home_s = s["home_attack"].get(home, 1.0) * s["away_defence"].get(away, 1.0) * s["league_avg_home"]
        lambda_away_s = s["away_attack"].get(away, 1.0) * s["home_defence"].get(home, 1.0) * s["league_avg_away"]

        probs_h = poisson.pmf(np.arange(max_goals + 1), lambda_home_s)
        probs_a = poisson.pmf(np.arange(max_goals + 1), lambda_away_s)

        best = (0, 0)
        best_p = 0.0
        over_prob = 0.0

        for h in range(max_goals + 1):
            for a in range(max_goals + 1):
                p = probs_h[h] * probs_a[a]
                if p > best_p:
                    best_p = p
                    best = (h, a)
                if over_under_threshold is not None and h + a > over_under_threshold:
                    over_prob += p

        res = {"score": f"{best[0]}-{best[1]}"}
        if over_under_threshold is not None:
            res["over"] = over_prob
            res["under"] = 1 - over_prob
        return res

    # --- Corners ---
    corners_pred = predict_stat("corners", over_under_threshold=10.5)
    if corners_pred:
        predictions["corners"] = corners_pred

    # --- Shots on Target ---
    shots_pred = predict_stat("shots", over_under_threshold=20.5)
    if shots_pred:
        predictions["shots"] = shots_pred

    # --- Expected Goals (xG) ---
    xg_pred = predict_stat("xg", over_under_threshold=2.5)
    if xg_pred:
        predictions["xg"] = xg_pred

    return predictions

# ================================
# MAIN UI
# ================================
uploaded_file = st.file_uploader("Upload League CSV", type=["csv"])

if uploaded_file is not None:
    file_bytes = uploaded_file.getvalue()
    df = load_csv(file_bytes)
    st.success(f"Loaded {len(df):,} matches.")

    with st.expander("Preview Data", expanded=False):
        st.dataframe(df.head(10))

    st.subheader("Map Required Columns")
    guessed = detect_columns(df)
    cols = st.columns(5)
    date_col = cols[0].selectbox("Date", df.columns, index=_safe_index(df, guessed.get("Date")))
    home_col = cols[1].selectbox("Home Team", df.columns, index=_safe_index(df, guessed.get("HomeTeam")))
    away_col = cols[2].selectbox("Away Team", df.columns, index=_safe_index(df, guessed.get("AwayTeam")))
    hg_col   = cols[3].selectbox("Home Goals (FTHG)", df.columns, index=_safe_index(df, guessed.get("FTHG")))
    ag_col   = cols[4].selectbox("Away Goals (FTAG)", df.columns, index=_safe_index(df, guessed.get("FTAG")))

    st.subheader("Optional: Corners, Shots, xG")
    opt_cols = st.columns(6)
    hc_col = opt_cols[0].selectbox("Home Corners (HC)", [""] + list(df.columns), index=0)
    ac_col = opt_cols[1].selectbox("Away Corners (AC)", [""] + list(df.columns), index=0)
    hs_col = opt_cols[2].selectbox("Home Shots (HS)", [""] + list(df.columns), index=0)
    as_col = opt_cols[3].selectbox("Away Shots (AS)", [""] + list(df.columns), index=0)
    hxg_col = opt_cols[4].selectbox("Home xG (HxG)", [""] + list(df.columns), index=0)
    axg_col = opt_cols[5].selectbox("Away xG (AxG)", [""] + list(df.columns), index=0)

    if st.button("Train Model"):
        stats = compute_team_stats(
            df, home_col, away_col, hg_col, ag_col,
            hc_col or None, ac_col or None,
            hs_col or None, as_col or None,
            hxg_col or None, axg_col or None
        )
        teams = sorted(set(df[home_col]).union(df[away_col]))
        st.session_state.stats = stats
        st.session_state.teams = teams
        st.success("Model trained!")

    if "stats" in st.session_state:
        st.subheader("Predict Match")
        c1, c2 = st.columns(2)
        home_team = c1.selectbox("Home Team", st.session_state.teams, key="ph")
        away_team = c2.selectbox("Away Team", st.session_state.teams, key="pa")

        if home_team == away_team:
            st.error("Select different teams.")
        elif st.button("Predict"):
            pred = predict_match(home_team, away_team, st.session_state.stats)

            st.markdown(f"### **{home_team} vs {away_team}**")

            # Goals
            g = pred["goals"]
            st.markdown(f"""
            #### Goals
            **Score:** `{g['score']}` | **Result:** **{g['result']}**  
            - Home Win: `{g['home_win']:.1%}`  
            - Draw: `{g['draw']:.1%}`  
            - Away Win: `{g['away_win']:.1%}`
            """)

            # Corners
            if "corners" in pred:
                c = pred["corners"]
                st.markdown(f"""
                #### Corners
                **Most Likely:** `{c['score']}`  
                - **Over 10.5:** `{c['over']:.1%}`  
                - **Under 10.5:** `{c['under']:.1%}`
                """)

            # Shots
            if "shots" in pred:
                s = pred["shots"]
                st.markdown(f"""
                #### Shots on Target
                **Most Likely:** `{s['score']}`  
                - **Over 20.5:** `{s['over']:.1%}`  
                - **Under 20.5:** `{s['under']:.1%}`
                """)

            # xG
            if "xg" in pred:
                x = pred["xg"]
                st.markdown(f"""
                #### Expected Goals (xG)
                **Most Likely:** `{x['score']}`  
                - **Over 2.5 xG:** `{x['over']:.1%}`  
                - **Under 2.5 xG:** `{x['under']:.1%}`
                """)

# Clear Cache
if st.button("Clear Cache"):
    st.cache_data.clear()
    st.success("Cache cleared!")

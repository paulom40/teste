# app.py
import streamlit as st
import pandas as pd
import numpy as np
from scipy.stats import poisson
import io
from typing import Dict, Any

# -------------------------------------------------
# PAGE CONFIG
# -------------------------------------------------
st.set_page_config(page_title="Football Predictor", layout="wide")
st.title("Football Match Outcome Predictor")
st.markdown("""
**Poisson-based predictions** from any **football-data.co.uk** CSV.

**Required:** Home/away teams + goals (FTHG/FTAG)  
**Optional:** Corners (HC/AC), Shots on target (HS/AS), Expected Goals (HxG/AxG)
""")

# -------------------------------------------------
# HELPERS
# -------------------------------------------------
def _safe_index(df: pd.DataFrame, col: str):
    return df.columns.get_loc(col) if col in df.columns else 0

# -------------------------------------------------
# DATA LOADER
# -------------------------------------------------
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
        if "date" in lower:                     mapping["Date"] = col
        elif "home" in lower and "team" in lower: mapping["HomeTeam"] = col
        elif "away" in lower and "team" in lower: mapping["AwayTeam"] = col
        elif lower in ["fthg", "hgoals"]:        mapping["FTHG"] = col
        elif lower in ["ftag", "agoals"]:        mapping["FTAG"] = col
        elif lower in ["hc", "homecorners"]:     mapping["HC"] = col
        elif lower in ["ac", "awaycorners"]:     mapping["AC"] = col
        elif lower in ["hs", "homeshotsontarget"]: mapping["HS"] = col
        elif lower in ["as", "awayshotsontarget"]: mapping["AS"] = col
        elif lower in ["hxg", "home_xg"]:        mapping["HxG"] = col
        elif lower in ["axg", "away_xg"]:        mapping["AxG"] = col
    return mapping

# -------------------------------------------------
# MODEL
# -------------------------------------------------
@st.cache_data(show_spinner="Training model...")
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
    stats: Dict[str, Any] = {}

    # ----- REQUIRED: GOALS -----
    required = [home_col, away_col, hg_col, ag_col]
    missing = [c for c in required if c not in _df.columns]
    if missing:
        raise ValueError(f"Missing required column(s): {', '.join(missing)}")

    clean = _df[[home_col, away_col, hg_col, ag_col]].dropna()
    if clean.empty:
        raise ValueError("No valid matches after removing NaN from goal columns.")

    avg_home = clean[hg_col].mean()
    avg_away = clean[ag_col].mean()
    if avg_home == 0 or avg_away == 0:
        raise ValueError("League average goals are zero – check the data.")

    stats["goals"] = {
        "league_avg_home": avg_home,
        "league_avg_away": avg_away,
        "home_attack": (clean.groupby(home_col)[hg_col].mean() / avg_home).fillna(1.0).to_dict(),
        "away_attack": (clean.groupby(away_col)[ag_col].mean() / avg_away).fillna(1.0).to_dict(),
        "home_defence": (clean.groupby(home_col)[ag_col].mean() / avg_away).fillna(1.0).to_dict(),
        "away_defence": (clean.groupby(away_col)[hg_col].mean() / avg_home).fillna(1.0).to_dict(),
    }

    # ----- OPTIONAL STATS (generic helper) -----
    def add_optional(name: str, h_col: str, a_col: str):
        if h_col and a_col and h_col in _df.columns and a_col in _df.columns:
            sub = _df[[home_col, away_col, h_col, a_col]].dropna()
            if not sub.empty:
                avg_h = sub[h_col].mean()
                avg_a = sub[a_col].mean()
                if avg_h > 0 and avg_a > 0:
                    stats[name] = {
                        "league_avg_home": avg_h,
                        "league_avg_away": avg_a,
                        "home_attack": (sub.groupby(home_col)[h_col].mean() / avg_h).fillna(1.0).to_dict(),
                        "away_attack": (sub.groupby(away_col)[a_col].mean() / avg_a).fillna(1.0).to_dict(),
                        "home_defence": (sub.groupby(home_col)[a_col].mean() / avg_a).fillna(1.0).to_dict(),
                        "away_defence": (sub.groupby(away_col)[h_col].mean() / avg_h).fillna(1.0).to_dict(),
                    }

    add_optional("corners", hc_col, ac_col)
    add_optional("shots",   hs_col, as_col)
    add_optional("xg",      hxg_col, axg_col)

    return stats


@st.cache_data(show_spinner=False)
def predict_match(home: str, away: str, stats: Dict[str, Any]) -> Dict[str, Any]:
    predictions: Dict[str, Any] = {}

    # ----- GOALS (always present) -----
    g = stats["goals"]
    lambda_home = g["home_attack"].get(home, 1.0) * g["away_defence"].get(away, 1.0) * g["league_avg_home"]
    lambda_away = g["away_attack"].get(away, 1.0) * g["home_defence"].get(home, 1.0) * g["league_avg_away"]

    max_g = 10
    hp = poisson.pmf(np.arange(max_g + 1), lambda_home)
    ap = poisson.pmf(np.arange(max_g + 1), lambda_away)

    prob_h = prob_d = prob_a = 0.0
    best = (0, 0)
    best_p = 0.0

    for h in range(max_g + 1):
        for a in range(max_g + 1):
            p = hp[h] * ap[a]
            if h > a:   prob_h += p
            elif h == a: prob_d += p
            else:       prob_a += p
            if p > best_p:
                best_p = p
                best = (h, a)

    result = "H" if prob_h > max(prob_d, prob_a) else "D" if prob_d > max(prob_h, prob_a) else "A"
    predictions["goals"] = {
        "score": f"{best[0]}-{best[1]}",
        "home_win": prob_h,
        "draw": prob_d,
        "away_win": prob_a,
        "result": result
    }

    # ----- OPTIONAL STATS (generic) -----
    def predict_optional(name: str, threshold: float = None, fmt_float: bool = False):
        if name not in stats:
            return None
        s = stats[name]
        lh = s["home_attack"].get(home, 1.0) * s["away_defence"].get(away, 1.0) * s["league_avg_home"]
        la = s["away_attack"].get(away, 1.0) * s["home_defence"].get(home, 1.0) * s["league_avg_away"]

        probs_h = poisson.pmf(np.arange(max_g + 1), lh)
        probs_a = poisson.pmf(np.arange(max_g + 1), la)

        best_h = best_a = 0
        best_p = 0.0
        over = 0.0

        for h in range(max_g + 1):
            for a in range(max_g + 1):
                p = probs_h[h] * probs_a[a]
                if p > best_p:
                    best_p = p
                    best_h, best_a = h, a
                if threshold and h + a > threshold:
                    over += p

        score = f"{best_h:.1f}-{best_a:.1f}" if fmt_float else f"{best_h}-{best_a}"
        res = {"score": score}
        if threshold is not None:
            res["over"] = over
            res["under"] = 1 - over
        return res

    predictions["corners"] = predict_optional("corners", 10.5)
    predictions["shots"]   = predict_optional("shots",   20.5)
    predictions["xg"]      = predict_optional("xg",      2.5, fmt_float=True)

    # Keep only keys that have a value
    return {k: v for k, v in predictions.items() if v is not None}

# -------------------------------------------------
# UI
# -------------------------------------------------
uploaded_file = st.file_uploader("Upload CSV", type=["csv"])

if uploaded_file:
    df = load_csv(uploaded_file.getvalue())
    st.success(f"Loaded {len(df)} rows.")

    with st.expander("Preview"):
        st.dataframe(df.head(10))

    # ---- REQUIRED COLUMNS ----
    st.subheader("Required Columns")
    guessed = detect_columns(df)
    c1, c2, c3, c4, c5 = st.columns(5)
    date_col = c1.selectbox("Date", df.columns, index=_safe_index(df, guessed.get("Date")))
    home_col = c2.selectbox("Home Team", df.columns, index=_safe_index(df, guessed.get("HomeTeam")))
    away_col = c3.selectbox("Away Team", df.columns, index=_safe_index(df, guessed.get("AwayTeam")))
    hg_col   = c4.selectbox("Home Goals (FTHG)", df.columns, index=_safe_index(df, guessed.get("FTHG")))
    ag_col   = c5.selectbox("Away Goals (FTAG)", df.columns, index=_safe_index(df, guessed.get("FTAG")))

    # ---- OPTIONAL COLUMNS ----
    st.subheader("Optional Columns")
    o1, o2, o3, o4, o5, o6 = st.columns(6)
    hc_col  = o1.selectbox("Home Corners (HC)", [""] + list(df.columns))
    ac_col  = o2.selectbox("Away Corners (AC)", [""] + list(df.columns))
    hs_col  = o3.selectbox("Home Shots (HS)",   [""] + list(df.columns))
    as_col  = o4.selectbox("Away Shots (AS)",   [""] + list(df.columns))
    hxg_col = o5.selectbox("Home xG (HxG)",     [""] + list(df.columns))
    axg_col = o6.selectbox("Away xG (AxG)",     [""] + list(df.columns))

    # ---- TRAIN ----
    if st.button("Train Model"):
        try:
            stats = compute_team_stats(
                df, home_col, away_col, hg_col, ag_col,
                hc_col or None, ac_col or None,
                hs_col or None, as_col or None,
                hxg_col or None, axg_col or None
            )
            teams = sorted(set(df[home_col]).union(df[away_col]))
            st.session_state.stats = stats
            st.session_state.teams = teams
            st.success("Model trained successfully!")
        except Exception as e:
            st.error(f"Training failed: {e}")

    # ---- PREDICT ----
    if st.session_state.get("stats") and st.session_state.get("teams"):
        st.subheader("Predict a Match")
        t1, t2 = st.columns(2)
        home_team = t1.selectbox("Home Team", st.session_state.teams, key="ph")
        away_team = t2.selectbox("Away Team", st.session_state.teams, key="pa")

        if home_team == away_team:
            st.error("Select two different teams.")
        elif st.button("Predict"):
            try:
                pred = predict_match(home_team, away_team, st.session_state.stats)
                st.markdown(f"### {home_team} vs {away_team}")

                # Goals (always shown)
                g = pred["goals"]
                st.markdown(f"""
                #### Goals
                **{g['score']}** → **{g['result']}**  
                H: `{g['home_win']:.1%}` | D: `{g['draw']:.1%}` | A: `{g['away_win']:.1%}`
                """)

                # Optional sections
                for name, label, thresh in [
                    ("corners", "Corners", 10.5),
                    ("shots",   "Shots on Target", 20.5),
                    ("xg",      "Expected Goals (xG)", 2.5)
                ]:
                    if name in pred:
                        p = pred[name]
                        st.markdown(f"""
                        #### {label}
                        **{p['score']}**  
                        Over {thresh}: `{p['over']:.1%}` | Under {thresh}: `{p['under']:.1%}`
                        """)
            except Exception as e:
                st.error(f"Prediction error: {e}")

# -------------------------------------------------
# CACHE CLEAR
# -------------------------------------------------
if st.button("Clear Cache"):
    st.cache_data.clear()
    st.success("Cache cleared!")

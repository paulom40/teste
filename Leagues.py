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
Upload any **football-data.co.uk** CSV (Premier League, La Liga, etc.)  
and get **instant Poisson-based predictions** using **cached modeling**.
""")

# ================================
# UTILS: DATA LOADER
# ================================
@st.cache_data(show_spinner="Loading CSV...")
def load_csv(uploaded_file_bytes: bytes) -> pd.DataFrame:
    """Load CSV with UTF-8 or Latin1 fallback."""
    try:
        return pd.read_csv(io.BytesIO(uploaded_file_bytes), encoding="utf-8")
    except UnicodeDecodeError:
        return pd.read_csv(io.BytesIO(uploaded_file_bytes), encoding="latin1")

@st.cache_data(show_spinner=False)
def detect_columns(df: pd.DataFrame) -> Dict[str, str]:
    """Auto-detect standard column names."""
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
    ag_col: str
) -> Dict[str, Any]:
    """Calculate attack/defence strengths."""
    league_avg_home = _df[hg_col].mean()
    league_avg_away = _df[ag_col].mean()

    home_attack = _df.groupby(home_col)[hg_col].mean() / league_avg_home
    away_attack = _df.groupby(away_col)[ag_col].mean() / league_avg_away
    home_defence = _df.groupby(home_col)[ag_col].mean() / league_avg_away
    away_defence = _df.groupby(away_col)[hg_col].mean() / league_avg_home

    return {
        "league_avg_home": league_avg_home,
        "league_avg_away": league_avg_away,
        "home_attack": home_attack.to_dict(),
        "away_attack": away_attack.to_dict(),
        "home_defence": home_defence.to_dict(),
        "away_defence": away_defence.to_dict(),
    }

@st.cache_data(show_spinner=False)
def predict_match(
    home: str,
    away: str,
    stats: Dict[str, Any],
    max_goals: int = 10
) -> Dict[str, Any]:
    """Predict match using Poisson distribution."""
    la_home = stats["league_avg_home"]
    la_away = stats["league_avg_away"]

    lambda_home = stats["home_attack"].get(home, 1.0) * stats["away_defence"].get(away, 1.0) * la_home
    lambda_away = stats["away_attack"].get(away, 1.0) * stats["home_defence"].get(home, 1.0) * la_away

    home_probs = poisson.pmf(np.arange(max_goals + 1), lambda_home)
    away_probs = poisson.pmf(np.arange(max_goals + 1), lambda_away)

    prob_home = prob_draw = prob_away = 0.0
    best_score = (0, 0)
    best_prob = 0.0

    for h in range(max_goals + 1):
        for a in range(max_goals + 1):
            p = home_probs[h] * away_probs[a]
            if h > a:
                prob_home += p
            elif h == a:
                prob_draw += p
            else:
                prob_away += p

            if p > best_prob:
                best_prob = p
                best_score = (h, a)

    result = "H" if prob_home > max(prob_draw, prob_away) \
             else "D" if prob_draw > max(prob_home, prob_away) \
             else "A"

    return {
        "score": f"{best_score[0]}-{best_score[1]}",
        "home_win": prob_home,
        "draw": prob_draw,
        "away_win": prob_away,
        "result": result
    }

# ================================
# MAIN UI
# ================================
uploaded_file = st.file_uploader("Upload League CSV (e.g., E0.csv, SP1.csv)", type=["csv"])

if uploaded_file is not None:
    file_bytes = uploaded_file.getvalue()
    df = load_csv(file_bytes)
    st.success(f"Loaded {len(df):,} matches.")

    # Preview
    with st.expander("Preview Data", expanded=False):
        st.dataframe(df.head(10))

    # Column Mapping
    st.subheader("Map Required Columns")
    guessed = detect_columns(df)

    cols = st.columns(5)
    date_col = cols[0].selectbox("Date", df.columns, index=_safe_index(df, guessed.get("Date")))
    home_col = cols[1].selectbox("Home Team", df.columns, index=_safe_index(df, guessed.get("HomeTeam")))
    away_col = cols[2].selectbox("Away Team", df.columns, index=_safe_index(df, guessed.get("AwayTeam")))
    hg_col   = cols[3].selectbox("Home Goals (FTHG)", df.columns, index=_safe_index(df, guessed.get("FTHG")))
    ag_col   = cols[4].selectbox("Away Goals (FTAG)", df.columns, index=_safe_index(df, guessed.get("FTAG")))

    # Train Model
    if st.button("Train Poisson Model"):
        stats = compute_team_stats(df, home_col, away_col, hg_col, ag_col)
        teams = sorted(set(df[home_col]).union(df[away_col]))
        st.session_state.stats = stats
        st.session_state.teams = teams
        st.success("Model trained and cached!")

    # Prediction Section
    if "stats" in st.session_state:
        st.subheader("Predict a Match")
        col1, col2 = st.columns(2)
        home_team = col1.selectbox("Home Team", st.session_state.teams, key="pred_home")
        away_team = col2.selectbox("Away Team", st.session_state.teams, key="pred_away")

        if home_team == away_team:
            st.error("Please select two different teams.")
        elif st.button("Predict Outcome"):
            pred = predict_match(home_team, away_team, st.session_state.stats)
            st.markdown(f"""
            ### **{home_team} vs {away_team}**
            **Most Likely Score:** `{pred['score']}`  
            **Predicted Result:** **{pred['result']}**  
            - **Home Win**: `{pred['home_win']:.1%}`  
            - **Draw**:     `{pred['draw']:.1%}`  
            - **Away Win**: `{pred['away_win']:.1%}`
            """)

# Helper
def _safe_index(df: pd.DataFrame, col: str):
    """Return column index safely."""
    return df.columns.get_loc(col) if col in df.columns else 0

# Optional: Clear Cache Button
if st.button("Clear All Cache"):
    st.cache_data.clear()
    st.success("Cache cleared!")

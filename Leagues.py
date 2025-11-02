# app.py
import streamlit as st
import pandas as pd
import numpy as np
from scipy.stats import poisson
import io
from typing import Dict, Any
import requests
from PIL import Image
from io import BytesIO

# ================================
# CONFIG
# ================================
st.set_page_config(page_title="Football Predictor", layout="wide")
st.title("Football Match Outcome Predictor")
st.markdown("""
**Poisson-based predictions** from **football-data.co.uk** CSVs.

**Predicts:**
- **Goals** (FTHG/FTAG) – Required
- **Corners** (HC/AC) – Auto if present
- **Shots on Target** (HS/AS) – Auto if present
- **Expected Goals (xG)** – Auto if present

**Logos:** Auto-fetched
""")

# ================================
# LOGO CACHE
# ================================
@st.cache_data(ttl=3600)
def get_team_logo(team_name: str) -> str:
    team_clean = team_name.strip().lower().replace(" ", "_").replace(".", "").replace("'", "")
    replacements = {
        "man_utd": "Manchester_United_F.C.", "manchester_united": "Manchester_United_F.C.",
        "man_city": "Manchester_City_F.C.", "manchester_city": "Manchester_City_F.C.",
        "arsenal": "Arsenal_F.C.", "chelsea": "Chelsea_F.C.", "liverpool": "Liverpool_F.C.",
        "spurs": "Tottenham_Hotspur_F.C.", "tottenham": "Tottenham_Hotspur_F.C.",
        "nottm_forest": "Nottingham_Forest_F.C.", "nottmforest": "Nottingham_Forest_F.C.",
        "nott'm_forest": "Nottingham_Forest_F.C.",
    }
    wiki_name = replacements.get(team_clean, team_name.replace(" ", "_").replace("'", "") + "_F.C.")
    url = f"https://en.wikipedia.org/wiki/File:{wiki_name}_logo.svg"
    try:
        if requests.head(url, timeout=5).status_code == 200:
            return f"https://en.wikipedia.org/wiki/File:{wiki_name}_logo.svg"
    except:
        pass
    fd_url = f"https://www.football-data.co.uk/mmz4281/logos/{team_clean}.gif"
    try:
        if requests.head(fd_url, timeout=5).status_code == 200:
            return fd_url
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

# ================================
# HELPERS
# ================================
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
        elif lower in ["hc", "homecorners"]:     mapping["HC"] = col
        elif lower in ["ac", "awaycorners"]:     mapping["AC"] = col
        elif lower in ["hs", "homeshotsontarget"]: mapping["HS"] = col
        elif lower in ["as", "awayshotsontarget"]: mapping["AS"] = col
        elif lower in ["hxg", "home_xg"]:        mapping["HxG"] = col
        elif lower in ["axg", "away_xg"]:        mapping["AxG"] = col
    return mapping

# ================================
# MODEL
# ================================
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
    df = _df.copy()

    # Convert all numeric columns
    for col in [hg_col, ag_col, hc_col, ac_col, hs_col, as_col, hxg_col, axg_col]:
        if col and col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')

    # REQUIRED: GOALS
    # Select only rows where both goals are valid
    valid_mask = df[hg_col].notna() & df[ag_col].notna()
    clean = df[valid_mask][[home_col, away_col, hg_col, ag_col]]

    if len(clean) == 0:
        raise ValueError("No matches with valid FTHG and FTAG values.")
    if len(clean) < 5:
        raise ValueError(f"Only {len(clean)} valid matches. Need at least 5 to train.")

    st.info(f"Using **{len(clean)} valid matches** for training.")

    avg_home = clean[hg_col].mean()
    avg_away = clean[ag_col].mean()
    if avg_home == 0 or avg_away == 0:
        raise ValueError("League average goals are zero.")

    stats = {
        "goals": {
            "league_avg_home": avg_home,
            "league_avg_away": avg_away,
            "home_attack": (clean.groupby(home_col)[hg_col].mean() / avg_home).fillna(1.0).to_dict(),
            "away_attack": (clean.groupby(away_col)[ag_col].mean() / avg_away).fillna(1.0).to_dict(),
            "home_defence": (clean.groupby(home_col)[ag_col].mean() / avg_away).fillna(1.0).to_dict(),
            "away_defence": (clean.groupby(away_col)[hg_col].mean() / avg_home).fillna(1.0).to_dict(),
        }
    }

    # OPTIONAL: CORNERS, SHOTS, xG
    def add(name, h_col, a_col):
        if h_col and a_col and h_col in df.columns and a_col in df.columns:
            sub_mask = df[h_col].notna() & df[a_col].notna()
            sub = df[sub_mask][[home_col, away_col, h_col, a_col]]
            if len(sub) >= 5:
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

    add("corners", hc_col, ac_col)
    add("shots",   hs_col, as_col)
    add("xg",      hxg_col, axg_col)

    return stats

@st.cache_data(show_spinner=False)
def predict_match(home: str, away: str, stats: Dict[str, Any]) -> Dict[str, Any]:
    if not isinstance(stats, dict) or "goals" not in stats:
        raise ValueError("Invalid model. Re-train.")

    predictions = {}
    max_g = 10

    # GOALS
    g = stats["goals"]
    lambda_home = g["home_attack"].get(home, 1.0) * g["away_defence"].get(away, 1.0) * g["league_avg_home"]
    lambda_away = g["away_attack"].get(away, 1.0) * g["home_defence"].get(home, 1.0) * g["league_avg_away"]
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

    # GENERIC PREDICTOR
    def predict_stat(name, threshold=None, is_float=False):
        if name not in stats:
            return None
        s = stats[name]
        lh = s["home_attack"].get(home, 1.0) * s["away_defence"].get(away, 1.0) * s["league_avg_home"]
        la = s["away_attack"].get(away, 1.0) * s["home_defence"].get(home, 1.0) * s["league_avg_away"]
        ph = poisson.pmf(np.arange(max_g + 1), lh)
        pa = poisson.pmf(np.arange(max_g + 1), la)
        best_h = best_a = 0
        best_p = 0.0
        over = 0.0
        for h in range(max_g + 1):
            for a in range(max_g + 1):
                p = ph[h] * pa[a]
                if p > best_p:
                    best_p = p
                    best_h, best_a = h, a
                if threshold and h + a > threshold:
                    over += p
        score = f"{best_h:.1f}-{best_a:.1f}" if is_float else f"{best_h}-{best_a}"
        res = {"score": score}
        if threshold:
            res["over"] = over
            res["under"] = 1 - over
        return res

    if "corners" in stats:
        predictions["corners"] = predict_stat("corners", 10.5)
    if "shots" in stats:
        predictions["shots"] = predict_stat("shots", 20.5)
    if "xg" in stats:
        predictions["xg"] = predict_stat("xg", 2.5, True)

    return predictions

# ================================
# UI
# ================================
uploaded_file = st.file_uploader("Upload CSV", type=["csv"])

if uploaded_file:
    df = load_csv(uploaded_file.getvalue())
    st.success(f"Loaded {len(df)} rows.")

    with st.expander("Preview Data"):
        st.dataframe(df.head(10))

    st.subheader("Required: Select FTHG & FTAG")
    guessed = detect_columns(df)
    c1, c2, c3, c4, c5 = st.columns(5)
    home_col = c2.selectbox("Home Team", df.columns, index=_safe_index(df, guessed.get("HomeTeam")))
    away_col = c3.selectbox("Away Team", df.columns, index=_safe_index(df, guessed.get("AwayTeam")))
    hg_col   = c4.selectbox("Home Goals (FTHG)", df.columns, index=_safe_index(df, guessed.get("FTHG")))
    ag_col   = c5.selectbox("Away Goals (FTAG)", df.columns, index=_safe_index(df, guessed.get("FTAG")))

    # Validate
    try:
        df[hg_col] = pd.to_numeric(df[hg_col], errors='coerce')
        df[ag_col] = pd.to_numeric(df[ag_col], errors='coerce')
        valid_matches = df[hg_col].notna() & df[ag_col].notna()
        valid_count = valid_matches.sum()
        if valid_count == 0:
            st.error("No valid numeric values in FTHG or FTAG columns.")
        elif valid_count < 5:
            st.warning(f"Only **{valid_count} valid matches**. Need at least 5.")
        else:
            st.success(f"**{valid_count} valid matches** found!")
    except:
        st.error("Cannot convert FTHG/FTAG to numbers.")

    st.subheader("Optional Columns")
    o1, o2, o3, o4, o5, o6 = st.columns(6)
    hc_col  = o1.selectbox("HC", [""] + list(df.columns))
    ac_col  = o2.selectbox("AC", [""] + list(df.columns))
    hs_col  = o3.selectbox("HS", [""] + list(df.columns))
    as_col  = o4.selectbox("AS", [""] + list(df.columns))
    hxg_col = o5.selectbox("HxG", [""] + list(df.columns))
    axg_col = o6.selectbox("AxG", [""] + list(df.columns))

    train_disabled = valid_count < 5 if 'valid_count' in locals() else True
    if st.button("Train Model", disabled=train_disabled):
        with st.spinner("Training..."):
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

    if st.button("Clear Model & Cache"):
        for key in list(st.session_state.keys()):
            del st.session_state[key]
        st.cache_data.clear()
        st.success("Cleared!")

    if st.session_state.get("stats") and st.session_state.get("teams"):
        st.subheader("Predict Match")
        t1, t2 = st.columns(2)
        home_team = t1.selectbox("Home", st.session_state.teams, key="ph")
        away_team = t2.selectbox("Away", st.session_state.teams, key="pa")

        if home_team == away_team:
            st.error("Select different teams.")
        elif st.button("Predict"):
            pred = predict_match(home_team, away_team, st.session_state.stats)

            # LOGOS
            col1, col2, col3 = st.columns([1, 2, 1])
            with col1:
                logo = get_team_logo(home_team)
                if logo and load_image(logo):
                    st.image(load_image(logo), width=80)
                else:
                    st.markdown(f"**{home_team}**")
            with col2:
                st.markdown(f"### **{home_team} vs {away_team}**")
            with col3:
                logo = get_team_logo(away_team)
                if logo and load_image(logo):
                    st.image(load_image(logo), width=80)
                else:
                    st.markdown(f"**{away_team}**")

            # RESULTS
            g = pred["goals"]
            st.markdown(f"""
            #### Goals
            **{g['score']}** → **{g['result']}**  
            H: `{g['home_win']:.1%}` | D: `{g['draw']:.1%}` | A: `{g['away_win']:.1%}`
            """)

            if "corners" in pred:
                c = pred["corners"]
                st.markdown(f"""
                #### Corners
                **{c['score']}**  
                Over 10.5: `{c['over']:.1%}` | Under: `{c['under']:.1%}`
                """)

            if "shots" in pred:
                s = pred["shots"]
                st.markdown(f"""
                #### Shots on Target
                **{s['score']}**  
                Over 20.5: `{s['over']:.1%}` | Under: `{s['under']:.1%}`
                """)

            if "xg" in pred:
                x = pred["xg"]
                st.markdown(f"""
                #### Expected Goals (xG)
                **{x['score']}**  
                Over 2.5: `{x['over']:.1%}` | Under: `{x['under']:.1%}`
                """)

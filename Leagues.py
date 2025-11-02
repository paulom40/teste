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

**Required:** Home/Away Teams + **FTHG/FTAG**  
**Optional:** Corners, Shots, xG  
**Logos:** Automatically fetched
""")

# ================================
# LOGO CACHE
# ================================
@st.cache_data(ttl=3600)
def get_team_logo(team_name: str) -> str:
    """Return image URL for team logo. Uses Wikipedia + football-data.co.uk."""
    team_clean = team_name.strip().lower().replace(" ", "_").replace(".", "")
    
    # Common replacements
    replacements = {
        "manchester_united": "Manchester_United_F.C.",
        "man_utd": "Manchester_United_F.C.",
        "manchester_city": "Manchester_City_F.C.",
        "man_city": "Manchester_City_F.C.",
        "arsenal": "Arsenal_F.C.",
        "chelsea": "Chelsea_F.C.",
        "liverpool": "Liverpool_F.C.",
        "tottenham": "Tottenham_Hotspur_F.C.",
        "spurs": "Tottenham_Hotspur_F.C.",
        "leicester": "Leicester_City_F.C.",
        "leicester_city": "Leicester_City_F.C.",
        "west_ham": "West_Ham_United_F.C.",
        "everton": "Everton_F.C.",
        "southampton": "Southampton_F.C.",
        "brighton": "Brighton_&_Hove_Albion_F.C.",
        "newcastle": "Newcastle_United_F.C.",
        "crystal_palace": "Crystal_Palace_F.C.",
        "wolves": "Wolverhampton_Wanderers_F.C.",
        "wolverhampton_wanderers": "Wolverhampton_Wanderers_F.C.",
        "aston_villa": "Aston_Villa_F.C.",
        "leeds": "Leeds_United_F.C.",
        "leeds_united": "Leeds_United_F.C.",
        "burnley": "Burnley_F.C.",
        "brentford": "Brentford_F.C.",
        "fulham": "Fulham_F.C.",
        "nottingham_forest": "Nottingham_Forest_F.C.",
        "luton": "Luton_Town_F.C.",
        "luton_town": "Luton_Town_F.C.",
        "sheffield_united": "Sheffield_United_F.C.",
        "sheffield_wed": "Sheffield_Wednesday_F.C.",
        "sheffield_wednesday": "Sheffield_Wednesday_F.C.",
        "birmingham": "Birmingham_City_F.C.",
        "birmingham_city": "Birmingham_City_F.C.",
        "cardiff": "Cardiff_City_F.C.",
        "cardiff_city": "Cardiff_City_F.C.",
        "swansea": "Swansea_City_A.F.C.",
        "swansea_city": "Swansea_City_A.F.C.",
        "stoke": "Stoke_City_F.C.",
        "stoke_city": "Stoke_City_F.C.",
        "hull": "Hull_City_A.F.C.",
        "hull_city": "Hull_City_A.F.C.",
        "norwich": "Norwich_City_F.C.",
        "norwich_city": "Norwich_City_F.C.",
        "watford": "Watford_F.C.",
        "reading": "Reading_F.C.",
        "blackburn": "Blackburn_Rovers_F.C.",
        "blackburn_rovers": "Blackburn_Rovers_F.C.",
        "preston": "Preston_North_End_F.C.",
        "preston_north_end": "Preston_North_End_F.C.",
        "millwall": "Millwall_F.C.",
        "qpr": "Queens_Park_Rangers_F.C.",
        "queens_park_rangers": "Queens_Park_Rangers_F.C.",
        "coventry": "Coventry_City_F.C.",
        "coventry_city": "Coventry_City_F.C.",
        "huddersfield": "Huddersfield_Town_A.F.C.",
        "huddersfield_town": "Huddersfield_Town_A.F.C.",
        "middlesbrough": "Middlesbrough_F.C.",
        "sunderland": "Sunderland_A.F.C.",
        "west_brom": "West_Bromwich_Albion_F.C.",
        "west_bromwich_albion": "West_Bromwich_Albion_F.C.",
        "wigan": "Wigan_Athletic_F.C.",
        "wigan_athletic": "Wigan_Athletic_F.C.",
        "derby": "Derby_County_F.C.",
        "derby_county": "Derby_County_F.C.",
        "bolton": "Bolton_Wanderers_F.C.",
        "bolton_wanderers": "Bolton_Wanderers_F.C.",
        "ipswich": "Ipswich_Town_F.C.",
        "ipswich_town": "Ipswich_Town_F.C.",
        "charlton": "Charlton_Athletic_F.C.",
        "charlton_athletic": "Charlton_Athletic_F.C.",
        "portsmouth": "Portsmouth_F.C.",
        "bournemouth": "A.F.C._Bournemouth",
        "afc_bournemouth": "A.F.C._Bournemouth",
        "bristol_city": "Bristol_City_F.C.",
        "peterborough": "Peterborough_United_F.C.",
        "peterborough_united": "Peterborough_United_F.C.",
        "oxford": "Oxford_United_F.C.",
        "oxford_united": "Oxford_United_F.C.",
        "wycombe": "Wycombe_Wanderers_F.C.",
        "wycombe_wanderers": "Wycombe_Wanderers_F.C.",
        "rotherham": "Rotherham_United_F.C.",
        "rotherham_united": "Rotherham_United_F.C.",
        "accrington": "Accrington_Stanley_F.C.",
        "accrington_stanley": "Accrington_Stanley_F.C.",
        "burton": "Burton_Albion_F.C.",
        "burton_albion": "Burton_Albion_F.C.",
        "fleetwood": "Fleetwood_Town_F.C.",
        "fleetwood_town": "Fleetwood_Town_F.C.",
        "shrewsbury": "Shrewsbury_Town_F.C.",
        "shrewsbury_town": "Shrewsbury_Town_F.C.",
        "doncaster": "Doncaster_Rovers_F.C.",
        "doncaster_rovers": "Doncaster_Rovers_F.C.",
        "gillingham": "Gillingham_F.C.",
        "wimbledon": "AFC_Wimbledon",
        "afc_wimbledon": "AFC_Wimbledon",
        "crewe": "Crewe_Alexandra_F.C.",
        "crewe_alexandra": "Crewe_Alexandra_F.C.",
        "morecambe": "Morecambe_F.C.",
        "crawley": "Crawley_Town_F.C.",
        "crawley_town": "Crawley_Town_F.C.",
        "colchester": "Colchester_United_F.C.",
        "colchester_united": "Colchester_United_F.C.",
        "northampton": "Northampton_Town_F.C.",
        "northampton_town": "Northampton_Town_F.C.",
        "walsall": "Walsall_F.C.",
        "bradford": "Bradford_City_A.F.C.",
        "bradford_city": "Bradford_City_A.F.C.",
        "scunthorpe": "Scunthorpe_United_F.C.",
        "scunthorpe_united": "Scunthorpe_United_F.C.",
        "oldham": "Oldham_Athletic_A.F.C.",
        "oldham_athletic": "Oldham_Athletic_A.F.C.",
        "rochdale": "Rochdale_A.F.C.",
        "tranmere": "Tranmere_Rovers_F.C.",
        "tranmere_rovers": "Tranmere_Rovers_F.C.",
        "carlisle": "Carlisle_United_F.C.",
        "carlisle_united": "Carlisle_United_F.C.",
        "exeter": "Exeter_City_F.C.",
        "exeter_city": "Exeter_City_F.C.",
        "plymouth": "Plymouth_Argyle_F.C.",
        "plymouth_argyle": "Plymouth_Argyle_F.C.",
        "portsmouth": "Portsmouth_F.C.",
        "salford": "Salford_City_F.C.",
        "salford_city": "Salford_City_F.C.",
        "harrogate": "Harrogate_Town_A.F.C.",
        "harrogate_town": "Harrogate_Town_A.F.C.",
        "barrow": "Barrow_A.F.C.",
        "forest_green": "Forest_Green_Rovers_F.C.",
        "forest_green_rovers": "Forest_Green_Rovers_F.C.",
        "newport": "Newport_County_A.F.C.",
        "newport_county": "Newport_County_A.F.C.",
        "stevenage": "Stevenage_F.C.",
        "grimsby": "Grimsby_Town_F.C.",
        "grimsby_town": "Grimsby_Town_F.C.",
        "cheltenham": "Cheltenham_Town_F.C.",
        "cheltenham_town": "Cheltenham_Town_F.C.",
    }

    wiki_name = replacements.get(team_clean, None)
    if not wiki_name:
        # Try direct match
        wiki_name = team_name.replace(" ", "_") + "_F.C."

    url = f"https://en.wikipedia.org/wiki/File:{wiki_name}_logo.svg"
    
    try:
        response = requests.head(url, timeout=5)
        if response.status_code == 200:
            return f"https://en.wikipedia.org/wiki/File:{wiki_name}_logo.svg"
    except:
        pass

    # Fallback: football-data.co.uk
    fd_url = f"https://www.football-data.co.uk/mmz4281/logos/{team_clean}.gif"
    try:
        response = requests.head(fd_url, timeout=5)
        if response.status_code == 200:
            return fd_url
    except:
        pass

    return None

@st.cache_data(ttl=3600)
def load_image(url: str):
    try:
        response = requests.get(url, timeout=10)
        img = Image.open(BytesIO(response.content))
        img = img.convert("RGBA")
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
    stats = {}

    clean = _df[[home_col, away_col, hg_col, ag_col]].dropna()
    if clean.empty:
        raise ValueError("No valid matches after removing NaN from goal columns.")

    avg_home = clean[hg_col].mean()
    avg_away = clean[ag_col].mean()
    if avg_home == 0 or avg_away == 0:
        raise ValueError("League average goals are zero.")

    stats["goals"] = {
        "league_avg_home": avg_home,
        "league_avg_away": avg_away,
        "home_attack": (clean.groupby(home_col)[hg_col].mean() / avg_home).fillna(1.0).to_dict(),
        "away_attack": (clean.groupby(away_col)[ag_col].mean() / avg_away).fillna(1.0).to_dict(),
        "home_defence": (clean.groupby(home_col)[ag_col].mean() / avg_away).fillna(1.0).to_dict(),
        "away_defence": (clean.groupby(away_col)[hg_col].mean() / avg_home).fillna(1.0).to_dict(),
    }

    def add(name, h_col, a_col):
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

    add("corners", hc_col, ac_col)
    add("shots",   hs_col, as_col)
    add("xg",      hxg_col, axg_col)

    return stats

@st.cache_data(show_spinner=False)
def predict_match(home: str, away: str, stats: Dict[str, Any]) -> Dict[str, Any]:
    if not isinstance(stats, dict) or "goals" not in stats:
        raise ValueError("Invalid model. Re-train.")

    predictions = {}
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

    def predict(name, thresh=None, is_float=False):
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
                if thresh and h + a > thresh:
                    over += p
        score = f"{best_h:.1f}-{best_a:.1f}" if is_float else f"{best_h}-{best_a}"
        res = {"score": score}
        if thresh:
            res["over"] = over
            res["under"] = 1 - over
        return res

    predictions["corners"] = predict("corners", 10.5)
    predictions["shots"]   = predict("shots",   20.5)
    predictions["xg"]      = predict("xg",      2.5, True)

    return {k: v for k, v in predictions.items() if v is not None}

# ================================
# UI
# ================================
uploaded_file = st.file_uploader("Upload CSV", type=["csv"])

if uploaded_file:
    df = load_csv(uploaded_file.getvalue())
    st.success(f"Loaded {len(df)} rows.")

    with st.expander("Preview Data"):
        st.dataframe(df.head())

    st.subheader("Required: Select FTHG & FTAG")
    guessed = detect_columns(df)
    c1, c2, c3, c4, c5 = st.columns(5)
    home_col = c2.selectbox("Home Team", df.columns, index=_safe_index(df, guessed.get("HomeTeam")))
    away_col = c3.selectbox("Away Team", df.columns, index=_safe_index(df, guessed.get("AwayTeam")))
    hg_col   = c4.selectbox("Home Goals (FTHG)", df.columns, index=_safe_index(df, guessed.get("FTHG")))
    ag_col   = c5.selectbox("Away Goals (FTAG)", df.columns, index=_safe_index(df, guessed.get("FTAG")))

    valid = (
        hg_col in df.columns and ag_col in df.columns and
        not df[hg_col].isna().all() and not df[ag_col].isna().all()
    )

    if not valid:
        st.error("Please select valid FTHG and FTAG columns with real numbers.")
    else:
        st.success("Goal columns valid!")

    st.subheader("Optional Columns")
    o1, o2, o3, o4, o5, o6 = st.columns(6)
    hc_col  = o1.selectbox("HC", [""] + list(df.columns))
    ac_col  = o2.selectbox("AC", [""] + list(df.columns))
    hs_col  = o3.selectbox("HS", [""] + list(df.columns))
    as_col  = o4.selectbox("AS", [""] + list(df.columns))
    hxg_col = o5.selectbox("HxG", [""] + list(df.columns))
    axg_col = o6.selectbox("AxG", [""] + list(df.columns))

    if st.button("Train Model", disabled=not valid):
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
                st.success("Model trained!")
            except Exception as e:
                st.error(f"Training failed: {e}")

    if st.button("Clear Model & Cache"):
        for key in list(st.session_state.keys()):
            del st.session_state[key]
        st.cache_data.clear()
        st.success("Cleared!")

    if st.session_state.get("stats") and st.session_state.get("teams"):
        if not isinstance(st.session_state.stats, dict) or "goals" not in st.session_state.stats:
            st.error("Model corrupted. Click **Clear Model & Cache** and re-train.")
        else:
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
                    home_logo_url = get_team_logo(home_team)
                    if home_logo_url:
                        img = load_image(home_logo_url)
                        if img:
                            st.image(img, width=80)
                        else:
                            st.markdown(f"**{home_team}**")
                    else:
                        st.markdown(f"**{home_team}**")
                with col2:
                    st.markdown(f"### **{home_team} vs {away_team}**")
                with col3:
                    away_logo_url = get_team_logo(away_team)
                    if away_logo_url:
                        img = load_image(away_logo_url)
                        if img:
                            st.image(img, width=80)
                        else:
                            st.markdown(f"**{away_team}**")
                    else:
                        st.markdown(f"**{away_team}**")

                g = pred["goals"]
                st.markdown(f"""
                #### Goals
                **{g['score']}** → **{g['result']}**  
                H: `{g['home_win']:.1%}` | D: `{g['draw']:.1%}` | A: `{g['away_win']:.1%}`
                """)

                for name, label, thresh in [
                    ("corners", "Corners", 10.5),
                    ("shots", "Shots", 20.5),
                    ("xg", "xG", 2.5)
                ]:
                    if name in pred:
                        p = pred[name]
                        st.markdown(f"""
                        #### {label}
                        **{p['score']}**  
                        Over {thresh}: `{p['over']:.1%}` | Under: `{p['under']:.1%}`
                        """)

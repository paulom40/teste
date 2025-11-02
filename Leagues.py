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
import base64
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

# ================================
# CONFIG
# ================================
st.set_page_config(page_title="Football Predictor", layout="wide")
st.title("Football Match Outcome Predictor")
st.markdown("""
**Poisson-based predictions** from **football-data.co.uk** CSVs.

**Predicts:**
- **Full-Time** (FTHG/FTAG)
- **Half-Time** (HTHG/HTAG)
- **BTTS** (Both Teams To Score)
- **Corners, Shots, xG**

**Interactive Charts + Print-to-PDF**
""")

# ================================
# LOGO CACHE
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

# ================================
# PRINT CSS
# ================================
print_css = """
<style>
@media print {
    body { font-family: Arial; margin: 1in; }
    .print-title { font-size: 24px; font-weight: bold; text-align: center; margin-bottom: 20px; }
    .team-box { text-align: center; }
    .logo { width: 80px; height: 80px; }
    .prediction { margin: 20px 0; padding: 15px; border: 1px solid #ccc; border-radius: 8px; background: #f9f9f9; }
    .score { font-size: 20px; font-weight: bold; }
    .prob { font-size: 14px; color: #555; }
    .no-print { display: none; }
    .stPlotlyChart { display: none; }
}
</style>
"""

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
# MODEL
# ================================
@st.cache_data(show_spinner="Training model...")
def compute_team_stats(
    _df: pd.DataFrame,
    home_col: str,
    away_col: str,
    hg_col: str,
    ag_col: str,
    hthg_col: str = None,
    htag_col: str = None,
    hc_col: str = None,
    ac_col: str = None,
    hs_col: str = None,
    as_col: str = None,
    hxg_col: str = None,
    axg_col: str = None
) -> Dict[str, Any]:
    df = _df.copy()
    for col in [hg_col, ag_col, hthg_col, htag_col, hc_col, ac_col, hs_col, as_col, hxg_col, axg_col]:
        if col and col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')

    stats = {}

    # FULL-TIME
    ft_mask = df[hg_col].notna() & df[ag_col].notna()
    clean_ft = df[ft_mask][[home_col, away_col, hg_col, ag_col]]
    if len(clean_ft) < 5:
        raise ValueError(f"Only {len(clean_ft)} valid FT matches. Need at least 5.")
    avg_home = clean_ft[hg_col].mean()
    avg_away = clean_ft[ag_col].mean()
    stats["goals"] = {
        "league_avg_home": avg_home,
        "league_avg_away": avg_away,
        "home_attack": (clean_ft.groupby(home_col)[hg_col].mean() / avg_home).fillna(1.0).to_dict(),
        "away_attack": (clean_ft.groupby(away_col)[ag_col].mean() / avg_away).fillna(1.0).to_dict(),
        "home_defence": (clean_ft.groupby(home_col)[ag_col].mean() / avg_away).fillna(1.0).to_dict(),
        "away_defence": (clean_ft.groupby(away_col)[hg_col].mean() / avg_home).fillna(1.0).to_dict(),
    }

    # HALF-TIME
    if hthg_col and htag_col and hthg_col in df.columns and htag_col in df.columns:
        ht_mask = df[hthg_col].notna() & df[htag_col].notna()
        clean_ht = df[ht_mask][[home_col, away_col, hthg_col, htag_col]]
        if len(clean_ht) >= 5:
            avg_h = clean_ht[hthg_col].mean()
            avg_a = clean_ht[htag_col].mean()
            if avg_h > 0 and avg_a > 0:
                stats["half_time"] = {
                    "league_avg_home": avg_h,
                    "league_avg_away": avg_a,
                    "home_attack": (clean_ht.groupby(home_col)[hthg_col].mean() / avg_h).fillna(1.0).to_dict(),
                    "away_attack": (clean_ht.groupby(away_col)[htag_col].mean() / avg_a).fillna(1.0).to_dict(),
                    "home_defence": (clean_ht.groupby(home_col)[htag_col].mean() / avg_a).fillna(1.0).to_dict(),
                    "away_defence": (clean_ht.groupby(away_col)[hthg_col].mean() / avg_h).fillna(1.0).to_dict(),
                }

    # OPTIONAL
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

# ================================
# PREDICT MATCH + CHARTS DATA
# ================================
@st.cache_data(show_spinner=False)
def predict_match(home: str, away: str, stats: Dict[str, Any]) -> Dict[str, Any]:
    max_g = 10
    max_ht = 5
    predictions = {}
    chart_data = {}

    # FULL-TIME
    g = stats["goals"]
    lambda_home = g["home_attack"].get(home, 1.0) * g["away_defence"].get(away, 1.0) * g["league_avg_home"]
    lambda_away = g["away_attack"].get(away, 1.0) * g["home_defence"].get(home, 1.0) * g["league_avg_away"]
    hp = poisson.pmf(np.arange(max_g + 1), lambda_home)
    ap = poisson.pmf(np.arange(max_g + 1), lambda_away)

    # Score matrix
    matrix = np.outer(hp, ap)
    chart_data["ft_matrix"] = pd.DataFrame(matrix, index=range(max_g+1), columns=range(max_g+1))

    prob_h = prob_d = prob_a = btts_yes = 0.0
    best = (0, 0)
    best_p = 0.0
    for h in range(max_g + 1):
        for a in range(max_g + 1):
            p = hp[h] * ap[a]
            if h > a:   prob_h += p
            elif h == a: prob_d += p
            else:       prob_a += p
            if h > 0 and a > 0: btts_yes += p
            if p > best_p:
                best_p = p
                best = (h, a)

    result = "H" if prob_h > max(prob_d, prob_a) else "D" if prob_d > max(prob_h, prob_a) else "A"
    btts_result = "Yes" if btts_yes > 0.5 else "No"

    predictions["goals"] = {
        "score": f"{best[0]}-{best[1]}",
        "home_win": prob_h,
        "draw": prob_d,
        "away_win": prob_a,
        "result": result,
        "btts_yes": btts_yes,
        "btts_no": 1 - btts_yes,
        "btts_result": btts_result
    }

    # HALF-TIME
    if "half_time" in stats:
        ht = stats["half_time"]
        lh = ht["home_attack"].get(home, 1.0) * ht["away_defence"].get(away, 1.0) * ht["league_avg_home"]
        la = ht["away_attack"].get(away, 1.0) * ht["home_defence"].get(home, 1.0) * ht["league_avg_away"]
        ph = poisson.pmf(np.arange(max_ht + 1), lh)
        pa = poisson.pmf(np.arange(max_ht + 1), la)
        ht_matrix = np.outer(ph, pa)
        chart_data["ht_matrix"] = pd.DataFrame(ht_matrix, index=range(max_ht+1), columns=range(max_ht+1))

        prob_h = prob_d = prob_a = 0.0
        best = (0, 0)
        best_p = 0.0
        for h in range(max_ht + 1):
            for a in range(max_ht + 1):
                p = ph[h] * pa[a]
                if h > a:   prob_h += p
                elif h == a: prob_d += p
                else:       prob_a += p
                if p > best_p:
                    best_p = p
                    best = (h, a)
        result = "H" if prob_h > max(prob_d, prob_a) else "D" if prob_d > max(prob_h, prob_a) else "A"
        predictions["half_time"] = {
            "score": f"{best[0]}-{best[1]}",
            "home_win": prob_h,
            "draw": prob_d,
            "away_win": prob_a,
            "result": result
        }

    # GENERIC
    def predict_stat(name, threshold=None):
        if name not in stats: return None
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
        res = {"score": f"{best_h}-{best_a}"}
        if threshold:
            res["over"] = over
            res["under"] = 1 - over
        return res

    if "corners" in stats: predictions["corners"] = predict_stat("corners", 10.5)
    if "shots" in stats:   predictions["shots"]   = predict_stat("shots",   20.5)
    if "xg" in stats:      predictions["xg"]      = predict_stat("xg",      2.5)

    predictions["chart_data"] = chart_data
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

    try:
        valid_count = (pd.to_numeric(df[hg_col], errors='coerce').notna() & 
                      pd.to_numeric(df[ag_col], errors='coerce').notna()).sum()
        if valid_count < 5:
            st.warning(f"Only {valid_count} valid matches. Need at least 5.")
        else:
            st.success(f"{valid_count} valid matches!")
    except:
        st.error("Cannot convert goals to numbers.")

    st.subheader("Optional Columns")
    o1, o2, o3, o4, o5, o6, o7, o8 = st.columns(8)
    hthg_col = o1.selectbox("HTHG", [""] + list(df.columns))
    htag_col = o2.selectbox("HTAG", [""] + list(df.columns))
    hc_col   = o3.selectbox("HC",   [""] + list(df.columns))
    ac_col   = o4.selectbox("AC",   [""] + list(df.columns))
    hs_col   = o5.selectbox("HS",   [""] + list(df.columns))
    as_col   = o6.selectbox("AS",   [""] + list(df.columns))
    hxg_col  = o7.selectbox("HxG",  [""] + list(df.columns))
    axg_col  = o8.selectbox("AxG",  [""] + list(df.columns))

    if st.button("Train Model", disabled=valid_count < 5 if 'valid_count' in locals() else True):
        with st.spinner("Training..."):
            try:
                stats = compute_team_stats(df, home_col, away_col, hg_col, ag_col,
                                         hthg_col or None, htag_col or None,
                                         hc_col or None, ac_col or None,
                                         hs_col or None, as_col or None,
                                         hxg_col or None, axg_col or None)
                teams = sorted(set(df[home_col]).union(df[away_col]))
                st.session_state.stats = stats
                st.session_state.teams = teams
                st.session_state.prediction = None
                st.success("Model trained!")
            except Exception as e:
                st.error(f"Training failed: {e}")

    if st.button("Clear Model & Cache"):
        for key in list(st.session_state.keys()):
            del st.session_state[key]
        st.cache_data.clear()
        st.success("Cleared!")

    # PREDICTION + CHARTS
    if st.session_state.get("stats") and st.session_state.get("teams"):
        st.subheader("Predict Match")
        t1, t2 = st.columns(2)
        home_team = t1.selectbox("Home", st.session_state.teams, key="ph")
        away_team = t2.selectbox("Away", st.session_state.teams, key="pa")

        if home_team == away_team:
            st.error("Select different teams.")
        else:
            predict_key = f"predict_{home_team}_{away_team}"
            if st.button("Predict", key=predict_key):
                pred = predict_match(home_team, away_team, st.session_state.stats)
                st.session_state.prediction = pred
                st.session_state.match = (home_team, away_team)

        if st.session_state.get("prediction"):
            home_team, away_team = st.session_state.match
            pred = st.session_state.prediction
            g = pred["goals"]
            chart_data = pred.get("chart_data", {})

            # LOGOS
            logo1 = get_team_logo(home_team)
            logo2 = get_team_logo(away_team)
            img1 = load_image(logo1) if logo1 else None
            img2 = load_image(logo2) if logo2 else None

            # === CHARTS ===
            col1, col2 = st.columns([1, 1])

            with col1:
                st.markdown("### Full-Time Score Matrix")
                if "ft_matrix" in chart_data:
                    fig = px.imshow(
                        chart_data["ft_matrix"],
                        labels=dict(x=f"{away_team} Goals", y=f"{home_team} Goals", color="Probability"),
                        x=[str(i) for i in range(11)],
                        y=[str(i) for i in range(11)],
                        color_continuous_scale="Blues",
                        text_auto=".1%"
                    )
                    fig.update_layout(height=500, title_text="Most Likely: " + g["score"])
                    st.plotly_chart(fig, use_container_width=True)

            with col2:
                st.markdown("### BTTS & Result Probabilities")
                labels = ['Home Win', 'Draw', 'Away Win', 'BTTS Yes', 'BTTS No']
                values = [g['home_win'], g['draw'], g['away_win'], g['btts_yes'], g['btts_no']]
                fig = go.Figure(data=[go.Pie(labels=labels, values=values, hole=0.4)])
                fig.update_traces(textinfo='percent+label')
                fig.update_layout(height=500)
                st.plotly_chart(fig, use_container_width=True)

            # Half-Time Chart
            if "half_time" in pred and "ht_matrix" in chart_data:
                st.markdown("### Half-Time Score Matrix")
                ht = pred["half_time"]
                fig = px.imshow(
                    chart_data["ht_matrix"],
                    labels=dict(x=f"{away_team} HT Goals", y=f"{home_team} HT Goals", color="Probability"),
                    x=[str(i) for i in range(6)],
                    y=[str(i) for i in range(6)],
                    color_continuous_scale="Greens",
                    text_auto=".1%"
                )
                fig.update_layout(height=400, title_text="HT Most Likely: " + ht["score"])
                st.plotly_chart(fig, use_container_width=True)

            # === SUMMARY + PRINT ===
            st.markdown(print_css, unsafe_allow_html=True)
            st.markdown("### Summary (Print with Ctrl+P)")

            def img_to_base64(img):
                if not img: return ""
                buffered = BytesIO()
                img.save(buffered, format="PNG")
                return base64.b64encode(buffered.getvalue()).decode()

            print_html = f"""
            <div class="print-title">{home_team} vs {away_team}</div>
            <div style="display: flex; justify-content: center; gap: 50px; margin: 20px 0;">
                <div class="team-box">
                    {f'<img src="data:image/png;base64,{img_to_base64(img1)}" class="logo">' if img1 else f'<b>{home_team}</b>'}
                </div>
                <div style="font-size: 24px; font-weight: bold; align-self: center;">VS</div>
                <div class="team-box">
                    {f'<img src="data:image/png;base64,{img_to_base64(img2)}" class="logo">' if img2 else f'<b>{away_team}</b>'}
                </div>
            </div>
            """

            print_html += f"""
            <div class="prediction">
                <div class="score">Full-Time: {g['score']} → {g['result']}</div>
                <div class="prob">H: {g['home_win']:.1%} | D: {g['draw']:.1%} | A: {g['away_win']:.1%}</div>
            </div>
            <div class="prediction">
                <div class="score">BTTS: {g['btts_result']}</div>
                <div class="prob">Yes: {g['btts_yes']:.1%} | No: {g['btts_no']:.1%}</div>
            </div>
            """

            if "half_time" in pred:
                ht = pred["half_time"]
                print_html += f"""
                <div class="prediction">
                    <div class="score">Half-Time: {ht['score']} → {ht['result']}</div>
                    <div class="prob">H: {ht['home_win']:.1%} | D: {ht['draw']:.1%} | A: {ht['away_win']:.1%}</div>
                </div>
                """

            st.markdown(print_html, unsafe_allow_html=True)
            st.markdown("**Click below → Ctrl+P → Save as PDF**")
            st.markdown('<button onclick="window.print()" class="no-print" style="padding:12px 24px; font-size:16px; background:#4CAF50; color:white; border:none; border-radius:5px; cursor:pointer;">Print / Save as PDF</button>', unsafe_allow_html=True)

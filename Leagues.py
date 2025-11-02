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
- **Over/Under 2.5 Goals**
- **Corners (Total + Over 10.5)**
- **xG (Home/Away/Total + Over 2.5)** **NEW**

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

    # FULL-TIME GOALS
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

    # CORNERS
    if hc_col and ac_col and hc_col in df.columns and ac_col in df.columns:
        corner_mask = df[hc_col].notna() & df[ac_col].notna()
        clean_c = df[corner_mask][[home_col, away_col, hc_col, ac_col]]
        if len(clean_c) >= 5:
            avg_hc = clean_c[hc_col].mean()
            avg_ac = clean_c[ac_col].mean()
            if avg_hc > 0 and avg_ac > 0:
                stats["corners"] = {
                    "league_avg_home": avg_hc,
                    "league_avg_away": avg_ac,
                    "home_attack": (clean_c.groupby(home_col)[hc_col].mean() / avg_hc).fillna(1.0).to_dict(),
                    "away_attack": (clean_c.groupby(away_col)[ac_col].mean() / avg_ac).fillna(1.0).to_dict(),
                    "home_defence": (clean_c.groupby(home_col)[ac_col].mean() / avg_ac).fillna(1.0).to_dict(),
                    "away_defence": (clean_c.groupby(away_col)[hc_col].mean() / avg_hc).fillna(1.0).to_dict(),
                }

    # xG
    if hxg_col and axg_col and hxg_col in df.columns and axg_col in df.columns:
        xg_mask = df[hxg_col].notna() & df[axg_col].notna()
        clean_xg = df[xg_mask][[home_col, away_col, hxg_col, axg_col]]
        if len(clean_xg) >= 5:
            avg_hxg = clean_xg[hxg_col].mean()
            avg_axg = clean_xg[axg_col].mean()
            if avg_hxg > 0 and avg_axg > 0:
                stats["xg"] = {
                    "league_avg_home": avg_hxg,
                    "league_avg_away": avg_axg,
                    "home_attack": (clean_xg.groupby(home_col)[hxg_col].mean() / avg_hxg).fillna(1.0).to_dict(),
                    "away_attack": (clean_xg.groupby(away_col)[axg_col].mean() / avg_axg).fillna(1.0).to_dict(),
                    "home_defence": (clean_xg.groupby(home_col)[axg_col].mean() / avg_axg).fillna(1.0).to_dict(),
                    "away_defence": (clean_xg.groupby(away_col)[hxg_col].mean() / avg_hxg).fillna(1.0).to_dict(),
                }

    return stats

# ================================
# PREDICT MATCH + xG
# ================================
@st.cache_data(show_spinner=False)
def predict_match(home: str, away: str, stats: Dict[str, Any]) -> Dict[str, Any]:
    max_g = 10
    max_c = 15
    predictions = {
        "goals": {
            "score": "N/A", "result": "N/A",
            "home_win": 0, "draw": 0, "away_win": 0,
            "btts_yes": 0, "btts_no": 1, "btts_result": "N/A",
            "over_25": 0, "under_25": 1, "over_under_result": "N/A"
        }
    }
    chart_data = {}

    # FULL-TIME GOALS
    if "goals" in stats:
        g = stats["goals"]
        lambda_home = g["home_attack"].get(home, 1.0) * g["away_defence"].get(away, 1.0) * g["league_avg_home"]
        lambda_away = g["away_attack"].get(away, 1.0) * g["home_defence"].get(home, 1.0) * g["league_avg_away"]
        hp = poisson.pmf(np.arange(max_g + 1), lambda_home)
        ap = poisson.pmf(np.arange(max_g + 1), lambda_away)

        matrix = np.outer(hp, ap)
        chart_data["ft_matrix"] = pd.DataFrame(
            matrix,
            index=[f"{home} {i}" for i in range(max_g + 1)],
            columns=[f"{away} {i}" for i in range(max_g + 1)]
        )

        prob_h = prob_d = prob_a = btts_yes = over_25 = 0.0
        best = (0, 0)
        best_p = 0.0
        for h in range(max_g + 1):
            for a in range(max_g + 1):
                p = hp[h] * ap[a]
                if h > a:   prob_h += p
                elif h == a: prob_d += p
                else:       prob_a += p
                if h > 0 and a > 0: btts_yes += p
                if h + a > 2.5: over_25 += p
                if p > best_p:
                    best_p = p
                    best = (h, a)

        result = "H" if prob_h > max(prob_d, prob_a) else "D" if prob_d > max(prob_h, prob_a) else "A"
        btts_result = "Yes" if btts_yes > 0.5 else "No"
        over_under_result = "Over" if over_25 > 0.5 else "Under"

        predictions["goals"] = {
            "score": f"{best[0]}-{best[1]}",
            "home_win": prob_h, "draw": prob_d, "away_win": prob_a,
            "result": result,
            "btts_yes": btts_yes, "btts_no": 1 - btts_yes, "btts_result": btts_result,
            "over_25": over_25, "under_25": 1 - over_25, "over_under_result": over_under_result
        }

    # CORNERS
    if "corners" in stats:
        c = stats["corners"]
        lambda_hc = c["home_attack"].get(home, 1.0) * c["away_defence"].get(away, 1.0) * c["league_avg_home"]
        lambda_ac = c["away_attack"].get(away, 1.0) * c["home_defence"].get(home, 1.0) * c["league_avg_away"]
        hc_probs = poisson.pmf(np.arange(max_c + 1), lambda_hc)
        ac_probs = poisson.pmf(np.arange(max_c + 1), lambda_ac)

        total_probs = np.zeros(max_c + 1)
        best_total = 0
        best_p = 0.0
        over_10_5 = 0.0
        for h in range(max_c + 1):
            for a in range(max_c + 1):
                p = hc_probs[h] * ac_probs[a]
                total = h + a
                if total <= max_c:
                    total_probs[total] += p
                if total > 10.5:
                    over_10_5 += p
                if p > best_p:
                    best_p = p
                    best_total = total

        corner_result = "Over" if over_10_5 > 0.5 else "Under"
        predictions["corners"] = {
            "total": best_total,
            "over_10_5": over_10_5,
            "under_10_5": 1 - over_10_5,
            "result": corner_result,
            "distribution": total_probs.tolist()
        }
        chart_data["corner_dist"] = pd.Series(total_probs, index=range(max_c + 1))

    # xG
    if "xg" in stats:
        xg = stats["xg"]
        lambda_hxg = xg["home_attack"].get(home, 1.0) * xg["away_defence"].get(away, 1.0) * xg["league_avg_home"]
        lambda_axg = xg["away_attack"].get(away, 1.0) * xg["home_defence"].get(home, 1.0) * xg["league_avg_away"]

        # Expected xG values
        home_xg = lambda_hxg
        away_xg = lambda_axg
        total_xg = home_xg + away_xg
        over_25_xg = poisson.sf(2, total_xg)  # P(total_xg > 2.5)

        predictions["xg"] = {
            "home_xg": home_xg,
            "away_xg": away_xg,
            "total_xg": total_xg,
            "over_25_xg": over_25_xg,
            "under_25_xg": 1 - over_25_xg,
            "result": "Over" if over_25_xg > 0.5 else "Under"
        }

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
    hc_col   = o3.selectbox("HC (Home Corners)", [""] + list(df.columns))
    ac_col   = o4.selectbox("AC (Away Corners)", [""] + list(df.columns))
    hs_col   = o5.selectbox("HS", [""] + list(df.columns))
    as_col   = o6.selectbox("AS", [""] + list(df.columns))
    hxg_col  = o7.selectbox("HxG (Home xG)", [""] + list(df.columns))
    axg_col  = o8.selectbox("AxG (Away xG)", [""] + list(df.columns))

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
            g = pred.get("goals", {})
            c = pred.get("corners", {})
            x = pred.get("xg", {})
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
                        color_continuous_scale="Blues",
                        text_auto=".1%"
                    )
                    fig.update_layout(height=500, title_text=f"Most Likely: {g.get('score', 'N/A')}")
                    st.plotly_chart(fig, use_container_width=True)

            with col2:
                st.markdown("### Key Markets")
                pie_labels = ['Home Win', 'Draw']
                pie_values = [g.get('home_win', 0), g.get('draw', 0)]
                if g.get('btts_yes') is not None:
                    pie_labels.append('BTTS Yes')
                    pie_values.append(g['btts_yes'])
                if g.get('over_25') is not None:
                    pie_labels.append('Over 2.5')
                    pie_values.append(g['over_25'])
                if len(pie_values) > 0:
                    fig = go.Figure(data=[go.Pie(labels=pie_labels, values=pie_values, hole=0.4)])
                    fig.update_traces(textinfo='percent+label')
                    fig.update_layout(height=500)
                    st.plotly_chart(fig, use_container_width=True)

            # CORNERS
            if c and "distribution" in c:
                st.markdown("### Corner Kicks Distribution")
                fig = px.bar(
                    x=range(len(c["distribution"])),
                    y=c["distribution"],
                    labels=dict(x="Total Corners", y="Probability"),
                    title=f"Most Likely: {c['total']} | Over 10.5: {c['over_10_5']:.1%}"
                )
                fig.update_layout(height=400)
                st.plotly_chart(fig, use_container_width=True)

            # xG
            if x:
                st.markdown("### Expected Goals (xG)")
                fig = go.Figure(data=[
                    go.Bar(name='Home xG', x=['xG'], y=[x['home_xg']], marker_color='lightgreen'),
                    go.Bar(name='Away xG', x=['xG'], y=[x['away_xg']], marker_color='salmon')
                ])
                fig.add_hline(y=2.5, line_dash="dash", line_color="gray", annotation_text="2.5 Line")
                fig.update_layout(barmode='stack', height=400, title_text=f"Total xG: {x['total_xg']:.2f} | Over 2.5: {x['over_25_xg']:.1%}")
                st.plotly_chart(fig, use_container_width=True)

            # PRINT SECTION
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

            if g:
                print_html += f"""
                <div class="prediction">
                    <div class="score">Full-Time: {g.get('score', 'N/A')} to {g.get('result', 'N/A')}</div>
                    <div class="prob">H: {g.get('home_win', 0):.1%} | D: {g.get('draw', 0):.1%} | A: {g.get('away_win', 0):.1%}</div>
                </div>
                """
                if g.get('btts_result'):
                    print_html += f"""
                    <div class="prediction">
                        <div class="score">BTTS: {g['btts_result']}</div>
                        <div class="prob">Yes: {g.get('btts_yes', 0):.1%} | No: {g.get('btts_no', 0):.1%}</div>
                    </div>
                    """
                if g.get('over_under_result'):
                    print_html += f"""
                    <div class="prediction">
                        <div class="score">Over/Under 2.5: {g['over_under_result']} 2.5</div>
                        <div class="prob">Over: {g.get('over_25', 0):.1%} | Under: {g.get('under_25', 0):.1%}</div>
                    </div>
                    """

            if c:
                print_html += f"""
                <div class="prediction">
                    <div class="score">Corners: {c['total']} (Most Likely)</div>
                    <div class="prob">Over 10.5: {c['over_10_5']:.1%} | Under: {c['under_10_5']:.1%}</div>
                </div>
                """

            if x:
                print_html += f"""
                <div class="prediction">
                    <div class="score">xG: {home_team} {x['home_xg']:.2f} – {away_team} {x['away_xg']:.2f}</div>
                    <div class="prob">Total: {x['total_xg']:.2f} | Over 2.5: {x['over_25_xg']:.1%}</div>
                </div>
                """

            st.markdown(print_html, unsafe_allow_html=True)

            # PRINT BUTTON
            st.markdown("""
            <div class="no-print" style="margin:20px 0;">
                <button onclick="window.print()" style="
                    padding:12px 24px; font-size:16px; background:#4CAF50; color:white; 
                    border:none; border-radius:5px; cursor:pointer;
                ">Print / Save as PDF</button>
            </div>
            """, unsafe_allow_html=True)

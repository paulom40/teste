# app.py
import streamlit as st
import pandas as pd
import numpy as np
from scipy.stats import poisson
import io
from typing import Dict, Any, List
import requests
from PIL import Image
from io import BytesIO
import base64
import plotly.graph_objects as go
import plotly.express as px
import re
import os
import tempfile

# --- PDF EXPORT ---
# Try different PDF libraries in order of preference
PDF_AVAILABLE = False
PDF_LIBRARY = None

try:
    from xhtml2pdf import pisa
    PDF_AVAILABLE = True
    PDF_LIBRARY = "xhtml2pdf"
    st.success("✓ xhtml2pdf available for PDF export")
except ImportError:
    st.warning("✗ xhtml2pdf not available")

if not PDF_AVAILABLE:
    try:
        from weasyprint import HTML
        PDF_AVAILABLE = True
        PDF_LIBRARY = "weasyprint"
        st.success("✓ WeasyPrint available for PDF export")
    except ImportError:
        st.warning("✗ WeasyPrint not available")

if not PDF_AVAILABLE:
    try:
        import pdfkit
        PDF_AVAILABLE = True
        PDF_LIBRARY = "pdfkit"
        st.info("✓ PDFKit available (but requires wkhtmltopdf)")
    except ImportError:
        st.warning("✗ PDFKit not available")

# ================================
# CONFIG
# ================================
st.set_page_config(page_title="Football Predictor", layout="wide")
st.title("Football Match Outcome Predictor")
st.markdown("""
**Full Prediction Suite + Export to PDF**

**Predicts:**
- Full-Time Score | BTTS | Over 2.5
- Corners | xG | Shots on Target
- **Goal Timing (1–15, 16–30, ..., 76–90)** — **Minute-Level Precision**

**Export to PDF with one click**
""")

# Show PDF export status
if PDF_AVAILABLE:
    st.success(f"✅ PDF Export: {PDF_LIBRARY} is ready to use!")
else:
    st.error("❌ PDF Export: No PDF libraries available. Install xhtml2pdf.")

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
# MODEL
# ================================
@st.cache_data(show_spinner="Training model...")
def compute_team_stats(
    _df: pd.DataFrame,
    home_col: str, away_col: str, hg_col: str, ag_col: str,
    hthg_col=None, htag_col=None, hc_col=None, ac_col=None,
    hs_col=None, as_col=None, hxg_col=None, axg_col=None
) -> Dict[str, Any]:
    df = _df.copy()
    for col in [hg_col, ag_col, hthg_col, htag_col, hc_col, ac_col, hs_col, as_col, hxg_col, axg_col]:
        if col and col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')

    stats = {}

    # === GOALS ===
    ft_mask = df[hg_col].notna() & df[ag_col].notna()
    clean_ft = df[ft_mask][[home_col, away_col, hg_col, ag_col]]
    if len(clean_ft) < 5:
        raise ValueError(f"Only {len(clean_ft)} valid matches.")
    avg_home = clean_ft[hg_col].mean()
    avg_away = clean_ft[ag_col].mean()
    stats["goals"] = {
        "league_avg_home": avg_home, "league_avg_away": avg_away,
        "home_attack": (clean_ft.groupby(home_col)[hg_col].mean() / avg_home).fillna(1.0).to_dict(),
        "away_attack": (clean_ft.groupby(away_col)[ag_col].mean() / avg_away).fillna(1.0).to_dict(),
        "home_defence": (clean_ft.groupby(home_col)[ag_col].mean() / avg_away).fillna(1.0).to_dict(),
        "away_defence": (clean_ft.groupby(away_col)[hg_col].mean() / avg_home).fillna(1.0).to_dict(),
    }

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

    # === CORNERS ===
    if hc_col and ac_col and hc_col in df.columns and ac_col in df.columns:
        c_mask = df[hc_col].notna() & df[ac_col].notna()
        clean_c = df[c_mask][[home_col, away_col, hc_col, ac_col]]
        if len(clean_c) >= 5:
            avg_hc = clean_c[hc_col].mean()
            avg_ac = clean_c[ac_col].mean()
            if avg_hc > 0 and avg_ac > 0:
                stats["corners"] = {
                    "league_avg_home": avg_hc, "league_avg_away": avg_ac,
                    "home_attack": (clean_c.groupby(home_col)[hc_col].mean() / avg_hc).fillna(1.0).to_dict(),
                    "away_attack": (clean_c.groupby(away_col)[ac_col].mean() / avg_ac).fillna(1.0).to_dict(),
                    "home_defence": (clean_c.groupby(home_col)[ac_col].mean() / avg_ac).fillna(1.0).to_dict(),
                    "away_defence": (clean_c.groupby(away_col)[hc_col].mean() / avg_hc).fillna(1.0).to_dict(),
                }

    # === xG ===
    if hxg_col and axg_col and hxg_col in df.columns and axg_col in df.columns:
        xg_mask = df[hxg_col].notna() & df[axg_col].notna()
        clean_xg = df[xg_mask][[home_col, away_col, hxg_col, axg_col]]
        if len(clean_xg) >= 5:
            avg_hxg = clean_xg[hxg_col].mean()
            avg_axg = clean_xg[axg_col].mean()
            if avg_hxg > 0 and avg_axg > 0:
                stats["xg"] = {
                    "league_avg_home": avg_hxg, "league_avg_away": avg_axg,
                    "home_attack": (clean_xg.groupby(home_col)[hxg_col].mean() / avg_hxg).fillna(1.0).to_dict(),
                    "away_attack": (clean_xg.groupby(away_col)[axg_col].mean() / avg_axg).fillna(1.0).to_dict(),
                    "home_defence": (clean_xg.groupby(home_col)[axg_col].mean() / avg_axg).fillna(1.0).to_dict(),
                    "away_defence": (clean_xg.groupby(away_col)[hxg_col].mean() / avg_hxg).fillna(1.0).to_dict(),
                }

    # === SHOTS ON TARGET ===
    if hs_col and as_col and hs_col in df.columns and as_col in df.columns:
        s_mask = df[hs_col].notna() & df[as_col].notna()
        clean_s = df[s_mask][[home_col, away_col, hs_col, as_col]]
        if len(clean_s) >= 5:
            avg_hs = clean_s[hs_col].mean()
            avg_as = clean_s[as_col].mean()
            if avg_hs > 0 and avg_as > 0:
                stats["shots"] = {
                    "league_avg_home": avg_hs, "league_avg_away": avg_as,
                    "home_attack": (clean_s.groupby(home_col)[hs_col].mean() / avg_hs).fillna(1.0).to_dict(),
                    "away_attack": (clean_s.groupby(away_col)[as_col].mean() / avg_as).fillna(1.0).to_dict(),
                    "home_defence": (clean_s.groupby(home_col)[as_col].mean() / avg_as).fillna(1.0).to_dict(),
                    "away_defence": (clean_s.groupby(away_col)[hs_col].mean() / avg_hs).fillna(1.0).to_dict(),
                }

    return stats

# ================================
# PREDICT MATCH
# ================================
@st.cache_data(show_spinner=False)
def predict_match(home: str, away: str, stats: Dict[str, Any]) -> Dict[str, Any]:
    max_g = 10
    max_c = 15
    max_s = 20
    predictions = {
        "goals": {"score": "N/A", "result": "N/A", "home_win": 0, "draw": 0, "away_win": 0,
                  "btts_yes": 0, "btts_no": 1, "btts_result": "N/A",
                  "over_25": 0, "under_25": 1, "over_under_result": "N/A"}
    }
    chart_data = {}

    # === GOALS ===
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
                if h > a: prob_h += p
                elif h == a: prob_d += p
                else: prob_a += p
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

    # === CORNERS ===
    if "corners" in stats:
        c = stats["corners"]
        lambda_hc = c["home_attack"].get(home, 1.0) * c["away_defence"].get(away, 1.0) * c["league_avg_home"]
        lambda_ac = c["away_attack"].get(away, 1.0) * c["home_defence"].get(home, 1.0) * c["league_avg_away"]
        hc_probs = poisson.pmf(np.arange(max_c + 1), lambda_hc)
        ac_probs = poisson.pmf(np.arange(max_c + 1), lambda_ac)

        total_probs = np.zeros(max_c + 1)
        best_total = int(round(lambda_hc + lambda_ac))
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
        if best_p == 0:
            best_total = int(round(lambda_hc + lambda_ac))

        predictions["corners"] = {
            "total": best_total,
            "over_10_5": over_10_5,
            "under_10_5": 1 - over_10_5,
            "result": "Over" if over_10_5 > 0.5 else "Under",
            "distribution": total_probs.tolist()
        }
        chart_data["corner_dist"] = pd.Series(total_probs, index=range(max_c + 1))

    # === xG ===
    if "xg" in stats:
        xg = stats["xg"]
        lambda_hxg = xg["home_attack"].get(home, 1.0) * xg["away_defence"].get(away, 1.0) * xg["league_avg_home"]
        lambda_axg = xg["away_attack"].get(away, 1.0) * xg["home_defence"].get(home, 1.0) * xg["league_avg_away"]
        total_xg = lambda_hxg + lambda_axg
        over_25_xg = poisson.sf(2, total_xg)

        predictions["xg"] = {
            "home_xg": lambda_hxg,
            "away_xg": lambda_axg,
            "total_xg": total_xg,
            "over_25_xg": over_25_xg,
            "under_25_xg": 1 - over_25_xg,
            "result": "Over" if over_25_xg > 0.5 else "Under"
        }

    # === SHOTS ON TARGET ===
    if "shots" in stats:
        s = stats["shots"]
        lambda_hs = s["home_attack"].get(home, 1.0) * s["away_defence"].get(away, 1.0) * s["league_avg_home"]
        lambda_as = s["away_attack"].get(away, 1.0) * s["home_defence"].get(home, 1.0) * s["league_avg_away"]
        hs_probs = poisson.pmf(np.arange(max_s + 1), lambda_hs)
        as_probs = poisson.pmf(np.arange(max_s + 1), lambda_as)

        total_probs = np.zeros(max_s + 1)
        best_total = int(round(lambda_hs + lambda_as))
        best_p = 0.0
        over_8_5 = 0.0
        for h in range(max_s + 1):
            for a in range(max_s + 1):
                p = hs_probs[h] * as_probs[a]
                total = h + a
                if total <= max_s:
                    total_probs[total] += p
                if total > 8.5:
                    over_8_5 += p
                if p > best_p:
                    best_p = p
                    best_total = total
        if best_p == 0:
            best_total = int(round(lambda_hs + lambda_as))

        predictions["shots"] = {
            "total": best_total,
            "over_8_5": over_8_5,
            "under_8_5": 1 - over_8_5,
            "result": "Over" if over_8_5 > 0.5 else "Under",
            "distribution": total_probs.tolist()
        }
        chart_data["shots_dist"] = pd.Series(total_probs, index=range(max_s + 1))

    # === GOAL TIMING ===
    if "goal_timing" in stats and "goals" in stats:
        t = stats["goal_timing"]
        lambda_home = g["home_attack"].get(home, 1.0) * g["away_defence"].get(away, 1.0) * g["league_avg_home"]
        lambda_away = g["away_attack"].get(away, 1.0) * g["home_defence"].get(home, 1.0) * g["league_avg_away"]
        total_lambda = lambda_home + lambda_away
        expected = np.array(t["prob"]) * total_lambda
        most_likely_idx = np.argmax(expected)
        predictions["goal_timing"] = {
            "intervals": t["intervals"],
            "expected_goals": expected.tolist(),
            "most_likely": t["intervals"][most_likely_idx]
        }

    predictions["chart_data"] = chart_data
    return predictions

# ================================
# EXPORT TO PDF - UNIVERSAL VERSION
# ================================
def export_to_pdf(html_content: str, filename: str = "prediction.pdf"):
    if PDF_LIBRARY == "xhtml2pdf":
        try:
            # Create PDF using xhtml2pdf
            result = BytesIO()
            pdf = pisa.CreatePDF(BytesIO(html_content.encode('utf-8')), result)
            
            if not pdf.err:
                return result.getvalue()
            else:
                st.error(f"xhtml2pdf error: {pdf.err}")
                return None
                
        except Exception as e:
            st.error(f"xhtml2pdf failed: {str(e)}")
            return None
            
    elif PDF_LIBRARY == "weasyprint":
        try:
            with tempfile.NamedTemporaryFile(delete=False, suffix=".html", mode='w', encoding='utf-8') as f:
                f.write(html_content)
                html_path = f.name
            
            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as f:
                pdf_path = f.name
            
            HTML(html_path).write_pdf(pdf_path)
            
            with open(pdf_path, "rb") as f:
                pdf_bytes = f.read()
            
            os.unlink(html_path)
            os.unlink(pdf_path)
            
            return pdf_bytes
            
        except Exception as e:
            st.error(f"WeasyPrint failed: {str(e)}")
            return None
            
    elif PDF_LIBRARY == "pdfkit":
        try:
            import subprocess
            try:
                subprocess.run(['wkhtmltopdf', '--version'], capture_output=True, check=True)
                wkhtmltopdf_available = True
            except (subprocess.CalledProcessError, FileNotFoundError):
                wkhtmltopdf_available = False
                
            if wkhtmltopdf_available:
                with tempfile.NamedTemporaryFile(delete=False, suffix=".html") as f:
                    f.write(html_content.encode('utf-8'))
                    html_path = f.name
                pdf_path = html_path.replace(".html", ".pdf")
                
                pdfkit.from_file(html_path, pdf_path)
                
                with open(pdf_path, "rb") as f:
                    pdf_bytes = f.read()
                
                os.unlink(html_path)
                os.unlink(pdf_path)
                
                return pdf_bytes
            else:
                st.error("wkhtmltopdf not available")
                return None
                
        except Exception as e:
            st.error(f"PDFKit failed: {str(e)}")
            return None
    else:
        st.error("No PDF library available")
        return None

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
    hs_col   = o5.selectbox("HS (Home Shots)", [""] + list(df.columns))
    as_col   = o6.selectbox("AS (Away Shots)", [""] + list(df.columns))
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
            s = pred.get("shots", {})
            t = pred.get("goal_timing", {})
            chart_data = pred.get("chart_data", {})

            logo1 = get_team_logo(home_team)
            logo2 = get_team_logo(away_team)
            img1 = load_image(logo1) if logo1 else None
            img2 = load_image(logo2) if logo2 else None

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

            if c and "distribution" in c:
                st.markdown("### Corner Kicks Distribution")
                fig = px.bar(x=range(len(c["distribution"])), y=c["distribution"],
                            labels=dict(x="Total Corners", y="Probability"),
                            title=f"Most Likely: {c['total']} | Over 10.5: {c['over_10_5']:.1%}")
                fig.update_layout(height=400)
                st.plotly_chart(fig, use_container_width=True)

            if x:
                st.markdown("### Expected Goals (xG)")
                fig = go.Figure(data=[
                    go.Bar(name='Home xG', x=['xG'], y=[x['home_xg']], marker_color='lightgreen'),
                    go.Bar(name='Away xG', x=['xG'], y=[x['away_xg']], marker_color='salmon')
                ])
                fig.add_hline(y=2.5, line_dash="dash", line_color="gray")
                fig.update_layout(barmode='stack', height=400, title_text=f"Total: {x['total_xg']:.2f} | Over 2.5: {x['over_25_xg']:.1%}")
                st.plotly_chart(fig, use_container_width=True)

            if s and "distribution" in s:
                st.markdown("### Shots on Target Distribution")
                fig = px.bar(x=range(len(s["distribution"])), y=s["distribution"],
                            labels=dict(x="Total Shots", y="Probability"),
                            title=f"Most Likely: {s['total']} | Over 8.5: {s['over_8_5']:.1%}")
                fig.update_layout(height=400)
                st.plotly_chart(fig, use_container_width=True)

            if t:
                st.markdown("### Goal Timing (15-min Intervals)")
                fig = go.Figure(data=[
                    go.Bar(x=t["intervals"], y=t["expected_goals"],
                           text=[f"{g:.2f}" for g in t["expected_goals"]], textposition="auto",
                           marker_color="#FF6B6B")
                ])
                fig.update_layout(title=f"<b>Most Likely: {t['most_likely']}</b>",
                                  xaxis_title="Time Interval", yaxis_title="Expected Goals", height=450)
                st.plotly_chart(fig, use_container_width=True)

            # === PRINT + EXPORT ===
            st.markdown(print_css, unsafe_allow_html=True)
            st.markdown("### Prediction Summary (Print or Export)")

            def img_to_base64(img):
                if not img: return ""
                buffered = BytesIO()
                img.save(buffered, format="PNG")
                return base64.b64encode(buffered.getvalue()).decode()

            print_html = f"""
            <div class="print-title">{home_team} vs {away_team}</div>
            <div style="display:flex; justify-content:center; gap:50px; margin:20px 0;">
                <div class="team-box">
                    {f'<img src="data:image/png;base64,{img_to_base64(img1)}" class="logo">' if img1 else f'<b>{home_team}</b>'}
                </div>
                <div style="font-size:24px; font-weight:bold; align-self:center;">VS</div>
                <div class="team-box">
                    {f'<img src="data:image/png;base64,{img_to_base64(img2)}" class="logo">' if img2 else f'<b>{away_team}</b>'}
                </div>
            </div>
            """

            if g.get('score') != 'N/A':
                print_html += f"""
                <div class="prediction">
                    <div class="score">Full-Time: {g['score']} to {g['result']}</div>
                    <div class="prob">H: {g['home_win']:.1%} | D: {g['draw']:.1%} | A: {g['away_win']:.1%}</div>
                </div>
                """
                if g.get('btts_result'):
                    print_html += f"""
                    <div class="prediction">
                        <div class="score">BTTS: {g['btts_result']}</div>
                        <div class="prob">Yes: {g['btts_yes']:.1%} | No: {g['btts_no']:.1%}</div>
                    </div>
                    """
                if g.get('over_under_result'):
                    print_html += f"""
                    <div class="prediction">
                        <div class="score">Over/Under 2.5: {g['over_under_result']} 2.5</div>
                        <div class="prob">Over: {g['over_25']:.1%} | Under: {g['under_25']:.1%}</div>
                    </div>
                    """

            if c:
                total_c = c.get('total')
                if total_c is not None:
                    print_html += f"""
                    <div class="prediction">
                        <div class="score">Corners: {total_c} (Most Likely)</div>
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

            if s:
                total_s = s.get('total')
                if total_s is not None:
                    print_html += f"""
                    <div class="prediction">
                        <div class="score">Shots on Target: {total_s} (Most Likely)</div>
                        <div class="prob">Over 8.5: {s['over_8_5']:.1%} | Under: {s['under_8_5']:.1%}</div>
                    </div>
                    """

            if t:
                eg = t["expected_goals"]
                print_html += f"""
                <div class="prediction">
                    <div class="score">Goal Timing: <b>{t['most_likely']}</b></div>
                    <div class="prob" style="font-size:13px; line-height:1.4;">
                        1–15: {eg[0]:.2f} 16–30: {eg[1]:.2f} 31–45: {eg[2]:.2f}<br>
                        46–60: {eg[3]:.2f} 61–75: {eg[4]:.2f} 76–90: {eg[5]:.2f}
                    </div>
                </div>
                """

            st.markdown(print_html, unsafe_allow_html=True)

            # EXPORT TO PDF BUTTON
            if PDF_AVAILABLE:
                if st.button("Export to PDF"):
                    full_html = f"""
                    <!DOCTYPE html><html><head><meta charset="utf-8">
                    <style>
                        body {{ font-family: Arial; padding: 40px; }}
                        .title {{ font-size: 28px; text-align: center; font-weight: bold; margin-bottom: 20px; }}
                        .team {{ text-align: center; }}
                        .prediction {{ margin: 20px 0; padding: 15px; border: 1px solid #ccc; border-radius: 8px; background: #f9f9f9; }}
                        .score {{ font-weight: bold; }}
                        .prob {{ font-size: 13px; color: #555; }}
                    </style></head><body>
                    {print_html}
                    </body></html>
                    """
                    with st.spinner("Generating PDF..."):
                        pdf_bytes = export_to_pdf(full_html)
                        if pdf_bytes:
                            st.download_button(
                                label="Download PDF",
                                data=pdf_bytes,
                                file_name=f"{home_team}_vs_{away_team}_prediction.pdf",
                                mime="application/pdf"
                            )
            else:
                st.warning("PDF export is not available. Install xhtml2pdf in requirements.txt")

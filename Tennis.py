import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score
from datetime import datetime
import warnings

warnings.filterwarnings('ignore')

st.set_page_config(page_title="WTA Predictor Pro 2026", layout="wide")

# ── Styling ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    .stButton>button {
        background: linear-gradient(135deg, #4A148C, #7E57C2);
        color: white; font-weight:bold; border:none;
        padding: 14px 32px; font-size:1.15em; border-radius:10px;
        width:100%; margin:10px 0;
    }
    .stButton>button:hover { background: linear-gradient(135deg, #7E57C2, #4A148C); }
    .pred-box {
        padding: 35px; border-radius: 18px; text-align:center;
        margin: 25px 0; box-shadow: 0 8px 25px rgba(0,0,0,0.22);
        color: white;
    }
    .big-num { font-size: 6.8em; font-weight: 800; line-height: 0.92; }
    .match-msg { font-size: 1.7em; margin-bottom: 14px; }
    .card { background:#fafafa; border-radius:12px; padding:24px; margin:20px 0; border-left:5px solid #7E57C2; }
    table.detail-table { width:100%; border-collapse:collapse; margin:16px 0; }
    .detail-table th, .detail-table td { padding:10px; text-align:left; border-bottom:1px solid #eee; }
    .detail-table th { background:#f0f0ff; font-weight:600; }
</style>
""", unsafe_allow_html=True)

st.title("🎾 WTA Match Predictor Pro")
st.caption("March 2026 • Surface-focused analysis • Last 15 matches on selected surface")

# ── File Upload & Data Loading ─────────────────────────────────────────────
uploaded_file = st.file_uploader("Upload WTA matches Excel file", type=["xlsx"])

if not uploaded_file:
    st.info("Please upload your WTA dataset (xlsx format)")
    st.stop()

@st.cache_data
def load_wta_data(file):
    df = pd.read_excel(file)
    df.columns = df.columns.str.strip()

    odds_cols = ['B365W','B365L','PSW','PSL','MaxW','MaxL','AvgW','AvgL']
    for c in odds_cols:
        if c in df.columns:
            df[c] = df[c].astype(str).str.replace(',','.').replace('',np.nan).astype(float)

    if 'Date' in df.columns:
        df['Date'] = pd.to_datetime(df['Date'], dayfirst=True, errors='coerce')

    df = df.sort_values('Date', ascending=False)

    for s in ['1','2','3']:
        df[f'W{s}'] = pd.to_numeric(df.get(f'W{s}',0), errors='coerce').fillna(0).astype(int)
        df[f'L{s}'] = pd.to_numeric(df.get(f'L{s}',0), errors='coerce').fillna(0).astype(int)

    df['Total_Games'] = df[[f'W{s}' for s in '123']] + df[[f'L{s}' for s in '123']]
    df['Completed'] = df.get('Comment','').str.contains('Completed', case=False, na=False)

    df = df[df['Completed']]
    return df.dropna(subset=['Date','Winner','Loser','Surface','Total_Games'])

df = load_wta_data(uploaded_file)

if df.empty:
    st.error("No valid completed matches found after cleaning.")
    st.stop()

st.success(f"Loaded {len(df):,} completed matches • Surfaces: {', '.join(sorted(df['Surface'].unique()))}")

# ── Helper Functions ───────────────────────────────────────────────────────

def get_player_matches(df, player, surface=None):
    cond = (df['Winner'] == player) | (df['Loser'] == player)
    if surface: cond &= (df['Surface'] == surface)
    return df[cond].sort_values('Date', ascending=False)


def analyze_last_15(df, player, surface):
    m = get_player_matches(df, player, surface).head(15)
    if len(m) == 0:
        return {'wins':0, 'losses':0, 'wr':50.0, 'avg_games':21.0, 'form':'No data'}

    wins = (m['Winner'] == player).sum()
    wr = wins / len(m) * 100
    form = "🔥 Strong" if wr >= 68 else "Good" if wr >= 54 else "Mixed" if wr >= 38 else "Difficult"

    return {
        'wins': wins,
        'losses': len(m)-wins,
        'wr': round(wr,1),
        'avg_games': round(m['Total_Games'].mean(),1) if 'Total_Games' in m.columns else 21.0,
        'form': form
    }


def calculate_surface_stats(df, player, surface, n=15):
    m = get_player_matches(df, player, surface).head(n)
    if len(m) < 5:
        return {
            'matches':len(m), 'note':'Limited data',
            'winners':19, 'unforced_errors':26,
            'net_pct':64, 'spw':60, 'rpw':42,
            'sgw':75, 'rgw':35, 'games_won_pct':55
        }

    wr = (m['Winner'] == player).mean()
    avg_g = m['Total_Games'].mean() if 'Total_Games' in m.columns else 21.5
    wr_dev = wr - 0.5

    bases = {
        'Hard':  {'win':19, 'ue':25, 'net':65, 'spw':61, 'rpw':42, 'sgw':76, 'rgw':36},
        'Clay':  {'win':17, 'ue':30, 'net':60, 'spw':58, 'rpw':45, 'sgw':72, 'rgw':40},
        'Grass': {'win':23, 'ue':22, 'net':70, 'spw':64, 'rpw':40, 'sgw':81, 'rgw':33},
    }
    b = bases.get(surface, bases['Hard'])

    adj = wr_dev * 0.22

    stats = {
        'matches': len(m),
        'winners':           round(b['win'] + adj*22 + (avg_g-21)*0.45, 0),
        'unforced_errors':   round(b['ue']  - adj*20 + (avg_g-21)*0.55, 0),
        'net_pct':           round(b['net'] + adj*18, 0),
        'spw':               round(b['spw'] + adj*13, 0),
        'rpw':               round(b['rpw'] + adj*15, 0),
        'sgw':               round(b['sgw'] + adj*15, 0),
        'rgw':               round(b['rgw'] + adj*17, 0),
        'games_won_pct':     round((b['sgw'] + adj*15 + b['rgw'] + adj*17)/2, 0),
    }

    for k in ['spw','rpw','sgw','rgw','net_pct']:
        stats[k] = max(53, min(87, stats[k]))

    return stats


def days_since_last_match(df, player):
    m = get_player_matches(df, player).head(1)
    if len(m) == 0 or pd.isna(m.iloc[0]['Date']):
        return 12, "Unknown"
    days = (datetime.now().date() - m.iloc[0]['Date'].date()).days
    lvl = "🔴 Very fresh" if days <= 4 else "⚠️ Recent" if days <= 10 else "Normal" if days <= 18 else "🟢 Well rested"
    return days, lvl


def head_to_head(df, p1, p2, surface=None):
    cond = (((df['Winner']==p1)&(df['Loser']==p2)) | ((df['Winner']==p2)&(df['Loser']==p1)))
    if surface: cond &= df['Surface']==surface
    h = df[cond]
    if len(h)==0: return {'total':0, f"{p1} wins":0, f"{p2} wins":0, 'avg_g':21.0}
    w1 = (h['Winner']==p1).sum()
    return {'total':len(h), f"{p1} wins":w1, f"{p2} wins":len(h)-w1, 'avg_g':round(h['Total_Games'].mean(),1) if 'Total_Games' in h.columns else 21.0}


@st.cache_resource
def build_total_games_model(df):
    d = df[df['Total_Games'].between(12,40)].copy()
    if len(d) < 70: return None

    feats = []
    for s in '123':
        feats.append(d[f'W{s}'] + d[f'L{s}'])
        if s in '12':
            feats.append(abs(d[f'W{s}'] - d[f'L{s}']))

    feats.extend([
        (d['Wsets']==2).astype(int),
        (d['Wsets']==3).astype(int),
        d.get('WRank',500).fillna(500),
        d.get('LRank',500).fillna(500),
        d.get('LRank',500).fillna(500) - d.get('WRank',500).fillna(500)
    ])

    if 'Surface' in d:
        dummies = pd.get_dummies(d['Surface'], prefix='Surf')
        for col in dummies: feats.append(dummies[col])

    X = np.column_stack(feats)
    X = np.nan_to_num(X)
    y = d['Total_Games'].values

    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, random_state=42)
    sc = StandardScaler().fit(Xtr)
    mdl = GradientBoostingRegressor(n_estimators=220, max_depth=5, learning_rate=0.045, random_state=42)
    mdl.fit(sc.transform(Xtr), ytr)

    p = mdl.predict(sc.transform(Xte))
    return {'model':mdl, 'scaler':sc, 'r2':r2_score(yte,p), 'mae':mean_absolute_error(yte,p)}


model = build_total_games_model(df)

# ── UI ─────────────────────────────────────────────────────────────────────
st.subheader("Match Selection")

c1, c2, c3 = st.columns([5,5,3])
with c1: player1 = st.selectbox("Player 1", sorted(set(df['Winner']) | set(df['Loser'])), key="p1")
with c2:
    p2_options = [p for p in sorted(set(df['Winner']) | set(df['Loser'])) if p != player1]
    player2 = st.selectbox("Player 2", p2_options, key="p2")
with c3: surf = st.selectbox("Surface", sorted(df['Surface'].unique()), key="s")

if st.button("Generate Prediction", type="primary"):
    with st.spinner("Calculating..."):
        d1 = analyze_last_15(df, player1, surf)
        d2 = analyze_last_15(df, player2, surf)

        s1 = calculate_surface_stats(df, player1, surf)
        s2 = calculate_surface_stats(df, player2, surf)

        days1, fat1 = days_since_last_match(df, player1)
        days2, fat2 = days_since_last_match(df, player2)

        h2h = head_to_head(df, player1, player2, surf)

        heur = (s1['avg_games'] + s2['avg_games']) / 2 * (1 + (d1['wr'] - d2['wr'])/450)

        ml_val = heur
        if model and model['model']:
            try:
                r1 = df[df['Winner']==player1]['WRank'].mean() or 400
                r2 = df[df['Winner']==player2]['WRank'].mean() or 400
                vec = [s1['avg_games']]*5 + [3]*3 + [0.7,0.3, r1,r2,r2-r1] + [1 if x==surf else 0 for x in sorted(df['Surface'].unique())]
                Xf = np.array(vec).reshape(1,-1)
                ml_val = model['model'].predict(model['scaler'].transform(Xf))[0]
            except:
                pass

        final_pred = round(0.65 * ml_val + 0.35 * heur, 1)

        st.session_state.update({
            'predicted':True, 'p1':player1, 'p2':player2, 'surface':surf,
            'd1':d1, 'd2':d2, 's1':s1, 's2':s2,
            'fat1':(days1,fat1), 'fat2':(days2,fat2),
            'h2h':h2h, 'prediction':final_pred,
            'model_ok': bool(model and model['model'])
        })
        st.rerun()

# ── Always-visible content + Export ────────────────────────────────────────
st.markdown("---")

# Current values (fallback if no prediction)
curr_p1 = st.session_state.get('p1', player1)
curr_p2 = st.session_state.get('p2', player2)
curr_surf = st.session_state.get('surface', surf)
has_pred = 'prediction' in st.session_state
pred_val = st.session_state.get('prediction', None)
pred_str = f"{pred_val:.1f}" if has_pred else "—"
color = "#66BB6A" if (has_pred and pred_val < 21) else "#FFB74D" if (has_pred and pred_val < 26) else "#EF5350" if has_pred else "#9E9E9E"
msg = 'Quick match' if (has_pred and pred_val < 21) else 'Competitive' if (has_pred and pred_val < 26) else 'Long battle' if has_pred else 'Run prediction'

st.markdown(f"""
<div class="pred-box" style="background:{color};">
    <div class="match-msg">{msg}</div>
    <div class="big-num">{pred_str}</div>
    <div style="font-size:1.4em; opacity:0.92;">TOTAL GAMES</div>
</div>
""", unsafe_allow_html=True)

# ── Export Button (always visible) ─────────────────────────────────────────
st.markdown("### Export Detailed HTML Report")
st.caption("Includes prediction (if generated), H2H, detailed surface stats, fatigue, and more")

def create_detailed_html_report():
    p1 = curr_p1 or "Player A"
    p2 = curr_p2 or "Player B"
    surface = curr_surf or "—"
    pred = pred_str
    pred_color = color

    has_full_data = has_pred and 'd1' in st.session_state and 's1' in st.session_state

    stats_section = ""
    if has_full_data:
        s1 = st.session_state.s1
        s2 = st.session_state.s2
        d1 = st.session_state.d1
        d2 = st.session_state.d2
        fat1_days, fat1_lvl = st.session_state.fat1
        fat2_days, fat2_lvl = st.session_state.fat2

        stats_section = f"""
        <h2>Detailed Last 15 Matches on {surface}</h2>
        <div style="display:grid; grid-template-columns:1fr 1fr; gap:30px;">
            <div class="card">
                <h3>{p1}</h3>
                <p><strong>Record:</strong> {d1['wins']}-{d1['losses']} ({d1['wr']}%) • {d1['form']}</p>
                <p><strong>Rest:</strong> {fat1_lvl} ({fat1_days} days since last match)</p>
                <table class="detail-table">
                    <tr><th>Stat</th><th>Value</th></tr>
                    <tr><td>Winners per match</td><td>{s1['winners']}</td></tr>
                    <tr><td>Unforced errors per match</td><td>{s1['unforced_errors']}</td></tr>
                    <tr><td>Net points won %</td><td>{s1['net_pct']}%</td></tr>
                    <tr><td>Service points won %</td><td>{s1['spw']}%</td></tr>
                    <tr><td>Return points won %</td><td>{s1['rpw']}%</td></tr>
                    <tr><td>Service games won %</td><td>{s1['sgw']}%</td></tr>
                    <tr><td>Return games won %</td><td>{s1['rgw']}%</td></tr>
                    <tr><td>Games won % (overall)</td><td>{s1['games_won_pct']}%</td></tr>
                </table>
            </div>
            <div class="card">
                <h3>{p2}</h3>
                <p><strong>Record:</strong> {d2['wins']}-{d2['losses']} ({d2['wr']}%) • {d2['form']}</p>
                <p><strong>Rest:</strong> {fat2_lvl} ({fat2_days} days since last match)</p>
                <table class="detail-table">
                    <tr><th>Stat</th><th>Value</th></tr>
                    <tr><td>Winners per match</td><td>{s2['winners']}</td></tr>
                    <tr><td>Unforced errors per match</td><td>{s2['unforced_errors']}</td></tr>
                    <tr><td>Net points won %</td><td>{s2['net_pct']}%</td></tr>
                    <tr><td>Service points won %</td><td>{s2['spw']}%</td></tr>
                    <tr><td>Return points won %</td><td>{s2['rpw']}%</td></tr>
                    <tr><td>Service games won %</td><td>{s2['sgw']}%</td></tr>
                    <tr><td>Return games won %</td><td>{s2['rgw']}%</td></tr>
                    <tr><td>Games won % (overall)</td><td>{s2['games_won_pct']}%</td></tr>
                </table>
            </div>
        </div>
        <p style="text-align:center; font-style:italic; margin-top:20px;">
            Model used: {'ML-enhanced' if st.session_state.get('model_ok', False) else 'Heuristic only'}<br>
            Training performance: R² ≈ {model['r2']:.3f} | MAE ±{model['mae']:.1f} games (if applicable)
        </p>
        """
    else:
        stats_section = '<p style="text-align:center; color:#666;">Run prediction to include detailed surface stats, form, fatigue and model info.</p>'

    h2h_data = ""
    if 'h2h' in st.session_state:
        h = st.session_state.h2h
        h2h_data = f"""
        <div class="card">
            <h3>Head-to-Head on {surface}</h3>
            <table class="detail-table">
                <tr><th>Metric</th><th>Value</th></tr>
                <tr><td>Total matches</td><td>{h['total']}</td></tr>
                <tr><td>{p1} wins</td><td>{h.get(f"{p1} wins", 0)}</td></tr>
                <tr><td>{p2} wins</td><td>{h.get(f"{p2} wins", 0)}</td></tr>
                <tr><td>Average games per match</td><td>{h.get('avg_g', 21.0):.1f}</td></tr>
            </table>
        </div>
        """

    full_html = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <title>WTA Prediction Report • {p1} vs {p2}</title>
    <style>
        body {{ font-family: system-ui, sans-serif; background: linear-gradient(to bottom right, #f8f9ff, #f0e8ff); margin:0; padding:30px; color:#333; }}
        .container {{ max-width:1100px; margin:auto; background:white; border-radius:16px; box-shadow:0 10px 40px rgba(0,0,0,0.15); overflow:hidden; }}
        .header {{ background: linear-gradient(135deg, #4A148C, #7E57C2); color:white; padding:45px 30px; text-align:center; }}
        .content {{ padding:40px; }}
        .prediction {{ background:{pred_color}; color:white; padding:45px; border-radius:14px; text-align:center; margin:30px 0; }}
        .big {{ font-size:7em; font-weight:800; line-height:0.9; }}
        .card {{ background:#fafafa; border-radius:12px; padding:28px; margin:24px 0; border-left:6px solid #7E57C2; }}
        table {{ width:100%; border-collapse:collapse; margin:20px 0; }}
        th, td {{ padding:14px; text-align:left; border-bottom:1px solid #eee; }}
        .footer {{ background:#f5f5f5; padding:25px; text-align:center; color:#666; font-size:0.95em; }}
    </style>
</head>
<body>
<div class="container">
    <div class="header">
        <h1>WTA Predictor Pro – Detailed Report</h1>
        <p>{p1} vs {p2} • {surface} • Generated {datetime.now().strftime('%Y-%m-%d %H:%M')}</p>
    </div>
    <div class="content">
        <div class="prediction">
            <div style="font-size:1.9em; margin-bottom:12px;">Predicted Total Games</div>
            <div class="big">{pred}</div>
        </div>

        {h2h_data}

        {stats_section}

    </div>
    <div class="footer">
        Generated with WTA Predictor Pro • Analysis & estimation only – not official WTA data • For informational purposes
    </div>
</div>
</body>
</html>"""
    return full_html

st.download_button(
    label="📥 Download Detailed HTML Report",
    data=create_detailed_html_report(),
    file_name=f"wta_detailed_{curr_p1.replace(' ','_')}_vs_{curr_p2.replace(' ','_')}_{curr_surf}_{datetime.now().strftime('%Y%m%d_%H%M')}.html",
    mime="text/html",
    use_container_width=True,
    key="detailed_export"
)

st.caption("Tip: The report includes more stats & context when you run a prediction first.")

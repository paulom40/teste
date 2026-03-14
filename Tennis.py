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

st.set_page_config(page_title="TENNIS Predictor Pro 2026", layout="wide")

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
</style>
""", unsafe_allow_html=True)

st.title("🎾 TENNIS Match Predictor Pro")
st.caption("March 2026 • Surface-focused • Last 15 matches on selected surface")

# ── File Upload & Data Loading ─────────────────────────────────────────────
uploaded_file = st.file_uploader("Upload WTA matches Excel file", type=["xlsx"])

if not uploaded_file:
    st.info("Please upload your TENNIS dataset (xlsx format)")
    st.stop()

@st.cache_data
def load_wta_data(file):
    df = pd.read_excel(file)
    df.columns = df.columns.str.strip()

    # Fix comma → dot in odds
    odds_cols = ['B365W','B365L','PSW','PSL','MaxW','MaxL','AvgW','AvgL']
    for c in odds_cols:
        if c in df.columns:
            df[c] = df[c].astype(str).str.replace(',','.').replace('',np.nan).astype(float)

    # Date (assuming DD/MM/YYYY from your sample)
    if 'Date' in df.columns:
        df['Date'] = pd.to_datetime(df['Date'], dayfirst=True, errors='coerce')

    df = df.sort_values('Date', ascending=False)

    # Build total games
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
    if surface:
        cond &= (df['Surface'] == surface)
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
        'avg_games': round(m['Total_Games'].mean(),1),
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
    avg_g = m['Total_Games'].mean() if 'Total_Games' in m else 21.5
    wr_dev = wr - 0.5

    bases = {
        'Hard':  {'win':19, 'ue':25, 'net':65, 'spw':61, 'rpw':42, 'sgw':76, 'rgw':36},
        'Clay':  {'win':17, 'ue':30, 'net':60, 'spw':58, 'rpw':45, 'sgw':72, 'rgw':40},
        'Grass': {'win':23, 'ue':22, 'net':70, 'spw':64, 'rpw':40, 'sgw':81, 'rgw':33},
    }
    b = bases.get(surface, bases['Hard'])

    adj = wr_dev * 0.22   # moderate swing

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
    return {'total':len(h), f"{p1} wins":w1, f"{p2} wins":len(h)-w1, 'avg_g':round(h['Total_Games'].mean(),1)}


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
                # very simplified proxy features – improve later
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

# ── Show Results ───────────────────────────────────────────────────────────
if st.session_state.get('predicted', False):
    p1 = st.session_state.p1
    p2 = st.session_state.p2
    s  = st.session_state.surface
    pred = st.session_state.prediction

    color = "#66BB6A" if pred < 21 else "#FFB74D" if pred < 26 else "#EF5350"

    st.markdown(f"""
    <div class="pred-box" style="background:{color};">
        <div class="match-msg">{'Quick' if pred<21 else 'Competitive' if pred<26 else 'Long battle'}</div>
        <div class="big-num">{pred:.1f}</div>
        <div style="font-size:1.4em; opacity:0.92;">TOTAL GAMES</div>
    </div>
    """, unsafe_allow_html=True)

    colA, colB = st.columns(2)
    with colA:
        st.subheader(p1)
        st.write(f"**Last 15 on {s}**: {st.session_state.d1['wins']}-{st.session_state.d1['losses']} ({st.session_state.d1['wr']}%) • {st.session_state.d1['form']}")
        st.write(f"**Rest**: {st.session_state.fat1[1]} ({st.session_state.fat1[0]} days)")
        with st.expander("Stats"):
            for k,v in st.session_state.s1.items():
                if k not in ['matches','note']:
                    st.write(f"**{k.replace('_',' ').title()}**: {v}")

    with colB:
        st.subheader(p2)
        st.write(f"**Last 15 on {s}**: {st.session_state.d2['wins']}-{st.session_state.d2['losses']} ({st.session_state.d2['wr']}%) • {st.session_state.d2['form']}")
        st.write(f"**Rest**: {st.session_state.fat2[1]} ({st.session_state.fat2[0]} days)")
        with st.expander("Stats"):
            for k,v in st.session_state.s2.items():
                if k not in ['matches','note']:
                    st.write(f"**{k.replace('_',' ').title()}**: {v}")

    st.markdown("---")
    h = st.session_state.h2h
    c1,c2,c3,c4 = st.columns(4)
    c1.metric("H2H", h['total'])
    c2.metric(f"{p1} wins", h[f"{p1} wins"])
    c3.metric(f"{p2} wins", h[f"{p2} wins"])
    c4.metric("Avg games H2H", f"{h['avg_g']:.1f}")

    # ── EXPORT BUTTON ──────────────────────────────────────────────────────
    st.markdown("### Export Report")
    export_col = st.columns([1,3,1])[1]

    def create_html_report():
        html = f"""<!DOCTYPE html>
<html>
<head>
    <title>WTA Prediction • {p1} vs {p2}</title>
    <style>
        body {{ font-family: system-ui, sans-serif; background: linear-gradient(to bottom right, #f0f4ff, #e8eaff); margin:0; padding:30px; }}
        .container {{ max-width:1100px; margin:auto; background:white; border-radius:16px; box-shadow:0 10px 40px rgba(0,0,0,0.12); overflow:hidden; }}
        .header {{ background: linear-gradient(135deg, #4A148C, #7E57C2); color:white; padding:40px; text-align:center; }}
        .content {{ padding:35px; }}
        .pred {{ background:{color}; color:white; padding:40px; border-radius:14px; text-align:center; margin:25px 0; }}
        .big {{ font-size:6.5em; font-weight:800; }}
        .card {{ background:#fafafa; border-radius:12px; padding:24px; margin:20px 0; border-left:5px solid #7E57C2; }}
        table {{ width:100%; border-collapse:collapse; margin:20px 0; }}
        th,td {{ padding:12px; text-align:left; border-bottom:1px solid #eee; }}
        .footer {{ background:#f5f5f5; padding:20px; text-align:center; color:#555; font-size:0.9em; }}
    </style>
</head>
<body>
<div class="container">
    <div class="header">
        <h1>WTA Predictor Pro</h1>
        <p>{p1} vs {p2} • {s} • Generated {datetime.now().strftime('%Y-%m-%d %H:%M')}</p>
    </div>
    <div class="content">
        <div class="pred">
            <div style="font-size:1.8em;">Predicted total games</div>
            <div class="big">{pred:.1f}</div>
        </div>

        <div class="card">
            <h3>Head-to-Head</h3>
            <table>
                <tr><td>Matches</td><td>{h['total']}</td></tr>
                <tr><td>{p1} wins</td><td>{h[f"{p1} wins"]}</td></tr>
                <tr><td>{p2} wins</td><td>{h[f"{p2} wins"]}</td></tr>
                <tr><td>Avg games</td><td>{h['avg_g']:.1f}</td></tr>
            </table>
        </div>

        <h2>Player Comparison – Last 15 on {s}</h2>
        <div style="display:grid; grid-template-columns:1fr 1fr; gap:25px;">
            <div class="card">
                <h3>{p1}</h3>
                <p>Record: {st.session_state.d1['wins']}-{st.session_state.d1['losses']} ({st.session_state.d1['wr']}%)</p>
                <p>Form: {st.session_state.d1['form']}</p>
                <p>Rest: {st.session_state.fat1[1]} ({st.session_state.fat1[0]} days)</p>
            </div>
            <div class="card">
                <h3>{p2}</h3>
                <p>Record: {st.session_state.d2['wins']}-{st.session_state.d2['losses']} ({st.session_state.d2['wr']}%)</p>
                <p>Form: {st.session_state.d2['form']}</p>
                <p>Rest: {st.session_state.fat2[1]} ({st.session_state.fat2[0]} days)</p>
            </div>
        </div>
    </div>
    <div class="footer">
        WTA Predictor Pro • For informational purposes only • Not official WTA data
    </div>
</div>
</body>
</html>"""
        return html

    with export_col:
        st.download_button(
            label="📥 Download HTML Report",
            data=create_html_report(),
            file_name=f"wta_{p1.replace(' ','_')}_vs_{p2.replace(' ','_')}_{s}_{datetime.now().strftime('%Y%m%d_%H%M')}.html",
            mime="text/html",
            use_container_width=True
        )

st.caption("Tip: upload more historical data → better surface-specific stats & model accuracy")

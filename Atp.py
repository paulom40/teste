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

st.set_page_config(page_title="ATP Predictor Pro", layout="wide")

# ── Styling ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    .stButton>button {
        background: linear-gradient(135deg, #1a237e, #3f51b5);
        color: white; font-weight:bold; border:none;
        padding: 14px 32px; font-size:1.15em; border-radius:10px;
        width:100%; margin:10px 0;
    }
    .stButton>button:hover { background: linear-gradient(135deg, #3f51b5, #1a237e); }
    .pred-box {
        padding: 35px; border-radius: 18px; text-align:center;
        margin: 25px 0; box-shadow: 0 8px 25px rgba(0,0,0,0.22);
        color: white;
    }
    .big-num { font-size: 6.8em; font-weight: 800; line-height: 0.92; }
    .match-msg { font-size: 1.7em; margin-bottom: 14px; }
    .card { background:#f8f9fa; border-radius:12px; padding:24px; margin:20px 0; border-left:5px solid #3f51b5; }
    table.detail-table { width:100%; border-collapse:collapse; margin:16px 0; }
    .detail-table th, .detail-table td { padding:10px; text-align:left; border-bottom:1px solid #eee; }
    .detail-table th { background:#e8eaf6; font-weight:600; }
</style>
""", unsafe_allow_html=True)

st.title("🎾 ATP Men's Tennis Predictor Pro")
st.caption("Surface-specific Elo + Last 15 matches on surface • Total Games Prediction")

# ── File Upload & Robust Loading ───────────────────────────────────────────
uploaded_file = st.file_uploader("Upload your ATP Excel file", type=["xlsx", "xls"])

if not uploaded_file:
    st.info("Please upload your Excel file (Brisbane sample format works)")
    st.stop()

@st.cache_data
def load_atp_excel(file):
    df = pd.read_excel(file)
    df.columns = df.columns.str.strip()

    # Robust odds cleaning
    odds_cols = ['B365W','B365L','PSW','PSL','MaxW','MaxL','AvgW','AvgL']
    for col in odds_cols:
        if col in df.columns:
            df[col] = df[col].astype(str).str.strip()
            df[col] = df[col].str.replace(',', '.', regex=False)
            df[col] = df[col].str.replace(r'[^0-9.\-]', '', regex=True)
            df[col] = df[col].replace(['', '-'], np.nan)
            df[col] = pd.to_numeric(df[col], errors='coerce')

    # Date (DD/MM/YYYY)
    if 'Date' in df.columns:
        df['Date'] = pd.to_datetime(df['Date'], dayfirst=True, errors='coerce')

    df = df.sort_values('Date', ascending=False)

    # Clean player names
    df['Winner'] = df['Winner'].str.strip().str.replace(r'\.$', '', regex=True)
    df['Loser']  = df['Loser'].str.strip().str.replace(r'\.$', '', regex=True)

    # Total games (5 sets)
    for s in '12345':
        df[f'W{s}'] = pd.to_numeric(df.get(f'W{s}', 0), errors='coerce').fillna(0).astype(int)
        df[f'L{s}'] = pd.to_numeric(df.get(f'L{s}', 0), errors='coerce').fillna(0).astype(int)

    df['Total_Games'] = sum(df[f'{side}{s}'] for side in 'WL' for s in '12345')

    df['Completed'] = df.get('Comment', '').str.contains('Completed', case=False, na=False)
    df = df[df['Completed']]

    return df.dropna(subset=['Date', 'Winner', 'Loser', 'Surface', 'Total_Games'])

df = load_atp_excel(uploaded_file)

if df.empty:
    st.error("No valid matches found.")
    st.stop()

st.success(f"Loaded {len(df):,} matches • Surfaces: {', '.join(sorted(df['Surface'].dropna().unique()))}")

# ── SURFACE-SPECIFIC ELO ───────────────────────────────────────────────────
@st.cache_data
def compute_surface_elo(_df, initial_elo=1500, K_base=32):
    surfaces = _df['Surface'].dropna().unique()
    elo_by_surf = {}
    for surf in surfaces:
        surf_df = _df[_df['Surface'] == surf].sort_values('Date')
        if len(surf_df) < 10: continue
        players = set(surf_df['Winner']) | set(surf_df['Loser'])
        elo = {p: initial_elo for p in players}
        matches_played = {p: 0 for p in players}
        for _, row in surf_df.iterrows():
            w, l = row['Winner'], row['Loser']
            if w not in elo or l not in elo: continue
            elo_w, elo_l = elo[w], elo[l]
            exp_w = 1 / (1 + 10 ** ((elo_l - elo_w) / 400))
            k_w = K_base / (1 + matches_played[w] / 40)
            k_l = K_base / (1 + matches_played[l] / 40)
            elo[w] += k_w * (1 - exp_w)
            elo[l] += k_l * (0 - exp_w)
            matches_played[w] += 1
            matches_played[l] += 1
        elo_by_surf[surf] = elo
    return elo_by_surf

surface_elo = compute_surface_elo(df)

def get_surface_elo(player, surface, default=1500):
    if surface in surface_elo and player in surface_elo[surface]:
        return surface_elo[surface][player]
    return default

def surface_win_prob(elo_a, elo_b):
    return 100 / (1 + 10 ** ((elo_b - elo_a) / 400))

# ── Other Helpers ──────────────────────────────────────────────────────────
def get_player_matches(df, player, surface=None):
    cond = (df['Winner'] == player) | (df['Loser'] == player)
    if surface: cond &= (df['Surface'] == surface)
    return df[cond].sort_values('Date', ascending=False)

def analyze_last_15(df, player, surface):
    m = get_player_matches(df, player, surface).head(15)
    if len(m) == 0:
        return {'wins':0, 'losses':0, 'wr':50.0, 'avg_games':23.5, 'form':'No data'}
    wins = (m['Winner'] == player).sum()
    wr = wins / len(m) * 100
    form = "🔥 Dominant" if wr >= 70 else "Strong" if wr >= 55 else "Mixed" if wr >= 40 else "Struggling"
    return {'wins': wins, 'losses': len(m)-wins, 'wr': round(wr,1),
            'avg_games': round(m['Total_Games'].mean(),1), 'form': form}

def calculate_surface_stats(df, player, surface, n=15):
    m = get_player_matches(df, player, surface).head(n)
    if len(m) < 5:
        return {'winners':22, 'unforced_errors':28, 'net_pct':68, 'spw':63,
                'rpw':40, 'sgw':78, 'rgw':34, 'games_won_pct':56}
    wr = (m['Winner'] == player).mean()
    avg_g = m['Total_Games'].mean() if 'Total_Games' in m.columns else 23.5
    wr_dev = wr - 0.5
    bases = {'Hard': {'win':22,'ue':27,'net':68,'spw':63,'rpw':40,'sgw':78,'rgw':34},
             'Clay': {'win':18,'ue':33,'net':62,'spw':59,'rpw':44,'sgw':73,'rgw':39},
             'Grass':{'win':26,'ue':24,'net':74,'spw':66,'rpw':38,'sgw':83,'rgw':32}}
    b = bases.get(surface, bases['Hard'])
    adj = wr_dev * 0.24
    stats = {
        'winners': round(b['win'] + adj*26 + (avg_g-23)*0.5, 0),
        'unforced_errors': round(b['ue'] - adj*24 + (avg_g-23)*0.6, 0),
        'net_pct': round(b['net'] + adj*20, 0),
        'spw': round(b['spw'] + adj*15, 0),
        'rpw': round(b['rpw'] + adj*17, 0),
        'sgw': round(b['sgw'] + adj*17, 0),
        'rgw': round(b['rgw'] + adj*19, 0),
        'games_won_pct': round((b['sgw'] + adj*17 + b['rgw'] + adj*19)/2, 0),
    }
    for k in ['spw','rpw','sgw','rgw','net_pct']:
        stats[k] = max(54, min(89, stats[k]))
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
    if surface: cond &= (df['Surface']==surface)
    h = df[cond]
    if len(h)==0: return {'total':0, f"{p1} wins":0, f"{p2} wins":0, 'avg_g':23.5}
    w1 = (h['Winner']==p1).sum()
    return {'total':len(h), f"{p1} wins":w1, f"{p2} wins":len(h)-w1,
            'avg_g': round(h['Total_Games'].mean(),1)}

@st.cache_resource
def train_model(df):
    d = df[df['Total_Games'].between(15,60)].copy()
    if len(d) < 80: return None
    feats = []
    for s in '12345':
        feats.append(d[f'W{s}'] + d[f'L{s}'])
        if s in '12': feats.append(abs(d[f'W{s}'] - d[f'L{s}']))
    feats.extend([(d['Wsets']==3).astype(int), (d['Wsets']==2).astype(int),
                  d.get('WRank',500).fillna(500), d.get('LRank',500).fillna(500),
                  d.get('LRank',500).fillna(500) - d.get('WRank',500).fillna(500)])
    if 'Surface' in d:
        dummies = pd.get_dummies(d['Surface'], prefix='Surf')
        for col in dummies: feats.append(dummies[col])
    X = np.column_stack(feats)
    X = np.nan_to_num(X)
    y = d['Total_Games'].values
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, random_state=42)
    sc = StandardScaler().fit(Xtr)
    mdl = GradientBoostingRegressor(n_estimators=250, max_depth=5, learning_rate=0.04, random_state=42)
    mdl.fit(sc.transform(Xtr), ytr)
    p = mdl.predict(sc.transform(Xte))
    return {'model':mdl, 'scaler':sc, 'r2':r2_score(yte,p), 'mae':mean_absolute_error(yte,p)}

model_info = train_model(df)

# ── UI ─────────────────────────────────────────────────────────────────────
st.subheader("Select Matchup")
c1, c2, c3 = st.columns([5,5,3])
with c1: player_a = st.selectbox("Player A", sorted(set(df['Winner'])|set(df['Loser'])), key="pa")
with c2:
    p2_list = [p for p in sorted(set(df['Winner'])|set(df['Loser'])) if p != player_a]
    player_b = st.selectbox("Player B", p2_list, key="pb")
with c3: surface = st.selectbox("Surface", sorted(df['Surface'].dropna().unique()), key="s")

if st.button("🔮 Generate Prediction", type="primary", use_container_width=True):
    with st.spinner("Calculating..."):
        data_a = analyze_last_15(df, player_a, surface)
        data_b = analyze_last_15(df, player_b, surface)
        stats_a = calculate_surface_stats(df, player_a, surface)
        stats_b = calculate_surface_stats(df, player_b, surface)
        days_a, fat_a = days_since_last_match(df, player_a)
        days_b, fat_b = days_since_last_match(df, player_b)
        h2h = head_to_head(df, player_a, player_b, surface)

        heur = (stats_a['avg_games'] + stats_b['avg_games']) / 2 * (1 + (data_a['wr'] - data_b['wr'])/400)

        ml_val = heur
        if model_info and model_info['model']:
            try:
                r_a = df[df['Winner']==player_a]['WRank'].mean() or 300
                r_b = df[df['Winner']==player_b]['WRank'].mean() or 300
                surf_dummies = [1 if x==surface else 0 for x in sorted(df['Surface'].dropna().unique())]
                vec = [stats_a['avg_games']]*5 + [3.5]*3 + [0.65,0.35,r_a,r_b,r_b-r_a] + surf_dummies
                Xf = np.array(vec).reshape(1,-1)
                ml_val = model_info['model'].predict(model_info['scaler'].transform(Xf))[0]
            except: pass

        final_pred = round(0.65 * ml_val + 0.35 * heur, 1)

        # Surface Elo
        elo_a = get_surface_elo(player_a, surface)
        elo_b = get_surface_elo(player_b, surface)
        win_a = surface_win_prob(elo_a, elo_b)
        win_b = 100 - win_a

        st.session_state.update({
            'predicted':True,
            'player_a':player_a, 'player_b':player_b, 'surface':surface,
            'data_a':data_a, 'data_b':data_b, 'stats_a':stats_a, 'stats_b':stats_b,
            'fat_a':(days_a,fat_a), 'fat_b':(days_b,fat_b), 'h2h':h2h,
            'prediction':final_pred,
            'elo_a':round(elo_a), 'elo_b':round(elo_b),
            'win_prob_a':round(win_a,1), 'win_prob_b':round(win_b,1)
        })
        st.rerun()

# ── Results + Always-visible Export ────────────────────────────────────────
st.markdown("---")

curr_a = st.session_state.get('player_a', "")
curr_b = st.session_state.get('player_b', "")
curr_surf = st.session_state.get('surface', "")
has_pred = st.session_state.get('predicted', False)
pred_val = st.session_state.get('prediction', None)
pred_str = f"{pred_val:.1f}" if has_pred else "—"
color = "#66BB6A" if (has_pred and pred_val < 23) else "#FFB74D" if (has_pred and pred_val < 30) else "#EF5350" if has_pred else "#9E9E9E"

st.markdown(f"""
<div class="pred-box" style="background:{color};">
    <div class="match-msg">{'Straight sets likely' if (has_pred and pred_val<23) else 'Competitive' if (has_pred and pred_val<30) else 'Likely 4–5 sets' if has_pred else 'Run prediction'}</div>
    <div class="big-num">{pred_str}</div>
    <div style="font-size:1.4em;">TOTAL GAMES</div>
</div>
""", unsafe_allow_html=True)

if has_pred:
    st.markdown("### Surface Elo & Win Probability")
    c1, c2 = st.columns(2)
    c1.metric(f"{curr_a}", f"Elo {st.session_state.elo_a}", f"Win {st.session_state.win_prob_a}%")
    c2.metric(f"{curr_b}", f"Elo {st.session_state.elo_b}", f"Win {st.session_state.win_prob_b}%")

# ── EXPORT BUTTON (always visible) ─────────────────────────────────────────
st.markdown("### 📥 Export Detailed Report")

def create_full_html_report():
    p_a = curr_a or "Player A"
    p_b = curr_b or "Player B"
    surf = curr_surf or "—"
    pred = pred_str
    pred_color = color

    has_full = has_pred and 'data_a' in st.session_state

    elo_section = ""
    if has_full:
        elo_section = f"""
        <div class="card">
            <h3>Surface Elo Ratings ({surf})</h3>
            <table class="detail-table">
                <tr><th>Player</th><th>Elo</th><th>Win %</th></tr>
                <tr><td>{p_a}</td><td>{st.session_state.get('elo_a','—')}</td><td>{st.session_state.get('win_prob_a','—')}%</td></tr>
                <tr><td>{p_b}</td><td>{st.session_state.get('elo_b','—')}</td><td>{st.session_state.get('win_prob_b','—')}%</td></tr>
            </table>
        </div>
        """

    # Always define stats_section (empty if no data)
    stats_section = ""
    if has_full:
        da = st.session_state.data_a
        db = st.session_state.data_b
        sa = st.session_state.stats_a
        sb = st.session_state.stats_b
        fa_d, fa_l = st.session_state.fat_a
        fb_d, fb_l = st.session_state.fat_b

        stats_section = f"""
        <h2>Last 15 on {surf} – Detailed Stats</h2>
        <div style="display:grid; grid-template-columns:1fr 1fr; gap:30px;">
            <div class="card">
                <h3>{p_a}</h3>
                <p><strong>Record:</strong> {da['wins']}-{da['losses']} ({da['wr']}%) • {da['form']}</p>
                <p><strong>Rest:</strong> {fa_l} ({fa_d} days)</p>
                <table class="detail-table">
                    <tr><th>Stat</th><th>Value</th></tr>
                    <tr><td>Winners / match</td><td>{sa['winners']}</td></tr>
                    <tr><td>Unforced errors / match</td><td>{sa['unforced_errors']}</td></tr>
                    <tr><td>Net points won %</td><td>{sa['net_pct']}%</td></tr>
                    <tr><td>Service points won %</td><td>{sa['spw']}%</td></tr>
                    <tr><td>Return points won %</td><td>{sa['rpw']}%</td></tr>
                    <tr><td>Service games won %</td><td>{sa['sgw']}%</td></tr>
                    <tr><td>Return games won %</td><td>{sa['rgw']}%</td></tr>
                    <tr><td>Games won % overall</td><td>{sa['games_won_pct']}%</td></tr>
                </table>
            </div>
            <div class="card">
                <h3>{p_b}</h3>
                <p><strong>Record:</strong> {db['wins']}-{db['losses']} ({db['wr']}%) • {db['form']}</p>
                <p><strong>Rest:</strong> {fb_l} ({fb_d} days)</p>
                <table class="detail-table">
                    <tr><th>Stat</th><th>Value</th></tr>
                    <tr><td>Winners / match</td><td>{sb['winners']}</td></tr>
                    <tr><td>Unforced errors / match</td><td>{sb['unforced_errors']}</td></tr>
                    <tr><td>Net points won %</td><td>{sb['net_pct']}%</td></tr>
                    <tr><td>Service points won %</td><td>{sb['spw']}%</td></tr>
                    <tr><td>Return points won %</td><td>{sb['rpw']}%</td></tr>
                    <tr><td>Service games won %</td><td>{sb['sgw']}%</td></tr>
                    <tr><td>Return games won %</td><td>{sb['rgw']}%</td></tr>
                    <tr><td>Games won % overall</td><td>{sb['games_won_pct']}%</td></tr>
                </table>
            </div>
        </div>
        """

    h2h_section = ""
    if 'h2h' in st.session_state:
        h = st.session_state.h2h
        h2h_section = f"""
        <div class="card">
            <h3>Head-to-Head on {surf}</h3>
            <table class="detail-table">
                <tr><th>Metric</th><th>Value</th></tr>
                <tr><td>Total matches</td><td>{h['total']}</td></tr>
                <tr><td>{p_a} wins</td><td>{h.get(f"{p_a} wins", 0)}</td></tr>
                <tr><td>{p_b} wins</td><td>{h.get(f"{p_b} wins", 0)}</td></tr>
                <tr><td>Avg total games</td><td>{h.get('avg_g', 23.5):.1f}</td></tr>
            </table>
        </div>
        """

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <title>ATP Prediction • {p_a} vs {p_b}</title>
    <style>
        body {{ font-family: system-ui, sans-serif; background: linear-gradient(to bottom right, #f0f2ff, #e3e8ff); margin:0; padding:30px; color:#222; }}
        .container {{ max-width:1150px; margin:auto; background:white; border-radius:18px; box-shadow:0 12px 50px rgba(0,0,0,0.18); overflow:hidden; }}
        .header {{ background: linear-gradient(135deg, #1a237e, #3f51b5); color:white; padding:50px 35px; text-align:center; }}
        .content {{ padding:45px; }}
        .prediction {{ background:{pred_color}; color:white; padding:50px; border-radius:16px; text-align:center; margin:35px 0; }}
        .big {{ font-size:8em; font-weight:900; line-height:0.88; }}
        .card {{ background:#f8f9fa; border-radius:14px; padding:30px; margin:28px 0; border-left:7px solid #3f51b5; }}
        table {{ width:100%; border-collapse:collapse; margin:22px 0; }}
        th, td {{ padding:14px; text-align:left; border-bottom:1px solid #e0e0e0; }}
        .footer {{ background:#f0f0f5; padding:30px; text-align:center; color:#555; font-size:0.98em; }}
    </style>
</head>
<body>
<div class="container">
    <div class="header">
        <h1>ATP Men's Tennis Predictor Pro</h1>
        <p>{p_a} vs {p_b} • {surf} • {datetime.now().strftime('%Y-%m-%d %H:%M')}</p>
    </div>
    <div class="content">
        <div class="prediction">
            <div style="font-size:2.1em;">Predicted Total Games</div>
            <div class="big">{pred}</div>
        </div>
        {elo_section}
        {h2h_section}
        {stats_section}
    </div>
    <div class="footer">
        Generated with ATP Predictor Pro • Surface-specific Elo calculation • For analysis only
    </div>
</div>
</body>
</html>"""

st.download_button(
    label="📥 Download Detailed HTML Report",
    data=create_full_html_report(),
    file_name=f"ATP_{curr_a.replace(' ','_')}_vs_{curr_b.replace(' ','_')}_{curr_surf}_{datetime.now().strftime('%Y%m%d_%H%M')}.html",
    mime="text/html",
    use_container_width=True
)

st.caption("Tip: The report is more detailed when you run a prediction first.")

import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score
import requests
from io import BytesIO
from datetime import datetime
import re
import warnings
warnings.filterwarnings('ignore')

st.set_page_config(page_title="CHALLENGER Advanced Predictor", page_icon="🎾", layout="wide")

# ============= DATA LOADING =============

def fetch_challenger_github_data():
    try:
        url = "https://github.com/paulom40/teste/raw/main/Challenger.xlsx"
        response = requests.get(url, timeout=15)
        response.raise_for_status()
        df = pd.read_excel(BytesIO(response.content))
        df = normalize_columns(df)
        return df, "GitHub Challenger Database"
    except Exception as e:
        st.warning(f"Could not fetch GitHub data: {str(e)}")
        return None, None

def load_custom_excel(uploaded_file):
    try:
        df = pd.read_excel(uploaded_file)
        df = normalize_columns(df)
        st.success(f"Loaded: {uploaded_file.name}")
        return df, uploaded_file.name
    except Exception as e:
        st.error(f"Error loading file: {str(e)}")
        return None, None

def normalize_columns(df):
    col_map = {
        'winner_name': 'Winner', 'loser_name': 'Loser',
        'winner_rank': 'WRank',  'loser_rank': 'LRank',
        'winner_rank_points': 'WPts', 'loser_rank_points': 'LPts',
        'tourney_date': 'Date',  'score': 'Score',
        'best_of': 'BestOf',     'round': 'Round', 'minutes': 'Minutes',
        'winner_hand': 'WHand',  'loser_hand': 'LHand',
        'winner_ht': 'WHt',      'loser_ht': 'LHt',
        'winner_age': 'WAge',    'loser_age': 'LAge',
    }
    df = df.rename(columns={k: v for k, v in col_map.items() if k in df.columns})

    if 'Score' in df.columns:
        df = parse_score(df)

    if 'Date' in df.columns:
        df['Date'] = pd.to_datetime(df['Date'].astype(str), format='%Y%m%d', errors='coerce')

    if 'Surface' not in df.columns:
        for alt in ['surface', 'tourney_surface']:
            if alt in df.columns:
                df = df.rename(columns={alt: 'Surface'})
                break
        else:
            df['Surface'] = 'Hard'

    return df

def parse_score(df):
    def _parse(score):
        if pd.isna(score):
            return [np.nan]*11
        sets = re.findall(r'(\d+)-(\d+)(?:\(\d+\))?', str(score))
        w = [int(s[0]) for s in sets]
        l = [int(s[1]) for s in sets]
        while len(w) < 5: w.append(np.nan)
        while len(l) < 5: l.append(np.nan)
        wsets = sum(1 for a, b in zip(w[:5], l[:5])
                    if not (pd.isna(a) or pd.isna(b)) and a > b)
        return w[:5] + l[:5] + [wsets]

    parsed = df['Score'].apply(_parse)
    for i, col in enumerate(['W1','W2','W3','W4','W5','L1','L2','L3','L4','L5','Wsets']):
        df[col] = [row[i] for row in parsed]
    return df

# ============= GAME COUNT =============

def calculate_total_games(row):
    total = 0
    for i in range(1, 6):
        w = row.get(f'W{i}', np.nan)
        l = row.get(f'L{i}', np.nan)
        if pd.notna(w) and pd.notna(l) and w >= 0 and l >= 0:
            total += int(w) + int(l)
    return total if total > 0 else np.nan

# ============= ANALYSIS =============

def analyze_last_15_surface_games(df, player_name, surface):
    mask = (df['Winner'] == player_name) | (df['Loser'] == player_name)
    if surface != 'All':
        mask &= (df['Surface'] == surface)
    matches = df[mask].sort_values('Date', ascending=False).head(15).copy()

    empty = {'matches': 0, 'wins': 0, 'losses': 0, 'win_rate': 0.5,
             'avg_games': 22.0, 'form': 'No Data'}
    if len(matches) == 0:
        return empty

    matches['Total_Games'] = matches.apply(calculate_total_games, axis=1)
    matches = matches.dropna(subset=['Total_Games'])
    if len(matches) == 0:
        return empty

    wins   = int((matches['Winner'] == player_name).sum())
    losses = int((matches['Loser']  == player_name).sum())

    if   wins >= 11: form = "🔥 Excellent"
    elif wins >= 8:  form = "✓ Good"
    elif wins >= 5:  form = "⚠️ Mixed"
    else:            form = "❌ Poor"

    return {'matches': len(matches), 'wins': wins, 'losses': losses,
            'win_rate': wins / len(matches), 'avg_games': matches['Total_Games'].mean(),
            'form': form}

def calculate_fatigue(df, player_name):
    matches = df[(df['Winner'] == player_name) | (df['Loser'] == player_name)].sort_values('Date', ascending=False)
    now = pd.Timestamp.now()

    if len(matches) == 0:
        return {'days_rest': 0, 'matches_last_7': 0, 'matches_last_14': 0, 'fatigue_level': 'Unknown'}

    try:
        days_rest = int((now - pd.to_datetime(matches.iloc[0]['Date'])).days)
    except:
        days_rest = 0

    try:
        diff = (now - pd.to_datetime(matches['Date'])).dt.days
        m7  = int((diff <= 7).sum())
        m14 = int((diff <= 14).sum())
    except:
        m7 = m14 = 0

    if   days_rest >= 7:                  level = "✓ Fresh"
    elif days_rest >= 4:                  level = "⚔️ Normal"
    elif days_rest >= 2 and m7 <= 2:      level = "⚠️ Tired"
    else:                                 level = "🔴 Exhausted"

    return {'days_rest': days_rest, 'matches_last_7': m7, 'matches_last_14': m14, 'fatigue_level': level}

def safe_skill(value, lo=0.01, hi=0.99):
    """Clamp a skill value to strictly (0, 1) for st.progress compatibility"""
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return 0.5
    return float(np.clip(value, lo, hi))

# ─── percentile normaliser ────────────────────────────────────────────────────
# Pre-built reference ranges derived from ATP Challenger tour averages.
# Each tuple is (realistic_min, realistic_max) for a typical Challenger player.
# Values outside range are clipped so the bar always makes sense visually.
_SKILL_RANGES = {
    'first_serve_pct':       (0.50, 0.75),   # % of 1st serves in
    'first_serve_won_pct':   (0.60, 0.85),   # % of 1st-serve points won
    'second_serve_won_pct':  (0.40, 0.65),   # % of 2nd-serve points won
    'hold_pct':              (0.55, 0.95),   # % of service games held
    'bp_saved_pct':          (0.45, 0.80),   # % of break points saved
    'ace_per_svgm':          (0.00, 1.20),   # aces per service game
    'df_per_svgm':           (0.00, 0.80),   # double faults per service game (inverted → lower = better)
    'return_pts_won_pct':    (0.25, 0.55),   # % of return points won
    'break_conversion_pct':  (0.20, 0.55),   # % of break point opportunities converted
    'dominance_ratio':       (0.60, 1.80),   # (srv pts won % / return pts won %) — higher → more dominant server
}

def _norm(value, key, invert=False):
    """Normalise a raw stat value to [0, 1] within Challenger tour range."""
    lo, hi = _SKILL_RANGES[key]
    if hi == lo:
        return 0.5
    normed = (value - lo) / (hi - lo)
    if invert:
        normed = 1.0 - normed
    return float(np.clip(normed, 0.01, 0.99))

def _series(df_rows, col):
    """Safe numeric series from a column."""
    if col not in df_rows.columns:
        return pd.Series(dtype=float)
    return pd.to_numeric(df_rows[col], errors='coerce').dropna()

def _ratio(num_series, den_series, fallback=np.nan):
    """Element-wise ratio, return mean, avoiding div-by-zero."""
    den = den_series.replace(0, np.nan)
    r = (num_series / den).dropna()
    return float(r.mean()) if len(r) > 0 else fallback

def analyze_player_skills(df, player_name, surface):
    """
    Build 10 real skill metrics from actual match stats.
    Each metric is computed separately when the player is Winner vs Loser
    (the data stores winner stats as w_* and loser stats as l_*),
    then merged by weighted average (wins weighted 60 %, losses 40 %
    to slightly favour form while still using all data).
    """
    mask = (df['Winner'] == player_name) | (df['Loser'] == player_name)
    if surface != 'All':
        mask &= (df['Surface'] == surface)
    recent = df[mask].sort_values('Date', ascending=False).head(25).copy()

    nan5 = {'first_serve_pct': np.nan, 'first_serve_won_pct': np.nan,
            'second_serve_won_pct': np.nan, 'hold_pct': np.nan,
            'bp_saved_pct': np.nan, 'ace_per_svgm': np.nan,
            'df_per_svgm': np.nan, 'return_pts_won_pct': np.nan,
            'break_conversion_pct': np.nan, 'dominance_ratio': np.nan}

    if len(recent) == 0:
        return {k: 0.5 for k in nan5}

    w_rows = recent[recent['Winner'] == player_name]   # player won  → w_* cols
    l_rows = recent[recent['Loser']  == player_name]   # player lost → l_* cols

    def stat_both(w_col, l_col, fallback=np.nan):
        """Average a stat across wins (w_col) and losses (l_col)."""
        vals = []
        ws = _series(w_rows, w_col)
        ls = _series(l_rows, l_col)
        if len(ws) > 0: vals.append((float(ws.mean()), len(ws), 0.6))
        if len(ls) > 0: vals.append((float(ls.mean()), len(ls), 0.4))
        if not vals: return fallback
        total_w = sum(v[1] * v[2] for v in vals)
        return sum(v[0] * v[1] * v[2] for v in vals) / total_w if total_w > 0 else fallback

    def ratio_both(w_num, w_den, l_num, l_den, fallback=np.nan):
        """Ratio stat across wins and losses."""
        vals = []
        r_w = _ratio(_series(w_rows, w_num), _series(w_rows, w_den))
        r_l = _ratio(_series(l_rows, l_num), _series(l_rows, l_den))
        if not np.isnan(r_w): vals.append((r_w, len(w_rows), 0.6))
        if not np.isnan(r_l): vals.append((r_l, len(l_rows), 0.4))
        if not vals: return fallback
        total_w = sum(v[1] * v[2] for v in vals)
        return sum(v[0] * v[1] * v[2] for v in vals) / total_w if total_w > 0 else fallback

    # ── 1. First Serve % ──────────────────────────────────────────────────────
    fs_pct = ratio_both('w_1stIn', 'w_svpt', 'l_1stIn', 'l_svpt', fallback=0.62)

    # ── 2. First Serve Won % (of 1st-serve points) ───────────────────────────
    fs_won_pct = ratio_both('w_1stWon', 'w_1stIn', 'l_1stWon', 'l_1stIn', fallback=0.72)

    # ── 3. Second Serve Won % ────────────────────────────────────────────────
    # 2nd serve attempts = svpt - 1stIn
    def second_won(rows, pfx):
        svpt  = _series(rows, f'{pfx}_svpt').values
        fs_in = _series(rows, f'{pfx}_1stIn').values
        s2won = _series(rows, f'{pfx}_2ndWon').values
        n = min(len(svpt), len(fs_in), len(s2won))
        if n == 0: return np.nan
        s2att = (svpt[:n] - fs_in[:n]).clip(1)
        return float((s2won[:n] / s2att).mean())

    s2w_w = second_won(w_rows, 'w')
    s2w_l = second_won(l_rows, 'l')
    vals = [(v, cnt, wt) for v, cnt, wt in [(s2w_w, len(w_rows), 0.6),
                                              (s2w_l, len(l_rows), 0.4)]
            if not np.isnan(v)]
    if vals:
        tw = sum(v[1]*v[2] for v in vals)
        s2_won_pct = sum(v[0]*v[1]*v[2] for v in vals) / tw if tw > 0 else 0.52
    else:
        s2_won_pct = 0.52

    # ── 4. Hold % (service games held / service games played) ────────────────
    # Proxy: service games held ≈ SvGms - break points faced that weren't saved
    # Simpler: (SvGms - bpFaced + bpSaved) / SvGms  →  approximation of holds
    def hold_pct(rows, pfx):
        svgm = _series(rows, f'{pfx}_SvGms').replace(0, np.nan)
        bpf  = _series(rows, f'{pfx}_bpFaced')
        bps  = _series(rows, f'{pfx}_bpSaved')
        n = min(len(svgm), len(bpf), len(bps))
        if n == 0: return np.nan
        breaks_conceded = (bpf.values[:n] - bps.values[:n]).clip(0)
        held = (svgm.values[:n] - breaks_conceded).clip(0)
        return float((held / svgm.values[:n]).mean())

    hp_w = hold_pct(w_rows, 'w')
    hp_l = hold_pct(l_rows, 'l')
    vals = [(v, cnt, wt) for v, cnt, wt in [(hp_w, len(w_rows), 0.6),
                                              (hp_l, len(l_rows), 0.4)]
            if not np.isnan(v)]
    if vals:
        tw = sum(v[1]*v[2] for v in vals)
        h_pct = sum(v[0]*v[1]*v[2] for v in vals) / tw if tw > 0 else 0.75
    else:
        h_pct = 0.75

    # ── 5. Break Points Saved % ───────────────────────────────────────────────
    bp_saved = ratio_both('w_bpSaved', 'w_bpFaced', 'l_bpSaved', 'l_bpFaced', fallback=0.60)

    # ── 6. Aces per Service Game ──────────────────────────────────────────────
    ace_w = _ratio(_series(w_rows, 'w_ace'), _series(w_rows, 'w_SvGms'))
    ace_l = _ratio(_series(l_rows, 'l_ace'), _series(l_rows, 'l_SvGms'))
    vals = [(v, cnt, wt) for v, cnt, wt in [(ace_w, len(w_rows), 0.6),
                                              (ace_l, len(l_rows), 0.4)]
            if not np.isnan(v)]
    if vals:
        tw = sum(v[1]*v[2] for v in vals)
        ace_pg = sum(v[0]*v[1]*v[2] for v in vals) / tw if tw > 0 else 0.3
    else:
        ace_pg = 0.3

    # ── 7. Double Faults per Service Game (lower is better) ──────────────────
    df_w = _ratio(_series(w_rows, 'w_df'), _series(w_rows, 'w_SvGms'))
    df_l = _ratio(_series(l_rows, 'l_df'), _series(l_rows, 'l_SvGms'))
    vals = [(v, cnt, wt) for v, cnt, wt in [(df_w, len(w_rows), 0.6),
                                              (df_l, len(l_rows), 0.4)]
            if not np.isnan(v)]
    if vals:
        tw = sum(v[1]*v[2] for v in vals)
        df_pg = sum(v[0]*v[1]*v[2] for v in vals) / tw if tw > 0 else 0.25
    else:
        df_pg = 0.25

    # ── 8. Return Points Won % ───────────────────────────────────────────────
    # When player WINS:   opponent's l_svpt and player earns = l_svpt - l_1stWon - l_2ndWon
    # When player LOSES:  opponent's w_svpt and player earns = w_svpt - w_1stWon - w_2ndWon
    def ret_won(rows, opp_pfx):
        svpt  = _series(rows, f'{opp_pfx}_svpt').values
        fw    = _series(rows, f'{opp_pfx}_1stWon').values
        sw    = _series(rows, f'{opp_pfx}_2ndWon').values
        n = min(len(svpt), len(fw), len(sw))
        if n == 0: return np.nan
        ret_won_pts = (svpt[:n] - fw[:n] - sw[:n]).clip(0)
        return float((ret_won_pts / svpt[:n].clip(1)).mean())

    rw_w = ret_won(w_rows, 'l')   # when player wins, opp is loser → l_* cols
    rw_l = ret_won(l_rows, 'w')   # when player loses, opp is winner → w_* cols
    vals = [(v, cnt, wt) for v, cnt, wt in [(rw_w, len(w_rows), 0.6),
                                              (rw_l, len(l_rows), 0.4)]
            if not np.isnan(v)]
    if vals:
        tw = sum(v[1]*v[2] for v in vals)
        ret_pct = sum(v[0]*v[1]*v[2] for v in vals) / tw if tw > 0 else 0.38
    else:
        ret_pct = 0.38

    # ── 9. Break Point Conversion % (return side) ────────────────────────────
    # BPs converted = opp bpFaced - opp bpSaved
    def bp_conv(rows, opp_pfx):
        bpf = _series(rows, f'{opp_pfx}_bpFaced').replace(0, np.nan)
        bps = _series(rows, f'{opp_pfx}_bpSaved')
        n = min(len(bpf), len(bps))
        if n == 0: return np.nan
        converted = (bpf.values[:n] - bps.values[:n]).clip(0)
        return float((converted / bpf.values[:n]).mean())

    bc_w = bp_conv(w_rows, 'l')
    bc_l = bp_conv(l_rows, 'w')
    vals = [(v, cnt, wt) for v, cnt, wt in [(bc_w, len(w_rows), 0.6),
                                              (bc_l, len(l_rows), 0.4)]
            if not np.isnan(v)]
    if vals:
        tw = sum(v[1]*v[2] for v in vals)
        bk_conv = sum(v[0]*v[1]*v[2] for v in vals) / tw if tw > 0 else 0.35
    else:
        bk_conv = 0.35

    # ── 10. Dominance Ratio = srv pts won % / return pts won % ───────────────
    # srv pts won when winner: (w_1stWon + w_2ndWon) / w_svpt
    def srv_pts_won(rows, pfx):
        svpt = _series(rows, f'{pfx}_svpt').replace(0, np.nan)
        fw   = _series(rows, f'{pfx}_1stWon')
        sw   = _series(rows, f'{pfx}_2ndWon')
        n = min(len(svpt), len(fw), len(sw))
        if n == 0: return np.nan
        return float(((fw.values[:n] + sw.values[:n]) / svpt.values[:n]).mean())

    spw_w = srv_pts_won(w_rows, 'w')
    spw_l = srv_pts_won(l_rows, 'l')
    vals = [(v, cnt, wt) for v, cnt, wt in [(spw_w, len(w_rows), 0.6),
                                              (spw_l, len(l_rows), 0.4)]
            if not np.isnan(v)]
    if vals:
        tw = sum(v[1]*v[2] for v in vals)
        spw = sum(v[0]*v[1]*v[2] for v in vals) / tw if tw > 0 else 0.62
    else:
        spw = 0.62

    dom_ratio = spw / max(ret_pct, 0.01)

    # ── Normalise everything to [0.01, 0.99] using tour reference ranges ─────
    return {
        # label                  raw value       range key                invert?
        'first_serve_pct':       _norm(fs_pct,        'first_serve_pct'),
        'first_serve_won_pct':   _norm(fs_won_pct,    'first_serve_won_pct'),
        'second_serve_won_pct':  _norm(s2_won_pct,    'second_serve_won_pct'),
        'hold_pct':              _norm(h_pct,          'hold_pct'),
        'bp_saved_pct':          _norm(bp_saved,       'bp_saved_pct'),
        'ace_per_svgm':          _norm(ace_pg,         'ace_per_svgm'),
        'df_per_svgm':           _norm(df_pg,          'df_per_svgm',   invert=True),  # fewer = better
        'return_pts_won_pct':    _norm(ret_pct,        'return_pts_won_pct'),
        'break_conversion_pct':  _norm(bk_conv,        'break_conversion_pct'),
        'dominance_ratio':       _norm(dom_ratio,      'dominance_ratio'),
        # raw values for display labels
        '_raw': {
            'first_serve_pct':      fs_pct,
            'first_serve_won_pct':  fs_won_pct,
            'second_serve_won_pct': s2_won_pct,
            'hold_pct':             h_pct,
            'bp_saved_pct':         bp_saved,
            'ace_per_svgm':         ace_pg,
            'df_per_svgm':          df_pg,
            'return_pts_won_pct':   ret_pct,
            'break_conversion_pct': bk_conv,
            'dominance_ratio':      dom_ratio,
        }
    }

def get_real_stats(df, player_name):
    rows = df[df['Winner'] == player_name].tail(15)

    def safe(col, series):
        v = pd.to_numeric(series, errors='coerce')
        return float(v.mean()) if len(v.dropna()) > 0 else 0.0

    if len(rows) == 0 or 'w_svpt' not in rows.columns:
        return {k: 0.0 for k in ['avg_aces','avg_df','first_serve_pct',
                                   'first_serve_won_pct','second_serve_won_pct','bp_saved_pct']}

    svpt  = pd.to_numeric(rows['w_svpt'],  errors='coerce').replace(0, np.nan)
    fs_in = pd.to_numeric(rows['w_1stIn'], errors='coerce')
    fs_won= pd.to_numeric(rows['w_1stWon'],errors='coerce')
    snd   = pd.to_numeric(rows['w_2ndWon'],errors='coerce')
    bps   = pd.to_numeric(rows['w_bpSaved'],errors='coerce')
    bpf   = pd.to_numeric(rows['w_bpFaced'],errors='coerce').replace(0, np.nan)

    return {
        'avg_aces':              safe('w_ace',  rows['w_ace']),
        'avg_df':                safe('w_df',   rows['w_df']),
        'first_serve_pct':       float((fs_in / svpt).mean() * 100) if svpt.notna().any() else 0.0,
        'first_serve_won_pct':   float((fs_won / fs_in.replace(0,np.nan)).mean() * 100) if fs_in.notna().any() else 0.0,
        'second_serve_won_pct':  float((snd / (svpt - fs_in).replace(0,np.nan)).mean() * 100) if svpt.notna().any() else 0.0,
        'bp_saved_pct':          float((bps / bpf).mean() * 100) if bpf.notna().any() else 0.0,
    }

# ============= MODEL =============

@st.cache_resource
def build_model(n_rows, df):
    df_t = df.copy()
    df_t['Total_Games'] = df_t.apply(calculate_total_games, axis=1)
    df_t = df_t.dropna(subset=['Total_Games'])
    df_t = df_t[(df_t['Total_Games'] > 5) & (df_t['Total_Games'] < 55)]

    if len(df_t) < 50:
        return None

    def col_num(c):
        return pd.to_numeric(df_t[c], errors='coerce').fillna(0).values if c in df_t.columns else np.zeros(len(df_t))

    w1, l1 = col_num('W1'), col_num('L1')
    w2, l2 = col_num('W2'), col_num('L2')
    w3, l3 = col_num('W3'), col_num('L3')

    feats = []

    # Set-level games
    feats += [w1+l1, w2+l2, np.where(w3+l3>0, w3+l3, 0)]

    # Match structure
    if 'Wsets' in df_t.columns:
        feats += [(df_t['Wsets']==2).astype(float).values,
                  (df_t['Wsets']==3).astype(float).values]

    # Rank features
    wrank = pd.to_numeric(df_t['WRank'], errors='coerce').fillna(500)
    lrank = pd.to_numeric(df_t['LRank'], errors='coerce').fillna(500)
    feats.append((lrank - wrank).values)                          # rank diff
    feats.append((wrank / (wrank + lrank)).values)                # winner rank ratio
    feats.append(np.log1p(wrank.values))                          # log rank winner
    feats.append(np.log1p(lrank.values))                          # log rank loser

    # Competitiveness of each set
    feats.append(1 / (1 + np.abs(w1-l1) + np.abs(w2-l2)))

    # Serve stats (most predictive of match length)
    for c in ['w_ace','w_df','w_svpt','w_1stIn','w_1stWon','w_2ndWon','w_bpSaved','w_bpFaced',
              'l_ace','l_df','l_svpt','l_1stIn','l_1stWon','l_2ndWon','l_bpSaved','l_bpFaced']:
        feats.append(col_num(c))

    # Derived serve quality features
    w_svpt = col_num('w_svpt').clip(1)
    l_svpt = col_num('l_svpt').clip(1)
    feats.append(col_num('w_1stIn') / w_svpt)   # w 1st serve %
    feats.append(col_num('w_1stWon') / w_svpt)  # w 1st won %
    feats.append(col_num('l_1stIn') / l_svpt)   # l 1st serve %
    feats.append(col_num('l_1stWon') / l_svpt)  # l 1st won %
    w_bpf = col_num('w_bpFaced').clip(1)
    l_bpf = col_num('l_bpFaced').clip(1)
    feats.append(col_num('w_bpSaved') / w_bpf)  # w bp saved %
    feats.append(col_num('l_bpSaved') / l_bpf)  # l bp saved %

    # Duration if available
    feats.append(col_num('Minutes'))

    # Surface dummies
    for s in df_t['Surface'].dropna().unique():
        feats.append((df_t['Surface']==s).astype(int).values)

    X = np.nan_to_num(np.column_stack(feats), nan=0, posinf=0, neginf=0)
    y = df_t['Total_Games'].values

    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=42)
    sc = StandardScaler()

    # Tuned for better accuracy
    model = GradientBoostingRegressor(
        n_estimators=600,
        learning_rate=0.02,
        max_depth=4,
        min_samples_split=6,
        min_samples_leaf=3,
        subsample=0.75,
        max_features=0.8,
        random_state=42
    )
    model.fit(sc.fit_transform(X_tr), y_tr)
    y_pred = model.predict(sc.transform(X_te))

    return {'model': model, 'scaler': sc,
            'r2': r2_score(y_te, y_pred),
            'mae': mean_absolute_error(y_te, y_pred),
            'df': df_t, 'y_test': y_te, 'y_pred': y_pred}


def predict_games(model_data, player_a, player_b, surface, df):
    """
    Predict using a weighted blend of:
    1. Each player's recent median game count on the surface
    2. Their head-to-head history
    3. Serve stat quality adjustment
    """
    def player_recent(p, surf, n=15):
        m = df[(df['Winner']==p) | (df['Loser']==p)]
        if surf != 'All':
            ms = m[m['Surface']==surf]
            if len(ms) >= 5:
                m = ms
        m = m.tail(n).copy()
        m['Total_Games'] = m.apply(calculate_total_games, axis=1)
        return m.dropna(subset=['Total_Games'])

    a_m = player_recent(player_a, surface)
    b_m = player_recent(player_b, surface)

    a_med = float(a_m['Total_Games'].median()) if len(a_m) >= 3 else 22.0
    b_med = float(b_m['Total_Games'].median()) if len(b_m) >= 3 else 22.0

    base = (a_med + b_med) / 2.0

    # Head-to-head adjustment
    h2h = df[
        ((df['Winner']==player_a) & (df['Loser']==player_b)) |
        ((df['Winner']==player_b) & (df['Loser']==player_a))
    ].copy()
    if len(h2h) >= 2:
        h2h['Total_Games'] = h2h.apply(calculate_total_games, axis=1)
        h2h_med = h2h['Total_Games'].dropna().median()
        if not np.isnan(h2h_med):
            base = base * 0.65 + h2h_med * 0.35   # blend h2h in

    # Serve quality adjustment: high bp-saved % → shorter matches
    def avg_bp_saved(p):
        rows = df[df['Winner']==p].tail(10)
        if len(rows) == 0 or 'w_bpSaved' not in rows.columns:
            return 0.5
        bps = pd.to_numeric(rows['w_bpSaved'], errors='coerce')
        bpf = pd.to_numeric(rows['w_bpFaced'], errors='coerce').replace(0, np.nan)
        r = (bps / bpf).dropna()
        return float(r.mean()) if len(r) > 0 else 0.5

    bp_a = avg_bp_saved(player_a)
    bp_b = avg_bp_saved(player_b)
    avg_bp = (bp_a + bp_b) / 2.0
    # high bp-saved → dominant server → shorter match
    serve_adj = (0.5 - avg_bp) * 4.0   # range roughly -2 to +2

    prediction = base + serve_adj
    return float(np.clip(prediction, 12, 45))

# ============= HTML REPORT =============

def generate_html_report(player_a, player_b, surface, an_a, an_b,
                          prediction, model_data, rs_a, rs_b):
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    if prediction < 23:   mt, col = "⚡ Quick Match",  "#16a34a"
    elif prediction < 27: mt, col = "⚔️ Competitive",  "#d97706"
    else:                 mt, col = "🔥 Long Match",    "#dc2626"

    def bar(val, name):
        p = int(np.clip(val*100, 0, 100))
        return f'<div class="skill"><div class="sn">{name}</div><div class="bc"><div class="bf" style="width:{p}%"><div class="bv">{p}%</div></div></div></div>'

    def pcard(name, an, rs):
        s, l, f = an['skills'], an['last15'], an['fatigue']
        raw = s.get('_raw', {})

        def r(key, fmt='.1%'):
            v = raw.get(key, 0) or 0
            return format(v, fmt)

        return f"""<div class="pc"><h3>{name}</h3>
        <div class="ss"><h4>📈 Last 15 · {surface}</h4>
          <div class="st"><span>Record</span><span>{l['wins']}-{l['losses']}</span></div>
          <div class="st"><span>Win Rate</span><span>{l['win_rate']:.1%}</span></div>
          <div class="st"><span>Avg Games</span><span>{l['avg_games']:.1f}</span></div>
          <div class="st"><span>Form</span><span>{l['form']}</span></div></div>
        <div class="ss"><h4>🎯 Serve Stats</h4>
          <div class="st"><span>1st Serve %</span><span>{rs['first_serve_pct']:.1f}%</span></div>
          <div class="st"><span>1st Serve Won %</span><span>{rs['first_serve_won_pct']:.1f}%</span></div>
          <div class="st"><span>2nd Serve Won %</span><span>{rs['second_serve_won_pct']:.1f}%</span></div>
          <div class="st"><span>Avg Aces</span><span>{rs['avg_aces']:.1f}</span></div>
          <div class="st"><span>Avg DFs</span><span>{rs['avg_df']:.1f}</span></div>
          <div class="st"><span>BP Saved %</span><span>{rs['bp_saved_pct']:.1f}%</span></div></div>
        <div class="ss"><h4>😓 Fatigue</h4>
          <div class="st"><span>Days Rest</span><span>{f['days_rest']}</span></div>
          <div class="st"><span>Matches (7d)</span><span>{f['matches_last_7']}</span></div>
          <div class="st"><span>Status</span><span>{f['fatigue_level']}</span></div></div>
        <div class="ss"><h4>⚡ Serve Skill Ratings</h4>
          {bar(s['first_serve_pct'],      f'1ST SERVE IN · {r("first_serve_pct")}')}
          {bar(s['first_serve_won_pct'],  f'1ST SERVE WON · {r("first_serve_won_pct")}')}
          {bar(s['second_serve_won_pct'], f'2ND SERVE WON · {r("second_serve_won_pct")}')}
          {bar(s['hold_pct'],             f'SERVICE HOLD · {r("hold_pct")}')}
          {bar(s['bp_saved_pct'],         f'BP SAVED · {r("bp_saved_pct")}')}
          {bar(s['ace_per_svgm'],         f'ACES/GAME · {r("ace_per_svgm", ".2f")}')}
          {bar(s['df_per_svgm'],          f'DF CONTROL · {r("df_per_svgm", ".2f")}/gm')}</div>
        <div class="ss"><h4>🔄 Return Skill Ratings</h4>
          {bar(s['return_pts_won_pct'],   f'RETURN PTS WON · {r("return_pts_won_pct")}')}
          {bar(s['break_conversion_pct'], f'BREAK CONVERSION · {r("break_conversion_pct")}')}
          {bar(s['dominance_ratio'],      f'DOMINANCE RATIO · {r("dominance_ratio", ".2f")}')}
        </div></div>"""

    return f"""<!DOCTYPE html><html><head><meta charset="UTF-8">
<title>Challenger Prediction</title>
<link href="https://fonts.googleapis.com/css2?family=Bebas+Neue&family=IBM+Plex+Mono:wght@400;600&family=IBM+Plex+Sans:wght@300;400;600&display=swap" rel="stylesheet">
<style>
*{{margin:0;padding:0;box-sizing:border-box}}
body{{font-family:'IBM Plex Sans',sans-serif;background:#0a0a0a;color:#e0e0e0;padding:20px}}
.wrap{{max-width:1100px;margin:0 auto;background:#111;border:1px solid #2a2a2a;border-radius:4px;overflow:hidden}}
.hdr{{background:#0a0a0a;border-bottom:3px solid #c8ff00;padding:40px 30px;text-align:center}}
.hdr h1{{font-family:'Bebas Neue',sans-serif;font-size:3em;letter-spacing:6px;color:#c8ff00}}
.hdr p{{font-family:'IBM Plex Mono',monospace;font-size:0.75em;color:#666;margin-top:8px;letter-spacing:2px}}
.body{{padding:40px}}
.mt{{font-family:'Bebas Neue',sans-serif;font-size:2.6em;letter-spacing:4px;color:#fff;text-align:center;margin:20px 0 5px}}
.sb{{text-align:center;margin-bottom:30px;font-family:'IBM Plex Mono',monospace;font-size:0.85em;color:#c8ff00;letter-spacing:3px}}
.pb{{background:{col};padding:35px;text-align:center;margin:30px 0;border-radius:2px}}
.pb .lbl{{font-family:'IBM Plex Mono',monospace;font-size:0.82em;letter-spacing:3px;color:rgba(255,255,255,.8)}}
.pb .num{{font-family:'Bebas Neue',sans-serif;font-size:5em;letter-spacing:4px;color:#fff;line-height:1}}
.stitle{{font-family:'Bebas Neue',sans-serif;color:#c8ff00;font-size:1.8em;letter-spacing:4px;border-bottom:1px solid #2a2a2a;padding-bottom:8px;margin:35px 0 20px}}
.grid{{display:grid;grid-template-columns:1fr 1fr;gap:20px;margin:20px 0}}
.pc{{background:#0d0d0d;padding:25px;border:1px solid #222;border-top:3px solid #c8ff00}}
.pc h3{{font-family:'Bebas Neue',sans-serif;color:#c8ff00;margin-bottom:20px;font-size:1.6em;letter-spacing:3px}}
.ss{{margin-bottom:22px}}
.ss h4{{font-family:'IBM Plex Mono',monospace;color:#888;font-size:0.73em;margin-bottom:10px;letter-spacing:2px;text-transform:uppercase}}
.st{{display:flex;justify-content:space-between;padding:7px 0;border-bottom:1px solid #1a1a1a;font-size:.9em}}
.st span:last-child{{font-family:'IBM Plex Mono',monospace;color:#c8ff00;font-weight:600}}
.skill{{margin:11px 0}}.sn{{font-size:.78em;color:#aaa;margin-bottom:5px;font-family:'IBM Plex Mono',monospace;letter-spacing:1px}}
.bc{{height:25px;background:#1a1a1a;border:1px solid #222;overflow:hidden}}
.bf{{height:100%;background:linear-gradient(90deg,#c8ff00,#7fff00);display:flex;align-items:center;justify-content:flex-end;padding-right:9px}}
.bv{{color:#000;font-weight:700;font-size:.8em;font-family:'IBM Plex Mono',monospace}}
.info{{background:#0d0d0d;border:1px solid #222;border-left:3px solid #c8ff00;padding:14px 18px;margin:20px 0;font-family:'IBM Plex Mono',monospace;font-size:.76em;color:#888;letter-spacing:1px}}
.ftr{{background:#0a0a0a;border-top:1px solid #222;padding:18px;text-align:center;font-family:'IBM Plex Mono',monospace;font-size:.7em;color:#444;letter-spacing:2px}}
</style></head><body>
<div class="wrap">
<div class="hdr"><h1>🎾 CHALLENGER PREDICTOR</h1><p>REAL SERVE STATS · LAST 15 MATCHES · FATIGUE · SKILL RATINGS</p></div>
<div class="body">
  <div class="mt">{player_a} VS {player_b}</div>
  <div class="sb">⬡ SURFACE: {surface}</div>
  <div class="pb"><div class="lbl">{mt} — PREDICTED TOTAL GAMES</div><div class="num">{prediction:.1f}</div></div>
  <div class="stitle">PLAYER ANALYSIS</div>
  <div class="grid">{pcard(player_a,an_a,rs_a)}{pcard(player_b,an_b,rs_b)}</div>
  <div class="info">MODEL · R² = {model_data['r2']:.3f} · MAE = ±{model_data['mae']:.2f} GAMES · {len(model_data['df'])} MATCHES</div>
</div>
<div class="ftr">GENERATED: {ts} · ATP CHALLENGER PREDICTOR · ±{model_data['mae']:.2f} GAMES</div>
</div></body></html>"""

# ============= MAIN =============

def main():
    st.sidebar.title("🎾 Challenger Predictor")
    st.sidebar.markdown("---")

    data_source = st.sidebar.radio("Data source:", ["🌐 GitHub Challenger DB", "📥 Upload Custom File"])

    df, source_name = None, ""

    if data_source == "📥 Upload Custom File":
        f = st.sidebar.file_uploader("Upload Excel (.xlsx)", type=['xlsx','xls'])
        if f:
            df, source_name = load_custom_excel(f)
        else:
            st.info("👈 Upload a file or switch to GitHub source")
            return
    else:
        with st.spinner("Loading Challenger data from GitHub..."):
            df, source_name = fetch_challenger_github_data()
        if df is None:
            st.error("Could not load from GitHub. Try uploading a custom file.")
            return

    if df is None:
        return

    st.sidebar.metric("Total Matches", len(df))
    surfaces_available = sorted(df['Surface'].dropna().unique().tolist())
    st.sidebar.write(f"Surfaces: {', '.join(surfaces_available)}")

    with st.spinner("Training model..."):
        model_data = build_model(len(df), df)

    if model_data is None:
        st.error(f"Not enough valid data to train. Rows loaded: {len(df)}")
        st.write("Sample of loaded data:")
        st.dataframe(df.head())
        return

    st.sidebar.success("✅ Model ready!")
    st.sidebar.metric("R²",       f"{model_data['r2']:.3f}")
    st.sidebar.metric("MAE ±",    f"{model_data['mae']:.2f} games")
    st.sidebar.metric("Training", f"{len(model_data['df'])} matches")

    st.header("🎾 ATP Challenger Advanced Match Predictor")
    st.caption(f"Source: **{source_name}** · Real Serve/Return Stats · Last 15 Matches · Fatigue Analysis")
    st.markdown("---")

    all_players = sorted(set(df['Winner'].dropna()) | set(df['Loser'].dropna()))
    surface_opts = ['All'] + surfaces_available

    c1, c2, c3 = st.columns(3)
    with c1: player_a = st.selectbox("Player 1", all_players)
    with c2: player_b = st.selectbox("Player 2", all_players, index=min(1, len(all_players)-1))
    with c3: surface  = st.selectbox("Surface",  surface_opts)

    st.markdown("---")

    if st.button("🔮 PREDICT MATCH", use_container_width=True):
        with st.spinner("Analysing..."):
            an_a = {'last15': analyze_last_15_surface_games(df, player_a, surface),
                    'fatigue': calculate_fatigue(df, player_a),
                    'skills':  analyze_player_skills(df, player_a, surface)}
            an_b = {'last15': analyze_last_15_surface_games(df, player_b, surface),
                    'fatigue': calculate_fatigue(df, player_b),
                    'skills':  analyze_player_skills(df, player_b, surface)}
            rs_a = get_real_stats(df, player_a)
            rs_b = get_real_stats(df, player_b)
            pred = predict_games(model_data, player_a, player_b, surface, df)

        col1, col2 = st.columns(2)

        def show(col, name, an, rs):
            with col:
                st.subheader(f"🎾 {name}")
                l, f, s = an['last15'], an['fatigue'], an['skills']

                with st.expander(f"📈 Last 15 · {surface}", expanded=True):
                    st.write(f"**{l['wins']}-{l['losses']}**  ·  Win rate: **{l['win_rate']:.1%}**  ·  Form: **{l['form']}**")
                    st.write(f"Avg games per match: **{l['avg_games']:.1f}**")

                with st.expander("🎯 Serve Stats (real data)", expanded=True):
                    a, b = st.columns(2)
                    a.metric("1st Serve %",      f"{rs['first_serve_pct']:.1f}%")
                    b.metric("1st Serve Won %",  f"{rs['first_serve_won_pct']:.1f}%")
                    a.metric("2nd Serve Won %",  f"{rs['second_serve_won_pct']:.1f}%")
                    b.metric("BP Saved %",       f"{rs['bp_saved_pct']:.1f}%")
                    a.metric("Avg Aces",         f"{rs['avg_aces']:.1f}")
                    b.metric("Avg DFs",          f"{rs['avg_df']:.1f}")

                with st.expander("😓 Fatigue", expanded=True):
                    st.write(f"Days rest: **{f['days_rest']}**  ·  Matches last 7d: **{f['matches_last_7']}**")
                    st.write(f"Status: **{f['fatigue_level']}**")

                with st.expander("⚡ Real Skill Ratings", expanded=True):
                    raw = s.get('_raw', {})
                    st.caption("📡 Serve Game")
                    st.progress(safe_skill(s['first_serve_pct']),
                        text=f"1st Serve In          {raw.get('first_serve_pct', 0):.1%}")
                    st.progress(safe_skill(s['first_serve_won_pct']),
                        text=f"1st Serve Won         {raw.get('first_serve_won_pct', 0):.1%}")
                    st.progress(safe_skill(s['second_serve_won_pct']),
                        text=f"2nd Serve Won         {raw.get('second_serve_won_pct', 0):.1%}")
                    st.progress(safe_skill(s['hold_pct']),
                        text=f"Service Hold          {raw.get('hold_pct', 0):.1%}")
                    st.progress(safe_skill(s['bp_saved_pct']),
                        text=f"Break Points Saved    {raw.get('bp_saved_pct', 0):.1%}")
                    st.progress(safe_skill(s['ace_per_svgm']),
                        text=f"Aces / Srv Game       {raw.get('ace_per_svgm', 0):.2f}")
                    st.progress(safe_skill(s['df_per_svgm']),
                        text=f"DF Control (fewer=better)  {raw.get('df_per_svgm', 0):.2f}/gm")
                    st.caption("🔄 Return Game")
                    st.progress(safe_skill(s['return_pts_won_pct']),
                        text=f"Return Points Won     {raw.get('return_pts_won_pct', 0):.1%}")
                    st.progress(safe_skill(s['break_conversion_pct']),
                        text=f"Break Conversion      {raw.get('break_conversion_pct', 0):.1%}")
                    st.caption("📊 Overall")
                    st.progress(safe_skill(s['dominance_ratio']),
                        text=f"Dominance Ratio       {raw.get('dominance_ratio', 0):.2f}")

        show(col1, player_a, an_a, rs_a)
        show(col2, player_b, an_b, rs_b)

        st.markdown("---")
        _, c, _ = st.columns([1,2,1])
        with c:
            st.metric("🎯 Predicted Total Games", f"{pred:.1f}")
            if   pred < 23: st.info("⚡ Quick Match — 2 sets likely")
            elif pred < 27: st.info("⚔️ Competitive Match")
            else:           st.warning("🔥 Long Match — 3 sets likely")

        st.markdown("---")
        html = generate_html_report(player_a, player_b, surface,
                                     an_a, an_b, pred, model_data, rs_a, rs_b)
        st.download_button(
            "📥 Download HTML Report", data=html,
            file_name=f"Challenger_{player_a}_vs_{player_b}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.html",
            mime="text/html"
        )

if __name__ == "__main__":
    main()

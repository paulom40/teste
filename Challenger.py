import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import GradientBoostingRegressor, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import mean_absolute_error, r2_score, accuracy_score
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

# ============= ELO RATING SYSTEM =============

def compute_elo_ratings(df, k=32, initial=1500):
    """
    Compute Elo ratings for every player by replaying matches in date order.
    Returns a dict {player_name: elo} reflecting each player's rating
    BEFORE their last match (so it's usable as a pre-match feature).
    Also returns a per-match elo series for feature building.
    """
    df_sorted = df.sort_values('Date').reset_index(drop=True)
    elo = {}
    # Store elo BEFORE each match for each row
    elo_winner_pre = []
    elo_loser_pre  = []

    for _, row in df_sorted.iterrows():
        w = row['Winner']
        l = row['Loser']
        ew = elo.get(w, initial)
        el = elo.get(l, initial)
        elo_winner_pre.append(ew)
        elo_loser_pre.append(el)
        # Expected scores
        exp_w = 1 / (1 + 10**((el - ew) / 400))
        exp_l = 1 - exp_w
        # Update
        elo[w] = ew + k * (1 - exp_w)
        elo[l] = el + k * (0 - exp_l)

    df_sorted['elo_winner_pre'] = elo_winner_pre
    df_sorted['elo_loser_pre']  = elo_loser_pre
    return elo, df_sorted



def compute_rolling30_elo(df, k=40, initial=1500, window=30):
    """
    Rolling Elo that re-computes each player's rating using only their
    last `window` matches. Highly responsive to recent form.
    Returns: elo_final, history {player: [(date,elo)]}, df_out
    """
    from collections import deque
    df_s = df.sort_values('Date').reset_index(drop=True)
    player_results = {}
    elo = {}
    history = {}
    pre_w_list, pre_l_list = [], []

    def recalc(results_deque):
        rating = initial
        for outcome, opp_elo in results_deque:
            exp = 1 / (1 + 10 ** ((opp_elo - rating) / 400))
            rating += k * (outcome - exp)
        return rating

    for _, row in df_s.iterrows():
        w, l = row['Winner'], row['Loser']
        date = row['Date']
        ew = elo.get(w, initial)
        el = elo.get(l, initial)
        pre_w_list.append(ew)
        pre_l_list.append(el)
        for player, outcome, opp_pre in [(w, 1, el), (l, 0, ew)]:
            if player not in player_results:
                player_results[player] = deque()
            player_results[player].append((outcome, opp_pre))
            if len(player_results[player]) > window:
                player_results[player].popleft()
            new_elo = recalc(player_results[player])
            elo[player] = new_elo
            if player not in history:
                history[player] = []
            history[player].append((date, new_elo))

    df_s['elo30_winner'] = pre_w_list
    df_s['elo30_loser']  = pre_l_list
    return elo, history, df_s


def get_player_elo30_stats(player, elo_final, history, df, window=30):
    """Rich stats dict for a player based on rolling-30 Elo."""
    current_elo = elo_final.get(player, 1500)
    if player in history and len(history[player]) > 0:
        hist_df = pd.DataFrame(history[player], columns=['Date', 'Elo'])
        hist_df['Date'] = pd.to_datetime(hist_df['Date'])
        hist_df = hist_df.sort_values('Date').drop_duplicates('Date', keep='last')
    else:
        hist_df = pd.DataFrame(columns=['Date', 'Elo'])
    recent_hist = hist_df.tail(window)
    peak_elo   = float(recent_hist['Elo'].max()) if len(recent_hist) > 0 else current_elo
    trough_elo = float(recent_hist['Elo'].min()) if len(recent_hist) > 0 else current_elo
    last10 = hist_df.tail(10)
    if len(last10) >= 3:
        slope = float(np.polyfit(np.arange(len(last10)), last10['Elo'].values, 1)[0])
    else:
        slope = 0.0
    if   slope >  8: trend = ('Rising',  'green')
    elif slope < -8: trend = ('Falling', 'red')
    else:            trend = ('Stable',  'gray')
    matches = df[(df['Winner']==player)|(df['Loser']==player)].sort_values('Date').tail(window)
    wins   = int((matches['Winner']==player).sum())
    losses = int((matches['Loser']==player).sum())
    surf_stats = {}
    for surf, grp in matches.groupby('Surface'):
        sw = int((grp['Winner']==player).sum())
        sl = int((grp['Loser']==player).sum())
        surf_stats[surf] = {'wins': sw, 'losses': sl, 'win_rate': sw/max(sw+sl,1)}
    return {
        'current_elo': current_elo, 'peak_elo': peak_elo, 'trough_elo': trough_elo,
        'slope': slope, 'trend': trend,
        'wins': wins, 'losses': losses,
        'win_rate': wins / max(wins+losses, 1),
        'hist_df': hist_df, 'surf_stats': surf_stats, 'n_matches': wins+losses,
    }


def render_elo_tab(df, model_data, player_a, player_b, surface, all_players):
    """Render the full Rolling-30 Elo tab."""
    import plotly.graph_objects as go

    st.subheader("📡 Rolling-30 Elo System")
    st.caption(
        "Each player's Elo is recalculated using only their **last 30 matches**. "
        "K-factor = 40. Highly responsive to current form vs career history."
    )

    with st.spinner("Computing rolling-30 Elo..."):
        elo_final, history, df_elo30 = compute_rolling30_elo(df)

    pa_stats = get_player_elo30_stats(player_a, elo_final, history, df)
    pb_stats = get_player_elo30_stats(player_b, elo_final, history, df)

    # ── Player comparison ─────────────────────────────────────────────────────
    st.markdown("### 🆚 Head-to-Head Elo Comparison")
    col_a, col_mid, col_b = st.columns([5, 2, 5])

    def elo_card(col, name, stats, color):
        with col:
            delta = stats['current_elo'] - 1500
            delta_str = f"+{delta:.0f}" if delta >= 0 else f"{delta:.0f}"
            trend_icons = {'Rising': '📈', 'Falling': '📉', 'Stable': '➡️'}
            ti = trend_icons.get(stats['trend'][0], '➡️')
            st.markdown(
                f"<div style='background:#0d0d0d;border:1px solid #2a2a2a;"
                f"border-top:4px solid {color};padding:20px;border-radius:4px;text-align:center'>"
                f"<div style='font-size:0.7em;color:#888;letter-spacing:3px;font-family:monospace'>"
                f"ROLLING-30 ELO</div>"
                f"<div style='font-size:3em;font-weight:900;color:{color};margin:6px 0'>"
                f"{stats['current_elo']:.0f}</div>"
                f"<div style='font-size:0.82em;color:#aaa;font-family:monospace'>"
                f"{delta_str} vs baseline</div>"
                f"<div style='font-size:1em;margin-top:10px;color:{color}'>"
                f"{ti} {stats['trend'][0]}</div>"
                f"</div>", unsafe_allow_html=True
            )
            st.markdown(f"**{name}**")
            c1, c2 = st.columns(2)
            c1.metric("Record (30)",   f"{stats['wins']}-{stats['losses']}")
            c2.metric("Win Rate",      f"{stats['win_rate']:.1%}")
            c1.metric("Peak Elo",      f"{stats['peak_elo']:.0f}")
            c2.metric("Trough Elo",    f"{stats['trough_elo']:.0f}")
            c1.metric("Trend /match",  f"{stats['slope']:+.1f}")
            c2.metric("Matches used",  f"{stats['n_matches']}")

    elo_card(col_a, player_a, pa_stats, '#c8ff00')
    elo_card(col_b, player_b, pb_stats, '#ff7043')

    with col_mid:
        st.markdown("<div style='height:60px'></div>", unsafe_allow_html=True)
        elo_diff = pa_stats['current_elo'] - pb_stats['current_elo']
        exp_a    = 1 / (1 + 10**(-elo_diff / 400))
        fav      = player_a if elo_diff >= 0 else player_b
        fav_prob = max(exp_a, 1-exp_a)
        st.markdown(
            f"<div style='text-align:center;padding:18px;background:#111;"
            f"border:1px solid #333;border-radius:4px'>"
            f"<div style='font-size:0.68em;color:#888;letter-spacing:2px;font-family:monospace'>ELO EDGE</div>"
            f"<div style='font-size:1.1em;font-weight:700;color:#fff;margin:6px 0'>{fav}</div>"
            f"<div style='font-size:2em;font-weight:900;color:#c8ff00'>{fav_prob:.1%}</div>"
            f"<div style='font-size:0.68em;color:#666;font-family:monospace'>WIN PROB</div>"
            f"<div style='font-size:0.75em;color:#555;margin-top:8px;font-family:monospace'>"
            f"Gap: {abs(elo_diff):.0f} pts</div>"
            f"</div>", unsafe_allow_html=True
        )

    # ── Trajectory chart ──────────────────────────────────────────────────────
    st.markdown("### 📈 Elo Trajectory (Last 30 Matches Each)")
    fig = go.Figure()
    for name, stats, color, dash in [
        (player_a, pa_stats, '#c8ff00', 'solid'),
        (player_b, pb_stats, '#ff7043', 'dash'),
    ]:
        hdf = stats['hist_df'].tail(30)
        if len(hdf) == 0:
            continue
        # Add shaded area under line
        fig.add_trace(go.Scatter(
            x=list(hdf['Date']) + list(hdf['Date'])[::-1],
            y=list(hdf['Elo']) + [1500]*len(hdf),
            fill='toself', fillcolor=color.replace('#','rgba(').replace('c8ff00','200,255,0,0.06)').replace('ff7043','255,112,67,0.06)'),
            line=dict(width=0), showlegend=False, hoverinfo='skip'
        ))
        fig.add_trace(go.Scatter(
            x=hdf['Date'], y=hdf['Elo'],
            mode='lines+markers', name=name,
            line=dict(color=color, width=2.5, dash=dash),
            marker=dict(size=5, color=color, line=dict(color='#111',width=1)),
            hovertemplate=f"<b>{name}</b><br>%{{x|%Y-%m-%d}}<br>Elo: %{{y:.0f}}<extra></extra>"
        ))
    fig.add_hline(y=1500, line_dash='dot', line_color='#333',
                  annotation_text='1500 baseline', annotation_font_color='#555',
                  annotation_position='bottom right')
    fig.update_layout(
        template='plotly_dark', paper_bgcolor='#0a0a0a', plot_bgcolor='#0d0d0d',
        font=dict(family='IBM Plex Mono', size=11, color='#aaa'),
        legend=dict(bgcolor='#111', bordercolor='#333', borderwidth=1,
                    orientation='h', yanchor='bottom', y=1.02, xanchor='right', x=1),
        xaxis=dict(gridcolor='#1a1a1a', title=''), yaxis=dict(gridcolor='#1a1a1a', title='Rolling-30 Elo'),
        margin=dict(l=40, r=20, t=40, b=30), height=380,
    )
    st.plotly_chart(fig, use_container_width=True)

    # ── Surface breakdown ─────────────────────────────────────────────────────
    st.markdown("### 🏟️ Surface Performance (Last 30 Matches)")
    s_col_a, s_col_b = st.columns(2)
    def surface_table(col, name, stats):
        with col:
            st.markdown(f"**{name}**")
            if not stats['surf_stats']:
                st.caption("No data"); return
            rows = []
            for surf, s in sorted(stats['surf_stats'].items()):
                wr = s['win_rate']
                bar = '█' * int(wr*10) + '░' * (10-int(wr*10))
                rows.append({'Surface': surf, 'W-L': f"{s['wins']}-{s['losses']}",
                             'Win%': f"{wr:.0%}", 'Form': bar})
            st.dataframe(pd.DataFrame(rows).set_index('Surface'), use_container_width=True)
    surface_table(s_col_a, player_a, pa_stats)
    surface_table(s_col_b, player_b, pb_stats)

    # ── Leaderboard ───────────────────────────────────────────────────────────
    st.markdown("### 🏆 Player Leaderboard — Rolling-30 Elo")
    leaderboard = []
    for p in all_players:
        if p not in elo_final: continue
        matches = df[(df['Winner']==p)|(df['Loser']==p)].tail(30)
        if len(matches) < 3: continue
        wins   = int((matches['Winner']==p).sum())
        losses = int((matches['Loser']==p).sum())
        elo_v  = elo_final[p]
        if p in history and len(history[p]) >= 10:
            h     = [e for _, e in history[p]]
            trend_pt = np.mean(h[-5:]) - np.mean(h[-10:-5])
            ti = '📈' if trend_pt > 15 else ('📉' if trend_pt < -15 else '➡️')
        else:
            trend_pt, ti = 0.0, '➡️'
        leaderboard.append({
            'Player':  p,
            'Elo':     round(elo_v),
            'W-L':     f"{wins}-{losses}",
            'Win%':    f"{wins/max(wins+losses,1):.0%}",
            'Trend':   ti,
            'Δ(5gm)':  f"{trend_pt:+.0f}",
        })
    if leaderboard:
        lb_df = (pd.DataFrame(leaderboard)
                   .sort_values('Elo', ascending=False)
                   .reset_index(drop=True))
        lb_df.index += 1; lb_df.index.name = 'Rank'

        def highlight(row):
            if row['Player'] == player_a: return ['background-color:#1a2a00;color:#c8ff00']*len(row)
            if row['Player'] == player_b: return ['background-color:#2a0d00;color:#ff7043']*len(row)
            return ['']*len(row)

        st.dataframe(lb_df.style.apply(highlight, axis=1),
                     use_container_width=True,
                     height=min(650, 45 + len(lb_df)*35))
        st.caption(f"🟡 {player_a}   🟠 {player_b}   ·   {len(lb_df)} ranked players")

    # ── H2H history ───────────────────────────────────────────────────────────
    h2h = df[
        ((df['Winner']==player_a)&(df['Loser']==player_b)) |
        ((df['Winner']==player_b)&(df['Loser']==player_a))
    ].sort_values('Date', ascending=False)
    if len(h2h) > 0:
        st.markdown(f"### ⚔️ Head-to-Head: {player_a} vs {player_b}")
        h2h_show = h2h.copy()
        h2h_show['Date']   = pd.to_datetime(h2h_show['Date']).dt.strftime('%Y-%m-%d')
        h2h_show['Result'] = h2h_show.apply(lambda r: f"✅ {r['Winner']} def. {r['Loser']}", axis=1)
        st.dataframe(h2h_show[['Date','Result','Score','Surface']].reset_index(drop=True),
                     use_container_width=True)
        a_w = int((h2h['Winner']==player_a).sum())
        b_w = len(h2h) - a_w
        st.info(f"Overall H2H: **{player_a} {a_w} – {b_w} {player_b}** ({len(h2h)} matches)")
    else:
        st.info(f"No H2H matches between {player_a} and {player_b} in the dataset.")


def rolling_player_stats(df_sorted, player, before_idx, window=20, surface=None):
    """
    Compute rolling stats for a player using only matches BEFORE before_idx.
    This is the key to avoiding data leakage.
    """
    mask = ((df_sorted['Winner'] == player) | (df_sorted['Loser'] == player))
    mask &= (df_sorted.index < before_idx)
    if surface and surface != 'All':
        surf_mask = mask & (df_sorted['Surface'] == surface)
        if surf_mask.sum() >= 5:
            mask = surf_mask
    hist = df_sorted[mask].tail(window)

    if len(hist) == 0:
        return None

    w_rows = hist[hist['Winner'] == player]
    l_rows = hist[hist['Loser']  == player]

    def ratio(num_col, den_col, rows, pfx):
        if len(rows) == 0: return np.nan
        num = pd.to_numeric(rows.get(f'{pfx}_{num_col}', pd.Series()), errors='coerce')
        den = pd.to_numeric(rows.get(f'{pfx}_{den_col}', pd.Series()), errors='coerce').replace(0, np.nan)
        r = (num / den).dropna()
        return float(r.mean()) if len(r) > 0 else np.nan

    def merge(v_w, v_l, nw, nl):
        vals = [(v, n, w) for v, n, w in [(v_w,nw,0.6),(v_l,nl,0.4)] if not (v is None or np.isnan(v))]
        if not vals: return np.nan
        tw = sum(x[1]*x[2] for x in vals)
        return sum(x[0]*x[1]*x[2] for x in vals)/tw if tw > 0 else np.nan

    nw, nl = len(w_rows), len(l_rows)

    # Serve pts won = (1stWon + 2ndWon) / svpt
    def srv_won(rows, pfx):
        s  = pd.to_numeric(rows.get(f'{pfx}_svpt', pd.Series()), errors='coerce').replace(0,np.nan)
        fw = pd.to_numeric(rows.get(f'{pfx}_1stWon', pd.Series()), errors='coerce')
        sw = pd.to_numeric(rows.get(f'{pfx}_2ndWon', pd.Series()), errors='coerce')
        n  = min(len(s), len(fw), len(sw))
        if n == 0: return np.nan
        return float(((fw.values[:n]+sw.values[:n])/s.values[:n]).mean())

    spw = merge(srv_won(w_rows,'w'), srv_won(l_rows,'l'), nw, nl)

    # Return pts won = (opp_svpt - opp_1stWon - opp_2ndWon) / opp_svpt
    def ret_won(opp_rows, pfx):
        s  = pd.to_numeric(opp_rows.get(f'{pfx}_svpt', pd.Series()), errors='coerce').replace(0,np.nan)
        fw = pd.to_numeric(opp_rows.get(f'{pfx}_1stWon', pd.Series()), errors='coerce')
        sw = pd.to_numeric(opp_rows.get(f'{pfx}_2ndWon', pd.Series()), errors='coerce')
        n  = min(len(s), len(fw), len(sw))
        if n == 0: return np.nan
        return float(((s.values[:n]-fw.values[:n]-sw.values[:n]).clip(0)/s.values[:n]).mean())

    rpw = merge(ret_won(w_rows,'l'), ret_won(l_rows,'w'), nw, nl)

    # BP saved %
    bps_v = merge(ratio('bpSaved','bpFaced',w_rows,'w'), ratio('bpSaved','bpFaced',l_rows,'l'), nw, nl)

    # BP converted %
    def bp_conv(opp_rows, pfx):
        bpf = pd.to_numeric(opp_rows.get(f'{pfx}_bpFaced', pd.Series()), errors='coerce').replace(0,np.nan)
        bps_c = pd.to_numeric(opp_rows.get(f'{pfx}_bpSaved', pd.Series()), errors='coerce')
        n = min(len(bpf), len(bps_c))
        if n == 0: return np.nan
        return float(((bpf.values[:n]-bps_c.values[:n]).clip(0)/bpf.values[:n]).mean())

    bpc_v = merge(bp_conv(w_rows,'l'), bp_conv(l_rows,'w'), nw, nl)

    # 1st serve %
    fs_pct = merge(ratio('1stIn','svpt',w_rows,'w'), ratio('1stIn','svpt',l_rows,'l'), nw, nl)

    # Ace rate
    ace_r = merge(ratio('ace','svpt',w_rows,'w'), ratio('ace','svpt',l_rows,'l'), nw, nl)

    # DF rate (lower = better)
    df_r  = merge(ratio('df','svpt',w_rows,'w'),  ratio('df','svpt',l_rows,'l'),  nw, nl)

    win_rate = nw / max(nw+nl, 1)

    return {
        'srv_won':  spw  if spw  is not None and not np.isnan(spw)  else 0.62,
        'ret_won':  rpw  if rpw  is not None and not np.isnan(rpw)  else 0.38,
        'bp_saved': bps_v if bps_v is not None and not np.isnan(bps_v) else 0.60,
        'bp_conv':  bpc_v if bpc_v is not None and not np.isnan(bpc_v) else 0.35,
        'fs_pct':   fs_pct if fs_pct is not None and not np.isnan(fs_pct) else 0.62,
        'ace_rate': ace_r if ace_r is not None and not np.isnan(ace_r) else 0.05,
        'df_rate':  df_r  if df_r  is not None and not np.isnan(df_r)  else 0.05,
        'win_rate': win_rate,
        'n':        nw+nl,
    }


@st.cache_resource
def build_model(n_rows, df):
    """
    Train two models:
    1. Winner classifier  — GBM trained on rolling pre-match differential features
    2. Games regressor   — GBM trained on match stats to predict total games
    Uses rolling features (no data leakage) and symmetric mirrored rows for balance.
    """
    df_t = df.copy()
    df_t['Total_Games'] = df_t.apply(calculate_total_games, axis=1)
    df_t = df_t.dropna(subset=['Total_Games'])
    df_t = df_t[(df_t['Total_Games'] > 5) & (df_t['Total_Games'] < 55)]

    if len(df_t) < 50:
        return None

    # ── Step 1: Compute Elo ratings in temporal order ─────────────────────────
    _, df_elo = compute_elo_ratings(df_t)
    df_elo = df_elo.reset_index(drop=True)

    # ── Step 2: Build rolling pre-match features for every match ──────────────
    X_clf_rows, y_clf = [], []

    for idx, row in df_elo.iterrows():
        w = row['Winner']
        l = row['Loser']
        surf = row.get('Surface', 'Hard')

        sw = rolling_player_stats(df_elo, w, idx, window=20, surface=surf)
        sl = rolling_player_stats(df_elo, l, idx, window=20, surface=surf)

        if sw is None or sl is None:
            continue

        elo_w = row['elo_winner_pre']
        elo_l = row['elo_loser_pre']

        wrank = float(pd.to_numeric(row.get('WRank', 300), errors='coerce') or 300)
        lrank = float(pd.to_numeric(row.get('LRank', 300), errors='coerce') or 300)

        def feat_vec(s_a, s_b, elo_a, elo_b, rank_a, rank_b):
            """Differential feature vector: positive = favours player A."""
            return [
                s_a['srv_won']  - s_b['srv_won'],        # serve dominance diff
                s_a['ret_won']  - s_b['ret_won'],        # return dominance diff
                s_a['bp_saved'] - s_b['bp_saved'],       # bp saved diff
                s_a['bp_conv']  - s_b['bp_conv'],        # bp converted diff
                s_a['fs_pct']   - s_b['fs_pct'],         # 1st serve % diff
                s_a['ace_rate'] - s_b['ace_rate'],       # ace rate diff
                s_b['df_rate']  - s_a['df_rate'],        # df rate diff (inverted)
                s_a['win_rate'] - s_b['win_rate'],       # form diff
                (elo_a - elo_b) / 400,                   # elo diff (normalised)
                np.log1p(rank_b) - np.log1p(rank_a),    # rank diff (log, inverted)
                s_a['srv_won'] + s_a['ret_won'],         # A total dominance
                s_b['srv_won'] + s_b['ret_won'],         # B total dominance
                # Interaction: serve + return combo
                (s_a['srv_won'] - s_b['srv_won']) * (s_a['ret_won'] - s_b['ret_won']),
                # Elo × rank combined signal
                ((elo_a - elo_b) / 400) * (np.log1p(rank_b) - np.log1p(rank_a)),
            ]

        # Original row: winner=A → label 1
        fv = feat_vec(sw, sl, elo_w, elo_l, wrank, lrank)
        X_clf_rows.append(fv)
        y_clf.append(1)

        # Mirrored row: winner=B → label 0  (balances the dataset perfectly)
        fv_mirror = feat_vec(sl, sw, elo_l, elo_w, lrank, wrank)
        X_clf_rows.append(fv_mirror)
        y_clf.append(0)

    if len(X_clf_rows) < 80:
        return None

    X_clf = np.nan_to_num(np.array(X_clf_rows, dtype=float), nan=0, posinf=0, neginf=0)
    y_clf = np.array(y_clf)

    # ── Step 3: Train classifier ──────────────────────────────────────────────
    X_tr, X_te, y_tr, y_te = train_test_split(X_clf, y_clf, test_size=0.2,
                                               random_state=42, stratify=y_clf)
    sc_clf = StandardScaler()
    X_tr_s = sc_clf.fit_transform(X_tr)
    X_te_s  = sc_clf.transform(X_te)

    clf = GradientBoostingClassifier(
        n_estimators=500,
        learning_rate=0.03,
        max_depth=3,
        min_samples_split=8,
        min_samples_leaf=4,
        subsample=0.8,
        max_features=0.8,
        random_state=42
    )
    clf.fit(X_tr_s, y_tr)
    clf_acc = accuracy_score(y_te, clf.predict(X_te_s))

    # Cross-val for robustness estimate
    cv_scores = cross_val_score(clf, sc_clf.transform(X_clf), y_clf, cv=5, scoring='accuracy')

    # ── Step 4: Build games regressor (uses match stats, no leakage issue) ────
    def col_num(c):
        return pd.to_numeric(df_elo[c], errors='coerce').fillna(0).values if c in df_elo.columns else np.zeros(len(df_elo))

    w1, l1 = col_num('W1'), col_num('L1')
    w2, l2 = col_num('W2'), col_num('L2')
    w3, l3 = col_num('W3'), col_num('L3')
    wrank_v = pd.to_numeric(df_elo['WRank'], errors='coerce').fillna(300).values
    lrank_v = pd.to_numeric(df_elo['LRank'], errors='coerce').fillna(300).values

    reg_feats = [
        w1+l1, w2+l2, np.where(w3+l3>0, w3+l3, 0),
        (df_elo['Wsets']==2).astype(float).values if 'Wsets' in df_elo.columns else np.zeros(len(df_elo)),
        (df_elo['Wsets']==3).astype(float).values if 'Wsets' in df_elo.columns else np.zeros(len(df_elo)),
        lrank_v - wrank_v,
        wrank_v / (wrank_v + lrank_v + 1),
        np.log1p(wrank_v), np.log1p(lrank_v),
        1 / (1 + np.abs(w1-l1) + np.abs(w2-l2)),
        df_elo['elo_winner_pre'].values, df_elo['elo_loser_pre'].values,
        (df_elo['elo_winner_pre'] - df_elo['elo_loser_pre']).values,
    ]
    w_svpt = col_num('w_svpt').clip(1)
    l_svpt = col_num('l_svpt').clip(1)
    w_bpf  = col_num('w_bpFaced').clip(1)
    l_bpf  = col_num('l_bpFaced').clip(1)
    for c in ['w_ace','w_df','w_svpt','w_1stIn','w_1stWon','w_2ndWon','w_bpSaved','w_bpFaced',
              'l_ace','l_df','l_svpt','l_1stIn','l_1stWon','l_2ndWon','l_bpSaved','l_bpFaced']:
        reg_feats.append(col_num(c))
    reg_feats += [
        col_num('w_1stIn')/w_svpt, col_num('w_1stWon')/w_svpt,
        (col_num('w_1stWon')+col_num('w_2ndWon'))/w_svpt,
        col_num('l_1stIn')/l_svpt, col_num('l_1stWon')/l_svpt,
        (col_num('l_1stWon')+col_num('l_2ndWon'))/l_svpt,
        col_num('w_bpSaved')/w_bpf, col_num('l_bpSaved')/l_bpf,
        col_num('Minutes'),
    ]
    for surf in df_elo['Surface'].dropna().unique():
        reg_feats.append((df_elo['Surface']==surf).astype(int).values)

    X_reg = np.nan_to_num(np.column_stack(reg_feats), nan=0, posinf=0, neginf=0)
    y_reg = df_elo['Total_Games'].values

    Xr_tr, Xr_te, yr_tr, yr_te = train_test_split(X_reg, y_reg, test_size=0.2, random_state=42)
    sc_reg = StandardScaler()
    reg = GradientBoostingRegressor(
        n_estimators=600, learning_rate=0.02, max_depth=4,
        min_samples_split=6, min_samples_leaf=3,
        subsample=0.75, max_features=0.8, random_state=42
    )
    reg.fit(sc_reg.fit_transform(Xr_tr), yr_tr)
    yr_pred = reg.predict(sc_reg.transform(Xr_te))

    return {
        'clf':          clf,
        'sc_clf':       sc_clf,
        'clf_acc':      clf_acc,
        'cv_acc':       float(cv_scores.mean()),
        'cv_std':       float(cv_scores.std()),
        'reg':          reg,
        'sc_reg':       sc_reg,
        'r2':           r2_score(yr_te, yr_pred),
        'mae':          mean_absolute_error(yr_te, yr_pred),
        'df':           df_elo,
        'y_test':       yr_te,
        'y_pred':       yr_pred,
        'elo_ratings':  compute_elo_ratings(df_elo)[0],  # final ratings
    }


def predict_games(model_data, player_a, player_b, surface, df):
    """Predict total games using each player's recent median + h2h + serve quality."""
    def player_recent(p, surf, n=15):
        m = df[(df['Winner']==p) | (df['Loser']==p)]
        if surf != 'All':
            ms = m[m['Surface']==surf]
            if len(ms) >= 5: m = ms
        m = m.tail(n).copy()
        m['Total_Games'] = m.apply(calculate_total_games, axis=1)
        return m.dropna(subset=['Total_Games'])

    a_m = player_recent(player_a, surface)
    b_m = player_recent(player_b, surface)
    a_med = float(a_m['Total_Games'].median()) if len(a_m) >= 3 else 22.0
    b_med = float(b_m['Total_Games'].median()) if len(b_m) >= 3 else 22.0
    base  = (a_med + b_med) / 2.0

    h2h = df[
        ((df['Winner']==player_a) & (df['Loser']==player_b)) |
        ((df['Winner']==player_b) & (df['Loser']==player_a))
    ].copy()
    if len(h2h) >= 2:
        h2h['Total_Games'] = h2h.apply(calculate_total_games, axis=1)
        h2h_med = h2h['Total_Games'].dropna().median()
        if not np.isnan(h2h_med):
            base = base * 0.65 + h2h_med * 0.35

    def avg_bp_saved(p):
        rows = df[df['Winner']==p].tail(10)
        if len(rows) == 0 or 'w_bpSaved' not in rows.columns: return 0.5
        bps = pd.to_numeric(rows['w_bpSaved'], errors='coerce')
        bpf = pd.to_numeric(rows['w_bpFaced'], errors='coerce').replace(0, np.nan)
        r = (bps / bpf).dropna()
        return float(r.mean()) if len(r) > 0 else 0.5

    serve_adj = (0.5 - (avg_bp_saved(player_a) + avg_bp_saved(player_b)) / 2) * 4.0
    return float(np.clip(base + serve_adj, 12, 45))

# ============= MATCH RESULT PREDICTION =============

def predict_match_result(player_a, player_b, surface, df, skills_a, skills_b, model_data):
    """
    Uses the trained GBM classifier with rolling pre-match differential features.
    Falls back to a rule-based approach if not enough history for either player.
    """
    df_elo = model_data['df']
    clf    = model_data['clf']
    sc_clf = model_data['sc_clf']
    elo    = model_data['elo_ratings']

    n_matches = len(df_elo)

    sa = rolling_player_stats(df_elo, player_a, n_matches, window=20, surface=surface)
    sb = rolling_player_stats(df_elo, player_b, n_matches, window=20, surface=surface)

    elo_a = elo.get(player_a, 1500)
    elo_b = elo.get(player_b, 1500)

    rank_a = float(pd.to_numeric(
        df_elo[(df_elo['Winner']==player_a)|(df_elo['Loser']==player_a)].apply(
            lambda r: r['WRank'] if r['Winner']==player_a else r['LRank'], axis=1
        ), errors='coerce').dropna().median() or 300)
    rank_b = float(pd.to_numeric(
        df_elo[(df_elo['Winner']==player_b)|(df_elo['Loser']==player_b)].apply(
            lambda r: r['WRank'] if r['Winner']==player_b else r['LRank'], axis=1
        ), errors='coerce').dropna().median() or 300)

    if sa is None: sa = {'srv_won':0.62,'ret_won':0.38,'bp_saved':0.60,'bp_conv':0.35,
                         'fs_pct':0.62,'ace_rate':0.05,'df_rate':0.05,'win_rate':0.5,'n':0}
    if sb is None: sb = {'srv_won':0.62,'ret_won':0.38,'bp_saved':0.60,'bp_conv':0.35,
                         'fs_pct':0.62,'ace_rate':0.05,'df_rate':0.05,'win_rate':0.5,'n':0}

    fv = [
        sa['srv_won']  - sb['srv_won'],
        sa['ret_won']  - sb['ret_won'],
        sa['bp_saved'] - sb['bp_saved'],
        sa['bp_conv']  - sb['bp_conv'],
        sa['fs_pct']   - sb['fs_pct'],
        sa['ace_rate'] - sb['ace_rate'],
        sb['df_rate']  - sa['df_rate'],
        sa['win_rate'] - sb['win_rate'],
        (elo_a - elo_b) / 400,
        np.log1p(rank_b) - np.log1p(rank_a),
        sa['srv_won'] + sa['ret_won'],
        sb['srv_won'] + sb['ret_won'],
        (sa['srv_won'] - sb['srv_won']) * (sa['ret_won'] - sb['ret_won']),
        ((elo_a - elo_b) / 400) * (np.log1p(rank_b) - np.log1p(rank_a)),
    ]

    X = np.nan_to_num(np.array([fv]), nan=0, posinf=0, neginf=0)
    prob_a = float(clf.predict_proba(sc_clf.transform(X))[0][1])
    prob_b = 1.0 - prob_a

    winner   = player_a if prob_a >= 0.5 else player_b
    loser    = player_b if winner == player_a else player_a
    win_prob = prob_a   if winner == player_a else prob_b

    if win_prob >= 0.72:   confidence = 'High'
    elif win_prob >= 0.60: confidence = 'Medium'
    else:                  confidence = 'Low'

    # Head-to-head
    h2h = df[
        ((df['Winner']==player_a) & (df['Loser']==player_b)) |
        ((df['Winner']==player_b) & (df['Loser']==player_a))
    ]
    h2h_a_wins = int((h2h['Winner'] == player_a).sum())
    h2h_total  = len(h2h)

    # Factor contributions (signed, positive = favours A)
    factor_defs = [
        ('Serve dominance',       fv[0] * 5),
        ('Return game',           fv[1] * 5),
        ('Break pts saved',       fv[2] * 3),
        ('Break conversion',      fv[3] * 3),
        ('1st Serve %',           fv[4] * 2),
        ('Ace rate',              fv[5] * 2),
        ('DF control',            fv[6] * 2),
        ('Recent form',           fv[7] * 3),
        ('Elo rating',            fv[8] * 4),
        ('World ranking',         fv[9] * 3),
    ]
    if h2h_total >= 2:
        h2h_pct   = h2h_a_wins / h2h_total
        h2h_score = (h2h_pct - 0.5) * 4
        factor_defs.append(('Head-to-head', h2h_score))

    factors = [(label, score, player_a if score > 0 else player_b)
               for label, score in factor_defs if abs(score) > 0.01]
    factors  = sorted(factors, key=lambda x: abs(x[1]), reverse=True)

    # Scoreline generation
    margin    = abs(prob_a - 0.5) * 2   # 0..1
    p3sets    = float(np.clip(1.0 - margin * 2.5, 0.05, 0.70))
    seed      = int(abs(hash(player_a + player_b + surface)) % (2**31))
    rng       = np.random.RandomState(seed)
    three_sets = bool(rng.random() < p3sets)

    def set_sc(dominant, close):
        if close:            opts, wts = [(7,6),(7,5),(6,4)], [0.35,0.35,0.30]
        elif dominant > 0.5: opts, wts = [(6,1),(6,2),(6,3)], [0.20,0.45,0.35]
        else:                opts, wts = [(6,3),(6,4),(7,5)], [0.35,0.40,0.25]
        return opts[rng.choice(len(opts), p=wts)]

    close = margin < 0.15
    s1 = set_sc(margin, close)
    s2 = set_sc(margin, close and three_sets)
    if three_sets:
        s3 = set_sc(margin * 0.4, True)
        score_parts = [f"{s1[0]}-{s1[1]}", f"{s3[1]}-{s3[0]}", f"{s2[0]}-{s2[1]}"]
    else:
        score_parts = [f"{s1[0]}-{s1[1]}", f"{s2[0]}-{s2[1]}"]

    return {
        'winner':          winner,
        'loser':           loser,
        'win_prob':        win_prob,
        'win_prob_a':      prob_a,
        'predicted_score': ' '.join(score_parts),
        'three_sets':      three_sets,
        'confidence':      confidence,
        'factors':         factors,
        'h2h_total':       h2h_total,
        'h2h_a_wins':      h2h_a_wins,
        'elo_a':           elo_a,
        'elo_b':           elo_b,
        'sta':             sa,
        'stb':             sb,
    }


# ============= HTML REPORT =============

def generate_html_report(player_a, player_b, surface, an_a, an_b,
                          prediction, model_data, rs_a, rs_b, result):
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    if prediction < 23:   mt, col = "Quick Match",  "#16a34a"
    elif prediction < 27: mt, col = "Competitive",  "#d97706"
    else:                 mt, col = "Long Match",    "#dc2626"

    conf_color = {'High': '#16a34a', 'Medium': '#d97706', 'Low': '#dc2626'}[result['confidence']]
    prob_a = result['win_prob_a'] * 100
    prob_b = (1 - result['win_prob_a']) * 100

    factor_rows = ''
    for label, score, adv in result['factors']:
        pct = min(100, int(abs(score) / 3.0 * 100))
        bar_col = '#c8ff00' if adv == player_a else '#ff5555'
        side = f'&larr; {player_a}' if adv == player_a else f'{player_b} &rarr;'
        factor_rows += (
            f'<div class="frow"><div class="flabel">{label}</div>'
            f'<div class="fbar-wrap"><div class="fbar" style="width:{pct}%;background:{bar_col}"></div></div>'
            f'<div class="fside" style="color:{bar_col}">{side}</div></div>'
        )

    h2h_note = (
        f"H2H: {result['h2h_a_wins']}-{result['h2h_total']-result['h2h_a_wins']} in {result['h2h_total']} meetings"
        if result['h2h_total'] > 0 else "No previous H2H matches"
    )

    def skill_bar(val, name):
        p = int(np.clip(val * 100, 0, 100))
        return (
            f'<div class="skill">'
            f'<div class="sn">{name}</div>'
            f'<div class="bc"><div class="bf" style="width:{p}%">'
            f'<div class="bv">{p}%</div></div></div></div>'
        )

    def pcard(name, an, rs):
        s, l, f = an['skills'], an['last15'], an['fatigue']
        raw = s.get('_raw', {})

        def r(key, fmt='.1%'):
            v = raw.get(key, 0) or 0
            return format(v, fmt)

        return (
            f'<div class="pc"><h3>{name}</h3>'
            f'<div class="ss"><h4>Last 15 - {surface}</h4>'
            f'<div class="st"><span>Record</span><span>{l["wins"]}-{l["losses"]}</span></div>'
            f'<div class="st"><span>Win Rate</span><span>{l["win_rate"]:.1%}</span></div>'
            f'<div class="st"><span>Avg Games</span><span>{l["avg_games"]:.1f}</span></div>'
            f'<div class="st"><span>Form</span><span>{l["form"]}</span></div></div>'
            f'<div class="ss"><h4>Serve Stats</h4>'
            f'<div class="st"><span>1st Serve %</span><span>{rs["first_serve_pct"]:.1f}%</span></div>'
            f'<div class="st"><span>1st Serve Won %</span><span>{rs["first_serve_won_pct"]:.1f}%</span></div>'
            f'<div class="st"><span>2nd Serve Won %</span><span>{rs["second_serve_won_pct"]:.1f}%</span></div>'
            f'<div class="st"><span>Avg Aces</span><span>{rs["avg_aces"]:.1f}</span></div>'
            f'<div class="st"><span>Avg DFs</span><span>{rs["avg_df"]:.1f}</span></div>'
            f'<div class="st"><span>BP Saved %</span><span>{rs["bp_saved_pct"]:.1f}%</span></div></div>'
            f'<div class="ss"><h4>Fatigue</h4>'
            f'<div class="st"><span>Days Rest</span><span>{f["days_rest"]}</span></div>'
            f'<div class="st"><span>Matches (7d)</span><span>{f["matches_last_7"]}</span></div>'
            f'<div class="st"><span>Status</span><span>{f["fatigue_level"]}</span></div></div>'
            f'<div class="ss"><h4>Serve Skills</h4>'
            + skill_bar(s['first_serve_pct'],      f'1ST SERVE IN - {r("first_serve_pct")}')
            + skill_bar(s['first_serve_won_pct'],  f'1ST SERVE WON - {r("first_serve_won_pct")}')
            + skill_bar(s['second_serve_won_pct'], f'2ND SERVE WON - {r("second_serve_won_pct")}')
            + skill_bar(s['hold_pct'],             f'SERVICE HOLD - {r("hold_pct")}')
            + skill_bar(s['bp_saved_pct'],         f'BP SAVED - {r("bp_saved_pct")}')
            + skill_bar(s['ace_per_svgm'],         f'ACES/GAME - {r("ace_per_svgm", ".2f")}')
            + skill_bar(s['df_per_svgm'],          f'DF CONTROL - {r("df_per_svgm", ".2f")}/gm')
            + f'</div><div class="ss"><h4>Return Skills</h4>'
            + skill_bar(s['return_pts_won_pct'],   f'RETURN PTS WON - {r("return_pts_won_pct")}')
            + skill_bar(s['break_conversion_pct'], f'BREAK CONVERSION - {r("break_conversion_pct")}')
            + skill_bar(s['dominance_ratio'],      f'DOMINANCE RATIO - {r("dominance_ratio", ".2f")}')
            + '</div></div>'
        )

    pa_pct = f"{prob_a:.1f}"
    pb_pct = f"{prob_b:.1f}"

    css = f"""
*{{margin:0;padding:0;box-sizing:border-box}}
body{{font-family:'IBM Plex Sans',sans-serif;background:#0a0a0a;color:#e0e0e0;padding:20px}}
.wrap{{max-width:1100px;margin:0 auto;background:#111;border:1px solid #2a2a2a;border-radius:4px;overflow:hidden}}
.hdr{{background:#0a0a0a;border-bottom:3px solid #c8ff00;padding:40px 30px;text-align:center}}
.hdr h1{{font-family:'Bebas Neue',sans-serif;font-size:3em;letter-spacing:6px;color:#c8ff00}}
.hdr p{{font-family:'IBM Plex Mono',monospace;font-size:0.75em;color:#666;margin-top:8px;letter-spacing:2px}}
.body{{padding:40px}}
.mt{{font-family:'Bebas Neue',sans-serif;font-size:2.6em;letter-spacing:4px;color:#fff;text-align:center;margin:20px 0 5px}}
.sb{{text-align:center;margin-bottom:20px;font-family:'IBM Plex Mono',monospace;font-size:0.85em;color:#c8ff00;letter-spacing:3px}}
.result-box{{background:#0d0d0d;border:1px solid #2a2a2a;border-top:4px solid #c8ff00;padding:30px;margin:20px 0}}
.result-winner{{font-family:'Bebas Neue',sans-serif;font-size:2.2em;letter-spacing:4px;color:#c8ff00;text-align:center}}
.result-score{{font-family:'Bebas Neue',sans-serif;font-size:3em;letter-spacing:6px;color:#fff;text-align:center;margin:10px 0}}
.result-conf{{text-align:center;font-family:'IBM Plex Mono',monospace;font-size:0.8em;letter-spacing:2px;color:{conf_color};margin-bottom:20px}}
.prob-wrap{{display:flex;height:36px;border:1px solid #333;overflow:hidden;margin:16px 0}}
.prob-a{{background:#c8ff00;display:flex;align-items:center;justify-content:center;font-family:'IBM Plex Mono',monospace;font-size:0.82em;font-weight:700;color:#000;width:{pa_pct}%}}
.prob-b{{background:#ff5555;display:flex;align-items:center;justify-content:center;font-family:'IBM Plex Mono',monospace;font-size:0.82em;font-weight:700;color:#fff;flex:1}}
.prob-labels{{display:flex;justify-content:space-between;font-family:'IBM Plex Mono',monospace;font-size:0.75em;color:#888;margin-top:4px}}
.factors{{margin-top:18px}}
.frow{{display:flex;align-items:center;gap:10px;margin:8px 0}}
.flabel{{font-family:'IBM Plex Mono',monospace;font-size:0.75em;color:#aaa;width:180px;flex-shrink:0}}
.fbar-wrap{{flex:1;height:18px;background:#1a1a1a;border:1px solid #222;overflow:hidden}}
.fbar{{height:100%}}
.fside{{font-family:'IBM Plex Mono',monospace;font-size:0.72em;width:130px;text-align:right;flex-shrink:0}}
.pb{{background:{col};padding:28px;text-align:center;margin:20px 0}}
.pb .lbl{{font-family:'IBM Plex Mono',monospace;font-size:0.82em;letter-spacing:3px;color:rgba(255,255,255,.8)}}
.pb .num{{font-family:'Bebas Neue',sans-serif;font-size:4em;letter-spacing:4px;color:#fff;line-height:1}}
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
"""

    html = (
        f'<!DOCTYPE html><html><head><meta charset="UTF-8">'
        f'<title>Challenger Prediction</title>'
        f'<link href="https://fonts.googleapis.com/css2?family=Bebas+Neue&family=IBM+Plex+Mono:wght@400;600&family=IBM+Plex+Sans:wght@300;400;600&display=swap" rel="stylesheet">'
        f'<style>{css}</style></head><body>'
        f'<div class="wrap">'
        f'<div class="hdr"><h1>CHALLENGER PREDICTOR</h1>'
        f'<p>MATCH RESULT - REAL SERVE STATS - LAST 15 MATCHES - SKILL RATINGS</p></div>'
        f'<div class="body">'
        f'<div class="mt">{player_a} VS {player_b}</div>'
        f'<div class="sb">SURFACE: {surface} &nbsp;-&nbsp; {h2h_note}</div>'
        f'<div class="result-box">'
        f'<div class="result-winner">PREDICTED WINNER: {result["winner"]}</div>'
        f'<div class="result-score">{result["winner"]} {result["predicted_score"]}</div>'
        f'<div class="result-conf">CONFIDENCE: {result["confidence"]} - WIN PROBABILITY: {result["win_prob"]:.1%}</div>'
        f'<div class="prob-wrap">'
        f'<div class="prob-a">{prob_a:.0f}% {player_a}</div>'
        f'<div class="prob-b">{player_b} {prob_b:.0f}%</div>'
        f'</div>'
        f'<div class="prob-labels"><span>{player_a}</span><span>{player_b}</span></div>'
        f'<div class="factors">'
        f'<div style="font-family:\'IBM Plex Mono\',monospace;font-size:0.75em;color:#666;letter-spacing:2px;margin-bottom:10px">DECISIVE FACTORS</div>'
        f'{factor_rows}'
        f'</div></div>'
        f'<div class="pb"><div class="lbl">{mt} - PREDICTED TOTAL GAMES</div>'
        f'<div class="num">{prediction:.1f}</div></div>'
        f'<div class="stitle">PLAYER ANALYSIS</div>'
        f'<div class="grid">{pcard(player_a,an_a,rs_a)}{pcard(player_b,an_b,rs_b)}</div>'
        f'<div class="info">WINNER ACCURACY = {model_data["clf_acc"]:.1%} - CV ACCURACY = {model_data["cv_acc"]:.1%} - GAMES MAE = +/-{model_data["mae"]:.2f} - {len(model_data["df"])} MATCHES</div>'
        f'</div>'
        f'<div class="ftr">GENERATED: {ts} - ATP CHALLENGER PREDICTOR - +/-{model_data["mae"]:.2f} GAMES</div>'
        f'</div></body></html>'
    )
    return html


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
    st.sidebar.metric("Winner Accuracy",  f"{model_data['clf_acc']:.1%}")
    st.sidebar.metric("CV Accuracy",      f"{model_data['cv_acc']:.1%} ±{model_data['cv_std']:.2f}")
    st.sidebar.metric("Games MAE ±",      f"{model_data['mae']:.2f}")
    st.sidebar.metric("Training matches", f"{len(model_data['df'])}")

    st.header("🎾 ATP Challenger Advanced Match Predictor")
    st.caption(f"Source: **{source_name}** · ML Winner Prediction · Rolling-30 Elo · Real Serve Stats")
    st.markdown("---")

    all_players = sorted(set(df['Winner'].dropna()) | set(df['Loser'].dropna()))
    surface_opts = ['All'] + surfaces_available

    c1, c2, c3 = st.columns(3)
    with c1: player_a = st.selectbox("Player 1", all_players)
    with c2: player_b = st.selectbox("Player 2", all_players, index=min(1, len(all_players)-1))
    with c3: surface  = st.selectbox("Surface",  surface_opts)

    st.markdown("---")

    # ── TABS ──────────────────────────────────────────────────────────────────
    tab_predict, tab_elo = st.tabs(["🔮 Match Prediction", "📡 Rolling-30 Elo"])

    # ══════════════════════════════════════════════════════════════════════════
    # TAB 1 — MATCH PREDICTION
    # ══════════════════════════════════════════════════════════════════════════
    with tab_predict:
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
                pred   = predict_games(model_data, player_a, player_b, surface, df)
                result = predict_match_result(
                    player_a, player_b, surface, df,
                    an_a['skills'], an_b['skills'], model_data
                )

            # ── MATCH RESULT ──────────────────────────────────────────────────
            st.markdown("---")
            st.subheader("🏆 MATCH PREDICTION")

            conf_emoji = {'High': '🟢', 'Medium': '🟡', 'Low': '🔴'}
            prob_a = result['win_prob_a']
            prob_b = 1.0 - prob_a

            win_col, lose_col = st.columns(2)
            with win_col:
                elo_w = int(result.get('elo_a', 1500) if result['winner'] == player_a else result.get('elo_b', 1500))
                st.success(f"### 🏆 {result['winner']}")
                ca, cb = st.columns(2)
                ca.metric("Win Probability", f"{result['win_prob']:.1%}")
                cb.metric("Elo Rating", f"{elo_w:.0f}")
            with lose_col:
                elo_l = int(result.get('elo_b', 1500) if result['loser'] == player_b else result.get('elo_a', 1500))
                st.error(f"### {result['loser']}")
                ca, cb = st.columns(2)
                ca.metric("Win Probability", f"{1-result['win_prob']:.1%}")
                cb.metric("Elo Rating", f"{elo_l:.0f}")

            st.markdown(
                f"<div style='text-align:center;padding:22px;background:#0d0d0d;border:1px solid #333;"
                f"border-top:3px solid #c8ff00;margin:12px 0;border-radius:4px;'>"
                f"<div style='font-size:0.75em;color:#888;letter-spacing:3px;font-family:monospace;"
                f"text-transform:uppercase'>Predicted Score</div>"
                f"<div style='font-size:2.6em;font-weight:900;color:#fff;letter-spacing:6px;margin:8px 0'>"
                f"{result['winner']}  {result['predicted_score']}</div>"
                f"<div style='font-size:0.82em;color:#aaa;font-family:monospace'>"
                f"{'3 sets' if result['three_sets'] else '2 sets'} &nbsp;·&nbsp; "
                f"Confidence: {conf_emoji[result['confidence']]} <strong>{result['confidence']}</strong>"
                f"&nbsp;·&nbsp; Model accuracy: {model_data['cv_acc']:.1%}</div>"
                f"</div>",
                unsafe_allow_html=True
            )

            min_pct = 8
            pa_show = max(min_pct, prob_a * 100)
            st.markdown(
                f"<div style='margin:12px 0'>"
                f"<div style='display:flex;height:34px;border-radius:4px;overflow:hidden;border:1px solid #333'>"
                f"<div style='width:{pa_show:.1f}%;background:#c8ff00;display:flex;align-items:center;"
                f"justify-content:center;font-size:0.8em;font-weight:700;color:#000;font-family:monospace;"
                f"white-space:nowrap;padding:0 6px'>{player_a} {prob_a:.0%}</div>"
                f"<div style='flex:1;background:#ff5555;display:flex;align-items:center;"
                f"justify-content:center;font-size:0.8em;font-weight:700;color:#fff;font-family:monospace;"
                f"white-space:nowrap;padding:0 6px'>{player_b} {prob_b:.0%}</div>"
                f"</div></div>",
                unsafe_allow_html=True
            )

            h2h_str = (
                f"H2H: **{result['h2h_a_wins']}-{result['h2h_total']-result['h2h_a_wins']}** "
                f"in favour of {player_a} ({result['h2h_total']} meetings)"
                if result['h2h_total'] > 0 else "No previous H2H in dataset"
            )
            elo_diff = int(result.get('elo_a', 1500) - result.get('elo_b', 1500))
            elo_lead = player_a if elo_diff >= 0 else player_b
            st.caption(f"{h2h_str}  ·  Elo gap: **{abs(elo_diff)} pts** in favour of {elo_lead}")

            with st.expander("📊 Why this prediction? (factor breakdown)", expanded=True):
                st.caption("Bar length = factor strength. Direction = favoured player.")
                for label, score, adv in result['factors']:
                    strength = float(np.clip(abs(score) / 5.0, 0.02, 0.99))
                    favour   = f"← **{adv}**" if score != 0 else "Even"
                    icon = "🟢" if adv == player_a else "🔴"
                    st.progress(strength, text=f"{icon} {label:<26} {favour}")

            st.markdown("---")
            st.subheader("📈 GAME TOTAL FORECAST")
            _, c, _ = st.columns([1,2,1])
            with c:
                st.metric("Expected Total Games", f"{pred:.1f}")
                if   pred < 23: st.info("⚡ Quick Match — 2 sets likely")
                elif pred < 27: st.info("⚔️ Competitive Match")
                else:           st.warning("🔥 Long Match — 3 sets likely")

            st.markdown("---")
            st.subheader("📊 PLAYER ANALYSIS")
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
            html = generate_html_report(player_a, player_b, surface,
                                         an_a, an_b, pred, model_data, rs_a, rs_b, result)
            st.download_button(
                "📥 Download HTML Report", data=html,
                file_name=f"Challenger_{player_a}_vs_{player_b}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.html",
                mime="text/html"
            )

    # ══════════════════════════════════════════════════════════════════════════
    # TAB 2 — ROLLING-30 ELO
    # ══════════════════════════════════════════════════════════════════════════
    with tab_elo:
        render_elo_tab(df, model_data, player_a, player_b, surface, all_players)

if __name__ == "__main__":
    main()

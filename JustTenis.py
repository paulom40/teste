import os
import re
import time
import warnings
from collections import defaultdict
from datetime import datetime, timedelta
import io

import numpy as np
import pandas as pd
import streamlit as st

from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score, f1_score, log_loss, accuracy_score

import requests

warnings.filterwarnings('ignore')

st.set_page_config(page_title="🎾 ATP & Challenger Predictor", page_icon="🎾", layout="wide")

st.markdown("""
    <style>
    .prediction-card {
        background: linear-gradient(135deg, #1e3c72 0%, #2a5298 100%);
        color: white;
        padding: 20px;
        border-radius: 15px;
        margin: 15px 0;
        box-shadow: 0 8px 16px rgba(0,0,0,0.3);
    }
    .confidence-high {color: #00ff88; font-weight: bold; font-size: 1.15em;}
    .confidence-medium {color: #ffd700; font-weight: bold; font-size: 1.15em;}
    .confidence-low {color: #ff6b6b; font-weight: bold; font-size: 1.15em;}
    </style>
""", unsafe_allow_html=True)


# ==============================================================================
# IMPROVED SURFACE DETECTION
# ==============================================================================

# Comprehensive tournament-to-surface mapping
TOURNAMENT_SURFACE_MAP = {
    # ATP 1000 Clay Events
    'monte carlo': 'Clay',
    'monaco': 'Clay',
    'madrid': 'Clay',
    'rome': 'Clay',
    'italian': 'Clay',
    
    # ATP 500 Clay Events
    'barcelona': 'Clay',
    'rio': 'Clay',
    'de janeiro': 'Clay',
    'buenos aires': 'Clay',
    'hamburg': 'Clay',
    'munich': 'Clay',
    'estoril': 'Clay',
    'geneva': 'Clay',
    'gstaad': 'Clay',
    'bastad': 'Clay',
    'umag': 'Clay',
    'kitzbuhel': 'Clay',
    
    # Challenger Clay Events (common locations)
    'oeiras': 'Clay',
    'quito': 'Clay',
    'santiago': 'Clay',
    'bogota': 'Clay',
    'santa cruz': 'Clay',
    'cordenons': 'Clay',
    'francavilla': 'Clay',
    'vicenza': 'Clay',
    'perugia': 'Clay',
    'prostejov': 'Clay',
    'bordeaux': 'Clay',
    'aix': 'Clay',
    'banja luka': 'Clay',
    'lyon': 'Clay',
    'orleans': 'Clay',
    'mouilleron': 'Clay',
    
    # ATP 1000 Grass Events
    'wimbledon': 'Grass',
    'queens': 'Grass',
    'halle': 'Grass',
    
    # ATP 500 Grass Events
    'stuttgart': 'Grass',
    's-hertogenbosch': 'Grass',
    'eastbourne': 'Grass',
    'newport': 'Grass',
    'mallorca': 'Grass',
    
    # All other major tournaments are Hard
    'australian open': 'Hard',
    'us open': 'Hard',
    'indian wells': 'Hard',
    'miami': 'Hard',
    'shanghai': 'Hard',
    'paris': 'Hard',
    'cincinnati': 'Hard',
    'canada': 'Hard',
    'dubai': 'Hard',
    'doha': 'Hard',
    'acapulco': 'Hard',
    'washington': 'Hard',
    'tokyo': 'Hard',
    'beijing': 'Hard',
    'vienna': 'Hard',
    'basel': 'Hard',
}

def detect_surface_from_tournament(tournament_name, surface_hint=None):
    """
    Enhanced surface detection using tournament name and optional hint.
    
    Args:
        tournament_name: Name of the tournament
        surface_hint: Optional surface hint from API or data source
    
    Returns:
        'Clay', 'Grass', or 'Hard'
    """
    if pd.isna(tournament_name):
        return surface_hint if surface_hint in ['Clay', 'Grass', 'Hard'] else 'Hard'
    
    tournament_lower = str(tournament_name).lower()
    
    # Check explicit keywords first
    if any(word in tournament_lower for word in ['clay', 'terre battue', 'antuka']):
        return 'Clay'
    if any(word in tournament_lower for word in ['grass', 'lawn', 'rasen']):
        return 'Grass'
    
    # Check tournament mapping
    for key, surface in TOURNAMENT_SURFACE_MAP.items():
        if key in tournament_lower:
            return surface
    
    # Check for common patterns
    # European cities in spring = likely clay
    if any(month in tournament_lower for month in ['april', 'may', 'june']):
        if any(country in tournament_lower for country in ['spain', 'portugal', 'italy', 'france', 'croatia', 'serbia']):
            return 'Clay'
    
    # Use surface hint if provided and valid
    if surface_hint in ['Clay', 'Grass', 'Hard']:
        return surface_hint
    
    # Default to Hard if uncertain
    return 'Hard'


def normalize_surface(s):
    """Legacy function for backward compatibility"""
    if pd.isna(s):
        return "Hard"
    s = str(s).lower()
    if "clay" in s:
        return "Clay"
    if "grass" in s:
        return "Grass"
    return "Hard"


# ==============================================================================
# ENHANCED ELO WITH MOMENTUM AND RECENT PERFORMANCE
# ==============================================================================

def calculate_enhanced_elo(df, k=32, surface_k=30, window=20):
    """
    Enhanced ELO calculation with surface-specific ratings and momentum.
    """
    players = set(df['winner'].dropna().unique()) | set(df['loser'].dropna().unique())

    elo = {p: 1500.0 for p in players}
    welo = {p: 1500.0 for p in players}
    surface_elo = {p: {'Hard':1500.0, 'Clay':1500.0, 'Grass':1500.0} for p in players}
    history = {p: [] for p in players}
    surface_history = {p: {'Hard':[], 'Clay':[], 'Grass':[]} for p in players}

    df_sorted = df.sort_values('date').copy()

    for _, row in df_sorted.iterrows():
        w = row['winner']
        l = row['loser']
        surf = row.get('surface', 'Hard')

        if surf not in ['Hard', 'Clay', 'Grass']:
            surf = 'Hard'

        if pd.isna(w) or pd.isna(l):
            continue

        history[w].append(1)
        history[l].append(0)
        
        surface_history[w][surf].append(1)
        surface_history[l][surf].append(0)

        # Calculate recent form (last N matches)
        w_recent = history[w][-window:]
        l_recent = history[l][-window:]

        w_form = sum(w_recent) / len(w_recent) if w_recent else 0.5
        l_form = sum(l_recent) / len(l_recent) if l_recent else 0.5

        # Calculate surface-specific form
        w_surf_recent = surface_history[w][surf][-10:]
        l_surf_recent = surface_history[l][surf][-10:]
        
        w_surf_form = sum(w_surf_recent) / len(w_surf_recent) if w_surf_recent else 0.5
        l_surf_form = sum(l_surf_recent) / len(l_surf_recent) if l_surf_recent else 0.5

        # Momentum bonus (stronger form bonus)
        r1 = elo[w] + 80 * (w_form - 0.5) + 40 * (w_surf_form - 0.5)
        r2 = elo[l] + 80 * (l_form - 0.5) + 40 * (l_surf_form - 0.5)

        exp1 = 1 / (1 + 10 ** ((r2 - r1) / 400))

        # Update general ELO
        elo[w] += k * (1 - exp1)
        elo[l] += k * (0 - (1 - exp1))

        # Update surface-specific ELO (higher K-factor for surface specialization)
        s1 = surface_elo[w][surf]
        s2 = surface_elo[l][surf]
        exp_s1 = 1 / (1 + 10 ** ((s2 - s1) / 400))

        surface_elo[w][surf] += surface_k * (1 - exp_s1)
        surface_elo[l][surf] += surface_k * (0 - (1 - exp_s1))

        # Weighted ELO (more weight on recent)
        welo[w] = welo[w] * 0.85 + elo[w] * 0.15
        welo[l] = welo[l] * 0.85 + elo[l] * 0.15

    return elo, welo, surface_elo, surface_history


# ==============================================================================
# LOAD HISTÓRICO
# ==============================================================================

@st.cache_data(ttl=3600)
def load_historical_data(file_path):
    df = pd.read_excel(file_path)
    df.columns = [str(c).strip().lower().replace(' ', '_') for c in df.columns]

    rename_map = {
        'winner_name':'winner',
        'loser_name':'loser',
        'tourney_date':'date',
        'winner_rank':'wrank',
        'loser_rank':'lrank'
    }
    df.rename(columns={k:v for k,v in rename_map.items() if k in df.columns}, inplace=True)

    df['date'] = pd.to_datetime(df['date'], errors='coerce')

    # IMPROVED: Use tournament-aware surface detection
    if 'surface' in df.columns and 'tourney_name' in df.columns:
        df['surface'] = df.apply(
            lambda row: detect_surface_from_tournament(
                row.get('tourney_name'), 
                normalize_surface(row.get('surface'))
            ), 
            axis=1
        )
    elif 'surface' in df.columns:
        df['surface'] = df['surface'].apply(normalize_surface)
    else:
        df['surface'] = "Hard"

    if 'score' in df.columns:
        def total_games_from_score(x):
            if pd.isna(x):
                return 22
            nums = [int(n) for n in re.findall(r'\d+', str(x))]
            return sum(nums) if nums else 22
        df['total_games'] = df['score'].apply(total_games_from_score)
    else:
        df['total_games'] = 22

    return df


# ==============================================================================
# ENHANCED PLAYER STATS
# ==============================================================================

def compute_player_stats(df):
    stats = {}
    elo, welo, surface_elo, surface_history = calculate_enhanced_elo(df)
    
    players = set(df['winner'].dropna().unique()) | set(df['loser'].dropna().unique())
    for player in players:
        matches = df[(df['winner'] == player) | (df['loser'] == player)]
        if len(matches) == 0:
            continue
            
        wins = len(df[df['winner'] == player])
        total = wins + len(df[df['loser'] == player])
        win_rate = wins / total if total > 0 else 0.5
        
        # Surface-specific stats
        surface_stats = {}
        surface_match_counts = {}
        for surf in ['Hard', 'Clay', 'Grass']:
            surf_matches = matches[matches['surface'] == surf]
            surface_match_counts[surf] = len(surf_matches)
            if len(surf_matches) > 0:
                surf_wins = len(surf_matches[surf_matches['winner'] == player])
                surface_stats[surf] = surf_wins / len(surf_matches)
            else:
                surface_stats[surf] = win_rate
        
        # Recent form (last 10 matches overall)
        recent = matches.sort_values('date', ascending=False).head(10)
        recent_form = len(recent[recent['winner'] == player]) / len(recent) if len(recent) > 0 else win_rate
        
        # Very recent form (last 5 matches - for momentum)
        very_recent = matches.sort_values('date', ascending=False).head(5)
        very_recent_form = len(very_recent[very_recent['winner'] == player]) / len(very_recent) if len(very_recent) > 0 else win_rate
        
        # Average ranking (lower is better)
        avg_rank = 150.0
        if 'wrank' in matches.columns:
            winner_ranks = matches[matches['winner'] == player]['wrank'].dropna()
            loser_ranks = matches[matches['loser'] == player]['lrank'].dropna()
            all_ranks = pd.concat([winner_ranks, loser_ranks])
            if len(all_ranks) > 0:
                avg_rank = float(all_ranks.mean())
        
        stats[player] = {
            'win_rate': win_rate,
            'avg_rank': avg_rank,
            'recent_form': recent_form,
            'very_recent_form': very_recent_form,
            'avg_games': float(matches['total_games'].mean()),
            'elo': float(elo[player]),
            'welo': float(welo[player]),
            'surface_elo': surface_elo[player],
            'surface_win_rate': surface_stats,
            'surface_match_count': surface_match_counts,
            'matches_played': float(total)
        }
    return stats


# ==============================================================================
# H2H
# ==============================================================================

def build_h2h_dict(df):
    h2h = defaultdict(int)
    h2h_surface = defaultdict(lambda: {'Hard': 0, 'Clay': 0, 'Grass': 0})
    
    for _, row in df.iterrows():
        if pd.notna(row['winner']) and pd.notna(row['loser']):
            pair = (row['winner'], row['loser'])
            surf = row.get('surface', 'Hard')
            h2h[pair] += 1
            h2h_surface[pair][surf] += 1
    
    return h2h, h2h_surface


# ==============================================================================
# ENHANCED FEATURES (with surface-specific h2h and more context)
# ==============================================================================

def build_features(p1, p2, surface, player_stats, h2h, h2h_surface, match=None):
    if p1 not in player_stats or p2 not in player_stats:
        return None

    s1 = player_stats[p1]
    s2 = player_stats[p2]

    surf = surface if surface in ['Hard', 'Clay', 'Grass'] else 'Hard'

    # H2H stats
    h2h_p1 = h2h.get((p1, p2), 0)
    h2h_p2 = h2h.get((p2, p1), 0)
    h2h_ratio = (h2h_p1 + 1) / (h2h_p1 + h2h_p2 + 2)
    
    # Surface-specific H2H
    h2h_surf_p1 = h2h_surface.get((p1, p2), {}).get(surf, 0)
    h2h_surf_p2 = h2h_surface.get((p2, p1), {}).get(surf, 0)
    h2h_surf_ratio = (h2h_surf_p1 + 0.5) / (h2h_surf_p1 + h2h_surf_p2 + 1)

    # Surface experience factor
    surf_exp_p1 = s1['surface_match_count'][surf]
    surf_exp_p2 = s2['surface_match_count'][surf]
    surf_exp_ratio = (surf_exp_p1 + 10) / (surf_exp_p2 + 10)

    feat = [
        # ELO ratios (higher weight)
        s1['elo'] / (s2['elo'] + 1),
        s1['welo'] / (s2['welo'] + 1),
        s1['surface_elo'][surf] / (s2['surface_elo'][surf] + 1),

        # Win rates and form
        s1['win_rate'] - s2['win_rate'],
        s1['recent_form'] - s2['recent_form'],
        s1['very_recent_form'] - s2['very_recent_form'],  # NEW: momentum
        s1['surface_win_rate'][surf] - s2['surface_win_rate'][surf],

        # Rankings (inverse ratio - lower rank is better)
        (s2['avg_rank'] + 1) / (s1['avg_rank'] + 1),

        # H2H features
        h2h_ratio,
        h2h_surf_ratio,  # NEW: surface-specific H2H

        # Game length prediction
        (s1['avg_games'] + s2['avg_games']) / 2,

        # Experience
        (s1['matches_played'] + 1) / (s2['matches_played'] + 1),
        surf_exp_ratio,  # NEW: surface experience
    ]

    # Odds-based probability
    prob1 = prob2 = 0.5
    if match is not None:
        odd1 = match.get("odd_p1")
        odd2 = match.get("odd_p2")
        if odd1 and odd2 and odd1 > 1 and odd2 > 1:
            prob1 = 1.0 / odd1
            prob2 = 1.0 / odd2

    feat.extend([prob1, prob2, prob1 - prob2])

    return feat


# ==============================================================================
# CROSS-VALIDATION
# ==============================================================================

def cross_val_metrics(model, X, y, name=""):
    if len(np.unique(y)) < 2:
        st.write(f"⚠️ {name}: apenas uma classe presente.")
        return

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    aucs, f1s, logs, accs = [], [], [], []

    for train_idx, test_idx in skf.split(X, y):
        X_tr, X_te = X[train_idx], X[test_idx]
        y_tr, y_te = y[train_idx], y[test_idx]

        m = GradientBoostingClassifier(
            n_estimators=model.n_estimators,
            max_depth=model.max_depth,
            learning_rate=model.learning_rate,
            random_state=42
        )
        m.fit(X_tr, y_tr)

        probs = m.predict_proba(X_te)[:, 1]
        preds = (probs >= 0.5).astype(int)

        aucs.append(roc_auc_score(y_te, probs))
        logs.append(log_loss(y_te, probs))
        f1s.append(f1_score(y_te, preds))
        accs.append(accuracy_score(y_te, preds))

    st.write(
        f"🔍 {name} | AUC: {np.mean(aucs):.3f} | "
        f"F1: {np.mean(f1s):.3f} | "
        f"Acc: {np.mean(accs):.3f} | "
        f"LogLoss: {np.mean(logs):.3f}"
    )


# ==============================================================================
# FUNÇÃO PARA LIMPAR NaN / INF
# ==============================================================================

def clean_xy(X, y):
    X = np.array(X)
    y = np.array(y)
    mask = np.isfinite(X).all(axis=1)
    return X[mask], y[mask]


# ==============================================================================
# ENHANCED MODEL TRAINING
# ==============================================================================

def train_models(df, player_stats, h2h, h2h_surface):

    progress = st.progress(0)
    step = 0
    total_steps = 6

    X_winner, y_winner = [], []
    X_ou, y_ou = [], []
    X_sets, y_sets = [], []
    X_hcap, y_hcap = [], []

    for _, row in df.iterrows():
        w = row['winner']
        l = row['loser']
        if pd.isna(w) or pd.isna(l):
            continue

        surf = row.get('surface', 'Hard')
        total_games = row.get('total_games', 22)
        score = str(row.get('score', "")) if 'score' in df.columns else ""

        # AMOSTRA 1: p1 = winner
        feat1 = build_features(w, l, surf, player_stats, h2h, h2h_surface, match=None)
        if feat1 is not None:
            X_winner.append(feat1)
            y_winner.append(1)

            X_ou.append(feat1)
            y_ou.append(1 if total_games > 21.5 else 0)

            sets = score.split()
            if len(sets) == 2:
                X_sets.append(feat1)
                y_sets.append(0)
            elif len(sets) >= 3:
                X_sets.append(feat1)
                y_sets.append(1)

            if total_games <= 20:
                X_hcap.append(feat1)
                y_hcap.append(1)
            elif total_games >= 24:
                X_hcap.append(feat1)
                y_hcap.append(0)

        # AMOSTRA 2: p1 = loser
        feat2 = build_features(l, w, surf, player_stats, h2h, h2h_surface, match=None)
        if feat2 is not None:
            X_winner.append(feat2)
            y_winner.append(0)

    progress.progress((step := step + 1) / total_steps)

    # LIMPAR NaN / INF
    X_winner, y_winner = clean_xy(X_winner, y_winner)
    X_ou, y_ou = clean_xy(X_ou, y_ou)

    if len(X_sets):
        X_sets, y_sets = clean_xy(X_sets, y_sets)

    if len(X_hcap):
        X_hcap, y_hcap = clean_xy(X_hcap, y_hcap)

    # Enhanced model parameters
    model_winner = GradientBoostingClassifier(
        n_estimators=250, max_depth=5, learning_rate=0.04, random_state=42,
        subsample=0.8, min_samples_split=20, min_samples_leaf=10
    )
    model_winner.fit(X_winner, y_winner)
    progress.progress((step := step + 1) / total_steps)
    cross_val_metrics(model_winner, X_winner, y_winner, "Winner")
    progress.progress((step := step + 1) / total_steps)

    model_ou = GradientBoostingClassifier(
        n_estimators=200, max_depth=4, learning_rate=0.04, random_state=42,
        subsample=0.8
    )
    model_ou.fit(X_ou, y_ou)
    cross_val_metrics(model_ou, X_ou, y_ou, "Over/Under 21.5")
    progress.progress((step := step + 1) / total_steps)

    model_sets = None
    if len(X_sets) and len(np.unique(y_sets)) == 2:
        model_sets = GradientBoostingClassifier(
            n_estimators=200, max_depth=4, learning_rate=0.04, random_state=42,
            subsample=0.8
        )
        model_sets.fit(X_sets, y_sets)
        cross_val_metrics(model_sets, X_sets, y_sets, "Sets 2–0 vs 2–1")
    progress.progress((step := step + 1) / total_steps)

    model_hcap = None
    if len(X_hcap) and len(np.unique(y_hcap)) == 2:
        model_hcap = GradientBoostingClassifier(
            n_estimators=200, max_depth=4, learning_rate=0.04, random_state=42,
            subsample=0.8
        )
        model_hcap.fit(X_hcap, y_hcap)
        cross_val_metrics(model_hcap, X_hcap, y_hcap, "Handicap -2.5")
    progress.progress((step := step + 1) / total_steps)

    return model_winner, model_ou, model_sets, model_hcap


# ==============================================================================
# PREDIÇÃO (inclui ELO + WELO)
# ==============================================================================
def predict_match(model_winner, model_ou, model_sets, model_hcap,
                  player_stats, h2h, h2h_surface, match):

    p1 = match['player1']
    p2 = match['player2']
    surface = match['surface']

    feat = build_features(p1, p2, surface, player_stats, h2h, h2h_surface, match=match)
    if feat is None:
        return None

    X = np.array([feat])

    # Winner
    probs_w = model_winner.predict_proba(X)[0]
    p1_prob = probs_w[1]
    p2_prob = 1 - p1_prob

    # Over/Under
    probs_ou = model_ou.predict_proba(X)[0]
    over_prob = probs_ou[1]

    # Sets
    sets_pred = None
    if model_sets is not None:
        probs_sets = model_sets.predict_proba(X)[0]
        sets_pred = {
            'label': "2–1 / jogo longo" if probs_sets[1] > probs_sets[0] else "2–0 / vitória clara",
            'conf': max(probs_sets)
        }

    # Handicap
    hcap_pred = None
    if model_hcap is not None:
        probs_h = model_hcap.predict_proba(X)[0]
        hcap_pred = {
            'label': f"{p1} -2.5" if probs_h[1] > probs_h[0] else f"{p2} +2.5",
            'conf': max(probs_h)
        }

    # ELO + WELO
    elo_p1 = player_stats[p1]['elo']
    elo_p2 = player_stats[p2]['elo']
    welo_p1 = player_stats[p1]['welo']
    welo_p2 = player_stats[p2]['welo']
    
    # Surface ELO
    surf_elo_p1 = player_stats[p1]['surface_elo'][surface]
    surf_elo_p2 = player_stats[p2]['surface_elo'][surface]

    return {
        'winner': p1 if p1_prob > p2_prob else p2,
        'winner_conf': max(p1_prob, p2_prob),
        'p1_prob': p1_prob,
        'p2_prob': p2_prob,
        'ou': "Over 21.5" if over_prob > 0.5 else "Under 21.5",
        'ou_conf': max(over_prob, 1 - over_prob),
        'sets': sets_pred,
        'handicap': hcap_pred,
        'elo_p1': elo_p1,
        'elo_p2': elo_p2,
        'welo_p1': welo_p1,
        'welo_p2': welo_p2,
        'surf_elo_p1': surf_elo_p1,
        'surf_elo_p2': surf_elo_p2
    }


# ==============================================================================
# IMPROVED SCRAPER SOFASCORE
# ==============================================================================

def scrape_matches_sofascore(days_ahead=0):
    try:
        target_date = (datetime.utcnow() + timedelta(days=days_ahead)).strftime("%Y-%m-%d")

        url = f"https://api.sofascore.com/api/v1/sport/tennis/scheduled-events/{target_date}"

        headers = {
            "User-Agent": (
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/123.0.0.0 Safari/537.36"
            ),
            "Accept": "application/json, text/plain, */*",
            "Accept-Language": "en-US,en;q=0.9",
            "Origin": "https://www.sofascore.com",
            "Referer": "https://www.sofascore.com/"
        }

        r = requests.get(url, headers=headers, timeout=10)

        if r.status_code != 200:
            st.error(f"Erro SofaScore: HTTP {r.status_code}")
            return []

        data = r.json()

        if "events" not in data:
            return []

        matches = []

        for ev in data["events"]:
            try:
                tournament = ev["tournament"]["name"]
                category = ev["tournament"]["category"]["name"]

                if "WTA" in category.upper():
                    continue

                p1 = ev["homeTeam"]["name"]
                p2 = ev["awayTeam"]["name"]

                # Get surface from API if available
                api_surface = ev.get("groundType", "")
                
                # IMPROVED: Use tournament-aware surface detection
                surface = detect_surface_from_tournament(tournament, normalize_surface(api_surface))

                matches.append({
                    "tournament": tournament,
                    "player1": p1,
                    "player2": p2,
                    "surface": surface,
                    "type": category,
                    "odd_p1": None,
                    "odd_p2": None
                })

            except:
                continue

        return matches

    except Exception as e:
        st.error(f"Erro SofaScore: {e}")
        return []


# ==============================================================================
# APP
# ==============================================================================

def main():
    st.title("🎾 ATP & Challenger Tennis Predictor - ENHANCED")
    st.markdown("**Improved Surface Detection + Enhanced ELO + Surface-Specific H2H + Momentum**")

    with st.sidebar:
        uploaded_file = st.file_uploader("Upload Historical Data (Excel)", type=['xlsx'])
        st.markdown("---")

        col1, col2 = st.columns(2)
        with col1:
            if st.button("📅 Today"):
                st.session_state.matches = scrape_matches_sofascore(0)
        with col2:
            if st.button("📅 Tomorrow"):
                st.session_state.matches = scrape_matches_sofascore(1)

    if 'matches' not in st.session_state:
        st.session_state.matches = []

    if uploaded_file:
        with st.spinner("Training enhanced model..."):
            temp_path = "/tmp/tennis_data.xlsx"
            with open(temp_path, 'wb') as f:
                f.write(uploaded_file.read())
            
            df = load_historical_data(temp_path)
            st.write(f"📁 Loaded {len(df)} matches")
            
            # Show surface distribution
            st.write("🎾 Surface distribution in training data:")
            st.write(df['surface'].value_counts())

            player_stats = compute_player_stats(df)
            st.write(f"👥 Computed stats for {len(player_stats)} players")

            h2h, h2h_surface = build_h2h_dict(df)

            model_w, model_ou, model_sets, model_hcap = train_models(df, player_stats, h2h, h2h_surface)

            st.session_state.player_stats = player_stats
            st.session_state.h2h = h2h
            st.session_state.h2h_surface = h2h_surface
            st.session_state.model_winner = model_w
            st.session_state.model_ou = model_ou
            st.session_state.model_sets = model_sets
            st.session_state.model_hcap = model_hcap
            st.session_state.models_trained = True

            st.success("✅ Enhanced model trained successfully!")

    if st.session_state.get('matches') and st.session_state.get('models_trained'):
        st.header("🎯 Predictions")

        results_export = []

        for m in st.session_state.matches:
            pred = predict_match(
                st.session_state.model_winner,
                st.session_state.model_ou,
                st.session_state.model_sets,
                st.session_state.model_hcap,
                st.session_state.player_stats,
                st.session_state.h2h,
                st.session_state.h2h_surface,
                m
            )

            if pred is None:
                continue

            results_export.append({
                "Tournament": m['tournament'],
                "Player1": m['player1'],
                "Player2": m['player2'],
                "Surface": m['surface'],
                "Odd_P1": m.get("odd_p1"),
                "Odd_P2": m.get("odd_p2"),
                "Winner": pred['winner'],
                "Winner_Prob": pred['winner_conf'],
                "P1_Prob": pred['p1_prob'],
                "P2_Prob": pred['p2_prob'],
                "ELO_P1": pred['elo_p1'],
                "ELO_P2": pred['elo_p2'],
                "WELO_P1": pred['welo_p1'],
                "WELO_P2": pred['welo_p2'],
                "SurfELO_P1": pred['surf_elo_p1'],
                "SurfELO_P2": pred['surf_elo_p2'],
                "OU": pred['ou'],
                "OU_Prob": pred['ou_conf'],
                "Sets": pred['sets']['label'] if pred['sets'] else "",
                "Sets_Prob": pred['sets']['conf'] if pred['sets'] else "",
                "Handicap": pred['handicap']['label'] if pred['handicap'] else "",
                "Handicap_Prob": pred['handicap']['conf'] if pred['handicap'] else ""
            })

            conf_class = (
                "confidence-high" if pred['winner_conf'] >= 0.65 else
                "confidence-medium" if pred['winner_conf'] >= 0.55 else
                "confidence-low"
            )

            sets_line = ""
            if pred['sets'] is not None:
                sets_line = f"<br>🧩 Sets: {pred['sets']['label']} ({pred['sets']['conf']:.1%})"

            hcap_line = ""
            if pred['handicap'] is not None:
                hcap_line = f"<br>📏 Handicap -2.5: {pred['handicap']['label']} ({pred['handicap']['conf']:.1%})"

            st.markdown(f"""
            <div class="prediction-card">
                <b>{m['type']} • {m['surface']} Surface</b><br>
                <h3>{m['tournament']}</h3>
                <h3>{m['player1']} vs {m['player2']}</h3>

                🏆 Winner: <span class="{conf_class}">
                    {pred['winner']} ({pred['winner_conf']:.1%})
                </span><br>

                📊 {m['player1']}: {pred['p1_prob']:.1%} |
                {m['player2']}: {pred['p2_prob']:.1%}<br>

                📈 ELO: {pred['elo_p1']:.0f} | {pred['elo_p2']:.0f}<br>
                🔥 WELO: {pred['welo_p1']:.0f} | {pred['welo_p2']:.0f}<br>
                🎾 {m['surface']} ELO: {pred['surf_elo_p1']:.0f} | {pred['surf_elo_p2']:.0f}<br>

                🎲 O/U 21.5: {pred['ou']} ({pred['ou_conf']:.1%})
                {sets_line}
                {hcap_line}
            </div>
            """, unsafe_allow_html=True)

        if results_export:
            df_export = pd.DataFrame(results_export)
            buffer = io.BytesIO()
            df_export.to_excel(buffer, index=False, engine='openpyxl')
            buffer.seek(0)

            st.download_button(
                label="📥 Exportar previsões para Excel",
                data=buffer,
                file_name="tennis_predictions_enhanced.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
            )


if __name__ == "__main__":
    main()

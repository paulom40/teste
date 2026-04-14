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
from sklearn.ensemble import VotingClassifier, GradientBoostingClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score, accuracy_score, log_loss
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
import requests

warnings.filterwarnings('ignore')
st.set_page_config(page_title="🎾 ATP Predictor v2", page_icon="🎾", layout="wide")

# ==============================================================================
# VALUE BETTING SETTINGS
# ==============================================================================
MIN_EDGE = 0.05          # 5% de edge mínimo recomendado
KELLY_FRACTION = 0.25    # Staking conservador

# ==============================================================================
# SURFACE MAP (mantido e melhorado)
# ==============================================================================
TOURNAMENT_SURFACE_MAP = {
    'monte carlo': 'Clay', 'madrid': 'Clay', 'rome': 'Clay', 'barcelona': 'Clay',
    'munich': 'Clay', 'estoril': 'Clay', 'geneva': 'Clay', 'oeiras': 'Clay',
    'santa cruz': 'Clay', 'tallahassee': 'Clay', 'busan': 'Hard', 'wuning': 'Hard',
    # ... (podes adicionar mais)
}

def detect_surface_from_tournament(tournament_name, surface_hint=None):
    if pd.isna(tournament_name):
        return surface_hint if surface_hint in ['Clay', 'Grass', 'Hard'] else 'Hard'
    
    t = str(tournament_name).lower()
    for key, surf in TOURNAMENT_SURFACE_MAP.items():
        if key in t:
            return surf
    if any(x in t for x in ['clay', 'terre', 'antuka']):
        return 'Clay'
    if any(x in t for x in ['grass', 'lawn']):
        return 'Grass'
    return surface_hint if surface_hint in ['Clay', 'Grass', 'Hard'] else 'Hard'

# ==============================================================================
# ENHANCED ELO (mantido)
# ==============================================================================
def calculate_enhanced_elo(df, k=32, surface_k=35, window=20):
    # ... (mantém a tua função original - está boa)
    players = set(df['winner'].dropna().unique()) | set(df['loser'].dropna().unique())
    elo = {p: 1500.0 for p in players}
    welo = {p: 1500.0 for p in players}
    surface_elo = {p: {'Hard':1500.0, 'Clay':1500.0, 'Grass':1500.0} for p in players}
    
    history = {p: [] for p in players}
    surface_history = {p: {'Hard':[], 'Clay':[], 'Grass':[]} for p in players}
    
    df_sorted = df.sort_values('date').copy()
    for _, row in df_sorted.iterrows():
        w, l, surf = row['winner'], row['loser'], row.get('surface', 'Hard')
        if pd.isna(w) or pd.isna(l): continue
        if surf not in ['Hard', 'Clay', 'Grass']: surf = 'Hard'
        
        history[w].append(1); history[l].append(0)
        surface_history[w][surf].append(1); surface_history[l][surf].append(0)
        
        w_form = sum(history[w][-window:]) / len(history[w][-window:]) if history[w] else 0.5
        l_form = sum(history[l][-window:]) / len(history[l][-window:]) if history[l] else 0.5
        
        # Surface ELO update
        s1 = surface_elo[w][surf] + 60*(w_form-0.5)
        s2 = surface_elo[l][surf] + 60*(l_form-0.5)
        exp_s1 = 1 / (1 + 10 ** ((s2 - s1) / 400))
        surface_elo[w][surf] += surface_k * (1 - exp_s1)
        surface_elo[l][surf] += surface_k * (0 - (1 - exp_s1))
        
        # General ELO
        r1 = surface_elo[w][surf] + 50*(w_form-0.5)
        r2 = surface_elo[l][surf] + 50*(l_form-0.5)
        exp1 = 1 / (1 + 10 ** ((r2 - r1) / 400))
        elo[w] += k * 0.75 * (1 - exp1)
        elo[l] += k * 0.75 * (0 - (1 - exp1))
        
        welo[w] = welo[w] * 0.78 + surface_elo[w][surf] * 0.22
        welo[l] = welo[l] * 0.78 + surface_elo[l][surf] * 0.22
    
    return elo, welo, surface_elo

# ==============================================================================
# PLAYER STATS + NOVAS FEATURES
# ==============================================================================
def compute_player_stats(df):
    elo, welo, surface_elo = calculate_enhanced_elo(df)
    stats = {}
    players = set(df['winner'].dropna().unique()) | set(df['loser'].dropna().unique())
    
    for player in players:
        matches = df[(df['winner'] == player) | (df['loser'] == player)].copy()
        if len(matches) == 0: continue
            
        wins = len(df[df['winner'] == player])
        total = wins + len(df[df['loser'] == player])
        
        surface_stats = {}
        surface_count = {}
        for surf in ['Hard', 'Clay', 'Grass']:
            surf_m = matches[matches['surface'] == surf]
            surface_count[surf] = len(surf_m)
            surface_stats[surf] = len(surf_m[surf_m['winner'] == player]) / len(surf_m) if len(surf_m) > 0 else 0.5
        
        recent = matches.sort_values('date', ascending=False).head(10)
        very_recent = matches.sort_values('date', ascending=False).head(5)
        
        stats[player] = {
            'win_rate': wins / total if total > 0 else 0.5,
            'recent_form': len(recent[recent['winner'] == player]) / len(recent) if len(recent)>0 else 0.5,
            'very_recent_form': len(very_recent[very_recent['winner'] == player]) / len(very_recent) if len(very_recent)>0 else 0.5,
            'elo': float(elo[player]),
            'welo': float(welo[player]),
            'surface_elo': surface_elo[player],
            'surface_win_rate': surface_stats,
            'surface_match_count': surface_count,
            'matches_played': float(total)
        }
    return stats

# ==============================================================================
# BUILD FEATURES - MELHORADO
# ==============================================================================
def build_features(p1, p2, surface, player_stats, h2h, h2h_surface):
    if p1 not in player_stats or p2 not in player_stats:
        return None
    
    s1 = player_stats[p1]
    s2 = player_stats[p2]
    surf = surface if surface in ['Hard','Clay','Grass'] else 'Hard'
    
    h2h_p1 = h2h.get((p1,p2), 0)
    h2h_surf_p1 = h2h_surface.get((p1,p2), {}).get(surf, 0)
    h2h_surf_ratio = (h2h_surf_p1 + 0.5) / (h2h_surf_p1 + h2h_surface.get((p2,p1), {}).get(surf, 0) + 1)
    
    feat = [
        s1['surface_elo'][surf] / (s2['surface_elo'][surf] + 1),
        s1['surface_elo'][surf] / (s2['surface_elo'][surf] + 1),
        s1['welo'] / (s2['welo'] + 1),
        s1['elo'] / (s2['elo'] + 1),
        s1['surface_win_rate'][surf] - s2['surface_win_rate'][surf],
        s1['very_recent_form'] - s2['very_recent_form'],           # Momentum
        s1['recent_form'] - s2['recent_form'],
        (s1['surface_win_rate'][surf] * (s1['surface_match_count'][surf] + 5) / 
         (s2['surface_win_rate'][surf] * (s2['surface_match_count'][surf] + 5))),  # Experience weighted
        abs(s1['elo'] - s2['elo']) / 100,                         # Closeness
        1 if s1['matches_played'] > 80 else 0,                    # Experience flag
        h2h_surf_ratio,
        h2h_surf_ratio,
    ]
    return feat

# ==============================================================================
# TRAINING - ENSEMBLE
# ==============================================================================
def train_models(df, player_stats, h2h, h2h_surface):
    X, y = [], []
    for _, row in df.iterrows():
        if pd.isna(row['winner']) or pd.isna(row['loser']): continue
        feat = build_features(row['winner'], row['loser'], row['surface'], player_stats, h2h, h2h_surface)
        if feat:
            X.append(feat)
            y.append(1)
        feat2 = build_features(row['loser'], row['winner'], row['surface'], player_stats, h2h, h2h_surface)
        if feat2:
            X.append(feat2)
            y.append(0)
    
    X = np.array(X)
    y = np.array(y)
    
    models = [
        ('gb', GradientBoostingClassifier(n_estimators=350, max_depth=6, learning_rate=0.03, subsample=0.8, random_state=42)),
        ('xgb', XGBClassifier(n_estimators=400, max_depth=5, learning_rate=0.025, subsample=0.8, random_state=42, eval_metric='logloss')),
        ('lgb', LGBMClassifier(n_estimators=400, max_depth=6, learning_rate=0.03, subsample=0.8, random_state=42, verbose=-1))
    ]
    
    ensemble = VotingClassifier(models, voting='soft')
    ensemble.fit(X, y)
    
    # Cross validation
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    aucs = [roc_auc_score(y[test], ensemble.predict_proba(X[test])[:,1]) for train, test in skf.split(X,y)]
    st.write(f"✅ Ensemble AUC: {np.mean(aucs):.4f}")
    
    return ensemble

# ==============================================================================
# PREDICTION
# ==============================================================================
def predict_match(model, player_stats, h2h, h2h_surface, match):
    p1, p2, surface = match['player1'], match['player2'], match['surface']
    feat = build_features(p1, p2, surface, player_stats, h2h, h2h_surface)
    if feat is None:
        return None
    
    X = np.array([feat])
    prob_p1 = model.predict_proba(X)[0][1]
    prob_p2 = 1 - prob_p1
    
    # Value Betting
    odd_p1 = match.get('odd_p1')
    odd_p2 = match.get('odd_p2')
    
    edge_p1 = prob_p1 * (odd_p1 - 1) - (1 - prob_p1) if odd_p1 else 0
    edge_p2 = prob_p2 * (odd_p2 - 1) - (1 - prob_p2) if odd_p2 else 0
    
    if edge_p1 > MIN_EDGE:
        recommendation = p1
        edge = edge_p1
        kelly = max(0, (prob_p1 * odd_p1 - 1) / (odd_p1 - 1) * KELLY_FRACTION)
    elif edge_p2 > MIN_EDGE:
        recommendation = p2
        edge = edge_p2
        kelly = max(0, (prob_p2 * odd_p2 - 1) / (odd_p2 - 1) * KELLY_FRACTION)
    else:
        recommendation = "No Value Bet"
        edge = 0
        kelly = 0
    
    return {
        'winner': p1 if prob_p1 > prob_p2 else p2,
        'p1_prob': prob_p1,
        'p2_prob': prob_p2,
        'recommendation': recommendation,
        'edge': edge,
        'kelly': kelly
    }

# ==============================================================================
# STREAMLIT APP
# ==============================================================================
def main():
    st.title("🎾 ATP & Challenger Predictor v2 - Enhanced Ensemble + Value Betting")
    st.markdown("**Melhorias:** Ensemble + Value Edge + Novas Features + Kelly Staking")

    uploaded_file = st.file_uploader("Upload Historical Data (Excel)", type=['xlsx'])
    
    if uploaded_file:
        with st.spinner("A treinar modelo melhorado..."):
            df = pd.read_excel(uploaded_file)
            df.columns = [str(c).strip().lower().replace(' ', '_') for c in df.columns]
            df['date'] = pd.to_datetime(df['date'], errors='coerce')
            df['surface'] = df.apply(lambda row: detect_surface_from_tournament(row.get('tournament',''), row.get('surface')), axis=1)
            
            player_stats = compute_player_stats(df)
            h2h = defaultdict(int)
            h2h_surface = defaultdict(lambda: {'Hard':0,'Clay':0,'Grass':0})
            
            # Build H2H
            for _, row in df.iterrows():
                if pd.notna(row['winner']) and pd.notna(row['loser']):
                    h2h[(row['winner'], row['loser'])] += 1
                    surf = row.get('surface', 'Hard')
                    h2h_surface[(row['winner'], row['loser'])][surf] += 1
            
            model = train_models(df, player_stats, h2h, h2h_surface)
            
            st.session_state.model = model
            st.session_state.player_stats = player_stats
            st.session_state.h2h = h2h
            st.session_state.h2h_surface = h2h_surface
            st.success("✅ Modelo Ensemble treinado com sucesso!")

    # Previsões
    if st.session_state.get('model') and st.button("🔄 Carregar jogos de hoje"):
        matches = scrape_matches_sofascore(0)  # usa a tua função original
        results = []
        
        for m in matches:
            pred = predict_match(st.session_state.model, st.session_state.player_stats,
                               st.session_state.h2h, st.session_state.h2h_surface, m)
            if pred:
                results.append({
                    "Tournament": m['tournament'],
                    "Player1": m['player1'],
                    "Player2": m['player2'],
                    "Surface": m['surface'],
                    "Prob_P1": f"{pred['p1_prob']:.1%}",
                    "Prob_P2": f"{pred['p2_prob']:.1%}",
                    "Recommendation": pred['recommendation'],
                    "Edge": f"{pred['edge']:.1%}",
                    "Kelly_%": f"{pred['kelly']:.1%}"
                })
        
        df_res = pd.DataFrame(results)
        st.dataframe(df_res, use_container_width=True)
        
        buffer = io.BytesIO()
        df_res.to_excel(buffer, index=False)
        st.download_button("📥 Download Previsões", buffer.getvalue(), "predictions_v2.xlsx")

if __name__ == "__main__":
    main()

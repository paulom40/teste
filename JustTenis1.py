import os
import re
import warnings
from collections import defaultdict
from datetime import datetime, timedelta
import io
import numpy as np
import pandas as pd
import streamlit as st
from sklearn.ensemble import VotingClassifier, GradientBoostingClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
import requests

warnings.filterwarnings('ignore')
st.set_page_config(page_title="🎾 ATP Predictor v2.1", page_icon="🎾", layout="wide")

# ==============================================================================
# CONFIGURAÇÕES
# ==============================================================================
MIN_EDGE = 0.05
KELLY_FRACTION = 0.25

# ==============================================================================
# SURFACE DETECTION
# ==============================================================================
TOURNAMENT_SURFACE_MAP = {
    'monte carlo': 'Clay', 'madrid': 'Clay', 'rome': 'Clay', 'barcelona': 'Clay',
    'munich': 'Clay', 'estoril': 'Clay', 'geneva': 'Clay', 'oeiras': 'Clay',
    'santa cruz': 'Clay', 'tallahassee': 'Clay', 'busan': 'Hard', 'wuning': 'Hard',
}

def detect_surface_from_tournament(tournament_name, surface_hint=None):
    if pd.isna(tournament_name):
        return surface_hint if surface_hint in ['Clay', 'Grass', 'Hard'] else 'Hard'
    t = str(tournament_name).lower()
    for key, surf in TOURNAMENT_SURFACE_MAP.items():
        if key in t: return surf
    if any(x in t for x in ['clay', 'terre', 'antuka']): return 'Clay'
    if any(x in t for x in ['grass', 'lawn']): return 'Grass'
    return surface_hint if surface_hint in ['Clay', 'Grass', 'Hard'] else 'Hard'

# ==============================================================================
# ELO (mantido simples e funcional)
# ==============================================================================
def calculate_enhanced_elo(df, k=32, surface_k=35):
    players = set(df['winner'].dropna().unique()) | set(df['loser'].dropna().unique())
    elo = {p: 1500.0 for p in players}
    welo = {p: 1500.0 for p in players}
    surface_elo = {p: {'Hard':1500.0, 'Clay':1500.0, 'Grass':1500.0} for p in players}
    
    df_sorted = df.sort_values('date').copy()
    for _, row in df_sorted.iterrows():
        w, l = row['winner'], row['loser']
        surf = row.get('surface', 'Hard')
        if surf not in ['Hard', 'Clay', 'Grass']: surf = 'Hard'
        if pd.isna(w) or pd.isna(l): continue
            
        # Simple ELO update
        r1 = surface_elo[w][surf]
        r2 = surface_elo[l][surf]
        exp = 1 / (1 + 10 ** ((r2 - r1) / 400))
        surface_elo[w][surf] += surface_k * (1 - exp)
        surface_elo[l][surf] += surface_k * (0 - (1 - exp))
        
        elo[w] += k * (1 - exp)
        elo[l] += k * (0 - (1 - exp))
        welo[w] = welo[w] * 0.8 + surface_elo[w][surf] * 0.2
        welo[l] = welo[l] * 0.8 + surface_elo[l][surf] * 0.2
    return elo, welo, surface_elo

# ==============================================================================
# PLAYER STATS
# ==============================================================================
def compute_player_stats(df):
    elo, welo, surface_elo = calculate_enhanced_elo(df)
    stats = {}
    for player in set(df['winner'].dropna().unique()) | set(df['loser'].dropna().unique()):
        matches = df[(df['winner'] == player) | (df['loser'] == player)]
        if len(matches) == 0: continue
            
        wins = len(df[df['winner'] == player])
        total = len(matches)
        
        surface_stats = {}
        surface_count = {}
        for surf in ['Hard', 'Clay', 'Grass']:
            m = matches[matches['surface'] == surf]
            surface_count[surf] = len(m)
            surface_stats[surf] = len(m[m['winner'] == player]) / len(m) if len(m) > 0 else 0.5
        
        recent = matches.sort_values('date', ascending=False).head(8)
        very_recent = matches.sort_values('date', ascending=False).head(4)
        
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
# FEATURES MELHORADAS
# ==============================================================================
def build_features(p1, p2, surface, player_stats, h2h, h2h_surface):
    if p1 not in player_stats or p2 not in player_stats:
        return None
    s1 = player_stats[p1]
    s2 = player_stats[p2]
    surf = surface if surface in ['Hard','Clay','Grass'] else 'Hard'
    
    h2h_surf_ratio = 0.5
    pair = (p1, p2)
    if pair in h2h_surface:
        h2h_surf_ratio = (h2h_surface[pair].get(surf, 0) + 0.5) / (h2h_surface[pair].get(surf, 0) + h2h_surface.get((p2,p1), {}).get(surf, 0) + 1)
    
    return [
        s1['surface_elo'][surf] / (s2['surface_elo'][surf] + 1),
        s1['welo'] / (s2['welo'] + 1),
        s1['surface_win_rate'][surf] - s2['surface_win_rate'][surf],
        s1['very_recent_form'] - s2['very_recent_form'],
        s1['recent_form'] - s2['recent_form'],
        abs(s1['elo'] - s2['elo']) / 100,
        h2h_surf_ratio,
    ]

# ==============================================================================
# TRAINING
# ==============================================================================
def train_models(df, player_stats, h2h, h2h_surface):
    X, y = [], []
    for _, row in df.iterrows():
        if pd.isna(row.get('winner')) or pd.isna(row.get('loser')): continue
        feat = build_features(row['winner'], row['loser'], row.get('surface','Hard'), player_stats, h2h, h2h_surface)
        if feat:
            X.append(feat)
            y.append(1)
            X.append(build_features(row['loser'], row['winner'], row.get('surface','Hard'), player_stats, h2h, h2h_surface))
            y.append(0)
    
    X, y = np.array(X), np.array(y)
    
    ensemble = VotingClassifier([
        ('gb', GradientBoostingClassifier(n_estimators=300, max_depth=5, learning_rate=0.04, random_state=42)),
        ('xgb', XGBClassifier(n_estimators=350, max_depth=5, learning_rate=0.03, random_state=42)),
        ('lgb', LGBMClassifier(n_estimators=350, max_depth=5, learning_rate=0.03, random_state=42, verbose=-1))
    ], voting='soft')
    
    ensemble.fit(X, y)
    st.success(f"✅ Ensemble treinado com {len(X)} amostras")
    return ensemble

# ==============================================================================
# MAIN APP
# ==============================================================================
def main():
    st.title("🎾 ATP & Challenger Predictor v2.1 - Fixed & Improved")
    
    uploaded_file = st.file_uploader("📁 Upload do teu histórico (Excel)", type=['xlsx'])
    
    if uploaded_file:
        with st.spinner("Processando dados e treinando modelo..."):
            df = pd.read_excel(uploaded_file)
            
            # === LIMPEZA ROBUSTA DE COLUNAS ===
            df.columns = [str(c).strip().lower().replace(' ', '_').replace('-','_') for c in df.columns]
            
            # Renomear colunas comuns
            col_map = {
                'tourney_date': 'date',
                'tournament_date': 'date',
                'date': 'date',
                'winner_name': 'winner',
                'loser_name': 'loser',
                'tourney_name': 'tournament',
                'surface': 'surface'
            }
            for old, new in col_map.items():
                if old in df.columns and new not in df.columns:
                    df.rename(columns={old: new}, inplace=True)
            
            # Garantir coluna date
            if 'date' not in df.columns and 'tourney_date' in df.columns:
                df['date'] = df['tourney_date']
            df['date'] = pd.to_datetime(df['date'], errors='coerce')
            
            # Surface
            if 'surface' in df.columns:
                df['surface'] = df.apply(lambda row: detect_surface_from_tournament(row.get('tournament'), row.get('surface')), axis=1)
            else:
                df['surface'] = 'Clay'  # default para a maioria dos teus dados
            
            st.write(f"✅ Carregados **{len(df)}** jogos | Colunas: {list(df.columns)}")
            
            player_stats = compute_player_stats(df)
            h2h = defaultdict(int)
            h2h_surface = defaultdict(lambda: {'Hard':0, 'Clay':0, 'Grass':0})
            
            for _, row in df.iterrows():
                if pd.notna(row.get('winner')) and pd.notna(row.get('loser')):
                    pair = (row['winner'], row['loser'])
                    h2h[pair] += 1
                    surf = row.get('surface', 'Hard')
                    h2h_surface[pair][surf] += 1
            
            model = train_models(df, player_stats, h2h, h2h_surface)
            
            # Guardar no session_state
            st.session_state.model = model
            st.session_state.player_stats = player_stats
            st.session_state.h2h = h2h
            st.session_state.h2h_surface = h2h_surface
            st.session_state.df = df

if __name__ == "__main__":
    main()

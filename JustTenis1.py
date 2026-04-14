import warnings
from collections import defaultdict
from datetime import datetime, timedelta
import io
import numpy as np
import pandas as pd
import streamlit as st
import requests
from lightgbm import LGBMClassifier

warnings.filterwarnings('ignore')

st.set_page_config(page_title="🎾 ATP Predictor v2.4 - Fast", page_icon="🎾", layout="wide")

# ==============================================================================
# CONFIGURAÇÕES
# ==============================================================================
MIN_CONFIDENCE_STRONG = 0.78
MIN_CONFIDENCE_GOOD = 0.68
MIN_CONFIDENCE_MEDIUM = 0.60

# ==============================================================================
# SURFACE DETECTION
# ==============================================================================
TOURNAMENT_SURFACE_MAP = {
    'monte carlo': 'Clay', 'madrid': 'Clay', 'rome': 'Clay', 'barcelona': 'Clay',
    'munich': 'Clay', 'estoril': 'Clay', 'geneva': 'Clay', 'oeiras': 'Clay',
    'santa cruz': 'Clay', 'tallahassee': 'Clay', 'busan': 'Hard', 'wuning': 'Hard',
    'quito': 'Clay', 'santiago': 'Clay'
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
# ELO E STATS (simplificado)
# ==============================================================================
def calculate_enhanced_elo(df):
    players = set(df['winner'].dropna().unique()) | set(df['loser'].dropna().unique())
    elo = {p: 1500.0 for p in players}
    surface_elo = {p: {'Hard':1500.0, 'Clay':1500.0, 'Grass':1500.0} for p in players}
    
    df_sorted = df.sort_values('date').copy()
    for _, row in df_sorted.iterrows():
        w, l = row['winner'], row['loser']
        surf = row.get('surface', 'Hard')
        if surf not in ['Hard', 'Clay', 'Grass']: surf = 'Hard'
        if pd.isna(w) or pd.isna(l): continue
            
        r1 = surface_elo[w][surf]
        r2 = surface_elo[l][surf]
        exp = 1 / (1 + 10 ** ((r2 - r1) / 400))
        
        surface_elo[w][surf] += 32 * (1 - exp)
        surface_elo[l][surf] += 32 * (0 - (1 - exp))
        elo[w] += 25 * (1 - exp)
        elo[l] += 25 * (0 - (1 - exp))
    return elo, surface_elo

def compute_player_stats(df):
    elo, surface_elo = calculate_enhanced_elo(df)
    stats = {}
    for player in set(df['winner'].dropna()) | set(df['loser'].dropna()):
        matches = df[(df['winner'] == player) | (df['loser'] == player)].copy()
        if len(matches) == 0: continue
            
        recent = matches.sort_values('date', ascending=False).head(7)
        very_recent = matches.sort_values('date', ascending=False).head(4)
        
        surface_stats = {}
        for surf in ['Hard', 'Clay', 'Grass']:
            m = matches[matches['surface'] == surf]
            surface_stats[surf] = len(m[m['winner'] == player]) / len(m) if len(m) > 0 else 0.5
        
        stats[player] = {
            'elo': float(elo[player]),
            'surface_elo': surface_elo[player],
            'surface_win_rate': surface_stats,
            'very_recent_form': len(very_recent[very_recent['winner'] == player]) / len(very_recent) if len(very_recent)>0 else 0.5,
            'recent_form': len(recent[recent['winner'] == player]) / len(recent) if len(recent)>0 else 0.5,
        }
    return stats

# ==============================================================================
# FEATURES (reduzidas)
# ==============================================================================
def build_features(p1, p2, surface, player_stats, h2h_surface):
    if p1 not in player_stats or p2 not in player_stats: return None
    s1 = player_stats[p1]
    s2 = player_stats[p2]
    surf = surface if surface in ['Hard','Clay','Grass'] else 'Hard'
    
    h2h_surf_ratio = 0.5
    pair = (p1, p2)
    if pair in h2h_surface:
        total = h2h_surface[pair].get(surf, 0) + h2h_surface.get((p2,p1), {}).get(surf, 0) + 1
        h2h_surf_ratio = (h2h_surface[pair].get(surf, 0) + 0.5) / total
    
    return [
        s1['surface_elo'][surf] / (s2['surface_elo'][surf] + 1),
        s1['surface_win_rate'][surf] - s2['surface_win_rate'][surf],
        s1['very_recent_form'] - s2['very_recent_form'],
        abs(s1['elo'] - s2['elo']) / 100,
        h2h_surf_ratio,
    ]

# ==============================================================================
# SCRAPER
# ==============================================================================
def scrape_matches_sofascore(days_ahead=0):
    try:
        target_date = (datetime.utcnow() + timedelta(days=days_ahead)).strftime("%Y-%m-%d")
        url = f"https://api.sofascore.com/api/v1/sport/tennis/scheduled-events/{target_date}"
        headers = {"User-Agent": "Mozilla/5.0"}
        
        r = requests.get(url, headers=headers, timeout=12)
        if r.status_code != 200: return []
        
        data = r.json()
        matches = []
        for ev in data.get("events", []):
            try:
                if "WTA" in str(ev.get("tournament", {}).get("category", {}).get("name", "")).upper(): 
                    continue
                matches.append({
                    "tournament": ev["tournament"]["name"],
                    "player1": ev["homeTeam"]["name"],
                    "player2": ev["awayTeam"]["name"],
                    "surface": detect_surface_from_tournament(ev["tournament"]["name"], ev.get("groundType"))
                })
            except:
                continue
        return matches
    except:
        return []

# ==============================================================================
# TRAINING (LIGHTGBM - RÁPIDO)
# ==============================================================================
def train_models(df, player_stats, h2h_surface):
    X, y = [], []
    for _, row in df.iterrows():
        if pd.isna(row.get('winner')) or pd.isna(row.get('loser')): continue
        surf = row.get('surface', 'Hard')
        feat = build_features(row['winner'], row['loser'], surf, player_stats, h2h_surface)
        if feat:
            X.append(feat)
            y.append(1)
            X.append(build_features(row['loser'], row['winner'], surf, player_stats, h2h_surface))
            y.append(0)
    
    model = LGBMClassifier(
        n_estimators=280,
        max_depth=6,
        learning_rate=0.045,
        num_leaves=32,
        subsample=0.85,
        colsample_bytree=0.85,
        random_state=42,
        verbose=-1
    )
    model.fit(np.array(X), np.array(y))
    return model

# ==============================================================================
# PREDICT
# ==============================================================================
def predict_match(model, player_stats, h2h_surface, match):
    p1 = match['player1']
    p2 = match['player2']
    surface = match['surface']
    
    feat = build_features(p1, p2, surface, player_stats, h2h_surface)
    if feat is None: return None
    
    prob_p1 = model.predict_proba([feat])[0][1]
    prob_p2 = 1 - prob_p1
    winner_pred = p1 if prob_p1 > prob_p2 else p2
    confidence = max(prob_p1, prob_p2)
    
    if confidence >= MIN_CONFIDENCE_STRONG:
        recommendation = f"✅ STRONG BET {winner_pred}"
    elif confidence >= MIN_CONFIDENCE_GOOD:
        recommendation = f"🟢 Bom Valor {winner_pred}"
    elif confidence >= MIN_CONFIDENCE_MEDIUM:
        recommendation = f"🟡 {winner_pred}"
    else:
        recommendation = "⚪ Sem Recomendação"
    
    return {
        'Tournament': match['tournament'],
        'Player1': p1,
        'Player2': p2,
        'Surface': surface,
        'Prob_P1': prob_p1,
        'Prob_P2': prob_p2,
        'Predicted_Winner': winner_pred,
        'Confidence': confidence,
        'Recommendation': recommendation
    }

# ==============================================================================
# MAIN
# ==============================================================================
def main():
    st.title("🎾 ATP Predictor v2.4 - Versão Rápida")
    st.caption("Treino otimizado com LightGBM (muito mais rápido)")

    uploaded_file = st.file_uploader("📁 Upload do teu ficheiro histórico (Excel)", type=['xlsx'])
    
    if uploaded_file and 'model' not in st.session_state:
        with st.spinner("A treinar modelo (versão rápida)..."):
            df = pd.read_excel(uploaded_file)
            df.columns = [str(c).strip().lower().replace(' ', '_').replace('-', '_') for c in df.columns]
            
            if 'tourney_date' in df.columns: df.rename(columns={'tourney_date': 'date'}, inplace=True)
            if 'winner_name' in df.columns: df.rename(columns={'winner_name': 'winner'}, inplace=True)
            if 'loser_name' in df.columns: df.rename(columns={'loser_name': 'loser'}, inplace=True)
            if 'tourney_name' in df.columns: df.rename(columns={'tourney_name': 'tournament'}, inplace=True)
            
            df['date'] = pd.to_datetime(df['date'], errors='coerce')
            df['surface'] = df.apply(lambda row: detect_surface_from_tournament(row.get('tournament'), row.get('surface')), axis=1)
            
            player_stats = compute_player_stats(df)
            h2h_surface = defaultdict(lambda: {'Hard':0, 'Clay':0, 'Grass':0})
            
            for _, row in df.iterrows():
                if pd.notna(row.get('winner')) and pd.notna(row.get('loser')):
                    pair = (row['winner'], row['loser'])
                    surf = row.get('surface', 'Hard')
                    h2h_surface[pair][surf] += 1
            
            model = train_models(df, player_stats, h2h_surface)
            
            st.session_state.model = model
            st.session_state.player_stats = player_stats
            st.session_state.h2h_surface = h2h_surface
            st.success("✅ Modelo rápido treinado com sucesso!")

    col1, col2 = st.columns(2)
    with col1:
        if st.button("📅 HOJE", use_container_width=True):
            st.session_state.current_matches = scrape_matches_sofascore(0)
    with col2:
        if st.button("📅 AMANHÃ", use_container_width=True):
            st.session_state.current_matches = scrape_matches_sofascore(1)

    if st.session_state.get('current_matches'):
        results = [predict_match(st.session_state.model, st.session_state.player_stats, 
                               st.session_state.h2h_surface, m) 
                  for m in st.session_state.current_matches if predict_match(...) is not None]
        
        results = [r for r in results if r is not None]
        
        if results:
            df_show = pd.DataFrame(results)
            styled = df_show.style.format({
                'Prob_P1': '{:.1%}',
                'Prob_P2': '{:.1%}',
                'Confidence': '{:.1%}'
            })
            
            st.subheader("🎯 Previsões")
            st.dataframe(styled, use_container_width=True, hide_index=True, height=650)
            
            buffer = io.BytesIO()
            df_show.to_excel(buffer, index=False)
            st.download_button("📥 Baixar Excel", buffer.getvalue(), 
                             f"previsoes_rapidas_{datetime.now().strftime('%Y-%m-%d')}.xlsx",
                             use_container_width=True)

if __name__ == "__main__":
    main()

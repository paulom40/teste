import warnings
from collections import defaultdict
from datetime import datetime, timedelta
import io
import numpy as np
import pandas as pd
import streamlit as st
import requests
from sklearn.ensemble import VotingClassifier, GradientBoostingClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier

warnings.filterwarnings('ignore')
st.set_page_config(page_title="🎾 ATP Predictor v2.2", page_icon="🎾", layout="wide")

# ==============================================================================
# CONFIG
# ==============================================================================
MIN_EDGE = 0.055   # 5.5% de edge mínimo
KELLY_FRACTION = 0.22

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
# ELO + PLAYER STATS
# ==============================================================================
def calculate_enhanced_elo(df):
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
            
        r1 = surface_elo[w][surf]
        r2 = surface_elo[l][surf]
        exp = 1 / (1 + 10 ** ((r2 - r1) / 400))
        
        surface_elo[w][surf] += 35 * (1 - exp)
        surface_elo[l][surf] += 35 * (0 - (1 - exp))
        elo[w] += 28 * (1 - exp)
        elo[l] += 28 * (0 - (1 - exp))
        
        welo[w] = welo[w] * 0.78 + surface_elo[w][surf] * 0.22
        welo[l] = welo[l] * 0.78 + surface_elo[l][surf] * 0.22
    return elo, welo, surface_elo

def compute_player_stats(df):
    elo, welo, surface_elo = calculate_enhanced_elo(df)
    stats = {}
    for player in set(df['winner'].dropna()) | set(df['loser'].dropna()):
        matches = df[(df['winner'] == player) | (df['loser'] == player)].copy()
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
            'elo': float(elo[player]),
            'welo': float(welo[player]),
            'surface_elo': surface_elo[player],
            'surface_win_rate': surface_stats,
            'surface_match_count': surface_count,
            'very_recent_form': len(very_recent[very_recent['winner'] == player]) / len(very_recent) if len(very_recent)>0 else 0.5,
            'recent_form': len(recent[recent['winner'] == player]) / len(recent) if len(recent)>0 else 0.5,
        }
    return stats

# ==============================================================================
# FEATURES
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
        s1['welo'] / (s2['welo'] + 1),
        s1['surface_win_rate'][surf] - s2['surface_win_rate'][surf],
        s1['very_recent_form'] - s2['very_recent_form'],
        s1['recent_form'] - s2['recent_form'],
        abs(s1['elo'] - s2['elo']) / 100,
        h2h_surf_ratio,
    ]

# ==============================================================================
# SOFASCORE SCRAPER
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
                if "WTA" in ev["tournament"]["category"]["name"].upper(): continue
                p1 = ev["homeTeam"]["name"]
                p2 = ev["awayTeam"]["name"]
                tournament = ev["tournament"]["name"]
                surface_hint = ev.get("groundType", "")
                
                surface = detect_surface_from_tournament(tournament, surface_hint)
                
                matches.append({
                    "tournament": tournament,
                    "player1": p1,
                    "player2": p2,
                    "surface": surface,
                    "date": target_date
                })
            except:
                continue
        return matches
    except:
        return []

# ==============================================================================
# TRAINING
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
    
    X, y = np.array(X), np.array(y)
    
    ensemble = VotingClassifier([
        ('gb', GradientBoostingClassifier(n_estimators=320, max_depth=5, learning_rate=0.035, random_state=42)),
        ('xgb', XGBClassifier(n_estimators=380, max_depth=5, learning_rate=0.028, random_state=42)),
        ('lgb', LGBMClassifier(n_estimators=380, max_depth=5, learning_rate=0.03, verbose=-1, random_state=42))
    ], voting='soft')
    
    ensemble.fit(X, y)
    return ensemble

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
    
    # Value Betting
    recommendation = "Sem Value"
    edge = 0
    kelly = 0
    
    # Aqui podes adicionar odds manualmente depois se quiseres
    
    return {
        'tournament': match['tournament'],
        'player1': p1,
        'player2': p2,
        'surface': surface,
        'prob_p1': prob_p1,
        'prob_p2': prob_p2,
        'winner_pred': p1 if prob_p1 > prob_p2 else p2,
        'recommendation': recommendation,
        'edge': edge,
        'confidence': max(prob_p1, prob_p2)
    }

# ==============================================================================
# STREAMLIT APP
# ==============================================================================
def main():
    st.title("🎾 ATP & Challenger Predictor v2.2")
    st.markdown("**Ensemble + Surface ELO + Value Betting**")

    uploaded_file = st.file_uploader("Upload Historical Data (Excel)", type=['xlsx'])
    
    if uploaded_file and 'model' not in st.session_state:
        with st.spinner("Treinando modelo..."):
            df = pd.read_excel(uploaded_file)
            df.columns = [str(c).strip().lower().replace(' ', '_').replace('-','_') for c in df.columns]
            
            # Column mapping
            if 'tourney_date' in df.columns: df = df.rename(columns={'tourney_date': 'date'})
            if 'winner_name' in df.columns: df = df.rename(columns={'winner_name': 'winner'})
            if 'loser_name' in df.columns: df = df.rename(columns={'loser_name': 'loser'})
            if 'tourney_name' in df.columns: df = df.rename(columns={'tourney_name': 'tournament'})
            
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
            st.success("✅ Modelo treinado com sucesso!")

    col1, col2 = st.columns(2)
    with col1:
        if st.button("📅 Previsões de HOJE", use_container_width=True):
            matches = scrape_matches_sofascore(0)
            st.session_state.current_matches = matches
    with col2:
        if st.button("📅 Previsões de AMANHÃ", use_container_width=True):
            matches = scrape_matches_sofascore(1)
            st.session_state.current_matches = matches

    if st.session_state.get('current_matches'):
        results = []
        for m in st.session_state.current_matches:
            pred = predict_match(st.session_state.model, st.session_state.player_stats, 
                               st.session_state.h2h_surface, m)
            if pred:
                results.append(pred)
        
        df_show = pd.DataFrame(results)
        df_show = df_show.round(4)
        st.dataframe(df_show, use_container_width=True)
        
        # Export
        buffer = io.BytesIO()
        df_show.to_excel(buffer, index=False)
        st.download_button("📥 Download Excel", buffer.getvalue(), 
                          f"previsoes_{datetime.now().strftime('%Y-%m-%d')}.xlsx",
                          mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")

if __name__ == "__main__":
    main()

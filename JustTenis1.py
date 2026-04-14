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

st.set_page_config(page_title="🎾 ATP Predictor v2.9 Strong Calibration", page_icon="🎾", layout="wide")

# ==============================================================================
# CONFIG - CALIBRAÇÃO FORTE
# ==============================================================================
WINNER_SMOOTH = 0.52     # Muito mais conservador
OU_SMOOTH = 0.58
MIN_CONFIDENCE_STRONG = 0.69
MIN_CONFIDENCE_GOOD = 0.62

# ==============================================================================
# SURFACE DETECTION
# ==============================================================================
TOURNAMENT_SURFACE_MAP = {
    'monte carlo': 'Clay', 'madrid': 'Clay', 'rome': 'Clay', 'barcelona': 'Clay',
    'munich': 'Clay', 'estoril': 'Clay', 'geneva': 'Clay', 'oeiras': 'Clay',
    'santa cruz': 'Clay', 'tallahassee': 'Clay', 'busan': 'Hard', 'wuning': 'Hard'
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
# ELO + STATS
# ==============================================================================
def calculate_recent_elo(df, recent_matches=20):
    players = set(df['winner'].dropna().unique()) | set(df['loser'].dropna().unique())
    surface_elo = {p: {'Hard':1500.0, 'Clay':1500.0, 'Grass':1500.0} for p in players}
    recent_form = {p: [] for p in players}
    
    for _, row in df.sort_values('date').iterrows():
        w, l = row['winner'], row['loser']
        surf = row.get('surface', 'Hard')
        if surf not in ['Hard', 'Clay', 'Grass']: surf = 'Hard'
        if pd.isna(w) or pd.isna(l): continue
        
        recent_form[w].append(1)
        recent_form[l].append(0)
        if len(recent_form[w]) > recent_matches: recent_form[w].pop(0)
        if len(recent_form[l]) > recent_matches: recent_form[l].pop(0)
        
        w_recent = sum(recent_form[w]) / len(recent_form[w]) if recent_form[w] else 0.5
        l_recent = sum(recent_form[l]) / len(recent_form[l]) if recent_form[l] else 0.5
        
        r1 = surface_elo[w][surf] + 55 * (w_recent - 0.5)
        r2 = surface_elo[l][surf] + 55 * (l_recent - 0.5)
        
        exp = 1 / (1 + 10 ** ((r2 - r1) / 400))
        
        surface_elo[w][surf] += 30 * (1 - exp)
        surface_elo[l][surf] += 30 * (0 - (1 - exp))
    
    return surface_elo, recent_form

def compute_player_stats(df, recent_matches=20):
    surface_elo, recent_form = calculate_recent_elo(df, recent_matches)
    stats = {}
    for player in set(df['winner'].dropna()) | set(df['loser'].dropna()):
        matches = df[(df['winner'] == player) | (df['loser'] == player)].copy()
        if len(matches) == 0:
            stats[player] = {'surface_elo': {'Hard':1500,'Clay':1500,'Grass':1500},
                           'surface_win_rate': {'Hard':0.5,'Clay':0.5,'Grass':0.5},
                           'very_recent_form': 0.5, 'recent_20_form': 0.5, 'avg_games': 22}
            continue
            
        recent = matches.sort_values('date', ascending=False).head(recent_matches)
        very_recent = matches.sort_values('date', ascending=False).head(5)
        
        surface_stats = {}
        for surf in ['Hard', 'Clay', 'Grass']:
            m = matches[matches['surface'] == surf]
            surface_stats[surf] = len(m[m['winner'] == player]) / len(m) if len(m) > 0 else 0.5
        
        stats[player] = {
            'surface_elo': surface_elo[player],
            'surface_win_rate': surface_stats,
            'very_recent_form': len(very_recent[very_recent['winner'] == player]) / len(very_recent) if len(very_recent)>0 else 0.5,
            'recent_20_form': len(recent[recent['winner'] == player]) / len(recent) if len(recent)>0 else 0.5,
            'avg_games': float(matches.get('total_games', pd.Series([22])).mean())
        }
    return stats

# ==============================================================================
# FEATURES + BUILD
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
        s1.get('recent_20_form', 0.5) - s2.get('recent_20_form', 0.5),
        s1.get('very_recent_form', 0.5) - s2.get('very_recent_form', 0.5),
        abs(s1['surface_elo'][surf] - s2['surface_elo'][surf]) / 180,
        h2h_surf_ratio,
        (s1.get('avg_games', 22) + s2.get('avg_games', 22)) / 2
    ]

# ==============================================================================
# SCRAPER, TRAINING, PREDICT
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
                if "WTA" in str(ev.get("tournament", {}).get("category", {}).get("name", "")).upper(): continue
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

def train_models(df, player_stats, h2h_surface):
    X, y_winner, y_ou = [], [], []
    for _, row in df.iterrows():
        if pd.isna(row.get('winner')) or pd.isna(row.get('loser')): continue
        surf = row.get('surface', 'Hard')
        total_games = row.get('total_games', 22)
        feat = build_features(row['winner'], row['loser'], surf, player_stats, h2h_surface)
        if feat:
            X.append(feat)
            y_winner.append(1)
            y_ou.append(1 if total_games > 21.5 else 0)
            X.append(build_features(row['loser'], row['winner'], surf, player_stats, h2h_surface))
            y_winner.append(0)
            y_ou.append(1 if total_games > 21.5 else 0)
    
    X = np.array(X)
    
    model_winner = LGBMClassifier(n_estimators=180, max_depth=4, learning_rate=0.035,
                                  num_leaves=16, reg_alpha=3.0, reg_lambda=3.0, random_state=42, verbose=-1)
    model_ou = LGBMClassifier(n_estimators=150, max_depth=4, learning_rate=0.04,
                              num_leaves=16, reg_alpha=2.5, reg_lambda=2.5, random_state=42, verbose=-1)
    
    model_winner.fit(X, y_winner)
    model_ou.fit(X, y_ou)
    return model_winner, model_ou

def predict_match(model_winner, model_ou, player_stats, h2h_surface, match):
    p1 = match['player1']
    p2 = match['player2']
    surface = match['surface']
    
    feat = build_features(p1, p2, surface, player_stats, h2h_surface)
    if feat is None: return None
    
    raw_p = model_winner.predict_proba([feat])[0][1]
    prob_p1 = 0.5 + (raw_p - 0.5) * WINNER_SMOOTH
    prob_p1 = max(0.08, min(0.92, prob_p1))   # Limites mais conservadores
    prob_p2 = 1 - prob_p1
    
    raw_ou = model_ou.predict_proba([feat])[0][1]
    ou_prob = 0.5 + (raw_ou - 0.5) * OU_SMOOTH
    ou_prob = max(0.20, min(0.80, ou_prob))
    
    winner_pred = p1 if prob_p1 > prob_p2 else p2
    confidence = max(prob_p1, prob_p2)
    
    if confidence >= MIN_CONFIDENCE_STRONG:
        rec = f"✅ STRONG {winner_pred}"
    elif confidence >= MIN_CONFIDENCE_GOOD:
        rec = f"🟢 {winner_pred}"
    else:
        rec = f"🟡 {winner_pred}"
    
    return {
        'Tournament': match['tournament'],
        'Player1': p1,
        'Player2': p2,
        'Surface': surface,
        'Prob_P1': prob_p1,
        'Prob_P2': prob_p2,
        'Predicted_Winner': winner_pred,
        'Confidence': confidence,
        'Recommendation': rec,
        'OU': "Over 21.5" if ou_prob > 0.5 else "Under 21.5",
        'OU_Prob': ou_prob
    }

# ==============================================================================
# MAIN APP (mesmo da versão anterior)
# ==============================================================================
def main():
    st.title("🎾 ATP Predictor v2.9 - Calibração Forte")
    st.caption("Últimos 20 jogos + Probabilidades mais realistas")

    uploaded_file = st.file_uploader("📁 Upload do teu ficheiro histórico (Excel)", type=['xlsx'])
    
    if uploaded_file and 'model_winner' not in st.session_state:
        with st.spinner("A treinar..."):
            df = pd.read_excel(uploaded_file)
            df.columns = [str(c).strip().lower().replace(' ', '_').replace('-', '_') for c in df.columns]
            
            if 'tourney_date' in df.columns: df.rename(columns={'tourney_date': 'date'}, inplace=True)
            if 'winner_name' in df.columns: df.rename(columns={'winner_name': 'winner'}, inplace=True)
            if 'loser_name' in df.columns: df.rename(columns={'loser_name': 'loser'}, inplace=True)
            if 'tourney_name' in df.columns: df.rename(columns={'tourney_name': 'tournament'}, inplace=True)
            
            df['date'] = pd.to_datetime(df['date'], errors='coerce')
            
            if 'total_games' not in df.columns and 'score' in df.columns:
                def get_games(s):
                    nums = [int(n) for n in str(s).replace(' ', '').split('-') if n.isdigit()]
                    return sum(nums) if nums else 22
                df['total_games'] = df['score'].apply(get_games)
            elif 'total_games' not in df.columns:
                df['total_games'] = 22
            
            max_rows = st.slider("Máximo de jogos para treino", 5000, len(df), min(11000, len(df)), 1000)
            if len(df) > max_rows:
                df = df.sort_values('date', ascending=False).head(max_rows).copy()
            
            df['surface'] = df.apply(lambda row: detect_surface_from_tournament(row.get('tournament'), row.get('surface')), axis=1)
            
            player_stats = compute_player_stats(df, RECENT_MATCHES)
            h2h_surface = defaultdict(lambda: {'Hard':0, 'Clay':0, 'Grass':0})
            for _, row in df.iterrows():
                if pd.notna(row.get('winner')) and pd.notna(row.get('loser')):
                    pair = (row['winner'], row['loser'])
                    h2h_surface[pair][row.get('surface', 'Hard')] += 1
            
            model_winner, model_ou = train_models(df, player_stats, h2h_surface)
            
            st.session_state.model_winner = model_winner
            st.session_state.model_ou = model_ou
            st.session_state.player_stats = player_stats
            st.session_state.h2h_surface = h2h_surface
            st.success("✅ Modelo treinado!")

    col1, col2 = st.columns(2)
    with col1:
        if st.button("📅 HOJE", use_container_width=True):
            st.session_state.current_matches = scrape_matches_sofascore(0)
    with col2:
        if st.button("📅 AMANHÃ", use_container_width=True):
            st.session_state.current_matches = scrape_matches_sofascore(1)

    if st.session_state.get('current_matches'):
        results = [predict_match(st.session_state.model_winner, st.session_state.model_ou,
                                st.session_state.player_stats, st.session_state.h2h_surface, m) 
                  for m in st.session_state.current_matches]
        results = [r for r in results if r is not None]
        
        if results:
            df_show = pd.DataFrame(results)
            styled = df_show.style.format({
                'Prob_P1': '{:.1%}',
                'Prob_P2': '{:.1%}',
                'Confidence': '{:.1%}',
                'OU_Prob': '{:.1%}'
            })
            st.subheader("🎯 Previsões")
            st.dataframe(styled, use_container_width=True, hide_index=True, height=700)
            
            buffer = io.BytesIO()
            df_show.to_excel(buffer, index=False)
            st.download_button("📥 Baixar Excel", buffer.getvalue(), 
                             f"previsoes_{datetime.now().strftime('%Y-%m-%d')}.xlsx",
                             use_container_width=True)

if __name__ == "__main__":
    main()

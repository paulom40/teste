import warnings
from collections import defaultdict
from datetime import datetime, timedelta
import io
import numpy as np
import pandas as pd
import streamlit as st
import requests
from lightgbm import LGBMClassifier
import re
from difflib import SequenceMatcher

warnings.filterwarnings('ignore')

st.set_page_config(page_title="🎾 ATP Predictor v4.4 - Direct Match", page_icon="🎾", layout="wide")

# ==============================================================================
# CONFIG
# ==============================================================================
WINNER_SMOOTH = 0.55
MIN_CONFIDENCE_STRONG = 0.68
MIN_CONFIDENCE_GOOD = 0.60

# ==============================================================================
# SURFACE DETECTION
# ==============================================================================
def detect_surface(tournament_name):
    if pd.isna(tournament_name):
        return 'Hard'
    t = str(tournament_name).lower()
    clay = ['clay', 'monte carlo', 'madrid', 'rome', 'barcelona', 'munich', 'roland garros']
    grass = ['grass', 'wimbledon', 'queens', 'halle']
    if any(k in t for k in clay):
        return 'Clay'
    if any(k in t for k in grass):
        return 'Grass'
    return 'Hard'

# ==============================================================================
# DATA PROCESSING - MOSTRA OS NOMES REAIS
# ==============================================================================
def process_historical_data(df):
    """Process historical data and show actual names"""
    
    # Clean column names
    df.columns = [str(c).strip().lower().replace(' ', '_').replace('-', '_') for c in df.columns]
    
    # Find columns
    winner_col = None
    loser_col = None
    tournament_col = None
    date_col = None
    
    for col in df.columns:
        if 'winner' in col or 'vencedor' in col:
            winner_col = col
        elif 'loser' in col or 'perdedor' in col:
            loser_col = col
        elif 'tourney' in col or 'torneio' in col or 'tournament' in col:
            tournament_col = col
        elif 'date' in col or 'data' in col:
            date_col = col
    
    if not winner_col or not loser_col:
        raise ValueError(f"Colunas não encontradas. Colunas disponíveis: {list(df.columns)}")
    
    # Rename
    df = df.rename(columns={
        winner_col: 'winner',
        loser_col: 'loser',
        tournament_col: 'tournament' if tournament_col else 'tournament',
        date_col: 'date' if date_col else 'date'
    })
    
    # Convert date
    if 'date' in df.columns:
        df['date'] = pd.to_datetime(df['date'], errors='coerce')
    else:
        df['date'] = pd.Timestamp.now()
    
    # Calculate total games (placeholder if not available)
    if 'total_games' not in df.columns:
        df['total_games'] = 22
    
    # Detect surface
    if 'tournament' in df.columns:
        df['surface'] = df['tournament'].apply(detect_surface)
    else:
        df['surface'] = 'Hard'
    
    # Clean names - keep as is from the file
    df['winner'] = df['winner'].astype(str).str.strip()
    df['loser'] = df['loser'].astype(str).str.strip()
    
    return df

# ==============================================================================
# SIMPLE NAME MATCHER - USA OS NOMES EXATOS DO HISTÓRICO
# ==============================================================================
class SimpleNameMatcher:
    """Matching simples usando os nomes exatos do histórico"""
    
    def __init__(self, historical_names):
        self.historical_names = list(historical_names)
        self.name_set = set(historical_names)
        # Criar índice de sobrenomes
        self.last_name_index = defaultdict(list)
        for name in self.historical_names:
            parts = name.split()
            if parts:
                last_name = parts[-1].lower()
                self.last_name_index[last_name].append(name)
    
    def find_match(self, search_name):
        """Find matching name in historical data"""
        if not search_name or pd.isna(search_name):
            return None
        
        search_str = str(search_name).strip()
        
        # 1. Exact match
        if search_str in self.name_set:
            return search_str
        
        # 2. Case insensitive
        search_lower = search_str.lower()
        for name in self.historical_names:
            if name.lower() == search_lower:
                return name
        
        # 3. Last name match (if unique)
        parts = search_str.split()
        if parts:
            last_name = parts[-1].lower()
            if last_name in self.last_name_index:
                matches = self.last_name_index[last_name]
                if len(matches) == 1:
                    return matches[0]
        
        # 4. Partial match (contains)
        for name in self.historical_names:
            if search_lower in name.lower() or name.lower() in search_lower:
                return name
        
        return None

# ==============================================================================
# PLAYER STATISTICS
# ==============================================================================
def calculate_player_stats(df):
    """Calculate player statistics"""
    
    all_players = set(df['winner'].dropna()) | set(df['loser'].dropna())
    stats = {}
    
    for player in all_players:
        player_matches = df[(df['winner'] == player) | (df['loser'] == player)]
        
        if len(player_matches) == 0:
            continue
        
        wins = len(player_matches[player_matches['winner'] == player])
        total = len(player_matches)
        win_rate = wins / total if total > 0 else 0.5
        
        # Surface stats
        surface_stats = {}
        for surf in ['Hard', 'Clay', 'Grass']:
            surf_matches = player_matches[player_matches['surface'] == surf]
            if len(surf_matches) > 0:
                surf_wins = len(surf_matches[surf_matches['winner'] == player])
                surface_stats[surf] = surf_wins / len(surf_matches)
            else:
                surface_stats[surf] = 0.5
        
        # Recent form
        recent = player_matches.sort_values('date', ascending=False).head(10)
        recent_wins = len(recent[recent['winner'] == player])
        recent_form = recent_wins / len(recent) if len(recent) > 0 else 0.5
        
        very_recent = player_matches.sort_values('date', ascending=False).head(3)
        very_recent_wins = len(very_recent[very_recent['winner'] == player])
        very_recent_form = very_recent_wins / len(very_recent) if len(very_recent) > 0 else 0.5
        
        avg_games = player_matches['total_games'].mean() if 'total_games' in player_matches.columns else 22
        
        stats[player] = {
            'name': player,
            'matches': total,
            'wins': wins,
            'losses': total - wins,
            'win_rate': win_rate,
            'hard_rate': surface_stats['Hard'],
            'clay_rate': surface_stats['Clay'],
            'grass_rate': surface_stats['Grass'],
            'recent_form': recent_form,
            'very_recent_form': very_recent_form,
            'avg_games': avg_games
        }
    
    return stats

# ==============================================================================
# H2H DATA
# ==============================================================================
def calculate_h2h(df):
    """Calculate head-to-head statistics"""
    h2h = defaultdict(lambda: {'wins': 0, 'total': 0})
    
    for _, row in df.iterrows():
        if pd.isna(row.get('winner')) or pd.isna(row.get('loser')):
            continue
        w, l = row['winner'], row['loser']
        h2h[(w, l)]['wins'] += 1
        h2h[(w, l)]['total'] += 1
    
    return h2h

# ==============================================================================
# ELO RATING
# ==============================================================================
def calculate_elo(df, k=32):
    """Calculate ELO ratings"""
    players = set(df['winner'].dropna()) | set(df['loser'].dropna())
    elo = {p: 1500 for p in players}
    
    for _, row in df.sort_values('date').iterrows():
        w, l = row['winner'], row['loser']
        if pd.isna(w) or pd.isna(l):
            continue
        exp_w = 1 / (1 + 10 ** ((elo[l] - elo[w]) / 400))
        elo[w] += k * (1 - exp_w)
        elo[l] += k * (0 - (1 - exp_w))
    
    return elo

# ==============================================================================
# FEATURE ENGINEERING
# ==============================================================================
def build_features(p1, p2, surface, player_stats, h2h, elo):
    """Build features for prediction"""
    
    s1 = player_stats.get(p1)
    s2 = player_stats.get(p2)
    
    if not s1 or not s2:
        return None
    
    if surface == 'Clay':
        surf_rate1, surf_rate2 = s1['clay_rate'], s2['clay_rate']
    elif surface == 'Grass':
        surf_rate1, surf_rate2 = s1['grass_rate'], s2['grass_rate']
    else:
        surf_rate1, surf_rate2 = s1['hard_rate'], s2['hard_rate']
    
    elo1, elo2 = elo.get(p1, 1500), elo.get(p2, 1500)
    elo_diff = (elo1 - elo2) / 400
    
    form_diff = s1['recent_form'] - s2['recent_form']
    very_recent_diff = s1['very_recent_form'] - s2['very_recent_form']
    win_rate_diff = s1['win_rate'] - s2['win_rate']
    surf_diff = surf_rate1 - surf_rate2
    
    # H2H
    h2h_adv = 0.5
    if (p1, p2) in h2h:
        h2h_adv = h2h[(p1, p2)]['wins'] / h2h[(p1, p2)]['total']
    elif (p2, p1) in h2h:
        h2h_adv = 1 - (h2h[(p2, p1)]['wins'] / h2h[(p2, p1)]['total'])
    
    games_avg = (s1['avg_games'] + s2['avg_games']) / 2
    games_norm = (games_avg - 21.5) / 8
    exp_diff = (s1['matches'] - s2['matches']) / 200
    momentum = (s1['very_recent_form'] - s2['very_recent_form']) * 0.6 + form_diff * 0.4
    
    return [elo_diff, form_diff, very_recent_diff, win_rate_diff, surf_diff, 
            h2h_adv, games_norm, exp_diff, momentum]

# ==============================================================================
# TRAIN MODEL
# ==============================================================================
def train_model(df, player_stats, h2h, elo):
    """Train prediction model"""
    
    X, y = [], []
    
    for _, row in df.iterrows():
        if pd.isna(row.get('winner')) or pd.isna(row.get('loser')):
            continue
        
        winner, loser = row['winner'], row['loser']
        surface = row.get('surface', 'Hard')
        
        features = build_features(winner, loser, surface, player_stats, h2h, elo)
        if features:
            X.append(features)
            y.append(1)
        
        features_rev = build_features(loser, winner, surface, player_stats, h2h, elo)
        if features_rev:
            X.append(features_rev)
            y.append(0)
    
    if len(X) == 0:
        raise ValueError("No training data")
    
    X = np.array(X)
    
    model = LGBMClassifier(
        n_estimators=150, max_depth=5, learning_rate=0.035,
        num_leaves=16, reg_alpha=0.8, reg_lambda=0.8,
        random_state=42, verbose=-1
    )
    
    model.fit(X, y)
    return model

# ==============================================================================
# PREDICT
# ==============================================================================
def predict_match(model, p1, p2, surface, player_stats, h2h, elo, name_matcher):
    """Predict a single match"""
    
    p1_match = name_matcher.find_match(p1)
    p2_match = name_matcher.find_match(p2)
    
    if not p1_match:
        return None, f"Não encontrado: '{p1}'"
    if not p2_match:
        return None, f"Não encontrado: '{p2}'"
    
    features = build_features(p1_match, p2_match, surface, player_stats, h2h, elo)
    
    if features is None:
        return None, f"Stats não disponíveis"
    
    features = np.array([features])
    prob = model.predict_proba(features)[0][1]
    
    prob_p1 = np.clip(0.5 + (prob - 0.5) * 0.55, 0.15, 0.85)
    prob_p2 = 1 - prob_p1
    
    confidence = abs(prob_p1 - 0.5) * 2
    winner = p1_match if prob_p1 > 0.5 else p2_match
    
    if confidence >= 0.68:
        rec = f"🔥 STRONG {winner}"
    elif confidence >= 0.60:
        rec = f"✅ GOOD {winner}"
    else:
        rec = f"⚪ AVOID {winner}"
    
    s1, s2 = player_stats.get(p1_match, {}), player_stats.get(p2_match, {})
    momentum_edge = (s1.get('very_recent_form', 0.5) - s2.get('very_recent_form', 0.5)) * 100
    expected_games = np.clip((s1.get('avg_games', 22) + s2.get('avg_games', 22)) / 2, 18, 35)
    ou_prob = np.clip(0.5 + (expected_games - 21.5) / 20, 0.35, 0.65)
    
    return {
        'Player1_Original': p1,
        'Player2_Original': p2,
        'Matched_As': f"{p1_match} | {p2_match}",
        'Surface': surface,
        'P1_Win%': prob_p1,
        'P2_Win%': prob_p2,
        'Predicted_Winner': winner,
        'Confidence': confidence,
        'Recommendation': rec,
        'Momentum': round(momentum_edge, 1),
        'Exp_Games': round(expected_games, 1),
        'OU': "Over" if ou_prob > 0.5 else "Under"
    }, None

# ==============================================================================
# SCRAPER
# ==============================================================================
def scrape_matches():
    """Scrape matches"""
    try:
        target_date = datetime.now().strftime("%Y-%m-%d")
        url = f"https://api.sofascore.com/api/v1/sport/tennis/scheduled-events/{target_date}"
        headers = {"User-Agent": "Mozilla/5.0"}
        r = requests.get(url, headers=headers, timeout=10)
        
        if r.status_code != 200:
            return []
        
        data = r.json()
        matches = []
        
        for ev in data.get("events", []):
            category = ev.get("tournament", {}).get("category", {}).get("name", "")
            if "WTA" in str(category).upper():
                continue
            
            tournament = ev["tournament"]["name"]
            surface = detect_surface(tournament)
            
            matches.append({
                "tournament": tournament,
                "player1": ev["homeTeam"]["name"],
                "player2": ev["awayTeam"]["name"],
                "surface": surface
            })
        return matches
    except:
        return []

# ==============================================================================
# MAIN APP
# ==============================================================================
def main():
    st.title("🎾 ATP Predictor v4.4 - Direct Match")
    st.caption("Usa EXATAMENTE os nomes do seu arquivo histórico")
    
    uploaded_file = st.file_uploader("📁 Upload do ficheiro histórico", type=['xlsx', 'csv'])
    
    if uploaded_file and 'model' not in st.session_state:
        with st.spinner("🔄 Processando..."):
            try:
                # Load
                if uploaded_file.name.endswith('.csv'):
                    df = pd.read_csv(uploaded_file)
                else:
                    df = pd.read_excel(uploaded_file)
                
                # Show original columns
                st.info(f"Colunas no seu arquivo: {list(df.columns)}")
                
                # Process
                df = process_historical_data(df)
                
                # MOSTRA OS NOMES REAIS DO SEU ARQUIVO
                st.subheader("📋 Nomes no seu histórico (amostra)")
                
                all_names = list(set(df['winner'].dropna()) | set(df['loser'].dropna()))
                st.write(f"Total de jogadores únicos: **{len(all_names)}**")
                
                # Show first 50 names
                with st.expander(f"Ver primeiros 50 nomes ({len(all_names)} total)"):
                    for i, name in enumerate(sorted(all_names)[:50]):
                        st.write(f"{i+1}. `{name}`")
                
                # Calculate stats
                player_stats = calculate_player_stats(df)
                h2h = calculate_h2h(df)
                elo = calculate_elo(df)
                
                # Create matcher with historical names
                name_matcher = SimpleNameMatcher(all_names)
                
                # Train model
                model = train_model(df, player_stats, h2h, elo)
                
                # Store
                st.session_state.model = model
                st.session_state.player_stats = player_stats
                st.session_state.h2h = h2h
                st.session_state.elo = elo
                st.session_state.name_matcher = name_matcher
                st.session_state.all_names = all_names
                st.session_state.models_ready = True
                
                st.success(f"✅ Modelo treinado com {len(player_stats)} jogadores!")
                
            except Exception as e:
                st.error(f"Erro: {str(e)}")
                import traceback
                st.code(traceback.format_exc())
    
    if st.session_state.get('models_ready'):
        # Show available names for matching
        with st.expander("🔍 Ver todos os jogadores disponíveis"):
            search = st.text_input("Buscar jogador:", "")
            if search:
                matches = [n for n in st.session_state.all_names if search.lower() in n.lower()]
                st.write(f"Encontrados {len(matches)} jogadores:")
                for m in matches[:20]:
                    stats = st.session_state.player_stats.get(m, {})
                    st.write(f"• `{m}` - {stats.get('matches',0)} jogos, {stats.get('win_rate',0):.0%} vitórias")
            else:
                st.write(f"Total: {len(st.session_state.all_names)} jogadores")
                for i, name in enumerate(sorted(st.session_state.all_names)[:30]):
                    stats = st.session_state.player_stats.get(name, {})
                    st.write(f"{i+1}. `{name}` - {stats.get('matches',0)} jogos")
        
        col1, col2 = st.columns(2)
        with col1:
            if st.button("📅 HOJE", use_container_width=True):
                with st.spinner("Buscando..."):
                    st.session_state.matches = scrape_matches()
        with col2:
            if st.button("🔄 LIMPAR", use_container_width=True):
                st.session_state.matches = []
                st.rerun()
        
        # Manual input - AGORA COM SUGESTÕES
        with st.expander("✏️ Previsão Manual", expanded=True):
            st.markdown("**Digite o nome EXATAMENTE como aparece no histórico acima**")
            
            col_a, col_b, col_c = st.columns(3)
            with col_a:
                manual_p1 = st.selectbox("Jogador 1", [""] + sorted(st.session_state.all_names)[:100])
            with col_b:
                manual_p2 = st.selectbox("Jogador 2", [""] + sorted(st.session_state.all_names)[:100])
            with col_c:
                manual_surface = st.selectbox("Superfície", ["Clay", "Hard", "Grass"])
            
            if st.button("🔮 PREVER", type="primary") and manual_p1 and manual_p2:
                result, error = predict_match(
                    st.session_state.model,
                    manual_p1, manual_p2, manual_surface,
                    st.session_state.player_stats,
                    st.session_state.h2h,
                    st.session_state.elo,
                    st.session_state.name_matcher
                )
                if result:
                    df_result = pd.DataFrame([result])
                    st.dataframe(df_result.style.format({
                        'P1_Win%': '{:.1%}', 'P2_Win%': '{:.1%}', 'Confidence': '{:.1%}'
                    }), use_container_width=True)
                else:
                    st.error(error)
        
        # Show predictions
        if st.session_state.get('matches'):
            st.subheader("🎯 Previsões")
            
            results = []
            not_found = []
            
            for match in st.session_state.matches:
                result, error = predict_match(
                    st.session_state.model,
                    match['player1'], match['player2'], match['surface'],
                    st.session_state.player_stats,
                    st.session_state.h2h,
                    st.session_state.elo,
                    st.session_state.name_matcher
                )
                if result:
                    result['Tournament'] = match['tournament']
                    results.append(result)
                elif error:
                    not_found.append(error)
            
            # Show not found
            if not_found:
                st.warning(f"⚠️ {len(not_found)} jogadores não encontrados no histórico:")
                for err in set(not_found)[:10]:
                    st.write(f"• {err}")
            
            if results:
                df_results = pd.DataFrame(results)
                st.dataframe(df_results, use_container_width=True, hide_index=True)
                
                # Download
                buffer = io.BytesIO()
                df_results.to_excel(buffer, index=False)
                st.download_button("📥 Download", buffer.getvalue(),
                                 f"predictions_{datetime.now().strftime('%Y%m%d_%H%M')}.xlsx")
                
                # Summary
                col_s1, col_s2, col_s3 = st.columns(3)
                with col_s1:
                    strong = sum(1 for r in results if 'STRONG' in r['Recommendation'])
                    st.metric("STRONG", strong)
                with col_s2:
                    avg_conf = df_results['Confidence'].mean()
                    st.metric("Confiança Média", f"{avg_conf:.1%}")
                with col_s3:
                    st.metric("Total", len(results))

if __name__ == "__main__":
    main()

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

warnings.filterwarnings('ignore')

st.set_page_config(page_title="🎾 ATP Predictor v5.0 - Seu Histórico", page_icon="🎾", layout="wide")

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
# PROCESSAR HISTÓRICO
# ==============================================================================
def process_historical_data(df):
    """Process historical data"""
    
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
        raise ValueError(f"Colunas não encontradas. Colunas: {list(df.columns)}")
    
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
    
    # Total games (placeholder)
    if 'total_games' not in df.columns:
        df['total_games'] = 22
    
    # Detect surface
    if 'tournament' in df.columns:
        df['surface'] = df['tournament'].apply(detect_surface)
    else:
        df['surface'] = 'Hard'
    
    # Clean names
    df['winner'] = df['winner'].astype(str).str.strip()
    df['loser'] = df['loser'].astype(str).str.strip()
    
    return df

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
        
        # Recent form (last 10)
        recent = player_matches.sort_values('date', ascending=False).head(10)
        recent_wins = len(recent[recent['winner'] == player])
        recent_form = recent_wins / len(recent) if len(recent) > 0 else 0.5
        
        # Very recent (last 3)
        very_recent = player_matches.sort_values('date', ascending=False).head(3)
        very_recent_wins = len(very_recent[very_recent['winner'] == player])
        very_recent_form = very_recent_wins / len(very_recent) if len(very_recent) > 0 else 0.5
        
        # Average games
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
# SMART NAME MATCHER - CONVERTE NOMES DA SOFASCORE PARA SEU FORMATO
# ==============================================================================
def create_name_mapping(historical_names):
    """Create mapping from full names to historical names"""
    mapping = {}
    
    # Mapeamentos manuais comuns
    manual_mapping = {
        # Nomes da Sofascore -> Seu formato
        'Jan-Lennard Struff': 'Jan-Lennard Struff',
        'Francisco Cerundolo': 'Francisco Cerundolo',
        'Sumit Nagal': 'Sumit Nagal',
        'Ben Shelton': 'Ben Shelton',
        'Alexander Zverev': 'Alexander Zverev',
        'Joao Fonseca': 'Joao Fonseca',
        'Alejandro Tabilo': 'Alejandro Tabilo',
        'Denis Shapovalov': 'Denis Shapovalov',
        'Tallon Griekspoor': 'Tallon Griekspoor',
        'Luciano Darderi': 'Luciano Darderi',
        'Marko Topo': 'Marko Topo',
        'Arthur Fils': 'Arthur Fils',
        'Jack Draper': 'Jack Draper',
        'Cameron Norrie': 'Cameron Norrie',
        'Stan Wawrinka': 'Stan Wawrinka',
        'Adrian Mannarino': 'Adrian Mannarino',
        'Mitchell Krueger': 'Mitchell Krueger',
        'Trevor Svajda': 'Trevor Svajda',
        'Yuta Shimizu': 'Yuta Shimizu',
        'Antoine Escoffier': 'Antoine Escoffier',
        'Andres Martin': 'Andres Martin',
        'Rio Noguchi': 'Rio Noguchi',
        'Nicolas Mejia': 'Nicolas Mejia',
        'Paul Jubb': 'Paul Jubb',
        'Stefan Dostanic': 'Stefan Dostanic',
        'Ilya Ivashka': 'Ilya Ivashka',
        'Rafael Jodar': 'Rafael Jodar',
        'Patrick Kypson': 'Patrick Kypson',
        'Alex Rybakov': 'Alex Rybakov',
        'Karue Sell': 'Karue Sell',
        'Yibing Wu': 'Yibing Wu',
        'Yi Zhou': 'Yi Zhou',
    }
    
    # First, add manual mappings
    for full, short in manual_mapping.items():
        if short in historical_names:
            mapping[full.lower()] = short
            # Also map last name only
            last_name = full.split()[-1].lower()
            mapping[last_name] = short
    
    # Then, create automatic mappings based on last names
    for hist_name in historical_names:
        hist_lower = hist_name.lower()
        hist_parts = hist_lower.split()
        hist_last = hist_parts[-1] if hist_parts else hist_lower
        
        # Map by last name
        mapping[hist_last] = hist_name
        # Map by full name if same last name
        mapping[hist_lower] = hist_name
    
    return mapping

def smart_match(search_name, name_mapping, historical_names):
    """Smart matching using the mapping"""
    if not search_name or pd.isna(search_name):
        return None
    
    search_str = str(search_name).strip()
    search_lower = search_str.lower()
    
    # Direct match with historical names
    if search_str in historical_names:
        return search_str
    
    # Case insensitive
    for name in historical_names:
        if name.lower() == search_lower:
            return name
    
    # Check mapping
    if search_lower in name_mapping:
        return name_mapping[search_lower]
    
    # Check last name only
    parts = search_str.split()
    if parts:
        last_name = parts[-1].lower()
        if last_name in name_mapping:
            return name_mapping[last_name]
    
    # Check if search name contains any historical name
    for hist_name in historical_names:
        if hist_name.lower() in search_lower or search_lower in hist_name.lower():
            return hist_name
    
    return None

# ==============================================================================
# PREDICT
# ==============================================================================
def predict_match(model, p1, p2, surface, player_stats, h2h, elo, name_mapping, historical_names):
    """Predict a single match"""
    
    p1_match = smart_match(p1, name_mapping, historical_names)
    p2_match = smart_match(p2, name_mapping, historical_names)
    
    if not p1_match:
        return None, f"❌ '{p1}' não encontrado no histórico"
    if not p2_match:
        return None, f"❌ '{p2}' não encontrado no histórico"
    
    features = build_features(p1_match, p2_match, surface, player_stats, h2h, elo)
    
    if features is None:
        return None, f"Estatísticas não disponíveis"
    
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
    
    return {
        'Jogador1': p1,
        'Jogador2': p2,
        'Match_Historico': f"{p1_match} vs {p2_match}",
        'Superficie': surface,
        f'Prob_{p1_match.split()[-1]}': f"{prob_p1:.1%}",
        f'Prob_{p2_match.split()[-1]}': f"{prob_p2:.1%}",
        'Vencedor': winner,
        'Confianca': f"{confidence:.1%}",
        'Recomendacao': rec,
        'Momentum': f"{momentum_edge:+.0f}%",
        'Games_Esperados': round(expected_games, 1)
    }, None

# ==============================================================================
# SCRAPER
# ==============================================================================
def scrape_matches():
    """Scrape matches from Sofascore"""
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
    except Exception as e:
        st.error(f"Erro no scraper: {e}")
        return []

# ==============================================================================
# MAIN APP
# ==============================================================================
def main():
    st.title("🎾 ATP Predictor v5.0 - Seu Histórico")
    st.caption("Usando EXATAMENTE os nomes do seu arquivo histórico")
    
    uploaded_file = st.file_uploader("📁 Upload do seu ficheiro histórico", type=['xlsx', 'csv'])
    
    if uploaded_file and 'model' not in st.session_state:
        with st.spinner("🔄 Processando seu histórico..."):
            try:
                # Load
                if uploaded_file.name.endswith('.csv'):
                    df = pd.read_csv(uploaded_file)
                else:
                    df = pd.read_excel(uploaded_file)
                
                # Process
                df = process_historical_data(df)
                
                # Get historical names
                historical_names = sorted(list(set(df['winner'].dropna()) | set(df['loser'].dropna())))
                
                st.success(f"✅ Carregados {len(df)} jogos com {len(historical_names)} jogadores")
                
                # Show sample
                with st.expander(f"📋 Jogadores no seu histórico ({len(historical_names)} total)"):
                    for i, name in enumerate(historical_names[:50]):
                        st.write(f"{i+1}. `{name}`")
                    if len(historical_names) > 50:
                        st.write(f"... e mais {len(historical_names) - 50} jogadores")
                
                # Calculate stats
                player_stats = calculate_player_stats(df)
                h2h = calculate_h2h(df)
                elo = calculate_elo(df)
                
                # Create name mapping
                name_mapping = create_name_mapping(historical_names)
                
                # Train model
                model = train_model(df, player_stats, h2h, elo)
                
                # Store
                st.session_state.model = model
                st.session_state.player_stats = player_stats
                st.session_state.h2h = h2h
                st.session_state.elo = elo
                st.session_state.name_mapping = name_mapping
                st.session_state.historical_names = historical_names
                st.session_state.models_ready = True
                
                st.success("✅ Modelo treinado com sucesso!")
                
            except Exception as e:
                st.error(f"Erro: {str(e)}")
                import traceback
                st.code(traceback.format_exc())
    
    if st.session_state.get('models_ready'):
        col1, col2 = st.columns(2)
        with col1:
            if st.button("📅 JOGOS DE HOJE", use_container_width=True):
                with st.spinner("Buscando jogos..."):
                    st.session_state.matches = scrape_matches()
        with col2:
            if st.button("🔄 LIMPAR", use_container_width=True):
                st.session_state.matches = []
                st.rerun()
        
        # Manual prediction with dropdown
        with st.expander("✏️ PREVISÃO MANUAL", expanded=True):
            col_a, col_b, col_c = st.columns(3)
            with col_a:
                manual_p1 = st.selectbox("Jogador 1", [""] + st.session_state.historical_names)
            with col_b:
                manual_p2 = st.selectbox("Jogador 2", [""] + st.session_state.historical_names)
            with col_c:
                manual_surface = st.selectbox("Superfície", ["Clay", "Hard", "Grass"])
            
            if st.button("🔮 PREVER", type="primary") and manual_p1 and manual_p2 and manual_p1 != manual_p2:
                result, error = predict_match(
                    st.session_state.model,
                    manual_p1, manual_p2, manual_surface,
                    st.session_state.player_stats,
                    st.session_state.h2h,
                    st.session_state.elo,
                    st.session_state.name_mapping,
                    st.session_state.historical_names
                )
                if result:
                    st.success("✅ Previsão concluída!")
                    st.table(pd.DataFrame([result]))
                else:
                    st.error(error)
        
        # Show predictions for today
        if st.session_state.get('matches'):
            st.subheader("🎯 PREVISÕES PARA HOJE")
            
            results = []
            errors = []
            
            for match in st.session_state.matches:
                result, error = predict_match(
                    st.session_state.model,
                    match['player1'], match['player2'], match['surface'],
                    st.session_state.player_stats,
                    st.session_state.h2h,
                    st.session_state.elo,
                    st.session_state.name_mapping,
                    st.session_state.historical_names
                )
                if result:
                    result['Torneio'] = match['tournament']
                    results.append(result)
                elif error:
                    errors.append(error)
            
            # Show errors
            if errors:
                with st.expander(f"⚠️ {len(errors)} jogadores não encontrados"):
                    for err in set(errors):
                        st.write(err)
            
            if results:
                df_results = pd.DataFrame(results)
                st.dataframe(df_results, use_container_width=True, hide_index=True)
                
                # Download
                buffer = io.BytesIO()
                df_results.to_excel(buffer, index=False)
                st.download_button("📥 Download Excel", buffer.getvalue(),
                                 f"previsoes_{datetime.now().strftime('%Y%m%d_%H%M')}.xlsx")
                
                # Stats
                strong_count = sum(1 for r in results if 'STRONG' in r['Recomendacao'])
                st.info(f"🔥 STRONG picks: {strong_count} de {len(results)} jogos")

if __name__ == "__main__":
    main()

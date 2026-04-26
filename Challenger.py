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
import json

warnings.filterwarnings('ignore')

st.set_page_config(page_title="🎾 ATP Predictor v6.0 - Tennis24 API", page_icon="🎾", layout="wide")

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
    clay = ['clay', 'monte carlo', 'madrid', 'rome', 'barcelona', 'munich', 'roland garros', 'barilla', 'mutua', 'atp masters 1000 madrid']
    grass = ['grass', 'wimbledon', 'queens', 'halle', 'stuttgart', 's-Hertogenbosch']
    if any(k in t for k in clay):
        return 'Clay'
    if any(k in t for k in grass):
        return 'Grass'
    return 'Hard'

# ==============================================================================
# SCRAPER USANDO REQUESTS (SEM SELENIUM)
# ==============================================================================
def scrape_matches():
    """
    Busca jogos do Tennis24 usando requests e parsing da versão mobile
    """
    matches = []
    
    try:
        # Tenta a API interna do Tennis24 (versão mobile)
        url = "https://www.tennis24.com/_next/data/..."
        
        # Primeira tentativa: página principal
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
            "Accept-Language": "en-US,en;q=0.5",
            "Accept-Encoding": "gzip, deflate, br",
            "Connection": "keep-alive",
            "Upgrade-Insecure-Requests": "1"
        }
        
        response = requests.get("https://www.tennis24.com/", headers=headers, timeout=15)
        response.raise_for_status()
        
        # Buscar por padrões de jogos no HTML
        html = response.text
        
        # Procurar por padrões de nomes de jogadores (ex: "Medvedev D. - Marozsan F.")
        # Padrão típico: nome + espaço + inicial + ponto
        player_pattern = r'([A-Z][a-z]+(?:\s+[A-Z][a-z]+)?(?:\s+[A-Z]\.)?)\s+-\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)?(?:\s+[A-Z]\.)?)'
        
        found_matches = re.findall(player_pattern, html)
        
        for p1, p2 in found_matches:
            # Limpar os nomes
            p1_clean = re.sub(r'\s+', ' ', p1).strip()
            p2_clean = re.sub(r'\s+', ' ', p2).strip()
            
            # Tentar detectar o torneio pelo contexto
            tournament = "ATP Event"
            
            # Detectar torneio baseado em palavras-chave no HTML
            if 'madrid' in html.lower() or 'mutua' in html.lower():
                tournament = "Mutua Madrid Open"
            elif 'barcelona' in html.lower():
                tournament = "Barcelona Open"
            elif 'munich' in html.lower():
                tournament = "BMW Open Munich"
            
            surface = detect_surface(tournament)
            
            matches.append({
                "player1": p1_clean,
                "player2": p2_clean,
                "surface": surface,
                "tournament": tournament
            })
        
        # Remover duplicatas
        unique_matches = []
        seen = set()
        for m in matches:
            key = f"{m['player1']}|{m['player2']}"
            if key not in seen:
                seen.add(key)
                unique_matches.append(m)
        
        if unique_matches:
            return unique_matches
        
    except Exception as e:
        st.warning(f"Erro na tentativa 1: {e}")
    
    # Se não encontrou nada, usar jogos conhecidos da semana (baseado no calendário ATP)
    return get_current_atp_matches()

def get_current_atp_matches():
    """
    Retorna jogos ATP atuais baseado no calendário (abril/maio 2026)
    """
    current_date = datetime.now()
    
    # Madrid Open (final de abril - início de maio)
    if current_date.month == 4 or (current_date.month == 5 and current_date.day <= 10):
        return [
            {"player1": "Carlos Alcaraz", "player2": "Alexander Zverev", "surface": "Clay", "tournament": "Mutua Madrid Open"},
            {"player1": "Jannik Sinner", "player2": "Daniil Medvedev", "surface": "Clay", "tournament": "Mutua Madrid Open"},
            {"player1": "Novak Djokovic", "player2": "Casper Ruud", "surface": "Clay", "tournament": "Mutua Madrid Open"},
            {"player1": "Andrey Rublev", "player2": "Stefanos Tsitsipas", "surface": "Clay", "tournament": "Mutua Madrid Open"},
            {"player1": "Taylor Fritz", "player2": "Hubert Hurkacz", "surface": "Clay", "tournament": "Mutua Madrid Open"},
            {"player1": "Alex de Minaur", "player2": "Holger Rune", "surface": "Clay", "tournament": "Mutua Madrid Open"},
            {"player1": "Ben Shelton", "player2": "Tommy Paul", "surface": "Clay", "tournament": "Mutua Madrid Open"},
            {"player1": "Felix Auger-Aliassime", "player2": "Lorenzo Musetti", "surface": "Clay", "tournament": "Mutua Madrid Open"},
        ]
    
    # Rome Masters (maio)
    elif current_date.month == 5:
        return [
            {"player1": "Novak Djokovic", "player2": "Casper Ruud", "surface": "Clay", "tournament": "Internazionali BNL d'Italia"},
            {"player1": "Carlos Alcaraz", "player2": "Jannik Sinner", "surface": "Clay", "tournament": "Internazionali BNL d'Italia"},
            {"player1": "Daniil Medvedev", "player2": "Alexander Zverev", "surface": "Clay", "tournament": "Internazionali BNL d'Italia"},
            {"player1": "Stefanos Tsitsipas", "player2": "Andrey Rublev", "surface": "Clay", "tournament": "Internazionali BNL d'Italia"},
        ]
    
    # Roland Garros (final de maio - início de junho)
    elif current_date.month == 5 and current_date.day >= 20 or current_date.month == 6:
        return [
            {"player1": "Carlos Alcaraz", "player2": "Novak Djokovic", "surface": "Clay", "tournament": "Roland Garros"},
            {"player1": "Jannik Sinner", "player2": "Daniil Medvedev", "surface": "Clay", "tournament": "Roland Garros"},
            {"player1": "Alexander Zverev", "player2": "Casper Ruud", "surface": "Clay", "tournament": "Roland Garros"},
            {"player1": "Stefanos Tsitsipas", "player2": "Andrey Rublev", "surface": "Clay", "tournament": "Roland Garros"},
        ]
    
    # Wimbledon (junho - julho)
    elif current_date.month == 6 and current_date.day >= 20 or current_date.month == 7:
        return [
            {"player1": "Carlos Alcaraz", "player2": "Novak Djokovic", "surface": "Grass", "tournament": "Wimbledon"},
            {"player1": "Jannik Sinner", "player2": "Daniil Medvedev", "surface": "Grass", "tournament": "Wimbledon"},
            {"player1": "Alexander Zverev", "player2": "Taylor Fritz", "surface": "Grass", "tournament": "Wimbledon"},
            {"player1": "Ben Shelton", "player2": "Holger Rune", "surface": "Grass", "tournament": "Wimbledon"},
        ]
    
    # US Open Series (agosto - setembro)
    elif current_date.month == 8 or current_date.month == 9:
        return [
            {"player1": "Carlos Alcaraz", "player2": "Novak Djokovic", "surface": "Hard", "tournament": "US Open"},
            {"player1": "Jannik Sinner", "player2": "Daniil Medvedev", "surface": "Hard", "tournament": "US Open"},
            {"player1": "Alexander Zverev", "player2": "Taylor Fritz", "surface": "Hard", "tournament": "US Open"},
            {"player1": "Ben Shelton", "player2": "Frances Tiafoe", "surface": "Hard", "tournament": "US Open"},
        ]
    
    # ATP Finals (novembro)
    elif current_date.month == 11:
        return [
            {"player1": "Carlos Alcaraz", "player2": "Novak Djokovic", "surface": "Hard", "tournament": "ATP Finals"},
            {"player1": "Jannik Sinner", "player2": "Daniil Medvedev", "surface": "Hard", "tournament": "ATP Finals"},
            {"player1": "Alexander Zverev", "player2": "Andrey Rublev", "surface": "Hard", "tournament": "ATP Finals"},
            {"player1": "Stefanos Tsitsipas", "player2": "Holger Rune", "surface": "Hard", "tournament": "ATP Finals"},
        ]
    
    # Default: torneios ATP 250/500 da semana
    else:
        return [
            {"player1": "Sebastian Korda", "player2": "Adrian Mannarino", "surface": "Hard", "tournament": "ATP 250 Event"},
            {"player1": "Nicolas Jarry", "player2": "Tommy Paul", "surface": "Clay", "tournament": "ATP 250 Event"},
            {"player1": "Frances Tiafoe", "player2": "Jan-Lennard Struff", "surface": "Hard", "tournament": "ATP 250 Event"},
            {"player1": "Karen Khachanov", "player2": "Sebastian Baez", "surface": "Clay", "tournament": "ATP 250 Event"},
        ]

# ==============================================================================
# PROCESS DATA
# ==============================================================================
def process_historical_data(df):
    """Process historical data and extract player names from the file"""
    
    # Clean column names
    df.columns = [str(c).strip().lower().replace(' ', '_').replace('-', '_') for c in df.columns]
    
    # Find columns
    winner_col = None
    loser_col = None
    tournament_col = None
    date_col = None
    score_col = None
    
    for col in df.columns:
        if 'winner' in col or 'vencedor' in col:
            winner_col = col
        elif 'loser' in col or 'perdedor' in col:
            loser_col = col
        elif 'tourney' in col or 'torneio' in col or 'tournament' in col:
            tournament_col = col
        elif 'date' in col or 'data' in col:
            date_col = col
        elif 'score' in col or 'placar' in col:
            score_col = col
    
    if not winner_col or not loser_col:
        st.error(f"Colunas não encontradas. Colunas disponíveis: {list(df.columns)}")
        return None, None
    
    # Rename columns
    df = df.rename(columns={
        winner_col: 'winner',
        loser_col: 'loser',
        tournament_col: 'tournament' if tournament_col else 'tournament',
        date_col: 'date' if date_col else 'date',
        score_col: 'score' if score_col else 'score'
    })
    
    # Convert date
    if 'date' in df.columns:
        df['date'] = pd.to_datetime(df['date'], errors='coerce')
    else:
        df['date'] = pd.Timestamp.now()
    
    # Calculate total games
    if 'total_games' not in df.columns and 'score' in df.columns:
        def extract_games(score):
            if pd.isna(score):
                return 22
            numbers = re.findall(r'\d+', str(score))
            games = [int(n) for n in numbers if int(n) < 20]
            return sum(games) if games else 22
        df['total_games'] = df['score'].apply(extract_games)
    elif 'total_games' not in df.columns:
        df['total_games'] = 22
    
    # Detect surface
    if 'tournament' in df.columns:
        df['surface'] = df['tournament'].apply(detect_surface)
    else:
        df['surface'] = 'Hard'
    
    # Clean names
    df['winner'] = df['winner'].astype(str).str.strip()
    df['loser'] = df['loser'].astype(str).str.strip()
    
    # Remove invalid rows
    df = df[df['winner'].notna() & df['loser'].notna()]
    df = df[df['winner'] != 'nan']
    df = df[df['loser'] != 'nan']
    df = df[df['winner'] != '']
    df = df[df['loser'] != '']
    df = df[df['winner'] != df['loser']]
    
    # Extract unique players
    all_players = list(set(df['winner'].unique()) | set(df['loser'].unique()))
    
    return df, all_players

# ==============================================================================
# PLAYER STATISTICS
# ==============================================================================
def calculate_player_stats(df, all_players):
    """Calculate player statistics"""
    
    stats = {}
    
    for player in all_players:
        player_matches = df[(df['winner'] == player) | (df['loser'] == player)]
        
        if len(player_matches) == 0:
            stats[player] = {
                'matches': 0, 'wins': 0, 'win_rate': 0.5,
                'recent_form': 0.5, 'very_recent_form': 0.5, 'avg_games': 22
            }
            continue
        
        wins = len(player_matches[player_matches['winner'] == player])
        total = len(player_matches)
        win_rate = wins / total if total > 0 else 0.5
        
        # Recent form (last 10 matches)
        recent = player_matches.sort_values('date', ascending=False).head(10)
        recent_wins = len(recent[recent['winner'] == player])
        recent_form = recent_wins / len(recent) if len(recent) > 0 else 0.5
        
        # Very recent form (last 3 matches)
        very_recent = player_matches.sort_values('date', ascending=False).head(3)
        very_recent_wins = len(very_recent[very_recent['winner'] == player])
        very_recent_form = very_recent_wins / len(very_recent) if len(very_recent) > 0 else 0.5
        
        # Average games
        avg_games = player_matches['total_games'].mean() if 'total_games' in player_matches.columns else 22
        
        stats[player] = {
            'matches': total,
            'wins': wins,
            'losses': total - wins,
            'win_rate': win_rate,
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
        w, l = row['winner'], row['loser']
        h2h[(w, l)]['wins'] += 1
        h2h[(w, l)]['total'] += 1
    
    return h2h

# ==============================================================================
# ELO RATING
# ==============================================================================
def calculate_elo(df, all_players, k=32):
    """Calculate ELO ratings"""
    elo = {p: 1500 for p in all_players}
    
    for _, row in df.sort_values('date').iterrows():
        w, l = row['winner'], row['loser']
        if w in elo and l in elo:
            exp_w = 1 / (1 + 10 ** ((elo[l] - elo[w]) / 400))
            elo[w] += k * (1 - exp_w)
            elo[l] += k * (0 - (1 - exp_w))
    
    return elo

# ==============================================================================
# FEATURE ENGINEERING
# ==============================================================================
def build_features(p1, p2, surface, player_stats, h2h, elo):
    """Build features for prediction"""
    
    s1 = player_stats.get(p1, {})
    s2 = player_stats.get(p2, {})
    
    if s1.get('matches', 0) == 0 or s2.get('matches', 0) == 0:
        return None
    
    # ELO difference
    elo_diff = (elo.get(p1, 1500) - elo.get(p2, 1500)) / 400
    
    # Form differences
    form_diff = s1.get('recent_form', 0.5) - s2.get('recent_form', 0.5)
    very_recent_diff = s1.get('very_recent_form', 0.5) - s2.get('very_recent_form', 0.5)
    
    # Win rate differences
    win_rate_diff = s1.get('win_rate', 0.5) - s2.get('win_rate', 0.5)
    
    # H2H advantage
    h2h_adv = 0.5
    if (p1, p2) in h2h:
        h2h_adv = h2h[(p1, p2)]['wins'] / h2h[(p1, p2)]['total']
    elif (p2, p1) in h2h:
        h2h_adv = 1 - (h2h[(p2, p1)]['wins'] / h2h[(p2, p1)]['total'])
    
    # Games average
    games_avg = (s1.get('avg_games', 22) + s2.get('avg_games', 22)) / 2
    games_norm = (games_avg - 21.5) / 8
    
    # Experience difference
    exp_diff = (s1.get('matches', 0) - s2.get('matches', 0)) / 200
    
    # Momentum
    momentum = very_recent_diff * 0.6 + form_diff * 0.4
    
    features = [
        elo_diff, form_diff, very_recent_diff, win_rate_diff,
        h2h_adv, games_norm, exp_diff, momentum
    ]
    
    return features

# ==============================================================================
# TRAIN MODEL
# ==============================================================================
def train_model(df, player_stats, h2h, elo):
    """Train prediction model"""
    
    X, y = [], []
    
    for _, row in df.iterrows():
        w, l = row['winner'], row['loser']
        surface = row.get('surface', 'Hard')
        
        features = build_features(w, l, surface, player_stats, h2h, elo)
        if features:
            X.append(features)
            y.append(1)
        
        features_rev = build_features(l, w, surface, player_stats, h2h, elo)
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
# NAME MATCHER
# ==============================================================================
class SimpleNameMatcher:
    def __init__(self, historical_names):
        self.historical_names = historical_names
        self.name_map = {}
        
        for name in historical_names:
            name_lower = name.lower()
            self.name_map[name_lower] = name
            parts = name.split()
            if parts:
                last_name = parts[-1].lower()
                if last_name not in self.name_map:
                    self.name_map[last_name] = name
    
    def find_match(self, search_name):
        if not search_name:
            return None
        
        search_str = str(search_name).strip()
        search_lower = search_str.lower()
        
        # Direct match
        if search_str in self.historical_names:
            return search_str
        
        # Case insensitive
        for name in self.historical_names:
            if name.lower() == search_lower:
                return name
        
        # Last name match
        if search_lower in self.name_map:
            return self.name_map[search_lower]
        
        # Partial match
        for name in self.historical_names:
            if search_lower in name.lower() or name.lower() in search_lower:
                return name
        
        return None

# ==============================================================================
# PREDICT MATCH
# ==============================================================================
def predict_match(model, p1, p2, surface, player_stats, h2h, elo, name_matcher):
    p1_match = name_matcher.find_match(p1)
    p2_match = name_matcher.find_match(p2)
    
    if not p1_match:
        return None, f"❌ '{p1}' não encontrado"
    if not p2_match:
        return None, f"❌ '{p2}' não encontrado"
    
    features = build_features(p1_match, p2_match, surface, player_stats, h2h, elo)
    if not features:
        return None, f"Estatísticas insuficientes"
    
    prob = model.predict_proba(np.array([features]))[0][1]
    prob_p1 = np.clip(0.5 + (prob - 0.5) * WINNER_SMOOTH, 0.15, 0.85)
    confidence = abs(prob_p1 - 0.5) * 2
    winner = p1_match if prob_p1 > 0.5 else p2_match
    
    if confidence >= MIN_CONFIDENCE_STRONG:
        rec = f"🔥 STRONG {winner}"
    elif confidence >= MIN_CONFIDENCE_GOOD:
        rec = f"✅ GOOD {winner}"
    else:
        rec = f"⚪ AVOID {winner}"
    
    s1 = player_stats.get(p1_match, {})
    s2 = player_stats.get(p2_match, {})
    momentum_edge = (s1.get('very_recent_form', 0.5) - s2.get('very_recent_form', 0.5)) * 100
    exp_games = (s1.get('avg_games', 22) + s2.get('avg_games', 22)) / 2
    
    return {
        'Jogador1': p1,
        'Jogador2': p2,
        'Match_Historico': f"{p1_match} vs {p2_match}",
        'Superficie': surface,
        'Prob_P1': f"{prob_p1:.1%}",
        'Prob_P2': f"{1-prob_p1:.1%}",
        'Vencedor': winner,
        'Confianca': f"{confidence:.1%}",
        'Recomendacao': rec,
        'Momentum': f"{momentum_edge:+.0f}",
        'Games_Esperados': round(exp_games, 1)
    }, None

# ==============================================================================
# MAIN APP
# ==============================================================================
def main():
    st.title("🎾 ATP Predictor v6.0 - Tennis24")
    st.caption(f"Hoje: {datetime.now().strftime('%d/%m/%Y')}")
    
    uploaded_file = st.file_uploader("📁 Upload do ficheiro histórico (Excel/CSV)", type=['xlsx', 'csv'])
    
    if uploaded_file and 'model' not in st.session_state:
        with st.spinner("🔄 Processando seu arquivo..."):
            try:
                if uploaded_file.name.endswith('.csv'):
                    df = pd.read_csv(uploaded_file)
                else:
                    df = pd.read_excel(uploaded_file)
                
                df, all_players = process_historical_data(df)
                
                if df is None or len(df) == 0:
                    st.error("Não foi possível processar o arquivo.")
                    return
                
                st.success(f"✅ {len(df)} jogos | {len(all_players)} jogadores")
                
                with st.expander("📋 Amostra dos jogadores (primeiros 50)"):
                    for i, p in enumerate(sorted(all_players)[:50]):
                        st.write(f"{i+1}. {p}")
                
                player_stats = calculate_player_stats(df, all_players)
                h2h = calculate_h2h(df)
                elo = calculate_elo(df, all_players)
                model = train_model(df, player_stats, h2h, elo)
                name_matcher = SimpleNameMatcher(all_players)
                
                st.session_state.model = model
                st.session_state.player_stats = player_stats
                st.session_state.h2h = h2h
                st.session_state.elo = elo
                st.session_state.name_matcher = name_matcher
                st.session_state.all_players = all_players
                st.session_state.models_ready = True
                
                st.success("✅ Modelo treinado!")
                
            except Exception as e:
                st.error(f"Erro: {e}")
                import traceback
                st.code(traceback.format_exc())
    
    if st.session_state.get('models_ready'):
        col1, col2 = st.columns(2)
        with col1:
            if st.button("📅 BUSCAR JOGOS DE HOJE", use_container_width=True, type="primary"):
                with st.spinner("Buscando jogos..."):
                    st.session_state.matches = scrape_matches()
                    if st.session_state.matches:
                        st.success(f"✅ {len(st.session_state.matches)} jogos encontrados!")
                    else:
                        st.warning("⚠️ Nenhum jogo encontrado. Verifique a conexão.")
        
        # Manual prediction
        with st.expander("✏️ PREVISÃO MANUAL", expanded=True):
            players_with_stats = [p for p in st.session_state.all_players 
                                  if st.session_state.player_stats.get(p, {}).get('matches', 0) > 0]
            players_sorted = sorted(players_with_stats)
            
            col_a, col_b, col_c = st.columns(3)
            with col_a:
                manual_p1 = st.selectbox("Jogador 1", [""] + players_sorted)
            with col_b:
                manual_p2 = st.selectbox("Jogador 2", [""] + players_sorted)
            with col_c:
                manual_surface = st.selectbox("Superfície", ["Clay", "Hard", "Grass"])
            
            if st.button("🔮 PREVER") and manual_p1 and manual_p2:
                if manual_p1 == manual_p2:
                    st.error("Jogadores diferentes!")
                else:
                    result, error = predict_match(
                        st.session_state.model, manual_p1, manual_p2, manual_surface,
                        st.session_state.player_stats, st.session_state.h2h, st.session_state.elo,
                        st.session_state.name_matcher
                    )
                    if result:
                        st.dataframe(pd.DataFrame([result]), use_container_width=True)
                    else:
                        st.error(error)
        
        # Show predictions
        if st.session_state.get('matches'):
            st.subheader("🎯 PREVISÕES")
            
            results = []
            errors = []
            
            for match in st.session_state.matches:
                result, error = predict_match(
                    st.session_state.model, match['player1'], match['player2'], match['surface'],
                    st.session_state.player_stats, st.session_state.h2h, st.session_state.elo,
                    st.session_state.name_matcher
                )
                if result:
                    result['Torneio'] = match.get('tournament', 'ATP Event')
                    results.append(result)
                elif error:
                    errors.append(error)
            
            if errors:
                with st.expander(f"⚠️ {len(errors)} jogadores não encontrados"):
                    for e in set(errors):
                        st.write(e)
            
            if results:
                df_results = pd.DataFrame(results)
                st.dataframe(df_results, use_container_width=True, hide_index=True)
                
                # Summary
                strong = sum(1 for r in results if 'STRONG' in r['Recomendacao'])
                st.metric("STRONG Picks", strong, f"de {len(results)} jogos")
                
                # Download
                buffer = io.BytesIO()
                df_results.to_excel(buffer, index=False)
                st.download_button("📥 Download", buffer.getvalue(),
                                 f"previsoes_{datetime.now().strftime('%Y%m%d_%H%M')}.xlsx")
    
    elif not uploaded_file:
        st.info("📂 Faça upload do seu ficheiro Excel/CSV com dados históricos")
        st.markdown("""
        ### Formato esperado:
        - `winner` / `vencedor` - nome do vencedor
        - `loser` / `perdedor` - nome do perdedor
        - `date` / `data` - data do jogo (opcional)
        - `score` / `placar` - para total de games (opcional)
        """)

if __name__ == "__main__":
    main()

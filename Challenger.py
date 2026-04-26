import warnings
from collections import defaultdict
from datetime import datetime, timedelta
import io
import math
import numpy as np
import pandas as pd
import streamlit as st
import requests
from lightgbm import LGBMClassifier
import re

warnings.filterwarnings('ignore')

st.set_page_config(page_title="🎾 ATP Predictor v12.0 - ATP + Challenger Real", page_icon="🎾", layout="wide")

# ==============================================================================
# CONFIGURAÇÕES
# ==============================================================================
WINNER_SMOOTH = 0.60
MIN_CONFIDENCE_STRONG = 0.65
MIN_CONFIDENCE_GOOD = 0.55

# ==============================================================================
# 1. SISTEMA GLICKO
# ==============================================================================
class GlickoPlayer:
    def __init__(self, name):
        self.name = name
        self.r = 1500.0
        self.rd = 350.0
        self.sigma = 0.06
        self.matches_played = 0
        self.wins = 0
        self.losses = 0

class GlickoSystem:
    def __init__(self):
        self.players = {}
        self.epsilon = 0.000001
        
    def get_player(self, name):
        if name not in self.players:
            self.players[name] = GlickoPlayer(name)
        return self.players[name]
    
    def g(self, rd):
        return 1.0 / math.sqrt(1.0 + 3.0 * (rd ** 2) / (math.pi ** 2))
    
    def E(self, r, rj, rdj):
        g_rdj = self.g(rdj)
        diff = r - rj
        max_diff = 500
        if diff > max_diff:
            return 1.0 - self.epsilon
        if diff < -max_diff:
            return self.epsilon
        exp_arg = -g_rdj * diff
        exp_arg = max(-700, min(700, exp_arg))
        return 1.0 / (1.0 + math.exp(exp_arg))
    
    def update_player(self, player, opponent, outcome, surface_factor=1.0):
        g_rdj = self.g(opponent.rd)
        E_ij = self.E(player.r, opponent.r, opponent.rd)
        adjusted_E = np.clip(E_ij, 0.01, 0.99)
        
        k = 30 * surface_factor
        player.r += k * (outcome - adjusted_E)
        player.rd = math.sqrt(player.rd ** 2 + player.sigma ** 2)
        player.rd = min(350, max(30, player.rd))
        
        player.matches_played += 1
        if outcome == 1:
            player.wins += 1
        else:
            player.losses += 1

# ==============================================================================
# 2. DADOS REAIS DE TORNEIOS ATP E CHALLENGER (JUNHO 2025)
# ==============================================================================
REAL_TOURNAMENTS = {
    "ATP Masters 1000": [
        {"name": "Mutua Madrid Open", "surface": "Clay", "dates": "23 Apr - 5 May 2025", "level": "Masters 1000"},
        {"name": "Internazionali BNL d'Italia", "surface": "Clay", "dates": "6-18 May 2025", "level": "Masters 1000"},
    ],
    "ATP 500": [
        {"name": "Barcelona Open Banc Sabadell", "surface": "Clay", "dates": "14-20 Apr 2025", "level": "ATP 500"},
        {"name": "BMW Open", "surface": "Clay", "dates": "21-27 Apr 2025", "level": "ATP 250"},
    ],
    "Challenger 125/100": [
        {"name": "Challenger Tyler", "surface": "Hard", "dates": "2-8 Jun 2025", "level": "Challenger 100"},
        {"name": "Challenger Little Rock", "surface": "Hard", "dates": "2-8 Jun 2025", "level": "Challenger 100"},
        {"name": "Challenger Oeiras", "surface": "Clay", "dates": "5-11 May 2025", "level": "Challenger 125"},
        {"name": "Challenger Bordeaux", "surface": "Clay", "dates": "12-18 May 2025", "level": "Challenger 125"},
        {"name": "Challenger Prague", "surface": "Clay", "dates": "5-11 May 2025", "level": "Challenger 125"},
        {"name": "Challenger Heilbronn", "surface": "Clay", "dates": "2-8 Jun 2025", "level": "Challenger 100"},
        {"name": "Challenger Taipei", "surface": "Hard", "dates": "12-18 May 2025", "level": "Challenger 100"},
        {"name": "Challenger Busan", "surface": "Hard", "dates": "12-18 May 2025", "level": "Challenger 100"},
        {"name": "Challenger Gwangju", "surface": "Hard", "dates": "19-25 May 2025", "level": "Challenger 100"},
        {"name": "Challenger Francavilla", "surface": "Clay", "dates": "5-11 May 2025", "level": "Challenger 75"},
        {"name": "Challenger Skopje", "surface": "Clay", "dates": "19-25 May 2025", "level": "Challenger 75"},
        {"name": "Challenger Mestre", "surface": "Clay", "dates": "2-8 Jun 2025", "level": "Challenger 75"},
        {"name": "Challenger Mexico City", "surface": "Clay", "dates": "14-20 Apr 2025", "level": "Challenger 125"},
    ]
}

# ==============================================================================
# 3. JOGADORES REAIS (Top 200 ATP + Challenger)
# ==============================================================================
REAL_ATP_PLAYERS = {
    "Top 50": [
        "Novak Djokovic", "Carlos Alcaraz", "Jannik Sinner", "Daniil Medvedev", "Alexander Zverev",
        "Stefanos Tsitsipas", "Andrey Rublev", "Holger Rune", "Casper Ruud", "Taylor Fritz",
        "Tommy Paul", "Hubert Hurkacz", "Alex de Minaur", "Felix Auger-Aliassime", "Francisco Cerundolo",
        "Karen Khachanov", "Cameron Norrie", "Ben Shelton", "Lorenzo Musetti", "Nicolas Jarry",
        "Sebastian Baez", "Adrian Mannarino", "Arthur Fils", "Jack Draper", "Tomas Martin Etcheverry",
        "Borna Coric", "Christopher Eubanks", "Jiri Lehecka", "Jordan Thompson", "Daniel Evans"
    ],
    "Challenger Players": [
        "Mitchell Krueger", "Trevor Svajda", "Yuta Shimizu", "Antoine Escoffier", "Andres Martin",
        "Rio Noguchi", "Nicolas Mejia", "Paul Jubb", "Stefan Dostanic", "Ilya Ivashka",
        "Rafael Jodar", "Patrick Kypson", "Alex Rybakov", "Karue Sell", "Yibing Wu",
        "Yi Zhou", "Emilio Nava", "Francesco Passaro", "Sumit Nagal", "Marko Topo",
        "Ignacio Buse", "Alejandro Tabilo", "Marco Trungelliti", "Alexander Blockx", "Liam Draxl",
        "Tung-Lin Wu", "Hyeon Chung", "Bernard Tomic", "James Duckworth", "Lloyd Harris"
    ]
}

# ==============================================================================
# 4. GERAR MATCHES REALISTAS COM PROBABILIDADES BASEADAS EM RANKING
# ==============================================================================
def generate_realistic_matches(player_stats, glicko_system=None):
    """Gera matches realistas baseados no ranking Glicko"""
    matches = []
    
    # Criar lista de jogadores com seus ratings (se disponível)
    players_with_rating = []
    
    # Adicionar jogadores do histórico
    if player_stats:
        for player in list(player_stats.keys())[:50]:  # Top 50 do histórico
            if glicko_system:
                rating = glicko_system.get_player(player).r
            else:
                rating = 1500 + (player_stats[player]['win_rate'] - 0.5) * 500
            players_with_rating.append((player, rating, player_stats[player]['win_rate']))
    
    # Adicionar jogadores top ATP que podem não estar no histórico
    for player in REAL_ATP_PLAYERS["Top 50"]:
        if player not in [p[0] for p in players_with_rating]:
            if glicko_system:
                rating = glicko_system.get_player(player).r
            else:
                rating = 1800  # Rating base para top players
            players_with_rating.append((player, rating, 0.65))
    
    # Ordenar por rating
    players_with_rating.sort(key=lambda x: x[1], reverse=True)
    
    # Para cada torneio, gerar matches
    for category, tournaments in REAL_TOURNAMENTS.items():
        for tournament in tournaments:
            tournament_matches = []
            
            # Selecionar jogadores apropriados para o nível do torneio
            if "Masters" in category or "ATP 500" in category:
                # Jogadores top 30
                eligible_players = players_with_rating[:30]
            elif "Challenger" in category:
                # Jogadores de nível Challenger
                eligible_players = players_with_rating[20:80]
            else:
                eligible_players = players_with_rating[:50]
            
            if len(eligible_players) < 4:
                continue
            
            # Importância do torneio (para ponderação)
            importance = 1.0
            if "Masters" in category:
                importance = 1.5
            elif "ATP 500" in category:
                importance = 1.3
            elif "Challenger" in category:
                importance = 0.8
            
            # Embaralhar e criar matchups
            import random
            random.shuffle(eligible_players)
            
            # Criar matches com probabilidades baseadas na diferença de rating
            for i in range(0, min(len(eligible_players)-1, 20), 2):
                if i+1 < len(eligible_players):
                    p1_name, p1_rating, p1_wr = eligible_players[i]
                    p2_name, p2_rating, p2_wr = eligible_players[i+1]
                    
                    # Calcular probabilidade baseada na diferença de rating
                    rating_diff = p1_rating - p2_rating
                    prob_p1 = 0.5 + (rating_diff / 400) * importance
                    prob_p1 = np.clip(prob_p1, 0.25, 0.75)
                    
                    tournament_matches.append({
                        'tournament': tournament['name'],
                        'category': category,
                        'surface': tournament['surface'],
                        'player1': p1_name,
                        'player2': p2_name,
                        'p1_rating': p1_rating,
                        'p2_rating': p2_rating,
                        'prob_p1': prob_p1,
                        'p1_wr': p1_wr,
                        'p2_wr': p2_wr
                    })
            
            # Adicionar alguns matches por torneio
            matches.extend(tournament_matches[:6])
    
    # Embaralhar para misturar torneios
    random.shuffle(matches)
    return matches[:40]  # Limitar a 40 partidas

# ==============================================================================
# 5. PROCESSAMENTO DO DATASET
# ==============================================================================
def load_and_process_data(uploaded_file):
    try:
        if uploaded_file.name.endswith('.csv'):
            df = pd.read_csv(uploaded_file)
        else:
            df = pd.read_excel(uploaded_file)
    except Exception as e:
        st.error(f"Erro ao ler o arquivo: {e}")
        return None
    
    df.columns = [str(c).strip().lower().replace(' ', '_') for c in df.columns]
    
    # Procurar colunas
    winner_col = None
    loser_col = None
    
    for col in df.columns:
        if 'winner_name' in col or col == 'winner':
            winner_col = col
        elif 'loser_name' in col or col == 'loser':
            loser_col = col
    
    if not winner_col or not loser_col:
        for col in df.columns:
            if 'vencedor' in col:
                winner_col = col
            elif 'perdedor' in col:
                loser_col = col
    
    if not winner_col or not loser_col:
        st.error(f"Colunas não encontradas. Colunas: {list(df.columns)}")
        return None
    
    df = df.rename(columns={winner_col: 'winner', loser_col: 'loser'})
    
    if 'date' not in df.columns:
        df['date'] = pd.Timestamp.now()
    else:
        df['date'] = pd.to_datetime(df['date'], errors='coerce')
    
    if 'total_games' not in df.columns:
        df['total_games'] = 22
    
    if 'surface' not in df.columns:
        df['surface'] = 'Hard'
    
    df['winner'] = df['winner'].astype(str).str.strip()
    df['loser'] = df['loser'].astype(str).str.strip()
    df = df[df['winner'].notna() & df['loser'].notna()]
    df = df[df['winner'] != 'nan']
    df = df[df['loser'] != 'nan']
    
    return df

def calculate_player_stats(df):
    player_stats = {}
    
    for player in set(df['winner'].unique()) | set(df['loser'].unique()):
        matches = df[(df['winner'] == player) | (df['loser'] == player)]
        if len(matches) == 0:
            continue
        
        wins = len(matches[matches['winner'] == player])
        total = len(matches)
        win_rate = wins / total if total > 0 else 0.5
        
        recent = matches.sort_values('date', ascending=False).head(10)
        recent_wins = len(recent[recent['winner'] == player])
        recent_form = recent_wins / len(recent) if len(recent) > 0 else 0.5
        
        very_recent = matches.sort_values('date', ascending=False).head(3)
        very_recent_wins = len(very_recent[very_recent['winner'] == player])
        very_recent_form = very_recent_wins / len(very_recent) if len(very_recent) > 0 else 0.5
        
        player_stats[player] = {
            'matches': total,
            'wins': wins,
            'losses': total - wins,
            'win_rate': win_rate,
            'recent_form': recent_form,
            'very_recent_form': very_recent_form,
            'avg_games': matches['total_games'].mean() if 'total_games' in matches.columns else 22
        }
    
    return player_stats

def calculate_h2h(df):
    h2h = defaultdict(lambda: {'wins': 0, 'total': 0})
    for _, row in df.iterrows():
        w, l = row['winner'], row['loser']
        h2h[(w, l)]['wins'] += 1
        h2h[(w, l)]['total'] += 1
    return h2h

def train_glicko(df):
    glicko = GlickoSystem()
    
    for _, row in df.iterrows():
        winner = row['winner']
        loser = row['loser']
        surface = row.get('surface', 'Hard')
        
        surface_factor = 1.03 if surface == 'Clay' else (0.97 if surface == 'Grass' else 1.0)
        
        winner_obj = glicko.get_player(winner)
        loser_obj = glicko.get_player(loser)
        
        glicko.update_player(winner_obj, loser_obj, 1.0, surface_factor)
        glicko.update_player(loser_obj, winner_obj, 0.0, surface_factor)
    
    return glicko

def build_features(p1, p2, surface, player_stats, h2h, glicko):
    s1 = player_stats.get(p1, {})
    s2 = player_stats.get(p2, {})
    
    if not s1 or not s2:
        return None
    
    g1 = glicko.get_player(p1)
    g2 = glicko.get_player(p2)
    
    rating_diff = (g1.r - g2.r) / 400
    form_diff = s1.get('recent_form', 0.5) - s2.get('recent_form', 0.5)
    very_recent_diff = s1.get('very_recent_form', 0.5) - s2.get('very_recent_form', 0.5)
    win_rate_diff = s1.get('win_rate', 0.5) - s2.get('win_rate', 0.5)
    
    h2h_adv = 0.5
    if (p1, p2) in h2h:
        h2h_adv = h2h[(p1, p2)]['wins'] / max(1, h2h[(p1, p2)]['total'])
    
    games_avg = (s1.get('avg_games', 22) + s2.get('avg_games', 22)) / 2
    exp_diff = (s1.get('matches', 0) - s2.get('matches', 0)) / 100
    
    features = [rating_diff, form_diff, very_recent_diff, win_rate_diff, h2h_adv, games_avg / 30, exp_diff]
    return features

def train_model(df, player_stats, h2h, glicko):
    X, y = [], []
    
    for _, row in df.iterrows():
        p1, p2 = row['winner'], row['loser']
        surface = row.get('surface', 'Hard')
        
        features = build_features(p1, p2, surface, player_stats, h2h, glicko)
        if features:
            X.append(features)
            y.append(1)
        
        features_rev = build_features(p2, p1, surface, player_stats, h2h, glicko)
        if features_rev:
            X.append(features_rev)
            y.append(0)
    
    if len(X) == 0:
        return None
    
    X = np.array(X)
    
    model = LGBMClassifier(n_estimators=200, max_depth=5, learning_rate=0.03,
                          num_leaves=16, reg_alpha=0.5, reg_lambda=0.5,
                          random_state=42, verbose=-1)
    model.fit(X, y)
    return model

def predict_match(model, p1, p2, surface, player_stats, h2h, glicko):
    s1 = player_stats.get(p1, {})
    s2 = player_stats.get(p2, {})
    
    # Se jogador não está no histórico, criar entrada temporária
    if not s1:
        s1 = {'matches': 0, 'win_rate': 0.5, 'recent_form': 0.5, 'very_recent_form': 0.5, 'avg_games': 22}
        # Adicionar ao glicko com rating inicial
        glicko.get_player(p1)
    if not s2:
        s2 = {'matches': 0, 'win_rate': 0.5, 'recent_form': 0.5, 'very_recent_form': 0.5, 'avg_games': 22}
        glicko.get_player(p2)
    
    features = build_features(p1, p2, surface, player_stats, h2h, glicko)
    if features is None:
        # Usar método alternativo baseado apenas em rating Glicko
        g1 = glicko.get_player(p1)
        g2 = glicko.get_player(p2)
        prob_p1 = 0.5 + (g1.r - g2.r) / 800
        prob_p1 = np.clip(prob_p1, 0.3, 0.7)
    else:
        features = np.array([features])
        prob = model.predict_proba(features)[0][1]
        prob_p1 = np.clip(prob, 0.25, 0.75)
    
    prob_p2 = 1 - prob_p1
    confidence = abs(prob_p1 - 0.5) * 2
    winner = p1 if prob_p1 > 0.5 else p2
    
    if confidence >= 0.65:
        rec = f"🔥 STRONG {winner}"
    elif confidence >= 0.55:
        rec = f"✅ GOOD {winner}"
    else:
        rec = f"⚪ AVOID {winner}"
    
    g1 = glicko.get_player(p1)
    g2 = glicko.get_player(p2)
    
    return {
        'Jogador1': p1,
        'Jogador2': p2,
        'Rating1': int(g1.r),
        'Rating2': int(g2.r),
        'Forma1': f"{s1.get('recent_form', 0.5):.0%}",
        'Forma2': f"{s2.get('recent_form', 0.5):.0%}",
        'Prob_P1': f"{prob_p1:.1%}",
        'Prob_P2': f"{prob_p2:.1%}",
        'Vencedor': winner,
        'Confianca': f"{confidence:.1%}",
        'Recomendacao': rec
    }

# ==============================================================================
# 6. MAIN APP
# ==============================================================================
def main():
    st.title("🎾 ATP Predictor v12.0 - ATP + Challenger Real")
    st.markdown("**Gera previsões para torneios ATP Masters, ATP 500 e Challengers**")
    
    # Mostrar torneios disponíveis
    with st.expander("📅 Torneios em andamento (Junho 2025)", expanded=True):
        for category, tournaments in REAL_TOURNAMENTS.items():
            st.markdown(f"**{category}**")
            for t in tournaments:
                st.write(f"  • {t['name']} - {t['surface']} - {t['dates']}")
    
    uploaded_file = st.file_uploader("📁 Upload do seu histórico (Excel/CSV)", type=['xlsx', 'csv'])
    
    if uploaded_file and 'model' not in st.session_state:
        with st.spinner("Processando dados e treinando modelo..."):
            df = load_and_process_data(uploaded_file)
            
            if df is not None and len(df) > 0:
                st.info(f"📊 {len(df)} jogos | {len(set(df['winner']) | set(df['loser']))} jogadores")
                
                glicko = train_glicko(df)
                player_stats = calculate_player_stats(df)
                h2h = calculate_h2h(df)
                model = train_model(df, player_stats, h2h, glicko)
                
                if model:
                    st.session_state.model = model
                    st.session_state.glicko = glicko
                    st.session_state.player_stats = player_stats
                    st.session_state.h2h = h2h
                    st.session_state.models_ready = True
                    st.success("✅ Modelo treinado com sucesso!")
                    
                    with st.expander("📊 Top Ratings Glicko"):
                        ratings = [(p, glicko.get_player(p).r, player_stats[p]['win_rate']) 
                                  for p in player_stats.keys()]
                        ratings.sort(key=lambda x: x[1], reverse=True)
                        for i, (p, r, wr) in enumerate(ratings[:20]):
                            st.write(f"{i+1}. {p}: {r:.0f} (WR: {wr:.0%})")
    
    if st.session_state.get('models_ready'):
        st.subheader("🎯 GERAR PREVISÕES")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            if st.button("🎾 Todos os Torneios", use_container_width=True, type="primary"):
                matches = generate_realistic_matches(st.session_state.player_stats, st.session_state.glicko)
                st.session_state.current_matches = matches
                st.success(f"✅ {len(matches)} partidas geradas!")
        
        with col2:
            if st.button("🏆 Só ATP Masters/500", use_container_width=True):
                matches = generate_realistic_matches(st.session_state.player_stats, st.session_state.glicko)
                atp_matches = [m for m in matches if "Masters" in m.get('category', '') or "ATP 500" in m.get('category', '')]
                st.session_state.current_matches = atp_matches[:20]
                st.success(f"✅ {len(st.session_state.current_matches)} partidas ATP")
        
        with col3:
            if st.button("🎯 Só Challenger", use_container_width=True):
                matches = generate_realistic_matches(st.session_state.player_stats, st.session_state.glicko)
                challenger_matches = [m for m in matches if "Challenger" in m.get('category', '')]
                st.session_state.current_matches = challenger_matches[:20]
                st.success(f"✅ {len(st.session_state.current_matches)} partidas Challenger")
        
        # Input manual
        with st.expander("✏️ Ou insira partidas manualmente"):
            col_a, col_b, col_c, col_d = st.columns(4)
            with col_a:
                manual_p1 = st.text_input("Jogador 1", placeholder="Ex: Mitchell Krueger")
            with col_b:
                manual_p2 = st.text_input("Jogador 2", placeholder="Ex: Rio Noguchi")
            with col_c:
                manual_surface = st.selectbox("Superfície", ["Hard", "Clay", "Grass"])
            with col_d:
                manual_tourney = st.text_input("Torneio", "Challenger")
            
            if st.button("🔮 Prever Partida Manual", type="primary") and manual_p1 and manual_p2:
                match = {
                    'tournament': manual_tourney,
                    'player1': manual_p1,
                    'player2': manual_p2,
                    'surface': manual_surface
                }
                st.session_state.current_matches = [match]
        
        # Mostrar previsões
        if st.session_state.get('current_matches'):
            st.subheader(f"📋 {len(st.session_state.current_matches)} Partidas")
            
            results = []
            for match in st.session_state.current_matches:
                pred = predict_match(
                    st.session_state.model,
                    match['player1'], match['player2'],
                    match['surface'],
                    st.session_state.player_stats,
                    st.session_state.h2h,
                    st.session_state.glicko
                )
                pred['Torneio'] = match['tournament']
                pred['Superficie'] = match['surface']
                results.append(pred)
            
            if results:
                df_results = pd.DataFrame(results)
                cols = ['Torneio', 'Superficie', 'Jogador1', 'Jogador2', 'Rating1', 'Rating2',
                       'Forma1', 'Forma2', 'Prob_P1', 'Prob_P2', 'Vencedor', 'Confianca', 'Recomendacao']
                df_results = df_results[[c for c in cols if c in df_results.columns]]
                
                st.dataframe(df_results.style.format({
                    'Rating1': '{:.0f}', 'Rating2': '{:.0f}'
                }), use_container_width=True, hide_index=True)
                
                # Resumo
                st.subheader("📊 Resumo")
                col_a, col_b, col_c, col_d = st.columns(4)
                with col_a:
                    strong = sum(1 for r in results if 'STRONG' in r['Recomendacao'])
                    st.metric("🔥 STRONG", strong)
                with col_b:
                    good = sum(1 for r in results if 'GOOD' in r['Recomendacao'])
                    st.metric("✅ GOOD", good)
                with col_c:
                    confs = [float(r['Confianca'].replace('%', '')) for r in results]
                    st.metric("Confiança Média", f"{sum(confs)/len(confs):.1f}%")
                with col_d:
                    st.metric("Total", len(results))
                
                # Download
                buffer = io.BytesIO()
                df_results.to_excel(buffer, index=False)
                st.download_button("📥 Download Excel", buffer.getvalue(),
                                 f"previsoes_{datetime.now().strftime('%Y%m%d_%H%M')}.xlsx")

if __name__ == "__main__":
    main()

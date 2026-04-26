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
from bs4 import BeautifulSoup
import random

warnings.filterwarnings('ignore')

st.set_page_config(page_title="🎾 ATP Predictor v11.0 - Challenger Focus", page_icon="🎾", layout="wide")

# ==============================================================================
# CONFIGURAÇÕES
# ==============================================================================
WINNER_SMOOTH = 0.65  # Reduzido para permitir mais variação
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
        
        # Ajuste por superfície
        adjusted_E = np.clip(E_ij, 0.01, 0.99)
        
        # Atualização do rating
        k = 30 * surface_factor
        player.r += k * (outcome - adjusted_E)
        
        # Atualizar RD
        player.rd = math.sqrt(player.rd ** 2 + player.sigma ** 2)
        player.rd = min(350, max(30, player.rd))
        
        # Contagem de partidas
        player.matches_played += 1
        if outcome == 1:
            player.wins += 1
        else:
            player.losses += 1

# ==============================================================================
# 2. LISTA DE TORNEIOS CHALLENGER 2025
# ==============================================================================
CHALLENGER_TOURNAMENTS = [
    # USA Challengers
    {"name": "Challenger Tyler", "location": "Tyler, USA", "surface": "Hard", "dates": "June 2025"},
    {"name": "Challenger Little Rock", "location": "Little Rock, USA", "surface": "Hard", "dates": "June 2025"},
    {"name": "Challenger Tallahassee", "location": "Tallahassee, USA", "surface": "Clay", "dates": "April 2025"},
    {"name": "Challenger Savannah", "location": "Savannah, USA", "surface": "Clay", "dates": "April 2025"},
    {"name": "Challenger Sarasota", "location": "Sarasota, USA", "surface": "Clay", "dates": "April 2025"},
    
    # European Challengers
    {"name": "Challenger Oeiras", "location": "Oeiras, Portugal", "surface": "Clay", "dates": "May 2025"},
    {"name": "Challenger Bordeaux", "location": "Bordeaux, France", "surface": "Clay", "dates": "May 2025"},
    {"name": "Challenger Prague", "location": "Prague, Czech Republic", "surface": "Clay", "dates": "May 2025"},
    {"name": "Challenger Heilbronn", "location": "Heilbronn, Germany", "surface": "Clay", "dates": "June 2025"},
    {"name": "Challenger Francavilla", "location": "Francavilla, Italy", "surface": "Clay", "dates": "May 2025"},
    {"name": "Challenger Mestre", "location": "Mestre, Italy", "surface": "Clay", "dates": "June 2025"},
    {"name": "Challenger Skopje", "location": "Skopje, North Macedonia", "surface": "Clay", "dates": "May 2025"},
    
    # Asian Challengers
    {"name": "Challenger Taipei", "location": "Taipei, Taiwan", "surface": "Hard", "dates": "May 2025"},
    {"name": "Challenger Busan", "location": "Busan, South Korea", "surface": "Hard", "dates": "May 2025"},
    {"name": "Challenger Guangzhou", "location": "Guangzhou, China", "surface": "Hard", "dates": "May 2025"},
    {"name": "Challenger Shenzhen", "location": "Shenzhen, China", "surface": "Hard", "dates": "May 2025"},
    {"name": "Challenger Wuxi", "location": "Wuxi, China", "surface": "Hard", "dates": "May 2025"},
    {"name": "Challenger Gwangju", "location": "Gwangju, South Korea", "surface": "Hard", "dates": "May 2025"},
    
    # South American Challengers
    {"name": "Challenger Mexico City", "location": "Mexico City, Mexico", "surface": "Clay", "dates": "April 2025"},
    {"name": "Challenger Santos", "location": "Santos, Brazil", "surface": "Clay", "dates": "May 2025"},
    {"name": "Challenger Buenos Aires", "location": "Buenos Aires, Argentina", "surface": "Clay", "dates": "April 2025"},
    {"name": "Challenger Santiago", "location": "Santiago, Chile", "surface": "Clay", "dates": "March 2025"},
]

# ==============================================================================
# 3. JOGADORES DO SEU HISTÓRICO (Organizados por nível aproximado)
# ==============================================================================
HISTORICAL_PLAYERS_LIST = [
    "Mitchell Krueger", "Trevor Svajda", "Yuta Shimizu", "Antoine Escoffier", "Andres Martin",
    "Rio Noguchi", "Nicolas Mejia", "Paul Jubb", "Stefan Dostanic", "Ilya Ivashka",
    "Rafael Jodar", "Patrick Kypson", "Alex Rybakov", "Karue Sell", "Yibing Wu",
    "Yi Zhou", "Emilio Nava", "Francesco Passaro", "Sumit Nagal", "Marko Topo",
    "Ignacio Buse", "Alejandro Tabilo", "Marco Trungelliti", "Alexander Blockx", "Liam Draxl"
]

# ==============================================================================
# 4. GERAR MATCHES REALISTAS DE CHALLENGER
# ==============================================================================
def generate_challenger_matches(player_stats):
    """Gera matches realistas de Challenger baseados nos jogadores do histórico"""
    matches = []
    
    # Filtrar jogadores com pelo menos 5 partidas no histórico
    active_players = [p for p in HISTORICAL_PLAYERS_LIST if p in player_stats and player_stats[p]['matches'] >= 3]
    
    if len(active_players) < 4:
        active_players = [p for p in HISTORICAL_PLAYERS_LIST if p in player_stats]
    
    if len(active_players) < 2:
        active_players = HISTORICAL_PLAYERS_LIST
    
    # Para cada torneio Challenger, gerar matches
    for tournament in CHALLENGER_TOURNAMENTS[:10]:  # Limitar a 10 torneios
        # Embaralhar jogadores
        players_copy = active_players.copy()
        random.shuffle(players_copy)
        
        # Criar matches baseados em ranking aproximado (winners vs losers baseado no rating)
        tournament_matches = []
        
        # Separar jogadores por força (baseado no win_rate do histórico)
        players_with_rating = []
        for p in players_copy[:20]:  # Pegar até 20 jogadores por torneio
            if p in player_stats:
                rating = player_stats[p]['win_rate'] * 100
            else:
                rating = 50
            players_with_rating.append((p, rating))
        
        players_with_rating.sort(key=lambda x: x[1], reverse=True)
        
        # Criar matches: favorito vs underdog para simular chaves de torneio
        n = len(players_with_rating)
        for i in range(0, n-1, 2):
            if i+1 < n:
                p1 = players_with_rating[i][0]
                p2 = players_with_rating[i+1][0]
                
                # Calcular probabilidade baseada no histórico
                if p1 in player_stats and p2 in player_stats:
                    p1_rate = player_stats[p1]['win_rate']
                    p2_rate = player_stats[p2]['win_rate']
                    # Probabilidade baseada na diferença de win rate
                    prob_p1 = 0.5 + (p1_rate - p2_rate)
                    prob_p1 = np.clip(prob_p1, 0.3, 0.7)
                else:
                    prob_p1 = 0.5
                
                tournament_matches.append({
                    'tournament': tournament['name'],
                    'player1': p1,
                    'player2': p2,
                    'surface': tournament['surface'],
                    'prob_p1': prob_p1
                })
        
        # Adicionar alguns matches do torneio
        matches.extend(tournament_matches[:5])  # 5 matches por torneio
    
    # Embaralhar e limitar
    random.shuffle(matches)
    return matches[:30]

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
        # Tentar com nomes em português
        for col in df.columns:
            if 'vencedor' in col:
                winner_col = col
            elif 'perdedor' in col:
                loser_col = col
    
    if not winner_col or not loser_col:
        st.error(f"Colunas não encontradas. Colunas disponíveis: {list(df.columns)}")
        return None
    
    df = df.rename(columns={winner_col: 'winner', loser_col: 'loser'})
    
    # Data
    if 'date' not in df.columns and 'tourney_date' in df.columns:
        df['date'] = pd.to_datetime(df['tourney_date'], errors='coerce')
    elif 'date' not in df.columns:
        df['date'] = pd.Timestamp.now()
    else:
        df['date'] = pd.to_datetime(df['date'], errors='coerce')
    
    # Total games
    if 'total_games' not in df.columns:
        df['total_games'] = 22
    
    # Surface
    if 'surface' not in df.columns:
        df['surface'] = 'Hard'
    
    # Clean names
    df['winner'] = df['winner'].astype(str).str.strip()
    df['loser'] = df['loser'].astype(str).str.strip()
    
    # Remover linhas vazias
    df = df[df['winner'].notna() & df['loser'].notna()]
    df = df[df['winner'] != 'nan']
    df = df[df['loser'] != 'nan']
    
    return df

def calculate_player_stats(df):
    """Calcula estatísticas detalhadas dos jogadores"""
    player_stats = {}
    
    for player in set(df['winner'].unique()) | set(df['loser'].unique()):
        matches = df[(df['winner'] == player) | (df['loser'] == player)]
        
        if len(matches) == 0:
            continue
        
        wins = len(matches[matches['winner'] == player])
        total = len(matches)
        win_rate = wins / total if total > 0 else 0.5
        
        # Forma recente (últimos 10)
        recent = matches.sort_values('date', ascending=False).head(10)
        recent_wins = len(recent[recent['winner'] == player])
        recent_form = recent_wins / len(recent) if len(recent) > 0 else 0.5
        
        # Forma muito recente (últimos 3)
        very_recent = matches.sort_values('date', ascending=False).head(3)
        very_recent_wins = len(very_recent[very_recent['winner'] == player])
        very_recent_form = very_recent_wins / len(very_recent) if len(very_recent) > 0 else 0.5
        
        # Estatísticas por superfície
        surface_stats = {}
        for surf in ['Hard', 'Clay', 'Grass']:
            surf_matches = matches[matches['surface'] == surf]
            if len(surf_matches) > 0:
                surf_wins = len(surf_matches[surf_matches['winner'] == player])
                surface_stats[surf] = surf_wins / len(surf_matches)
            else:
                surface_stats[surf] = 0.5
        
        # Média de games
        avg_games = matches['total_games'].mean() if 'total_games' in matches.columns else 22
        
        player_stats[player] = {
            'matches': total,
            'wins': wins,
            'losses': total - wins,
            'win_rate': win_rate,
            'recent_form': recent_form,
            'very_recent_form': very_recent_form,
            'hard_rate': surface_stats['Hard'],
            'clay_rate': surface_stats['Clay'],
            'grass_rate': surface_stats['Grass'],
            'avg_games': avg_games
        }
    
    return player_stats

def calculate_h2h(df):
    """Calcula histórico de confrontos diretos"""
    h2h = defaultdict(lambda: {'wins': 0, 'total': 0, 'surface_wins': defaultdict(int)})
    
    for _, row in df.iterrows():
        w, l = row['winner'], row['loser']
        surface = row.get('surface', 'Hard')
        
        h2h[(w, l)]['wins'] += 1
        h2h[(w, l)]['total'] += 1
        h2h[(w, l)]['surface_wins'][surface] += 1
    
    return h2h

def train_glicko(df):
    """Treina o sistema Glicko"""
    glicko = GlickoSystem()
    
    for _, row in df.iterrows():
        winner = row['winner']
        loser = row['loser']
        surface = row.get('surface', 'Hard')
        
        surface_factor = 1.0
        if surface == 'Clay':
            surface_factor = 1.03
        elif surface == 'Grass':
            surface_factor = 0.97
        
        winner_obj = glicko.get_player(winner)
        loser_obj = glicko.get_player(loser)
        
        # Atualizar vencedor
        glicko.update_player(winner_obj, loser_obj, 1.0, surface_factor)
        # Atualizar perdedor
        glicko.update_player(loser_obj, winner_obj, 0.0, surface_factor)
    
    return glicko

def build_features(p1, p2, surface, player_stats, h2h, glicko):
    """Constrói features para predição"""
    
    s1 = player_stats.get(p1, {})
    s2 = player_stats.get(p2, {})
    
    if not s1 or not s2:
        return None
    
    # Ratings Glicko
    g1 = glicko.get_player(p1)
    g2 = glicko.get_player(p2)
    
    # Features base
    elo_diff = (g1.r - g2.r) / 400
    rd_diff = (g2.rd - g1.rd) / 350
    
    # Forma
    form_diff = s1.get('recent_form', 0.5) - s2.get('recent_form', 0.5)
    very_recent_diff = s1.get('very_recent_form', 0.5) - s2.get('very_recent_form', 0.5)
    
    # Win rates
    total_win_diff = s1.get('win_rate', 0.5) - s2.get('win_rate', 0.5)
    
    # Win rate por superfície
    if surface == 'Clay':
        surf_rate1 = s1.get('clay_rate', 0.5)
        surf_rate2 = s2.get('clay_rate', 0.5)
    elif surface == 'Grass':
        surf_rate1 = s1.get('grass_rate', 0.5)
        surf_rate2 = s2.get('grass_rate', 0.5)
    else:
        surf_rate1 = s1.get('hard_rate', 0.5)
        surf_rate2 = s2.get('hard_rate', 0.5)
    
    surf_diff = surf_rate1 - surf_rate2
    
    # H2H
    h2h_adv = 0.5
    if (p1, p2) in h2h:
        h2h_adv = h2h[(p1, p2)]['wins'] / max(1, h2h[(p1, p2)]['total'])
    elif (p2, p1) in h2h:
        h2h_adv = 1 - (h2h[(p2, p1)]['wins'] / max(1, h2h[(p2, p1)]['total']))
    
    # Experiência
    exp_diff = (s1.get('matches', 0) - s2.get('matches', 0)) / 100
    
    # Games
    games_avg = (s1.get('avg_games', 22) + s2.get('avg_games', 22)) / 2
    games_norm = (games_avg - 21.5) / 10
    
    # Momentum
    momentum = very_recent_diff * 0.5 + form_diff * 0.3
    
    features = [
        elo_diff, rd_diff, form_diff, very_recent_diff,
        total_win_diff, surf_diff, h2h_adv, exp_diff, games_norm, momentum
    ]
    
    return features

def train_model(df, player_stats, h2h, glicko):
    """Treina o modelo LightGBM"""
    
    X, y = [], []
    
    for _, row in df.iterrows():
        p1 = row['winner']
        p2 = row['loser']
        surface = row.get('surface', 'Hard')
        
        # Features para o vencedor
        features = build_features(p1, p2, surface, player_stats, h2h, glicko)
        if features:
            X.append(features)
            y.append(1)
        
        # Features para o perdedor (invertido)
        features_rev = build_features(p2, p1, surface, player_stats, h2h, glicko)
        if features_rev:
            X.append(features_rev)
            y.append(0)
    
    if len(X) == 0:
        return None
    
    X = np.array(X)
    
    model = LGBMClassifier(
        n_estimators=200,
        max_depth=6,
        learning_rate=0.03,
        num_leaves=20,
        reg_alpha=0.5,
        reg_lambda=0.5,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        verbose=-1
    )
    
    model.fit(X, y)
    return model

def predict_match(model, p1, p2, surface, player_stats, h2h, glicko):
    """Prediz uma partida"""
    
    features = build_features(p1, p2, surface, player_stats, h2h, glicko)
    
    if features is None:
        return None, f"Jogador não encontrado no histórico"
    
    features = np.array([features])
    
    # Probabilidade do modelo
    prob = model.predict_proba(features)[0][1]
    
    # Calibração
    prob_p1 = np.clip(prob, 0.25, 0.75)
    prob_p2 = 1 - prob_p1
    
    # Confiança (baseada na diferença de probabilidade)
    confidence = abs(prob_p1 - 0.5) * 2
    
    winner = p1 if prob_p1 > 0.5 else p2
    
    # Recomendação
    if confidence >= MIN_CONFIDENCE_STRONG:
        rec = f"🔥 STRONG {winner}"
    elif confidence >= MIN_CONFIDENCE_GOOD:
        rec = f"✅ GOOD {winner}"
    else:
        rec = f"⚪ AVOID {winner}"
    
    # Estatísticas para display
    s1 = player_stats.get(p1, {})
    s2 = player_stats.get(p2, {})
    g1 = glicko.get_player(p1)
    g2 = glicko.get_player(p2)
    
    return {
        'Jogador1': p1,
        'Jogador2': p2,
        'Rating1': f"{g1.r:.0f}",
        'Rating2': f"{g2.r:.0f}",
        'Forma1': f"{s1.get('recent_form', 0.5):.0%}",
        'Forma2': f"{s2.get('recent_form', 0.5):.0%}",
        'Prob_P1': f"{prob_p1:.1%}",
        'Prob_P2': f"{prob_p2:.1%}",
        'Vencedor': winner,
        'Confianca': f"{confidence:.1%}",
        'Recomendacao': rec,
        'H2H': f"{h2h.get((p1,p2), {}).get('wins', 0)}-{h2h.get((p2,p1), {}).get('wins', 0)}"
    }, None

# ==============================================================================
# 6. FUNÇÃO PRINCIPAL
# ==============================================================================
def get_challenger_matches(player_stats):
    """Gera matches de Challenger"""
    return generate_challenger_matches(player_stats)

# ==============================================================================
# 7. MAIN APP
# ==============================================================================
def main():
    st.title("🎾 ATP Predictor v11.0 - Challenger Focus")
    st.markdown("**Sistema de Rating Glicko + LightGBM para Challengers e ATP**")
    
    uploaded_file = st.file_uploader("📁 Upload do seu histórico (Excel/CSV)", type=['xlsx', 'csv'])
    
    if uploaded_file and 'model' not in st.session_state:
        with st.spinner("Processando dados e treinando modelo..."):
            df = load_and_process_data(uploaded_file)
            
            if df is not None and len(df) > 0:
                st.info(f"📊 Dataset: {len(df)} jogos | {len(set(df['winner']) | set(df['loser']))} jogadores")
                
                # Treinar Glicko
                glicko = train_glicko(df)
                
                # Calcular estatísticas
                player_stats = calculate_player_stats(df)
                h2h = calculate_h2h(df)
                
                # Treinar modelo
                model = train_model(df, player_stats, h2h, glicko)
                
                if model:
                    st.session_state.model = model
                    st.session_state.glicko = glicko
                    st.session_state.player_stats = player_stats
                    st.session_state.h2h = h2h
                    st.session_state.models_ready = True
                    st.success(f"✅ Modelo treinado com {len(player_stats)} jogadores!")
                    
                    # Mostrar top jogadores
                    with st.expander("📊 Top 20 Jogadores (Rating Glicko)"):
                        ratings = [(p, glicko.get_player(p).r, player_stats[p]['win_rate']) 
                                  for p in player_stats.keys()]
                        ratings.sort(key=lambda x: x[1], reverse=True)
                        for i, (player, rating, wr) in enumerate(ratings[:20]):
                            st.write(f"{i+1}. {player}: {rating:.0f} (WR: {wr:.0%})")
    
    if st.session_state.get('models_ready'):
        st.subheader("🎯 PREVISÕES PARA CHALLENGER")
        
        col1, col2 = st.columns(2)
        with col1:
            if st.button("🎾 Gerar Matches Challenger", use_container_width=True, type="primary"):
                matches = get_challenger_matches(st.session_state.player_stats)
                st.session_state.current_matches = matches
                st.success(f"✅ {len(matches)} partidas geradas!")
        
        with col2:
            if st.button("📝 Inserir Manualmente", use_container_width=True):
                st.session_state.show_manual = not st.session_state.get('show_manual', False)
        
        # Input manual
        if st.session_state.get('show_manual', False):
            with st.expander("✏️ Inserir Partidas", expanded=True):
                num = st.number_input("Número de partidas", 1, 10, 3)
                manual_matches = []
                
                for i in range(num):
                    st.markdown(f"**Partida {i+1}**")
                    cols = st.columns(4)
                    with cols[0]:
                        p1 = st.text_input(f"Jogador 1", key=f"m_p1_{i}")
                    with cols[1]:
                        p2 = st.text_input(f"Jogador 2", key=f"m_p2_{i}")
                    with cols[2]:
                        surf = st.selectbox("Superfície", ["Hard", "Clay", "Grass"], key=f"m_surf_{i}")
                    with cols[3]:
                        tourney = st.text_input("Torneio", "Challenger", key=f"m_tourney_{i}")
                    
                    if p1 and p2:
                        manual_matches.append({
                            'tournament': tourney,
                            'player1': p1,
                            'player2': p2,
                            'surface': surf
                        })
                    st.markdown("---")
                
                if st.button("Prever", type="primary") and manual_matches:
                    st.session_state.current_matches = manual_matches
        
        # Mostrar previsões
        if st.session_state.get('current_matches'):
            st.subheader(f"📋 {len(st.session_state.current_matches)} Partidas")
            
            results = []
            not_found = []
            
            for match in st.session_state.current_matches:
                pred, error = predict_match(
                    st.session_state.model,
                    match['player1'], match['player2'],
                    match['surface'],
                    st.session_state.player_stats,
                    st.session_state.h2h,
                    st.session_state.glicko
                )
                
                if pred:
                    pred['Torneio'] = match['tournament']
                    pred['Superficie'] = match['surface']
                    results.append(pred)
                elif error:
                    not_found.append(error)
            
            if not_found:
                st.warning(f"⚠️ {len(set(not_found))} jogadores não encontrados")
                for err in set(not_found):
                    st.write(f"• {err}")
            
            if results:
                df_results = pd.DataFrame(results)
                cols = ['Torneio', 'Superficie', 'Jogador1', 'Jogador2', 'Rating1', 'Rating2',
                       'Forma1', 'Forma2', 'H2H', 'Prob_P1', 'Prob_P2', 'Vencedor', 
                       'Confianca', 'Recomendacao']
                df_results = df_results[[c for c in cols if c in df_results.columns]]
                
                st.dataframe(df_results, use_container_width=True, hide_index=True)
                
                # Resumo
                col_s1, col_s2, col_s3 = st.columns(3)
                with col_s1:
                    strong = sum(1 for r in results if 'STRONG' in r['Recomendacao'])
                    st.metric("🔥 STRONG", strong)
                with col_s2:
                    avg_conf = sum(float(r['Confianca'].replace('%', '')) for r in results) / len(results)
                    st.metric("Confiança Média", f"{avg_conf:.1f}%")
                with col_s3:
                    st.metric("Total", len(results))
                
                # Download
                buffer = io.BytesIO()
                df_results.to_excel(buffer, index=False)
                st.download_button("📥 Download Excel", buffer.getvalue(),
                                 f"previsoes_{datetime.now().strftime('%Y%m%d_%H%M')}.xlsx")

if __name__ == "__main__":
    main()

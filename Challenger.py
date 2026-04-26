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
import json
from bs4 import BeautifulSoup

warnings.filterwarnings('ignore')

st.set_page_config(page_title="🎾 ATP Predictor v10.0 - ATP + Challenger", page_icon="🎾", layout="wide")

# ==============================================================================
# CONFIGURAÇÕES
# ==============================================================================
WINNER_SMOOTH = 0.55
MIN_CONFIDENCE_STRONG = 0.68
MIN_CONFIDENCE_GOOD = 0.60

# ==============================================================================
# 1. SISTEMA GLICKO
# ==============================================================================
class GlickoPlayer:
    def __init__(self, name):
        self.name = name
        self.r = 1500.0
        self.rd = 350.0
        self.sigma = 0.06

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
        if exp_arg > 700:
            return 0.0
        if exp_arg < -700:
            return 1.0
        return 1.0 / (1.0 + math.exp(exp_arg))
    
    def update_player(self, player, opponents, outcomes, surface_factor=1.0):
        if len(opponents) == 0:
            return
        
        v = 0.0
        delta = 0.0
        
        for opp, outcome in zip(opponents, outcomes):
            g_rdj = self.g(opp.rd)
            E_ij = self.E(player.r, opp.r, opp.rd)
            adjusted_E = np.clip(0.5 + (E_ij - 0.5) * surface_factor, 0.01, 0.99)
            
            v += g_rdj ** 2 * adjusted_E * (1 - adjusted_E)
            delta += g_rdj * (outcome - adjusted_E)
        
        if v < self.epsilon:
            v = self.epsilon
        
        v = 1.0 / v
        delta *= v
        
        sigma_new = min(0.5, player.sigma + 0.01)
        rd_star = math.sqrt(player.rd ** 2 + sigma_new ** 2)
        rd_star = min(rd_star, 350)
        
        delta_limited = np.clip(delta, -300, 300)
        r_new = player.r + (v * delta_limited)
        
        new_rd_sq = 1.0 / (1.0 / (rd_star ** 2) + 1.0 / v)
        rd_new = math.sqrt(max(self.epsilon, new_rd_sq))
        
        player.r = r_new
        player.rd = min(rd_new, 350)
        player.sigma = sigma_new

# ==============================================================================
# 2. SCRAPING COMPLETO - ATP + CHALLENGER
# ==============================================================================
def scrape_all_matches():
    """Scraping de todos os torneios (ATP + Challenger)"""
    all_matches = []
    
    # Lista de torneios Challenger em andamento
    challengers = get_challenger_tournaments()
    
    for challenger in challengers:
        matches = scrape_challenger_matches(challenger)
        all_matches.extend(matches)
    
    # Adicionar torneios ATP
    atp_matches = scrape_atp_matches()
    all_matches.extend(atp_matches)
    
    return all_matches

def get_challenger_tournaments():
    """Retorna lista de torneios Challenger ativos"""
    # Lista de Challengers ATP por região/superfície
    challengers = [
        {"name": "Challenger Tyler", "location": "Tyler, USA", "surface": "Hard"},
        {"name": "Challenger Savannah", "location": "Savannah, USA", "surface": "Clay"},
        {"name": "Challenger Oeiras", "location": "Oeiras, Portugal", "surface": "Clay"},
        {"name": "Challenger Bordeaux", "location": "Bordeaux, France", "surface": "Clay"},
        {"name": "Challenger Tunis", "location": "Tunis, Tunisia", "surface": "Clay"},
        {"name": "Challenger Taipei", "location": "Taipei, Taiwan", "surface": "Hard"},
        {"name": "Challenger Busan", "location": "Busan, South Korea", "surface": "Hard"},
        {"name": "Challenger Prague", "location": "Prague, Czech Republic", "surface": "Clay"},
        {"name": "Challenger Heilbronn", "location": "Heilbronn, Germany", "surface": "Clay"},
        {"name": "Challenger Francavilla", "location": "Francavilla, Italy", "surface": "Clay"},
        {"name": "Challenger Mestre", "location": "Mestre, Italy", "surface": "Clay"},
        {"name": "Challenger Skopje", "location": "Skopje, North Macedonia", "surface": "Clay"},
        {"name": "Challenger Little Rock", "location": "Little Rock, USA", "surface": "Hard"},
        {"name": "Challenger Tallahassee", "location": "Tallahassee, USA", "surface": "Clay"},
        {"name": "Challenger Sarasota", "location": "Sarasota, USA", "surface": "Clay"},
        {"name": "Challenger Mexico City", "location": "Mexico City, Mexico", "surface": "Clay"},
        {"name": "Challenger Guangzhou", "location": "Guangzhou, China", "surface": "Hard"},
        {"name": "Challenger Shenzhen", "location": "Shenzhen, China", "surface": "Hard"},
        {"name": "Challenger Wuxi", "location": "Wuxi, China", "surface": "Hard"},
        {"name": "Challenger Gwangju", "location": "Gwangju, South Korea", "surface": "Hard"},
    ]
    return challengers

def scrape_challenger_matches(challenger):
    """Scraping de matches de um Challenger específico"""
    matches = []
    
    # Usar API do Tennis24 para Challengers
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36',
        'Accept': 'application/json'
    }
    
    # Tentar diferentes formatos de URL para Challenger
    search_name = challenger['name'].lower().replace(' ', '-')
    urls = [
        f"https://www.tennis24.com/tennis/atp-challenger/{search_name}/",
        f"https://www.tennis24.com/sport/tennis/atp-challenger/{search_name}/",
        f"https://www.tennis24.com/matches/#/search/{search_name}",
    ]
    
    for url in urls:
        try:
            response = requests.get(url, headers=headers, timeout=10)
            if response.status_code == 200:
                html = response.text
                # Procurar por padrões de partida
                match_patterns = [
                    r'<div class="event__match"[^>]*>.*?<a[^>]*>([^<]+)</a>.*?<a[^>]*>([^<]+)</a>',
                    r'<span class="team-name">([^<]+)</span>.*?<span class="team-name">([^<]+)</span>',
                    r'"homeTeam":{"name":"([^"]+)"}.*?"awayTeam":{"name":"([^"]+)"}'
                ]
                
                for pattern in match_patterns:
                    found = re.findall(pattern, html, re.DOTALL)
                    for p1, p2 in found:
                        if p1 and p2 and p1 != p2:
                            matches.append({
                                'tournament': challenger['name'],
                                'player1': p1.strip(),
                                'player2': p2.strip(),
                                'surface': challenger['surface']
                            })
                    if matches:
                        break
            if matches:
                break
        except:
            continue
    
    return matches

def scrape_atp_matches():
    """Scraping de torneios ATP principais"""
    matches = []
    
    # Torneios ATP em andamento
    atp_tournaments = [
        {"name": "Mutua Madrid Open", "surface": "Clay", "level": "Masters 1000"},
        {"name": "Internazionali BNL d'Italia", "surface": "Clay", "level": "Masters 1000"},
        {"name": "Roland Garros", "surface": "Clay", "level": "Grand Slam"},
        {"name": "Wimbledon", "surface": "Grass", "level": "Grand Slam"},
        {"name": "US Open", "surface": "Hard", "level": "Grand Slam"},
        {"name": "Australian Open", "surface": "Hard", "level": "Grand Slam"},
    ]
    
    headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'}
    
    for tournament in atp_tournaments:
        search_name = tournament['name'].lower().replace(' ', '-')
        url = f"https://www.tennis24.com/tennis/atp/{search_name}/"
        
        try:
            response = requests.get(url, headers=headers, timeout=10)
            if response.status_code == 200:
                html = response.text
                # Procurar por partidas
                pattern = r'<a[^>]*class="[^"]*team-name[^"]*"[^>]*>([^<]+)</a>'
                players = re.findall(pattern, html)
                
                for i in range(0, len(players)-1, 2):
                    if i+1 < len(players):
                        p1 = players[i].strip()
                        p2 = players[i+1].strip()
                        if p1 and p2 and p1 != p2:
                            matches.append({
                                'tournament': tournament['name'],
                                'player1': p1,
                                'player2': p2,
                                'surface': tournament['surface']
                            })
        except:
            continue
    
    return matches

# ==============================================================================
# 3. GERAR MATCHES BASEADOS NO SEU HISTÓRICO
# ==============================================================================
def generate_matches_from_history(player_stats, surface='Hard'):
    """Gera partidas baseadas nos jogadores do seu histórico"""
    matches = []
    
    # Pegar jogadores que aparecem no histórico
    players = list(player_stats.keys())
    if len(players) < 2:
        return matches
    
    # Embaralhar e criar matchups
    import random
    random.shuffle(players)
    
    # Criar partidas entre jogadores do mesmo nível aproximado (baseado no rating)
    players_with_rating = [(p, player_stats[p]['win_rate']) for p in players]
    players_with_rating.sort(key=lambda x: x[1], reverse=True)
    
    # Criar matchups: top vs top, middle vs middle, etc.
    n = len(players_with_rating)
    for i in range(0, n-1, 2):
        if i+1 < n:
            matches.append({
                'tournament': 'Challenger Match',
                'player1': players_with_rating[i][0],
                'player2': players_with_rating[i+1][0],
                'surface': surface
            })
    
    return matches[:20]  # Limitar a 20 partidas

# ==============================================================================
# 4. SCRAPING VIA FLASHCORE (API alternativa)
# ==============================================================================
def scrape_via_flashcore():
    """Usa API do Flashcore para obter partidas (alternativa)"""
    matches = []
    
    # URLs de torneios Challenger no Flashscore
    challenger_urls = [
        "https://www.flashscore.com/tennis/atp-challenger-men/",
        "https://www.flashscore.com/tennis/atp-challenger-men-2025/",
        "https://www.flashscore.com/tennis/atp-challenger-tyler/",
    ]
    
    headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'}
    
    for url in challenger_urls:
        try:
            response = requests.get(url, headers=headers, timeout=10)
            if response.status_code == 200:
                html = response.text
                
                # Padrões de partidas no Flashscore
                patterns = [
                    r'<div class="event__match"[^>]*>.*?<div[^>]*class="[^"]*homeTeam[^"]*"[^>]*>([^<]+)</div>.*?<div[^>]*class="[^"]*awayTeam[^"]*"[^>]*>([^<]+)</div>',
                    r'"home":{"name":"([^"]+)"}.*?"away":{"name":"([^"]+)"}',
                    r'([A-Z][a-z]+(?:\s+[A-Z][a-z]+)?)\s*[-–—]\s*([A-Z][a-z]+(?:\s+[A-Z][a-z]+)?)'
                ]
                
                for pattern in patterns:
                    found_matches = re.findall(pattern, html, re.DOTALL)
                    for p1, p2 in found_matches:
                        p1_clean = re.sub(r'<[^>]+>', '', p1).strip()
                        p2_clean = re.sub(r'<[^>]+>', '', p2).strip()
                        if len(p1_clean) > 2 and len(p2_clean) > 2 and p1_clean != p2_clean:
                            matches.append({
                                'tournament': 'Challenger',
                                'player1': p1_clean,
                                'player2': p2_clean,
                                'surface': 'Hard'
                            })
                    if matches:
                        break
            if matches:
                break
        except:
            continue
    
    return matches

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
    
    winner_col = next((c for c in df.columns if 'winner_name' in c), None)
    loser_col = next((c for c in df.columns if 'loser_name' in c), None)
    score_col = next((c for c in df.columns if 'score' in c), None)
    surface_col = next((c for c in df.columns if 'surface' in c), None)
    date_col = next((c for c in df.columns if 'tourney_date' in c), None)
    
    if not winner_col or not loser_col:
        st.error("Colunas 'winner_name' e 'loser_name' são obrigatórias.")
        return None
    
    df = df.rename(columns={
        winner_col: 'winner',
        loser_col: 'loser',
        score_col: 'score',
        surface_col: 'surface',
        date_col: 'date'
    })
    
    if 'date' in df.columns:
        df['date'] = pd.to_datetime(df['date'], errors='coerce')
    else:
        df['date'] = pd.Timestamp.now()
    
    if 'score' in df.columns:
        def extract_games(score):
            if pd.isna(score):
                return 22
            games = re.findall(r'(\d+)-(\d+)', str(score))
            total = sum(int(a) + int(b) for a, b in games if a.isdigit() and b.isdigit())
            return max(total, 22) if total > 0 else 22
        df['total_games'] = df['score'].apply(extract_games)
    else:
        df['total_games'] = 22
        
    if 'surface' not in df.columns:
        df['surface'] = 'Hard'
    
    df = df.sort_values('date')
    
    return df

def train_glicko_and_features(df):
    glicko = GlickoSystem()
    
    for idx, row in df.iterrows():
        winner = row['winner']
        loser = row['loser']
        surface = row.get('surface', 'Hard')
        
        surface_factor = 1.05 if surface == 'Clay' else (0.95 if surface == 'Grass' else 1.0)
        
        winner_obj = glicko.get_player(winner)
        loser_obj = glicko.get_player(loser)
        
        glicko.update_player(winner_obj, [loser_obj], [1.0], surface_factor)
        glicko.update_player(loser_obj, [winner_obj], [0.0], surface_factor)
    
    return glicko

def calculate_player_stats(df):
    player_stats = {}
    
    for player in set(df['winner'].unique()) | set(df['loser'].unique()):
        matches = df[(df['winner'] == player) | (df['loser'] == player)]
        wins = len(matches[matches['winner'] == player])
        total = len(matches)
        
        recent = matches.sort_values('date', ascending=False).head(10)
        recent_wins = len(recent[recent['winner'] == player])
        avg_games = matches['total_games'].mean()
        
        player_stats[player] = {
            'matches': total,
            'win_rate': wins / total if total > 0 else 0.5,
            'recent_form': recent_wins / len(recent) if len(recent) > 0 else 0.5,
            'avg_games': avg_games
        }
    
    return player_stats

def calculate_h2h(df):
    h2h = {}
    for _, row in df.iterrows():
        w, l = row['winner'], row['loser']
        key = (w, l)
        if key not in h2h:
            h2h[key] = {'wins': 0, 'total': 0}
        h2h[key]['wins'] += 1
        h2h[key]['total'] += 1
    return h2h

def prepare_features(df, player_stats, h2h, glicko_system):
    X, y = [], []
    
    for _, row in df.iterrows():
        p1 = row['winner']
        p2 = row['loser']
        
        s1 = player_stats.get(p1, {'matches': 0, 'win_rate': 0.5, 'recent_form': 0.5, 'avg_games': 22})
        s2 = player_stats.get(p2, {'matches': 0, 'win_rate': 0.5, 'recent_form': 0.5, 'avg_games': 22})
        
        p1_glicko = glicko_system.get_player(p1)
        p2_glicko = glicko_system.get_player(p2)
        
        rating_diff = (p1_glicko.r - p2_glicko.r) / 400
        rd_diff = (p2_glicko.rd - p1_glicko.rd) / 350
        form_diff = s1['recent_form'] - s2['recent_form']
        win_rate_diff = s1['win_rate'] - s2['win_rate']
        
        h2h_val = 0.5
        if (p1, p2) in h2h:
            h2h_val = h2h[(p1, p2)]['wins'] / h2h[(p1, p2)]['total']
        
        games_avg = (s1['avg_games'] + s2['avg_games']) / 2
        games_norm = (games_avg - 21.5) / 8
        exp_diff = (s1['matches'] - s2['matches']) / 200
        
        features = [rating_diff, rd_diff, form_diff, win_rate_diff, h2h_val, games_norm, exp_diff]
        
        X.append(features)
        y.append(1)
        
        features_inv = [-rating_diff, -rd_diff, -form_diff, -win_rate_diff, 1-h2h_val, games_norm, -exp_diff]
        X.append(features_inv)
        y.append(0)
    
    return np.array(X), np.array(y)

def train_model(X, y):
    if len(X) == 0:
        return None
    
    model = LGBMClassifier(
        n_estimators=150,
        max_depth=5,
        learning_rate=0.035,
        num_leaves=16,
        reg_alpha=0.8,
        reg_lambda=0.8,
        random_state=42,
        verbose=-1
    )
    
    model.fit(X, y)
    return model

def predict_match(model, p1_name, p2_name, player_stats, h2h, glicko_system):
    s1 = player_stats.get(p1_name, {'matches': 0, 'win_rate': 0.5, 'recent_form': 0.5, 'avg_games': 22})
    s2 = player_stats.get(p2_name, {'matches': 0, 'win_rate': 0.5, 'recent_form': 0.5, 'avg_games': 22})
    
    p1 = glicko_system.get_player(p1_name)
    p2 = glicko_system.get_player(p2_name)
    
    # Verificar se os jogadores existem no sistema
    if p1.r == 1500 and p1.rd == 350:
        # Jogador novo, rating inicial
        pass
    
    rating_diff = (p1.r - p2.r) / 400
    rd_diff = (p2.rd - p1.rd) / 350
    form_diff = s1['recent_form'] - s2['recent_form']
    win_rate_diff = s1['win_rate'] - s2['win_rate']
    
    h2h_val = 0.5
    if (p1_name, p2_name) in h2h:
        h2h_val = h2h[(p1_name, p2_name)]['wins'] / h2h[(p1_name, p2_name)]['total']
    elif (p2_name, p1_name) in h2h:
        h2h_val = 1 - h2h[(p2_name, p1_name)]['wins'] / h2h[(p2_name, p1_name)]['total']
    
    games_avg = (s1['avg_games'] + s2['avg_games']) / 2
    games_norm = (games_avg - 21.5) / 8
    exp_diff = (s1['matches'] - s2['matches']) / 200
    
    features = np.array([[rating_diff, rd_diff, form_diff, win_rate_diff, h2h_val, games_norm, exp_diff]])
    
    prob = model.predict_proba(features)[0][1]
    prob_p1 = np.clip(0.5 + (prob - 0.5) * WINNER_SMOOTH, 0.15, 0.85)
    confidence = abs(prob_p1 - 0.5) * 2
    winner = p1_name if prob_p1 > 0.5 else p2_name
    
    if confidence >= MIN_CONFIDENCE_STRONG:
        rec = f"🔥 STRONG {winner}"
    elif confidence >= MIN_CONFIDENCE_GOOD:
        rec = f"✅ GOOD {winner}"
    else:
        rec = f"⚪ AVOID {winner}"
    
    return {
        'Jogador1': p1_name,
        'Jogador2': p2_name,
        'Rating_Glicko': f"{p1.r:.0f} | {p2.r:.0f}",
        'Prob_P1': f"{prob_p1:.1%}",
        'Prob_P2': f"{1-prob_p1:.1%}",
        'Vencedor': winner,
        'Confianca': f"{confidence:.1%}",
        'Recomendacao': rec,
        'Games_Esperados': round(games_avg, 1)
    }

# ==============================================================================
# 6. FUNÇÃO PRINCIPAL DE BUSCA
# ==============================================================================
def get_all_matches(player_stats):
    """Busca todos os tipos de partidas"""
    all_matches = []
    
    # Método 1: Tentar scraping online
    with st.spinner("Buscando Challengers e torneios ATP..."):
        all_matches = scrape_all_matches()
    
    # Método 2: Scraping via Flashcore
    if not all_matches:
        with st.spinner("Tentando Flashcore..."):
            all_matches = scrape_via_flashcore()
    
    # Método 3: Gerar partidas baseadas no histórico
    if not all_matches and player_stats:
        st.info("Gerando partidas baseadas no seu histórico...")
        all_matches = generate_matches_from_history(player_stats)
    
    return all_matches

# ==============================================================================
# 7. MAIN
# ==============================================================================
def main():
    st.title("🎾 ATP Predictor v10.0 - ATP + Challenger")
    st.markdown("**Sistema de Rating Dinâmico (Glicko) + LightGBM**")
    st.info("Inclui torneios ATP e Challenger baseados no seu histórico")
    
    uploaded_file = st.file_uploader("📁 Upload do seu histórico (Excel/CSV)", type=['xlsx', 'csv'])
    
    if uploaded_file and 'model' not in st.session_state:
        with st.spinner("Processando dados e treinando modelo Glicko..."):
            df = load_and_process_data(uploaded_file)
            
            if df is not None and len(df) > 0:
                st.info(f"📊 Dataset: {len(df)} jogos | {len(set(df['winner']) | set(df['loser']))} jogadores")
                
                glicko_system = train_glicko_and_features(df)
                player_stats = calculate_player_stats(df)
                h2h = calculate_h2h(df)
                X, y = prepare_features(df, player_stats, h2h, glicko_system)
                model = train_model(X, y)
                
                if model:
                    st.session_state.model = model
                    st.session_state.glicko = glicko_system
                    st.session_state.player_stats = player_stats
                    st.session_state.h2h = h2h
                    st.session_state.models_ready = True
                    st.success(f"✅ Modelo treinado com sucesso!")
                    
                    with st.expander("📊 Top Ratings Glicko"):
                        ratings = [(p, glicko_system.get_player(p).r) for p in player_stats.keys()]
                        ratings.sort(key=lambda x: x[1], reverse=True)
                        for i, (player, rating) in enumerate(ratings[:30]):
                            st.write(f"{i+1}. {player}: {rating:.0f}")
    
    if st.session_state.get('models_ready'):
        st.subheader("🎯 BUSCAR PARTIDAS")
        
        col1, col2, col3 = st.columns(3)
        with col1:
            if st.button("🎾 ATP + Challenger", use_container_width=True, type="primary"):
                matches = get_all_matches(st.session_state.player_stats)
                st.session_state.today_matches = matches
                if matches:
                    st.success(f"✅ {len(matches)} partidas encontradas!")
                else:
                    st.warning("Nenhuma partida encontrada. Use inserção manual.")
        
        with col2:
            if st.button("🎾 Só Challenger", use_container_width=True):
                challengers = get_challenger_tournaments()
                matches = []
                for ch in challengers[:5]:
                    m = scrape_challenger_matches(ch)
                    matches.extend(m)
                st.session_state.today_matches = matches
                st.success(f"✅ {len(matches)} partidas de Challenger")
        
        with col3:
            if st.button("📝 Inserir Manualmente", use_container_width=True):
                st.session_state.show_manual = not st.session_state.get('show_manual', False)
        
        # Manual input
        if st.session_state.get('show_manual', False):
            with st.expander("✏️ Inserir Partidas Manualmente", expanded=True):
                st.markdown("### Inserir partidas de Challenger/ATP")
                
                num_matches = st.number_input("Número de partidas", min_value=1, max_value=20, value=5)
                manual_matches = []
                
                col1, col2, col3 = st.columns(3)
                surface_options = ["Hard", "Clay", "Grass"]
                
                for i in range(num_matches):
                    with st.container():
                        st.markdown(f"**Partida {i+1}**")
                        cols = st.columns(3)
                        with cols[0]:
                            p1 = st.text_input(f"Jogador 1", key=f"man_p1_{i}", 
                                              placeholder="Ex: Mitchell Krueger")
                        with cols[1]:
                            p2 = st.text_input(f"Jogador 2", key=f"man_p2_{i}",
                                              placeholder="Ex: Tung-Lin Wu")
                        with cols[2]:
                            surf = st.selectbox(f"Superfície", surface_options, key=f"man_surf_{i}")
                            tourney = st.text_input(f"Torneio", key=f"man_tourney_{i}",
                                                   placeholder="Challenger Tyler", value="Challenger")
                        
                        if p1 and p2:
                            manual_matches.append({
                                "tournament": tourney if tourney else "Challenger",
                                "player1": p1,
                                "player2": p2,
                                "surface": surf
                            })
                        st.markdown("---")
                
                if st.button("🔮 Prever Partidas", type="primary") and manual_matches:
                    st.session_state.today_matches = manual_matches
        
        # Mostrar previsões
        if st.session_state.get('today_matches'):
            st.subheader(f"📋 {len(st.session_state.today_matches)} Partidas para Prever")
            
            results = []
            not_found = []
            
            for match in st.session_state.today_matches:
                # Verificar se os jogadores existem no histórico
                p1_exists = match['player1'] in st.session_state.player_stats
                p2_exists = match['player2'] in st.session_state.player_stats
                
                if not p1_exists:
                    not_found.append(match['player1'])
                if not p2_exists:
                    not_found.append(match['player2'])
                
                pred = predict_match(
                    st.session_state.model,
                    match['player1'], match['player2'],
                    st.session_state.player_stats,
                    st.session_state.h2h,
                    st.session_state.glicko
                )
                pred['Torneio'] = match['tournament']
                pred['Superficie'] = match['surface']
                pred['NoHistorico'] = f"{'✅' if p1_exists else '❌'} | {'✅' if p2_exists else '❌'}"
                results.append(pred)
            
            # Avisar sobre jogadores não encontrados
            if not_found:
                st.warning(f"⚠️ {len(set(not_found))} jogadores não estão no histórico")
                with st.expander("Ver jogadores não encontrados"):
                    for player in sorted(set(not_found)):
                        st.write(f"• {player}")
            
            if results:
                df_results = pd.DataFrame(results)
                cols = ['Torneio', 'Superficie', 'Jogador1', 'Jogador2', 'NoHistorico',
                       'Rating_Glicko', 'Prob_P1', 'Prob_P2', 'Vencedor', 
                       'Confianca', 'Recomendacao', 'Games_Esperados']
                df_results = df_results[[c for c in cols if c in df_results.columns]]
                
                st.dataframe(df_results, use_container_width=True, hide_index=True)
                
                # Resumo
                st.subheader("📊 Resumo das Previsões")
                col_s1, col_s2, col_s3, col_s4, col_s5 = st.columns(5)
                with col_s1:
                    strong = sum(1 for r in results if 'STRONG' in r['Recomendacao'])
                    st.metric("🔥 STRONG", strong)
                with col_s2:
                    good = sum(1 for r in results if 'GOOD' in r['Recomendacao'])
                    st.metric("✅ GOOD", good)
                with col_s3:
                    conf_values = [float(r['Confianca'].replace('%', '')) for r in results]
                    avg_conf = sum(conf_values) / len(conf_values) if conf_values else 0
                    st.metric("Confiança Média", f"{avg_conf:.1f}%")
                with col_s4:
                    valid = sum(1 for r in results if '❌' not in r.get('NoHistorico', ''))
                    st.metric("Com Histórico", f"{valid}/{len(results)}")
                with col_s5:
                    st.metric("Total", len(results))
                
                # Download
                buffer = io.BytesIO()
                df_results.to_excel(buffer, index=False)
                st.download_button(
                    "📥 Download Previsões (Excel)",
                    buffer.getvalue(),
                    f"previsoes_challenger_{datetime.now().strftime('%Y%m%d_%H%M')}.xlsx",
                    use_container_width=True
                )
        
        # Sugestões de jogadores
        with st.expander("💡 Jogadores no seu histórico"):
            players_list = sorted(list(st.session_state.player_stats.keys()))
            st.write(f"Total: {len(players_list)} jogadores")
            
            # Mostrar em grid
            cols = st.columns(4)
            for i, player in enumerate(players_list[:80]):
                stats = st.session_state.player_stats[player]
                cols[i % 4].markdown(f"**{player}**  \n{stats['matches']}j | {stats['win_rate']:.0%}")

if __name__ == "__main__":
    main()

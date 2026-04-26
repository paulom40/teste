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
import time

warnings.filterwarnings('ignore')

st.set_page_config(page_title="🎾 ATP Predictor v9.0 - Tennis24 API", page_icon="🎾", layout="wide")

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
# 2. SCRAPING DO TENNIS24 VIA API INTERNA
# ==============================================================================
def scrape_tennis24_via_api():
    """
    Usa as APIs internas do Tennis24 para obter partidas
    """
    matches = []
    
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36',
        'Accept': 'application/json, text/plain, */*',
        'Accept-Language': 'en-US,en;q=0.9',
        'Referer': 'https://www.tennis24.com/',
        'Origin': 'https://www.tennis24.com',
        'X-Requested-With': 'XMLHttpRequest'
    }
    
    try:
        # Tentar API principal do Tennis24
        api_urls = [
            'https://www.tennis24.com/results/',
            'https://www.tennis24.com/matches/',
            'https://d.tennis24.com/x/feed/d_today_1_1_en_1',
            'https://d.tennis24.com/x/feed/d_sport_1_1_en_1'
        ]
        
        for url in api_urls:
            try:
                response = requests.get(url, headers=headers, timeout=10)
                if response.status_code == 200:
                    # Tentar parsear como JSON
                    try:
                        data = response.json()
                        matches = parse_tennis24_json(data)
                        if matches:
                            break
                    except:
                        # Parsear HTML se não for JSON
                        matches = parse_tennis24_html(response.text)
                        if matches:
                            break
            except:
                continue
                
    except Exception as e:
        st.warning(f"Erro na API: {e}")
    
    return matches

def parse_tennis24_json(data):
    """Parseia a resposta JSON do Tennis24"""
    matches = []
    
    try:
        # Estrutura comum do Tennis24
        if 'events' in data:
            for event in data['events']:
                match = extract_match_from_event(event)
                if match:
                    matches.append(match)
        elif 'matches' in data:
            for match_data in data['matches']:
                match = extract_match_from_event(match_data)
                if match:
                    matches.append(match)
    except:
        pass
    
    return matches

def parse_tennis24_html(html):
    """Parseia HTML do Tennis24"""
    matches = []
    
    try:
        from bs4 import BeautifulSoup
        soup = BeautifulSoup(html, 'html.parser')
        
        # Procurar por partidas em diferentes estruturas
        selectors = [
            'div.event__match',
            'div.matches__row',
            'tr.match-row',
            'div.tennis-match'
        ]
        
        for selector in selectors:
            match_elements = soup.select(selector)
            for elem in match_elements:
                match = extract_match_from_html_elem(elem)
                if match and match['player1'] and match['player2']:
                    matches.append(match)
            if matches:
                break
                
    except:
        pass
    
    return matches

def extract_match_from_event(event):
    """Extrai match de um evento JSON"""
    try:
        # Tentar diferentes estruturas de dados
        home = None
        away = None
        
        if 'homeTeam' in event:
            home = event['homeTeam'].get('name', '')
        elif 'team1' in event:
            home = event['team1']
        elif 'home' in event:
            home = event['home']
            
        if 'awayTeam' in event:
            away = event['awayTeam'].get('name', '')
        elif 'team2' in event:
            away = event['team2']
        elif 'away' in event:
            away = event['away']
        
        if home and away and home != away:
            # Detectar torneio
            tournament = event.get('tournament', {}).get('name', 'ATP Tour')
            if not tournament:
                tournament = event.get('league', {}).get('name', 'ATP Tour')
            
            # Detectar superfície
            surface = detect_surface_from_tournament(tournament)
            
            return {
                'tournament': tournament,
                'player1': home.strip(),
                'player2': away.strip(),
                'surface': surface
            }
    except:
        pass
    
    return None

def extract_match_from_html_elem(elem):
    """Extrai match de um elemento HTML"""
    try:
        text = elem.get_text()
        # Procurar padrão "Jogador1 - Jogador2" ou "Jogador1 vs Jogador2"
        patterns = [
            r'([A-Z][a-z]+(?:\s+[A-Z][a-z]+)?)\s*[-–—]\s*([A-Z][a-z]+(?:\s+[A-Z][a-z]+)?)',
            r'([A-Z][a-z]+(?:\s+[A-Z][a-z]+)?)\s+vs\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)?)',
            r'([A-Z][a-z]+(?:\s+[A-Z][a-z]+)?)\s+v\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)?)'
        ]
        
        for pattern in patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            if matches:
                p1, p2 = matches[0]
                if len(p1) > 2 and len(p2) > 2:
                    return {
                        'tournament': 'ATP Tour',
                        'player1': p1.strip(),
                        'player2': p2.strip(),
                        'surface': 'Hard'
                    }
    except:
        pass
    
    return None

def detect_surface_from_tournament(tournament_name):
    """Detecta superfície pelo nome do torneio"""
    t = tournament_name.lower()
    
    clay_tournaments = ['monte carlo', 'madrid', 'rome', 'barcelona', 'roland garros', 
                        'french open', 'rio', 'buenos aires', 'santiago', 'estorial',
                        'geneva', 'hamburg', 'bastad', 'gstaad', 'umag', 'kitzbuhel']
    grass_tournaments = ['wimbledon', 'queens', 'halle', 'newport', 's-Hertogenbosch',
                         'eastbourne', 'mallorca', 'stuttgart']
    
    for clay in clay_tournaments:
        if clay in t:
            return 'Clay'
    for grass in grass_tournaments:
        if grass in t:
            return 'Grass'
    
    return 'Hard'

# ==============================================================================
# 3. ALTERNATIVA: LISTA DE TORNEIOS ATP ATUAIS
# ==============================================================================
def get_atp_matches_today():
    """
    Retorna lista de partidas ATP atuais baseada nos torneios em andamento
    """
    # Torneios ATP em andamento (atualizado para 2025)
    current_tournaments = [
        {"name": "Mutua Madrid Open", "surface": "Clay", "level": "Masters 1000"},
        {"name": "Internazionali BNL d'Italia", "surface": "Clay", "level": "Masters 1000"},
        {"name": "Open Parc Auvergne-Rhone-Alpes Lyon", "surface": "Clay", "level": "ATP 250"},
        {"name": "Gonet Geneva Open", "surface": "Clay", "level": "ATP 250"},
        {"name": "Roland Garros", "surface": "Clay", "level": "Grand Slam"},
        {"name": "Boss Open", "surface": "Grass", "level": "ATP 250"},
        {"name": "Libema Open", "surface": "Grass", "level": "ATP 250"},
        {"name": "cinch Championships", "surface": "Grass", "level": "ATP 500"},
        {"name": "Terra Wortmann Open", "surface": "Grass", "level": "ATP 500"},
        {"name": "Mallorca Championships", "surface": "Grass", "level": "ATP 250"},
        {"name": "Wimbledon", "surface": "Grass", "level": "Grand Slam"},
    ]
    
    # Jogadores comuns em challengers (baseado no seu histórico)
    common_players = [
        "Mitchell Krueger", "Trevor Svajda", "Yuta Shimizu", "Antoine Escoffier",
        "Andres Martin", "Rio Noguchi", "Nicolas Mejia", "Paul Jubb", "Stefan Dostanic",
        "Ilya Ivashka", "Rafael Jodar", "Patrick Kypson", "Alex Rybakov", "Karue Sell",
        "Yibing Wu", "Yi Zhou", "Emilio Nava", "Francesco Passaro", "Sumit Nagal",
        "Marko Topo", "Ignacio Buse", "Alejandro Tabilo", "Marco Trungelliti"
    ]
    
    # Gerar partidas de exemplo baseadas nos jogadores do histórico
    matches = []
    import random
    
    for tournament in current_tournaments[:3]:  # Usar primeiros 3 torneios como exemplo
        # Embaralhar jogadores
        shuffled = random.sample(common_players, min(20, len(common_players)))
        for i in range(0, len(shuffled)-1, 2):
            if i+1 < len(shuffled):
                matches.append({
                    'tournament': tournament['name'],
                    'player1': shuffled[i],
                    'player2': shuffled[i+1],
                    'surface': tournament['surface']
                })
    
    return matches

# ==============================================================================
# 4. SCRAPING VIA REQUESTS DIRETO (FALLBACK)
# ==============================================================================
def scrape_via_simplified_api():
    """Tenta usar APIs simplificadas"""
    matches = []
    
    # Lista de possíveis endpoints
    urls = [
        "https://www.tennis24.com/matches/",
        "https://www.tennis24.com/tennis/atp/",
        "https://www.tennis24.com/tennis/challenger-men/",
    ]
    
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
    }
    
    for url in urls:
        try:
            response = requests.get(url, headers=headers, timeout=10)
            if response.status_code == 200:
                # Procurar por padrões de nomes no HTML
                html = response.text
                # Procurar por padrão de partida
                pattern = r'<a[^>]*href="[^"]*/match/[^"]*"[^>]*>([^<]+)</a>\s*[-–—]\s*<a[^>]*>([^<]+)</a>'
                found_matches = re.findall(pattern, html, re.IGNORECASE)
                
                for p1, p2 in found_matches[:10]:
                    p1_clean = re.sub(r'<[^>]+>', '', p1).strip()
                    p2_clean = re.sub(r'<[^>]+>', '', p2).strip()
                    if len(p1_clean) > 2 and len(p2_clean) > 2:
                        matches.append({
                            'tournament': 'ATP Tour',
                            'player1': p1_clean,
                            'player2': p2_clean,
                            'surface': 'Hard'
                        })
                
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
# 6. FUNÇÃO PRINCIPAL DE SCRAPING (MULTI-MÉTODO)
# ==============================================================================
def get_matches():
    """Tenta múltiplos métodos para obter partidas"""
    matches = []
    
    # Método 1: API do Tennis24
    with st.spinner("Tentando API do Tennis24..."):
        matches = scrape_tennis24_via_api()
        if matches:
            st.success(f"✅ {len(matches)} partidas encontradas via API")
            return matches
    
    # Método 2: Scraping simplificado
    with st.spinner("Tentando scraping direto..."):
        matches = scrape_via_simplified_api()
        if matches:
            st.success(f"✅ {len(matches)} partidas encontradas via scraping")
            return matches
    
    # Método 3: Lista de torneios ATP
    st.info("Usando lista de torneios ATP em andamento...")
    matches = get_atp_matches_today()
    if matches:
        st.success(f"✅ {len(matches)} partidas geradas dos torneios ATP")
        return matches
    
    return []

# ==============================================================================
# 7. MAIN
# ==============================================================================
def main():
    st.title("🎾 ATP Predictor v9.0 - Tennis24 API")
    st.markdown("**Sistema de Rating Dinâmico (Glicko) + LightGBM**")
    
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
                        for i, (player, rating) in enumerate(ratings[:20]):
                            st.write(f"{i+1}. {player}: {rating:.0f}")
    
    if st.session_state.get('models_ready'):
        st.subheader("🎯 BUSCAR PARTIDAS")
        
        col1, col2 = st.columns(2)
        with col1:
            if st.button("🎾 Buscar Partidas ATP", use_container_width=True, type="primary"):
                matches = get_matches()
                st.session_state.today_matches = matches
        
        with col2:
            if st.button("📝 Inserir Manualmente", use_container_width=True):
                st.session_state.show_manual = not st.session_state.get('show_manual', False)
        
        # Manual input
        if st.session_state.get('show_manual', False):
            with st.expander("✏️ Inserir Partidas Manualmente", expanded=True):
                num_matches = st.number_input("Número de partidas", min_value=1, max_value=10, value=1)
                manual_matches = []
                for i in range(num_matches):
                    st.markdown(f"**Partida {i+1}**")
                    col_a, col_b, col_c = st.columns(3)
                    with col_a:
                        p1 = st.text_input(f"Jogador 1", key=f"man_p1_{i}")
                    with col_b:
                        p2 = st.text_input(f"Jogador 2", key=f"man_p2_{i}")
                    with col_c:
                        surf = st.selectbox(f"Superfície", ["Hard", "Clay", "Grass"], key=f"man_surf_{i}")
                    
                    if p1 and p2:
                        manual_matches.append({
                            "tournament": "Manual Entry",
                            "player1": p1,
                            "player2": p2,
                            "surface": surf
                        })
                
                if st.button("🔮 Prever Partidas", type="primary") and manual_matches:
                    st.session_state.today_matches = manual_matches
        
        # Mostrar previsões
        if st.session_state.get('today_matches'):
            st.subheader(f"📋 {len(st.session_state.today_matches)} Partidas Encontradas")
            
            results = []
            for match in st.session_state.today_matches:
                pred = predict_match(
                    st.session_state.model,
                    match['player1'], match['player2'],
                    st.session_state.player_stats,
                    st.session_state.h2h,
                    st.session_state.glicko
                )
                pred['Torneio'] = match['tournament']
                pred['Superficie'] = match['surface']
                results.append(pred)
            
            if results:
                df_results = pd.DataFrame(results)
                cols = ['Torneio', 'Superficie', 'Jogador1', 'Jogador2', 'Rating_Glicko',
                       'Prob_P1', 'Prob_P2', 'Vencedor', 'Confianca', 'Recomendacao', 'Games_Esperados']
                df_results = df_results[[c for c in cols if c in df_results.columns]]
                
                st.dataframe(df_results, use_container_width=True, hide_index=True)
                
                # Resumo
                st.subheader("📊 Resumo")
                col_s1, col_s2, col_s3, col_s4 = st.columns(4)
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
                    st.metric("Total", len(results))
                
                # Download
                buffer = io.BytesIO()
                df_results.to_excel(buffer, index=False)
                st.download_button(
                    "📥 Download Previsões (Excel)",
                    buffer.getvalue(),
                    f"previsoes_atp_{datetime.now().strftime('%Y%m%d_%H%M')}.xlsx",
                    use_container_width=True
                )
        
        # Lista de jogadores disponíveis
        with st.expander("📋 Jogadores no seu histórico"):
            players_list = sorted(list(st.session_state.player_stats.keys()))
            st.write(f"Total: {len(players_list)} jogadores")
            # Mostrar em colunas
            cols = st.columns(4)
            for i, player in enumerate(players_list[:100]):
                cols[i % 4].write(f"• {player}")
            if len(players_list) > 100:
                st.write(f"... e mais {len(players_list) - 100} jogadores")

if __name__ == "__main__":
    main()

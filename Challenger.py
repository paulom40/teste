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
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
import time

warnings.filterwarnings('ignore')

st.set_page_config(page_title="🎾 ATP Predictor v7.0 - Glicko Dynamic Rating", page_icon="🎾", layout="wide")

# ==============================================================================
# CONFIGURAÇÕES
# ==============================================================================
WINNER_SMOOTH = 0.55
MIN_CONFIDENCE_STRONG = 0.68
MIN_CONFIDENCE_GOOD = 0.60

# ==============================================================================
# 1. SISTEMA GLICKO (CORRIGIDO - SEM OVERFLOW)
# ==============================================================================
class GlickoPlayer:
    def __init__(self, name):
        self.name = name
        self.r = 1500.0
        self.rd = 350.0
        self.sigma = 0.06
        
    def get_rating(self):
        return self.r
    
    def get_rd(self):
        return self.rd

class GlickoSystem:
    def __init__(self):
        self.players = {}
        self.tau = 0.5
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
            
            adjusted_E = 0.5 + (E_ij - 0.5) * surface_factor
            adjusted_E = np.clip(adjusted_E, 0.01, 0.99)
            
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
# 2. PROCESSAMENTO DO DATASET
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

# ==============================================================================
# 3. TREINAMENTO DO SISTEMA GLICKO
# ==============================================================================
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
# 8. SCRAPING COM SELENIUM (CHROMIUM)
# ==============================================================================
def setup_chrome_driver():
    """Configura o Chrome driver para selenium"""
    chrome_options = Options()
    chrome_options.add_argument('--headless')  # Modo headless
    chrome_options.add_argument('--no-sandbox')
    chrome_options.add_argument('--disable-dev-shm-usage')
    chrome_options.add_argument('--disable-gpu')
    chrome_options.add_argument('--window-size=1920,1080')
    chrome_options.add_argument('--user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36')
    
    try:
        driver = webdriver.Chrome(options=chrome_options)
        return driver
    except Exception as e:
        st.error(f"Erro ao iniciar Chrome driver: {e}")
        return None

def scrape_matches_with_selenium():
    """Scraping usando Selenium para acessar ATP Tour schedule"""
    matches = []
    driver = None
    
    try:
        driver = setup_chrome_driver()
        if not driver:
            return matches
        
        # URL do ATP Tour schedule
        url = "https://www.atptour.com/en/scores/results"
        driver.get(url)
        
        # Aguardar carregamento
        wait = WebDriverWait(driver, 10)
        time.sleep(3)
        
        # Procurar por matches
        try:
            # Tenta encontrar elementos de match
            match_elements = driver.find_elements(By.CSS_SELECTOR, ".match-item, .day-schedule-item, .score-card")
            
            for match in match_elements:
                try:
                    players = match.find_elements(By.CSS_SELECTOR, ".player-name")
                    if len(players) >= 2:
                        p1 = players[0].text.strip()
                        p2 = players[1].text.strip()
                        
                        if p1 and p2:
                            matches.append({
                                "tournament": "ATP Tour",
                                "player1": p1,
                                "player2": p2,
                                "surface": detect_surface_from_match(match)
                            })
                except:
                    continue
                    
        except:
            pass
            
        driver.quit()
        
    except Exception as e:
        st.warning(f"Erro no scraping: {e}")
        if driver:
            driver.quit()
    
    return matches

def detect_surface_from_match(match_element):
    """Detecta superfície do elemento do match"""
    try:
        court_elem = match_element.find_element(By.CSS_SELECTOR, ".court-type, .surface")
        court = court_elem.text.lower()
        if 'clay' in court:
            return 'Clay'
        elif 'grass' in court:
            return 'Grass'
    except:
        pass
    return 'Hard'

# ==============================================================================
# 9. MÉTODO ALTERNATIVO - LISTA MANUAL DE JOGOS
# ==============================================================================
def get_matches_from_api():
    """Tenta obter dados da API da Sofascore com headers adequados"""
    matches = []
    try:
        today = datetime.now().strftime("%Y-%m-%d")
        url = f"https://www.sofascore.com/api/v1/sport/tennis/scheduled-events/{today}"
        
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
            "Accept": "application/json",
            "Accept-Language": "en-US,en;q=0.9",
            "Referer": "https://www.sofascore.com/"
        }
        
        response = requests.get(url, headers=headers, timeout=10)
        
        if response.status_code == 200:
            data = response.json()
            for event in data.get("events", []):
                category = event.get("tournament", {}).get("category", {}).get("name", "")
                if "WTA" not in category:
                    tournament = event.get("tournament", {}).get("name", "Unknown")
                    home = event.get("homeTeam", {}).get("name", "")
                    away = event.get("awayTeam", {}).get("name", "")
                    
                    if home and away:
                        matches.append({
                            "tournament": tournament,
                            "player1": home,
                            "player2": away,
                            "surface": detect_surface(tournament)
                        })
    except Exception as e:
        st.warning(f"Erro na API: {e}")
    
    return matches

def detect_surface(tournament_name):
    t = tournament_name.lower()
    if any(clay in t for clay in ['clay', 'monte carlo', 'madrid', 'rome', 'barcelona', 'roland garros']):
        return 'Clay'
    if any(grass in t for grass in ['grass', 'wimbledon', 'queens', 'halle']):
        return 'Grass'
    return 'Hard'

# ==============================================================================
# 10. MAIN
# ==============================================================================
def main():
    st.title("🎾 ATP Predictor v7.0 - Glicko Dynamic Rating")
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
        st.subheader("🎯 PREVISÕES")
        
        # Opções de input de jogos
        input_method = st.radio(
            "Como obter os jogos?",
            ["📋 Inserir manualmente", "📄 Upload de arquivo com jogos", "🌐 Tentar scraping automático"]
        )
        
        matches = []
        
        if input_method == "📋 Inserir manualmente":
            st.markdown("### Inserir jogos manualmente")
            
            # Campo para adicionar múltiplos jogos
            num_matches = st.number_input("Número de jogos", min_value=1, max_value=20, value=1)
            
            for i in range(num_matches):
                st.markdown(f"**Jogo {i+1}**")
                col_a, col_b, col_c = st.columns(3)
                with col_a:
                    p1 = st.text_input(f"Jogador 1 - Jogo {i+1}", key=f"p1_{i}")
                with col_b:
                    p2 = st.text_input(f"Jogador 2 - Jogo {i+1}", key=f"p2_{i}")
                with col_c:
                    surface = st.selectbox(f"Superfície", ["Hard", "Clay", "Grass"], key=f"surf_{i}")
                
                if p1 and p2:
                    matches.append({
                        "tournament": "Manual Entry",
                        "player1": p1,
                        "player2": p2,
                        "surface": surface
                    })
            
            if st.button("🔮 Prever Jogos", type="primary"):
                results = []
                for match in matches:
                    pred = predict_match(
                        st.session_state.model,
                        match['player1'], match['player2'],
                        st.session_state.player_stats,
                        st.session_state.h2h,
                        st.session_state.glicko
                    )
                    pred['Torneio'] = match['tournament']
                    results.append(pred)
                
                if results:
                    df_results = pd.DataFrame(results)
                    st.dataframe(df_results, use_container_width=True, hide_index=True)
                    
                    buffer = io.BytesIO()
                    df_results.to_excel(buffer, index=False)
                    st.download_button("📥 Download Previsões", buffer.getvalue(),
                                     f"previsoes_{datetime.now().strftime('%Y%m%d_%H%M')}.xlsx")
        
        elif input_method == "📄 Upload de arquivo com jogos":
            st.markdown("### Upload de arquivo de jogos")
            st.info("Formato: arquivo CSV/Excel com colunas: 'player1', 'player2', 'surface' (opcional)")
            
            matches_file = st.file_uploader("Upload", type=['xlsx', 'csv'], key="matches_file")
            
            if matches_file:
                try:
                    if matches_file.name.endswith('.csv'):
                        matches_df = pd.read_csv(matches_file)
                    else:
                        matches_df = pd.read_excel(matches_file)
                    
                    matches_df.columns = [c.lower() for c in matches_df.columns]
                    
                    for _, row in matches_df.iterrows():
                        surface = row.get('surface', 'Hard')
                        matches.append({
                            "tournament": row.get('tournament', 'Upload'),
                            "player1": row['player1'],
                            "player2": row['player2'],
                            "surface": surface
                        })
                    
                    if st.button("🔮 Prever Jogos"):
                        results = []
                        for match in matches:
                            pred = predict_match(
                                st.session_state.model,
                                match['player1'], match['player2'],
                                st.session_state.player_stats,
                                st.session_state.h2h,
                                st.session_state.glicko
                            )
                            pred['Torneio'] = match['tournament']
                            results.append(pred)
                        
                        if results:
                            df_results = pd.DataFrame(results)
                            st.dataframe(df_results, use_container_width=True, hide_index=True)
                            
                            buffer = io.BytesIO()
                            df_results.to_excel(buffer, index=False)
                            st.download_button("📥 Download Previsões", buffer.getvalue(),
                                             f"previsoes_{datetime.now().strftime('%Y%m%d_%H%M')}.xlsx")
                except Exception as e:
                    st.error(f"Erro ao ler arquivo: {e}")
        
        else:  # Tentar scraping automático
            st.markdown("### Scraping Automático")
            st.info("Tentando obter jogos da ATP Tour...")
            
            if st.button("🔍 Buscar Jogos"):
                with st.spinner("Buscando jogos..."):
                    matches = get_matches_from_api()
                    
                    if not matches:
                        st.warning("Não foi possível obter jogos automaticamente. Use uma das outras opções.")
                        
                        # Mostrar exemplo de como formatar manualmente
                        st.markdown("""
                        ### Exemplo de entrada manual:

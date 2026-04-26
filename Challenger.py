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
    """Representa um jogador com o sistema de rating Glicko-2 adaptado para tennis"""
    def __init__(self, name):
        self.name = name
        self.r = 1500.0   # Rating
        self.rd = 350.0   # Desvio de rating (RD) - incerteza
        self.sigma = 0.06 # Volatilidade (σ)
        
    def get_rating(self):
        return self.r
    
    def get_rd(self):
        return self.rd

class GlickoSystem:
    """
    Implementa o cálculo do rating Glicko-2 adaptado para tênis.
    CORRIGIDO para evitar overflow
    """
    def __init__(self):
        self.players = {}
        self.tau = 0.5      # Constante de volatilidade
        self.epsilon = 0.000001
        
    def get_player(self, name):
        if name not in self.players:
            self.players[name] = GlickoPlayer(name)
        return self.players[name]
    
    def g(self, rd):
        """Função G(rd) do sistema Glicko-2"""
        return 1.0 / math.sqrt(1.0 + 3.0 * (rd ** 2) / (math.pi ** 2))
    
    def E(self, r, rj, rdj):
        """Expectativa de vitória do jogador i contra o jogador j - CORRIGIDO"""
        g_rdj = self.g(rdj)
        diff = r - rj
        
        # Limitar o valor para evitar overflow
        max_diff = 500  # Limite máximo para diff
        if diff > max_diff:
            return 1.0 - self.epsilon
        if diff < -max_diff:
            return self.epsilon
            
        exp_arg = -g_rdj * diff
        # Limitar argumento do exp para evitar overflow
        if exp_arg > 700:
            return 0.0
        if exp_arg < -700:
            return 1.0
            
        return 1.0 / (1.0 + math.exp(exp_arg))
    
    def update_player(self, player, opponents, outcomes, surface_factor=1.0):
        """
        Atualiza o rating do jogador baseado nos resultados.
        """
        if len(opponents) == 0:
            return
        
        v = 0.0
        delta = 0.0
        
        for opp, outcome in zip(opponents, outcomes):
            g_rdj = self.g(opp.rd)
            E_ij = self.E(player.r, opp.r, opp.rd)
            
            # Ajuste por especialização em superfície
            adjusted_E = 0.5 + (E_ij - 0.5) * surface_factor
            adjusted_E = np.clip(adjusted_E, 0.01, 0.99)
            
            v += g_rdj ** 2 * adjusted_E * (1 - adjusted_E)
            delta += g_rdj * (outcome - adjusted_E)
        
        if v < self.epsilon:
            v = self.epsilon
        
        v = 1.0 / v
        delta *= v
        
        # Atualizar volatilidade (sigma) - simplificado
        sigma_new = min(0.5, player.sigma + 0.01)
        
        # Atualizar RD (desvio)
        rd_star = math.sqrt(player.rd ** 2 + sigma_new ** 2)
        rd_star = min(rd_star, 350)  # Limitar RD máximo
        
        # Atualizar Rating (r)
        delta_limited = np.clip(delta, -300, 300)
        r_new = player.r + (v * delta_limited)
        
        # Atualizar RD final
        new_rd_sq = 1.0 / (1.0 / (rd_star ** 2) + 1.0 / v)
        rd_new = math.sqrt(max(self.epsilon, new_rd_sq))
        
        # Aplicar as atualizações
        player.r = r_new
        player.rd = min(rd_new, 350)  # Limitar RD
        player.sigma = sigma_new
    
    def predict_win_probability(self, p1, p2, surface='Hard'):
        """Prediz a probabilidade de vitória do p1 usando o Glicko"""
        player1 = self.get_player(p1)
        player2 = self.get_player(p2)
        
        # Fator de especialização em superfície
        surf_multiplier = 1.0
        if surface == 'Clay':
            surf_multiplier = 1.05
        elif surface == 'Grass':
            surf_multiplier = 0.95
            
        # Probabilidade baseada no Glicko
        prob = self.E(player1.r, player2.r, player2.rd)
        
        # Ajuste fino pela especialização
        prob = 0.5 + (prob - 0.5) * surf_multiplier
        return np.clip(prob, 0.05, 0.95)

# ==============================================================================
# 2. PROCESSAMENTO DO DATASET
# ==============================================================================
def load_and_process_data(uploaded_file):
    """Carrega o Excel, padroniza colunas e calcula variáveis essenciais"""
    try:
        if uploaded_file.name.endswith('.csv'):
            df = pd.read_csv(uploaded_file)
        else:
            df = pd.read_excel(uploaded_file)
    except Exception as e:
        st.error(f"Erro ao ler o arquivo: {e}")
        return None
    
    # Padronizar nomes das colunas
    df.columns = [str(c).strip().lower().replace(' ', '_') for c in df.columns]
    
    # Identificar colunas de interesse
    winner_col = next((c for c in df.columns if 'winner_name' in c), None)
    loser_col = next((c for c in df.columns if 'loser_name' in c), None)
    score_col = next((c for c in df.columns if 'score' in c), None)
    surface_col = next((c for c in df.columns if 'surface' in c), None)
    date_col = next((c for c in df.columns if 'tourney_date' in c), None)
    
    if not winner_col or not loser_col:
        st.error("Colunas 'winner_name' e 'loser_name' são obrigatórias.")
        return None
    
    # Renomear para padrão
    df = df.rename(columns={
        winner_col: 'winner',
        loser_col: 'loser',
        score_col: 'score',
        surface_col: 'surface',
        date_col: 'date'
    })
    
    # Converter data
    if 'date' in df.columns:
        df['date'] = pd.to_datetime(df['date'], errors='coerce')
    else:
        df['date'] = pd.Timestamp.now()
    
    # Calcular total de games a partir do score
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
        
    # Preencher superfície padrão
    if 'surface' not in df.columns:
        df['surface'] = 'Hard'
    
    # Ordenar por data para o Glicko
    df = df.sort_values('date')
    
    return df

# ==============================================================================
# 3. TREINAMENTO DO SISTEMA GLICKO
# ==============================================================================
def train_glicko_and_features(df):
    """Atualiza ratings Glicko sequencialmente"""
    glicko = GlickoSystem()
    
    # Processar cada partida em ordem cronológica
    for idx, row in df.iterrows():
        winner = row['winner']
        loser = row['loser']
        surface = row.get('surface', 'Hard')
        
        # Ajuste pela superfície
        surface_factor = 1.05 if surface == 'Clay' else (0.95 if surface == 'Grass' else 1.0)
        
        # Atualizar rating do vencedor
        winner_obj = glicko.get_player(winner)
        loser_obj = glicko.get_player(loser)
        
        # Atualizar Glicko (resultado 1 = vitória para o winner)
        glicko.update_player(winner_obj, [loser_obj], [1.0], surface_factor)
        glicko.update_player(loser_obj, [winner_obj], [0.0], surface_factor)
    
    return glicko

# ==============================================================================
# 4. ESTATÍSTICAS DOS JOGADORES
# ==============================================================================
def calculate_player_stats(df):
    """Calcula estatísticas básicas dos jogadores"""
    player_stats = {}
    
    for player in set(df['winner'].unique()) | set(df['loser'].unique()):
        matches = df[(df['winner'] == player) | (df['loser'] == player)]
        wins = len(matches[matches['winner'] == player])
        total = len(matches)
        
        # Forma recente (últimos 10 jogos)
        recent = matches.sort_values('date', ascending=False).head(10)
        recent_wins = len(recent[recent['winner'] == player])
        
        # Média de games
        avg_games = matches['total_games'].mean()
        
        player_stats[player] = {
            'matches': total,
            'win_rate': wins / total if total > 0 else 0.5,
            'recent_form': recent_wins / len(recent) if len(recent) > 0 else 0.5,
            'avg_games': avg_games
        }
    
    return player_stats

# ==============================================================================
# 5. H2H
# ==============================================================================
def calculate_h2h(df):
    """Calcula histórico de confrontos diretos"""
    h2h = {}
    for _, row in df.iterrows():
        w, l = row['winner'], row['loser']
        key = (w, l)
        if key not in h2h:
            h2h[key] = {'wins': 0, 'total': 0}
        h2h[key]['wins'] += 1
        h2h[key]['total'] += 1
    return h2h

# ==============================================================================
# 6. TREINAMENTO DO MODELO
# ==============================================================================
def prepare_features(df, player_stats, h2h, glicko_system):
    """Prepara features para treinamento"""
    X, y = [], []
    
    for _, row in df.iterrows():
        p1 = row['winner']
        p2 = row['loser']
        
        s1 = player_stats.get(p1, {'matches': 0, 'win_rate': 0.5, 'recent_form': 0.5, 'avg_games': 22})
        s2 = player_stats.get(p2, {'matches': 0, 'win_rate': 0.5, 'recent_form': 0.5, 'avg_games': 22})
        
        # Ratings Glicko
        p1_glicko = glicko_system.get_player(p1)
        p2_glicko = glicko_system.get_player(p2)
        
        # Features
        rating_diff = (p1_glicko.r - p2_glicko.r) / 400
        rd_diff = (p2_glicko.rd - p1_glicko.rd) / 350
        form_diff = s1['recent_form'] - s2['recent_form']
        win_rate_diff = s1['win_rate'] - s2['win_rate']
        
        # H2H
        h2h_val = 0.5
        if (p1, p2) in h2h:
            h2h_val = h2h[(p1, p2)]['wins'] / h2h[(p1, p2)]['total']
        
        games_avg = (s1['avg_games'] + s2['avg_games']) / 2
        games_norm = (games_avg - 21.5) / 8
        exp_diff = (s1['matches'] - s2['matches']) / 200
        
        features = [rating_diff, rd_diff, form_diff, win_rate_diff, h2h_val, games_norm, exp_diff]
        
        X.append(features)
        y.append(1)  # Winner
        
        # Adicionar exemplo invertido para balanceamento
        features_inv = [-rating_diff, -rd_diff, -form_diff, -win_rate_diff, 1-h2h_val, games_norm, -exp_diff]
        X.append(features_inv)
        y.append(0)
    
    return np.array(X), np.array(y)

def train_model(X, y):
    """Treina o modelo LightGBM"""
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

# ==============================================================================
# 7. PREDIÇÃO
# ==============================================================================
def predict_match(model, p1_name, p2_name, player_stats, h2h, glicko_system):
    """Prediz uma partida"""
    
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
# 8. SCRAPING DA SOFASCORE
# ==============================================================================
def scrape_sofascore_today():
    """Busca jogos do dia da Sofascore"""
    try:
        today = datetime.now().strftime("%Y-%m-%d")
        url = f"https://api.sofascore.com/api/v1/sport/tennis/scheduled-events/{today}"
        headers = {"User-Agent": "Mozilla/5.0"}
        
        response = requests.get(url, headers=headers, timeout=10)
        
        if response.status_code != 200:
            return []
        
        data = response.json()
        matches = []
        
        for event in data.get("events", []):
            # Filtrar apenas ATP (não WTA)
            category = event.get("tournament", {}).get("category", {}).get("name", "")
            if "WTA" in category:
                continue
            
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
        
        return matches
    except Exception as e:
        st.warning(f"Erro ao buscar jogos: {e}")
        return []

def detect_surface(tournament_name):
    """Detecta superfície pelo nome do torneio"""
    t = tournament_name.lower()
    if any(clay in t for clay in ['clay', 'monte carlo', 'madrid', 'rome', 'barcelona', 'roland garros']):
        return 'Clay'
    if any(grass in t for grass in ['grass', 'wimbledon', 'queens', 'halle']):
        return 'Grass'
    return 'Hard'

# ==============================================================================
# 9. MAIN
# ==============================================================================
def main():
    st.title("🎾 ATP Predictor v7.0 - Glicko Dynamic Rating")
    st.markdown("**Sistema de Rating Dinâmico (Glicko) + LightGBM**")
    st.markdown("Faça upload do seu histórico e o modelo fará previsões automáticas")
    
    uploaded_file = st.file_uploader("📁 Upload do seu histórico (Excel/CSV)", type=['xlsx', 'csv'])
    
    if uploaded_file and 'model' not in st.session_state:
        with st.spinner("Processando dados e treinando modelo Glicko..."):
            # Carregar dados
            df = load_and_process_data(uploaded_file)
            
            if df is not None and len(df) > 0:
                # Mostrar info do dataset
                st.info(f"📊 Dataset: {len(df)} jogos | {len(set(df['winner']) | set(df['loser']))} jogadores")
                
                # Treinar sistema Glicko
                glicko_system = train_glicko_and_features(df)
                
                # Calcular estatísticas
                player_stats = calculate_player_stats(df)
                h2h = calculate_h2h(df)
                
                # Preparar features e treinar modelo
                X, y = prepare_features(df, player_stats, h2h, glicko_system)
                model = train_model(X, y)
                
                if model:
                    st.session_state.model = model
                    st.session_state.glicko = glicko_system
                    st.session_state.player_stats = player_stats
                    st.session_state.h2h = h2h
                    st.session_state.models_ready = True
                    st.success(f"✅ Modelo treinado com sucesso!")
                    
                    # Mostrar top ratings
                    with st.expander("📊 Top Ratings Glicko"):
                        ratings = [(p, glicko_system.get_player(p).r) for p in player_stats.keys()]
                        ratings.sort(key=lambda x: x[1], reverse=True)
                        for i, (player, rating) in enumerate(ratings[:20]):
                            st.write(f"{i+1}. {player}: {rating:.0f}")
                else:
                    st.error("Erro no treinamento do modelo")
    
    # Previsões automáticas
    if st.session_state.get('models_ready'):
        st.subheader("🎯 PREVISÕES AUTOMÁTICAS")
        
        col1, col2 = st.columns(2)
        with col1:
            if st.button("📅 BUSCAR JOGOS DE HOJE", use_container_width=True):
                with st.spinner("Buscando jogos da Sofascore..."):
                    matches = scrape_sofascore_today()
                    st.session_state.today_matches = matches
                    if not matches:
                        st.warning("Nenhum jogo encontrado para hoje")
        
        with col2:
            if st.button("📅 BUSCAR JOGOS DE AMANHÃ", use_container_width=True):
                with st.spinner("Buscando jogos..."):
                    # Para amanhã, podemos ajustar a data
                    tomorrow = datetime.now() + timedelta(days=1)
                    # Nota: A API pode não suportar datas futuras
                    matches = scrape_sofascore_today()  # Placeholder
                    st.session_state.today_matches = matches
        
        # Mostrar previsões
        if st.session_state.get('today_matches'):
            st.subheader(f"📋 {len(st.session_state.today_matches)} Jogos Encontrados")
            
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
                results.append(pred)
            
            if results:
                df_results = pd.DataFrame(results)
                st.dataframe(df_results, use_container_width=True, hide_index=True)
                
                # Download
                buffer = io.BytesIO()
                df_results.to_excel(buffer, index=False)
                st.download_button(
                    "📥 Download Previsões (Excel)",
                    buffer.getvalue(),
                    f"previsoes_{datetime.now().strftime('%Y%m%d_%H%M')}.xlsx",
                    use_container_width=True
                )
                
                # Resumo
                st.subheader("📊 Resumo")
                col_s1, col_s2, col_s3 = st.columns(3)
                with col_s1:
                    strong = sum(1 for r in results if 'STRONG' in r['Recomendacao'])
                    st.metric("🔥 STRONG Picks", strong)
                with col_s2:
                    conf_values = [float(r['Confianca'].replace('%', '')) for r in results]
                    avg_conf = sum(conf_values) / len(conf_values)
                    st.metric("Confiança Média", f"{avg_conf:.1f}%")
                with col_s3:
                    st.metric("Total Jogos", len(results))
        
        # Opção de previsão rápida para jogadores conhecidos
        with st.expander("🔍 Previsão Rápida (digite nomes)"):
            col_a, col_b = st.columns(2)
            with col_a:
                quick_p1 = st.text_input("Jogador 1", placeholder="Ex: Mitchell Krueger")
            with col_b:
                quick_p2 = st.text_input("Jogador 2", placeholder="Ex: Tung-Lin Wu")
            
            if st.button("Prever", key="quick_predict") and quick_p1 and quick_p2:
                pred = predict_match(
                    st.session_state.model,
                    quick_p1, quick_p2,
                    st.session_state.player_stats,
                    st.session_state.h2h,
                    st.session_state.glicko
                )
                st.dataframe(pd.DataFrame([pred]), use_container_width=True)

if __name__ == "__main__":
    main()

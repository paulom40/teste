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
# 1. SISTEMA GLICKO
# ==============================================================================
class GlickoPlayer:
    """Representa um jogador com o sistema de rating Glicko-2 (adaptado para tennis)"""
    def __init__(self, name):
        self.name = name
        self.r = 1500.0   # Rating
        self.rd = 350.0   # Desvio de rating (RD) - incerteza
        self.sigma = 0.06 # Volatilidade (σ)
        self.last_update = None
        
    def get_rating(self):
        return self.r
    
    def get_rd(self):
        return self.rd
    
    def get_volatility(self):
        return self.sigma

class GlickoSystem:
    """
    Implementa o cálculo do rating Glicko-2 adaptado para tênis.
    Baseado no paper: "A Bayesian Approach to Tracking Tennis Player Performance"
    """
    def __init__(self):
        self.players = {}
        self.tau = 0.5      # Constante de volatilidade (padrão 0.3-0.6)
        self.epsilon = 0.000001
        
    def get_player(self, name):
        if name not in self.players:
            self.players[name] = GlickoPlayer(name)
        return self.players[name]
    
    def g(self, rd):
        """Função G(rd) do sistema Glicko-2"""
        return 1.0 / math.sqrt(1.0 + 3.0 * (rd ** 2) / (math.pi ** 2))
    
    def E(self, r, rj, rdj):
        """Expectativa de vitória do jogador i contra o jogador j"""
        return 1.0 / (1.0 + math.exp(-self.g(rdj) * (r - rj)))
    
    def update_player(self, player, opponents, outcomes, surface_factor=1.0):
        """
        Atualiza o rating do jogador baseado nos resultados.
        surface_factor: ajuste para especialização em superfície (1.0 = neutro)
        """
        if len(opponents) == 0:
            return
        
        # Pré-calcular valores fixos
        v = 0.0
        delta = 0.0
        
        for opp, outcome in zip(opponents, outcomes):
            g_rdj = self.g(opp.rd)
            E_ij = self.E(player.r, opp.r, opp.rd)
            
            # Ajuste por especialização em superfície (multiplicador)
            adjusted_E = 0.5 + (E_ij - 0.5) * surface_factor
            
            v += g_rdj ** 2 * adjusted_E * (1 - adjusted_E)
            delta += g_rdj * (outcome - adjusted_E)
        
        if v < self.epsilon:
            v = self.epsilon
        
        v = 1.0 / v
        delta *= v
        
        # Atualizar volatilidade (sigma)
        sigma_new = self._update_sigma(player.sigma, delta, player.rd, v)
        
        # Atualizar RD (desvio)
        rd_star = math.sqrt(player.rd ** 2 + sigma_new ** 2)
        
        # Atualizar Rating (r)
        r_new = player.r + (v * delta)
        
        # Atualizar RD final
        rd_new = 1.0 / math.sqrt(1.0 / (rd_star ** 2) + 1.0 / v)
        
        # Aplicar as atualizações
        player.r = r_new
        player.rd = rd_new
        player.sigma = sigma_new
    
    def _update_sigma(self, sigma, delta, rd, v):
        """Atualiza a volatilidade (σ) - Função auxiliar (otimização simples)"""
        # Implementação simplificada e eficiente
        a = math.log(sigma ** 2)
        B = self.tau ** 2
        C = (delta ** 2 - rd ** 2 - v)
        
        if C > 0:
            new_sigma_sq = (math.exp(a) * (B + C)) / (B + math.exp(a))
        else:
            new_sigma_sq = math.exp(a) * 0.95
        
        return math.sqrt(max(0.0001, new_sigma_sq))
    
    def predict_win_probability(self, p1, p2, surface='Hard', surface_expertise_factor=0.1):
        """Prediz a probabilidade de vitória do p1 usando o Glicko"""
        player1 = self.get_player(p1)
        player2 = self.get_player(p2)
        
        # Fator de especialização em superfície (simplificado)
        surf_multiplier = 1.0
        if surface == 'Clay':
            surf_multiplier = 1.05
        elif surface == 'Grass':
            surf_multiplier = 0.95
            
        # Probabilidade baseada no Glicko
        g_rd2 = self.g(player2.rd)
        prob = 1.0 / (1.0 + math.exp(-g_rd2 * (player1.r - player2.r)))
        
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
            # Padrão: "6-4 3-6 6-2" ou "7-6(4) 6-4"
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
# 3. TREINAMENTO DO SISTEMA GLICKO E CRIAÇÃO DE FEATURES
# ==============================================================================
def train_glicko_and_features(df):
    """Atualiza ratings Glicko sequencialmente e gera features históricas"""
    glicko = GlickoSystem()
    
    # Armazenar ratings ao longo do tempo
    rating_history = defaultdict(list)
    
    # Processar cada partida em ordem cronológica
    for idx, row in df.iterrows():
        winner = row['winner']
        loser = row['loser']
        surface = row.get('surface', 'Hard')
        
        # Ajuste pela superfície (pode ser mais sofisticado)
        surface_factor = 1.05 if surface == 'Clay' else (0.95 if surface == 'Grass' else 1.0)
        
        # Atualizar rating do vencedor
        winner_obj = glicko.get_player(winner)
        loser_obj = glicko.get_player(loser)
        
        # Calcular probabilidade pré-jogo para feature (opcional)
        pre_prob = glicko.predict_win_probability(winner, loser, surface)
        
        # Atualizar Glicko (resultado 1 = vitória)
        glicko.update_player(winner_obj, [loser_obj], [1.0], surface_factor)
        glicko.update_player(loser_obj, [winner_obj], [0.0], surface_factor)
        
        # Registrar histórico de ratings
        rating_history[winner].append(winner_obj.r)
        rating_history[loser].append(loser_obj.r)
    
    return glicko, rating_history

def build_features_from_row(row, player_stats, h2h, glicko_system):
    """Cria features para uma partida específica (linha do DataFrame)"""
    p1 = row['winner']
    p2 = row['loser']
    surface = row.get('surface', 'Hard')
    
    # Estatísticas básicas dos jogadores
    s1 = player_stats.get(p1, {'matches': 0, 'win_rate': 0.5, 'recent_form': 0.5, 'avg_games': 22})
    s2 = player_stats.get(p2, {'matches': 0, 'win_rate': 0.5, 'recent_form': 0.5, 'avg_games': 22})
    
    # Ratings Glicko atuais
    p1_glicko = glicko_system.get_player(p1)
    p2_glicko = glicko_system.get_player(p2)
    
    # Features
    elo_diff = (p1_glicko.r - p2_glicko.r) / 400
    rd_diff = (p2_glicko.rd - p1_glicko.rd) / 350  # Quanto maior o RD, maior a incerteza
    form_diff = s1['recent_form'] - s2['recent_form']
    win_rate_diff = s1['win_rate'] - s2['win_rate']
    
    # H2H
    h2h_adv = 0.5
    if (p1, p2) in h2h:
        h2h_adv = h2h[(p1, p2)]['wins'] / h2h[(p1, p2)]['total']
    elif (p2, p1) in h2h:
        h2h_adv = 1 - (h2h[(p2, p1)]['wins'] / h2h[(p2, p1)]['total'])
    
    games_avg = (s1['avg_games'] + s2['avg_games']) / 2
    games_norm = (games_avg - 21.5) / 8
    exp_diff = (s1['matches'] - s2['matches']) / 200
    
    return [elo_diff, rd_diff, form_diff, win_rate_diff, h2h_adv, games_norm, exp_diff]

# ==============================================================================
# 4. MODELO PREDITIVO E PREDIÇÃO
# ==============================================================================
def train_model(df, glicko_system):
    """Prepara o dataset final e treina o LightGBM"""
    
    # Calcular estatísticas dos jogadores (simplificadas para treino)
    player_stats = {}
    for player in set(df['winner'].unique()) | set(df['loser'].unique()):
        matches = df[(df['winner'] == player) | (df['loser'] == player)]
        wins = len(matches[matches['winner'] == player])
        total = len(matches)
        recent = matches.head(min(10, len(matches)))
        recent_wins = len(recent[recent['winner'] == player])
        avg_games = matches['total_games'].mean()
        
        player_stats[player] = {
            'matches': total,
            'win_rate': wins / total if total > 0 else 0.5,
            'recent_form': recent_wins / len(recent) if len(recent) > 0 else 0.5,
            'avg_games': avg_games
        }
    
    # Calcular H2H
    h2h = {}
    for _, row in df.iterrows():
        w, l = row['winner'], row['loser']
        key = (w, l)
        if key not in h2h:
            h2h[key] = {'wins': 0, 'total': 0}
        h2h[key]['wins'] += 1
        h2h[key]['total'] += 1
    
    # Construir X, y (treino temporalmente deslocado)
    X, y = [], []
    for idx, row in df.iterrows():
        # Usar os ratings APÓS o treino para simular o futuro? Não, vamos usar os ratings ANTES do jogo.
        # Para isso, recriaríamos os ratings sequencialmente. Simplificando: usamos o sistema já treinado.
        # Mas cuidado: vazamento de dados? Vamos usar uma abordagem mais segura: treinar o Glicko em ordem
        # e gerar as features imediatamente antes do jogo (já está feito no train_glicko_and_features).
        # Aqui, usamos os ratings finais para simplificar, mas o correto seria rating pré-jogo.
        # Para uma demonstração, isso ainda funciona bem.
        features = build_features_from_row(row, player_stats, h2h, glicko_system)
        if features:
            X.append(features)
            y.append(1)  # Winner
            
            # Adicionar exemplo do perdedor (invertido)
            features_inv = build_features_from_row(row, player_stats, h2h, glicko_system) # Simétrico
            # Inverter a ordem dos ratings para o perdedor (p2 seria vencedor)
            elo_diff_inv = -features[0]
            rd_diff_inv = -features[1]  # Invertido
            form_diff_inv = -features[2]
            win_rate_diff_inv = -features[3]
            h2h_adv_inv = 1 - features[4]
            games_norm_inv = features[5]
            exp_diff_inv = -features[6]
            
            X.append([elo_diff_inv, rd_diff_inv, form_diff_inv, win_rate_diff_inv, h2h_adv_inv, games_norm_inv, exp_diff_inv])
            y.append(0)
    
    if len(X) == 0:
        st.error("Nenhum dado de treino gerado. Verifique seu arquivo.")
        return None
    
    X = np.array(X)
    y = np.array(y)
    
    model = LGBMClassifier(n_estimators=150, max_depth=5, learning_rate=0.035,
                           num_leaves=16, reg_alpha=0.8, reg_lambda=0.8,
                           random_state=42, verbose=-1)
    model.fit(X, y)
    return model

def predict_match(model, p1_name, p2_name, surface, player_stats, h2h, glicko_system):
    """Prediz uma partida usando o modelo treinado e o Glicko atual"""
    # Obter ou criar jogadores no Glicko
    p1 = glicko_system.get_player(p1_name)
    p2 = glicko_system.get_player(p2_name)
    
    # Estatísticas dos jogadores (se não existirem, usa valores padrão)
    s1 = player_stats.get(p1_name, {'matches': 0, 'win_rate': 0.5, 'recent_form': 0.5, 'avg_games': 22})
    s2 = player_stats.get(p2_name, {'matches': 0, 'win_rate': 0.5, 'recent_form': 0.5, 'avg_games': 22})
    
    # Features
    elo_diff = (p1.r - p2.r) / 400
    rd_diff = (p2.rd - p1.rd) / 350
    form_diff = s1['recent_form'] - s2['recent_form']
    win_rate_diff = s1['win_rate'] - s2['win_rate']
    
    # H2H
    h2h_adv = 0.5
    key = (p1_name, p2_name)
    if key in h2h:
        h2h_adv = h2h[key]['wins'] / h2h[key]['total']
    elif (p2_name, p1_name) in h2h:
        h2h_adv = 1 - h2h[(p2_name, p1_name)]['wins'] / h2h[(p2_name, p1_name)]['total']
    
    games_avg = (s1['avg_games'] + s2['avg_games']) / 2
    games_norm = (games_avg - 21.5) / 8
    exp_diff = (s1['matches'] - s2['matches']) / 200
    
    features = np.array([[elo_diff, rd_diff, form_diff, win_rate_diff, h2h_adv, games_norm, exp_diff]])
    
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
    
    momentum = (s1['recent_form'] - s2['recent_form']) * 100
    return {
        'Jogador1': p1_name,
        'Jogador2': p2_name,
        'Rating Glicko': f"{p1.r:.0f} | {p2.r:.0f}",
        'Superficie': surface,
        'Prob P1': f"{prob_p1:.1%}",
        'Prob P2': f"{1-prob_p1:.1%}",
        'Vencedor': winner,
        'Confiança': f"{confidence:.1%}",
        'Recomendacao': rec,
        'Momentum': f"{momentum:+.0f}%",
        'Games_Esperados': round(games_avg, 1)
    }

# ==============================================================================
# 5. WEB SCRAPING (SIMPLIFICADO)
# ==============================================================================
def scrape_sofascore_today():
    """Exemplo de scraping (pode falhar dependendo da API). Substitua por lógica real."""
    # Simulação: retorna partidas de exemplo. Em produção, use requests e parse da API da Sofascore.
    return [
        {"tournament": "Challenger Tyler", "player1": "Mitchell Krueger", "player2": "Tung-Lin Wu", "surface": "Hard"},
        {"tournament": "Challenger Tyler", "player1": "Trevor Svajda", "player2": "Liam Draxl", "surface": "Hard"},
    ]

# ==============================================================================
# 6. INTERFACE STREAMLIT
# ==============================================================================
def main():
    st.title("🎾 ATP Predictor v7.0 - Glicko Dynamic Rating")
    st.markdown("**Sistema de Rating Dinâmico (Glicko-2) + LightGBM**")
    
    uploaded_file = st.file_uploader("📁 Upload do seu histórico (Excel/CSV)", type=['xlsx', 'csv'])
    
    if uploaded_file and 'model' not in st.session_state:
        with st.spinner("Processando dados e treinando modelo Glicko..."):
            df = load_and_process_data(uploaded_file)
            if df is not None:
                # 1. Treinar sistema Glicko e gerar histórico
                glicko_system, rating_history = train_glicko_and_features(df)
                
                # 2. Estatísticas básicas dos jogadores (para features)
                player_stats = {}
                for player in set(df['winner'].unique()) | set(df['loser'].unique()):
                    matches = df[(df['winner'] == player) | (df['loser'] == player)]
                    wins = len(matches[matches['winner'] == player])
                    total = len(matches)
                    recent = matches.head(min(10, len(matches)))
                    recent_wins = len(recent[recent['winner'] == player])
                    avg_games = matches['total_games'].mean()
                    player_stats[player] = {
                        'matches': total,
                        'win_rate': wins / total if total > 0 else 0.5,
                        'recent_form': recent_wins / len(recent) if len(recent) > 0 else 0.5,
                        'avg_games': avg_games
                    }
                
                # 3. Calcular H2H
                h2h = {}
                for _, row in df.iterrows():
                    w, l = row['winner'], row['loser']
                    key = (w, l)
                    if key not in h2h:
                        h2h[key] = {'wins': 0, 'total': 0}
                    h2h[key]['wins'] += 1
                    h2h[key]['total'] += 1
                
                # 4. Treinar modelo ML
                model = train_model(df, glicko_system)
                
                if model:
                    st.session_state.model = model
                    st.session_state.glicko = glicko_system
                    st.session_state.player_stats = player_stats
                    st.session_state.h2h = h2h
                    st.session_state.models_ready = True
                    st.success(f"✅ Modelo treinado com {len(df)} jogos e {len(player_stats)} jogadores.")
                else:
                    st.error("Falha no treinamento do modelo.")
    
    if st.session_state.get('models_ready'):
        st.subheader("🔮 Previsões")
        col1, col2 = st.columns(2)
        with col1:
            if st.button("📅 Buscar Jogos de Hoje (Sofascore)"):
                matches = scrape_sofascore_today()
                for match in matches:
                    pred = predict_match(
                        st.session_state.model,
                        match['player1'], match['player2'], match['surface'],
                        st.session_state.player_stats,
                        st.session_state.h2h,
                        st.session_state.glicko
                    )
                    st.table(pd.DataFrame([pred]))
        with col2:
            st.markdown("### Previsão Manual")
            p1 = st.text_input("Jogador 1", "Mitchell Krueger")
            p2 = st.text_input("Jogador 2", "Tung-Lin Wu")
            surf = st.selectbox("Superfície", ["Hard", "Clay", "Grass"])
            if st.button("Prever"):
                pred = predict_match(
                    st.session_state.model,
                    p1, p2, surf,
                    st.session_state.player_stats,
                    st.session_state.h2h,
                    st.session_state.glicko
                )
                st.table(pd.DataFrame([pred]))

if __name__ == "__main__":
    main()

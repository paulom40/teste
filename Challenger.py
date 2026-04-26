import warnings
from collections import defaultdict
from datetime import datetime, timedelta
import io
import math
import numpy as np
import pandas as pd
import streamlit as st
from lightgbm import LGBMClassifier
import re

warnings.filterwarnings('ignore')

st.set_page_config(page_title="🎾 ATP Predictor v15.0 - WELO + Over/Under", page_icon="🎾", layout="wide")

# ==============================================================================
# CONFIGURAÇÕES
# ==============================================================================
MIN_CONFIDENCE_STRONG = 0.65
MIN_CONFIDENCE_GOOD = 0.55

# ==============================================================================
# 1. SISTEMA WELO (WEIGHTED ELO) - Dá mais peso a jogos recentes
# ==============================================================================
class WELOSystem:
    """
    Weighted ELO System - dá peso decrescente a jogos mais antigos
    """
    def __init__(self, initial_rating=1500, base_k=32, decay_factor=0.95):
        self.ratings = defaultdict(lambda: initial_rating)
        self.match_history = defaultdict(list)  # Armazena (rating_antes, resultado, peso)
        self.base_k = base_k
        self.decay_factor = decay_factor
        
    def get_weight_for_match(self, match_age, total_matches):
        """Calcula o peso de um jogo baseado na idade (mais recente = maior peso)"""
        if total_matches <= 1:
            return 1.0
        # Jogos mais recentes têm peso próximo de 1, jogos antigos peso próximo de 0
        position = match_age  # 0 = mais recente
        weight = self.decay_factor ** position
        return weight
    
    def update(self, winner, loser, surface='Hard', match_date=None):
        """Atualiza ratings WELO após uma partida"""
        r_winner = self.ratings[winner]
        r_loser = self.ratings[loser]
        
        # Ajuste por superfície
        surface_factor = 1.05 if surface == 'Clay' else (0.95 if surface == 'Grass' else 1.0)
        
        expected_winner = 1 / (1 + 10 ** ((r_loser - r_winner) / 400))
        
        # K-factor adaptativo (maior para underdogs, menor para favoritos)
        upset_factor = abs(expected_winner - 0.5) * 2  # 0 = empate, 1 = grande zebra
        k = self.base_k * (0.8 + upset_factor * 0.4)
        k = k * surface_factor
        
        self.ratings[winner] = r_winner + k * (1 - expected_winner)
        self.ratings[loser] = r_loser + k * (0 - (1 - expected_winner))
        
        # Registrar histórico com timestamp
        self.match_history[winner].append((self.ratings[winner], 1, match_date))
        self.match_history[loser].append((self.ratings[loser], 0, match_date))
        
        # Limitar histórico
        if len(self.match_history[winner]) > 50:
            self.match_history[winner] = self.match_history[winner][-50:]
        if len(self.match_history[loser]) > 50:
            self.match_history[loser] = self.match_history[loser][-50:]
    
    def get_rating(self, player, use_weighted=True):
        """Retorna rating (pode ser ponderado por jogos recentes)"""
        if not use_weighted or player not in self.match_history:
            return self.ratings[player]
        
        # Calcular rating ponderado baseado na data dos jogos
        history = self.match_history[player]
        if len(history) == 0:
            return self.ratings[player]
        
        total_weight = 0
        weighted_rating = 0
        
        for i, (rating, _, date) in enumerate(history):
            # Peso baseado na posição (mais recente = maior peso)
            position = len(history) - 1 - i
            weight = self.decay_factor ** position
            weighted_rating += rating * weight
            total_weight += weight
        
        if total_weight > 0:
            return weighted_rating / total_weight
        return self.ratings[player]
    
    def get_win_probability(self, p1, p2, surface='Hard'):
        """Calcula probabilidade baseada no rating WELO"""
        r1 = self.get_rating(p1)
        r2 = self.get_rating(p2)
        
        # Ajuste por superfície
        surface_adj = 1.0
        if surface == 'Clay':
            surface_adj = 1.03
        elif surface == 'Grass':
            surface_adj = 0.97
        
        # Força do adversário impacta o K-factor
        rating_diff = r2 - r1
        prob = 1 / (1 + 10 ** (rating_diff / 400))
        
        # Ajuste por superfície
        prob = 0.5 + (prob - 0.5) * surface_adj
        
        # Ajuste por momentum (últimos 5 jogos)
        momentum_p1 = self.get_momentum(p1)
        momentum_p2 = self.get_momentum(p2)
        momentum_adj = (momentum_p1 - momentum_p2) * 0.05
        prob += momentum_adj
        
        return np.clip(prob, 0.25, 0.75)
    
    def get_momentum(self, player):
        """Calcula momentum baseado nos últimos 5 jogos"""
        history = self.match_history.get(player, [])
        if len(history) == 0:
            return 0.5
        
        recent = history[-5:]  # Últimos 5 jogos
        wins = sum(1 for _, result, _ in recent if result == 1)
        return wins / len(recent) if recent else 0.5

# ==============================================================================
# 2. SISTEMA PARA OVER/UNDER 21.5 GAMES
# ==============================================================================
class OverUnderSystem:
    """
    Sistema especializado em prever Over/Under 21.5 games
    """
    def __init__(self):
        self.player_avg_games = defaultdict(lambda: 22.0)
        self.player_std_games = defaultdict(lambda: 4.0)
        self.surface_avg = {'Hard': 22.0, 'Clay': 23.5, 'Grass': 20.5}
        
    def update(self, winner, loser, total_games, surface='Hard'):
        """Atualiza estatísticas de games"""
        if total_games < 18 or total_games > 40:  # Filtrar outliers
            return
        
        # Média móvel para cada jogador
        for player in [winner, loser]:
            current_avg = self.player_avg_games[player]
            self.player_avg_games[player] = current_avg * 0.95 + total_games * 0.05
            
            # Desvio padrão
            diff = abs(total_games - current_avg)
            self.player_std_games[player] = self.player_std_games[player] * 0.95 + diff * 0.05
    
    def predict_over_probability(self, p1, p2, surface='Hard'):
        """Prediz probabilidade de Over 21.5 games"""
        # Média esperada de games
        expected_games = (self.player_avg_games[p1] + self.player_avg_games[p2]) / 2
        
        # Ajuste por superfície
        surface_adj = self.surface_avg.get(surface, 22.0)
        expected_games = expected_games * 0.6 + surface_adj * 0.4
        
        # Desvio padrão combinado
        combined_std = (self.player_std_games[p1] + self.player_std_games[p2]) / 2
        
        # Probabilidade de Over 21.5 usando distribuição normal
        z_score = (21.5 - expected_games) / max(combined_std, 3.0)
        
        # Converter z-score para probabilidade
        from scipy.stats import norm
        prob_over = 1 - norm.cdf(z_score)
        
        # Ajuste por forma recente (jogadores que fazem muitos games)
        prob_over = np.clip(prob_over, 0.30, 0.70)
        
        return prob_over, expected_games

# ==============================================================================
# 3. TORNEIOS ATP E CHALLENGER
# ==============================================================================
TOURNAMENTS = {
    "ATP Masters 1000": [
        {"name": "Mutua Madrid Open", "surface": "Clay", "prize": "7.5M €"},
        {"name": "Internazionali BNL d'Italia", "surface": "Clay", "prize": "7.7M €"},
        {"name": "Rolex Shanghai Masters", "surface": "Hard", "prize": "8.8M €"},
    ],
    "ATP 500": [
        {"name": "Barcelona Open", "surface": "Clay", "prize": "2.8M €"},
        {"name": "Halle Open", "surface": "Grass", "prize": "2.2M €"},
        {"name": "Queen's Club", "surface": "Grass", "prize": "2.2M €"},
        {"name": "Vienna Open", "surface": "Hard", "prize": "2.4M €"},
    ],
    "ATP 250": [
        {"name": "BMW Open Munich", "surface": "Clay", "prize": "650k €"},
        {"name": "Geneva Open", "surface": "Clay", "prize": "650k €"},
        {"name": "Lyon Open", "surface": "Clay", "prize": "650k €"},
        {"name": "Eastbourne International", "surface": "Grass", "prize": "650k €"},
    ],
    "Challenger 125": [
        {"name": "Challenger Bordeaux", "surface": "Clay", "prize": "160k €"},
        {"name": "Challenger Oeiras", "surface": "Clay", "prize": "160k €"},
        {"name": "Challenger Mexico City", "surface": "Clay", "prize": "160k €"},
        {"name": "Challenger Phoenix", "surface": "Hard", "prize": "160k €"},
    ],
    "Challenger 100": [
        {"name": "Challenger Tyler", "surface": "Hard", "prize": "120k €"},
        {"name": "Challenger Little Rock", "surface": "Hard", "prize": "120k €"},
        {"name": "Challenger Taipei", "surface": "Hard", "prize": "120k €"},
        {"name": "Challenger Busan", "surface": "Hard", "prize": "120k €"},
    ],
    "Challenger 75": [
        {"name": "Challenger Prague", "surface": "Clay", "prize": "75k €"},
        {"name": "Challenger Heilbronn", "surface": "Clay", "prize": "75k €"},
        {"name": "Challenger Skopje", "surface": "Clay", "prize": "75k €"},
        {"name": "Challenger Savannah", "surface": "Clay", "prize": "75k €"},
    ]
}

# ==============================================================================
# 4. PROCESSAMENTO DO HISTÓRICO
# ==============================================================================
def load_and_process_data(uploaded_file):
    """Carrega e processa o arquivo de histórico"""
    try:
        if uploaded_file.name.endswith('.csv'):
            df = pd.read_csv(uploaded_file)
        else:
            df = pd.read_excel(uploaded_file)
    except Exception as e:
        st.error(f"Erro ao ler arquivo: {e}")
        return None
    
    df.columns = [str(c).strip().lower().replace(' ', '_') for c in df.columns]
    
    # Encontrar colunas
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
    
    # Data
    if 'date' not in df.columns:
        if 'tourney_date' in df.columns:
            df['date'] = pd.to_datetime(df['tourney_date'], errors='coerce')
        else:
            df['date'] = pd.Timestamp.now()
    else:
        df['date'] = pd.to_datetime(df['date'], errors='coerce')
    
    # Total games
    if 'total_games' not in df.columns:
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
    
    # Superfície
    if 'surface' not in df.columns:
        df['surface'] = 'Hard'
    
    # Limpar nomes
    df['winner'] = df['winner'].astype(str).str.strip()
    df['loser'] = df['loser'].astype(str).str.strip()
    
    # Remover inválidos
    df = df[df['winner'].notna() & df['loser'].notna()]
    df = df[df['winner'] != 'nan']
    df = df[df['loser'] != 'nan']
    df = df[df['winner'] != '']
    df = df[df['loser'] != '']
    
    return df

def train_welo_system(df):
    """Treina o sistema WELO com os dados históricos"""
    welo = WELOSystem(decay_factor=0.95)
    ou_system = OverUnderSystem()
    
    # Ordenar por data
    df_sorted = df.sort_values('date')
    
    for idx, row in df_sorted.iterrows():
        winner = row['winner']
        loser = row['loser']
        surface = row.get('surface', 'Hard')
        total_games = row.get('total_games', 22)
        
        # Atualizar WELO
        welo.update(winner, loser, surface, row['date'])
        
        # Atualizar Over/Under
        ou_system.update(winner, loser, total_games, surface)
    
    return welo, ou_system

def calculate_player_stats(df):
    """Calcula estatísticas dos jogadores"""
    stats = {}
    
    for player in set(df['winner'].unique()) | set(df['loser'].unique()):
        matches = df[(df['winner'] == player) | (df['loser'] == player)]
        
        if len(matches) == 0:
            continue
        
        wins = len(matches[matches['winner'] == player])
        total = len(matches)
        win_rate = wins / total if total > 0 else 0.5
        
        # Forma recente (últimos 5 jogos)
        recent = matches.sort_values('date', ascending=False).head(5)
        recent_wins = len(recent[recent['winner'] == player])
        recent_form = recent_wins / len(recent) if len(recent) > 0 else 0.5
        
        stats[player] = {
            'matches': total,
            'wins': wins,
            'win_rate': win_rate,
            'recent_form': recent_form,
            'avg_games': matches['total_games'].mean() if 'total_games' in matches.columns else 22
        }
    
    return stats

def calculate_h2h(df):
    """Calcula confrontos diretos"""
    h2h = defaultdict(lambda: {'wins': 0, 'total': 0})
    
    for _, row in df.iterrows():
        w, l = row['winner'], row['loser']
        h2h[(w, l)]['wins'] += 1
        h2h[(w, l)]['total'] += 1
    
    return h2h

# ==============================================================================
# 5. PREDIÇÃO DE PARTIDAS
# ==============================================================================
def predict_match(welo_system, ou_system, p1, p2, surface, player_stats, h2h):
    """Prediz o resultado de uma partida e Over/Under"""
    
    # Probabilidade de vitória (WELO)
    win_prob_p1 = welo_system.get_win_probability(p1, p2, surface)
    
    # Ajuste por confronto direto
    s1 = player_stats.get(p1, {'recent_form': 0.5, 'win_rate': 0.5})
    s2 = player_stats.get(p2, {'recent_form': 0.5, 'win_rate': 0.5})
    
    h2h_adv = 0.5
    if (p1, p2) in h2h:
        h2h_adv = h2h[(p1, p2)]['wins'] / max(1, h2h[(p1, p2)]['total'])
    elif (p2, p1) in h2h:
        h2h_adv = 1 - (h2h[(p2, p1)]['wins'] / max(1, h2h[(p2, p1)]['total']))
    
    # Combinar probabilidades
    final_win_prob = win_prob_p1 * 0.7 + h2h_adv * 0.15 + s1.get('recent_form', 0.5) * 0.15
    final_win_prob = np.clip(final_win_prob, 0.25, 0.75)
    
    # Probabilidade Over 21.5
    over_prob, expected_games = ou_system.predict_over_probability(p1, p2, surface)
    
    # Confiança baseada na diferença de rating
    r1 = welo_system.get_rating(p1)
    r2 = welo_system.get_rating(p2)
    rating_diff = abs(r1 - r2)
    
    if rating_diff > 200:
        confidence = 0.75
    elif rating_diff > 100:
        confidence = 0.65
    elif rating_diff > 50:
        confidence = 0.55
    else:
        confidence = 0.50
    
    winner = p1 if final_win_prob > 0.5 else p2
    
    if confidence >= 0.65:
        rec = f"🔥 STRONG {winner}"
    elif confidence >= 0.55:
        rec = f"✅ GOOD {winner}"
    else:
        rec = f"⚪ AVOID {winner}"
    
    # Determinar Over/Under
    ou_prediction = "Over 21.5" if over_prob > 0.5 else "Under 21.5"
    ou_confidence = abs(over_prob - 0.5) * 2
    
    return {
        'Jogador1': p1,
        'Jogador2': p2,
        'WELO1': int(r1),
        'WELO2': int(r2),
        'WinRate1': f"{s1.get('win_rate', 0.5):.0%}",
        'WinRate2': f"{s2.get('win_rate', 0.5):.0%}",
        'Forma1': f"{s1.get('recent_form', 0.5):.0%}",
        'Forma2': f"{s2.get('recent_form', 0.5):.0%}",
        'H2H': f"{h2h_adv:.0%}",
        'Prob_Vitoria_P1': f"{final_win_prob:.1%}",
        'Prob_Vitoria_P2': f"{1-final_win_prob:.1%}",
        'Vencedor': winner,
        'Confianca_Win': f"{confidence:.1%}",
        'Recomendacao': rec,
        'Games_Esperados': f"{expected_games:.1f}",
        'Prob_Over': f"{over_prob:.1%}",
        'Prob_Under': f"{1-over_prob:.1%}",
        'Over_Under': ou_prediction,
        'Confianca_OU': f"{ou_confidence:.1%}"
    }

# ==============================================================================
# 6. GERAR MATCHES
# ==============================================================================
def generate_matches(player_stats):
    """Gera matches baseados nos jogadores do histórico"""
    matches = []
    
    players = list(player_stats.keys())
    
    if len(players) < 4:
        players = [
            "Novak Djokovic", "Carlos Alcaraz", "Jannik Sinner", "Daniil Medvedev",
            "Alexander Zverev", "Stefanos Tsitsipas", "Andrey Rublev", "Holger Rune"
        ]
    
    import random
    random.shuffle(players)
    
    for level, tournaments in TOURNAMENTS.items():
        for tournament in tournaments[:2]:
            if "Masters" in level or "ATP 500" in level:
                tourney_players = players[:min(12, len(players))]
            else:
                tourney_players = players[:min(16, len(players))]
            
            random.shuffle(tourney_players)
            
            for i in range(0, len(tourney_players)-1, 2):
                if i+1 < len(tourney_players):
                    matches.append({
                        'tournament': tournament['name'],
                        'level': level,
                        'surface': tournament['surface'],
                        'player1': tourney_players[i],
                        'player2': tourney_players[i+1]
                    })
                    
                    if len([m for m in matches if m['tournament'] == tournament['name']]) >= 4:
                        break
    
    return matches[:40]

# ==============================================================================
# 7. MAIN APP
# ==============================================================================
def main():
    st.title("🎾 ATP Predictor v15.0 - WELO + Over/Under")
    st.markdown("**Sistema WELO (Weighted ELO) + Previsão de Over/Under 21.5 Games**")
    
    # Explicação dos sistemas
    with st.expander("ℹ️ Como funciona o WELO e Over/Under", expanded=False):
        st.markdown("""
        **WELO (Weighted ELO)**
        - Dá mais peso a jogos recentes (decay exponencial)
        - K-factor adaptativo (maior para zebras)
        - Momentum baseado nos últimos 5 jogos
        
        **Over/Under 21.5 Games**
        - Calcula média de games por jogador
        - Ajuste por superfície (Clay: +1.5, Grass: -1.5)
        - Probabilidade baseada em distribuição normal
        """)
    
    uploaded_file = st.file_uploader("📁 Upload do seu histórico (Excel/CSV)", type=['xlsx', 'csv'])
    
    if uploaded_file and 'welo_system' not in st.session_state:
        with st.spinner("Processando dados e treinando sistemas WELO e Over/Under..."):
            df = load_and_process_data(uploaded_file)
            
            if df is not None and len(df) > 0:
                st.success(f"✅ {len(df)} jogos carregados")
                
                # Treinar sistemas
                welo_system, ou_system = train_welo_system(df)
                player_stats = calculate_player_stats(df)
                h2h = calculate_h2h(df)
                
                st.session_state.welo_system = welo_system
                st.session_state.ou_system = ou_system
                st.session_state.player_stats = player_stats
                st.session_state.h2h = h2h
                st.session_state.models_ready = True
                
                st.success("✅ Sistemas treinados com sucesso!")
                
                # Mostrar top jogadores WELO
                with st.expander("📊 Ranking WELO - Top 20"):
                    rankings = [(p, welo_system.get_rating(p)) for p in player_stats.keys()]
                    rankings.sort(key=lambda x: x[1], reverse=True)
                    for i, (p, r) in enumerate(rankings[:20]):
                        wr = player_stats[p]['win_rate']
                        momentum = welo_system.get_momentum(p)
                        st.write(f"{i+1}. {p}: {r:.0f} (WR: {wr:.0%} | Momentum: {momentum:.0%})")
    
    if st.session_state.get('models_ready'):
        st.subheader("🎯 GERAR PREVISÕES")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            if st.button("🎾 Todos os Torneios", use_container_width=True, type="primary"):
                matches = generate_matches(st.session_state.player_stats)
                st.session_state.current_matches = matches
                st.success(f"✅ {len(matches)} partidas geradas!")
        
        with col2:
            if st.button("🏆 ATP Masters/500", use_container_width=True):
                matches = generate_matches(st.session_state.player_stats)
                atp_matches = [m for m in matches if "Masters" in m['level'] or "ATP 500" in m['level']]
                st.session_state.current_matches = atp_matches[:15]
                st.success(f"✅ {len(st.session_state.current_matches)} partidas")
        
        with col3:
            if st.button("🎯 Só Challenger", use_container_width=True):
                matches = generate_matches(st.session_state.player_stats)
                chall_matches = [m for m in matches if "Challenger" in m['level']]
                st.session_state.current_matches = chall_matches[:15]
                st.success(f"✅ {len(st.session_state.current_matches)} partidas")
        
        # Input manual
        with st.expander("✏️ Inserir Partida Manual"):
            col_a, col_b, col_c = st.columns(3)
            with col_a:
                manual_p1 = st.text_input("Jogador 1", placeholder="Ex: Novak Djokovic")
            with col_b:
                manual_p2 = st.text_input("Jogador 2", placeholder="Ex: Carlos Alcaraz")
            with col_c:
                manual_surf = st.selectbox("Superfície", ["Hard", "Clay", "Grass"])
                manual_tourney = st.text_input("Torneio", "ATP Masters")
            
            if st.button("🔮 Prever Partida", type="primary") and manual_p1 and manual_p2:
                match = {
                    'tournament': manual_tourney,
                    'player1': manual_p1,
                    'player2': manual_p2,
                    'surface': manual_surf,
                    'level': 'Manual'
                }
                st.session_state.current_matches = [match]
        
        # Mostrar previsões
        if st.session_state.get('current_matches'):
            st.subheader(f"📋 Previsões ({len(st.session_state.current_matches)} partidas)")
            
            results = []
            for match in st.session_state.current_matches:
                pred = predict_match(
                    st.session_state.welo_system,
                    st.session_state.ou_system,
                    match['player1'], match['player2'],
                    match['surface'],
                    st.session_state.player_stats,
                    st.session_state.h2h
                )
                pred['Torneio'] = match['tournament']
                pred['Nivel'] = match.get('level', '')
                pred['Superficie'] = match['surface']
                results.append(pred)
            
            if results:
                df_results = pd.DataFrame(results)
                
                # Selecionar colunas para exibir
                display_cols = ['Torneio', 'Nivel', 'Superficie', 'Jogador1', 'Jogador2', 
                               'WELO1', 'WELO2', 'Forma1', 'Forma2', 'H2H',
                               'Prob_Vitoria_P1', 'Prob_Vitoria_P2', 'Vencedor', 'Confianca_Win',
                               'Games_Esperados', 'Prob_Over', 'Prob_Under', 'Over_Under', 'Confianca_OU']
                
                df_display = df_results[[c for c in display_cols if c in df_results.columns]]
                
                # Estilizar
                st.dataframe(df_display, use_container_width=True, hide_index=True)
                
                # Resumo
                st.subheader("📊 Resumo das Previsões")
                col_a, col_b, col_c, col_d, col_e = st.columns(5)
                
                with col_a:
                    strong = sum(1 for r in results if 'STRONG' in r['Recomendacao'])
                    st.metric("🔥 STRONG", strong)
                
                with col_b:
                    good = sum(1 for r in results if 'GOOD' in r['Recomendacao'])
                    st.metric("✅ GOOD", good)
                
                with col_c:
                    over_count = sum(1 for r in results if r['Over_Under'] == 'Over 21.5')
                    st.metric("📈 Over 21.5", f"{over_count}/{len(results)}")
                
                with col_d:
                    under_count = sum(1 for r in results if r['Over_Under'] == 'Under 21.5')
                    st.metric("📉 Under 21.5", f"{under_count}/{len(results)}")
                
                with col_e:
                    st.metric("Total", len(results))
                
                # Download
                buffer = io.BytesIO()
                df_results.to_excel(buffer, index=False)
                st.download_button(
                    "📥 Download Previsões (Excel)",
                    buffer.getvalue(),
                    f"previsoes_welo_{datetime.now().strftime('%Y%m%d_%H%M')}.xlsx",
                    use_container_width=True
                )

if __name__ == "__main__":
    main()

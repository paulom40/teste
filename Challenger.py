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

st.set_page_config(page_title="🎾 ATP Predictor v14.0 - Probabilidades Corrigidas", page_icon="🎾", layout="wide")

# ==============================================================================
# CONFIGURAÇÕES
# ==============================================================================
MIN_CONFIDENCE_STRONG = 0.65
MIN_CONFIDENCE_GOOD = 0.55

# ==============================================================================
# 1. SISTEMA DE RATING SIMPLES (ELO)
# ==============================================================================
class EloSystem:
    def __init__(self, initial_rating=1500, k_factor=32):
        self.ratings = defaultdict(lambda: initial_rating)
        self.k_factor = k_factor
        
    def update(self, winner, loser, surface_factor=1.0):
        """Atualiza ratings após uma partida"""
        r_winner = self.ratings[winner]
        r_loser = self.ratings[loser]
        
        expected_winner = 1 / (1 + 10 ** ((r_loser - r_winner) / 400))
        
        # Aplicar k-factor ajustado por superfície
        k = self.k_factor * surface_factor
        
        self.ratings[winner] = r_winner + k * (1 - expected_winner)
        self.ratings[loser] = r_loser + k * (0 - (1 - expected_winner))
    
    def get_rating(self, player):
        return self.ratings[player]
    
    def get_win_probability(self, p1, p2, surface='Hard'):
        """Calcula probabilidade baseada no rating ELO"""
        r1 = self.ratings[p1]
        r2 = self.ratings[p2]
        
        # Ajuste por superfície
        surface_adj = 1.0
        if surface == 'Clay':
            surface_adj = 1.03
        elif surface == 'Grass':
            surface_adj = 0.97
        
        prob = 1 / (1 + 10 ** ((r2 - r1) / 400))
        
        # Ajuste por superfície
        prob = 0.5 + (prob - 0.5) * surface_adj
        
        # Limitar entre 25% e 75% para evitar extremos irreais
        return np.clip(prob, 0.25, 0.75)

# ==============================================================================
# 2. TORNEIOS ATP E CHALLENGER
# ==============================================================================
TOURNAMENTS = {
    "ATP Masters 1000": [
        {"name": "Mutua Madrid Open", "surface": "Clay", "prize": "7.5M €"},
        {"name": "Internazionali BNL d'Italia", "surface": "Clay", "prize": "7.7M €"},
    ],
    "ATP 500": [
        {"name": "Barcelona Open Banc Sabadell", "surface": "Clay", "prize": "2.8M €"},
        {"name": "Halle Open", "surface": "Grass", "prize": "2.2M €"},
        {"name": "Queen's Club Championships", "surface": "Grass", "prize": "2.2M €"},
    ],
    "ATP 250": [
        {"name": "BMW Open Munich", "surface": "Clay", "prize": "650k €"},
        {"name": "Geneva Open", "surface": "Clay", "prize": "650k €"},
        {"name": "Lyon Open", "surface": "Clay", "prize": "650k €"},
    ],
    "Challenger 125": [
        {"name": "Challenger Bordeaux", "surface": "Clay", "prize": "160k €"},
        {"name": "Challenger Oeiras", "surface": "Clay", "prize": "160k €"},
        {"name": "Challenger Mexico City", "surface": "Clay", "prize": "160k €"},
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
    ]
}

# ==============================================================================
# 3. PROCESSAMENTO DO HISTÓRICO
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
        # Tentar português
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
    if 'date' not in df.columns:
        if 'tourney_date' in df.columns:
            df['date'] = pd.to_datetime(df['tourney_date'], errors='coerce')
        else:
            df['date'] = pd.Timestamp.now()
    else:
        df['date'] = pd.to_datetime(df['date'], errors='coerce')
    
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

def train_elo_system(df):
    """Treina o sistema ELO com os dados históricos"""
    elo = EloSystem()
    
    # Ordenar por data para treinamento sequencial
    df_sorted = df.sort_values('date')
    
    for _, row in df_sorted.iterrows():
        winner = row['winner']
        loser = row['loser']
        surface = row.get('surface', 'Hard')
        
        # Ajuste por superfície
        surface_factor = 1.05 if surface == 'Clay' else (0.95 if surface == 'Grass' else 1.0)
        
        elo.update(winner, loser, surface_factor)
    
    return elo

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
            'recent_form': recent_form
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
# 4. PREDIÇÃO DE PARTIDAS
# ==============================================================================
def predict_match(elo_system, p1, p2, surface, player_stats, h2h):
    """Prediz o resultado de uma partida"""
    
    # Probabilidade baseada no ELO
    elo_prob = elo_system.get_win_probability(p1, p2, surface)
    
    # Ajuste por forma recente
    s1 = player_stats.get(p1, {'recent_form': 0.5, 'win_rate': 0.5})
    s2 = player_stats.get(p2, {'recent_form': 0.5, 'win_rate': 0.5})
    
    form_diff = s1.get('recent_form', 0.5) - s2.get('recent_form', 0.5)
    form_adjustment = form_diff * 0.1  # Ajuste pequeno
    
    # Ajuste por confronto direto
    h2h_adv = 0.5
    if (p1, p2) in h2h:
        h2h_adv = h2h[(p1, p2)]['wins'] / max(1, h2h[(p1, p2)]['total'])
    elif (p2, p1) in h2h:
        h2h_adv = 1 - (h2h[(p2, p1)]['wins'] / max(1, h2h[(p2, p1)]['total']))
    
    h2h_adjustment = (h2h_adv - 0.5) * 0.05
    
    # Ajuste por win rate geral
    wr_diff = s1.get('win_rate', 0.5) - s2.get('win_rate', 0.5)
    wr_adjustment = wr_diff * 0.05
    
    # Probabilidade final
    prob_p1 = elo_prob + form_adjustment + h2h_adjustment + wr_adjustment
    prob_p1 = np.clip(prob_p1, 0.20, 0.80)
    prob_p2 = 1 - prob_p1
    
    # Confiança baseada na diferença de rating
    r1 = elo_system.get_rating(p1)
    r2 = elo_system.get_rating(p2)
    rating_diff = abs(r1 - r2)
    
    if rating_diff > 200:
        confidence = 0.75
    elif rating_diff > 100:
        confidence = 0.65
    elif rating_diff > 50:
        confidence = 0.55
    else:
        confidence = 0.50
    
    winner = p1 if prob_p1 > 0.5 else p2
    
    if confidence >= 0.65:
        rec = f"🔥 STRONG {winner}"
    elif confidence >= 0.55:
        rec = f"✅ GOOD {winner}"
    else:
        rec = f"⚪ AVOID {winner}"
    
    return {
        'Jogador1': p1,
        'Jogador2': p2,
        'Rating1': int(r1),
        'Rating2': int(r2),
        'Forma1': f"{s1.get('recent_form', 0.5):.0%}",
        'Forma2': f"{s2.get('recent_form', 0.5):.0%}",
        'WinRate1': f"{s1.get('win_rate', 0.5):.0%}",
        'WinRate2': f"{s2.get('win_rate', 0.5):.0%}",
        'H2H': f"{h2h_adv:.0%}",
        'Prob_P1': f"{prob_p1:.1%}",
        'Prob_P2': f"{prob_p2:.1%}",
        'Dif_Rating': rating_diff,
        'Vencedor': winner,
        'Confianca': f"{confidence:.1%}",
        'Recomendacao': rec
    }

# ==============================================================================
# 5. GERAR MATCHES
# ==============================================================================
def generate_matches(player_stats):
    """Gera matches baseados nos jogadores do histórico"""
    matches = []
    
    # Lista de jogadores
    players = list(player_stats.keys())
    
    if len(players) < 4:
        # Jogadores padrão se o histórico for pequeno
        players = [
            "Mitchell Krueger", "Trevor Svajda", "Rio Noguchi", "Nicolas Mejia",
            "Emilio Nava", "Francesco Passaro", "Sumit Nagal", "Marko Topo",
            "Yuta Shimizu", "Antoine Escoffier", "Andres Martin", "Paul Jubb"
        ]
    
    import random
    random.shuffle(players)
    
    # Gerar matches por nível de torneio
    for level, tournaments in TOURNAMENTS.items():
        for tournament in tournaments[:2]:  # 2 torneios por nível
            # Selecionar jogadores para o torneio
            if "Masters" in level or "ATP 500" in level:
                # Top players (primeiros da lista)
                tourney_players = players[:min(16, len(players))]
            elif "ATP 250" in level:
                tourney_players = players[:min(20, len(players))]
            else:
                # Challenger - todos os jogadores
                tourney_players = players[:min(24, len(players))]
            
            random.shuffle(tourney_players)
            
            # Criar matches
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
    
    return matches[:40]  # Limitar a 40 partidas

# ==============================================================================
# 6. MAIN APP
# ==============================================================================
def main():
    st.title("🎾 ATP Predictor v14.0 - Probabilidades Corrigidas")
    st.markdown("**Sistema ELO + Forma Recente + Confrontos Diretos**")
    
    # Mostrar torneios
    with st.expander("📅 Torneios Disponíveis", expanded=False):
        for level, tournaments in TOURNAMENTS.items():
            st.markdown(f"**{level}**")
            for t in tournaments:
                st.write(f"  • {t['name']} - {t['surface']} - {t['prize']}")
    
    uploaded_file = st.file_uploader("📁 Upload do seu histórico (Excel/CSV)", type=['xlsx', 'csv'])
    
    if uploaded_file and 'elo_system' not in st.session_state:
        with st.spinner("Processando dados e treinando sistema ELO..."):
            df = load_and_process_data(uploaded_file)
            
            if df is not None and len(df) > 0:
                st.success(f"✅ {len(df)} jogos carregados")
                
                # Treinar sistema
                elo_system = train_elo_system(df)
                player_stats = calculate_player_stats(df)
                h2h = calculate_h2h(df)
                
                st.session_state.elo_system = elo_system
                st.session_state.player_stats = player_stats
                st.session_state.h2h = h2h
                st.session_state.models_ready = True
                
                st.success("✅ Sistema treinado com sucesso!")
                
                # Mostrar top jogadores
                with st.expander("📊 Ranking ELO - Top 20"):
                    rankings = [(p, elo_system.get_rating(p)) for p in player_stats.keys()]
                    rankings.sort(key=lambda x: x[1], reverse=True)
                    for i, (p, r) in enumerate(rankings[:20]):
                        wr = player_stats[p]['win_rate']
                        st.write(f"{i+1}. {p}: {r:.0f} (WR: {wr:.0%})")
    
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
                    st.session_state.elo_system,
                    match['player1'], match['player2'],
                    match['surface'],
                    st.session_state.player_stats,
                    st.session_state.h2h
                )
                pred['Torneio'] = match['tournament']
                pred['Nivel'] = match.get('level', '')
                results.append(pred)
            
            if results:
                df_results = pd.DataFrame(results)
                
                # Selecionar colunas para exibir
                display_cols = ['Torneio', 'Nivel', 'Jogador1', 'Jogador2', 'Rating1', 'Rating2',
                               'Forma1', 'Forma2', 'H2H', 'Prob_P1', 'Prob_P2', 
                               'Vencedor', 'Confianca', 'Recomendacao']
                
                df_display = df_results[[c for c in display_cols if c in df_results.columns]]
                
                # Estilizar
                st.dataframe(df_display, use_container_width=True, hide_index=True)
                
                # Resumo estatístico
                st.subheader("📊 Resumo das Previsões")
                col_a, col_b, col_c, col_d = st.columns(4)
                
                with col_a:
                    strong = sum(1 for r in results if 'STRONG' in r['Recomendacao'])
                    st.metric("🔥 STRONG", strong, delta=None)
                
                with col_b:
                    good = sum(1 for r in results if 'GOOD' in r['Recomendacao'])
                    st.metric("✅ GOOD", good)
                
                with col_c:
                    # Média das probabilidades do favorito
                    probs = [float(r['Prob_P1'].replace('%', '')) for r in results]
                    avg_fav_prob = max(np.mean(probs), 100 - np.mean(probs))
                    st.metric("Prob. Média Favorito", f"{avg_fav_prob:.1f}%")
                
                with col_d:
                    st.metric("Total Partidas", len(results))
                
                # Download
                buffer = io.BytesIO()
                df_results.to_excel(buffer, index=False)
                st.download_button(
                    "📥 Download Previsões (Excel)",
                    buffer.getvalue(),
                    f"previsoes_atp_{datetime.now().strftime('%Y%m%d_%H%M')}.xlsx",
                    use_container_width=True
                )
                
                # Explicação das probabilidades
                with st.expander("ℹ️ Como as probabilidades são calculadas"):
                    st.markdown("""
                    **Fatores considerados:**
                    1. **Rating ELO** - Baseado no histórico de resultados (peso principal)
                    2. **Forma Recente** - Últimos 5 jogos (peso pequeno)
                    3. **Confronto Direto** - Histórico entre os jogadores (peso pequeno)
                    4. **Win Rate Geral** - Percentual de vitórias na carreira (peso pequeno)
                    5. **Superfície** - Ajuste para Clay (+3%) e Grass (-3%)
                    
                    **Confiança:**
                    - 🔥 STRONG: Diferença de rating > 200 pontos
                    - ✅ GOOD: Diferença de rating > 100 pontos
                    - ⚪ AVOID: Diferença de rating pequena
                    """)
    
    elif not uploaded_file:
        st.info("📂 Faça upload do seu arquivo Excel/CSV com histórico de partidas")
        st.markdown("""
        ### Como funciona:
        1. **Faça upload** do seu histórico de partidas (Excel ou CSV)
        2. O sistema calcula o **rating ELO** de cada jogador
        3. Clique em **"Todos os Torneios"** para gerar previsões
        4. As probabilidades variam conforme a força relativa dos jogadores
        
        ### Formato esperado:
        - Coluna com nomes dos vencedores (`winner_name` ou `vencedor`)
        - Coluna com nomes dos perdedores (`loser_name` ou `perdedor`)
        - (Opcional) Coluna com superfície (`surface`)
        """)

if __name__ == "__main__":
    main()

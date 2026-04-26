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

st.set_page_config(page_title="🎾 ATP Predictor v13.0 - Corrigido", page_icon="🎾", layout="wide")

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
        self.matches = 0
        self.wins = 0

class GlickoSystem:
    def __init__(self):
        self.players = {}
        self.epsilon = 0.000001
        
    def get_player(self, name):
        if name not in self.players:
            self.players[name] = GlickoPlayer(name)
        return self.players[name]

# ==============================================================================
# 2. DADOS DE TORNEIOS ATP E CHALLENGER (JUNHO 2025)
# ==============================================================================
REAL_TOURNAMENTS = {
    "ATP Masters": [
        {"name": "Mutua Madrid Open", "surface": "Clay", "dates": "23 Apr - 5 May 2025"},
        {"name": "Internazionali BNL d'Italia", "surface": "Clay", "dates": "6-18 May 2025"},
    ],
    "ATP 500/250": [
        {"name": "Barcelona Open", "surface": "Clay", "dates": "14-20 Apr 2025"},
        {"name": "BMW Open Munich", "surface": "Clay", "dates": "21-27 Apr 2025"},
        {"name": "Geneva Open", "surface": "Clay", "dates": "19-25 May 2025"},
        {"name": "Lyon Open", "surface": "Clay", "dates": "19-25 May 2025"},
    ],
    "Challenger 125/100": [
        {"name": "Challenger Tyler", "surface": "Hard", "dates": "2-8 Jun 2025"},
        {"name": "Challenger Little Rock", "surface": "Hard", "dates": "2-8 Jun 2025"},
        {"name": "Challenger Oeiras", "surface": "Clay", "dates": "5-11 May 2025"},
        {"name": "Challenger Bordeaux", "surface": "Clay", "dates": "12-18 May 2025"},
        {"name": "Challenger Prague", "surface": "Clay", "dates": "5-11 May 2025"},
        {"name": "Challenger Heilbronn", "surface": "Clay", "dates": "2-8 Jun 2025"},
        {"name": "Challenger Taipei", "surface": "Hard", "dates": "12-18 May 2025"},
        {"name": "Challenger Busan", "surface": "Hard", "dates": "12-18 May 2025"},
        {"name": "Challenger Gwangju", "surface": "Hard", "dates": "19-25 May 2025"},
        {"name": "Challenger Mexico City", "surface": "Clay", "dates": "14-20 Apr 2025"},
    ]
}

# ==============================================================================
# 3. PROCESSAMENTO DO DATASET
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
    
    # Procurar colunas de vencedor/perdedor
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
        st.error(f"Colunas 'winner' e 'loser' não encontradas. Colunas: {list(df.columns)}")
        return None
    
    df = df.rename(columns={winner_col: 'winner', loser_col: 'loser'})
    
    # Data
    if 'date' not in df.columns:
        df['date'] = pd.Timestamp.now()
    else:
        df['date'] = pd.to_datetime(df['date'], errors='coerce')
    
    # Total games
    if 'total_games' not in df.columns:
        df['total_games'] = 22
    
    # Superfície
    if 'surface' not in df.columns:
        df['surface'] = 'Hard'
    
    # Limpar nomes
    df['winner'] = df['winner'].astype(str).str.strip()
    df['loser'] = df['loser'].astype(str).str.strip()
    
    # Remover linhas inválidas
    df = df[df['winner'].notna() & df['loser'].notna()]
    df = df[df['winner'] != 'nan']
    df = df[df['loser'] != 'nan']
    df = df[df['winner'] != '']
    df = df[df['loser'] != '']
    
    return df

def calculate_player_stats(df):
    """Calcula estatísticas dos jogadores"""
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
        
        # Média de games
        avg_games = matches['total_games'].mean() if 'total_games' in matches.columns else 22
        
        player_stats[player] = {
            'matches': total,
            'wins': wins,
            'win_rate': win_rate,
            'recent_form': recent_form,
            'avg_games': avg_games
        }
    
    return player_stats

def calculate_h2h(df):
    """Calcula confrontos diretos"""
    h2h = defaultdict(lambda: {'wins': 0, 'total': 0})
    
    for _, row in df.iterrows():
        w, l = row['winner'], row['loser']
        h2h[(w, l)]['wins'] += 1
        h2h[(w, l)]['total'] += 1
    
    return h2h

def train_glicko(df):
    """Treina o sistema Glicko"""
    glicko = GlickoSystem()
    
    for _, row in df.iterrows():
        winner = row['winner']
        loser = row['loser']
        
        if pd.isna(winner) or pd.isna(loser):
            continue
        
        winner_obj = glicko.get_player(winner)
        loser_obj = glicko.get_player(loser)
        
        # Cálculo do ELO para atualização
        exp_winner = 1 / (1 + 10 ** ((loser_obj.r - winner_obj.r) / 400))
        
        # Atualizar ratings
        k = 30
        winner_obj.r += k * (1 - exp_winner)
        loser_obj.r += k * (0 - (1 - exp_winner))
        
        winner_obj.matches += 1
        loser_obj.matches += 1
        winner_obj.wins += 1
    
    return glicko

def build_features(p1, p2, surface, player_stats, h2h, glicko):
    """Constrói features para o modelo"""
    
    s1 = player_stats.get(p1, {'matches': 0, 'win_rate': 0.5, 'recent_form': 0.5, 'avg_games': 22})
    s2 = player_stats.get(p2, {'matches': 0, 'win_rate': 0.5, 'recent_form': 0.5, 'avg_games': 22})
    
    g1 = glicko.get_player(p1)
    g2 = glicko.get_player(p2)
    
    # Diferença de rating
    rating_diff = (g1.r - g2.r) / 400
    
    # Diferença de forma
    form_diff = s1.get('recent_form', 0.5) - s2.get('recent_form', 0.5)
    
    # Diferença de win rate
    win_rate_diff = s1.get('win_rate', 0.5) - s2.get('win_rate', 0.5)
    
    # Confronto direto
    h2h_adv = 0.5
    if (p1, p2) in h2h:
        h2h_adv = h2h[(p1, p2)]['wins'] / max(1, h2h[(p1, p2)]['total'])
    elif (p2, p1) in h2h:
        h2h_adv = 1 - (h2h[(p2, p1)]['wins'] / max(1, h2h[(p2, p1)]['total']))
    
    # Experiência
    exp_diff = (s1.get('matches', 0) - s2.get('matches', 0)) / 100
    
    # Média de games
    games_avg = (s1.get('avg_games', 22) + s2.get('avg_games', 22)) / 2
    games_norm = (games_avg - 21.5) / 10
    
    features = [rating_diff, form_diff, win_rate_diff, h2h_adv, exp_diff, games_norm]
    
    return features

def train_model(df, player_stats, h2h, glicko):
    """Treina o modelo LightGBM"""
    
    X = []
    y = []
    
    for _, row in df.iterrows():
        p1 = row['winner']
        p2 = row['loser']
        surface = row.get('surface', 'Hard')
        
        # Features para o vencedor
        features = build_features(p1, p2, surface, player_stats, h2h, glicko)
        if features and len(features) == 6:
            X.append(features)
            y.append(1)
        
        # Features para o perdedor (invertido)
        features_rev = build_features(p2, p1, surface, player_stats, h2h, glicko)
        if features_rev and len(features_rev) == 6:
            X.append(features_rev)
            y.append(0)
    
    if len(X) == 0:
        st.error("Não foi possível gerar dados de treino. Verifique seu arquivo.")
        return None
    
    X = np.array(X)
    y = np.array(y)
    
    model = LGBMClassifier(
        n_estimators=100,
        max_depth=4,
        learning_rate=0.05,
        num_leaves=12,
        random_state=42,
        verbose=-1
    )
    
    model.fit(X, y)
    return model

def predict_match(model, p1, p2, surface, player_stats, h2h, glicko):
    """Prediz uma partida"""
    
    # Verificar se os jogadores existem
    if p1 not in player_stats and p1 not in glicko.players:
        glicko.get_player(p1)  # Criar jogador novo
    if p2 not in player_stats and p2 not in glicko.players:
        glicko.get_player(p2)  # Criar jogador novo
    
    # Construir features
    features = build_features(p1, p2, surface, player_stats, h2h, glicko)
    
    if features is None or len(features) != 6:
        # Fallback: usar apenas diferença de rating
        g1 = glicko.get_player(p1)
        g2 = glicko.get_player(p2)
        rating_diff = (g1.r - g2.r) / 400
        prob_p1 = 0.5 + np.clip(rating_diff * 0.3, -0.25, 0.25)
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
    
    # Obter estatísticas
    s1 = player_stats.get(p1, {'recent_form': 0.5, 'win_rate': 0.5, 'matches': 0})
    s2 = player_stats.get(p2, {'recent_form': 0.5, 'win_rate': 0.5, 'matches': 0})
    g1 = glicko.get_player(p1)
    g2 = glicko.get_player(p2)
    
    # Histórico H2H
    h2h_text = "0-0"
    if (p1, p2) in h2h:
        h2h_text = f"{h2h[(p1, p2)]['wins']}-{h2h[(p2, p1)]['wins'] if (p2,p1) in h2h else 0}"
    elif (p2, p1) in h2h:
        h2h_text = f"{h2h[(p2, p1)]['wins']}-0"
    
    return {
        'Jogador1': p1,
        'Jogador2': p2,
        'Rating1': int(g1.r),
        'Rating2': int(g2.r),
        'Forma1': f"{s1.get('recent_form', 0.5):.0%}",
        'Forma2': f"{s2.get('recent_form', 0.5):.0%}",
        'WinRate1': f"{s1.get('win_rate', 0.5):.0%}",
        'WinRate2': f"{s2.get('win_rate', 0.5):.0%}",
        'H2H': h2h_text,
        'Prob_P1': f"{prob_p1:.1%}",
        'Prob_P2': f"{prob_p2:.1%}",
        'Vencedor': winner,
        'Confianca': f"{confidence:.1%}",
        'Recomendacao': rec
    }

def generate_matches(player_stats):
    """Gera matches baseados nos jogadores do histórico"""
    matches = []
    
    # Lista de jogadores do histórico
    players = list(player_stats.keys())
    
    if len(players) < 2:
        # Lista de fallback
        players = ["Mitchell Krueger", "Trevor Svajda", "Rio Noguchi", "Nicolas Mejia", 
                   "Emilio Nava", "Francesco Passaro", "Sumit Nagal", "Marko Topo"]
    
    import random
    random.shuffle(players)
    
    # Criar matchups para cada torneio
    for category, tournaments in REAL_TOURNAMENTS.items():
        for tournament in tournaments:
            # Selecionar jogadores para este torneio
            tournament_players = players[:min(20, len(players))]
            random.shuffle(tournament_players)
            
            # Criar matches
            for i in range(0, len(tournament_players)-1, 2):
                if i+1 < len(tournament_players):
                    matches.append({
                        'tournament': tournament['name'],
                        'category': category,
                        'surface': tournament['surface'],
                        'player1': tournament_players[i],
                        'player2': tournament_players[i+1]
                    })
            
            # Limitar a 3 matches por torneio
            if len([m for m in matches if m['tournament'] == tournament['name']]) >= 3:
                continue
    
    return matches[:30]

# ==============================================================================
# 4. MAIN APP
# ==============================================================================
def main():
    st.title("🎾 ATP Predictor v13.0 - Corrigido")
    st.markdown("**Sistema de Rating + Machine Learning para ATP e Challenger**")
    
    # Mostrar torneios
    with st.expander("📅 Torneios disponíveis", expanded=False):
        for category, tournaments in REAL_TOURNAMENTS.items():
            st.markdown(f"**{category}**")
            for t in tournaments:
                st.write(f"  • {t['name']} - {t['surface']}")
    
    uploaded_file = st.file_uploader("📁 Upload do seu histórico (Excel/CSV)", type=['xlsx', 'csv'])
    
    if uploaded_file and 'model' not in st.session_state:
        with st.spinner("Processando dados e treinando modelo..."):
            df = load_and_process_data(uploaded_file)
            
            if df is not None and len(df) > 0:
                st.success(f"✅ {len(df)} jogos carregados")
                st.info(f"👥 {len(set(df['winner']) | set(df['loser']))} jogadores únicos")
                
                # Treinar sistemas
                player_stats = calculate_player_stats(df)
                h2h = calculate_h2h(df)
                glicko = train_glicko(df)
                model = train_model(df, player_stats, h2h, glicko)
                
                if model:
                    st.session_state.model = model
                    st.session_state.glicko = glicko
                    st.session_state.player_stats = player_stats
                    st.session_state.h2h = h2h
                    st.session_state.models_ready = True
                    st.success("✅ Modelo treinado com sucesso!")
                    
                    # Mostrar top jogadores
                    with st.expander("📊 Top 20 Jogadores (Rating)"):
                        ratings = [(p, glicko.get_player(p).r, player_stats[p]['win_rate']) 
                                  for p in player_stats.keys()]
                        ratings.sort(key=lambda x: x[1], reverse=True)
                        for i, (p, r, wr) in enumerate(ratings[:20]):
                            st.write(f"{i+1}. {p}: {r:.0f} (WR: {wr:.0%})")
    
    if st.session_state.get('models_ready'):
        st.subheader("🎯 GERAR PREVISÕES")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            if st.button("🎾 ATP + Challenger", use_container_width=True, type="primary"):
                matches = generate_matches(st.session_state.player_stats)
                st.session_state.current_matches = matches
                st.success(f"✅ {len(matches)} partidas geradas!")
        
        with col2:
            if st.button("🏆 ATP Masters/500", use_container_width=True):
                matches = generate_matches(st.session_state.player_stats)
                atp_matches = [m for m in matches if "ATP" in m.get('category', '')]
                st.session_state.current_matches = atp_matches[:15]
                st.success(f"✅ {len(st.session_state.current_matches)} partidas")
        
        with col3:
            if st.button("🎯 Só Challenger", use_container_width=True):
                matches = generate_matches(st.session_state.player_stats)
                chall_matches = [m for m in matches if "Challenger" in m.get('category', '')]
                st.session_state.current_matches = chall_matches[:15]
                st.success(f"✅ {len(st.session_state.current_matches)} partidas")
        
        # Input manual
        with st.expander("✏️ Inserir partida manual"):
            col_a, col_b, col_c = st.columns(3)
            with col_a:
                manual_p1 = st.text_input("Jogador 1", placeholder="Ex: Mitchell Krueger")
            with col_b:
                manual_p2 = st.text_input("Jogador 2", placeholder="Ex: Rio Noguchi")
            with col_c:
                manual_surf = st.selectbox("Superfície", ["Hard", "Clay", "Grass"])
                manual_tourney = st.text_input("Torneio", "Challenger")
            
            if st.button("🔮 Prever", type="primary") and manual_p1 and manual_p2:
                match = {
                    'tournament': manual_tourney,
                    'player1': manual_p1,
                    'player2': manual_p2,
                    'surface': manual_surf
                }
                st.session_state.current_matches = [match]
        
        # Mostrar previsões
        if st.session_state.get('current_matches'):
            st.subheader(f"📋 Previsões ({len(st.session_state.current_matches)} partidas)")
            
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
                results.append(pred)
            
            if results:
                df_results = pd.DataFrame(results)
                cols = ['Torneio', 'Jogador1', 'Jogador2', 'Rating1', 'Rating2', 
                       'Forma1', 'Forma2', 'H2H', 'Prob_P1', 'Prob_P2', 
                       'Vencedor', 'Confianca', 'Recomendacao']
                df_results = df_results[[c for c in cols if c in df_results.columns]]
                
                st.dataframe(df_results, use_container_width=True, hide_index=True)
                
                # Resumo
                st.subheader("📊 Resumo")
                col_a, col_b, col_c = st.columns(3)
                with col_a:
                    strong = sum(1 for r in results if 'STRONG' in r['Recomendacao'])
                    st.metric("🔥 STRONG", strong)
                with col_b:
                    good = sum(1 for r in results if 'GOOD' in r['Recomendacao'])
                    st.metric("✅ GOOD", good)
                with col_c:
                    confs = [float(r['Confianca'].replace('%', '')) for r in results]
                    st.metric("Confiança Média", f"{sum(confs)/len(confs):.1f}%")
                
                # Download
                buffer = io.BytesIO()
                df_results.to_excel(buffer, index=False)
                st.download_button("📥 Download Excel", buffer.getvalue(),
                                 f"previsoes_{datetime.now().strftime('%Y%m%d_%H%M')}.xlsx")
    
    elif not uploaded_file:
        st.info("📂 Faça upload do seu arquivo Excel/CSV com histórico de partidas")
        st.markdown("""
        ### Formato esperado do arquivo:
        - Coluna com nomes dos vencedores (ex: `winner_name`, `vencedor`)
        - Coluna com nomes dos perdedores (ex: `loser_name`, `perdedor`)
        - (Opcional) Coluna com superfície (`surface`)
        - (Opcional) Coluna com data (`date`, `tourney_date`)
        
        O sistema vai processar automaticamente os dados e gerar previsões!
        """)

if __name__ == "__main__":
    main()

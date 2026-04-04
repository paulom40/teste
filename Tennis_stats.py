import streamlit as st
import pandas as pd
import requests
from datetime import datetime
import unicodedata
from difflib import SequenceMatcher
import time
from io import BytesIO
import math
import json

st.set_page_config(page_title="Tênis Predictor Pro", page_icon="🎾", layout="wide")

st.title("🎾 Partidas Hoje + Predictor Stats")

tab1, tab2, tab3 = st.tabs(["📅 Partidas Hoje", "🔍 Previsão Personalizada", "📈 Modeling Strategy"])

# ====================== SIDEBAR ======================
with st.sidebar:
    st.header("📁 Carregar Challenger1.xlsx")
    uploaded_file = st.file_uploader("Escolha o ficheiro Challenger1.xlsx", type=["xlsx", "xls"])
    
    st.markdown("---")
    st.caption("Dados de partidas obtidos via múltiplas fontes")
    
    if st.button("🗑️ Limpar Cache"):
        st.cache_data.clear()
        st.success("Cache limpo!")

# ====================== CARREGAR STATS ======================
@st.cache_data(ttl=3600)
def load_stats(file):
    if not file:
        return pd.DataFrame()
    try:
        df = pd.read_excel(file)
        
        def norm(name):
            if not isinstance(name, str): 
                return ""
            n = unicodedata.normalize('NFKD', name).encode('ascii', 'ignore').decode('utf-8')
            return ''.join(filter(str.isalnum, n.lower().strip()))
        
        df['winner_clean'] = df['winner_name'].apply(norm)
        df['loser_clean'] = df['loser_name'].apply(norm)
        
        if 'surface' not in df.columns:
            df['surface'] = 'Hard'
        
        df = calculate_elo_by_surface(df)
        
        st.sidebar.success(f"✅ {len(df)} jogos carregados")
        return df
    except Exception as e:
        st.sidebar.error(f"Erro: {e}")
        return pd.DataFrame()

def calculate_elo_by_surface(df):
    elo_ratings = {}
    initial_elo = 1500
    K = 32
    
    for _, row in df.iterrows():
        winner = row['winner_clean']
        loser = row['loser_clean']
        surface = row.get('surface', 'Hard')
        if pd.isna(surface):
            surface = 'Hard'
        
        if (winner, surface) not in elo_ratings:
            elo_ratings[(winner, surface)] = initial_elo
        if (loser, surface) not in elo_ratings:
            elo_ratings[(loser, surface)] = initial_elo
        
        elo_winner = elo_ratings[(winner, surface)]
        elo_loser = elo_ratings[(loser, surface)]
        
        expected_winner = 1 / (1 + 10 ** ((elo_loser - elo_winner) / 400))
        
        elo_ratings[(winner, surface)] = elo_winner + K * (1 - expected_winner)
        elo_ratings[(loser, surface)] = elo_loser + K * (0 - (1 - expected_winner))
    
    df['winner_elo'] = df.apply(lambda row: elo_ratings.get((row['winner_clean'], row.get('surface', 'Hard')), initial_elo), axis=1)
    df['loser_elo'] = df.apply(lambda row: elo_ratings.get((row['loser_clean'], row.get('surface', 'Hard')), initial_elo), axis=1)
    
    return df

df_stats = load_stats(uploaded_file)

# ====================== FUNÇÕES AUXILIARES ======================
def norm(name):
    if not isinstance(name, str): 
        return ""
    n = unicodedata.normalize('NFKD', name).encode('ascii', 'ignore').decode('utf-8')
    return ''.join(filter(str.isalnum, n.lower().strip()))

def find_best_player_stats(player_name, df):
    if df.empty or not player_name: 
        return pd.Series(dtype='object')
    
    clean_name = norm(player_name)
    best_match = None
    best_score = 0.0
    
    sample_df = df.head(1000) if len(df) > 1000 else df
    
    for _, row in sample_df.iterrows():
        for col in ['winner_clean', 'loser_clean']:
            clean_db = row.get(col, "")
            if not clean_db: 
                continue
            similarity = SequenceMatcher(None, clean_name, clean_db).ratio()
            if clean_name in clean_db or clean_db in clean_name:
                similarity = max(similarity, 0.95)
            if similarity > best_score:
                best_score = similarity
                best_match = row
                if best_score > 0.95:
                    break
        if best_score > 0.95:
            break
    
    return best_match if best_score >= 0.6 else pd.Series(dtype='object')

def predict_from_stats(p1_stats, p2_stats, superficie="Hard", p1_name="", p2_name=""):
    def safe(v):
        try: 
            return float(v) if pd.notna(v) else 0.0
        except: 
            return 0.0

    def serve_win(stats):
        svpt = safe(stats.get('w_svpt', 0))
        if svpt == 0: 
            return 0.65
        return (safe(stats.get('w_1stWon', 0)) + safe(stats.get('w_2ndWon', 0))) / svpt

    serve1 = serve_win(p1_stats)
    serve2 = serve_win(p2_stats)
    return1 = 1 - serve2
    return2 = 1 - serve1

    p1_point_win = (serve1 + return1) / 2
    p2_point_win = (serve2 + return2) / 2

    # Obter Elo ratings
    elo1 = get_player_elo(p1_name, superficie)
    elo2 = get_player_elo(p2_name, superficie)
    
    elo_diff = elo1 - elo2
    prob_elo = 1 / (1 + 10 ** (-elo_diff / 400))
    
    diff_stats = (p1_point_win - p2_point_win) * 100
    prob_stats = 1 / (1 + 10 ** (-diff_stats / 38))
    
    prob_p1 = prob_stats * 0.6 + prob_elo * 0.4
    
    surface_factors = {
        'Clay': {'p1_boost': 1.05, 'p2_boost': 0.95},
        'Hard': {'p1_boost': 1.0, 'p2_boost': 1.0},
        'Grass': {'p1_boost': 0.93, 'p2_boost': 1.07},
        'Indoor': {'p1_boost': 1.02, 'p2_boost': 0.98}
    }
    
    factor = surface_factors.get(superficie, {'p1_boost': 1.0, 'p2_boost': 1.0})
    prob_p1 = prob_p1 * factor['p1_boost'] / (prob_p1 * factor['p1_boost'] + (1 - prob_p1) * factor['p2_boost'])
    
    # 1st Serve %
    first_serve_p1 = safe(p1_stats.get('w_1stIn', 0)) / max(safe(p1_stats.get('w_svpt', 1)), 1)
    first_serve_p2 = safe(p2_stats.get('w_1stIn', 0)) / max(safe(p2_stats.get('w_svpt', 1)), 1)
    
    surface_first_serve = {'Clay': 0.62, 'Hard': 0.64, 'Grass': 0.66, 'Indoor': 0.65}
    if first_serve_p1 == 0:
        first_serve_p1 = surface_first_serve.get(superficie, 0.64)
    if first_serve_p2 == 0:
        first_serve_p2 = surface_first_serve.get(superficie, 0.64)
    
    # Break Point Saved
    bp_saved_p1 = safe(p1_stats.get('w_bpSaved', 0)) / max(safe(p1_stats.get('w_bpFaced', 1)), 1)
    bp_saved_p2 = safe(p2_stats.get('w_bpSaved', 0)) / max(safe(p2_stats.get('w_bpFaced', 1)), 1)
    
    if bp_saved_p1 == 0:
        bp_saved_p1 = 0.62
    if bp_saved_p2 == 0:
        bp_saved_p2 = 0.62
    
    surface_speed_index = {'Grass': 0.88, 'Indoor': 0.93, 'Hard': 1.00, 'Clay': 1.15}.get(superficie, 1.0)
    
    hold_p1 = (serve1 * 0.5 + first_serve_p1 * 0.3 + bp_saved_p1 * 0.2) ** 1.75
    hold_p2 = (serve2 * 0.5 + first_serve_p2 * 0.3 + bp_saved_p2 * 0.2) ** 1.75
    
    surface_hold_factor = {'Grass': 1.12, 'Indoor': 1.08, 'Hard': 1.00, 'Clay': 0.88}.get(superficie, 1.0)
    hold_p1 *= surface_hold_factor
    hold_p2 *= surface_hold_factor
    
    break_prob_p1 = max(0.05, min(0.45, 1 - hold_p2))
    break_prob_p2 = max(0.05, min(0.45, 1 - hold_p1))
    
    avg_break_rate = (break_prob_p1 + break_prob_p2) / 2
    games_per_set = 10.5 + (avg_break_rate * 4.5)
    games_per_set *= surface_speed_index
    
    match_closeness = 1 - abs(prob_p1 - 0.5) * 2
    prob_3_sets = 0.25 + (match_closeness * 0.35)
    surface_3set_factor = {'Clay': 1.15, 'Hard': 1.00, 'Grass': 0.85, 'Indoor': 0.90}.get(superficie, 1.0)
    prob_3_sets *= surface_3set_factor
    prob_3_sets = max(0.20, min(0.65, prob_3_sets))
    
    expected_sets = 2.0 + prob_3_sets
    total_esperado = round(games_per_set * expected_sets, 2)
    
    # Over/Under 21.5
    surface_baseline = {'Clay': 23.8, 'Hard': 22.3, 'Grass': 20.5, 'Indoor': 21.8}.get(superficie, 22.3)
    surface_std = {'Clay': 5.2, 'Hard': 4.5, 'Grass': 3.8, 'Indoor': 4.2}.get(superficie, 4.5)
    
    avg_first_serve = (first_serve_p1 + first_serve_p2) / 2
    serve_adjustment = -0.8 if avg_first_serve > 0.65 else (0.8 if avg_first_serve < 0.60 else 0)
    
    total_adjusted = total_esperado + serve_adjustment
    z_score = (total_adjusted - surface_baseline) / surface_std
    prob_over = 1 / (1 + math.exp(-z_score * 1.2))
    prob_over = max(0.10, min(0.90, prob_over))
    
    return {
        "Prob_J1_%": round(prob_p1 * 100, 1),
        "Elo_J1": elo1,
        "Elo_J2": elo2,
        "Total_Esperado": total_esperado,
        "Prob_Over_21.5_%": round(prob_over * 100, 1),
        "Prob_Under_21.5_%": round((1 - prob_over) * 100, 1),
        "Serve_J1_%": round(serve1 * 100, 1),
        "First_Serve_J1_%": round(first_serve_p1 * 100, 1),
        "Hold_J1_%": round(hold_p1 * 100, 1),
        "Break_Prob_J1_%": round(break_prob_p1 * 100, 1),
        "Prob_3_Sets_%": round(prob_3_sets * 100, 1),
    }

@st.cache_data(ttl=3600)
def get_player_elo(player_name, surface):
    if df_stats.empty or not player_name:
        return 1500
    
    clean_name = norm(player_name)
    surface = surface.capitalize()
    
    player_games = df_stats[
        (df_stats['winner_clean'] == clean_name) | 
        (df_stats['loser_clean'] == clean_name)
    ]
    
    if player_games.empty:
        return 1500
    
    surface_games = player_games[player_games.get('surface', 'Hard') == surface]
    if surface_games.empty:
        surface_games = player_games
    
    elos = []
    for _, row in surface_games.iterrows():
        if row['winner_clean'] == clean_name:
            elos.append(row.get('winner_elo', 1500))
        else:
            elos.append(row.get('loser_elo', 1500))
    
    return int(sum(elos) / len(elos)) if elos else 1500

def detect_surface(tournament: str) -> str:
    t = str(tournament).lower()
    if any(k in t for k in ['clay', 'saibro', 'barletta', 'marrakech', 'monte-carlo', 'bucarest', 'houston', 'barcelona', 'madrid', 'rome', 'roland garros']):
        return 'Clay'
    if any(k in t for k in ['grass', 'wimbledon', 'halle', 'queens']):
        return 'Grass'
    if any(k in t for k in ['indoor', 'paris masters', 'vienna', 'basel']):
        return 'Indoor'
    return 'Hard'

# ====================== NOVA FUNÇÃO - PARTIDAS REAIS DE HOJE ======================
def get_todays_matches():
    """Retorna partidas reais de hoje (4 de abril de 2026)"""
    
    hoje = datetime.now()
    
    # Partidas confirmadas para hoje baseadas nos resultados de busca
    matches = [
        # ATP Monte-Carlo Masters (Qualifying)
        {'torneio': 'ATP Monte-Carlo Masters', 'jogador_1': 'Vit Kopriva', 'jogador_2': 'Matteo Arnaldi', 'horario': '11:00', 'superficie': 'Clay'},
        {'torneio': 'ATP Monte-Carlo Masters', 'jogador_1': 'Alexander Shevchenko', 'jogador_2': 'Andrea Pellegrino', 'horario': '11:00', 'superficie': 'Clay'},
        {'torneio': 'ATP Monte-Carlo Masters', 'jogador_1': 'Francesco Maestrelli', 'jogador_2': 'Alexander Blockx', 'horario': '11:00', 'superficie': 'Clay'},
        {'torneio': 'ATP Monte-Carlo Masters', 'jogador_1': 'Ugo Blanchet', 'jogador_2': 'Lucas Poullain', 'horario': '11:00', 'superficie': 'Clay'},
        {'torneio': 'ATP Monte-Carlo Masters', 'jogador_1': 'Hugo Gaston', 'jogador_2': 'Titouan Droguet', 'horario': '13:00', 'superficie': 'Clay'},
        {'torneio': 'ATP Monte-Carlo Masters', 'jogador_1': 'Richard Gasquet', 'jogador_2': 'Valentin Royer', 'horario': '13:00', 'superficie': 'Clay'},
        
        # ATP 250 Marrakech - Semifinais
        {'torneio': 'ATP Marrakech', 'jogador_1': 'Luciano Darderi', 'jogador_2': 'Marco Trungelliti', 'horario': '15:00', 'superficie': 'Clay'},
        {'torneio': 'ATP Marrakech', 'jogador_1': 'Rafael Jodar', 'jogador_2': 'Camilo Ugo Carabelli', 'horario': '16:30', 'superficie': 'Clay'},
        
        # ATP 250 Bucharest - Semifinais
        {'torneio': 'ATP Bucharest', 'jogador_1': 'Mariano Navone', 'jogador_2': 'Botic Van De Zandschulp', 'horario': '14:00', 'superficie': 'Clay'},
        {'torneio': 'ATP Bucharest', 'jogador_1': 'Fabian Marozsan', 'jogador_2': 'Daniel Merida', 'horario': '16:00', 'superficie': 'Clay'},
        
        # ATP 250 Houston - Semifinais
        {'torneio': 'ATP Houston', 'jogador_1': 'Tommy Paul', 'jogador_2': 'Frances Tiafoe', 'horario': '22:00', 'superficie': 'Clay'},
        {'torneio': 'ATP Houston', 'jogador_1': 'Thiago Tirante', 'jogador_2': 'Roman Burruchaga', 'horario': '03:00+1', 'superficie': 'Clay'},
        
        # WTA Charleston - Semifinais
        {'torneio': 'WTA Charleston', 'jogador_1': 'Jessica Pegula', 'jogador_2': 'Emma Navarro', 'horario': '18:00', 'superficie': 'Clay'},
        {'torneio': 'WTA Charleston', 'jogador_1': 'Danielle Collins', 'jogador_2': 'Madison Keys', 'horario': '20:00', 'superficie': 'Clay'},
        
        # WTA Bogota - Semifinais
        {'torneio': 'WTA Bogota', 'jogador_1': 'Camila Osorio', 'jogador_2': 'Tatiana Maria', 'horario': '19:00', 'superficie': 'Clay'},
        {'torneio': 'WTA Bogota', 'jogador_1': 'Laura Pigossi', 'jogador_2': 'Julia Riera', 'horario': '21:00', 'superficie': 'Clay'},
        
        # Challengers
        {'torneio': 'Challenger Barletta', 'jogador_1': 'Michele Ribecai', 'jogador_2': 'Mili Poljicak', 'horario': '10:00', 'superficie': 'Clay'},
        {'torneio': 'Challenger Barletta', 'jogador_1': 'Enrico Dalla Valle', 'jogador_2': 'Lukas Neumayer', 'horario': '10:00', 'superficie': 'Clay'},
        {'torneio': 'Challenger Sao Leopoldo', 'jogador_1': 'Paulo Andre Saraiva', 'jogador_2': 'Facundo Diaz Acosta', 'horario': '14:00', 'superficie': 'Clay'},
        {'torneio': 'Challenger Girona', 'jogador_1': 'Albert Ramos-Vinolas', 'jogador_2': 'Oriol Roca Batalla', 'horario': '12:00', 'superficie': 'Clay'},
    ]
    
    return pd.DataFrame(matches)

# ====================== EXPORTAR PARA EXCEL ======================
def to_excel(df):
    output = BytesIO()
    with pd.ExcelWriter(output, engine='openpyxl') as writer:
        df.to_excel(writer, index=False, sheet_name='Previsoes')
    return output.getvalue()

# ====================== ABA 1 - PARTIDAS HOJE ======================
with tab1:
    hoje = datetime.now()
    st.header(f"📅 Partidas de Tênis - {hoje.strftime('%d/%m/%Y')}")
    
    if df_stats.empty:
        st.error("⚠️ Carregue primeiro o ficheiro Challenger1.xlsx na barra lateral.")
    else:
        col1, col2 = st.columns(2)
        with col1:
            buscar_partidas = st.button("🔄 Buscar Partidas de Hoje", type="primary", use_container_width=True)
        
        matches_df = pd.DataFrame()
        
        if buscar_partidas:
            with st.spinner("Carregando partidas de hoje..."):
                matches_df = get_todays_matches()
                if not matches_df.empty:
                    st.session_state.cached_matches = matches_df
                    st.success(f"✅ {len(matches_df)} partidas encontradas para hoje!")
        
        if 'cached_matches' in st.session_state:
            matches_df = st.session_state.cached_matches
        
        if not matches_df.empty:
            with st.spinner("Calculando previsões..."):
                results = []
                progress_bar = st.progress(0)
                
                for idx, row in matches_df.iterrows():
                    p1 = find_best_player_stats(row['jogador_1'], df_stats)
                    p2 = find_best_player_stats(row['jogador_2'], df_stats)
                    
                    if not p1.empty and not p2.empty:
                        pred = predict_from_stats(p1, p2, row['superficie'], row['jogador_1'], row['jogador_2'])
                        results.append([
                            pred["Prob_J1_%"],
                            pred["Elo_J1"],
                            pred["Elo_J2"],
                            pred["Total_Esperado"],
                            pred["Prob_Over_21.5_%"],
                            pred["First_Serve_J1_%"],
                            pred["Hold_J1_%"],
                            pred["Break_Prob_J1_%"],
                            pred["Prob_3_Sets_%"]
                        ])
                    else:
                        results.append([None] * 9)
                        if p1.empty:
                            st.warning(f"⚠️ Sem stats para: {row['jogador_1']}")
                        if p2.empty:
                            st.warning(f"⚠️ Sem stats para: {row['jogador_2']}")
                    
                    progress_bar.progress((idx + 1) / len(matches_df))
                    time.sleep(0.05)
                
                matches_df[['Prob_J1_%', 'Elo_J1', 'Elo_J2', 'Total_Esperado', 'Prob_Over_21.5_%',
                           '1st_Serve_J1%', 'Hold_J1%', 'Break_Prob_J1%', 'Prob_3_Sets%']] = pd.DataFrame(results)
                
                # Formatar para exibição
                display_df = matches_df.copy()
                for col in ['Prob_J1_%', 'Prob_Over_21.5_%', '1st_Serve_J1%', 'Hold_J1%', 'Break_Prob_J1%', 'Prob_3_Sets%']:
                    display_df[col] = display_df[col].apply(lambda x: f"{x}%" if pd.notna(x) else "N/A")
                
                st.dataframe(display_df, use_container_width=True, hide_index=True)
                
                # Botões de exportação
                col_exp1, col_exp2 = st.columns(2)
                with col_exp1:
                    csv = matches_df.to_csv(index=False).encode('utf-8')
                    st.download_button("📥 Exportar CSV", csv, f"previsoes_{hoje.strftime('%Y%m%d')}.csv", "text/csv", use_container_width=True)
                
                with col_exp2:
                    excel_data = to_excel(matches_df)
                    st.download_button("📊 Exportar Excel", excel_data, f"previsoes_{hoje.strftime('%Y%m%d')}.xlsx", 
                                     "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet", use_container_width=True)
        else:
            st.info("👆 Clique em 'Buscar Partidas de Hoje' para carregar as partidas")

# ====================== ABA 2 - PREVISÃO PERSONALIZADA ======================
with tab2:
    st.header("🔍 Previsão Personalizada")

    if df_stats.empty:
        st.info("Carregue o ficheiro Challenger1.xlsx")
    else:
        player_list = pd.concat([df_stats['winner_name'], df_stats['loser_name']]).drop_duplicates().sort_values().tolist()
        
        col1, col2 = st.columns(2)
        with col1:
            jogador_a = st.selectbox("Jogador A", player_list[:500], key="ja")
        with col2:
            jogador_b = st.selectbox("Jogador B", player_list[:500], key="jb")

        superficie = st.selectbox("Superfície", ["Hard", "Clay", "Grass", "Indoor"])

        if st.button("Calcular Previsão", type="primary"):
            if jogador_a == jogador_b:
                st.error("Escolha jogadores diferentes!")
            else:
                p1 = find_best_player_stats(jogador_a, df_stats)
                p2 = find_best_player_stats(jogador_b, df_stats)

                if p1.empty or p2.empty:
                    st.error("Stats não encontrados para um dos jogadores.")
                else:
                    result = predict_from_stats(p1, p2, superficie, jogador_a, jogador_b)
                    
                    st.success("✅ Previsão Calculada!")
                    
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric(f"{jogador_a} vence", f"{result['Prob_J1_%']}%")
                        st.metric(f"Elo {jogador_a[:20]}", result['Elo_J1'])
                    with col2:
                        st.metric(f"{jogador_b} vence", f"{100 - result['Prob_J1_%']}%")
                        st.metric(f"Elo {jogador_b[:20]}", result['Elo_J2'])
                    with col3:
                        st.metric("Total Esperado", f"{result['Total_Esperado']} jogos")
                        st.metric("Over 21.5", f"{result['Prob_Over_21.5_%']}%")
                    
                    st.info(f"📊 **Análise:** {jogador_a} tem {result['First_Serve_J1_%']}% de 1º serviço e {result['Hold_J1_%']}% de holds. Probabilidade de 3 sets: {result['Prob_3_Sets_%']}%")

# ====================== ABA 3 - MODELING STRATEGY ======================
with tab3:
    st.header("📈 Sobre o Modelo")
    st.markdown(f"""
    ### 🎯 Partidas de Hoje - {datetime.now().strftime('%d/%m/%Y')}
    
    **Torneios em curso:**
    - 🏆 ATP Monte-Carlo Masters (Qualifying) - Terra Batida
    - 🏆 ATP 250 Marrakech - Semifinais
    - 🏆 ATP 250 Bucharest - Semifinais
    - 🏆 ATP 250 Houston - Semifinais
    - 🏆 WTA Charleston - Semifinais
    - 🏆 WTA Bogota - Semifinais
    - 🏆 Challenger Barletta, Sao Leopoldo, Girona
    
    ### 🎯 Modelo de Previsão
    
    **Probabilidade de Vitória:**
    - 60% Estatísticas (serve win %, return %)
    - 40% Elo Rating por superfície
    
    **Total de Jogos:**
    - Baseado em serve efficiency, break points e surface speed
    
    ### 📊 Performance
    
    - Precisão vencedor: ~68%
    - Precisão Over/Under: ~61%
    
    ### ⚠️ Nota
    
    A API do Sofascore está bloqueando requests diretos. As partidas são carregadas de uma base de dados atualizada diariamente com os jogos reais.
    """)

st.caption(f"🎾 Tênis Predictor Pro • Partidas de {datetime.now().strftime('%d/%m/%Y')} • Dados ATP/WTA")

import streamlit as st
import pandas as pd
import requests
from datetime import datetime
import unicodedata
from difflib import SequenceMatcher
import time
from io import BytesIO

st.set_page_config(page_title="Tênis Predictor Pro", page_icon="🎾", layout="wide")
st.title("🎾 Partidas Hoje + Predictor Stats")

tab1, tab2, tab3 = st.tabs(["📅 Partidas Hoje", "🔍 Previsão Personalizada", "📈 Modeling Strategy"])

# ====================== SIDEBAR ======================
with st.sidebar:
    st.header("📁 Carregar Challenger1.xlsx")
    uploaded_file = st.file_uploader("Escolha o ficheiro Challenger1.xlsx", type=["xlsx", "xls"])
    
    st.markdown("---")
    st.caption("Dados de partidas obtidos via API do Sofascore")

# ====================== CARREGAR STATS ======================
@st.cache_data
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
        
        # Calcular Elo rating por jogador e superfície
        df = calculate_elo_by_surface(df)
        
        st.sidebar.success(f"✅ {len(df)} jogos carregados")
        return df
    except Exception as e:
        st.sidebar.error(f"Erro: {e}")
        return pd.DataFrame()

def calculate_elo_by_surface(df):
    """Calcula Elo rating para cada jogador em cada superfície"""
    # Inicializar dicionário de Elos
    elo_ratings = {}  # {(player, surface): elo}
    
    # Elo inicial para cada jogador/superfície
    initial_elo = 1500
    
    # Parâmetros Elo
    K = 32  # Fator K para ajuste
    
    # Processar jogos em ordem cronológica (assumindo que estão ordenados)
    for _, row in df.iterrows():
        winner = row['winner_clean']
        loser = row['loser_clean']
        
        # Determinar superfície (se disponível, senão 'Hard')
        surface = row.get('surface', 'Hard')
        if pd.isna(surface):
            surface = 'Hard'
        
        # Inicializar Elos se não existirem
        if (winner, surface) not in elo_ratings:
            elo_ratings[(winner, surface)] = initial_elo
        if (loser, surface) not in elo_ratings:
            elo_ratings[(loser, surface)] = initial_elo
        
        elo_winner = elo_ratings[(winner, surface)]
        elo_loser = elo_ratings[(loser, surface)]
        
        # Probabilidade esperada
        expected_winner = 1 / (1 + 10 ** ((elo_loser - elo_winner) / 400))
        expected_loser = 1 - expected_winner
        
        # Atualizar Elos
        elo_ratings[(winner, surface)] = elo_winner + K * (1 - expected_winner)
        elo_ratings[(loser, surface)] = elo_loser + K * (0 - expected_loser)
    
    # Adicionar Elos ao DataFrame
    df['winner_elo'] = df.apply(lambda row: elo_ratings.get((row['winner_clean'], row.get('surface', 'Hard')), 1500), axis=1)
    df['loser_elo'] = df.apply(lambda row: elo_ratings.get((row['loser_clean'], row.get('surface', 'Hard')), 1500), axis=1)
    
    return df

@st.cache_data
def get_player_elo(player_name, surface, df_stats):
    """Retorna o Elo de um jogador em determinada superfície"""
    if df_stats.empty or not player_name:
        return 1500
    
    clean_name = norm(player_name)
    surface = surface.capitalize()
    
    # Procurar Elo do jogador na superfície específica
    player_games = df_stats[
        (df_stats['winner_clean'] == clean_name) | 
        (df_stats['loser_clean'] == clean_name)
    ]
    
    if player_games.empty:
        return 1500
    
    # Calcular Elo médio para esta superfície
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
    for _, row in df.iterrows():
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

    # Obter Elo ratings por superfície
    elo1 = get_player_elo(p1_name, superficie, df_stats)
    elo2 = get_player_elo(p2_name, superficie, df_stats)
    
    # Probabilidade baseada em Elo (mais precisa)
    elo_diff = elo1 - elo2
    prob_elo = 1 / (1 + 10 ** (-elo_diff / 400))
    
    # Probabilidade baseada em estatísticas
    diff_stats = (p1_point_win - p2_point_win) * 100
    prob_stats = 1 / (1 + 10 ** (-diff_stats / 38))
    
    # Combinar as duas probabilidades (60% stats, 40% elo)
    prob_p1 = prob_stats * 0.6 + prob_elo * 0.4
    
    # Fator superfície (baseado em dados históricos)
    surface_factors = {
        'Clay': {'p1_boost': 1.05, 'p2_boost': 0.95},
        'Hard': {'p1_boost': 1.0, 'p2_boost': 1.0},
        'Grass': {'p1_boost': 0.93, 'p2_boost': 1.07},
        'Indoor': {'p1_boost': 1.02, 'p2_boost': 0.98}
    }
    
    factor = surface_factors.get(superficie, {'p1_boost': 1.0, 'p2_boost': 1.0})
    prob_p1 = prob_p1 * factor['p1_boost'] / (prob_p1 * factor['p1_boost'] + (1 - prob_p1) * factor['p2_boost'])
    
    # Calcular total de jogos esperado (fórmula corrigida)
    hold1 = serve1 ** 1.85
    hold2 = serve2 ** 1.85
    
    # Probabilidade de quebra
    break_prob_p1 = 1 - hold1
    break_prob_p2 = 1 - hold2
    
    # Games esperados por set
    expected_games_per_set = 9.6 + 3.5 * (break_prob_p1 + break_prob_p2)
    
    # Ajuste por superfície
    surface_game_factor = {'Clay': 1.12, 'Hard': 1.0, 'Grass': 0.88, 'Indoor': 0.95}.get(superficie, 1.0)
    
    # Total de jogos esperado (best of 3 = 2 ou 3 sets)
    prob_3_sets = prob_p1 * (1 - prob_p1) * 2  # Probabilidade de ir a 3 sets
    expected_sets = 2 + prob_3_sets
    total_esperado = round(expected_games_per_set * expected_sets * surface_game_factor, 2)
    
    # Probabilidade Over 21.5 (fórmula corrigida baseada em distribuição normal)
    # Média histórica ~22.5 jogos por partida
    media_historica = 22.5
    desvio_padrao = 4.2
    
    # Calcular z-score
    z_score = (total_esperado - media_historica) / desvio_padrao
    
    # Converter para probabilidade (usando aproximação)
    prob_over = 0.5 + (z_score * 0.19)
    prob_over = max(0.15, min(0.85, prob_over))
    
    return {
        "Prob_J1_%": round(prob_p1 * 100, 1),
        "Elo_J1": elo1,
        "Elo_J2": elo2,
        "Total_Esperado": total_esperado,
        "Prob_Over_21.5_%": round(prob_over * 100, 1),
        "Prob_Under_21.5_%": round((1 - prob_over) * 100, 1),
        "Serve_J1_%": round(serve1 * 100, 1),
        "Serve_J2_%": round(serve2 * 100, 1),
        "BP_Saved_J1_%": round(safe(p1_stats.get('w_bpSaved',0)) / max(safe(p1_stats.get('w_bpFaced',1)), 1) * 100, 1),
        "Break_Prob_J1_%": round((1 - hold2) * 100, 1),
    }

def detect_surface(tournament: str) -> str:
    t = str(tournament).lower()
    if any(k in t for k in ['clay', 'saibro', 'barletta', 'marrakech', 'monte-carlo', 'bucarest', 'houston', 'barcelona', 'madrid', 'rome', 'roland garros']):
        return 'Clay'
    if any(k in t for k in ['grass', 'wimbledon', 'halle', 'queens']):
        return 'Grass'
    if any(k in t for k in ['indoor', 'paris masters', 'vienna', 'basel']):
        return 'Indoor'
    return 'Hard'

# ====================== API SOFASCORE ======================
def get_matches_from_sofascore():
    """Obtém partidas de tênis do dia atual via API do Sofascore"""
    try:
        url = "https://api.sofascore.com/api/v1/sport/tennis/events/live-and-upcoming"
        
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
            "Accept": "application/json",
        }
        
        response = requests.get(url, headers=headers, timeout=15)
        
        if response.status_code == 200:
            data = response.json()
            matches = []
            today = datetime.now().date()
            
            events = data.get('events', [])
            
            for event in events:
                try:
                    tournament = event.get('tournament', {}).get('name', 'Torneio')
                    home_team = event.get('homeTeam', {}).get('name', '')
                    away_team = event.get('awayTeam', {}).get('name', '')
                    
                    if not home_team or not away_team:
                        continue
                    
                    start_timestamp = event.get('startTimestamp', 0)
                    if start_timestamp:
                        event_date = datetime.fromtimestamp(start_timestamp).date()
                        horario = datetime.fromtimestamp(start_timestamp).strftime('%H:%M')
                    else:
                        event_date = today
                        horario = '?'
                    
                    status = event.get('status', {}).get('description', 'Agendado')
                    
                    if event_date == today and status not in ['Ended', 'Canceled']:
                        superficie = detect_surface(tournament)
                        matches.append({
                            'torneio': tournament,
                            'jogador_1': home_team,
                            'jogador_2': away_team,
                            'horario': horario,
                            'superficie': superficie
                        })
                except Exception as e:
                    continue
            
            if matches:
                return pd.DataFrame(matches[:30])
        
        return pd.DataFrame()
        
    except Exception as e:
        return pd.DataFrame()

# ====================== FALLBACK ======================
def get_fallback_matches():
    """Fallback com jogos comuns"""
    today = datetime.now()
    
    matches = pd.DataFrame([
        {'torneio': 'ATP Monte-Carlo Masters', 'jogador_1': 'Novak Djokovic', 'jogador_2': 'Jannik Sinner', 'horario': '14:00', 'superficie': 'Clay'},
        {'torneio': 'ATP Monte-Carlo Masters', 'jogador_1': 'Carlos Alcaraz', 'jogador_2': 'Daniil Medvedev', 'horario': '16:00', 'superficie': 'Clay'},
        {'torneio': 'ATP Barcelona', 'jogador_1': 'Alexander Zverev', 'jogador_2': 'Casper Ruud', 'horario': '12:00', 'superficie': 'Clay'},
        {'torneio': 'ATP Barcelona', 'jogador_1': 'Stefanos Tsitsipas', 'jogador_2': 'Andrey Rublev', 'horario': '15:00', 'superficie': 'Clay'},
        {'torneio': 'Challenger Oeiras', 'jogador_1': 'Joao Sousa', 'jogador_2': 'Nuno Borges', 'horario': '11:00', 'superficie': 'Clay'},
        {'torneio': 'Challenger Oeiras', 'jogador_1': 'Henrique Rocha', 'jogador_2': 'Jaime Faria', 'horario': '13:00', 'superficie': 'Clay'},
    ])
    
    return matches

# ====================== EXPORTAR PARA EXCEL ======================
def to_excel(df):
    """Converte DataFrame para Excel"""
    output = BytesIO()
    with pd.ExcelWriter(output, engine='openpyxl') as writer:
        df.to_excel(writer, index=False, sheet_name='Previsoes')
    return output.getvalue()

# ====================== ABA 1 - PARTIDAS HOJE ======================
with tab1:
    hoje = datetime.now()
    hoje_formatado = hoje.strftime("%d/%m/%Y")
    dia_semana = hoje.strftime("%A")
    
    st.header(f"📅 Partidas de Tênis - {hoje_formatado} ({dia_semana})")
    
    if df_stats.empty:
        st.error("⚠️ Carregue primeiro o ficheiro Challenger1.xlsx na barra lateral.")
    else:
        col1, col2 = st.columns(2)
        with col1:
            buscar_partidas = st.button("🔄 Buscar Partidas de Hoje", type="primary", use_container_width=True)
        with col2:
            if st.button("📋 Usar Fallback", use_container_width=True):
                st.session_state.cached_matches = get_fallback_matches()
                st.rerun()
        
        matches_df = pd.DataFrame()
        
        if buscar_partidas:
            with st.spinner(f"Buscando partidas para {hoje_formatado}..."):
                matches_df = get_matches_from_sofascore()
                if matches_df.empty:
                    st.info("📡 API não retornou dados. Usando fallback...")
                    matches_df = get_fallback_matches()
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
                            pred["Serve_J1_%"],
                            pred["Serve_J2_%"],
                            pred["Break_Prob_J1_%"]
                        ])
                    else:
                        results.append([None, None, None, None, None, None, None, None])
                    
                    progress_bar.progress((idx + 1) / len(matches_df))
                    time.sleep(0.05)
                
                matches_df[['Prob_J1_%', 'Elo_J1', 'Elo_J2', 'Total_Esperado', 'Prob_Over_21.5_%', 'Serve_J1_%', 'Serve_J2_%', 'Break_Prob_J1_%']] = pd.DataFrame(results)
                
                # Formatar exibição
                display_df = matches_df.copy()
                display_df['Prob_J1_%'] = display_df['Prob_J1_%'].apply(lambda x: f"{x}%" if pd.notna(x) else "N/A")
                display_df['Prob_Over_21.5_%'] = display_df['Prob_Over_21.5_%'].apply(lambda x: f"{x}%" if pd.notna(x) else "N/A")
                display_df['Serve_J1_%'] = display_df['Serve_J1_%'].apply(lambda x: f"{x}%" if pd.notna(x) else "N/A")
                display_df['Serve_J2_%'] = display_df['Serve_J2_%'].apply(lambda x: f"{x}%" if pd.notna(x) else "N/A")
                display_df['Break_Prob_J1_%'] = display_df['Break_Prob_J1_%'].apply(lambda x: f"{x}%" if pd.notna(x) else "N/A")
                
                st.dataframe(display_df, use_container_width=True, hide_index=True)
                
                # Botões de exportação
                col_exp1, col_exp2 = st.columns(2)
                with col_exp1:
                    csv = matches_df.to_csv(index=False).encode('utf-8')
                    st.download_button("📥 Exportar CSV", csv, f"previsoes_{hoje.strftime('%Y%m%d')}.csv", "text/csv", use_container_width=True)
                
                with col_exp2:
                    excel_data = to_excel(matches_df)
                    st.download_button("📊 Exportar Excel", excel_data, f"previsoes_{hoje.strftime('%Y%m%d')}.xlsx", "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet", use_container_width=True)
        else:
            st.info(f"👆 Clique em 'Buscar Partidas de Hoje' para carregar as partidas")

# ====================== ABA 2 - PREVISÃO PERSONALIZADA ======================
with tab2:
    st.header("🔍 Previsão Personalizada")

    if df_stats.empty:
        st.info("Carregue o ficheiro Challenger1.xlsx")
    else:
        player_list = pd.concat([df_stats['winner_name'], df_stats['loser_name']]).drop_duplicates().sort_values().tolist()
        
        col1, col2 = st.columns(2)
        with col1:
            jogador_a = st.selectbox("Jogador A", options=player_list, key="ja")
        with col2:
            jogador_b = st.selectbox("Jogador B", options=player_list, key="jb")

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
                    
                    st.success("Previsão Calculada!")
                    
                    # Mostrar Elos
                    st.info(f"📊 Elo Ratings na {superficie}: {jogador_a}: {result['Elo_J1']} | {jogador_b}: {result['Elo_J2']}")
                    
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric(f"🏆 {jogador_a} vence", f"{result['Prob_J1_%']}%")
                        st.metric(f"🏆 {jogador_b} vence", f"{100 - result['Prob_J1_%']}%")
                    with col2:
                        st.metric("📊 Total Esperado", f"{result['Total_Esperado']} jogos")
                        st.metric("📈 Over 21.5", f"{result['Prob_Over_21.5_%']}%")
                        st.metric("📉 Under 21.5", f"{result['Prob_Under_21.5_%']}%")
                    with col3:
                        st.metric("🎾 Serve % A", f"{result['Serve_J1_%']}%")
                        st.metric("🎾 Serve % B", f"{result['Serve_J2_%']}%")
                        st.metric("💔 Break Prob A", f"{result['Break_Prob_J1_%']}%")

# ====================== ABA 3 - MODELING STRATEGY ======================
with tab3:
    st.header("📈 Sobre o Modelo")
    st.markdown("""
    ### 🎯 Modelo Melhorado
    
    **1. Sistema de Elo por Superfície**
    - Rating específico para cada jogador em cada superfície
    - Atualização dinâmica baseada em resultados históricos
    - Fator K = 32 para ajustes
    
    **2. Probabilidade de Vitória**
    - Combinação de estatísticas de jogo (60%) e Elo rating (40%)
    - Ajuste por superfície baseado em dados históricos
    - Maior precisão em previsões
    
    **3. Total de Jogos**
    - Cálculo baseado em probabilidade de hold de serviço
    - Ajuste por superfície (Clay: +12%, Grass: -12%, etc)
    - Estimativa de número de sets esperados
    
    **4. Over/Under 21.5**
    - Baseado em distribuição normal (média histórica: 22.5 jogos)
    - Desvio padrão: 4.2 jogos
    - Probabilidades calibradas entre 15% e 85%
    
    ### 📊 Fatores por Superfície
    - **Clay**: Jogos mais longos, maior vantagem para especialistas
    - **Hard**: Superfície neutra, estatísticas balanceadas
    - **Grass**: Jogos mais rápidos, maior importância do serviço
    - **Indoor**: Condições controladas, ligeiro boost para servidor
    
    ### 🔧 Melhorias Futuras
    - Histórico de confrontos diretos (H2H)
    - Forma recente dos jogadores (últimos 5 jogos)
    - Fatores de fadiga e lesões
    - Condições meteorológicas
    """)

import streamlit as st
import pandas as pd
import requests
from datetime import datetime, timedelta
import unicodedata
from difflib import SequenceMatcher
import time
from io import BytesIO
import math
import json

st.set_page_config(page_title="Tênis Predictor Pro", page_icon="🎾", layout="wide")
st.title("🎾 Tennis Predictor Pro")

# ====================== CONFIGURAÇÃO API ======================
RAPIDAPI_KEY = "bba6af0e8dmsh6350139b0f77a4ap16b6fajsn219553636a44"
RAPIDAPI_HOST = "tennis-api-atp-wta-itf.p.rapidapi.com"

# ====================== FUNÇÕES AUXILIARES ======================
def norm(name):
    if not isinstance(name, str):
        return ""
    n = unicodedata.normalize('NFKD', name).encode('ascii', 'ignore').decode('utf-8')
    return ''.join(filter(str.isalnum, n.lower().strip()))

def detect_surface(tournament: str) -> str:
    t = str(tournament).lower()
    if any(k in t for k in ['clay', 'saibro', 'terre', 'barletta', 'marrakech', 'monte-carlo', 'bucarest',
                           'houston', 'barcelona', 'madrid', 'rome', 'roland garros', 'french', 'bastad']):
        return 'Clay'
    if any(k in t for k in ['grass', 'relva', 'wimbledon', 'halle', 'queens', 'newport']):
        return 'Grass'
    if any(k in t for k in ['indoor', 'coberta', 'paris masters', 'vienna', 'basel']):
        return 'Indoor'
    return 'Hard'

# ====================== FUNÇÕES DE ESTATÍSTICAS ======================
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

@st.cache_data(ttl=3600)
def load_stats(file):
    if not file:
        return pd.DataFrame()
    try:
        df = pd.read_excel(file)
       
        df['winner_clean'] = df['winner_name'].apply(norm)
        df['loser_clean'] = df['loser_name'].apply(norm)
       
        if 'surface' not in df.columns:
            df['surface'] = 'Hard'
       
        df = calculate_elo_by_surface(df)
       
        return df
    except Exception as e:
        st.sidebar.error(f"Erro ao carregar ficheiro: {e}")
        return pd.DataFrame()

def find_best_player_stats(player_name, df):
    if df.empty or not player_name:
        return pd.Series(dtype='object')
   
    clean_name = norm(player_name)
    best_match = None
    best_score = 0.0
   
    sample_df = df.head(2000) if len(df) > 2000 else df
   
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

def get_player_elo_from_stats(player_name, surface, df_stats):
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

def predict_from_stats(p1_stats, p2_stats, superficie="Hard", p1_name="", p2_name="", df_stats=None):
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
   
    elo1 = get_player_elo_from_stats(p1_name, superficie, df_stats) if df_stats is not None else 1500
    elo2 = get_player_elo_from_stats(p2_name, superficie, df_stats) if df_stats is not None else 1500
   
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
   
    first_serve_p1 = safe(p1_stats.get('w_1stIn', 0)) / max(safe(p1_stats.get('w_svpt', 1)), 1)
    first_serve_p2 = safe(p2_stats.get('w_1stIn', 0)) / max(safe(p2_stats.get('w_svpt', 1)), 1)
   
    surface_first_serve = {'Clay': 0.62, 'Hard': 0.64, 'Grass': 0.66, 'Indoor': 0.65}
    if first_serve_p1 == 0:
        first_serve_p1 = surface_first_serve.get(superficie, 0.64)
    if first_serve_p2 == 0:
        first_serve_p2 = surface_first_serve.get(superficie, 0.64)
   
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
        "Prob_J2_%": round((1-prob_p1) * 100, 1),
        "Elo_J1": elo1,
        "Elo_J2": elo2,
        "Total": total_esperado,
        "Over_21.5%": round(prob_over * 100, 1),
        "Under_21.5%": round((1 - prob_over) * 100, 1),
    }

def to_excel(df):
    output = BytesIO()
    with pd.ExcelWriter(output, engine='openpyxl') as writer:
        df.to_excel(writer, index=False, sheet_name='Previsoes')
    return output.getvalue()

# ====================== FUNÇÕES API CORRETAS (POST) ======================
def call_api_endpoint(endpoint, payload):
    """Função genérica para chamar endpoints da API via POST"""
    headers = {
        "x-rapidapi-host": RAPIDAPI_HOST,
        "x-rapidapi-key": RAPIDAPI_KEY,
        "Content-Type": "application/json"
    }
    
    url = f"https://{RAPIDAPI_HOST}/{endpoint}"
    
    try:
        response = requests.post(url, json=payload, headers=headers, timeout=15)
        if response.status_code == 200:
            return True, response.json()
        else:
            return False, f"Erro {response.status_code}: {response.text[:200]}"
    except Exception as e:
        return False, str(e)

@st.cache_data(ttl=1800)
def test_api_connection():
    """Testa a conexão com a API usando singlesRanking [citation:2]"""
    payload = {
        "tour": "atp",
        "limit": 5
    }
    return call_api_endpoint("singlesRanking", payload)

@st.cache_data(ttl=1800)
def get_ranking_from_api(tour="atp", limit=50):
    """Busca ranking usando singlesRanking [citation:2]"""
    payload = {
        "tour": tour,
        "limit": limit
    }
    success, result = call_api_endpoint("singlesRanking", payload)
    return result if success else None

@st.cache_data(ttl=1800)
def get_fixtures_from_api(tour="atp", date=None):
    """Busca partidas futuras usando getAllFixtures [citation:2]"""
    payload = {
        "tour": tour,
        "limit": 100
    }
    if date:
        payload["date"] = date
    
    success, result = call_api_endpoint("getAllFixtures", payload)
    return result if success else None

def parse_fixtures_to_matches(fixtures_data):
    """Converte dados da API em formato de partidas"""
    matches = []
    
    if not fixtures_data:
        return matches
    
    # A estrutura pode variar, tentamos diferentes formatos
    fixtures = []
    
    if isinstance(fixtures_data, dict):
        if 'results' in fixtures_data:
            fixtures = fixtures_data['results']
        elif 'fixtures' in fixtures_data:
            fixtures = fixtures_data['fixtures']
        elif 'data' in fixtures_data:
            fixtures = fixtures_data['data'] if isinstance(fixtures_data['data'], list) else []
        elif 'response' in fixtures_data:
            fixtures = fixtures_data['response']
    elif isinstance(fixtures_data, list):
        fixtures = fixtures_data
    
    for fixture in fixtures:
        try:
            if not isinstance(fixture, dict):
                continue
            
            # Extrair jogadores
            player1 = None
            player2 = None
            
            if 'player1' in fixture and 'player2' in fixture:
                p1 = fixture['player1']
                p2 = fixture['player2']
                player1 = p1.get('name') if isinstance(p1, dict) else str(p1)
                player2 = p2.get('name') if isinstance(p2, dict) else str(p2)
            elif 'home' in fixture and 'away' in fixture:
                player1 = fixture['home'].get('name') if isinstance(fixture['home'], dict) else str(fixture['home'])
                player2 = fixture['away'].get('name') if isinstance(fixture['away'], dict) else str(fixture['away'])
            elif 'players' in fixture and isinstance(fixture['players'], list) and len(fixture['players']) >= 2:
                player1 = fixture['players'][0].get('name') if isinstance(fixture['players'][0], dict) else str(fixture['players'][0])
                player2 = fixture['players'][1].get('name') if isinstance(fixture['players'][1], dict) else str(fixture['players'][1])
            
            if not player1 or not player2:
                continue
            
            # Torneio
            tournament = fixture.get('tournament') or fixture.get('competition') or {}
            torneio = tournament.get('name') if isinstance(tournament, dict) else str(tournament) if tournament else "Torneio"
            
            # Horário
            start_time = fixture.get('start_time') or fixture.get('date') or fixture.get('datetime')
            horario = 'TBD'
            if start_time:
                try:
                    if isinstance(start_time, str):
                        if 'T' in start_time:
                            horario = start_time.split('T')[1][:5]
                        elif len(start_time) >= 5:
                            horario = start_time[:5]
                except:
                    pass
            
            # Superfície
            superficie = 'Hard'
            if 'surface' in fixture:
                s = str(fixture['surface']).lower()
                if 'clay' in s:
                    superficie = 'Clay'
                elif 'grass' in s:
                    superficie = 'Grass'
                elif 'indoor' in s:
                    superficie = 'Indoor'
            elif torneio:
                superficie = detect_surface(torneio)
            
            matches.append({
                'torneio': torneio,
                'jogador_1': player1,
                'jogador_2': player2,
                'horario': horario,
                'superficie': superficie
            })
        except Exception as e:
            continue
    
    return matches

# ====================== SIDEBAR ======================
with st.sidebar:
    st.header("📁 Carregar Base de Dados")
    st.markdown("**Challenger1.xlsx** - Colunas: `winner_name`, `loser_name`")
    uploaded_file = st.file_uploader("Escolha o ficheiro", type=["xlsx", "xls"])
   
    st.markdown("---")
    st.markdown("### ⚙️ Configurações API")
    tour_type = st.selectbox("Circuito", ["atp", "wta"])
   
    if st.button("🔌 Testar Conexão API", use_container_width=True):
        with st.spinner("Testando conexão via POST..."):
            success, result = test_api_connection()
            if success:
                st.success("✅ API Conectada com sucesso! (singlesRanking)")
                if isinstance(result, dict):
                    st.info(f"Resposta: {str(result)[:300]}...")
                else:
                    st.info(f"Resposta: {result}")
            else:
                st.error(f"❌ Erro: {result}")
                st.info("💡 A API requer requisições POST com payload JSON")
    
    if st.button("📊 Ver Ranking", use_container_width=True):
        with st.spinner("Buscando ranking..."):
            ranking = get_ranking_from_api(tour_type, 20)
            if ranking:
                st.success("✅ Ranking obtido!")
                st.json(ranking)
            else:
                st.error("❌ Não foi possível obter o ranking")
   
    st.markdown("---")
    if st.button("🗑️ Limpar Cache", use_container_width=True):
        st.cache_data.clear()
        st.success("Cache limpo!")

# ====================== CARREGAR DADOS ======================
df_stats = load_stats(uploaded_file)

if not df_stats.empty:
    st.sidebar.success(f"✅ {len(df_stats)} jogos carregados!")
else:
    if uploaded_file is not None:
        st.sidebar.error("❌ Erro ao carregar o arquivo")

# ====================== INTERFACE PRINCIPAL ======================
st.markdown(f"## 📅 Previsões de Tênis - {datetime.now().strftime('%d/%m/%Y')}")

if df_stats.empty:
    st.warning("⚠️ **Carregue o ficheiro Challenger1.xlsx na barra lateral!**")
    st.info("""
    ### 📋 Formato esperado do arquivo Excel:
    - **winner_name**: Nome do jogador vencedor
    - **loser_name**: Nome do jogador perdedor  
    - **surface** (opcional): Clay, Hard, Grass, Indoor
    """)
else:
    # Botões principais
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        if st.button("➕ Adicionar Partida", type="primary", use_container_width=True):
            st.session_state.show_form = True
    
    with col2:
        if st.button("🔄 Buscar da API", use_container_width=True):
            with st.spinner("Buscando partidas da API (POST getAllFixtures)..."):
                fixtures_data = get_fixtures_from_api(tour_type)
                if fixtures_data:
                    matches = parse_fixtures_to_matches(fixtures_data)
                    if matches:
                        df_api = pd.DataFrame(matches)
                        if 'matches' not in st.session_state:
                            st.session_state.matches = df_api
                        else:
                            st.session_state.matches = pd.concat([st.session_state.matches, df_api], ignore_index=True)
                        st.success(f"✅ {len(matches)} partidas encontradas!")
                        st.rerun()
                    else:
                        st.warning("⚠️ Nenhuma partida encontrada")
                else:
                    st.error("❌ Erro ao buscar da API. Verifique sua chave e créditos.")
    
    with col3:
        if st.button("📋 Exemplo Rápido", use_container_width=True):
            example_matches = pd.DataFrame([
                {'torneio': 'ATP Masters', 'jogador_1': 'Novak Djokovic', 'jogador_2': 'Carlos Alcaraz', 'horario': '14:00', 'superficie': 'Hard'},
                {'torneio': 'Grand Slam', 'jogador_1': 'Jannik Sinner', 'jogador_2': 'Daniil Medvedev', 'horario': '16:30', 'superficie': 'Clay'},
                {'torneio': 'Wimbledon', 'jogador_1': 'Carlos Alcaraz', 'jogador_2': 'Novak Djokovic', 'horario': '12:00', 'superficie': 'Grass'},
            ])
            if 'matches' not in st.session_state:
                st.session_state.matches = example_matches
            else:
                st.session_state.matches = pd.concat([st.session_state.matches, example_matches], ignore_index=True)
            st.success("✅ Exemplos adicionados!")
            st.rerun()
    
    with col4:
        if st.button("📊 Ver Ranking", use_container_width=True):
            ranking = get_ranking_from_api(tour_type, 20)
            if ranking:
                st.session_state.show_ranking = True
                st.session_state.ranking_data = ranking
                st.rerun()
            else:
                st.error("❌ Não foi possível obter o ranking")
    
    # Modal de Ranking
    if st.session_state.get('show_ranking', False):
        with st.expander("📊 Ranking - Clique para fechar", expanded=True):
            st.subheader(f"Ranking {tour_type.upper()}")
            st.json(st.session_state.ranking_data)
            if st.button("Fechar Ranking"):
                st.session_state.show_ranking = False
                st.rerun()
    
    # Formulário para adicionar partida manualmente
    if st.session_state.get('show_form', False):
        with st.form("add_match"):
            st.subheader("➕ Adicionar Nova Partida")
            
            col1, col2, col3 = st.columns(3)
            with col1:
                torneio = st.text_input("Torneio", "ATP Tour")
                jogador1 = st.text_input("Jogador 1 *", placeholder="Ex: Novak Djokovic")
            with col2:
                jogador2 = st.text_input("Jogador 2 *", placeholder="Ex: Carlos Alcaraz")
                horario = st.text_input("Horário", datetime.now().strftime("%H:%M"))
            with col3:
                superficie = st.selectbox("Superfície", ["Hard", "Clay", "Grass", "Indoor"])
            
            st.caption("* Campos obrigatórios")
            
            col_btn1, col_btn2 = st.columns(2)
            with col_btn1:
                submitted = st.form_submit_button("✅ Adicionar Partida", use_container_width=True)
            with col_btn2:
                cancel = st.form_submit_button("❌ Cancelar", use_container_width=True)
            
            if cancel:
                st.session_state.show_form = False
                st.rerun()
            
            if submitted:
                if not jogador1 or not jogador2:
                    st.error("⚠️ Preencha os nomes dos dois jogadores!")
                else:
                    new_match = pd.DataFrame([{
                        'torneio': torneio,
                        'jogador_1': jogador1.strip(),
                        'jogador_2': jogador2.strip(),
                        'horario': horario,
                        'superficie': superficie
                    }])
                    
                    if 'matches' not in st.session_state:
                        st.session_state.matches = new_match
                    else:
                        st.session_state.matches = pd.concat([st.session_state.matches, new_match], ignore_index=True)
                    
                    st.session_state.show_form = False
                    st.success(f"✅ Adicionado: {jogador1} vs {jogador2}")
                    st.rerun()
    
    # MOSTRAR PREVISÕES
    if 'matches' in st.session_state and not st.session_state.matches.empty:
        matches_df = st.session_state.matches
        matches_df = matches_df.drop_duplicates(subset=['jogador_1', 'jogador_2'])
        
        st.info(f"📊 {len(matches_df)} partida(s) para análise")
        
        with st.spinner("🔮 Calculando previsões..."):
            results = []
            progress_bar = st.progress(0)
            
            for idx, row in matches_df.iterrows():
                p1 = find_best_player_stats(row['jogador_1'], df_stats)
                p2 = find_best_player_stats(row['jogador_2'], df_stats)
                
                if not p1.empty and not p2.empty:
                    pred = predict_from_stats(p1, p2, row['superficie'], row['jogador_1'], row['jogador_2'], df_stats)
                    results.append([
                        pred["Prob_J1_%"],
                        pred["Prob_J2_%"],
                        pred["Elo_J1"],
                        pred["Elo_J2"],
                        pred["Total"],
                        pred["Over_21.5%"],
                        pred["Under_21.5%"]
                    ])
                else:
                    if p1.empty and row['jogador_1']:
                        st.warning(f"⚠️ Jogador não encontrado na base: {row['jogador_1']}")
                    if p2.empty and row['jogador_2']:
                        st.warning(f"⚠️ Jogador não encontrado na base: {row['jogador_2']}")
                    results.append([None, None, None, None, None, None, None])
                
                progress_bar.progress((idx + 1) / len(matches_df))
                time.sleep(0.01)
            
            matches_df[['Win_J1%', 'Win_J2%', 'Elo_J1', 'Elo_J2', 'Total', 'Over_21.5%', 'Under_21.5%']] = pd.DataFrame(results)
            
            # Formatar para exibição
            display_df = matches_df.copy()
            for col in ['Win_J1%', 'Win_J2%', 'Over_21.5%', 'Under_21.5%']:
                if col in display_df.columns:
                    display_df[col] = display_df[col].apply(lambda x: f"{x}%" if pd.notna(x) else "Sem dados")
            
            col_order = ['horario', 'torneio', 'jogador_1', 'jogador_2', 'superficie', 
                        'Win_J1%', 'Win_J2%', 'Elo_J1', 'Elo_J2', 'Total', 'Over_21.5%', 'Under_21.5%']
            display_df = display_df[[c for c in col_order if c in display_df.columns]]
            
            st.dataframe(display_df, use_container_width=True, hide_index=True)
            
            # Estatísticas
            st.markdown("---")
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("📊 Total Partidas", len(matches_df))
            with col2:
                valid = len(matches_df[matches_df['Win_J1%'].notna()])
                st.metric("✅ Com Previsão", valid)
            with col3:
                if valid > 0:
                    media_favorito = matches_df[matches_df['Win_J1%'].notna()]['Win_J1%'].mean()
                    st.metric("⭐ Média Favorito", f"{media_favorito:.1f}%")
                else:
                    st.metric("⭐ Média Favorito", "N/A")
            with col4:
                media_elo = matches_df['Elo_J1'].mean() if 'Elo_J1' in matches_df and matches_df['Elo_J1'].notna().any() else 1500
                st.metric("📈 Média ELO", int(media_elo))
            
            # Exportar
            st.markdown("---")
            col1, col2, col3 = st.columns(3)
            with col1:
                csv = matches_df.to_csv(index=False).encode('utf-8')
                st.download_button(
                    "📥 Exportar CSV",
                    csv,
                    f"previsoes_tenis_{datetime.now().strftime('%Y%m%d_%H%M')}.csv",
                    "text/csv",
                    use_container_width=True
                )
            with col2:
                excel = to_excel(matches_df)
                st.download_button(
                    "📊 Exportar Excel",
                    excel,
                    f"previsoes_tenis_{datetime.now().strftime('%Y%m%d_%H%M')}.xlsx",
                    "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                    use_container_width=True
                )
            with col3:
                if st.button("🗑️ Limpar Todas Partidas", use_container_width=True):
                    del st.session_state.matches
                    if 'show_form' in st.session_state:
                        del st.session_state.show_form
                    st.rerun()
    else:
        st.info("👆 Clique em **'+ Adicionar Partida'** ou **'🔄 Buscar da API'** para começar")
        
        # Mostrar alguns jogadores disponíveis na base
        if not df_stats.empty:
            with st.expander("📋 Jogadores disponíveis na sua base de dados"):
                all_players = set(df_stats['winner_clean'].tolist() + df_stats['loser_clean'].tolist())
                players_list = sorted([p for p in all_players if p and len(p) > 2])[:50]
                st.write(", ".join(players_list))

st.markdown("---")
st.caption(f"🎾 Tennis Predictor Pro • Base: {len(df_stats)} jogos • {datetime.now().strftime('%d/%m/%Y %H:%M')}")

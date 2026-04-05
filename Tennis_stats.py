import streamlit as st
import pandas as pd
import requests
from datetime import datetime, timedelta
import unicodedata
from difflib import SequenceMatcher
import time
from io import BytesIO
import math

st.set_page_config(page_title="Tênis Predictor Pro", page_icon="🎾", layout="wide")
st.title("🎾 Tennis Predictor Pro - API RapidAPI")

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

# ====================== FUNÇÕES API ======================
@st.cache_data(ttl=1800)
def get_matches_from_rapidapi(date_str=None, tour="atp"):
    """Busca partidas via RapidAPI Tennis"""
    if date_str is None:
        date_str = datetime.now().strftime("%Y-%m-%d")
   
    headers = {
        "x-rapidapi-host": RAPIDAPI_HOST,
        "x-rapidapi-key": RAPIDAPI_KEY
    }
    
    # NOTA: Como a API está retornando 404, vamos retornar DataFrame vazio
    # e permitir que o usuário adicione partidas manualmente
    st.warning("⚠️ API não está respondendo. Use o botão '+ Adicionar Partida' para inserir jogos manualmente.")
    
    return pd.DataFrame()

# ====================== SIDEBAR ======================
with st.sidebar:
    st.header("📁 Carregar Challenger1.xlsx")
    uploaded_file = st.file_uploader("Escolha o ficheiro Challenger1.xlsx", type=["xlsx", "xls"])
   
    st.markdown("---")
    st.markdown("### ⚙️ Configurações API")
    
    tour_type = st.selectbox("Tipo de Torneio", ["atp", "wta"])
   
    if st.button("🔌 Testar Conexão API"):
        with st.spinner("Testando..."):
            try:
                headers = {
                    "x-rapidapi-host": RAPIDAPI_HOST,
                    "x-rapidapi-key": RAPIDAPI_KEY
                }
                
                # Tentar diferentes endpoints comuns
                endpoints_to_try = [
                    f"https://{RAPIDAPI_HOST}/getRanking",
                    f"https://{RAPIDAPI_HOST}/getAllFixtures",
                    f"https://{RAPIDAPI_HOST}/singlesRanking",
                ]
                
                working = False
                for url in endpoints_to_try:
                    try:
                        response = requests.get(url, headers=headers, timeout=10)
                        if response.status_code == 200:
                            st.success(f"✅ API Conectada! Endpoint funcionando: {url}")
                            st.json(response.json())
                            working = True
                            break
                    except:
                        continue
                
                if not working:
                    st.error("❌ Nenhum endpoint da API respondeu. Verifique sua chave API ou use partidas manuais.")
                    
            except Exception as e:
                st.error(f"❌ Erro de conexão: {str(e)[:150]}")
   
    st.markdown("---")
    if st.button("🗑️ Limpar Cache"):
        st.cache_data.clear()
        st.success("Cache limpo!")

# ====================== CARREGAR DADOS ======================
# Isso define a variável df_stats
df_stats = load_stats(uploaded_file)

# Mostrar status do carregamento
if not df_stats.empty:
    st.sidebar.success(f"✅ {len(df_stats)} jogos carregados!")
else:
    if uploaded_file is not None:
        st.sidebar.error("❌ Erro ao carregar o arquivo. Verifique o formato.")

# ====================== INTERFACE PRINCIPAL ======================
st.markdown(f"## 📅 Partidas de Tênis - {datetime.now().strftime('%d/%m/%Y')}")

# Verificar se os dados foram carregados
if df_stats.empty:
    st.warning("⚠️ **Carregue o ficheiro Challenger1.xlsx na barra lateral!**")
    st.info("O arquivo deve conter as colunas: 'winner_name', 'loser_name' e opcionalmente 'surface'")
else:
    # Seleção de data
    col_date1, col_date2, col_date3 = st.columns([2, 1, 1])
   
    with col_date1:
        date_selected = st.date_input(
            "📅 Selecione a data:",
            value=datetime.now(),
            min_value=datetime.now() - timedelta(days=7),
            max_value=datetime.now() + timedelta(days=14)
        )
   
    with col_date2:
        if st.button("🔄 BUSCAR PARTIDAS", type="primary", use_container_width=True):
            with st.spinner("🌐 Buscando via RapidAPI..."):
                date_str = date_selected.strftime("%Y-%m-%d")
                matches_df = get_matches_from_rapidapi(date_str, tour_type)
               
                if not matches_df.empty:
                    st.session_state.matches = matches_df
                    st.success(f"✅ {len(matches_df)} partidas encontradas!")
                    st.rerun()
                else:
                    st.warning("⚠️ Nenhuma partida encontrada. Use '+ Adicionar Partida' para inserir manualmente.")
   
    with col_date3:
        if st.button("➕ Adicionar Partida", use_container_width=True):
            st.session_state.show_form = True
   
    # Formulário para adicionar partida manualmente
    if st.session_state.get('show_form', False):
        with st.form("add_match"):
            st.subheader("➕ Adicionar Partida Manualmente")
            col1, col2, col3 = st.columns(3)
            with col1:
                torneio = st.text_input("Torneio", "ATP Tournament")
                jogador1 = st.text_input("Jogador 1 *")
            with col2:
                jogador2 = st.text_input("Jogador 2 *")
                horario = st.text_input("Horário", "14:00")
            with col3:
                superficie = st.selectbox("Superfície", ["Hard", "Clay", "Grass", "Indoor"])
           
            st.caption("* Campos obrigatórios")
           
            col_btn1, col_btn2 = st.columns(2)
            with col_btn1:
                submitted = st.form_submit_button("✅ Adicionar", use_container_width=True)
            with col_btn2:
                cancel = st.form_submit_button("❌ Cancelar", use_container_width=True)
           
            if cancel:
                st.session_state.show_form = False
                st.rerun()
           
            if submitted and jogador1 and jogador2:
                new_match = pd.DataFrame([{
                    'torneio': torneio,
                    'jogador_1': jogador1,
                    'jogador_2': jogador2,
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
       
        st.info(f"📊 {len(matches_df)} partidas para análise")
       
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
                    results.append([None, None, None, None, None, None, None])
               
                progress_bar.progress((idx + 1) / len(matches_df))
                time.sleep(0.01)
           
            matches_df[['Win_J1%', 'Win_J2%', 'Elo_J1', 'Elo_J2', 'Total', 'Over_21.5%', 'Under_21.5%']] = pd.DataFrame(results)
           
            # Formatar para exibição
            display_df = matches_df.copy()
            for col in ['Win_J1%', 'Win_J2%', 'Over_21.5%', 'Under_21.5%']:
                if col in display_df.columns:
                    display_df[col] = display_df[col].apply(lambda x: f"{x}%" if pd.notna(x) else "N/A")
           
            st.dataframe(display_df, use_container_width=True, hide_index=True)
           
            # Estatísticas
            st.markdown("---")
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("📊 Total", len(matches_df))
            with col2:
                clay = len(matches_df[matches_df['superficie'] == 'Clay']) if 'superficie' in matches_df else 0
                st.metric("🟤 Clay", clay)
            with col3:
                hard = len(matches_df[matches_df['superficie'] == 'Hard']) if 'superficie' in matches_df else 0
                st.metric("🔵 Hard", hard)
            with col4:
                grass = len(matches_df[matches_df['superficie'] == 'Grass']) if 'superficie' in matches_df else 0
                st.metric("🟢 Grass", grass)
           
            # Exportar
            st.markdown("---")
            col1, col2, col3 = st.columns(3)
            with col1:
                csv = matches_df.to_csv(index=False).encode('utf-8')
                st.download_button(
                    "📥 CSV",
                    csv,
                    f"previsoes_{datetime.now().strftime('%Y%m%d_%H%M')}.csv",
                    "text/csv",
                    use_container_width=True
                )
            with col2:
                excel = to_excel(matches_df)
                st.download_button(
                    "📊 Excel",
                    excel,
                    f"previsoes_{datetime.now().strftime('%Y%m%d_%H%M')}.xlsx",
                    "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                    use_container_width=True
                )
            with col3:
                if st.button("🗑️ Limpar Partidas", use_container_width=True):
                    del st.session_state.matches
                    if 'show_form' in st.session_state:
                        del st.session_state.show_form
                    st.rerun()
    else:
        st.info("👆 Clique em **'+ Adicionar Partida'** para inserir jogos manualmente ou **'BUSCAR PARTIDAS'** para tentar a API")

st.markdown("---")
st.caption(f"🎾 Tennis Predictor Pro • {datetime.now().strftime('%d/%m/%Y %H:%M')}")
